# daqio/ao_runner.py
from __future__ import annotations
import asyncio
from typing import Iterable, Optional, Callable, Awaitable, List, Dict, Union
from datetime import datetime, timezone
from time import monotonic_ns

import numpy as np
import nidaqmx
from nidaqmx.system import System
from nidaqmx.constants import AcquisitionType, RegenerationMode
from nidaqmx.stream_writers import AnalogMultiChannelWriter

PublishFn = Callable[[Dict], Awaitable[None]]

ArrayLike = Union[List[float], np.ndarray]
PerChanWave = Union[ArrayLike, Dict[str, ArrayLike], np.ndarray]  # (S,), (S,C), or {ch: (S,)}


class _SyncMapper:
    """
    Live linear map between device sample index k and system wall time (UTC epoch seconds):
        t_sys(k) ≈ a + b * k
    We refine 'b' (slope) against host monotonic time and keep continuity by
    re-pinning 'a' at the current sample k (no timestamp jumps).
    """
    __slots__ = ("Fs", "a_ns", "b_ns_per_sample", "mono0_ns", "alpha")

    def __init__(self, Fs_actual: float, time_format: str):
        self.Fs = float(Fs_actual)
        # Pin intercept to current system clock at start (UTC); caller can format later
        self.a_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        self.mono0_ns = monotonic_ns()
        # Initial slope from requested Fs
        self.b_ns_per_sample = int(round((1.0 / self.Fs) * 1e9))
        # Smoothing for drift (2% new, 98% old)
        self.alpha = 0.02

    def update(self, k_now: int) -> None:
        """Refine slope using monotonic vs device elapsed; re-pin intercept at k_now."""
        if k_now <= 0:
            return
        mono_now_ns = monotonic_ns()
        mono_elapsed_ns = mono_now_ns - self.mono0_ns
        dev_elapsed_ns = int(round((k_now / self.Fs) * 1e9))
        if dev_elapsed_ns <= 0:
            return
        # Ratio ~ 1 + drift; reject absurd outliers
        ratio = mono_elapsed_ns / dev_elapsed_ns
        if 0.98 <= ratio <= 1.02:
            est_b_ns = max(1, int(round(ratio * (1e9 / self.Fs))))
            self.b_ns_per_sample = int(round(
                (1 - self.alpha) * self.b_ns_per_sample + self.alpha * est_b_ns
            ))
            # Re-pin intercept so mapping is continuous at (k_now, wall_now)
            wall_now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
            self.a_ns = wall_now_ns - self.b_ns_per_sample * int(k_now)

    def ts_str_for_k(self, k: int, time_format: str) -> str:
        wall_s = (self.a_ns + self.b_ns_per_sample * int(k)) / 1e9
        # Use naive datetime formatting to keep your original behavior (no timezone suffix)
        return datetime.fromtimestamp(wall_s).strftime(time_format)

class AsyncAORunner:
    """
    Async NI-DAQmx analog-output runner.

    Modes:
      - Random mode (no waveform provided): writes random values every interval.
      - Waveform mode (waveform provided): plays the waveform at
        sample_rate = frequency * len(waveform), repeating for `cycles` (0=forever).
    """

    def __init__(
        self,
        device: str,
        channels: Optional[Iterable[str]] = None,
        *,
        # Random mode params (ignored if waveform provided)
        interval: float = 0.5,
        low: float = 0.0,                     # <-- added (you use self.low below)
        high: float = 1.0,
        seed: Optional[int] = None,
        # Waveform mode params
        waveform: Optional[PerChanWave] = None,
        waveform_cycles_hz: Optional[float] = None,  # NEW user-facing knob (cycles/sec)
        frequency: float = 1.0,                      # legacy alias (cycles/sec)
        cycles: int = 0,             # 0 = run forever
        # Common
        publish: Optional[PublishFn] = None,
        time_format: str = "%Y-%m-%d %H:%M:%S.%f",
    ) -> None:
        self.device = device
        self.channels = list(channels) if channels else None

        # random-mode
        self.interval = float(interval)
        self.low = float(low)
        self.high = float(high)
        self._rng = np.random.default_rng(seed)

        # waveform-mode
        self.waveform = waveform
        # prefer new knob if provided; fall back to legacy 'frequency'
        self.frequency = float(waveform_cycles_hz) if waveform_cycles_hz is not None else float(frequency)
        self.cycles = int(cycles)

        # common
        self.publish = publish
        self.time_format = time_format

        # internals
        self._task: Optional[nidaqmx.Task] = None
        self._ao_ch_names: List[str] = []
        self._runner: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._stopping = asyncio.Event()

        # prepared waveform (samples x channels)
        self._wf_matrix: Optional[np.ndarray] = None   # (S, C) for value lookup
        self._wf_matrix_T: Optional[np.ndarray] = None # (C, S) for DMA write
        self._sample_period: Optional[float] = None    # seconds between samples
        self._samples_per_cycle: Optional[int] = None


    # ---------- lifecycle ----------
    async def start(self) -> None:
        async with self._lock:
            if self._task is not None:
                return
            # discover device / channels
            system = System.local()
            dev = next((d for d in system.devices if d.name == self.device), None)
            if dev is None:
                raise ValueError(f"Device {self.device} not found.")

            if self.channels is None:
                self._ao_ch_names = [ch.name for ch in dev.ao_physical_chans]
                if not self._ao_ch_names:
                    raise ValueError(f"No AO channels found on {self.device}.")
            else:
                self._ao_ch_names = list(self.channels)

            # Prepare mode
            if self.waveform is not None:
                self._prepare_waveform_mode()
            else:
                # random mode; nothing to precompute
                pass

            # Create task
            self._task = nidaqmx.Task()
            for ch in self._ao_ch_names:
                # Keep the same range selection behavior as before for compatibility
                self._task.ao_channels.add_ao_voltage_chan(
                    ch,
                    min_val=min(self.low, -10.0),
                    max_val=max(self.high, 10.0),
                )

            self._stopping.clear()
            self._runner = asyncio.create_task(self._run_loop(), name="AsyncAORunner")

    async def stop(self, safe_zero: bool = True) -> None:
        async with self._lock:
            self._stopping.set()
            if self._runner:
                self._runner.cancel()
                try:
                    await self._runner
                except asyncio.CancelledError:
                    pass
                self._runner = None

            if self._task:
                if safe_zero:
                    try:
                        self._task.write([0.0] * len(self._ao_ch_names))
                    except Exception:
                        pass
                self._task.close()
                self._task = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    # ---------- preparation ----------
    def _prepare_waveform_mode(self) -> None:
        """
        Normalize user-provided waveform into a (S, C) float array and compute timing.
        sample_rate = frequency * S  -> sample_period = 1 / sample_rate
        """
        wf = self.waveform
        chs = self._ao_ch_names

        def to_1d(x: ArrayLike) -> np.ndarray:
            arr = np.asarray(x, dtype=float).reshape(-1)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError("Waveform arrays must be 1-D and non-empty.")
            return arr

        if isinstance(wf, dict):
            # dict: channel -> 1D array
            series = []
            for ch in chs:
                if ch not in wf:
                    raise ValueError(f"Waveform dict missing channel: {ch}")
                series.append(to_1d(wf[ch]))
            lengths = {len(s) for s in series}
            if len(lengths) != 1:
                raise ValueError("All per-channel waveforms must have the same length.")
            matrix = np.column_stack(series)  # (S, C)
        else:
            arr = np.asarray(wf)
            if arr.ndim == 1:
                vec = to_1d(arr)
                matrix = np.column_stack([vec for _ in chs])  # same waveform on all channels
            elif arr.ndim == 2:
                # (S, C) or (C, S) — assume (S, C); reorder if needed
                S, C = arr.shape
                if C == len(chs):
                    matrix = arr.astype(float)
                elif S == len(chs):
                    matrix = arr.T.astype(float)
                else:
                    raise ValueError(
                        f"Waveform 2D shape {arr.shape} does not match channels {len(chs)}."
                    )
            else:
                raise ValueError("Waveform must be 1-D, 2-D, or dict of 1-D arrays.")

        S = matrix.shape[0]
        if self.frequency <= 0.0:
            raise ValueError("frequency must be > 0 when using waveform mode.")
        sample_rate = self.frequency * S
        self._sample_period = 1.0 / sample_rate
        self._samples_per_cycle = S
        self._wf_matrix = matrix
        self._wf_matrix_T = matrix.T.copy(order="C")  # (C, S) once, for efficient DMA write

    # ---------- main loop ----------
    async def _run_loop(self) -> None:
        assert self._task is not None
        try:
            if self._wf_matrix is not None:
                await self._run_waveform_mode()
            else:
                await self._run_random_mode()
        finally:
            pass  # stop() handles zeroing and close

    # ---------------- random mode (unchanged semantics) ----------------
    async def _run_random_mode(self) -> None:
        while not self._stopping.is_set():
            values = self._rng.uniform(self.low, self.high, size=len(self._ao_ch_names)).tolist()
            self._task.write(values)
            if self.publish:
                ts = datetime.now().strftime(self.time_format)
                payload = {
                    "timestamp": ts,
                    "channel_values": {ch: float(v) for ch, v in zip(self._ao_ch_names, values)},
                }
                await self.publish(payload)
            await asyncio.sleep(self.interval)

    # ---------------- waveform mode: HW-timed with safe fallback ----------------
    async def _run_waveform_mode(self) -> None:
        """
        Preferred path: hardware-timed, regenerative AO; publish per-sample based on
        device sample counter 'n', with system-time timestamps in the same format.

        If a major exception occurs during initial setup/start, fall back to the
        original software-timed loop to preserve behavior.
        """
        try:
            await self._run_waveform_mode_hw()
        except Exception as e:
            # Fallback for "big exception happens in the beginning"
            # (e.g., misconfigured device, timing not supported, start failure, etc.)
            warn = f"[AsyncAORunner] HW-timed AO failed early ({type(e).__name__}: {e}). Falling back to software-timed mode."
            print(warn)
            await self._run_waveform_mode_software_fallback()

    async def _run_waveform_mode_hw(self) -> None:
        """
        Hardware-timed regenerative AO with per-sample publishing.
        Uses the device's total_samp_per_chan_generated to derive:
          - which waveform row was (just) output
          - the system-timestamp to publish with (using a monotonic-calibrated mapper)
        Adds lightweight alignment checks and respects self.cycles.
        """
        assert self._wf_matrix is not None and self._wf_matrix_T is not None
        assert self._sample_period is not None and self._samples_per_cycle is not None

        S = self._samples_per_cycle
        Fs_req = 1.0 / self._sample_period  # = frequency * S

        # Configure hardware-timed, continuous AO with regeneration
        self._task.timing.cfg_samp_clk_timing(
            rate=Fs_req,
            sample_mode=AcquisitionType.CONTINUOUS
        )
        self._task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

        # Pre-load one cycle and start (auto_start=False for glitch-free start)
        writer = AnalogMultiChannelWriter(self._task.out_stream, auto_start=False)
        writer.write_many_sample(self._wf_matrix_T, timeout=10.0)

        # Anchors and mapping (system clock + monotonic for drift)
        mapper_ready = False
        try:
            # Pin mapping to system time at start
            mapper = _SyncMapper(Fs_actual=Fs_req, time_format=self.time_format)
            self._task.start()
            # Query coerced rate (actual)
            Fs = float(self._task.timing.samp_clk_rate)
            # Update mapper slope to the actual rate immediately
            mapper.b_ns_per_sample = int(round((1.0 / Fs) * 1e9))
            mapper_ready = True
        except Exception:
            # Don't leave the task partially started
            raise

        # Lightweight alignment checks
        ppm_margin = 1000  # growth bound margin (0.1%)
        last_n = 0
        last_mono_ns = monotonic_ns()
        publishes_since_cal = 0
        # To avoid busy spinning while still being per-sample faithful, we poll frequently
        # and publish any samples that have been generated since the last poll.
        while not self._stopping.is_set():
            # How many samples have actually gone out?
            n = int(self._task.out_stream.total_samp_per_chan_generated)

            # --- Alignment checks ---
            if n < last_n:
                # Task likely restarted or error in driver; we won't attempt to "recover" silently.
                raise RuntimeError("AO sample counter decreased; task restart or device error.")
            mono_now_ns = monotonic_ns()
            dt = (mono_now_ns - last_mono_ns) / 1e9
            if dt > 0:
                max_expected = Fs * dt * (1 + ppm_margin / 1e6)
                if (n - last_n) > (max_expected + 10):  # +10 for coarse polling slack
                    print(f"[AsyncAORunner] AlignmentWarning: AO count grew faster than expected "
                          f"(Δn={n-last_n}, dt={dt:.6f}s, Fs={Fs:.3f}Hz).")

            # Periodically refine mapping slope vs monotonic
            publishes_since_cal += 1
            if publishes_since_cal >= 5:
                mapper.update(k_now=n)
                publishes_since_cal = 0
            last_mono_ns = mono_now_ns

            # Publish every new sample between last_n and n (per-sample behavior like original)
            if self.publish and n > last_n:
                # Cap per-iteration to avoid pathological burst (keeps loop responsive)
                # This does not drop samples; we just iterate this loop again quickly.
                batch_limit = max(1, int(0.02 * Fs))  # ~20 ms worth at most per tick
                target_n = n
                cursor = last_n
                while cursor < target_n and not self._stopping.is_set():
                    upper = min(target_n, cursor + batch_limit)
                    for k in range(cursor + 1, upper + 1):
                        row_idx = (k - 1) % S  # sample just generated
                        row = self._wf_matrix[row_idx, :]  # (C,)
                        ts = mapper.ts_str_for_k(k, self.time_format)
                        payload = {
                            "timestamp": ts,
                            "channel_values": {
                                ch: float(v) for ch, v in zip(self._ao_ch_names, row.tolist())
                            },
                        }
                        await self.publish(payload)
                    cursor = upper

            last_n = n

            # Respect finite cycles, if requested
            if self.cycles:
                cycles_done = n // S
                if cycles_done >= self.cycles:
                    break

            # Light sleep to avoid busy spin; bounded by a fraction of sample period
            # If Fs is very low, this is also fine (we'll publish per-sample as they appear).
            await asyncio.sleep(min(0.005, max(0.001, self._sample_period * 0.25)))

    async def _run_waveform_mode_software_fallback(self) -> None:
        """
        Original software-timed loop (your previous behavior), kept intact for fallback.
        """
        assert self._wf_matrix is not None and self._sample_period is not None and self._samples_per_cycle is not None
        S = self._samples_per_cycle
        sp = self._sample_period

        cycle_count = 0
        idx = 0
        while not self._stopping.is_set():
            row = self._wf_matrix[idx, :]  # shape (C,)
            # On-demand write (software-timed)
            self._task.write(row.tolist())

            if self.publish:
                ts = datetime.now().strftime(self.time_format)
                payload = {
                    "timestamp": ts,
                    "channel_values": {ch: float(v) for ch, v in zip(self._ao_ch_names, row.tolist())},
                }
                await self.publish(payload)

            idx += 1
            if idx >= S:
                idx = 0
                cycle_count += 1
                if self.cycles and cycle_count >= self.cycles:
                    break
            await asyncio.sleep(sp)

    # ---------- optional controls ----------
    async def set_frequency(self, frequency: float) -> None:
        """Update cycles-per-second for waveform mode (recompute sample period)."""
        if frequency <= 0:
            raise ValueError("frequency must be > 0.")
        self.frequency = float(frequency)
        if self._wf_matrix is not None:
            S = self._wf_matrix.shape[0]
            self._sample_period = 1.0 / (self.frequency * S)

    async def write_once(self, values: Dict[str, float]) -> None:
        """Immediate point write (holds until next scheduled write)."""
        if not self._task:
            raise RuntimeError("Runner not started.")
        ordered = [values.get(ch, 0.0) for ch in self._ao_ch_names]
        self._task.write(ordered)
