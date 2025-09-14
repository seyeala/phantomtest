# daqio/ao_runner.py
from __future__ import annotations
import asyncio
from typing import Iterable, Optional, Callable, Awaitable, List, Dict, Union
from datetime import datetime

import numpy as np
import nidaqmx
from nidaqmx.system import System

PublishFn = Callable[[Dict], Awaitable[None]]

ArrayLike = Union[List[float], np.ndarray]
PerChanWave = Union[ArrayLike, Dict[str, ArrayLike], np.ndarray]  # (S,), (S,C), or {ch: (S,)}

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
        low: float = 0.0,
        high: float = 1.0,
        seed: Optional[int] = None,
        # Waveform mode params
        waveform: Optional[PerChanWave] = None,
        frequency: float = 1.0,      # cycles per second (complete waveform repeats per second)
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
        self.frequency = float(frequency)
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
        self._wf_matrix: Optional[np.ndarray] = None   # shape: (S, C)
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
        # Convert waveform into matrix (S, C)
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

    async def _run_random_mode(self) -> None:
        while not self._stopping.is_set():
            values = self._rng.uniform(self.low, self.high, size=len(self._ao_ch_names)).tolist()
            self._task.write(values)
            if self.publish:
                ts = datetime.now().strftime(self.time_format)
                # Cast each value to a native Python float for downstream consumers
                payload = {
                    "timestamp": ts,
                    "channel_values": {
                        ch: float(v) for ch, v in zip(self._ao_ch_names, values)
                    },
                }
                await self.publish(payload)
            await asyncio.sleep(self.interval)

    async def _run_waveform_mode(self) -> None:
        assert self._wf_matrix is not None and self._sample_period is not None and self._samples_per_cycle is not None
        S = self._samples_per_cycle
        sp = self._sample_period

        cycle_count = 0
        idx = 0
        while not self._stopping.is_set():
            row = self._wf_matrix[idx, :]  # shape (C,)
            self._task.write(row.tolist())

            if self.publish:
                ts = datetime.now().strftime(self.time_format)
                # Convert per-channel samples to native floats before publishing
                payload = {
                    "timestamp": ts,
                    "channel_values": {
                        ch: float(v) for ch, v in zip(self._ao_ch_names, row.tolist())
                    },
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
