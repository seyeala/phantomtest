# daqio/ai_reader.py
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Callable, Awaitable

import numpy as np
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType  # CHANGED: add AcquisitionType

# Reuse your existing helpers
from .config import load_yaml, load_output_config

# Optional publisher hook type (same idea as your publish_ai)
PublishFn = Callable[[Dict[str, Any]], Awaitable[None]]

@dataclass
class AIConfig:
    device: str
    channels: List[str]
    freq: float              # Hz; interpreted as the device sample clock for read_average()
    averages: int            # number of kept samples contributing to the mean
    omissions: int           # number of skipped intervals between kept samples
    terminal: str = "RSE"

class AIReader:
    """
    NI-DAQmx Analog Input reader (object-oriented).

    - read_once(): On-demand immediate single sample (unchanged).
    - read_average(): Hardware-timed, buffered acquisition at 'freq' with optional
      decimation via 'omissions' for precise spacing and low jitter. Publishes once.

    Usage:
        reader = AIReader.from_yaml("configs/config_test.yml")
        with reader:  # opens an on-demand task for read_once(); read_average creates its own timed task
            channel_values, log = reader.read_average()
    """

    def __init__(
        self,
        device: str,
        channels: Iterable[str],
        *,
        freq: float,
        averages: int,
        omissions: int,
        terminal: str = "RSE",
        publish: Optional[PublishFn] = None,         # optional async publisher
        time_format: Optional[str] = None,           # strftime format; if None, taken from output YAML at read time
        output_config_path: Optional[str | Path] = None,  # e.g., configs/daqI_output.yml
    ) -> None:
        self.cfg = AIConfig(
            device=str(device),
            channels=list(channels),
            freq=float(freq),
            averages=int(averages),
            omissions=int(omissions),
            terminal=str(terminal),
        )
        self.publish = publish
        self.time_format = time_format
        self.output_config_path = Path(output_config_path) if output_config_path else None

        self._task: Optional[nidaqmx.Task] = None   # on-demand task for read_once()
        self._open = False

    # ---------- Construction helpers ----------
    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        section: str = "daqI",
        publish: Optional[PublishFn] = None,
        time_format: Optional[str] = None,
        output_config_path: Optional[str | Path] = None,
    ) -> "AIReader":
        data = load_yaml(path)
        if section not in data:
            raise ValueError(f"Expected '{section}' section in {path}")
        d = data[section]
        required = ("device", "channels", "freq", "averages", "omissions")
        missing = [k for k in required if d.get(k) in (None, [], "")]
        if missing:
            raise ValueError(f"Missing required keys in {section}: {', '.join(missing)}")
        return cls(
            d["device"],
            d["channels"],
            freq=d["freq"],
            averages=d["averages"],
            omissions=d["omissions"],
            terminal=d.get("terminal", "RSE"),
            publish=publish,
            time_format=time_format,
            output_config_path=output_config_path,
        )

    # ---------- Context management ----------
    def __enter__(self) -> "AIReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------- Lifecycle ----------
    def open(self) -> None:
        """
        Open an on-demand task for immediate reads (read_once).
        NOTE: read_average creates its own temporary hardware-timed task, so we
        do not attach timing here to preserve read_once() behavior.
        """
        if self._open:
            return
        term = TerminalConfiguration[self.cfg.terminal]
        t = nidaqmx.Task()
        for ch in self.cfg.channels:
            t.ai_channels.add_ai_voltage_chan(
                ch, min_val=-10.0, max_val=10.0, terminal_config=term
            )
        # No timing config -> on-demand
        self._task = t
        self._open = True

    def close(self) -> None:
        if self._task is not None:
            try:
                self._task.close()
            finally:
                self._task = None
        self._open = False

    # ---------- Acquisition ----------
    def read_once(self) -> Dict[str, float]:
        """
        Single immediate sample across all configured channels (on-demand).
        Returns {channel: value}.
        """
        if not self._open or self._task is None:
            raise RuntimeError("AIReader is not open. Use 'with reader:' or call open().")
        vals = self._task.read()
        # NI-DAQmx returns a scalar for single-channel, or list for multi
        if not isinstance(vals, list):
            vals = [vals]
        # Ensure values are plain Python floats
        result = {ch: float(val) for ch, val in zip(self.cfg.channels, vals)}

        # Optionally publish the single-shot result
        if self.publish:
            ts_format = self._resolve_time_format(use_output_yaml=False)
            ts_pub = datetime.now().strftime(ts_format)
            payload = {"timestamp": ts_pub, "channel_values": result}
            self._publish_now(payload)

        return result

    def read_average(
        self,
        *,
        use_output_yaml: bool = True,
    ) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Hardware-timed block acquisition with optional decimation.

        Behavior preserved:
          - Takes `averages` kept samples.
          - Spacing between kept samples matches previous software-timed logic:
                dt_keep = (omissions + 1) / freq
          - Prints one line per channel for each kept sample.
          - Returns ({channel: mean_voltage}, log_of_kept_samples)
          - Publishes once per batch if `publish` is set.

        Implementation:
          - Configure a temporary FINITE, hardware-timed task at `freq` (Hz).
          - For each kept sample, read (omissions + 1) samples per channel and
            keep the last sample from that mini-block (true "skip" semantics).
        """
        if self.cfg.freq <= 0:
            raise ValueError("freq must be > 0 for hardware-timed acquisition.")
        if self.cfg.averages <= 0:
            raise ValueError("averages must be a positive integer.")
        if self.cfg.omissions < 0:
            raise ValueError("omissions must be >= 0.")

        # Decide timestamp format for the final summary (payload), possibly from output YAML
        ts_format = self._resolve_time_format(use_output_yaml)

        base_rate = float(self.cfg.freq)             # device sample clock (Hz)
        stride = int(self.cfg.omissions) + 1         # samples per kept point
        total_samps = self.cfg.averages * stride     # samples per channel for the whole batch
        term = TerminalConfiguration[self.cfg.terminal]

        kept_rows: List[List[float]] = []            # shape: [averages, n_chan]
        log: List[Dict[str, Any]] = []
        n_chan = len(self.cfg.channels)

        # Timeout per mini-read: give generous leeway for very low rates
        per_block_timeout = max(10.0, 2.0 * stride / base_rate)
        total_timeout = max(10.0, 2.0 * total_samps / base_rate)

        # Create a dedicated, timed task so read_once() remains on-demand
        with nidaqmx.Task() as t:  # CHANGED: temporary hardware-timed task
            for ch in self.cfg.channels:
                t.ai_channels.add_ai_voltage_chan(
                    ch, min_val=-10.0, max_val=10.0, terminal_config=term
                )

            # Hardware timing: finite acquisition of the whole batch
            t.timing.cfg_samp_clk_timing(
                rate=base_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=total_samps,
            )

            # Explicit start avoids implicit auto-starts and makes timing clearer
            t.start()  # CHANGED: start timed acquisition

            # Read 'stride' samples at a time; keep the last of each block
            # This preserves your "omissions" semantics while using the HW clock
            read_so_far = 0
            for _ in range(self.cfg.averages):
                block = t.read(number_of_samples_per_channel=stride, timeout=per_block_timeout)

                # Normalize return shape:
                # - single channel: block is a list[float] of length=stride -> wrap to [list]
                # - multi channel:  block is list[list[float]] with shape [n_chan][stride]
                if n_chan == 1:
                    # last sample of the single channel
                    last_vals = [block[-1]]
                else:
                    last_vals = [block[ch_idx][-1] for ch_idx in range(n_chan)]

                ts_print = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                #for ch, val in zip(self.cfg.channels, last_vals):
                    #print(f"{ts_print} {ch}: {val:.6f} V")
                log.append(
                    {
                        "timestamp": ts_print,
                        "channel_values": {ch: float(v) for ch, v in zip(self.cfg.channels, last_vals)},
                    }
                )
                kept_rows.append(last_vals)
                read_so_far += stride

            # Drain any residual samples if caller asked for more reads than we consumed
            # (normally unnecessary because FINITE + exact total_samps keeps counts aligned)
            if read_so_far < total_samps:
                try:
                    _ = t.read(number_of_samples_per_channel=(total_samps - read_so_far),
                               timeout=total_timeout)
                except Exception:
                    # Non-fatal: acquisition likely already complete
                    pass

            # Task auto-stops on FINITE completion; context manager closes it

        arr = np.asarray(kept_rows, dtype=float)     # shape: [averages, n_chan]
        means = np.nanmean(arr, axis=0)
        channel_values = {ch: float(val) for ch, val in zip(self.cfg.channels, means)}
        #for ch, val in channel_values.items():
            #print(f"{ch}: {val:.6f} V")

        # Publish once per batch (optional)
        if self.publish:
            ts_pub = datetime.now().strftime(ts_format)
            payload = {"timestamp": ts_pub, "channel_values": channel_values}
            self._publish_now(payload)

        return channel_values, log

    # ---------- Utilities ----------
    def _resolve_time_format(self, use_output_yaml: bool) -> str:
        if self.time_format:
            return self.time_format
        if use_output_yaml:
            cfg_path = (
                self.output_config_path
                if self.output_config_path
                else Path(__file__).resolve().parent.parent / "configs" / "daqI_output.yml"
            )
            ts_format, _, _ = load_output_config(cfg_path)
            return ts_format
        # Fallback
        return "%Y-%m-%d %H:%M:%S.%f"

    def _publish_now(self, payload: Dict[str, Any]) -> None:
        if not self.publish:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.publish(payload))  # no loop: run ad-hoc
        else:
            loop.create_task(self.publish(payload))  # fire-and-forget inside a running loop
