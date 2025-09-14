# daqio/ai_reader.py
from __future__ import annotations
import time
import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Callable, Awaitable

import numpy as np
import nidaqmx
from nidaqmx.constants import TerminalConfiguration

# Reuse your existing helpers
from .config import load_yaml, load_output_config
# Optional publisher hook type (same idea as your publish_ai)
PublishFn = Callable[[Dict[str, Any]], Awaitable[None]]

@dataclass
class AIConfig:
    device: str
    channels: List[str]
    freq: float
    averages: int
    omissions: int
    terminal: str = "RSE"

class AIReader:
    """
    NI-DAQmx Analog Input reader (object-oriented).

    - One-shot batch acquisition with averaging (like your current script).
    - No internal consumer/logger.
    - Optional async `publish` hook called once per batch.
    - Keeps I/O type separation (AI only).
    - Published payloads share the same schema as analog output:
      {"timestamp": str, "channel_values": {channel: value}}

    Usage:
        reader = AIReader.from_yaml("configs/config_test.yml")
        with reader:  # opens task
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

        self._task: Optional[nidaqmx.Task] = None
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
        if self._open:
            return
        term = TerminalConfiguration[self.cfg.terminal]
        t = nidaqmx.Task()
        for ch in self.cfg.channels:
            t.ai_channels.add_ai_voltage_chan(
                ch, min_val=-10.0, max_val=10.0, terminal_config=term
            )
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
    def read_once(self) -> Dict[str, Any]:
        """
        Single immediate sample across all configured channels.
        Returns the same payload schema used for publishing and AO:
        {"timestamp": ts, "channel_values": {channel: value}}.
        """
        if not self._open or self._task is None:
            raise RuntimeError("AIReader is not open. Use 'with reader:' or call open().")
        vals = self._task.read()
        if not isinstance(vals, list):
            vals = [vals]
        channel_values = dict(zip(self.cfg.channels, vals))
        ts_format = self._resolve_time_format(use_output_yaml=False)
        ts_pub = datetime.now().strftime(ts_format)
        payload = {"timestamp": ts_pub, "channel_values": channel_values}

        if self.publish:
            self._publish_now(payload)

        return payload

    def read_average(
        self,
        *,
        use_output_yaml: bool = True,
    ) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Collect `averages` samples at 1/freq, skipping `omissions` intervals between reads.
        Prints each read with a high-res timestamp.
        Returns:
            channel_values: {channel: mean_voltage}
            log:     [{"timestamp": ts, "values": {...}}, ...] for each read
        Also publishes a summary payload once (if `publish` set).
        """
        if not self._open or self._task is None:
            raise RuntimeError("AIReader is not open. Use 'with reader:' or call open().")

        # Decide timestamp format for the final summary (payload), possibly from output YAML
        ts_format = self._resolve_time_format(use_output_yaml)

        sample_interval = 1.0 / self.cfg.freq
        log: List[Dict[str, Any]] = []
        batch: List[List[float]] = []

        for _ in range(self.cfg.averages):
            ts_print = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            vals = self._task.read()
            if not isinstance(vals, list):
                vals = [vals]
            for ch, val in zip(self.cfg.channels, vals):
                print(f"{ts_print} {ch}: {val:.6f} V")
            log.append({"timestamp": ts_print, "values": dict(zip(self.cfg.channels, vals))})
            batch.append(vals)
            time.sleep(sample_interval * (self.cfg.omissions + 1))

        arr = np.asarray(batch, dtype=float)
        means = np.nanmean(arr, axis=0)
        channel_values = dict(zip(self.cfg.channels, means))
        for ch, val in channel_values.items():
            print(f"{ch}: {val:.6f} V")

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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.publish(payload))  # no loop: run ad-hoc
        else:
            loop.create_task(self.publish(payload))  # fire-and-forget inside a running loop
