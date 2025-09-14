"""Random analog-output helper for NI-DAQmx.

This module provides utilities for driving one or more analog-output (AO)
channels with random voltages. Configuration may be supplied either
programmatically or via a YAML file. The configuration must come from a
``daqO`` section; reusing ``daqI`` entries for outputs can trigger NI-DAQmx
``I/O type`` errors or energize unintended channels. A small command-line
interface is also provided::

    python -m daqio.daqO --config configs/config_test.yml

Safety
------
All outputs are reset to ``0 V`` when the loop exits, even when the user
interrupts the program with ``Ctrl+C``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import nidaqmx
from nidaqmx.system import System

from .config import load_yaml, load_output_config
from .publisher import publish_ao, start_ao_consumer


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load analog-output configuration from ``path``.

    The YAML file may contain the following keys, either at the top level or
    nested under a ``daqO`` section:

    ``device`` (str)
        NI-DAQmx device name, e.g. ``Dev1`` or ``cDAQ1Mod1``.
    ``channels`` (sequence of str, optional)
        Explicit AO channel names.  If omitted, all AO channels on the
        device are used.
    ``interval`` (float)
        Seconds to wait between updates.
    ``low`` / ``high`` (float)
        Voltage bounds for the generated random numbers.
    ``seed`` (int, optional)
        Seed for the random-number generator.

    Parameters
    ----------
    path:
        Path to a YAML file containing the configuration.

    Returns
    -------
    dict
        Dictionary with keys suitable for :func:`write_random`.
    """

    data = load_yaml(path)
    cfg = data
    required = {"device", "interval", "low", "high"}
    if not required.issubset(cfg.keys()) and "daqO" in data:
        cfg = data["daqO"]
    return {
        "device": cfg.get("device"),
        "channels": cfg.get("channels"),
        "interval": cfg.get("interval"),
        "low": cfg.get("low"),
        "high": cfg.get("high"),
        "seed": cfg.get("seed"),
    }


# ---------------------------------------------------------------------------
# Random output loop
# ---------------------------------------------------------------------------

async def write_random(
    dev: str,
    interval: float,
    low: float,
    high: float,
    seed: Optional[int] = None,
    channels: Optional[Iterable[str]] = None,
    output_config: Optional[str] = None,
) -> None:
    """Continuously write random voltages to analog outputs.

    Parameters
    ----------
    dev:
        NI-DAQmx device name, e.g. ``Dev1``.
    interval:
        Seconds between updates.
    low, high:
        Voltage bounds for the random numbers.
    seed:
        Seed for the random-number generator.  If ``None``, a random seed
        is used.
    channels:
        Optional iterable of channel names.  If ``None``, all AO channels
        on ``dev`` are used.

    Notes
    -----
    For safety, all outputs are reset to ``0 V`` when the function exits.
    """

    rng = np.random.default_rng(seed)

    system = System.local()
    device = next((d for d in system.devices if d.name == dev), None)
    if device is None:
        raise ValueError(f"Device {dev} not found.")

    if channels is None:
        ao_channels: List[str] = [ch.name for ch in device.ao_physical_chans]
        if not ao_channels:
            raise ValueError(f"No analog OUTPUT channels found on {dev}.")
    else:
        ao_channels = list(channels)

    print(f"Will drive AO channels on {dev}: {', '.join(ao_channels)}")
    print(
        f"Random range: [{low:.3f}, {high:.3f}] V, update every {interval:.3f} s",
    )
    print("Press Ctrl+C to stop.\n")

    cfg_path = (
        Path(output_config)
        if output_config
        else Path(__file__).resolve().parent.parent / "configs" / "daqO_output.yml"
    )
    ts_format, csv_path, columns = load_output_config(cfg_path)
    if columns == ["timestamp"]:
        columns = ["timestamp", *ao_channels]

    consumer_task = start_ao_consumer(csv_path, columns)

    with nidaqmx.Task() as task:
        for ch in ao_channels:
            task.ao_channels.add_ao_voltage_chan(
                ch,
                min_val=min(low, -10.0),
                max_val=max(high, 10.0),
            )
        try:
            while True:
                values = rng.uniform(low, high, size=len(ao_channels)).tolist()
                task.write(values)
                ts = datetime.now().strftime(ts_format)
                payload = {
                    "timestamp": ts,
                    # Ensure downstream consumers receive plain Python floats
                    "channel_values": {c: float(v) for c, v in zip(ao_channels, values)},
                }
                await publish_ao(payload)
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped. Setting outputs to 0 V for safety.")
            try:
                task.write([0.0] * len(ao_channels))
            except Exception:
                pass
        finally:
            consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer_task


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write random voltages to NI-DAQmx analog outputs",
    )
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--dev", help="Device name (e.g., Dev1)")
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Explicit channel list (default: all AO channels)",
    )
    parser.add_argument("--interval", type=float, help="Seconds between updates")
    parser.add_argument("--low", type=float, help="Low end of random range in volts")
    parser.add_argument("--high", type=float, help="High end of random range in volts")
    parser.add_argument("--seed", type=int, help="RNG seed (optional)")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point for ``python -m daqio.daqO``."""

    args = _parse_args(argv)

    cfg = {}
    if args.config:
        cfg.update(load_config(args.config))

    # Command-line flags override config file
    for key in ["device", "channels", "interval", "low", "high", "seed"]:
        arg_val = getattr(args, key if key != "device" else "dev")
        if arg_val is not None:
            cfg[key] = arg_val

    required = ["device", "interval", "low", "high"]
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(f"Missing required configuration: {missing_str}")

    asyncio.run(
        write_random(
            cfg["device"],
            float(cfg["interval"]),
            float(cfg["low"]),
            float(cfg["high"]),
            seed=None if cfg.get("seed") in (None, "") else int(cfg["seed"]),
            channels=cfg.get("channels"),
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI use
    main()
