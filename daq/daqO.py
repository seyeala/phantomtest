"""NI-DAQmx analog output helper.

This module provides a small utility for writing random voltages to
multiple analog-output (AO) channels of a National Instruments device.
Configuration is supplied through a YAML file.

Example YAML configuration::

    device: Dev1
    channels:
      - Dev1/ao0
      - Dev1/ao1
    range: [0.0, 3.0]   # [low, high] in volts
    interval: 0.5       # seconds between updates
    seed: 1234          # optional RNG seed

Run the module from the command line::

    python -m daq.daqO --config config.yml

This will continuously write random voltages between the specified low
and high values to all listed channels until interrupted with
``Ctrl+C``.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Any, Dict

import nidaqmx

from .utils import load_yaml


@dataclass
class Config:
    """Configuration parameters for the analog-output task."""

    device: str
    channels: list[str]
    low: float
    high: float
    interval: float
    seed: int | None = None


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from ``path``.

    The YAML file must define the device name, list of channel names,
    voltage range and update interval.  Optionally a random-number seed
    can be provided.

    Parameters
    ----------
    path:
        Path to a YAML configuration file.

    Returns
    -------
    dict
        Dictionary containing the parsed configuration with keys
        ``device``, ``channels``, ``low``, ``high``, ``interval`` and
        optional ``seed``.
    """

    data = load_yaml(path)

    device = data.get("device")
    channels = data.get("channels")
    interval = data.get("interval")
    seed = data.get("seed")

    # Range may be given as [low, high] or as explicit keys
    if "range" in data:
        try:
            low, high = data["range"]
        except Exception as exc:  # noqa: BLE001 - simple validation
            raise ValueError("range must be a sequence of two numbers") from exc
    else:
        low = data.get("low")
        high = data.get("high")

    if (
        device is None
        or not channels
        or low is None
        or high is None
        or interval is None
    ):
        raise ValueError(
            "Config must include device, channels, range (low/high) and interval"
        )

    config = Config(
        device=str(device),
        channels=list(channels),
        low=float(low),
        high=float(high),
        interval=float(interval),
        seed=int(seed) if seed is not None else None,
    )
    return config.__dict__


# ---------------------------------------------------------------------------
# Task setup
# ---------------------------------------------------------------------------

def setup_task(config: Dict[str, Any]) -> nidaqmx.Task:
    """Create and configure an analog-output task.

    Parameters
    ----------
    config:
        Configuration dictionary as returned by :func:`load_config`.

    Returns
    -------
    nidaqmx.Task
        A task with all AO channels added and voltage limits set.
    """

    task = nidaqmx.Task()
    low, high = config["low"], config["high"]
    for ch in config["channels"]:
        task.ao_channels.add_ao_voltage_chan(ch, min_val=low, max_val=high)
    return task


# ---------------------------------------------------------------------------
# Output loop
# ---------------------------------------------------------------------------

def write_random(task: nidaqmx.Task, config: Dict[str, Any]) -> None:
    """Continuously write random voltages to all channels.

    The function runs indefinitely until interrupted with ``Ctrl+C``.  It
    will attempt to set all outputs to 0 V when stopping for safety.

    Parameters
    ----------
    task:
        Configured :class:`nidaqmx.Task` containing AO channels.
    config:
        Configuration dictionary with ``low``, ``high``, ``interval`` and
        optional ``seed`` keys.
    """

    rng = random.Random(config.get("seed"))
    low, high = config["low"], config["high"]
    interval = config["interval"]
    n_channels = len(config["channels"])

    try:
        while True:
            values = [rng.uniform(low, high) for _ in range(n_channels)]
            task.write(values)
            time.sleep(interval)
    except KeyboardInterrupt:
        try:
            task.write([0.0] * n_channels)
        except Exception:  # noqa: BLE001 - best effort to reset outputs
            pass


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Write random voltages to NI-DAQmx analog outputs",
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    with setup_task(cfg) as task:
        write_random(task, cfg)


if __name__ == "__main__":  # pragma: no cover - manual use
    main()
