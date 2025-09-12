"""NI-DAQmx analog input helper.

This module provides a convenience wrapper for reading voltages from
multiple analog-input (AI) channels of a National Instruments device.
Configuration is supplied through a YAML file.

Example YAML configuration::

    device: Dev1
    channels:
      - Dev1/ai0
      - Dev1/ai1
    freq: 10          # sample frequency in Hz
    averages: 5       # number of samples to average
    terminal: RSE     # optional terminal configuration

Run the module from the command line::

    python -m daq.daqI --config config.yml

The script will acquire the requested number of samples from each
channel, compute the mean voltage per channel and print the results.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict

import nidaqmx
from nidaqmx.constants import TerminalConfiguration
import numpy as np

from .utils import load_yaml


@dataclass
class Config:
    """Configuration parameters for the analog-input task."""

    device: str
    channels: list[str]
    freq: float
    averages: int
    terminal: str = "RSE"


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from ``path``.

    The YAML file must define the device name, list of channel names,
    sample frequency and number of samples to average. Optionally a
    terminal configuration can be specified. Valid values are those
    accepted by :class:`nidaqmx.constants.TerminalConfiguration` (e.g.
    ``RSE``, ``DIFF``).

    Parameters
    ----------
    path:
        Path to a YAML configuration file.

    Returns
    -------
    dict
        Dictionary containing the parsed configuration with keys
        ``device``, ``channels``, ``freq``, ``averages`` and optional
        ``terminal``.
    """

    data = load_yaml(path)

    device = data.get("device")
    channels = data.get("channels")
    freq = data.get("freq")
    averages = data.get("averages")
    terminal = data.get("terminal", "RSE")

    if device is None or not channels or freq is None or averages is None:
        raise ValueError(
            "Config must include device, channels, freq and averages"
        )

    config = Config(
        device=str(device),
        channels=list(channels),
        freq=float(freq),
        averages=int(averages),
        terminal=str(terminal),
    )
    return config.__dict__


# ---------------------------------------------------------------------------
# Task setup
# ---------------------------------------------------------------------------


def setup_task(config: Dict[str, Any]) -> nidaqmx.Task:
    """Create and configure an analog-input task.

    Parameters
    ----------
    config:
        Configuration dictionary as returned by :func:`load_config`.

    Returns
    -------
    nidaqmx.Task
        A task with all AI channels added using the requested terminal
        configuration and a ±10 V range.
    """

    term = TerminalConfiguration[config["terminal"]]
    task = nidaqmx.Task()
    for ch in config["channels"]:
        task.ai_channels.add_ai_voltage_chan(
            ch, min_val=-10.0, max_val=10.0, terminal_config=term
        )
    return task


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def read_average(task: nidaqmx.Task, config: Dict[str, Any]) -> Dict[str, float]:
    """Acquire samples and compute the mean voltage per channel.

    Parameters
    ----------
    task:
        Configured :class:`nidaqmx.Task` containing AI channels.
    config:
        Configuration dictionary with ``freq`` and ``averages`` keys.

    Returns
    -------
    dict
        Mapping of channel names to mean voltages in volts.
    """

    sample_interval = 1.0 / config["freq"]
    batch: list[list[float]] = []
    for _ in range(config["averages"]):
        vals = task.read()
        if not isinstance(vals, list):
            vals = [vals]
        batch.append(vals)
        time.sleep(sample_interval)

    arr = np.asarray(batch, dtype=float)
    means = np.nanmean(arr, axis=0)

    results = dict(zip(config["channels"], means))
    for ch, val in results.items():
        print(f"{ch}: {val:.6f} V")
    return results


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Read NI-DAQmx analog inputs and report average voltage",
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    with setup_task(cfg) as task:
        read_average(task, cfg)


if __name__ == "__main__":  # pragma: no cover - manual use
    main()
