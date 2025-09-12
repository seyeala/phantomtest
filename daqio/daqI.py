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

    python -m daqio.daqI --config configs/config_test.yml

The script will acquire the requested number of samples from each
channel, compute the mean voltage per channel and print the results.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import nidaqmx
from nidaqmx.constants import TerminalConfiguration
import numpy as np

from .config import load_yaml, parse_args_with_config as _parse_args_with_config


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


def load_config(data_or_path: Dict[str, Any] | str) -> Dict[str, Any]:
    """Load configuration from ``data_or_path``.

    The argument may either be a path to a YAML file or a dictionary with
    configuration values.  The configuration must define the device name,
    list of channel names, sample frequency and number of samples to
    average.  Optionally a terminal configuration can be specified.  Valid
    values are those accepted by
    :class:`nidaqmx.constants.TerminalConfiguration` (e.g. ``RSE``, ``DIFF``).

    Parameters
    ----------
    data_or_path:
        Either a mapping of configuration values or a path to a YAML file
        containing them.

    Returns
    -------
    dict
        Dictionary containing the parsed configuration with keys
        ``device``, ``channels``, ``freq``, ``averages`` and optional
        ``terminal``.
    """

    if isinstance(data_or_path, str):
        data = load_yaml(data_or_path)
    else:
        data = dict(data_or_path)

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


def parse_args_with_config(
    default_config_path: str | None = None,
    argv: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Parse CLI arguments and optional YAML configuration.

    Parameters
    ----------
    default_config_path:
        Optional path to a default configuration file.  Users may override
        it with ``--config`` on the command line.
    argv:
        Optional argument sequence.  If ``None`` the arguments are read from
        :data:`sys.argv`.

    Returns
    -------
    dict
        Configuration dictionary ready for use by the module.
    """

    parser = argparse.ArgumentParser(
        description="Read NI-DAQmx analog inputs and report average voltage",
    )
    parser.add_argument("--device", help="Device name (e.g., Dev1)")
    parser.add_argument("--channels", nargs="+", help="Analog input channels")
    parser.add_argument("--freq", type=float, help="Sample frequency in Hz")
    parser.add_argument("--averages", type=int, help="Number of samples to average")
    parser.add_argument("--terminal", help="Terminal configuration")

    raw_cfg = _parse_args_with_config(
        parser, default_config_path, section="daqI", argv=argv
    )
    return load_config(raw_cfg)


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


def main(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Command-line entry point."""

    cfg = parse_args_with_config("configs/config_test.yml", argv=argv)
    with setup_task(cfg) as task:
        read_average(task, cfg)
    return cfg


if __name__ == "__main__":  # pragma: no cover - manual use
    main()
