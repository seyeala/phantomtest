"""NI-DAQmx analog input helper.

This module provides a convenience wrapper for reading voltages from
multiple analog-input (AI) channels of a National Instruments device.
Configuration is supplied through a YAML file. The YAML data must contain
a dedicated ``daqI`` section for AI channels only. Analog outputs belong
in a separate ``daqO`` section; mixing them may raise NI-DAQmx ``I/O type``
errors or drive unintended channels.

Example YAML configuration::

    device: Dev1
    channels:
      - Dev1/ai0
      - Dev1/ai1
    freq: 10          # sample frequency in Hz
    averages: 5       # number of samples to average
    omissions: 0      # number of sample intervals to skip between reads
    terminal: RSE     # optional terminal configuration

Run the module from the command line::

    python -m daqio.daqI --config configs/config_test.yml

The script will acquire the requested number of samples from each
channel, compute the mean voltage per channel and print the channel values.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import nidaqmx
from nidaqmx.constants import TerminalConfiguration
import numpy as np

from .config import (
    load_yaml,
    parse_args_with_config as _parse_args_with_config,
    load_output_config,
)
from .publisher import publish_ai


@dataclass
class Config:
    """Configuration parameters for the analog-input task."""

    device: str
    channels: list[str]
    freq: float
    averages: int
    omissions: int
    terminal: str = "RSE"


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_config(data_or_path: Dict[str, Any] | str) -> Dict[str, Any]:
    """Load configuration from ``data_or_path``.

    The argument may either be a path to a YAML file or a dictionary with
    configuration values.  The configuration must define the device name,
    list of channel names, sample frequency, number of samples to average and
    the number of sample intervals to skip between reads.  Optionally a
    terminal configuration can be specified.  Valid values are those accepted by
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
        ``device``, ``channels``, ``freq``, ``averages``, ``omissions`` and
        optional ``terminal``.
    """

    if isinstance(data_or_path, str):
        data = load_yaml(data_or_path)
    else:
        data = dict(data_or_path)

    device = data.get("device")
    channels = data.get("channels")
    freq = data.get("freq")
    averages = data.get("averages")
    omissions = data.get("omissions")
    terminal = data.get("terminal", "RSE")

    if (
        device is None
        or not channels
        or freq is None
        or averages is None
        or omissions is None
    ):
        raise ValueError(
            "Config must include device, channels, freq, averages and omissions"
        )

    config = Config(
        device=str(device),
        channels=list(channels),
        freq=float(freq),
        averages=int(averages),
        omissions=int(omissions),
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
    parser.add_argument(
        "--omissions",
        type=int,
        help="Number of sample intervals to skip between reads",
    )
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


def read_average(
    task: nidaqmx.Task, config: Dict[str, Any], output_config: str | None = None
) -> tuple[Dict[str, float], list[dict[str, Any]]]:
    """Acquire samples and compute the mean voltage per channel.

    Parameters
    ----------
    task:
        Configured :class:`nidaqmx.Task` containing AI channels.
    config:
        Configuration dictionary with ``freq``, ``averages`` and ``omissions`` keys.

    Returns
    -------
    tuple of (dict, list of dict)
        Mapping of channel names to mean voltages in volts and a list of
        timestamped sample readings for optional post-processing.
    """

    cfg_path = (
        Path(output_config)
        if output_config
        else Path(__file__).resolve().parent.parent / "configs" / "daqI_output.yml"
    )
    ts_format, _, _ = load_output_config(cfg_path)

    sample_interval = 1.0 / config["freq"]
    batch: list[list[float]] = []
    log: list[dict[str, Any]] = []
    for _ in range(config["averages"]):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        vals = task.read()
        if not isinstance(vals, list):
            vals = [vals]
        for ch, val in zip(config["channels"], vals):
            print(f"{ts} {ch}: {val:.6f} V")
        log.append({"timestamp": ts, "values": dict(zip(config["channels"], vals))})
        batch.append(vals)
        time.sleep(sample_interval * (config["omissions"] + 1))

    arr = np.asarray(batch, dtype=float)
    means = np.nanmean(arr, axis=0)

    channel_values = dict(zip(config["channels"], means))
    for ch, val in channel_values.items():
        print(f"{ch}: {val:.6f} V")

    ts = datetime.now().strftime(ts_format)
    payload = {"timestamp": ts, "channel_values": channel_values}
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(publish_ai(payload))
    else:
        loop.create_task(publish_ai(payload))

    return channel_values, log


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
