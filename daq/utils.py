"""Utility helpers for NI-DAQmx modules.

This module centralises small helper functions that are shared between
the :mod:`daq.daqI` and :mod:`daq.daqO` modules.  Placing them here keeps the
other modules focused on their core logic and avoids repetition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import argparse
import yaml

try:  # pragma: no cover - nidaqmx might not be installed during tests
    from nidaqmx.system import System
except Exception:  # noqa: BLE001 - optional dependency
    System = None  # type: ignore[assignment]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file from ``path``.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed YAML content.  If the file is empty an empty dictionary is
        returned instead.
    """

    with open(Path(path), "r", encoding="utf-8") as fh:
        data: Dict[str, Any] | None = yaml.safe_load(fh)
    return data or {}


def list_devices() -> List[str]:
    """Return names of detected NI-DAQmx devices.

    The function attempts to query the local NI-DAQmx system for connected
    devices.  If the :mod:`nidaqmx` package is unavailable or an error occurs
    during discovery an empty list is returned.
    """

    if System is None:
        return []

    try:
        system = System.local()
        return [dev.name for dev in system.devices]
    except Exception:  # noqa: BLE001 - best effort device discovery
        return []


def first_device() -> Optional[str]:
    """Return the first detected NI-DAQmx device name if available."""

    devices = list_devices()
    return devices[0] if devices else None


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    default_config_path: str | None = None,
    *,
    section: str | None = None,
    argv: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Parse command-line arguments with optional YAML defaults.

    Parameters
    ----------
    parser:
        Argument parser instance with all module-specific options already
        added (except ``--config`` which is handled here).
    default_config_path:
        Path to a default YAML configuration file.  May be ``None`` if no
        default is desired.
    section:
        Optional top-level section within the YAML file to load.  This allows
        a single configuration file to host multiple module configurations.
    argv:
        Optional explicit argument list for testing.  If ``None`` the
        arguments are taken from :data:`sys.argv`.

    Returns
    -------
    dict
        Dictionary containing the merged configuration values.  Any
        command-line options override values loaded from the configuration
        file.
    """

    parser.add_argument(
        "--config",
        default=default_config_path,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args(argv)

    cfg: Dict[str, Any] = {}
    config_path = getattr(args, "config", None)
    if config_path:
        data = load_yaml(config_path)
        if section:
            cfg.update(data.get(section, {}))
        else:
            cfg.update(data)

    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        cfg[key] = value

    return cfg


__all__ = [
    "load_yaml",
    "list_devices",
    "first_device",
    "parse_args_with_config",
]

