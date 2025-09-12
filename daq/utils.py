"""Utility helpers for NI-DAQmx modules.

This module centralises small helper functions that are shared between
the :mod:`daq.daqI` and :mod:`daq.daqO` modules.  Placing them here keeps the
other modules focused on their core logic and avoids repetition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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


__all__ = ["load_yaml", "list_devices", "first_device"]

