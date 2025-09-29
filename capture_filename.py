"""Utilities for parsing Pico capture filenames."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

__all__ = ["CaptureFilenameMetadata", "parse_capture_filename"]


@dataclass(frozen=True)
class CaptureFilenameMetadata:
    """Structured information parsed from a capture filename.

    Attributes
    ----------
    timestamp:
        The hardware timestamp embedded in the filename.
    channels:
        Mapping of channel names to their recorded floating point values.
    stem:
        The stem portion of the filename (without extension).
    extension:
        The file extension including the leading dot.
    """

    timestamp: datetime
    channels: Dict[str, float]
    stem: str
    extension: str


_CAPTURE_FILENAME_RE = re.compile(
    r"""
    ^
    (?P<stem>.+?)
    __
    (?P<hwts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:\.\d+)?)
    (?P<channel_suffix>(?:__[A-Za-z0-9]+_-?[0-9]+(?:p[0-9]+)?)*)
    (?P<extension>\.[A-Za-z0-9]+)
    $
    """,
    re.VERBOSE,
)

_CHANNEL_RE = re.compile(r"__([A-Za-z0-9]+)_(-?[0-9]+(?:p[0-9]+)?)")


def _value_to_float(raw: str) -> float:
    """Convert a DAQ channel value token into a float."""
    normalized = raw.replace("p", ".")
    try:
        return float(normalized)
    except ValueError as exc:  # pragma: no cover - defensive, should not happen with regex
        raise ValueError(f"Invalid channel value token: {raw!r}") from exc


def parse_capture_filename(filename: str | Path) -> CaptureFilenameMetadata:
    """Parse a capture filename and extract timestamp/channel metadata.

    Parameters
    ----------
    filename:
        Filename or path to parse. The parent directories are ignored; only the
        final path component is inspected.

    Returns
    -------
    CaptureFilenameMetadata
        Parsed timestamp and channel mapping information.

    Raises
    ------
    ValueError
        If the filename does not conform to the expected capture pattern.
    """

    name = Path(filename).name
    match = _CAPTURE_FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Unrecognised capture filename format: {name!r}")

    hwts = match.group("hwts")
    timestamp: datetime
    for fmt in ("%Y-%m-%d-%H-%M-%S.%f", "%Y-%m-%d-%H-%M-%S"):
        try:
            timestamp = datetime.strptime(hwts, fmt)
            break
        except ValueError:
            continue
    else:  # pragma: no cover - regex should enforce correctness
        raise ValueError(f"Invalid timestamp token in capture filename: {hwts!r}")

    channel_suffix = match.group("channel_suffix")
    channels: Dict[str, float] = {}
    if channel_suffix:
        for channel_match in _CHANNEL_RE.finditer(channel_suffix):
            channel, raw_value = channel_match.groups()
            channels[channel] = _value_to_float(raw_value)

    stem = f"{match.group('stem')}__{hwts}"
    extension = match.group("extension")
    return CaptureFilenameMetadata(
        timestamp=timestamp,
        channels=channels,
        stem=stem,
        extension=extension,
    )
