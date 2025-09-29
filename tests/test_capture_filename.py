from datetime import datetime
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture_filename import parse_capture_filename


@pytest.mark.parametrize(
    "filename, expected_ts, expected_channels",
    [
        (
            "M09-D19-H09-M47-S43-U.098__2025-09-19-09-47-43.057497__ai1_3p698__ai2_5p005__ai3_3p789.npz",
            datetime(2025, 9, 19, 9, 47, 43, 57497),
            {"ai1": 3.698, "ai2": 5.005, "ai3": 3.789},
        ),
        (
            "prefix__2024-01-02-03-04-05.6__ai0_-1p500__ai12_0p000.csv",
            datetime(2024, 1, 2, 3, 4, 5, 600000),
            {"ai0": -1.5, "ai12": 0.0},
        ),
        (
            "abc__2023-07-08-09-10-11__ai1_3.npz",
            datetime(2023, 7, 8, 9, 10, 11),
            {"ai1": 3.0},
        ),
    ],
)
def test_parse_capture_filename(filename, expected_ts, expected_channels):
    metadata = parse_capture_filename(filename)
    assert metadata.timestamp == expected_ts
    assert metadata.channels == expected_channels


def test_parse_capture_filename_invalid():
    with pytest.raises(ValueError):
        parse_capture_filename("not_a_capture_file.txt")
