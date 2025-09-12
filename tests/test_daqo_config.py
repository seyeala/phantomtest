"""Tests for :func:`daqio.daqO.load_config`."""

from __future__ import annotations

from pathlib import Path

from daqio.daqO import load_config


def _write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(content)
    return path


def test_load_config_flat(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
device: Dev1
channels:
  - Dev1/ao0
  - Dev1/ao1
interval: 0.5
low: -5
high: 5
seed: 123
""",
    )
    cfg = load_config(path)
    assert cfg["device"] == "Dev1"
    assert cfg["channels"] == ["Dev1/ao0", "Dev1/ao1"]
    assert cfg["interval"] == 0.5
    assert cfg["low"] == -5
    assert cfg["high"] == 5
    assert cfg["seed"] == 123


def test_load_config_nested(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
daqO:
  device: Dev1
  channels:
    - Dev1/ao0
    - Dev1/ao1
  interval: 0.5
  low: -5
  high: 5
  seed: 123
""",
    )
    cfg = load_config(path)
    assert cfg["device"] == "Dev1"
    assert cfg["channels"] == ["Dev1/ao0", "Dev1/ao1"]
    assert cfg["interval"] == 0.5
    assert cfg["low"] == -5
    assert cfg["high"] == 5
    assert cfg["seed"] == 123
