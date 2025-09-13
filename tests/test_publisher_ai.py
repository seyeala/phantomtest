import asyncio
import contextlib
from pathlib import Path

import pytest

from daqio import publisher
import daqio.daqI as daqI
from daqio.daqI import read_average


async def _run_consumer(tmp_path: Path):
    csv_file = tmp_path / "ai.csv"
    columns = ["timestamp", "c1", "c2"]
    task = publisher.start_ai_consumer(str(csv_file), columns)
    payload = {"timestamp": "t0", "results": {"c1": 1, "c2": 2}}
    await publisher.publish_ai(payload)
    await publisher._get_ai_queue().join()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return csv_file


def test_ai_consumer_writes_csv(tmp_path):
    csv_file = asyncio.run(_run_consumer(tmp_path))
    lines = csv_file.read_text().splitlines()
    assert lines[0] == "timestamp,c1,c2"
    assert lines[1] == "t0,1,2"


class DummyTask:
    def __init__(self, values):
        self._values = values

    def read(self):
        return self._values


def test_read_average_publish(monkeypatch):
    captured = {}

    async def fake_publish(data):
        captured["data"] = data

    monkeypatch.setattr(daqI, "publish_ai", fake_publish)
    monkeypatch.setattr(daqI, "load_yaml", lambda path: {"timestamp_format": "%Y"})
    monkeypatch.setattr(daqI.time, "sleep", lambda s: None)
    cfg = {"freq": 1.0, "averages": 1, "channels": ["c1", "c2"]}
    task = DummyTask([1.0, 2.0])
    read_average(task, cfg)
    assert captured["data"]["results"] == {"c1": 1.0, "c2": 2.0}
    assert captured["data"]["timestamp"].isdigit()
