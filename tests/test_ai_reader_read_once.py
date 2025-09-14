"""Tests for :mod:`daqio.ai_reader` single-shot acquisition publishing."""

from __future__ import annotations

from daqio.ai_reader import AIReader


class DummyTask:
    """Minimal task stub returning pre-defined values."""

    def __init__(self, values):
        self._values = values

    def read(self):  # noqa: D401 - simple stub
        return self._values


def test_read_once_publish():
    captured: dict = {}

    async def fake_publish(data):
        captured["data"] = data

    reader = AIReader(
        device="Dev1",
        channels=["c1", "c2"],
        freq=1.0,
        averages=1,
        omissions=0,
        publish=fake_publish,
    )
    reader._task = DummyTask([1.0, 2.0])  # type: ignore[assignment]
    reader._open = True

    results = reader.read_once()

    assert results == {"c1": 1.0, "c2": 2.0}
    assert captured["data"]["results"] == {"c1": 1.0, "c2": 2.0}
    assert captured["data"]["timestamp"]

