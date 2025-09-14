"""Manage queues and CSV writers for analog I/O.

Public API: publish_ao, publish_ai, start_ao_consumer, start_ai_consumer,
            get_latest_ao, get_latest_ai.
"""
import asyncio
from pathlib import Path
import csv
import os
from typing import Dict, List, Optional

# ---- queue sizing (avoid unbounded growth) ----
_MAX = int(os.getenv("DAQIO_QUEUE_MAX", "1024"))

_ao_queue: Optional[asyncio.Queue] = None
_ai_queue: Optional[asyncio.Queue] = None

# ---- latest snapshots for instant access (same process) ----
_latest_ai: Optional[Dict] = None
_latest_ao: Optional[Dict] = None


def _get_ao_queue() -> asyncio.Queue:
    """Return the singleton queue for analog-output messages."""
    global _ao_queue
    if _ao_queue is None:
        _ao_queue = asyncio.Queue(maxsize=_MAX)
    return _ao_queue


def _get_ai_queue() -> asyncio.Queue:
    """Return the singleton queue for analog-input messages."""
    global _ai_queue
    if _ai_queue is None:
        _ai_queue = asyncio.Queue(maxsize=_MAX)
    return _ai_queue


def get_latest_ai() -> Dict | None:
    """Return the most recent AI payload (do not mutate)."""
    return _latest_ai


def get_latest_ao() -> Dict | None:
    """Return the most recent AO payload (do not mutate)."""
    return _latest_ao


async def _put_drop_oldest(q: asyncio.Queue, item: Dict) -> None:
    """Non-blocking put with drop-oldest if queue is full (keeps DAQ loops from stalling)."""
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        try:
            _ = q.get_nowait()
            q.task_done()
        except asyncio.QueueEmpty:
            pass
        q.put_nowait(item)


async def publish_ao(data: Dict) -> None:
    """Publish analog-output data to the queue and update latest snapshot."""
    global _latest_ao
    _latest_ao = data
    await _put_drop_oldest(_get_ao_queue(), data)


async def publish_ai(data: Dict) -> None:
    """Publish analog-input channel values to the queue and update latest snapshot."""
    global _latest_ai
    _latest_ai = data
    await _put_drop_oldest(_get_ai_queue(), data)


async def _ao_consumer(csv_path: str, columns: List[str]) -> None:
    queue = _get_ao_queue()
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        while True:
            item = await queue.get()
            row = {}
            for col in columns:
                if col == "timestamp":
                    row[col] = item.get("timestamp")
                else:
                    row[col] = item.get("channel_values", {}).get(col)
            writer.writerow(row)
            fh.flush()
            queue.task_done()


async def _ai_consumer(csv_path: str, columns: List[str]) -> None:
    queue = _get_ai_queue()
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        while True:
            item = await queue.get()
            row = {}
            for col in columns:
                if col == "timestamp":
                    row[col] = item.get("timestamp")
                else:
                    row[col] = item.get("channel_values", {}).get(col)
            writer.writerow(row)
            fh.flush()
            queue.task_done()


def start_ao_consumer(csv_path: str, columns: List[str]) -> asyncio.Task:
    """Start background task writing queue entries to CSV."""
    loop = asyncio.get_running_loop()
    return loop.create_task(_ao_consumer(csv_path, columns))


def start_ai_consumer(csv_path: str, columns: List[str]) -> asyncio.Task:
    """Start background task writing AI queue entries to CSV."""
    loop = asyncio.get_running_loop()
    return loop.create_task(_ai_consumer(csv_path, columns))
