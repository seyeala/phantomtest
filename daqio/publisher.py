import asyncio
from pathlib import Path
import csv
from typing import Dict, List

_ao_queue: asyncio.Queue | None = None
_ai_queue: asyncio.Queue | None = None


def _get_ao_queue() -> asyncio.Queue:
    """Return the singleton queue for analog-output messages."""
    global _ao_queue
    if _ao_queue is None:
        _ao_queue = asyncio.Queue()
    return _ao_queue


def _get_ai_queue() -> asyncio.Queue:
    """Return the singleton queue for analog-input messages."""
    global _ai_queue
    if _ai_queue is None:
        _ai_queue = asyncio.Queue()
    return _ai_queue


async def publish_ao(data: Dict) -> None:
    """Publish analog-output data to the queue."""
    await _get_ao_queue().put(data)


async def publish_ai(result: Dict) -> None:
    """Publish analog-input averaging results to the queue."""
    await _get_ai_queue().put(result)


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
                    row[col] = item.get("results", {}).get(col)
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
