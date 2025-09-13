import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path so "daqio" package is importable even
# though this test module sits inside a directory named "tests/daqio" which
# would otherwise shadow the real package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daqio import publisher
from daqio.ao_runner import AsyncAORunner
from daqio.ai_reader import AIReader

try:  # Skip entire module if NI-DAQmx is unavailable
    from nidaqmx.system import System
    _system = System.local()
    _devices = list(_system.devices)
    if not _devices:
        raise RuntimeError("No NI-DAQmx devices detected")
    _device = _devices[0]
    if not _device.ao_physical_chans or not _device.ai_physical_chans:
        raise RuntimeError("Device lacks required AO/AI channels")
except Exception as e:  # pragma: no cover - skip if hardware missing
    pytest.skip(f"NI-DAQmx system unavailable: {e}", allow_module_level=True)


async def queue_printer(get_queue):
    q = get_queue()
    try:
        while True:
            item = await q.get()
            print(item)
            q.task_done()
    except asyncio.CancelledError:
        pass


def _make_reader_writer(pressures):
    dev_name = _device.name
    ao_ch = _device.ao_physical_chans[0].name
    ai_ch = _device.ai_physical_chans[0].name

    runner = AsyncAORunner(
        device=dev_name,
        channels=[ao_ch],
        waveform=pressures,
        frequency=0.1,
        publish=publisher.publish_ao,
    )
    reader = AIReader(
        device=dev_name,
        channels=[ai_ch],
        freq=0.1,
        averages=1,
        omissions=0,
        publish=publisher.publish_ai,
    )
    return runner, reader


async def ai_loop(reader):
    try:
        while True:
            await asyncio.to_thread(reader.read_once)
            await asyncio.sleep(10.0)
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_waveform_io():
    pressures = np.loadtxt(Path(__file__).resolve().parent / "daqio" / "apressure.csv")
    runner, reader = _make_reader_writer(pressures)

    with reader:
        async with runner:
            tasks = [
                asyncio.create_task(ai_loop(reader)),
                asyncio.create_task(queue_printer(publisher._get_ao_queue)),
                asyncio.create_task(queue_printer(publisher._get_ai_queue)),
            ]
            try:
                await asyncio.sleep(60.0)
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
