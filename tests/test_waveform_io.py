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
from daqio.config import load_yaml

CONFIG_PATH = ROOT / "configs" / "config_test.yml"

try:  # Skip entire module if NI-DAQmx is unavailable or config incomplete
    cfg = load_yaml(CONFIG_PATH)
    ao_cfg = cfg["daqO"]
    ai_cfg = cfg["daqI"]
    _ao_device = ao_cfg["device"]
    _ao_channels = ao_cfg["channels"]
    _ai_device = ai_cfg["device"]
    _ai_channels = ai_cfg["channels"]

    from nidaqmx.system import System

    system = System.local()
    devices = {dev.name: dev for dev in system.devices}
    if _ao_device not in devices or _ai_device not in devices:
        raise RuntimeError("Configured NI-DAQmx devices not detected")
    if not devices[_ao_device].ao_physical_chans or not devices[_ai_device].ai_physical_chans:
        raise RuntimeError("Configured devices lack required AO/AI channels")
except Exception as e:  # pragma: no cover - skip if hardware or config missing
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


def _make_reader_writer(pressures, ao_device, ao_channels, ai_device, ai_channels):
    runner = AsyncAORunner(
        device=ao_device,
        channels=ao_channels,
        waveform=pressures,
        frequency=0.1,
        publish=publisher.publish_ao,
    )
    reader = AIReader(
        device=ai_device,
        channels=ai_channels,
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
    runner, reader = _make_reader_writer(
        pressures, _ao_device, _ao_channels, _ai_device, _ai_channels
    )

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


if __name__ == "__main__":
    asyncio.run(test_waveform_io())
