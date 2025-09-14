import asyncio
import os
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


def _check_nidaq_available():
    """
    Returns (True, payload) when NI-DAQmx + config look good.
    Returns (False, Exception) otherwise.
    """
    try:
        cfg = load_yaml(CONFIG_PATH)
        ao_cfg = cfg["daqO"]
        ai_cfg = cfg["daqI"]

        _ao_device = ao_cfg["device"]
        _ai_device = ai_cfg["device"]

        from nidaqmx.system import System
        system = System.local()
        devices = {dev.name: dev for dev in system.devices}

        if _ao_device not in devices or _ai_device not in devices:
            raise RuntimeError("Configured NI-DAQmx devices not detected")

        if (not devices[_ao_device].ao_physical_chans
                or not devices[_ai_device].ai_physical_chans):
            raise RuntimeError("Configured devices lack required AO/AI channels")

        return True, (cfg, devices)
    except Exception as e:
        return False, e


_is_ok, _payload = _check_nidaq_available()
if not _is_ok:
    # If running under pytest, skip; otherwise print a message and exit cleanly.
    if "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(f"NI-DAQmx system unavailable: {_payload}", allow_module_level=True)
    else:
        print(f"[test_waveform_io] NI-DAQmx system unavailable: {_payload}")
        sys.exit(0)
else:
    cfg, _devices = _payload
    ao_cfg = cfg["daqO"]
    ai_cfg = cfg["daqI"]
    _ao_device = ao_cfg["device"]
    _ao_channels = ao_cfg["channels"]
    _ai_device = ai_cfg["device"]
    _ai_channels = ai_cfg["channels"]


async def queue_printer(get_queue):
    q = get_queue()
    try:
        while True:
            item = await q.get()
            print(item)
            q.task_done()
    except asyncio.CancelledError:
        pass


# --- Local AO publish throttle to avoid console spam at high Fs ---
def _mk_ao_throttler(max_hz: float = 20.0):
    """
    Returns an async function that wraps publisher.publish_ao but only forwards
    at most 'max_hz' payloads per second (best-effort).
    """
    min_interval = 1.0 / max_hz if max_hz > 0 else 0.0
    last_ts = 0.0

    async def publish_ao_throttled(payload: dict):
        nonlocal last_ts
        now = asyncio.get_running_loop().time()
        if min_interval == 0.0 or (now - last_ts) >= min_interval:
            last_ts = now
            await publisher.publish_ao(payload)
        # else: drop this payload to keep output readable

    return publish_ao_throttled


def _make_reader_writer(pressures: np.ndarray):
    """
    Build AO runner + AI reader.

    AO waveform frequency (cycles/sec) is taken from YAML `daqO.waveform_cycles_hz`
    if present, otherwise defaults to 1.0 Hz. This decouples AO from AI.
    """
    ao_cycles_hz = float(1)

    runner = AsyncAORunner(
        device=_ao_device,
        channels=_ao_channels,
        waveform=pressures,
        waveform_cycles_hz=ao_cycles_hz,             # <-- independent knob
        publish=_mk_ao_throttler(max_hz=20.0),       # throttle AO publishes to ~20 Hz
    )

    reader = AIReader(
        **ai_cfg,
        publish=publisher.publish_ai,
    )
    return runner, reader


async def ai_loop(reader: AIReader, delay: float):
    """
    Drive AI reads according to config using the built-in averaging.
    """
    try:
        while True:
            # AIReader.read_average() handles its own (hardware) timing for the batch;
            # delay here just spaces out successive batches.
            await asyncio.to_thread(reader.read_average)
            await asyncio.sleep(delay)
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_waveform_io():
    pressures = np.loadtxt(Path(__file__).resolve().parent / "daqio" / "apressure.csv")
    runner, reader = _make_reader_writer(pressures)

    # Space AI batches roughly by its configured freq (simple heuristic):
    delay = max(0.0, 1.0 / float(ai_cfg["freq"]))

    with reader:
        async with runner:
            tasks = [
                asyncio.create_task(ai_loop(reader, delay)),
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
