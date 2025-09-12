import importlib
import pytest

MODULES = [
    "labjack",
    "labjack.ljm",
    "mcculw",
    "nidaqmx",
]

@pytest.mark.parametrize("name", MODULES)
def test_daq_module_available(name):
    try:
        module = importlib.import_module(name)
    except Exception as e:
        pytest.skip(f"{name} not available: {e}")
    assert module is not None
