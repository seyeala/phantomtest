import pytest

try:
    from nidaqmx.system import System
except Exception as e:
    pytest.skip(f"nidaqmx not available: {e}", allow_module_level=True)


def test_device_names_and_types():
    try:
        system = System.local()
        devices = list(system.devices)
    except Exception as e:
        pytest.skip(f"NI-DAQmx system unavailable: {e}")
    if not devices:
        pytest.skip("No NI-DAQmx devices detected.")
    for device in devices:
        assert device.name
        assert device.product_type
