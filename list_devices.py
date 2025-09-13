"""List NI-DAQmx devices and their product types."""

from __future__ import annotations

from daqio.config import list_devices

try:  # pragma: no cover - nidaqmx may be unavailable
    from nidaqmx.system import System
except Exception:  # noqa: BLE001 - optional dependency
    System = None  # type: ignore[assignment]


def main() -> None:
    """Print each detected device name and its product type."""
    if System is None:
        print("nidaqmx not available.")
        return
    try:
        system = System.local()
    except Exception:  # noqa: BLE001 - best effort system access
        print("NI-DAQmx system unavailable.")
        return
    for name in list_devices():
        product_type = system.devices[name].product_type
        print(f"{name}: {product_type}")


if __name__ == "__main__":
    main()
