"""List NI-DAQmx device names and product types."""

from __future__ import annotations

from daqio.config import list_devices

try:  # pragma: no cover - nidaqmx is optional
    from nidaqmx.system import System
except Exception:  # noqa: BLE001 - best effort
    System = None  # type: ignore[assignment]


def main() -> None:
    """Print each detected NI-DAQmx device and its product type."""

    names = list_devices()
    if not names or System is None:
        print("No NI-DAQmx devices detected.")
        return

    try:
        system = System.local()
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"NI-DAQmx system unavailable: {exc}")
        return

    for device in system.devices:
        print(f"{device.name}: {device.product_type}")


if __name__ == "__main__":
    main()
