import importlib

modules = [
    "labjack",
    "labjack.ljm",
    "mcculw",
    "nidaqmx",
]

for name in modules:
    try:
        importlib.import_module(name)
        status = "[OK]"
    except Exception:
        status = "[FAIL]"
    print(f"{name}: {status}")
