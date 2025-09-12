from nidaqmx.system import System

system = System.local()
devices = list(system.devices)

if not devices:
    print("No NI‑DAQmx devices detected.")
else:
    for dev in devices:
        print(f"{dev.name}: {dev.product_type}")
