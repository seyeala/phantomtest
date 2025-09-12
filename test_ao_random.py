import argparse
import time
import numpy as np
import nidaqmx
from nidaqmx.system import System

def main(dev, interval_s, low, high, seed):
    rng = np.random.default_rng(seed)

    # find device
    system = System.local()
    device = next((d for d in system.devices if d.name == dev), None)
    if not device:
        print(f"Device {dev} not found.")
        return

    # list AO channels (names only, not repr)
    ao_channels = [ch.name for ch in device.ao_physical_chans]
    if not ao_channels:
        print(f"No analog OUTPUT channels found on {dev}. "
              f"(E.g., NI-9263 has AO, USB-6009 has AO0..1; some devices may have none.)")
        return

    print(f"Will drive AO channels on {dev}: {', '.join(ao_channels)}")
    print(f"Random range: [{low:.3f}, {high:.3f}] V, update every {interval_s:.3f} s")
    print("Press Ctrl+C to stop.\n")

    # Use a single task with all AOs. Set a wide range that works on most boards.
    # (9263 supports ±10 V; USB-6009 AO is 0–5 V—writing 0–3 V is safe for both.)
    with nidaqmx.Task() as task:
        for ch in ao_channels:
            # Wide ±10 V range usually accepted by NI devices; we only write 0–3 V.
            task.ao_channels.add_ao_voltage_chan(ch, min_val=-10.0, max_val=10.0)

        # Loop: generate randoms and write vector to all channels
        try:
            while True:
                values = rng.uniform(low, high, size=len(ao_channels)).tolist()
                task.write(values)  # writes one sample per channel
                # pretty print
                line = " | ".join(f"{ch.split('/')[-1]}={v:5.3f} V" for ch, v in zip(ao_channels, values))
                print(f"{time.strftime('%H:%M:%S')}  {line}")
                time.sleep(interval_s)
        except KeyboardInterrupt:
            print("\nStopped. Setting outputs to 0 V for safety.")
            try:
                task.write([0.0]*len(ao_channels))
            except Exception:
                pass

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Write random voltages (0–3 V by default) to all AO channels.")
    p.add_argument("--dev", required=True, help="Device name (e.g., Dev1, cDAQ1Mod1)")
    p.add_argument("--interval", type=float, required=True, help="Seconds between updates (e.g., 1.0)")
    p.add_argument("--low", type=float, default=0.0, help="Low end of random range in volts (default 0.0)")
    p.add_argument("--high", type=float, default=3.0, help="High end of random range in volts (default 3.0)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (optional, for reproducibility)")
    args = p.parse_args()
    main(args.dev, args.interval, args.low, args.high, args.seed)
