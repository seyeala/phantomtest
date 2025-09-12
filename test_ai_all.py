import nidaqmx
from nidaqmx.system import System
import time
import numpy as np
import argparse

def main(dev, freq, n_avg):
    # Discover channels on the selected device
    system = System.local()
    device = next((d for d in system.devices if d.name == dev), None)
    if not device:
        print(f"Device {dev} not found.")
        return

    ai_channels = [f"{dev}/{ch}" for ch in device.ai_physical_chans]
    if not ai_channels:
        print(f"No analog input channels found on {dev}.")
        return

    print(f"Measuring channels: {ai_channels}")
    sample_interval = 1.0 / freq

    with nidaqmx.Task() as task:
        for ch in ai_channels:
            task.ai_channels.add_ai_voltage_chan(ch, min_val=-10.0, max_val=10.0)

        while True:
            samples = []
            for _ in range(n_avg):
                samples.append(task.read())
                time.sleep(sample_interval)

            arr = np.array(samples)  # shape (n_avg, n_channels)
            mean_vals = np.mean(arr, axis=0)
            print("Averaged:", ", ".join(f"{v:.4f} V" for v in mean_vals))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=str, required=True, help="Device name, e.g. Dev1 or cDAQ1Mod1")
    parser.add_argument("--freq", type=float, required=True, help="Sample frequency in Hz")
    parser.add_argument("--n", type=int, required=True, help="Number of samples to average")
    args = parser.parse_args()

    main(args.dev, args.freq, args.n)
