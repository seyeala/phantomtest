import time
import argparse
import numpy as np
import nidaqmx
from nidaqmx.system import System
from nidaqmx.constants import TerminalConfiguration


def main(dev, freq, n_avg):
    # Find device
    system = System.local()
    device = next((d for d in system.devices if d.name == dev), None)
    if not device:
        print(f"Device {dev} not found.")
        return

    # Get AI channel *names*
    ai_channels = [ch.name for ch in device.ai_physical_chans]
    if not ai_channels:
        print(
            f"No analog input channels on {dev}. (This is expected for AO-only modules like NI-9263.)"
        )
        return

    print("Measuring channels:", ai_channels)
    sample_interval = 1.0 / float(freq)

    with nidaqmx.Task() as task:
        # Add all AI channels, set ±10 V and default to RSE wiring (adjust if you wired differential)
        for ch in ai_channels:
            task.ai_channels.add_ai_voltage_chan(
                ch, min_val=-10.0, max_val=10.0, terminal_config=TerminalConfiguration.RSE
            )

        while True:
            batch = []
            for _ in range(n_avg):
                vals = task.read()  # list of len = n_channels
                if not isinstance(vals, list):  # single-channel edge case
                    vals = [vals]
                batch.append(vals)
                time.sleep(sample_interval)

            arr = np.asarray(batch, dtype=float)  # shape (n_avg, n_channels)
            means = np.nanmean(arr, axis=0)
            print("Averaged:", ", ".join(f"{v:.4f} V" for v in means))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dev", required=True, help="Device name (e.g., Dev1, cDAQ1Mod1)")
    p.add_argument("--freq", type=float, required=True, help="Sample frequency in Hz")
    p.add_argument("--n", type=int, required=True, help="Number of samples to average")
    args = p.parse_args()
    main(args.dev, args.freq, args.n)
