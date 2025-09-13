import argparse
import asyncio
from daqio.daqO import write_random


def main(dev, interval_s, low, high, seed, channels=None):
    """Wrapper used by the original script.

    All heavy lifting now lives in :func:`daqio.daqO.write_random` which is
    reused by the command-line helper as well as tests.
    """

    asyncio.run(
        write_random(dev, interval_s, low, high, seed=seed, channels=channels)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Write random voltages (0–3 V by default) to all AO channels.",
    )
    p.add_argument("--dev", required=True, help="Device name (e.g., Dev1, cDAQ1Mod1)")
    p.add_argument(
        "--interval", type=float, required=True, help="Seconds between updates (e.g., 1.0)",
    )
    p.add_argument(
        "--low", type=float, default=0.0, help="Low end of random range in volts (default 0.0)",
    )
    p.add_argument(
        "--high", type=float, default=3.0, help="High end of random range in volts (default 3.0)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="RNG seed (optional, for reproducibility)",
    )
    p.add_argument(
        "--channels",
        nargs="+",
        help="Explicit channel list; defaults to all AO channels on the device",
    )
    args = p.parse_args()
    main(
        args.dev,
        args.interval,
        args.low,
        args.high,
        args.seed,
        channels=args.channels,
    )
