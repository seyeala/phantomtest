# -*- coding: utf-8 -*-
"""
run_waveform_and_pico_capture.py
Orchestrates NI-DAQmx AI/AO and PicoScope captures with AI-suffixed filenames.
- Prefers multi-shot via your existing automation/capture_multi_shot.py (Option B).
- Falls back to one-shot via capture_single_shot.py if multi-shot is not provided.
- Starts background tasks that log AI and AO publications from daqio.publisher to CSV.

Requirements:
- Lives in repo 1 (PhantomTest)
- Can dynamically import repo 2 files by path (no need to install as a package)

Stop keys:
- Press ENTER (preferred) or Ctrl+C to stop the AI/AO run and shut down cleanly.

Outputs (all in one folder):
- <outdir>/ai_log.csv
- <outdir>/ao_log.csv
- <outdir>/<timestamp>__...__<AI values>.csv/.npz  (Pico captures; names include AI channel values)
"""
import argparse
import asyncio
import importlib.util
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# --- Make sure we can import the 'daqio' package from repo 1 root ---
REPO1_ROOT = Path(__file__).resolve().parents[1]  # adjust if you place this elsewhere
if str(REPO1_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO1_ROOT))

# --- Repo 1: helpers ---
from daqio import publisher
from daqio.ao_runner import AsyncAORunner
from daqio.ai_reader import AIReader
from daqio.config import load_yaml


# -----------------------
# Utility: check NI-DAQmx
# -----------------------
def _check_nidaq_available(cfg_path: Path):
    """
    Quick sanity check for NI-DAQmx presence & channels in config.
    Returns (ok: bool, payload_or_error).
    """
    try:
        cfg = load_yaml(cfg_path)
        ao_cfg = cfg["daqO"]
        ai_cfg = cfg["daqI"]

        from nidaqmx.system import System  # noqa
        system = System.local()
        devices = {dev.name: dev for dev in system.devices}

        ao_dev = ao_cfg["device"]
        ai_dev = ai_cfg["device"]

        if ao_dev not in devices or ai_dev not in devices:
            raise RuntimeError("Configured NI-DAQmx devices not detected")

        if (not devices[ao_dev].ao_physical_chans) or (not devices[ai_dev].ai_physical_chans):
            raise RuntimeError("Required AO/AI channels not present on configured devices")

        return True, (cfg, devices)
    except Exception as e:
        return False, e




# ----------------------------------
# Keypress listener (Enter to stop)
# ----------------------------------
def _start_enter_listener(stop_flag: threading.Event):
    def _reader():
        try:
            print("[keys] Press ENTER to stop...")
            input()
        except Exception:
            pass
        stop_flag.set()

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t


# ----------------------------------------------
# Dynamic import helpers (import by FILE PATH)
# ----------------------------------------------
def _load_module_from_path(module_name: str, file_path: Path):
    """Load a module from an explicit file path and register it under 'module_name'."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


async def capture_loop(args, outdir: Path, stop_evt: asyncio.Event) -> None:
    """Run Pico capture (multi-shot preferred) without blocking the event loop."""
    if stop_evt.is_set():
        return

    did_capture = False

    multi_py_path: Optional[Path] = None
    if args.multi_py:
        multi_py_path = Path(args.multi_py)
    elif args.capture_py:
        candidate = Path(args.capture_py).parent / "capture_multi_shot.py"
        if candidate.exists():
            multi_py_path = candidate

    if multi_py_path and multi_py_path.exists():
        print(f"== Step 3: Performing multi-shot Pico captures using: {multi_py_path}")
        try:
            if args.capture_py is None:
                raise RuntimeError("--capture-py must be provided when using --multi-py")
            single_mod = _load_module_from_path("capture_single_shot", Path(args.capture_py))
            print(f"[capture] single-shot module loaded from: {single_mod.__file__}")

            multi_mod = _load_module_from_path("capture_multi_shot", multi_py_path)
            print(f"[capture] multi-shot module loaded from: {multi_mod.__file__}")

            multi_cfg_path = Path(args.multi_config) if args.multi_config else (multi_py_path.parent / "capture_multi.yml")
            import yaml  # local import
            with open(multi_cfg_path, "r") as f:
                multi_cfg = yaml.safe_load(f)

            if args.captures is not None:
                multi_cfg["captures"] = int(args.captures)
            if args.rest_ms is not None:
                multi_cfg["rest_ms"] = float(args.rest_ms)
            if args.break_on_key:
                multi_cfg["break_on_key"] = True

            multi_cfg["timestamp_filenames"] = True
            multi_cfg["csv_path"] = str(outdir / "capture.csv")
            multi_cfg["numpy_path"] = str(outdir / "capture.npz")
            multi_cfg["daq_source"] = "ai"
            multi_cfg["name_embed"] = args.name_embed
            multi_cfg["name_maxlen"] = int(args.name_maxlen)

            if stop_evt.is_set():
                return
            await asyncio.to_thread(multi_mod.main, multi_cfg)
            did_capture = True
            print("[ok] Multi-shot capture phase complete.")
        except Exception as e:
            print(f"[warning] Multi-shot capture failed: {e}")

    if not did_capture and not stop_evt.is_set():
        if not args.capture_py:
            print("[info] No capture script provided; skipping Pico capture.")
        else:
            print("== Step 3: Performing single-shot Pico capture with AI-in-name...")
            try:
                cap_mod = _load_module_from_path("capture_single_shot_inproc", Path(args.capture_py))
                cap_cfg_path = Path(args.capture_config) if args.capture_config else (Path(args.capture_py).parent / "capture_config_test.yml")
                import yaml  # local import
                with open(cap_cfg_path, "r") as f:
                    cap_cfg = yaml.safe_load(f)

                cap_cfg["timestamp_filenames"] = True
                cap_cfg["csv_path"] = str(outdir / "capture.csv")
                cap_cfg["numpy_path"] = str(outdir / "capture.npz")
                cap_cfg["daq_source"] = "ai"
                cap_cfg["name_embed"] = args.name_embed
                cap_cfg["name_maxlen"] = int(args.name_maxlen)

                if stop_evt.is_set():
                    return
                await asyncio.to_thread(cap_mod.main, cap_cfg)
                print("[ok] Pico single-shot capture complete.")
            except Exception as e:
                print(f"[warning] Pico single-shot capture failed: {e}")


# ---------------
# Main async run
# ---------------
async def _run(args):
    print("== Step 1: Validating NI-DAQmx configuration...")
    ok, payload = _check_nidaq_available(Path(args.daq_config))
    if not ok:
        print(f"[warning] NI-DAQmx unavailable: {payload}")
        cfg = load_yaml(Path(args.daq_config))
        ai_cfg = cfg.get("daqI", {})
        ao_cfg = cfg.get("daqO", {})
    else:
        cfg, _devices = payload
        ai_cfg = cfg["daqI"]
        ao_cfg = cfg["daqO"]

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ai_csv = str(outdir / "ai_log.csv")
    ao_csv = str(outdir / "ao_log.csv")

    pressures: Optional[np.ndarray]
    if args.waveform_csv and Path(args.waveform_csv).exists():
        pressures = np.loadtxt(args.waveform_csv)
        print(f"Loaded AO waveform from {args.waveform_csv} with {len(pressures)} samples")
    else:
        n = 2000
        pressures = (np.sin(np.linspace(0, 2 * np.pi, n)) * 2.0).astype(np.float32)
        print("No waveform CSV found; using a synthetic sine waveform.")

    print("== Step 2: Taking pre-capture AI snapshot (for filename embedding)...")
    try:
        with AIReader(**ai_cfg, publish=publisher.publish_ai) as reader:
            await asyncio.to_thread(reader.read_average)
        print("[ok] AI snapshot published.")
    except Exception as e:
        print(f"[warning] Could not take AI snapshot before capture: {e}")

    print("== Step 3: Starting AI/AO CSV consumer tasks...")
    ai_columns = ["timestamp"] + [str(ch) for ch in ai_cfg.get("channels", [])]
    ao_columns = ["timestamp"] + [str(ch) for ch in ao_cfg.get("channels", [])]
    ai_csv_task = publisher.start_ai_consumer(ai_csv, ai_columns)
    ao_csv_task = publisher.start_ao_consumer(ao_csv, ao_columns)
    print(f"[ok] CSV consumers started: {ai_csv} | {ao_csv}")

    print("== Step 4: Starting capture and AI/AO run...")
    stop_evt = asyncio.Event()

    def _sigint_handler(*_):
        if not stop_evt.is_set():
            print("\n[signal] Ctrl+C received; stopping...")
            stop_evt.set()

    signal.signal(signal.SIGINT, _sigint_handler)
    try:
        signal.signal(signal.SIGTERM, _sigint_handler)
    except Exception:
        pass

    key_flag = threading.Event()
    _start_enter_listener(key_flag)

    ao_cycles_hz = float(args.ao_cycles_hz)
    runner = AsyncAORunner(
        device=ao_cfg["device"],
        channels=ao_cfg["channels"],
        waveform=pressures,
        waveform_cycles_hz=ao_cycles_hz,
        publish=publisher.publish_ao,
    )
    reader = AIReader(**ai_cfg, publish=publisher.publish_ai)

    ai_delay = max(0.0, 1.0 / float(ai_cfg.get("freq", 10)))

    async def _ai_loop():
        with reader:
            try:
                while not stop_evt.is_set():
                    await asyncio.to_thread(reader.read_average)
                    await asyncio.sleep(ai_delay)
            except asyncio.CancelledError:
                pass

    async def _ao_loop():
        async with runner:
            try:
                while not stop_evt.is_set():
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass

    ai_task = asyncio.create_task(_ai_loop())
    ao_task = asyncio.create_task(_ao_loop())
    capture_task = asyncio.create_task(capture_loop(args, outdir, stop_evt))

    tasks = [ai_task, ao_task, capture_task, ai_csv_task, ao_csv_task]

    print("[run] AI/AO active. Press ENTER or Ctrl+C to stop.")
    if args.duration > 0:
        try:
            await asyncio.wait_for(asyncio.to_thread(key_flag.wait), timeout=args.duration)
        except asyncio.TimeoutError:
            print(f"[run] Duration {args.duration}s elapsed; stopping...")
            stop_evt.set()
    else:
        await asyncio.to_thread(key_flag.wait)
        stop_evt.set()

    print("== Step 5: Stopping tasks...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    print("[ok] Shutdown complete. All files in:", outdir)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI/AO runner with multi-shot Pico capture and CSV loggers (Option B)")
    p.add_argument("--daq-config", default=str(REPO1_ROOT / "configs" / "config_test.yml"),
                   help="Path to repo1 YAML for NI-DAQmx AI/AO")

    # --- Multi-shot (preferred) ---
    p.add_argument("--multi-py", default="",
                   help="Path to repo2 automation/capture_multi_shot.py (preferred). "
                        "If omitted, will look next to --capture-py; if not found, falls back to single-shot.")
    p.add_argument("--multi-config", default="",
                   help="Path to capture_multi.yml (defaults to alongside --multi-py)")
    p.add_argument("--captures", type=int, default=None,
                   help="Override number of captures for multi-shot")
    p.add_argument("--rest-ms", type=float, default=None,
                   help="Override rest between captures in milliseconds")
    p.add_argument("--break-on-key", action="store_true",
                   help="Stop early if a key is pressed during the rest period (multi-shot only)")

    # --- Single-shot (fallback) ---
    p.add_argument("--capture-py", default="",
                   help="Path to repo2 automation/capture_single_shot.py (fallback to one-shot)")
    p.add_argument("--capture-config", default="",
                   help="Path to capture_config_test.yml (defaults to alongside --capture-py)")

    # --- Common ---
    p.add_argument("--outdir", default=str(REPO1_ROOT / "runs"),
                   help="Output directory for all files (logs + capture)")
    p.add_argument("--waveform-csv", default=str(REPO1_ROOT / "tests" / "daqio" / "apressure.csv"),
                   help="CSV/ASCII file with one column waveform for AO (fallback: sine wave)")
    p.add_argument("--ao-cycles-hz", type=float, default=1.0,
                   help="Waveform cycles per second for AO replay")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Run time in seconds (0 → until ENTER)")
    p.add_argument("--name-embed", choices=["full", "mini", "none"], default="full",
                   help="How much AI info to embed in Pico filename")
    p.add_argument("--name-maxlen", type=int, default=160,
                   help="Max length for the DAQ suffix in filenames")
    return p


def main():
    parser = _build_argparser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
