# -*- coding: utf-8 -*-
"""
run_waveform_and_pico_capture.py
Orchestrates NI-DAQmx AI/AO and a one-shot PicoScope capture with AI-suffixed filenames.
Also starts two separate processes that log AI and AO publications from daqio.publisher to CSV.

Requirements:
- Lives in repo 1 (PhantomTest)
- Dynamically imports 'capture_single_shot.py' from repo 2 (automation folder) in-process

Stop keys:
- Press ENTER (preferred) or Ctrl+C to stop the AI/AO run and shut down cleanly.

Outputs (all in one folder):
- <outdir>/ai_log.csv
- <outdir>/ao_log.csv
- <outdir>/<timestamp>__...__<AI values>.csv   (PicoScope capture, if CSV enabled in its config)
- <outdir>/<timestamp>__...__<AI values>.npz   (PicoScope capture, if NumPy enabled in its config)
"""

import argparse
import asyncio
import csv
import importlib.util
import multiprocessing as mp
import os
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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

# -------------------------
# CSV writer process target
# -------------------------
def _csv_writer_loop(kind: str, q: mp.Queue, out_csv: str):
    """
    Separate process: consumes flattened rows and appends to CSV.
    kind: "AI" or "AO"
    q: multiprocessing.Queue with dict rows: {"timestamp": str, "device": str, "channel": str, "value": float}
       Sentinel: None → clean exit
    """
    try:
        first = True
        with open(out_csv, "a", newline="") as f:
            w = csv.writer(f)
            while True:
                item = q.get()
                if item is None:
                    break
                # Ensure minimal schema
                ts = item.get("timestamp", "")
                dev = item.get("device", "")
                ch  = item.get("channel", "")
                val = item.get("value", "")
                if first:
                    w.writerow(["source", "timestamp", "device", "channel", "value"])
                    first = False
                w.writerow([kind, ts, dev, ch, val])
    except Exception as e:
        # Last resort: print to stderr. Parent process can't catch this easily.
        print(f"[{kind} CSV writer] Error: {e}", file=sys.stderr)

# ---------------------------------------
# Async forwarders: async-queue → mp-queue
# ---------------------------------------
async def _forward_ai(async_get_queue, out_q: mp.Queue, stop: asyncio.Event, default_device: str):
    q = async_get_queue()
    try:
        while not stop.is_set():
            payload = await q.get()
            # Flatten AI payload: results is a dict channel->value
            results = payload.get("results") if isinstance(payload, dict) else None
            ts = payload.get("timestamp") if isinstance(payload, dict) else None
            dev = payload.get("device") if isinstance(payload, dict) else None
            if isinstance(results, dict):
                iso = ts if isinstance(ts, str) else datetime.utcnow().isoformat()
                devname = dev or default_device or ""
                for ch, val in results.items():
                    out_q.put({"timestamp": iso, "device": devname, "channel": str(ch), "value": float(val)})
            q.task_done()
    except asyncio.CancelledError:
        pass

async def _forward_ao(async_get_queue, out_q: mp.Queue, stop: asyncio.Event, default_device: str):
    q = async_get_queue()
    try:
        while not stop.is_set():
            payload = await q.get()
            # Flatten AO payload: channel_values is a dict channel->value
            chvals = payload.get("channel_values") if isinstance(payload, dict) else None
            ts = payload.get("timestamp") if isinstance(payload, dict) else None
            dev = payload.get("device") if isinstance(payload, dict) else None
            if isinstance(chvals, dict):
                iso = ts if isinstance(ts, str) else datetime.utcnow().isoformat()
                devname = dev or default_device or ""
                for ch, val in chvals.items():
                    out_q.put({"timestamp": iso, "device": devname, "channel": str(ch), "value": float(val)})
            q.task_done()
    except asyncio.CancelledError:
        pass

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
# Dynamic import of repo2's capture_single_shot.py
# ----------------------------------------------
def _load_capture_module(capture_py_path: Path):
    spec = importlib.util.spec_from_file_location("capture_single_shot_inproc", str(capture_py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {capture_py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["capture_single_shot_inproc"] = mod
    spec.loader.exec_module(mod)
    return mod  # must expose: build_argparser/apply_overrides/main OR directly 'main'

# ---------------
# Main async run
# ---------------
async def _run(args):
    print("== Step 1: Validating NI-DAQmx configuration...")
    ok, payload = _check_nidaq_available(Path(args.daq_config))
    if not ok:
        print(f"[warning] NI-DAQmx unavailable: {payload}")
        # You may still want to attempt Pico capture; continue.
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

    # Load waveform for AO
    pressures: Optional[np.ndarray] = None
    if args.waveform_csv and Path(args.waveform_csv).exists():
        pressures = np.loadtxt(args.waveform_csv)
        print(f"Loaded AO waveform from {args.waveform_csv} with {len(pressures)} samples")
    else:
        # Fallback: small sine wave if file not present
        n = 2000
        pressures = (np.sin(np.linspace(0, 2 * np.pi, n)) * 2.0).astype(np.float32)  # +/-2 V
        print("No waveform CSV found; using a synthetic sine waveform.")

    # -------------------------------
    # Step 2: Pre-capture AI snapshot
    # -------------------------------
    # We do a one-off AI read and publish so capture_single_shot can embed it.
    print("== Step 2: Taking pre-capture AI snapshot (for filename embedding)...")
    try:
        with AIReader(**ai_cfg, publish=publisher.publish_ai) as reader:
            # read_average is blocking; does its own HW timing and publishes the result
            await asyncio.to_thread(reader.read_average)
        print("[ok] AI snapshot published.")
    except Exception as e:
        print(f"[warning] Could not take AI snapshot before capture: {e}")

    # ----------------------------------
    # Step 3: Pico single-shot capture
    # ----------------------------------
    if args.capture_py is None:
        print("[info] --capture-py not provided; skipping Pico capture.")
    else:
        print("== Step 3: Performing PicoScope single-shot capture with AI-in-name...")
        try:
            cap_mod = _load_capture_module(Path(args.capture_py))

            # Resolve capture config path (default next to capture script if not provided)
            cap_cfg_path = Path(args.capture_config) if args.capture_config else (Path(args.capture_py).parent / "capture_config_test.yml")
            with open(cap_cfg_path, "r") as f:
                import yaml  # local import to avoid global dependency if unused
                cap_cfg = yaml.safe_load(f)

            # Force outputs into our outdir + ensure timestamped filenames + AI embedding
            cap_cfg["timestamp_filenames"] = True
            cap_cfg["csv_path"] = str(outdir / "capture.csv")
            cap_cfg["numpy_path"] = str(outdir / "capture.npz")
            cap_cfg["daq_source"] = "ai"
            cap_cfg["name_embed"] = args.name_embed  # "full" (recommended) or "mini"
            cap_cfg["name_maxlen"] = int(args.name_maxlen)

            # Run capture in-process (so it can see daqio.publisher latest AI)
            cap_mod.main(cap_cfg)
            print("[ok] Pico capture complete.")
        except Exception as e:
            print(f"[warning] Pico capture failed: {e}")

    # -----------------------------------------
    # Step 4: Start AI/AO logging to CSV (fork)
    # -----------------------------------------
    print("== Step 4: Starting AI/AO CSV writer processes...")
    mp_ctx = mp.get_context("spawn")  # Windows-safe
    ai_q: mp.Queue = mp_ctx.Queue(maxsize=2048)
    ao_q: mp.Queue = mp_ctx.Queue(maxsize=2048)

    ai_proc = mp_ctx.Process(target=_csv_writer_loop, args=("AI", ai_q, ai_csv), daemon=True)
    ao_proc = mp_ctx.Process(target=_csv_writer_loop, args=("AO", ao_q, ao_csv), daemon=True)
    ai_proc.start()
    ao_proc.start()
    print(f"[ok] CSV writers started: {ai_csv} | {ao_csv}")

    # ---------------------------------------------
    # Step 5: Start forwarders and AI/AO run itself
    # ---------------------------------------------
    print("== Step 5: Starting AI/AO run...")
    stop_evt = asyncio.Event()

    # graceful Ctrl+C → stop_evt
    def _sigint_handler(*_):
        if not stop_evt.is_set():
            print("\n[signal] Ctrl+C received; stopping...")
            stop_evt.set()
    signal.signal(signal.SIGINT, _sigint_handler)
    # SIGTERM too (when available)
    try:
        signal.signal(signal.SIGTERM, _sigint_handler)
    except Exception:
        pass

    # Also accept ENTER to stop:
    key_flag = threading.Event()
    _start_enter_listener(key_flag)

    # Create AO runner + AI reader
    ao_cycles_hz = float(args.ao_cycles_hz)
    runner = AsyncAORunner(
        device=ao_cfg["device"],
        channels=ao_cfg["channels"],
        waveform=pressures,
        waveform_cycles_hz=ao_cycles_hz,
        publish=publisher.publish_ao,  # we'll throttle in CSV stage instead of here
    )
    reader = AIReader(**ai_cfg, publish=publisher.publish_ai)

    # Forward daqio.publisher's async queues into mp queues
    forward_ai_task = asyncio.create_task(_forward_ai(publisher._get_ai_queue, ai_q, stop_evt, ai_cfg.get("device", "")))
    forward_ao_task = asyncio.create_task(_forward_ao(publisher._get_ao_queue, ao_q, stop_evt, ao_cfg.get("device", "")))

    # heuristic spacing between AI batches
    ai_delay = max(0.0, 1.0 / float(ai_cfg.get("freq", 10)))

    async def _ai_loop():
        with reader:
            try:
                while not stop_evt.is_set():
                    await asyncio.to_thread(reader.read_average)  # blocking call off-thread
                    await asyncio.sleep(ai_delay)
            except asyncio.CancelledError:
                pass

    async def _ao_loop():
        async with runner:
            try:
                while not stop_evt.is_set():
                    # AO runner drives waveform internally (hardware-timed) and publishes values
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass

    ai_task = asyncio.create_task(_ai_loop())
    ao_task = asyncio.create_task(_ao_loop())

    print("[run] AI/AO active. Press ENTER or Ctrl+C to stop.")
    # Wait for keypress or duration
    if args.duration > 0:
        try:
            await asyncio.wait_for(asyncio.to_thread(key_flag.wait), timeout=args.duration)
        except asyncio.TimeoutError:
            print(f"[run] Duration {args.duration}s elapsed; stopping...")
            stop_evt.set()
    else:
        # no duration → wait for Enter
        await asyncio.to_thread(key_flag.wait)
        stop_evt.set()

    # -----------------------
    # Step 6: Graceful shutdown
    # -----------------------
    print("== Step 6: Stopping tasks and writer processes...")
    for t in (ai_task, ao_task, forward_ai_task, forward_ao_task):
        t.cancel()
    await asyncio.gather(ai_task, ao_task, forward_ai_task, forward_ao_task, return_exceptions=True)

    # Signal writers to exit and join
    ai_q.put(None)
    ao_q.put(None)
    ai_proc.join(timeout=5.0)
    ao_proc.join(timeout=5.0)
    print("[ok] Shutdown complete. All files in:", outdir)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI/AO runner with in-process Pico capture and CSV loggers")
    p.add_argument("--daq-config", default=str(REPO1_ROOT / "configs" / "config_test.yml"),
                   help="Path to repo1 YAML for NI-DAQmx AI/AO")
    p.add_argument("--capture-py", default=None,
                   help="Path to repo2 automation/capture_single_shot.py (omit to skip capture)")
    p.add_argument("--capture-config", default=None,
                   help="Path to capture_config_test.yml (defaults to alongside --capture-py)")
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
    # Use a fresh event loop
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
