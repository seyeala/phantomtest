"""Interactive asynchronous analog I/O demo.

This script drives analog-output (AO) channels with random voltages while
simultaneously reading analog-input (AI) channels and publishing results to CSV
files. It prints status messages and lets the user stop the process by pressing
Enter.
"""

import asyncio

from daqio.config import load_yaml
from daqio.daqI import load_config as load_ai_config, setup_task, read_average
from daqio.daqO import write_random
from daqio import publisher


async def ai_worker(cfg: dict) -> None:
    """Continuously read AI channels and publish averages."""
    print("AI worker: starting analog input sampling")
    with setup_task(cfg) as task:
        while True:
            await asyncio.to_thread(read_average, task, cfg)


async def ao_worker(cfg: dict) -> None:
    """Drive AO channels with random voltages."""
    print("AO worker: driving random analog outputs")
    await write_random(
        cfg["device"],
        cfg["interval"],
        cfg["low"],
        cfg["high"],
        seed=cfg.get("seed"),
        channels=cfg["channels"],
    )


async def wait_for_quit(tasks: list[asyncio.Task]) -> None:
    """Wait for the user to press Enter and then cancel tasks."""
    await asyncio.to_thread(input, "Press Enter to stop...\n")
    print("Stopping tasks...")
    for t in tasks:
        t.cancel()


async def main() -> None:
    data = load_yaml("configs/config_test.yml")
    cfg_ai = load_ai_config(data["daqI"])
    cfg_ao = data["daqO"]

    print("Launching asynchronous analog I/O tasks")
    print("Press Enter at any time to stop")

    ao_writer = publisher.start_ao_consumer("ao.csv", ["timestamp", *cfg_ao["channels"]])
    ai_writer = publisher.start_ai_consumer("ai.csv", ["timestamp", *cfg_ai["channels"]])

    tasks: list[asyncio.Task] = [
        asyncio.create_task(ao_worker(cfg_ao)),
        asyncio.create_task(ai_worker(cfg_ai)),
        ao_writer,
        ai_writer,
    ]

    tasks.append(asyncio.create_task(wait_for_quit(tasks.copy())))

    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print("All tasks stopped")


if __name__ == "__main__":
    asyncio.run(main())
