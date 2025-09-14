"""Interactive asynchronous analog I/O demo with keypress exit.

Continuously writes random analog outputs while reading analog inputs and
publishing channel values. Press Enter to stop the demo gracefully.
"""

import asyncio

from daqio.config import load_yaml
from daqio.daqI import load_config as load_ai_config, setup_task, read_average
from daqio.daqO import write_random
from daqio import publisher


async def ai_worker(cfg: dict) -> None:
    """Continuously read AI channels and publish averages."""
    print("AI worker started; reading inputs.")
    with setup_task(cfg) as task:
        while True:
            await asyncio.to_thread(read_average, task, cfg)


async def ao_worker(cfg: dict) -> None:
    """Continuously write random voltages to AO channels."""
    print("AO worker started; writing random outputs.")
    await write_random(
        cfg["device"],
        cfg["interval"],
        cfg["low"],
        cfg["high"],
        seed=cfg.get("seed"),
        channels=cfg["channels"],
    )


async def wait_for_quit() -> None:
    """Pause until the user presses Enter."""
    await asyncio.to_thread(input, "Press Enter to quit...\n")


async def main() -> None:
    data = load_yaml("configs/config_test.yml")
    cfg_ai = load_ai_config(data["daqI"])
    cfg_ao = data["daqO"]

    # Background CSV writers
    ao_writer = publisher.start_ao_consumer("ao.csv", ["timestamp", *cfg_ao["channels"]])
    ai_writer = publisher.start_ai_consumer("ai.csv", ["timestamp", *cfg_ai["channels"]])

    tasks = [
        asyncio.create_task(ao_worker(cfg_ao)),
        asyncio.create_task(ai_worker(cfg_ai)),
        ao_writer,
        ai_writer,
    ]

    print("Analog I/O tasks running.")
    print("Press Enter to stop.\n")

    await wait_for_quit()
    print("Stopping tasks...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
