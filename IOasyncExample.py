"""Asynchronous analog I/O demo.

This example concurrently drives analog-output (AO) channels while reading
analog-input (AI) channels.  The YAML configuration file contains separate
``daqO`` and ``daqI`` sections.  **The ``daqI`` section is strictly for
inputs** – reusing its channel list for outputs is unsafe and may damage
equipment.  Always use the ``daqO`` section for any AO tasks.
"""

import asyncio

from daqio.daqO import write_random
from daqio.daqI import load_config as load_ai_config, setup_task, read_average
from daqio.config import load_yaml
from daqio import publisher


async def ai_loop(cfg: dict):
    """Continuously read analog inputs and publish results."""
    with setup_task(cfg) as task:
        while True:
            await asyncio.to_thread(read_average, task, cfg)
            # read_average sleeps internally according to cfg["freq"]


async def queue_reader(q: asyncio.Queue, label: str):
    """Extra reader that prints every published item."""
    while True:
        item = await q.get()
        print(f"{label} received:", item)
        q.task_done()


async def main():
    data = load_yaml("configs/config_test.yml")
    cfg_ai = load_ai_config(data["daqI"])
    cfg_ao = data["daqO"]

    # background writers to CSV
    ao_writer = publisher.start_ao_consumer(
        "ao.csv", ["timestamp", *cfg_ao["channels"]]
    )
    ai_writer = publisher.start_ai_consumer(
        "ai.csv", ["timestamp", *cfg_ai["channels"]]
    )

    tasks = [
        asyncio.create_task(
            write_random(
                cfg_ao["device"], 0.1, -1.0, 1.0, channels=cfg_ao["channels"]
            )
        ),
        asyncio.create_task(ai_loop(cfg_ai)),
        asyncio.create_task(queue_reader(publisher._get_ao_queue(), "AO reader")),
        asyncio.create_task(queue_reader(publisher._get_ai_queue(), "AI reader")),
        ao_writer,
        ai_writer,
    ]

    # Run all tasks until cancelled (Ctrl+C)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
