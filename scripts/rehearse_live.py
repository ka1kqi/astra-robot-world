"""Opt-in live Astra rehearsal. Uses configured API credentials and incurs API usage."""

import argparse
import asyncio
import json
from pathlib import Path
import time

from astra_world.astra import AstraAdapter
from astra_world.simulation import SimulationRuntime

PROMPTS = [
    "Build a sorting station with colored blocks and two bins.",
    "Put the red blocks in the left bin.",
    "Now add an obstacle between the robot and that bin.",
    "Try again.",
]
EXPECTED = ["build_sorting_station", "sort_blocks", "add_obstacle", "retry_last_task"]


async def rehearse(repeats):
    adapter = AstraAdapter.from_env()
    if not adapter.configured:
        raise SystemExit("Configure Astra in .env before running a live rehearsal.")
    reports = []
    for trial in range(repeats):
        runtime = SimulationRuntime(headless=True).start()
        history, events = [], []
        try:
            for index, prompt in enumerate(PROMPTS):
                calls = []

                async def execute(name, args):
                    result = await asyncio.wrap_future(runtime.submit(name, args))
                    calls.append({"tool": name, "arguments": args, "result": result})
                    return result

                start = time.monotonic()
                history.append({"role": "user", "content": prompt})
                reply = await adapter.run_turn(history, execute, events.append)
                state = runtime.snapshot()
                record = {
                    "trial": trial + 1,
                    "prompt": prompt,
                    "reply": reply,
                    "calls": calls,
                    "wall_seconds": round(time.monotonic() - start, 3),
                    "state": state,
                }
                reports.append(record)
                Path("artifacts").mkdir(exist_ok=True)
                Path("artifacts/live-rehearsal.json").write_text(
                    json.dumps(
                        {
                            "mode": "live_astra",
                            "model": adapter.model,
                            "turns": reports,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                assert EXPECTED[index] in [c["tool"] for c in calls], record["reply"]
                assert all(c["result"]["ok"] for c in calls), calls
                if index in (1, 3):
                    reds = [b for b in state["blocks"] if b["color"] == "red"]
                    assert len(reds) == 2 and all(b["bin"] == "left_bin" for b in reds)
                    assert all(
                        b["bin"] is None for b in state["blocks"] if b["color"] != "red"
                    )
                if index == 3:
                    retry = next(
                        c["result"] for c in calls if c["tool"] == "retry_last_task"
                    )
                    assert state["obstacles"] and any(
                        r != "direct" for r in retry["payload"]["routes"]
                    )
                print(
                    f"Trial {trial + 1}, step {index + 1}: passed ({record['wall_seconds']}s) — {reply}",
                    flush=True,
                )
        finally:
            runtime.close()
    print(
        f"{repeats} live four-prompt rehearsals passed. Log: artifacts/live-rehearsal.json",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    asyncio.run(rehearse(parser.parse_args().repeats))
