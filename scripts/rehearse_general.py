"""Opt-in live Astra scene/stack/pack rehearsal; uses configured API credits."""

import asyncio, json
from pathlib import Path
from astra_world.astra import AstraAdapter
from astra_world.simulation import SimulationRuntime


async def main():
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    a = AstraAdapter.from_env()
    reports = []
    scenarios = [
        [
            "Create a Panda stacking scene with a red small_box at [0.4,-0.12,0.021] and a blue small_box at [0.4,0.12,0.021].",
            "Stack the red block on the blue block.",
            "Add a wall obstacle at [0.65,-0.3,0.15], scaled [0.2,0.2,0.5].",
        ],
        [
            "Create a Panda packing scene: a red block at [0.4,-0.12,0.021], a blue short_cylinder at [0.55,-0.12,0.021], and a tray at [0.48,0.35,0.025].",
            "Pack both objects into the tray, placing their centers at [0.43,0.35,0.03] and [0.53,0.35,0.03].",
        ],
        [
            "Build a scene without a robot with a ball dropping from one meter, then simulate for one second."
        ],
    ]
    for prompts in scenarios:
        r = SimulationRuntime(headless=True).start()
        history = []
        try:
            for prompt in prompts:
                calls = []

                async def execute(name, args):
                    result = await asyncio.wrap_future(r.submit(name, args))
                    calls.append(dict(tool=name, arguments=args, result=result))
                    return result

                history.append(dict(role="user", content=prompt))
                reply = await a.run_turn(history, execute, lambda e: None)
                reports.append(
                    dict(prompt=prompt, reply=reply, calls=calls, state=r.snapshot())
                )
                Path("artifacts/general-live.json").write_text(
                    json.dumps(reports, indent=2)
                )
                assert calls and all(c["result"]["ok"] for c in calls), calls
                if prompt.startswith("Stack"):
                    assert any(c["tool"] == "pick_place" for c in calls)
                if prompt.startswith("Pack"):
                    assert sum(c["tool"] == "pick_place" for c in calls) == 2
                if prompt.startswith("Add"):
                    assert any(c["tool"] == "add_entity" for c in calls)
                if prompt.startswith("Build a scene"):
                    assert any(c["tool"] == "simulate" for c in calls)
                    assert r.snapshot()["entities"][0]["position"][2] < 0.1
                print(
                    json.dumps(
                        dict(
                            prompt=prompt,
                            reply=reply,
                            tools=[
                                (
                                    c["tool"],
                                    c["result"]["ok"],
                                    c["result"].get("detail"),
                                )
                                for c in calls
                            ],
                        )
                    ),
                    flush=True,
                )
        finally:
            r.close()


if __name__ == "__main__":
    asyncio.run(main())
