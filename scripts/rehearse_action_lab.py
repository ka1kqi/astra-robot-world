"""Opt-in live Astra action discovery. Uses API credits; no HTTP/native instance is started."""

import asyncio
import json
from pathlib import Path
import numpy as np
from astra_world.astra import AstraAdapter
from astra_world.simulation import SimulationRuntime


async def main():
    import astra_world.action_lab as lab_module

    lab_module.ACTIONS_DIR = Path("artifacts/rehearsal-actions").resolve()
    runtime = SimulationRuntime(headless=True).start()
    reports = []
    try:

        async def execute(name, args):
            result = await asyncio.wrap_future(runtime.submit(name, args))
            reports.append({"name": name, "arguments": args, "result": result})
            Path("artifacts").mkdir(exist_ok=True)
            Path("artifacts/action-discovery.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(
                name,
                result["ok"],
                result.get("payload", {}).get("state", ""),
                flush=True,
            )
            return result

        await execute(
            "create_world",
            {
                "name": "Action discovery tower",
                "robot": "panda",
                "entities": [
                    {
                        "id": "red",
                        "asset_id": "small_box",
                        "color": "red",
                        "position": [0.4, -0.12, 0.02],
                    },
                    {
                        "id": "green",
                        "asset_id": "small_box",
                        "color": "green",
                        "position": [0.4, -0.12, 0.06],
                    },
                ],
            },
        )
        await execute("simulate", {"duration": 0.5})
        before = runtime.snapshot()["entities"]
        history = [
            {
                "role": "user",
                "content": "Create and save an action named discovered_topple to dislodge the green block from the red support. "
                "Use Action Lab with five trials. For this experiment, first test a candidate whose horizontal contact stroke travels only 0.005 meters "
                "after approaching the green block from the left at a 0.09 meter offset. Retreat upward afterward. "
                "If that does not achieve the goal, use the physical result to revise the stroke length and verify the new candidate. "
                "Save only after verification. Do not run it in the live world.",
            }
        ]
        adapter = AstraAdapter.from_env()
        await adapter.run_turn(history, execute, lambda event: None)
        np.testing.assert_array_equal(
            [e["position"] for e in runtime.snapshot()["entities"]],
            [e["position"] for e in before],
        )
        trials = [r for r in reports if r["name"] == "test_action"]
        assert len(trials) >= 2, (
            "Expected a failed short-stroke trial followed by a revision."
        )
        assert trials[0]["result"]["payload"]["verified"] is False
        assert trials[-1]["result"]["payload"]["verified"] is True
        assert any(r["name"] == "save_action" and r["result"]["ok"] for r in reports)
        print(
            "Live failure -> revised program -> three successful trials -> saved action passed.",
            flush=True,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
