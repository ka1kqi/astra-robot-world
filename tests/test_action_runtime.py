import numpy as np
from astra_world.simulation import SimulationRuntime


def test_runtime_draft_test_save_run_circle(tmp_path, monkeypatch):
    import astra_world.action_lab as module

    monkeypatch.setattr(module, "ACTIONS_DIR", tmp_path)
    runtime = SimulationRuntime(headless=True).start()

    def command(name, args):
        return runtime.submit(name, args).result(30)

    try:
        assert command(
            "create_world",
            {"name": "Circle workspace", "robot": "panda", "entities": []},
        )["ok"]
        draft = command(
            "draft_action",
            {
                "name": "circle",
                "goal": {
                    "kind": "circle",
                    "target_position": [0.45, 0, 0.3],
                    "radius": 0.06,
                },
            },
        )
        assert draft["ok"], draft
        program = {
            "steps": [
                {
                    "op": "move_to_pose",
                    "position": [0.45 + 0.06 * np.cos(t), 0.06 * np.sin(t), 0.3],
                    "speed": 0.12,
                }
                for t in np.linspace(0, 2 * np.pi, 17)
            ]
        }
        before = runtime.snapshot()["robot"]["joints"]
        result = command(
            "test_action",
            {"draft_id": draft["payload"]["draft_id"], "program": program},
        )
        assert result["ok"], result
        assert result["payload"]["verified"], result
        np.testing.assert_array_equal(runtime.snapshot()["robot"]["joints"], before)
        saved = command("save_action", {"draft_id": draft["payload"]["draft_id"]})
        assert saved["ok"], saved
        assert command("list_actions", {})["payload"]["actions"]
        ran = command("run_action", {"name": "circle"})
        assert ran["ok"], ran
    finally:
        runtime.close()
