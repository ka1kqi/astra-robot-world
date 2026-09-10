import numpy as np
from astra_world.simulation import SimulationRuntime


def test_create_simulate_reset_and_return_to_sorting():
    runtime = SimulationRuntime(headless=True).start()
    try:
        created = runtime.submit(
            "create_world",
            {
                "name": "Ball drop",
                "entities": [
                    {"id": "ball_1", "asset_id": "ball", "position": [0, 0, 1]}
                ],
            },
        ).result(10)
        assert created["ok"], created
        first = runtime.snapshot()
        assert first["kind"] == "general" and first["robot"]["type"] == "none"
        assert abs(first["entities"][0]["position"][2] - 1) < 0.001
        result = runtime.submit("simulate", {"duration": 1}).result(10)
        assert result["ok"], result
        assert runtime.snapshot()["entities"][0]["position"][2] < 0.2
        assert runtime.submit("reset_world", {}).result(10)["ok"]
        assert abs(runtime.snapshot()["entities"][0]["position"][2] - 1) < 0.001
        rejected = runtime.submit("sort_blocks", {}).result(10)
        assert rejected["error_code"] == "unsupported_scene"
        assert runtime.submit("build_sorting_station", {}).result(10)["ok"]
        assert len(runtime.snapshot()["blocks"]) == 6
    finally:
        runtime.close()


def test_unknown_asset_rejects_without_replacing_world():
    runtime = SimulationRuntime(headless=True).start()
    try:
        rev = runtime.snapshot()["scene_revision"]
        result = runtime.submit(
            "create_world",
            {
                "name": "Bad",
                "entities": [{"id": "x", "asset_id": "missing", "position": [0, 0, 1]}],
            },
        ).result(10)
        assert not result["ok"]
        assert runtime.snapshot()["scene_revision"] == rev
    finally:
        runtime.close()


def test_save_load_current_state_and_reject_traversal(tmp_path, monkeypatch):
    import astra_world.general_session as module

    monkeypatch.setattr(module, "SCENARIO_DIR", tmp_path)
    runtime = SimulationRuntime(headless=True).start()
    try:
        assert runtime.submit(
            "create_world",
            {
                "name": "Experiment",
                "entities": [{"id": "ball", "asset_id": "ball", "position": [0, 0, 1]}],
            },
        ).result(10)["ok"]
        assert runtime.submit("simulate", {"duration": 0.2}).result(10)["ok"]
        pos = runtime.snapshot()["entities"][0]["position"]
        assert runtime.submit("save_scenario", {"name": "drop"}).result(10)["ok"]
        assert runtime.submit("simulate", {"duration": 0.5}).result(10)["ok"]
        assert runtime.submit("load_scenario", {"name": "drop"}).result(10)["ok"]
        np.testing.assert_allclose(
            runtime.snapshot()["entities"][0]["position"], pos, atol=1e-6
        )
        bad = runtime.submit("save_scenario", {"name": "../escape"}).result(10)
        assert bad["error_code"] == "invalid_arguments"
    finally:
        runtime.close()


def test_add_entity_keeps_current_positions_and_rejects_overlap():
    runtime = SimulationRuntime(headless=True).start()
    try:
        assert runtime.submit(
            "create_world",
            {
                "name": "Drop",
                "entities": [{"id": "ball", "asset_id": "ball", "position": [0, 0, 1]}],
            },
        ).result(10)["ok"]
        assert runtime.submit("simulate", {"duration": 0.2}).result(10)["ok"]
        pos = runtime.snapshot()["entities"][0]["position"]
        result = runtime.submit(
            "add_entity", {"id": "wall", "asset_id": "wall", "position": [2, 0, 1]}
        ).result(10)
        assert result["ok"], result
        np.testing.assert_allclose(runtime.snapshot()["entities"][0]["position"], pos)
        rev = runtime.snapshot()["scene_revision"]
        bad = runtime.submit(
            "add_entity", {"id": "overlap", "asset_id": "ball", "position": pos}
        ).result(10)
        assert not bad["ok"]
        assert runtime.snapshot()["scene_revision"] == rev
    finally:
        runtime.close()
