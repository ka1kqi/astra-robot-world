from astra_world.world_builder import build_world
from astra_world.world_spec import WorldSpec
from astra_world.general_manipulation import pick_place
import numpy as np
import pytest


@pytest.mark.parametrize("rotation", [[np.pi / 2, 0, 0], [0, np.pi / 2, 0], [np.pi, 0, 0]])
def test_pick_stacks_cube_resting_on_another_face(rotation):
    spec = station().spec.model_copy(deep=True)
    spec.entities[0].rotation = rotation
    w = build_world(spec)
    result = pick_place(w, "red", [0.4, 0.12, 0.06])
    assert result["ok"], result
    np.testing.assert_allclose(w.data.body("entity_red").xpos, [0.4, 0.12, 0.06], atol=0.012)


def test_sideways_cylinder_still_requires_a_different_grasp():
    spec = station().spec.model_copy(deep=True)
    spec.entities[0].asset_id = "short_cylinder"
    spec.entities[0].rotation = [np.pi / 2, 0, 0]
    w = build_world(spec)
    before = w.data.qpos.copy()
    assert pick_place(w, "red", [0.4, 0.12, 0.06])["error_code"] == "unsupported_grasp"
    np.testing.assert_array_equal(w.data.qpos, before)


def station(extra=None):
    return build_world(
        WorldSpec(
            name="Panda blocks",
            robot="panda",
            entities=[
                {
                    "id": "red",
                    "asset_id": "small_box",
                    "position": [0.4, -0.12, 0.021],
                    "color": "red",
                },
                {
                    "id": "blue",
                    "asset_id": "small_box",
                    "position": [0.4, 0.12, 0.021],
                    "color": "blue",
                },
                *(extra or []),
            ],
        )
    )


def test_panda_stacks_small_blocks_physically():
    w = station()
    result = pick_place(w, "red", [0.4, 0.12, 0.06])
    assert result["ok"], result
    trace = result["payload"]["command_trace"]
    assert [c["op"] for c in trace] == ["gripper", "execute_transport", "move", "gripper", "move", "execute_transport", "move", "gripper", "move"]
    assert all(c["status"] == "completed" for c in trace)
    assert trace[5]["held_id"] == "red"
    assert trace[5]["position"][:2] == [0.4, 0.12]
    np.testing.assert_allclose(
        w.data.body("entity_red").xpos, [0.4, 0.12, 0.06], atol=0.012
    )
    np.testing.assert_allclose(
        w.data.body("entity_blue").xpos, [0.4, 0.12, 0.02], atol=0.008
    )


def test_pick_requires_panda_and_known_grasp():
    w = build_world(
        WorldSpec(
            name="No robot",
            entities=[
                {"id": "red", "asset_id": "small_box", "position": [0.4, 0, 0.02]}
            ],
        )
    )
    result = pick_place(w, "red", [0.4, 0.2, 0.02])
    assert result["error_code"] == "unsupported_robot"
    w = station([{"id": "large", "asset_id": "table", "position": [3, 0, 0.5]}])
    before = w.data.qpos.copy()
    assert pick_place(w, "large", [0.4, 0.2, 0.02])["error_code"] == "unsupported_grasp"
    np.testing.assert_array_equal(w.data.qpos, before)


def test_panda_packs_block_and_cylinder_into_tray():
    w = build_world(
        WorldSpec(
            name="Pack parts",
            robot="panda",
            entities=[
                {
                    "id": "part",
                    "asset_id": "block",
                    "position": [0.4, -0.12, 0.021],
                    "color": "red",
                },
                {
                    "id": "can",
                    "asset_id": "short_cylinder",
                    "position": [0.55, -0.12, 0.021],
                    "color": "blue",
                },
                {
                    "id": "tray",
                    "asset_id": "tray",
                    "position": [0.48, 0.35, 0.025],
                    "color": "gray",
                },
            ],
        )
    )
    for name, target in [("part", [0.43, 0.35, 0.03]), ("can", [0.53, 0.35, 0.03])]:
        result = pick_place(w, name, target)
        assert result["ok"], result
        np.testing.assert_allclose(
            w.data.body("entity_" + name).xpos, target, atol=0.015
        )


def test_deep_support_intersection_is_rejected():
    from astra_world.motion import colliding

    w = station(
        [
            {
                "id": "support",
                "asset_id": "box",
                "fixed": True,
                "position": [0.4, -0.12, -0.038],
            }
        ]
    )
    assert colliding(w, w.data, "red") is not None
