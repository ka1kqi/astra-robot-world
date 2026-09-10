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


def test_stack_blue_then_green_handles_release_impact_without_stopping():
    import mujoco

    w = build_world(WorldSpec(name="Three block stack", robot="panda", entities=[
        {"id": "red_block", "asset_id": "small_box", "position": [.4, -.12, .02]},
        {"id": "blue_block", "asset_id": "small_box", "position": [.55, -.12, .02]},
        {"id": "green_block", "asset_id": "small_box", "position": [.4, .04, .02]},
        {"id": "tray", "asset_id": "tray", "position": [.48, .35, .025]},
    ]))
    for _ in range(1000):
        mujoco.mj_step(w.model, w.data)
    result = pick_place(w, "blue_block", [.4, -.12, .0598922])
    assert result["ok"], result
    blue = w.data.body("entity_blue_block").xpos.copy()
    result = pick_place(w, "green_block", blue + [0, 0, .04])
    assert result["ok"], result
    red = w.data.body("entity_red_block").xpos
    blue = w.data.body("entity_blue_block").xpos
    green = w.data.body("entity_green_block").xpos
    np.testing.assert_allclose(red, [.4, -.12, .02], atol=.005)
    np.testing.assert_allclose(blue - red, [0, 0, .04], atol=.005)
    np.testing.assert_allclose(green - blue, [0, 0, .04], atol=.005)


def test_releasing_object_permits_only_residual_finger_contact(monkeypatch):
    import mujoco
    from astra_world.motion import Controller, MotionError, colliding, solve_ik

    w = station()
    q = solve_ik(w, [.4, -.12, .025], w.arm_q)
    w.data.qpos[:7] = q
    w.data.qpos[7:9] = 0
    mujoco.mj_forward(w.model, w.data)
    ctl = Controller(w)
    ctl.held_id = "red"
    checked = []

    def opening(seconds):
        assert ctl.held_id is None
        assert ctl.contact_ids == ("red",)
        assert colliding(w, w.data, contact_ids=ctl.contact_ids) is None
        # Releasing never permits the target to intersect the palm.
        w.data.joint("red_joint").qpos[:3] = w.data.body("hand").xpos
        mujoco.mj_forward(w.model, w.data)
        assert colliding(w, w.data, contact_ids=ctl.contact_ids) is not None
        checked.append(True)
        raise MotionError("cancelled", "Stopped while opening")

    monkeypatch.setattr(ctl, "step", opening)
    with pytest.raises(MotionError):
        ctl.release()
    assert checked
    assert ctl.held_id is None
    assert ctl.contact_ids == ()
