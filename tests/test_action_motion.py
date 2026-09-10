from threading import Event

import mujoco
import numpy as np

from astra_world.action_contracts import ActionProgram
from astra_world.world_builder import build_world
from astra_world.world_spec import WorldSpec


def test_custom_grasp_lifts_sideways_cube_and_releases():
    from astra_world.action_motion import execute_program
    w = build_world(WorldSpec(name="Sideways grasp", robot="panda", entities=[
        {"id": "green", "asset_id": "small_box", "position": [0.4, -0.12, 0.02], "rotation": [np.pi / 2, 0, 0]}
    ]))
    seen_heights = []
    result = execute_program(w, {"steps": [
        {"op": "gripper", "opened": True},
        {"op": "move_to_pose", "position": [0.4, -0.12, 0.24], "speed": 0.2},
        {"op": "move_to_pose", "position": [0.4, -0.12, 0.024], "speed": 0.2},
        {"op": "gripper", "opened": False, "object_id": "green"},
        {"op": "move_to_pose", "position": [0.4, -0.12, 0.20], "speed": 0.2},
        {"op": "gripper", "opened": True},
        {"op": "wait", "seconds": 1},
    ]}, tick=lambda: seen_heights.append(float(w.data.body("entity_green").xpos[2])))
    assert result["ok"], result
    assert max(seen_heights) > 0.17
    assert w.data.body("entity_green").xpos[2] < 0.03


def test_custom_grasp_requires_contact_and_a_free_target():
    from astra_world.action_motion import execute_program
    w = tower()
    result = execute_program(w, {"steps": [{"op": "gripper", "opened": False, "object_id": "green"}]})
    assert result["error_code"] == "grasp_failed", result
    spec = w.spec.model_copy(deep=True)
    spec.entities[1].fixed = True
    w = build_world(spec)
    before = w.data.qpos.copy()
    result = execute_program(w, {"steps": [{"op": "gripper", "opened": False, "object_id": "green"}]})
    assert result["error_code"] == "fixed_target", result
    np.testing.assert_array_equal(w.data.qpos, before)


def tower():
    return build_world(
        WorldSpec(
            name="Tower",
            robot="panda",
            entities=[
                {
                    "id": "red",
                    "asset_id": "small_box",
                    "position": [0.4, -0.12, 0.02],
                    "color": "red",
                },
                {
                    "id": "green",
                    "asset_id": "small_box",
                    "position": [0.4, -0.12, 0.06],
                    "color": "green",
                },
            ],
        )
    )


def tower_program():
    return ActionProgram(
        steps=[
            {"op": "gripper", "opened": False},
            {
                "op": "move_to_pose",
                "reference_id": "green",
                "position": [-0.09, 0, 0.18],
            },
            {
                "op": "move_to_pose",
                "reference_id": "green",
                "position": [-0.09, 0, 0.005],
            },
            {
                "op": "contact_stroke",
                "direction": [1, 0, 0],
                "distance": 0.17,
                "speed": 0.06,
                "contact_ids": ["green"],
            },
            {
                "op": "contact_stroke",
                "direction": [0, 0, 1],
                "distance": 0.13,
                "speed": 0.06,
                "contact_ids": ["green"],
            },
            {"op": "wait", "seconds": 1},
        ]
    )


def test_finger_stroke_physically_dislodges_upper_block(monkeypatch):
    from astra_world.action_motion import execute_program

    w = tower()
    # All live object movement must occur inside the physics engine. No execution
    # code may assign object qpos/qvel or apply forces directly to object DOFs.
    physical_step = mujoco.mj_step
    last_qpos = w.data.qpos[9:].copy()
    last_qvel = w.data.qvel[9:].copy()

    def checked_step(model, data, *args, **kwargs):
        nonlocal last_qpos, last_qvel
        if data is w.data:
            np.testing.assert_array_equal(data.qpos[9:], last_qpos)
            np.testing.assert_array_equal(data.qvel[9:], last_qvel)
            assert not data.qfrc_applied[9:].any()
            assert not data.xfrc_applied.any()
        physical_step(model, data, *args, **kwargs)
        if data is w.data:
            last_qpos = data.qpos[9:].copy()
            last_qvel = data.qvel[9:].copy()

    monkeypatch.setattr(mujoco, "mj_step", checked_step)
    result = execute_program(w, tower_program())
    assert result["ok"], result
    green = w.data.body("entity_green").xpos
    red = w.data.body("entity_red").xpos
    assert green[2] < 0.025
    assert np.linalg.norm(green[:2] - red[:2]) > 0.039
    assert np.linalg.norm(w.data.joint("green_joint").qvel) < 0.02
    assert result["payload"]["max_contact_force"] > 0


def test_cancel_and_duration_limit_hold_arm():
    from astra_world.action_motion import execute_program

    w = tower()
    cancel = Event()
    cancel.set()
    result = execute_program(
        w, {"steps": [{"op": "wait", "seconds": 1}]}, cancel=cancel
    )
    assert result["error_code"] == "cancelled"
    np.testing.assert_allclose(w.data.ctrl[:7], w.arm_q)
    result = execute_program(
        w, {"steps": [{"op": "wait", "seconds": 1}]}, max_seconds=0.1
    )
    assert result["error_code"] == "time_limit"
    assert w.data.time <= 0.102
    np.testing.assert_allclose(w.data.ctrl[:7], w.arm_q)


def test_invalid_reference_rejects_before_moving():
    from astra_world.action_motion import execute_program

    w = tower()
    before = w.data.qpos.copy()
    result = execute_program(
        w,
        {
            "steps": [
                {"op": "gripper", "opened": False},
                {
                    "op": "move_to_pose",
                    "position": [0, 0, 0.2],
                    "reference_id": "missing",
                },
            ]
        },
    )
    assert result["error_code"] == "unknown_object"
    np.testing.assert_array_equal(w.data.qpos, before)


def test_only_declared_fingers_can_contact_the_target():
    from astra_world.motion import colliding, solve_ik

    w = tower()
    q = solve_ik(w, [0.4, -0.12, 0.065], w.arm_q)
    assert q is not None
    w.data.qpos[:7] = q
    w.data.qpos[7:9] = 0
    mujoco.mj_forward(w.model, w.data)
    assert colliding(w, w.data) is not None
    assert colliding(w, w.data, contact_ids=["green"]) is None
    assert colliding(w, w.data, contact_ids=["red"]) is not None
    # Move the target into the palm in the test fixture. Naming the target
    # must never grant permission for a hand or arm collision.
    w.data.joint("green_joint").qpos[:3] = w.data.body("hand").xpos
    mujoco.mj_forward(w.model, w.data)
    assert colliding(w, w.data, contact_ids=["green"]) is not None


def test_heavy_contact_exceeds_force_limit():
    from astra_world.action_motion import execute_program

    spec = tower().spec.model_copy(deep=True)
    spec.entities[0].fixed = True
    spec.entities[1].mass = 100
    w = build_world(spec)
    result = execute_program(w, tower_program())
    assert result["error_code"] == "contact_force", result
    assert result["payload"]["max_contact_force"] > 40
    contacts = result["payload"]["diagnostics"]["force_contacts"]
    assert contacts and contacts[0]["force_newtons"] > 0
    assert len(contacts[0]["body_pair"]) == 2
    np.testing.assert_allclose(w.data.ctrl[:7], w.arm_q)


def test_explicit_orientation_controls_cartesian_pose():
    from scipy.spatial.transform import Rotation

    from astra_world.action_motion import execute_program

    w = tower()
    result = execute_program(
        w,
        {
            "steps": [
                {
                    "op": "move_to_pose",
                    "position": [0.4, -0.1, 0.3],
                    "rotation": [3.141592653589793, 0, 0.3],
                    "speed": 0.1,
                }
            ]
        },
    )
    assert result["ok"], result
    expected = Rotation.from_euler("XYZ", [np.pi, 0, 0.3]).as_matrix()
    actual = w.data.site("grasp").xmat.reshape(3, 3)
    assert Rotation.from_matrix(expected @ actual.T).magnitude() < 0.025
    np.testing.assert_allclose(w.data.site("grasp").xpos, [0.4, -0.1, 0.3], atol=0.003)


def test_stroke_remains_cartesian_and_holds_orientation():
    from astra_world.action_motion import execute_program

    w = tower()
    assert execute_program(
        w,
        {
            "steps": [
                {"op": "move_to_pose", "position": [0.35, -0.2, 0.3], "speed": 0.1}
            ]
        },
    )["ok"]
    start = w.data.site("grasp").xpos.copy()
    rotation = w.data.site("grasp").xmat.copy()
    positions = []
    orientations = []
    result = execute_program(
        w,
        {
            "steps": [
                {
                    "op": "contact_stroke",
                    "direction": [1, 1, 0],
                    "distance": 0.12,
                    "speed": 0.08,
                    "contact_ids": ["green"],
                }
            ]
        },
        tick=lambda: (
            positions.append(w.data.site("grasp").xpos.copy()),
            orientations.append(w.data.site("grasp").xmat.copy()),
        ),
    )
    assert result["ok"], result
    offsets = np.asarray(positions) - start
    assert np.max(np.abs(offsets[:, 0] - offsets[:, 1])) < 0.003
    assert np.max(np.abs(offsets[:, 2])) < 0.003
    assert max(np.linalg.norm(r - rotation) for r in orientations) < 0.025
    assert np.linalg.norm(offsets[-1]) > 0.115


def test_pick_place_composes_and_obeys_program_deadline():
    from astra_world.action_motion import execute_program

    spec = tower().spec.model_copy(deep=True)
    spec.entities.pop()
    w = build_world(spec)
    program = {
        "steps": [
            {"op": "pick_place", "object_id": "red", "position": [0.4, 0.12, 0.02]}
        ]
    }
    result = execute_program(w, program)
    assert result["ok"], result
    np.testing.assert_allclose(
        w.data.body("entity_red").xpos, [0.4, 0.12, 0.02], atol=0.015
    )
    w = build_world(spec)
    result = execute_program(w, program, max_seconds=0.1)
    assert result["error_code"] == "time_limit", result
    assert w.data.time <= 0.102
    np.testing.assert_allclose(w.data.ctrl[:7], w.arm_q)


def test_prohibited_live_contact_stops_and_holds():
    from astra_world.action_motion import execute_program

    w = tower()
    # Begin with the target intersecting the palm to exercise runtime policing,
    # independently of the approach planner's collision prevention.
    w.data.joint("green_joint").qpos[:3] = w.data.body("hand").xpos
    mujoco.mj_forward(w.model, w.data)
    result = execute_program(
        w,
        {
            "steps": [
                {
                    "op": "contact_stroke",
                    "direction": [1, 0, 0],
                    "distance": 0.01,
                    "contact_ids": ["green"],
                }
            ]
        },
    )
    assert result["error_code"] == "unexpected_contact", result
    np.testing.assert_allclose(w.data.ctrl[:7], w.arm_q)


def test_mid_motion_cancel_stops_at_next_physics_step():
    from astra_world.action_motion import execute_program

    w = tower()
    cancel = Event()
    stopped_at = None

    def tick():
        nonlocal stopped_at
        if w.data.time >= 0.05:
            stopped_at = w.data.qpos.copy()
            cancel.set()

    result = execute_program(
        w,
        {"steps": [{"op": "move_to_pose", "position": [0.4, -0.1, 0.3]}]},
        cancel=cancel,
        tick=tick,
    )
    assert result["error_code"] == "cancelled"
    assert stopped_at is not None
    np.testing.assert_array_equal(w.data.qpos, stopped_at)
    np.testing.assert_allclose(w.data.ctrl[:7], w.arm_q)


def test_fixed_contact_target_rejects_before_motion():
    from astra_world.action_motion import execute_program

    spec = tower().spec.model_copy(deep=True)
    spec.entities[1].fixed = True
    w = build_world(spec)
    before = w.data.qpos.copy()
    result = execute_program(w, tower_program())
    assert result["error_code"] == "fixed_target"
    np.testing.assert_array_equal(w.data.qpos, before)


def circle_program():
    angles = np.linspace(0, 2 * np.pi, 17)
    return ActionProgram(
        steps=[
            {
                "op": "move_to_pose",
                "position": [0.45 + 0.06 * np.cos(t), 0.06 * np.sin(t), 0.3],
                "speed": 0.12,
            }
            for t in angles
        ]
    )


def test_horizontal_circle_tracks_full_loop():
    from astra_world.action_motion import execute_program

    w = tower()
    positions = []
    result = execute_program(
        w,
        circle_program(),
        tick=lambda: positions.append(w.data.site("grasp").xpos.copy()),
    )
    assert result["ok"], result
    points = np.asarray(positions)
    start = np.flatnonzero(np.linalg.norm(points - [0.51, 0, 0.3], axis=1) < 0.002)[0]
    points = points[start:]
    offsets = points[:, :2] - [0.45, 0]
    radii = np.linalg.norm(offsets, axis=1)
    winding = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
    assert winding[-1] - winding[0] > 6.20
    assert np.max(np.abs(radii - 0.06)) < 0.005
    assert np.max(np.abs(points[:, 2] - 0.3)) < 0.003
    assert result["payload"]["elapsed_seconds"] < 30
