import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from astra_world.world_builder import build_world
from astra_world.world_spec import WorldSpec


def world():
    return build_world(WorldSpec(name="Approaches", robot="panda", entities=[
        {"id": "red", "asset_id": "small_box", "position": [0.4, -0.12, 0.02]},
    ]))


def test_candidates_are_independent_and_preserve_full_integration_state():
    from astra_world.action_lab import PhysicsSnapshot
    from astra_world.tools import CheckApproachesArgs, dispatch
    from astra_world.general_session import GeneralSession
    w = world()
    w.data.qvel[:] = 0.003
    w.data.qacc_warmstart[:] = 0.007
    w.data.qfrc_applied[:] = 0.009
    w.data.xfrc_applied[:] = 0.011
    w.data.ctrl[7] = 41
    before = PhysicsSnapshot.capture(w)
    session = GeneralSession(w)
    args = CheckApproachesArgs(candidates=[
        {"name": "unreachable", "waypoints": [{"position": [1.9, 0, 1.9]}]},
        {"name": "high", "waypoints": [{"position": [0.4, -0.12, 0.3]}]},
    ])
    same, result = dispatch(session, "check_approaches", args)
    assert same is session
    assert result["ok"]
    bad, good = result["payload"]["candidates"]
    assert bad["error_code"] == "unreachable"
    assert bad["failed_waypoint_index"] == 0
    assert good["reachable"] is True and good["clear"] is True
    assert result["payload"]["physical_trial"] is False
    np.testing.assert_array_equal(PhysicsSnapshot.capture(w).state, before.state)
    assert w.revision == before.revision
    assert not hasattr(session, "action_lab")


def test_blocking_geometry_is_reported_without_moving_live_world():
    from astra_world.approaches import check_approaches
    from astra_world.tools import CheckApproachesArgs
    w = world()
    # Place an object across the palm in the captured starting state.
    w.data.joint("red_joint").qpos[:3] = w.data.body("hand").xpos
    mujoco.mj_forward(w.model, w.data)
    result = check_approaches(w, CheckApproachesArgs(candidates=[
        {"name": "blocked", "waypoints": [{"position": [0.4, -0.12, 0.3]}]},
    ]))
    candidate = result["payload"]["candidates"][0]
    assert candidate["error_code"] == "no_path"
    assert candidate["reachable"] is True and candidate["clear"] is False
    assert candidate["failed_waypoint_index"] == 0
    collision = candidate["collision"]
    assert any("red" in name for name in collision["geometry_pair"])
    assert collision["penetration"] > 0
    assert len(collision["position"]) == 3


@pytest.mark.parametrize("candidates", [
    [],
    [{"name": "too-many", "waypoints": [{"position": [0.4, 0, 0.3]}]}] * 9,
    [{"name": "empty", "waypoints": []}],
    [{"name": "too-long", "waypoints": [{"position": [0.4, 0, 0.3]}] * 9}],
    [{"name": "nan", "waypoints": [{"position": [float("nan"), 0, 0.3]}]}],
    [{"name": "rotation", "waypoints": [{"position": [0.4, 0, 0.3], "rotation": [0, float("inf"), 0]}]}],
    [{"name": "underground", "waypoints": [{"position": [0.4, 0, -0.1]}]}],
    [{"name": "far", "waypoints": [{"position": [3, 0, 0.3]}]}],
])
def test_invalid_and_unbounded_inputs_are_rejected(candidates):
    from astra_world.tools import CheckApproachesArgs
    with pytest.raises(ValidationError):
        CheckApproachesArgs(candidates=candidates)


def test_motion_failure_reports_resolved_pose_and_step():
    from astra_world.action_motion import execute_program
    w = world()
    result = execute_program(w, {"steps": [
        {"op": "wait", "seconds": 0.05},
        {"op": "move_to_pose", "position": [1.5, 0, 1.7], "reference_id": "red"},
    ]})
    assert result["error_code"] == "unreachable"
    diagnostics = result["payload"]["diagnostics"]
    assert diagnostics["failed_step_index"] == 1
    np.testing.assert_allclose(diagnostics["requested_pose"]["position"], [1.9, -0.12, 1.72])


def test_route_checks_later_waypoints_and_motion_matches_collision_report():
    from astra_world.approaches import check_approaches
    from astra_world.action_motion import execute_program
    w = world()
    w.data.joint("red_joint").qpos[:3] = [0.4, -0.12, 0.2]
    mujoco.mj_forward(w.model, w.data)
    route = [{"position": [0.4, -0.12, 0.35]}, {"position": [0.4, -0.12, 0.13]}]
    result = check_approaches(w, {"candidates": [{"name": "descent", "waypoints": route}]})
    candidate = result["payload"]["candidates"][0]
    assert candidate["error_code"] == "no_path"
    assert candidate["failed_waypoint_index"] == 1
    assert candidate["checked_waypoints"] == 1
    # The object starts across the palm to exercise controller preflight errors.
    w.data.joint("red_joint").qpos[:3] = w.data.body("hand").xpos
    mujoco.mj_forward(w.model, w.data)
    motion = execute_program(w, {"steps": [{"op": "move_to_pose", **route[0]}]})
    assert motion["error_code"] == "no_path"
    diagnostics = motion["payload"]["diagnostics"]
    assert diagnostics["failed_step_index"] == 0
    assert any("red" in name for name in diagnostics["collision"]["geometry_pair"])


def test_approaches_never_step_physics_or_grant_hypothetical_grasp(monkeypatch):
    from astra_world.approaches import check_approaches
    from astra_world.tools import CheckApproachesArgs
    w = world()

    def forbidden(*args, **kwargs):
        pytest.fail("A kinematic preflight must never step physics")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    args = {"candidates": [{"name": "high", "waypoints": [
        {"position": [0.4, 0, 0.3], "rotation": [np.pi, 0, 0.3]},
    ]}]}
    result = check_approaches(w, args)
    assert result["payload"]["candidates"][0]["clear"]
    with pytest.raises(ValidationError):
        CheckApproachesArgs(**args, held_id="red")


def test_runtime_contact_diagnostics_include_collision_and_stroke_pose():
    from astra_world.action_motion import execute_program
    w = world()
    w.data.joint("red_joint").qpos[:3] = w.data.body("hand").xpos
    mujoco.mj_forward(w.model, w.data)
    result = execute_program(w, {"steps": [{
        "op": "contact_stroke", "direction": [1, 0, 0], "distance": 0.01,
        "contact_ids": ["red"],
    }]})
    assert result["error_code"] == "unexpected_contact"
    diagnostics = result["payload"]["diagnostics"]
    assert diagnostics["failed_step_index"] == 0
    assert diagnostics["stroke_sample_index"] == 1
    assert len(diagnostics["requested_pose"]["position"]) == 3
    assert any("red" in name for name in diagnostics["collision"]["geometry_pair"])
