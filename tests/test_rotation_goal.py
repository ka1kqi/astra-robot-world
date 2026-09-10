import math
from copy import deepcopy

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from astra_world.action_contracts import GoalSpec, MotionStep
from astra_world.action_lab import ActionLab, PhysicsSnapshot, SavedAction, _Evaluator
from astra_world.world_builder import build_world
from astra_world.world_spec import WorldSpec


def world(rotation=(0, 0, 0)):
    return build_world(WorldSpec(name='Rotate', robot='panda', entities=[
        {'id': 'red', 'asset_id': 'small_box', 'position': [.4, -.12, .02], 'rotation': rotation},
        {'id': 'other', 'asset_id': 'small_box', 'position': [.55, .15, .02]},
    ]))


def goal(**changes):
    return GoalSpec.model_validate(dict(kind='rotate', object_id='red', target_rotation=[0, 0, math.pi / 4], **changes))


def evaluator(w, g):
    return _Evaluator(w, g, {e.id: w.data.body(w.body_name(e.id)).xpos.copy() for e in w.spec.entities})


def set_pose(w, rotation, position=None):
    q = Rotation.from_euler('XYZ', rotation).as_quat()
    w.data.joint('red_joint').qpos[3:] = q[[3, 0, 1, 2]]
    if position is not None:
        w.data.joint('red_joint').qpos[:3] = position
    mujoco.mj_forward(w.model, w.data)


def settle(e):
    e.tick()
    e.world.data.time += .5
    e.tick()
    return e.success()


def test_rotation_goal_requires_full_orientation_and_final_position():
    w = world()
    e = evaluator(w, goal())
    assert not settle(e)
    assert e.latest['angular_error'] == pytest.approx(math.pi / 4)
    set_pose(w, [0, 0, math.pi / 4])
    assert settle(e)
    set_pose(w, [.2, 0, math.pi / 4])
    assert not settle(e)  # Correct yaw alone cannot hide a tilted object.
    set_pose(w, [0, 0, math.pi / 4], [.5, -.12, .02])
    assert not settle(e)  # In-place is the default when no destination is supplied.
    e = evaluator(w, goal(target_position=[.5, -.12, .02]))
    assert settle(e)


def test_rotation_error_wraps_pi_and_settling_resets_when_target_is_lost():
    w = world()
    g = GoalSpec(kind='rotate', object_id='red', target_rotation=[0, 0, -math.pi + .01])
    e = evaluator(w, g)
    set_pose(w, [0, 0, math.pi - .01])
    e.tick()
    assert e.latest['angular_error'] == pytest.approx(.02)
    w.data.time += .49
    e.tick()
    assert not e.success()
    set_pose(w, [0, 0, 0])
    e.tick()
    set_pose(w, [0, 0, math.pi - .01])
    assert settle(e)


def test_rotation_and_primitive_contracts_reject_invalid_requests():
    assert MotionStep(op='pick_place', object_id='red', position=[.4, 0, .02], target_rotation=[0, 0, .5]).target_rotation == (0, 0, .5)
    for payload in [
        {'kind': 'rotate', 'object_id': 'red'},
        {'kind': 'rotate', 'object_id': 'red', 'target_rotation': [0, 0, float('nan')]},
        {'kind': 'rotate', 'object_id': 'red', 'target_rotation': [0, 0, 0], 'angular_tolerance': .5},
        {'kind': 'rotate', 'object_id': 'red', 'target_rotation': [0, 0, 0], 'support_id': 'other'},
        {'kind': 'displace', 'object_id': 'red', 'target_position': [.4, 0, .02], 'target_rotation': [0, 0, .5]},
    ]:
        with pytest.raises(ValidationError): GoalSpec.model_validate(payload)
    with pytest.raises(ValidationError):
        MotionStep(op='wait', target_rotation=[0, 0, .5])


def test_rotation_measurements_are_required_for_saved_evidence(tmp_path):
    w = world()
    lab = ActionLab(w, tmp_path)
    before = PhysicsSnapshot.capture(w).state
    drafted = lab.draft('rotate_red', goal())
    identifier = drafted['payload']['draft_id']
    result = lab.test(identifier, {'steps': [
        {'op': 'pick_place', 'object_id': 'red', 'reference_id': 'red', 'position': [0, 0, 0],
         'target_rotation': [0, 0, math.pi / 4]},
        {'op': 'wait', 'seconds': .6},
    ]})
    assert result['payload']['verified'], result
    assert result['payload']['trials_used'] == 3
    assert lab.save(identifier)['ok']
    saved = lab._load(tmp_path / 'rotate_red.json').model_dump(mode='json')
    np.testing.assert_array_equal(PhysicsSnapshot.capture(w).state, before)
    for changes in [
        {'rotation': [0, 0, 0], 'angular_error': 0},
        {'angular_error': None},
        {'position': [.6, -.12, .02], 'distance_to_goal': 0},
        {'angular_speed': 1},
        {'settled_seconds': .1},
    ]:
        forged = deepcopy(saved)
        forged['verification'][0]['measurements'].update(changes)
        with pytest.raises(ValidationError): SavedAction.model_validate(forged)


def test_already_rotated_goal_rejects_draft_and_wait_cannot_satisfy_new_rotation(tmp_path):
    lab = ActionLab(world([0, 0, math.pi / 4]), tmp_path)
    assert lab.draft('unchanged', goal())['error_code'] == 'goal_already_satisfied'
    lab = ActionLab(world(), tmp_path)
    identifier = lab.draft('rotation', goal())['payload']['draft_id']
    report = lab.test(identifier, {'steps': [{'op': 'wait', 'seconds': 1}]})
    assert report['payload']['verified'] is False
    assert report['payload']['trials'][0]['error_code'] == 'goal_not_met'
