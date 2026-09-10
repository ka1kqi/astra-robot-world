"""Trial isolation and fixed physical predicates, independent of motion planning."""
from threading import Event

import mujoco
import numpy as np
import pytest

from astra_world.action_contracts import ActionProgram, GoalSpec
from astra_world.world_builder import build_world
from astra_world.world_spec import EntitySpec, WorldSpec


WAIT = ActionProgram(steps=[{'op': 'wait', 'seconds': 1.5}])


def integration_state(world):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(world.model, kind))
    mujoco.mj_getState(world.model, world.data, state, kind)
    return state


@pytest.fixture
def lab(tmp_path, monkeypatch):
    from astra_world.action_lab import ActionLab
    world = build_world(WorldSpec(name='fall', robot='panda', entities=[
        EntitySpec(id='target', asset_id='box', scale=(1/3, 1/3, 1/3), position=(.4, -.12, .3)),
        EntitySpec(id='other', asset_id='box', scale=(1/3, 1/3, 1/3), position=(.6, .25, .02)),
    ]))
    result = ActionLab(world, actions_dir=tmp_path)
    return result


def physics_wait(world, program, cancel=None, tick=None, status=None, max_seconds=30):
    """Physical free fall isolates trial measurements from the robot planner."""
    start = world.data.time
    for step in program.steps:
        for _ in range(round(step.seconds / world.model.opt.timestep)):
            if cancel and cancel.is_set():
                return {'ok': False, 'error_code': 'cancelled', 'payload': {}}
            if world.data.time - start >= max_seconds:
                return {'ok': False, 'error_code': 'time_budget', 'payload': {}}
            mujoco.mj_step(world.model, world.data)
            if tick:
                tick()
    return {'ok': True, 'payload': {'elapsed_seconds': world.data.time - start}}


def draft_drop(lab, budget=5):
    return lab.draft('drop', GoalSpec(kind='displace', object_id='target',
                                     target_position=(.4, -.12, .02)), budget)['payload']['draft_id']


def test_success_requires_three_fresh_trials_and_preserves_integration_state(lab, monkeypatch):
    from astra_world import action_lab
    starts = []
    def execute(world, program, **kwargs):
        starts.append(integration_state(world))
        return physics_wait(world, program, **kwargs)
    monkeypatch.setattr(action_lab, 'execute_program', execute)
    original = integration_state(lab.world)
    result = lab.test(draft_drop(lab), WAIT)
    assert result['ok'] and result['payload']['verified']
    assert result['payload']['trials_used'] == 3
    for start in starts[:3]:
        np.testing.assert_array_equal(start, original)
    np.testing.assert_array_equal(integration_state(lab.world), original)
    assert result['payload']['trials'][-1]['measurements']['settled_seconds'] >= .5


def test_exhausted_budget_cannot_save_unconfirmed_candidate(lab, monkeypatch):
    from astra_world import action_lab
    monkeypatch.setattr(action_lab, 'execute_program', physics_wait)
    draft_id = draft_drop(lab, 3)
    # An initial failed candidate consumes the slot needed for two confirmations.
    assert not lab.test(draft_id, ActionProgram(steps=[{'op': 'wait', 'seconds': .02}]))['payload']['verified']
    result = lab.test(draft_id, WAIT)
    assert not result['payload']['verified']
    assert result['payload']['trials_used'] == 3
    assert not lab.save(draft_id)['ok']
    assert not lab.test(draft_id, WAIT)['ok']


def test_cancelled_trial_is_terminal_and_live_world_unchanged(lab, monkeypatch):
    from astra_world import action_lab
    stop = Event()
    before = integration_state(lab.world)
    def execute(world, program, **kwargs):
        stop.set()
        return physics_wait(world, program, **kwargs)
    monkeypatch.setattr(action_lab, 'execute_program', execute)
    draft_id = draft_drop(lab)
    result = lab.test(draft_id, WAIT, cancel=stop)
    assert not result['payload']['verified']
    assert result['payload']['state'] == 'cancelled'
    assert not lab.test(draft_id, WAIT)['ok']
    np.testing.assert_array_equal(integration_state(lab.world), before)


def test_fixed_target_rejected_before_consuming_trials(lab):
    lab.world.spec.entities[0].fixed = True
    fixed = build_world(lab.world.spec)
    from astra_world.action_lab import ActionLab
    result = ActionLab(fixed).draft('push', GoalSpec(kind='displace', object_id='target', target_position=(.5, 0, .02)))
    assert not result['ok'] and result['error_code'] == 'fixed_target'


def test_program_cannot_define_success_predicate(lab):
    result = lab.test(draft_drop(lab), {'steps': [{'op': 'wait'}], 'goal_score': 1})
    assert not result['ok']
    assert lab.snapshot()['drafts'][0]['trials_used'] == 0


def test_motion_ok_does_not_claim_toppling_when_upper_still_supported(tmp_path, monkeypatch):
    from astra_world import action_lab
    monkeypatch.setattr(action_lab, 'execute_program', physics_wait)
    world = build_world(WorldSpec(name='tower', robot='panda', entities=[
        EntitySpec(id='red', asset_id='box', scale=(1/3, 1/3, 1/3), position=(.4, -.12, .02)),
        EntitySpec(id='green', asset_id='box', scale=(1/3, 1/3, 1/3), position=(.4, -.12, .06)),
    ]))
    lab = action_lab.ActionLab(world, actions_dir=tmp_path)
    created = lab.draft('topple', GoalSpec(kind='topple', object_id='green', support_id='red'))
    assert created['ok'], created
    result = lab.test(created['payload']['draft_id'], WAIT)
    assert not result['payload']['verified']
    measurements = result['payload']['trials'][0]['measurements']
    assert measurements['supported_by_support'] and not measurements['on_ground']


def test_unrelated_object_motion_fails_even_when_target_reaches_goal(lab, monkeypatch):
    from astra_world import action_lab
    monkeypatch.setattr(action_lab, 'execute_program', physics_wait)
    lab.world.data.joint('other_joint').qvel[0] = 1
    result = lab.test(draft_drop(lab), WAIT)
    assert not result['payload']['verified']
    assert not result['payload']['trials'][0]['measurements']['preserved']


def test_saved_record_is_validated_and_run_rechecks_current_geometry(lab, monkeypatch):
    from astra_world import action_lab
    monkeypatch.setattr(action_lab, 'execute_program', physics_wait)
    draft_id = draft_drop(lab)
    assert lab.test(draft_id, WAIT)['payload']['verified']
    assert lab.save(draft_id)['ok']
    assert lab.list_actions()['payload']['actions'][0]['name'] == 'drop'
    # The current target is now a different size: old verification does not apply.
    spec = lab.world.spec.model_copy(deep=True)
    spec.entities[0].scale = (2, 2, 2)
    changed = build_world(spec)
    before = integration_state(changed)
    assert not lab.run(changed, 'drop')['ok']
    np.testing.assert_array_equal(integration_state(changed), before)
    (lab.actions_dir / 'forged.json').write_text('{"name":"forged","verified":true}')
    listing = lab.list_actions()['payload']
    assert len(listing['actions']) == 1 and listing['invalid_actions']


def test_wall_deadline_stops_additional_trials(lab, monkeypatch):
    from astra_world import action_lab
    monkeypatch.setattr(action_lab, 'execute_program', physics_wait)
    draft_id = draft_drop(lab)
    monkeypatch.setattr(action_lab.time, 'monotonic', lambda: 1e20)
    result = lab.test(draft_id, WAIT)
    assert not result['ok'] and result['error_code'] == 'wall_budget'
    assert result['payload']['trials_used'] == 0


def test_saved_action_revalidates_then_runs_live_and_rejects_repeat_at_goal(lab, monkeypatch):
    from astra_world import action_lab
    monkeypatch.setattr(action_lab, 'execute_program', physics_wait)
    draft_id = draft_drop(lab)
    assert lab.test(draft_id, WAIT)['payload']['verified']
    assert lab.save(draft_id)['ok']
    result = lab.run(lab.world, 'drop')
    assert result['ok'], result
    assert lab.world.data.time == pytest.approx(1.5)
    assert not lab.run(lab.world, 'drop')['ok']


def test_scene_change_during_revalidation_aborts_live_replay(lab, monkeypatch):
    from astra_world import action_lab
    monkeypatch.setattr(action_lab, 'execute_program', physics_wait)
    draft_id = draft_drop(lab)
    assert lab.test(draft_id, WAIT)['payload']['verified']
    assert lab.save(draft_id)['ok']
    def changed_scene(message):
        if 'isolated trial' in message:
            lab.world.revision += 1
    result = lab.run(lab.world, 'drop', status=changed_scene)
    assert not result['ok'] and result['error_code'] == 'stale_scene'
    assert lab.world.data.time == 0


def test_verified_tower_program_passes_three_physical_trials(tmp_path):
    from astra_world.action_lab import ActionLab
    world = build_world(WorldSpec(name='tower', robot='panda', entities=[
        EntitySpec(id='red', asset_id='small_box', position=(.4, -.12, .02)),
        EntitySpec(id='green', asset_id='small_box', position=(.4, -.12, .06)),
        EntitySpec(id='unrelated', asset_id='small_box', position=(.55, .25, .02)),
    ]))
    lab = ActionLab(world, actions_dir=tmp_path)
    before = integration_state(world)
    draft_id = lab.draft('topple', GoalSpec(kind='topple', object_id='green', support_id='red'))['payload']['draft_id']
    program = ActionProgram(steps=[
        {'op': 'gripper', 'opened': False},
        {'op': 'move_to_pose', 'reference_id': 'green', 'position': [-.09, 0, .18]},
        {'op': 'move_to_pose', 'reference_id': 'green', 'position': [-.09, 0, .005]},
        {'op': 'contact_stroke', 'direction': [1, 0, 0], 'distance': .17, 'speed': .06, 'contact_ids': ['green']},
        {'op': 'contact_stroke', 'direction': [0, 0, 1], 'distance': .13, 'speed': .06, 'contact_ids': ['green']},
        {'op': 'wait', 'seconds': 1},
    ])
    result = lab.test(draft_id, program)
    assert result['payload']['verified'], result
    measurements = result['payload']['trials'][-1]['measurements']
    assert measurements['on_ground'] and not measurements['supported_by_support']
    assert measurements['preserved']
    np.testing.assert_array_equal(integration_state(world), before)


def test_circle_requires_measured_sweep_and_three_confirmations(tmp_path):
    from astra_world.action_lab import ActionLab
    world = build_world(WorldSpec(name='circle', robot='panda'))
    lab = ActionLab(world, actions_dir=tmp_path)
    before = integration_state(world)
    draft_id = lab.draft('circle', GoalSpec(kind='circle', target_position=(.45, 0, .3), radius=.06, plane='xy'))['payload']['draft_id']
    assert not lab.test(draft_id, WAIT)['payload']['verified']
    program = ActionProgram(steps=[
        {'op': 'move_to_pose', 'position': [.45+.06*np.cos(angle), .06*np.sin(angle), .3], 'speed': .12}
        for angle in np.linspace(0, 2*np.pi, 17)
    ])
    result = lab.test(draft_id, program)
    assert result['payload']['verified'], result
    assert result['payload']['trials_used'] == 4
    assert result['payload']['trials'][-1]['measurements']['circle']['success']
    assert len(result['payload']['trials'][-1]['measurements']['trace']) <= 256
    assert lab.save(draft_id)['ok']
    assert lab.list_actions()['payload']['actions'][0]['goal']['kind'] == 'circle'
    np.testing.assert_array_equal(integration_state(world), before)


def test_cancelled_circle_without_trace_is_reported_without_crashing(tmp_path, monkeypatch):
    from astra_world import action_lab
    world = build_world(WorldSpec(name='circle', robot='panda'))
    lab = action_lab.ActionLab(world, actions_dir=tmp_path)
    draft_id = lab.draft('circle', GoalSpec(kind='circle', target_position=(.45, 0, .3)))['payload']['draft_id']
    stop = Event()
    def stopped(world, program, **kwargs):
        stop.set()
        return {'ok': False, 'error_code': 'cancelled', 'payload': {}}
    monkeypatch.setattr(action_lab, 'execute_program', stopped)
    result = lab.test(draft_id, WAIT, cancel=stop)
    assert result['payload']['state'] == 'cancelled'
    assert not result['payload']['verified']


def test_trial_snapshot_includes_model_physics_and_integration_state(lab, monkeypatch):
    from astra_world import action_lab
    lab.world.model.geom_friction[:] = [.63, .007, .0003]
    lab.world.data.qacc_warmstart[:] = .125
    lab.world.data.qfrc_applied[:] = .01
    lab.world.data.xfrc_applied[:] = .002
    lab.world.data.ctrl[0] += .03
    expected_state = integration_state(lab.world)
    expected_friction = lab.world.model.geom_friction.copy()
    draft_id = draft_drop(lab)
    captured = []
    def inspect(world, program, **kwargs):
        captured.append((world.model.geom_friction.copy(), integration_state(world)))
        return {'ok': False, 'error_code': 'deliberate_failure', 'payload': {}}
    monkeypatch.setattr(action_lab, 'execute_program', inspect)
    # Changing the live model after capture must not change subsequent trials.
    lab.world.model.geom_friction[:] = [.9, .003, .0001]
    lab.test(draft_id, WAIT)
    np.testing.assert_array_equal(captured[0][0], expected_friction)
    np.testing.assert_array_equal(captured[0][1], expected_state)


def test_half_second_settling_can_be_saved_at_nonzero_simulation_time(lab, monkeypatch):
    from astra_world import action_lab
    monkeypatch.setattr(action_lab, 'execute_program', physics_wait)
    lab.world.data.time = 100.
    draft_id = draft_drop(lab)
    result = lab.test(draft_id, ActionProgram(steps=[{'op': 'wait', 'seconds': .872}]))
    assert result['payload']['verified'], result
    assert lab.save(draft_id)['ok']
