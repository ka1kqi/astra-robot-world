"""Extraction requires a changed support relation, not merely a moved stack."""
import json

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from astra_world.action_contracts import ActionProgram, GoalSpec
from astra_world.action_lab import ActionLab, _Evaluator, SavedAction
from astra_world.world_builder import build_world
from astra_world.world_spec import EntitySpec, WorldSpec


def goal(**changes):
    return GoalSpec.model_validate(dict(kind='extract', object_id='blue', supported_id='red', landing_id='tray', **changes))


def scene(fixed_tray=True):
    world = build_world(WorldSpec(name='extraction', robot='panda', entities=[
        EntitySpec(id='tray', asset_id='tray', position=(.65, .35, .025), fixed=fixed_tray),
        EntitySpec(id='blue', asset_id='small_box', position=(.65, .35, .03)),
        EntitySpec(id='red', asset_id='small_box', position=(.65, .35, .07)),
        EntitySpec(id='green', asset_id='small_box', position=(.76, .43, .03)),
    ]))
    advance(world, .3)
    return world


def advance(world, seconds, evaluator=None):
    for _ in range(round(seconds / world.model.opt.timestep)):
        mujoco.mj_step(world.model, world.data)
        if evaluator:
            evaluator.tick()


def shift(world, identifier, dx):
    world.data.joint(f'{identifier}_joint').qpos[0] += dx
    mujoco.mj_forward(world.model, world.data)


def evaluator_for(world, **changes):
    positions = {e.id: world.data.body(world.body_name(e.id)).xpos.copy() for e in world.spec.entities}
    return _Evaluator(world, goal(preserve_ids=['tray', 'green'], **changes), positions)


def test_extract_allows_red_to_land_on_tray_and_requires_half_second_settling():
    world = scene()
    evaluator = evaluator_for(world)
    shift(world, 'blue', -.12)
    advance(world, .4, evaluator)
    assert not evaluator.success()
    advance(world, .7, evaluator)
    assert evaluator.success(), evaluator.latest
    measured = evaluator.latest['extraction']
    assert measured['upper_on_landing'] and not measured['upper_supported_by_lower']
    assert measured['lower_displacement'] > .1
    assert measured['horizontal_separation'] > .1
    assert evaluator.latest['preserved']


@pytest.mark.parametrize('case', ['whole_stack', 'upper_only', 'wrong_landing', 'green_moved', 'tray_moved', 'wrong_destination'])
def test_extract_rejects_outcomes_that_do_not_meet_all_predicates(case):
    world = scene(fixed_tray=case != 'tray_moved')
    evaluator = evaluator_for(world, **({'target_position': (.9, .35, .03)} if case == 'wrong_destination' else {}))
    if case == 'upper_only':
        shift(world, 'red', -.12)
    else:
        shift(world, 'blue', -.12)
    if case == 'whole_stack':
        shift(world, 'red', -.12)
    if case == 'wrong_landing':
        shift(world, 'red', .3)
    if case == 'green_moved':
        shift(world, 'green', -.04)
    if case == 'tray_moved':
        shift(world, 'tray', -.04)
    advance(world, 1.2, evaluator)
    assert not evaluator.success(), evaluator.latest


def test_extract_draft_preserves_tray_green_and_checks_initial_support(tmp_path):
    world = scene()
    lab = ActionLab(world, tmp_path)
    result = lab.draft('extract_blue', goal())
    assert result['ok'], result
    assert result['payload']['goal']['preserve_ids'] == ['green', 'tray']
    shift(world, 'red', -.12)
    advance(world, .5)
    result = lab.draft('no_stack', goal())
    assert not result['ok'] and result['error_code'] == 'precondition_failed'


def test_extraction_roles_and_measurement_bounds_are_validated(tmp_path):
    base = dict(kind='extract', object_id='blue', supported_id='red', landing_id='tray')
    assert GoalSpec.model_validate(base).kind == 'extract'
    for changes in [dict(supported_id=None), dict(landing_id=None), dict(supported_id='blue'),
                    dict(landing_id='red'), dict(landing_id='blue'), dict(support_id='green'),
                    dict(preserve_ids=['red']), dict(preserve_ids=['blue']),
                    dict(min_displacement=0), dict(min_displacement=float('nan')),
                    dict(kind='displace', target_position=(.5, .3, .03))]:
        with pytest.raises(ValidationError):
            GoalSpec.model_validate(base | changes)
    world = scene()
    for identifier in ['blue', 'red']:
        spec = world.spec.model_copy(deep=True)
        next(e for e in spec.entities if e.id == identifier).fixed = True
        result = ActionLab(build_world(spec), tmp_path / identifier).draft('fixed', goal())
        assert not result['ok']
    result = ActionLab(world, tmp_path).draft('unknown', GoalSpec.model_validate(base | {'landing_id': 'missing'}))
    assert not result['ok'] and result['error_code'] == 'unknown_entity'


def test_saved_extraction_validates_new_references_and_rebinds_roles(tmp_path, monkeypatch):
    from astra_world import action_lab
    world = scene()
    lab = ActionLab(world, tmp_path)
    # Isolate evaluator/save/replay behavior from the robot planner: physically settle
    # the upper object after a controlled lower-object relocation in each trial.
    def relocate(world, program, tick=None, **kwargs):
        shift(world, program.steps[0].reference_id, -.12)
        for _ in range(round(1.2 / world.model.opt.timestep)):
            mujoco.mj_step(world.model, world.data)
            if tick:
                tick()
        return {'ok': True, 'payload': {}}
    monkeypatch.setattr(action_lab, 'execute_program', relocate)
    draft = lab.draft('extract_blue', goal())['payload']['draft_id']
    program = ActionProgram(steps=[dict(op='move_to_pose', reference_id='blue', position=(0, 0, .1))])
    assert lab.test(draft, program)['payload']['verified']
    assert lab.save(draft)['ok']
    record = json.loads((tmp_path / 'extract_blue.json').read_text())
    forged = json.loads(json.dumps(record))
    forged['program']['steps'] = [{'op': 'gripper', 'opened': False, 'object_id': 'red'}]
    with pytest.raises(ValidationError):
        SavedAction.model_validate(forged)
    for role in ['supported_id', 'landing_id']:
        forged = json.loads(json.dumps(record))
        forged['goal'][role] = 'missing'
        with pytest.raises(ValidationError):
            SavedAction.model_validate(forged)
    for changes in [None, {'upper_on_landing': False}, {'upper_supported_by_lower': True},
                    {'upper_linear_speed': .2}, {'horizontal_separation': .001}]:
        forged = json.loads(json.dumps(record))
        if changes is None:
            forged['verification'][0]['measurements']['extraction'] = None
        else:
            forged['verification'][0]['measurements']['extraction'].update(changes)
        with pytest.raises(ValidationError):
            SavedAction.model_validate(forged)
    bindings = {'blue': 'lower', 'red': 'upper', 'tray': 'landing', 'green': 'other'}
    spec = world.spec.model_copy(deep=True)
    for entity in spec.entities:
        entity.id = bindings[entity.id]
    rebound = build_world(spec)
    advance(rebound, .3)
    result = lab.run(rebound, 'extract_blue', bindings=bindings)
    assert result['ok'], result
    assert result['payload']['result']['measurements']['preservation_displacements'].keys() == {'landing', 'other'}
    assert not lab.run(rebound, 'extract_blue', bindings=bindings)['ok']
    assert not lab.run(world, 'extract_blue', bindings={'red': 'blue'})['ok']


def test_extraction_proposal_preserves_landing_and_validates_named_upper(monkeypatch):
    import asyncio
    from astra_world import action_proposals
    from astra_world.astra import AstraAdapter, ProviderError
    response_goal = goal().model_dump(mode='json')
    async def respond(self, messages):
        return {'tool_calls': [{'function': {'name': 'submit_action_proposal', 'arguments': json.dumps({
            'status': 'ready', 'name': 'extract_blue', 'interpretation': 'Extract blue; red lands on tray.',
            'goal': response_goal,
        })}}]}
    monkeypatch.setattr(action_proposals._ProposalAdapter, 'respond', respond)
    adapter = AstraAdapter('https://example.test/v1', 'test', 'fixture')
    world = {'entities': [e.model_dump(mode='json') for e in scene().spec.entities]}
    proposal = asyncio.run(action_proposals.interpret_action(adapter, 'Extract blue', world, []))
    assert proposal.goal.preserve_ids == ['green', 'tray']
    response_goal['supported_id'] = 'missing'
    with pytest.raises(ProviderError):
        asyncio.run(action_proposals.interpret_action(adapter, 'Extract blue', world, []))
    response_goal['supported_id'] = 'red'
    next(e for e in world['entities'] if e['id'] == 'red')['fixed'] = True
    proposal = asyncio.run(action_proposals.interpret_action(adapter, 'Extract blue', world, []))
    assert proposal.status == 'unsupported'


def test_upper_motion_resets_extraction_settling_window():
    world = scene()
    evaluator = evaluator_for(world)
    shift(world, 'blue', -.12)
    advance(world, 1.1, evaluator)
    assert evaluator.success()
    world.data.joint('red_joint').qvel[3] = .3
    mujoco.mj_forward(world.model, world.data)
    assert not evaluator.success()
    assert evaluator.latest['settled_seconds'] == 0


@pytest.mark.parametrize('step', [
    {'op': 'pick_place', 'object_id': 'red', 'position': (.53, .35, .03)},
    {'op': 'gripper', 'opened': False, 'object_id': 'red'},
    {'op': 'contact_stroke', 'contact_ids': ['red'], 'direction': (1, 0, 0)},
])
def test_extraction_rejects_deliberate_upper_manipulation_before_spending_trial(tmp_path, step):
    lab = ActionLab(scene(), tmp_path)
    draft = lab.draft('extract_blue', goal())['payload']['draft_id']
    result = lab.test(draft, ActionProgram(steps=[step]))
    assert not result['ok'] and result['error_code'] == 'invalid_program'
    assert result['payload']['trials_used'] == 0
