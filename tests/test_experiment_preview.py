"""Visible scratch trials preserve live physics and retain bounded replay evidence."""
from threading import Thread
import time

import mujoco
import numpy as np
import pytest

from astra_world.action_contracts import ActionProgram, GoalSpec
from astra_world.action_lab import ActionLab, PhysicsSnapshot
from astra_world.general_session import GeneralSession
from astra_world.simulation import SimulationRuntime
from astra_world.world_builder import build_world
from astra_world.world_spec import EntitySpec, WorldSpec


def make_world():
    return build_world(WorldSpec(name='fall', robot='panda', entities=[
        EntitySpec(id='target', asset_id='small_box', position=(.4, -.12, .3)),
    ]))


def state(world):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(world.model, kind))
    mujoco.mj_getState(world.model, world.data, values, kind)
    return values


def draft(lab):
    return lab.draft('drop', GoalSpec(kind='displace', object_id='target',
                                    target_position=(.4, -.12, .02)))['payload']['draft_id']


def test_preview_observes_moving_scratch_and_captures_failure_then_success(tmp_path):
    world = make_world()
    before = state(world)
    events = []
    def preview(event, scratch, metadata):
        assert scratch.data is not world.data
        np.testing.assert_array_equal(state(world), before)
        events.append((event, float(scratch.data.time), metadata.copy()))
    lab = ActionLab(world, actions_dir=tmp_path, preview=preview)
    identifier = draft(lab)
    lab.notebook.add_note(identifier, 'Let the target settle under gravity.')
    lab.test(identifier, ActionProgram(steps=[{'op': 'wait', 'seconds': .02}]))
    result = lab.test(identifier, ActionProgram(steps=[{'op': 'wait', 'seconds': 1.5}]))
    assert result['payload']['verified']
    ends = [metadata for event, _, metadata in events if event == 'end']
    assert [item['goal_success'] for item in ends] == [False, True, True, True]
    assert len({item['id'] for item in ends}) == 4
    assert all(item['experiment_id'] == identifier for item in ends)
    assert all(item['hypothesis'] == 'Let the target settle under gravity.' for item in ends)
    assert any(t > .5 for event, t, _ in events if event == 'tick')
    np.testing.assert_array_equal(state(world), before)


def test_preview_failure_cannot_interrupt_measurement(tmp_path):
    def broken(*args):
        raise RuntimeError('renderer unavailable')
    lab = ActionLab(make_world(), actions_dir=tmp_path, preview=broken)
    result = lab.test(draft(lab), ActionProgram(steps=[{'op': 'wait', 'seconds': 1.5}]))
    assert result['payload']['verified']


def test_clip_store_bounds_frames_trials_bytes_and_copies_metadata():
    from astra_world.experiment_preview import ExperimentClips
    clips = ExperimentClips(max_bytes=12, max_trials=2, max_frames=2)
    clips.begin({'id': 'a', 'goal': {'kind': 'circle'}})
    for frame in (b'1111', b'2222', b'3333'):
        clips.append('a', frame, .1)
    assert clips.frame('a', 0) == b'2222'
    assert clips.frame('a', 1) == b'3333'
    clips.end('a', {'goal_success': False, 'error_code': 'goal_not_met', 'sim_seconds': .3})
    clips.begin({'id': 'b'})
    clips.append('b', b'4444', .1)
    clips.append('b', b'5555', .2)
    assert clips.frame('a') is None  # Byte cap evicts the older completed clip.
    clips.begin({'id': 'c'})
    clips.begin({'id': 'd'})
    metadata = clips.metadata()
    assert [trial['id'] for trial in metadata['trials']] == ['c', 'd']
    metadata['trials'][0]['state'] = 'corrupted'
    assert clips.metadata()['trials'][0]['state'] == 'testing'
    assert clips.frame('d', -1) is None
    clips.append('d', b'x' * 13, .1)
    assert not clips.metadata()['available']


def test_state_token_detects_physics_changes_without_revision_and_ignores_status(tmp_path):
    runtime = SimulationRuntime(headless=True)
    runtime._session = GeneralSession(make_world())
    runtime._session.action_lab = ActionLab(runtime._session.world, actions_dir=tmp_path)
    runtime._publish()
    first = runtime.snapshot()['state_token']
    runtime._status('New notes')
    assert runtime.snapshot()['state_token'] == first
    runtime._session.world.data.ctrl[0] += .01
    runtime._publish()
    second = runtime.snapshot()['state_token']
    assert second != first
    runtime._session.world.model.opt.gravity[0] += .01
    runtime._publish()
    third = runtime.snapshot()['state_token']
    assert third != second
    runtime._session.world.model.geom_friction[0, 0] += .01
    runtime._publish()
    assert runtime.snapshot()['state_token'] != third


def test_runtime_paces_preview_and_stop_interrupts_wait(tmp_path, monkeypatch):
    from astra_world.rendering import WorldRenderer
    monkeypatch.setattr(WorldRenderer, 'render', lambda self, world: b'jpeg')
    runtime = SimulationRuntime(headless=True, render_frames=True)
    runtime._session = GeneralSession(make_world())
    world = runtime._session.world
    scratch = PhysicsSnapshot.capture(world).clone()
    runtime._preview('begin', scratch, {'id': 'trial', 'experiment_id': 'experiment'})
    scratch.data.time += 2
    timer = Thread(target=lambda: (time.sleep(.05), runtime.stop()))
    timer.start()
    start = time.monotonic()
    runtime._preview('tick', scratch, {'id': 'trial'})
    elapsed = time.monotonic() - start
    timer.join()
    assert .03 <= elapsed < .5
    runtime._preview('end', scratch, {'id': 'trial', 'goal_success': False, 'error_code': 'cancelled', 'sim_seconds': 2})
    assert runtime.experiment_frame('trial') == b'jpeg'
    assert runtime.experiment_metadata()['trials'][0]['state'] == 'cancelled'
    runtime._preview_renderer.close()


def test_runtime_render_error_keeps_failed_trial_metadata(tmp_path, monkeypatch):
    from astra_world.rendering import WorldRenderer
    def unavailable(self, world):
        raise RuntimeError('no GL context')
    monkeypatch.setattr(WorldRenderer, 'render', unavailable)
    runtime = SimulationRuntime(headless=True, render_frames=True)
    runtime._session = GeneralSession(make_world())
    lab = ActionLab(runtime._session.world, actions_dir=tmp_path, preview=runtime._preview)
    result = lab.test(draft(lab), ActionProgram(steps=[{'op': 'wait', 'seconds': .02}]))
    assert result['payload']['trials_used'] == 1
    metadata = runtime.experiment_metadata()
    assert not metadata['available'] and 'no GL context' in metadata['error']
    assert metadata['trials'][0]['error_code'] == 'goal_not_met'


def test_owner_rejects_stale_token_even_when_cached_snapshot_matches(tmp_path, monkeypatch):
    from astra_world import action_lab
    monkeypatch.setattr(action_lab, 'ACTIONS_DIR', tmp_path)
    runtime = SimulationRuntime(headless=True).start()
    try:
        assert runtime.submit('create_world', make_world().spec.model_dump()).result(10)['ok']
        token = runtime.snapshot()['state_token']
        revision = runtime.snapshot()['scene_revision']
        assert runtime.submit('simulate', {'duration': .02}).result(10)['ok']
        assert runtime.snapshot()['scene_revision'] == revision
        with runtime._lock:
            runtime._snapshot['state_token'] = token  # A stale HTTP observation cannot authorize execution.
        result = runtime.submit('draft_action', {
            'name': 'drop', 'goal': {'kind': 'displace', 'object_id': 'target',
                                   'target_position': [.4, -.12, .02]},
        }, expected_state_token=token).result(10)
        assert not result['ok'] and result['error_code'] == 'stale_scene'
        assert runtime.snapshot()['action_lab']['drafts'] == []
    finally:
        runtime.close()


def test_actual_scratch_jpegs_remain_replayable_and_live_frame_stays_unchanged(tmp_path):
    from io import BytesIO
    from PIL import Image
    runtime = SimulationRuntime(headless=True, render_frames=True)
    runtime._session = GeneralSession(make_world())
    lab = ActionLab(runtime._session.world, actions_dir=tmp_path, preview=runtime._preview)
    runtime._session.action_lab = lab
    runtime._render_frame()
    live_frame = runtime.frame()
    before = state(runtime._session.world)
    runtime._publish()
    token = runtime.snapshot()['state_token']
    try:
        identifier = draft(lab)
        lab.test(identifier, ActionProgram(steps=[{'op': 'wait', 'seconds': .02}]))
        result = lab.test(identifier, ActionProgram(steps=[{'op': 'wait', 'seconds': .9}]))
        assert result['payload']['verified']
        trials = runtime.experiment_metadata()['trials']
        assert [trial['state'] for trial in trials] == ['failed', 'succeeded', 'succeeded', 'succeeded']
        for trial in trials:
            frame = runtime.experiment_frame(trial['id'])
            assert isinstance(frame, bytes)
            assert Image.open(BytesIO(frame)).format == 'JPEG'
            assert trial['frame_count'] >= 2
        assert runtime.experiment_frame(trials[1]['id'], 0) != runtime.experiment_frame(trials[1]['id'])
        assert runtime.frame() == live_frame
        np.testing.assert_array_equal(state(runtime._session.world), before)
        runtime._publish()
        assert runtime.snapshot()['state_token'] == token
    finally:
        if runtime._preview_renderer:
            runtime._preview_renderer.close()
        if runtime._renderer:
            runtime._renderer.close()


def test_status_publication_preserves_solver_warmstart_and_applied_forces():
    runtime = SimulationRuntime(headless=True)
    runtime._session = GeneralSession(make_world())
    world = runtime._session.world
    world.data.qacc_warmstart[:] = .125
    world.data.qfrc_applied[:] = .01
    world.data.xfrc_applied[:] = .002
    before = state(world)
    runtime._status('Recording a trial note')
    np.testing.assert_array_equal(state(world), before)


def test_status_observation_cannot_apply_control_callback_to_live_state():
    runtime = SimulationRuntime(headless=True)
    runtime._session = GeneralSession(make_world())
    before = state(runtime._session.world)
    def control(model, data):
        data.ctrl[0] += .01
    mujoco.set_mjcb_control(control)
    try:
        runtime._status('Recording a trial note')
    finally:
        mujoco.set_mjcb_control(None)
    np.testing.assert_array_equal(state(runtime._session.world), before)


def test_append_to_retained_clip_never_evicts_itself_or_breaks_byte_accounting():
    from astra_world.experiment_preview import ExperimentClips
    clips = ExperimentClips(max_bytes=8, max_trials=2, max_frames=3)
    clips.begin({'id': 'old'})
    clips.append('old', b'1111', .1)
    clips.begin({'id': 'new'})
    clips.append('new', b'2222', .1)
    clips.append('old', b'3333', .2)
    assert clips.frame('old') == b'3333'
    assert clips.frame('new') is None
    assert clips.metadata()['trials'][0]['frame_count'] == 2
