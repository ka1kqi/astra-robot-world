import mujoco
import numpy as np
from fastapi.testclient import TestClient

from astra_world.action_lab import PhysicsSnapshot
from astra_world.astra import AstraAdapter
from astra_world.general_session import GeneralSession
from astra_world.server import create_app
from astra_world.simulation import SimulationRuntime
from astra_world.world_builder import build_world
from astra_world.world_spec import WorldSpec


def runtime():
    rt = SimulationRuntime(headless=True, render_frames=True)
    rt._session = GeneralSession(build_world(WorldSpec(name='replay', robot='panda', entities=[])))
    return rt


def test_recording_preserves_physics_and_separates_revalidation_and_no_motion():
    rt = runtime()
    w = rt._session.world
    before = PhysicsSnapshot.capture(w).state
    try:
        rt._arm_action_recording('run_action', {'name': 'test-motion'})
        np.testing.assert_array_equal(PhysicsSnapshot.capture(w).state, before)
        scratch = PhysicsSnapshot.capture(w).clone()
        rt._preview('begin', scratch, {'id': 'validation'})
        rt._preview('end', scratch, {'id': 'validation', 'goal_success': False})
        assert rt.action_replay_metadata()['clip'] is None
        rt._finish_action_recording({'ok': False})
        assert rt.action_replay_metadata()['clip'] is None
        rt._arm_action_recording('pick_place', {'object_id': 'cube'})
        mujoco.mj_step(w.model, w.data)
        expected = PhysicsSnapshot.capture(w).state
        rt._record_action_frame()
        rt._finish_action_recording({'ok': False, 'error_code': 'unexpected_contact'})
        np.testing.assert_array_equal(PhysicsSnapshot.capture(w).state, expected)
        clip = rt.action_replay_metadata()['clip']
        assert clip['tool'] == 'pick_place'
        assert clip['state'] == 'failed'
        assert clip['frame_count'] >= 2
        assert rt.action_replay_frame(clip['id'], 0).startswith(b'\xff\xd8')
        assert rt.action_replay_frame('stale', 0) is None
        rt._arm_action_recording('pick_place', {'object_id': 'missing'})
        rt._finish_action_recording({'ok': False})
        assert rt.action_replay_metadata()['clip']['id'] == clip['id']
        rt._arm_action_recording('check_approaches', {})
        assert rt._action_recording is None
    finally:
        if rt._renderer: rt._renderer.close()
        if rt._preview_renderer: rt._preview_renderer.close()


def test_recording_limits_disclose_truncation():
    from astra_world.experiment_preview import ExperimentClips
    clips = ExperimentClips(max_frames=2, max_trials=1, max_bytes=8)
    clips.begin({'id': 'one'})
    for i in range(4): clips.append('one', b'jpeg', i)
    clip = clips.metadata()['trials'][0]
    assert clip['truncated'] is True
    assert clip['dropped_frames'] == 2
    assert clip['frame_times'] == [2, 3]


def test_action_replay_http_is_read_only_and_validates_indices():
    rt = runtime()
    rt._action_clips.begin({'id': 'one', 'tool': 'pick_place'})
    rt._action_clips.append('one', b'jpeg', 0)
    before = PhysicsSnapshot.capture(rt._session.world).state
    with TestClient(create_app(rt, adapter=AstraAdapter()), base_url="http://127.0.0.1") as client:
        assert client.get('/action-replay').json()['clip']['id'] == 'one'
        assert client.get('/frame.jpg?view=action&clip_id=one&frame=0').content == b'jpeg'
        assert client.get('/frame.jpg?view=action&clip_id=stale&frame=0').status_code == 503
        for index in [-1, 300, 'abc']:
            assert client.get(f'/frame.jpg?view=action&frame={index}').status_code == 422
    np.testing.assert_array_equal(PhysicsSnapshot.capture(rt._session.world).state, before)


def test_owner_records_only_live_ticks_and_retains_clip_after_rejected_or_readonly_command(tmp_path, monkeypatch):
    from astra_world import action_lab, simulation
    from astra_world.rendering import WorldRenderer
    monkeypatch.setattr(action_lab, 'ACTIONS_DIR', tmp_path)
    monkeypatch.setattr(WorldRenderer, 'render', lambda self, world: f'frame:{world.data.time}'.encode())
    dispatch = simulation.dispatch

    def controlled_dispatch(session, name, arguments, **kwargs):
        if name != 'pick_place':
            return dispatch(session, name, arguments, **kwargs)
        scratch = PhysicsSnapshot.capture(session.world).clone()
        kwargs['preview']('begin', scratch, {'id': 'scratch'})
        kwargs['preview']('end', scratch, {'id': 'scratch', 'goal_success': False})
        if arguments.object_id == 'move':
            mujoco.mj_step(session.world.model, session.world.data)
            kwargs['tick']()
            return session, {'ok': True, 'scene_revision': session.world.revision}
        return session, {'ok': False, 'scene_revision': session.world.revision, 'error_code': 'unknown_object'}

    monkeypatch.setattr(simulation, 'dispatch', controlled_dispatch)
    rt = SimulationRuntime(headless=True, render_frames=True).start()
    try:
        assert rt.submit('create_world', {'name': 'copy', 'robot': 'panda', 'entities': []}).result(10)['ok']
        assert not rt.submit('pick_place', {'object_id': 'missing', 'target_position': [.4, 0, .3]}).result(10)['ok']
        assert rt.action_replay_metadata()['clip'] is None
        assert rt.submit('pick_place', {'object_id': 'move', 'target_position': [.4, 0, .3]}).result(10)['ok']
        clip = rt.action_replay_metadata()['clip']
        assert clip['state'] == 'succeeded'
        assert clip['frame_count'] >= 2
        assert rt.experiment_metadata()['latest_trial_id'] == 'scratch'
        assert rt.submit('simulate', {'duration': .02}).result(10)['ok']
        assert rt.action_replay_metadata()['clip']['id'] == clip['id']
        assert not rt.submit('pick_place', {'object_id': 'missing', 'target_position': [.4, 0, .3]}).result(10)['ok']
        assert rt.action_replay_metadata()['clip']['id'] == clip['id']
    finally:
        rt.close()
