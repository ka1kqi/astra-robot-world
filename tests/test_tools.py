import time

import pytest


@pytest.fixture
def runtime():
    from astra_world.simulation import SimulationRuntime
    worker = SimulationRuntime(headless=True)
    worker.start()
    yield worker
    worker.close()


def test_invalid_commands_do_not_mutate_world(runtime):
    before = runtime.snapshot()
    for name, args in [('sort_blocks', {'color': 'pink'}),
                       ('build_sorting_station', {'surprise': 1}),
                       ('observe_world', {'seed': 3}), ('unknown', {})]:
        result = runtime.submit(name, args).result(timeout=1)
        assert not result['ok']
        assert result['error_code'] in {'invalid_arguments', 'unknown_tool'}
    assert runtime.snapshot() == before


def test_build_observe_and_snapshot_are_isolated(runtime):
    result = runtime.submit('build_sorting_station', {'seed': 11, 'block_colors': ['blue']}).result(timeout=10)
    assert result['ok']
    observation = runtime.submit('observe_world', {}).result(timeout=1)['payload']
    assert observation['seed'] == 11
    assert len(observation['blocks']) == 1
    observation['blocks'].clear()
    assert len(runtime.snapshot()['blocks']) == 1
    assert runtime.submit('list_assets', {}).result(timeout=1)['payload']['assets']


def test_busy_observation_and_prompt_stop(runtime):
    action = runtime.submit('sort_blocks', {'color': 'red', 'destination_id': 'left_bin'})
    busy = runtime.submit('build_sorting_station', {}).result(timeout=1)
    assert busy['error_code'] == 'busy'
    assert runtime.submit('observe_world', {}).result(timeout=1)['ok']
    deadline = time.monotonic() + 2
    while runtime.snapshot()['status'] == 'Ready' and time.monotonic() < deadline:
        time.sleep(.001)
    assert not action.done(), 'Expected an in-progress physical action before Stop'
    start = time.monotonic()
    assert runtime.stop()['ok']
    result = action.result(timeout=2)
    assert time.monotonic() - start < .25
    assert result['error_code'] == 'cancelled'
    assert runtime.submit('build_sorting_station', {}).result(timeout=10)['ok']


def test_shutdown_rejects_work(runtime):
    runtime.close()
    result = runtime.submit('sort_blocks', {}).result(timeout=1)
    assert result['error_code'] == 'shutdown'


def test_viewer_replacement_waits_for_macos_async_close(monkeypatch):
    from types import SimpleNamespace
    import mujoco.viewer
    from astra_world.simulation import SimulationRuntime
    runtime = SimulationRuntime()
    closed = []
    runtime._viewer = SimpleNamespace(close=lambda: closed.append(True))
    replacement = object()
    attempts = []

    def launch(model, data):
        attempts.append((model, data))
        if len(attempts) < 3:
            raise RuntimeError('another MuJoCo viewer is already open')
        return replacement

    monkeypatch.setattr(mujoco.viewer, 'launch_passive', launch)
    world = SimpleNamespace(model=object(), data=object())
    assert runtime._open_viewer(world)
    assert runtime._viewer is replacement
    assert closed == [True]
    assert len(attempts) == 3


def test_viewer_replacement_does_not_retry_unrelated_errors(monkeypatch):
    import mujoco.viewer
    from types import SimpleNamespace
    from astra_world.simulation import SimulationRuntime

    def fail(*args):
        raise RuntimeError('missing graphics context')

    monkeypatch.setattr(mujoco.viewer, 'launch_passive', fail)
    with pytest.raises(RuntimeError, match='missing graphics context'):
        SimulationRuntime()._open_viewer(SimpleNamespace(model=None, data=None))


def test_viewer_replacement_can_be_cancelled_before_reopening(monkeypatch):
    from types import SimpleNamespace
    from astra_world.simulation import SimulationRuntime
    runtime = SimulationRuntime()
    runtime._busy = True
    runtime.stop()
    assert not runtime._open_viewer(SimpleNamespace(model=None, data=None))
