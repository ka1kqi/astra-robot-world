"""Strict command arguments and dispatch on the simulation owner thread."""
from .contracts import StrictModel, SceneSpec, SortGoal, ObstacleSpec
from .scene import build_scene, observe


class EmptyArgs(StrictModel):
    pass


ARGUMENTS = {
    'list_assets': EmptyArgs,
    'build_sorting_station': SceneSpec,
    'observe_world': EmptyArgs,
    'sort_blocks': SortGoal,
    'add_obstacle': ObstacleSpec,
    'retry_last_task': EmptyArgs,
    'stop': EmptyArgs,
}


def error(code, detail, revision=0):
    return {'ok': False, 'scene_revision': revision, 'error_code': code, 'detail': detail}


def dispatch(session, name, arguments, *, automatic_obstacle=False, cancel=None, tick=None, status=None):
    """Return the possibly replaced session and the action's measured result."""
    if name == 'build_sorting_station':
        from .session import Session
        session = Session(build_scene(arguments, revision=session.world.revision + 1))
        return session, {'ok': True, 'scene_revision': session.world.revision, 'payload': observe(session.world)}
    if name == 'sort_blocks':
        result = session.sort(arguments, cancel=cancel, tick=tick, status=status)
    elif name == 'add_obstacle':
        result = session.add_obstacle(None if automatic_obstacle else arguments)
    elif name == 'retry_last_task':
        result = session.retry(cancel=cancel, tick=tick, status=status)
    else:
        raise ValueError(f'Not a mutating command: {name}')
    return session, result
