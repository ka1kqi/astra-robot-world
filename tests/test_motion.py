import numpy as np
from threading import Event
from astra_world.scene import build_scene
from astra_world.contracts import ObstacleSpec
from astra_world.motion import plan_transport, Controller, MotionError
import pytest


def test_blocked_target_has_no_route_without_mutation():
    w = build_scene(
        obstacles=[
            ObstacleSpec(position=(0.4, 0.12, 0.19), half_extents=(0.075, 0.025, 0.19))
        ]
    )
    before = w.data.qpos.copy()
    assert plan_transport(w, [0.4, 0.12, 0.24]) is None
    assert np.array_equal(before, w.data.qpos)


def test_cancel_holds_arm():
    w = build_scene()
    event = Event()
    event.set()
    with pytest.raises(MotionError, match="stopped"):
        Controller(w, event).step(2)
    np.testing.assert_allclose(w.data.ctrl[:7], w.arm_q)
