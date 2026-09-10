import numpy as np
from astra_world.circle_measure import evaluate_circle
from astra_world.action_contracts import GoalSpec


def test_requires_measured_complete_circle_not_endpoint_or_half_arc():
    goal = GoalSpec(kind="circle", target_position=[0.45, 0, 0.3], radius=0.06)
    angles = np.linspace(0, 2 * np.pi, 101)
    circle = np.c_[
        0.45 + 0.06 * np.cos(angles), 0.06 * np.sin(angles), np.full(101, 0.3)
    ]
    assert evaluate_circle(circle, goal)["success"]
    assert not evaluate_circle(circle[:40], goal)["success"]
    assert not evaluate_circle(circle[[0, -1]], goal)["success"]
    assert not evaluate_circle(circle * 2, goal)["success"]


def test_approach_is_excluded_but_shortcut_is_rejected():
    goal = GoalSpec(kind="circle", target_position=[0.45, 0, 0.3], radius=0.06)
    angles = np.linspace(0, 2 * np.pi, 101)
    circle = np.c_[
        0.45 + 0.06 * np.cos(angles), 0.06 * np.sin(angles), np.full(101, 0.3)
    ]
    assert evaluate_circle(np.vstack([[0, 0, 1], circle]), goal)["success"]
    circle[50] = [0.45, 0, 0.3]
    assert not evaluate_circle(circle, goal)["success"]
