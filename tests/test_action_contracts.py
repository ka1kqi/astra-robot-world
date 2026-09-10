import pytest
from pydantic import ValidationError
from astra_world.action_contracts import ActionProgram, MotionStep, GoalSpec


def test_rejects_unsafe_or_incomplete_steps():
    for step in [
        dict(
            op="contact_stroke",
            direction=[1, 0, 0],
            distance=0.5,
            contact_ids=["green"],
        ),
        dict(
            op="contact_stroke",
            direction=[0, 0, 0],
            distance=0.1,
            contact_ids=["green"],
        ),
        dict(op="move_to_pose", position=[float("nan"), 0, 0]),
        dict(op="gripper"),
        dict(op="gripper", opened=True, object_id="green"),
        dict(op="move_to_pose", position=[0.4, 0, 0.2], object_id="green"),
        dict(op="eval", code="pass"),
    ]:
        with pytest.raises(ValidationError):
            MotionStep.model_validate(step)


def test_program_and_goals_are_bounded():
    with pytest.raises(ValidationError):
        ActionProgram(steps=[])
    with pytest.raises(ValidationError):
        GoalSpec(kind="topple", object_id="green")
    with pytest.raises(ValidationError):
        GoalSpec(kind="displace", object_id="green")
    p = ActionProgram(
        steps=[
            dict(op="move_to_pose", reference_id="green", position=[0, -0.1, 0.06]),
            dict(op="gripper", opened=False),
            dict(
                op="contact_stroke",
                direction=[0, 1, 0],
                distance=0.15,
                contact_ids=["green"],
            ),
        ]
    )
    assert len(p.steps) == 3


def test_goal_cannot_exempt_unrelated_objects():
    with pytest.raises(ValidationError):
        GoalSpec(
            kind="circle",
            target_position=[0.45, 0, 0.3],
            object_id="red",
            support_id="green",
        )
    with pytest.raises(ValidationError):
        GoalSpec(
            kind="displace",
            target_position=[0.45, 0, 0.03],
            object_id="red",
            support_id="green",
        )
