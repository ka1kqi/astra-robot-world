"""Typed, bounded programs Astra can compose; never executable Python."""

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
import math

Number = Annotated[float, Field(allow_inf_nan=False)]
Vector = tuple[Number, Number, Number]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MotionStep(Contract):
    op: Literal["move_to_pose", "gripper", "contact_stroke", "wait", "pick_place"]
    position: Vector | None = None
    reference_id: Identifier | None = None
    rotation: Vector | None = None
    target_rotation: Vector | None = None
    opened: bool | None = None
    direction: Vector | None = None
    distance: float = Field(default=0.1, gt=0, le=0.25)
    speed: float = Field(default=0.06, ge=0.01, le=0.2)
    seconds: float = Field(default=1.0, ge=0.02, le=5.0)
    contact_ids: list[Identifier] = Field(default_factory=list, max_length=8)
    object_id: Identifier | None = None

    @model_validator(mode="after")
    def meaningful(self):
        if self.target_rotation is not None and self.op != "pick_place":
            raise ValueError("target_rotation is only allowed on pick_place; use rotation for gripper poses.")
        if self.op in ("move_to_pose", "pick_place") and self.position is None:
            raise ValueError("This step requires a position.")
        if self.position is not None and any(abs(x) > 2 for x in self.position):
            raise ValueError("Position or relative offset must be within two meters.")
        if self.op == "gripper" and self.opened is None:
            raise ValueError("Gripper requires opened true or false.")
        if self.op == "pick_place" and self.object_id is None:
            raise ValueError("Pick/place requires object_id.")
        if self.object_id is not None and self.op != "pick_place":
            if self.op != "gripper" or self.opened is not False:
                raise ValueError("A grasp object_id is only allowed when closing the gripper.")
        if self.op == "contact_stroke":
            if (
                not self.contact_ids
                or self.direction is None
                or math.sqrt(sum(x * x for x in self.direction)) < 1e-6
            ):
                raise ValueError(
                    "Contact stroke requires target contact_ids and nonzero direction."
                )
        elif self.contact_ids:
            raise ValueError(
                "contact_ids are only allowed during contact_stroke; use gripper opened=false with object_id for a grasp."
            )
        return self


class ActionProgram(Contract):
    version: Literal[1] = 1
    steps: list[MotionStep] = Field(min_length=1, max_length=24)


class GoalSpec(Contract):
    kind: Literal["topple", "displace", "circle", "extract", "rotate"]
    object_id: Identifier | None = None
    radius: float = Field(default=0.06, ge=0.02, le=0.12)
    plane: Literal["xy", "xz", "yz"] = "xy"
    support_id: Identifier | None = None
    supported_id: Identifier | None = None
    landing_id: Identifier | None = None
    min_displacement: float = Field(default=0.04, ge=0.01, le=0.5)
    target_position: Vector | None = None
    target_rotation: Vector | None = None
    angular_tolerance: float = Field(default=math.radians(5), ge=math.radians(.5), le=math.radians(15))
    tolerance: float = Field(default=0.04, ge=0.005, le=0.1)
    preserve_ids: list[Identifier] = Field(default_factory=list, max_length=64)
    preserve_tolerance: float = Field(default=0.02, ge=0.001, le=0.05)

    @model_validator(mode="after")
    def complete(self):
        if self.kind == "rotate":
            if self.target_rotation is None or self.support_id is not None:
                raise ValueError("Rotate requires target_rotation and cannot exempt a support object.")
        elif self.target_rotation is not None:
            raise ValueError("Only rotation goals can declare target_rotation.")
        if self.kind == "extract":
            roles = (self.object_id, self.supported_id, self.landing_id)
            if not all(roles) or len(set(roles)) != 3 or self.support_id:
                raise ValueError(
                    "Extract requires distinct object_id, supported_id, and landing_id; no support_id."
                )
            if {self.object_id, self.supported_id} & set(self.preserve_ids):
                raise ValueError(
                    "Extraction lower and upper objects cannot also be preserved."
                )
        elif self.supported_id or self.landing_id:
            raise ValueError(
                "Only extraction goals may declare supported_id and landing_id."
            )
        if self.kind == "circle" and (self.object_id or self.support_id):
            raise ValueError(
                "Circle goals cannot exempt scene objects from preservation."
            )
        if self.kind == "displace" and self.support_id:
            raise ValueError("Only a topple goal may declare a movable support.")
        if self.kind != "circle" and not self.object_id:
            raise ValueError("Object goals require object_id.")
        if self.kind == "circle" and self.target_position is None:
            raise ValueError("Circle requires a center target_position.")
        if self.kind == "topple" and (
            not self.support_id or self.support_id == self.object_id
        ):
            raise ValueError("Topple requires a distinct support_id.")
        if self.kind == "displace" and self.target_position is None:
            raise ValueError("Displace requires target_position.")
        return self


class DraftActionArgs(Contract):
    name: Identifier
    goal: GoalSpec
    trial_budget: int = Field(default=5, ge=3, le=10)


class TestActionArgs(Contract):
    draft_id: str = Field(min_length=1, max_length=64)
    program: ActionProgram


class SaveActionArgs(Contract):
    draft_id: str = Field(min_length=1, max_length=64)


class RunActionArgs(Contract):
    name: Identifier
    bindings: dict[Identifier, Identifier] = Field(default_factory=dict, max_length=64)


class SearchNotebookArgs(Contract):
    query: str = Field(default="", max_length=200)
    limit: int = Field(default=20, ge=1, le=50)


class ReadNotebookArgs(Contract):
    experiment_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    limit: int = Field(default=50, ge=1, le=50)
    after_id: int = Field(default=0, ge=0)


class NoteArgs(Contract):
    experiment_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    text: str = Field(min_length=1, max_length=3000)
    trial_number: int | None = Field(default=None, ge=1, le=10)
