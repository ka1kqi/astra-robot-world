"""Isolated, bounded physical trials for typed reusable robot actions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Literal
from uuid import uuid4

import mujoco
import numpy as np
from pydantic import Field, model_validator
from scipy.spatial.transform import Rotation

from .action_contracts import (
    ActionProgram,
    Contract,
    DraftActionArgs,
    GoalSpec,
    Identifier,
    RunActionArgs,
)
from .action_motion import execute_program
from .circle_measure import evaluate_circle
from .world_builder import GeneralWorld
from .world_spec import WorldSpec
from .action_notebook import ActionNotebook

ACTIONS_DIR = Path(__file__).resolve().parents[2] / "actions"
MAX_SIM_SECONDS = 30.0
MAX_WALL_SECONDS = 300.0
SETTLED_SECONDS = 0.5
_STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def _state(world):
    state = np.empty(mujoco.mj_stateSize(world.model, _STATE))
    mujoco.mj_getState(world.model, world.data, state, _STATE)
    return state


@dataclass(frozen=True)
class PhysicsSnapshot:
    spec_json: str
    state: np.ndarray
    revision: int
    model: mujoco.MjModel

    @classmethod
    def capture(cls, world):
        state = _state(world)
        state.setflags(write=False)
        return cls(
            world.spec.model_dump_json(), state, world.revision, deepcopy(world.model)
        )

    def clone(self):
        model = deepcopy(self.model)
        world = GeneralWorld(
            model,
            mujoco.MjData(model),
            WorldSpec.model_validate_json(self.spec_json),
            self.revision,
        )
        mujoco.mj_setState(world.model, world.data, self.state, _STATE)
        # Forward computes observations, but can change the solver warm start.
        mujoco.mj_forward(world.model, world.data)
        mujoco.mj_setState(world.model, world.data, self.state, _STATE)
        return world


class ExtractionMeasurements(Contract):
    upper_supported_by_lower: bool
    upper_on_landing: bool
    lower_displacement: float = Field(ge=0)
    horizontal_separation: float = Field(ge=0)
    upper_position: list[float] = Field(min_length=3, max_length=3)
    upper_displacement: float = Field(ge=0)
    upper_linear_speed: float = Field(ge=0)
    upper_angular_speed: float = Field(ge=0)


class Measurements(Contract):
    position: list[float]
    rotation: list[float]
    displacement: float
    rotation_change: float
    linear_speed: float
    angular_speed: float
    settled_seconds: float
    on_ground: bool
    supported_by_support: bool
    preserved: bool
    preservation_displacements: dict[str, float]
    goal_score: float
    distance_to_goal: float | None = None
    extraction: ExtractionMeasurements | None = None
    circle: dict | None = None
    trace: list[list[float]] = Field(default_factory=list, max_length=256)


class TrialReport(Contract):
    trial_number: int = Field(ge=1)
    confirmation: bool
    ok: bool
    goal_success: bool
    error_code: str | None = None
    detail: str | None = None
    measurements: Measurements
    motion: dict = Field(default_factory=dict)
    sim_seconds: float = Field(ge=0, le=MAX_SIM_SECONDS + 0.01)


class SavedAction(Contract):
    experiment_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    version: Literal[1] = 1
    name: Identifier
    goal: GoalSpec
    program: ActionProgram
    verified: Literal[True] = True
    tested_on: WorldSpec
    initial_positions: dict[str, list[float]]
    verification: list[TrialReport] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def confirmed(self):
        _validate_goal_program(self.goal, self.program)
        for report in self.verification:
            measured = report.measurements
            if self.goal.kind == "circle":
                achieved = measured.circle is not None and measured.circle.get(
                    "success"
                )
            else:
                achieved = measured.settled_seconds + 1e-9 >= SETTLED_SECONDS
            if self.goal.kind == "extract":
                extraction = measured.extraction
                achieved = achieved and extraction is not None and (
                    extraction.lower_displacement >= self.goal.min_displacement
                    and extraction.horizontal_separation >= self.goal.min_displacement
                    and extraction.upper_on_landing
                    and not extraction.upper_supported_by_lower
                    and extraction.upper_linear_speed <= 0.015
                    and extraction.upper_angular_speed <= 0.2
                    and measured.linear_speed <= 0.015
                    and measured.angular_speed <= 0.2
                    and (
                        self.goal.target_position is None
                        or (
                            measured.distance_to_goal is not None
                            and measured.distance_to_goal <= self.goal.tolerance
                        )
                    )
                )
            if not (
                report.ok and report.goal_success and measured.preserved and achieved
            ):
                raise ValueError(
                    "Saved actions require three successful physical verification reports."
                )
        ids = {entity.id for entity in self.tested_on.entities}
        references = _references(self.program) | set(self.goal.preserve_ids)
        if self.goal.object_id:
            references.add(self.goal.object_id)
        for identifier in (
            self.goal.support_id, self.goal.supported_id, self.goal.landing_id
        ):
            if identifier:
                references.add(identifier)
        if not references <= ids or set(self.initial_positions) != ids:
            raise ValueError("Saved references must match the tested scene.")
        if any(
            len(position) != 3 or not np.isfinite(position).all()
            for position in self.initial_positions.values()
        ):
            raise ValueError("Saved initial positions must be finite vectors.")
        return self


def _references(program):
    return {
        identifier
        for step in program.steps
        for identifier in [step.reference_id, step.object_id, *step.contact_ids]
        if identifier
    }


def _validate_goal_program(goal, program):
    if goal.kind == "extract" and any(
        step.object_id == goal.supported_id or goal.supported_id in step.contact_ids
        for step in program.steps
    ):
        raise ValueError(
            "Extraction must manipulate the lower object; the supported upper object may only move passively."
        )


def _movable_goal_ids(goal):
    return {goal.object_id, goal.support_id, goal.supported_id} - {None}


def _rests_on(world, upper, lower):
    """A near-contact normal must support the upper body against gravity."""
    for contact in world.data.contact:
        a, b = (int(world.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        if contact.dist <= 0.002:
            normal_z = float(contact.frame[2])
            if (a == lower and b == upper and normal_z > 0.5) or (
                b == lower and a == upper and normal_z < -0.5
            ):
                return True
    return False


@dataclass
class Draft:
    draft_id: str
    name: str
    goal: GoalSpec
    initial: PhysicsSnapshot
    positions: dict
    trial_budget: int
    started: float = field(default_factory=time.monotonic)
    state: str = "draft"
    program: ActionProgram | None = None
    verified: bool = False
    trials: list = field(default_factory=list)
    confirmations: list = field(default_factory=list)
    best_result: dict | None = None

    def view(self):
        return {
            "draft_id": self.draft_id,
            "name": self.name,
            "goal": self.goal.model_dump(mode="json"),
            "trial_budget": self.trial_budget,
            "trials_used": len(self.trials),
            "trials_remaining": self.trial_budget - len(self.trials),
            "state": self.state,
            "verified": self.verified,
            "program": self.program.model_dump(mode="json") if self.program else None,
            "best_result": deepcopy(self.best_result),
            "trials": deepcopy(self.trials),
        }


class _Stop:
    """Event-compatible cancellation combines the user's Stop and wall deadline."""

    def __init__(self, cancel, deadline):
        self.cancel, self.deadline = cancel, deadline

    def is_set(self):
        return (
            bool(self.cancel and self.cancel.is_set())
            or time.monotonic() >= self.deadline
        )


class _Evaluator:
    def __init__(self, world, goal, positions):
        self.world, self.goal, self.positions = world, goal, positions
        self.target = world.model.body(
            "hand" if goal.kind == "circle" else world.body_name(goal.object_id)
        ).id
        self.initial_position = (
            world.data.site("grasp").xpos
            if goal.kind == "circle"
            else world.data.body(self.target).xpos
        ).copy()
        self.trace = [self.initial_position.tolist()] if goal.kind == "circle" else []
        self.support = (
            world.model.body(world.body_name(goal.support_id)).id
            if goal.support_id
            else None
        )
        self.upper = (
            world.model.body(world.body_name(goal.supported_id)).id
            if goal.supported_id
            else None
        )
        self.landing = (
            world.model.body(world.body_name(goal.landing_id)).id
            if goal.landing_id
            else None
        )
        self.ground = world.model.geom("general_ground").id
        self.initial_rotation = world.data.body(self.target).xmat.reshape(3, 3).copy()
        self.settled_since = None
        self.max_displacements = {identifier: 0.0 for identifier in goal.preserve_ids}
        self.latest = self.measure()

    def measure(self):
        world, goal = self.world, self.goal
        body = world.data.body(self.target)
        position = (
            world.data.site("grasp").xpos if goal.kind == "circle" else body.xpos
        ).copy()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            world.model, world.data, mujoco.mjtObj.mjOBJ_BODY, self.target, velocity, 0
        )
        linear_speed, angular_speed = (
            float(np.linalg.norm(velocity[3:])),
            float(np.linalg.norm(velocity[:3])),
        )
        on_ground, supported = False, False
        for contact in world.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            bodies = {int(world.model.geom_bodyid[a]), int(world.model.geom_bodyid[b])}
            if (
                self.target not in bodies
                or contact.dist > 0.002
                or abs(float(contact.frame[2])) < 0.5
            ):
                continue
            on_ground |= self.ground in (a, b)
            if self.support in bodies:
                supported |= world.data.body(self.support).xpos[2] < position[2]
        for identifier in goal.preserve_ids:
            displacement = float(
                np.linalg.norm(
                    world.data.body(world.body_name(identifier)).xpos
                    - self.positions[identifier]
                )
            )
            self.max_displacements[identifier] = max(
                self.max_displacements[identifier], displacement
            )
        preserved = all(
            distance <= goal.preserve_tolerance
            for distance in self.max_displacements.values()
        )
        distance = (
            float(np.linalg.norm(position - goal.target_position))
            if goal.kind in ("displace", "extract") and goal.target_position is not None
            else None
        )
        extraction = None
        if goal.kind == "extract":
            upper = world.data.body(self.upper)
            upper_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                world.model, world.data, mujoco.mjtObj.mjOBJ_BODY,
                self.upper, upper_velocity, 0
            )
            lower_displacement = float(np.linalg.norm(position - self.initial_position))
            separation = float(np.linalg.norm(position[:2] - upper.xpos[:2]))
            extraction = {
                "upper_supported_by_lower": _rests_on(world, self.upper, self.target),
                "upper_on_landing": _rests_on(world, self.upper, self.landing),
                "lower_displacement": lower_displacement,
                "horizontal_separation": separation,
                "upper_position": upper.xpos.tolist(),
                "upper_displacement": float(
                    np.linalg.norm(upper.xpos - self.positions[goal.supported_id])
                ),
                "upper_linear_speed": float(np.linalg.norm(upper_velocity[3:])),
                "upper_angular_speed": float(np.linalg.norm(upper_velocity[:3])),
            }
            predicate = (
                lower_displacement >= goal.min_displacement
                and separation >= goal.min_displacement
                and not extraction["upper_supported_by_lower"]
                and extraction["upper_on_landing"]
                and (distance is None or distance <= goal.tolerance)
            )
        elif goal.kind == "topple":
            predicate = on_ground and not supported
        elif goal.kind == "displace":
            predicate = distance <= goal.tolerance
        else:
            predicate = False
        stable = linear_speed <= 0.015 and angular_speed <= 0.2
        if extraction is not None:
            stable = (
                stable
                and extraction["upper_linear_speed"] <= 0.015
                and extraction["upper_angular_speed"] <= 0.2
            )
        now = float(world.data.time)
        if predicate and stable and preserved:
            if self.settled_since is None:
                self.settled_since = now
        else:
            self.settled_since = None
        duration = (
            0.0 if self.settled_since is None else max(0.0, now - self.settled_since)
        )
        if goal.kind == "topple":
            score = (
                0.35 * (not supported)
                + 0.35 * on_ground
                + 0.3 * min(1.0, duration / SETTLED_SECONDS)
            )
        elif goal.kind == "extract":
            score = (
                0.2 * min(1.0, extraction["lower_displacement"] / goal.min_displacement)
                + 0.2 * min(1.0, extraction["horizontal_separation"] / goal.min_displacement)
                + 0.15 * (not extraction["upper_supported_by_lower"])
                + 0.15 * extraction["upper_on_landing"]
                + 0.3 * min(1.0, duration / SETTLED_SECONDS)
            )
        elif goal.kind == "circle":
            score = 0.0
        else:
            initial_distance = float(
                np.linalg.norm(
                    np.asarray(self.positions[goal.object_id]) - goal.target_position
                )
            )
            score = 0.7 * max(
                0.0, 1.0 - distance / max(initial_distance, goal.tolerance)
            ) + 0.3 * min(1.0, duration / SETTLED_SECONDS)
        rotation = Rotation.from_matrix(body.xmat.reshape(3, 3))
        return Measurements(
            position=position.tolist(),
            rotation=rotation.as_euler("XYZ").tolist(),
            displacement=float(np.linalg.norm(position - self.initial_position)),
            rotation_change=float(
                (
                    rotation * Rotation.from_matrix(self.initial_rotation).inv()
                ).magnitude()
            ),
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            settled_seconds=duration,
            on_ground=bool(on_ground),
            supported_by_support=bool(supported),
            preserved=preserved,
            preservation_displacements=dict(self.max_displacements),
            goal_score=float(score if preserved else 0.0),
            distance_to_goal=distance,
            extraction=extraction,
        ).model_dump(mode="json")

    def tick(self):
        self.latest = self.measure()
        if self.goal.kind == "circle":
            self.trace.append(self.latest["position"])

    def success(self):
        self.tick()
        if self.goal.kind == "circle":
            result = evaluate_circle(self.trace, self.goal)
            self.latest["circle"] = result
            self.latest["trace"] = [
                self.trace[index]
                for index in np.linspace(
                    0, len(self.trace) - 1, min(256, len(self.trace)), dtype=int
                )
            ]
            self.latest["goal_score"] = (
                min(1.0, result["swept_radians"] / result["required_radians"])
                if self.latest["preserved"]
                else 0.0
            )
            return self.latest["preserved"] and result["success"]
        return (
            self.latest["preserved"]
            and self.latest["settled_seconds"] + 1e-9 >= SETTLED_SECONDS
        )


def _goal_context(world, goal):
    if not isinstance(world, GeneralWorld) or world.spec.robot != "panda":
        return "unsupported_world", "Action Lab requires a general scene with Panda."
    entities = {entity.id: entity for entity in world.spec.entities}
    required = set(goal.preserve_ids)
    if goal.object_id:
        required.add(goal.object_id)
    required.update(
        identifier
        for identifier in (goal.support_id, goal.supported_id, goal.landing_id)
        if identifier
    )
    if not required <= entities.keys():
        return "unknown_entity", "The goal references an entity outside this scene."
    if goal.kind == "circle":
        return None
    target = world.model.body(world.body_name(goal.object_id))
    if world.model.body_jntnum[target.id] == 0:
        return "fixed_target", "A fixed target cannot be displaced by robot contact."
    if goal.kind == "extract":
        upper = world.model.body(world.body_name(goal.supported_id))
        if world.model.body_jntnum[upper.id] == 0:
            return "fixed_target", "The supported upper object must be movable to settle on the landing surface."
    if goal.object_id in goal.preserve_ids:
        return "invalid_goal", "The displaced target cannot also be preserved."
    return None


class ActionLab:
    def __init__(self, world, actions_dir=None, preview=None):
        self.world = world
        self.preview = preview
        self.actions_dir = Path(actions_dir) if actions_dir is not None else ACTIONS_DIR
        self._drafts: dict[str, Draft] = {}
        self.notebook = ActionNotebook(self.actions_dir / "notebook.sqlite3")

    def _result(self, ok, payload=None, error_code=None, detail=None, world=None):
        result = {
            "ok": bool(ok),
            "scene_revision": (world or self.world).revision,
            "payload": payload or {},
        }
        if error_code:
            result.update(error_code=error_code, detail=detail or error_code)
        return result

    def snapshot(self):
        return {"drafts": [draft.view() for draft in self._drafts.values()]}

    def _latest_note(self, experiment_id):
        text, after_id = None, 0
        while entries := self.notebook.read(experiment_id, limit=100, after_id=after_id):
            for entry in entries:
                if entry["kind"] == "note" and entry["source"] == "astra":
                    text = entry["payload"].get("text")
            after_id = entries[-1]["id"]
        return text

    def draft(self, name, goal, trial_budget=5):
        try:
            args = DraftActionArgs(name=name, goal=goal, trial_budget=trial_budget)
        except ValueError as exc:
            return self._result(False, error_code="invalid_goal", detail=str(exc))
        error = _goal_context(self.world, args.goal)
        if error:
            return self._result(False, error_code=error[0], detail=error[1])
        initial = PhysicsSnapshot.capture(self.world)
        scratch = initial.clone()
        goal = args.goal.model_copy(deep=True)
        # The server fixes preservation for every unrelated entity, even if omitted by Astra.
        goal.preserve_ids = sorted(
            set(goal.preserve_ids)
            | {
                entity.id
                for entity in scratch.spec.entities
                if entity.id not in _movable_goal_ids(goal)
            }
        )
        positions = {
            entity.id: scratch.data.body(scratch.body_name(entity.id)).xpos.tolist()
            for entity in scratch.spec.entities
        }
        evaluator = _Evaluator(scratch, goal, positions)
        if goal.kind == "extract" and not evaluator.latest["extraction"][
            "upper_supported_by_lower"
        ]:
            return self._result(
                False, error_code="precondition_failed",
                detail="The supported upper object must initially rest on the lower extraction object.",
            )
        if goal.kind == "topple" and not evaluator.latest["supported_by_support"]:
            return self._result(
                False,
                error_code="precondition_failed",
                detail="The upper target must initially rest on the declared support.",
            )
        if (
            goal.kind == "displace"
            and evaluator.latest["distance_to_goal"] <= goal.tolerance
        ):
            return self._result(
                False,
                error_code="goal_already_satisfied",
                detail="The target already occupies the requested region.",
            )
        draft = Draft(
            uuid4().hex, args.name, goal, initial, positions, args.trial_budget
        )
        self.notebook.record(
            draft.draft_id,
            "draft",
            {
                "name": draft.name,
                "goal": goal.model_dump(mode="json"),
                "scene": scratch.spec.model_dump(mode="json"),
                "initial_positions": positions,
                "trial_budget": trial_budget,
            },
        )
        self._drafts[draft.draft_id] = draft
        return self._result(True, draft.view())

    def _execute(
        self,
        world,
        goal,
        positions,
        program,
        stop,
        *,
        tick=None,
        status=None,
        trial_number=1,
        confirmation=False,
        preview_metadata=None,
    ):
        evaluator = _Evaluator(world, goal, positions)
        start = float(world.data.time)
        preview_failed = False

        def preview(event, report=None):
            nonlocal preview_failed
            if self.preview and preview_metadata and not preview_failed:
                try:
                    self.preview(event, world, {**preview_metadata, **(report or {})})
                except Exception:
                    # A display failure cannot alter a physical result.
                    preview_failed = True
                    logging.getLogger(__name__).exception("Trial preview failed")

        def progress():
            evaluator.tick()
            preview("tick")
            if tick:
                tick()

        preview("begin")
        try:
            motion = execute_program(
                world,
                program,
                cancel=stop,
                tick=progress,
                status=status,
                max_seconds=MAX_SIM_SECONDS,
            )
        except Exception as exc:
            motion = {
                "ok": False,
                "error_code": "execution_error",
                "detail": str(exc),
                "payload": {},
            }
        success = evaluator.success()
        elapsed = float(world.data.time - start)
        if elapsed > MAX_SIM_SECONDS + float(world.model.opt.timestep):
            motion = {
                "ok": False,
                "error_code": "sim_budget",
                "detail": "Trial exceeded its simulated time budget.",
                "payload": {},
            }
        if stop.is_set():
            reason = "wall_budget" if time.monotonic() >= stop.deadline else "cancelled"
            motion = {
                **motion,
                "ok": False,
                "error_code": reason,
                "detail": "Experiment stopped.",
            }
        ok = bool(motion.get("ok"))
        report = {
            "trial_number": trial_number,
            "confirmation": confirmation,
            "ok": ok,
            "goal_success": ok and success,
            "error_code": motion.get("error_code")
            if not ok
            else (None if success else "goal_not_met"),
            "detail": motion.get("detail")
            if not ok
            else (
                None
                if success
                else "Physical outcome did not meet the fixed goal; include at least 0.5 seconds settled at the goal."
            ),
            "measurements": evaluator.latest,
            "motion": motion.get("payload", {}),
            "sim_seconds": elapsed,
        }
        preview("end", report)
        return report

    def test(self, draft_id, program, cancel=None, status=None):
        draft = self._drafts.get(draft_id)
        if draft is None:
            return self._result(False, error_code="unknown_draft")
        if draft.state in ("cancelled", "exhausted", "saved", "verified"):
            return self._result(
                False, draft.view(), "draft_closed", "This draft has finished."
            )
        try:
            program = ActionProgram.model_validate(
                program.model_dump() if isinstance(program, ActionProgram) else program
            )
            _validate_goal_program(draft.goal, program)
            ids = {
                entity.id
                for entity in WorldSpec.model_validate_json(
                    draft.initial.spec_json
                ).entities
            }
            if not _references(program) <= ids:
                raise ValueError(
                    "Program references an entity outside the captured scene."
                )
        except ValueError as exc:
            return self._result(False, draft.view(), "invalid_program", str(exc))
        stop = _Stop(cancel, draft.started + MAX_WALL_SECONDS)
        if stop.is_set():
            draft.state = "cancelled" if cancel and cancel.is_set() else "exhausted"
            self.notebook.record(
                draft_id, "state", {"state": draft.state, "verified": False}
            )
            return self._result(
                False,
                draft.view(),
                "cancelled" if draft.state == "cancelled" else "wall_budget",
            )
        draft.program, draft.verified, draft.confirmations = program, False, []
        for confirmation in (False, True, True):
            if len(draft.trials) >= draft.trial_budget or stop.is_set():
                draft.state = "cancelled" if cancel and cancel.is_set() else "exhausted"
                break
            draft.state = "testing"
            self.notebook.record(
                draft_id,
                "candidate",
                {
                    "trial_number": len(draft.trials) + 1,
                    "confirmation": confirmation,
                    "program": program.model_dump(mode="json"),
                },
            )
            if status:
                status(
                    f"Action Lab trial {len(draft.trials) + 1}/{draft.trial_budget}"
                    + (" — confirmation" if confirmation else "")
                )
            report = self._execute(
                draft.initial.clone(),
                draft.goal,
                draft.positions,
                program,
                stop,
                status=status,
                trial_number=len(draft.trials) + 1,
                confirmation=confirmation,
                preview_metadata={
                    "id": uuid4().hex,
                    "experiment_id": draft_id,
                    "name": draft.name,
                    "goal": draft.goal.model_dump(mode="json"),
                    "trial_number": len(draft.trials) + 1,
                    "trial_budget": draft.trial_budget,
                    "confirmation": confirmation,
                    "hypothesis": self._latest_note(draft_id),
                },
            )
            draft.trials.append(report)
            self.notebook.record(
                draft_id,
                "trial",
                {
                    "trial_number": report["trial_number"],
                    "program": program.model_dump(mode="json"),
                    "report": report,
                },
            )
            if (
                draft.best_result is None
                or report["measurements"]["goal_score"]
                > draft.best_result["measurements"]["goal_score"]
            ):
                draft.best_result = deepcopy(report)
            if not report["goal_success"]:
                draft.state = (
                    "cancelled"
                    if report["error_code"] == "cancelled"
                    else (
                        "exhausted"
                        if len(draft.trials) >= draft.trial_budget
                        or report["error_code"] == "wall_budget"
                        else "failed"
                    )
                )
                break
            draft.confirmations.append(report)
            if len(draft.confirmations) == 3:
                draft.verified, draft.state = True, "verified"
            if status:
                status(
                    "Action Lab measured goal achieved."
                    if draft.verified
                    else "Candidate passed; checking a fresh snapshot."
                )
        self.notebook.record(
            draft_id,
            "state",
            {
                "state": draft.state,
                "verified": draft.verified,
                "trials_used": len(draft.trials),
            },
        )
        if status:
            status(
                "Action Lab verified."
                if draft.verified
                else f"Action Lab {draft.state}."
            )
        return self._result(True, draft.view())

    def save(self, draft_id):
        draft = self._drafts.get(draft_id)
        if draft is None:
            return self._result(False, error_code="unknown_draft")
        if not draft.verified or draft.program is None:
            return self._result(
                False,
                draft.view(),
                "not_verified",
                "Three successful runs are required before saving.",
            )
        try:
            record = SavedAction(
                experiment_id=draft_id,
                name=draft.name,
                goal=draft.goal,
                program=draft.program,
                tested_on=WorldSpec.model_validate_json(draft.initial.spec_json),
                initial_positions=draft.positions,
                verification=draft.confirmations,
            )
            self.actions_dir.mkdir(parents=True, exist_ok=True)
            destination = self.actions_dir / f"{draft.name}.json"
            temporary = self.actions_dir / f".{draft.name}.{uuid4().hex}.tmp"
            temporary.write_text(
                json.dumps(record.model_dump(mode="json"), indent=2, allow_nan=False)
            )
            temporary.replace(destination)
        except (OSError, ValueError) as exc:
            return self._result(False, error_code="save_failed", detail=str(exc))
        self.notebook.record(
            draft_id, "saved", {"name": draft.name, "file": f"{draft.name}.json"}
        )
        draft.state = "saved"
        return self._result(
            True, {"action": record.model_dump(mode="json"), "draft_id": draft_id}
        )

    def _load(self, path):
        if path.is_symlink() or path.stat().st_size > 2_000_000:
            raise ValueError("Action file is not a bounded local JSON record.")
        action = SavedAction.model_validate_json(path.read_text())
        if path.stem != action.name:
            raise ValueError("Action name must match its file.")
        return action

    def list_actions(self):
        actions, invalid = [], []
        for path in sorted(self.actions_dir.glob("*.json")):
            try:
                actions.append(self._load(path).model_dump(mode="json"))
            except (OSError, ValueError) as exc:
                invalid.append({"name": path.stem, "error": str(exc)})
        return self._result(True, {"actions": actions, "invalid_actions": invalid})

    def run(self, world, name, bindings=None, cancel=None, tick=None, status=None):
        try:
            args = RunActionArgs(name=name, bindings=bindings or {})
            saved = self._load(self.actions_dir / f"{args.name}.json")
            source_ids = {entity.id for entity in saved.tested_on.entities}
            if not set(args.bindings) <= source_ids:
                raise ValueError("Bindings contain unknown saved entity IDs.")
            mapped = {
                identifier: args.bindings.get(identifier, identifier)
                for identifier in source_ids
            }
            if len(set(mapped.values())) != len(mapped):
                raise ValueError("Each saved entity requires a distinct binding.")
            if (
                not isinstance(world, GeneralWorld)
                or world.spec.robot != saved.tested_on.robot
                or world.spec.gravity != saved.tested_on.gravity
            ):
                raise ValueError("Robot and gravity must match the tested scene.")
            current = {entity.id: entity for entity in world.spec.entities}
            for entity in saved.tested_on.entities:
                target = current.get(mapped[entity.id])
                if target is None:
                    raise ValueError(f"Missing entity binding: {mapped[entity.id]}")
                fields = ("asset_id", "scale", "mass", "friction", "fixed")
                if any(getattr(entity, key) != getattr(target, key) for key in fields):
                    raise ValueError(f"Untested physical geometry for {target.id}.")
            goal_data = saved.goal.model_dump()
            for key in ("object_id", "support_id", "supported_id", "landing_id"):
                if goal_data[key]:
                    goal_data[key] = mapped[goal_data[key]]
            goal_data["preserve_ids"] = sorted(
                {mapped[identifier] for identifier in goal_data["preserve_ids"]}
                | {
                    identifier
                    for identifier in current
                    if identifier
                    not in (goal_data["object_id"], goal_data["support_id"], goal_data["supported_id"])
                }
            )
            goal = GoalSpec.model_validate(goal_data)
            program_data = saved.program.model_dump()
            for step in program_data["steps"]:
                for key in ("reference_id", "object_id"):
                    if step[key]:
                        step[key] = mapped[step[key]]
                step["contact_ids"] = [
                    mapped[identifier] for identifier in step["contact_ids"]
                ]
            program = ActionProgram.model_validate(program_data)
        except (OSError, ValueError) as exc:
            return self._result(
                False, error_code="precondition_failed", detail=str(exc), world=world
            )
        error = _goal_context(world, goal)
        if error:
            return self._result(
                False, error_code=error[0], detail=error[1], world=world
            )
        initial = PhysicsSnapshot.capture(world)
        scratch = initial.clone()
        positions = {
            entity.id: scratch.data.body(scratch.body_name(entity.id)).xpos.tolist()
            for entity in scratch.spec.entities
        }
        evaluator = _Evaluator(scratch, goal, positions)
        if goal.kind == "extract" and not evaluator.latest["extraction"][
            "upper_supported_by_lower"
        ]:
            return self._result(
                False, error_code="precondition_failed",
                detail="The supported upper object must initially rest on the lower extraction object.",
            )
        if goal.kind == "topple" and not evaluator.latest["supported_by_support"]:
            return self._result(
                False,
                error_code="precondition_failed",
                detail="The target is no longer supported by the declared support.",
                world=world,
            )
        if (
            goal.kind == "displace"
            and evaluator.latest["distance_to_goal"] <= goal.tolerance
        ):
            return self._result(
                False,
                error_code="goal_already_satisfied",
                detail="The target already occupies the requested region.",
                world=world,
            )
        stop = _Stop(cancel, time.monotonic() + MAX_WALL_SECONDS)
        if status:
            status(
                "Checking the saved action against the current scene in an isolated trial."
            )
        trial = self._execute(
            scratch, goal, positions, program, stop, status=status,
            preview_metadata={
                "id": uuid4().hex, "experiment_id": saved.experiment_id,
                "name": name, "goal": goal.model_dump(mode="json"),
                "trial_number": 1, "trial_budget": 1, "confirmation": True,
                "revalidation": True,
            },
        )
        if not trial["goal_success"]:
            return self._result(
                False,
                {"trial": trial},
                trial["error_code"] or "revalidation_failed",
                "Saved action did not pass in the current scene.",
                world=world,
            )
        if (
            initial.spec_json != world.spec.model_dump_json()
            or initial.revision != world.revision
            or not np.array_equal(initial.state, _state(world))
        ):
            return self._result(
                False,
                {"trial": trial},
                "stale_scene",
                "The live scene changed during revalidation.",
                world=world,
            )
        if stop.is_set():
            return self._result(False, error_code="cancelled", world=world)
        if status:
            status("Running the verified action in the live scene.")
        live = self._execute(
            world, goal, positions, program, stop, tick=tick, status=status
        )
        return self._result(
            live["goal_success"],
            {
                "name": name,
                "verified": live["goal_success"],
                "trial": trial,
                "result": live,
            },
            None if live["goal_success"] else live["error_code"],
            live.get("detail"),
            world=world,
        )
