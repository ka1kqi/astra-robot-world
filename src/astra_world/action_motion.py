"""Bounded typed actions, driven exclusively through the Panda actuators."""

import math

import mujoco
import numpy as np
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from .action_contracts import ActionProgram
from .approaches import collision_details, path_obstruction
from .motion import (
    DOWN,
    Controller,
    MotionError,
    body_name,
    colliding,
    solve_ik,
)


class ActionController(Controller):
    def __init__(self, world, cancel=None, tick=None, status=None, max_seconds=30):
        super().__init__(world, cancel, tick, status)
        self.started = float(world.data.time)
        self.deadline = self.started + max_seconds
        self.contact_ids = ()
        self.max_contact_force = 0.0
        self.diagnostics = {}
        names = [
            "hand",
            "left_finger",
            "right_finger",
            *[f"link{j}" for j in range(1, 8)],
        ]
        self.moving_bodies = {world.model.body(name).id for name in names}

    def guard(self):
        if self.cancel.is_set():
            raise MotionError(
                "cancelled", "Action stopped; holding the current arm position."
            )
        if self.world.data.time >= self.deadline - 1e-9:
            raise MotionError("time_limit", "Action reached its simulated time limit.")

    def check_loads(self):
        w = self.world
        # Aggregate loads across every robot contact; static base/ground support
        # is excluded. A compound object can generate multiple contact points.
        load = 0.0
        contact_loads = []
        force = np.zeros(6)
        for i, contact in enumerate(w.data.contact):
            if not any(
                int(w.model.geom_bodyid[g]) in self.moving_bodies for g in contact.geom
            ):
                continue
            mujoco.mj_contactForce(w.model, w.data, i, force)
            magnitude = float(np.linalg.norm(force[:3]))
            load += magnitude
            contact_loads.append((magnitude, i))
        self.max_contact_force = max(self.max_contact_force, load)
        if load > 40:
            self.diagnostics["force_contacts"] = [
                {
                    "force_newtons": magnitude,
                    "body_pair": [w.model.body(int(w.model.geom_bodyid[g])).name for g in w.data.contact[index].geom],
                    "position": w.data.contact[index].pos.tolist(),
                }
                for magnitude, index in sorted(contact_loads, reverse=True)[:5]
            ]
            raise MotionError(
                "contact_force", "Robot contact load exceeded the 40 N action limit."
            )

    def step(self, seconds):
        w = self.world
        for _ in range(max(1, math.ceil(seconds / w.model.opt.timestep))):
            self.guard()
            w.data.qfrc_applied[:7] = w.data.qfrc_bias[:7]
            mujoco.mj_step(w.model, w.data)
            if not np.isfinite(w.data.qpos).all() or not np.isfinite(w.data.qvel).all():
                raise MotionError("unstable_physics", "The simulation became unstable.")
            contact = colliding(w, w.data, self.held_id, self.contact_ids)
            if contact:
                self.diagnostics["collision"] = collision_details(w, w.data, contact)
                raise MotionError(
                    "unexpected_contact",
                    f"Unexpected contact between {contact[0]} and {contact[1]}.",
                )
            self.check_loads()
            self.tick()

    def pose(self, position, rotation, speed):
        self.guard()
        self.diagnostics["requested_pose"] = {
            "position": np.asarray(position).tolist(),
            "rotation_matrix": np.asarray(rotation).tolist(),
            "coordinate_frame": "world",
        }
        q = solve_ik(self.world, position, self.world.arm_q, rotation)
        if q is None:
            raise MotionError(
                "unreachable", "No joint solution was found for this position and wrist orientation within the bounded IK search. Try another wrist orientation or approach; this does not prove the object is unreachable."
            )
        obstruction = path_obstruction(self.world, [q], self.held_id)
        if obstruction:
            self.diagnostics.update(obstruction)
            pair = obstruction["collision"]["geometry_pair"]
            raise MotionError("no_path", f"The approach path is obstructed by contact between {pair[0]} and {pair[1]}.")
        distance = np.linalg.norm(
            np.asarray(position) - self.world.data.site("grasp").xpos
        )
        self.move_q(q, max(0.3, 1.5 * distance / speed))

    def grasp(self, opened, object_id=None):
        if object_id is not None:
            if self.held_id is not None and self.held_id != object_id:
                raise MotionError("invalid_grasp", "Release the held object before grasping another.")
            body = self.world.model.body(body_name(self.world, object_id))
            if body.jntnum == 0:
                raise MotionError("fixed_target", "Cannot grasp a fixed object.")
            self.held_id = object_id
        if opened:
            # Released objects may fall onto supports. Permit only residual
            # finger contact while opening, without treating the object as held.
            self.contact_ids = (self.held_id,) if self.held_id else ()
            self.held_id = None
            try:
                self.gripper(True)
            finally:
                self.contact_ids = ()
            return
        self.gripper(False)
        if self.held_id is not None:
            world = self.world
            target = world.model.body(body_name(world, self.held_id)).id
            fingers = {world.model.body(n).id for n in ("left_finger", "right_finger")}
            touching = set()
            for contact in world.data.contact:
                a, b = (int(world.model.geom_bodyid[g]) for g in contact.geom)
                if contact.dist <= 0 and target in (a, b):
                    touching.update({a, b} & fingers)
            if touching != fingers:
                raise MotionError("grasp_failed", "Both fingers must contact the declared grasp target.")

    def stroke(self, direction, distance, speed, rotation, contact_ids):
        w = self.world
        start = w.data.site("grasp").xpos.copy()
        direction = np.asarray(direction) / np.linalg.norm(direction)
        count = max(1, math.ceil(distance / 0.002))
        self.contact_ids = tuple(contact_ids)
        try:
            q = w.arm_q
            for i in range(1, count + 1):
                self.guard()
                position = start + direction * distance * i / count
                self.diagnostics.update(
                    stroke_sample_index=i,
                    requested_pose={
                        "position": position.tolist(),
                        "rotation_matrix": np.asarray(rotation).tolist(),
                        "coordinate_frame": "world",
                    },
                )
                target = solve_ik(w, position, q, rotation)
                if target is None:
                    raise MotionError(
                        "unreachable",
                        "No joint solution was found at this stroke sample. Try a different wrist orientation, stroke direction, or starting pose.",
                    )
                duration = max(
                    distance / count / speed, float(np.max(np.abs(target - q))) / 0.65
                )
                steps = max(1, math.ceil(duration / w.model.opt.timestep))
                for j in range(1, steps + 1):
                    w.data.ctrl[:7] = q + (target - q) * j / steps
                    self.step(w.model.opt.timestep)
                    if np.max(np.abs(w.arm_q - w.data.ctrl[:7])) > 0.1:
                        raise MotionError(
                            "tracking_error", "Arm could not track the contact stroke."
                        )
                q = target
            # Allow actuator lag to settle while declared finger contact remains valid.
            self.step(0.15)
            if np.linalg.norm(w.data.site("grasp").xpos - position) > 0.015:
                raise MotionError(
                    "tracking_error", "Fingertips could not reach the stroke endpoint."
                )
        finally:
            self.contact_ids = ()


def execute_program(
    world, program, cancel=None, tick=None, status=None, max_seconds=30
):
    """Execute relative poses against one immutable initial observation.

    Rotations are intrinsic XYZ Euler radians, matching WorldSpec. Omitted
    approach orientation is downward; omitted stroke orientation holds its
    initial orientation. Named gripper grasps permit target/finger contact until
    release; contact_stroke allows its declared finger contacts for that stroke.
    """
    ctl = None
    completed = 0

    def result(ok, **kwargs):
        return dict(
            ok=ok,
            scene_revision=world.revision,
            payload={
                "completed_steps": completed,
                "program": program.model_dump(mode="json") if isinstance(program, ActionProgram) else None,
                "elapsed_seconds": float(world.data.time) - ctl.started if ctl else 0.0,
                "max_contact_force": ctl.max_contact_force if ctl else 0.0,
                "diagnostics": ctl.diagnostics if ctl and not ok else {},
            },
            **kwargs,
        )

    try:
        program = ActionProgram.model_validate(
            program.model_dump() if isinstance(program, ActionProgram) else program
        )
        if not math.isfinite(max_seconds) or max_seconds <= 0 or max_seconds > 30:
            raise MotionError(
                "invalid_program",
                "Execution duration must be positive and at most 30 seconds.",
            )
        if getattr(getattr(world, "spec", None), "robot", None) != "panda":
            raise MotionError(
                "unsupported_robot",
                "Generated actions require a general world with a Panda robot.",
            )
        initial = {
            e["id"]: np.asarray(e["position"]) for e in world.observe()["entities"]
        }
        for step in program.steps:
            for name in [step.reference_id, step.object_id, *step.contact_ids]:
                if name is not None and name not in initial:
                    raise MotionError("unknown_object", f"No object named {name}.")
            for name in step.contact_ids:
                if world.model.body(body_name(world, name)).jntnum == 0:
                    raise MotionError(
                        "fixed_target", f"Contact target {name} is fixed."
                    )
            if step.position is not None:
                position = np.asarray(step.position) + (
                    initial[step.reference_id] if step.reference_id else 0
                )
                if np.any(np.abs(position) > 2) or position[2] < 0:
                    raise MotionError(
                        "invalid_target",
                        "Resolved pose must lie above ground within two meters of the origin.",
                    )
        ctl = ActionController(world, cancel, tick, status, max_seconds)
        for index, step in enumerate(program.steps):
            ctl.diagnostics = {"failed_step_index": index, "op": step.op}
            ctl.guard()
            ctl.status(f"Step {index + 1}/{len(program.steps)}: {step.op}.")
            rotation = (
                Rotation.from_euler("XYZ", step.rotation).as_matrix()
                if step.rotation is not None
                else None
            )
            if step.op in ("move_to_pose", "pick_place"):
                position = np.asarray(step.position) + (
                    initial[step.reference_id] if step.reference_id else 0
                )
            if step.op == "move_to_pose":
                ctl.pose(position, DOWN if rotation is None else rotation, step.speed)
            elif step.op == "contact_stroke":
                ctl.stroke(
                    step.direction,
                    step.distance,
                    step.speed,
                    world.data.site("grasp").xmat.reshape(3, 3).copy()
                    if rotation is None
                    else rotation,
                    step.contact_ids,
                )
            elif step.op == "gripper":
                ctl.grasp(step.opened, step.object_id)
            elif step.op == "wait":
                ctl.step(step.seconds)
            elif step.op == "pick_place":
                from .general_manipulation import pick_place

                if ctl.held_id is not None:
                    raise MotionError("invalid_grasp", "Release the custom grasp before using pick_place.")

                def bounded_tick():
                    ctl.check_loads()
                    ctl.tick()
                    ctl.guard()

                placed = pick_place(
                    world,
                    step.object_id,
                    position,
                    ctl.cancel,
                    bounded_tick,
                    ctl.status,
                )
                if not placed["ok"]:
                    raise MotionError(placed["error_code"], placed["detail"])
            completed += 1
        return result(True)
    except (ValidationError, TypeError, ValueError) as exc:
        return result(False, error_code="invalid_program", detail=str(exc))
    except MotionError as exc:
        if ctl is not None:
            world.data.ctrl[:7] = world.arm_q
            ctl.status(str(exc))
        return result(False, error_code=exc.code, detail=str(exc))
