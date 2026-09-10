"""Composable, physically validated pick/place for supported Panda props."""

import numpy as np
from .catalog import get_asset
from .motion import (
    Controller,
    MotionError,
    execute_transport,
    solve_ik,
    body_name,
    joint_name,
)


def pick_place(world, object_id, target_position, cancel=None, tick=None, status=None):
    command_trace = []

    def result(ok, **kwargs):
        kwargs["payload"] = {**kwargs.get("payload", {}), "command_trace": command_trace}
        return {"ok": ok, "scene_revision": world.revision, **kwargs}

    def command(op, parameters, function):
        entry = {"op": op, **parameters, "status": "started"}
        command_trace.append(entry)
        try:
            value = function()
        except MotionError:
            entry["status"] = "failed"
            raise
        entry["status"] = "completed"
        if value is not None:
            entry["result"] = value
        return value

    if world.spec.robot != "panda":
        return result(
            False,
            error_code="unsupported_robot",
            detail="Pick/place needs a Panda robot in this world.",
        )
    entity = next((e for e in world.spec.entities if e.id == object_id), None)
    if entity is None:
        return result(
            False, error_code="unknown_object", detail=f"No object named {object_id}."
        )
    asset = get_asset(entity.asset_id)
    dimensions = np.array(asset["bounds"]) * entity.scale
    fixed = entity.fixed if entity.fixed is not None else asset["default_fixed"]
    if (
        fixed
        or "pinch_candidate" not in asset.get("capabilities", [])
        or max(dimensions[:2]) > 0.06
        or dimensions[2] < 0.025
        or dimensions[2] > 0.06
        or (
            entity.mass
            if entity.mass is not None
            else asset["default_mass"] * np.prod(entity.scale)
        )
        > 0.1
    ):
        return result(
            False,
            error_code="unsupported_grasp",
            detail="This skill supports free, small pinch-candidate props, 2.5–6 cm tall and at most 100 g. Other assets can be simulated but do not have a validated grasp.",
        )
    target = np.asarray(target_position, dtype=float)
    if (
        target.shape != (3,)
        or not np.isfinite(target).all()
        or target[2] < dimensions[2] / 2 - 0.002
    ):
        return result(
            False,
            error_code="invalid_target",
            detail="Target is the object center and must be finite and above the ground.",
        )
    ctl = Controller(world, cancel, tick, status)

    def gripper(opened):
        return command("gripper", {"opened": opened}, lambda: ctl.gripper(opened))

    def move(position, seconds=1.2):
        return command("move", {"position": list(position), "seconds": seconds}, lambda: ctl.move(position, seconds))

    def transport(position, held_id=None):
        return command("execute_transport", {"position": list(position), "held_id": held_id}, lambda: execute_transport(ctl, position, held_id))
    body = body_name(world, object_id)
    try:
        # Validate target reachability before disturbing the source object.
        if solve_ik(world, target + [0, 0, 0.018], world.arm_q) is None:
            raise MotionError(
                "unreachable", "The requested placement is outside the Panda workspace."
            )
        rotation = world.data.body(body).xmat.reshape(3, 3)
        # A cube has six equivalent support faces; local +Z is not special.
        cube_on_face = (
            entity.asset_id == "small_box"
            and np.allclose(dimensions, dimensions[0], atol=1e-6, rtol=0)
            and np.max(np.abs(rotation[2])) >= 0.99
        )
        if rotation[2, 2] < 0.99 and not cube_on_face:
            raise MotionError(
                "unsupported_grasp", "This grasp requires an upright prop or a cube resting flat on any face."
            )
        pos = world.data.body(body).xpos.copy()
        height = max(0.24, pos[2] + 0.18, target[2] + 0.18)
        ctl.status(f"Reaching for {object_id}.")
        gripper(True)
        transport([pos[0], pos[1], height])
        move(pos + [0, 0, 0.004], 1.5)
        ctl.held_id = object_id
        gripper(False)
        move([pos[0], pos[1], height], 1.5)
        if world.data.body(body).xpos[2] < height - 0.06:
            raise MotionError(
                "grasp_failed", f"{object_id} did not lift with the fingers."
            )
        route = transport([target[0], target[1], height], object_id)
        move(target + [0, 0, 0.018], 1.5)
        gripper(True)
        ctl.held_id = None
        move([target[0], target[1], height])
        stable = 0.0
        for _ in range(round(3 / world.model.opt.timestep)):
            ctl.step(world.model.opt.timestep)
            actual = world.data.body(body).xpos
            velocity = world.data.joint(joint_name(world, object_id)).qvel
            if (
                np.linalg.norm(actual[:2] - target[:2]) < 0.025
                and abs(actual[2] - target[2]) < 0.015
                and np.linalg.norm(velocity[:3]) < 0.02
                and np.linalg.norm(velocity[3:]) < 0.2
            ):
                stable += world.model.opt.timestep
                if stable >= 0.5:
                    ctl.status(f"Placed {object_id}.")
                    return result(
                        True,
                        payload={
                            "object_id": object_id,
                            "position": actual.tolist(),
                            "route": route,
                            "settled_seconds": stable,
                        },
                    )
            else:
                stable = 0.0
        raise MotionError(
            "placement_failed",
            f"{object_id} did not settle at the requested position. It needs a stable support surface.",
        )
    except MotionError as exc:
        world.data.ctrl[:7] = world.arm_q
        ctl.status(str(exc))
        return result(False, error_code=exc.code, detail=str(exc))
