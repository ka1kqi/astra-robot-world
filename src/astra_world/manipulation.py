"""Physical Panda manipulation. Objects move only through contact forces."""

import numpy as np
from .scene import BINS, bin_for
from .motion import Controller, MotionError, execute_transport


def wait_settled(ctl, name, destination_id):
    stable = 0.0
    elapsed = 0.0
    dt = ctl.world.model.opt.timestep
    while elapsed < 3.0:
        ctl.step(dt)
        elapsed += dt
        velocity = ctl.world.data.joint(name + "_joint").qvel
        if (
            bin_for(ctl.world, name) == destination_id
            and np.linalg.norm(velocity[:3]) < 0.02
            and np.linalg.norm(velocity[3:]) < 0.2
        ):
            stable += dt
            if stable >= 0.5:
                return
        else:
            stable = 0.0
    raise MotionError(
        "placement_failed",
        f"{name} did not remain settled inside the bin for half a second.",
    )


def sort_blocks(world, goal, cancel=None, tick=None, status=None):
    ctl = Controller(world, cancel, tick, status)
    completed = []
    routes = []
    try:
        names = [
            name
            for name, color in zip(world.block_ids, world.spec.block_colors)
            if color == goal.color
        ]
        center = BINS[goal.destination_id]
        slots = [
            center + np.array([x, y, 0.24])
            for y in (-0.04, 0.04)
            for x in (-0.043, 0.043)
        ]
        occupied = [
            world.data.body(n).xpos
            for n in world.block_ids
            if bin_for(world, n) == goal.destination_id
        ]
        slots = [
            slot
            for slot in slots
            if all(np.linalg.norm(slot[:2] - pos[:2]) > 0.055 for pos in occupied)
        ]
        remaining = [n for n in names if bin_for(world, n) != goal.destination_id]
        if len(remaining) > len(slots):
            raise MotionError(
                "bin_capacity",
                "The destination has insufficient free slots (four blocks per bin).",
            )
        for index, name in enumerate(names):
            if bin_for(world, name) == goal.destination_id:
                wait_settled(ctl, name, goal.destination_id)
                completed.append(name)
                continue
            ctl.status(f"Picking up {name.replace('_', ' ')}.")
            pos = world.data.body(name).xpos.copy()
            ctl.gripper(True)
            execute_transport(ctl, [pos[0], pos[1], 0.24])
            ctl.move([pos[0], pos[1], pos[2] + 0.004], 1.5)
            ctl.held_id = name
            ctl.gripper(False)
            ctl.move([pos[0], pos[1], 0.24], 1.5)
            if world.data.body(name).xpos[2] < 0.17:
                raise MotionError(
                    "grasp_failed", f"{name} did not lift with the gripper."
                )
            target = slots.pop(0)
            ctl.status("Carrying the block to the bin.")
            routes.append(execute_transport(ctl, target, name))
            ctl.move([target[0], target[1], 0.047], 1.5)
            ctl.gripper(True)
            ctl.held_id = None
            ctl.move(target)
            wait_settled(ctl, name, goal.destination_id)
            completed.append(name)
        ctl.status(
            f"Sorted {len(completed)} {goal.color} blocks into the {goal.destination_id.replace('_', ' ')}."
        )
        return {
            "ok": True,
            "scene_revision": world.revision,
            "payload": {"completed": completed, "routes": routes},
        }
    except MotionError as exc:
        world.data.ctrl[:7] = world.arm_q
        ctl.status(str(exc))
        return {
            "ok": False,
            "scene_revision": world.revision,
            "error_code": exc.code,
            "detail": str(exc),
            "payload": {"completed": completed, "routes": routes},
        }
