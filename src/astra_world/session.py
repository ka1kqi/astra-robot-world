"""Conversation-independent task memory and transactional scene changes."""

import mujoco
import numpy as np
from .contracts import ObstacleSpec
from .scene import build_scene, BINS
from .manipulation import sort_blocks


def capture(world):
    return {
        world.model.joint(i).name: (
            world.data.joint(i).qpos.copy(),
            world.data.joint(i).qvel.copy(),
        )
        for i in range(world.model.njnt)
    }


def restore(world, snapshot):
    for name, (qpos, qvel) in snapshot.items():
        world.data.joint(name).qpos[:] = qpos
        world.data.joint(name).qvel[:] = qvel
    world.data.ctrl[:7] = world.arm_q
    world.data.ctrl[7] = 255
    mujoco.mj_forward(world.model, world.data)


class Session:
    def __init__(self, world):
        self.world = world
        self.goal = None
        self.trial = None
        self.last_paths = []

    def result(self, ok, **kwargs):
        return {"ok": ok, "scene_revision": self.world.revision, **kwargs}

    def sort(self, goal, cancel=None, tick=None, status=None):
        self.goal, self.trial = goal, capture(self.world)
        result = sort_blocks(self.world, goal, cancel, tick, status)
        self.last_paths = list(self.world.paths)
        return result

    def add_obstacle(self, spec=None):
        if self.world.obstacles:
            return self.result(
                False,
                error_code="obstacle_limit",
                detail="This demo supports one obstacle. Build a new station to reset it.",
            )
        # Position the barrier across the previous source-to-left-bin corridor.
        sign = -1 if self.goal and self.goal.destination_id == "right_bin" else 1
        spec = spec or ObstacleSpec(
            position=(0.40, sign * 0.17, 0.19), half_extents=(0.075, 0.025, 0.19)
        )
        p, size = np.array(spec.position), np.array(spec.half_extents)
        try:
            for name in self.world.block_ids:
                positions = [self.world.data.body(name).xpos]
                if self.trial:
                    positions.append(self.trial[name + "_joint"][0][:3])
                if any(np.all(np.abs(pos - p) < size + 0.024) for pos in positions):
                    raise ValueError(
                        "Obstacle overlaps a current or restored block position."
                    )
            for center in BINS.values():
                if np.all(np.abs(center[:2] - p[:2]) < size[:2] + 0.15):
                    raise ValueError("Obstacle overlaps a bin.")
            candidate = build_scene(self.world.spec, [spec], self.world.revision + 1)
            restore(candidate, capture(self.world))
            obstacle_geom = candidate.model.geom("obstacle_0").id
            if any(
                obstacle_geom in c.geom and c.dist < -0.001
                for c in candidate.data.contact
            ):
                raise ValueError("Obstacle overlaps the robot or another object.")
            if self.trial:
                current = capture(candidate)
                restore(candidate, self.trial)
                if any(
                    obstacle_geom in c.geom and c.dist < -0.001
                    for c in candidate.data.contact
                ):
                    raise ValueError(
                        "Obstacle overlaps the robot at the start of the repeated trial."
                    )
                restore(candidate, current)
            self.world = candidate
            return self.result(
                True,
                payload={
                    "obstacle": spec.model_dump(),
                    "message": "Obstacle added. Try again to reset the blocks and repeat the sorting task.",
                },
            )
        except ValueError as exc:
            return self.result(False, error_code="invalid_placement", detail=str(exc))

    def retry(self, cancel=None, tick=None, status=None):
        if self.trial is None:
            return self.result(
                False,
                error_code="no_previous_task",
                detail="Run a sorting task before trying again.",
            )
        if status:
            status("Resetting the blocks for another attempt; keeping the obstacle.")
        restore(self.world, self.trial)
        self.world.revision += 1
        self.world.paths = []
        return sort_blocks(self.world, self.goal, cancel, tick, status)
