"""General scene lifecycle, bounded physics stepping, and reproducible snapshots."""

import json
from pathlib import Path

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .assets import ROOT
from .world_builder import build_world
from .world_spec import WorldSpec

SCENARIO_DIR = ROOT / "scenarios"


class JointState(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    qpos: list[float]
    qvel: list[float]


class SavedScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: int = Field(default=1, ge=1, le=1)
    spec: WorldSpec
    joints: dict[str, JointState]
    ctrl: list[float]
    sim_time: float = Field(ge=0)


class GeneralSession:
    def __init__(self, world):
        self.world = world
        self.last_paths = []

    def result(self, ok=True, **kwargs):
        return {"ok": ok, "scene_revision": self.world.revision, **kwargs}

    def simulate(self, duration, cancel=None, tick=None, status=None):
        world = self.world
        steps = max(1, round(duration / world.model.opt.timestep))
        start = world.data.time
        if status:
            status(f"Simulating {duration:g} seconds.")
        for _ in range(steps):
            if cancel and cancel.is_set():
                return self.result(
                    False,
                    error_code="cancelled",
                    detail="Simulation paused.",
                    payload={"duration": world.data.time - start},
                )
            if world.spec.robot == "panda":
                world.data.qfrc_applied[:7] = world.data.qfrc_bias[:7]
            mujoco.mj_step(world.model, world.data)
            if not np.isfinite(world.data.qpos).all():
                return self.result(
                    False,
                    error_code="unstable_physics",
                    detail="The world became unstable. Reset it before continuing.",
                )
            if tick:
                tick()
        return self.result(
            payload={"duration": world.data.time - start, "world": world.observe()}
        )

    def reset(self):
        self.world = build_world(self.world.spec, revision=self.world.revision + 1)
        return self.result(payload=self.world.observe())

    def add_entity(self, entity):
        source = self.world
        spec = source.spec.model_copy(
            update={"entities": [*source.spec.entities, entity]}
        )
        candidate = build_world(spec, revision=source.revision + 1)
        for i in range(source.model.njnt):
            name = source.model.joint(i).name
            candidate.data.joint(name).qpos[:] = source.data.joint(name).qpos
            candidate.data.joint(name).qvel[:] = source.data.joint(name).qvel
        for i in range(source.model.nu):
            actuator_id = candidate.model.actuator(source.model.actuator(i).name).id
            candidate.data.ctrl[actuator_id] = source.data.ctrl[i]
        candidate.data.time = source.data.time
        mujoco.mj_forward(candidate.model, candidate.data)
        body_id = candidate.model.body(candidate.body_name(entity.id)).id
        new_geoms = [
            g
            for g in range(candidate.model.ngeom)
            if candidate.model.geom_bodyid[g] == body_id
            and candidate.model.geom_contype[g]
        ]
        other_geoms = [
            g
            for g in range(candidate.model.ngeom)
            if candidate.model.geom_bodyid[g] != body_id
            and candidate.model.geom_contype[g]
        ]
        for first in new_geoms:
            for second in other_geoms:
                # Convex distance can return zero for coincident centers, including
                # deeply intersecting ellipsoids. Handle that degeneracy explicitly.
                coincident = (
                    candidate.model.geom_type[second] != mujoco.mjtGeom.mjGEOM_PLANE
                    and np.linalg.norm(
                        candidate.data.geom_xpos[first]
                        - candidate.data.geom_xpos[second]
                    )
                    < 1e-6
                )
                distance = mujoco.mj_geomDistance(
                    candidate.model, candidate.data, first, second, 0.002, None
                )
                if coincident or distance < -0.001:
                    raise ValueError(
                        "The new entity overlaps an existing object, robot, or ground."
                    )
        self.world = candidate
        return self.result(payload=candidate.observe())

    def save(self, name):
        w = self.world
        saved = SavedScenario(
            spec=w.spec,
            joints={
                w.model.joint(i).name: JointState(
                    qpos=w.data.joint(i).qpos.tolist(),
                    qvel=w.data.joint(i).qvel.tolist(),
                )
                for i in range(w.model.njnt)
            },
            ctrl=w.data.ctrl.tolist(),
            sim_time=w.data.time,
        )
        SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
        path = SCENARIO_DIR / (name + ".json")
        temp = path.with_suffix(".tmp")
        temp.write_text(saved.model_dump_json(indent=2) + "\n")
        temp.replace(path)
        return self.result(
            payload={
                "name": name,
                "path": str(path),
                "message": "Saved current scene and physics state.",
            }
        )


def load_scenario(name, revision):
    path = SCENARIO_DIR / (name + ".json")
    if not path.exists():
        raise ValueError(f"No saved scenario named {name}.")
    saved = SavedScenario.model_validate_json(path.read_text())
    world = build_world(saved.spec, revision=revision)
    expected = {world.model.joint(i).name for i in range(world.model.njnt)}
    if set(saved.joints) != expected or len(saved.ctrl) != world.model.nu:
        raise ValueError("Saved state does not match the scene joints or actuators.")
    for name, state in saved.joints.items():
        joint = world.data.joint(name)
        if len(state.qpos) != len(joint.qpos) or len(state.qvel) != len(joint.qvel):
            raise ValueError("Saved joint state has invalid dimensions.")
        if len(state.qpos) == 7 and not np.isclose(
            np.linalg.norm(state.qpos[3:]), 1, atol=1e-4
        ):
            raise ValueError("Saved object orientation is not a unit quaternion.")
        joint.qpos[:] = state.qpos
        joint.qvel[:] = state.qvel
    world.data.ctrl[:] = saved.ctrl
    world.data.time = saved.sim_time
    mujoco.mj_forward(world.model, world.data)
    return GeneralSession(world)
