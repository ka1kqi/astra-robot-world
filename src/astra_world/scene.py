"""Scene compilation and observations. Coordinates are meters, Z up."""

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .assets import PANDA
from .contracts import SceneSpec, ObstacleSpec

HOME = np.array([0, -0.3, 0, -2.1, 0, 1.8, 0.785398])
COLORS = {"red": ".9 .16 .18 1", "blue": ".16 .39 .9 1", "green": ".1 .65 .43 1"}
BINS = {
    "left_bin": np.array([0.46, 0.40, 0.0]),
    "right_bin": np.array([0.46, -0.40, 0.0]),
}


@dataclass
class World:
    model: mujoco.MjModel
    data: mujoco.MjData
    spec: SceneSpec
    obstacles: list[ObstacleSpec]
    revision: int = 1
    paths: list = field(default_factory=list)

    @property
    def block_ids(self):
        return [f"block_{i}" for i in range(len(self.spec.block_colors))]

    @property
    def arm_q(self):
        return np.array([self.data.joint(f"joint{i}").qpos[0] for i in range(1, 8)])


def _geom(parent, name, pos, size, rgba, **attrs):
    ET.SubElement(
        parent,
        "geom",
        name=name,
        type="box",
        pos=" ".join(map(str, pos)),
        size=" ".join(map(str, size)),
        rgba=rgba,
        **attrs,
    )


def build_scene(spec=None, obstacles=None, revision=1):
    spec = spec or SceneSpec()
    obstacles = obstacles or []
    if not PANDA.exists():
        raise FileNotFoundError(
            "Run uv run python scripts/fetch_assets.py to install Panda assets."
        )
    root = ET.parse(PANDA).getroot()
    root.find("compiler").set("meshdir", str(PANDA.parent / "assets"))
    root.remove(root.find("keyframe"))
    option = root.find("option")
    option.set("timestep", ".002")
    option.set("cone", "elliptic")
    option.set("iterations", "80")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(visual, "headlight", diffuse=".7 .7 .7", ambient=".3 .3 .3")
    wb = root.find("worldbody")
    _geom(wb, "table", [0.35, 0, -0.035], [0.6, 0.62, 0.035], ".82 .85 .89 1")
    ET.SubElement(
        wb,
        "geom",
        name="floor",
        type="plane",
        pos="0 0 -.72",
        size="3 3 .1",
        rgba=".15 .19 .25 1",
    )
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(
        hand, "site", name="grasp", pos="0 0 .103", size=".003", rgba="0 0 0 0"
    )
    # Higher fingertip friction supports a real two-finger pinch of a light cube.
    for node in root.findall(".//default[@class='collision']//geom"):
        node.set("friction", "1.2 .01 .001")
    rng = np.random.default_rng(spec.seed)
    for i, color in enumerate(spec.block_colors):
        x = 0.36 + (i % 3) * 0.11 + rng.uniform(-0.004, 0.004)
        y = -0.14 + (i // 3) * 0.095 + rng.uniform(-0.004, 0.004)
        body = ET.SubElement(wb, "body", name=f"block_{i}", pos=f"{x} {y} .021")
        ET.SubElement(body, "freejoint", name=f"block_{i}_joint")
        _geom(
            body,
            f"block_{i}_geom",
            [0, 0, 0],
            [0.02] * 3,
            COLORS[color],
            mass=".04",
            friction="1.2 .01 .001",
            condim="4",
        )
    for name, pos in BINS.items():
        body = ET.SubElement(wb, "body", name=name, pos=" ".join(map(str, pos)))
        _geom(body, name + "_floor", [0, 0, 0.005], [0.14, 0.14, 0.005], ".3 .4 .55 1")
        for axis in (0, 1):
            for side in (-1, 1):
                p = [0, 0, 0.04]
                p[axis] = side * 0.135
                s = [0.145, 0.145, 0.04]
                s[axis] = 0.01
                _geom(body, f"{name}_wall_{axis}_{side}", p, s, ".4 .5 .66 1")
    for i, obstacle in enumerate(obstacles):
        p, s = np.array(obstacle.position), np.array(obstacle.half_extents)
        if (
            np.any(s <= 0)
            or np.any(s > 0.4)
            or not (0.15 < p[0] < 0.8)
            or abs(p[1]) > 0.48
            or abs(p[2] - s[2]) > 0.001
        ):
            raise ValueError(
                "Obstacle must rest on the tabletop, have positive half-extents <= 0.4 m, and be inside the work area."
            )
        if np.any(p[:2] - s[:2] < [-0.25, -0.62]) or np.any(
            p[:2] + s[:2] > [0.95, 0.62]
        ):
            raise ValueError("The entire obstacle must fit on the tabletop.")
        _geom(wb, f"obstacle_{i}", p, s, "1 .57 .12 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    world = World(model, data, spec, obstacles, revision)
    for i, q in enumerate(HOME, 1):
        data.joint(f"joint{i}").qpos[0] = q
        data.ctrl[i - 1] = q
    data.joint("finger_joint1").qpos[0] = 0.04
    data.joint("finger_joint2").qpos[0] = 0.04
    data.ctrl[7] = 255
    for _ in range(400):
        mujoco.mj_step(model, data)
    return world


def bin_for(world, block_id):
    pos = world.data.body(block_id).xpos
    rot = world.data.body(block_id).xmat.reshape(3, 3)
    extent = np.abs(rot) @ np.array([0.02] * 3)
    for name, center in BINS.items():
        if (
            np.all(np.abs(pos[:2] - center[:2]) + extent[:2] < 0.125)
            and 0.015 < pos[2] < 0.065
        ):
            return name
    return None


def observe(world):
    if hasattr(world, 'observe'):
        return world.observe()
    blocks = []
    for name, color in zip(world.block_ids, world.spec.block_colors):
        blocks.append(
            {
                "id": name,
                "color": color,
                "position": world.data.body(name).xpos.tolist(),
                "bin": bin_for(world, name),
            }
        )
    return {
        "scene_revision": world.revision,
        "seed": world.spec.seed,
        "sim_time": world.data.time,
        "blocks": blocks,
        "bins": {k: v.tolist() for k, v in BINS.items()},
        "obstacles": [o.model_dump() for o in world.obstacles],
        "robot": {"joints": world.arm_q.tolist()},
        "paths": world.paths,
    }
