"""Compile catalog assets into independent MuJoCo worlds without settling them."""

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .assets import PANDA
from .catalog import get_asset
from .scene import HOME
from .world_spec import WorldSpec

COLORS = {
    "red": (0.9, 0.16, 0.18, 1),
    "blue": (0.16, 0.39, 0.9, 1),
    "green": (0.1, 0.65, 0.43, 1),
    "yellow": (0.98, 0.8, 0.15, 1),
    "orange": (1, 0.45, 0.1, 1),
    "white": (0.92, 0.93, 0.95, 1),
    "gray": (0.55, 0.59, 0.65, 1),
    "black": (0.08, 0.1, 0.13, 1),
    "purple": (0.6, 0.25, 0.8, 1),
    "cyan": (0.1, 0.8, 0.85, 1),
    "brown": (0.45, 0.25, 0.12, 1),
}


def _numbers(values):
    return " ".join(str(float(value)) for value in values)


@dataclass
class GeneralWorld:
    model: mujoco.MjModel
    data: mujoco.MjData
    spec: WorldSpec
    revision: int = 1
    paths: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)

    @property
    def arm_q(self):
        if self.spec.robot == "none":
            return np.empty(0)
        return np.array([self.data.joint(f"joint{i}").qpos[0] for i in range(1, 8)])

    @property
    def block_ids(self):
        return []

    @staticmethod
    def body_name(entity_id):
        return f"entity_{entity_id}"

    @staticmethod
    def joint_name(entity_id):
        return f"{entity_id}_joint"

    def observe(self):
        mujoco.mj_forward(self.model, self.data)
        entities = []
        for entity in self.spec.entities:
            body = self.data.body(self.body_name(entity.id))
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body.id, velocity, 0
            )
            entities.append(
                {
                    "id": entity.id,
                    "asset_id": entity.asset_id,
                    "color": entity.color,
                    "scale": list(entity.scale),
                    "mass": float(self.model.body_mass[body.id]),
                    "position": body.xpos.tolist(),
                    "rotation": Rotation.from_matrix(body.xmat.reshape(3, 3))
                    .as_euler("XYZ")
                    .tolist(),
                    "velocity": velocity[3:].tolist(),
                    "fixed": bool(self.model.body_jntnum[body.id] == 0),
                }
            )
        return {
            "scene_revision": self.revision,
            "kind": "general",
            "name": self.spec.name,
            "seed": self.spec.seed,
            "sim_time": self.data.time,
            "entities": entities,
            "blocks": [],
            "bins": {},
            "obstacles": [],
            "robot": {"type": self.spec.robot, "joints": self.arm_q.tolist()},
            "paths": self.paths,
        }


def build_world(spec: WorldSpec, revision=1):
    # Validate again at the compilation boundary, including mutated entity lists.
    spec = WorldSpec.model_validate(spec.model_dump())
    if spec.robot == "panda":
        if not PANDA.exists():
            raise FileNotFoundError(
                "Run uv run python scripts/fetch_assets.py to install Panda assets."
            )
        root = ET.parse(PANDA).getroot()
        root.find("compiler").set("meshdir", str(PANDA.parent / "assets"))
        hand = root.find(".//body[@name='hand']")
        ET.SubElement(
            hand, "site", name="grasp", pos="0 0 .103", size=".003", rgba="0 0 0 0"
        )
        for node in root.findall(".//default[@class='collision']//geom"):
            node.set("friction", "1.2 .01 .001")
        keyframe = root.find("keyframe")
        if keyframe is not None:
            root.remove(keyframe)
    else:
        root = ET.Element("mujoco", model=spec.name)
        ET.SubElement(root, "compiler")
        ET.SubElement(root, "option")
        ET.SubElement(root, "worldbody")
    root.find("compiler").set("angle", "radian")
    root.find("compiler").set("eulerseq", "xyz")
    option = root.find("option")
    option.set("gravity", _numbers(spec.gravity))
    option.set("timestep", ".002")
    option.set("iterations", "80")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(visual, "headlight", diffuse=".7 .7 .7", ambient=".3 .3 .3")
    wb = root.find("worldbody")
    ET.SubElement(wb, "light", pos="0 0 5", dir="0 0 -1", diffuse=".8 .8 .8")
    ET.SubElement(
        wb,
        "geom",
        name="general_ground",
        type="plane",
        size="20 20 .1",
        rgba=".15 .19 .25 1",
    )
    asset_root = root.find("asset")
    if asset_root is None:
        asset_root = ET.SubElement(root, "asset")
    for entity in spec.entities:
        definition = get_asset(entity.asset_id)
        fixed = (
            entity.fixed if entity.fixed is not None else definition["default_fixed"]
        )
        scale = np.asarray(entity.scale)
        extent = np.abs(Rotation.from_euler("XYZ", entity.rotation).as_matrix()) @ (
            np.asarray(definition["bounds"]) * scale / 2
        )
        if np.any(np.abs(entity.position) + extent > 20):
            raise ValueError(
                f"Entity {entity.id} extends beyond the 20 meter world bounds"
            )
        body = ET.SubElement(
            wb,
            "body",
            name=f"entity_{entity.id}",
            pos=_numbers(entity.position),
            euler=_numbers(entity.rotation),
        )
        if not fixed:
            ET.SubElement(body, "freejoint", name=f"{entity.id}_joint")
        parts = definition["geometries"]
        mass = (
            entity.mass
            if entity.mass is not None
            else min(100, definition["default_mass"] * float(np.prod(scale)))
        )
        for index, part in enumerate(parts):
            kind = part["type"]
            part_rotation = Rotation.from_euler(
                "XYZ", part.get("euler", [0, 0, 0]), degrees=True
            ).as_matrix()
            # Entity-axis scaling must be expressed in each rotated part's axes.
            part_scaling = part_rotation.T @ np.diag(scale) @ part_rotation
            local_scale = np.diag(part_scaling)
            if not np.allclose(part_scaling, np.diag(local_scale), atol=1e-10):
                raise ValueError(
                    f"{entity.asset_id} scale would shear a rotated geometry; use uniform scale"
                )
            if kind == "capsule" and not np.allclose(local_scale, local_scale[0]):
                raise ValueError(
                    f"{entity.asset_id} requires uniform scale to preserve capsule geometry"
                )
            name = f"entity_{entity.id}_geom_{index}"
            attrs = {
                "name": name,
                "type": kind,
                "pos": _numbers(np.asarray(part.get("pos", [0, 0, 0])) * scale),
                "euler": _numbers(np.deg2rad(part.get("euler", [0, 0, 0]))),
                "rgba": _numbers(COLORS[entity.color]),
                "mass": str(mass / len(parts)),
                "friction": f"{entity.friction if entity.friction is not None else 0.8} .005 .0001",
            }
            size = np.asarray(part["size"])
            if kind == "mesh":
                ET.SubElement(
                    asset_root,
                    "mesh",
                    name=name + "_mesh",
                    file=part["file"],
                    scale=_numbers(size * local_scale),
                )
                attrs["mesh"] = name + "_mesh"
            elif kind == "sphere":
                # Ellipsoids preserve requested anisotropic scale.
                attrs["type"] = "ellipsoid"
                attrs["size"] = _numbers(size[0] * local_scale)
            elif kind in ("cylinder", "capsule"):
                if not np.isclose(local_scale[0], local_scale[1]):
                    raise ValueError(
                        f"{entity.asset_id} requires equal local x/y scale for circular cross sections"
                    )
                attrs["size"] = _numbers(
                    [size[0] * local_scale[0], size[1] * local_scale[2]]
                )
            else:
                attrs["size"] = _numbers(size * local_scale)
            if definition.get("visual_file"):
                attrs["rgba"] = "0 0 0 0"
                attrs["group"] = "3"
            ET.SubElement(body, "geom", **attrs)
        if definition.get("visual_file"):
            visual_name = f"entity_{entity.id}_visual"
            ET.SubElement(
                asset_root,
                "mesh",
                name=visual_name + "_mesh",
                file=definition["visual_file"],
                scale=_numbers(scale),
            )
            ET.SubElement(
                body,
                "geom",
                name=visual_name,
                type="mesh",
                mesh=visual_name + "_mesh",
                rgba=_numbers(COLORS[entity.color]),
                contype="0",
                conaffinity="0",
                group="2",
                mass="0",
            )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    if spec.robot == "panda":
        for i, q in enumerate(HOME, 1):
            data.joint(f"joint{i}").qpos[0] = q
            data.ctrl[i - 1] = q
        for name in ("finger_joint1", "finger_joint2"):
            data.joint(name).qpos[0] = 0.04
        data.ctrl[7] = 255
    mujoco.mj_forward(model, data)
    return GeneralWorld(model, data, spec, revision=revision)
