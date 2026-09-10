import mujoco
import numpy as np
import pytest
from pydantic import ValidationError


def entity(asset_id="ball", **kwargs):
    from astra_world.world_spec import EntitySpec

    return EntitySpec(id="test_object", asset_id=asset_id, position=(0, 0, 1), **kwargs)


def test_falling_ball_preserves_initial_pose_and_observes_motion():
    from astra_world.world_spec import WorldSpec
    from astra_world.world_builder import build_world

    world = build_world(WorldSpec(name="fall", entities=[entity()]), revision=3)
    assert world.data.time == 0
    assert world.observe()["entities"][0]["position"] == [0, 0, 1]
    for _ in range(100):
        mujoco.mj_step(world.model, world.data)
    observation = world.observe()
    assert observation["entities"][0]["position"][2] < 0.9
    assert observation["entities"][0]["velocity"][2] < -1
    assert observation["scene_revision"] == 3
    assert world.arm_q.size == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": (0, 1, 1)},
        {"position": (float("nan"), 0, 0)},
        {"mass": 101},
        {"friction": -1},
        {"asset_id": "unknown"},
        {"id": "bad id"},
    ],
)
def test_rejects_invalid_entity(kwargs):
    from astra_world.world_spec import EntitySpec

    with pytest.raises(ValidationError):
        EntitySpec(
            **dict({"id": "a", "asset_id": "ball", "position": (0, 0, 1)}, **kwargs)
        )


def test_rejects_duplicate_ids_and_too_many_entities():
    from astra_world.world_spec import WorldSpec, EntitySpec

    with pytest.raises(ValidationError):
        WorldSpec(name="duplicate", entities=[entity(), entity()])
    entities = [
        EntitySpec(id=f"e{i}", asset_id="box", position=(0, 0, 1)) for i in range(65)
    ]
    with pytest.raises(ValidationError):
        WorldSpec(name="many", entities=entities)
    assert len(WorldSpec(name="limit", entities=entities[:64]).entities) == 64


def test_ball_drops_into_hollow_bin():
    from astra_world.world_spec import WorldSpec, EntitySpec
    from astra_world.world_builder import build_world

    world = build_world(
        WorldSpec(
            name="container",
            entities=[
                EntitySpec(
                    id="container", asset_id="bin", position=(0, 0, 0.09), fixed=True
                ),
                entity(),
            ],
        )
    )
    for _ in range(1200):
        mujoco.mj_step(world.model, world.data)
    position = world.observe()["entities"][1]["position"]
    assert abs(position[0]) < 0.1 and abs(position[1]) < 0.1
    assert 0.02 < position[2] < 0.15


def test_all_catalog_geometry_builds_and_fixed_pose_stays_fixed():
    from astra_world.world_spec import WorldSpec, EntitySpec
    from astra_world.world_builder import build_world

    ids = [
        "box",
        "sphere",
        "cylinder",
        "capsule",
        "table",
        "bin",
        "ramp",
        "wall",
        "platform",
        "domino",
        "peg",
        "ball",
    ]
    world = build_world(
        WorldSpec(
            name="shapes",
            entities=[
                EntitySpec(
                    id=f"e{i}", asset_id=asset, position=(i - 6, 0, 1), fixed=True
                )
                for i, asset in enumerate(ids)
            ],
        )
    )
    before = world.observe()["entities"]
    for _ in range(10):
        mujoco.mj_step(world.model, world.data)
    assert world.observe()["entities"] == before
    assert world.model.ngeom > len(ids)


def test_ball_rolls_down_rotated_ramp():
    from astra_world.world_spec import WorldSpec, EntitySpec
    from astra_world.world_builder import build_world

    world = build_world(
        WorldSpec(
            name="ramp",
            entities=[
                EntitySpec(
                    id="ramp",
                    asset_id="ramp",
                    position=(0, 0, 0.3),
                    rotation=(0, 0.6, 0),
                    fixed=True,
                ),
                EntitySpec(id="ball", asset_id="ball", position=(-0.1, 0, 0.6)),
            ],
        )
    )
    for _ in range(350):
        mujoco.mj_step(world.model, world.data)
    assert world.observe()["entities"][1]["position"][0] > -0.02


def test_panda_world_has_optional_robot_and_preserves_entity_pose():
    from astra_world.world_spec import WorldSpec
    from astra_world.world_builder import build_world
    from astra_world.assets import PANDA

    if not PANDA.exists():
        pytest.skip("Panda assets unavailable")
    world = build_world(WorldSpec(name="robot", robot="panda", entities=[entity()]))
    assert world.arm_q.shape == (7,)
    assert world.model.site("grasp").pos.tolist() == [0, 0, 0.103]
    assert world.body_name("test_object") == "entity_test_object"
    assert world.joint_name("test_object") == "test_object_joint"
    assert world.observe()["robot"]["type"] == "panda"
    assert world.observe()["entities"][0]["position"] == [0, 0, 1]
    for _ in range(100):
        mujoco.mj_step(world.model, world.data)
    assert np.all(np.isfinite(world.data.qpos))


def test_entity_extent_cannot_cross_world_boundary():
    from astra_world.world_spec import WorldSpec, EntitySpec
    from astra_world.world_builder import build_world

    with pytest.raises(ValueError, match="bounds"):
        build_world(
            WorldSpec(
                name="oversize",
                entities=[
                    EntitySpec(
                        id="wall",
                        asset_id="wall",
                        position=(20, 0, 0),
                        scale=(10, 10, 10),
                    )
                ],
            )
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("mass", "2"),
        ("friction", "0.5"),
        ("fixed", "false"),
        ("position", (True, 0, 1)),
        ("scale", ("1", 1, 1)),
    ],
)
def test_numeric_and_boolean_values_are_strict(field, value):
    from astra_world.world_spec import EntitySpec

    with pytest.raises(ValidationError):
        EntitySpec(
            **dict(
                {"id": "strict", "asset_id": "ball", "position": (0, 0, 1)},
                **{field: value},
            )
        )


def test_rotation_observation_matches_requested_three_axis_pose():
    from astra_world.world_spec import WorldSpec
    from astra_world.world_builder import build_world

    rotation = (0.2, 0.3, 0.4)
    world = build_world(
        WorldSpec(name="rotation", entities=[entity("box", rotation=rotation)])
    )
    np.testing.assert_allclose(
        world.observe()["entities"][0]["rotation"], rotation, atol=1e-12
    )


def test_imported_visual_mesh_is_separate_from_collision_hull(tmp_path, monkeypatch):
    import subprocess
    import sys
    import trimesh
    from astra_world import catalog
    from astra_world.world_builder import build_world
    from astra_world.world_spec import EntitySpec, WorldSpec

    # An L shape has a visible concavity its collision hull cannot represent.
    horizontal = trimesh.creation.box(extents=(0.3, 0.1, 0.1))
    vertical = trimesh.creation.box(extents=(0.1, 0.2, 0.1))
    vertical.apply_translation((-0.1, 0.15, 0))
    source = tmp_path / "concave.obj"
    trimesh.util.concatenate([horizontal, vertical]).export(source)
    destination = tmp_path / "catalog"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_asset.py",
            str(source),
            "--id",
            "concave",
            "--license",
            "CC0",
            "--output-dir",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    monkeypatch.setattr(catalog, "CUSTOM_CATALOG_PATH", destination / "catalog.json")
    world = build_world(
        WorldSpec(
            name="mesh",
            entities=[
                EntitySpec(
                    id="object",
                    asset_id="concave",
                    position=(0, 0, 1),
                    mass=2,
                )
            ],
        )
    )
    assert world.model.nmesh == 2
    collision = world.model.geom("entity_object_geom_0")
    visual = world.model.geom("entity_object_visual")
    assert collision.contype[0] == 1
    assert collision.rgba[3] == 0
    assert visual.contype[0] == visual.conaffinity[0] == 0
    assert visual.rgba[3] == 1
    assert visual.group[0] == 2
    assert world.model.body("entity_object").mass[0] == pytest.approx(2)
    # Keeping the source surface preserves more triangles than the convex hull.
    assert (
        world.model.mesh_facenum[visual.dataid[0]]
        > world.model.mesh_facenum[collision.dataid[0]]
    )
    assert catalog.get_asset("concave")["readiness"] == "imported"


def test_rotated_wheel_scales_in_entity_axes_and_contacts_ground():
    from astra_world.world_builder import build_world
    from astra_world.world_spec import EntitySpec, WorldSpec

    world = build_world(
        WorldSpec(
            name="wheel",
            entities=[
                EntitySpec(
                    id="wheel",
                    asset_id="wheel",
                    position=(0, 0, 0.5),
                    scale=(2, 1, 2),
                )
            ],
        )
    )
    wheel = world.model.geom("entity_wheel_geom_0")
    np.testing.assert_allclose(wheel.size[:2], [0.2, 0.025])
    for _ in range(900):
        mujoco.mj_step(world.model, world.data)
    # The doubled radius supports the disk at z=.2; its axle width stays .05.
    assert world.observe()["entities"][0]["position"][2] == pytest.approx(
        0.2, abs=0.005
    )


@pytest.mark.parametrize(
    "asset_id,scale,message",
    [
        ("wheel", (2, 2, 1), "circular cross sections"),
        ("ramp", (2, 1, 1), "shear"),
        ("capsule", (1, 1, 2), "uniform scale"),
    ],
)
def test_rejects_unrepresentable_part_scaling(asset_id, scale, message):
    from astra_world.world_builder import build_world
    from astra_world.world_spec import WorldSpec

    with pytest.raises(ValueError, match=message):
        build_world(WorldSpec(name="scale", entities=[entity(asset_id, scale=scale)]))
