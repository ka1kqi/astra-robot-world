"""Physical routing through the catalog-built world, without sorting-station bins."""

import numpy as np
import pytest

from astra_world.general_manipulation import pick_place
from astra_world.motion import Controller
from astra_world.world_builder import build_world
from astra_world.world_spec import WorldSpec


@pytest.mark.parametrize("with_barrier", [False, True], ids=["open", "obstructed"])
def test_general_panda_routes_around_wall_and_places_block_physically(with_barrier):
    entities = [
        {
            "id": "part",
            "asset_id": "small_box",
            "position": [0.4, -0.12, 0.021],
            "color": "red",
        }
    ]
    if with_barrier:
        # The wall catalog bounds are 1.2 × .06 × .5 m; these scales produce
        # the proven sorting barrier's half-extents .075 × .025 × .19 m.
        entities.append(
            {
                "id": "barrier",
                "asset_id": "wall",
                "position": [0.4, 0.17, 0.19],
                "scale": [0.125, 5 / 6, 0.76],
                "color": "orange",
            }
        )
    world = build_world(WorldSpec(name="General obstacle routing", robot="panda", entities=entities))
    target = np.array([0.4, 0.35, 0.021])
    wall_id = world.model.geom("entity_barrier_geom_0").id if with_barrier else None
    if with_barrier:
        np.testing.assert_allclose(world.model.geom_size[wall_id], [0.075, 0.025, 0.19])

    lift_heights = []

    def check_physics():
        lift_heights.append(world.data.body("entity_part").xpos[2])
        for contact in world.data.contact:
            if wall_id is not None and wall_id in contact.geom:
                assert contact.dist >= -0.0008, "Robot or carried part penetrated the barrier"

    result = pick_place(world, "part", target, tick=check_physics)
    assert result["ok"], result
    assert max(lift_heights) > 0.18, "The object must be physically lifted before transport"
    if with_barrier:
        assert result["payload"]["route"] != "direct"
        transport = np.asarray(world.paths[-1]["points"])
        assert len(transport) > 2
        assert np.max(np.abs(transport[:, 0] - 0.4)) > 0.075 + 0.04
    else:
        assert result["payload"]["route"] == "direct"

    # Independently recheck a further half-second of settled physical state.
    ctl = Controller(world, tick=check_physics)
    for _ in range(round(0.5 / world.model.opt.timestep)):
        ctl.step(world.model.opt.timestep)
        np.testing.assert_allclose(world.data.body("entity_part").xpos, target, atol=0.012)
        assert np.linalg.norm(world.data.joint("part_joint").qvel[:3]) < 0.02
    if with_barrier:
        np.testing.assert_array_equal(world.data.body("entity_barrier").xpos, [0.4, 0.17, 0.19])
        assert world.model.body_jntnum[world.model.body("entity_barrier").id] == 0
    assert np.isfinite(world.data.qpos).all()
