import numpy as np
from astra_world.scene import build_scene, bin_for
from astra_world.contracts import SortGoal, ObstacleSpec
from astra_world.session import Session


def test_retry_needs_previous_task():
    s = Session(build_scene())
    assert s.retry()["error_code"] == "no_previous_task"


def test_obstacle_and_retry_physical_sort():
    session = Session(build_scene())
    first = session.sort(SortGoal())
    assert first["ok"], first
    added = session.add_obstacle()
    assert added["ok"], added
    assert session.world.obstacles
    again = session.retry()
    assert again["ok"], again
    assert session.world.obstacles
    assert any(route != "direct" for route in again["payload"]["routes"])
    assert bin_for(session.world, "block_0") == "left_bin"
    assert bin_for(session.world, "block_3") == "left_bin"


def test_overlapping_obstacle_is_transactional():
    session = Session(build_scene())
    original = session.world
    pos = original.data.body("block_0").xpos
    result = session.add_obstacle(
        ObstacleSpec(
            position=(float(pos[0]), float(pos[1]), 0.1), half_extents=(0.04, 0.04, 0.1)
        )
    )
    assert not result["ok"]
    assert session.world is original


def test_obstacle_must_fit_table_not_just_center():
    session = Session(build_scene())
    result = session.add_obstacle(
        ObstacleSpec(position=(0.79, 0, 0.1), half_extents=(0.3, 0.025, 0.1))
    )
    assert not result["ok"]
    assert not session.world.obstacles


def test_near_block_barrier_stops_before_penetrating_contact():
    from astra_world.contracts import SceneSpec

    session = Session(build_scene(SceneSpec(block_colors=["red"])))
    pos = session.world.data.body("block_0").xpos
    added = session.add_obstacle(
        ObstacleSpec(
            position=(float(pos[0]), float(pos[1] + 0.065), 0.05),
            half_extents=(0.025, 0.025, 0.05),
        )
    )
    assert added["ok"], added
    penetrations = []

    def inspect():
        w = session.world
        geom = w.model.geom("obstacle_0").id
        penetrations.extend(
            c.dist for c in w.data.contact if geom in c.geom and c.dist < -0.0008
        )

    result = session.sort(SortGoal(), tick=inspect)
    assert not result["ok"] and result["error_code"] == "no_path", result
    assert not penetrations
    np.testing.assert_allclose(session.world.data.ctrl[:7], session.world.arm_q)
