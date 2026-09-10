import numpy as np
from astra_world.scene import build_scene, bin_for
from astra_world.contracts import SceneSpec, SortGoal
from astra_world.motion import solve_ik
from astra_world.manipulation import sort_blocks


def test_unreachable_ik_fails_without_changing_world():
    w = build_scene(SceneSpec(block_colors=["red"]))
    before = w.data.qpos.copy()
    assert solve_ik(w, [4, 0, 0.2], w.arm_q) is None
    assert np.array_equal(before, w.data.qpos)


def test_physical_sort():
    w = build_scene()
    other = {
        name: w.data.body(name).xpos.copy()
        for name, c in zip(w.block_ids, w.spec.block_colors)
        if c != "red"
    }
    result = sort_blocks(w, SortGoal())
    assert result["ok"], result
    assert bin_for(w, "block_0") == "left_bin"
    assert bin_for(w, "block_3") == "left_bin"
    for name, pos in other.items():
        assert np.linalg.norm(w.data.body(name).xpos - pos) < 0.005


def test_bin_capacity_failure_does_not_move_robot():
    w = build_scene(SceneSpec(block_colors=["red"] * 5))
    before = w.data.qpos.copy()
    result = sort_blocks(w, SortGoal())
    assert result["error_code"] == "bin_capacity"
    assert np.array_equal(before, w.data.qpos)


def test_already_sorted_block_is_observed_settling():
    import mujoco
    from astra_world.scene import BINS

    w = build_scene(SceneSpec(block_colors=["red"]))
    w.data.joint("block_0_joint").qpos[:3] = BINS["left_bin"] + [0, 0, 0.03]
    mujoco.mj_forward(w.model, w.data)
    before = w.data.time
    result = sort_blocks(w, SortGoal())
    assert result["ok"], result
    assert w.data.time - before >= 0.5 - 1e-6
