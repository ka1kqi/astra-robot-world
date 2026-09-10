import numpy as np
import pytest
from pydantic import ValidationError
from astra_world.contracts import SceneSpec
from astra_world.scene import build_scene, observe


@pytest.mark.parametrize("seed", [7, 11])
def test_station_settles_with_six_blocks(seed):
    world = build_scene(SceneSpec(seed=seed))
    state = observe(world)
    assert len(state["blocks"]) == 6
    assert set(state["bins"]) == {"left_bin", "right_bin"}
    assert all(abs(b["position"][2] - 0.02) < 0.002 for b in state["blocks"])
    assert np.isfinite(world.data.qpos).all()
    assert all(b["bin"] is None for b in state["blocks"])


def test_rejects_unsupported_scene():
    with pytest.raises(ValidationError):
        SceneSpec(block_colors=["purple"])
    with pytest.raises(ValidationError):
        SceneSpec(block_colors=["red"] * 13)
