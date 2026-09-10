from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Color = Literal["red", "blue", "green"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SceneSpec(StrictModel):
    seed: int = Field(default=7, ge=0, le=2**32 - 1)
    block_colors: list[Color] = Field(
        default_factory=lambda: ["red", "blue", "green"] * 2,
        min_length=1,
        max_length=12,
    )


class SortGoal(StrictModel):
    color: Color = "red"
    destination_id: Literal["left_bin", "right_bin"] = "left_bin"


class ObstacleSpec(StrictModel):
    position: tuple[float, float, float] = (0.48, 0.10, 0.16)
    half_extents: tuple[float, float, float] = (0.065, 0.025, 0.16)
