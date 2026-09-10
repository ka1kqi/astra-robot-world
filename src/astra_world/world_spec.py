"""Validated, renderer-independent descriptions for general physics worlds."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Scalar = Annotated[float, Field(allow_inf_nan=False, strict=True)]
Vector3 = tuple[Scalar, Scalar, Scalar]
Scale = Annotated[float, Field(ge=0.05, le=10, allow_inf_nan=False, strict=True)]
Color = Literal[
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "white",
    "gray",
    "black",
    "purple",
    "cyan",
    "brown",
]


class EntitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    asset_id: str
    position: Vector3
    rotation: Vector3 = (0, 0, 0)
    scale: tuple[Scale, Scale, Scale] = (1, 1, 1)
    color: Color = "gray"
    mass: float | None = Field(
        default=None, gt=0, le=100, allow_inf_nan=False, strict=True
    )
    friction: float | None = Field(
        default=None, ge=0, le=10, allow_inf_nan=False, strict=True
    )
    fixed: bool | None = Field(default=None, strict=True)

    @field_validator("asset_id")
    @classmethod
    def known_asset(cls, value):
        from .catalog import get_asset

        try:
            get_asset(value)
        except KeyError:
            raise ValueError(f"Unknown asset_id: {value}") from None
        return value

    @field_validator("position")
    @classmethod
    def bounded_position(cls, value):
        if any(abs(x) > 20 for x in value):
            raise ValueError("Entity position must be within 20 meters of the origin")
        return value


class WorldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    name: str = Field(min_length=1, max_length=120)
    seed: int = Field(default=7, ge=0, le=2**32 - 1, strict=True)
    gravity: Vector3 = (0, 0, -9.81)
    robot: Literal["none", "panda"] = "none"
    entities: list[EntitySpec] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def unique_entities(self):
        ids = [entity.id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("Entity ids must be unique")
        if any(abs(x) > 100 for x in self.gravity):
            raise ValueError("Gravity components must be within 100 m/s²")
        return self
