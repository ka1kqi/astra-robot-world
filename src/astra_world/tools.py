from .action_contracts import SearchNotebookArgs, ReadNotebookArgs, NoteArgs

"""Strict command arguments and dispatch on the simulation owner thread."""

from .contracts import StrictModel, SceneSpec, SortGoal, ObstacleSpec
from .scene import build_scene, observe
from .world_spec import WorldSpec, EntitySpec
from pydantic import Field, field_validator
from .action_contracts import (
    DraftActionArgs,
    TestActionArgs,
    SaveActionArgs,
    RunActionArgs,
)


class EmptyArgs(StrictModel):
    pass


class SearchArgs(StrictModel):
    query: str = Field(default="", max_length=200)
    limit: int = Field(default=20, ge=1, le=50)


class AssetArgs(StrictModel):
    asset_id: str = Field(min_length=1, max_length=100)


class SimulateArgs(StrictModel):
    duration: float = Field(gt=0, le=20)


class ScenarioArgs(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class PickPlaceArgs(StrictModel):
    object_id: str
    target_position: tuple[float, float, float]
    target_rotation: tuple[float, float, float] | None = None


class ApproachWaypoint(StrictModel):
    position: tuple[float, float, float]
    rotation: tuple[float, float, float] | None = None

    @field_validator("position")
    @classmethod
    def bounded_position(cls, position):
        if any(abs(value) > 2 for value in position) or position[2] < 0:
            raise ValueError("World pose must be above ground within two meters of the origin.")
        return position


class ApproachCandidate(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    waypoints: list[ApproachWaypoint] = Field(min_length=1, max_length=8)


class CheckApproachesArgs(StrictModel):
    candidates: list[ApproachCandidate] = Field(min_length=1, max_length=8)


ARGUMENTS = {
    "check_approaches": CheckApproachesArgs,
    "search_action_notes": SearchNotebookArgs,
    "read_action_notes": ReadNotebookArgs,
    "write_action_note": NoteArgs,
    "draft_action": DraftActionArgs,
    "test_action": TestActionArgs,
    "save_action": SaveActionArgs,
    "list_actions": EmptyArgs,
    "run_action": RunActionArgs,
    "search_assets": SearchArgs,
    "describe_asset": AssetArgs,
    "create_world": WorldSpec,
    "add_entity": EntitySpec,
    "simulate": SimulateArgs,
    "reset_world": EmptyArgs,
    "save_scenario": ScenarioArgs,
    "load_scenario": ScenarioArgs,
    "pick_place": PickPlaceArgs,
    "list_assets": EmptyArgs,
    "build_sorting_station": SceneSpec,
    "observe_world": EmptyArgs,
    "sort_blocks": SortGoal,
    "add_obstacle": ObstacleSpec,
    "retry_last_task": EmptyArgs,
    "stop": EmptyArgs,
}


def error(code, detail, revision=0):
    return {
        "ok": False,
        "scene_revision": revision,
        "error_code": code,
        "detail": detail,
    }


def dispatch(
    session,
    name,
    arguments,
    *,
    automatic_obstacle=False,
    cancel=None,
    tick=None,
    status=None,
    preview=None,
):
    """Return the possibly replaced session and the action's measured result."""
    from .general_session import GeneralSession, load_scenario

    if name == "check_approaches":
        from .approaches import check_approaches

        return session, check_approaches(session.world, arguments)
    if name in ("search_action_notes", "read_action_notes", "write_action_note"):
        from .action_lab import ActionLab

        if not hasattr(session, "action_lab"):
            session.action_lab = ActionLab(session.world)
        notebook = session.action_lab.notebook
        if name == "search_action_notes":
            payload = {"entries": notebook.search(arguments.query, arguments.limit)}
        elif name == "read_action_notes":
            entries = notebook.read(
                arguments.experiment_id, arguments.limit, arguments.after_id
            )
            payload = {
                "entries": entries,
                "next_after_id": entries[-1]["id"] if entries else arguments.after_id,
            }
        else:
            payload = {
                "entry_id": notebook.add_note(
                    arguments.experiment_id, arguments.text, arguments.trial_number
                ),
                "source": "astra",
                "message": "Saved an interpretation note; measured trial evidence is unchanged.",
            }
        return session, {
            "ok": True,
            "scene_revision": session.world.revision,
            "payload": payload,
        }
    if name in ("draft_action", "test_action", "save_action", "run_action"):
        if not isinstance(session, GeneralSession):
            return session, error(
                "unsupported_scene",
                "Action Lab needs a general world with a Panda.",
                session.world.revision,
            )
        from .action_lab import ActionLab

        if not hasattr(session, "action_lab"):
            session.action_lab = ActionLab(session.world)
        lab = session.action_lab
        lab.world = session.world
        if preview is not None:
            lab.preview = preview
        if name == "draft_action":
            result = lab.draft(arguments.name, arguments.goal, arguments.trial_budget)
        elif name == "test_action":
            result = lab.test(
                arguments.draft_id, arguments.program, cancel=cancel, status=status
            )
        elif name == "save_action":
            result = lab.save(arguments.draft_id)
        else:
            result = lab.run(
                session.world,
                arguments.name,
                bindings=arguments.bindings,
                cancel=cancel,
                tick=tick,
                status=status,
            )
        return session, result
    if name == "create_world":
        from .world_builder import build_world

        session = GeneralSession(
            build_world(arguments, revision=session.world.revision + 1)
        )
        return session, session.result(payload=observe(session.world))
    if name == "load_scenario":
        session = load_scenario(arguments.name, session.world.revision + 1)
        return session, session.result(payload=observe(session.world))
    if name in ("simulate", "reset_world", "save_scenario", "pick_place", "add_entity"):
        if not isinstance(session, GeneralSession):
            return session, error(
                "unsupported_scene",
                "Create a general world before using this tool.",
                session.world.revision,
            )
        if name == "add_entity":
            return session, session.add_entity(arguments)
        if name == "simulate":
            return session, session.simulate(arguments.duration, cancel, tick, status)
        if name == "reset_world":
            return session, session.reset()
        if name == "save_scenario":
            return session, session.save(arguments.name)
        from .general_manipulation import pick_place

        return session, pick_place(
            session.world,
            arguments.object_id,
            arguments.target_position,
            cancel,
            tick,
            status,
            target_rotation=arguments.target_rotation,
        )
    if isinstance(session, GeneralSession) and name in (
        "sort_blocks",
        "add_obstacle",
        "retry_last_task",
    ):
        return session, error(
            "unsupported_scene",
            "This preset tool needs the sorting station. Use general scene and object tools here.",
            session.world.revision,
        )
    if name == "build_sorting_station":
        from .session import Session

        session = Session(build_scene(arguments, revision=session.world.revision + 1))
        return session, {
            "ok": True,
            "scene_revision": session.world.revision,
            "payload": observe(session.world),
        }
    if name == "sort_blocks":
        result = session.sort(arguments, cancel=cancel, tick=tick, status=status)
    elif name == "add_obstacle":
        result = session.add_obstacle(None if automatic_obstacle else arguments)
    elif name == "retry_last_task":
        result = session.retry(cancel=cancel, tick=tick, status=status)
    else:
        raise ValueError(f"Not a mutating command: {name}")
    return session, result
