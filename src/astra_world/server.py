"""Local browser API. Physics is accessed exclusively through the runtime boundary."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit
import json
import os
import re
from .action_contracts import Identifier, Vector
from .action_proposals import interpret_action
from .viewport import ViewportInput
from copy import deepcopy
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .astra import AstraAdapter, ConfigurationError, ProviderError, TOOL_NAMES


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=8000)

    @field_validator("message")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class VisualChatRequest(ChatRequest):
    viewport: ViewportInput | None = None


class CreateActionRequest(ChatRequest):
    goal_kind: Literal["topple", "circle"] = "topple"
    target_id: Identifier | None = None
    support_id: Identifier | None = None
    center: Vector = (0.45, 0, 0.3)
    radius: float = Field(default=0.06, ge=0.02, le=0.12)
    plane: Literal["xy", "xz", "yz"] = "xy"
    trial_budget: int = Field(default=5, ge=3, le=10)


class ProposeActionRequest(ChatRequest):
    trial_budget: int = Field(default=5, ge=3, le=10)


class StartActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposal_id: str = Field(pattern=r"^[a-f0-9]{32}$")


def _local_origin(value):
    """Parse an exact local HTTP origin, rejecting ambiguous authorities."""
    if not value or any(char.isspace() for char in value):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or re.fullmatch(r"(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]+)?", parsed.netloc, re.IGNORECASE) is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path or parsed.query or parsed.fragment
            or any(char in value for char in ("@", "\\", "?", "#"))
        ):
            return None
        port = parsed.port
        if port == 0:
            return None
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        return parsed.scheme, parsed.hostname, port
    except ValueError:
        return None


def create_app(runtime, *, adapter=None):
    adapter = adapter or AstraAdapter.from_env()
    history, turns = [], []
    proposals = {}
    active = None
    active_turn = None
    web = Path(__file__).resolve().parents[2] / "web"

    @asynccontextmanager
    async def lifespan(app):
        yield
        if active and not active.done():
            runtime.stop()
            active.cancel()
            await asyncio.gather(active, return_exceptions=True)

    app = FastAPI(title="Astra Robot World", lifespan=lifespan)

    @app.middleware("http")
    async def local_browser_boundary(request, call_next):
        hosts = request.headers.getlist("host")
        expected = (
            _local_origin(f"{request.scope['scheme']}://{hosts[0]}")
            if len(hosts) == 1 else None
        )
        if expected is None:
            return JSONResponse({"detail": "A local Host header is required."}, status_code=400)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origins = request.headers.getlist("origin")
            if origins and (len(origins) != 1 or _local_origin(origins[0]) != expected):
                return JSONResponse({"detail": "Cross-origin state changes are not allowed."}, status_code=403)
        return await call_next(request)

    def busy():
        return active is not None and not active.done()

    def new_turn(message, mode):
        nonlocal active_turn
        if busy():
            raise HTTPException(409, "An action is running. Wait or press Stop.")
        turn = {
            "id": uuid4().hex,
            "message": message,
            "mode": mode,
            "status": "running",
            "events": [],
        }
        turns.append(turn)
        active_turn = turn
        return turn

    async def execute(turn, name, args, *, expected_state_token=None):
        call_id = uuid4().hex
        turn["events"].append({"type": "tool_start", "name": name, "call_id": call_id, "arguments": deepcopy(args)})
        if name == "retry_last_task":
            turn["events"].append(
                {
                    "type": "status",
                    "text": "Resetting the blocks for another attempt; keeping the obstacle.",
                }
            )
        if name == "stop":
            result = runtime.stop()
        else:
            # shield the worker future: asyncio cancellation must not cancel its result receiver.
            result = await asyncio.shield(
                asyncio.wrap_future(
                    runtime.submit(
                        name, args, expected_state_token=expected_state_token
                    )
                    if expected_state_token is not None
                    else runtime.submit(name, args)
                )
            )
        turn["events"].append({"type": "tool_result", "name": name, "call_id": call_id, "result": result})
        return result

    async def run(turn, name=None, args=None, viewport=None):
        user_message = None
        try:
            if name:
                result = await execute(turn, name, args or {})
                turn["status"] = "completed" if result.get("ok") else "failed"
            else:
                user_message = {"role": "user", "content": turn["message"]}
                if viewport:
                    user_message = viewport.message(
                        turn["message"], responses=adapter.uses_responses
                    )
                history.append(user_message)
                await adapter.run_turn(
                    history, lambda n, a: execute(turn, n, a), turn["events"].append
                )
                action_failed = any(
                    event["type"] == "tool_result" and not event["result"].get("ok")
                    for event in turn["events"]
                )
                turn["status"] = "failed" if action_failed else "completed"
        except asyncio.CancelledError:
            turn["status"] = "cancelled"
            turn["events"].append(
                {
                    "type": "status",
                    "text": "Stopped the current action.",
                }
            )
        except (ConfigurationError, ProviderError) as exc:
            turn["status"] = "failed"
            turn["events"].append({"type": "error", "text": str(exc)})
        except Exception:
            turn["status"] = "failed"
            turn["events"].append(
                {
                    "type": "error",
                    "text": "The action could not complete. Inspect the world state and try again.",
                }
            )
        finally:
            if viewport and user_message:
                user_message["content"] = turn["message"] + "\n[The viewport image from this earlier turn has been omitted.]"

    async def create_action_turn(
        turn, body, *, approved_goal=None, expected_state_token=None, action_name=None
    ):
        saved = False
        try:
            async with asyncio.timeout(300):
                world = runtime.snapshot()
                goal = {
                    "kind": "topple",
                    "object_id": body.target_id,
                    "support_id": body.support_id,
                    "preserve_ids": [
                        e["id"]
                        for e in world.get("entities", [])
                        if e["id"] not in (body.target_id, body.support_id)
                    ],
                }
                if body.goal_kind == "circle":
                    goal = {
                        "kind": "circle",
                        "target_position": body.center,
                        "radius": body.radius,
                        "plane": body.plane,
                        "preserve_ids": [e["id"] for e in world.get("entities", [])],
                    }
                if approved_goal is not None:
                    goal = deepcopy(approved_goal)
                from .action_lab import ACTIONS_DIR
                from .action_notebook import ActionNotebook

                notebook = ActionNotebook(ACTIONS_DIR / "notebook.sqlite3")
                related = await asyncio.to_thread(notebook.related, goal["kind"], 3)
                drafted = await execute(
                    turn,
                    "draft_action",
                    {
                        "name": action_name
                        or body.goal_kind
                        + "_"
                        + (body.target_id or "gripper")[:35]
                        + "_"
                        + turn["id"][:8],
                        "goal": goal,
                        "trial_budget": body.trial_budget,
                    },
                    expected_state_token=expected_state_token,
                )
                if not drafted.get("ok"):
                    turn["status"] = "failed"
                    return
                draft_id = drafted["payload"]["draft_id"]

                async def experiment(name, args):
                    nonlocal saved
                    allowed = {
                        "check_approaches",
                        "observe_world",
                        "search_assets",
                        "describe_asset",
                        "list_assets",
                        "list_actions",
                        "test_action",
                        "save_action",
                        "search_action_notes",
                        "read_action_notes",
                        "write_action_note",
                    }
                    if (
                        name == "write_action_note"
                        and args.get("experiment_id") != draft_id
                    ):
                        result = {
                            "ok": False,
                            "error_code": "experiment_scope",
                            "detail": "Write notes only to this experiment.",
                        }
                        turn["events"].append(
                            {"type": "tool_result", "name": name, "result": result}
                        )
                        return result
                    if name not in allowed or (
                        name in ("test_action", "save_action")
                        and args.get("draft_id") != draft_id
                    ):
                        result = {
                            "ok": False,
                            "error_code": "experiment_scope",
                            "detail": "Use only this draft for testing and saving. Live scene mutations and new drafts are disabled during action creation.",
                        }
                        turn["events"].append(
                            {"type": "tool_result", "name": name, "result": result}
                        )
                        return result
                    result = await execute(turn, name, args)
                    if name == "save_action" and result.get("ok"):
                        saved = True
                    return result

                action_history = [
                    {
                        "role": "user",
                        "content": "Create and verify a reusable action for this request: "
                        + body.message
                        + "\nThe server already drafted the fixed measurable goal. Use this draft; do not create another. "
                        "Test candidate programs, revise from measured failures, then save when verified. Do not execute in the live world. "
                        "Continue autonomously within the available trial budget. Draft result: "
                        + json.dumps(drafted)
                        + "\nRelated persistent records (read their experiments for context; notes are hypotheses): "
                        + json.dumps(related)
                        + "\nUse write_action_note with experiment_id="
                        + draft_id
                        + " to record hypotheses and lessons. Measured evidence is recorded automatically.",
                    }
                ]
                for _ in range(3):
                    await adapter.run_turn(
                        action_history, experiment, turn["events"].append
                    )
                    if saved:
                        break
                    action_history.append(
                        {
                            "role": "user",
                            "content": "Continue the authorized experiment if trials remain; revise the candidate or save a verified action. If exhausted or impossible, state that explicitly.",
                        }
                    )
                turn["status"] = "completed" if saved else "failed"
                if not saved:
                    turn["events"].append(
                        {
                            "type": "status",
                            "text": "No verified action was saved. Review the trial diagnostics.",
                        }
                    )
        except (asyncio.CancelledError, TimeoutError) as exc:
            runtime.stop()
            turn["status"] = (
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            )
            turn["events"].append(
                {
                    "type": "status",
                    "text": "Action creation stopped."
                    if isinstance(exc, asyncio.CancelledError)
                    else "Action creation reached its five-minute budget.",
                }
            )
        except (ConfigurationError, ProviderError) as exc:
            turn["status"] = "failed"
            turn["events"].append({"type": "error", "text": str(exc)})
        except Exception:
            runtime.stop()
            turn["status"] = "failed"
            turn["events"].append(
                {
                    "type": "error",
                    "text": "Action creation could not complete. Inspect the trial results.",
                }
            )

    async def propose_turn(turn, proposal, world):
        try:
            from .action_lab import ACTIONS_DIR
            from .action_notebook import ActionNotebook

            notebook = ActionNotebook(ACTIONS_DIR / "notebook.sqlite3")
            related = []
            for kind in ("topple", "displace", "circle", "extract", "rotate"):
                related.extend(await asyncio.to_thread(notebook.related, kind, 1))
            result = await interpret_action(
                adapter, proposal["message"], world, related
            )
            proposal.update(result.model_dump(mode="json"))
            turn["events"].append({"type": "assistant", "text": result.interpretation})
            if result.clarification:
                turn["events"].append(
                    {"type": "assistant", "text": result.clarification}
                )
            turn["status"] = "completed"
        except asyncio.CancelledError:
            proposal["status"] = "cancelled"
            turn["status"] = "cancelled"
        except (ConfigurationError, ProviderError) as exc:
            proposal["status"] = "failed"
            proposal["interpretation"] = str(exc)
            turn["status"] = "failed"
            turn["events"].append({"type": "error", "text": str(exc)})
        except Exception:
            proposal["status"] = "failed"
            turn["status"] = "failed"
            turn["events"].append(
                {"type": "error", "text": "Could not interpret this action request."}
            )

    @app.post("/actions/propose", status_code=202)
    async def propose_action(body: ProposeActionRequest):
        nonlocal active
        if not adapter.configured:
            raise HTTPException(503, "Configure Astra before proposing an action.")
        world = runtime.snapshot()
        if (
            world.get("kind") != "general"
            or world.get("robot", {}).get("type") != "panda"
        ):
            raise HTTPException(409, "Create a general Panda scene first.")
        if world.get("busy"):
            raise HTTPException(409, "Wait for the current motion or press Stop.")
        turn = new_turn(body.message, "action_proposal")
        proposal = {
            "id": uuid4().hex,
            "turn_id": turn["id"],
            "message": body.message,
            "status": "interpreting",
            "interpretation": "Interpreting your request…",
            "goal": None,
            "name": "",
            "clarification": None,
            "limitations": [],
            "scene_revision": world["scene_revision"],
            "state_token": world.get("state_token"),
            "trial_budget": body.trial_budget,
        }
        proposals[proposal["id"]] = proposal
        while len(proposals) > 20:
            proposals.pop(next(iter(proposals)))
        active = asyncio.create_task(propose_turn(turn, proposal, deepcopy(world)))
        return {"turn_id": turn["id"], "proposal_id": proposal["id"]}

    @app.post("/actions/start", status_code=202)
    async def start_action(body: StartActionRequest):
        nonlocal active
        proposal = proposals.get(body.proposal_id)
        if not proposal or proposal["status"] != "ready":
            raise HTTPException(
                409, "Only a ready, unused proposal can start experiments."
            )
        world = runtime.snapshot()
        if (
            not proposal["state_token"]
            or world.get("state_token") != proposal["state_token"]
        ):
            proposal["status"] = "stale"
            raise HTTPException(
                409, "The scene changed. Propose the action again before starting."
            )
        if world.get("busy"):
            raise HTTPException(409, "Wait for the current motion or press Stop.")
        goal = deepcopy(proposal["goal"])
        legacy = CreateActionRequest(
            message=proposal["message"],
            goal_kind="circle" if goal["kind"] == "circle" else "topple",
            trial_budget=proposal["trial_budget"],
        )
        turn = new_turn(proposal["message"], "action_lab")
        proposal["status"] = "started"
        proposal["experiment_turn_id"] = turn["id"]

        async def approved_run():
            await create_action_turn(
                turn,
                legacy,
                approved_goal=goal,
                expected_state_token=proposal["state_token"],
                action_name=proposal["name"][:48] + "_" + turn["id"][:8],
            )
            if any(
                event.get("result", {}).get("error_code") == "stale_scene"
                for event in turn["events"]
            ):
                proposal["status"] = "stale"

        active = asyncio.create_task(approved_run())
        return {"turn_id": turn["id"]}

    @app.get("/health")
    async def health():
        return {
            "app": "astra-robot-world",
            "pid": os.getpid(),
            "project": str(Path(__file__).resolve().parents[2]),
            "native_viewer": not getattr(runtime, "headless", True),
        }

    @app.get("/frame.jpg")
    async def frame(
        view: Literal["live", "experiment", "action"] = "live",
        trial_id: str | None = Query(default=None, max_length=64),
        clip_id: str | None = Query(default=None, max_length=64),
        frame: int | None = Query(default=None, ge=0, le=299),
    ):
        if view == "action":
            data = (
                runtime.action_replay_frame(clip_id, frame)
                if hasattr(runtime, "action_replay_frame") else None
            )
        elif view == "experiment":
            data = (
                runtime.experiment_frame(trial_id, frame)
                if hasattr(runtime, "experiment_frame")
                else None
            )
        else:
            data = runtime.frame() if hasattr(runtime, "frame") else None
        if data is None:
            raise HTTPException(
                503, "The simulation renderer has not produced a frame."
            )
        return Response(
            data, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    @app.get("/action-replay")
    async def action_replay():
        if hasattr(runtime, "action_replay_metadata"):
            return runtime.action_replay_metadata()
        return {"clip": None, "available": False, "error": None}

    @app.get("/experiments")
    async def experiments():
        if hasattr(runtime, "experiment_metadata"):
            return runtime.experiment_metadata()
        return {
            "trials": [],
            "latest_trial_id": None,
            "available": False,
            "error": None,
        }

    @app.get("/action-notes")
    async def action_notes(
        query: str = Query(default="", max_length=200),
        experiment_id: str | None = Query(default=None, pattern=r"^[a-f0-9]{32}$"),
        after_id: int = Query(default=0, ge=0),
    ):
        from .action_lab import ACTIONS_DIR
        from .action_notebook import ActionNotebook

        book = ActionNotebook(ACTIONS_DIR / "notebook.sqlite3")
        if experiment_id:
            entries = await asyncio.to_thread(book.read, experiment_id, 50, after_id)
        else:
            entries = await asyncio.to_thread(book.search, query, 20)
        return {
            "ok": True,
            "payload": {
                "entries": entries,
                "next_after_id": entries[-1]["id"]
                if entries and experiment_id
                else after_id,
            },
        }

    @app.get("/actions")
    async def actions():
        return {
            "ok": True,
            "payload": {"actions": runtime.snapshot().get("actions", [])},
        }

    @app.post("/actions/create", status_code=202)
    async def create_action(body: CreateActionRequest):
        nonlocal active
        if not adapter.configured:
            raise HTTPException(503, "Configure Astra before creating an action.")
        world = runtime.snapshot()
        ids = {e["id"] for e in world.get("entities", [])}
        if (
            world.get("kind") != "general"
            or world.get("robot", {}).get("type") != "panda"
        ):
            raise HTTPException(409, "Create a general Panda scene first.")
        if body.goal_kind == "topple" and (
            body.target_id == body.support_id
            or not {body.target_id, body.support_id} <= ids
        ):
            raise HTTPException(
                422, "Select distinct existing upper and lower objects."
            )
        turn = new_turn(body.message, "action_lab")
        active = asyncio.create_task(create_action_turn(turn, body))
        return {"turn_id": turn["id"]}

    @app.get("/state")
    async def state():
        return {
            "world": runtime.snapshot(),
            "proposals": deepcopy(list(proposals.values())),
            "busy": busy(),
            "turns": turns[-100:],
            "provider": {
                "configured": adapter.configured,
                "message": "Live Astra is configured. Requests use your selected provider and model."
                if adapter.configured
                else "Live Astra is not configured. Set ASTRA_BASE_URL, ASTRA_MODEL and ASTRA_API_KEY, then restart. Use manual controls to test the physics.",
            },
        }

    @app.get("/assets")
    async def assets(query: str = Query(default="", max_length=200)):
        try:
            return await asyncio.wait_for(
                asyncio.shield(
                    asyncio.wrap_future(
                        runtime.submit("search_assets", {"query": query, "limit": 50})
                    )
                ),
                timeout=10,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                503, "Asset search is taking too long. Try again."
            ) from None

    @app.post("/chat", status_code=202)
    async def chat(body: VisualChatRequest):
        nonlocal active
        if not adapter.configured:
            raise HTTPException(
                503,
                "Live Astra is not configured. Set ASTRA_BASE_URL, ASTRA_MODEL and ASTRA_API_KEY, then restart.",
            )
        turn = new_turn(body.message, "astra")
        if body.viewport:
            turn["viewport"] = body.viewport.model_dump(exclude={"image"})
        active = asyncio.create_task(run(turn, viewport=body.viewport))
        return {"turn_id": turn["id"]}

    @app.post("/tools/{name}", status_code=202)
    async def manual(name: str, args: dict):
        nonlocal active
        if name not in TOOL_NAMES:
            raise HTTPException(404, "Unknown tool")
        if name == "stop":
            return await stop()
        turn = new_turn(name.replace("_", " "), "manual")
        active = asyncio.create_task(run(turn, name, args))
        return {"turn_id": turn["id"]}

    @app.post("/stop")
    async def stop():
        result = runtime.stop()
        if busy():
            active.cancel()
            # A task cancelled before its first tick never reaches its own handler.
            if active_turn and active_turn["status"] == "running":
                active_turn["status"] = "cancelled"
            for proposal in proposals.values():
                if (
                    active_turn
                    and active_turn["id"]
                    in (proposal.get("turn_id"), proposal.get("experiment_turn_id"))
                    and proposal["status"] in ("interpreting", "started")
                ):
                    proposal["status"] = "cancelled"
        return result

    @app.get("/")
    async def index():
        return FileResponse(web / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/{asset}")
    async def asset(asset: str):
        if asset not in {"app.js", "styles.css", "architecture.html"}:
            raise HTTPException(404)
        return FileResponse(web / asset, headers={"Cache-Control": "no-store"})

    return app
