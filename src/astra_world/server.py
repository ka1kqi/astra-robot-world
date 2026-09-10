"""Local browser API. Physics is accessed exclusively through the runtime boundary."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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


def create_app(runtime, *, adapter=None):
    adapter = adapter or AstraAdapter.from_env()
    history, turns = [], []
    active = None
    web = Path(__file__).resolve().parents[2] / "web"

    @asynccontextmanager
    async def lifespan(app):
        yield
        if active and not active.done():
            runtime.stop()
            active.cancel()
            await asyncio.gather(active, return_exceptions=True)

    app = FastAPI(title="Astra Robot World", lifespan=lifespan)

    def busy():
        return active is not None and not active.done()

    def new_turn(message, mode):
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
        return turn

    async def execute(turn, name, args):
        turn["events"].append({"type": "tool_start", "name": name})
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
                asyncio.wrap_future(runtime.submit(name, args))
            )
        turn["events"].append({"type": "tool_result", "name": name, "result": result})
        return result

    async def run(turn, name=None, args=None):
        try:
            if name:
                result = await execute(turn, name, args or {})
                turn["status"] = "completed" if result.get("ok") else "failed"
            else:
                history.append({"role": "user", "content": turn["message"]})
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
                    "text": "Stopped. The robot is holding its position.",
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

    @app.get("/state")
    async def state():
        return {
            "world": runtime.snapshot(),
            "busy": busy(),
            "turns": turns[-100:],
            "provider": {
                "configured": adapter.configured,
                "message": "Live Astra is configured. Requests use your selected provider and model."
                if adapter.configured
                else "Live Astra is not configured. Set ASTRA_BASE_URL, ASTRA_MODEL and ASTRA_API_KEY, then restart. Use manual controls to test the physics.",
            },
        }

    @app.post("/chat", status_code=202)
    async def chat(body: ChatRequest):
        nonlocal active
        if not adapter.configured:
            raise HTTPException(
                503,
                "Live Astra is not configured. Set ASTRA_BASE_URL, ASTRA_MODEL and ASTRA_API_KEY, then restart.",
            )
        turn = new_turn(body.message, "astra")
        active = asyncio.create_task(run(turn))
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
        return result

    @app.get("/")
    async def index():
        return FileResponse(web / "index.html")

    @app.get("/{asset}")
    async def asset(asset: str):
        if asset not in {"app.js", "styles.css"}:
            raise HTTPException(404)
        return FileResponse(web / asset)

    return app
