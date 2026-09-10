from concurrent.futures import Future
import time
import pytest
from fastapi.testclient import TestClient
from astra_world.astra import AstraAdapter
from astra_world.action_proposals import ActionProposal
from astra_world.server import create_app


class Runtime:
    headless = True

    def __init__(self):
        self.token = "original"
        self.calls = []

    def snapshot(self):
        return {
            "kind": "general",
            "robot": {"type": "panda"},
            "entities": [],
            "scene_revision": 1,
            "state_token": self.token,
            "busy": False,
        }

    def submit(self, name, args, **kwargs):
        self.calls.append((name, args, kwargs))
        f = Future()
        f.set_result(
            {
                "ok": False,
                "scene_revision": 1,
                "error_code": "fixture_stopped",
                "detail": "End fixture before execution.",
            }
        )
        return f

    def stop(self):
        return {"ok": True}

    def frame(self):
        return b"live"

    def experiment_frame(self, trial_id, frame):
        return b"trial" if trial_id == "abc" and frame == 0 else None

    def experiment_metadata(self):
        return {
            "trials": [{"id": "abc", "frame_count": 1}],
            "latest_trial_id": "abc",
            "available": True,
            "error": None,
        }


def wait(client):
    for _ in range(100):
        s = client.get("/state").json()
        if not s["busy"]:
            return s
        time.sleep(0.01)
    raise AssertionError("turn did not finish")


def test_proposal_is_readonly_and_start_freezes_goal_and_state(monkeypatch):
    import astra_world.server as server

    async def interpret(*args):
        return ActionProposal(
            status="ready",
            name="circle",
            interpretation="Circle at 6cm radius.",
            goal={"kind": "circle", "target_position": [0.45, 0, 0.3], "radius": 0.06},
        )

    monkeypatch.setattr(server, "interpret_action", interpret)
    runtime = Runtime()
    with TestClient(
        create_app(runtime, adapter=AstraAdapter("http://fixture", "model", "fixture")),
        base_url="http://127.0.0.1",
    ) as client:
        assert (
            client.post("/actions/propose", json={"message": "Circle"}).status_code
            == 202
        )
        proposal = wait(client)["proposals"][-1]
        assert not runtime.calls
        assert proposal["status"] == "ready"
        assert (
            client.post(
                "/actions/start", json={"proposal_id": proposal["id"], "goal": {}}
            ).status_code
            == 422
        )
        runtime.token = "changed"
        assert (
            client.post(
                "/actions/start", json={"proposal_id": proposal["id"]}
            ).status_code
            == 409
        )
        assert not runtime.calls
        client.post("/actions/propose", json={"message": "Circle again"})
        proposal = wait(client)["proposals"][-1]
        assert (
            client.post(
                "/actions/start", json={"proposal_id": proposal["id"]}
            ).status_code
            == 202
        )
        wait(client)
        assert runtime.calls[0][0] == "draft_action"
        assert runtime.calls[0][1]["goal"]["radius"] == 0.06
        assert runtime.calls[0][2]["expected_state_token"] == "changed"
        assert (
            client.post(
                "/actions/start", json={"proposal_id": proposal["id"]}
            ).status_code
            == 409
        )


def test_unsupported_proposal_cannot_start(monkeypatch):
    import astra_world.server as server

    async def interpret(*args):
        return ActionProposal(
            status="unsupported", name="wave", interpretation="No wave evaluator yet."
        )

    monkeypatch.setattr(server, "interpret_action", interpret)
    runtime = Runtime()
    with TestClient(
        create_app(runtime, adapter=AstraAdapter("http://fixture", "model", "fixture")),
        base_url="http://127.0.0.1",
    ) as client:
        client.post("/actions/propose", json={"message": "Wave hello"})
        proposal = wait(client)["proposals"][-1]
        assert (
            client.post(
                "/actions/start", json={"proposal_id": proposal["id"]}
            ).status_code
            == 409
        )
        assert not runtime.calls


def test_preview_route_selects_recorded_frames_without_changing_live():
    with TestClient(create_app(Runtime(), adapter=AstraAdapter()), base_url="http://127.0.0.1") as client:
        assert client.get("/frame.jpg").content == b"live"
        assert (
            client.get("/frame.jpg?view=experiment&trial_id=abc&frame=0").content
            == b"trial"
        )
        assert client.get("/frame.jpg?view=experiment&frame=999").status_code == 422
        assert client.get("/frame.jpg?view=unknown").status_code == 422
        assert (
            client.get(
                "/frame.jpg?view=experiment&trial_id=missing&frame=0"
            ).status_code
            == 503
        )
        assert client.get("/experiments").json()["latest_trial_id"] == "abc"


def test_immediate_stop_terminalizes_unstarted_proposal(monkeypatch):
    import asyncio, httpx
    import astra_world.server as server

    async def interpret(*args):
        await asyncio.sleep(10)

    monkeypatch.setattr(server, "interpret_action", interpret)

    async def exercise():
        app = create_app(
            Runtime(), adapter=AstraAdapter("http://fixture", "model", "fixture")
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            await client.post("/actions/propose", json={"message": "Circle"})
            await client.post("/stop")
            await asyncio.sleep(0.01)
            state = (await client.get("/state")).json()
            assert not state["busy"]
            assert state["proposals"][-1]["status"] == "cancelled"
            assert state["turns"][-1]["status"] == "cancelled"

    asyncio.run(exercise())
