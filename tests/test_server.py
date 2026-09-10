from concurrent.futures import Future
import time

from fastapi.testclient import TestClient
from astra_world.server import create_app
from astra_world.astra import AstraAdapter


class Runtime:
    def __init__(self):
        self.pending = Future()
        self.stopped = False

    def submit(self, name, args):
        return self.pending

    def snapshot(self):
        return {"scene_revision": 1, "objects": []}

    def stop(self):
        self.stopped = True
        return {"ok": True, "scene_revision": 1}


def test_manual_action_busy_stop_and_missing_provider():
    runtime = Runtime()
    with TestClient(create_app(runtime, adapter=AstraAdapter("", "", ""))) as client:
        state = client.get("/state").json()
        assert state["provider"]["configured"] is False
        assert (
            client.post("/chat", json={"message": "Build a scene"}).status_code == 503
        )
        assert client.post("/chat", json={"message": "  "}).status_code == 422
        assert client.post("/tools/not_a_tool", json={}).status_code == 404
        assert client.post("/tools/build_sorting_station", json={}).status_code == 202
        assert (
            client.post(
                "/tools/sort_blocks",
                json={"color": "red", "destination_id": "left_bin"},
            ).status_code
            == 409
        )
        assert client.post("/stop").json()["ok"]
        assert runtime.stopped
        for _ in range(20):
            state = client.get("/state").json()
            if not state["busy"]:
                break
            time.sleep(0.01)
        assert not state["busy"]
        assert state["turns"][-1]["status"] == "cancelled"


def test_tool_failure_visible_and_static_ui_served():
    runtime = Runtime()
    runtime.pending.set_result(
        {
            "ok": False,
            "scene_revision": 1,
            "error_code": "no_previous_task",
            "detail": "Nothing to retry.",
        }
    )
    with TestClient(create_app(runtime, adapter=AstraAdapter("", "", ""))) as client:
        assert client.get("/").status_code == 200
        assert client.get("/app.js").status_code == 200
        assert client.post("/tools/retry_last_task", json={}).status_code == 202
        for _ in range(20):
            state = client.get("/state").json()
            if not state["busy"]:
                break
            time.sleep(0.01)
        assert state["turns"][-1]["status"] == "failed"
        assert (
            state["turns"][-1]["events"][-1]["result"]["error_code"]
            == "no_previous_task"
        )


def test_live_turn_preserves_failed_action_status():
    import httpx

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        message = {"role": "assistant", "content": "No safe path was found."}
        if calls == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "sort_blocks",
                            "arguments": '{"color":"red","destination_id":"left_bin"}',
                        },
                    }
                ],
            }
        return httpx.Response(200, json={"choices": [{"message": message}]})

    runtime = Runtime()
    runtime.pending.set_result(
        {
            "ok": False,
            "scene_revision": 1,
            "error_code": "no_path",
            "detail": "Blocked.",
        }
    )
    adapter = AstraAdapter(
        "https://example.test/v1",
        "model",
        "fixture",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with TestClient(create_app(runtime, adapter=adapter)) as client:
        assert (
            client.post("/chat", json={"message": "Sort red blocks"}).status_code == 202
        )
        for _ in range(20):
            state = client.get("/state").json()
            if not state["busy"]:
                break
            time.sleep(0.01)
        assert state["turns"][-1]["status"] == "failed"
        assert state["turns"][-1]["events"][-1]["text"] == "No safe path was found."


def test_malformed_astra_arguments_are_visible_and_fail_the_turn():
    import httpx

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            output = [
                {
                    "type": "function_call",
                    "call_id": "invalid-sort",
                    "name": "sort_blocks",
                    "arguments": "not valid json",
                }
            ]
        else:
            output = [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "I could not perform that action.",
                        }
                    ],
                }
            ]
        return httpx.Response(200, json={"status": "completed", "output": output})

    runtime = Runtime()
    adapter = AstraAdapter(
        "https://example.test/v1",
        "gpt-6-astra",
        "fixture",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with TestClient(create_app(runtime, adapter=adapter)) as client:
        assert (
            client.post("/chat", json={"message": "Sort red blocks"}).status_code == 202
        )
        for _ in range(20):
            state = client.get("/state").json()
            if not state["busy"]:
                break
            time.sleep(0.01)
        turn = state["turns"][-1]
        assert turn["status"] == "failed"
        assert any(
            event.get("result", {}).get("error_code") == "invalid_tool"
            for event in turn["events"]
        )
        assert turn["events"][-1]["text"] == "I could not perform that action."
