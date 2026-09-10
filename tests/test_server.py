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


def test_tool_arguments_visible_before_completion_and_linked_to_result():
    runtime = Runtime()
    args = {"object_id": "green", "target_position": [0.4, -0.12, 0.06]}
    with TestClient(create_app(runtime, adapter=AstraAdapter("", "", "")), base_url="http://127.0.0.1") as client:
        client.post("/tools/pick_place", json=args).raise_for_status()
        for _ in range(50):
            events = client.get("/state").json()["turns"][-1]["events"]
            if events:
                break
            time.sleep(.01)
        assert events[0]["arguments"] == args
        assert events[0]["call_id"]
        runtime.pending.set_result({"ok": True, "payload": {"route": "direct"}})
        for _ in range(50):
            events = client.get("/state").json()["turns"][-1]["events"]
            if len(events) == 2:
                break
            time.sleep(.01)
        assert events[1]["call_id"] == events[0]["call_id"]
        for path in ("/", "/app.js", "/styles.css"):
            assert client.get(path).headers["cache-control"] == "no-store"


def test_manual_action_busy_stop_and_missing_provider():
    runtime = Runtime()
    with TestClient(create_app(runtime, adapter=AstraAdapter("", "", "")), base_url="http://127.0.0.1") as client:
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
    with TestClient(create_app(runtime, adapter=AstraAdapter("", "", "")), base_url="http://127.0.0.1") as client:
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
    with TestClient(create_app(runtime, adapter=adapter), base_url="http://127.0.0.1") as client:
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
    with TestClient(create_app(runtime, adapter=adapter), base_url="http://127.0.0.1") as client:
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


def test_asset_search_is_read_only_and_available_during_action():
    class AssetRuntime(Runtime):
        def submit(self, name, args):
            if name == "search_assets":
                result = Future()
                result.set_result(
                    {
                        "ok": True,
                        "scene_revision": 1,
                        "payload": {
                            "assets": [{"id": "ball", "name": "Ball"}],
                            "count": 1,
                        },
                    }
                )
                return result
            return super().submit(name, args)

    with TestClient(
        create_app(AssetRuntime(), adapter=AstraAdapter("", "", "")),
        base_url="http://127.0.0.1",
    ) as client:
        assert client.post("/tools/build_sorting_station", json={}).status_code == 202
        response = client.get("/assets?query=ball")
        assert response.status_code == 200
        assert response.json()["payload"]["assets"][0]["id"] == "ball"
        assert len(client.get("/state").json()["turns"]) == 1


def test_embedded_frame_and_action_creation_validation():
    runtime = Runtime()
    runtime.frame = lambda: b"jpeg-frame"
    with TestClient(create_app(runtime, adapter=AstraAdapter("", "", "")), base_url="http://127.0.0.1") as client:
        response = client.get("/frame.jpg")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content == b"jpeg-frame"
        assert client.get("/actions").json()["payload"]["actions"] == []
        assert (
            client.post(
                "/actions/create",
                json={
                    "message": "Topple it",
                    "target_id": "green",
                    "support_id": "red",
                    "trial_budget": 5,
                },
            ).status_code
            == 503
        )
        assert (
            client.post(
                "/actions/create",
                json={
                    "message": "Topple it",
                    "target_id": "green",
                    "support_id": "red",
                    "trial_budget": 100,
                },
            ).status_code
            == 422
        )
