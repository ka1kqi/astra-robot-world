import asyncio
import json

import httpx
import pytest

from astra_world.astra import AstraAdapter, ProviderError, ConfigurationError


@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch, tmp_path):
    import astra_world.astra as module

    monkeypatch.setattr(module, "DOTENV_PATH", tmp_path / ".env")


def test_missing_configuration_is_explicit(monkeypatch):
    for name in ("ASTRA_BASE_URL", "ASTRA_MODEL", "ASTRA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    adapter = AstraAdapter.from_env()
    assert not adapter.configured
    with pytest.raises(ConfigurationError):
        asyncio.run(adapter.respond([]))


def test_provider_receives_actual_tool_result_and_preserves_conversation():
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
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
        else:
            assert (
                json.loads(body["messages"][-1]["content"])["error_code"] == "no_path"
            )
            message = {"role": "assistant", "content": "No safe route was found."}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = AstraAdapter(
                "https://example.test/v1", "configured-model", "test-key", client=client
            )
            history = [{"role": "user", "content": "Put red blocks in the left bin"}]

            async def execute(name, args):
                return {
                    "ok": False,
                    "scene_revision": 1,
                    "error_code": "no_path",
                    "detail": "blocked",
                }

            result = await adapter.run_turn(history, execute)
            assert result == "No safe route was found."
            assert history[-1]["content"] == result
            assert requests[0]["model"] == "configured-model"

    asyncio.run(run())


def test_loop_limit_bounds_physics_calls():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "x",
                                    "type": "function",
                                    "function": {
                                        "name": "observe_world",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = AstraAdapter(
                "https://example.test/v1", "model", "key", client=client
            )
            calls = []

            async def execute(name, args):
                calls.append(name)
                return {"ok": True, "scene_revision": 1}

            with pytest.raises(ProviderError, match="12"):
                await adapter.run_turn([], execute)
            assert len(calls) == 12

    asyncio.run(run())


def test_provider_error_never_exposes_response_or_credentials():
    def handler(request):
        return httpx.Response(401, text="provider secret details test-key")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = AstraAdapter(
                "https://example.test/v1", "model", "test-key", client=client
            )
            with pytest.raises(ProviderError) as error:
                await adapter.respond([])
            assert "401" in str(error.value)
            assert "test-key" not in str(error.value)

    asyncio.run(run())


def test_cancellation_repairs_history_for_the_next_provider_turn():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "cancel-me",
                                    "type": "function",
                                    "function": {
                                        "name": "sort_blocks",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = AstraAdapter(
                "https://example.test/v1", "model", "key", client=client
            )
            history = []

            async def execute(name, args):
                raise asyncio.CancelledError()

            with pytest.raises(asyncio.CancelledError):
                await adapter.run_turn(history, execute)
            assert history[-1]["tool_call_id"] == "cancel-me"
            assert json.loads(history[-1]["content"])["error_code"] == "cancelled"

    asyncio.run(run())


def test_dotenv_configuration_uses_environment_overrides(monkeypatch, tmp_path):
    import astra_world.astra as module

    env_file = tmp_path / ".env"
    env_file.write_text(
        "ASTRA_BASE_URL=https://example.test/v1\nASTRA_MODEL=file-model\nASTRA_API_KEY=fixture-only\n"
    )
    monkeypatch.setattr(module, "DOTENV_PATH", env_file, raising=False)
    for name in ("ASTRA_BASE_URL", "ASTRA_MODEL", "ASTRA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ASTRA_MODEL", "environment-model")
    adapter = AstraAdapter.from_env()
    assert adapter.configured
    assert adapter.base_url == "https://example.test/v1"
    assert adapter.model == "environment-model"
    assert "ASTRA_API_KEY" not in __import__("os").environ
    assert "fixture-only" not in repr(adapter)


def test_provider_output_is_bounded():
    def handler(request):
        assert json.loads(request.content)["max_completion_tokens"] == 2048
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Ready."}}]},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = AstraAdapter(
                "https://example.test/v1", "model", "fixture", client=client
            )
            assert (await adapter.respond([]))["content"] == "Ready."

    asyncio.run(run())


@pytest.mark.parametrize("name", [[], {}, None, 42])
def test_malformed_tool_name_returns_error_without_dispatch(name):
    requests = 0

    def handler(request):
        nonlocal requests
        requests += 1
        if requests == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "bad",
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                ],
            }
        else:
            result = json.loads(json.loads(request.content)["messages"][-1]["content"])
            assert result["error_code"] == "invalid_tool"
            message = {"role": "assistant", "content": "The tool request was invalid."}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = AstraAdapter(
                "https://example.test/v1", "model", "fixture", client=client
            )

            async def execute(name, args):
                pytest.fail("Malformed tool must never reach the runtime")

            assert (
                await adapter.run_turn([], execute) == "The tool request was invalid."
            )

    asyncio.run(run())


def test_astra_responses_replays_reasoning_and_tool_results():
    requests = []
    reasoning = {
        "type": "reasoning",
        "id": "rs_fixture",
        "summary": [],
        "encrypted_content": "opaque-fixture",
    }
    function = {
        "type": "function_call",
        "id": "fc_fixture",
        "call_id": "call_fixture",
        "name": "observe_world",
        "arguments": "{}",
        "status": "completed",
    }

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/responses"
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
        assert body["max_output_tokens"] == 4096
        assert body["reasoning"] == {"effort": "low"}
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["name"] == "list_assets"
        assert body["tools"][0]["strict"] is False
        if len(requests) == 1:
            output = [reasoning, function]
        else:
            assert reasoning in body["input"]
            assert function in body["input"]
            result = body["input"][-1]
            assert result["type"] == "function_call_output"
            assert result["call_id"] == "call_fixture"
            assert json.loads(result["output"])["scene_revision"] == 4
            output = [
                {
                    "type": "message",
                    "id": "msg_fixture",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The station is ready.",
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                }
            ]
        return httpx.Response(200, json={"status": "completed", "output": output})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = AstraAdapter(
                "https://example.test/v1", "gpt-6-astra", "fixture", client=client
            )
            history = [{"role": "user", "content": "Observe the station"}]
            events = []

            async def execute(name, args):
                return {"ok": True, "scene_revision": 4}

            assert (
                await adapter.run_turn(history, execute, events.append)
                == "The station is ready."
            )
            assert history[-1]["phase"] == "final_answer"
            assert "opaque-fixture" not in json.dumps(events)

    asyncio.run(run())


def test_astra_responses_cancellation_keeps_valid_function_output():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "cancelled_call",
                        "name": "observe_world",
                        "arguments": "{}",
                    }
                ],
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = AstraAdapter(
                "https://example.test/v1", "gpt-6-astra", "fixture", client=client
            )
            history = []

            async def execute(name, args):
                raise asyncio.CancelledError()

            with pytest.raises(asyncio.CancelledError):
                await adapter.run_turn(history, execute)
            assert history[-1]["type"] == "function_call_output"
            assert history[-1]["call_id"] == "cancelled_call"
            assert json.loads(history[-1]["output"])["error_code"] == "cancelled"

    asyncio.run(run())
