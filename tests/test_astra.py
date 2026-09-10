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


@pytest.mark.parametrize("tool_name,limit", [("observe_world", 36), ("pick_place", 12)])
def test_loop_budget_reserves_a_tool_free_summary(tool_name, limit):
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        message = {"role": "assistant", "content": "The last action completed; the tool budget is reached."}
        if body.get("tools"):
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": str(len(requests)), "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }]}
        return httpx.Response(200, json={"choices": [{"message": message}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = AstraAdapter("https://example.test/v1", "model", "key", client=client)
            calls = []
            async def execute(name, args):
                calls.append(name)
                return {"ok": True, "scene_revision": 1}
            answer = await adapter.run_turn([], execute)
            assert "last action completed" in answer
            assert len(calls) == limit
            assert requests[-1].get("tools") == []
            assert requests[-1].get("tool_choice") == "none"
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


def test_general_world_tools_expose_validated_scene_schema():
    from astra_world.astra import TOOLS

    functions = {item["function"]["name"]: item["function"] for item in TOOLS}
    assert {
        "search_assets",
        "describe_asset",
        "create_world",
        "simulate",
        "reset_world",
        "save_scenario",
        "load_scenario",
    } <= functions.keys()
    scene = functions["create_world"]["parameters"]
    assert scene["properties"]["entities"]["maxItems"] == 64
    assert scene["properties"]["robot"]["enum"] == ["none", "panda"]
    assert (
        functions["simulate"]["parameters"]["properties"]["duration"]["maximum"] == 20
    )


def test_generic_pick_place_schema_requires_object_and_target_center():
    from astra_world.astra import TOOLS

    function = next(
        item["function"] for item in TOOLS if item["function"]["name"] == "pick_place"
    )
    assert function["parameters"]["required"] == ["object_id", "target_position"]
    assert function["parameters"]["properties"]["target_position"]["minItems"] == 3


def test_add_entity_uses_entity_contract_and_vector_schema():
    from astra_world.astra import TOOLS
    from astra_world.world_spec import EntitySpec

    function = next(
        item["function"] for item in TOOLS if item["function"]["name"] == "add_entity"
    )
    schema = function["parameters"]
    assert schema["required"] == EntitySpec.model_json_schema()["required"]
    assert schema["properties"]["position"]["items"]["type"] == "number"
    assert schema["properties"]["position"]["minItems"] == 3
    assert schema["additionalProperties"] is False


def test_experiment_calls_do_not_consume_live_budget_and_summary_cannot_execute():
    requests = 0
    def handler(request):
        nonlocal requests
        requests += 1
        body = json.loads(request.content)
        if requests == 1:
            names = ["write_action_note"] * 16 + ["pick_place"] * 12
        else:
            assert body['tools'] == [] and body['tool_choice'] == 'none'
            names = ['pick_place']  # Even a noncompliant provider cannot run it.
        calls = [{"id": f"{requests}-{i}", "type": "function", "function": {"name": name, "arguments": "{}"}} for i, name in enumerate(names)]
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = AstraAdapter('https://example.test/v1', 'model', 'key', client=client)
            called = []
            history = []
            async def execute(name, args):
                called.append(name)
                return {'ok': True}
            answer = await adapter.run_turn(history, execute)
            assert called.count('write_action_note') == 16
            assert called.count('pick_place') == 12
            assert 'No further tools were run' in answer
            assert not any(item.get('role') == 'developer' for item in history)
    asyncio.run(run())
