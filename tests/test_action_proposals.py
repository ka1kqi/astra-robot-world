import asyncio, json
import httpx
import pytest
from astra_world.astra import AstraAdapter, ProviderError
from astra_world.action_proposals import interpret_action, ActionProposal
from pydantic import ValidationError

WORLD = {
    "kind": "general",
    "name": "Test",
    "scene_revision": 1,
    "robot": {"type": "panda", "joints": []},
    "entities": [],
}


def test_proposal_adapter_has_only_readonly_submission_not_robot_tools():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
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
                                    "id": "p",
                                    "function": {
                                        "name": "submit_action_proposal",
                                        "arguments": json.dumps(
                                            {
                                                "status": "ready",
                                                "name": "circle",
                                                "interpretation": "Trace a horizontal gripper circle.",
                                                "goal": {
                                                    "kind": "circle",
                                                    "target_position": [0.45, 0, 0.3],
                                                },
                                                "limitations": [],
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    adapter = AstraAdapter(
        "https://example.test/v1",
        "test",
        "fixture",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = asyncio.run(interpret_action(adapter, "Draw a circle", WORLD, []))
    assert result.goal.kind == "circle"
    assert [t["function"]["name"] for t in seen[0]["tools"]] == [
        "submit_action_proposal"
    ]
    assert "No simulation tool will be executed" in seen[0]["messages"][0]["content"]


def test_unsupported_proposal_cannot_carry_executable_goal():
    with pytest.raises(ValidationError):
        ActionProposal(
            status="unsupported",
            name="spin",
            interpretation="Unsupported",
            goal={"kind": "circle", "target_position": [0.4, 0, 0.3]},
        )
    with pytest.raises(ValidationError):
        ActionProposal(status="ready", name="spin", interpretation="Spin")
