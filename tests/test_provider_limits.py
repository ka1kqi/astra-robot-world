import asyncio
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from astra_world.astra import AstraAdapter, ProviderError


SUCCESS = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}


def run_responses(responses, *, on_retry=None, allow_tools=True):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return responses[min(len(requests) - 1, len(responses) - 1)]

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            adapter = AstraAdapter("https://provider.invalid", "test", "secret-key", client=client)
            return await adapter.respond([], on_retry=on_retry, allow_tools=allow_tools)

    return run, requests


@pytest.fixture
def sleeps(monkeypatch):
    waits = []

    async def sleep(delay):
        waits.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return waits


def limited(code=None, *, kind=None, retry_after=None):
    return httpx.Response(429, json={"error": {"code": code, "type": kind, "message": "secret-key private provider details"}}, headers={} if retry_after is None else {"Retry-After": retry_after})


@pytest.mark.parametrize("code,kind", [
    ("credit_balance_exhausted", None),
    ("organization_spend_limit_exceeded", None),
    ("project_spend_limit_exceeded", None),
    ("organization_usage_limit_exceeded", None),
    ("insufficient_quota", None),
    (None, "insufficient_quota"),
])
def test_quota_errors_are_actionable_and_never_retried(code, kind, sleeps):
    run, requests = run_responses([limited(code, kind=kind, retry_after="1")])
    with pytest.raises(ProviderError, match="billing|credits|quota|spending limit|usage limit") as caught:
        asyncio.run(run())
    assert "secret-key" not in str(caught.value)
    assert len(requests) == 1
    assert sleeps == []


def test_transient_retry_preserves_tool_free_request_and_reports_wait(sleeps):
    statuses = []
    run, requests = run_responses([limited("rate_limit_exceeded", retry_after="2"), httpx.Response(200, json=SUCCESS)], on_retry=statuses.append, allow_tools=False)
    assert asyncio.run(run())["content"] == "done"
    assert sleeps == [2]
    assert len(requests) == 2 and requests[0] == requests[1]
    assert requests[0]["tools"] == []
    assert requests[0]["tool_choice"] == "none"
    assert len(statuses) == 1 and "2" in statuses[0]
    assert "secret-key" not in statuses[0]


@pytest.mark.parametrize("code,kind", [("rate_limit_exceeded", None), ("slow_down", None), (None, "rate_limit_error")])
def test_retry_budget_is_two_with_exponential_backoff(code, kind, sleeps):
    run, requests = run_responses([limited(code, kind=kind)])
    with pytest.raises(ProviderError, match="rate limit"):
        asyncio.run(run())
    assert len(requests) == 3
    assert len(sleeps) == 2
    assert 1 <= sleeps[0] <= 1.5
    assert 2 <= sleeps[1] <= 2.5


@pytest.mark.parametrize("response", [limited(), httpx.Response(429, text="secret-key"), limited("secret-key", retry_after="NaN"), limited(retry_after="-1")])
def test_unknown_limit_does_not_guess_quota_or_retry(response, sleeps):
    run, requests = run_responses([response])
    with pytest.raises(ProviderError, match="HTTP 429") as caught:
        asyncio.run(run())
    assert "secret-key" not in str(caught.value)
    assert len(requests) == 1 and sleeps == []


def test_unknown_limit_with_valid_retry_hint_can_recover(sleeps):
    run, requests = run_responses([limited(retry_after="0"), httpx.Response(200, json=SUCCESS)])
    assert asyncio.run(run())["content"] == "done"
    assert sleeps == [0] and len(requests) == 2


@pytest.mark.parametrize("wait", ["16", "120", "1e100"])
def test_long_retry_after_never_retries_early(wait, sleeps):
    run, requests = run_responses([limited("rate_limit_exceeded", retry_after=wait)])
    with pytest.raises(ProviderError, match="wait"):
        asyncio.run(run())
    assert sleeps == [] and len(requests) == 1


def test_retry_after_http_date_is_honored(sleeps):
    future = datetime.now(timezone.utc) + timedelta(seconds=10)
    run, _ = run_responses([limited("slow_down", retry_after=format_datetime(future, usegmt=True)), httpx.Response(200, json=SUCCESS)])
    assert asyncio.run(run())["content"] == "done"
    assert len(sleeps) == 1 and 8 <= sleeps[0] <= 10


def test_cancellation_during_backoff_does_not_send_again():
    requests = []

    async def run():
        entered = asyncio.Event()
        def handle(request):
            requests.append(request)
            return limited("rate_limit_exceeded", retry_after="15")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            adapter = AstraAdapter("https://provider.invalid", "test", "secret-key", client=client)
            task = asyncio.create_task(adapter.respond([], on_retry=lambda _: entered.set()))
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    asyncio.run(run())
    assert len(requests) == 1


def test_retry_after_live_tool_result_never_reexecutes_robot_action(sleeps):
    requests = []
    executed = []
    first = {"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call1", "type": "function", "function": {
            "name": "sort_blocks", "arguments": '{"color":"red","destination_id":"left_bin"}'
        }}],
    }}]}

    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json=first)
        if len(requests) == 2:
            return limited("rate_limit_exceeded", retry_after="0")
        return httpx.Response(200, json=SUCCESS)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            adapter = AstraAdapter("https://provider.invalid", "test", "secret-key", client=client)
            async def execute(name, arguments):
                executed.append((name, arguments))
                return {"ok": True, "scene_revision": 2}
            return await adapter.run_turn([], execute)

    assert asyncio.run(run()) == "done"
    assert executed == [("sort_blocks", {"color": "red", "destination_id": "left_bin"})]
    assert len(requests) == 3
    assert requests[1] == requests[2]
    assert requests[2]["messages"][-1]["role"] == "tool"
    assert json.loads(requests[2]["messages"][-1]["content"]) == {"ok": True, "scene_revision": 2}
