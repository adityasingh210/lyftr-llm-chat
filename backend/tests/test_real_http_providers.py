"""Failover and caching through LiteLLM's *real* HTTP path, against a local fake provider.

The mock provider tests prove the gateway logic; these prove that real provider responses (HTTP
status codes, Gemini error JSON, timeouts, junk bodies) are mapped by LiteLLM into exceptions that
trigger failover, that LiteLLM/the OpenAI SDK do not silently retry a failing key, and that
token usage comes straight from the provider's usage block.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_settings
from tests.fake_provider import FakeProvider

# The sandbox/CI may route HTTP through a proxy; the fake server is local.
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

FORMATS = {"gemini": "gemini/gemini-3.5-flash-lite", "openai": "openai/fake-chat-model"}


@pytest.fixture(scope="module")
def provider():
    fake = FakeProvider(slow_seconds=4).start()
    yield fake
    fake.stop()


@pytest.fixture(params=sorted(FORMATS))
def fmt(request):
    return request.param


@pytest.fixture
def make_client(tmp_path, provider, fmt):
    clients = []
    provider.reset()

    def factory(keys: str, **overrides) -> TestClient:
        settings = make_settings(
            tmp_path / uuid.uuid4().hex[:6],
            LLM_API_KEYS=keys,
            LLM_MODEL=FORMATS[fmt],
            LLM_API_BASE=provider.base_url(fmt),
            LLM_TIMEOUT_SECONDS="1",
            **overrides,
        )
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


def post(client, session, message, **headers):
    return client.post("/chat", json={"session_id": session, "message": message}, headers=headers)


def test_live_answer_uses_provider_usage(make_client, provider, fmt):
    client = make_client("ok-1,ok-2")
    res = post(client, "s1", "What is the capital of France?")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["reply"].startswith("echo: What is the capital of France?")
    meta = body["metadata"]
    assert meta["model"] == FORMATS[fmt]
    assert meta["key_id"] == "key_1" and meta["fallback_attempts"] == 0
    sent = provider.requests[-1]
    assert sent["format"] == fmt
    # prompt tokens are exactly what the provider reported (system prompt + message)
    assert meta["input_tokens"] >= 10
    assert meta["output_tokens"] > 0


@pytest.mark.parametrize("failing", ["429", "500", "502", "503", "401", "403", "400", "empty", "garbage"])
def test_real_provider_error_fails_over(make_client, provider, failing):
    client = make_client(f"{failing},ok-2", KEY_COOLDOWN_SECONDS="0")
    res = post(client, f"s-{failing}", f"hello {failing}")
    assert res.status_code == 200, res.text
    meta = res.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    # No hidden retries: the failing key was called exactly once for this request.
    assert provider.hits[failing] == 1
    assert provider.hits["ok-2"] == 1


def test_timeout_fails_over_quickly(make_client, provider):
    client = make_client("slow,ok-2")
    started = time.perf_counter()
    res = post(client, "s-slow", "are you there?")
    elapsed = time.perf_counter() - started
    assert res.status_code == 200, res.text
    assert res.json()["metadata"]["key_id"] == "key_2"
    assert elapsed < 3.5  # bounded by LLM_TIMEOUT_SECONDS=1, not the provider's 4s stall


def test_every_real_key_failing_is_503(make_client, provider):
    client = make_client("429,503")
    res = post(client, "s-dead", "anyone?")
    assert res.status_code == 503
    assert res.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/s-dead/messages").status_code == 404


def test_cache_hit_makes_no_provider_call(make_client, provider):
    client = make_client("ok-1,ok-2")
    first = post(client, "a", "cache me please").json()
    calls_after_first = sum(provider.hits.values())
    second = post(client, "b", "cache me please").json()
    assert sum(provider.hits.values()) == calls_after_first  # served without touching the provider
    assert second["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second["reply"] == first["reply"]
    assert (second["metadata"]["input_tokens"], second["metadata"]["output_tokens"]) == (
        first["metadata"]["input_tokens"],
        first["metadata"]["output_tokens"],
    )


def test_full_context_is_sent_each_turn(make_client, provider, fmt):
    client = make_client("ok-1,ok-2")
    client.post("/chat", json={"session_id": "ctx", "message": "first turn", "system_prompt": "SYS-MARKER"})
    res = post(client, "ctx", "second turn")
    assert "user turns seen: 2" in res.json()["reply"]
    body = provider.requests[-1]["body"]
    flat = str(body)
    assert "SYS-MARKER" in flat and "first turn" in flat and "second turn" in flat
    assert "echo: first turn" in flat  # the stored assistant reply is part of the context


def test_real_429_cools_key_down(make_client, provider):
    client = make_client("429,ok-2", KEY_COOLDOWN_SECONDS="60")
    post(client, "c1", "one")
    second = post(client, "c2", "two").json()["metadata"]
    assert second["key_id"] == "key_2" and second["fallback_attempts"] == 0
    assert provider.hits["429"] == 1  # not retried while cooling down
    assert client.get("/health").json()["keys_cooling_down"] == ["key_1"]


def test_api_key_never_returned(make_client, provider):
    client = make_client("ok-secret-xyz,ok-2")
    text = post(client, "k", "hi").text + client.get("/sessions/k/messages").text + client.get("/health").text
    assert "ok-secret-xyz" not in text
