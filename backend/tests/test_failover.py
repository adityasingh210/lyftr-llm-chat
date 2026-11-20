"""Key-pool failover: X-Simulate-Failure hook (section 4) and real provider errors."""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest

from app.llm import InvalidSimulationHeader, parse_simulation_header
from app.main import create_app
from tests.conftest import make_settings


def sid() -> str:
    return f"s-{uuid.uuid4().hex[:8]}"


# ---- header parsing ------------------------------------------------------------------------
KEYS = ["key_1", "key_2", "key_3"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, {}),
        ("", {}),
        ("key_1=429", {"key_1": 429}),
        ("key_1=429,key_2=500", {"key_1": 429, "key_2": 500}),
        (" key_1 = 502 , key_3=503 ", {"key_1": 502, "key_3": 503}),
        ("*=429", {"key_1": 429, "key_2": 429, "key_3": 429}),
        ("*=500,key_2=429", {"key_1": 500, "key_2": 429, "key_3": 500}),
        ("key_9=429", {}),  # unknown keys are ignored
    ],
)
def test_parse_simulation_header(raw, expected):
    assert parse_simulation_header(raw, KEYS) == expected


@pytest.mark.parametrize("raw", ["key_1", "key_1=abc", "key_1=404", "key_1=200"])
def test_parse_simulation_header_rejects_garbage(raw):
    with pytest.raises(InvalidSimulationHeader):
        parse_simulation_header(raw, KEYS)


# ---- simulated failover through the API ----------------------------------------------------
@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_key_1_failure_fails_over_to_key_2(chat, status):
    res = chat(sid(), f"failover {status}", headers={"X-Simulate-Failure": f"key_1={status}"})
    assert res.status_code == 200
    meta = res.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert res.json()["reply"]  # a real answer, not an error wrapped in a 200


def test_simulated_request_is_not_served_from_cache(chat):
    chat(sid(), "cached question")
    res = chat(sid(), "cached question", headers={"X-Simulate-Failure": "key_1=429"}).json()
    assert res["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert res["metadata"]["key_id"] == "key_2"


@pytest.mark.parametrize("header", ["*=429", "key_1=429,key_2=500", "*=503"])
def test_all_keys_failing_returns_503_and_writes_nothing(chat, client, header):
    s = sid()
    res = chat(s, "doomed", headers={"X-Simulate-Failure": header})
    assert res.status_code == 503
    assert res.json() == {"detail": "all providers unavailable"}
    assert client.get(f"/sessions/{s}/messages").status_code == 404


def test_failed_turn_does_not_touch_existing_history(chat, client):
    s = sid()
    chat(s, "turn one")
    assert chat(s, "turn two", headers={"X-Simulate-Failure": "*=429"}).status_code == 503
    messages = client.get(f"/sessions/{s}/messages").json()["messages"]
    assert [m["content"] for m in messages if m["role"] == "user"] == ["turn one"]


def test_simulation_is_per_request_only(chat):
    chat(sid(), "warmup", headers={"X-Simulate-Failure": "key_1=429"})
    meta = chat(sid(), "next request", headers={}).json()["metadata"]
    assert meta["key_id"] == "key_1"  # simulated failures don't cool keys down
    assert meta["fallback_attempts"] == 0


def test_header_ignored_when_simulation_disabled(client_factory):
    client = client_factory(ENABLE_FAILURE_SIMULATION="false")
    res = client.post("/chat", json={"session_id": "s", "message": "hi"}, headers={"X-Simulate-Failure": "*=429"})
    assert res.status_code == 200
    assert res.json()["metadata"]["key_id"] == "key_1"


def test_malformed_simulation_header_is_422(chat):
    res = chat(sid(), "hi", headers={"X-Simulate-Failure": "key_1=418"})
    assert res.status_code == 422
    assert "detail" in res.json()


def test_failover_with_three_keys(client_factory):
    client = client_factory(LLM_API_KEYS="mock-a,mock-b,mock-c")
    res = client.post(
        "/chat", json={"session_id": "s", "message": "three"}, headers={"X-Simulate-Failure": "key_1=429,key_2=503"}
    )
    meta = res.json()["metadata"]
    assert meta["key_id"] == "key_3" and meta["fallback_attempts"] == 2


# ---- real (non-simulated) provider errors --------------------------------------------------
def test_real_rate_limit_fails_over_and_cools_key_down(client_factory):
    client = client_factory(LLM_API_KEYS="mock-fail-429,mock-good", KEY_COOLDOWN_SECONDS="60")
    first = client.post("/chat", json={"session_id": "a", "message": "one"}).json()["metadata"]
    assert first["key_id"] == "key_2" and first["fallback_attempts"] == 1

    # key_1 is now cooling down, so it is tried last and the healthy key answers first.
    second = client.post("/chat", json={"session_id": "b", "message": "two"}).json()["metadata"]
    assert second["key_id"] == "key_2" and second["fallback_attempts"] == 0


def test_real_5xx_fails_over(client_factory):
    client = client_factory(LLM_API_KEYS="mock-fail-500,mock-good")
    meta = client.post("/chat", json={"session_id": "a", "message": "x"}).json()["metadata"]
    assert meta["key_id"] == "key_2" and meta["fallback_attempts"] == 1


def test_real_total_outage_is_clean_503(client_factory):
    client = client_factory(LLM_API_KEYS="mock-fail-429,mock-fail-503")
    res = client.post("/chat", json={"session_id": "a", "message": "x"})
    assert res.status_code == 503
    assert res.json() == {"detail": "all providers unavailable"}


def test_mixed_models_across_keys(client_factory):
    client = client_factory(LLM_API_KEYS="mock-a,mock-b", LLM_KEY_MODELS="mock/model-a,mock/model-b")
    res = client.post(
        "/chat", json={"session_id": "s", "message": "mixed"}, headers={"X-Simulate-Failure": "key_1=429"}
    ).json()
    assert res["metadata"]["model"] == "mock/model-b"


# ---- concurrency ---------------------------------------------------------------------------
def test_concurrent_turns_in_one_session_stay_ordered(tmp_path):
    app = create_app(make_settings(tmp_path))
    app.state.store.init()

    async def run() -> list[dict]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await asyncio.gather(
                *(client.post("/chat", json={"session_id": "c", "message": f"m{i}"}) for i in range(5))
            )
            res = await client.get("/sessions/c/messages")
            return res.json()["messages"]

    messages = asyncio.run(run())
    assert len(messages) == 10
    assert [m["role"] for m in messages] == ["user", "assistant"] * 5
