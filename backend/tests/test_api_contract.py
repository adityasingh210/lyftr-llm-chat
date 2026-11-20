"""Contract tests for /health, /chat and /sessions/{id}/messages (assignment sections 3-4)."""

from __future__ import annotations

import uuid

import pytest


def sid() -> str:
    return f"s-{uuid.uuid4().hex[:8]}"


# ---- /health -------------------------------------------------------------------------------
def test_health_reports_pool_and_cache(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["keys_configured"] == 2
    assert body["cache_enabled"] is True


def test_health_is_503_without_keys(client_factory):
    client = client_factory(LLM_API_KEYS="")
    res = client.get("/health")
    assert res.status_code == 503
    assert res.json()["keys_configured"] == 0


# ---- /chat response shape ------------------------------------------------------------------
def test_chat_response_matches_contract(chat):
    res = chat("s-7f3a", "What is the capital of France?")
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == "s-7f3a"
    assert isinstance(body["reply"], str) and body["reply"]
    meta = body["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == "mock/test-model"
    assert isinstance(meta["latency_ms"], int) and meta["latency_ms"] > 0
    assert isinstance(meta["input_tokens"], int) and meta["input_tokens"] > 0
    assert isinstance(meta["output_tokens"], int) and meta["output_tokens"] > 0
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0


def test_api_keys_never_leak(client, chat):
    texts = [
        client.get("/health").text,
        chat("s-leak", "hi").text,
        chat("s-leak2", "hi", headers={"X-Simulate-Failure": "*=429"}).text,
        client.get("/sessions/s-leak/messages").text,
    ]
    for text in texts:
        assert "mock-key-one" not in text and "mock-key-two" not in text


# ---- validation (422) ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "s1"},  # missing message
        {"message": "hi"},  # missing session_id
        {"session_id": "s1", "message": ""},
        {"session_id": "s1", "message": "   \n\t "},  # empty after trimming
        {"session_id": "", "message": "hi"},
        {"session_id": "   ", "message": "hi"},
        {"session_id": "x" * 129, "message": "hi"},
        {"session_id": "s1", "message": "a" * 4001},
        {"session_id": "s1", "message": 123},
        {"session_id": "s1", "message": "hi", "system_prompt": "p" * 4001},
    ],
)
def test_invalid_payloads_return_422_json(client, payload):
    res = client.post("/chat", json=payload)
    assert res.status_code == 422
    assert "detail" in res.json()
    assert client.get("/sessions/s1/messages").status_code == 404  # nothing persisted


def test_malformed_json_returns_422(client):
    res = client.post("/chat", content=b"{not json", headers={"Content-Type": "application/json"})
    assert res.status_code == 422
    assert "detail" in res.json()


def test_message_is_trimmed_and_limit_applies_after_trim(chat, client):
    res = chat("s-trim", "  " + "a" * 4000 + "  ")
    assert res.status_code == 200
    stored = client.get("/sessions/s-trim/messages").json()["messages"][0]["content"]
    assert stored == "a" * 4000


def test_session_id_boundaries(chat):
    assert chat("x", "hi").status_code == 200
    assert chat("y" * 128, "hi").status_code == 200


# ---- caching -------------------------------------------------------------------------------
def test_same_first_message_in_new_session_hits_cache(chat):
    first = chat(sid(), "Tell me a fun fact about owls").json()
    second = chat(sid(), "Tell me a fun fact about owls").json()

    assert first["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second["reply"] == first["reply"]
    assert second["metadata"]["key_id"] is None
    assert second["metadata"]["fallback_attempts"] == 0
    assert second["metadata"]["input_tokens"] == first["metadata"]["input_tokens"]
    assert second["metadata"]["output_tokens"] == first["metadata"]["output_tokens"]
    assert second["metadata"]["latency_ms"] < first["metadata"]["latency_ms"]


def test_different_message_is_never_served_from_cache(chat):
    chat(sid(), "message number one")
    res = chat(sid(), "message number two").json()
    assert res["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_key_includes_history(chat):
    s = sid()
    chat(s, "first")
    chat(sid(), "second")  # "second" as a first message is now cached
    res = chat(s, "second").json()  # same text, but different message list (has history)
    assert res["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_key_includes_system_prompt(chat):
    chat(sid(), "same text", system_prompt="You are a pirate.")
    res = chat(sid(), "same text", system_prompt="You are a poet.").json()
    assert res["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    res = chat(sid(), "same text", system_prompt="You are a poet.").json()
    assert res["metadata"]["resolution_source"] == "LLM_CACHE"


def test_in_memory_cache_also_works(client_factory):
    client = client_factory(CACHE_TYPE="local")
    a = client.post("/chat", json={"session_id": "a", "message": "ping"}).json()
    b = client.post("/chat", json={"session_id": "b", "message": "ping"}).json()
    assert a["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert b["metadata"]["resolution_source"] == "LLM_CACHE"


def test_cache_disabled(client_factory):
    client = client_factory(CACHE_TYPE="none")
    assert client.get("/health").json()["cache_enabled"] is False
    client.post("/chat", json={"session_id": "a", "message": "ping"})
    b = client.post("/chat", json={"session_id": "b", "message": "ping"}).json()
    assert b["metadata"]["resolution_source"] == "LIVE_INFERENCE"


# ---- tokens --------------------------------------------------------------------------------
def test_tokens_are_per_request_not_cumulative(chat):
    a = chat(sid(), "abcdefgh one").json()["metadata"]
    b = chat(sid(), "abcdefgh two").json()["metadata"]
    assert abs(a["input_tokens"] - b["input_tokens"]) <= 2  # equal-length first messages

    s = sid()
    t1 = chat(s, "hello there").json()["metadata"]
    t2 = chat(s, "ok").json()["metadata"]
    # Second turn's prompt includes the history, so input grows, but output is just this reply.
    assert t2["input_tokens"] > t1["input_tokens"]
    assert t2["output_tokens"] < t1["output_tokens"] + t2["output_tokens"]
    assert t2["input_tokens"] < t1["input_tokens"] + t1["output_tokens"] + t2["input_tokens"]


# ---- multi-turn state ----------------------------------------------------------------------
def test_multi_turn_memory_and_history_endpoint(chat, client):
    s = sid()
    chat(s, "My name is Sudarshan")
    reply = chat(s, "What is my name?").json()["reply"]
    assert "My name is Sudarshan" in reply  # mock model echoes the first turn it was sent

    res = client.get(f"/sessions/{s}/messages")
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == s
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert body["messages"][0]["content"] == "My name is Sudarshan"
    assert "metadata" not in body["messages"][0]
    meta = body["messages"][1]["metadata"]
    assert meta["resolution_source"] in {"LIVE_INFERENCE", "LLM_CACHE"}
    assert {"input_tokens", "output_tokens"} <= meta.keys()
    assert all(m["role"] != "system" for m in body["messages"])


def test_sessions_are_isolated(chat, client):
    a, b = sid(), sid()
    chat(a, "alpha secret")
    reply_b = chat(b, "what did I say?").json()["reply"]
    assert "alpha secret" not in reply_b
    assert len(client.get(f"/sessions/{b}/messages").json()["messages"]) == 2


def test_unknown_session_history_is_404(client):
    res = client.get("/sessions/does-not-exist/messages")
    assert res.status_code == 404
    assert res.json() == {"detail": "session not found"}


def test_system_prompt_persists_for_session(chat, client):
    s = sid()
    first = chat(s, "hi", system_prompt="You are terse.").json()["metadata"]
    second = chat(s, "hi again").json()["metadata"]
    store = client.app.state.store
    session = store._get_session(s)
    assert session["system_prompt"] == "You are terse."
    assert first["input_tokens"] > 0 and second["input_tokens"] > first["input_tokens"]


def test_history_and_cache_survive_restart(client_factory):
    first_app = client_factory()
    s = sid()
    original = first_app.post("/chat", json={"session_id": s, "message": "persist me"}).json()

    restarted = client_factory()  # same DB file and cache dir, new process state
    history = restarted.get(f"/sessions/{s}/messages").json()["messages"]
    assert [m["content"] for m in history] == ["persist me", original["reply"]]
    again = restarted.post("/chat", json={"session_id": sid(), "message": "persist me"}).json()
    assert again["metadata"]["resolution_source"] == "LLM_CACHE"


# ---- misc ----------------------------------------------------------------------------------
def test_unknown_route_is_json(client):
    res = client.get("/nope")
    assert res.status_code == 404
    assert res.json() == {"detail": "Not Found"}


def test_cors_allows_frontend_origin_and_simulation_header(client):
    res = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-simulate-failure",
        },
    )
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://localhost:3000"
