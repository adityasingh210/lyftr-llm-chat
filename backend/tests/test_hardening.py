"""Hardening features: request ids, body limit, rate limit, path-safe session ids, context window."""

from __future__ import annotations

from urllib.parse import quote

from app.main import history_for_context


def test_request_id_is_generated_and_echoed(client):
    res = client.get("/health")
    assert len(res.headers["x-request-id"]) >= 8
    res = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert res.headers["x-request-id"] == "trace-123"


def test_oversized_body_is_413_json(client):
    res = client.post(
        "/chat",
        content=b'{"session_id":"s","message":"' + b"a" * 70_000 + b'"}',
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 413
    assert res.json() == {"detail": "request body too large"}


def test_rate_limit_is_off_by_default(client):
    for i in range(8):
        assert client.post("/chat", json={"session_id": "r", "message": f"m{i}"}).status_code == 200


def test_rate_limit_returns_429_json_with_cors(client_factory):
    client = client_factory(RATE_LIMIT_PER_MINUTE="2")
    origin = {"Origin": "http://localhost:3000"}
    assert client.post("/chat", json={"session_id": "r", "message": "1"}, headers=origin).status_code == 200
    assert client.post("/chat", json={"session_id": "r", "message": "2"}, headers=origin).status_code == 200
    res = client.post("/chat", json={"session_id": "r", "message": "3"}, headers=origin)
    assert res.status_code == 429
    assert res.json()["detail"].startswith("rate limit exceeded")
    assert int(res.headers["retry-after"]) >= 1
    assert res.headers["access-control-allow-origin"] == "http://localhost:3000"  # browser can read it
    # other endpoints are not limited, and the rejected turn was not stored
    assert client.get("/health").status_code == 200
    assert len(client.get("/sessions/r/messages").json()["messages"]) == 4


def test_session_ids_with_slashes_and_unicode(chat, client):
    for session in ["team/alpha/42", "नमस्ते-सत्र", "has space", "emoji-🚀"]:
        assert chat(session, "hi").status_code == 200
        res = client.get(f"/sessions/{quote(session, safe='')}/messages")
        assert res.status_code == 200, session
        assert res.json()["session_id"] == session


def test_unicode_and_injection_like_content_roundtrips(chat, client):
    text = "Robert'); DROP TABLE messages;-- <script>alert(1)</script> 你好 🚀"
    assert chat("weird", text).status_code == 200
    stored = client.get("/sessions/weird/messages").json()["messages"][0]["content"]
    assert stored == text


def test_history_window_never_starts_with_assistant():
    history = [
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert history_for_context(history) == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_odd_history_window_still_works(client_factory):
    client = client_factory(MAX_HISTORY_MESSAGES="3")
    for i in range(4):
        res = client.post("/chat", json={"session_id": "w", "message": f"turn {i}"})
        assert res.status_code == 200
    # full history is still stored even though only a window is sent as context
    assert len(client.get("/sessions/w/messages").json()["messages"]) == 8


def test_extra_params_are_part_of_cache_key(client_factory):
    a = client_factory(LLM_EXTRA_PARAMS='{"temperature": 0.1}')
    a.post("/chat", json={"session_id": "x1", "message": "same"})
    assert (
        a.post("/chat", json={"session_id": "x2", "message": "same"}).json()["metadata"]["resolution_source"]
        == "LLM_CACHE"
    )
    b = client_factory(LLM_EXTRA_PARAMS='{"temperature": 0.9}')  # same cache dir, different params
    assert (
        b.post("/chat", json={"session_id": "x3", "message": "same"}).json()["metadata"]["resolution_source"]
        == "LIVE_INFERENCE"
    )


def test_method_not_allowed_is_json(client):
    res = client.get("/chat")
    assert res.status_code == 405
    assert res.json() == {"detail": "Method Not Allowed"}


def test_unhandled_error_is_clean_500_with_cors(client, monkeypatch):
    from fastapi.testclient import TestClient

    async def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(client.app.state.store, "get_history", boom)
    raw = TestClient(client.app, raise_server_exceptions=False)
    res = raw.post("/chat", json={"session_id": "e", "message": "hi"}, headers={"Origin": "http://localhost:3000"})
    assert res.status_code == 500
    assert res.json() == {"detail": "internal server error"}
    assert "secret internal detail" not in res.text
    assert res.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert res.headers["x-request-id"]
