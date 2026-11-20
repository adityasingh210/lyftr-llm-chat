from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MOCK_LATENCY_MS", "80")  # live calls are measurably slower than cache hits

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


def make_settings(tmp_path, **overrides) -> Settings:
    env = {
        "LLM_API_KEYS": "mock-key-one,mock-key-two",
        "LLM_MODEL": "mock/test-model",
        "DATABASE_URL": f"sqlite:///{tmp_path}/app.db",
        "CACHE_DIR": str(tmp_path / "llm_cache"),
        "CACHE_TYPE": "disk",
        "ENABLE_FAILURE_SIMULATION": "true",
        "ALLOWED_ORIGINS": "http://localhost:3000",
    }
    env.update({k: str(v) for k, v in overrides.items()})
    return Settings.from_env(env)


@pytest.fixture
def client_factory(tmp_path) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(**overrides) -> TestClient:
        client = TestClient(create_app(make_settings(tmp_path, **overrides)))
        client.__enter__()  # run lifespan (schema creation)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(client_factory) -> TestClient:
    return client_factory()


@pytest.fixture
def chat(client):
    def _chat(session_id: str, message: str, headers: dict | None = None, **extra):
        return client.post("/chat", json={"session_id": session_id, "message": message, **extra}, headers=headers or {})

    return _chat
