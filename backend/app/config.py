"""Application configuration, read entirely from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."
# gemini-2.0-flash was shut down on 2026-06-01 and gemini-2.5-* is closed to new projects,
# so default to a current, free-tier Flash-Lite model. Override with LLM_MODEL.
DEFAULT_MODEL = "gemini/gemini-3.5-flash-lite"


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _json_object(value: str | None) -> dict[str, Any]:
    """Parse LLM_EXTRA_PARAMS, e.g. '{"temperature": 0.2, "reasoning_effort": "low"}'."""
    if not value or not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("LLM_EXTRA_PARAMS must be a JSON object")
    return parsed


def sqlite_path_from_url(url: str) -> str:
    """Convert a SQLAlchemy-style SQLite URL into a filesystem path.

    sqlite:////data/app.db -> /data/app.db   (absolute)
    sqlite:///./app.db     -> ./app.db       (relative)
    sqlite:///:memory:     -> :memory:
    A bare path is returned unchanged.
    """
    prefix = "sqlite:///"
    if url.startswith(prefix):
        return url[len(prefix) :] or ":memory:"
    if url.startswith("sqlite://"):
        raise ValueError(f"Unsupported DATABASE_URL: {url!r}")
    return url


@dataclass(frozen=True)
class KeyConfig:
    key_id: str  # public id: key_1, key_2, ...
    api_key: str  # secret, never returned or logged
    model: str  # LiteLLM model string served by this key
    api_base: str | None = None  # optional custom endpoint (proxy, OpenAI-compatible host, ...)


@dataclass(frozen=True)
class Settings:
    keys: list[KeyConfig] = field(default_factory=list)
    database_path: str = "./data/app.db"
    allowed_origins: list[str] = field(default_factory=list)
    enable_failure_simulation: bool = False
    cache_type: str = "disk"  # "disk" (persists under /data) or "local" (in-memory)
    cache_dir: str = "./data/llm_cache"
    cache_ttl_seconds: int = 24 * 3600
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    llm_timeout_seconds: float = 30.0
    max_history_messages: int = 40
    max_output_tokens: int = 4096
    key_cooldown_seconds: float = 30.0
    extra_llm_params: dict[str, Any] = field(default_factory=dict)
    rate_limit_per_minute: int = 0  # per client IP; 0 disables
    max_body_bytes: int = 64 * 1024

    @property
    def models(self) -> list[str]:
        return sorted({k.model for k in self.keys})

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ if env is None else env)

        api_keys = _csv(env.get("LLM_API_KEYS"))
        default_model = (env.get("LLM_MODEL") or DEFAULT_MODEL).strip()
        api_base = (env.get("LLM_API_BASE") or "").strip() or None
        # Optional per-key models, aligned with LLM_API_KEYS (lets the pool span several models).
        per_key_models = _csv(env.get("LLM_KEY_MODELS"))
        keys = [
            KeyConfig(
                key_id=f"key_{i}",
                api_key=key,
                model=per_key_models[i - 1] if i - 1 < len(per_key_models) else default_model,
                api_base=api_base,
            )
            for i, key in enumerate(api_keys, start=1)
        ]

        return cls(
            keys=keys,
            database_path=sqlite_path_from_url(env.get("DATABASE_URL", "sqlite:///./data/app.db")),
            allowed_origins=_csv(env.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")),
            enable_failure_simulation=_bool(env.get("ENABLE_FAILURE_SIMULATION")),
            cache_type=env.get("CACHE_TYPE", "disk").strip().lower(),
            cache_dir=env.get("CACHE_DIR", "./data/llm_cache"),
            cache_ttl_seconds=int(env.get("CACHE_TTL_SECONDS", str(24 * 3600))),
            default_system_prompt=env.get("DEFAULT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            llm_timeout_seconds=float(env.get("LLM_TIMEOUT_SECONDS", "30")),
            max_history_messages=int(env.get("MAX_HISTORY_MESSAGES", "40")),
            max_output_tokens=int(env.get("MAX_OUTPUT_TOKENS", "4096")),
            key_cooldown_seconds=float(env.get("KEY_COOLDOWN_SECONDS", "30")),
            extra_llm_params=_json_object(env.get("LLM_EXTRA_PARAMS")),
            rate_limit_per_minute=int(env.get("RATE_LIMIT_PER_MINUTE", "0") or 0),
            max_body_bytes=int(env.get("MAX_BODY_BYTES", str(64 * 1024))),
        )
