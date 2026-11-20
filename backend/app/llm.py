"""LLM gateway: ordered key pool with automatic failover, built on LiteLLM.

Design
------
* Every API key is a separate LiteLLM deployment (key_id -> api_key + model).
* A request walks the pool in order (key_1, key_2, ...). Keys that recently hit a real 429 or an
  auth error are cooled down and tried last, so a dead key doesn't add latency to every request.
* Any provider error (429 / quota, 5xx, timeout, connection error, auth error) moves on to the
  next key. If every key fails, AllProvidersUnavailable is raised and the API returns 503.
* Caching is LiteLLM's built-in cache (disk by default, in-memory optional). LiteLLM keys the
  cache on the model + full message list (it deliberately ignores api_key), so the same
  conversation hits the cache no matter which key produced the original answer.
* X-Simulate-Failure is injected through LiteLLM's own `mock_response` hook, so a simulated 429
  raises the exact same `litellm.RateLimitError` a real provider would and travels the same
  failover path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import litellm
from litellm.caching.caching import Cache, LiteLLMCacheType

from .config import KeyConfig, Settings
from .mock_provider import register_mock_provider

logger = logging.getLogger("app.llm")

SIMULATABLE_STATUSES = {429, 500, 502, 503}
COOLDOWN_STATUSES = {401, 403, 429}


class AllProvidersUnavailable(Exception):
    """Every key in the pool failed for this request."""


class InvalidSimulationHeader(ValueError):
    pass


class EmptyReplyError(Exception):
    """The provider answered 200 but without usable text; treated like any other key failure."""


@dataclass
class Attempt:
    key_id: str
    status: int | None
    error: str
    simulated: bool


@dataclass
class LLMResult:
    reply: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_hit: bool
    key_id: str | None
    fallback_attempts: int
    attempts: list[Attempt] = field(default_factory=list)


def parse_simulation_header(raw: str | None, key_ids: list[str]) -> dict[str, int]:
    """Parse `key_1=429,key_2=500` / `*=429` into {key_id: status}.

    Unknown key ids are ignored (they can't affect the pool); malformed pairs or unsupported
    statuses raise InvalidSimulationHeader so the caller gets a clear 422 instead of a silent no-op.
    """
    if raw is None or not raw.strip():
        return {}
    result: dict[str, int] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise InvalidSimulationHeader(f"invalid X-Simulate-Failure entry {pair!r}; expected <key_id>=<status>")
        target, _, status_raw = pair.partition("=")
        target, status_raw = target.strip(), status_raw.strip()
        try:
            status = int(status_raw)
        except ValueError:
            raise InvalidSimulationHeader(f"invalid status {status_raw!r} in X-Simulate-Failure") from None
        if status not in SIMULATABLE_STATUSES:
            raise InvalidSimulationHeader(
                f"unsupported status {status} in X-Simulate-Failure; allowed: {sorted(SIMULATABLE_STATUSES)}"
            )
        if target == "*":
            for key_id in key_ids:
                result[key_id] = status
        elif target in key_ids:
            result[target] = status
    return result


def _simulated_exception(status: int, key: KeyConfig) -> Exception:
    provider = key.model.split("/", 1)[0] if "/" in key.model else "openai"
    message = f"Simulated HTTP {status} for {key.key_id} (X-Simulate-Failure)"
    if status == 429:
        return litellm.RateLimitError(message=message, llm_provider=provider, model=key.model)
    if status == 503:
        return litellm.ServiceUnavailableError(message=message, llm_provider=provider, model=key.model)
    if status == 500:
        return litellm.InternalServerError(message=message, llm_provider=provider, model=key.model)
    return litellm.APIError(
        status_code=status,
        message=message,
        llm_provider=provider,
        model=key.model,
        request=httpx.Request("POST", "https://simulated.invalid"),
    )


def _status_of(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    if isinstance(exc, litellm.Timeout | asyncio.TimeoutError):
        return 408
    return None


def _usage_int(usage: Any, name: str) -> int:
    value = getattr(usage, name, None) if usage is not None else None
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    return int(value or 0)


class LLMGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.keys = list(settings.keys)
        self._cooldown_until: dict[str, float] = {}
        register_mock_provider()
        litellm.suppress_debug_info = True
        litellm.drop_params = True
        litellm.num_retries = 0  # failover is ours; never retry the same key inside LiteLLM
        self.cache_enabled = self._configure_cache()

    # ---- cache -----------------------------------------------------------------------------
    def _configure_cache(self) -> bool:
        cache_type = self.settings.cache_type
        if cache_type in {"none", "off", "disabled"}:
            litellm.cache = None
            return False
        if cache_type == "disk":
            os.makedirs(self.settings.cache_dir, exist_ok=True)
            litellm.cache = Cache(
                type=LiteLLMCacheType.DISK,
                disk_cache_dir=self.settings.cache_dir,
                ttl=self.settings.cache_ttl_seconds,
            )
        else:
            litellm.cache = Cache(type=LiteLLMCacheType.LOCAL, ttl=self.settings.cache_ttl_seconds)
        litellm.enable_cache()  # ensure the caching callbacks are registered
        return True

    # ---- key ordering ----------------------------------------------------------------------
    def _ordered_keys(self) -> list[KeyConfig]:
        now = time.monotonic()
        healthy = [k for k in self.keys if self._cooldown_until.get(k.key_id, 0) <= now]
        cooling = [k for k in self.keys if self._cooldown_until.get(k.key_id, 0) > now]
        return healthy + cooling

    def _mark_failure(self, key: KeyConfig, status: int | None) -> None:
        if status in COOLDOWN_STATUSES and self.settings.key_cooldown_seconds > 0:
            self._cooldown_until[key.key_id] = time.monotonic() + self.settings.key_cooldown_seconds

    def _mark_success(self, key: KeyConfig) -> None:
        self._cooldown_until.pop(key.key_id, None)

    def cooling_down(self) -> list[str]:
        now = time.monotonic()
        return sorted(k for k, v in self._cooldown_until.items() if v > now)

    # ---- completion ------------------------------------------------------------------------
    def _call_kwargs(self) -> dict[str, Any]:
        # Only parameters that are identical for every attempt go here, so the cache key
        # (model + messages + these) stays stable across keys.
        return {"max_tokens": self.settings.max_output_tokens, **self.settings.extra_llm_params}

    async def _attempt(
        self,
        key: KeyConfig,
        messages: list[dict[str, str]],
        params: dict[str, Any],
        simulated_status: int | None,
        skip_cache_read: bool,
    ) -> tuple[Any, str]:
        """One provider call for one key. Raises on any failure, including a malformed/empty reply."""
        call: dict[str, Any] = {
            "model": key.model,
            "messages": messages,
            "api_key": key.api_key,
            "timeout": self.settings.llm_timeout_seconds,
            "num_retries": 0,
            "caching": self.cache_enabled,
            **params,
        }
        if key.api_base:
            call["api_base"] = key.api_base
        if skip_cache_read:
            call["cache"] = {"no-cache": True}
        if simulated_status is not None:
            call["mock_response"] = _simulated_exception(simulated_status, key)

        # Belt and braces: LiteLLM's timeout is per HTTP read; this bounds the whole attempt
        # (connect + retries inside the SDK + parsing) so one hung key can't stall the request.
        response = await asyncio.wait_for(litellm.acompletion(**call), timeout=self.settings.llm_timeout_seconds + 5)
        try:
            reply = (response.choices[0].message.content or "").strip()
        except (AttributeError, IndexError, TypeError) as exc:
            raise EmptyReplyError("provider returned no choices") from exc
        if not reply:
            finish = getattr(response.choices[0], "finish_reason", None)
            raise EmptyReplyError(f"provider returned an empty reply (finish_reason={finish})")
        return response, reply

    async def complete(self, messages: list[dict[str, str]], simulate: dict[str, int] | None = None) -> LLMResult:
        simulate = simulate or {}
        attempts: list[Attempt] = []
        params = self._call_kwargs()

        for key in self._ordered_keys():
            simulated_status = simulate.get(key.key_id)
            try:
                # A failure-simulation request exercises the provider path, so it never reads the
                # cache (a hit would hide failover). Fresh answers are still written to the cache.
                response, reply = await self._attempt(
                    key, messages, params, simulated_status, skip_cache_read=bool(simulate)
                )
            except Exception as exc:  # noqa: BLE001 - any provider failure triggers failover
                status = _status_of(exc)
                attempts.append(
                    Attempt(
                        key_id=key.key_id,
                        status=status,
                        error=type(exc).__name__,
                        simulated=simulated_status is not None,
                    )
                )
                logger.warning(
                    "llm attempt failed key_id=%s status=%s error=%s simulated=%s",
                    key.key_id,
                    status,
                    type(exc).__name__,
                    simulated_status is not None,
                )
                if simulated_status is None:
                    self._mark_failure(key, status)
                continue

            cache_hit = bool((getattr(response, "_hidden_params", None) or {}).get("cache_hit"))
            if not cache_hit:
                self._mark_success(key)
                await self._write_cache(response, key.model, messages, params)

            usage = getattr(response, "usage", None)
            return LLMResult(
                reply=reply,
                model=key.model,
                input_tokens=_usage_int(usage, "prompt_tokens"),
                output_tokens=_usage_int(usage, "completion_tokens"),
                cache_hit=cache_hit,
                key_id=None if cache_hit else key.key_id,
                fallback_attempts=0 if cache_hit else len(attempts),
                attempts=attempts,
            )

        raise AllProvidersUnavailable(attempts)

    async def _write_cache(self, response: Any, model: str, messages: list[dict], params: dict[str, Any]) -> None:
        """Write the fresh response to LiteLLM's cache before we return.

        LiteLLM schedules its own cache write as a background task, which can land *after* an
        immediate follow-up request. Writing synchronously here (same cache, same key) makes the
        "second identical request is a cache hit" guarantee deterministic.
        """
        if not self.cache_enabled or litellm.cache is None:
            return
        try:
            await litellm.cache.async_add_cache(response, model=model, messages=messages, **params)
        except Exception:  # noqa: BLE001 - caching must never break a successful reply
            logger.exception("failed to write llm cache")
