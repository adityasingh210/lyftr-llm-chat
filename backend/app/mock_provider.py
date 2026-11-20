"""An offline LiteLLM provider ("mock/<name>") for local development, demos and tests.

It is registered through LiteLLM's custom-provider hook, so requests to it travel through the
exact same LiteLLM code path as a real provider: the same cache lookup/write, the same exception
types and the same usage accounting. That keeps tests honest without needing real API keys.

Special api_key values let you exercise *real* (non-simulated) failure handling:
  mock-fail-429   -> always raises litellm.RateLimitError
  mock-fail-500   -> always raises litellm.InternalServerError
  mock-fail-503   -> always raises litellm.ServiceUnavailableError
Any other key answers normally.
"""

from __future__ import annotations

import asyncio
import os
import time

import litellm
from litellm import CustomLLM


def _count(messages: list[dict], text: str | None = None) -> int:
    if text is not None:
        return litellm.token_counter(model="gpt-3.5-turbo", text=text)
    return litellm.token_counter(model="gpt-3.5-turbo", messages=messages)


class MockLLM(CustomLLM):
    def _respond(self, model: str, messages: list[dict], api_key: str | None) -> litellm.ModelResponse:
        key = api_key or ""
        if key.startswith("mock-fail-429"):
            raise litellm.RateLimitError(message="mock quota exhausted", llm_provider="mock", model=model)
        if key.startswith("mock-fail-500"):
            raise litellm.InternalServerError(message="mock server error", llm_provider="mock", model=model)
        if key.startswith("mock-fail-503"):
            raise litellm.ServiceUnavailableError(message="mock unavailable", llm_provider="mock", model=model)

        user_turns = [m["content"] for m in messages if m.get("role") == "user"]
        latest = user_turns[-1] if user_turns else ""
        reply = f'You said: "{latest}".'
        if len(user_turns) > 1:
            reply += f' Earlier in this chat you said: "{user_turns[0]}".'
        reply += f" (mock model, turn {len(user_turns)})"

        prompt_tokens = _count(messages)
        completion_tokens = _count([], text=reply)
        return litellm.ModelResponse(
            model=model,
            created=int(time.time()),
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    def completion(self, model, messages, api_key=None, **kwargs):  # type: ignore[override]
        time.sleep(float(os.getenv("MOCK_LATENCY_MS", "350")) / 1000)
        return self._respond(model, messages, api_key)

    async def acompletion(self, model, messages, api_key=None, **kwargs):  # type: ignore[override]
        await asyncio.sleep(float(os.getenv("MOCK_LATENCY_MS", "350")) / 1000)
        return self._respond(model, messages, api_key)


_registered = False


def register_mock_provider() -> None:
    global _registered
    if _registered:
        return
    existing = [p for p in (litellm.custom_provider_map or []) if p.get("provider") != "mock"]
    litellm.custom_provider_map = existing + [{"provider": "mock", "custom_handler": MockLLM()}]
    _registered = True
