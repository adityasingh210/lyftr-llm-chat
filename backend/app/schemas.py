from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_CHARS = 4000
MAX_SESSION_ID_CHARS = 128
MAX_SYSTEM_PROMPT_CHARS = 4000


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=MAX_SESSION_ID_CHARS)
    message: str
    system_prompt: str | None = Field(default=None, max_length=MAX_SYSTEM_PROMPT_CHARS)

    @field_validator("session_id")
    @classmethod
    def session_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session_id must not be blank")
        return value

    @field_validator("message")
    @classmethod
    def message_length_after_trim(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("message must not be empty")
        if len(trimmed) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message must be at most {MAX_MESSAGE_CHARS} characters after trimming")
        return trimmed

    @field_validator("system_prompt")
    @classmethod
    def blank_system_prompt_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ChatMetadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata
