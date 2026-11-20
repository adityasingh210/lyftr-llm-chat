"""FastAPI application: /health, /chat, /sessions/{session_id}/messages."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .db import Store
from .llm import AllProvidersUnavailable, InvalidSimulationHeader, LLMGateway, parse_simulation_header
from .schemas import ChatMetadata, ChatRequest, ChatResponse

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


_handler = logging.StreamHandler()
_handler.addFilter(_RequestIdFilter())
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"))
logger = logging.getLogger("app")
logger.handlers = [_handler]
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logger.propagate = False

ALL_PROVIDERS_UNAVAILABLE = "all providers unavailable"


class SessionLocks:
    """One asyncio.Lock per session so concurrent turns in the same chat stay ordered."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._users: dict[str, int] = {}

    @asynccontextmanager
    async def hold(self, session_id: str) -> AsyncIterator[None]:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        self._users[session_id] = self._users.get(session_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._users[session_id] -= 1
            if self._users[session_id] == 0:
                self._users.pop(session_id, None)
                self._locks.pop(session_id, None)


class RateLimiter:
    """Sliding-window limit per client IP for POST /chat (protects the key pool's quota)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}

    def check(self, client: str) -> float | None:
        """Return None if allowed, else seconds until the next request is allowed."""
        if self.per_minute <= 0:
            return None
        now = time.monotonic()
        window = self._hits.setdefault(client, deque())
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return max(0.0, 60 - (now - window[0]))
        window.append(now)
        if len(self._hits) > 10_000:  # bound memory under a flood of distinct IPs
            for ip in [ip for ip, w in self._hits.items() if not w or now - w[-1] >= 60]:
                self._hits.pop(ip, None)
        return None


async def _guarded(call_next: Any, request: Request) -> Any:
    """Turn any unhandled exception into a JSON 500 *inside* the CORS middleware, so browsers can
    read it, and never leak the exception text or a stack trace to the client."""
    try:
        return await call_next(request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("unhandled error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})


def history_for_context(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Messages to send as context: drop any leading assistant turn left by the history window."""
    start = 0
    while start < len(history) and history[start]["role"] != "user":
        start += 1
    return [{"role": m["role"], "content": m["content"]} for m in history[start:]]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.database_path)
    gateway = LLMGateway(settings)
    locks = SessionLocks()
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        store.init()
        logger.info(
            "startup: keys=%d models=%s cache=%s db=%s failure_simulation=%s",
            len(settings.keys),
            settings.models,
            settings.cache_type,
            settings.database_path,
            settings.enable_failure_simulation,
        )
        if len(settings.keys) < 2:
            logger.warning("LLM_API_KEYS has %d key(s); at least 2 are expected for failover", len(settings.keys))
        yield

    docs_enabled = os.getenv("ENABLE_DOCS", "false").lower() == "true"
    app = FastAPI(
        title="LLM Chatbot with Key Failover and Response Caching",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.gateway = gateway

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        try:
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > settings.max_body_bytes:
                response: Any = JSONResponse(status_code=413, content={"detail": "request body too large"})
            elif request.method == "POST" and request.url.path == "/chat":
                client = request.client.host if request.client else "unknown"
                retry_after = limiter.check(client)
                if retry_after is not None:
                    response = JSONResponse(
                        status_code=429,
                        content={"detail": "rate limit exceeded, slow down"},
                        headers={"Retry-After": str(int(retry_after) + 1)},
                    )
                else:
                    response = await _guarded(call_next, request)
            else:
                response = await _guarded(call_next, request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_var.reset(token)

    # Added last => outermost, so even 413/429 responses from request_context carry CORS headers.
    origins = settings.allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX") or None,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Simulate-Failure", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    # ---- error handling: always JSON with a `detail` field, never a stack trace -------------
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = []
        for err in exc.errors():
            err = dict(err)
            err.pop("ctx", None)  # may hold exception objects that aren't JSON-serialisable
            if isinstance(err.get("input"), str) and len(err["input"]) > 200:
                err["input"] = err["input"][:200] + "..."
            errors.append(err)
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    # ---- routes ----------------------------------------------------------------------------
    @app.get("/health")
    async def health() -> JSONResponse:
        db_ok = await store.aping()
        keys_configured = len(gateway.keys)
        ok = db_ok and keys_configured > 0
        body: dict[str, Any] = {
            "status": "ok" if ok else "unavailable",
            "keys_configured": keys_configured,
            "cache_enabled": gateway.cache_enabled,
            "database": "ok" if db_ok else "unreachable",
            "models": settings.models,
            "failure_simulation_enabled": settings.enable_failure_simulation,
            "keys_cooling_down": gateway.cooling_down(),
        }
        return JSONResponse(status_code=200 if ok else 503, content=body)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        body: ChatRequest,
        x_simulate_failure: str | None = Header(default=None, alias="X-Simulate-Failure"),
    ) -> ChatResponse:
        started = time.perf_counter()

        simulate: dict[str, int] = {}
        if settings.enable_failure_simulation and x_simulate_failure:
            try:
                simulate = parse_simulation_header(x_simulate_failure, [k.key_id for k in gateway.keys])
            except InvalidSimulationHeader as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None

        async with locks.hold(body.session_id):
            session = await store.get_session(body.session_id)
            system_prompt = body.system_prompt or (session or {}).get("system_prompt") or settings.default_system_prompt
            history = await store.get_history(body.session_id, limit=settings.max_history_messages)

            messages = [{"role": "system", "content": system_prompt}]
            messages += history_for_context(history)
            messages.append({"role": "user", "content": body.message})

            try:
                result = await gateway.complete(messages, simulate=simulate)
            except AllProvidersUnavailable:
                logger.error("all providers unavailable session=%s", body.session_id)
                raise HTTPException(status_code=503, detail=ALL_PROVIDERS_UNAVAILABLE) from None

            metadata = ChatMetadata(
                resolution_source="LLM_CACHE" if result.cache_hit else "LIVE_INFERENCE",
                model=result.model,
                latency_ms=0,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                key_id=result.key_id,
                fallback_attempts=result.fallback_attempts,
            )
            stored_metadata = metadata.model_dump()
            stored_metadata["failed_attempts"] = [
                {"key_id": a.key_id, "status": a.status, "error": a.error, "simulated": a.simulated}
                for a in result.attempts
            ]

            # Only a successful turn is written, and user + assistant rows go in one transaction.
            latency_ms = int(round((time.perf_counter() - started) * 1000))
            metadata.latency_ms = latency_ms
            stored_metadata["latency_ms"] = latency_ms
            await store.append_exchange(
                body.session_id,
                body.message,
                result.reply,
                stored_metadata,
                body.system_prompt or (session or {}).get("system_prompt"),
            )

        logger.info(
            "chat session=%s source=%s key=%s fallbacks=%d latency_ms=%d in=%d out=%d",
            body.session_id,
            metadata.resolution_source,
            metadata.key_id,
            metadata.fallback_attempts,
            metadata.latency_ms,
            metadata.input_tokens,
            metadata.output_tokens,
        )
        return ChatResponse(session_id=body.session_id, reply=result.reply, metadata=metadata)

    # `:path` so client-generated ids containing "/" (sent URL-encoded) still resolve.
    @app.get("/sessions/{session_id:path}/messages")
    async def session_messages(session_id: str) -> dict[str, Any]:
        history = await store.get_history(session_id)
        if not history:
            raise HTTPException(status_code=404, detail="session not found")
        return {"session_id": session_id, "messages": history}

    return app


app = create_app()
