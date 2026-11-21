"""A real HTTP server that impersonates LLM providers, for testing LiteLLM's network path.

It speaks two wire formats:
  * Gemini   POST /models/{model}:generateContent   (Gemini 3 response shape, incl. thoughts)
  * OpenAI   POST /v1/chat/completions               (OpenAI-compatible: Groq, OpenRouter, ...)

Behaviour is chosen by the API key the client sends (Bearer token, x-goog-api-key or ?key=):
  ok*       success, echoing the last user message
  429       quota / rate limit           500 / 502 / 503   server errors
  401 / 403 auth errors                  400               bad request (e.g. invalid key on Gemini)
  slow      sleeps longer than the client timeout
  empty     200 with an empty completion
  garbage   200 with an HTML body
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

ERRORS = {
    "429": (429, "RESOURCE_EXHAUSTED", "Resource has been exhausted (e.g. check quota)."),
    "500": (500, "INTERNAL", "An internal error has occurred."),
    "502": (502, "UNAVAILABLE", "Bad gateway."),
    "503": (503, "UNAVAILABLE", "The model is overloaded. Please try again later."),
    "401": (401, "UNAUTHENTICATED", "Request had invalid authentication credentials."),
    "403": (403, "PERMISSION_DENIED", "Permission denied on resource project."),
    "400": (400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."),
}


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class FakeProvider:
    def __init__(self, slow_seconds: float = 3.0) -> None:
        self.slow_seconds = slow_seconds
        self.hits: Counter[str] = Counter()
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # silence
                pass

            def _key(self) -> str:
                auth = self.headers.get("authorization", "")
                if auth.lower().startswith("bearer "):
                    return auth[7:].strip()
                if self.headers.get("x-goog-api-key"):
                    return self.headers["x-goog-api-key"]
                return parse_qs(urlparse(self.path).query).get("key", [""])[0]

            def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("content-length", 0) or 0))
                payload = json.loads(raw or b"{}")
                key = self._key()
                path = urlparse(self.path).path
                fmt = "gemini" if ":generateContent" in path else "openai"
                with provider._lock:
                    provider.hits[key] += 1
                    provider.requests.append({"key": key, "format": fmt, "path": path, "body": payload})

                if key in ERRORS:
                    status, code, message = ERRORS[key]
                    body = {"error": {"code": status, "message": message, "status": code}}
                    if fmt == "openai":
                        body = {"error": {"message": message, "type": code.lower(), "code": str(status)}}
                    return self._send(status, json.dumps(body).encode())
                if key == "slow":
                    time.sleep(provider.slow_seconds)
                if key == "garbage":
                    return self._send(200, b"<html><body>502 Bad Gateway</body></html>", "text/html")

                texts = provider._texts(fmt, payload)
                last_user = texts["users"][-1] if texts["users"] else ""
                reply = "" if key == "empty" else f"echo: {last_user} (user turns seen: {len(texts['users'])})"
                prompt_tokens = _tokens(" ".join(texts["all"]))
                completion_tokens = _tokens(reply) if reply else 0
                if fmt == "gemini":
                    body = {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": reply, "thoughtSignature": "c2lnbmF0dXJl"}],
                                },
                                "finishReason": "STOP",
                                "index": 0,
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": prompt_tokens,
                            "candidatesTokenCount": completion_tokens,
                            "thoughtsTokenCount": 7,
                            "totalTokenCount": prompt_tokens + completion_tokens + 7,
                        },
                        "modelVersion": path.split("/models/")[-1].split(":")[0],
                        "responseId": "fake-resp",
                    }
                else:
                    body = {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": payload.get("model", "fake"),
                        "choices": [
                            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                self._send(200, json.dumps(body).encode())

        class QuietServer(ThreadingHTTPServer):
            def handle_error(self, request: Any, client_address: Any) -> None:
                # The "slow" key's client times out on purpose; writing to the closed socket
                # afterwards raises BrokenPipeError, which is expected and not worth a traceback.
                pass

        self.server = QuietServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @staticmethod
    def _texts(fmt: str, payload: dict[str, Any]) -> dict[str, list[str]]:
        users: list[str] = []
        everything: list[str] = []
        if fmt == "gemini":
            system = payload.get("system_instruction") or payload.get("systemInstruction") or {}
            everything += [p.get("text", "") for p in system.get("parts", [])]
            for content in payload.get("contents", []):
                text = " ".join(p.get("text", "") for p in content.get("parts", []))
                everything.append(text)
                if content.get("role") == "user":
                    users.append(text)
        else:
            for message in payload.get("messages", []):
                content = message.get("content")
                text = content if isinstance(content, str) else " ".join(c.get("text", "") for c in content or [])
                everything.append(text)
                if message.get("role") == "user":
                    users.append(text)
        return {"users": users, "all": everything}

    # ---- lifecycle ----
    def start(self) -> FakeProvider:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def reset(self) -> None:
        with self._lock:
            self.hits.clear()
            self.requests.clear()

    def base_url(self, fmt: str) -> str:
        return f"http://127.0.0.1:{self.port}" + ("/v1" if fmt == "openai" else "")
