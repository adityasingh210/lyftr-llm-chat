"""Black-box check of a running backend against the assignment's API contract.

Usage:  python scripts/verify_api.py https://your-backend.example.com
Stdlib only. Needs ENABLE_FAILURE_SIMULATION=true on the target for the failover checks.
Each run uses fresh session ids and a random message suffix, so it is safe to repeat.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
RUN = uuid.uuid4().hex[:6]
failures: list[str] = []


def call(method: str, path: str, body=None, headers=None, raw: bytes | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=90) as res:
            status, text = res.status, res.read().decode()
    except urllib.error.HTTPError as err:
        status, text = err.code, err.read().decode()
    elapsed = int((time.perf_counter() - started) * 1000)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {"_raw": text[:200]}
    return status, payload, elapsed


def check(name: str, condition: bool, info: object = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -> {info}" if not condition and info else ""))
    if not condition:
        failures.append(name)


def sid(tag: str) -> str:
    return f"verify-{RUN}-{tag}"


print(f"Target: {BASE}\n\n/health")
status, body, _ = call("GET", "/health")
check("200", status == 200, body)
check("keys_configured >= 2", isinstance(body.get("keys_configured"), int) and body["keys_configured"] >= 2, body)
check("cache_enabled true", body.get("cache_enabled") is True, body)

print("\nPOST /chat (live)")
q = f"In one short sentence, what is the capital of France? [{RUN}]"
status, first, _ = call("POST", "/chat", {"session_id": sid("a"), "message": q})
meta = first.get("metadata", {})
check("200", status == 200, first)
check("session_id echoed", first.get("session_id") == sid("a"))
check("reply non-empty", bool(first.get("reply")))
check("LIVE_INFERENCE", meta.get("resolution_source") == "LIVE_INFERENCE", meta)
for field, typ in [("model", str), ("latency_ms", int), ("input_tokens", int), ("output_tokens", int), ("fallback_attempts", int)]:
    check(f"metadata.{field} is {typ.__name__}", isinstance(meta.get(field), typ), meta.get(field))
check("key_id like key_N", isinstance(meta.get("key_id"), str) and meta["key_id"].startswith("key_"), meta.get("key_id"))

print("\nPOST /chat (cache hit in a new session)")
status, second, _ = call("POST", "/chat", {"session_id": sid("b"), "message": q})
m2 = second.get("metadata", {})
check("LLM_CACHE", m2.get("resolution_source") == "LLM_CACHE", m2)
check("identical reply", second.get("reply") == first.get("reply"))
check("key_id null", m2.get("key_id") is None)
check("fallback_attempts 0", m2.get("fallback_attempts") == 0)
check("same tokens as original", (m2.get("input_tokens"), m2.get("output_tokens")) == (meta.get("input_tokens"), meta.get("output_tokens")))
check("faster than live", m2.get("latency_ms", 1e9) < meta.get("latency_ms", 0), (m2.get("latency_ms"), meta.get("latency_ms")))

print("\nPOST /chat (different message is live)")
status, other, _ = call("POST", "/chat", {"session_id": sid("c"), "message": f"Name one primary color. [{RUN}]"})
check("LIVE_INFERENCE", other.get("metadata", {}).get("resolution_source") == "LIVE_INFERENCE", other)

print("\nMulti-turn + history")
call("POST", "/chat", {"session_id": sid("m"), "message": f"My favourite number is 47. Just say OK. [{RUN}]"})
status, turn2, _ = call("POST", "/chat", {"session_id": sid("m"), "message": "What is my favourite number? Reply with the number only."})
check("remembers earlier turn", "47" in turn2.get("reply", ""), turn2.get("reply"))
status, hist, _ = call("GET", f"/sessions/{sid('m')}/messages")
roles = [m.get("role") for m in hist.get("messages", [])]
check("history 200", status == 200, hist)
check("history oldest-first user/assistant x2", roles == ["user", "assistant", "user", "assistant"], roles)
check("assistant metadata present", "metadata" in (hist.get("messages") or [{}, {}])[1])
status, body, _ = call("GET", f"/sessions/{sid('nope')}/messages")
check("unknown session 404", status == 404 and body.get("detail") == "session not found", (status, body))

print("\nValidation (422)")
for label, payload in [
    ("missing message", {"session_id": sid("v")}),
    ("blank message", {"session_id": sid("v"), "message": "   "}),
    ("message > 4000", {"session_id": sid("v"), "message": "a" * 4001}),
    ("empty session_id", {"session_id": "", "message": "hi"}),
    ("session_id > 128", {"session_id": "x" * 129, "message": "hi"}),
]:
    status, body, _ = call("POST", "/chat", payload)
    check(label, status == 422 and "detail" in body, (status, body))
status, body, _ = call("POST", "/chat", raw=b"{oops")
check("malformed JSON", status == 422 and "detail" in body, (status, body))

print("\nFailover (X-Simulate-Failure)")
for header in ["key_1=429", "key_1=500"]:
    status, body, _ = call("POST", "/chat", {"session_id": sid(header[-3:]), "message": f"Say hi. ({header}) [{RUN}]"}, {"X-Simulate-Failure": header})
    m = body.get("metadata", {})
    check(f"{header} -> 200 key_2", status == 200 and m.get("key_id") == "key_2" and m.get("fallback_attempts") == 1, (status, m))
status, body, _ = call("POST", "/chat", {"session_id": sid("all"), "message": f"Doomed [{RUN}]"}, {"X-Simulate-Failure": "*=429"})
check("*=429 -> 503", status == 503 and body == {"detail": "all providers unavailable"}, (status, body))
status, _, _ = call("GET", f"/sessions/{sid('all')}/messages")
check("failed request not stored", status == 404, status)

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
