"""Concurrency / soak test: mixed traffic against a running backend, then consistency checks.

Usage: python scripts/load_test.py http://localhost:8000 [requests=300] [concurrency=20]
Stdlib only. Use against a backend running the mock model (or be ready to spend real quota).
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 300
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 20
RUN = uuid.uuid4().hex[:6]
SHARED_SESSIONS = [f"load-{RUN}-shared-{i}" for i in range(5)]  # hammered concurrently


def post(session: str, message: str, header: str | None = None) -> tuple[int, dict, float]:
    req = urllib.request.Request(
        BASE + "/chat",
        data=json.dumps({"session_id": session, "message": message}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", **({"X-Simulate-Failure": header} if header else {})},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            status, body = res.status, json.loads(res.read())
    except urllib.error.HTTPError as err:
        status, body = err.code, json.loads(err.read() or b"{}")
    return status, body, (time.perf_counter() - started) * 1000


def job(i: int) -> tuple[str, int, dict, float, str]:
    kind = random.choices(["new", "repeat", "shared", "failover", "dead"], weights=[30, 25, 25, 15, 5])[0]
    if kind == "new":
        session, message, header = f"load-{RUN}-{i}", f"unique question {RUN}-{i}", None
    elif kind == "repeat":
        session, message, header = f"load-{RUN}-{i}", f"popular question {random.randint(1, 5)} {RUN}", None
    elif kind == "shared":
        session, message, header = random.choice(SHARED_SESSIONS), f"turn {i}", None
    elif kind == "failover":
        session, message, header = f"load-{RUN}-{i}", f"failover {i}", "key_1=429"
    else:
        session, message, header = f"load-{RUN}-dead-{i}", f"dead {i}", "*=503"
    status, body, ms = post(session, message, header)
    return kind, status, body, ms, session


def main() -> None:
    print(f"{TOTAL} requests, concurrency {CONCURRENCY}, target {BASE}")
    started = time.perf_counter()
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        results = list(pool.map(job, range(TOTAL)))
    wall = time.perf_counter() - started

    problems: list[str] = []
    by_kind: dict[str, Counter] = {}
    latencies: dict[str, list[float]] = {"LIVE_INFERENCE": [], "LLM_CACHE": []}
    shared_ok: Counter = Counter()
    for kind, status, body, ms, session in results:
        by_kind.setdefault(kind, Counter())[status] += 1
        expected = 503 if kind == "dead" else 200
        if status != expected:
            problems.append(f"{kind}: HTTP {status} {body}")
            continue
        if status == 200:
            meta = body["metadata"]
            latencies[meta["resolution_source"]].append(meta["latency_ms"])
            if kind == "failover" and meta["key_id"] != "key_2":
                problems.append(f"failover answered by {meta['key_id']}")
            if kind == "shared":
                shared_ok[session] += 1

    # Consistency: every successful turn in a shared session is stored exactly once, in pairs.
    for session in SHARED_SESSIONS:
        if not shared_ok[session]:
            continue
        with urllib.request.urlopen(f"{BASE}/sessions/{session}/messages", timeout=30) as res:
            messages = json.loads(res.read())["messages"]
        roles = [m["role"] for m in messages]
        if len(messages) != 2 * shared_ok[session] or roles != ["user", "assistant"] * shared_ok[session]:
            problems.append(f"{session}: {len(messages)} stored for {shared_ok[session]} turns")
    for kind, status, _body, _ms, session in results:
        if kind == "dead":
            try:
                urllib.request.urlopen(f"{BASE}/sessions/{session}/messages", timeout=30)
                problems.append(f"{session}: failed turn was stored")
            except urllib.error.HTTPError as err:
                if err.code != 404:
                    problems.append(f"{session}: history returned {err.code}")

    print(f"wall time {wall:.1f}s  throughput {TOTAL / wall:.1f} req/s")
    for kind, counts in sorted(by_kind.items()):
        print(f"  {kind:9s} {dict(counts)}")
    for source, values in latencies.items():
        if values:
            values.sort()
            p95 = values[min(len(values) - 1, int(len(values) * 0.95))]
            print(f"  {source:15s} n={len(values):3d}  p50={statistics.median(values):.0f}ms  p95={p95:.0f}ms")
    print("LOAD TEST PASSED" if not problems else f"{len(problems)} PROBLEMS:\n  " + "\n  ".join(problems[:20]))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
