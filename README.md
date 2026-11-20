# Relay — LLM Chatbot with Key Failover and Response Caching

A small full-stack AI chat app: React frontend, FastAPI backend, LiteLLM as the LLM gateway, SQLite for session history. Every reply shows where it came from (`LIVE_INFERENCE` or `LLM_CACHE`), which model answered, server latency and the tokens used by **that request only**. When an API key is rate-limited, out of quota or returning 5xx, the backend fails over to the next key without the user noticing.

```
browser ──► React (Vite, nginx) ──► FastAPI ──► LLMGateway ──► LiteLLM ──► provider (key_1 → key_2 → …)
                                       │             │
                                       ▼             └──► LiteLLM cache (disk, /data/llm_cache)
                                   SQLite (/data/app.db)
```

| Path | What it is |
|---|---|
| `backend/app/main.py` | FastAPI app, the three endpoints, JSON-only error handling |
| `backend/app/llm.py` | Key pool, ordered failover, cooldowns, `X-Simulate-Failure`, cache wiring |
| `backend/app/db.py` | SQLite schema (auto-created) and atomic turn persistence |
| `backend/app/mock_provider.py` | Offline `mock/*` LiteLLM provider for tests and keyless demos |
| `backend/tests/` | 112 pytest tests: contract, failover, cache, real-HTTP provider errors, hardening |
| `frontend/src/` | React UI (`App.tsx`), API client (`api.ts`), 12 Vitest tests |
| `scripts/verify_api.py` | Black-box contract checker you can point at any deployed backend |
| `scripts/e2e_ui.py` | Playwright browser test of the UI contract |
| `scripts/load_test.py` | Concurrency soak test with consistency checks |
| `.github/workflows/ci.yml` | CI: lint, tests, `make up`, contract + browser checks, restart persistence |

---

## 1. Live URLs

| | URL |
|---|---|
| Frontend | `https://<your-frontend>.vercel.app` |
| Backend | `https://<your-backend>.onrender.com` |
| Health | `https://<your-backend>.onrender.com/health` |
| Repository | `https://github.com/<you>/<repo>` |
| Demo video | `<link>` |

> The deployed backend runs with `ENABLE_FAILURE_SIMULATION=true` so failover can be tested on demand.

---

## 2. Run locally

### With Docker

```bash
cp .env.example .env          # then put at least 2 real keys in LLM_API_KEYS
make up                       # builds both images and starts the stack
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000/health
- `make down` stops it (the `app-data` volume keeps history and cache); `make clean` also deletes the volume.
- `make logs`, `make test` are also available.

No `make` on Windows? Run `docker compose up -d --build` and `docker compose down` directly (that's all the targets do).

**Which model and keys?** The default is `gemini/gemini-3.5-flash-lite`, a free-tier model. The assignment's example `gemini-2.0-flash` was shut down on 1 June 2026, and `gemini-2.5-*` is closed to new projects. **Gemini rate limits apply per Google Cloud project, not per API key.** Create each key in a *different* project, or mix providers with `LLM_KEY_MODELS` (e.g. Gemini + Groq). Otherwise one real 429 exhausts every key at once.

**No API keys handy?** Set `LLM_API_KEYS=mock-a,mock-b` and `LLM_MODEL=mock/demo`. The mock provider is registered inside LiteLLM, so caching, failover and token counting work exactly as with a real model.

### Without Docker

Backend (Python 3.11+):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export LLM_API_KEYS=key-one,key-two LLM_MODEL=gemini/gemini-3.5-flash-lite \
       DATABASE_URL=sqlite:///./data/app.db ENABLE_FAILURE_SIMULATION=true
uvicorn app.main:app --reload --port 8000
pytest                                                  # 112 tests, no keys or internet needed
```

Frontend (Node 20+):

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev     # http://localhost:3000
npm test
```

### Environment variables

All configuration is through environment variables; `.env.example` lists each one with a placeholder.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_API_KEYS` | — | Comma-separated key pool (≥ 2). Ids are `key_1`, `key_2`, … in order |
| `LLM_MODEL` | `gemini/gemini-3.5-flash-lite` | LiteLLM model used by every key |
| `LLM_API_BASE` | — | Optional custom endpoint for every key (LiteLLM proxy, OpenAI-compatible host) |
| `LLM_EXTRA_PARAMS` | — | Extra LiteLLM params as JSON, e.g. `{"temperature": 0.3}` (part of the cache key) |
| `LLM_KEY_MODELS` | — | Optional per-key models, aligned with `LLM_API_KEYS` (e.g. `gemini/gemini-3.5-flash-lite,groq/llama-3.1-8b-instant`) |
| `DATABASE_URL` | `sqlite:///./data/app.db` | SQLite location; `sqlite:////data/app.db` in Docker |
| `ALLOWED_ORIGINS` | `http://localhost:3000,…` | CORS allow-list |
| `ALLOWED_ORIGIN_REGEX` | — | Optional regex for extra origins (e.g. Vercel previews) |
| `ENABLE_FAILURE_SIMULATION` | `false` | Enables the `X-Simulate-Failure` hook |
| `CACHE_TYPE` | `disk` | `disk`, `local` (in-memory) or `none` |
| `CACHE_DIR` / `CACHE_TTL_SECONDS` | `./data/llm_cache` / `86400` | Disk cache location and TTL |
| `DEFAULT_SYSTEM_PROMPT` | `You are a helpful, concise assistant.` | Used when a session has no custom prompt |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-attempt provider timeout |
| `MAX_HISTORY_MESSAGES` | `40` | How many stored messages are sent as context |
| `MAX_OUTPUT_TOKENS` | `4096` | Completion cap (generous because Gemini 3 "thinking" tokens count towards it) |
| `RATE_LIMIT_PER_MINUTE` | `0` (off) | Per-IP limit on `POST /chat`, returns JSON 429 with `Retry-After` |
| `MAX_BODY_BYTES` | `65536` | Larger request bodies get a JSON 413 |
| `KEY_COOLDOWN_SECONDS` | `30` | How long a key that returned a real 429/401/403 is tried last |
| `ENABLE_DOCS` | `false` | Serve Swagger UI at `/docs` |
| `API_BASE_URL` (frontend container) | `http://localhost:8000` | Backend URL injected into `config.js` at container start |
| `VITE_API_BASE_URL` (frontend build) | — | Build-time fallback for static hosts (Vercel/Netlify) |

### Deploying

- **Backend → Render**: New → Blueprint → this repo (`render.yaml`), or a Docker web service with root `backend/`. Set `LLM_API_KEYS` and `ALLOWED_ORIGINS=https://<your-frontend>.vercel.app`. Render injects `PORT`; the image listens on it. On the free plan the filesystem resets on redeploy; attach a disk at `/data` (paid) for durable history. Railway or Fly.io with a volume at `/data` work the same way.
- **Frontend → Vercel**: import the repo, root directory `frontend`, env `VITE_API_BASE_URL=https://<your-backend>.onrender.com`. `vercel.json` handles SPA routing.
- Then run `python scripts/verify_api.py https://<your-backend>` to confirm the live backend meets the contract.
- Free Render instances sleep after ~15 idle minutes, and the first request then takes up to a minute. The UI shows "waking up backend…" and waits up to 2 minutes. To keep it warm during review, point a free uptime pinger (e.g. UptimeRobot, cron-job.org) at `/health` every 10 minutes.

---

## 3. Design decisions

### Key fallback

Each API key is its own LiteLLM deployment (`key_id → api_key + model`). `LLMGateway.complete()` walks the pool **in order** and calls `litellm.acompletion` once per key with `num_retries=0`, so LiteLLM never retries the same key behind our back.

- **What triggers failover:** any provider error. That includes 429 / quota (`RateLimitError`), 5xx (`InternalServerError`, `ServiceUnavailableError`, `APIError`), timeouts and connection errors. It also covers 401/403, so a revoked key doesn't take the app down, and an empty completion.
- **All keys failed:** the API returns `503 {"detail": "all providers unavailable"}` and **nothing is written** to history.
- **Cooldown (circuit breaker):** a key that returns a *real* 429/401/403 is moved to the back of the order for `KEY_COOLDOWN_SECONDS`. A dead key then costs one failed call, not one per request. A success clears the cooldown. Simulated failures never cool a key down, so the test hook stays per-request.
- **Hard attempt timeout:** each attempt is wrapped in `asyncio.wait_for(LLM_TIMEOUT_SECONDS + 5)` on top of LiteLLM's own timeout, so a hung key costs seconds, not the whole request. A 200 with no usable text (empty completion, no choices, junk body) is also treated as a failure for that key.
- **Verified against real HTTP:** `tests/test_real_http_providers.py` runs a local fake provider that speaks both the Gemini (`generateContent`, Gemini 3 response shape with thought tokens) and OpenAI wire formats. The tests cover real 429/500/502/503/401/403/400 responses, a stalled provider, an empty completion and an HTML body; each one fails over. They also assert that the failing key received **exactly one** request, so neither LiteLLM nor the OpenAI SDK retries a dead key behind our back.
- **Why an explicit loop rather than LiteLLM's Router fallbacks:** the contract needs the exact `key_id` that answered and an exact `fallback_attempts` count. It also needs deterministic per-request failure injection. The Router shuffles deployments and keeps cooldown state shared across requests, which makes both harder to report and to test. The loop is about 40 lines, fully observable (every attempt is logged as `key_id`/status, never the key), and still uses LiteLLM for every provider call, for exception mapping and for caching.
- **`X-Simulate-Failure`:** parsed into `{key_id: status}` (`*` expands to every key; unknown keys are ignored; malformed values return 422). A simulated key is called with LiteLLM's own `mock_response=<litellm.RateLimitError | InternalServerError | …>`. The simulated 429 is therefore the same exception type a real provider raises and goes through the same failover code. Simulated requests skip the cache *read* (a cache hit would hide failover; e.g. `key_1=429` must report `key_id=key_2`). Their fresh answers are still cached. The hook is ignored unless `ENABLE_FAILURE_SIMULATION=true`.

### Caching

- LiteLLM's built-in cache (`litellm.cache = Cache(type="disk")`), stored in the Docker volume next to the database, so it survives restarts. `CACHE_TYPE=local` switches to in-memory.
- **Cache key:** LiteLLM hashes `model + messages + call params`, where `messages` is the full list sent to the LLM (system prompt + history + new message). It deliberately excludes `api_key`, so an answer produced by `key_2` is reused even when `key_1` is healthy again. A different message, a different history or a different system prompt therefore never matches.
- **Hit detection:** LiteLLM marks hits in `response._hidden_params["cache_hit"]`. On a hit we return the identical reply, the token usage stored with it, `key_id: null` and `fallback_attempts: 0`. Cache hits measure ~2–10 ms against hundreds of ms for live calls.
- **Deterministic writes:** LiteLLM writes to the cache in a background task, which can land *after* an immediate follow-up request. After a live success the gateway also awaits `litellm.cache.async_add_cache(...)` with the same key. "Second identical request is a cache hit" then holds even for back-to-back scripted calls.
- Pinned `litellm==1.77.7`. The newest releases dropped the disk cache type and crash on the cache-hit path without proxy-only extras installed.

### SQLite session storage

- Stdlib `sqlite3`, WAL mode, schema created on startup (`CREATE TABLE IF NOT EXISTS`). Each query runs in a worker thread so the event loop never blocks.
- The required `messages` table (plus an index on `(session_id, id)`) and a small `sessions` table that stores each session's system prompt.
- **A turn is written only after the LLM succeeds,** and the user row and assistant row go in one transaction. A 422/503 therefore never leaves half a turn behind. The assistant row's `metadata` column holds the response metadata plus the list of failed attempts for that turn.
- **Per-session lock:** concurrent requests for the same `session_id` are serialised, so the history each one reads and writes stays consistent. Different sessions still run in parallel.
- `GET /sessions/{id}/messages` returns history oldest-first without the system prompt; `404` if the session has no stored turns.
- `DATABASE_URL=sqlite:////data/app.db` lives on the named compose volume `app-data`, so `make down && make up` keeps every conversation.

### API and error handling

- Exactly `GET /health`, `POST /chat` and `GET /sessions/{session_id}/messages` (Swagger is opt-in via `ENABLE_DOCS`). The history route accepts ids containing `/` when URL-encoded.
- Every response carries an `X-Request-ID` (echoed if the client sends one), and every log line includes it, so a failover in the logs can be traced to the exact request.
- CORS is the outermost middleware, so even 413/429/500 responses carry CORS headers and the browser can show the real message instead of a CORS error.
- Every error is JSON with `detail`, including 404/405 and a catch-all 500 handler; stack traces are only logged. Validation runs on the trimmed message, so 4000 characters plus surrounding whitespace is accepted and stored trimmed.
- Gemini 3 models "think": their thought tokens are billed as output and LiteLLM reports them in `completion_tokens`, so `output_tokens` includes them.
- `latency_ms` is measured server-side with `perf_counter` from request start to just before the response.
- `/health` returns 200 only when the database answers a query and at least one key is loaded, otherwise 503 with the same JSON shape.
- API keys never appear in responses or logs, and only `key_N` ids are exposed. A test asserts this.

---

## 4. Verification

**Quickest:** `python scripts/verify_api.py <backend-url>` runs about 35 contract checks (health, live/cache, tokens, multi-turn, history, 422s, failover, 503) and prints PASS/FAIL.

**Cache hit**

```bash
B=http://localhost:8000
curl -s -XPOST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"a1","message":"What is the capital of France?"}'
curl -s -XPOST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"b1","message":"What is the capital of France?"}'
# 1st: "resolution_source":"LIVE_INFERENCE","key_id":"key_1"   2nd: "LLM_CACHE", same reply, "key_id":null, much lower latency_ms
```

In the UI: send a message, reload the page (new session) and send the same message. The chip turns amber `LLM_CACHE`.

**Failover**

```bash
curl -s -XPOST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=429' \
     -d '{"session_id":"f1","message":"Say hi"}'                  # 200, key_id=key_2, fallback_attempts=1
curl -s -XPOST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=500' \
     -d '{"session_id":"f2","message":"Say hello"}'               # 200, key_id=key_2
curl -s -i -XPOST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: *=429' \
     -d '{"session_id":"f3","message":"Say hey"}'                 # 503 {"detail":"all providers unavailable"}
curl -s -i $B/sessions/f3/messages                                # 404, nothing was stored
```

In the UI: **Settings → Failover test hook** picks a preset that is sent as `X-Simulate-Failure`. The reply shows `key_2` and a red `failover ×1` chip; `every key → 429` shows a readable error with a Retry button. Backend logs show each attempt: `llm attempt failed key_id=key_1 status=429 error=RateLimitError simulated=True`.

**Per-message token usage**

Each assistant bubble shows `in <input_tokens> / out <output_tokens>` for that request. Two equal-length first messages in new sessions report about the same `input_tokens`. In one session, `input_tokens` grows with the history (it counts system + history + new message) while `output_tokens` is only that reply. `GET /sessions/{id}/messages` returns the same numbers per assistant message.

**Multi-turn**

```bash
curl -s -XPOST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"m1","message":"My name is Sudarshan."}'
curl -s -XPOST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"m1","message":"What is my name?"}'
curl -s $B/sessions/m1/messages
```

**Automated tests**

- `cd backend && pytest`: 112 tests, 96% line coverage, no keys or internet needed.
  - `test_api_contract.py`: response shape, every 422 case, cache hit/miss (history- and system-prompt-sensitive), restart persistence, tokens, multi-turn, history and 404s.
  - `test_failover.py`: header parsing; simulated 429/500/502/503; `*=429`; three-key chains; real 429/5xx with cooldown; nothing written on failure; concurrent turns in one session.
  - `test_real_http_providers.py`: the Gemini and OpenAI wire formats over real HTTP (see Design decisions).
  - `test_hardening.py`: request ids, 413, rate limit with CORS, ids with `/` or unicode, injection-like content, context window, clean 500.
- `cd frontend && npm test`: 12 Vitest tests. They cover the data-testids, input clearing, Enter to send, one session per page load, new chat, 422/429/503/network errors, retry, markdown rendering (and that raw HTML is never rendered), the cold-start state, and the system prompt and simulation header.
- `python scripts/e2e_ui.py http://localhost:3000`: a real-browser Playwright run through live → multi-turn → cache hit → failover → 503 → system prompt.
- `python scripts/load_test.py http://localhost:8000 300 20`: 300 mixed requests (new, repeated, same-session, failover, all-keys-down) at concurrency 20. It then checks that no request errored and that every successful turn was stored exactly once, in order, while failed turns were not stored. Local Docker run: 0 errors, live p50 ≈ 380 ms, cache p50 ≈ 6 ms.
- **CI** (`.github/workflows/ci.yml`) runs on every push: ruff, both test suites, then `make up` on the real images, `verify_api.py`, the Playwright run, and a `make down && make up` history-persistence check.

---

## 5. Setup Used

- **Editor:** VS Code (Git Bash + Docker Desktop on Windows 11)
- **AI tools:** Claude (Anthropic) for pair-programming the implementation, tests and documentation; all code reviewed and run locally.
- **Runtime:** Python 3.12, FastAPI 0.115, LiteLLM 1.77.7, React 18 + Vite 6 + TypeScript, nginx 1.27, Docker Compose.

---

## 6. Bonus features

- **Disk-backed LLM cache on the data volume**, so cache hits survive restarts and redeploys with a persistent disk, just like history.
- **Key cooldown / circuit breaker:** a key that returns a real 429/401/403 is tried last for a while, so a dead key doesn't add latency to every request.
- **Failover on auth errors and timeouts too,** not just 429/5xx. A revoked key or a hung provider is handled the same way.
- **Mixed-model pools** (`LLM_KEY_MODELS`) to spread the pool across providers, e.g. Gemini + Groq.
- **Offline `mock/*` provider** inside LiteLLM: the full test suite and a keyless local demo exercise the real caching and failover code paths.
- **Atomic turns and per-session locking,** so concurrent requests can't interleave or half-write a conversation.
- **UI:** failover simulator dropdown, `key_id` and `failover ×N` chips, live health badge, retry button on errors, character counter, "New chat" button.
- **Runtime frontend config:** one Docker image works against any backend (`API_BASE_URL` is written into `config.js` at container start).
- **Hardened images:** multi-stage builds, backend runs as a non-root user, Docker `HEALTHCHECK`s, and the frontend waits for the backend to be healthy.
- **`scripts/verify_api.py`, `scripts/e2e_ui.py` and `scripts/load_test.py`** to check a live deployment end to end in seconds.
- **Markdown replies** (lists, code, tables) rendered safely: raw HTML from the model is never injected into the page.
- **Cold-start aware UI:** shows "waking up backend…" while a sleeping free-tier backend boots, with a 2-minute request timeout instead of hanging forever.
- **Request ids, 413 body limit, optional per-IP rate limit** to protect the key pool's quota on a public URL.
- **CI pipeline** that builds the real Docker images and runs the full end-to-end checks on every push.
