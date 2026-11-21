import { FormEvent, KeyboardEvent, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ApiError, ChatMetadata, HealthResponse, getHealth, newSessionId, sendChat } from "./api";

const MAX_CHARS = 4000;

type Entry =
  | { id: number; role: "user"; content: string }
  | { id: number; role: "assistant"; content: string; metadata: ChatMetadata }
  | { id: number; role: "error"; content: string; retryText: string; userEntryId: number };

const SIMULATION_PRESETS: { label: string; value: string }[] = [
  { label: "No simulated failure", value: "" },
  { label: "key_1 → 429 (rate limit)", value: "key_1=429" },
  { label: "key_1 → 500 (server error)", value: "key_1=500" },
  { label: "key_1 → 503, key_2 → 502", value: "key_1=503,key_2=502" },
  { label: "every key → 429", value: "*=429" },
];

let nextId = 1;

function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
      }}
    >
      {text}
    </ReactMarkdown>
  );
}

type HealthState = "connecting" | "waking" | "ok" | "down";

function MetaBar({ meta }: { meta: ChatMetadata }) {
  const cached = meta.resolution_source === "LLM_CACHE";
  return (
    <div className="meta" aria-label="Response metadata">
      <span className={`chip chip-source ${cached ? "is-cache" : "is-live"}`} title="Resolution source">
        <span data-testid="meta-source">{meta.resolution_source}</span>
      </span>
      <span className="chip" title="Model">
        <span data-testid="meta-model">{meta.model}</span>
      </span>
      <span className="chip" title="Server-measured latency">
        <span data-testid="meta-latency">{meta.latency_ms} ms</span>
      </span>
      <span className="chip" title="Tokens for this request only">
        <span className="chip-label">in</span>
        <span data-testid="meta-input-tokens">{meta.input_tokens}</span>
        <span className="chip-sep">/</span>
        <span className="chip-label">out</span>
        <span data-testid="meta-output-tokens">{meta.output_tokens}</span>
      </span>
      {meta.key_id && (
        <span className="chip chip-dim" title="Key that answered">
          <span data-testid="meta-key-id">{meta.key_id}</span>
        </span>
      )}
      {meta.fallback_attempts > 0 && (
        <span className="chip chip-warn" title="Keys that failed before this one answered">
          <span data-testid="meta-fallbacks">
            failover ×{meta.fallback_attempts}
          </span>
        </span>
      )}
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState<string>(() => newSessionId());
  const [entries, setEntries] = useState<Entry[]>([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [simulate, setSimulate] = useState("");
  const [pending, setPending] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthState, setHealthState] = useState<HealthState>("connecting");
  // Only disable when the backend has told us simulation is off; if health is unknown, the header
  // is harmless (the server ignores it unless ENABLE_FAILURE_SIMULATION=true).
  const simulationDisabled = health !== null && health.failure_simulation_enabled === false;
  const activeSimulation = simulationDisabled ? "" : simulate;
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    // Poll /health until the backend answers. Free-tier hosts sleep when idle and can take up
    // to a minute to wake, so show that instead of a dead-looking UI.
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const started = Date.now();
    const slowHint = setTimeout(() => !cancelled && setHealthState((s) => (s === "connecting" ? "waking" : s)), 3000);
    const poll = async () => {
      try {
        const h = await getHealth();
        if (cancelled) return;
        setHealth(h);
        setHealthState(h.status === "ok" ? "ok" : "down");
      } catch {
        if (cancelled) return;
        if (Date.now() - started < 120_000) {
          setHealthState("waking");
          timer = setTimeout(poll, 4000);
        } else {
          setHealthState("down");
        }
      }
    };
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(slowHint);
      if (timer) clearTimeout(timer);
    };
  }, []);

  // Auto-grow the composer (works in every browser, not only those with CSS field-sizing).
  useLayoutEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [input]);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [entries, pending]);

  const send = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message || pending) return;

      const userEntry: Entry = { id: nextId++, role: "user", content: message };
      setEntries((prev) => [...prev, userEntry]);
      setInput("");
      setPending(true);
      try {
        const res = await sendChat(
          {
            session_id: sessionId,
            message,
            ...(systemPrompt.trim() ? { system_prompt: systemPrompt.trim() } : {}),
          },
          activeSimulation || undefined,
        );
        setEntries((prev) => [
          ...prev,
          { id: nextId++, role: "assistant", content: res.reply, metadata: res.metadata },
        ]);
      } catch (err) {
        const content =
          err instanceof ApiError ? err.message : "Something unexpected went wrong. Please try again.";
        setEntries((prev) => [
          ...prev,
          { id: nextId++, role: "error", content, retryText: message, userEntryId: userEntry.id },
        ]);
      } finally {
        setPending(false);
        inputRef.current?.focus();
      }
    },
    [pending, sessionId, activeSimulation, systemPrompt],
  );

  const retry = (entry: Extract<Entry, { role: "error" }>) => {
    // The failed turn was never stored server-side, so drop it locally and resend.
    setEntries((prev) => prev.filter((e) => e.id !== entry.id && e.id !== entry.userEntryId));
    void send(entry.retryText);
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void send(input);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send(input);
    }
  };

  const newChat = () => {
    setSessionId(newSessionId());
    setEntries([]);
    setInput("");
    inputRef.current?.focus();
  };

  const charCount = input.trim().length;
  const overLimit = charCount > MAX_CHARS;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            ⟢
          </span>
          <div>
            <h1>Relay</h1>
            <p className="tagline">LLM chat with key failover &amp; response caching</p>
          </div>
        </div>
        <div className="topbar-right">
          <span
            className={`status ${healthState === "ok" ? "ok" : healthState === "down" ? "down" : "unknown"}`}
            data-testid="health-status"
            title={health?.models?.join(", ")}
          >
            {healthState === "ok" && health
              ? `${health.keys_configured} keys · cache ${health.cache_enabled ? "on" : "off"}`
              : healthState === "waking"
                ? "waking up backend…"
                : healthState === "down"
                  ? health
                    ? "backend degraded"
                    : "backend unreachable"
                  : "connecting…"}
          </span>
          <button type="button" className="ghost" onClick={() => setShowSettings((s) => !s)} aria-expanded={showSettings}>
            Settings
          </button>
          <button type="button" className="ghost" onClick={newChat} data-testid="new-chat">
            New chat
          </button>
        </div>
      </header>

      {showSettings && (
        <section className="settings" aria-label="Session settings">
          <label className="field">
            <span>System prompt (this session)</span>
            <textarea
              data-testid="system-prompt-input"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="e.g. You are a concise assistant who answers in one sentence."
              rows={2}
            />
          </label>
          <label className="field">
            <span>
              Failover test hook <code>X-Simulate-Failure</code>
            </span>
            <select
              data-testid="simulate-failure"
              value={simulate}
              onChange={(e) => setSimulate(e.target.value)}
              disabled={simulationDisabled}
            >
              {SIMULATION_PRESETS.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
            </select>
            {simulationDisabled && (
              <small className="field-note" data-testid="simulation-disabled-note">
                Disabled on this backend. Set <code>ENABLE_FAILURE_SIMULATION=true</code> on the server and redeploy.
              </small>
            )}
          </label>
          <p className="session-id">
            session <code data-testid="session-id">{sessionId}</code>
          </p>
        </section>
      )}

      <main className="conversation" ref={listRef} aria-live="polite">
        {entries.length === 0 && (
          <div className="empty">
            <h2>Ask anything.</h2>
            <p>
              Each reply shows where it came from — a fresh <b>LIVE_INFERENCE</b> or an <b>LLM_CACHE</b> hit —
              plus model, latency and the tokens used by that single request.
            </p>
            <div className="suggestions">
              {["What is the capital of France?", "Explain caching in one paragraph.", "Give me 3 names for a cat."].map(
                (s) => (
                  <button key={s} type="button" onClick={() => void send(s)} disabled={pending}>
                    {s}
                  </button>
                ),
              )}
            </div>
          </div>
        )}

        {entries.map((entry) => {
          if (entry.role === "user") {
            return (
              <div key={entry.id} className="row row-user">
                <div className="bubble bubble-user" data-testid="user-message">
                  {entry.content}
                </div>
              </div>
            );
          }
          if (entry.role === "assistant") {
            return (
              <div key={entry.id} className="row row-assistant">
                <article className="bubble bubble-assistant" data-testid="assistant-message">
                  <div className="reply-text markdown">
                    <Markdown text={entry.content} />
                  </div>
                  <MetaBar meta={entry.metadata} />
                </article>
              </div>
            );
          }
          return (
            <div key={entry.id} className="row row-assistant">
              <div className="bubble bubble-error" role="alert" data-testid="error-message">
                <span>{entry.content}</span>
                <button type="button" className="link" onClick={() => retry(entry)} disabled={pending}>
                  Retry
                </button>
              </div>
            </div>
          );
        })}

        {pending && healthState === "waking" && (
          <p className="hint" data-testid="cold-start-hint">
            The backend is waking up from sleep (free hosting tier); the first reply can take up to a minute.
          </p>
        )}
        {pending && (
          <div className="row row-assistant">
            <div className="bubble bubble-assistant typing" data-testid="typing-indicator" aria-label="Assistant is typing">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
      </main>

      <form className="composer" onSubmit={onSubmit}>
        {activeSimulation && (
          <div className="sim-banner">
            Next message will send <code>X-Simulate-Failure: {activeSimulation}</code>
          </div>
        )}
        <div className="composer-box">
          <textarea
            ref={inputRef}
            data-testid="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Message Relay…  (Enter to send, Shift+Enter for a new line)"
            rows={1}
            aria-label="Message"
            autoFocus
          />
          <button type="submit" data-testid="send-button" disabled={pending || charCount === 0}>
            {pending ? "Sending…" : "Send"}
          </button>
        </div>
        <div className={`counter ${overLimit ? "over" : ""}`}>
          {charCount} / {MAX_CHARS}
        </div>
      </form>
    </div>
  );
}
