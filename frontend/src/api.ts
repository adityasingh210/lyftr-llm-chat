export type ResolutionSource = "LIVE_INFERENCE" | "LLM_CACHE";

export interface ChatMetadata {
  resolution_source: ResolutionSource;
  model: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  key_id: string | null;
  fallback_attempts: number;
}

export interface ChatResponse {
  session_id: string;
  reply: string;
  metadata: ChatMetadata;
}

export interface HealthResponse {
  status: string;
  keys_configured: number;
  cache_enabled: boolean;
  models?: string[];
  failure_simulation_enabled?: boolean;
}

declare global {
  interface Window {
    __APP_CONFIG__?: { API_BASE_URL?: string };
  }
}

export function apiBaseUrl(): string {
  const runtime = typeof window !== "undefined" ? window.__APP_CONFIG__?.API_BASE_URL : undefined;
  const base = runtime || import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
  return base.replace(/\/+$/, "");
}

export function newSessionId(): string {
  const random =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().replace(/-/g, "").slice(0, 12)
      : Math.random().toString(36).slice(2, 14);
  return `s-${random}`;
}

/** An error the UI can show as-is. */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number | null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function describeDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => {
        if (d && typeof d === "object" && "msg" in d) {
          const loc = Array.isArray((d as { loc?: unknown[] }).loc)
            ? (d as { loc: unknown[] }).loc.filter((p) => p !== "body").join(".")
            : "";
          const msg = String((d as { msg: unknown }).msg).replace(/^Value error, /, "");
          return loc ? `${loc}: ${msg}` : msg;
        }
        return null;
      })
      .filter(Boolean);
    return parts.length ? parts.join("; ") : null;
  }
  return null;
}

export function friendlyError(status: number, detail: unknown): string {
  const text = describeDetail(detail);
  if (status === 503) {
    return "All AI providers are unavailable right now (every API key failed). Please try again in a moment.";
  }
  if (status === 422) return `Your message couldn't be sent: ${text ?? "invalid request"}.`;
  if (status === 429) return "You're sending messages too quickly. Wait a few seconds and try again.";
  if (status === 413) return "That message is too large to send.";
  if (status === 404) return text ?? "Not found.";
  return text ? `Something went wrong (${status}): ${text}` : `Something went wrong (HTTP ${status}).`;
}

/** Generous: free-tier hosts can take ~a minute to wake up on the first request. */
export const REQUEST_TIMEOUT_MS = 120_000;

async function request<T>(path: string, init?: RequestInit, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${apiBaseUrl()}${path}`, { ...init, signal: controller.signal });
  } catch (err) {
    if (controller.signal.aborted) {
      throw new ApiError("The server took too long to respond. Please try again.", null);
    }
    throw new ApiError("Can't reach the server. Check your connection and try again.", null);
  } finally {
    clearTimeout(timer);
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    const detail = body && typeof body === "object" ? (body as { detail?: unknown }).detail : undefined;
    throw new ApiError(friendlyError(res.status, detail), res.status);
  }
  return body as T;
}

export function sendChat(
  payload: { session_id: string; message: string; system_prompt?: string },
  simulateFailure?: string,
): Promise<ChatResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (simulateFailure) headers["X-Simulate-Failure"] = simulateFailure;
  return request<ChatResponse>("/chat", { method: "POST", headers, body: JSON.stringify(payload) });
}

export function getHealth(timeoutMs = 15_000): Promise<HealthResponse> {
  return request<HealthResponse>("/health", undefined, timeoutMs);
}
