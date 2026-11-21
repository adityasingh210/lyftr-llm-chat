import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { friendlyError } from "./api";

type Handler = (url: string, init?: RequestInit) => { status: number; body: unknown };

function mockFetch(handler: Handler) {
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    const { status, body } = handler(url, init);
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

const health = { status: "ok", keys_configured: 2, cache_enabled: true, failure_simulation_enabled: true };

const okReply = (source = "LIVE_INFERENCE") => ({
  session_id: "x",
  reply: "The capital of France is Paris.",
  metadata: {
    resolution_source: source,
    model: "gemini/gemini-3.5-flash-lite",
    latency_ms: 812,
    input_tokens: 42,
    output_tokens: 9,
    key_id: source === "LLM_CACHE" ? null : "key_2",
    fallback_attempts: source === "LLM_CACHE" ? 0 : 1,
  },
});

afterEach(() => vi.unstubAllGlobals());

test("sends a message, clears the input and renders reply metadata with test ids", async () => {
  const fetchMock = mockFetch((url) => (url.endsWith("/health") ? { status: 200, body: health } : { status: 200, body: okReply() }));
  render(<App />);

  const input = screen.getByTestId("chat-input");
  await userEvent.type(input, "What is the capital of France?");
  await userEvent.click(screen.getByTestId("send-button"));

  expect(input).toHaveValue("");
  const reply = await screen.findByTestId("assistant-message");
  expect(reply).toHaveTextContent("The capital of France is Paris.");
  expect(within(reply).getByTestId("meta-source")).toHaveTextContent(/^LIVE_INFERENCE$/);
  expect(within(reply).getByTestId("meta-model")).toHaveTextContent("gemini/gemini-3.5-flash-lite");
  expect(within(reply).getByTestId("meta-latency")).toHaveTextContent("812 ms");
  expect(within(reply).getByTestId("meta-input-tokens")).toHaveTextContent(/^42$/);
  expect(within(reply).getByTestId("meta-output-tokens")).toHaveTextContent(/^9$/);

  const chatCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/chat"))!;
  const body = JSON.parse(String(chatCall[1]!.body));
  expect(body.message).toBe("What is the capital of France?");
  expect(body.session_id).toMatch(/^s-/);
  expect(body.system_prompt).toBeUndefined();
});

test("Enter sends; the same session id is reused across turns", async () => {
  const fetchMock = mockFetch((url) => (url.endsWith("/health") ? { status: 200, body: health } : { status: 200, body: okReply("LLM_CACHE") }));
  render(<App />);
  const input = screen.getByTestId("chat-input");

  await userEvent.type(input, "one{Enter}");
  await screen.findAllByTestId("assistant-message");
  await userEvent.type(input, "two{Enter}");
  await waitFor(() => expect(screen.getAllByTestId("assistant-message")).toHaveLength(2));

  const ids = fetchMock.mock.calls
    .filter(([url]) => String(url).endsWith("/chat"))
    .map(([, init]) => JSON.parse(String(init!.body)).session_id);
  expect(new Set(ids).size).toBe(1);
  expect(screen.getAllByTestId("meta-source")[0]).toHaveTextContent("LLM_CACHE");
});

test("a 503 is shown as a readable message in the chat, not a blank screen", async () => {
  mockFetch((url) =>
    url.endsWith("/health")
      ? { status: 200, body: health }
      : { status: 503, body: { detail: "all providers unavailable" } },
  );
  render(<App />);
  await userEvent.type(screen.getByTestId("chat-input"), "hello{Enter}");

  const error = await screen.findByTestId("error-message");
  expect(error).toHaveTextContent(/unavailable/i);
  expect(screen.getByTestId("chat-input")).toBeInTheDocument();
});

test("a 422 shows the validation reason", async () => {
  mockFetch((url) =>
    url.endsWith("/health")
      ? { status: 200, body: health }
      : { status: 422, body: { detail: [{ loc: ["body", "message"], msg: "Value error, message must be at most 4000 characters after trimming" }] } },
  );
  render(<App />);
  await userEvent.type(screen.getByTestId("chat-input"), "x{Enter}");
  expect(await screen.findByTestId("error-message")).toHaveTextContent("message: message must be at most 4000 characters");
});

test("network failure is handled gracefully", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
  render(<App />);
  await userEvent.type(screen.getByTestId("chat-input"), "x{Enter}");
  expect(await screen.findByTestId("error-message")).toHaveTextContent(/can't reach the server/i);
});

test("system prompt and simulation header are sent when set", async () => {
  const fetchMock = mockFetch((url) => (url.endsWith("/health") ? { status: 200, body: health } : { status: 200, body: okReply() }));
  render(<App />);
  await userEvent.click(screen.getByRole("button", { name: "Settings" }));
  await userEvent.type(screen.getByTestId("system-prompt-input"), "Be terse.");
  await waitFor(() => expect(screen.getByTestId("simulate-failure")).toBeInTheDocument());
  await userEvent.selectOptions(screen.getByTestId("simulate-failure"), "key_1=429");
  await userEvent.type(screen.getByTestId("chat-input"), "hi{Enter}");
  await screen.findByTestId("assistant-message");

  const [, init] = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/chat"))!;
  expect(JSON.parse(String(init!.body)).system_prompt).toBe("Be terse.");
  expect((init!.headers as Record<string, string>)["X-Simulate-Failure"]).toBe("key_1=429");
  expect(screen.getByTestId("meta-fallbacks")).toHaveTextContent("failover ×1");
});

test("friendlyError covers the documented statuses", () => {
  expect(friendlyError(503, "all providers unavailable")).toMatch(/unavailable/);
  expect(friendlyError(422, "bad")).toMatch(/bad/);
  expect(friendlyError(500, undefined)).toMatch(/500/);
});

test("replies render markdown but never raw HTML", async () => {
  const reply = {
    ...okReply(),
    reply: "Here is **bold**, a list:\n\n- one\n- two\n\n`code` <img src=x onerror=\"alert(1)\"> <script>alert(2)</script>",
  };
  mockFetch((url) => (url.endsWith("/health") ? { status: 200, body: health } : { status: 200, body: reply }));
  const { container } = render(<App />);
  await userEvent.type(screen.getByTestId("chat-input"), "md{Enter}");
  const message = await screen.findByTestId("assistant-message");
  expect(within(message).getByText("bold").tagName).toBe("STRONG");
  expect(within(message).getAllByRole("listitem")).toHaveLength(2);
  expect(container.querySelector("img")).toBeNull();
  expect(container.querySelector("script")).toBeNull();
  // metadata still sits inside the assistant-message container
  expect(within(message).getByTestId("meta-source")).toHaveTextContent("LIVE_INFERENCE");
});

test("429 from the API is explained to the user", async () => {
  mockFetch((url) =>
    url.endsWith("/health") ? { status: 200, body: health } : { status: 429, body: { detail: "rate limit exceeded" } },
  );
  render(<App />);
  await userEvent.type(screen.getByTestId("chat-input"), "fast{Enter}");
  expect(await screen.findByTestId("error-message")).toHaveTextContent(/too quickly/i);
});

test("retry resends the failed message and replaces the error", async () => {
  let calls = 0;
  mockFetch((url) => {
    if (url.endsWith("/health")) return { status: 200, body: health };
    calls += 1;
    return calls === 1 ? { status: 503, body: { detail: "all providers unavailable" } } : { status: 200, body: okReply() };
  });
  render(<App />);
  await userEvent.type(screen.getByTestId("chat-input"), "try again{Enter}");
  await userEvent.click(await screen.findByRole("button", { name: "Retry" }));
  await screen.findByTestId("assistant-message");
  expect(screen.queryByTestId("error-message")).toBeNull();
  expect(screen.getAllByTestId("user-message")).toHaveLength(1);
});

test("shows a waking-up state while the backend is cold", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    let healthCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        healthCalls += 1;
        if (healthCalls < 3) throw new TypeError("Failed to fetch");
        return new Response(JSON.stringify(health), { status: 200, headers: { "Content-Type": "application/json" } });
      }),
    );
    render(<App />);
    expect(await screen.findByText(/waking up backend/i)).toBeInTheDocument();
    await vi.advanceTimersByTimeAsync(9000);
    await waitFor(() => expect(screen.getByTestId("health-status")).toHaveTextContent("2 keys · cache on"));
  } finally {
    vi.useRealTimers();
  }
});

test("a new chat starts a new session id", async () => {
  const fetchMock = mockFetch((url) => (url.endsWith("/health") ? { status: 200, body: health } : { status: 200, body: okReply() }));
  render(<App />);
  await userEvent.type(screen.getByTestId("chat-input"), "a{Enter}");
  await screen.findByTestId("assistant-message");
  await userEvent.click(screen.getByTestId("new-chat"));
  expect(screen.queryByTestId("assistant-message")).toBeNull();
  await userEvent.type(screen.getByTestId("chat-input"), "b{Enter}");
  await screen.findByTestId("assistant-message");
  const ids = fetchMock.mock.calls
    .filter(([url]) => String(url).endsWith("/chat"))
    .map(([, init]) => JSON.parse(String(init!.body)).session_id);
  expect(ids[0]).not.toBe(ids[1]);
});

test("failover control is visible but disabled when the backend has simulation off", async () => {
  const fetchMock = mockFetch((url) =>
    url.endsWith("/health")
      ? { status: 200, body: { ...health, failure_simulation_enabled: false } }
      : { status: 200, body: okReply() },
  );
  render(<App />);
  await userEvent.click(screen.getByRole("button", { name: "Settings" }));
  await waitFor(() => expect(screen.getByTestId("simulate-failure")).toBeDisabled());
  expect(screen.getByTestId("simulation-disabled-note")).toHaveTextContent("ENABLE_FAILURE_SIMULATION=true");
  await userEvent.type(screen.getByTestId("chat-input"), "hi{Enter}");
  await screen.findByTestId("assistant-message");
  const [, init] = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/chat"))!;
  expect((init!.headers as Record<string, string>)["X-Simulate-Failure"]).toBeUndefined();
});
