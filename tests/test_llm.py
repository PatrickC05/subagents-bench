"""The chat client through httpx.MockTransport (real SSE parsing, retries, errors, Prime and Sail
requests), the captured Sail tool-call re-send chunks and Prime's reasoning_details."""

from __future__ import annotations

import json
import time

import httpx
import pytest

from subagents_bench import llm
from subagents_bench.config import ModelConfig
from subagents_bench.llm import (
    BillingError,
    ContextLengthError,
    Message,
    ChatClient,
    ProviderError,
    ToolCall,
    ToolResult,
    ToolSpec,
    _Accumulator,
)

BASH = ToolSpec("bash", "Run a command.", {"type": "object", "properties": {"command": {"type": "string"}}})
USER = [Message("user", "hi")]


# --- captured Sail chunks (nvidia/Gemma-4-31B-IT-NVFP4, balanced, 2026-10-11) -----------

CMD = "cd /tmp/.sb/arb/a0001/cand1 && \\\ngit status && git reflog --all | head -50"
ARGS = json.dumps({"command": CMD})
CALL_ID = "call_resp_01a12e3f-d477-749a-93fc-7067c70944b1_3"


def chunk(delta: dict, finish: str | None = None) -> dict:
    return {"id": "resp_x", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def call(index: int, args: str = ARGS, call_id: str | None = CALL_ID, name: str | None = "bash") -> dict:
    tc: dict = {"index": index, "type": "function", "function": {}}
    if call_id:
        tc["id"] = call_id
    if name:
        tc["function"]["name"] = name
    if args is not None:
        tc["function"]["arguments"] = args
    return {"tool_calls": [tc]}


def feed(chunks: list[dict]) -> _Accumulator:
    acc = _Accumulator()
    for c in chunks:
        acc.feed(c)
    return acc


def test_sail_full_repeat_under_another_index_is_one_call():
    acc = feed([chunk({"role": "assistant"}), chunk(call(3)), chunk(call(0)), chunk({}, "tool_calls")])
    (c,) = acc.tool_calls()
    assert (c.id, c.name, c.arguments) == (CALL_ID, "bash", {"command": CMD})
    assert acc.stream_repairs == 1 and acc.finish_reason == "tool_calls"


def test_repeat_with_reformatted_json_is_still_a_repeat():
    acc = feed([chunk(call(3)), chunk(call(0, args=json.dumps({"command": CMD}, indent=1)))])
    (c,) = acc.tool_calls()
    assert c.arguments == {"command": CMD}


def test_two_distinct_calls_and_trailing_empty_deltas_are_untouched():
    other = json.dumps({"path": "/app/README.md"})
    acc = feed([
        chunk(call(0, call_id="call_a")),
        chunk(call(1, args=other, call_id="call_b", name="read_file")),
        chunk({"tool_calls": [{"index": 0, "function": {}}, {"index": 1, "function": {}}]}),
        chunk({}, "tool_calls"),
    ])
    assert [(c.id, c.name) for c in acc.tool_calls()] == [("call_a", "bash"), ("call_b", "read_file")]
    assert acc.stream_repairs == 0


def test_fragmented_arguments_without_ids_still_concatenate():
    acc = feed([chunk(call(0, args='{"command": "ec', call_id="call_a")), chunk(call(0, args='ho hi"}', call_id=None, name=None))])
    (c,) = acc.tool_calls()
    assert c.arguments == {"command": "echo hi"} and acc.stream_repairs == 0


def test_same_id_on_different_calls_gets_unique_ids():
    acc = feed([
        chunk({"tool_calls": [{"index": 0, "id": "dup", "function": {"name": "bash", "arguments": '{"command": "a"}'}}]}),
        chunk({"tool_calls": [{"index": 1, "id": "dup2", "function": {"name": "bash", "arguments": '{"command": "b"}'}}]}),
    ])
    acc.slots[1].id = "dup"  # a provider that reuses one id
    calls = acc.tool_calls()
    assert [c.arguments["command"] for c in calls] == ["a", "b"]
    assert calls[0].id == "dup" and calls[1].id.startswith("dup_dup") and acc.stream_repairs == 1


def test_unparseable_arguments_are_kept_raw():
    (c,) = feed([chunk(call(0, args="{not json", call_id="x"))]).tool_calls()
    assert c.arguments == {"_raw": "{not json"}


# --- the client over MockTransport -----------------------------------------------------


class Pieces(httpx.SyncByteStream):
    """A body delivered piece by piece, optionally ``gap`` seconds apart or failing mid-way."""

    def __init__(self, pieces: list[bytes], gap: float = 0.0, fail_at: int | None = None):
        self.pieces, self.gap, self.fail_at = pieces, gap, fail_at

    def __iter__(self):
        for i, p in enumerate(self.pieces):
            if i == self.fail_at:
                raise httpx.ReadError("connection reset by peer")
            time.sleep(self.gap)
            yield p


def sse(*events: dict, done: bool = True, gap: float = 0.0, fail_at: int | None = None, split: int = 7) -> httpx.Response:
    body = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + (b"data: [DONE]\n\n" if done else b"")
    pieces = [body[i : i + split] for i in range(0, len(body), split)] if not gap else [body]
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Pieces(pieces, gap, fail_at))


def ok(text: str = "ok", **usage) -> httpx.Response:
    return sse(chunk({"content": text}), chunk({}, "stop"), {"choices": [], "usage": usage or {"prompt_tokens": 5, "completion_tokens": 1}})


def error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message}})


class Server:
    """Pops one response (or exception) per request."""

    def __init__(self, *script):
        self.script, self.requests = list(script), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class MaxRng:
    def uniform(self, a: float, b: float) -> float:
        return b


@pytest.fixture(autouse=True)
def api_keys(monkeypatch):
    monkeypatch.setenv("SAIL_API_KEY", "sk-test")
    monkeypatch.setenv("PRIME_API_KEY", "pit-test")
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)


def client(server: Server, sleeps: list | None = None, **kw) -> ChatClient:
    cfg = ModelConfig(model=kw.pop("model", "deepseek-ai/DeepSeek-V4.1-Flash"), **{"provider": "sail", **kw})
    return ChatClient(cfg, transport=httpx.MockTransport(server), sleep=(sleeps if sleeps is not None else []).append, rng=MaxRng())


def test_request_body_headers_and_reasoning_passback():
    server = Server(ok())
    history = [
        Message("user", "fix it"),
        Message("assistant", "", [ToolCall("c1", "bash", {"command": "ls"})], reasoning="think first"),
        Message("tool", tool_results=[ToolResult("c1", "a.txt")]),
    ]
    client(server, max_output_tokens=321, temperature=0.7).complete("SYS", history, [BASH])
    req = server.requests[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://api.sailresearch.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test" and req.headers["x-sail-completion-window"] == "balanced"
    assert "reasoning" not in body and "x-prime-team-id" not in req.headers
    assert (body["model"], body["max_tokens"], body["temperature"]) == ("deepseek-ai/DeepSeek-V4.1-Flash", 321, 0.7)
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["tools"][0]["function"]["name"] == "bash"
    sys_msg, user, asst, tool = body["messages"]
    assert sys_msg == {"role": "system", "content": "SYS"} and user["content"] == "fix it"
    assert asst["reasoning_content"] == "think first"
    assert asst["tool_calls"][0]["function"] == {"name": "bash", "arguments": '{"command": "ls"}'}
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}


def test_reply_usage_and_timings():
    usage = {"prompt_tokens": 120, "completion_tokens": 40, "prompt_tokens_details": {"cached_tokens": 100},
             "completion_tokens_details": {"reasoning_tokens": 30}}
    server = Server(sse(
        chunk({"role": "assistant"}),
        chunk({"reasoning_content": "hmm "}),
        chunk({"reasoning_content": "ok"}),
        chunk({"content": "Running."}),
        chunk(call(0, args='{"command": "ls"}', call_id="c9")),
        chunk({}, "tool_calls"),
        {"choices": [], "usage": usage},
    ))
    r = client(server).complete("s", USER, [BASH])
    assert (r.text, r.reasoning, r.finish_reason) == ("Running.", "hmm ok", "tool_calls")
    assert r.tool_calls == [ToolCall("c9", "bash", {"command": "ls"})]
    assert (r.prompt_tokens, r.cached_tokens, r.completion_tokens, r.reasoning_tokens) == (120, 100, 40, 30)
    assert r.t_start <= r.t_first_chunk <= r.t_end and (r.n_attempts, r.backoff_sec, r.stream_repairs) == (1, 0.0, 0)


def test_retryable_failures_then_success():
    sleeps: list[float] = []
    server = Server(
        error(429, "rate limited, quota exceeded for this minute"),
        httpx.ConnectError("refused"),
        sse(chunk({"content": "par"}), fail_at=1),  # broken mid-stream
        sse(chunk({"content": "partial"}), done=False),  # ended without finish_reason
        ok("fine"),
    )
    r = client(server, sleeps).complete("s", USER, [])
    assert r.text == "fine" and r.n_attempts == 5 and len(server.requests) == 5
    assert sleeps == [2.0, 4.0, 8.0, 15.0] and r.backoff_sec == 29.0  # full jitter, capped at 15 s


@pytest.mark.parametrize("bad", [
    sse(chunk({"content": "x"}), {"error": {"message": "upstream failed", "code": 500}}),
    httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"data: {not json\n\n"),
    error(503, "overloaded"),
])
def test_stream_errors_and_5xx_are_retried(bad):
    assert client(Server(bad, ok("fine"))).complete("s", USER, []).text == "fine"


def test_a_stream_error_after_a_tool_call_is_retried_asking_for_one_call():
    """Sail's Gemma-4 parsing streams the first of two tool calls (under index 3), then fails, on
    every attempt; the retry adds ONE_CALL_NOTE to the system prompt and is recorded."""
    two_calls = sse(chunk({"role": "assistant"}), chunk(call(3, args='{"command": "ls"}', call_id="x_3")),
                    chunk(call(3, args='{"command": "pwd"}')),
                    {"error": {"code": "", "message": "chat completion failed", "type": "upstream_error"}}, done=False)
    one = sse(chunk(call(0, args='{"command": "ls"}', call_id="c1")), chunk({}, "tool_calls"))
    server = Server(two_calls, one)
    r = client(server).complete("SYS", USER, [BASH])
    assert r.tool_calls == [ToolCall("c1", "bash", {"command": "ls"})] and r.one_call_retry and r.n_attempts == 2
    first, second = (json.loads(q.content)["messages"][0]["content"] for q in server.requests)
    assert first == "SYS" and second == "SYS" + llm.ONE_CALL_NOTE
    # an error before any tool call, or a request without tools, retries the same request
    server = Server(sse(chunk({"content": "x"}), {"error": {"message": "upstream failed"}}, done=False), ok("fine"))
    r = client(server).complete("SYS", USER, [BASH])
    assert not r.one_call_retry and json.loads(server.requests[1].content)["messages"][0]["content"] == "SYS"


def test_retries_exhausted_raise_provider_error():
    server = Server(*[error(503, "down")] * 5)
    with pytest.raises(ProviderError) as ei:
        client(server, [], max_retries=4, retry_budget_sec=1e9).complete("s", USER, [])
    assert ei.value.n_attempts == 5 and ei.value.status == 503 and len(server.requests) == 5
    assert ei.value.backoff_sec == 2 + 4 + 8 + 15


def test_no_retry_starts_past_the_retry_budget(monkeypatch):
    clock = [1000.0]  # backoff sleeps advance a fake clock
    monkeypatch.setattr(llm.time, "time", lambda: clock[0])
    server = Server(error(503, "down"), error(503, "down"), ok())
    c = client(server, retry_budget_sec=5.0)
    c._sleep = lambda s: clock.__setitem__(0, clock[0] + s)
    with pytest.raises(ProviderError, match="giving up after 2 attempt"):
        c.complete("s", USER, [])
    assert clock[0] == 1002.0  # waited 2 s; the next wait (4 s) would start attempt 3 at 6 s


# Prime Inference's answer to a call billed to an empty account (docs: inference/troubleshooting)
PRIME_FUNDS = ('{"error": {"message": "Insufficient balance (including overdraft). Please add funds to continue.", '
               '"type": "insufficient_quota", "code": "insufficient_funds"}}')


@pytest.mark.parametrize("status,message,kind", [
    (402, "payment required", BillingError),
    (400, "Insufficient account balance", BillingError),
    (403, "credit limit reached", BillingError),
    (429, "insufficient_quota", BillingError),
    (402, PRIME_FUNDS, BillingError),
    (429, PRIME_FUNDS, BillingError),
    (404, '{"error":{"message":"The model \'x/y\' does not exist","type":"server_error","code":"model_not_found"}}',
     ProviderError),
    (400, "The decoder prompt (length 9000) is longer than the maximum model length of 8192.", ContextLengthError),
    (413, "context_length_exceeded", ContextLengthError),
    (400, "invalid value for temperature", ProviderError),
    (401, "invalid api key", ProviderError),
])
def test_error_classification_never_retries(status, message, kind):
    server = Server(error(status, message), ok())
    with pytest.raises(kind) as ei:
        client(server).complete("s", USER, [])
    assert type(ei.value) is kind and ei.value.status == status and ei.value.n_attempts == 1 and len(server.requests) == 1


@pytest.mark.parametrize("provider,var", [("sail", "SAIL_API_KEY"), ("prime", "PRIME_API_KEY")])
def test_missing_key_fails_before_any_request(monkeypatch, provider, var):
    monkeypatch.delenv(var)
    server = Server()
    with pytest.raises(ProviderError, match=var):
        client(server, provider=provider).complete("s", USER, [])
    assert server.requests == []


def paced(pieces: list[bytes], gap: float) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Pieces(pieces, gap))


def test_keep_alive_bytes_without_data_fail_the_attempt():
    server = Server(paced([b": keep-alive\n\n"] * 60, 0.05), ok("fine"))
    t0 = time.time()
    r = client(server, request_timeout_sec=0.5).complete("s", USER, [])
    assert r.text == "fine" and r.n_attempts == 2 and time.time() - t0 < 2.0


def test_a_stall_after_some_data_also_fails_the_attempt():
    first = b"data: " + json.dumps(chunk({"content": "partial"})).encode() + b"\n\n"
    server = Server(paced([first] + [b": ping\n\n"] * 40, 0.05), ok("fine"))
    assert client(server, request_timeout_sec=0.5).complete("s", USER, []).n_attempts == 2


def test_a_slow_stream_that_keeps_sending_data_is_not_cut():
    events = [chunk({"content": f"w{i} "}) for i in range(12)] + [chunk({}, "stop")]
    pieces = [b"data: " + json.dumps(e).encode() + b"\n\n" for e in events] + [b"data: [DONE]\n\n"]
    r = client(Server(paced(pieces, 0.1)), request_timeout_sec=0.5).complete("s", USER, [])
    assert r.text.startswith("w0 w1") and r.n_attempts == 1


# --- Prime Inference ----------------------------------------------------------------------


def test_prime_url_key_and_team_header(monkeypatch):
    server = Server(ok(), ok())
    c = client(server, provider="prime", model="openai/gpt-5.6-luna")
    c.complete("s", USER, [])
    monkeypatch.setenv("PRIME_TEAM_ID", "team-7")
    c.complete("s", USER, [])
    first, second = server.requests
    assert str(first.url) == "https://api.pinference.ai/api/v1/chat/completions"
    assert first.headers["authorization"] == "Bearer pit-test" and "x-sail-completion-window" not in first.headers
    assert "x-prime-team-id" not in first.headers and second.headers["x-prime-team-id"] == "team-7"


# captured from anthropic/claude-sonnet-5.5 and openai/gpt-5.6-luna on Prime (2026-10-25), shortened: text
# and summaries stream in pieces, a signature or encrypted blob arrives whole
DETAILS = [
    [{"type": "reasoning.text", "text": "From", "format": "anthropic-claude-v1", "index": 0}],
    [{"type": "reasoning.text", "text": " the notes", "format": "anthropic-claude-v1", "index": 0}],
    [{"type": "reasoning.text", "signature": "CAQS7QYK", "format": "anthropic-claude-v1", "index": 0}],
    [{"type": "reasoning.encrypted", "data": "gAAAAABq", "id": "rs_02", "format": "azure-openai-responses-v1",
      "index": 1}],
]


def test_reasoning_details_are_merged_and_sent_back_instead_of_reasoning_content():
    server = Server(sse(
        chunk({"role": "assistant"}),
        *(chunk({"reasoning": d[0].get("text"), "reasoning_details": d}) for d in DETAILS[:3]),
        chunk({"reasoning_details": DETAILS[3]}),  # no reasoning text, as Terra and Argon sometimes stream
        chunk(call(0, args='{"command": "ls"}', call_id="c1")),
        chunk({}, "tool_calls"),
    ), ok())
    c = client(server, provider="prime", model="anthropic/claude-sonnet-5.5")
    r = c.complete("s", USER, [BASH])
    assert r.reasoning == "From the notes"
    assert r.reasoning_details == [
        {"type": "reasoning.text", "text": "From the notes", "signature": "CAQS7QYK", "format": "anthropic-claude-v1",
         "index": 0},
        {"type": "reasoning.encrypted", "data": "gAAAAABq", "id": "rs_02", "format": "azure-openai-responses-v1",
         "index": 1},
    ]
    history = [*USER, Message("assistant", r.text, r.tool_calls, reasoning=r.reasoning,
                              reasoning_details=r.reasoning_details),
               Message("tool", tool_results=[ToolResult("c1", "a.txt")])]
    c.complete("s", history, [BASH])
    asst = json.loads(server.requests[1].content)["messages"][2]
    assert asst["reasoning_details"] == r.reasoning_details and "reasoning_content" not in asst


def test_reasoning_details_only_chunk_is_data_not_a_stall():
    server = Server(paced([b"data: " + json.dumps(chunk({"reasoning_details": DETAILS[3]})).encode() + b"\n\n"] * 12
                          + [b"data: " + json.dumps(chunk({"content": "ok"}, "stop")).encode() + b"\n\ndata: [DONE]\n\n"],
                          0.05))
    r = client(server, provider="prime", request_timeout_sec=0.3).complete("s", USER, [])
    assert r.text == "ok" and r.n_attempts == 1 and r.t_first_chunk is not None


def test_effort_and_cache_breakpoints():
    server = Server(ok(), ok())
    history = [
        Message("user", "fix it"),
        Message("assistant", "", [ToolCall("c1", "bash", {"command": "ls"})]),
        Message("tool", tool_results=[ToolResult("c1", "a.txt"), ToolResult("c2", "b.txt")]),
    ]
    c = client(server, provider="prime", model="anthropic/claude-opus-5.5", reasoning_effort="high", cache_control=True)
    c.complete("SYS", history, [BASH])
    body = json.loads(server.requests[0].content)
    assert body["reasoning"] == {"effort": "high"} and "reasoning_effort" not in body
    mark = {"type": "ephemeral"}
    sys_msg, user, asst, tool1, tool2 = body["messages"]
    assert sys_msg["content"] == [{"type": "text", "text": "SYS", "cache_control": mark}]
    assert user["content"] == "fix it" and asst["content"] == "" and tool1["content"] == "a.txt"
    assert tool2["content"] == [{"type": "text", "text": "b.txt", "cache_control": mark}]  # only the newest
    assert history[2].tool_results[1].content == "b.txt"  # the history itself is unchanged
    c.complete("SYS", [*history, Message("assistant", "")], [])  # an empty text cannot carry a breakpoint
    assert json.loads(server.requests[1].content)["messages"][-1]["content"] == ""
