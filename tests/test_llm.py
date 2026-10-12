"""Unit tests for the LLM clients and credentials (no network).

The OpenAI-compatible client is driven through ``httpx.MockTransport`` with
SSE bodies delivered as separate byte chunks (deliberately split mid-line and
mid-JSON), so the tests exercise the real streaming parser, retry loop and
error classification. Backoff sleeps are injected and recorded, never slept.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from dataclasses import asdict
from types import SimpleNamespace

import httpx
import pytest

from subagents_bench import llm
from subagents_bench.config import ModelConfig
from subagents_bench.credentials import (
    describe_credentials,
    load_dotenv,
    parse_dotenv,
    prime_credentials,
    resolve_api_key,
)
from subagents_bench.llm import (
    AnthropicClient,
    BillingError,
    ContextLengthError,
    LLMError,
    Message,
    OpenAIClient,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    _cost,
    _reported_cost,
    make_client,
)
from subagents_bench.types import CHUNK_OFFSETS_CAP

SAIL_KEY = "sk-sail-test-123"
TOOLS = [
    ToolSpec(
        name="bash",
        description="Run a command.",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    )
]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ChunkStream(httpx.SyncByteStream):
    """Response body delivered as separate byte chunks; may fail mid-way."""

    def __init__(self, chunks: list[bytes], fail_after: int | None = None):
        self.chunks = chunks
        self.fail_after = fail_after

    def __iter__(self):
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise httpx.ReadError("connection reset by peer")
            yield c


def sse_bytes(*events: dict, done: bool = True) -> bytes:
    out = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
    return out + (b"data: [DONE]\n\n" if done else b"")


def split_every(data: bytes, n: int) -> list[bytes]:
    return [data[i : i + n] for i in range(0, len(data), n)]


def sse_response(*events: dict, done: bool = True, split: int = 7, headers: dict | None = None, fail_after=None):
    body = sse_bytes(*events, done=done)
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream", **(headers or {})},
        stream=ChunkStream(split_every(body, split), fail_after=fail_after),
    )


def delta(rid="chatcmpl-1", finish=None, **d):
    return {"id": rid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


def usage_chunk(rid="chatcmpl-1", **usage):
    return {"id": rid, "object": "chat.completion.chunk", "choices": [], "usage": usage}


def simple_ok(text="ok", usage=None):
    """A minimal valid streamed reply."""
    events = [delta(role="assistant", content=""), delta(content=text), delta(finish="stop")]
    if usage is not None:
        events.append(usage_chunk(**usage))
    return sse_response(*events)


class Server:
    """Scripted MockTransport handler: pops one response (or exception) per request."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(request)
        return item


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Hermetic credentials: test keys in env, empty HOME (no ~/.prime)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SAIL_API_KEY", SAIL_KEY)
    for k in ("PRIME_API_KEY", "PRIME_TEAM_ID", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def make(server: Server, sleeps: list | None = None, **cfg_kw) -> OpenAIClient:
    kw = {"model": "deepseek-ai/DeepSeek-V4.1-Flash", "provider": "sail", **cfg_kw}
    return make_client(
        ModelConfig(**kw),
        transport=httpx.MockTransport(server),
        sleep=(sleeps.append if sleeps is not None else (lambda s: None)),
        rng=random.Random(0),
    )


USER = [Message(role="user", content="hi")]


# ---------------------------------------------------------------------------
# make_client / config
# ---------------------------------------------------------------------------


def test_provider_defaults_and_no_mutation(env):
    cfg = ModelConfig(model="Qwen/Qwen3.5-2B", provider="prime")
    c = make_client(cfg)
    assert c.cfg.base_url == "https://api.pinference.ai/api/v1"
    assert c.cfg.api_key_env == "PRIME_API_KEY"
    assert cfg.base_url is None and cfg.api_key_env is None  # caller's cfg untouched
    for p in ("prime_intellect", "pinference"):
        assert make_client(ModelConfig(model="m", provider=p)).cfg.base_url == "https://api.pinference.ai/api/v1"
    s = make_client(ModelConfig(model="m", provider="sail"))
    assert (s.cfg.base_url, s.cfg.api_key_env) == ("https://api.sailresearch.com/v1", "SAIL_API_KEY")
    assert isinstance(s, OpenAIClient)


def test_explicit_base_url_and_key_env_win(env):
    c = make_client(ModelConfig(model="m", provider="prime", base_url="http://localhost:9/v1", api_key_env="MY_KEY"))
    assert c.cfg.base_url == "http://localhost:9/v1"
    assert c.cfg.api_key_env == "MY_KEY"


@pytest.mark.parametrize("provider", ["vllm", "openai", "openai_compatible"])
def test_generic_providers_need_base_url(env, provider):
    with pytest.raises(ValueError, match="base_url"):
        make_client(ModelConfig(model="m", provider=provider))
    c = make_client(ModelConfig(model="m", provider=provider, base_url="http://h:8000/v1"))
    assert isinstance(c, OpenAIClient)


def test_unknown_provider_and_anthropic_default(env):
    with pytest.raises(ValueError, match="unknown provider"):
        make_client(ModelConfig(model="m", provider="nope"))
    a = make_client(ModelConfig(model="claude-x", provider="anthropic"))
    assert isinstance(a, AnthropicClient)
    assert a.cfg.api_key_env == "ANTHROPIC_API_KEY"


def test_model_config_old_yaml_dict_still_loads():
    old = {
        "model": "Qwen/Qwen3.5-2B",
        "provider": "prime",
        "max_output_tokens": 4096,
        "temperature": 0.2,
        "price_in": 0.06,
        "price_out": 0.18,
        "price_cache_read": 0.054,
    }
    m = ModelConfig(**old)
    assert m.completion_window is None
    assert m.extra_headers == {}
    assert m.reasoning_passback is True
    assert m.context_window is None
    assert m.max_retries == 8
    assert m.request_timeout_sec == 600.0
    assert m.stream is True
    assert m.active_params_b is None and m.total_params_b is None
    # New fields round-trip through a plain dict.
    m2 = ModelConfig(**{**old, "completion_window": "balanced", "extra_headers": {"X-A": "1"}, "context_window": 16384})
    assert ModelConfig(**asdict(m2)) == m2


# ---------------------------------------------------------------------------
# Streaming assembly and timings
# ---------------------------------------------------------------------------


def test_stream_assembles_reasoning_content_and_tool_calls(env):
    events = [
        delta(role="assistant", content=""),  # role-only: not a delta chunk
        delta(reasoning_content="Let me "),
        delta(reasoning_content="think."),
        delta(content="Running "),
        delta(content="two commands."),
        # Two tool calls, interleaved fragments keyed by index; id/name only first.
        delta(tool_calls=[{"index": 0, "id": "call_a", "type": "function", "function": {"name": "bash", "arguments": ""}}]),
        delta(tool_calls=[{"index": 1, "id": "call_b", "type": "function", "function": {"name": "bash", "arguments": '{"comm'}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '{"command": "l'}}]),
        delta(tool_calls=[{"index": 1, "function": {"arguments": 'and": "pwd"}'}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": 's -la"}'}}]),
        delta(finish="tool_calls"),
        usage_chunk(prompt_tokens=120, completion_tokens=40),
    ]
    server = Server(sse_response(*events, split=5))
    resp = make(server).complete("sys", USER, TOOLS)

    assert resp.reasoning == "Let me think."
    assert resp.text == "Running two commands."
    assert [(tc.id, tc.name, tc.arguments) for tc in resp.tool_calls] == [
        ("call_a", "bash", {"command": "ls -la"}),
        ("call_b", "bash", {"command": "pwd"}),
    ]
    assert resp.finish_reason == "tool_calls"
    assert resp.response_id == "chatcmpl-1"
    t = resp.timing
    assert t.streamed is True
    # 2 reasoning + 2 content + 5 tool fragments; role-only, finish-only and usage chunks excluded.
    assert t.n_chunks == 9
    assert len(t.chunk_offsets_ms) == 9
    assert t.chunk_offsets_ms == sorted(t.chunk_offsets_ms)
    assert all(o >= 0 for o in t.chunk_offsets_ms)
    assert t.t_start <= t.t_request_sent <= t.t_headers <= t.t_first_chunk
    assert t.t_first_chunk == t.t_first_reasoning
    assert t.t_first_reasoning <= t.t_first_content <= t.t_first_tool_call <= t.t_last_chunk <= t.t_end
    assert len(resp.attempts) == 1 and resp.attempts[0].status == 200 and resp.attempts[0].error_type is None


def test_vllm_style_reasoning_field_and_name_fragments(env):
    events = [
        delta(reasoning="hmm"),
        delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "read_", "arguments": ""}}]),
        delta(tool_calls=[{"index": 0, "function": {"name": "file", "arguments": '{"path": "a"}'}}]),
        delta(finish="tool_calls"),
    ]
    resp = make(Server(sse_response(*events))).complete("s", USER, TOOLS)
    assert resp.reasoning == "hmm"
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].arguments == {"path": "a"}


def test_repeated_full_name_not_duplicated_and_missing_id_generated(env):
    events = [
        delta(tool_calls=[{"index": 0, "function": {"name": "bash", "arguments": '{"command":'}}]),
        delta(tool_calls=[{"index": 0, "function": {"name": "bash", "arguments": ' "ls"}'}}]),
        delta(finish="tool_calls"),
    ]
    resp = make(Server(sse_response(*events))).complete("s", USER, TOOLS)
    assert resp.tool_calls[0].name == "bash"
    assert resp.tool_calls[0].arguments == {"command": "ls"}
    assert resp.tool_calls[0].id.startswith("call_") and len(resp.tool_calls[0].id) > 5


def test_invalid_tool_arguments_kept_raw(env):
    events = [
        delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls'}}]),
        delta(tool_calls=[{"index": 1, "id": "c2", "function": {"name": "bash", "arguments": '"{\\"command\\": \\"pwd\\"}"'}}]),
        delta(tool_calls=[{"index": 2, "id": "c3", "function": {"name": "bash", "arguments": "[1, 2]"}}]),
        delta(finish="tool_calls"),
    ]
    resp = make(Server(sse_response(*events))).complete("s", USER, TOOLS)
    assert resp.tool_calls[0].arguments == {"_raw": '{"command": "ls'}
    assert resp.tool_calls[1].arguments == {"command": "pwd"}  # double-encoded JSON unwrapped
    assert resp.tool_calls[2].arguments == {"_raw": "[1, 2]"}


def test_chunk_offsets_capped(env):
    n = CHUNK_OFFSETS_CAP + 10
    events = [delta(content="x") for _ in range(n)] + [delta(finish="length")]
    resp = make(Server(sse_response(*events, split=4096))).complete("s", USER, [])
    assert resp.text == "x" * n
    assert resp.timing.n_chunks == n
    assert len(resp.timing.chunk_offsets_ms) == CHUNK_OFFSETS_CAP
    assert resp.finish_reason == "length"


def test_response_headers_without_cookies(env):
    server = Server(
        sse_response(
            delta(content="hi"),
            delta(finish="stop"),
            headers={"set-cookie": "session=secret", "x-request-id": "req-42"},
        )
    )
    resp = make(server).complete("s", USER, [])
    assert resp.response_headers["x-request-id"] == "req-42"
    assert "set-cookie" not in {k.lower() for k in resp.response_headers}
    assert "session=secret" not in json.dumps(resp.response_headers)


def test_finish_reason_without_done_is_accepted(env):
    server = Server(sse_response(delta(content="hi"), delta(finish="stop"), done=False))
    resp = make(server).complete("s", USER, [])
    assert resp.text == "hi" and resp.finish_reason == "stop"
    assert len(resp.attempts) == 1


def test_done_without_finish_reason_is_unknown_not_stop(env):
    server = Server(sse_response(delta(content="hi"), done=True))
    resp = make(server).complete("s", USER, [])
    assert resp.finish_reason == "unknown"
    assert len(resp.attempts) == 1


# ---------------------------------------------------------------------------
# Usage, estimation and cost
# ---------------------------------------------------------------------------


def test_usage_parsing_cached_reasoning_and_cost(env):
    usage = {
        "prompt_tokens": 2048,
        "completion_tokens": 300,
        "total_tokens": 2348,
        "prompt_tokens_details": {"cached_tokens": 1536},
        "completion_tokens_details": {"reasoning_tokens": 210},
        "cost": 0.5,
    }
    server = Server(simple_ok("done", usage=usage))
    resp = make(server, price_in=0.07, price_out=0.22, price_cache_read=0.005).complete("s", USER, [])
    u = resp.usage
    assert (u.prompt_tokens, u.completion_tokens, u.cached_tokens, u.reasoning_tokens) == (2048, 300, 1536, 210)
    assert u.uncached_prompt_tokens == 512
    assert resp.usage_source == "provider"
    assert resp.usage_raw == usage
    expected = (512 * 0.07 + 1536 * 0.005 + 300 * 0.22) / 1e6
    assert resp.cost_usd == pytest.approx(expected, rel=1e-12)
    # Provider-reported cost is captured verbatim but never used for cost_usd.
    assert resp.provider_reported_cost == 0.5
    assert resp.cost_usd != 0.5


def test_usage_without_reasoning_details_is_none(env):
    # Prime-style usage: no completion_tokens_details, cost 0.0.
    usage = {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55, "cost": 0.0}
    resp = make(Server(simple_ok(usage=usage))).complete("s", USER, [])
    assert resp.usage.reasoning_tokens is None
    assert resp.usage.cached_tokens == 0
    assert resp.provider_reported_cost == 0.0
    assert resp.usage_source == "provider"


def test_missing_usage_is_estimated(env):
    events = [delta(reasoning_content="abc"), delta(content="hello world"), delta(finish="stop")]
    resp = make(Server(sse_response(*events)), price_in=1.0, price_out=2.0).complete("system prompt", USER, TOOLS)
    assert resp.usage_source == "estimated"
    assert resp.usage_raw is None
    assert resp.provider_reported_cost is None
    assert resp.usage.prompt_tokens == math.ceil(resp.request.prompt_chars / 4)
    assert resp.usage.completion_tokens == math.ceil(len("abc" + "hello world") / 4) == 4
    assert resp.usage.cached_tokens == 0 and resp.usage.reasoning_tokens is None
    assert resp.cost_usd == pytest.approx((resp.usage.prompt_tokens * 1.0 + 4 * 2.0) / 1e6)


def test_cost_math():
    cfg = ModelConfig(model="m", provider="openai", price_in=1.0, price_out=2.0)
    assert _cost(cfg, Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)) == 3.0
    # Cached tokens fall back to price_in when price_cache_read is unset...
    assert _cost(cfg, Usage(prompt_tokens=1_000_000, cached_tokens=400_000)) == pytest.approx(1.0)
    # ...and use price_cache_read when set.
    cfg.price_cache_read = 0.1
    assert _cost(cfg, Usage(prompt_tokens=1_000_000, cached_tokens=400_000)) == pytest.approx(0.6 + 0.04)
    # No price configured -> unknown (None), never a made-up 0; an explicit 0.0 is free.
    assert _cost(ModelConfig(model="m"), Usage(prompt_tokens=10, completion_tokens=10)) is None
    assert _cost(ModelConfig(model="m", price_in=1.0), Usage(prompt_tokens=10)) is None
    assert _cost(ModelConfig(model="m", price_in=0.0, price_out=0.0), Usage(prompt_tokens=10, completion_tokens=10)) == 0.0


def test_reported_cost_variants():
    assert _reported_cost({"cost": 0.0123}) == 0.0123
    assert _reported_cost({"cost": {"total_usd": 0.5}}) == 0.5
    assert _reported_cost(SimpleNamespace(cost={"usd": 0.25})) == 0.25
    assert _reported_cost({"cost": "0.75"}) == 0.75
    assert _reported_cost({"cost": True}) is None
    assert _reported_cost({"prompt_tokens": 1}) is None
    assert _reported_cost(None) is None


# ---------------------------------------------------------------------------
# Retries and error classification
# ---------------------------------------------------------------------------


def test_429_retry_after_then_success(env):
    sleeps: list[float] = []
    server = Server(
        httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "rate limited"}}),
        simple_ok("fine", usage={"prompt_tokens": 3, "completion_tokens": 1}),
    )
    resp = make(server, sleeps).complete("s", USER, [])
    assert resp.text == "fine"
    assert sleeps == [7.0]  # Retry-After (7s) exceeds the first jitter ceiling (2s)
    a0, a1 = resp.attempts
    assert (a0.attempt, a0.status, a0.error_type, a0.retry_after_sec, a0.backoff_sec) == (1, 429, "RateLimited", 7.0, 7.0)
    assert "rate limited" in a0.error
    assert (a1.attempt, a1.status, a1.error_type, a1.backoff_sec) == (2, 200, None, 0.0)
    assert a0.t_end <= a1.t_start
    assert resp.timing.t_start <= a0.t_start and a1.t_end <= resp.timing.t_end
    assert len(server.requests) == 2


def test_retry_after_ms_and_http_date():
    assert llm._parse_retry_after(httpx.Headers({"retry-after-ms": "1500"})) == 1.5
    assert llm._parse_retry_after(httpx.Headers({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})) == 0.0
    assert llm._parse_retry_after(httpx.Headers({"retry-after": "soon"})) is None
    assert llm._parse_retry_after(httpx.Headers({})) is None


def test_503_exhausts_retries(env):
    sleeps: list[float] = []
    server = Server(*[httpx.Response(503, text="upstream overloaded") for _ in range(4)])
    client = make(server, sleeps, max_retries=3)
    with pytest.raises(LLMError) as ei:
        client.complete("s", USER, [])
    e = ei.value
    assert type(e) is LLMError  # not a Billing/ContextLength subclass
    assert e.status == 503
    assert [a.status for a in e.attempts] == [503, 503, 503, 503]
    assert [a.error_type for a in e.attempts] == ["ServerError"] * 4
    assert [a.attempt for a in e.attempts] == [1, 2, 3, 4]
    assert len(sleeps) == 3 and len(server.requests) == 4
    # Full jitter under ceilings 2, 4, 8; the final attempt has no backoff.
    for s, ceiling in zip(sleeps, (2.0, 4.0, 8.0)):
        assert 0.0 <= s <= ceiling
    assert [a.backoff_sec for a in e.attempts] == sleeps + [0.0]
    assert "4 attempts" in str(e)
    # The error carries the RequestInfo of what was sent (exact request hashes for the failed-call record).
    ok = make(Server(simple_ok("x")), []).complete("s", USER, [])
    assert e.request is not None and e.request.msg_hashes == ok.request.msg_hashes
    assert e.request.system_hash == ok.request.system_hash and e.request.n_messages == 1


def test_backoff_ceiling_caps_at_120():
    c = OpenAIClient(ModelConfig(model="m", provider="vllm", base_url="http://x/v1"), rng=random.Random(1))
    assert all(c._backoff(k, None) <= 120.0 for k in range(1, 30))
    assert c._backoff(1, 50.0) == 50.0
    assert c._backoff(1, 10_000.0) == llm.RETRY_AFTER_MAX_SEC


def test_transport_error_retried(env):
    sleeps: list[float] = []
    server = Server(httpx.ConnectError("connection refused"), simple_ok("ok"))
    resp = make(server, sleeps).complete("s", USER, [])
    assert resp.text == "ok"
    assert resp.attempts[0].status is None
    assert resp.attempts[0].error_type == "ConnectError"
    assert len(sleeps) == 1 and resp.attempts[0].backoff_sec == sleeps[0]


def test_mid_stream_read_error_retried(env):
    first = sse_response(delta(content="partial"), delta(content=" more"), delta(finish="stop"), split=10, fail_after=2)
    server = Server(first, simple_ok("whole"))
    resp = make(server).complete("s", USER, [])
    assert resp.text == "whole"  # partial text from the failed attempt is discarded
    assert resp.attempts[0].error_type == "ReadError"
    assert resp.attempts[0].status == 200  # headers had arrived


def test_truncated_stream_retried(env):
    truncated = sse_response(delta(content="cut o"), done=False)
    server = Server(truncated, simple_ok("complete"))
    resp = make(server).complete("s", USER, [])
    assert resp.text == "complete"
    assert [a.error_type for a in resp.attempts] == ["TruncatedStream", None]


def test_unparseable_sse_retried(env):
    bad = httpx.Response(200, stream=ChunkStream([b"data: {not json\n\n"]))
    resp = make(Server(bad, simple_ok("ok"))).complete("s", USER, [])
    assert resp.attempts[0].error_type == "StreamParseError"
    assert resp.text == "ok"


def test_in_stream_error_object(env):
    err_stream = sse_response(delta(content="a"), {"error": {"message": "internal engine error", "code": 500}})
    resp = make(Server(err_stream, simple_ok("ok"))).complete("s", USER, [])
    assert resp.attempts[0].error_type == "StreamError" and resp.attempts[0].status == 500
    ctx_stream = sse_response({"error": {"message": "maximum context length exceeded", "code": 400}})
    with pytest.raises(ContextLengthError):
        make(Server(ctx_stream)).complete("s", USER, [])


@pytest.mark.parametrize(
    "status,body",
    [
        (402, {"error": "Payment Required"}),
        (400, {"error": {"message": "Insufficient balance. Please top up your account."}}),
        (403, {"error": {"message": "Your credit limit has been reached"}}),
        (429, {"error": {"message": "You exceeded your current quota"}}),
    ],
)
def test_billing_errors_not_retried(env, status, body):
    sleeps: list[float] = []
    server = Server(httpx.Response(status, json=body), simple_ok())
    with pytest.raises(BillingError) as ei:
        make(server, sleeps).complete("s", USER, [])
    assert ei.value.status == status
    assert len(ei.value.attempts) == 1 and ei.value.attempts[0].error_type == "Billing"
    assert sleeps == [] and len(server.requests) == 1
    assert ei.value.request is not None and ei.value.request.n_messages == 1


def test_context_length_error(env):
    body = {"error": {"message": "This model's maximum context length is 16384 tokens. However, you requested 20000 tokens."}}
    server = Server(httpx.Response(400, json=body))
    with pytest.raises(ContextLengthError) as ei:
        make(server).complete("s", USER, [])
    assert ei.value.status == 400 and len(ei.value.attempts) == 1
    assert isinstance(ei.value, LLMError) and not isinstance(ei.value, BillingError)


def test_other_4xx_fails_fast(env):
    sleeps: list[float] = []
    server = Server(httpx.Response(401, json={"error": "invalid api key"}))
    with pytest.raises(LLMError) as ei:
        make(server, sleeps).complete("s", USER, [])
    assert type(ei.value) is LLMError and ei.value.status == 401
    assert ei.value.attempts[0].error_type == "HTTP401"
    assert sleeps == []
    server = Server(httpx.Response(400, json={"error": "unknown field 'foo'"}))
    with pytest.raises(LLMError) as ei:
        make(server).complete("s", USER, [])
    assert type(ei.value) is LLMError


def test_missing_api_key_fails_before_request(env):
    env.delenv("SAIL_API_KEY")
    server = Server(simple_ok())
    with pytest.raises(LLMError, match="SAIL_API_KEY"):
        make(server).complete("s", USER, [])
    assert server.requests == []


def test_keyless_vllm_sends_no_auth(env):
    server = Server(simple_ok())
    client = make_client(
        ModelConfig(model="m", provider="vllm", base_url="http://gpu:8000/v1/"),
        transport=httpx.MockTransport(server),
    )
    client.complete("s", USER, [])
    req = server.requests[0]
    assert str(req.url) == "http://gpu:8000/v1/chat/completions"
    assert "authorization" not in req.headers


# ---------------------------------------------------------------------------
# Request headers and body
# ---------------------------------------------------------------------------


def test_sail_headers(env):
    server = Server(simple_ok())
    client = make(server, completion_window="balanced", extra_headers={"X-Extra": "1", "user-agent": "override/9"})
    client.complete("s", USER, [])
    h = server.requests[0].headers
    assert h["authorization"] == f"Bearer {SAIL_KEY}"
    assert h["x-sail-completion-window"] == "balanced"
    assert h["x-extra"] == "1"
    assert h.get_list("user-agent") == ["override/9"]  # extra_headers replace, not duplicate
    assert "x-prime-team-id" not in h
    assert str(server.requests[0].url) == "https://api.sailresearch.com/v1/chat/completions"


def test_default_user_agent_and_no_tier_header(env):
    server = Server(simple_ok())
    make(server).complete("s", USER, [])
    h = server.requests[0].headers
    assert h["user-agent"].startswith("subagents-bench/")
    assert "python" not in h["user-agent"].lower()
    assert "x-sail-completion-window" not in h


def test_prime_headers_from_env(env):
    env.setenv("PRIME_API_KEY", "pk-env")
    env.setenv("PRIME_TEAM_ID", "team-env")
    server = Server(simple_ok())
    client = make_client(ModelConfig(model="Qwen/Qwen3.5-2B", provider="prime"), transport=httpx.MockTransport(server))
    client.complete("s", USER, [])
    h = server.requests[0].headers
    assert h["authorization"] == "Bearer pk-env"
    assert h["x-prime-team-id"] == "team-env"
    assert h["user-agent"].startswith("subagents-bench/")


def test_prime_headers_from_cli_config(env, tmp_path):
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "config.json").write_text(json.dumps({"api_key": "pk-file", "team_id": "team-file"}))
    server = Server(simple_ok())
    client = make_client(ModelConfig(model="Qwen/Qwen3.5-2B", provider="prime"), transport=httpx.MockTransport(server))
    client.complete("s", USER, [])
    h = server.requests[0].headers
    assert h["authorization"] == "Bearer pk-file"
    assert h["x-prime-team-id"] == "team-file"


def _conversation(reasoning: str | None = "I should list files.") -> list[Message]:
    return [
        Message(role="user", content="list files"),
        Message(
            role="assistant",
            content=None,
            reasoning=reasoning,
            tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "ls"}), ToolCall(id="c2", name="bash", arguments={"command": "pwd"})],
        ),
        Message(role="tool", tool_results=[ToolResult(call_id="c1", content="a.txt"), ToolResult(call_id="c2", content="/app", is_error=True)]),
    ]


def test_request_body_mapping(env):
    server = Server(simple_ok())
    client = make(server, max_output_tokens=777, temperature=0.7, extra_body={"chat_template_kwargs": {"thinking": False}})
    client.complete("the system", _conversation(), TOOLS)
    body = server.bodies[0]
    assert body["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert body["max_tokens"] == 777 and body["temperature"] == 0.7
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"] == {"thinking": False}
    assert body["tools"] == [{"type": "function", "function": {"name": "bash", "description": "Run a command.", "parameters": TOOLS[0].parameters}}]
    msgs = body["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "tool"]
    assert msgs[0] == {"role": "system", "content": "the system"}
    asst = msgs[2]
    assert asst["content"] == ""
    assert [(tc["id"], tc["function"]["name"], json.loads(tc["function"]["arguments"])) for tc in asst["tool_calls"]] == [
        ("c1", "bash", {"command": "ls"}),
        ("c2", "bash", {"command": "pwd"}),
    ]
    assert asst["reasoning_content"] == "I should list files."
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}
    assert msgs[4] == {"role": "tool", "tool_call_id": "c2", "content": "/app"}


def test_reasoning_passback_toggle(env):
    on, off = Server(simple_ok()), Server(simple_ok())
    r_on = make(on, reasoning_passback=True).complete("s", _conversation(), [])
    r_off = make(off, reasoning_passback=False).complete("s", _conversation(), [])
    assert on.bodies[0]["messages"][2]["reasoning_content"] == "I should list files."
    assert "reasoning_content" not in off.bodies[0]["messages"][2]
    assert r_on.request.reasoning_passback is True and r_off.request.reasoning_passback is False
    # Only the assistant message's hash differs.
    assert r_on.request.msg_hashes[0] == r_off.request.msg_hashes[0]
    assert r_on.request.msg_hashes[1] != r_off.request.msg_hashes[1]
    assert r_on.request.msg_hashes[2:] == r_off.request.msg_hashes[2:]
    # Nothing to pass back -> no field even when enabled.
    none = Server(simple_ok())
    make(none).complete("s", _conversation(reasoning=None), [])
    assert "reasoning_content" not in none.bodies[0]["messages"][2]


def test_request_info_hashes(env):
    server = Server(simple_ok(), simple_ok(), simple_ok())
    client = make(server, completion_window="balanced", extra_body={"top_p": 0.9})
    conv = _conversation()
    a = client.complete("sys prompt", conv, TOOLS).request
    b = client.complete("sys prompt", conv, TOOLS).request
    longer = conv + [Message(role="user", content="next")]
    c = client.complete("sys prompt", longer, TOOLS).request

    assert a.msg_hashes == b.msg_hashes and a.tools_hash == b.tools_hash and a.system_hash == b.system_hash
    assert a.system_hash == hashlib.sha1(b"sys prompt").hexdigest()[:12]
    # One hash per wire message (the 2-result tool message is two wire messages).
    assert a.n_messages == 4 == len(a.msg_hashes) == len(a.msg_chars)
    assert len(set(a.msg_hashes)) == 4
    # Hash = sha1[:12] of the canonical JSON of each message as sent.
    sent = server.bodies[0]["messages"][1:]
    for m, h, n in zip(sent, a.msg_hashes, a.msg_chars):
        canon = json.dumps(m, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert h == hashlib.sha1(canon.encode()).hexdigest()[:12]
        assert n == len(canon)
    tools_canon = json.dumps(server.bodies[0]["tools"], sort_keys=True, separators=(",", ":"))
    assert a.tools_hash == hashlib.sha1(tools_canon.encode()).hexdigest()[:12]
    system_canon = json.dumps({"role": "system", "content": "sys prompt"}, sort_keys=True, separators=(",", ":"))
    assert a.prompt_chars == len(system_canon) + sum(a.msg_chars) + len(tools_canon)
    # Appending a message keeps the prefix hashes.
    assert c.msg_hashes[:4] == a.msg_hashes and c.n_messages == 5
    assert (a.n_tools, a.max_tokens, a.completion_window, a.extra_body) == (1, 8192, "balanced", {"top_p": 0.9})
    assert (a.model, a.provider, a.base_url) == ("deepseek-ai/DeepSeek-V4.1-Flash", "sail", "https://api.sailresearch.com/v1")


def test_non_streaming_path(env):
    body = {
        "id": "chatcmpl-ns",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_content": "thought",
                    "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "id"}'}}],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 2}},
    }
    server = Server(httpx.Response(200, json=body))
    resp = make(server, stream=False).complete("s", USER, TOOLS)
    sent = server.bodies[0]
    assert sent["stream"] is False and "stream_options" not in sent
    assert "accept" not in server.requests[0].headers or "event-stream" not in server.requests[0].headers["accept"]
    assert resp.text == "ok" and resp.reasoning == "thought"
    assert [(t.id, t.name, t.arguments) for t in resp.tool_calls] == [("t1", "bash", {"command": "id"})]
    assert resp.finish_reason == "tool_calls" and resp.response_id == "chatcmpl-ns"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.reasoning_tokens) == (10, 4, 2)
    t = resp.timing
    assert t.streamed is False and t.n_chunks == 0 and t.chunk_offsets_ms == []
    assert t.t_first_chunk is None and t.t_last_chunk is None and t.t_first_content is None
    assert t.t_start <= t.t_request_sent <= t.t_headers <= t.t_end


def test_streaming_request_answered_with_plain_json(env):
    # A server that ignores stream=true: parse the body, do not retry it as truncated.
    body = {"id": "x1", "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "plain"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    server = Server(httpx.Response(200, json=body))
    resp = make(server).complete("s", USER, [])
    assert server.bodies[0]["stream"] is True
    assert resp.text == "plain" and len(resp.attempts) == 1
    assert resp.timing.streamed is False and resp.timing.n_chunks == 0
    assert resp.usage_source == "provider" and resp.usage.prompt_tokens == 5


def test_user_agent_is_versioned():
    import subagents_bench

    assert llm.user_agent() == f"subagents-bench/{subagents_bench.__version__}"


def test_non_streaming_retries_server_error(env):
    ok = {"choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "x"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    server = Server(httpx.Response(502, text="bad gateway"), httpx.Response(200, json=ok))
    resp = make(server, stream=False).complete("s", USER, [])
    assert resp.text == "x" and [a.status for a in resp.attempts] == [502, 200]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_threads_share_one_client(env):
    def handler(request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][-1]["content"]
        return sse_response(
            delta(rid=f"id-{user}", content=f"echo {user}"),
            delta(rid=f"id-{user}", finish="stop"),
            usage_chunk(rid=f"id-{user}", prompt_tokens=len(user), completion_tokens=2),
            split=3,
        )

    client = make_client(ModelConfig(model="m", provider="sail"), transport=httpx.MockTransport(handler), sleep=lambda s: None)
    results: dict[int, object] = {}
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        try:
            barrier.wait()
            for j in range(5):
                tag = f"t{i}-{j}"
                r = client.complete("s", [Message(role="user", content=tag)], [])
                assert r.text == f"echo {tag}" and r.response_id == f"id-{tag}"
                assert r.usage.prompt_tokens == len(tag)
            results[i] = r
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []
    assert sorted(results) == list(range(8))


# ---------------------------------------------------------------------------
# Anthropic client (fake SDK)
# ---------------------------------------------------------------------------


class FakeAnthropicStream:
    def __init__(self, events, final):
        self.events, self.final = events, final
        self.response = SimpleNamespace(headers=httpx.Headers({"request-id": "req_1", "set-cookie": "x=y"}))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final


class FakeMessages:
    def __init__(self, events=None, final=None, error=None):
        self.events, self.final, self.error = events or [], final, error
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return FakeAnthropicStream(self.events, self.final)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.final


def _anthropic_final():
    return SimpleNamespace(
        id="msg_1",
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="thinking", thinking="plan"),
            SimpleNamespace(type="text", text="Let me look."),
            SimpleNamespace(type="tool_use", id="toolu_1", name="bash", input={"command": "ls"}),
        ],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=1000, cache_creation_input_tokens=20),
    )


def _anthropic_events():
    ev = SimpleNamespace
    return [
        ev(type="message_start"),
        ev(type="content_block_start", content_block=ev(type="thinking")),
        ev(type="content_block_delta", delta=ev(type="thinking_delta", thinking="plan")),
        ev(type="content_block_delta", delta=ev(type="signature_delta", signature="sig")),
        ev(type="content_block_delta", delta=ev(type="text_delta", text="Let me look.")),
        ev(type="content_block_start", content_block=ev(type="tool_use", id="toolu_1", name="bash")),
        ev(type="content_block_delta", delta=ev(type="input_json_delta", partial_json='{"command": "ls"}')),
        ev(type="message_stop"),
    ]


def test_anthropic_streaming_timings_and_usage(env):
    fake = FakeMessages(events=_anthropic_events(), final=_anthropic_final())
    cfg = ModelConfig(model="claude-x", provider="anthropic", price_in=3.0, price_out=15.0, price_cache_read=0.3, temperature=0.5)
    client = make_client(cfg, sdk_client=SimpleNamespace(messages=fake))
    resp = client.complete("sys", _conversation(), TOOLS)
    call = fake.calls[0]
    assert call["model"] == "claude-x" and call["temperature"] == 0.5
    assert call["tools"][0]["input_schema"] == TOOLS[0].parameters
    assert call["messages"][1]["content"][0] == {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "ls"}}
    assert call["messages"][2]["content"][1]["is_error"] is True
    assert resp.text == "Let me look." and resp.reasoning == "plan"
    assert [(t.id, t.name, t.arguments) for t in resp.tool_calls] == [("toolu_1", "bash", {"command": "ls"})]
    u = resp.usage
    assert (u.prompt_tokens, u.cached_tokens, u.completion_tokens, u.reasoning_tokens) == (1120, 1000, 50, None)
    assert resp.cost_usd == pytest.approx((120 * 3.0 + 1000 * 0.3 + 50 * 15.0) / 1e6)
    assert resp.finish_reason == "tool_use" and resp.response_id == "msg_1"
    t = resp.timing
    assert t.streamed is True
    assert t.n_chunks == 4  # thinking, text, tool_use start, input_json (signature excluded)
    assert t.t_start <= t.t_request_sent <= t.t_headers <= t.t_first_chunk == t.t_first_reasoning
    assert t.t_first_reasoning <= t.t_first_content <= t.t_first_tool_call <= t.t_last_chunk <= t.t_end
    assert resp.response_headers == {"request-id": "req_1"}
    assert resp.usage_raw["cache_read_input_tokens"] == 1000
    assert resp.request.reasoning_passback is False and resp.request.n_messages == 3


def test_anthropic_non_streaming(env):
    fake = FakeMessages(final=_anthropic_final())
    client = make_client(ModelConfig(model="claude-x", provider="anthropic", stream=False), sdk_client=SimpleNamespace(messages=fake))
    resp = client.complete("sys", USER, [])
    assert resp.timing.streamed is False
    assert resp.timing.t_request_sent is None and resp.timing.t_first_chunk is None and resp.timing.n_chunks == 0
    assert resp.usage.prompt_tokens == 1120 and resp.usage_source == "provider"


class FakeStatusError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize(
    "error,exc_type",
    [
        (FakeStatusError(400, "prompt is too long: 210000 tokens > 200000 maximum"), ContextLengthError),
        (FakeStatusError(400, "Your credit balance is too low"), BillingError),
        (FakeStatusError(529, "overloaded"), LLMError),
    ],
)
def test_anthropic_error_mapping(env, error, exc_type):
    client = make_client(ModelConfig(model="claude-x", provider="anthropic"), sdk_client=SimpleNamespace(messages=FakeMessages(error=error)))
    with pytest.raises(exc_type) as ei:
        client.complete("s", USER, [])
    assert type(ei.value) is exc_type
    assert ei.value.status == error.status_code and len(ei.value.attempts) == 1


def test_anthropic_non_sdk_exception_propagates(env):
    client = make_client(ModelConfig(model="claude-x", provider="anthropic"), sdk_client=SimpleNamespace(messages=FakeMessages(error=KeyError("bug"))))
    with pytest.raises(KeyError):
        client.complete("s", USER, [])


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_prime_credentials_from_home_config(env, tmp_path):
    assert prime_credentials() == (None, None)  # no file, no env
    (tmp_path / ".prime").mkdir()
    cfg_file = tmp_path / ".prime" / "config.json"
    cfg_file.write_text(json.dumps({"api_key": "pk-file", "team_id": "team-file", "user_id": "u"}))
    assert prime_credentials() == ("pk-file", "team-file")
    env.setenv("PRIME_TEAM_ID", "team-env")  # env wins per value
    assert prime_credentials() == ("pk-file", "team-env")
    env.setenv("PRIME_API_KEY", "pk-env")
    assert prime_credentials() == ("pk-env", "team-env")
    env.delenv("PRIME_API_KEY")
    env.delenv("PRIME_TEAM_ID")
    cfg_file.write_text(json.dumps({"api_key": "", "team_id": None}))
    assert prime_credentials() == (None, None)
    cfg_file.write_text("{not json")
    assert prime_credentials() == (None, None)


def test_resolve_api_key(env, tmp_path):
    assert resolve_api_key(ModelConfig(model="m", provider="sail")) == SAIL_KEY
    assert resolve_api_key(ModelConfig(model="m", provider="prime")) is None
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "config.json").write_text(json.dumps({"api_key": "pk-file"}))
    assert resolve_api_key(ModelConfig(model="m", provider="prime")) == "pk-file"
    env.setenv("CUSTOM_KEY", "ck")
    assert resolve_api_key(ModelConfig(model="m", provider="sail", api_key_env="CUSTOM_KEY")) == "ck"
    # Unset custom variable falls back to the provider default.
    assert resolve_api_key(ModelConfig(model="m", provider="sail", api_key_env="UNSET_KEY")) == SAIL_KEY
    assert resolve_api_key(ModelConfig(model="m", provider="vllm", base_url="http://x")) is None


def test_describe_credentials_never_leaks_values(env, tmp_path):
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "config.json").write_text(json.dumps({"api_key": "pk-secret"}))
    d = describe_credentials()
    assert d == {"SAIL_API_KEY": "set", "PRIME_API_KEY": "set", "PRIME_TEAM_ID": "missing"}
    assert describe_credentials(["OPENAI_API_KEY"]) == {"OPENAI_API_KEY": "missing"}
    assert SAIL_KEY not in json.dumps(d) and "pk-secret" not in json.dumps(d)


def test_dotenv_parsing(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "PLAIN=value1",
                "export EXPORTED=value2",
                "SPACED = spaced value  ",
                'DOUBLE="quoted # not a comment"',
                "SINGLE='single $literal \\n'",
                'ESCAPED="line1\\nline2 \\"q\\""',
                "INLINE=abc # trailing comment",
                "HASHED=abc#def",
                "EMPTY=",
                "NOEQUALS",
                "1BAD=x",
                'UNTERMINATED="oops',
                "URL=https://example.com/a?b=c=d",
            ]
        )
    )
    assert parse_dotenv(p) == {
        "PLAIN": "value1",
        "EXPORTED": "value2",
        "SPACED": "spaced value",
        "DOUBLE": "quoted # not a comment",
        "SINGLE": "single $literal \\n",
        "ESCAPED": 'line1\nline2 "q"',
        "INLINE": "abc",
        "HASHED": "abc#def",
        "EMPTY": "",
        "URL": "https://example.com/a?b=c=d",
    }
    assert parse_dotenv(tmp_path / "missing.env") == {}


def test_load_dotenv_does_not_override(tmp_path, monkeypatch, capsys):
    p = tmp_path / ".env"
    p.write_text("SB_TEST_A=from_file\nSB_TEST_B=from_file\n")
    monkeypatch.setenv("SB_TEST_A", "from_env")
    monkeypatch.delenv("SB_TEST_B", raising=False)
    status = load_dotenv(p)
    assert status == {"SB_TEST_A": "kept", "SB_TEST_B": "set"}
    import os

    assert os.environ["SB_TEST_A"] == "from_env"
    assert os.environ["SB_TEST_B"] == "from_file"
    assert load_dotenv(p, override=True) == {"SB_TEST_A": "overridden", "SB_TEST_B": "overridden"}
    assert os.environ["SB_TEST_A"] == "from_file"
    assert load_dotenv(tmp_path / "nope.env") == {}
    out = capsys.readouterr()
    assert "from_file" not in out.out + out.err
    monkeypatch.delenv("SB_TEST_B", raising=False)
