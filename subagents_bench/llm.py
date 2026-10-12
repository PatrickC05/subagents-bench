"""LLM clients: one ``complete()`` interface, full per-call measurements.

The orchestrator, subagents and arbiter share one message/tool vocabulary
(:mod:`subagents_bench.types`, re-exported here) so the same agent loop can
drive any provider. A client's ``complete(system, messages, tools)`` returns an
:class:`LLMResponse` that carries, besides the text and tool calls, everything
needed to recompute cost and latency offline: provider-reported usage (incl.
cached and reasoning tokens), streaming instants and per-chunk arrival
offsets, every HTTP attempt with its backoff, response headers, and hashes of
what was sent.

Backends:

* :class:`OpenAIClient` -- any OpenAI-compatible ``/chat/completions``
  endpoint (Sail, Prime Intellect, vLLM, OpenAI). Talks to it with a shared
  ``httpx.Client`` and parses the SSE stream itself (no OpenAI SDK), so
  timings are measured at the transport. Retries, backoff and error
  classification are implemented here.
* :class:`AnthropicClient` -- native Messages API through the ``anthropic``
  SDK (lazy import), streamed with ``messages.stream`` for timings.

Errors: :class:`BillingError` (never retried; aborts the sweep),
:class:`ContextLengthError` (the runtime may trim and retry) and
:class:`LLMError` (everything else, after retries). Each carries the list of
:class:`Attempt` records.

Clients are thread-safe: one instance is shared by concurrent subagent
threads. All per-call state is local to ``complete()``.
"""

from __future__ import annotations

import copy
import email.utils
import hashlib
import inspect
import json
import logging
import math
import random
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import ModelConfig
from .credentials import (
    DEFAULT_API_KEY_ENVS,
    PRIME_PROVIDERS,
    prime_credentials,
    resolve_api_key,
)
from .types import (
    CHUNK_OFFSETS_CAP,
    AgentStopped,
    Attempt,
    BillingError,
    CallTiming,
    ContextLengthError,
    LLMError,
    LLMResponse,
    Message,
    MessageRole,
    RequestInfo,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

__all__ = [
    "AgentStopped",
    "AnthropicClient",
    "Attempt",
    "BillingError",
    "CallTiming",
    "ContextLengthError",
    "DEFAULT_API_KEY_ENVS",
    "DEFAULT_BASE_URLS",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "Message",
    "MessageRole",
    "OpenAIClient",
    "RequestInfo",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "make_client",
    "resolve_model_config",
]

log = logging.getLogger(__name__)

# Named OpenAI-compatible endpoints: `provider: sail` + a model id is enough.
# An explicit `base_url` in the config always wins.
DEFAULT_BASE_URLS: dict[str, str] = {
    "sail": "https://api.sailresearch.com/v1",
    "prime": "https://api.pinference.ai/api/v1",
    "prime_intellect": "https://api.pinference.ai/api/v1",
    "pinference": "https://api.pinference.ai/api/v1",
}
# Generic OpenAI-compatible providers: the config must give base_url.
NEEDS_BASE_URL = frozenset({"openai", "vllm", "openai_compatible"})
OPENAI_COMPATIBLE = frozenset({*DEFAULT_BASE_URLS, *NEEDS_BASE_URL})
# Self-hosted servers often run without auth; others must have a key.
KEYLESS_PROVIDERS = frozenset({"vllm", "openai_compatible"})

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
BACKOFF_BASE_SEC = 2.0
BACKOFF_MAX_SEC = 120.0
# Upper bound on a server-supplied Retry-After we are willing to sleep.
RETRY_AFTER_MAX_SEC = 600.0
# Error bodies/messages stored on Attempt records are truncated to this.
ERROR_TEXT_MAX = 2000
_BILLING_WORDS = ("balance", "billing", "credit", "quota", "payment")
# A 429 is a rate limit (retried, honoring Retry-After) unless it carries an
# explicit billing marker: OpenAI-style "insufficient_quota" arrives as 429,
# but so do per-minute limits worded "... quota exceeded".
_BILLING_WORDS_429 = ("insufficient_quota", "insufficient quota", "billing", "balance", "payment", "credit")
# Explicit context-overflow phrases, matched after "_" -> " " (so the
# "context_length_exceeded" code matches too). Generic words such as
# "maximum" or "tokens" also appear in unrelated validation errors.
_CONTEXT_PHRASES = (
    "context length",
    "context window",
    "maximum context",
    "too long",
    "reduce the length",
    # vLLM engine: "The decoder prompt (length N) is longer than the maximum
    # model length of M. Make sure that `max_model_len` is no smaller ..."
    "maximum model length",
)
_MAX_TOKENS_WORDS = ("max tokens", "max new tokens", "max completion tokens")
_CONTEXT_STATUS = frozenset({400, 413})


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _harness_version() -> str:
    # Resolved lazily: the package __init__ imports this module before it
    # defines __version__.
    try:
        from . import __version__  # noqa: PLC0415

        return str(__version__)
    except ImportError:
        try:
            from importlib.metadata import version  # noqa: PLC0415

            return version("subagents-bench")
        except Exception:  # noqa: BLE001
            return "unknown"


def user_agent() -> str:
    # Always explicit: Prime's gateway 403s default library user agents.
    return f"subagents-bench/{_harness_version()}"


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing and char counts."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def build_request_info(
    cfg: ModelConfig,
    *,
    system: str,
    system_wire: Any,
    wire_messages: list[dict[str, Any]],
    tool_defs: list[dict[str, Any]],
    reasoning_passback: bool,
) -> RequestInfo:
    """RequestInfo over the messages exactly as they go on the wire.

    ``wire_messages`` excludes the system prompt; one runtime ``tool`` message
    with several results is several wire messages, so ``n_messages`` and the
    hash list count wire messages (what a prefix cache sees).
    """
    ser = [canonical_json(m) for m in wire_messages]
    tools_ser = canonical_json(tool_defs)
    prompt_chars = len(canonical_json(system_wire)) if system_wire else 0
    prompt_chars += sum(len(s) for s in ser)
    if tool_defs:
        prompt_chars += len(tools_ser)
    return RequestInfo(
        model=cfg.model,
        provider=cfg.provider,
        base_url=cfg.base_url,
        completion_window=getattr(cfg, "completion_window", None),
        max_tokens=cfg.max_output_tokens,
        temperature=cfg.temperature,
        extra_body=copy.deepcopy(dict(cfg.extra_body)),
        n_messages=len(wire_messages),
        n_tools=len(tool_defs),
        prompt_chars=prompt_chars,
        system_hash=short_hash(system),
        tools_hash=short_hash(tools_ser),
        msg_hashes=[short_hash(s) for s in ser],
        msg_chars=[len(s) for s in ser],
        reasoning_passback=reasoning_passback,
    )


def _cost(cfg: ModelConfig, usage: Usage) -> float | None:
    """USD from the config price table (prices are per 1M tokens); None when
    the config has no ``price_in``/``price_out`` (unknown, never a made-up 0).

    Cached prompt tokens use ``price_cache_read`` when set, else ``price_in``.
    """
    if cfg.price_in is None or cfg.price_out is None:
        return None
    cache_price = cfg.price_cache_read if cfg.price_cache_read is not None else cfg.price_in
    return (
        usage.uncached_prompt_tokens * cfg.price_in
        + usage.cached_tokens * cache_price
        + usage.completion_tokens * cfg.price_out
    ) / 1e6


def _reported_cost(usage: Any) -> float | None:
    """A provider-reported USD cost from a usage object/dict, verbatim.

    Accepts ``cost`` as a number (Prime sends ``0.0``) or nested under
    ``total_usd``/``usd``/``total``. Never used for ``cost_usd``.
    """
    if usage is None:
        return None

    def get(obj: Any, key: str) -> Any:
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    def num(v: Any) -> float | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return None

    cost = get(usage, "cost")
    if cost is None:
        return None
    direct = num(cost)
    if direct is not None:
        return direct
    for key in ("total_usd", "usd", "total"):
        v = num(get(cost, key))
        if v is not None:
            return v
    return None


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def _parse_usage(raw: dict[str, Any]) -> Usage | None:
    """Usage from an OpenAI-style usage dict; None if token counts are absent."""
    pt = _as_int(raw.get("prompt_tokens"))
    ct = _as_int(raw.get("completion_tokens"))
    if pt is None or ct is None:
        return None
    ptd = raw.get("prompt_tokens_details")
    cached = _as_int(ptd.get("cached_tokens")) if isinstance(ptd, dict) else None
    if cached is None:
        # DeepSeek's native API names it differently.
        cached = _as_int(raw.get("prompt_cache_hit_tokens"))
    ctd = raw.get("completion_tokens_details")
    reasoning = _as_int(ctd.get("reasoning_tokens")) if isinstance(ctd, dict) else None
    return Usage(
        prompt_tokens=pt,
        completion_tokens=ct,
        cached_tokens=cached or 0,
        reasoning_tokens=reasoning,
    )


def _estimate_usage(prompt_chars: int, completion_chars: int) -> Usage:
    """chars/4 fallback when the provider sent no usage (usage_source='estimated')."""
    return Usage(
        prompt_tokens=math.ceil(prompt_chars / 4),
        completion_tokens=math.ceil(completion_chars / 4),
        cached_tokens=0,
        reasoning_tokens=None,
    )


def _clean_headers(headers: Any) -> dict[str, str]:
    """Response headers as a plain dict, cookies removed."""
    if headers is None:
        return {}
    try:
        items = headers.items()
    except AttributeError:
        return {}
    return {str(k).lower(): str(v) for k, v in items if str(k).lower() != "set-cookie"}


def _parse_retry_after(headers: Any) -> float | None:
    """Seconds from ``retry-after-ms`` or ``retry-after`` (seconds or HTTP date)."""
    if headers is None:
        return None
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return max(float(ms) / 1000.0, 0.0)
        except ValueError:
            pass
    ra = headers.get("retry-after")
    if not ra:
        return None
    try:
        return max(float(ra), 0.0)
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(ra)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    return max(dt.timestamp() - time.time(), 0.0)


def _status_error_type(status: int) -> str:
    return {
        408: "Timeout",
        409: "Conflict",
        425: "TooEarly",
        429: "RateLimited",
        529: "Overloaded",
    }.get(status, "ServerError" if 500 <= status < 600 else f"HTTP{status}")


def _is_billing(status: int | None, low: str) -> bool:
    if status == 402:
        return True
    if status == 429:
        return any(w in low for w in _BILLING_WORDS_429)
    return status is not None and 400 <= status < 500 and any(w in low for w in _BILLING_WORDS)


def _is_context_overflow(status: int | None, low: str) -> bool:
    if status not in _CONTEXT_STATUS:
        return False
    words = low.replace("_", " ")
    if any(p in words for p in _CONTEXT_PHRASES):
        return True
    # "max_tokens + input tokens must be <= ..." style limits.
    return "input" in words and any(w in words for w in _MAX_TOKENS_WORDS)


def _classify(status: int | None, text: str) -> tuple[type[LLMError] | None, str]:
    """(exception class to raise now, or None to retry; error_type)."""
    low = text.lower()
    if _is_billing(status, low):
        return BillingError, "Billing"
    if _is_context_overflow(status, low):
        return ContextLengthError, "ContextLength"
    if status is None or status in RETRYABLE_STATUS:
        return None, "StreamError" if status is None else _status_error_type(status)
    return LLMError, f"HTTP{status}"


def _truncate(text: str, n: int = ERROR_TEXT_MAX) -> str:
    return text if len(text) <= n else text[:n] + f"...[{len(text) - n} more chars]"


class _AttemptFailure(Exception):
    """One failed HTTP attempt. ``exc_class`` None means retryable."""

    def __init__(
        self,
        *,
        status: int | None,
        error_type: str,
        message: str,
        exc_class: type[LLMError] | None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = _truncate(message)
        self.exc_class = exc_class
        self.retry_after = retry_after


def _parse_arguments(text: str) -> dict[str, Any]:
    """Tool-call arguments JSON -> dict; anything else -> ``{'_raw': text}``."""
    if not text.strip():
        return {}
    try:
        val = json.loads(text)
    except ValueError:
        return {"_raw": text}
    if isinstance(val, str):
        # Some models double-encode the arguments object.
        try:
            inner = json.loads(val)
        except ValueError:
            inner = None
        if isinstance(inner, dict):
            return inner
    return val if isinstance(val, dict) else {"_raw": text}


def _is_json_object(text: str) -> bool:
    """True if ``text`` is one complete JSON object (a whole arguments string)."""
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Client interface and factory
# ---------------------------------------------------------------------------


class LLMClient:
    """Base class: ``.cfg`` plus ``complete(system, messages, tools)``."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        raise NotImplementedError

    def close(self) -> None:
        """Release network resources (optional)."""


def resolve_model_config(cfg: ModelConfig) -> ModelConfig:
    """A shallow copy of ``cfg`` with the provider defaults filled in
    (``api_key_env``; ``base_url`` for OpenAI-compatible providers).

    Raises ValueError for an unknown provider or an OpenAI-compatible provider
    without a default URL and no ``base_url``. The caller's ``cfg`` is not
    modified.
    """
    provider = cfg.provider
    resolved = copy.copy(cfg)
    if resolved.api_key_env is None and provider in DEFAULT_API_KEY_ENVS:
        resolved.api_key_env = DEFAULT_API_KEY_ENVS[provider]
    if provider == "anthropic":
        return resolved
    if provider in OPENAI_COMPATIBLE:
        if not resolved.base_url:
            resolved.base_url = DEFAULT_BASE_URLS.get(provider)
        if not resolved.base_url:
            raise ValueError(f"provider {provider!r} needs an explicit base_url in the model config")
        return resolved
    raise ValueError(f"unknown provider: {provider!r}")


def make_client(cfg: ModelConfig, **client_kwargs: Any) -> LLMClient:
    """Build the client for ``cfg.provider``, filling provider defaults.

    The caller's ``cfg`` is not modified: the client gets the copy from
    :func:`resolve_model_config` (``client.cfg``). ``client_kwargs`` go to the
    client constructor (tests inject an ``http_client``/``transport``,
    ``sleep`` or ``rng``).
    """
    resolved = resolve_model_config(cfg)
    if resolved.provider == "anthropic":
        return AnthropicClient(resolved, **client_kwargs)
    return OpenAIClient(resolved, **client_kwargs)


# ---------------------------------------------------------------------------
# OpenAI-compatible (Sail, Prime, vLLM, OpenAI)
# ---------------------------------------------------------------------------


@dataclass
class _ToolSlot:
    id: str | None = None
    name: str = ""
    args: list[str] = field(default_factory=list)


@dataclass
class _Accumulator:
    """Assembles one response from stream chunks (or one non-stream body)."""

    t_sent: float
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tools: dict[int, _ToolSlot] = field(default_factory=dict)
    # tool-call id -> the slot index it first arrived under.
    id_index: dict[str, int] = field(default_factory=dict)
    # Deltas dropped because they repeated an already complete call.
    tool_call_repeats_dropped: int = 0
    # Calls dropped / re-identified in tool_calls() (same id twice).
    tool_call_duplicates_dropped: int = 0
    tool_call_ids_renamed: int = 0
    finish_reason: str | None = None
    usage_raw: dict[str, Any] | None = None
    response_id: str | None = None
    done: bool = False
    n_chunks: int = 0
    offsets_ms: list[float] = field(default_factory=list)
    t_first_chunk: float | None = None
    t_first_reasoning: float | None = None
    t_first_content: float | None = None
    t_first_tool_call: float | None = None
    t_last_chunk: float | None = None

    def feed_chunk(self, chunk: dict[str, Any], now: float) -> None:
        """One parsed SSE ``data:`` object."""
        self._meta(chunk)
        carried = False
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            kinds = self.add_delta(choice.get("delta") or {})
            carried = carried or bool(kinds)
            if "reasoning" in kinds and self.t_first_reasoning is None:
                self.t_first_reasoning = now
            if "content" in kinds and self.t_first_content is None:
                self.t_first_content = now
            if "tool" in kinds and self.t_first_tool_call is None:
                self.t_first_tool_call = now
            if choice.get("finish_reason"):
                self.finish_reason = str(choice["finish_reason"])
        if carried:
            # Only delta-carrying chunks count (not role-only, finish or usage chunks).
            self.n_chunks += 1
            if self.t_first_chunk is None:
                self.t_first_chunk = now
            self.t_last_chunk = now
            if len(self.offsets_ms) < CHUNK_OFFSETS_CAP:
                self.offsets_ms.append(round((now - self.t_sent) * 1000.0, 3))

    def feed_body(self, body: dict[str, Any]) -> None:
        """A complete non-streaming response body."""
        self._meta(body)
        self.done = True
        for choice in body.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            self.add_delta(choice.get("message") or {})
            if choice.get("finish_reason"):
                self.finish_reason = str(choice["finish_reason"])

    def _meta(self, obj: dict[str, Any]) -> None:
        if self.response_id is None and obj.get("id"):
            self.response_id = str(obj["id"])
        usage = obj.get("usage")
        if isinstance(usage, dict) and usage:
            self.usage_raw = usage

    def add_delta(self, delta: dict[str, Any]) -> set[str]:
        """Merge a delta (or full message); return which kinds it carried."""
        kinds: set[str] = set()
        # vLLM-style servers use `reasoning`, DeepSeek-style `reasoning_content`;
        # take one so a server sending both is not double-counted.
        r = delta.get("reasoning_content")
        if not (isinstance(r, str) and r):
            r = delta.get("reasoning")
        if isinstance(r, str) and r:
            self.reasoning.append(r)
            kinds.add("reasoning")
        c = delta.get("content")
        if isinstance(c, str) and c:
            self.text.append(c)
            kinds.add("content")
        tcs = delta.get("tool_calls")
        if isinstance(tcs, list):
            for pos, tc in enumerate(tcs):
                if not isinstance(tc, dict):
                    continue
                kinds.add("tool")
                idx = tc.get("index", pos)
                idx = idx if isinstance(idx, int) else pos
                fn = tc.get("function") or {}
                name = fn.get("name")
                args = fn.get("arguments")
                if args is not None and not isinstance(args, str):
                    args = json.dumps(args)
                tc_id = str(tc["id"]) if tc.get("id") else None
                # Sail re-sends each finished call in full under another
                # index with the same id (seen 2026-10-11 for Gemma-4-31B:
                # index 3, then index 0). Route by id, and drop a delta that
                # repeats the slot's complete name and arguments.
                if tc_id and tc_id in self.id_index and self.id_index[tc_id] != idx:
                    owner = self.tools[self.id_index[tc_id]]
                    have = "".join(owner.args)
                    if (not name or name == owner.name) and (
                        not args or args == have or (_is_json_object(args) and _is_json_object(have))
                    ):
                        self.tool_call_repeats_dropped += 1
                        continue
                    idx = self.id_index[tc_id]
                slot = self.tools.setdefault(idx, _ToolSlot())
                if tc_id and not slot.id:
                    slot.id = tc_id
                    self.id_index.setdefault(tc_id, idx)
                # Names normally arrive whole once; tolerate fragments, and
                # servers that repeat the full name on every fragment.
                if isinstance(name, str) and name and name != slot.name:
                    slot.name += name
                if args:
                    slot.args.append(args)
        return kinds

    def tool_calls(self) -> list[ToolCall]:
        """The assembled calls, in index order. Ids are unique: an exact
        duplicate (same id, name and arguments) is dropped, and a different
        call reusing an id gets a fresh one, so the conversation sent back
        to the provider never carries two tool calls with one id."""
        out: list[ToolCall] = []
        seen: dict[str, tuple[str, str]] = {}
        for idx in sorted(self.tools):
            slot = self.tools[idx]
            raw_args = "".join(slot.args)
            if not slot.name and not raw_args:
                continue
            call_id = slot.id or f"call_{uuid.uuid4().hex[:24]}"
            if call_id in seen:
                if seen[call_id] == (slot.name, raw_args):
                    self.tool_call_duplicates_dropped += 1
                    continue
                call_id = f"{call_id}_dup{uuid.uuid4().hex[:8]}"
                self.tool_call_ids_renamed += 1
            seen[call_id] = (slot.name, raw_args)
            out.append(ToolCall(id=call_id, name=slot.name, arguments=_parse_arguments(raw_args)))
        return out

    @property
    def stream_repairs(self) -> int:
        """How many tool-call repairs this response needed (0 normally)."""
        return self.tool_call_repeats_dropped + self.tool_call_duplicates_dropped + self.tool_call_ids_renamed

    def completion_chars(self) -> int:
        n = sum(len(s) for s in self.text) + sum(len(s) for s in self.reasoning)
        for slot in self.tools.values():
            n += len(slot.name) + sum(len(a) for a in slot.args)
        return n


@dataclass
class _AttemptOK:
    acc: _Accumulator
    status: int
    t_headers: float
    t_end: float
    headers: dict[str, str]
    streamed: bool


def _iter_sse_data(lines: Iterable[str]) -> Iterator[str]:
    """Yield the payload of each SSE ``data:`` line.

    OpenAI-compatible servers put one JSON object per ``data:`` line; treating
    each line as an event also tolerates servers that omit the blank-line
    event separator. Comments (``:``) and other fields are skipped.
    """
    for line in lines:
        if not line or line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if name != "data":
            continue
        yield value[1:] if value.startswith(" ") else value


def _stream_error(chunk: dict[str, Any]) -> _AttemptFailure | None:
    """An in-stream error object (vLLM/gateways send these mid-stream)."""
    err = chunk.get("error")
    if err is None and chunk.get("object") == "error":
        err = chunk
    if err is None:
        return None
    if isinstance(err, dict):
        msg = str(err.get("message") or err)
        code = _as_int(err.get("code")) or _as_int(err.get("status"))
    else:
        msg, code = str(err), None
    status = code if code is not None and 100 <= code < 600 else None
    exc, etype = _classify(status, msg)
    return _AttemptFailure(
        status=status,
        error_type="StreamError" if exc is None else etype,
        message=msg,
        exc_class=exc,
    )


class OpenAIClient(LLMClient):
    """OpenAI-compatible chat/completions over httpx with SSE streaming.

    Timing notes: ``t_request_sent`` is taken just before the request is handed
    to httpx, so on a fresh (unpooled) connection it includes connect + TLS;
    ``t_headers`` is when the status line and headers arrived; chunk instants
    are taken as each SSE ``data:`` line is read. A stream that delivers the
    whole reply in one chunk has ``t_first_chunk == t_last_chunk``.

    Constructor hooks (for tests): ``http_client`` (a ready ``httpx.Client``)
    or ``transport`` (e.g. ``httpx.MockTransport``), ``sleep`` (called with the
    backoff seconds) and ``rng`` (``random.Random`` for jitter).
    """

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
    ):
        super().__init__(cfg)
        if not cfg.base_url:
            raise ValueError("OpenAIClient needs cfg.base_url (use make_client for provider defaults)")
        timeout_sec = float(getattr(cfg, "request_timeout_sec", 600.0))
        self._timeout = httpx.Timeout(timeout_sec, connect=min(30.0, timeout_sec))
        self._owns_http = http_client is None
        # One pooled client shared by all threads (httpx.Client is thread-safe).
        self._http = http_client or httpx.Client(timeout=self._timeout, transport=transport)
        self._sleep = sleep or time.sleep
        self._rng = rng or random.Random()
        self._url = cfg.base_url.rstrip("/") + "/chat/completions"

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    # ---- request building --------------------------------------------------

    def _to_openai(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Runtime messages -> wire messages (system prompt excluded)."""
        passback = bool(getattr(self.cfg, "reasoning_passback", True))
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in m.tool_calls
                    ]
                if passback and m.reasoning:
                    msg["reasoning_content"] = m.reasoning
                out.append(msg)
            elif m.role == "tool":
                for tr in m.tool_results:
                    out.append({"role": "tool", "tool_call_id": tr.call_id, "content": tr.content})
            elif m.role == "system":
                out.append({"role": "system", "content": m.content or ""})
            else:
                out.append({"role": "user", "content": m.content or ""})
        return out

    @staticmethod
    def _tool_defs(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]

    def _body(self, system_msg: dict[str, Any] | None, wire: list[dict[str, Any]], tool_defs: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self.cfg
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": ([system_msg] if system_msg else []) + wire,
            "max_tokens": cfg.max_output_tokens,
            "temperature": cfg.temperature,
        }
        if tool_defs:
            body["tools"] = tool_defs
        body.update(copy.deepcopy(dict(cfg.extra_body)))
        if getattr(cfg, "stream", True):
            body["stream"] = True
            opts = body.get("stream_options")
            body["stream_options"] = {**(opts if isinstance(opts, dict) else {}), "include_usage": True}
        else:
            body["stream"] = False
            body.pop("stream_options", None)
        return body

    def _headers(self) -> httpx.Headers:
        cfg = self.cfg
        h = httpx.Headers({"Content-Type": "application/json", "User-Agent": user_agent()})
        if getattr(cfg, "stream", True):
            h["Accept"] = "text/event-stream"
        key = resolve_api_key(cfg)
        if key:
            h["Authorization"] = f"Bearer {key}"
        elif cfg.provider not in KEYLESS_PROVIDERS:
            where = cfg.api_key_env or DEFAULT_API_KEY_ENVS.get(cfg.provider, "an API key variable")
            hint = " (or log in with the prime CLI)" if cfg.provider in PRIME_PROVIDERS else ""
            raise LLMError(f"no API key for provider {cfg.provider!r}: set {where}{hint}")
        window = getattr(cfg, "completion_window", None)
        if window:
            h["X-Sail-Completion-Window"] = str(window)
        if cfg.provider in PRIME_PROVIDERS:
            team = prime_credentials()[1]
            if team:
                h["X-Prime-Team-ID"] = team
        for k, v in (getattr(cfg, "extra_headers", None) or {}).items():
            h[str(k)] = str(v)  # replaces case-insensitively
        return h

    # ---- one HTTP attempt ----------------------------------------------------

    def _attempt(self, body: dict[str, Any], headers: httpx.Headers) -> _AttemptOK:
        streaming = bool(body.get("stream"))
        t_sent = time.time()
        acc = _Accumulator(t_sent=t_sent)
        status: int | None = None
        try:
            with self._http.stream("POST", self._url, json=body, headers=headers, timeout=self._timeout) as resp:
                t_headers = time.time()
                status = resp.status_code
                if status >= 300:
                    text = resp.read().decode("utf-8", errors="replace")
                    exc, etype = _classify(status, text)
                    raise _AttemptFailure(
                        status=status,
                        error_type=etype,
                        message=text or resp.reason_phrase,
                        exc_class=exc,
                        retry_after=_parse_retry_after(resp.headers),
                    )
                # A server that ignores stream=true answers with a plain JSON body.
                use_sse = streaming and not resp.headers.get("content-type", "").lower().startswith("application/json")
                if use_sse:
                    for data in _iter_sse_data(resp.iter_lines()):
                        if data.strip() == "[DONE]":
                            acc.done = True
                            break
                        now = time.time()
                        try:
                            chunk = json.loads(data)
                        except ValueError:
                            raise _AttemptFailure(
                                status=status,
                                error_type="StreamParseError",
                                message=f"unparseable SSE data: {data[:200]!r}",
                                exc_class=None,
                            ) from None
                        if not isinstance(chunk, dict):
                            continue
                        failure = _stream_error(chunk)
                        if failure is not None:
                            raise failure
                        acc.feed_chunk(chunk, now)
                else:
                    raw = resp.read()
                    try:
                        parsed = json.loads(raw)
                    except ValueError:
                        raise _AttemptFailure(
                            status=status,
                            error_type="BadResponse",
                            message=f"non-JSON body: {raw[:200]!r}",
                            exc_class=None,
                        ) from None
                    failure = _stream_error(parsed) if isinstance(parsed, dict) else None
                    if failure is not None:
                        raise failure
                    if not isinstance(parsed, dict) or not parsed.get("choices"):
                        raise _AttemptFailure(
                            status=status,
                            error_type="BadResponse",
                            message=f"response without choices: {raw[:200]!r}",
                            exc_class=None,
                        )
                    acc.feed_body(parsed)
                t_end = time.time()
                headers_out = _clean_headers(resp.headers)
        except httpx.TransportError as e:
            # Connect/read timeouts, resets, protocol errors (incl. mid-stream).
            raise _AttemptFailure(
                status=status,
                error_type=type(e).__name__,
                message=str(e) or repr(e),
                exc_class=None,
            ) from None
        except httpx.HTTPError as e:  # e.g. decoding errors: not worth retrying
            raise _AttemptFailure(
                status=status, error_type=type(e).__name__, message=str(e) or repr(e), exc_class=LLMError
            ) from None
        if not acc.done and acc.finish_reason is None:
            raise _AttemptFailure(
                status=status,
                error_type="TruncatedStream",
                message=f"stream ended without finish_reason or [DONE] after {acc.n_chunks} delta chunks",
                exc_class=None,
            )
        return _AttemptOK(
            acc=acc, status=status, t_headers=t_headers, t_end=t_end, headers=headers_out, streamed=use_sse
        )

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        """Full-jitter exponential backoff, at least the server's Retry-After."""
        ceiling = min(BACKOFF_MAX_SEC, BACKOFF_BASE_SEC * 2 ** (attempt - 1))
        delay = self._rng.uniform(0.0, ceiling)
        if retry_after is not None:
            delay = max(delay, min(retry_after, RETRY_AFTER_MAX_SEC))
        return delay

    # ---- the call -------------------------------------------------------------

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        cfg = self.cfg
        system_msg = {"role": "system", "content": system} if system else None
        wire = self._to_openai(messages)
        tool_defs = self._tool_defs(tools)
        info = build_request_info(
            cfg,
            system=system,
            system_wire=system_msg,
            wire_messages=wire,
            tool_defs=tool_defs,
            reasoning_passback=bool(getattr(cfg, "reasoning_passback", True)),
        )
        body = self._body(system_msg, wire, tool_defs)
        try:
            headers = self._headers()
        except LLMError as e:  # no API key: nothing was sent
            e.request = info
            raise
        max_retries = max(int(getattr(cfg, "max_retries", 8)), 0)

        attempts: list[Attempt] = []
        t_start = time.time()
        n = 0
        while True:
            n += 1
            a_start = time.time()
            try:
                ok = self._attempt(body, headers)
            except _AttemptFailure as f:
                att = Attempt(
                    attempt=n,
                    t_start=a_start,
                    t_end=time.time(),
                    status=f.status,
                    error_type=f.error_type,
                    error=f.message,
                    retry_after_sec=f.retry_after,
                )
                attempts.append(att)
                where = f"{cfg.provider} {cfg.model}"
                status_txt = f"HTTP {f.status}" if f.status is not None else "no status"
                if f.exc_class is not None:
                    raise f.exc_class(
                        f"{where}: {f.error_type} ({status_txt}): {_truncate(f.message, 500)}",
                        attempts=attempts,
                        status=f.status,
                        request=info,
                    ) from None
                if n > max_retries:
                    raise LLMError(
                        f"{where}: giving up after {n} attempts; last {f.error_type} ({status_txt}): "
                        f"{_truncate(f.message, 500)}",
                        attempts=attempts,
                        status=f.status,
                        request=info,
                    ) from None
                att.backoff_sec = self._backoff(n, f.retry_after)
                log.warning(
                    "%s attempt %d failed (%s, %s); retrying in %.1fs",
                    where, n, f.error_type, status_txt, att.backoff_sec,
                )
                self._sleep(att.backoff_sec)
                continue
            attempts.append(Attempt(attempt=n, t_start=a_start, t_end=ok.t_end, status=ok.status))
            break
        t_end = time.time()

        acc = ok.acc
        tool_calls = acc.tool_calls()
        if acc.stream_repairs:
            log.warning(
                "%s %s: repaired streamed tool calls (%d repeated deltas, %d duplicate calls dropped, %d ids renamed)",
                cfg.provider, cfg.model, acc.tool_call_repeats_dropped, acc.tool_call_duplicates_dropped, acc.tool_call_ids_renamed,
            )
        text = "".join(acc.text)
        reasoning = "".join(acc.reasoning)
        usage = _parse_usage(acc.usage_raw) if acc.usage_raw else None
        usage_source = "provider"
        if usage is None:
            usage_source = "estimated"
            usage = _estimate_usage(info.prompt_chars, acc.completion_chars())
        timing = CallTiming(
            t_start=t_start,
            t_end=t_end,
            t_request_sent=acc.t_sent,
            t_headers=ok.t_headers,
            t_first_chunk=acc.t_first_chunk,
            t_first_reasoning=acc.t_first_reasoning,
            t_first_content=acc.t_first_content,
            t_first_tool_call=acc.t_first_tool_call,
            t_last_chunk=acc.t_last_chunk,
            n_chunks=acc.n_chunks,
            chunk_offsets_ms=acc.offsets_ms,
            streamed=ok.streamed,
        )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=acc.finish_reason or "unknown",
            cost_usd=_cost(cfg, usage),
            timing=timing,
            request=info,
            attempts=attempts,
            reasoning=reasoning,
            usage_source=usage_source,
            usage_raw=acc.usage_raw,
            provider_reported_cost=_reported_cost(acc.usage_raw),
            response_headers=ok.headers,
            response_id=acc.response_id,
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _accepts_kwarg(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class AnthropicClient(LLMClient):
    """Native Messages API via the ``anthropic`` SDK (lazy import).

    The SDK retries internally (``max_retries`` from the config), so only one
    :class:`Attempt` is recorded per call. Reasoning is not passed back:
    Anthropic thinking blocks need their signatures, which the runtime's
    :class:`Message` does not carry. ``sdk_client`` injects a fake for tests.
    """

    def __init__(self, cfg: ModelConfig, *, sdk_client: Any = None):
        super().__init__(cfg)
        self._client = sdk_client
        self._lock = threading.Lock()

    def _lazy(self) -> Any:
        with self._lock:
            if self._client is None:
                import anthropic  # noqa: PLC0415

                self._client = anthropic.Anthropic(
                    api_key=resolve_api_key(self.cfg),
                    base_url=self.cfg.base_url,
                    max_retries=int(getattr(self.cfg, "max_retries", 8)),
                    timeout=float(getattr(self.cfg, "request_timeout_sec", 600.0)),
                    default_headers=dict(getattr(self.cfg, "extra_headers", None) or {}) or None,
                )
            return self._client

    @staticmethod
    def _to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": blocks})
            elif m.role == "tool":
                blocks = [
                    {"type": "tool_result", "tool_use_id": tr.call_id, "content": tr.content, "is_error": tr.is_error}
                    for tr in m.tool_results
                ]
                out.append({"role": "user", "content": blocks})
            else:  # user (a mid-conversation "system" message is sent as user text)
                out.append({"role": "user", "content": m.content or ""})
        return out

    def _kwargs(self, fn: Callable[..., Any], system_blocks: Any, wire: Any, tool_defs: Any) -> dict[str, Any]:
        cfg = self.cfg
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_output_tokens,
            "system": system_blocks,
            "messages": wire,
        }
        if tool_defs:
            kwargs["tools"] = tool_defs
        # Params the SDK method does not name (temperature in recent SDKs,
        # provider extras) go through its `extra_body`.
        extra: dict[str, Any] = {}
        for k, v in {"temperature": cfg.temperature, **dict(cfg.extra_body)}.items():
            if _accepts_kwarg(fn, k) and k != "extra_body":
                kwargs[k] = v
            else:
                extra[k] = v
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        cfg = self.cfg
        client = self._lazy()
        wire = self._to_anthropic(messages)
        tool_defs = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        info = build_request_info(
            cfg,
            system=system,
            system_wire=system_blocks,
            wire_messages=wire,
            tool_defs=tool_defs,
            reasoning_passback=False,
        )
        streaming = bool(getattr(cfg, "stream", True))
        fn = client.messages.stream if streaming else client.messages.create
        kwargs = self._kwargs(fn, system_blocks, wire, tool_defs)

        t_start = time.time()
        t_sent = t_start
        t_headers = None
        first: dict[str, float] = {}
        offsets: list[float] = []
        n_chunks = 0
        t_last = None
        headers: dict[str, str] = {}
        try:
            if streaming:
                with fn(**kwargs) as stream:
                    t_headers = time.time()
                    for ev in stream:
                        now = time.time()
                        kind = None
                        etype = _get(ev, "type")
                        if etype == "content_block_start" and _get(_get(ev, "content_block"), "type") == "tool_use":
                            kind = "tool"
                        elif etype == "content_block_delta":
                            dtype = _get(_get(ev, "delta"), "type")
                            kind = {"text_delta": "content", "thinking_delta": "reasoning", "input_json_delta": "tool"}.get(dtype)
                        if kind is None:
                            continue
                        n_chunks += 1
                        first.setdefault("chunk", now)
                        first.setdefault(kind, now)
                        t_last = now
                        if len(offsets) < CHUNK_OFFSETS_CAP:
                            offsets.append(round((now - t_sent) * 1000.0, 3))
                    final = stream.get_final_message()
                    headers = _clean_headers(_get(_get(stream, "response"), "headers"))
            else:
                final = fn(**kwargs)
        except Exception as e:  # noqa: BLE001
            # SDK errors (anthropic.APIStatusError / APIConnectionError, or any
            # error object carrying an HTTP status); others are bugs: re-raise.
            if not (type(e).__module__.startswith("anthropic") or hasattr(e, "status_code")):
                raise
            status = _as_int(getattr(e, "status_code", None))
            text = f"{e} {getattr(e, 'body', '') or ''}"
            exc, etype = _classify(status, text)
            att = Attempt(
                attempt=1, t_start=t_start, t_end=time.time(), status=status,
                error_type=etype if exc is not None else type(e).__name__, error=_truncate(str(e)),
            )
            raise (exc or LLMError)(f"anthropic {cfg.model}: {type(e).__name__}: {_truncate(str(e), 500)}", attempts=[att], status=status) from None
        t_end = time.time()

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in _get(final, "content") or []:
            btype = _get(block, "type")
            if btype == "text":
                text_parts.append(_get(block, "text") or "")
            elif btype == "thinking":
                reasoning_parts.append(_get(block, "thinking") or "")
            elif btype == "tool_use":
                args = _get(block, "input")
                tool_calls.append(
                    ToolCall(id=_get(block, "id"), name=_get(block, "name"), arguments=args if isinstance(args, dict) else {"_raw": args})
                )

        u = _get(final, "usage")
        usage_raw: dict[str, Any] | None = None
        if u is not None:
            usage_raw = u.model_dump() if hasattr(u, "model_dump") else (dict(u) if isinstance(u, dict) else dict(vars(u)))
        inp = _as_int(_get(u, "input_tokens")) if u is not None else None
        out = _as_int(_get(u, "output_tokens")) if u is not None else None
        if inp is not None and out is not None:
            cache_read = _as_int(_get(u, "cache_read_input_tokens")) or 0
            cache_create = _as_int(_get(u, "cache_creation_input_tokens")) or 0
            usage = Usage(
                prompt_tokens=inp + cache_read + cache_create,
                completion_tokens=out,
                cached_tokens=cache_read,
                reasoning_tokens=None,
            )
            usage_source = "provider"
        else:
            completion_chars = sum(len(s) for s in text_parts + reasoning_parts) + sum(
                len(tc.name) + len(json.dumps(tc.arguments)) for tc in tool_calls
            )
            usage = _estimate_usage(info.prompt_chars, completion_chars)
            usage_source = "estimated"

        timing = CallTiming(
            t_start=t_start,
            t_end=t_end,
            t_request_sent=t_sent if streaming else None,
            t_headers=t_headers,
            t_first_chunk=first.get("chunk"),
            t_first_reasoning=first.get("reasoning"),
            t_first_content=first.get("content"),
            t_first_tool_call=first.get("tool"),
            t_last_chunk=t_last,
            n_chunks=n_chunks,
            chunk_offsets_ms=offsets,
            streamed=streaming,
        )
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=_get(final, "stop_reason") or "unknown",
            cost_usd=_cost(cfg, usage),
            timing=timing,
            request=info,
            attempts=[Attempt(attempt=1, t_start=t_start, t_end=t_end, status=200)],
            reasoning="".join(reasoning_parts),
            usage_source=usage_source,
            usage_raw=usage_raw,
            provider_reported_cost=None,
            response_headers=headers,
            response_id=_get(final, "id"),
            raw=final,
        )
