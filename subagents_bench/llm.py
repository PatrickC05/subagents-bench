"""Chat-completions client for Prime Inference and Sail (OpenAI-compatible, streamed SSE) and the
message types."""

from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .config import ModelConfig

# provider -> (chat-completions URL, API key variable). Prime Inference's optional PRIME_TEAM_ID bills a
# team (X-Prime-Team-ID) instead of the key owner's account.
PROVIDERS = {
    "prime": ("https://api.pinference.ai/api/v1/chat/completions", "PRIME_API_KEY"),
    "sail": ("https://api.sailresearch.com/v1/chat/completions", "SAIL_API_KEY"),
}
RETRYABLE_STATUS = frozenset({408, 409, 425, 429})
BACKOFF_BASE_SEC = 2.0
BACKOFF_MAX_SEC = 15.0
_BILLING_WORDS = ("balance", "billing", "credit", "payment", "insufficient_funds")
_BILLING_WORDS_429 = ("insufficient_quota", "insufficient_funds", "billing")
# Matched after "_" -> " " (so "context_length_exceeded" counts); vLLM says "maximum model length".
_CONTEXT_PHRASES = ("context length", "context window", "maximum context", "too long", "reduce the length",
                    "maximum model length")


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    reasoning: str | None = None  # sent back as reasoning_content on later turns, unless reasoning_details
    # Prime's gateway models: the reply's reasoning blocks (signed thinking, encrypted reasoning, Gemini's
    # thought signatures), sent back instead of reasoning_content, which those models ignore
    reasoning_details: list[dict[str, Any]] | None = None


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str
    reasoning: str = ""
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0  # includes cached_tokens
    cached_tokens: int = 0
    completion_tokens: int = 0  # includes reasoning_tokens
    reasoning_tokens: int | None = None
    t_start: float = 0.0
    t_first_chunk: float | None = None  # first chunk carrying content, reasoning or a tool call
    t_end: float = 0.0
    n_attempts: int = 1
    backoff_sec: float = 0.0
    stream_repairs: int = 0
    one_call_retry: bool = False  # answered by a retry that asked for exactly one tool call (ONE_CALL_NOTE)
    n_chunks: int = 0  # stream chunks carrying content, reasoning or a tool call (Sail: about 16 tokens each)


class LLMError(Exception):
    """A failed call; ``n_attempts``/``backoff_sec`` describe its retries."""

    def __init__(self, message: str, *, status: int | None = None, n_attempts: int = 1, backoff_sec: float = 0.0):
        super().__init__(message)
        self.status = status
        self.n_attempts = n_attempts
        self.backoff_sec = backoff_sec


class ProviderError(LLMError):
    """The provider could not serve the call: retries exhausted or a non-retryable error."""


class BillingError(LLMError):
    """402, or a 4xx about balance/billing/credit/payment. Never retried."""


class ContextLengthError(LLMError):
    """400/413 saying the prompt exceeds the model's context. Never retried."""


class CallInterrupted(Exception):
    """A call aborted through its :class:`CallCancel` (an interactive interrupt). Not an LLMError:
    nothing failed, so it never stops the episode."""


class CallCancel:
    """Lets another thread abort one streaming call: :meth:`cancel` makes the call stop reading at
    its next SSE line, close the HTTP stream and raise :class:`CallInterrupted` (also from a retry's
    backoff). ``acc`` is the attempt's partial reply, read by the recorder for the aborted call."""

    def __init__(self) -> None:
        self.t_start = time.time()
        self.acc: _Accumulator | None = None
        self.n_attempts = 0
        self.backoff_sec = 0.0
        self.t_first: float | None = None
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, sec: float) -> bool:
        return self._event.wait(sec)

    @property
    def in_tool_call(self) -> bool:
        """The reply has started a tool call (its reasoning is over)."""
        acc = self.acc
        return acc is not None and any(s.name or s.args for s in list(acc.slots.values()))

    def partial(self) -> dict[str, Any]:
        """What the aborted attempt streamed: data chunks (Sail: about 16 tokens each), characters, usage."""
        acc = self.acc
        if acc is None:
            return {"chunks": 0, "reasoning_chunks": 0, "text_chars": 0, "reasoning_chars": 0, "usage": {}}
        return {"chunks": acc.n_chunks, "reasoning_chunks": acc.n_reasoning_chunks,
                "text_chars": sum(len(x) for x in list(acc.text)),
                "reasoning_chars": sum(len(x) for x in list(acc.reasoning)), "usage": dict(acc.usage)}


class _Retry(Exception):
    def __init__(self, message: str, status: int | None = None, after_tool_call: bool = False):
        super().__init__(message)
        self.status = status
        self.after_tool_call = after_tool_call  # a stream error once a tool call had started streaming


# Sail's tool-call parsing for Gemma-4-31B fails when a reply holds more than one tool call: it
# streams the first (sometimes under index 3) and then an `upstream_error` "chat completion failed"
# chunk (2026-10-14: 3 of the 4 two-call replies in the interactive sweeps, every one of them again on
# replay; 415 one-call replies never failed). The same request fails the same way on every retry, so
# a retry after a stream error that came once a tool call had started adds this note to the system
# prompt, and the model answers with one call; that call records ``one_call_retry``.
ONE_CALL_NOTE = ("\n\n[Note: your previous reply to this request could not be delivered because it held more than "
                 "one tool call. Reply with exactly one tool call.]")


def _classify(status: int, text: str) -> LLMError | None:
    """The error to raise for a non-2xx answer, or None to retry it."""
    low = text.lower()
    msg = f"HTTP {status}: {text[:500]}"
    if status == 402 or (status == 429 and any(w in low for w in _BILLING_WORDS_429)) or (
        400 <= status < 500 and status != 429 and any(w in low for w in _BILLING_WORDS)
    ):
        return BillingError(msg, status=status)
    if status in (400, 413) and any(p in low.replace("_", " ") for p in _CONTEXT_PHRASES):
        return ContextLengthError(msg, status=status)
    if status in RETRYABLE_STATUS or status >= 500:
        return None
    return ProviderError(msg, status=status)


def _is_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def _parse_arguments(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        value = json.loads(text)
    except ValueError:
        return {"_raw": text}
    return value if isinstance(value, dict) else {"_raw": text}


@dataclass
class _Slot:
    id: str | None = None
    name: str = ""
    args: str = ""


class _Accumulator:
    """Assembles one streamed reply.

    Sail streams each finished tool call twice: under index 3, then in full under
    index 0 with the same id (Gemma-4-31B, 2026-10-11); keyed by index that ran every
    command twice and the duplicate id later failed the request. So deltas are routed
    by id, a full re-send of an existing call is dropped, and clashing ids get unique
    names; each repair counts in ``stream_repairs``.
    """

    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.details: dict[tuple[Any, Any], dict[str, Any]] = {}  # reasoning_details by (index, type)
        self.slots: dict[int, _Slot] = {}
        self.by_id: dict[str, int] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] = {}
        self.stream_repairs = 0
        self.n_chunks = 0  # chunks carrying content, reasoning or a tool call
        self.n_reasoning_chunks = 0

    def feed(self, chunk: dict[str, Any]) -> bool:
        """Merge one SSE chunk; True if it carried content, reasoning or tool-call deltas."""
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        carried = thinking = False
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                self.reasoning.append(reasoning)
                carried = thinking = True
            if isinstance(delta.get("reasoning_details"), list) and delta["reasoning_details"]:
                self._details(delta["reasoning_details"])
                carried = thinking = True
            if delta.get("content"):
                self.text.append(delta["content"])
                carried = True
            for pos, tc in enumerate(delta.get("tool_calls") or []):
                self._tool_delta(tc, pos)
                carried = True
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if carried:
            self.n_chunks += 1
            self.n_reasoning_chunks += thinking
        return carried

    def _details(self, items: list[Any]) -> None:
        """Merge reasoning_details deltas per (index, type): ``text`` and ``summary`` stream in pieces,
        the other fields (``signature``, encrypted ``data``, ``id``, ``format``) arrive whole."""
        for item in items:
            if not isinstance(item, dict):
                continue
            cur = self.details.setdefault((item.get("index", 0), item.get("type")), {})
            for k, v in item.items():
                if k in ("text", "summary") and isinstance(v, str):
                    cur[k] = cur.get(k, "") + v
                elif v is not None:
                    cur[k] = v

    def _tool_delta(self, tc: dict[str, Any], pos: int) -> None:
        idx = tc.get("index", pos)
        fn = tc.get("function") or {}
        name, args, tc_id = fn.get("name") or "", fn.get("arguments") or "", tc.get("id")
        if tc_id in self.by_id and self.by_id[tc_id] != idx:
            owner = self.slots[self.by_id[tc_id]]
            same_args = not args or args == owner.args or (_is_object(args) and _is_object(owner.args))
            if name in ("", owner.name) and same_args:
                self.stream_repairs += 1  # a re-send of a call we already have
                return
            idx = self.by_id[tc_id]
        slot = self.slots.setdefault(idx, _Slot())
        if tc_id and not slot.id:
            slot.id = tc_id
            self.by_id.setdefault(tc_id, idx)
        if name and name != slot.name:
            slot.name += name
        slot.args += args

    def tool_calls(self) -> list[ToolCall]:
        """The calls in index order, ids unique (exact duplicates dropped, clashes renamed)."""
        out: list[ToolCall] = []
        seen: dict[str, tuple[str, str]] = {}
        for idx in sorted(self.slots):
            s = self.slots[idx]
            if not s.name and not s.args:
                continue
            call_id = s.id or f"call_{uuid.uuid4().hex[:24]}"
            if call_id in seen:
                self.stream_repairs += 1
                if seen[call_id] == (s.name, s.args):
                    continue
                call_id = f"{call_id}_dup{uuid.uuid4().hex[:8]}"
            seen[call_id] = (s.name, s.args)
            out.append(ToolCall(call_id, s.name, _parse_arguments(s.args)))
        return out


def _wire(m: Message) -> list[dict[str, Any]]:
    if m.role == "tool":
        return [{"role": "tool", "tool_call_id": r.call_id, "content": r.content} for r in m.tool_results]
    msg: dict[str, Any] = {"role": m.role, "content": m.content or ""}
    if m.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in m.tool_calls
        ]
    if m.reasoning_details:
        msg["reasoning_details"] = m.reasoning_details
    elif m.reasoning:
        msg["reasoning_content"] = m.reasoning
    return [msg]


def _cached(content: Any) -> Any:
    """Text as one content part carrying a prompt-cache breakpoint (an empty text cannot carry one)."""
    if not isinstance(content, str) or not content:
        return content
    return [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]


class ChatClient:
    """One model of a provider (``PROVIDERS``); thread-safe (one pooled httpx client). Tests inject
    ``transport``, ``sleep`` and ``rng``."""

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ):
        self.cfg = cfg
        self._url, self._key_var = PROVIDERS[cfg.provider]
        self._http = httpx.Client(timeout=httpx.Timeout(cfg.request_timeout_sec), transport=transport)
        self._sleep = sleep
        self._rng = rng or random.Random()

    def close(self) -> None:
        self._http.close()

    def _body(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        """The request. With ``cache_control``, the system prompt and the newest message carry a
        prompt-cache breakpoint: Anthropic's models on Prime cache nothing without them, and each call
        reads the prefix the previous call's breakpoint wrote."""
        wire = [w for m in messages for w in _wire(m)]
        system_msg: dict[str, Any] = {"role": "system", "content": system}
        if self.cfg.cache_control:
            system_msg["content"] = _cached(system)
            if wire:
                wire[-1] = {**wire[-1], "content": _cached(wire[-1]["content"])}
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [system_msg] + wire,
            "max_tokens": self.cfg.max_output_tokens,
            "temperature": self.cfg.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.cfg.reasoning_effort is not None:  # Prime's form: also turns DeepSeek's thinking on
            body["reasoning"] = {"effort": self.cfg.reasoning_effort}
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        return body

    def _attempt(self, body: dict[str, Any], headers: dict[str, str],
                 cancel: CallCancel | None = None) -> tuple[_Accumulator, float | None]:
        """One HTTP attempt. Fails (retryable) once ``request_timeout_sec`` pass without a
        chunk carrying data: Sail attempts hung 468-2015 s on keep-alive bytes alone. With
        ``cancel``, a cancel stops it at the next line (CallInterrupted; leaving the ``with``
        closes the stream)."""
        acc = _Accumulator()
        t_first: float | None = None
        last = time.time()
        done = False
        if cancel is not None:
            cancel.acc = acc
            if cancel.is_set():
                raise CallInterrupted("interrupted before the request")
        try:
            with self._http.stream("POST", self._url, json=body, headers=headers) as resp:
                if resp.status_code >= 300:
                    text = resp.read().decode("utf-8", "replace")
                    raise _classify(resp.status_code, text) or _Retry(f"HTTP {resp.status_code}: {text[:300]}", resp.status_code)
                for line in resp.iter_lines():
                    if cancel is not None and cancel.is_set():
                        raise CallInterrupted("interrupted while streaming")
                    now = time.time()
                    carried = False
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        chunk = json.loads(data)
                        if chunk.get("error"):
                            raise _Retry(f"stream error: {str(chunk['error'])[:300]}", 200,
                                         after_tool_call=any(s.name or s.args for s in acc.slots.values()))
                        carried = acc.feed(chunk)
                    if carried:
                        t_first, last = t_first or now, now
                        if cancel is not None:
                            cancel.t_first = cancel.t_first or now
                    elif now - last > self.cfg.request_timeout_sec:
                        raise _Retry(f"stalled: no data for {now - last:.0f}s (keep-alive bytes only)", 200)
        except httpx.TransportError as e:
            if cancel is not None and cancel.is_set():
                raise CallInterrupted("interrupted while streaming") from None
            raise _Retry(f"{type(e).__name__}: {e}") from None
        except json.JSONDecodeError as e:
            raise _Retry(f"unparseable SSE data: {e}", 200) from None
        if not done and acc.finish_reason is None:
            raise _Retry("stream ended without finish_reason or [DONE]", 200)
        return acc, t_first

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec],
                 cancel: CallCancel | None = None) -> LLMResponse:
        """One call with retries (full-jitter backoff; no retry starts past ``retry_budget_sec``).
        ``cancel`` (interactive interrupts) aborts it: CallInterrupted, never retried."""
        key = os.environ.get(self._key_var)
        if not key:
            raise ProviderError(f"{self._key_var} is not set", n_attempts=0)
        headers = {"Authorization": f"Bearer {key}"}
        if self.cfg.provider == "sail":
            headers["X-Sail-Completion-Window"] = self.cfg.completion_window or "balanced"
        elif os.environ.get("PRIME_TEAM_ID"):
            headers["X-Prime-Team-ID"] = os.environ["PRIME_TEAM_ID"]
        body = self._body(system, messages, tools)
        t_start = time.time()
        attempts, backoff, one_call = 0, 0.0, False
        while True:
            attempts += 1
            if cancel is not None:
                cancel.n_attempts, cancel.backoff_sec = attempts, backoff
            try:
                acc, t_first = self._attempt(body, headers, cancel)
                break
            except _Retry as e:
                if cancel is not None and cancel.is_set():
                    raise CallInterrupted("interrupted") from None
                if e.after_tool_call and tools and not one_call:
                    one_call = True
                    body = self._body(system + ONE_CALL_NOTE, messages, tools)
                delay = self._rng.uniform(0.0, min(BACKOFF_MAX_SEC, BACKOFF_BASE_SEC * 2 ** (attempts - 1)))
                # a 429 (a billing one raised above) is a rate or concurrency limit: it waits for a free
                # slot until retry_budget_sec, past max_retries (Prime, 2026-10-26: "Too many concurrent
                # requests for this model (limit: 16)" failed 5 attempts in 10-20 s, issue #28)
                capped = attempts > self.cfg.max_retries and e.status != 429
                if capped or time.time() + delay - t_start > self.cfg.retry_budget_sec:
                    raise ProviderError(
                        f"{self.cfg.model}: giving up after {attempts} attempt(s) in {time.time() - t_start:.0f}s: {e}",
                        status=e.status, n_attempts=attempts, backoff_sec=backoff,
                    ) from None
                if cancel is None:
                    self._sleep(delay)
                elif cancel.wait(delay):
                    raise CallInterrupted("interrupted during a retry's backoff") from None
                backoff += delay
                if cancel is not None:
                    cancel.backoff_sec = backoff
            except LLMError as e:
                e.n_attempts, e.backoff_sec = attempts, backoff
                raise
        usage = acc.usage
        calls = acc.tool_calls()
        return LLMResponse(
            text="".join(acc.text),
            tool_calls=calls,
            finish_reason=acc.finish_reason or "unknown",
            reasoning="".join(acc.reasoning),
            reasoning_details=list(acc.details.values()),
            prompt_tokens=usage.get("prompt_tokens") or 0,
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            completion_tokens=usage.get("completion_tokens") or 0,
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            t_start=t_start,
            t_first_chunk=t_first,
            t_end=time.time(),
            n_attempts=attempts,
            backoff_sec=backoff,
            stream_repairs=acc.stream_repairs,
            one_call_retry=one_call,
            n_chunks=acc.n_chunks,
        )
