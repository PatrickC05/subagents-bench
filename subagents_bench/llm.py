"""Sail chat-completions client (OpenAI-compatible, streamed SSE) and the message types."""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .config import ModelConfig

SAIL_URL = "https://api.sailresearch.com/v1/chat/completions"
RETRYABLE_STATUS = frozenset({408, 409, 425, 429})
BACKOFF_BASE_SEC = 2.0
BACKOFF_MAX_SEC = 15.0
_BILLING_WORDS = ("balance", "billing", "credit", "payment")
_BILLING_WORDS_429 = ("insufficient_quota", "billing")
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
    reasoning: str | None = None  # sent back as reasoning_content on later turns


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str
    reasoning: str = ""
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


class _Retry(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


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
        self.slots: dict[int, _Slot] = {}
        self.by_id: dict[str, int] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] = {}
        self.stream_repairs = 0

    def feed(self, chunk: dict[str, Any]) -> bool:
        """Merge one SSE chunk; True if it carried content, reasoning or tool-call deltas."""
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        carried = False
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                self.reasoning.append(reasoning)
                carried = True
            if delta.get("content"):
                self.text.append(delta["content"])
                carried = True
            for pos, tc in enumerate(delta.get("tool_calls") or []):
                self._tool_delta(tc, pos)
                carried = True
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return carried

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
    if m.reasoning:
        msg["reasoning_content"] = m.reasoning
    return [msg]


class SailClient:
    """One Sail model; thread-safe (one pooled httpx client). Tests inject ``transport``,
    ``sleep`` and ``rng``."""

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ):
        self.cfg = cfg
        self._http = httpx.Client(timeout=httpx.Timeout(cfg.request_timeout_sec), transport=transport)
        self._sleep = sleep
        self._rng = rng or random.Random()

    def close(self) -> None:
        self._http.close()

    def _body(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}] + [w for m in messages for w in _wire(m)],
            "max_tokens": self.cfg.max_output_tokens,
            "temperature": self.cfg.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        return body

    def _attempt(self, body: dict[str, Any], headers: dict[str, str]) -> tuple[_Accumulator, float | None]:
        """One HTTP attempt. Fails (retryable) once ``request_timeout_sec`` pass without a
        chunk carrying data: Sail attempts hung 468-2015 s on keep-alive bytes alone."""
        acc = _Accumulator()
        t_first: float | None = None
        last = time.time()
        done = False
        try:
            with self._http.stream("POST", SAIL_URL, json=body, headers=headers) as resp:
                if resp.status_code >= 300:
                    text = resp.read().decode("utf-8", "replace")
                    raise _classify(resp.status_code, text) or _Retry(f"HTTP {resp.status_code}: {text[:300]}", resp.status_code)
                for line in resp.iter_lines():
                    now = time.time()
                    carried = False
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        chunk = json.loads(data)
                        if chunk.get("error"):
                            raise _Retry(f"stream error: {str(chunk['error'])[:300]}", 200)
                        carried = acc.feed(chunk)
                    if carried:
                        t_first, last = t_first or now, now
                    elif now - last > self.cfg.request_timeout_sec:
                        raise _Retry(f"stalled: no data for {now - last:.0f}s (keep-alive bytes only)", 200)
        except httpx.TransportError as e:
            raise _Retry(f"{type(e).__name__}: {e}") from None
        except json.JSONDecodeError as e:
            raise _Retry(f"unparseable SSE data: {e}", 200) from None
        if not done and acc.finish_reason is None:
            raise _Retry("stream ended without finish_reason or [DONE]", 200)
        return acc, t_first

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        """One call with retries (full-jitter backoff; no retry starts past ``retry_budget_sec``)."""
        key = os.environ.get("SAIL_API_KEY")
        if not key:
            raise ProviderError("SAIL_API_KEY is not set", n_attempts=0)
        headers = {"Authorization": f"Bearer {key}", "X-Sail-Completion-Window": self.cfg.completion_window}
        body = self._body(system, messages, tools)
        t_start = time.time()
        attempts, backoff = 0, 0.0
        while True:
            attempts += 1
            try:
                acc, t_first = self._attempt(body, headers)
                break
            except _Retry as e:
                delay = self._rng.uniform(0.0, min(BACKOFF_MAX_SEC, BACKOFF_BASE_SEC * 2 ** (attempts - 1)))
                if attempts > self.cfg.max_retries or time.time() + delay - t_start > self.cfg.retry_budget_sec:
                    raise ProviderError(
                        f"{self.cfg.model}: giving up after {attempts} attempt(s) in {time.time() - t_start:.0f}s: {e}",
                        status=e.status, n_attempts=attempts, backoff_sec=backoff,
                    ) from None
                self._sleep(delay)
                backoff += delay
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
        )
