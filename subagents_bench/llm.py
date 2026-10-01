"""Provider-agnostic LLM turn interface with usage accounting.

The orchestrator and subagents share one message/tool vocabulary so the same
agent loop can drive an Anthropic frontier model and a vLLM-served smaller
model without caring which is which. Each backend takes a list of
:class:`Message` plus :class:`ToolSpec` definitions and returns a
:class:`LLMResponse` carrying assistant text, any tool calls, and token usage.

Backends:

* :class:`AnthropicClient` -- native Messages API. Prompt caching is requested
  on the system block so long repository context is cheap across steps.
* :class:`OpenAIClient` -- any OpenAI-compatible ``/v1/chat/completions``
  endpoint, including a local **vLLM** server for subagents.

Both are thin and lazy-import their SDKs, so importing this module never
requires either SDK to be installed.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import ModelConfig
from .telemetry import Usage

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolSpec:
    name: str
    description: str
    # JSON schema for the tool's input object.
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
    role: MessageRole
    # Plain assistant/user text.
    content: str | None = None
    # For assistant turns that called tools.
    tool_calls: list[ToolCall] = field(default_factory=list)
    # For tool-result turns (role == "tool").
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: str
    cost_usd: float
    t_start: float
    t_end: float
    raw: Any = None


# Convenience providers for the endpoints we deploy against. Each is just an
# OpenAI-compatible server reached at a known base URL; naming them lets a
# config set `provider: sail` / `provider: prime` and a model id, nothing else.
# An explicit `base_url` in the config always wins.
DEFAULT_BASE_URLS: dict[str, str] = {
    "sail": "https://api.sailresearch.com/v1",
    "prime": "https://api.pinference.ai/api/v1",
    "prime_intellect": "https://api.pinference.ai/api/v1",
    "pinference": "https://api.pinference.ai/api/v1",
}

DEFAULT_API_KEY_ENVS: dict[str, str] = {
    "sail": "SAIL_API_KEY",
    "prime": "PRIME_API_KEY",
    "prime_intellect": "PRIME_API_KEY",
    "pinference": "PRIME_API_KEY",
}

OPENAI_COMPATIBLE = {"openai", "vllm", "openai_compatible", *DEFAULT_BASE_URLS}


def make_client(cfg: ModelConfig) -> "LLMClient":
    if cfg.provider == "anthropic":
        return AnthropicClient(cfg)
    if cfg.provider in OPENAI_COMPATIBLE:
        if cfg.base_url is None and cfg.provider in DEFAULT_BASE_URLS:
            cfg.base_url = DEFAULT_BASE_URLS[cfg.provider]
        if cfg.api_key_env is None and cfg.provider in DEFAULT_API_KEY_ENVS:
            cfg.api_key_env = DEFAULT_API_KEY_ENVS[cfg.provider]
        return OpenAIClient(cfg)
    raise ValueError(f"unknown provider: {cfg.provider}")


def _reported_cost(usage: Any) -> float | None:
    """Extract a provider-reported USD cost from an OpenAI-style usage object.

    Sail and Prime Intellect surface cost on the usage object (directly as
    ``cost``, or nested under ``cost.total_usd`` / a details object). Returns
    None when no cost field is present.
    """
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    if cost is None:
        return None
    if isinstance(cost, (int, float)):
        return float(cost)
    for attr in ("total_usd", "usd", "total"):
        val = getattr(cost, attr, None)
        if isinstance(val, (int, float)):
            return float(val)
    if isinstance(cost, dict):
        for key in ("total_usd", "usd", "total"):
            if isinstance(cost.get(key), (int, float)):
                return float(cost[key])
    return None


def _cost(cfg: ModelConfig, usage: Usage) -> float:
    cache_price = cfg.price_cache_read
    non_cached_in = max(usage.input_tokens - usage.cache_read_tokens, 0)
    cost = non_cached_in / 1e6 * cfg.price_in
    cost += usage.output_tokens / 1e6 * cfg.price_out
    if cache_price is not None:
        cost += usage.cache_read_tokens / 1e6 * cache_price
    else:
        # Fall back to full input price for cached tokens if unspecified.
        cost += usage.cache_read_tokens / 1e6 * cfg.price_in
    return cost


class LLMClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def _api_key(self) -> str | None:
        if self.cfg.api_key_env:
            return os.environ.get(self.cfg.api_key_env)
        return None

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self._client = None

    def _lazy(self):
        if self._client is None:
            import anthropic  # noqa: PLC0415

            key = self._api_key() or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key, base_url=self.cfg.base_url)
        return self._client

    def _to_anthropic(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
            elif m.role == "tool":
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.call_id,
                        "content": tr.content,
                        "is_error": tr.is_error,
                    }
                    for tr in m.tool_results
                ]
                out.append({"role": "user", "content": blocks})
            else:  # user
                out.append({"role": "user", "content": m.content or ""})
        return out

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        client = self._lazy()
        tool_defs = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]
        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        t0 = time.time()
        resp = client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_output_tokens,
            temperature=self.cfg.temperature,
            system=system_blocks,
            tools=tool_defs,
            messages=self._to_anthropic(messages),
            **self.cfg.extra_body,
        )
        t1 = time.time()

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens + getattr(u, "cache_read_input_tokens", 0) + getattr(u, "cache_creation_input_tokens", 0),
            output_tokens=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        )
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=resp.stop_reason or "end_turn",
            cost_usd=_cost(self.cfg, usage),
            t_start=t0,
            t_end=t1,
            raw=resp,
        )


# --------------------------------------------------------------------------
# OpenAI-compatible (covers vLLM)
# --------------------------------------------------------------------------


class OpenAIClient(LLMClient):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self._client = None

    def _lazy(self):
        if self._client is None:
            import openai  # noqa: PLC0415

            key = self._api_key() or os.environ.get("OPENAI_API_KEY") or "EMPTY"
            self._client = openai.OpenAI(api_key=key, base_url=self.cfg.base_url)
        return self._client

    def _to_openai(self, system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                out.append(msg)
            elif m.role == "tool":
                for tr in m.tool_results:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.call_id,
                            "content": tr.content,
                        }
                    )
            else:
                out.append({"role": "user", "content": m.content or ""})
        return out

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        client = self._lazy()
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        t0 = time.time()
        resp = client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_output_tokens,
            temperature=self.cfg.temperature,
            messages=self._to_openai(system, messages),
            tools=tool_defs or None,
            **self.cfg.extra_body,
        )
        t1 = time.time()

        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        u = resp.usage
        cached = 0
        if u is not None and getattr(u, "prompt_tokens_details", None) is not None:
            cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        usage = Usage(
            input_tokens=(u.prompt_tokens if u else 0),
            output_tokens=(u.completion_tokens if u else 0),
            cache_read_tokens=cached,
        )
        # Sail and Prime Intellect report a USD cost in the usage object; trust
        # it when the config hasn't set explicit prices, so hosted endpoints are
        # costed accurately without us tracking a price table per model.
        reported = _reported_cost(u)
        if reported is not None and self.cfg.price_in == 0 and self.cfg.price_out == 0:
            cost = reported
        else:
            cost = _cost(self.cfg, usage)
        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=choice.finish_reason or "stop",
            cost_usd=cost,
            t_start=t0,
            t_end=t1,
            raw=resp,
        )
