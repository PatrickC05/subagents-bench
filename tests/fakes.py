"""Scripted LLM clients for runtime tests (no network).

:class:`FakeClient` replays a per-model script of :class:`Reply` objects (or
exceptions to raise) and returns real :class:`~subagents_bench.types.LLMResponse`
objects -- usage, timing instants, attempts and a
:class:`~subagents_bench.types.RequestInfo` computed with the real
``build_request_info`` -- so the runtime's records are built exactly as in
production. A script may also be a callable ``(system, messages, tools,
client) -> Reply | Exception``, which lets concurrent agents sharing one model
get role-specific answers.

Unless a reply says otherwise, prompt tokens are ``ceil(prompt_chars / 4)``
of the request actually sent (so the context guard sees realistic numbers)
and completion tokens are ``ceil(output chars / 4)``. ``latency`` (seconds,
per client or per reply) is spent in ``time.sleep`` between the request and
the first chunk, which makes concurrency observable: every client tracks how
many of its calls were in flight at once (``max_in_flight``).

The second half holds the runtime tests' shared helpers: a LocalShell config
whose workdir is kept apart from the output dir (:func:`make_config`),
:func:`run`, and :func:`assert_valid`, the record checks every runtime test
applies.

All numbers produced here are test fixtures, not measurements.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from subagents_bench.config import ExperimentConfig
from subagents_bench.environment import LocalShellEnvironment
from subagents_bench.llm import _cost, build_request_info
from subagents_bench.runner import run_task
from subagents_bench.telemetry import _generic_wire, read_jsonl, validate_episode_dir
from subagents_bench.types import Attempt, CallTiming, LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage

_ids = itertools.count(1)


def tc(name: str, **arguments: Any) -> ToolCall:
    """A tool call with a unique id."""
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None  # None -> ceil(prompt_chars / 4)
    completion_tokens: int | None = None  # None -> ceil(output chars / 4)
    cached_tokens: int = 0
    reasoning_tokens: int | None = None
    reasoning: str = ""
    finish_reason: str | None = None
    latency: float | None = None  # overrides the client's latency
    failed_attempts: int = 0  # retried attempts before the successful one
    backoff_sec: float = 0.0  # recorded backoff per failed attempt (not slept)
    usage_source: str = "provider"


class ScriptExhausted(LLMError):
    """The test's script for a model ran out (a test bug)."""


Script = list | Callable[..., Any]


class FakeClient:
    def __init__(self, cfg: Any, script: Script | None = None, *, latency: float = 0.0):
        self.cfg = cfg
        self.latency = latency
        self._script = script if callable(script) else list(script or [])
        self._lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def _next(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Any:
        if callable(self._script):
            return self._script(system, messages, tools, self)
        with self._lock:
            if not self._script:
                return ScriptExhausted(f"script for model {self.cfg.model!r} is exhausted")
            return self._script.pop(0)

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.requests.append(
                {"system": system, "messages": copy.deepcopy(messages), "tools": [t.name for t in tools]}
            )
        try:
            return self._complete(system, messages, tools)
        finally:
            with self._lock:
                self.in_flight -= 1

    def _complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        t_start = time.time()
        item = self._next(system, messages, tools)
        latency = self.latency
        if isinstance(item, Reply) and item.latency is not None:
            latency = item.latency
        if latency:
            time.sleep(latency)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, Reply), f"script items must be Reply or an exception, got {item!r}"
        tool_defs = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]
        req = build_request_info(
            self.cfg,
            system=system,
            system_wire=system or None,
            wire_messages=_generic_wire(messages),
            tool_defs=tool_defs,
            reasoning_passback=bool(getattr(self.cfg, "reasoning_passback", True)),
        )
        out_chars = len(item.text) + len(item.reasoning) + sum(len(str(c.arguments)) + len(c.name) for c in item.tool_calls)
        usage = Usage(
            prompt_tokens=item.prompt_tokens if item.prompt_tokens is not None else math.ceil(req.prompt_chars / 4),
            completion_tokens=item.completion_tokens if item.completion_tokens is not None else max(math.ceil(out_chars / 4), 1),
            cached_tokens=item.cached_tokens,
            reasoning_tokens=item.reasoning_tokens,
        )
        t_end = time.time()
        # Split the measured interval into request -> first chunk -> last chunk.
        t_req = t_start
        t_first = t_start + (t_end - t_start) * 0.6
        attempts = [
            Attempt(attempt=i + 1, t_start=t_start, t_end=t_start, status=429, error_type="RateLimited",
                    error="rate limited", retry_after_sec=None, backoff_sec=item.backoff_sec)
            for i in range(item.failed_attempts)
        ]
        attempts.append(Attempt(attempt=item.failed_attempts + 1, t_start=t_req, t_end=t_end, status=200))
        timing = CallTiming(
            t_start=t_start,
            t_end=t_end,
            t_request_sent=t_req,
            t_headers=t_req,
            t_first_chunk=t_first,
            t_first_reasoning=t_first if item.reasoning else None,
            t_first_content=t_first if item.text else None,
            t_first_tool_call=t_first if item.tool_calls else None,
            t_last_chunk=t_end,
            n_chunks=3,
            chunk_offsets_ms=[(t_first - t_req) * 1000, (t_end - t_req) * 500 + (t_first - t_req) * 500, (t_end - t_req) * 1000],
            streamed=True,
        )
        finish = item.finish_reason or ("tool_calls" if item.tool_calls else "stop")
        return LLMResponse(
            text=item.text,
            tool_calls=[ToolCall(id=c.id, name=c.name, arguments=copy.deepcopy(c.arguments)) for c in item.tool_calls],
            usage=usage,
            finish_reason=finish,
            cost_usd=_cost(self.cfg, usage),
            timing=timing,
            request=req,
            attempts=attempts,
            reasoning=item.reasoning,
            usage_source=item.usage_source,  # type: ignore[arg-type]
            usage_raw={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
            if item.usage_source == "provider" else None,
            provider_reported_cost=None,
            response_headers={"content-type": "text/event-stream"},
            response_id=f"resp-{next(_ids)}",
        )


class FakeFactory:
    """``client_factory`` for the runtime: one FakeClient per model id."""

    def __init__(self, scripts: dict[str, Script] | None = None, latency: dict[str, float] | float = 0.0):
        self.scripts = scripts or {}
        self.latency = latency
        self.clients: dict[str, FakeClient] = {}
        self._lock = threading.Lock()

    def __call__(self, cfg: Any) -> FakeClient:
        with self._lock:
            if cfg.model not in self.clients:
                lat = self.latency.get(cfg.model, 0.0) if isinstance(self.latency, dict) else self.latency
                self.clients[cfg.model] = FakeClient(cfg, self.scripts.get(cfg.model), latency=lat)
            return self.clients[cfg.model]

    def __getitem__(self, model: str) -> FakeClient:
        return self.clients[model]


def last_user_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return ""


def tool_names(tools: list[ToolSpec]) -> list[str]:
    return [t.name for t in tools]


# ---------------------------------------------------------------------------
# runtime test helpers
# ---------------------------------------------------------------------------

ORCH = "orch-model"
SUB = "sub-model"


def make_config(tmp_path: Path, **overrides: Any) -> ExperimentConfig:
    """A config for LocalShell runs; ``overrides`` are deep-merged dicts."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    base: dict[str, Any] = {
        "name": "cond",
        "topology": "subagents",
        "orchestrator": {"model": ORCH, "provider": "sail", "price_in": 1.0, "price_out": 2.0, "price_cache_read": 0.1},
        "subagent": {
            "model": {"model": SUB, "provider": "sail", "price_in": 0.1, "price_out": 0.2},
            "strategy": "parallel",
            "scope": "hard",
            "max_concurrent": 3,
            "max_steps": 4,
        },
        "budget": {"orchestrator_steps": 5, "command_timeout_sec": 20, "max_command_timeout_sec": 60},
        "sandbox": {"kind": "local_shell", "workdir": str(work)},
        "output_dir": str(tmp_path / "runs"),
    }

    def merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    return ExperimentConfig.from_dict(merge(base, overrides))


def run(cfg: ExperimentConfig, factory: Any, instruction: str = "solve the task", **kw: Any):
    env = LocalShellEnvironment(cfg.sandbox.workdir)
    try:
        return run_task(cfg, "task-1", instruction, env, client_factory=factory, **kw)
    finally:
        env.close()


def records(episode_dir: Path) -> dict[str, Any]:
    d = Path(episode_dir)
    out: dict[str, Any] = {name: read_jsonl(d / f"{name}.jsonl") for name in
                           ("calls", "tools", "steps", "spawns", "dispatches", "arbitrations", "transcript")}
    out["episode"] = json.loads((d / "episode.json").read_text())
    return out


def assert_valid(episode_dir: Path) -> dict[str, Any]:
    """Every line of every file validates against schema.py, the analysis
    package's check-run finds no errors (totals = sums, contiguous agent_seq,
    references resolve), and each agent's events are time-ordered."""
    problems = validate_episode_dir(episode_dir)
    assert problems == [], problems
    from subagents_bench.analysis.validate import check_episode

    errors = [str(p) for p in check_episode(episode_dir) if p.level == "error"]
    assert errors == [], errors
    r = records(episode_dir)
    chains: dict[str, list[dict]] = {}
    for ev in r["calls"] + r["tools"]:
        chains.setdefault(ev["agent_id"], []).append(ev)
    for agent, evs in chains.items():
        seqs = sorted(e["agent_seq"] for e in evs)
        assert seqs == list(range(len(seqs))), (agent, seqs)
    assert sorted(e["seq"] for e in r["calls"] + r["tools"]) == list(range(len(r["calls"]) + len(r["tools"])))
    return r
