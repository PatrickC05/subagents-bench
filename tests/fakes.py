"""Scripted fake LLM clients and local-run helpers that produce real episodes offline.

A :class:`FakeClient` replays a script of :class:`Reply` objects (or exceptions to raise) and
returns real :class:`LLMResponse` objects, so the runtime records them as in production. A
script may also be a callable ``(system, messages, tools) -> Reply | Exception`` (concurrent
agents sharing one model). ``latency`` seconds are slept per call, which makes concurrency
observable (``max_in_flight``). :func:`run_local` runs one episode on the host shell.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from subagents_bench.config import ExperimentConfig, from_dict
from subagents_bench.environment import LocalShellEnvironment
from subagents_bench.llm import LLMResponse, Message, ToolCall, ToolSpec
from subagents_bench.orchestrator import Orchestrator
from subagents_bench.records import KINDS, EpisodeRecorder

ORCH, SUB = "orch-model", "sub-model"
_ids = itertools.count(1)


def tc(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(f"call_{next(_ids)}", name, arguments)


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None  # default: tool_calls or stop
    prompt_tokens: int = 100
    cached_tokens: int = 0
    completion_tokens: int = 10
    reasoning: str = ""
    latency: float | None = None


Script = list | Callable[..., Any]


class FakeClient:
    def __init__(self, cfg: Any, script: Script | None, latency: float = 0.0):
        self.cfg, self.latency = cfg, latency
        self.script = script if callable(script) else list(script or [])
        self.requests: list[dict[str, Any]] = []
        self.in_flight = self.max_in_flight = 0
        self._lock = threading.Lock()

    def close(self) -> None:
        pass

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        t_start = time.time()
        with self._lock:
            self.requests.append({"system": system, "messages": copy.deepcopy(messages), "tools": [t.name for t in tools]})
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            if callable(self.script):
                item = None
            elif self.script:
                item = self.script.pop(0)
            else:
                item = AssertionError(f"script for {self.cfg.model} is exhausted")
        try:
            item = self.script(system, messages, tools) if item is None else item
            time.sleep(item.latency if isinstance(item, Reply) and item.latency is not None else self.latency)
            if isinstance(item, BaseException):
                raise item
            return LLMResponse(
                text=item.text, tool_calls=copy.deepcopy(item.tool_calls),
                finish_reason=item.finish_reason or ("tool_calls" if item.tool_calls else "stop"),
                reasoning=item.reasoning, prompt_tokens=item.prompt_tokens, cached_tokens=item.cached_tokens,
                completion_tokens=item.completion_tokens, reasoning_tokens=len(item.reasoning) // 4 or None,
                t_start=t_start, t_first_chunk=time.time(), t_end=time.time(),
            )
        finally:
            with self._lock:
                self.in_flight -= 1


class FakeFactory:
    """``client_factory``: one FakeClient per model id, scripted by ``scripts[model]``."""

    def __init__(self, scripts: dict[str, Script], latency: float = 0.0):
        self.scripts, self.latency, self.clients = scripts, latency, {}

    def __call__(self, cfg: Any) -> FakeClient:
        if cfg.model not in self.clients:
            self.clients[cfg.model] = FakeClient(cfg, self.scripts.get(cfg.model), self.latency)
        return self.clients[cfg.model]

    def __getitem__(self, model: str) -> FakeClient:
        return self.clients[model]


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def make_config(**overrides: Any) -> ExperimentConfig:
    """A test config: fake models with prices, small budgets; ``overrides`` deep-merged."""
    base = {
        "name": "cond",
        "orchestrator": {"model": ORCH, "max_output_tokens": 1000, "price_in": 1.0, "price_cache_read": 0.1, "price_out": 2.0},
        "subagent": {"model": {"model": SUB, "price_in": 0.1, "price_cache_read": 0.01, "price_out": 0.2},
                     "max_steps": 4, "max_output_chars": 4000},
        "budget": {"command_timeout_sec": 20, "max_command_timeout_sec": 60},
        "tools": {"max_output_chars": 4000},
    }
    return from_dict(merge(base, overrides))


def run_local(cfg: ExperimentConfig, factory: FakeFactory, workdir: Path, episode_dir: Path,
              instruction: str = "solve the task", task_name: str = "task-1", trial_name: str | None = None,
              ) -> tuple[str, EpisodeRecorder]:
    """One episode against the host shell in ``workdir``, recorded in ``episode_dir``."""
    env = LocalShellEnvironment(os.path.realpath(workdir))
    rec = EpisodeRecorder(episode_dir, cfg, task_name=task_name, trial_name=trial_name)
    try:
        return Orchestrator(cfg, env, rec, env.workdir, factory).run(instruction), rec
    finally:
        env.close()


def records(episode_dir: Path) -> dict[str, Any]:
    d = Path(episode_dir)
    out: dict[str, Any] = {k: [json.loads(x) for x in (d / f"{k}.jsonl").read_text().splitlines()] for k in KINDS}
    out["episode"] = json.loads((d / "episode.json").read_text())
    return out


def assert_consistent(episode_dir: Path) -> dict[str, Any]:
    """Episode totals equal the JSONL sums, and every id referenced exists."""
    r = records(episode_dir)
    ep, calls = r["episode"], r["calls"]
    assert ep["exit_reason"] and ep["wall_clock_sec"] >= 0
    for role, t in ep["tokens_by_role"].items():
        mine = [c for c in calls if c["role"] == role]
        assert t["prompt"] == sum(c["prompt_tokens"] for c in mine)
        assert t["completion"] == sum(c["completion_tokens"] for c in mine)
        assert t["cached"] == sum(c["cached_tokens"] for c in mine)
    assert ep["n_calls"] == len(calls) and ep["orchestrator_steps"] == len(r["steps"])
    assert (ep["spawn_calls"], ep["dispatches"], ep["arbitrations"]) == (len(r["spawns"]), len(r["dispatches"]), len(r["arbitrations"]))
    assert ep["n_stream_repairs"] == sum(c["stream_repairs"] for c in calls)
    call_ids = {c["call_id"] for c in calls}
    dispatch_ids = {d["dispatch_id"] for d in r["dispatches"]}
    assert {s["call_id"] for s in r["steps"] if s["call_id"]} <= call_ids
    assert {t["call_id"] for t in r["tools"]} <= call_ids
    for s in r["spawns"]:
        assert set(s["dispatch_ids"]) <= dispatch_ids
        assert set(s["arbitration_ids"]) <= {a["arbitration_id"] for a in r["arbitrations"]}
    for a in r["arbitrations"]:
        for rnd in a["rounds"]:
            assert set(rnd["judge_call_ids"]) | set(rnd["debate_call_ids"]) <= call_ids
    assert len({c["call_id"] for c in calls}) == len(calls)
    return r
