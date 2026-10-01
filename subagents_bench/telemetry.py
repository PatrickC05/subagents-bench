"""Telemetry: the quantities this study actually reports.

Everything the paper needs to draw its Pareto curves is accumulated here:

* total tokens, split by orchestrator vs. subagents, and by in/out/cache;
* total generation cost in USD;
* subagent **dispatches** and **re-dispatches** (a re-dispatch is any dispatch
  after the first for a given logical subagent slot / retry);
* wall-clock time, from the first API call to episode end.

Every API call is timestamped so per-call latency can be reconstructed from the
event log, and the aggregate is emitted as a single JSON record per trial that
lines up with the SWE-bench / ExploitBench resolve score.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

Role = Literal["orchestrator", "subagent"]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens

    @property
    def total_tokens(self) -> int:
        # Cache-read tokens are already counted inside input_tokens by most
        # providers; we track them separately only for cost, not totals.
        return self.input_tokens + self.output_tokens


@dataclass
class CallEvent:
    """One LLM API call, timestamped for latency reconstruction."""

    role: Role
    model: str
    agent_id: str
    step: int
    t_start: float
    t_end: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float
    n_tool_calls: int = 0
    error: str | None = None

    @property
    def latency_sec(self) -> float:
        return self.t_end - self.t_start


@dataclass
class Telemetry:
    experiment: str
    benchmark: str
    task_id: str
    trial: int
    orchestrator_model: str
    subagent_model: str
    topology: str
    subtask_type: str

    # Counters.
    orchestrator_usage: Usage = field(default_factory=Usage)
    subagent_usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    orchestrator_steps: int = 0
    subagent_dispatches: int = 0
    subagent_redispatches: int = 0
    subagent_steps_total: int = 0

    # Timing.
    t_episode_start: float | None = None
    t_episode_end: float | None = None
    first_call_at: float | None = None

    # Outcome (filled by the scorer).
    resolved: bool | None = None
    exit_reason: str | None = None

    events: list[CallEvent] = field(default_factory=list)
    # Instruction-following: per spawn_subagents call, how many subagents the
    # orchestrator requested vs. how many it was granted.
    spawn_log: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- lifecycle ---------------------------------------------------------

    def start_episode(self) -> None:
        self.t_episode_start = time.time()

    def end_episode(self, exit_reason: str | None = None) -> None:
        self.t_episode_end = time.time()
        if exit_reason is not None:
            self.exit_reason = exit_reason

    # ---- recording ---------------------------------------------------------

    def record_call(self, event: CallEvent) -> None:
        with self._lock:
            self.events.append(event)
            if self.first_call_at is None:
                self.first_call_at = event.t_start
            usage = Usage(event.input_tokens, event.output_tokens, event.cache_read_tokens)
            if event.role == "orchestrator":
                self.orchestrator_usage.add(usage)
            else:
                self.subagent_usage.add(usage)
            self.cost_usd += event.cost_usd

    def record_orchestrator_step(self) -> None:
        with self._lock:
            self.orchestrator_steps += 1

    def record_dispatch(self, is_redispatch: bool, n_steps: int) -> None:
        with self._lock:
            self.subagent_dispatches += 1
            if is_redispatch:
                self.subagent_redispatches += 1
            self.subagent_steps_total += n_steps

    # ---- derived -----------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self.orchestrator_usage.total_tokens + self.subagent_usage.total_tokens

    @property
    def wall_clock_sec(self) -> float | None:
        if self.t_episode_start is None or self.t_episode_end is None:
            return None
        return self.t_episode_end - self.t_episode_start

    # ---- serialisation -----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """The one record per trial that feeds the Pareto analysis."""
        return {
            "experiment": self.experiment,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "trial": self.trial,
            "orchestrator_model": self.orchestrator_model,
            "subagent_model": self.subagent_model,
            "topology": self.topology,
            "subtask_type": self.subtask_type,
            "resolved": self.resolved,
            "exit_reason": self.exit_reason,
            "total_tokens": self.total_tokens,
            "orchestrator_tokens": self.orchestrator_usage.total_tokens,
            "subagent_tokens": self.subagent_usage.total_tokens,
            "orchestrator_input_tokens": self.orchestrator_usage.input_tokens,
            "orchestrator_output_tokens": self.orchestrator_usage.output_tokens,
            "subagent_input_tokens": self.subagent_usage.input_tokens,
            "subagent_output_tokens": self.subagent_usage.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "orchestrator_steps": self.orchestrator_steps,
            "subagent_dispatches": self.subagent_dispatches,
            "subagent_redispatches": self.subagent_redispatches,
            "subagent_steps_total": self.subagent_steps_total,
            "wall_clock_sec": self.wall_clock_sec,
            "n_calls": len(self.events),
            "instruction_following": self.instruction_following(),
            "spawn_log": self.spawn_log,
        }

    def instruction_following(self) -> float | None:
        """Mean granted/requested ratio across spawn calls (None if no spawns).

        This is the subagent instruction-following signal: when the config
        asks the orchestrator to spawn K in parallel, did it?
        """
        if not self.spawn_log:
            return None
        ratios = [
            (e["granted"] / e["requested"]) if e["requested"] else 1.0
            for e in self.spawn_log
        ]
        return round(sum(ratios) / len(ratios), 4)

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        summary_path = directory / "telemetry.json"
        with open(summary_path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        with open(directory / "calls.jsonl", "w") as f:
            for e in self.events:
                f.write(json.dumps(asdict(e)) + "\n")
        return summary_path
