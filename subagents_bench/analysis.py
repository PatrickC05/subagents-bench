"""Aggregate per-trial telemetry into the study's reported curves.

Reads every ``telemetry.json`` under a runs directory and produces:

* a per-configuration table of resolve rate, mean total tokens, mean cost, mean
  wall-clock, and subagent dispatch/re-dispatch counts;
* the (tokens, resolve_rate) and (wall_clock, resolve_rate) points that the
  paper plots as Pareto curves;
* the subagent instruction-following rate.

Pure stdlib so it runs anywhere the harness runs.
"""

from __future__ import annotations

import csv
import json
import statistics

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def load_trials(runs_dir: str | Path) -> list[dict[str, Any]]:
    runs_dir = Path(runs_dir)
    trials = []
    for path in runs_dir.rglob("telemetry.json"):
        with open(path) as f:
            trials.append(json.load(f))
    return trials


def _config_key(t: dict[str, Any]) -> tuple:
    return (
        t.get("experiment"),
        t.get("benchmark"),
        t.get("orchestrator_model"),
        t.get("subagent_model"),
        t.get("topology"),
        t.get("subtask_type"),
    )


@dataclass
class ConfigSummary:
    experiment: str
    benchmark: str
    orchestrator_model: str
    subagent_model: str
    topology: str
    subtask_type: str
    n_trials: int = 0
    n_resolved: int = 0
    tokens: list[int] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    wall: list[float] = field(default_factory=list)
    dispatches: list[int] = field(default_factory=list)
    redispatches: list[int] = field(default_factory=list)
    instr_follow: list[float] = field(default_factory=list)

    @property
    def resolve_rate(self) -> float | None:
        scored = self.n_trials
        return self.n_resolved / scored if scored else None

    def row(self) -> dict[str, Any]:
        def mean(xs):
            return round(statistics.fmean(xs), 4) if xs else None

        return {
            "experiment": self.experiment,
            "benchmark": self.benchmark,
            "orchestrator_model": self.orchestrator_model,
            "subagent_model": self.subagent_model,
            "topology": self.topology,
            "subtask_type": self.subtask_type,
            "n_trials": self.n_trials,
            "resolve_rate": round(self.resolve_rate, 4) if self.resolve_rate is not None else None,
            "mean_total_tokens": mean(self.tokens),
            "mean_cost_usd": mean(self.costs),
            "mean_wall_clock_sec": mean(self.wall),
            "mean_dispatches": mean(self.dispatches),
            "mean_redispatches": mean(self.redispatches),
            "mean_instruction_following": mean(self.instr_follow),
        }


def summarize(trials: list[dict[str, Any]]) -> list[ConfigSummary]:
    groups: dict[tuple, ConfigSummary] = {}
    for t in trials:
        key = _config_key(t)
        if key not in groups:
            groups[key] = ConfigSummary(
                experiment=t.get("experiment", ""),
                benchmark=t.get("benchmark", ""),
                orchestrator_model=t.get("orchestrator_model", ""),
                subagent_model=t.get("subagent_model", ""),
                topology=t.get("topology", ""),
                subtask_type=t.get("subtask_type", ""),
            )
        s = groups[key]
        s.n_trials += 1
        if t.get("resolved"):
            s.n_resolved += 1
        if t.get("total_tokens") is not None:
            s.tokens.append(t["total_tokens"])
        if t.get("cost_usd") is not None:
            s.costs.append(t["cost_usd"])
        if t.get("wall_clock_sec") is not None:
            s.wall.append(t["wall_clock_sec"])
        s.dispatches.append(t.get("subagent_dispatches", 0))
        s.redispatches.append(t.get("subagent_redispatches", 0))
        if t.get("instruction_following") is not None:
            s.instr_follow.append(t["instruction_following"])
    return list(groups.values())


def pareto_points(summaries: list[ConfigSummary], x: str = "mean_total_tokens") -> list[dict[str, Any]]:
    """Resolve-rate vs cost/latency points, one per configuration.

    ``x`` is one of ``mean_total_tokens``, ``mean_cost_usd``,
    ``mean_wall_clock_sec``. Points are sorted by x so plotting a Pareto
    frontier is a left-to-right scan.
    """
    rows = [s.row() for s in summaries]
    rows = [r for r in rows if r.get(x) is not None and r.get("resolve_rate") is not None]
    return sorted(rows, key=lambda r: r[x])


def write_csv(summaries: list[ConfigSummary], path: str | Path) -> Path:
    path = Path(path)
    rows = [s.row() for s in summaries]
    if not rows:
        path.write_text("")
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
