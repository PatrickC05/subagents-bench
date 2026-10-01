"""Unified configuration for subagent-orchestration experiments.

A single :class:`ExperimentConfig` describes one point in the design space the
project studies: which model orchestrates, which (possibly smaller) model is
served to its subagents, how many subagents it may spawn, what kind of subtask
it is nudged to delegate, which benchmark subset it runs on, and the
time/token budgets.

The same config object drives both Terminal-Bench 4.0 and ExploitBench; only
the ``benchmark`` field and a handful of benchmark-scoped knobs differ, so a
sweep can be expressed as a list of configs that vary one axis at a time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class Benchmark(str, Enum):
    TERMINAL_BENCH = "terminal_bench"
    EXPLOIT_BENCH = "exploit_bench"


class Topology(str, Enum):
    """How agents are wired together.

    * ``single`` -- baseline 1: the orchestrator model works alone with the
      same tools and budget. No subagents.
    * ``subagents`` -- the orchestrator may spawn subagents. When
      ``subagent.model`` equals ``orchestrator.model`` this is baseline 2
      (same-model subagents), which isolates the effect of *having* subagents
      from the effect of subagent *size*.
    """

    SINGLE = "single"
    SUBAGENTS = "subagents"


class SubtaskType(str, Enum):
    """Delegation behaviour nudged into the orchestrator's system prompt.

    This selects the *prefix* prepended to the orchestrator prompt. The exact
    wording lives in :mod:`subagents_bench.prompts`. ``none`` leaves the
    orchestrator free to decide; the others steer what it hands to subagents.
    """

    NONE = "none"
    PARALLEL_EXPLORATION = "parallel_exploration"
    LOCALIZE_THEN_FIX = "localize_then_fix"
    VERIFY = "verify"
    DECOMPOSE = "decompose"


@dataclass
class ModelConfig:
    """A single model endpoint.

    ``provider`` picks the client backend:

    * ``anthropic`` -- native Anthropic Messages API (typical orchestrator).
    * ``openai`` -- any OpenAI-compatible chat/completions endpoint. This also
      covers subagents served locally by **vLLM**: point ``base_url`` at the
      vLLM server (``http://host:8000/v1``).

    ``price_in``/``price_out`` are USD per 1M tokens, used for cost telemetry.
    For vLLM-served subagents on fixed hardware there is no API price; set both
    to an amortised estimate (or 0 to report tokens only).
    """

    model: str
    provider: str = "anthropic"
    base_url: str | None = None
    api_key_env: str | None = None
    max_output_tokens: int = 8192
    temperature: float = 0.0
    # USD per 1M tokens.
    price_in: float = 0.0
    price_out: float = 0.0
    price_cache_read: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentConfig:
    model: ModelConfig | None = None
    # Maximum number of subagents the orchestrator may have live at once.
    max_concurrent: int = 4
    # Hard cap on total dispatches across the whole episode (re-dispatches
    # included). None -> derive from max_concurrent * orchestrator steps.
    max_dispatches: int | None = None
    # Per-subagent step budget (a subagent runs its own tool loop).
    max_steps: int = 8
    # Per-subagent token budget; None -> inherit from BudgetConfig.
    token_budget: int | None = None


@dataclass
class BudgetConfig:
    # Orchestrator-visible step budget for the whole episode. The study fixes
    # this at 5 steps per episode.
    orchestrator_steps: int = 5
    # Total token budget across orchestrator + all subagents.
    total_token_budget: int = 2_000_000
    # Wall-clock timeout for the whole episode, seconds.
    timeout_sec: int = 1800


@dataclass
class SandboxConfig:
    """Where tools actually execute.

    * ``harbor`` -- the harness hands us a live ``BaseEnvironment`` (the
      SWE-bench / Terminal-Bench Docker container); we exec inside it.
    * ``exploitbench`` -- the ExploitBench container, driven through its MCP
      tool interface.
    * ``local_docker`` -- the standalone runner starts its own container from
      ``image`` for smoke tests and local development.
    """

    kind: str = "harbor"
    image: str | None = None
    workdir: str = "/app"
    network: bool = False


@dataclass
class ExperimentConfig:
    name: str
    benchmark: Benchmark = Benchmark.TERMINAL_BENCH
    topology: Topology = Topology.SUBAGENTS
    subtask_type: SubtaskType = SubtaskType.NONE

    orchestrator: ModelConfig = field(
        default_factory=lambda: ModelConfig(model="claude-opus-4-8", provider="anthropic")
    )
    subagent: SubagentConfig = field(default_factory=SubagentConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)

    # Benchmark task selection. For Terminal-Bench, ``task_ids`` filters by id
    # and ``difficulties`` by the task's difficulty label (the task-size axis).
    task_ids: list[str] = field(default_factory=list)
    difficulties: list[str] = field(default_factory=list)
    # How many trials per task (seeds), and where to write logs.
    n_trials: int = 1
    seed: int = 1
    output_dir: str = "runs"

    def resolved_subagent_model(self) -> ModelConfig:
        """The model a subagent uses, defaulting to the orchestrator's."""
        if self.subagent.model is not None:
            return self.subagent.model
        return self.orchestrator

    def is_same_model_baseline(self) -> bool:
        return (
            self.topology is Topology.SUBAGENTS
            and self.resolved_subagent_model().model == self.orchestrator.model
        )

    # ---- (de)serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict leaves enums as Enum members in some python versions; coerce.
        return _coerce_enums_to_values(d)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        raw = copy.deepcopy(raw)

        def model_cfg(d: dict[str, Any] | None) -> ModelConfig | None:
            if d is None:
                return None
            return ModelConfig(**d)

        orchestrator = model_cfg(raw.get("orchestrator")) or ModelConfig(
            model="claude-opus-4-8"
        )

        sub_raw = raw.get("subagent", {}) or {}
        subagent = SubagentConfig(
            model=model_cfg(sub_raw.get("model")),
            max_concurrent=sub_raw.get("max_concurrent", 4),
            max_dispatches=sub_raw.get("max_dispatches"),
            max_steps=sub_raw.get("max_steps", 8),
            token_budget=sub_raw.get("token_budget"),
        )

        budget = BudgetConfig(**(raw.get("budget", {}) or {}))
        sandbox = SandboxConfig(**(raw.get("sandbox", {}) or {}))

        return cls(
            name=raw["name"],
            benchmark=Benchmark(raw.get("benchmark", "terminal_bench")),
            topology=Topology(raw.get("topology", "subagents")),
            subtask_type=SubtaskType(raw.get("subtask_type", "none")),
            orchestrator=orchestrator,
            subagent=subagent,
            budget=budget,
            sandbox=sandbox,
            task_ids=raw.get("task_ids", []) or [],
            difficulties=raw.get("difficulties", []) or [],
            n_trials=raw.get("n_trials", 1),
            seed=raw.get("seed", 1),
            output_dir=raw.get("output_dir", "runs"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


def _coerce_enums_to_values(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _coerce_enums_to_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_enums_to_values(v) for v in obj]
    return obj
