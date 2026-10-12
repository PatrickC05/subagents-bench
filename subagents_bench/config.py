"""Unified configuration for subagent-orchestration experiments.

A single :class:`ExperimentConfig` describes one point in the design space the
study sweeps (``docs/DESIGN.md`` section 2): which model orchestrates, which
(usually smaller) model serves the subagents, the delegation **strategy**
(parallel / sequential / arbiter) and **scope** (easy / medium / hard tool
sets), the step/token budgets, the sandbox, and the task selection.

Loading is strict where it matters for reproducibility: unknown keys and bad
enum values raise :class:`ConfigError` with the offending dotted path, so a
typo in a sweep condition fails at preflight instead of silently running the
default. Fields added after the first YAML files were written all have
defaults, so those files keep loading. ``to_dict()``/``from_dict()`` round-trip
every field (enums become their string values; nested model configs are kept).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """An invalid experiment config (bad key, enum value or range)."""


class Benchmark(str, Enum):
    TERMINAL_BENCH = "terminal_bench"
    EXPLOIT_BENCH = "exploit_bench"


class Topology(str, Enum):
    """How agents are wired together.

    * ``single`` -- the orchestrator model works alone with the same tools and
      budget. No subagents (the ``single_*`` baselines).
    * ``subagents`` -- the orchestrator may spawn subagents. When the subagent
      model equals the orchestrator model this is the same-model control.
    """

    SINGLE = "single"
    SUBAGENTS = "subagents"


class Strategy(str, Enum):
    """How one ``spawn_subagents`` call runs its instructions.

    * ``parallel`` -- one subagent per granted instruction, all concurrently.
    * ``sequential`` -- one subagent per granted instruction, one at a time.
    * ``arbiter`` -- ``n_candidates`` subagents per instruction run
      concurrently; an LLM judge picks the best (confidence-aware debate).
    """

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ARBITER = "arbiter"


class Scope(str, Enum):
    """Delegation scope = the subagent tool set (the README's easy/medium/hard).

    * ``easy`` -- locate/retrieve + summarize: read_file, list_dir, search.
    * ``medium`` -- easy + execute/observe/diagnose: adds bash.
    * ``hard`` -- medium + implement/edit/verify: adds edit_file.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SubtaskType(str, Enum):
    """Legacy delegation nudge: selects a prefix for the orchestrator prompt
    (:data:`subagents_bench.prompts.SUBTASK_PREFIXES`). ``none`` in the study."""

    NONE = "none"
    PARALLEL_EXPLORATION = "parallel_exploration"
    LOCALIZE_THEN_FIX = "localize_then_fix"
    VERIFY = "verify"
    DECOMPOSE = "decompose"


SANDBOX_KINDS = ("harbor", "local_docker", "local_shell", "exploitbench")


@dataclass
class ModelConfig:
    """A single model endpoint.

    ``provider`` picks the client backend (:func:`subagents_bench.llm.make_client`):
    ``sail``, ``prime`` (aliases ``prime_intellect``, ``pinference``),
    ``vllm``/``openai``/``openai_compatible`` (need ``base_url``), or
    ``anthropic``.

    ``price_*`` are USD per 1M tokens and only feed the run-time ``cost_usd``
    estimate; the study recomputes cost offline from logged tokens. A missing
    ``price_in``/``price_out`` means "unknown": calls then record
    ``cost_usd: null`` rather than a made-up 0 (write ``0.0`` for a model
    that really is free, e.g. a local vLLM server).
    """

    model: str
    provider: str = "anthropic"
    base_url: str | None = None
    api_key_env: str | None = None
    max_output_tokens: int = 8192
    temperature: float = 0.0
    # USD per 1M tokens; None = unknown (cost_usd is then None).
    price_in: float | None = None
    price_out: float | None = None
    price_cache_read: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Sail service tier (asap | balanced | flex), sent as X-Sail-Completion-Window.
    completion_window: str | None = None
    # Extra HTTP request headers (sent last, so they override the defaults).
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Send prior assistant reasoning back as ``reasoning_content`` on later turns.
    reasoning_passback: bool = True
    # Model context window in tokens; enables the subagent context guard.
    context_window: int | None = None
    # Retries after the first HTTP attempt (so max_retries + 1 attempts total).
    max_retries: int = 8
    # Connect/read timeout per HTTP attempt (read = max gap between bytes).
    request_timeout_sec: float = 600.0
    # Stream SSE (timings per chunk); False -> one plain POST per attempt.
    stream: bool = True
    # Descriptive only (FLOPs accounting offline); None = unknown.
    active_params_b: float | None = None
    total_params_b: float | None = None


@dataclass
class ArbiterConfig:
    """Settings of the ``arbiter`` strategy (ignored by the others)."""

    # Candidate subagents per instruction (each counts against max_dispatches).
    n_candidates: int = 3
    # Max debate rounds; a round runs only if the judge is not confident/agreeing.
    debate_rounds: int = 1
    # Debate unless (agreement and confidence >= confidence_threshold).
    confidence_threshold: float = 0.8
    # Judge model; None -> the orchestrator model.
    model: ModelConfig | None = None
    # Hard scope: private workdir copy per candidate if `du` <= this, else shared_fallback.
    max_copy_mb: float = 500


@dataclass
class SubagentConfig:
    # None -> the orchestrator model (same-model control).
    model: ModelConfig | None = None
    strategy: Strategy = Strategy.PARALLEL
    scope: Scope = Scope.HARD
    # Max subagents LIVE at once, for every strategy.
    max_concurrent: int = 3
    # Total subagent runs over the episode (arbiter candidates count
    # individually). None -> max_concurrent * budget.orchestrator_steps.
    max_dispatches: int | None = None
    # LLM steps per subagent.
    max_steps: int = 8
    # Tokens (prompt + completion) per subagent; None = no per-subagent cap.
    token_budget: int | None = None
    # Tool-output truncation for subagents (chars).
    max_output_chars: int = 12000
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)

    def __post_init__(self) -> None:
        self.strategy = _enum(Strategy, self.strategy, "subagent.strategy")
        self.scope = _enum(Scope, self.scope, "subagent.scope")


@dataclass
class BudgetConfig:
    # Orchestrator LLM steps per episode (the study fixes 5).
    orchestrator_steps: int = 5
    # Safety cap on prompt + completion tokens across every agent.
    total_token_budget: int = 5_000_000
    # Safety cap on episode wall-clock, seconds (harbor's agent timeout is separate).
    timeout_sec: int = 3600
    # Default per-command timeout for shell-backed tools.
    command_timeout_sec: int = 180
    # Cap on a model-requested bash timeout_sec.
    max_command_timeout_sec: int = 900


@dataclass
class ToolsConfig:
    # Orchestrator tool-output truncation (chars).
    max_output_chars: int = 12000


@dataclass
class SandboxConfig:
    """Where tools execute.

    * ``harbor`` -- harbor hands the agent a live task container.
    * ``exploitbench`` -- the ExploitBench container via a host callable.
    * ``local_docker`` -- the local runner starts a container from ``image``.
    * ``local_shell`` -- the host shell (tests/offline demos only: unsandboxed).

    ``workdir: auto`` resolves to the container's working directory (``pwd``
    with the backend's default cwd, i.e. the task image's WORKDIR).
    """

    kind: str = "harbor"
    image: str | None = None
    workdir: str = "auto"
    network: bool = False

    def __post_init__(self) -> None:
        if self.kind not in SANDBOX_KINDS:
            raise ConfigError(f"sandbox.kind: {self.kind!r} is not one of {', '.join(SANDBOX_KINDS)}")


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
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)

    # Task selection (Terminal-Bench ids / difficulty labels), trials and the
    # local runner's output root. Sweeps leave these to harbor.
    task_ids: list[str] = field(default_factory=list)
    difficulties: list[str] = field(default_factory=list)
    n_trials: int = 1
    seed: int = 1
    output_dir: str = "runs"

    def __post_init__(self) -> None:
        self.benchmark = _enum(Benchmark, self.benchmark, "benchmark")
        self.topology = _enum(Topology, self.topology, "topology")
        self.subtask_type = _enum(SubtaskType, self.subtask_type, "subtask_type")

    # ---- derived -------------------------------------------------------------

    def resolved_subagent_model(self) -> ModelConfig:
        """The model a subagent uses, defaulting to the orchestrator's."""
        return self.subagent.model if self.subagent.model is not None else self.orchestrator

    def resolved_arbiter_model(self) -> ModelConfig:
        """The arbiter's judge model, defaulting to the orchestrator's."""
        m = self.subagent.arbiter.model
        return m if m is not None else self.orchestrator

    def resolved_max_dispatches(self) -> int:
        """Episode dispatch budget (``max_concurrent * orchestrator_steps`` if unset)."""
        cap = self.subagent.max_dispatches
        if cap is not None:
            return cap
        return self.subagent.max_concurrent * self.budget.orchestrator_steps

    def is_same_model_baseline(self) -> bool:
        return (
            self.topology is Topology.SUBAGENTS
            and self.resolved_subagent_model().model == self.orchestrator.model
        )

    def validate(self) -> list[str]:
        """Range problems (empty if the config is usable)."""
        p: list[str] = []
        s, b, a = self.subagent, self.budget, self.subagent.arbiter

        def at_least(name: str, value: Any, lo: float) -> None:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < lo:
                p.append(f"{name} must be >= {lo}, got {value!r}")

        at_least("budget.orchestrator_steps", b.orchestrator_steps, 1)
        at_least("budget.total_token_budget", b.total_token_budget, 1)
        at_least("budget.timeout_sec", b.timeout_sec, 1)
        at_least("budget.command_timeout_sec", b.command_timeout_sec, 1)
        at_least("budget.max_command_timeout_sec", b.max_command_timeout_sec, 1)
        if isinstance(b.command_timeout_sec, int) and isinstance(b.max_command_timeout_sec, int):
            if b.command_timeout_sec > b.max_command_timeout_sec:
                p.append("budget.command_timeout_sec must be <= budget.max_command_timeout_sec")
        at_least("tools.max_output_chars", self.tools.max_output_chars, 100)
        at_least("subagent.max_concurrent", s.max_concurrent, 1)
        if s.max_dispatches is not None:
            at_least("subagent.max_dispatches", s.max_dispatches, 0)
        at_least("subagent.max_steps", s.max_steps, 1)
        if s.token_budget is not None:
            at_least("subagent.token_budget", s.token_budget, 1)
        at_least("subagent.max_output_chars", s.max_output_chars, 100)
        at_least("subagent.arbiter.n_candidates", a.n_candidates, 1)
        at_least("subagent.arbiter.debate_rounds", a.debate_rounds, 0)
        at_least("subagent.arbiter.max_copy_mb", a.max_copy_mb, 0)
        if not isinstance(a.confidence_threshold, (int, float)) or not 0.0 <= a.confidence_threshold <= 1.0:
            p.append(f"subagent.arbiter.confidence_threshold must be in [0, 1], got {a.confidence_threshold!r}")
        for where, m in (("orchestrator", self.orchestrator), ("subagent.model", s.model), ("subagent.arbiter.model", a.model)):
            if m is None:
                continue
            at_least(f"{where}.max_output_tokens", m.max_output_tokens, 1)
            if m.context_window is not None:
                at_least(f"{where}.context_window", m.context_window, 1)
        return p

    def config_hash(self) -> str:
        """sha1[:12] of the canonical JSON of :meth:`to_dict`."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    # ---- (de)serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return _coerce_enums_to_values(asdict(self))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        """Build from a plain dict (YAML/JSON). Raises :class:`ConfigError`."""
        if not isinstance(raw, dict):
            raise ConfigError(f"config must be a mapping, got {type(raw).__name__}")
        raw = copy.deepcopy(raw)
        if not raw.get("name"):
            raise ConfigError("name: required (a non-empty string)")
        _check_keys(cls, raw, "")

        orchestrator = _model_cfg(raw.get("orchestrator"), "orchestrator") or ModelConfig(model="claude-opus-4-8")

        sub_raw = _mapping(raw.get("subagent"), "subagent")
        _check_keys(SubagentConfig, sub_raw, "subagent.")
        arb_raw = _mapping(sub_raw.pop("arbiter", None), "subagent.arbiter")
        _check_keys(ArbiterConfig, arb_raw, "subagent.arbiter.")
        arb_model = _model_cfg(arb_raw.pop("model", None), "subagent.arbiter.model")
        arbiter = ArbiterConfig(model=arb_model, **_drop_none(arb_raw, ArbiterConfig))
        sub_model = _model_cfg(sub_raw.pop("model", None), "subagent.model")
        subagent = SubagentConfig(model=sub_model, arbiter=arbiter, **_drop_none(sub_raw, SubagentConfig))

        budget = _build(BudgetConfig, raw.get("budget"), "budget")
        tools = _build(ToolsConfig, raw.get("tools"), "tools")
        sandbox = _build(SandboxConfig, raw.get("sandbox"), "sandbox")

        cfg = cls(
            name=str(raw["name"]),
            benchmark=raw.get("benchmark") or Benchmark.TERMINAL_BENCH,
            topology=raw.get("topology") or Topology.SUBAGENTS,
            subtask_type=raw.get("subtask_type") or SubtaskType.NONE,
            orchestrator=orchestrator,
            subagent=subagent,
            budget=budget,
            tools=tools,
            sandbox=sandbox,
            task_ids=list(raw.get("task_ids") or []),
            difficulties=list(raw.get("difficulties") or []),
            n_trials=1 if raw.get("n_trials") is None else raw["n_trials"],
            seed=1 if raw.get("seed") is None else raw["seed"],
            output_dir=raw.get("output_dir") or "runs",
        )
        problems = cfg.validate()
        if problems:
            raise ConfigError("; ".join(problems))
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        try:
            return cls.from_dict(raw)
        except ConfigError as e:
            raise ConfigError(f"{path}: {e}") from None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _enum(cls: type[Enum], value: Any, where: str) -> Any:
    if isinstance(value, cls):
        return value
    try:
        return cls(value)
    except ValueError:
        allowed = ", ".join(m.value for m in cls)  # type: ignore[attr-defined]
        raise ConfigError(f"{where}: {value!r} is not one of {allowed}") from None


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return dict(value)


def _check_keys(cls: type, raw: dict[str, Any], prefix: str) -> None:
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(str(k) for k in raw if k not in names)
    if unknown:
        raise ConfigError(
            f"unknown key(s) {', '.join(prefix + k for k in unknown)}; "
            f"allowed under {prefix.rstrip('.') or 'the top level'}: {', '.join(sorted(names))}"
        )


def _drop_none(raw: dict[str, Any], cls: type) -> dict[str, Any]:
    """Keys whose explicit YAML ``null`` should mean "use the default".

    Only fields whose type does not admit None are dropped; for Optional
    fields (e.g. ``max_dispatches: null``) None is a meaningful value.
    """
    optional = {f.name for f in dataclasses.fields(cls) if "None" in str(f.type)}
    return {k: v for k, v in raw.items() if v is not None or k in optional}


def _build(cls: type, value: Any, where: str) -> Any:
    raw = _mapping(value, where)
    _check_keys(cls, raw, where + ".")
    try:
        return cls(**_drop_none(raw, cls))
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from None


def _model_cfg(value: Any, where: str) -> ModelConfig | None:
    if value is None:
        return None
    raw = _mapping(value, where)
    _check_keys(ModelConfig, raw, where + ".")
    if not raw.get("model"):
        raise ConfigError(f"{where}.model: required (the provider's model id)")
    raw = _drop_none(raw, ModelConfig)
    for k in ("extra_body", "extra_headers"):
        if k in raw and not isinstance(raw[k], dict):
            raise ConfigError(f"{where}.{k}: expected a mapping")
    return ModelConfig(**raw)


def _coerce_enums_to_values(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _coerce_enums_to_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_enums_to_values(v) for v in obj]
    return obj
