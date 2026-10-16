"""Experiment configuration: one condition of the study, loaded strictly from YAML."""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TOPOLOGIES = ("single", "subagents")
STRATEGIES = ("parallel", "sequential", "arbiter")
SCOPES = ("easy", "medium", "hard")
SPAWN_MODES = ("blocking", "background")
HINT_DETAILS = ("files", "tests", "report")
SESSION_SCOPES = ("reported", "all")
TOOL_NAMES = ("bash", "edit_file", "read_file", "list_dir", "search")


class ConfigError(ValueError):
    """An invalid experiment config (unknown key, bad choice)."""


@dataclass
class ModelConfig:
    """A Sail model. Prices (USD per 1M tokens) only feed harbor's AgentContext cost."""

    model: str
    completion_window: str = "balanced"
    temperature: float = 0.7
    max_output_tokens: int = 4096
    max_retries: int = 4
    retry_budget_sec: float = 120.0
    request_timeout_sec: float = 180.0
    price_in: float | None = None
    price_cache_read: float | None = None
    price_out: float | None = None


@dataclass
class ArbiterConfig:
    n_candidates: int = 3
    debate_rounds: int = 1
    confidence_threshold: float = 0.8


@dataclass
class SubagentConfig:
    model: ModelConfig | None = None  # None: the orchestrator's model (same-model control)
    strategy: str = "parallel"
    scope: str = "hard"
    max_concurrent: int = 3
    max_dispatches: int = 15
    max_steps: int = 8
    max_output_chars: int = 12000
    delegation_hint: bool = False  # tell the orchestrator its steps are scarce and subagents' steps free
    # Interactive, background only: tell the orchestrator to stay available to the user (short steps,
    # a short instruction per bug at once, the subagent works out the problem itself)
    availability_hint: bool = False
    # Interactive, background only: tell the orchestrator to put one subagent in charge of each bug
    owner_hint: bool = False
    # Helper subagents a subagent may start itself (`spawn_subagents`, blocking, one level deep); 0: none
    max_children: int = 0
    # Interactive episodes only: `background` returns from spawn_subagents at once and delivers each
    # report as a message when its subagent ends; `blocking` waits for all of them, as in batch episodes.
    spawn_mode: str = "blocking"
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)


@dataclass
class BudgetConfig:
    orchestrator_steps: int = 5
    command_timeout_sec: int = 180
    max_command_timeout_sec: int = 900


@dataclass
class ToolsConfig:
    max_output_chars: int = 12000
    orchestrator_tools: list[str] | None = None  # None: all five; read-only tools make a "manager"


@dataclass
class InteractiveConfig:
    """A timed session: the task's subtasks arrive one by one as user messages (SWE-sweep)."""

    wall_clock_sec: float = 3600.0  # the episode ends here, wherever it is
    interval_sec: float = 300.0  # hint k (0-based) is sent at k * interval_sec
    max_hints: int | None = None  # None: as many as fit in the wall clock (and the task has)
    hint_detail: str = "tests"  # files | tests (+ the hidden tests' ids) | report (a generated bug report)
    order_seed: int = 0  # the subtasks' order, shuffled per task with this seed (same for every condition)
    vary_order_by_round: bool = True  # round k of a sweep uses order_seed + k - 1 (each round its own order)
    scope: str = "reported"  # reported: fix the bugs the user sends (the task's "fix all bugs" is set aside); all
    session_note: str | None = None  # a paragraph for the first message, before hint 0 (e.g. a testing rule)
    end_when_idle: bool = True  # end once every hint was sent, nothing runs and the orchestrator waits
    max_context_chars: int | None = None  # beyond this, elide old tool output and reports, drop old reasoning
    # Requests (docs/DESIGN.md, *Requests*): every user message gets an id and the agent answers with
    # `reply`; the user sees nothing else. Bugs then ask for an ETA when planned and a done reply.
    replies: bool = False
    questions_per_interval: int = 0  # code questions sent between consecutive bugs, evenly spaced (needs replies)
    status_every: int = 0  # 0: none; n: after every n-th bug, a status check halfway to the next asks for ETAs
    urgent_every: int = 0  # 0: none; n: bugs n, 2n, ... (0-based) are urgent: "fix this one before the others"
    # `reply` to a bug must carry eta_min or done: true, or it is refused (an error, nothing sent; needs replies)
    require_eta: bool = False
    # Workspace snapshots (subagents_bench.snapshots): the collect hook's diff against `baseline`, taken in
    # the background after every bash or edit_file call and every snapshot_every_sec, for `fix-times`.
    snapshots: bool = False
    snapshot_every_sec: float = 15.0
    # Interrupts (docs/DESIGN.md, *Interrupts*): a user message that comes due while the orchestrator's
    # LLM call runs aborts the call (its partial reply is dropped) and is delivered at once; the call is
    # made again with it. Never a call younger than interrupt_min_sec or one already streaming a tool
    # call, and never more than interrupt_max_in_row calls in a row. Subagent reports never interrupt.
    interrupt: bool = False
    interrupt_min_sec: float = 5.0
    interrupt_max_in_row: int = 3


@dataclass
class ExperimentConfig:
    name: str
    orchestrator: ModelConfig
    topology: str = "subagents"
    subagent: SubagentConfig = field(default_factory=SubagentConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    interactive: InteractiveConfig | None = None  # None: the batch episode (a step budget, `finish`)

    @property
    def subagent_model(self) -> ModelConfig:
        return self.subagent.model or self.orchestrator

    @property
    def orchestrator_tools(self) -> list[str]:
        allowed = self.tools.orchestrator_tools
        return [t for t in TOOL_NAMES if allowed is None or t in allowed]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_NESTED = {
    (ExperimentConfig, "orchestrator"): ModelConfig,
    (ExperimentConfig, "subagent"): SubagentConfig,
    (ExperimentConfig, "budget"): BudgetConfig,
    (ExperimentConfig, "tools"): ToolsConfig,
    (ExperimentConfig, "interactive"): InteractiveConfig,
    (SubagentConfig, "model"): ModelConfig,
    (SubagentConfig, "arbiter"): ArbiterConfig,
}


def _build(cls: type, raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where or 'config'}: expected a mapping, got {type(raw).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(str(k) for k in raw if k not in names)
    if unknown:
        raise ConfigError(f"unknown key(s): {', '.join(where + k for k in unknown)}")
    kwargs = {}
    for k, v in raw.items():
        sub = _NESTED.get((cls, k))
        kwargs[k] = _build(sub, v, f"{where}{k}.") if sub and v is not None else v
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"{where or 'config'}: {e}") from None


def from_dict(raw: Any) -> ExperimentConfig:
    """Build and check a config; raises :class:`ConfigError`."""
    cfg = _build(ExperimentConfig, raw, "")
    for where, value, choices in (
        ("topology", cfg.topology, TOPOLOGIES),
        ("subagent.strategy", cfg.subagent.strategy, STRATEGIES),
        ("subagent.scope", cfg.subagent.scope, SCOPES),
    ):
        if value not in choices:
            raise ConfigError(f"{where}: {value!r} is not one of {', '.join(choices)}")
    if not isinstance(cfg.subagent.delegation_hint, bool):
        raise ConfigError(f"subagent.delegation_hint: expected true or false, got {cfg.subagent.delegation_hint!r}")
    if not isinstance(cfg.subagent.owner_hint, bool):
        raise ConfigError(f"subagent.owner_hint: expected true or false, got {cfg.subagent.owner_hint!r}")
    mc = cfg.subagent.max_children
    if isinstance(mc, bool) or not isinstance(mc, int) or mc < 0:
        raise ConfigError(f"subagent.max_children: expected an integer >= 0, got {mc!r}")
    if mc and cfg.subagent.strategy == "arbiter":
        raise ConfigError("subagent.max_children: helper subagents are not supported with strategy arbiter")
    if not isinstance(cfg.subagent.availability_hint, bool):
        raise ConfigError(f"subagent.availability_hint: expected true or false, got {cfg.subagent.availability_hint!r}")
    bad = sorted(set(cfg.tools.orchestrator_tools or []) - set(TOOL_NAMES))
    if bad:
        raise ConfigError(f"tools.orchestrator_tools: unknown tool(s) {bad}; choose from {list(TOOL_NAMES)}")
    if cfg.subagent.spawn_mode not in SPAWN_MODES:
        raise ConfigError(f"subagent.spawn_mode: {cfg.subagent.spawn_mode!r} is not one of {', '.join(SPAWN_MODES)}")
    if cfg.interactive is not None:
        _check_interactive(cfg)
    elif cfg.subagent.spawn_mode != "blocking":
        raise ConfigError("subagent.spawn_mode: background needs an `interactive` episode")
    return cfg


def _check_interactive(cfg: ExperimentConfig) -> None:
    i = cfg.interactive
    if i.session_note is not None and not (isinstance(i.session_note, str) and i.session_note.strip()):
        raise ConfigError(f"interactive.session_note: expected null or non-empty text, got {i.session_note!r}")
    if i.scope not in SESSION_SCOPES:
        raise ConfigError(f"interactive.scope: {i.scope!r} is not one of {', '.join(SESSION_SCOPES)}")
    if i.hint_detail not in HINT_DETAILS:
        raise ConfigError(f"interactive.hint_detail: {i.hint_detail!r} is not one of {', '.join(HINT_DETAILS)}")
    for name in ("wall_clock_sec", "interval_sec", "snapshot_every_sec"):
        v = getattr(i, name)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < math.inf:
            raise ConfigError(f"interactive.{name}: expected a positive number, got {v!r}")
    for name, v, lo in (("max_hints", i.max_hints, 1), ("max_context_chars", i.max_context_chars, 10000)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < lo):
            raise ConfigError(f"interactive.{name}: expected null or an integer >= {lo}, got {v!r}")
    if isinstance(i.order_seed, bool) or not isinstance(i.order_seed, int):
        raise ConfigError(f"interactive.order_seed: expected an integer, got {i.order_seed!r}")
    for name in ("questions_per_interval", "status_every", "urgent_every"):
        v = getattr(i, name)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ConfigError(f"interactive.{name}: expected an integer >= 0, got {v!r}")
    if not i.replies and (i.questions_per_interval or i.status_every or i.urgent_every):
        raise ConfigError("interactive: questions, status checks and urgent bugs come with message ids and `reply`: "
                          "set replies: true")
    for name in ("end_when_idle", "vary_order_by_round", "replies", "require_eta", "snapshots", "interrupt"):
        if not isinstance(getattr(i, name), bool):
            raise ConfigError(f"interactive.{name}: expected true or false, got {getattr(i, name)!r}")
    v = i.interrupt_min_sec
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < math.inf:
        raise ConfigError(f"interactive.interrupt_min_sec: expected a number >= 0, got {v!r}")
    v = i.interrupt_max_in_row
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ConfigError(f"interactive.interrupt_max_in_row: expected an integer >= 1, got {v!r}")
    if i.require_eta and not i.replies:
        raise ConfigError("interactive.require_eta: it is a rule for `reply` to a bug: set replies: true")
    if cfg.topology == "subagents" and cfg.subagent.spawn_mode == "background" and cfg.subagent.strategy != "parallel":
        raise ConfigError("subagent.spawn_mode: background runs each instruction as its own subagent; "
                          "it needs strategy parallel")
    if cfg.topology == "subagents" and cfg.subagent.delegation_hint and cfg.subagent.spawn_mode != "background":
        raise ConfigError("subagent.delegation_hint: in an interactive episode it says background subagents keep "
                          "the orchestrator free, so it needs spawn_mode background")
    if cfg.subagent.owner_hint and not (cfg.interactive and cfg.topology == "subagents"
                                        and cfg.subagent.spawn_mode == "background"):
        raise ConfigError("subagent.owner_hint: it tells an interactive orchestrator to put background subagents "
                          "in charge of bugs, so it needs interactive, topology subagents and spawn_mode background")
    if cfg.subagent.availability_hint and not (cfg.interactive and cfg.topology == "subagents"
                                               and cfg.subagent.spawn_mode == "background"):
        raise ConfigError("subagent.availability_hint: it tells an interactive orchestrator to hand bugs to "
                          "background subagents, so it needs interactive, topology subagents and spawn_mode background")


def load_config(path: str | os.PathLike) -> ExperimentConfig:
    try:
        return from_dict(yaml.safe_load(Path(path).read_text()))
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None


def load_dotenv(path: str | os.PathLike) -> None:
    """Set the ``KEY=VALUE`` lines of ``path`` that are not already in the environment."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        key = key.strip()
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip("'\""))
