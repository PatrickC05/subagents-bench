"""Experiment configuration: one condition of the study, loaded strictly from YAML."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TOPOLOGIES = ("single", "subagents")
STRATEGIES = ("parallel", "sequential", "arbiter")
SCOPES = ("easy", "medium", "hard")
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
class ExperimentConfig:
    name: str
    orchestrator: ModelConfig
    topology: str = "subagents"
    subagent: SubagentConfig = field(default_factory=SubagentConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

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
    bad = sorted(set(cfg.tools.orchestrator_tools or []) - set(TOOL_NAMES))
    if bad:
        raise ConfigError(f"tools.orchestrator_tools: unknown tool(s) {bad}; choose from {list(TOOL_NAMES)}")
    return cfg


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
