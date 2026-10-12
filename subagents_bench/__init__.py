"""subagents-bench: measuring the cost and latency of a large orchestrator
directing smaller subagents on executable coding tasks (Terminal-Bench 2.0).

Public surface:

* :class:`~subagents_bench.config.ExperimentConfig` -- one point in the sweep.
* :class:`~subagents_bench.orchestrator.Orchestrator` -- the agent runtime.
* :class:`~subagents_bench.telemetry.EpisodeRecorder` -- the per-episode
  record files (``Telemetry`` is the older name).
* :func:`~subagents_bench.runner.run_task` -- run and record one episode.
* :mod:`subagents_bench.harness` -- harbor / ExploitBench adapters.
"""

# Defined before the submodule imports: llm.py reads it lazily for the
# User-Agent header.
__version__ = "0.1.0"  # keep in sync with pyproject.toml

from .config import (  # noqa: E402
    ArbiterConfig,
    Benchmark,
    BudgetConfig,
    ConfigError,
    ExperimentConfig,
    ModelConfig,
    SandboxConfig,
    Scope,
    Strategy,
    SubagentConfig,
    SubtaskType,
    ToolsConfig,
    Topology,
)
from .orchestrator import EpisodeResult, Orchestrator  # noqa: E402
from .runner import run_task  # noqa: E402
from .telemetry import EpisodeRecorder, Telemetry  # noqa: E402

__all__ = [
    "ArbiterConfig",
    "Benchmark",
    "BudgetConfig",
    "ConfigError",
    "EpisodeRecorder",
    "EpisodeResult",
    "ExperimentConfig",
    "ModelConfig",
    "Orchestrator",
    "SandboxConfig",
    "Scope",
    "Strategy",
    "SubagentConfig",
    "SubtaskType",
    "Telemetry",
    "ToolsConfig",
    "Topology",
    "__version__",
    "run_task",
]
