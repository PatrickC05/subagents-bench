"""subagents-bench: measuring the cost and latency of a frontier orchestrator
directing smaller subagents on executable coding tasks.

Public surface:

* :class:`~subagents_bench.config.ExperimentConfig` -- one point in the sweep.
* :class:`~subagents_bench.orchestrator.Orchestrator` -- the agent engine.
* :class:`~subagents_bench.telemetry.Telemetry` -- the reported metrics.
* :mod:`subagents_bench.harness` -- Terminal-Bench / ExploitBench adapters.
"""

from .config import (
    Benchmark,
    BudgetConfig,
    ExperimentConfig,
    ModelConfig,
    SandboxConfig,
    SubagentConfig,
    SubtaskType,
    Topology,
)
from .orchestrator import Orchestrator
from .telemetry import Telemetry

__all__ = [
    "Benchmark",
    "BudgetConfig",
    "ExperimentConfig",
    "ModelConfig",
    "SandboxConfig",
    "SubagentConfig",
    "SubtaskType",
    "Topology",
    "Orchestrator",
    "Telemetry",
]

__version__ = "0.1.0"
