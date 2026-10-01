"""Standalone runner for local development and smoke tests.

The real benchmark runs go through harbor (Terminal-Bench) or the ExploitBench
runner, which own task selection, the official Docker images, and scoring. This
module provides a minimal path to exercise the orchestrator end-to-end on one
hand-written task, in a local container (or, for offline tests, a host shell).

It also exposes :func:`run_task`, the shared "build env + telemetry + run
orchestrator" routine both the CLI and tests use.
"""

from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig
from .environment import Environment, LocalDockerEnvironment, LocalShellEnvironment
from .orchestrator import Orchestrator
from .telemetry import Telemetry


def build_local_environment(config: ExperimentConfig) -> Environment:
    sb = config.sandbox
    if sb.kind == "local_docker":
        if not sb.image:
            raise ValueError("sandbox.image required for local_docker")
        return LocalDockerEnvironment(sb.image, workdir=sb.workdir, network=sb.network)
    if sb.kind == "local_shell":
        return LocalShellEnvironment(workdir=sb.workdir)
    raise ValueError(
        f"standalone runner supports local_docker/local_shell, not {sb.kind!r}; "
        "use the harbor or exploitbench harness for that sandbox."
    )


def run_task(
    config: ExperimentConfig,
    task_id: str,
    instruction: str,
    env: Environment,
    *,
    resolved: bool | None = None,
) -> Telemetry:
    telemetry = Telemetry(
        experiment=config.name,
        benchmark=config.benchmark.value,
        task_id=task_id,
        trial=config.seed,
        orchestrator_model=config.orchestrator.model,
        subagent_model=config.resolved_subagent_model().model,
        topology=config.topology.value,
        subtask_type=config.subtask_type.value,
    )
    orch = Orchestrator(config, env, telemetry)
    orch.run(instruction)
    telemetry.resolved = resolved
    telemetry.write(Path(config.output_dir) / config.name / task_id)
    return telemetry


def run_local(config: ExperimentConfig, task_id: str, instruction: str) -> Telemetry:
    env = build_local_environment(config)
    try:
        return run_task(config, task_id, instruction, env)
    finally:
        close = getattr(env, "close", None)
        if close:
            close()
