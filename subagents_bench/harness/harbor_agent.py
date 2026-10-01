"""Harbor ``BaseAgent`` wrapper for Terminal-Bench 4.0 (and any harbor dataset).

Harbor hands each trial a live ``BaseEnvironment`` (the SWE-bench /
Terminal-Bench container at the base commit) and an instruction string. We wrap
that environment in :class:`HarborEnvironment`, run the orchestrator, and write
token/cost telemetry back into the ``AgentContext`` harbor reports.

Run it with::

    SUBAGENTS_BENCH_CONFIG=configs/unified.yaml \\
      harbor run -d "terminal-bench@head" \\
      --agent-import-path subagents_bench.harness.harbor_agent:OrchestratorAgent

The config is read from ``SUBAGENTS_BENCH_CONFIG`` (a path) or
``SUBAGENTS_BENCH_CONFIG_JSON`` (inline JSON) so a sweep can launch one harbor
job per config without editing code.
"""

from __future__ import annotations

import json
import os

from ..config import ExperimentConfig
from ..environment import HarborEnvironment
from ..orchestrator import Orchestrator
from ..telemetry import Telemetry

try:  # harbor is only present in the benchmark environment.
    from harbor.agents.base import BaseAgent  # type: ignore
    from harbor.models.agent.context import AgentContext  # type: ignore
    from harbor.environments.base import BaseEnvironment  # type: ignore
    _HARBOR = True
except Exception:  # noqa: BLE001 - allow import without harbor installed
    BaseAgent = object  # type: ignore
    AgentContext = object  # type: ignore
    BaseEnvironment = object  # type: ignore
    _HARBOR = False


def load_config() -> ExperimentConfig:
    path = os.environ.get("SUBAGENTS_BENCH_CONFIG")
    if path:
        return ExperimentConfig.from_yaml(path)
    inline = os.environ.get("SUBAGENTS_BENCH_CONFIG_JSON")
    if inline:
        return ExperimentConfig.from_dict(json.loads(inline))
    raise RuntimeError(
        "set SUBAGENTS_BENCH_CONFIG (path) or SUBAGENTS_BENCH_CONFIG_JSON (inline)"
    )


class OrchestratorAgent(BaseAgent):
    """Frontier orchestrator + subagents, as a harbor external agent."""

    def __init__(self, *args, **kwargs):
        if _HARBOR:
            super().__init__(*args, **kwargs)
        self.config = load_config()
        self._telemetry: Telemetry | None = None
        self._task_id = kwargs.get("task_id") or os.environ.get("HARBOR_TASK_ID", "unknown")
        self._env: HarborEnvironment | None = None

    @staticmethod
    def name() -> str:
        return "subagents-orchestrator"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment) -> None:  # noqa: D401 - harbor hook
        workdir = self.config.sandbox.workdir
        self._env = HarborEnvironment(environment, workdir=workdir)

    async def run(self, instruction: str, environment, context) -> None:  # noqa: D401
        cfg = self.config
        if self._env is None:
            self._env = HarborEnvironment(environment, workdir=cfg.sandbox.workdir)

        telemetry = Telemetry(
            experiment=cfg.name,
            benchmark=cfg.benchmark.value,
            task_id=self._task_id,
            trial=cfg.seed,
            orchestrator_model=cfg.orchestrator.model,
            subagent_model=cfg.resolved_subagent_model().model,
            topology=cfg.topology.value,
            subtask_type=cfg.subtask_type.value,
        )
        self._telemetry = telemetry

        orch = Orchestrator(cfg, self._env, telemetry)
        try:
            orch.run(instruction)
        finally:
            self._env.close()
            telemetry.write(os.path.join(cfg.output_dir, cfg.name, self._task_id))

        # Report usage/cost back to harbor's AgentContext.
        self._fill_context(context, telemetry)

    def _fill_context(self, context, telemetry: Telemetry) -> None:
        if not _HARBOR or context is None:
            return
        try:
            context.n_input_tokens = (
                telemetry.orchestrator_usage.input_tokens
                + telemetry.subagent_usage.input_tokens
            )
            context.n_output_tokens = (
                telemetry.orchestrator_usage.output_tokens
                + telemetry.subagent_usage.output_tokens
            )
            context.n_cache_tokens = (
                telemetry.orchestrator_usage.cache_read_tokens
                + telemetry.subagent_usage.cache_read_tokens
            )
            context.cost_usd = telemetry.cost_usd
            context.metadata = {**(context.metadata or {}), **telemetry.summary()}
        except Exception:  # noqa: BLE001 - never fail a trial over telemetry
            pass
