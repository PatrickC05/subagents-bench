"""Harbor ``BaseAgent`` running the orchestrator on a harbor task container.

Run it with harbor 0.24.0 (the sweep launcher does this for every job)::

    SUBAGENTS_BENCH_CONFIG=configs/sail/base.yaml \\
      harbor run -d terminal-bench@2.0 -i fix-git \\
      -a subagents_bench.harness.harbor_agent:OrchestratorAgent -e docker

The config comes from ``SUBAGENTS_BENCH_CONFIG`` (a YAML path) or
``SUBAGENTS_BENCH_CONFIG_JSON`` (inline JSON). The episode records are written
to ``self.logs_dir`` (``<trial>/agent/``).

Threading: harbor calls :meth:`OrchestratorAgent.run` on its event loop. The
agent captures that loop, wraps harbor's environment in a
:class:`~subagents_bench.environment.HarborEnvironment` bound to it, and runs
the synchronous orchestrator with ``await asyncio.to_thread(...)``; worker
threads submit each ``exec`` back to the loop. When harbor cancels the agent
(its agent timeout), the agent sets the episode's stop event, waits (shielded)
up to :attr:`OrchestratorAgent.CANCEL_GRACE_SEC` for the thread to unwind,
finalizes the episode with ``exit_reason="cancelled"`` (a no-op if the thread
already did) and re-raises. A :class:`~subagents_bench.types.BillingError` or
a harness bug is recorded by the orchestrator and re-raised so harbor records
the exception.

``AgentContext`` (tokens incl. cache, cache tokens, output tokens, cost,
per-model usage, and the episode summary in ``metadata``) is updated after
every LLM call and at the end.

The module imports without harbor installed (tests); ``BaseAgent`` is then a
minimal stand-in.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

from .. import __version__
from ..config import ExperimentConfig
from ..environment import HarborEnvironment
from ..orchestrator import Orchestrator
from ..runner import EnvFacts, probe_environment
from ..telemetry import EpisodeRecorder

log = logging.getLogger(__name__)

try:  # harbor is only installed in the benchmark environment.
    from harbor.agents.base import BaseAgent  # type: ignore
    from harbor.models.agent.context import ModelUsage  # type: ignore

    _HARBOR = True
except Exception:  # noqa: BLE001 - allow import without harbor installed
    _HARBOR = False
    ModelUsage = None  # type: ignore[assignment,misc]

    class BaseAgent:  # type: ignore[no-redef]
        """Stand-in with the attributes the agent uses (harbor absent)."""

        session_id: str | None = None
        context_id: Any = None

        def __init__(self, logs_dir: Any, model_name: str | None = None, logger: Any = None, *args: Any, **kwargs: Any):
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name
            self.logger = logger or log


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> ExperimentConfig:
    path = os.environ.get("SUBAGENTS_BENCH_CONFIG")
    if path:
        return ExperimentConfig.from_yaml(path)
    inline = os.environ.get("SUBAGENTS_BENCH_CONFIG_JSON")
    if inline:
        return ExperimentConfig.from_dict(json.loads(inline))
    raise RuntimeError("set SUBAGENTS_BENCH_CONFIG (YAML path) or SUBAGENTS_BENCH_CONFIG_JSON (inline JSON)")


def _task_identity(logs_dir: Path) -> tuple[str, str | None]:
    """(task_name, trial_name) from the trial dir (``<task>__<suffix>``)."""
    trial_dir = logs_dir.parent
    trial_name = trial_dir.name or None
    task = None
    try:
        cfg = json.loads((trial_dir / "config.json").read_text())
        t = cfg.get("task") or {}
        task = t.get("name") or (Path(t["path"]).name if t.get("path") else None)
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    if not task and trial_name and "__" in trial_name:
        task = trial_name.rsplit("__", 1)[0]
    return task or trial_name or "unknown", trial_name


class OrchestratorAgent(BaseAgent):
    """Large orchestrator + small subagents, as a harbor external agent."""

    # Seconds to wait for the orchestrator thread to unwind after a cancel.
    CANCEL_GRACE_SEC = 30.0

    def __init__(self, logs_dir: Any, model_name: str | None = None, logger: Any = None, *args: Any, **kwargs: Any):
        super().__init__(logs_dir, model_name, logger, *args, **kwargs)
        self.logs_dir = Path(logs_dir)
        try:
            from ..credentials import load_dotenv  # noqa: PLC0415

            load_dotenv(REPO_ROOT / ".env")  # never overrides the environment
        except Exception:  # noqa: BLE001
            log.debug("could not load .env", exc_info=True)
        self.config = load_config()
        self._env: HarborEnvironment | None = None
        self._facts: EnvFacts | None = None
        self.recorder: EpisodeRecorder | None = None
        self.stop_event: threading.Event | None = None
        # Tests inject a fake LLM client factory here.
        self.client_factory: Callable[[Any], Any] | None = None
        self._run_future: asyncio.Future | None = None
        # Harness state dir in the sandbox; None = /tmp/.sb (fresh per task
        # container). Tests running the "container" on the host override it.
        self.state_root: str | None = None

    @staticmethod
    def name() -> str:
        return "subagents-orchestrator"

    def version(self) -> str | None:
        return __version__

    async def setup(self, environment: Any) -> None:
        """Bind to harbor's loop and probe the container (workdir, arch,
        coreutils timeout) from a worker thread."""
        loop = asyncio.get_running_loop()
        wd = self.config.sandbox.workdir
        env = HarborEnvironment(environment, loop=loop, workdir=None if wd in (None, "", "auto") else wd)
        if self.state_root:
            env.state_root = self.state_root
        self._facts = await asyncio.to_thread(probe_environment, env, wd)
        self._env = env

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        loop = asyncio.get_running_loop()
        env = self._env
        if env is None or getattr(env, "_env", None) is not environment or getattr(env, "_loop", None) is not loop:
            await self.setup(environment)
            env = self._env
        assert env is not None and self._facts is not None
        facts = self._facts
        stop = threading.Event()
        env.stop_event = stop
        self.stop_event = stop
        task_name, trial_name = _task_identity(self.logs_dir)

        def on_event(kind: str, _rec: dict) -> None:
            if kind != "calls":
                return
            try:
                loop.call_soon_threadsafe(self._fill_context, context)
            except RuntimeError:  # loop closed
                pass

        recorder = EpisodeRecorder(
            self.logs_dir,
            config=self.config,
            task_name=task_name,
            instruction=instruction,
            sandbox_kind="harbor",
            workdir=facts.workdir,
            trial_name=trial_name,
            harbor_session_id=getattr(self, "session_id", None),
            harbor_trial_id=None if getattr(self, "context_id", None) is None else str(self.context_id),
            container_arch=facts.container_arch,
            has_timeout_binary=facts.has_timeout_binary,
            on_event=on_event,
        )
        self.recorder = recorder
        orch = Orchestrator(
            self.config, env, recorder, workdir=facts.workdir, stop_event=stop, client_factory=self.client_factory
        )
        fut = asyncio.ensure_future(asyncio.to_thread(orch.run, instruction))
        self._run_future = fut  # introspection (tests)
        try:
            await asyncio.shield(fut)
        except asyncio.CancelledError:
            stop.set()
            if not fut.done():
                try:
                    await asyncio.wait({fut}, timeout=self.CANCEL_GRACE_SEC)
                except asyncio.CancelledError:
                    pass
            if not fut.done():
                log.warning("orchestrator thread did not unwind within %.0fs of the cancel", self.CANCEL_GRACE_SEC)
            harness = orch.rt.harness
            recorder.finalize(
                "cancelled",
                error="cancelled by harbor (agent timeout or job cancellation)",
                harness_exec_n=harness.n if harness is not None else None,
                harness_exec_sec=harness.sec if harness is not None else None,
            )
            self._fill_context(context)
            raise
        except BaseException:
            # BillingError or a harness bug: already recorded and finalized
            # by the orchestrator; harbor records the exception.
            self._fill_context(context)
            raise
        self._fill_context(context)

    def _fill_context(self, context: Any) -> None:
        """Copy the recorded totals into harbor's AgentContext (never raises)."""
        rec = self.recorder
        if context is None or rec is None:
            return
        try:
            totals = rec.role_totals()
            t = totals["total"]
            context.n_input_tokens = int(t["n_input_tokens"])
            context.n_cache_tokens = int(t["n_cache_tokens"])
            context.n_output_tokens = int(t["n_output_tokens"])
            context.cost_usd = None if t["cost_usd"] is None else float(t["cost_usd"])
            usage = {}
            for model, m in totals["models"].items():
                usage[model] = ModelUsage(**m) if ModelUsage is not None else dict(m)
            context.model_usage = usage
            context.metadata = {**(context.metadata or {}), **rec.summary()}
        except Exception:  # noqa: BLE001 - never fail a trial over reporting
            log.exception("could not update the harbor AgentContext")
