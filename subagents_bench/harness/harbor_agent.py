"""Harbor external agent running the orchestrator in a task container.

    SUBAGENTS_BENCH_CONFIG=configs/sail/base.yaml harbor run -d terminal-bench@2.0 -i fix-git \\
      -a subagents_bench.harness.harbor_agent:OrchestratorAgent -e docker

The synchronous orchestrator runs in a worker thread; records go to harbor's ``logs_dir``
(the trial's ``agent/``). On harbor's cancel (its agent timeout) the agent stops the episode,
waits up to 30 s for the thread, kills every agent command still running in the container
(cancelling a ``docker exec`` client leaves the process running, and the verifier grades the
same container next), removes the arbiter copies and finalizes the episode as ``cancelled``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.models.agent.context import AgentContext, ModelUsage

from .. import __version__
from ..config import load_config, load_dotenv
from ..environment import HarborEnvironment
from ..llm import LLMError, SailClient
from ..orchestrator import Orchestrator
from ..records import EpisodeRecorder

REPO_ROOT = Path(__file__).resolve().parents[2]

# Each running command's wrapper wrote its pid, which is also its process group, to
# <state_root>/<agent>/pid_<n> (removed when the command ends): kill those groups.
KILL_SCRIPT = """n=0
for f in {root}/*/pid_*; do
  [ -f "$f" ] || continue
  read -r p < "$f" 2>/dev/null || continue
  case "$p" in ''|*[!0-9]*) continue ;; esac
  kill -KILL -- "-$p" 2>/dev/null && n=$((n + 1))
done
rm -rf -- {root}/arb
echo "__SB_KILLED $n"
"""


class ProviderUnavailableError(LLMError):
    """Raised after an episode ended with ``provider_error``, so harbor records an exception
    instead of grading an unattempted task (the trial is invalid and is rerun)."""


def kill_script(state_root: str) -> str:
    return KILL_SCRIPT.format(root=shlex.quote(state_root))


def task_name(trial_dir: Path) -> str:
    try:
        return Path(json.loads((trial_dir / "config.json").read_text())["task"]["path"]).name
    except (OSError, ValueError, KeyError, TypeError):
        return trial_dir.name.rsplit("__", 1)[0]


class OrchestratorAgent(BaseAgent):
    CANCEL_GRACE_SEC = 30.0
    KILL_TIMEOUT_SEC = 30

    def __init__(self, logs_dir: Path, *args: Any, **kwargs: Any):
        super().__init__(logs_dir, *args, **kwargs)
        load_dotenv(REPO_ROOT / ".env")
        path = os.environ.get("SUBAGENTS_BENCH_CONFIG")
        if not path:
            raise RuntimeError("set SUBAGENTS_BENCH_CONFIG to the condition's YAML config")
        self.config = load_config(path)
        self.client_factory = SailClient  # tests inject fakes
        self.state_root: str | None = None  # tests keep host-side shell state out of /tmp/.sb
        self.recorder: EpisodeRecorder | None = None

    @staticmethod
    def name() -> str:
        return "subagents-orchestrator"

    def version(self) -> str:
        return __version__

    async def setup(self, environment: Any) -> None:
        pass

    async def run(self, instruction: str, environment: Any, context: AgentContext) -> None:
        env = HarborEnvironment(environment, asyncio.get_running_loop())
        if self.state_root:
            env.state_root = self.state_root
        pwd = await asyncio.to_thread(env.exec, "pwd", 60)
        workdir = pwd.stdout.strip().splitlines()[-1]
        trial_dir = Path(self.logs_dir).parent
        rec = self.recorder = EpisodeRecorder(self.logs_dir, self.config, task_name=task_name(trial_dir),
                                              trial_name=trial_dir.name)
        orch = Orchestrator(self.config, env, rec, workdir, self.client_factory)
        fut = asyncio.ensure_future(asyncio.to_thread(orch.run, instruction))
        try:
            await asyncio.shield(fut)
        except asyncio.CancelledError:
            rec.stop_event.set()
            await asyncio.wait({fut}, timeout=self.CANCEL_GRACE_SEC)
            kill = await self._kill(environment, env.state_root)
            rec.finalize("cancelled", "cancelled by harbor (agent timeout)")  # a no-op if the thread finalized
            rec.update(cancel_kill=kill)
            raise
        finally:
            self._fill_context(context)
        if fut.result() == "billing_error":
            raise rec.failure
        if fut.result() == "provider_error":
            raise ProviderUnavailableError(f"episode ended with provider_error: {rec.failure}") from rec.failure

    async def _kill(self, environment: Any, state_root: str) -> dict[str, Any]:
        """SIGKILL the agent commands still running; returns the episode's ``cancel_kill``."""
        try:
            res = await asyncio.wait_for(environment.exec(kill_script(state_root), timeout_sec=self.KILL_TIMEOUT_SEC),
                                         self.KILL_TIMEOUT_SEC + 10)
        except Exception as e:  # recorded; the trial is graded either way
            return {"ok": False, "n_killed": None, "error": f"{type(e).__name__}: {e}"[:500]}
        m = re.search(r"__SB_KILLED (\d+)", res.stdout or "")
        if m is None:
            return {"ok": False, "n_killed": None, "error": f"unexpected output: {(res.stdout or '')[-300:]}"}
        return {"ok": True, "n_killed": int(m.group(1)), "error": None}

    def _fill_context(self, context: AgentContext) -> None:
        """Tokens and cost (from the config's prices) per model, and the episode in metadata."""
        rec = self.recorder
        if rec is None:
            return
        models = {m.model: m for m in (self.config.orchestrator, self.config.subagent_model)}
        usage: dict[str, ModelUsage] = {}
        for c in list(rec.records["calls"]):
            u = usage.setdefault(c["model"], ModelUsage(cost_usd=0.0))
            u.n_input_tokens += c["prompt_tokens"]
            u.n_cache_tokens += c["cached_tokens"]
            u.n_output_tokens += c["completion_tokens"]
            m = models[c["model"]]
            if None in (m.price_in, m.price_cache_read, m.price_out) or u.cost_usd is None:
                u.cost_usd = None
            else:
                u.cost_usd += ((c["prompt_tokens"] - c["cached_tokens"]) * m.price_in + c["cached_tokens"]
                               * m.price_cache_read + c["completion_tokens"] * m.price_out) / 1e6
        context.model_usage = usage
        context.n_input_tokens = sum(u.n_input_tokens for u in usage.values())
        context.n_cache_tokens = sum(u.n_cache_tokens for u in usage.values())
        context.n_output_tokens = sum(u.n_output_tokens for u in usage.values())
        costs = [u.cost_usd for u in usage.values()]
        context.cost_usd = None if None in costs else sum(costs)
        context.metadata = {k: v for k, v in rec.episode.items() if k != "config"}
