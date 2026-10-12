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
threads submit each ``exec`` back to the loop.

When harbor cancels the agent (its agent timeout), the agent sets the
episode's stop event, waits (shielded) up to
:attr:`OrchestratorAgent.CANCEL_GRACE_SEC` for the thread to unwind, kills
every agent command still running in the container (cancelling a
``docker exec`` client never stops the process inside, and harbor's
shared-mode verifier grades the same container next; see
:func:`kill_commands_script`), finalizes the episode with
``exit_reason="cancelled"`` and the kill's outcome in ``cancel_kill`` (the
exit reason stays the thread's if it finalized first) and re-raises.

A :class:`~subagents_bench.types.BillingError` or a harness bug is recorded
by the orchestrator and re-raised so harbor records the exception. An episode
that ended with ``exit_reason="provider_error"`` (the provider could not serve
the orchestrator) is finalized as usual and then
:class:`~subagents_bench.types.ProviderUnavailableError` is raised, so harbor
records an exception and skips the verifier instead of grading an
unattempted task.

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
import re
import shlex
import threading
from pathlib import Path
from typing import Any, Callable

from .. import __version__, workspace
from ..config import ExperimentConfig
from ..environment import HarborEnvironment
from ..orchestrator import Orchestrator
from ..runner import EnvFacts, probe_environment
from ..telemetry import EpisodeRecorder, provenance
from ..types import ProviderUnavailableError

log = logging.getLogger(__name__)

# Fingerprint the source as harbor imports the agent (telemetry caches it per
# process): this process's episodes run the modules loaded now, whatever is
# on disk when the first episode starts.
provenance()

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


# Kills every agent command still running in the sandbox. Each shell-backed
# tool call runs ``wrap_<nonce>.sh`` and the wrapper first writes its pid to
# ``<state_root>/<agent_id>/pid_<nonce>`` (tools._WRAPPER); the outer script
# removes the file when the command ends. For each pid file whose process is
# still that wrapper (checked by its command line, or by its parent's, i.e.
# ``timeout``'s or the outer script's, when the command ``exec``'d away from
# the wrapper; so a stale file with a recycled pid is skipped), SIGKILL the
# wrapper's process group (``timeout`` makes one per command) unless it is our
# own or init's, and the wrapper's process tree. The process table is read
# from /proc (Linux containers) or `ps` (other hosts). Prints
# ``__SB_KILLED <commands killed>``.
_KILL_SCRIPT = r"""
__sb_root=@ROOT@
__sb_tab=
if [ -d /proc/self ]; then
  for __sb_st in /proc/[0-9]*/stat; do
    read -r __sb_l < "$__sb_st" 2>/dev/null || continue
    set -- ${__sb_l##*) }
    __sb_tab="$__sb_tab${__sb_l%% *} $2 $3
"
  done
else
  __sb_tab=$(ps -A -o pid= -o ppid= -o pgid= 2>/dev/null)
fi
__sb_pgid() {
  local p pp g
  while read -r p pp g; do [ "$p" = "$1" ] && { echo "$g"; return; }; done <<< "$__sb_tab"
}
__sb_ppid() {
  local p pp g
  while read -r p pp g; do [ "$p" = "$1" ] && { echo "$pp"; return; }; done <<< "$__sb_tab"
}
__sb_tree() {
  local todo="$1" out= p c pp g
  while [ -n "$todo" ]; do
    set -- $todo; p=$1; shift; todo="$*"
    out="$out $p"
    while read -r c pp g; do [ "$pp" = "$p" ] && todo="$todo $c"; done <<< "$__sb_tab"
  done
  echo "$out"
}
__sb_args() {
  if [ -r "/proc/$1/cmdline" ]; then tr '\000' ' ' < "/proc/$1/cmdline" 2>/dev/null
  else ps -o args= -p "$1" 2>/dev/null; fi
}
__sb_self=$(__sb_pgid $$)
__sb_init=$(__sb_pgid 1)
__sb_n=0
for __sb_f in "$__sb_root"/*/pid_*; do
  [ -f "$__sb_f" ] || continue
  __sb_nonce=${__sb_f##*/pid_}
  read -r __sb_p < "$__sb_f" 2>/dev/null || continue
  case "$__sb_p" in ''|*[!0-9]*) continue ;; esac
  __sb_pp=$(__sb_ppid "$__sb_p")
  case "$(__sb_args "$__sb_p") $([ -n "$__sb_pp" ] && __sb_args "$__sb_pp")" in
    *"wrap_$__sb_nonce"*) ;;
    *) continue ;;
  esac
  __sb_g=$(__sb_pgid "$__sb_p")
  __sb_pids=$(__sb_tree "$__sb_p")
  if [ -n "$__sb_g" ] && [ "$__sb_g" -gt 1 ] 2>/dev/null && [ "$__sb_g" != "$__sb_self" ] && [ "$__sb_g" != "$__sb_init" ]; then
    kill -KILL -- "-$__sb_g" 2>/dev/null
  fi
  kill -KILL $__sb_pids 2>/dev/null
  __sb_n=$((__sb_n + 1))
done
echo "__SB_KILLED $__sb_n"
"""


def kill_commands_script(state_root: str) -> str:
    """bash script that SIGKILLs every agent command still running under
    ``state_root`` (see ``_KILL_SCRIPT``); prints ``__SB_KILLED <n>``."""
    return _KILL_SCRIPT.replace("@ROOT@", shlex.quote(state_root))


class OrchestratorAgent(BaseAgent):
    """Large orchestrator + small subagents, as a harbor external agent."""

    # Seconds to wait for the orchestrator thread to unwind after a cancel.
    CANCEL_GRACE_SEC = 30.0
    # Timeout of the cancel-time kill exec (we stop waiting 10 s later).
    CANCEL_KILL_TIMEOUT_SEC = 20

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
            # Before the verifier sees the container: stop what the agent left running.
            kill = await self._kill_agent_commands(environment, workspace.sb_root(env))
            recorder.finalize(
                "cancelled",
                error="cancelled by harbor (agent timeout or job cancellation)",
                cancel_kill=kill,
                **orch.rt.harness_exec_fields(),
            )
            if recorder.record.get("cancel_kill") != kill:  # the thread finalized first
                recorder.update_record(cancel_kill=kill)
            self._fill_context(context)
            raise
        except BaseException:
            # BillingError or a harness bug: already recorded and finalized
            # by the orchestrator; harbor records the exception.
            self._fill_context(context)
            raise
        self._fill_context(context)
        result = fut.result()
        if result.exit_reason == "provider_error":
            # Not an outcome of the condition: make harbor record an exception
            # (and skip the verifier) so the trial is invalid and is rerun.
            cause = result.cause
            raise ProviderUnavailableError(
                f"episode {result.episode_id} ended with exit_reason=provider_error: {result.error}",
                attempts=list(cause.attempts) if cause is not None else None,
                status=cause.status if cause is not None else None,
                request=cause.request if cause is not None else None,
            ) from cause

    async def _kill_agent_commands(self, environment: Any, state_root: str) -> dict[str, Any]:
        """Best-effort SIGKILL of every agent command still running in the
        container, through harbor's own environment (the HarborEnvironment
        wrapper refuses execs once the stop event is set). Never raises;
        returns ``EpisodeRecord.cancel_kill``."""
        info: dict[str, Any] = {"attempted": False, "ok": False, "n_killed": None, "error": None}
        exec_fn = getattr(environment, "exec", None)
        if not callable(exec_fn):
            info["error"] = "no harbor environment to exec in"
            return info
        try:
            task = asyncio.ensure_future(
                exec_fn(kill_commands_script(state_root), cwd="/", timeout_sec=self.CANCEL_KILL_TIMEOUT_SEC)
            )
        except Exception as e:  # noqa: BLE001 - best effort
            info["error"] = f"{type(e).__name__}: {e}"[:500]
            return info
        info["attempted"] = True
        try:
            # A separate task: a second cancel stops our wait, not the kill.
            done, _ = await asyncio.wait({task}, timeout=self.CANCEL_KILL_TIMEOUT_SEC + 10)
        except asyncio.CancelledError:
            info["error"] = "interrupted by another cancellation before the kill finished"
            return info
        if not done:
            task.cancel()
            info["error"] = f"no result within {self.CANCEL_KILL_TIMEOUT_SEC + 10}s"
            return info
        if task.cancelled():
            info["error"] = "the kill exec was cancelled"
            return info
        exc = task.exception()
        if exc is not None:
            info["error"] = f"{type(exc).__name__}: {exc}"[:500]
            return info
        res = task.result()
        out = getattr(res, "stdout", None) or ""
        m = re.search(r"__SB_KILLED (\d+)", out)
        if m is None:
            detail = (out + (getattr(res, "stderr", None) or "")).strip()
            info["error"] = f"unexpected kill output (exit {getattr(res, 'return_code', None)}): {detail}"[:500]
            return info
        info["ok"], info["n_killed"] = True, int(m.group(1))
        if info["n_killed"]:
            log.warning("killed %d agent command(s) still running after the cancel", info["n_killed"])
        return info

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
