"""OrchestratorAgent driven like harbor 0.24.0 drives it: a real ``BaseAgent``
subclass constructed with harbor's kwargs, ``setup``/``run`` awaited on an
event loop with a real ``AgentContext``, against a fake async environment whose
``exec`` coroutine runs bash locally in a temp dir. Covers the records landing
in ``logs_dir``, progressive AgentContext updates, harbor-style cancellation
(``asyncio.wait_for`` timeout) including the kill of commands left running,
and the BillingError / ProviderUnavailableError propagation."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from fakes import ORCH, SUB, FakeFactory, Reply, assert_valid, tc
from subagents_bench import tools
from subagents_bench.harness import harbor_agent
from subagents_bench.harness.harbor_agent import OrchestratorAgent, kill_commands_script
from subagents_bench.provenance import repo_root
from subagents_bench.types import BillingError, LLMError, ProviderUnavailableError

HAS_HARBOR = importlib.util.find_spec("harbor") is not None
if HAS_HARBOR:
    from harbor.models.agent.context import AgentContext, ModelUsage

pytestmark_harbor = pytest.mark.skipif(not HAS_HARBOR, reason="harbor is not installed")


@dataclass
class _Res:
    stdout: str | None
    stderr: str | None
    return_code: int


# Every process a FakeHarborEnv started; leftovers are killed after each test.
_PROCS: list[subprocess.Popen] = []


@pytest.fixture(autouse=True)
def _kill_leftover_processes():
    yield
    while _PROCS:
        proc = _PROCS.pop()
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass


class FakeHarborEnv:
    """Mimics harbor's async ``BaseEnvironment.exec`` with docker backend
    semantics: default cwd = the image WORKDIR; timeouts raise RuntimeError;
    and cancelling the exec (harbor's agent timeout) does NOT stop the
    command, just as killing a ``docker exec`` client leaves the process in
    the container running."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.n_exec = 0

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.n_exec += 1
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(
                ["bash", "-c", command], cwd=cwd or str(self.workdir), stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, start_new_session=True,
            )
            _PROCS.append(proc)
            deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
            while proc.poll() is None:
                if deadline is not None and time.monotonic() >= deadline:
                    proc.kill()  # the client only, like docker's exec client
                    raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
                await asyncio.sleep(0.01)  # a CancelledError here leaves the command running
            out.seek(0)
            err.seek(0)
            return _Res(out.read().decode(), err.read().decode(), proc.returncode)


def _in_git_worktree() -> bool:
    try:
        return subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"], capture_output=True).returncode == 0
    except OSError:
        return False


def trial(tmp_path, monkeypatch, config: dict) -> tuple[Path, Path, FakeHarborEnv]:
    work = tmp_path / "container_workdir"
    work.mkdir()
    trial_dir = tmp_path / "jobs" / "pilot__r1__cond" / "fix-git__AbC1234"
    logs = trial_dir / "agent"
    logs.mkdir(parents=True)
    (trial_dir / "config.json").write_text(json.dumps({"task": {"path": "fix-git", "source": "terminal-bench"}}))
    base = {
        "name": "cond",
        "topology": "subagents",
        "orchestrator": {"model": ORCH, "provider": "sail", "price_in": 1.0, "price_out": 2.0},
        "subagent": {"model": {"model": SUB, "provider": "sail", "price_in": 0.5, "price_out": 0.5},
                     "strategy": "parallel", "scope": "hard", "max_steps": 3},
        "budget": {"command_timeout_sec": 20, "max_command_timeout_sec": 60},
        "sandbox": {"kind": "harbor", "workdir": "auto"},
        **config,
    }
    monkeypatch.delenv("SUBAGENTS_BENCH_CONFIG", raising=False)
    monkeypatch.setenv("SUBAGENTS_BENCH_CONFIG_JSON", json.dumps(base))
    return logs, work, FakeHarborEnv(work)


def make_agent(logs: Path, factory) -> OrchestratorAgent:
    # The kwargs harbor's AgentFactory passes (logs_dir, model_name, logger, extra_env, ...).
    agent = OrchestratorAgent(logs_dir=logs, model_name=None, logger=None, extra_env={}, unknown_option="ignored")
    agent.session_id = "fix-git__AbC1234__agent"
    agent.context_id = "594025f3-7d65-4655-8576-4bee95002eae"
    agent.client_factory = factory
    # The fake "container" is the host: keep per-agent shell state out of /tmp/.sb.
    agent.state_root = str(logs.parent / "sb_state")
    return agent


@pytestmark_harbor
def test_harbor_run_writes_records_and_fills_context(tmp_path, monkeypatch):
    logs, work, env = trial(tmp_path, monkeypatch, {})
    seen_context: dict = {}
    ctx = AgentContext()

    def orch(system, messages, tools, client):
        n = sum(1 for m in messages if m.role == "assistant")
        if n == 0:
            return Reply("", [tc("bash", command="echo data > out.txt"),
                              tc("spawn_subagents", instructions=["append a line to out.txt"])],
                         prompt_tokens=1000, completion_tokens=100, cached_tokens=200)
        # Progressive update: AgentContext already reflects earlier calls.
        deadline = time.time() + 5
        while ctx.n_input_tokens is None and time.time() < deadline:
            time.sleep(0.02)
        seen_context["n_input_tokens"] = ctx.n_input_tokens
        return Reply("", [tc("finish", summary="done")], prompt_tokens=1500, completion_tokens=50)

    def sub(system, messages, tools, client):
        if len(messages) == 1:
            return Reply("", [tc("bash", command="echo more >> out.txt")], prompt_tokens=300, completion_tokens=30)
        return Reply("", [tc("submit", report="appended")], prompt_tokens=400, completion_tokens=20)

    f = FakeFactory({ORCH: orch, SUB: sub})
    agent = make_agent(logs, f)
    assert agent.name() == "subagents-orchestrator" and agent.version() == harbor_agent.__version__

    async def main():
        await agent.setup(env)
        await agent.run("Fix the repository.", env, ctx)

    asyncio.run(main())

    assert (work / "out.txt").read_text() == "data\nmore\n"
    r = assert_valid(logs)
    ep = r["episode"]
    assert ep["status"] == "done" and ep["exit_reason"] == "finished" and ep["final_summary"] == "done"
    assert (ep["task_name"], ep["trial_name"]) == ("fix-git", "fix-git__AbC1234")
    assert ep["harbor_session_id"] == "fix-git__AbC1234__agent"
    assert ep["harbor_trial_id"] == "594025f3-7d65-4655-8576-4bee95002eae"
    assert ep["sandbox_kind"] == "harbor" and ep["logs_dir"] == str(logs)
    assert os.path.realpath(ep["workdir"]) == os.path.realpath(work)  # resolved by `pwd` (workdir: auto)
    assert ep["container_arch"] == platform.machine()
    assert isinstance(ep["has_timeout_binary"], bool)
    assert ep["harbor_version"] == "0.24.0" and len(ep["source_sha256"]) == 64
    if _in_git_worktree():  # a ZIP/tarball copy has no commit (source_sha256 still identifies the code)
        assert ep["git_commit"] and isinstance(ep["git_dirty"], bool)
    assert ep["n_provider_errors"] == 0 and ep["cancel_kill"] is None
    assert ep["harness_exec_n"] > 0 and 0 < ep["harness_exec_sec"] <= ep["harness_exec_sum_sec"]
    assert ep["files_modified"] == [os.path.join(ep["workdir"], "out.txt")]
    # AgentContext: final totals from the records (input includes cache).
    calls = r["calls"]
    assert ctx.n_input_tokens == sum(c["prompt_tokens"] for c in calls) == 1000 + 1500 + 300 + 400
    assert ctx.n_cache_tokens == 200 and ctx.n_output_tokens == 100 + 50 + 30 + 20
    assert ctx.cost_usd == pytest.approx(sum(c["cost_usd"] for c in calls))
    assert set(ctx.model_usage) == {ORCH, SUB} and isinstance(ctx.model_usage[ORCH], ModelUsage)
    assert ctx.model_usage[SUB].n_input_tokens == 700 and ctx.model_usage[ORCH].n_cache_tokens == 200
    assert ctx.metadata["exit_reason"] == "finished" and ctx.metadata["dispatches"] == 1
    assert ctx.metadata["episode_id"] == ep["episode_id"]
    assert seen_context["n_input_tokens"] is not None and seen_context["n_input_tokens"] >= 1000
    ctx.model_dump_json()  # serializes cleanly for harbor's result.json


@pytestmark_harbor
def test_harbor_cancellation_finalizes_and_stops_thread(tmp_path, monkeypatch):
    logs, work, env = trial(tmp_path, monkeypatch, {})
    ctx = AgentContext()
    f = FakeFactory({ORCH: [Reply("", [tc("bash", command="echo started > s.txt")]),
                            Reply("never", latency=60.0)]})
    agent = make_agent(logs, f)
    t0 = time.time()

    async def main():
        await agent.setup(env)
        # Exactly how harbor applies its agent timeout.
        await asyncio.wait_for(agent.run("task", env, ctx), timeout=1.0)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(main())
    assert time.time() - t0 < 10
    ep = json.loads((logs / "episode.json").read_text())
    assert ep["status"] == "done" and ep["exit_reason"] == "cancelled"
    assert agent.stop_event.is_set()
    # The worker thread unwound on its own (it finalized first, so the agent's
    # fallback finalize -- which would set an error text -- was a no-op), and
    # the in-flight LLM call was abandoned and recorded.
    assert agent._run_future.done() and agent._run_future.exception() is None
    assert ep["error"] is None
    # The kill still ran (nothing was left running) and was added to the thread's record.
    assert ep["cancel_kill"] == {"attempted": True, "ok": True, "n_killed": 0, "error": None}
    r = assert_valid(logs)
    assert r["calls"][-1]["error"].startswith("cancelled") and r["steps"][-1]["call_id"] is None
    assert ctx.metadata["exit_reason"] == "cancelled"
    # Records written after finalization (none expected) would be dropped, not appended.
    assert agent.recorder.finalized and agent.recorder.dropped == 0


def _cancel_during_command(tmp_path, monkeypatch, command: str, topology: str = "single"):
    """Harbor's agent timeout (1.5 s) fires while the agent's bash command runs."""
    cfg = {"topology": topology}
    logs, work, env = trial(tmp_path, monkeypatch, cfg)
    ctx = AgentContext()
    if topology == "single":
        f = FakeFactory({ORCH: [Reply("", [tc("bash", command=command)])]})
    else:
        f = FakeFactory({ORCH: [Reply("", [tc("spawn_subagents", instructions=["run it"])])],
                         SUB: [Reply("", [tc("bash", command=command)])]})
    agent = make_agent(logs, f)
    t0 = time.time()

    async def main():
        await agent.setup(env)
        await asyncio.wait_for(agent.run("task", env, ctx), timeout=1.5)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(main())
    assert time.time() - t0 < 10 and agent._run_future.done()
    return logs, work, ctx, t0


@pytestmark_harbor
@pytest.mark.parametrize("topology", ["single", "subagents"])
def test_harbor_cancellation_kills_the_running_command(tmp_path, monkeypatch, topology):
    # Cancelling the exec leaves the command running (docker semantics), so
    # the agent must kill it before the verifier grades the container.
    logs, work, ctx, t0 = _cancel_during_command(
        tmp_path, monkeypatch, "sleep 3; echo late > late.txt", topology=topology
    )
    r = assert_valid(logs)
    ep = r["episode"]
    assert ep["exit_reason"] == "cancelled"
    assert ep["cancel_kill"] == {"attempted": True, "ok": True, "n_killed": 1, "error": None}
    assert ctx.metadata["cancel_kill"]["n_killed"] == 1
    bash = [t for t in r["tools"] if t["name"] == "bash"][0]
    assert bash["is_error"] and bash["error"] == "cancelled" and bash["duration_sec"] < 5
    time.sleep(max(0.0, t0 + 4.5 - time.time()))  # past the command's end
    assert not (work / "late.txt").exists()


@pytestmark_harbor
def test_without_the_kill_the_command_outlives_the_cancel(tmp_path, monkeypatch):
    # The control: the fake env really has docker semantics (what the kill fixes).
    monkeypatch.setattr(harbor_agent, "kill_commands_script", lambda root: "echo __SB_KILLED 0")
    logs, work, ctx, t0 = _cancel_during_command(tmp_path, monkeypatch, "sleep 3; echo late > late.txt")
    ep = json.loads((logs / "episode.json").read_text())
    assert ep["cancel_kill"] == {"attempted": True, "ok": True, "n_killed": 0, "error": None}
    time.sleep(max(0.0, t0 + 4.5 - time.time()))
    assert (work / "late.txt").read_text() == "late\n"


@pytestmark_harbor
def test_cancel_kill_failure_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(harbor_agent, "kill_commands_script", lambda root: "echo nope; exit 3")
    logs, work, ctx, t0 = _cancel_during_command(tmp_path, monkeypatch, "sleep 2")
    kill = json.loads((logs / "episode.json").read_text())["cancel_kill"]
    assert kill["attempted"] is True and kill["ok"] is False and kill["n_killed"] is None
    assert kill["error"].startswith("unexpected kill output (exit 3): nope")


def _start(args: list[str]) -> subprocess.Popen:
    proc = subprocess.Popen(args, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _PROCS.append(proc)
    return proc


def _run_kill(root: Path) -> str:
    return subprocess.run(["bash", "-c", kill_commands_script(str(root))], capture_output=True, text=True,
                          start_new_session=True, timeout=30).stdout


def _group_gone(pgid: int, wait_sec: float = 3.0) -> bool:
    deadline = time.monotonic() + wait_sec
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


def test_kill_script_kills_live_wrappers_and_skips_stale_pid_files(tmp_path):
    root = tmp_path / "sb state"  # a space: the root is quoted
    agent = root / "sub-1"
    agent.mkdir(parents=True)
    # A live command, as the wrapper runs it: `bash wrap_<nonce>.sh`, with a
    # background child in its process group.
    (agent / "wrap_abc123.sh").write_text("sleep 30 & sleep 30\n")
    live = _start(["bash", str(agent / "wrap_abc123.sh")])
    (agent / "pid_abc123").write_text(f"{live.pid}\n")
    # A stale pid file whose pid now belongs to an unrelated process.
    other = _start(["sleep", "30"])
    (agent / "pid_def456").write_text(f"{other.pid}\n")
    (root / "orchestrator").mkdir()
    (root / "orchestrator" / "pid_000000").write_text("not-a-pid\n")
    time.sleep(0.3)
    assert _run_kill(root).strip() == "__SB_KILLED 1"
    live.wait(timeout=5)
    assert live.returncode == -signal.SIGKILL and _group_gone(live.pid)
    assert other.poll() is None  # untouched
    assert _run_kill(tmp_path / "missing").strip() == "__SB_KILLED 0"


def test_kill_script_kills_a_command_that_exec_d_away_from_the_wrapper(tmp_path):
    # `exec prog` in the command replaces the wrapper's process, so its own
    # command line no longer names wrap_<nonce>; its parent's (timeout's, or
    # the outer script's as here) still does.
    root = tmp_path / "sb"
    agent = root / "orchestrator"
    agent.mkdir(parents=True)
    (agent / "wrap_ex1.sh").write_text(f'echo "$$" > "{agent}/pid_ex1"\nexec sleep 30\n')
    outer = _start(["bash", "-c", f'bash "{agent}/wrap_ex1.sh"; echo done'])
    deadline = time.monotonic() + 5
    while not (agent / "pid_ex1").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)
    assert _run_kill(root).strip() == "__SB_KILLED 1"
    outer.wait(timeout=5)
    assert outer.returncode == -signal.SIGKILL and _group_gone(outer.pid)


def test_kill_script_spares_its_own_process_group(tmp_path):
    # Without `timeout` the wrapper shares the exec's process group; when that
    # is the kill's own group, only the wrapper's process tree is killed.
    agent = tmp_path / "sb" / "orchestrator"
    agent.mkdir(parents=True)
    (agent / "wrap_n1.sh").write_text("sleep 30\n")
    marker = tmp_path / "survived"
    script = (
        f'bash "{agent}/wrap_n1.sh" & echo $! > "{agent}/pid_n1"; sleep 0.3\n'
        + kill_commands_script(str(tmp_path / "sb"))
        + f'\nwait; echo yes > "{marker}"'
    )
    t0 = time.monotonic()
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, start_new_session=True, timeout=30).stdout
    assert "__SB_KILLED 1" in out and marker.read_text() == "yes\n"
    assert time.monotonic() - t0 < 10  # the wrapper's sleep did not run out


@pytest.mark.docker
def test_kill_script_in_a_linux_container():
    # The /proc path and coreutils `timeout` process groups, as in a task container.
    import shutil
    import threading

    from subagents_bench import workspace
    from subagents_bench.environment import LocalDockerEnvironment
    from subagents_bench.tools import run_tool
    from subagents_bench.types import ToolContext

    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip("docker daemon not available")
    with LocalDockerEnvironment("ubuntu:24.04", workdir="/app") as denv:
        ctx = ToolContext(env=denv, agent_id="sub-1", initial_cwd="/app", max_output_chars=4000,
                          command_timeout_sec=60, max_command_timeout_sec=120)
        got: dict = {}
        cmd = "sleep 25 & sleep 25; echo late > /app/late.txt"
        th = threading.Thread(target=lambda: got.update(out=run_tool(ctx, "bash", {"command": cmd})))
        th.start()
        time.sleep(3)
        res = denv.exec(kill_commands_script(workspace.sb_root(denv)), timeout_sec=30)
        assert res.stdout.strip() == "__SB_KILLED 1", res.output
        th.join(timeout=20)
        assert not th.is_alive() and got["out"].exit_code == 137
        probe = denv.exec(
            "for p in /proc/[0-9]*; do tr '\\000' ' ' < $p/cmdline 2>/dev/null; echo; done | grep -c '^sleep 25 $'; "
            "test -e /app/late.txt && echo LATE; true"
        )
        assert probe.stdout.split() == ["0"], probe.output


def test_wrapper_still_writes_the_pid_files_the_kill_reads():
    # kill_commands_script relies on this convention of tools.py.
    assert 'echo "$$" > "$__sb_S/pid_{N}"' in tools._WRAPPER
    outer = tools._outer_script("/s/agent", "n0", "/app", "true", "true", "bash", 0, True)
    assert '"$__sb_S/wrap_n0.sh"' in outer and '"$__sb_S/pid_n0"' in outer  # run, then removed
    assert tools.agent_state_dir(type("E", (), {"state_root": "/s"})(), "sub-1") == "/s/sub-1"


@pytestmark_harbor
def test_provider_error_raises_after_finalizing(tmp_path, monkeypatch):
    logs, work, env = trial(tmp_path, monkeypatch, {"topology": "single"})
    ctx = AgentContext()
    down = LLMError("sail m: giving up after 9 attempts; last ServerError (HTTP 503): upstream down", status=503)
    f = FakeFactory({ORCH: [Reply("", [tc("list_dir")], prompt_tokens=10, completion_tokens=1), down]})
    agent = make_agent(logs, f)

    async def main():
        await agent.setup(env)
        await agent.run("task", env, ctx)

    with pytest.raises(ProviderUnavailableError) as ei:
        asyncio.run(main())
    # A dedicated exception type (harbor records it and skips the verifier), carrying the cause.
    assert isinstance(ei.value, LLMError) and ei.value.status == 503 and ei.value.__cause__ is down
    assert "exit_reason=provider_error" in str(ei.value)
    ep = assert_valid(logs)["episode"]
    assert ep["status"] == "done" and ep["exit_reason"] == "provider_error" and ep["n_provider_errors"] == 1
    assert ep["error"].startswith("LLMError: sail m: giving up") and "HTTP 503" in ep["error"]
    assert ctx.n_input_tokens == 10 and ctx.metadata["exit_reason"] == "provider_error"


@pytestmark_harbor
def test_subagent_provider_error_counts_but_does_not_raise(tmp_path, monkeypatch):
    logs, work, env = trial(tmp_path, monkeypatch, {})
    ctx = AgentContext()
    f = FakeFactory({
        ORCH: [Reply("", [tc("spawn_subagents", instructions=["look around"])]), Reply("", [tc("finish")])],
        SUB: [LLMError("sail s: HTTP401 (HTTP 401): invalid api key", status=401)],
    })
    agent = make_agent(logs, f)

    async def main():
        await agent.setup(env)
        await agent.run("task", env, ctx)

    asyncio.run(main())
    r = assert_valid(logs)
    ep = r["episode"]
    assert ep["exit_reason"] == "finished" and ep["n_provider_errors"] == 1
    (d,) = r["dispatches"]
    assert d["status"] == "error" and d["error"].startswith("LLMError:")
    assert ctx.metadata["n_provider_errors"] == 1


@pytestmark_harbor
def test_harbor_billing_error_propagates(tmp_path, monkeypatch):
    logs, work, env = trial(tmp_path, monkeypatch, {"topology": "single"})
    ctx = AgentContext()
    f = FakeFactory({ORCH: [Reply("", [tc("list_dir")], prompt_tokens=10, completion_tokens=1),
                            BillingError("account balance is zero", status=402)]})
    agent = make_agent(logs, f)

    async def main():
        await agent.setup(env)
        await agent.run("task", env, ctx)

    with pytest.raises(BillingError):
        asyncio.run(main())
    ep = assert_valid(logs)["episode"]
    assert ep["exit_reason"] == "billing_error" and "balance" in ep["error"]
    assert ctx.n_input_tokens == 10 and ctx.metadata["exit_reason"] == "billing_error"


@pytestmark_harbor
def test_run_without_setup_and_config_from_path(tmp_path, monkeypatch):
    logs, work, env = trial(tmp_path, monkeypatch, {"topology": "single"})
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(os.environ["SUBAGENTS_BENCH_CONFIG_JSON"])
    import yaml

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.safe_dump(json.loads(cfg_path.read_text())))
    monkeypatch.setenv("SUBAGENTS_BENCH_CONFIG", str(yaml_path))
    monkeypatch.delenv("SUBAGENTS_BENCH_CONFIG_JSON")
    ctx = AgentContext()
    agent = make_agent(logs, FakeFactory({ORCH: [Reply("", [tc("finish")])]}))
    asyncio.run(agent.run("task", env, ctx))
    assert assert_valid(logs)["episode"]["exit_reason"] == "finished"


def test_missing_config_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("SUBAGENTS_BENCH_CONFIG", raising=False)
    monkeypatch.delenv("SUBAGENTS_BENCH_CONFIG_JSON", raising=False)
    with pytest.raises(RuntimeError, match="SUBAGENTS_BENCH_CONFIG"):
        OrchestratorAgent(logs_dir=tmp_path)


def test_importable_and_constructible_without_harbor(tmp_path):
    code = (
        "import sys, json, os; sys.modules['harbor'] = None\n"
        "os.environ['SUBAGENTS_BENCH_CONFIG_JSON'] = json.dumps({'name': 'x', 'topology': 'single'})\n"
        "import subagents_bench.harness.harbor_agent as m\n"
        "assert m._HARBOR is False and m.ModelUsage is None\n"
        "a = m.OrchestratorAgent(logs_dir=sys.argv[1])\n"
        "assert a.name() == 'subagents-orchestrator' and a.config.name == 'x'\n"
        "print('ok')\n"
    )
    p = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parent.parent)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "ok"
