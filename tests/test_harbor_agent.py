"""OrchestratorAgent driven like harbor drives it (real BaseAgent, AgentContext and
``asyncio.wait_for`` timeouts) against a fake async environment that runs bash on the host
with docker semantics: cancelling an exec does not stop the command."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from harbor.models.agent.context import AgentContext

from fakes import ORCH, SUB, FakeFactory, Reply, assert_consistent, make_config, tc
from subagents_bench.harness.harbor_agent import OrchestratorAgent, ProviderUnavailableError, kill_script
from subagents_bench.llm import BillingError, ProviderError

PROCS: list[subprocess.Popen] = []


@pytest.fixture(autouse=True)
def kill_leftovers():
    yield
    for p in PROCS:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    PROCS.clear()


@dataclass
class Res:
    stdout: str
    stderr: str
    return_code: int


class FakeHarborEnv:
    def __init__(self, workdir: Path):
        self.workdir = workdir

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        with tempfile.TemporaryFile() as out:  # stderr merged into stdout, as in harbor
            proc = subprocess.Popen(["bash", "-c", command], cwd=cwd or self.workdir, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            PROCS.append(proc)
            deadline = time.monotonic() + (timeout_sec or 1e9)
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
                await asyncio.sleep(0.01)  # a cancel here leaves the command running
            out.seek(0)
            return Res(out.read().decode(), "", proc.returncode)


def trial(tmp_path, monkeypatch, scripts, **overrides):
    work = tmp_path / "work"
    work.mkdir(parents=True)
    trial_dir = tmp_path / "jobs/pilot__r1__cond/fix-git__AbC1234"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "config.json").write_text(json.dumps({"task": {"path": "fix-git"}}))
    cfg_path = tmp_path / "cond.yaml"
    cfg_path.write_text(yaml.safe_dump(make_config(**overrides).to_dict()))
    monkeypatch.setenv("SUBAGENTS_BENCH_CONFIG", str(cfg_path))
    agent = OrchestratorAgent(logs_dir=trial_dir / "agent", model_name=None, logger=None, extra_env={})
    agent.client_factory = FakeFactory(scripts)
    agent.state_root = str(tmp_path / "sb")
    return agent, FakeHarborEnv(work), work


def run(agent, env, timeout=None) -> AgentContext:
    ctx = AgentContext()

    async def main():
        await agent.setup(env)
        await asyncio.wait_for(agent.run("Fix the repo.", env, ctx), timeout)

    asyncio.run(main())
    return ctx


def test_run_writes_records_and_fills_the_context(tmp_path, monkeypatch):
    agent, env, work = trial(tmp_path, monkeypatch, {
        ORCH: [Reply("", [tc("bash", command="echo data > out.txt"), tc("spawn_subagents", instructions=["append"])],
                     prompt_tokens=1000, cached_tokens=200, completion_tokens=100),
               Reply("", [tc("finish")], prompt_tokens=1500, completion_tokens=50)],
        SUB: [Reply("", [tc("bash", command="echo more >> out.txt")], prompt_tokens=300, completion_tokens=30),
              Reply("", [tc("submit", report="appended")], prompt_tokens=400, completion_tokens=20)],
    })
    ctx = run(agent, env)
    assert (work / "out.txt").read_text() == "data\nmore\n"
    r = assert_consistent(agent.logs_dir)
    ep = r["episode"]
    assert (ep["exit_reason"], ep["task_name"], ep["trial_name"]) == ("finished", "fix-git", "fix-git__AbC1234")
    assert ctx.n_input_tokens == 3200 and ctx.n_cache_tokens == 200 and ctx.n_output_tokens == 200
    assert ctx.model_usage[SUB].n_input_tokens == 700
    expected = (800 * 1.0 + 200 * 0.1 + 100 * 2.0 + 1500 * 1.0 + 50 * 2.0 + 700 * 0.1 + 50 * 0.2) / 1e6
    assert ctx.cost_usd == pytest.approx(expected) and ctx.metadata["exit_reason"] == "finished"
    assert "config" not in ctx.metadata
    ctx.model_dump_json()


@pytest.mark.parametrize("topology", ["single", "subagents"])
def test_cancel_kills_the_running_command(tmp_path, monkeypatch, topology):
    if topology == "single":
        scripts = {ORCH: [Reply("", [tc("bash", command="sleep 3; echo late > late.txt")])]}
    else:
        scripts = {ORCH: [Reply("", [tc("spawn_subagents", instructions=["run it"])])],
                   SUB: [Reply("", [tc("bash", command="sleep 3; echo late > late.txt")])]}
    agent, env, work = trial(tmp_path, monkeypatch, scripts, topology=topology)
    t0 = time.time()
    with pytest.raises(asyncio.TimeoutError):
        run(agent, env, timeout=1.5)
    assert time.time() - t0 < 10
    r = assert_consistent(agent.logs_dir)
    assert r["episode"]["exit_reason"] == "cancelled"
    assert r["episode"]["cancel_kill"] == {"ok": True, "n_killed": 1, "error": None}
    time.sleep(max(0.0, t0 + 4.5 - time.time()))
    assert not (work / "late.txt").exists()


def test_cancel_during_an_llm_call_unwinds_the_thread(tmp_path, monkeypatch):
    agent, env, _ = trial(tmp_path, monkeypatch, {ORCH: [Reply("never", latency=60)]})
    with pytest.raises(asyncio.TimeoutError):
        run(agent, env, timeout=1.0)
    r = assert_consistent(agent.logs_dir)
    assert r["episode"]["error"] is None  # the thread finalized first: cancelled, not the fallback
    assert r["episode"]["cancel_kill"]["n_killed"] == 0 and r["calls"][0]["error"] == "cancelled"
    assert agent.recorder.dropped == 0


def test_provider_and_billing_errors_raise_after_finalizing(tmp_path, monkeypatch):
    agent, env, _ = trial(tmp_path / "p", monkeypatch, {
        ORCH: [Reply("", [tc("spawn_subagents", instructions=["x"])]), Reply("", [tc("finish")])],
        SUB: [ProviderError("sub-model: HTTP 401: invalid key", status=401)],
    })
    with pytest.raises(ProviderUnavailableError) as ei:
        run(agent, env)
    assert isinstance(ei.value.__cause__, ProviderError)
    assert assert_consistent(agent.logs_dir)["episode"]["exit_reason"] == "provider_error"
    agent, env, _ = trial(tmp_path / "b", monkeypatch, {ORCH: [BillingError("HTTP 402: no balance", status=402)]})
    with pytest.raises(BillingError):
        run(agent, env)
    assert assert_consistent(agent.logs_dir)["episode"]["exit_reason"] == "billing_error"


def test_missing_config(tmp_path, monkeypatch):
    monkeypatch.delenv("SUBAGENTS_BENCH_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="SUBAGENTS_BENCH_CONFIG"):
        OrchestratorAgent(logs_dir=tmp_path)


def test_kill_script_kills_live_command_groups_only(tmp_path):
    root = tmp_path / "sb state"
    (root / "sub-1").mkdir(parents=True)
    (root / "orchestrator").mkdir()
    (root / "arb/arb-1").mkdir(parents=True)
    live = subprocess.Popen(["bash", "-c", "sleep 30 & sleep 30"], start_new_session=True)
    other = subprocess.Popen(["sleep", "30"], start_new_session=True)
    PROCS.extend([live, other])
    (root / "sub-1/pid_abc").write_text(f"{live.pid}\n")
    (root / "orchestrator/pid_def").write_text("not-a-pid\n")
    out = subprocess.run(["bash", "-c", kill_script(str(root))], capture_output=True, text=True, timeout=30)
    assert out.stdout.strip() == "__SB_KILLED 1" and not (root / "arb").exists()
    live.wait(timeout=5)
    assert live.returncode == -signal.SIGKILL and other.poll() is None


@pytest.mark.docker
def test_kill_script_in_a_linux_container():
    import threading

    from subagents_bench.tools import run_tool
    from test_tools import DockerEnv, ctx_for

    d = DockerEnv("ubuntu:24.04")
    try:
        got = {}
        cmd = "sleep 25 & sleep 25; echo late > /tmp/late"
        th = threading.Thread(target=lambda: got.update(out=run_tool(ctx_for(d, command_timeout_sec=60), "bash", {"command": cmd})))
        th.start()
        time.sleep(2)
        assert d.exec(kill_script(d.state_root), 30).stdout.strip() == "__SB_KILLED 1"
        th.join(timeout=10)
        assert got["out"].exit_code == 137 and not got["out"].timed_out
        probe = d.exec("ls /tmp/late 2>&1; for p in /proc/[0-9]*; do tr '\\0' ' ' < $p/cmdline; echo; done | grep -c '^sleep 25'", 10)
        assert probe.stdout.split()[-1] == "0" and "No such file" in probe.stdout
    finally:
        d.close()
