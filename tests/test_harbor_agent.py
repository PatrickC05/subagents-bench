"""OrchestratorAgent driven like harbor 0.24.0 drives it: a real ``BaseAgent``
subclass constructed with harbor's kwargs, ``setup``/``run`` awaited on an
event loop with a real ``AgentContext``, against a fake async environment whose
``exec`` coroutine runs bash locally in a temp dir. Covers the records landing
in ``logs_dir``, progressive AgentContext updates, harbor-style cancellation
(``asyncio.wait_for`` timeout) and BillingError propagation."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from fakes import ORCH, SUB, FakeFactory, Reply, assert_valid, tc
from subagents_bench.harness import harbor_agent
from subagents_bench.harness.harbor_agent import OrchestratorAgent
from subagents_bench.types import BillingError

HAS_HARBOR = importlib.util.find_spec("harbor") is not None
if HAS_HARBOR:
    from harbor.models.agent.context import AgentContext, ModelUsage

pytestmark_harbor = pytest.mark.skipif(not HAS_HARBOR, reason="harbor is not installed")


@dataclass
class _Res:
    stdout: str | None
    stderr: str | None
    return_code: int


class FakeHarborEnv:
    """Mimics harbor's async ``BaseEnvironment.exec`` (docker backend
    semantics: default cwd = the image WORKDIR; timeouts raise RuntimeError)."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.n_exec = 0

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.n_exec += 1
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command, cwd=cwd or str(self.workdir),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds") from None
        except asyncio.CancelledError:
            proc.kill()
            raise
        return _Res(out.decode(), err.decode(), proc.returncode)


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
    assert ep["harbor_version"] == "0.24.0" and ep["git_commit"] and isinstance(ep["git_dirty"], bool)
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
    r = assert_valid(logs)
    assert r["calls"][-1]["error"].startswith("cancelled") and r["steps"][-1]["call_id"] is None
    assert ctx.metadata["exit_reason"] == "cancelled"
    # Records written after finalization (none expected) would be dropped, not appended.
    assert agent.recorder.finalized and agent.recorder.dropped == 0


@pytestmark_harbor
def test_harbor_cancellation_during_a_command(tmp_path, monkeypatch):
    logs, work, env = trial(tmp_path, monkeypatch, {"topology": "single"})
    ctx = AgentContext()
    f = FakeFactory({ORCH: [Reply("", [tc("bash", command="sleep 30; echo late > late.txt")])]})
    agent = make_agent(logs, f)
    t0 = time.time()

    async def main():
        await agent.setup(env)
        await asyncio.wait_for(agent.run("task", env, ctx), timeout=1.5)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(main())
    assert time.time() - t0 < 10 and agent._run_future.done()
    r = assert_valid(logs)
    assert r["episode"]["exit_reason"] == "cancelled"
    bash = [t for t in r["tools"] if t["name"] == "bash"][0]
    assert bash["is_error"] and bash["error"] == "cancelled" and bash["duration_sec"] < 5
    time.sleep(0.2)
    assert not (work / "late.txt").exists()


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
