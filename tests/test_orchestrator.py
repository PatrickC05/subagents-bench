"""End-to-end tests of the agent loop with scripted fake LLM clients.

No network or Docker: agents run against a LocalShellEnvironment in a temp dir,
and the LLM is a FakeClient that replays a fixed list of responses. This
exercises the tool loop, subagent dispatch/re-dispatch accounting, budget caps,
instruction-following logging, and telemetry.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from subagents_bench.config import (
    BudgetConfig,
    ExperimentConfig,
    ModelConfig,
    SandboxConfig,
    SubagentConfig,
    SubtaskType,
    Topology,
)
from subagents_bench.environment import LocalShellEnvironment
from subagents_bench.llm import LLMResponse, ToolCall
from subagents_bench.telemetry import Usage
import subagents_bench.orchestrator as orch_mod


class FakeClient:
    """Replays scripted responses. Keyed by model name so orchestrator and
    subagent clients can be scripted independently."""

    scripts: dict[str, list] = {}

    def __init__(self, cfg):
        self.cfg = cfg
        self._it = iter(FakeClient.scripts.get(cfg.model, []))

    def complete(self, system, messages, tools):
        try:
            text, tool_calls, usage = next(self._it)
        except StopIteration:
            text, tool_calls, usage = ("done", [], Usage(10, 5))
        import time

        t0 = time.time()
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason="tool_use" if tool_calls else "end_turn",
            cost_usd=self.cfg.price_in * usage.input_tokens / 1e6
            + self.cfg.price_out * usage.output_tokens / 1e6,
            t_start=t0,
            t_end=t0 + 0.01,
        )


_ids = itertools.count()


def _tc(name, args):
    return ToolCall(id=f"call-{next(_ids)}", name=name, arguments=args)


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    monkeypatch.setattr(orch_mod, "make_client", lambda cfg: FakeClient(cfg))
    FakeClient.scripts = {}
    yield
    FakeClient.scripts = {}


def _config(tmp_path, topology=Topology.SUBAGENTS, sub_model="small-sub"):
    return ExperimentConfig(
        name="test",
        topology=topology,
        subtask_type=SubtaskType.PARALLEL_EXPLORATION,
        orchestrator=ModelConfig(model="orch", provider="anthropic", price_in=1, price_out=2),
        subagent=SubagentConfig(
            model=ModelConfig(model=sub_model, provider="vllm", price_in=0, price_out=0),
            max_concurrent=2,
            max_steps=3,
        ),
        budget=BudgetConfig(orchestrator_steps=5, total_token_budget=10_000_000, timeout_sec=600),
        sandbox=SandboxConfig(kind="local_shell", workdir=str(tmp_path)),
        output_dir=str(tmp_path / "runs"),
    )


def test_single_agent_finishes_and_edits(tmp_path):
    cfg = _config(tmp_path, topology=Topology.SINGLE)
    FakeClient.scripts["orch"] = [
        ("writing file", [_tc("bash", {"command": "echo hi > a.txt"})], Usage(100, 20)),
        ("done", [_tc("finish", {"summary": "created a.txt"})], Usage(80, 10)),
    ]
    env = LocalShellEnvironment(str(tmp_path))
    from subagents_bench.runner import run_task

    t = run_task(cfg, "task-1", "create a.txt", env, resolved=True)

    assert (tmp_path / "a.txt").exists()
    assert t.exit_reason == "finished"
    assert t.orchestrator_steps == 2
    assert t.subagent_dispatches == 0
    assert t.total_tokens == 100 + 20 + 80 + 10
    assert t.cost_usd > 0


def test_subagent_dispatch_and_redispatch_accounting(tmp_path):
    cfg = _config(tmp_path)
    FakeClient.scripts["orch"] = [
        # Step 1: spawn 2 subagents in parallel.
        ("delegate", [_tc("spawn_subagents", {"instructions": ["look at X", "look at Y"]})], Usage(200, 40)),
        # Step 2: spawn 1 more (this dispatch -> re-dispatch).
        ("delegate again", [_tc("spawn_subagents", {"instructions": ["fix Z"]})], Usage(150, 30)),
        # Step 3: finish.
        ("done", [_tc("finish", {"summary": "fixed"})], Usage(90, 10)),
    ]
    FakeClient.scripts["small-sub"] = [
        ("report", [_tc("submit", {"report": "found it"})], Usage(50, 20)),
    ] * 10

    env = LocalShellEnvironment(str(tmp_path))
    from subagents_bench.runner import run_task

    t = run_task(cfg, "task-2", "fix the bug", env, resolved=False)

    assert t.subagent_dispatches == 3
    # First dispatch is not a re-dispatch; the next two are.
    assert t.subagent_redispatches == 2
    assert t.exit_reason == "finished"
    assert t.subagent_usage.total_tokens == 3 * (50 + 20)
    # Instruction following: requested 2 then 1, both fully granted.
    assert t.instruction_following() == 1.0
    assert t.spawn_log == [
        {"step": 1, "requested": 2, "granted": 2},
        {"step": 2, "requested": 1, "granted": 1},
    ]


def test_concurrency_cap_limits_grant(tmp_path):
    cfg = _config(tmp_path)  # max_concurrent=2
    FakeClient.scripts["orch"] = [
        ("delegate", [_tc("spawn_subagents", {"instructions": ["a", "b", "c", "d"]})], Usage(100, 10)),
        ("done", [_tc("finish", {})], Usage(50, 5)),
    ]
    FakeClient.scripts["small-sub"] = [("r", [_tc("submit", {"report": "ok"})], Usage(5, 5))] * 10

    env = LocalShellEnvironment(str(tmp_path))
    from subagents_bench.runner import run_task

    t = run_task(cfg, "task-3", "do work", env)
    # Requested 4, but max_concurrent caps the grant at 2.
    assert t.spawn_log[0] == {"step": 1, "requested": 4, "granted": 2}
    assert t.instruction_following() == 0.5
    assert t.subagent_dispatches == 2


def test_subagents_disabled_in_single_topology(tmp_path):
    cfg = _config(tmp_path, topology=Topology.SINGLE)
    FakeClient.scripts["orch"] = [
        ("try delegate", [_tc("spawn_subagents", {"instructions": ["x"]})], Usage(10, 10)),
        ("done", [_tc("finish", {})], Usage(10, 10)),
    ]
    env = LocalShellEnvironment(str(tmp_path))
    from subagents_bench.runner import run_task

    t = run_task(cfg, "task-4", "solo", env)
    assert t.subagent_dispatches == 0


def test_telemetry_written(tmp_path):
    cfg = _config(tmp_path, topology=Topology.SINGLE)
    FakeClient.scripts["orch"] = [("done", [_tc("finish", {})], Usage(10, 10))]
    env = LocalShellEnvironment(str(tmp_path))
    from subagents_bench.runner import run_task

    t = run_task(cfg, "task-5", "x", env, resolved=True)
    out = Path(cfg.output_dir) / "test" / "task-5" / "telemetry.json"
    assert out.exists()
    assert t.wall_clock_sec is not None and t.wall_clock_sec >= 0
