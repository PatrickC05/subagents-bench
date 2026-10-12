"""The orchestrator loop and spawn_subagents, end to end on LocalShell with
scripted fake models: exit reasons, step notes, grant rules and caps per
strategy, real concurrency, stop/billing/error handling, and record validity
(every file validates against schema.py and passes the analysis check-run)."""

from __future__ import annotations

import threading
import time

import pytest

from fakes import ORCH, SUB, FakeFactory, Reply, ScriptExhausted, assert_valid, make_config, records, run, tc
from subagents_bench.environment import LocalShellEnvironment
from subagents_bench.orchestrator import Orchestrator
from subagents_bench.telemetry import EpisodeRecorder
from subagents_bench.types import BillingError, LLMError, Message


def submit(report="done"):
    return Reply("", [tc("submit", report=report)])


# ---------------------------------------------------------------------------
# single topology and exit reasons
# ---------------------------------------------------------------------------


def test_single_topology_edits_and_finishes(tmp_path):
    cfg = make_config(tmp_path, topology="single")
    f = FakeFactory({ORCH: [
        Reply("look", [tc("bash", command="echo hi > a.txt && pwd"), tc("read_file", path="a.txt")], reasoning="plan A"),
        Reply("fix", [tc("edit_file", path="a.txt", old="hi", new="bye"), tc("finish", summary="changed a.txt")]),
    ]})
    res = run(cfg, f)
    work = tmp_path / "work"
    assert (work / "a.txt").read_text() == "bye\n"
    assert res.exit_reason == "finished" and res.final_summary == "changed a.txt"
    r = assert_valid(res.episode_dir)
    ep = r["episode"]
    assert ep["orchestrator_steps"] == 2 and ep["spawn_calls"] == 0 and ep["dispatches"] == 0
    assert ep["strategy"] is None and ep["subagent_model"] is None
    assert [t["name"] for t in r["tools"]] == ["bash", "read_file", "edit_file", "finish"]
    assert r["tools"][0]["cwd_after"] == str(work) and r["tools"][0]["exit_code"] == 0
    assert ep["files_modified"] == [str(work / "a.txt")] and ep["n_files_modified"] == 1
    # spawn_subagents is not offered without subagents; the step note is appended.
    req = f[ORCH].requests
    assert req[0]["tools"] == ["bash", "edit_file", "read_file", "list_dir", "search", "finish"]
    last_result = req[1]["messages"][-1].tool_results[-1].content
    assert last_result.endswith("[step 1/5 complete; 4 remaining]")
    # Reasoning is kept on the assistant message for passback.
    asst = req[1]["messages"][1]
    assert asst.role == "assistant" and asst.reasoning == "plan A"
    steps = r["steps"]
    assert steps[1]["finished"] and steps[1]["tool_names"] == ["edit_file", "finish"]
    assert steps[0]["llm_sec"] + steps[0]["tools_sec"] == pytest.approx(steps[0]["duration_sec"])


def test_max_steps_and_step_notes(tmp_path):
    cfg = make_config(tmp_path, topology="single", budget={"orchestrator_steps": 3})
    f = FakeFactory({ORCH: [Reply(f"s{i}", [tc("list_dir")]) for i in range(3)]})
    res = run(cfg, f)
    assert res.exit_reason == "max_steps" and res.final_summary == "s2"
    notes = [m.tool_results[-1].content.rsplit("\n", 1)[-1] for m in f[ORCH].requests[-1]["messages"] if m.role == "tool"]
    assert notes == ["[step 1/3 complete; 2 remaining]", "[step 2/3 complete; 1 remaining]"]
    assert assert_valid(res.episode_dir)["episode"]["orchestrator_steps"] == 3


def test_no_tool_call_ends_episode(tmp_path):
    cfg = make_config(tmp_path, topology="single")
    res = run(cfg, FakeFactory({ORCH: [Reply("I am done, nothing to do.")]}))
    assert res.exit_reason == "no_tool_call" and res.final_summary == "I am done, nothing to do."
    r = assert_valid(res.episode_dir)
    assert r["steps"][0]["n_tool_calls"] == 0 and r["tools"] == []


def test_tools_after_finish_are_not_run_and_unknown_tool_errors(tmp_path):
    cfg = make_config(tmp_path, topology="single")
    f = FakeFactory({ORCH: [
        Reply("", [tc("frobnicate", x=1)]),
        Reply("", [tc("finish", summary="s"), tc("bash", command="touch never.txt")]),
    ]})
    res = run(cfg, f)
    assert res.exit_reason == "finished"
    assert not (tmp_path / "work" / "never.txt").exists()
    r = assert_valid(res.episode_dir)
    assert [t["name"] for t in r["tools"]] == ["frobnicate", "finish"]
    assert r["tools"][0]["is_error"] and "unknown tool 'frobnicate'" in r["tools"][0]["error"]
    assert r["steps"][1]["tool_names"] == ["finish", "bash"]


def test_token_budget_cap(tmp_path):
    cfg = make_config(tmp_path, topology="single", budget={"total_token_budget": 1000})
    f = FakeFactory({ORCH: [Reply("", [tc("list_dir")], prompt_tokens=900, completion_tokens=200)] * 3})
    res = run(cfg, f)
    assert res.exit_reason == "token_budget"
    assert assert_valid(res.episode_dir)["episode"]["orchestrator_steps"] == 1


def test_timeout_safety_cap(tmp_path):
    cfg = make_config(tmp_path, topology="single", budget={"timeout_sec": 1})
    f = FakeFactory({ORCH: [Reply("", [tc("list_dir")], latency=1.1)] * 3})
    res = run(cfg, f)
    assert res.exit_reason == "timeout"
    assert assert_valid(res.episode_dir)["episode"]["orchestrator_steps"] == 1


def test_trials_get_separate_directories(tmp_path):
    cfg = make_config(tmp_path, topology="single")
    f = FakeFactory({ORCH: [Reply("", [tc("finish")]), Reply("", [tc("finish")])]})
    a, b = run(cfg, f, resolved=True), run(cfg, f)
    assert a.episode_dir != b.episode_dir
    assert a.episode_dir.parent == b.episode_dir.parent == tmp_path / "runs" / "cond" / "task-1"
    assert (a.record["reward"], a.record["resolved"]) == (1.0, True)
    assert (b.record["reward"], b.record["resolved"]) == (None, None)


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------


def test_billing_error_is_recorded_then_raised(tmp_path):
    cfg = make_config(tmp_path, topology="single")
    f = FakeFactory({ORCH: [Reply("", [tc("list_dir")]), BillingError("insufficient balance", status=402)]})
    with pytest.raises(BillingError):
        run(cfg, f)
    ep_dir = next((tmp_path / "runs" / "cond" / "task-1").iterdir())
    r = assert_valid(ep_dir)
    assert r["episode"]["exit_reason"] == "billing_error" and "insufficient balance" in r["episode"]["error"]
    assert r["calls"][-1]["error"].startswith("BillingError") and r["steps"][-1]["call_id"] is None


def test_billing_error_in_subagent_aborts_episode(tmp_path):
    cfg = make_config(tmp_path, subagent={"strategy": "sequential"})
    f = FakeFactory({
        ORCH: [Reply("", [tc("spawn_subagents", instructions=["a", "b"])])],
        SUB: [BillingError("quota exceeded", status=429)],
    })
    with pytest.raises(BillingError):
        run(cfg, f)
    r = assert_valid(next((tmp_path / "runs" / "cond" / "task-1").iterdir()))
    assert r["episode"]["exit_reason"] == "billing_error"
    assert [d["status"] for d in r["dispatches"]] == ["error", "cancelled"]
    assert r["dispatches"][1]["steps_used"] == 0 and "billing" in r["dispatches"][1]["error"]
    assert r["spawns"][0]["dispatch_ids"] == [d["dispatch_id"] for d in r["dispatches"]]


def test_provider_failure_ends_with_provider_error_without_raising(tmp_path):
    cfg = make_config(tmp_path, topology="single")
    down = LLMError("giving up after 9 attempts", status=503)
    f = FakeFactory({ORCH: [Reply("", [tc("list_dir")]), down]})
    res = run(cfg, f)
    # Its own outcome (not 'error'): the harbor agent raises ProviderUnavailableError from res.cause.
    assert res.exit_reason == "provider_error" and res.cause is down
    assert res.error == "LLMError: giving up after 9 attempts [HTTP 503]"  # class and status kept
    ep = assert_valid(res.episode_dir)["episode"]
    assert ep["exit_reason"] == "provider_error" and ep["error"] == res.error and ep["n_provider_errors"] == 1


def test_orchestrator_context_overflow_is_an_error_not_a_provider_failure(tmp_path):
    from subagents_bench.types import ContextLengthError

    cfg = make_config(tmp_path, topology="single")
    res = run(cfg, FakeFactory({ORCH: [ContextLengthError("prompt is too long", status=400)]}))
    assert res.exit_reason == "error" and res.cause is None
    assert res.error.startswith("ContextLengthError: prompt is too long")
    assert assert_valid(res.episode_dir)["episode"]["n_provider_errors"] == 0


def test_harness_exec_sec_is_the_union_of_concurrent_measurements(tmp_path, monkeypatch):
    # Parallel dispatches take their markers and scans concurrently: the sum
    # counts the overlap N times, the union (harness_exec_sec) once.
    from subagents_bench import workspace

    orig = workspace.make_marker

    def slow_marker(env, name):
        env.exec("sleep 0.3")  # through the same (metered) env as the marker itself
        return orig(env, name)

    monkeypatch.setattr(workspace, "make_marker", slow_marker)
    cfg = make_config(tmp_path, subagent={"strategy": "parallel", "max_concurrent": 3})
    f = FakeFactory({ORCH: [Reply("", [tc("spawn_subagents", instructions=["a", "b", "c"])]), Reply("", [tc("finish")])],
                     SUB: [Reply("", [tc("submit", report="r")])] * 3})
    res = run(cfg, f)
    ep = assert_valid(res.episode_dir)["episode"]
    # 5 slow markers (episode, group, 3 dispatches); the dispatches' three overlap.
    assert ep["harness_exec_sum_sec"] >= 5 * 0.3
    assert ep["harness_exec_sec"] < ep["harness_exec_sum_sec"] - 0.4
    assert ep["harness_exec_sec"] >= 3 * 0.3


def test_unexpected_exception_finalizes_then_raises(tmp_path):
    cfg = make_config(tmp_path, topology="single")
    f = FakeFactory({ORCH: [ValueError("bug in a client")]})
    with pytest.raises(ValueError):
        run(cfg, f)
    ep_dir = next((tmp_path / "runs" / "cond" / "task-1").iterdir())
    r = assert_valid(ep_dir)
    assert r["episode"]["exit_reason"] == "error" and "Traceback" in r["episode"]["error"]
    # Explicitly suppressed: the record is identical, nothing raises.
    res = run(cfg, FakeFactory({ORCH: [ValueError("bug")]}), raise_on_error=False)
    assert res.exit_reason == "error" and "ValueError: bug" in res.error


def test_stop_event_cancels_promptly(tmp_path):
    cfg = make_config(tmp_path)
    f = FakeFactory(
        {ORCH: [Reply("", [tc("spawn_subagents", instructions=["slow one", "slow two"])])],
         SUB: [Reply("", [tc("list_dir")], latency=0.05), Reply("", [tc("list_dir")], latency=0.05)] + [Reply("x", latency=30)] * 2},
    )
    env = LocalShellEnvironment(cfg.sandbox.workdir)
    env.workdir = cfg.sandbox.workdir
    stop = threading.Event()
    rec = EpisodeRecorder(tmp_path / "ep", config=cfg, task_name="t", instruction="i", workdir=cfg.sandbox.workdir)
    orch = Orchestrator(cfg, env, rec, stop_event=stop, client_factory=f)
    assert env.stop_event is stop
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("res", orch.run("go")))
    th.start()
    time.sleep(0.8)
    t0 = time.time()
    stop.set()
    th.join(timeout=10)
    assert not th.is_alive() and time.time() - t0 < 5
    res = out["res"]
    assert res.exit_reason == "cancelled"
    r = assert_valid(res.episode_dir)
    assert [d["status"] for d in r["dispatches"]] == ["cancelled", "cancelled"]
    assert any(c["error"] and c["error"].startswith("cancelled") for c in r["calls"])
    assert r["spawns"][0]["dispatch_ids"] == [d["dispatch_id"] for d in sorted(r["dispatches"], key=lambda d: d["dispatch_id"])]
    env.close()


# ---------------------------------------------------------------------------
# spawn_subagents: grants, strategies, concurrency
# ---------------------------------------------------------------------------


def test_parallel_runs_concurrently_and_respects_max_concurrent(tmp_path):
    cfg = make_config(tmp_path, subagent={"max_concurrent": 2})
    f = FakeFactory(
        {ORCH: [Reply("", [tc("spawn_subagents", instructions=["a", "b", "c", "d"])]), Reply("", [tc("finish")])],
         SUB: [submit("r")] * 2},
        latency={SUB: 0.4},
    )
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    sp = r["spawns"][0]
    assert (sp["requested"], sp["granted"], sp["cap_reason"], sp["concurrency"]) == (4, 2, "max_concurrent", 2)
    assert sp["instructions"] == ["a", "b", "c", "d"] and len(sp["dispatch_ids"]) == 2
    d1, d2 = r["dispatches"]
    assert d1["t_start"] < d2["t_end"] and d2["t_start"] < d1["t_end"]  # windows overlap
    assert f[SUB].max_in_flight == 2
    assert sp["duration_sec"] < sp["sum_dispatch_sec"]
    text = f[ORCH].requests[1]["messages"][-1].tool_results[0].content
    assert "ran 2 of 4 instruction(s)" in text and "instruction(s) 3, 4 were NOT run" in text
    assert "dispatch budget left: 8]" in text  # max_concurrent 2 * 5 steps - 2
    spawn_tool = [t for t in r["tools"] if t["name"] == "spawn_subagents"]
    assert len(spawn_tool) == 1 and spawn_tool[0]["group_id"] == sp["group_id"]
    # Allocated when the spawn started (before its members), written when it ended.
    assert spawn_tool[0]["seq"] < min(c["seq"] for c in r["calls"] if c["role"] == "subagent")
    assert spawn_tool[0]["n_exec"] >= 2  # group marker + files scan
    assert r["steps"][0]["spawned_group_ids"] == [sp["group_id"]]
    assert r["episode"]["instruction_following"] == pytest.approx(0.5)


def test_sequential_never_overlaps_and_keeps_order(tmp_path):
    cfg = make_config(tmp_path, subagent={"strategy": "sequential"})
    f = FakeFactory(
        {ORCH: [Reply("", [tc("spawn_subagents", instructions=["first", "second", "third"])]), Reply("", [tc("finish")])],
         SUB: lambda s, m, t, c: submit("report for " + m[0].content)},
        latency={SUB: 0.15},
    )
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    ds = r["dispatches"]
    assert [d["instruction"] for d in ds] == ["first", "second", "third"]
    for a, b in zip(ds, ds[1:], strict=False):
        assert a["t_end"] <= b["t_start"]
    assert f[SUB].max_in_flight == 1 and r["spawns"][0]["concurrency"] == 1
    assert ds[2]["queue_sec"] >= ds[0]["duration_sec"] + ds[1]["duration_sec"] - 0.05
    text = f[ORCH].requests[1]["messages"][-1].tool_results[0].content
    assert text.index("report for first") < text.index("report for second") < text.index("report for third")


def test_dispatch_budget_across_steps(tmp_path):
    cfg = make_config(tmp_path, subagent={"max_concurrent": 2, "max_dispatches": 3, "scope": "easy"})
    f = FakeFactory(
        {ORCH: [Reply("", [tc("spawn_subagents", instructions=["a", "b"])]),
                Reply("", [tc("spawn_subagents", instructions=["c", "d"])]),
                Reply("", [tc("spawn_subagents", instructions=["e"])]),
                Reply("", [tc("finish")])],
         SUB: [submit()] * 3},
    )
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    got = [(s["requested"], s["granted"], s["cap_reason"], s["dispatches_remaining_after"]) for s in r["spawns"]]
    assert got == [(2, 2, None, 1), (2, 1, "max_dispatches", 0), (1, 0, "max_dispatches", 0)]
    assert r["episode"]["dispatches"] == 3 and r["episode"]["redispatches"] == 1
    third = f[ORCH].requests[3]["messages"][-1].tool_results[0].content
    assert "ran 0 of 1" in third and "remaining dispatch budget" in third and third.endswith("dispatch budget left: 0]")


def test_invalid_spawn_arguments(tmp_path):
    cfg = make_config(tmp_path)
    f = FakeFactory({ORCH: [Reply("", [tc("spawn_subagents", instructions=[])]), Reply("", [tc("finish")])]})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    assert r["spawns"] == [] and r["tools"][0]["is_error"] and r["tools"][0]["group_id"] is None


def test_arbiter_grant_rule(tmp_path):
    cfg = make_config(tmp_path, subagent={"strategy": "arbiter", "scope": "easy", "max_concurrent": 3, "max_dispatches": 4})

    def orch(system, messages, tools, client):
        if "submit_judgement" in [t.name for t in tools]:
            return Reply("", [tc("submit_judgement", winner=0, confidence=1.0, agreement=True, rationale="same")])
        n_steps = sum(1 for m in messages if m.role == "assistant")
        if n_steps == 0:
            return Reply("", [tc("spawn_subagents", instructions=["x", "y"])])
        if n_steps == 1:
            return Reply("", [tc("spawn_subagents", instructions=["z"])])
        return Reply("", [tc("finish")])

    f = FakeFactory({ORCH: orch, SUB: [submit()] * 3})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    got = [(s["requested"], s["granted"], s["cap_reason"], s["concurrency"], len(s["dispatch_ids"])) for s in r["spawns"]]
    # 3 // 3 = 1 instruction per call (3 candidates); then 1 dispatch left < 3 candidates.
    assert got == [(2, 1, "max_concurrent", 3, 3), (1, 0, "max_dispatches", 0, 0)]
    assert r["spawns"][0]["dispatches_remaining_after"] == 1 and len(r["arbitrations"]) == 1


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def test_subagent_prompt_and_tools_follow_scope(tmp_path):
    cfg = make_config(tmp_path, subagent={"scope": "medium"})
    f = FakeFactory({ORCH: [Reply("", [tc("spawn_subagents", instructions=["check"])]), Reply("", [tc("finish")])],
                     SUB: [submit()]})
    res = run(cfg, f)
    assert_valid(res.episode_dir)
    sub_req = f[SUB].requests[0]
    assert sub_req["tools"] == ["read_file", "list_dir", "search", "bash", "submit"]
    assert "Do not modify project files" in sub_req["system"]
    orch_sys = f[ORCH].requests[0]["system"]
    assert "exactly 5 steps" in orch_sys and "Strategy: parallel" in orch_sys and "Scope: medium" in orch_sys
    assert "at most 3 instructions per spawn_subagents call" in orch_sys and "at most 15 subagent runs" in orch_sys
    assert "nohup" in orch_sys and "no TTY" in orch_sys


def test_script_exhaustion_is_visible(tmp_path):
    # A test whose script runs out ends the episode with an error, not silently.
    cfg = make_config(tmp_path, topology="single")
    res = run(cfg, FakeFactory({ORCH: [Reply("", [tc("list_dir")])]}))
    assert res.exit_reason == "error" and ScriptExhausted.__name__ in res.error


def test_orchestrator_messages_are_not_mutated_by_failed_calls(tmp_path):
    cfg = make_config(tmp_path, topology="single")
    f = FakeFactory({ORCH: [Reply("", [tc("finish")])]})
    res = run(cfg, f, instruction="do X")
    req = f[ORCH].requests[0]["messages"]
    assert req == [Message(role="user", content="do X")]
    assert records(res.episode_dir)["transcript"][0]["message_role"] == "system"


def test_cli_run_local_shell(tmp_path, monkeypatch, capsys):
    import yaml

    import subagents_bench.orchestrator as orch_mod
    from subagents_bench.cli import main

    cfg = make_config(tmp_path, topology="single")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg.to_dict()))
    f = FakeFactory({ORCH: [Reply("", [tc("bash", command="echo ok > cli.txt"), tc("finish", summary="s")])]})
    monkeypatch.setattr(orch_mod, "make_client", f)
    rc = main(["run", str(path), "--task-id", "demo", "--issue", "write cli.txt"])
    out = capsys.readouterr().out
    assert rc == 0 and "exit_reason=finished" in out
    assert (tmp_path / "work" / "cli.txt").read_text() == "ok\n"
    ep_dir = next((tmp_path / "runs" / "cond" / "demo").iterdir())
    assert str(ep_dir) in out
    assert assert_valid(ep_dir)["episode"]["instruction"] == "write cli.txt"
    # A harbor config cannot run locally.
    cfg2 = make_config(tmp_path, sandbox={"kind": "harbor", "workdir": "auto"})
    path.write_text(yaml.safe_dump(cfg2.to_dict()))
    assert main(["run", str(path), "--task-id", "demo", "--instruction", "x"]) == 2
    assert "local_docker or local_shell" in capsys.readouterr().err


def test_exploitbench_adapter_records_grade(tmp_path, monkeypatch):
    import json
    import subprocess

    from subagents_bench.harness import exploitbench_agent as eb

    box = tmp_path / "box"
    box.mkdir()
    monkeypatch.setattr(eb.CallableEnvironment, "state_root", str(tmp_path / "sb"))
    seen = []

    def exec_fn(command, timeout):
        seen.append(timeout)
        p = subprocess.run(["bash", "-c", command], cwd=box, capture_output=True, text=True, timeout=timeout)
        return p.stdout, p.stderr, p.returncode

    cfg = make_config(tmp_path, topology="single", benchmark="exploit_bench",
                      sandbox={"kind": "exploitbench", "workdir": "auto"})
    f = FakeFactory({ORCH: [Reply("", [tc("bash", command="echo pwned > flag"), tc("finish")])]})
    res = eb.run_exploitbench_trial(cfg, "cve-1", "exploit it", exec_fn,
                                    grade_fn=lambda: {"arbitrary_code_execution": True, "flags": 16},
                                    client_factory=f)
    assert (box / "flag").read_text() == "pwned\n"
    ep = assert_valid(res.episode_dir)["episode"]
    assert ep["sandbox_kind"] == "exploitbench" and ep["workdir"] == str(box.resolve())
    assert (ep["reward"], ep["resolved"]) == (1.0, True)
    assert json.loads((res.episode_dir / "exploitbench_grade.json").read_text()) == {"arbitrary_code_execution": True, "flags": 16}
    assert all(t is None or isinstance(t, int) for t in seen)


def test_client_construction_failure_is_recorded(tmp_path):
    cfg = make_config(tmp_path, topology="single")

    def factory(model_cfg):
        raise ValueError("provider 'vllm' needs an explicit base_url in the model config")

    with pytest.raises(ValueError):
        run(cfg, factory)
    ep_dir = next((tmp_path / "runs" / "cond" / "task-1").iterdir())
    ep = assert_valid(ep_dir)["episode"]
    assert ep["status"] == "done" and ep["exit_reason"] == "error" and "needs an explicit base_url" in ep["error"]


# ---------------------------------------------------------------------------
# unknown prices stay unknown
# ---------------------------------------------------------------------------


def test_unpriced_subagent_model_gives_unknown_cost_not_zero(tmp_path):
    # The subagent model has no price_in/price_out: its calls' cost_usd is None, and every
    # total that includes them is None; the priced orchestrator's own cost stays known.
    cfg = make_config(tmp_path, subagent={"model": {"model": SUB, "provider": "sail", "price_in": None, "price_out": None}})
    f = FakeFactory({
        ORCH: [Reply("", [tc("spawn_subagents", instructions=["look"])]), Reply("", [tc("finish")])],
        SUB: [submit("seen")],
    })
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)  # check-run accepts None costs
    sub_calls = [c for c in r["calls"] if c["role"] == "subagent"]
    assert sub_calls and all(c["cost_usd"] is None and c["price_in"] is None for c in sub_calls)
    orch_calls = [c for c in r["calls"] if c["role"] == "orchestrator"]
    assert all(isinstance(c["cost_usd"], float) and c["cost_usd"] > 0 for c in orch_calls)
    ep = r["episode"]
    assert ep["cost_usd"]["subagent"] is None and ep["cost_usd"]["total"] is None
    assert ep["cost_usd"]["orchestrator"] == pytest.approx(sum(c["cost_usd"] for c in orch_calls))
    assert ep["cost_usd"]["arbiter"] == 0.0  # no calls: an observed 0
    assert r["dispatches"][0]["cost_usd"] is None
    totals = res.recorder.role_totals()
    assert totals["total"]["cost_usd"] is None and totals["models"][SUB]["cost_usd"] is None
    assert totals["models"][ORCH]["cost_usd"] == pytest.approx(ep["cost_usd"]["orchestrator"])


def test_manager_orchestrator_has_only_its_configured_tools(tmp_path):
    # tools.orchestrator_tools restricts the orchestrator's own tools; a call to
    # a tool it was not given is an error, and the work goes through subagents.
    cfg = make_config(tmp_path, tools={"orchestrator_tools": ["read_file", "list_dir", "search"]})
    f = FakeFactory({
        ORCH: [
            Reply("", [tc("bash", command="echo hacked > a.txt")]),
            Reply("", [tc("spawn_subagents", instructions=["create a.txt containing ok"])]),
            Reply("", [tc("read_file", path="a.txt"), tc("finish", summary="done")]),
        ],
        SUB: [Reply("", [tc("bash", command="echo ok > a.txt")]), submit("wrote a.txt")],
    })
    res = run(cfg, f)
    work = tmp_path / "work"
    assert (work / "a.txt").read_text() == "ok\n"
    assert res.exit_reason == "finished"
    r = assert_valid(res.episode_dir)
    req = f[ORCH].requests
    assert req[0]["tools"] == ["read_file", "list_dir", "search", "spawn_subagents", "finish"]
    bash_rec = r["tools"][0]
    assert bash_rec["name"] == "bash" and bash_rec["is_error"] and "unknown tool 'bash'" in bash_rec["error"]
    orch_sys = req[0]["system"]
    assert "You cannot run commands or change files yourself" in orch_sys
    assert "- `bash`" not in orch_sys and "- `edit_file`" not in orch_sys
    assert "Subagents have these tools:" in orch_sys and "same tools as you" not in orch_sys
    # The subagent keeps the full hard-scope tool set.
    assert f[SUB].requests[0]["tools"] == ["read_file", "list_dir", "search", "bash", "edit_file", "submit"]
    assert r["episode"]["config"]["tools"]["orchestrator_tools"] == ["read_file", "list_dir", "search"]
    assert r["episode"]["dispatches"] == 1
