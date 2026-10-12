"""Orchestrator and subagents end to end on the host shell with scripted fake models."""

from __future__ import annotations

import threading
import time

from fakes import ORCH, SUB, FakeFactory, Reply, assert_consistent, make_config, run_local, tc
from subagents_bench.llm import BillingError, ContextLengthError, ProviderError
from subagents_bench.prompts import LAST_STEP_NOTE, LENGTH_CUTOFF_NOTE, LENGTH_CUTOFF_REPLY


def go(tmp_path, scripts, latency=0.0, **overrides):
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    f = FakeFactory(scripts, latency)
    exit_reason, rec = run_local(make_config(**overrides), f, work, tmp_path / "ep")
    return exit_reason, assert_consistent(tmp_path / "ep"), f


def tool_texts(request) -> list[str]:
    return [r.content for m in request["messages"] if m.role == "tool" for r in m.tool_results]


def test_single_agent_edits_and_finishes(tmp_path):
    exit_reason, r, f = go(tmp_path, {ORCH: [
        Reply("", [tc("bash", command="echo hi > a.txt"), tc("edit_file", path="a.txt", old="hi", new="bye")]),
        Reply("done", [tc("finish", summary="ok"), tc("bash", command="touch never")]),
    ]}, topology="single")
    assert exit_reason == "finished" and (tmp_path / "work/a.txt").read_text() == "bye\n"
    assert not (tmp_path / "work/never").exists()
    assert tool_texts(f[ORCH].requests[1])[-1].endswith("[step 1/5 complete; 4 remaining]")
    assert "spawn_subagents" not in f[ORCH].requests[0]["tools"] and "Subagents" not in f[ORCH].requests[0]["system"]
    ep = r["episode"]
    assert (ep["orchestrator_steps"], ep["strategy"], ep["subagent_model"]) == (2, None, None)
    assert ep["tokens_by_role"]["orchestrator"]["prompt"] == 200 and [s["finished"] for s in r["steps"]] == [False, True]
    assert [t["name"] for t in r["tools"]] == ["bash", "edit_file", "finish"]
    assert r["transcript"][0]["msg_role"] == "system" and r["transcript"][1]["content"] == "solve the task"


def test_max_steps_no_tool_call_and_dispatch_budget_note(tmp_path):
    exit_reason, r, f = go(tmp_path, {ORCH: [Reply("", [tc("list_dir")])] * 2}, budget={"orchestrator_steps": 2})
    assert exit_reason == "max_steps" and tool_texts(f[ORCH].requests[1])[-1].endswith(
        "[step 1/2 complete; 1 remaining; dispatch budget left: 15]")
    exit_reason, r, _ = go(tmp_path / "b", {ORCH: [Reply("I give up.")]})
    assert exit_reason == "no_tool_call" and r["episode"]["orchestrator_steps"] == 1


def test_cut_off_reply_uses_the_step_and_continues(tmp_path):
    exit_reason, r, f = go(tmp_path, {ORCH: [
        Reply("", finish_reason="length", reasoning="long thoughts"),
        Reply("", [tc("finish")]),
    ]})
    assert exit_reason == "finished" and r["episode"]["orchestrator_steps"] == 2
    asst, note = f[ORCH].requests[1]["messages"][-2:]
    assert asst.content == LENGTH_CUTOFF_REPLY and asst.reasoning == "long thoughts"
    assert note.role == "user" and note.content.startswith(LENGTH_CUTOFF_NOTE) and "[step 1/5" in note.content


def test_manager_has_only_read_tools_and_must_delegate(tmp_path):
    exit_reason, r, f = go(tmp_path, {ORCH: [Reply("", [tc("bash", command="ls")]), Reply("", [tc("finish")])]},
                           tools={"orchestrator_tools": ["read_file", "list_dir", "search"]})
    req = f[ORCH].requests[0]
    assert req["tools"] == ["read_file", "list_dir", "search", "spawn_subagents", "finish"]
    assert "must be delegated" in req["system"] and "`bash`: run" not in req["system"]
    assert tool_texts(f[ORCH].requests[1])[0].startswith("[error] unknown tool 'bash'")


def test_default_prompt_is_neutral_about_delegation(tmp_path):
    _, _, f = go(tmp_path, {ORCH: [Reply("", [tc("finish")])]})
    system = f[ORCH].requests[0]["system"]
    assert "separate model instances" in system and "one spawn_subagents call uses one of your steps" in system.lower()
    assert "must be delegated" not in system and SUB not in system


def test_parallel_spawn_runs_concurrently_within_the_grant(tmp_path):
    def sub(system, messages, tools):
        if len(messages) == 1:
            n = messages[0].content[-1]
            return Reply("", [tc("bash", command=f"echo {n} > f{n}.txt")])
        return Reply("", [tc("submit", report="did " + messages[0].content)])

    exit_reason, r, f = go(tmp_path, {
        ORCH: [Reply("", [tc("spawn_subagents", instructions=["write 1", "write 2", "write 3"])]), Reply("", [tc("finish")])],
        SUB: sub,
    }, latency=0.3, subagent={"max_concurrent": 2})
    assert exit_reason == "finished" and f[SUB].max_in_flight == 2
    (spawn,) = r["spawns"]
    assert (spawn["requested"], spawn["granted"], spawn["strategy"]) == (3, 2, "parallel")
    result = tool_texts(f[ORCH].requests[1])[0]
    assert "ran 2 of 3" in result and "were NOT run" in result and "did write 1" in result and "dispatch budget left: 13" in result
    assert sorted(d["status"] for d in r["dispatches"]) == ["submitted", "submitted"]
    # Parallel siblings share the workdir, so each one's scan also sees the other's file.
    assert {p.rsplit("/", 1)[1] for d in r["dispatches"] for p in d["files_modified"]} == {"f1.txt", "f2.txt"}
    assert r["tools"][0]["group_id"] == spawn["group_id"] == r["steps"][0]["group_ids"][0]


def test_sequential_runs_in_order_one_at_a_time(tmp_path):
    def sub(system, messages, tools):
        return Reply("", [tc("bash", command=f"echo {messages[0].content} >> log; cat log")]) if len(messages) == 1 \
            else Reply("", [tc("submit", report="ok")])

    _, r, f = go(tmp_path, {ORCH: [Reply("", [tc("spawn_subagents", instructions=["a", "b", "c"])]), Reply("", [tc("finish")])],
                            SUB: sub}, latency=0.1, subagent={"strategy": "sequential"})
    assert f[SUB].max_in_flight == 1 and (tmp_path / "work/log").read_text() == "a\nb\nc\n"
    ds = r["dispatches"]
    assert [d["instruction_index"] for d in ds] == [0, 1, 2] and ds[0]["t_end"] <= ds[1]["t_start"]


def test_dispatch_budget_across_steps(tmp_path):
    spawn = Reply("", [tc("spawn_subagents", instructions=["x", "y"])])
    _, r, f = go(tmp_path, {ORCH: [spawn, spawn, spawn, Reply("", [tc("finish")])], SUB: lambda *a: Reply("r")},
                 subagent={"max_dispatches": 3})
    assert [s["granted"] for s in r["spawns"]] == [2, 1, 0] and len(r["dispatches"]) == 3


def test_arbiter_grant_is_per_call_and_counts_every_candidate(tmp_path):
    spawn = Reply("", [tc("spawn_subagents", instructions=["a", "b"])])
    verdict = '{"winner": 0, "confidence": 1, "agreement": true}'

    def orch(system, messages, tools):
        return Reply(verdict) if system.startswith("You are an impartial") else orch_steps.pop(0)

    orch_steps = [spawn, spawn, Reply("", [tc("finish")])]
    _, r, f = go(tmp_path, {ORCH: orch, SUB: lambda *a: Reply("report")},
                 subagent={"strategy": "arbiter", "scope": "easy", "max_dispatches": 4})
    assert [s["granted"] for s in r["spawns"]] == [1, 0] and len(r["dispatches"]) == 3
    assert "at most 1 instruction(s) per call" in f[ORCH].requests[0]["system"]


def test_subagent_scope_cut_off_and_last_step_note(tmp_path):
    _, r, f = go(tmp_path, {
        ORCH: [Reply("", [tc("spawn_subagents", instructions=["look"])]), Reply("", [tc("finish")])],
        SUB: [Reply("", [tc("bash", command="ls")]), Reply("", finish_reason="length"),
              Reply("found it", [tc("list_dir")]), Reply("partial", [tc("search", pattern="x")])],
    }, subagent={"scope": "easy"})
    reqs = f[SUB].requests
    assert reqs[0]["tools"] == ["read_file", "list_dir", "search", "submit"]
    assert "not available to you" in tool_texts(reqs[1])[0]
    assert reqs[2]["messages"][-1].content.startswith(LENGTH_CUTOFF_NOTE)
    assert tool_texts(reqs[3])[-1].endswith(LAST_STEP_NOTE)
    (d,) = r["dispatches"]
    assert (d["status"], d["steps_used"], d["files_modified"]) == ("max_steps", 4, [])
    assert "partial" in tool_texts(f[ORCH].requests[1])[0]  # the last text is the report


def test_tools_after_submit_are_not_run(tmp_path):
    _, r, f = go(tmp_path, {
        ORCH: [Reply("", [tc("spawn_subagents", instructions=["go"])]), Reply("", [tc("finish")])],
        SUB: [Reply("", [tc("submit", report="R"), tc("bash", command="touch late")])],
    })
    assert not (tmp_path / "work/late").exists() and r["dispatches"][0]["status"] == "submitted"


def test_subagent_provider_failure_stops_the_episode_at_once(tmp_path):
    def sub(system, messages, tools):
        if "fail" in messages[0].content:
            return ProviderError("sub-model: giving up after 5 attempt(s)", status=503, n_attempts=5)
        return Reply("slow", latency=30)

    t0 = time.time()
    exit_reason, r, f = go(tmp_path, {ORCH: [Reply("", [tc("spawn_subagents", instructions=["fail", "slow"])])], SUB: sub})
    assert exit_reason == "provider_error" and time.time() - t0 < 5 and len(f[ORCH].requests) == 1
    ep = r["episode"]
    assert ep["n_provider_errors"] == 1 and ep["error"].startswith("ProviderError: sub-model: giving up")
    assert sorted(d["status"] for d in r["dispatches"]) == ["cancelled", "error"]
    errors = {c["error"]: c for c in r["calls"] if c["error"]}
    assert set(errors) == {"cancelled", "ProviderError: sub-model: giving up after 5 attempt(s)"}
    assert errors["ProviderError: sub-model: giving up after 5 attempt(s)"]["n_attempts"] == 5


def test_orchestrator_errors(tmp_path):
    cases = [(BillingError("HTTP 402: no balance", status=402), "billing_error"),
             (ProviderError("down"), "provider_error"),
             (ContextLengthError("HTTP 400: too long", status=400), "error")]
    for i, (err, expected) in enumerate(cases):
        exit_reason, r, _ = go(tmp_path / str(i), {ORCH: [Reply("", [tc("list_dir")]), err]})
        assert exit_reason == expected and r["episode"]["error"].startswith(type(err).__name__)
        assert r["calls"][-1]["error"].startswith(type(err).__name__) and r["steps"][-1]["call_id"] is None


def test_stop_event_cancels_promptly(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    f = FakeFactory({ORCH: [Reply("", [tc("bash", command="sleep 20")])]})
    from subagents_bench.environment import LocalShellEnvironment
    from subagents_bench.orchestrator import Orchestrator
    from subagents_bench.records import EpisodeRecorder

    env, rec = LocalShellEnvironment(str(work)), EpisodeRecorder(tmp_path / "ep", make_config(), task_name="t")
    threading.Timer(1.0, rec.stop_event.set).start()
    t0 = time.time()
    assert Orchestrator(make_config(), env, rec, str(work), f).run("x") == "cancelled"
    assert time.time() - t0 < 5
    r = assert_consistent(tmp_path / "ep")
    assert r["episode"]["error"] is None and r["tools"][0]["is_error"]
    env.close()
