"""Interactive episodes: timed hints from SWE-sweep subtasks, background subagents, ``wait``,
the deadline, and the interactivity analysis, end to end on the host shell with fake models."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fakes import ORCH, SUB, FakeFactory, Reply, assert_consistent, make_config, merge, tc
from subagents_bench.analysis.__main__ import main as analyze
from subagents_bench.analysis.interactivity import (condition_table, hint_rows, names_file, position_table,
                                                    tool_touches)
from subagents_bench.analysis.latency import critical_path
from subagents_bench.analysis.load import load_trial
from subagents_bench.config import ConfigError
from subagents_bench.environment import LocalShellEnvironment
from subagents_bench.hints import (HintError, Subtask, generate_reports, hint_text, load_reports, load_subtasks, n_slots,
                                   ordered, schedule, write_reports)
from subagents_bench.interactive import InteractiveOrchestrator
from subagents_bench.plots.__main__ import main as plot
from subagents_bench.prompts import clock, interactive_note, interactive_prompt
from subagents_bench.records import EpisodeRecorder

FAST = {"interactive": {"wall_clock_sec": 4.0, "interval_sec": 0.5}, "budget": {"orchestrator_steps": 80}}


def make_task(root: Path, n: int = 4, files: list[list[str]] | None = None) -> Path:
    d = root / "task-1"
    (d / "tests").mkdir(parents=True)
    subtasks = [{"id": f"b{i}", "f2p": [f"tests.test_m::test_bug_number_{i}[p0]", f"tests.test_m::test_bug_number_{i}[p1]"],
                 "target_files": files[i] if files else [f"src/m{i}.py"]} for i in range(n)]
    (d / "tests" / "swesweep_eval.json").write_text(json.dumps({"subtasks": subtasks}))
    for s in subtasks:
        (d / "tests" / "subtasks" / s["id"]).mkdir(parents=True)
        (d / "tests" / "subtasks" / s["id"] / "test.patch").write_text(f"+def test_bug_number_{s['id']}(): ...\n")
    (d / "task.toml").write_text("[agent]\ntimeout_sec = 43200\n")
    return d


def session(tmp_path, scripts, n_hints=4, latency=0.0, **overrides):
    """One interactive episode in ``tmp_path/ep`` -> (exit, records, factory, hints, seconds)."""
    cfg = make_config(**merge(FAST, overrides))
    task = make_task(tmp_path, max(n_hints, 1))
    hints = schedule(cfg.interactive, task, "task-1")[:n_hints]
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True, exist_ok=True)
    f = FakeFactory(scripts, latency)
    env = LocalShellEnvironment(os.path.realpath(work))
    rec = EpisodeRecorder(tmp_path / "ep", cfg, task_name="task-1")
    t0 = time.time()
    try:
        exit_reason = InteractiveOrchestrator(cfg, env, rec, env.workdir, hints, f).run("Fix every bug.")
    finally:
        env.close()
    return exit_reason, assert_consistent(tmp_path / "ep"), f, hints, time.time() - t0


def hint_of(messages) -> str | None:
    last = messages[-1]
    return last.content if last.role == "user" and "Hint:" in (last.content or "") else None


def target(text: str) -> str:
    return text.split("`")[1]  # the hint's first file


def delegating(sub_sleep: float = 0.3):
    """Orchestrator: one background subagent per hint, naming its file; otherwise wait. Subagent:
    sleep, edit the file, submit."""
    def orch(system, messages, tools):
        h = hint_of(messages)
        if h:  # one subagent per hint in the message
            return Reply("on it", [tc("spawn_subagents", instructions=[f"Fix the bug in {target(x)}"
                                                                       for x in h.split("Hint: ")[1:]])])
        return Reply("", [tc("wait")])

    def sub(system, messages, tools):
        path = messages[0].content.split()[-1]
        if len(messages) == 1:
            return Reply("", [tc("bash", command=f"sleep {sub_sleep}"), tc("edit_file", path=path, old="", new="x\n")])
        return Reply("", [tc("submit", report=f"fixed {path}")])

    return {ORCH: orch, SUB: sub}


def hints_in(r) -> list[dict]:
    return [x for x in r["inbox"] if x["kind"] == "hint"]


# --- the session ---------------------------------------------------------------------------------


def test_background_session_delivers_hints_and_reports_while_subagents_run(tmp_path):
    exit_reason, r, f, hints, _ = session(tmp_path, delegating(), subagent={"spawn_mode": "background"})
    assert exit_reason == "idle"
    first = f[ORCH].requests[0]
    assert first["messages"][0].content.startswith("Fix every bug.\n\n---\nIn this session you do not search the "
                                                   "repository for bugs on your own")
    assert "every rule above still applies.\n\nHint: There is a bug in `src/m" in first["messages"][0].content
    assert "wait" in first["tools"] and "finish" not in first["tools"] and "spawn_subagents" in first["tools"]
    assert "starts one subagent per instruction in the background" in first["system"]
    hs = hints_in(r)
    assert [h["hint_index"] for h in hs] == [0, 1, 2, 3] and all(h["t_delivered"] is not None for h in hs)
    assert all(h["t_delivered"] - h["t_ready"] < 0.3 for h in hs)  # the orchestrator was waiting, not busy
    assert [round(h["t_ready"] - r["episode"]["session_t0"], 3) for h in hs] == [0.0, 0.5, 1.0, 1.5]
    reports = [x for x in r["inbox"] if x["kind"] == "report"]
    assert len(reports) == 4 and all(x["status"] == "submitted" for x in reports)
    assert {s["spawn_mode"] for s in r["spawns"]} == {"background"}
    spawn_tools = [x for x in r["tools"] if x["name"] == "spawn_subagents"]
    assert all(x["t_end"] - x["t_start"] < 0.2 for x in spawn_tools)  # returns at once
    d1 = r["dispatches"][0]
    assert d1["t_end"] - d1["t_start"] >= 0.3 and d1["instruction"].startswith("Fix the bug in src/m")
    overlapping = [c for c in r["calls"] if c["role"] == "orchestrator" and d1["t_start"] < c["t_start"] < d1["t_end"]]
    assert overlapping  # the orchestrator kept going while its subagent ran
    edits = [x for x in r["tools"] if x["name"] == "edit_file"]
    assert edits and edits[0]["arguments"]["path"].startswith("src/m") and edits[0]["tool_call_id"]
    user_texts = [m["content"] for m in r["transcript"] if m["role"] == "orchestrator" and m["msg_role"] == "user"]
    assert any("Message from the user:\nAnother one, while you keep working" in t for t in user_texts)
    assert any("Report from subagent sub-1 (submitted, 2 steps) on: Fix the bug in" in t for t in user_texts)
    ep = r["episode"]
    assert (ep["n_hints"], ep["n_hints_delivered"], ep["n_elided_results"]) == (4, 4, 0)
    assert all((tmp_path / "work" / h.subtask.target_files[0]).read_text() == "x\n" for h in hints)
    assert critical_path(r)["tool"] <= ep["wall_clock_sec"]


def test_single_agent_hint_waits_for_its_own_command(tmp_path):
    def orch(system, messages, tools):
        if len(messages) == 1:
            return Reply("", [tc("bash", command="sleep 1.2")])
        return Reply("", [tc("wait")])

    exit_reason, r, f, _, _ = session(tmp_path, {ORCH: orch}, n_hints=2, topology="single")
    assert exit_reason == "idle"
    assert "spawn_subagents" not in f[ORCH].requests[0]["tools"] and "Subagents" not in f[ORCH].requests[0]["system"]
    h1 = hints_in(r)[1]
    assert h1["t_delivered"] - h1["t_ready"] > 0.5 and h1["step"] == 2  # queued behind the 1.2 s command
    note = f[ORCH].requests[1]["messages"][-2].tool_results[-1].content
    assert note.endswith("of 0:04 elapsed]")


def test_blocking_spawn_holds_the_hint_until_the_subagents_report(tmp_path):
    scripts = delegating(sub_sleep=1.2)
    exit_reason, r, f, _, _ = session(tmp_path, scripts, n_hints=2)
    assert exit_reason == "idle" and {s["spawn_mode"] for s in r["spawns"]} == {"blocking"}
    h1 = hints_in(r)[1]
    assert h1["t_delivered"] - h1["t_ready"] > 0.5
    assert not [x for x in r["inbox"] if x["kind"] == "report"]  # reports came back as the spawn's result
    assert "your step waits for all of them" in f[ORCH].requests[0]["system"]


def test_deadline_cancels_background_subagents(tmp_path):
    def orch(system, messages, tools):
        if len(messages) == 1:
            return Reply("", [tc("spawn_subagents", instructions=["run the long suite"])])
        return Reply("", [tc("wait")])

    sub = [Reply("", [tc("bash", command="sleep 30")])]
    exit_reason, r, _, _, sec = session(tmp_path, {ORCH: orch, SUB: sub}, n_hints=3,
                                        interactive={"wall_clock_sec": 1.5, "end_when_idle": False},
                                        subagent={"spawn_mode": "background"})
    assert exit_reason == "deadline" and r["episode"]["error"] is None and sec < 10
    assert r["dispatches"][0]["status"] == "cancelled"
    assert r["episode"]["n_hints_delivered"] == 3  # 0, 0.5 and 1.0 s, all before the 1.5 s deadline


def test_deadline_records_hints_never_delivered(tmp_path):
    scripts = {ORCH: [Reply("", [tc("bash", command="sleep 30")])]}
    exit_reason, r, _, _, sec = session(tmp_path, scripts, n_hints=3, topology="single",
                                        interactive={"wall_clock_sec": 1.5})
    assert exit_reason == "deadline" and sec < 10
    assert [h["t_delivered"] is None for h in hints_in(r)] == [False, True, True]
    assert (r["episode"]["n_hints"], r["episode"]["n_hints_delivered"]) == (3, 1)


def test_reply_without_tool_call_waits_and_finish_is_not_a_tool(tmp_path):
    script = [Reply("Looking into it."), Reply("", [tc("finish")]), Reply("", [tc("wait")])]
    exit_reason, r, f, _, _ = session(tmp_path, {ORCH: script}, n_hints=2, topology="single")
    assert exit_reason == "idle" and len(r["steps"]) == 3
    assert hints_in(r)[1]["step"] == 2 and r["steps"][1]["t_start"] - r["episode"]["session_t0"] >= 0.5
    finish = f[ORCH].requests[2]["messages"][-1].tool_results[0]
    assert finish.is_error and "unknown tool 'finish'" in finish.content


def test_spawn_beyond_the_free_slots_and_the_budget(tmp_path):
    def orch(system, messages, tools):
        n = sum(1 for m in messages if m.role == "assistant")
        if n == 0:
            return Reply("", [tc("spawn_subagents", instructions=["a", "b", "c"])])
        if n == 1:
            return Reply("", [tc("wait")])
        if n == 2:
            return Reply("", [tc("spawn_subagents", instructions=["d", "e"])])
        return Reply("", [tc("wait")])

    sub = lambda system, messages, tools: Reply("", [tc("submit", report="ok")])  # noqa: E731
    _, r, f, _, _ = session(tmp_path, {ORCH: orch, SUB: sub}, n_hints=1,
                            subagent={"spawn_mode": "background", "max_concurrent": 2, "max_dispatches": 3})
    texts = [x["content"] for m in r["transcript"] if m["role"] == "orchestrator" for x in m["tool_results"]]
    spawned = [t for t in texts if t.startswith("[spawn_subagents")]
    assert "started 2 of 3 instruction(s): sub-1, sub-2" in spawned[0] and "instruction(s) 3-3 were NOT run" in spawned[0]
    assert "started 1 of 2 instruction(s): sub-3" in spawned[1] and "dispatch budget left: 0" in spawned[1]
    assert [s["granted"] for s in r["spawns"]] == [2, 1]


def test_max_steps_stops_the_background_subagents(tmp_path):
    scripts = {ORCH: [Reply("", [tc("spawn_subagents", instructions=["long"])]), Reply("", [tc("list_dir")])],
               SUB: [Reply("", [tc("bash", command="sleep 30")])]}
    exit_reason, r, _, _, sec = session(tmp_path, scripts, n_hints=1, budget={"orchestrator_steps": 2},
                                        subagent={"spawn_mode": "background"})
    assert exit_reason == "max_steps" and sec < 10 and r["dispatches"][0]["status"] == "cancelled"


def test_old_tool_output_is_elided_beyond_the_context_limit(tmp_path):
    def orch(system, messages, tools):
        if sum(1 for m in messages if m.role == "assistant") < 4:
            return Reply("", [tc("bash", command="head -c 6000 /dev/zero | tr '\\0' a")])
        return Reply("", [tc("wait")])

    _, r, f, _, _ = session(tmp_path, {ORCH: orch}, n_hints=1, topology="single",
                            interactive={"max_context_chars": 10000})
    last = f[ORCH].requests[-1]["messages"]
    results = [m.tool_results[0].content for m in last if m.role == "tool"]
    assert results[0].startswith("[elided to keep the conversation within its size limit")
    assert results[-1].startswith("a" * 100)  # the newest output is never elided
    assert r["episode"]["n_elided_results"] >= 2


# --- hints ---------------------------------------------------------------------------------------


def test_schedule_order_slots_and_texts(tmp_path):
    task = make_task(tmp_path, 14, files=[[f"src/m{i}.py"] for i in range(13)] + [["a/x.py", "b/y.py"]])
    subs = load_subtasks(task)
    assert len(subs) == 14 and subs[0] == Subtask("b0", ("tests.test_m::test_bug_number_0[p0]",
                                                          "tests.test_m::test_bug_number_0[p1]"), ("src/m0.py",))
    assert ordered(subs, "task-1", 0) == ordered(list(reversed(subs)), "task-1", 0)
    assert ordered(subs, "task-1", 0) != ordered(subs, "task-1", 1) != ordered(subs, "task-2", 0)
    cfg = make_config(interactive={}).interactive
    assert (cfg.wall_clock_sec, cfg.interval_sec, n_slots(cfg)) == (3600.0, 300.0, 12)
    sched = schedule(cfg, task, "task-1")
    assert [h.t_offset for h in sched] == [300.0 * k for k in range(12)] and [h.index for h in sched] == list(range(12))
    assert len(schedule(make_config(interactive={"max_hints": 5}).interactive, task, "task-1")) == 5
    assert len(schedule(make_config(interactive={"wall_clock_sec": 3601}).interactive, task, "task-1")) == 13
    b0 = subs[0]
    assert hint_text(b0, "files") == "There is a bug in `src/m0.py`."
    assert hint_text(b0, "tests") == ("There is a bug in `src/m0.py`. Once it is fixed, the test "
                                      "`tests.test_m::test_bug_number_0` should pass; the version of it that checks "
                                      "this is not in the repository, so its name only describes the expected "
                                      "behaviour.")
    many = Subtask("x", tuple(f"t::test_{i}" for i in range(5)), ("a/x.py", "b/y.py"))
    assert hint_text(many, "tests").startswith("There is a bug whose fix touches `a/x.py` and `b/y.py`. Once it is "
                                               "fixed, the tests `t::test_0`, `t::test_1` and `t::test_2` (and 2 more)")
    assert hint_text(b0, "report", "  Clicking crashes.  ") == "Clicking crashes."
    with pytest.raises(HintError, match="no generated report"):
        hint_text(b0, "report")


def test_task_without_subtasks_is_a_hint_error(tmp_path):
    with pytest.raises(HintError, match="no SWE-sweep subtasks"):
        load_subtasks(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "swesweep_eval.json").write_text(json.dumps({"subtasks": []}))
    with pytest.raises(HintError, match="non-empty"):
        load_subtasks(tmp_path)


def test_generated_reports(tmp_path):
    task = make_task(tmp_path, 3)
    factory = FakeFactory({ORCH: [Reply("Report for b1."), Reply("Report for b2.")]})
    client = factory(make_config().orchestrator)
    reports = generate_reports(task, client, existing={"b0": "Kept."})
    assert reports == {"b0": "Kept.", "b1": "Report for b1.", "b2": "Report for b2."}
    prompt = client.requests[0]["messages"][0].content
    assert "Files the fix changed: src/m1.py" in prompt and "+def test_bug_number_b1" in prompt
    assert client.requests[0]["tools"] == [] and "Do not describe the fix" in client.requests[0]["system"]
    write_reports(tmp_path / "hints", "task-1", ORCH, reports)
    assert load_reports(tmp_path / "hints", "task-1") == reports
    cfg = make_config(interactive={"hint_detail": "report"}).interactive
    assert schedule(cfg, task, "task-1", tmp_path / "hints")[0].text.startswith(("Kept.", "Report for"))
    with pytest.raises(HintError, match="run `subagents-bench hints`"):
        schedule(cfg, task, "task-2", tmp_path / "hints")
    with pytest.raises(HintError, match="needs a hints dir"):
        schedule(cfg, task, "task-1", None)


@pytest.mark.parametrize("over, msg", [
    ({"subagent": {"spawn_mode": "background"}}, "background needs an `interactive` episode"),
    ({"subagent": {"spawn_mode": "later"}}, "spawn_mode: 'later' is not one of"),
    ({"interactive": {"hint_detail": "all"}}, "hint_detail: 'all' is not one of"),
    ({"interactive": {"interval_sec": 0}}, "interval_sec: expected a positive number"),
    ({"interactive": {"max_hints": 0}}, "max_hints: expected null or an integer >= 1"),
    ({"interactive": {"max_context_chars": 500}}, "max_context_chars: expected null or an integer >= 10000"),
    ({"interactive": {"end_when_idle": "yes"}}, "end_when_idle: expected true or false"),
    ({"interactive": {"walltime": 5}}, "unknown key"),
    ({"interactive": {}, "subagent": {"spawn_mode": "background", "strategy": "arbiter"}}, "needs strategy parallel"),
    ({"interactive": {}, "subagent": {"delegation_hint": True}}, "needs spawn_mode background"),
])
def test_bad_interactive_configs(over, msg):
    with pytest.raises(ConfigError, match=msg):
        make_config(**over)


def test_interactive_prompts():
    bg = interactive_prompt(make_config(interactive={}, subagent={"spawn_mode": "background", "delegation_hint": True}), "/w")
    assert "The session lasts 60 minutes" in bg and "New user messages and subagent reports are delivered" in bg
    assert "Your attention is the scarce resource" in bg and "at most 3 subagents run at once" in bg
    assert "Step budget" not in bg and "`finish`" not in bg and "`wait`: pause until a subagent report" in bg
    single = interactive_prompt(make_config(topology="single", interactive={"wall_clock_sec": 90}), "/w")
    assert "lasts 1.5 minutes" in single and "Subagents" not in single and "until a new user message arrives" in single
    blocking = interactive_prompt(make_config(interactive={}), "/w")
    assert "your step waits for all of them" in blocking and "subagent reports are delivered" not in blocking
    assert clock(0) == "0:00" and clock(3599.9) == "59:59" and clock(-1) == "0:00"
    assert interactive_note(65, 3600, [("sub-1", 5), ("sub-2", 125)], 7) == (
        "[1:05 of 60:00 elapsed; subagents running: sub-1 (0:05), sub-2 (2:05); dispatch budget left: 7]")
    assert interactive_note(5, 60, [], 3).startswith("[0:05 of 1:00 elapsed; subagents running: none")
    assert interactive_note(5, 60, None, None) == "[0:05 of 1:00 elapsed]"


# --- analysis --------------------------------------------------------------------------------------


def test_path_matching():
    assert names_file("cat /click/src/click/types.py", "src/click/types.py")
    assert names_file("sed -n 1,9p types.py", "src/click/types.py")  # after a cd into src/click
    assert names_file("vim click/types.py", "src/click/types.py")
    assert not names_file("cat src/other/types.py", "src/click/types.py")
    assert not names_file("cat mytypes.py types.pyc types.py.bak", "src/click/types.py")
    assert not names_file("cat otherclick/types.py", "src/click/types.py")
    f = ["src/click/core.py"]
    bash = lambda c: {"name": "bash", "arguments": {"command": c}}  # noqa: E731
    assert tool_touches({"name": "read_file", "arguments": {"path": "/click/src/click/core.py"}}, f) == (True, False)
    assert tool_touches({"name": "edit_file", "arguments": {"path": "src/click/core.py"}}, f) == (True, True)
    assert tool_touches({"name": "search", "arguments": {"pattern": "core.py"}}, f) == (False, False)
    assert tool_touches(bash("grep -n def src/click/core.py | head"), f) == (True, False)
    assert tool_touches(bash("cat src/click/core.py > /tmp/x 2>&1"), f) == (True, False)
    assert tool_touches(bash("sed -i 's/a/b/' src/click/core.py"), f) == (True, True)
    assert tool_touches(bash("echo x >> src/click/core.py"), f) == (True, True)
    assert tool_touches(bash("git apply fix.diff  # src/click/core.py"), f) == (True, True)
    assert tool_touches(bash("python3 - <<'EOF'\np='src/click/core.py'\nopen(p,'w').write(s)\nEOF"), f) == (True, True)
    assert tool_touches(bash("pytest tests/test_core.py"), f) == (False, False)


def trial_dir(jobs: Path, job: str, ep_src: Path, verdicts: dict[str, bool], visible_ok: bool = True) -> Path:
    d = jobs / job / "task-1__x"
    (d / "verifier").mkdir(parents=True)
    os.replace(ep_src, d / "agent")
    (d / "config.json").write_text(json.dumps({"task": {"path": "task-1"}}))
    n = sum(verdicts.values())
    (d / "result.json").write_text(json.dumps({
        "task_name": "task-1", "trial_name": d.name, "exception_info": None,
        "verifier_result": {"rewards": {"reward": n / 4 if visible_ok else 0.0, "n_resolved": n, "visible_ok": float(visible_ok)}},
        "verifier": {"started_at": "2026-10-13T00:00:00Z"}, "verifier_environment_mode": "separate"}))
    (d / "verifier" / "eval.json").write_text(json.dumps({"visible_ok": visible_ok, "subtasks": [
        {"subtask_id": k, "resolved": v, "partial": False} for k, v in verdicts.items()]}))
    return d


@pytest.fixture(scope="module")
def interactive_jobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ijobs")
    jobs = root / "jobs" / "si"
    out = {}
    for cond, over in {"bg": {"subagent": {"spawn_mode": "background"}}, "solo": {"topology": "single"}}.items():
        base = root / cond
        if cond == "bg":
            scripts = delegating()
        else:
            def solo(system, messages, tools):
                h = hint_of(messages)
                if h:  # every hint in the message: two can arrive together while a command runs
                    writes = "; ".join(f"echo fixed > {target(x)}" for x in h.split("Hint: ")[1:])
                    return Reply("", [tc("bash", command=f"sleep 0.7; mkdir -p src; {writes}")])
                return Reply("", [tc("wait")])
            scripts = {ORCH: solo}
        _, _, _, hints, _ = session(base, scripts, **over)
        verdicts = {h.subtask.id: k % 2 == 0 for k, h in enumerate(hints)}
        out[cond] = (trial_dir(jobs, f"si__r1__{cond}", base / "ep", verdicts), hints)
    return root, jobs, out


def test_hint_rows_measure_each_reaction(interactive_jobs):
    _, _, out = interactive_jobs
    d, hints = out["bg"]
    t = load_trial(d)
    rows = pd.DataFrame(hint_rows(t, {"condition": "bg", "trial_name": d.name}, True))
    assert list(rows["subtask_id"]) == [h.subtask.id for h in hints]
    assert (rows["delivery_sec"] < 0.3).all() and (rows["response_sec"] >= rows["delivery_sec"]).all()
    assert (rows["dispatch_sec"] >= rows["response_sec"]).all() and rows["touch_by"].eq("subagent").all()
    assert (rows["edit_sec"] >= rows["touch_sec"]).all() and (rows["edit_sec"] >= 0.3).all()  # after the sleep
    assert list(rows["busy_at_arrival"])[0] == "start" and set(rows["busy_at_arrival"][1:]) <= {"wait", "idle", "llm", "tool"}
    assert list(rows["resolved"]) == [True, False, True, False] and list(rows["counted"]) == [True, False, True, False]
    assert not rows["shared_files"].any() and rows["continued_after_next"].iloc[:3].notna().all()
    solo = pd.DataFrame(hint_rows(load_trial(out["solo"][0]), {"condition": "solo", "trial_name": "x"}, True))
    # each hint waits for the previous hint's 0.7 s command: delivered late, touched by the orchestrator itself
    assert (solo["delivery_sec"].iloc[1:] > 0).all() and solo["delivery_sec"].iloc[1] > 0.1
    assert solo["touch_by"].eq("orchestrator").all()
    assert solo["dispatch_sec"].isna().all()
    assert (solo["busy_at_arrival"].iloc[1:] == "tool").all()


def test_interactivity_tables_and_figure(interactive_jobs, tmp_path):
    root, jobs, _ = interactive_jobs
    out = tmp_path / "analysis"
    assert analyze([str(jobs), "--out", str(out)]) == 0
    inter = pd.read_csv(out / "interactivity.csv").set_index("condition")
    assert set(inter.index) == {"bg", "solo"} and (inter["n_hints"] == 4).all()
    assert inter.loc["bg", "hinted_fixed_frac"] == 0.5 and inter.loc["bg", "dispatch_frac"] == 1.0
    assert inter.loc["solo", "busy_at_arrival_frac"] == 1.0 and inter.loc["bg", "busy_at_arrival_frac"] < 1.0
    assert inter.loc["solo", "delivery_median_sec"] > inter.loc["bg", "delivery_median_sec"]
    pos = pd.read_csv(out / "hint_positions.csv")
    assert len(pos) == 8 and set(pos["hint_index"]) == {0, 1, 2, 3}
    eps = pd.read_csv(out / "episodes.csv")
    assert (eps["interactive"] == 1).all() and (eps["n_hints_delivered"] == 4).all()
    assert (eps["cp_total_sec"] <= eps["wall_clock_sec"] + 1e-3).all()
    assert "arguments" not in pd.read_csv(out / "tools.csv").columns
    figs = tmp_path / "figs"
    assert plot([str(out), "--out", str(figs)]) == 0
    assert (figs / "interactivity.png").stat().st_size > 10_000


def test_censored_medians_status_and_unshared_touches():
    def row(k, touch, shared=False, status=None, counted=None, pre_edited=False, valid=True):
        return {"condition": "c", "trial_name": "t1", "valid": valid, "topology": "single", "spawn_mode": None,
                "interval_sec": 300.0, "hint_index": k, "delivered": True, "busy_at_arrival": "tool" if k else "start",
                "shared_files": shared, "shares_any": shared, "delivery_sec": 1.0, "response_sec": 5.0,
                "dispatch_sec": None, "touch_sec": touch, "edit_sec": None, "pre_edited": pre_edited,
                "status_at_next": status, "counted": counted}
    hints = pd.DataFrame([row(0, 10.0, status="continued", counted=True), row(1, None, status="not_started"),
                          row(2, 400.0, shared=True, status="continued"), row(3, 20.0, status="stopped", counted=True),
                          row(4, 1.0, pre_edited=True, counted=False)])
    c = condition_table(hints).iloc[0]
    assert (c["n_hints"], c["n_later_hints"]) == (5, 4) and c["first_touch_median_sec"] == 10.0
    assert c["pre_edited_frac"] == 0.25  # hint 4 was edited before it was sent: left out of touch and edit
    assert c["touch_frac"] == 2 / 3 and c["touch_median_sec"] == 400.0  # [20, 400, never]
    assert c["touch_before_next"] == 1 / 3
    assert pd.isna(c["touch_median_sec_unshared"]) and c["touch_before_next_unshared"] == 0.5  # [never, 20]
    assert c["n_status_unshared"] == 3 and c["continued_at_next_frac"] == 1 / 3 and c["stopped_fixed_frac"] == 1.0
    assert c["not_started_at_next_frac"] == 1 / 3 and c["hinted_fixed_frac"] == 2 / 3
    assert pd.isna(c["edit_median_sec"]) and c["busy_at_arrival_frac"] == 1.0
    assert list(position_table(hints)["hint_index"]) == [0, 1, 2, 3, 4]
    invalid = pd.DataFrame([row(0, 1.0, valid=False)])
    assert condition_table(invalid).empty and position_table(invalid).empty  # no crash without valid hints


# --- harbor agent and sweep ---------------------------------------------------------------------------


def test_harbor_agent_runs_an_interactive_session(tmp_path, monkeypatch):
    from test_harbor_agent import FakeHarborEnv, run
    from subagents_bench.harness.harbor_agent import OrchestratorAgent

    task = make_task(tmp_path / "tasks", 2)
    work = tmp_path / "work"
    work.mkdir()
    trial = tmp_path / "jobs/si__r1__solo/task-1__AbC"
    (trial / "agent").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    (trial / "config.json").write_text(json.dumps({"task": {"path": "tasks/task-1"}}))  # relative to harbor's cwd
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(make_config(**merge(FAST, {"topology": "single"})).to_dict()))
    monkeypatch.setenv("SUBAGENTS_BENCH_CONFIG", str(cfg))
    agent = OrchestratorAgent(logs_dir=trial / "agent", model_name=None, logger=None, extra_env={})
    agent.client_factory = FakeFactory({ORCH: lambda s, m, t: Reply("", [tc("wait")])})
    agent.state_root = str(tmp_path / "sb")
    ctx = run(agent, FakeHarborEnv(work))
    r = assert_consistent(trial / "agent")
    assert r["episode"]["exit_reason"] == "idle" and r["episode"]["n_hints_delivered"] == 2
    assert r["episode"]["end_kill"] == {"ok": True, "n_killed": 0, "error": None}
    assert ctx.metadata["n_hints"] == 2 and r["inbox"][0]["subtask_id"] in {"b0", "b1"}
    assert ctx.metadata["end_kill"]["ok"] and ctx.metadata["hint_order_seed"] == 0  # round 1 of the job name
    (trial / "config.json").write_text(json.dumps({"task": {}}))
    with pytest.raises(Exception, match="names no task path"):
        asyncio.run(agent.run("x", FakeHarborEnv(work), None))


def interactive_spec(tmp_path: Path, **over) -> Path:
    from test_sweep import make_spec
    make_task(tmp_path / "proj" / "tasks", 3)
    spec = {"dataset": None, "dataset_path": "tasks", "tasks": ["task-1"],
            "harbor": {"n_concurrent": 1, "agent_timeout_multiplier": 0.1},
            "overrides": {"interactive": {"wall_clock_sec": 3600}},
            "conditions": {"solo": {"topology": "single"}}, **over}
    return make_spec(tmp_path, **spec)


def test_sweep_checks_interactive_tasks_and_timeouts(tmp_path):
    from subagents_bench import sweep
    spec = sweep.load_spec(interactive_spec(tmp_path))
    assert list(sweep.interactive(spec)) == ["solo"] and sweep.check_interactive(spec) == []
    spec.timeout_multiplier = 0.0833
    assert "would cancel a 3600s session" in sweep.check_interactive(spec)[0]
    spec.tasks = ["task-1", "task-2"]
    spec.timeout_multiplier = 0.1
    assert sweep.check_interactive(spec)[0].startswith("task-2: ") and "no SWE-sweep subtasks" in sweep.check_interactive(spec)[0]
    with pytest.raises(ConfigError, match="set dataset_path"):
        sweep.load_spec(interactive_spec(tmp_path / "b", dataset="terminal-bench@2.0", dataset_path=None))


def test_sweep_generates_reports_and_passes_the_hints_dir(tmp_path, monkeypatch, capsys):
    from subagents_bench import cli, sweep
    from test_sweep import Fake, FAKE
    import shlex
    import sys

    path = interactive_spec(tmp_path, overrides={"interactive": {"wall_clock_sec": 3600, "hint_detail": "report"}})
    spec = sweep.load_spec(path)
    assert "no generated reports" in sweep.check_interactive(spec)[0]
    replies = iter(["Report A.", "Report B.", "Report C."])
    monkeypatch.setattr(sweep, "SailClient", lambda m: FakeFactory({m.model: lambda s, ms, t: Reply(next(replies))})(m))
    assert cli.main(["hints", str(path)]) == 0
    assert "[OK] task-1: 3 reports (3 new)" in capsys.readouterr().out
    assert sorted(load_reports(sweep.hints_dir(spec), "task-1").values()) == ["Report A.", "Report B.", "Report C."]
    assert sweep.check_interactive(spec) == []
    assert cli.main(["hints", str(path)]) == 0 and "(0 new)" in capsys.readouterr().out  # kept, no new requests
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "harbor").write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE))} "$@"\n')
    (bin_dir / "harbor").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(sweep, "harbor_bin", lambda: str(bin_dir / "harbor"))
    fake = Fake(tmp_path)
    monkeypatch.setenv("FAKE_HARBOR_LOG", str(fake.log))
    monkeypatch.setenv("FAKE_HARBOR_MODES", str(fake.modes))
    monkeypatch.setattr(sweep, "ping", fake.ping)
    assert sweep.run(spec, sweep.plan(spec)) == 0
    assert fake.calls()[0]["hints_dir"] == str(sweep.hints_dir(spec))


# --- regressions from review ---------------------------------------------------------------------------


def test_nothing_is_delivered_after_the_last_step(tmp_path):
    scripts = {ORCH: [Reply("", [tc("bash", command="sleep 0.7")])] * 2}
    exit_reason, r, f, _, _ = session(tmp_path, scripts, n_hints=3, topology="single", budget={"orchestrator_steps": 2})
    assert exit_reason == "max_steps" and len(f[ORCH].requests) == 2
    assert [(h["hint_index"], h["step"]) for h in hints_in(r)] == [(0, 1), (1, 2), (2, None)]
    assert r["episode"]["n_hints_delivered"] == 2


def test_wait_timeouts():
    from subagents_bench.interactive import wait_timeout
    assert [wait_timeout(x, 3600) for x in (None, True, "soon", float("nan"))] == [None] * 4
    assert [wait_timeout(x, 3600) for x in (0, -5, "60", 12.5, 10 ** 400, float("inf"))] == [0.0, 0.0, 60.0, 12.5, 3600, 3600]


def test_wait_zero_checks_once_and_a_timed_wait_does_not_end_the_session(tmp_path):
    script = [Reply("", [tc("wait", timeout_sec=0)]), Reply("", [tc("wait", timeout_sec=1)]), Reply("", [tc("wait")])]
    exit_reason, r, f, _, sec = session(tmp_path, {ORCH: script}, n_hints=1, topology="single")
    waits = [x for x in r["tools"] if x["name"] == "wait"]
    assert exit_reason == "idle" and len(waits) == 3
    assert waits[0]["t_end"] - waits[0]["t_start"] < 0.2  # wait(0): a check, not a block
    assert waits[1]["t_end"] - waits[1]["t_start"] >= 0.9  # timed: runs its course though nothing is left


def test_wait_then_spawn_in_one_step_keeps_the_session(tmp_path):
    def orch(system, messages, tools):
        if sum(1 for m in messages if m.role == "assistant") == 0:
            return Reply("", [tc("wait"), tc("spawn_subagents", instructions=["late work"])])
        return Reply("", [tc("wait")])

    sub = [Reply("", [tc("bash", command="sleep 0.3")]), Reply("", [tc("submit", report="done")])]
    exit_reason, r, _, _, _ = session(tmp_path, {ORCH: orch, SUB: sub}, n_hints=1, subagent={"spawn_mode": "background"})
    assert exit_reason == "idle" and r["dispatches"][0]["status"] == "submitted"
    assert [x["kind"] for x in r["inbox"]] == ["hint", "report"]


def test_implicit_waits_are_recorded_as_wait_tools(tmp_path):
    _, r, _, _, _ = session(tmp_path, {ORCH: [Reply("ok"), Reply("ok")]}, n_hints=2, topology="single")
    waits = [x for x in r["tools"] if x["name"] == "wait"]
    assert len(waits) == 2 and waits[0]["arguments"] == {"implicit": True} and waits[0]["call_id"] == r["steps"][0]["call_id"]
    assert waits[0]["t_end"] - waits[0]["t_start"] >= 0.4  # until hint 1, at 0.5 s
    assert critical_path(r)["tool"] >= 0.4


def test_old_reports_and_reasoning_are_elided_too(tmp_path):
    def orch(system, messages, tools):  # one job at a time: each report is its own, older message
        n = sum(1 for m in messages if m.role == "assistant" and m.tool_calls[0].name == "spawn_subagents")
        if n < 4 and messages[-1].role == "user":
            return Reply("", [tc("spawn_subagents", instructions=[f"job {n}"])], reasoning="r" * 3000)
        return Reply("", [tc("wait")])

    sub = lambda system, messages, tools: Reply("", [tc("submit", report="R" * 5000)])  # noqa: E731
    _, r, f, _, _ = session(tmp_path, {ORCH: orch, SUB: sub}, n_hints=1, subagent={"spawn_mode": "background"},
                            interactive={"max_context_chars": 10000})
    users = [m.content for req in f[ORCH].requests for m in req["messages"][1:] if m.role == "user"]
    assert any("characters of an older report]" in u for u in users) and r["episode"]["n_elided_results"] >= 2
    assert f[ORCH].requests[-1]["messages"][1].reasoning is None


def test_scope_all_keeps_the_task_goal_and_rounds_vary_the_order(tmp_path):
    from subagents_bench.hints import order_seed
    _, _, f, _, _ = session(tmp_path, {ORCH: [Reply("", [tc("wait")])]}, n_hints=1, topology="single",
                            interactive={"scope": "all"})
    assert f[ORCH].requests[0]["messages"][0].content.startswith("Fix every bug.\n\nHint: ")
    cfg = make_config(interactive={"order_seed": 5}).interactive
    assert [order_seed(cfg, k) for k in (None, 1, 2, 3)] == [5, 5, 6, 7]
    assert order_seed(make_config(interactive={"order_seed": 5, "vary_order_by_round": False}).interactive, 3) == 5
    task = make_task(tmp_path / "t", 12)
    orders = {tuple(h.subtask.id for h in schedule(cfg, task, "task-1", round_=k)) for k in (1, 2, 3)}
    assert len(orders) == 3
    with pytest.raises(ConfigError, match="scope: 'some' is not one of"):
        make_config(interactive={"scope": "some"})


def test_blocking_prompt_has_no_step_budget_wording():
    blocking = interactive_prompt(make_config(interactive={}), "/w")
    assert "uses one of your steps" not in blocking and "returns when all of them have finished" in blocking
    assert "at most 3 instruction(s) per call are run" in blocking


def test_unfixed_and_pre_hint_work_in_hint_rows(tmp_path):
    def orch(system, messages, tools):
        if sum(1 for m in messages if m.role == "assistant") == 0:  # works on every file up front
            writes = "; ".join(f"echo x > src/m{i}.py" for i in range(3))
            return Reply("", [tc("bash", command=f"mkdir -p src; {writes}")])
        return Reply("", [tc("wait")])

    base = tmp_path / "e"
    _, _, _, hints, _ = session(base, {ORCH: orch}, n_hints=3, topology="single", interactive={"scope": "all"})
    d = tmp_path / "jobs" / "s__r1__c" / "task-1__x"
    (d / "verifier").mkdir(parents=True)
    os.replace(base / "ep", d / "agent")
    (d / "config.json").write_text(json.dumps({"task": {"path": "task-1"}}))
    (d / "verifier" / "eval.json").write_text(json.dumps({"visible_ok": False, "agent_build_ok": False, "subtasks": []}))
    rows = pd.DataFrame(hint_rows(load_trial(d), {"condition": "c", "trial_name": d.name}, True))
    assert list(rows["pre_edited"]) == [False, True, True] and rows["touch_sec"].iloc[1:].isna().all()
    assert list(rows["counted"]) == [False] * 3 and list(rows["graded"]) == [False] * 3  # the build broke: unfixed
    assert list(pd.DataFrame(hint_rows(load_trial(d), {}, False))["counted"]) == [None] * 3  # invalid: unknown


def test_status_at_next_and_ambiguous_basenames():
    from subagents_bench.analysis.interactivity import tool_touches as touches
    f = ["pint/facets/plain/quantity.py"]
    cd = {"name": "bash", "arguments": {"command": "cd pint && grep -n x quantity.py"}}
    assert touches(cd, f) == (True, False) and touches(cd, f, basename_ok=set()) == (False, False)
    seg = {"name": "bash", "arguments": {"command": "grep core.py src/core.py; sed -i 's/a/b/' parser.py"}}
    assert touches(seg, ["src/core.py"]) == (True, False)  # the write is to another file
