"""The monitor page (``subagents_bench.monitor``) on a small synthetic jobs dir: one graded requests
session (with a transcript), one graded bugs-only control (without), the comparison with an earlier
sweep (the same bugs, more bugs, unusable), the waiting page and the CLI's errors."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from subagents_bench.monitor import build, load_sweep
from subagents_bench.monitor.__main__ import main
from subagents_bench.monitor.transcripts import CAP

T0 = 1000.0
SWEEP = "mini"
TASK = "swe-sweep__demo"
ICFG = {"wall_clock_sec": 600, "interval_sec": 120, "max_hints": 2, "hint_detail": "tests", "replies": True,
        "questions_per_interval": 1, "status_every": 0, "urgent_every": 1}
CONDS = {"single_requests": {"topology": "single"},
         "single_bugsonly": {"topology": "single", "interactive": {"replies": False, "questions_per_interval": 0}}}
EVIL = "<script>alert(1)</script>"


def jl(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def sweep_json(jobs: Path) -> None:
    jobs.mkdir(parents=True, exist_ok=True)
    configs = {c: {"interactive": {**ICFG, **(o.get("interactive") or {})}} for c, o in CONDS.items()}
    (jobs / "sweep.json").write_text(json.dumps({"name": SWEEP, "spec": {
        "name": SWEEP, "tasks": [TASK], "rounds": 1, "conditions": CONDS}, "configs": configs}))


def hint(mid, idx, t, sub, urgent=False):
    return {"msg_id": mid, "kind": "hint", "hint_index": idx, "t_ready": T0 + t, "t_delivered": T0 + t + 2,
            "step": None, "subtask_id": sub, "target_files": [f"pkg/{sub}.py"], "f2p": [], "urgent": urgent}


def trial(jobs: Path, cond: str, replies: bool, transcript: bool) -> Path:
    d = jobs / f"{SWEEP}__r1__{cond}" / f"{TASK}__abc"
    (d / "agent").mkdir(parents=True)
    (d / "verifier").mkdir()
    (d / "verifier" / "eval.json").write_text(json.dumps({"visible_ok": True, "subtasks": [
        {"subtask_id": "b0", "resolved": True}, {"subtask_id": "b1", "resolved": False}]}))
    (d / "result.json").write_text(json.dumps({"task_name": TASK, "verifier_result": {"rewards": {"reward": 0.5}}}))
    (d / "config.json").write_text("{}")
    icfg = {**ICFG, **({} if replies else {"replies": False, "questions_per_interval": 0})}
    (d / "agent" / "episode.json").write_text(json.dumps({
        "task_name": TASK, "topology": "single", "config": {"interactive": icfg}, "session_t0": T0, "t_start": T0,
        "t_end": T0 + 600, "exit_reason": "deadline"}))
    inbox = [hint(1, 0, 0, "b0", urgent=replies), hint(3 if replies else 2, 1, 120, "b1")]
    if replies:
        inbox.insert(1, {"msg_id": 2, "kind": "question", "hint_index": 0, "t_ready": T0 + 60, "t_delivered": T0 + 61,
                         "step": None, "subtask_id": None, "target_files": None, "f2p": None, "urgent": False,
                         "question_id": "q2", "question": f"What does {EVIL} return?", "reference": "ref"})
    else:
        for m in inbox:
            m["msg_id"], m["urgent"] = None, False
    jl(d / "agent" / "inbox.jsonl", inbox)
    jl(d / "agent" / "replies.jsonl", [{"t": T0 + 20, "to": 1, "text": "on it", "eta_min": 2, "done": False},
                                       {"t": T0 + 70, "to": 2, "text": f"It returns {EVIL}", "eta_min": None,
                                        "done": False},
                                       {"t": T0 + 150, "to": 1, "text": "fixed", "eta_min": None, "done": True}]
       if replies else [])
    jl(d / "agent" / "tools.jsonl", [{"name": "edit_file", "role": "orchestrator", "arguments": {"path": "pkg/b0.py"},
                                      "t_start": T0 + 30, "t_end": T0 + 31}])
    jl(d / "agent" / "calls.jsonl", [{"role": "orchestrator", "model": "nope", "prompt_tokens": 10, "cached_tokens": 0,
                                      "completion_tokens": 1, "t_start": T0 + 3, "t_end": T0 + 9}])
    for k in ("steps", "spawns", "dispatches", "arbitrations"):
        jl(d / "agent" / f"{k}.jsonl", [])
    if transcript:
        orch = lambda r, **kw: {"agent_id": "orchestrator", "role": "orchestrator", "msg_role": r, **kw}  # noqa: E731
        jl(d / "agent" / "transcript.jsonl", [
            orch("system", t=T0, content="You are the orchestrator."),
            orch("user", t=T0 + 2, content="Request #1"),
            orch("assistant", t=T0 + 9, content="x" * (CAP + 5), tool_calls=[{"id": "c1", "name": "edit_file",
                                                                             "arguments": {"path": "pkg/b0.py"}}]),
            orch("tool", t=T0 + 31, tool_results=[{"call_id": "c1", "content": "ok"}]),
            orch("assistant", t=T0 + 70, content=f"Done: {EVIL}")])
    return d


@pytest.fixture
def jobs(tmp_path) -> Path:
    j = tmp_path / "jobs" / SWEEP
    sweep_json(j)
    trial(j, "single_requests", True, True)
    trial(j, "single_bugsonly", False, False)
    return j


def test_sweep_design_from_sweep_json(jobs):
    sw = load_sweep(jobs)
    assert (sw.wall, sw.interval, sw.n_bugs, sw.n_sessions) == (600.0, 120.0, 2, 2)
    assert sw.control("single_bugsonly") and not sw.control("single_requests")
    assert sw.twins() == {"single_bugsonly": "single_requests"}


def test_page_and_transcripts(jobs, tmp_path):
    out = tmp_path / "out"
    assert main([str(jobs), "--compare", str(tmp_path / "missing"), "--out", str(out)]) == 0
    page = (out / "index.html").read_text()
    assert "<title>Requests Sweep Monitor</title>" in page
    assert page.count('<details class="card') == 2 and page.count('<details class="card control') == 1
    assert "Single agent, bugs only" in page and "Bugs only vs with requests" in page
    assert "Fixed 1 of 2 bugs" in page
    assert "is unavailable: its jobs folder is missing" in page
    assert "What changed from the earlier runs" not in page  # the requests sweep's table only
    # model and question text is escaped, never markup
    assert EVIL not in page.split("<script>\n(() =>")[0] and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    srcs = re.findall(r'data-src="transcripts/([^"?]+)\?v=[0-9a-f]+"', page)
    assert len(srcs) == 1 and (out / "transcripts" / srcs[0]).is_file()  # the control has no transcript
    tx = json.loads((out / "transcripts" / srcs[0]).read_text())
    assert tx["sys"] == "You are the orchestrator." and len(tx["msgs"]) == 4 and tx["subs"] == []
    assert tx["urgent"] == [1] and tx["msgs"][1]["tc"][0][1] == "edit_file"
    long = tx["msgs"][1]["c"]
    assert len(long) > CAP and long[:CAP] == "x" * CAP and "[... 5 more characters not shown" in long
    assert tx["msgs"][3]["c"] == f"Done: {EVIL}"  # JSON keeps the text; the viewer sets textContent
    man = json.loads((out / "transcripts" / "manifest.json").read_text())
    assert [e["file"] for e in man["files"]] == srcs


def test_no_compare_no_transcripts(jobs, tmp_path):
    page = build(jobs, tmp_path / "o", transcripts=False)[0].read_text()
    assert "Compared with the last run" not in page and "data-src=" not in page
    assert not (tmp_path / "o" / "transcripts").exists()


def test_compare_with_an_earlier_sweep(jobs, tmp_path):
    """The run against itself under another name: every bug shared, groups of the same design."""
    other = tmp_path / "jobs" / "earlier"
    sweep_json(other)
    trial(other, "single_requests", True, False)
    page = build(jobs, tmp_path / "o", compare=other, transcripts=False)[0].read_text()
    assert "2 of 2 shared bugs have a graded session here" in page
    assert "the same first 2 bugs in the same order as the earlier run" in page
    assert '<table class="grid">' in page and "Bug by bug" in page


def test_empty_jobs_dir_waits(tmp_path):
    j = tmp_path / "jobs" / SWEEP
    sweep_json(j)
    page = build(j, tmp_path / "o")[0].read_text()
    assert "Waiting for the first session" in page and "<b>0</b> of 2 graded" in page
    assert f"<code>{SWEEP}</code>" in page and str(tmp_path) not in page  # no local path on a published page
    assert '<details class="card' not in page


def test_missing_sweep_json(tmp_path, capsys):
    assert main([str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    assert "--spec" in capsys.readouterr().err


def test_compare_leaves_out_edits_of_later_bugs(jobs, tmp_path):
    """An earlier sweep that sent 4 bugs: bug 2's only edit comes after bug 3, which targets the same
    file, has arrived, so it may be work on bug 3 and is left out (``_own_edits``); bug 1's edit stays."""
    from subagents_bench.monitor.build import rate_rows
    other = tmp_path / "jobs" / "earlier"
    other.mkdir(parents=True)
    icfg = {**ICFG, "max_hints": 4, "replies": False, "questions_per_interval": 0}
    (other / "sweep.json").write_text(json.dumps({"name": "earlier", "spec": {
        "name": "earlier", "tasks": [TASK], "rounds": 1, "conditions": {"single_bugsonly": {"topology": "single"}}},
        "configs": {"single_bugsonly": {"interactive": icfg}}}))
    d = trial(other, "single_bugsonly", False, False)
    ep = json.loads((d / "agent" / "episode.json").read_text())
    (d / "agent" / "episode.json").write_text(json.dumps({**ep, "config": {"interactive": icfg}}))
    inbox = [hint(None, i, 120 * i, f"b{i}") for i in range(4)]
    inbox[2]["target_files"] = ["pkg/b1.py"]
    for m in inbox:
        m["urgent"] = False
    jl(d / "agent" / "inbox.jsonl", inbox)
    jl(d / "agent" / "tools.jsonl", [
        {"name": "edit_file", "role": "orchestrator", "arguments": {"path": p}, "t_start": T0 + t, "t_end": T0 + t + 1}
        for p, t in (("pkg/b0.py", 30), ("pkg/b1.py", 250))])
    (d / "verifier" / "eval.json").write_text(json.dumps({"visible_ok": True, "subtasks": [
        {"subtask_id": f"b{i}", "resolved": i == 0} for i in range(4)]}))
    rows = {r["subtask_id"]: r for r in rate_rows(other, 2)}
    assert (rows["b0"]["edit_sec"], rows["b0"]["edit_sec_any"]) == (30, 30)
    assert rows["b1"]["edit_sec"] is None and rows["b1"]["edit_sec_any"] == 130
    page = build(jobs, tmp_path / "o", compare=other, transcripts=False)[0].read_text()
    assert "which sent 4 per session" in page and "Last-run first edits leave out possible work on bugs 3 to 4" in page
    assert "this leaves 1 of its 2 sends of bugs 1 to 2 edited instead of 2" in page
    assert "there, bugs 3 to 4 arrived after these" in page


def test_unusable_compare_sweep_says_why(jobs, tmp_path):
    bad, plain = tmp_path / "jobs" / "bad", tmp_path / "jobs" / "plain"
    bad.mkdir()
    (bad / "sweep.json").write_text(json.dumps({"name": "bad"}))
    sweep_json(plain)
    raw = json.loads((plain / "sweep.json").read_text())
    (plain / "sweep.json").write_text(json.dumps({**raw, "configs": {}}))
    page = build(jobs, tmp_path / "o", compare=bad, transcripts=False)[0].read_text()
    assert "is unavailable: its sweep.json is malformed" in page
    page = build(jobs, tmp_path / "o", compare=plain, transcripts=False)[0].read_text()
    assert f"is unavailable: its sweep {SWEEP} has no interactive condition" in page


def test_malformed_sweep_json(tmp_path, capsys):
    (tmp_path / "sweep.json").write_text(json.dumps({"name": "x"}))
    assert main([str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    assert "sweep.json is malformed" in capsys.readouterr().err
    (tmp_path / "sweep.json").write_text(json.dumps({"spec": {"name": "x", "conditions": {"a": {}}},
                                                     "configs": {"a": {"interactive": {"replies": True}}}}))
    assert main([str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    assert "its spec has no valid 'tasks'" in capsys.readouterr().err
