"""The monitor page (``subagents_bench.monitor``) on a small synthetic jobs dir: one graded requests
session (with a transcript), one graded bugs-only control (without), the comparison with an earlier
sweep (the same bugs, more bugs, unusable), the waiting page and the CLI's errors; the tab bar and
pages sharing one out dir; the requests sweep's own header; the levels sweep (labels, plan, the
"what each level adds" table, the ladder, a partial run, the comparison with PR #13's baseline)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from subagents_bench.monitor import Monitor, build, load_sweep
from subagents_bench.monitor.__main__ import main
from subagents_bench.monitor.build import LEVELS, SETTINGS, level_of, planned_rounds
from subagents_bench.monitor.transcripts import CAP, export_all, file_name

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


def sweep_json(jobs: Path, name: str = SWEEP) -> None:
    jobs.mkdir(parents=True, exist_ok=True)
    configs = {c: {"interactive": {**ICFG, **(o.get("interactive") or {})}} for c, o in CONDS.items()}
    (jobs / "sweep.json").write_text(json.dumps({"name": name, "spec": {
        "name": name, "tasks": [TASK], "rounds": 1, "conditions": CONDS}, "configs": configs}))


def hint(mid, idx, t, sub, urgent=False):
    return {"msg_id": mid, "kind": "hint", "hint_index": idx, "t_ready": T0 + t, "t_delivered": T0 + t + 2,
            "step": None, "subtask_id": sub, "target_files": [f"pkg/{sub}.py"], "f2p": [], "urgent": urgent}


def trial(jobs: Path, cond: str, replies: bool, transcript: bool, sweep: str = SWEEP, rnd: int = 1,
          icfg: dict | None = None, task: str = TASK) -> Path:
    d = jobs / f"{sweep}__r{rnd}__{cond}" / f"{task}__abc"
    (d / "agent").mkdir(parents=True)
    (d / "verifier").mkdir()
    (d / "verifier" / "eval.json").write_text(json.dumps({"visible_ok": True, "subtasks": [
        {"subtask_id": "b0", "resolved": True}, {"subtask_id": "b1", "resolved": False}]}))
    (d / "result.json").write_text(json.dumps({"task_name": task, "verifier_result": {"rewards": {"reward": 0.5}}}))
    (d / "config.json").write_text("{}")
    icfg = icfg or {**ICFG, **({} if replies else {"replies": False, "questions_per_interval": 0})}
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
    assert page.startswith("<title>mini Monitor</title>")  # a sweep without its own text: a header from the design
    assert "<h1>mini</h1>" in page and "2 conditions on demo, 1 round: 2 sessions" in page
    assert "Conditions without <code>reply</code> get bugs only" in page and "Controls (dashed cards)" in page
    assert '<nav class="tabs"' not in page  # no --nav, no tab bar
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


# --- the tab bar, pages sharing an out dir, transcript names -------------------------------------

SNAP = "Thu 16 Oct, 12:00 EDT"


@pytest.fixture
def fixed_snapshot(monkeypatch):
    import importlib
    monkeypatch.setattr(importlib.import_module("subagents_bench.monitor.build").time, "strftime", lambda *a: SNAP)


def test_nav_and_pages_sharing_an_out_dir(jobs, tmp_path, fixed_snapshot):
    """Two sweeps' pages in one out dir: the tab bar marks the current page, a page other than
    index.html is a whole document, and both pages' transcripts stay listed in the manifest."""
    lv = tmp_path / "jobs" / LEVELS_NAME
    levels_json(lv)
    level_trial(lv, LEVELS_NAME, "single_eta", 1, transcript=True)
    out = tmp_path / "out"
    nav = ["--nav", "Levels=index.html", "--nav", "Requests (archive)=requests.html"]
    assert main([str(lv), "--out", str(out), *nav]) == 0
    assert main([str(jobs), "--out", str(out), "--page", "requests.html", *nav]) == 0
    index, req = (out / "index.html").read_text(), (out / "requests.html").read_text()
    tabs = ('<nav class="tabs" aria-label="Sweeps"><a href="index.html"{}>Levels</a><a href="requests.html"{}>'
            'Requests (archive)</a></nav>\n<header>')
    cur = ' aria-current="page"'
    assert tabs.format(cur, "") in index and tabs.format("", cur) in req
    assert not index.startswith("<!doctype") and req.startswith('<!doctype html>\n<html lang="en">\n<meta charset')
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in req
    files = sorted(p.name for p in (out / "transcripts").glob("*.json") if p.name != "manifest.json")
    assert files == [f"{SWEEP}__r1__single_requests__{TASK}__abc.json",
                     f"{LEVELS_NAME}__r1__single_eta__{TASK}__abc.json"]
    man = json.loads((out / "transcripts" / "manifest.json").read_text())
    assert sorted(e["file"] for e in man["files"]) == files  # the requests page kept the levels page's entry
    assert re.search(rf'data-src="transcripts/{LEVELS_NAME}__r1__single_eta__', index)
    # writing the levels page again rewrites nothing: its entry was kept with its signature
    assert export_all([lv / f"{LEVELS_NAME}__r1__single_eta" / f"{TASK}__abc"], out / "transcripts")[1] == 0


def test_page_and_nav_errors(jobs, tmp_path):
    assert main([str(jobs), "--out", str(tmp_path / "o"), "--page", "../x.html"]) == 2
    assert main([str(jobs), "--out", str(tmp_path / "o"), "--page", "x.htm"]) == 2
    with pytest.raises(SystemExit):
        main([str(jobs), "--out", str(tmp_path / "o"), "--nav", "no-href"])


def test_transcript_names_never_collide_across_sweeps(tmp_path):
    job = f"{LEVELS_NAME}__r1__single_baseline"
    assert file_name(tmp_path / LEVELS_NAME / job / "t") == f"{job}__t.json"
    # a jobs dir not named after the sweep (a spec's jobs_dir) puts its name in front
    assert file_name(tmp_path / "rerun" / job / "t") == f"rerun__{job}__t.json"


# --- the requests sweep's page -------------------------------------------------------------------

REQUESTS_HEADER = (  # the template's header before it had fields, with this sweep's numbers
    '<header>\n  <div class="eyebrow">subagents-bench · requests sweep</div>\n'
    '  <h1>Is the agent there when you ask?</h1>\n'
    '  <p>A simulated user works with the agent on one repository for 10 minutes. A bug arrives every 2 minutes (2 in '
    'all), two questions about the code come between bugs, a status check asks how long the open bugs will take, and '
    'bug 3 is urgent. The user sees only what the agent sends with <code>reply</code>, so this measures how fast and '
    'how well questions are answered, whether bugs get fixed, and whether the agent\'s estimates of its own work hold. '
    'Two bugs-only controls get the same bugs at the same times and nothing else, so the gap to them is what the '
    'requests cost. Orchestrator: DeepSeek-V4.1-Flash. Subagents: Gemma-4-31B, 20 steps each, up to 3 at once.</p>\n'
    f'  <div class="strip"><span>Snapshot <b>{SNAP}</b></span><span><b>0</b> live</span><span><b>0</b> grading</span>'
    '<span><b>2</b> of 2 graded</span>')


def test_requests_page_keeps_its_text(tmp_path, fixed_snapshot):
    """The requests sweep renders as before the header had fields; the tab bar is the only addition."""
    j = tmp_path / "jobs" / "swe_interactive_requests"
    sweep_json(j, name="swe_interactive_requests")
    trial(j, "single_requests", True, False, sweep="swe_interactive_requests")
    trial(j, "single_bugsonly", False, False, sweep="swe_interactive_requests")
    plain = build(j, tmp_path / "a", transcripts=False)[0].read_text()
    assert plain.startswith("<title>Requests Sweep Monitor</title>") and REQUESTS_HEADER in plain
    assert '<div class="wrap">\n<header>' in plain and "What changed from the earlier runs" in plain
    assert "<span>Questions ready: demo not written yet</span></div>" in plain
    assert ("Medians count a message never answered as infinitely late. Controls (dashed cards) have no ids or "
            "replies: their bugs are numbered by position and show only pickup (waited to be read, first edit after) "
            "and fixes.</p>") in plain
    assert ("not fixes. Bugs only vs with requests: per-bug pickup from <code>interactivity.csv</code>, over the bugs "
            "after the first.</p>") in plain
    assert "the share it was said done before.</p>\n</footer>" in plain and "By level" not in plain
    nav = [("Levels", "index.html"), ("Requests (archive)", "requests.html")]
    tabbed = build(j, tmp_path / "a", transcripts=False, nav=nav)[0].read_text()
    bar = ('<nav class="tabs" aria-label="Sweeps"><a href="index.html" aria-current="page">Levels</a>'
           '<a href="requests.html">Requests (archive)</a></nav>\n')
    assert tabbed.replace(bar, "", 1) == plain


# --- the levels sweep ----------------------------------------------------------------------------

LEVELS_NAME = "swe_interactive_levels"
TASK2 = "swe-sweep__other"
LEVEL_ICFG = {"wall_clock_sec": 600, "interval_sec": 120, "max_hints": 2, "hint_detail": "tests", "replies": True,
              "questions_per_interval": 0, "status_every": 0, "urgent_every": 0, "order_seed": 0,
              "vary_order_by_round": False, "snapshots": True, "snapshot_every_sec": 15.0, "require_eta": True}
LEVEL_ADDS = {"baseline": {}, "eta": {"status_every": 2}, "questions": {"status_every": 2, "questions_per_interval": 1},
              "urgent": {"status_every": 2, "questions_per_interval": 1, "urgent_every": 1}}
SETTING_OVER = {"single": {"topology": "single"},
                "manager_background": {"subagent": {"spawn_mode": "background"},
                                       "tools": {"orchestrator_tools": ["read_file"]}},
                "background_hinted": {"subagent": {"spawn_mode": "background", "delegation_hint": True}}}


def levels_json(jobs: Path, name: str = LEVELS_NAME, rounds: int = 4, levels=tuple(LEVEL_ADDS)) -> None:
    """A levels sweep's sweep.json, its conditions in the spec's order (level by level)."""
    jobs.mkdir(parents=True, exist_ok=True)
    conds = {f"{s}_{lv}": {**o, **({"interactive": LEVEL_ADDS[lv]} if LEVEL_ADDS[lv] else {})}
             for lv in levels for s, o in SETTING_OVER.items()}
    configs = {c: {"orchestrator": {"model": "deepseek-ai/DeepSeek-V4.1-Flash"},
                   "topology": "single" if c.startswith("single") else "subagents",
                   "subagent": {"model": None, "max_concurrent": 60, "max_steps": 2000},
                   "interactive": {**LEVEL_ICFG, **LEVEL_ADDS[c.rpartition("_")[2]]}} for c in conds}
    (jobs / "sweep.json").write_text(json.dumps({"name": name, "spec": {
        "name": name, "tasks": [TASK, TASK2], "rounds": rounds, "conditions": conds,
        "harbor": {"environment": "modal"}}, "configs": configs}))


def level_trial(jobs: Path, sweep: str, cond: str, rnd: int, live: bool = False, transcript: bool = False) -> Path:
    """A session with its level's messages: bug #1 at 0:00, a question at 1:00 (L2 on), the second bug
    at 2:00 (urgent at L3), a status check at 3:00 (L1 on); bug #1 estimated and said done, the rest
    answered."""
    icfg = {**LEVEL_ICFG, **LEVEL_ADDS[cond.rpartition("_")[2]]}
    d = trial(jobs, cond, True, transcript, sweep=sweep, rnd=rnd, icfg=icfg)
    inbox, replies = [hint(1, 0, 0, "b0")], [{"t": T0 + 20, "to": 1, "text": "on it", "eta_min": 2, "done": False},
                                             {"t": T0 + 150, "to": 1, "text": "fixed", "eta_min": None, "done": True}]
    if icfg["questions_per_interval"]:
        inbox.append({"msg_id": 2, "kind": "question", "hint_index": 0, "t_ready": T0 + 60, "t_delivered": T0 + 61,
                      "step": None, "subtask_id": None, "target_files": None, "f2p": None, "urgent": False,
                      "question_id": "q2", "question": "What does f return?", "reference": "ref"})
        replies.append({"t": T0 + 70, "to": 2, "text": "None", "eta_min": None, "done": False})
    inbox.append(hint(len(inbox) + 1, 1, 120, "b1", urgent=bool(icfg["urgent_every"])))
    if icfg["status_every"]:
        inbox.append({"msg_id": len(inbox) + 1, "kind": "status", "hint_index": 0, "t_ready": T0 + 180,
                      "t_delivered": T0 + 181, "step": None, "subtask_id": None, "target_files": None, "f2p": None,
                      "urgent": False})
        replies.append({"t": T0 + 190, "to": len(inbox), "text": "#3: 5 min", "eta_min": None, "done": False})
    jl(d / "agent" / "inbox.jsonl", inbox)
    jl(d / "agent" / "replies.jsonl", sorted(replies, key=lambda x: x["t"]))
    if live:  # still running: no end, no result
        (d / "result.json").unlink()
        ep = json.loads((d / "agent" / "episode.json").read_text())
        (d / "agent" / "episode.json").write_text(json.dumps({k: v for k, v in ep.items()
                                                              if k not in ("t_end", "exit_reason")}))
    return d


def ladder_rows(page: str) -> dict[str, list[list[str]]]:
    """The "by level" ladder: setting -> its rows, as the cells' text (the level first)."""
    ladder = page.split("<h2>By level</h2>")[1].split("<h2>By condition</h2>")[0]
    out: dict[str, list[list[str]]] = {}
    for part in ladder.split('<tr class="grp-head">')[1:]:
        name = re.search(r"<b>(.*?)</b>", part)[1]
        out[name] = [[re.sub(r"<[^>]+>", "", c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row)]
                     for row in re.findall(r"<tr>(<td>L\d.*?)</tr>", part)]
    return out


def test_levels_names_and_plan(tmp_path):
    assert level_of("manager_background_questions") == ("manager_background", 2)
    assert level_of("single_baseline") == ("single", 0) and level_of("single_requests") is None
    conds = [f"{s}_{lv}" for lv in LEVELS for s in SETTINGS]
    plan = planned_rounds(LEVELS_NAME, conds, 4)
    assert plan["single_baseline"] == plan["background_hinted_eta"] == (1, 2)
    assert plan["manager_background_questions"] == plan["single_urgent"] == (1, 2, 3, 4)
    assert set(planned_rounds("swe_interactive_levels_smoke", conds, 1).values()) == {(1,)}
    assert set(planned_rounds("other", conds, 4).values()) == {(1, 2, 3, 4)}  # only the levels sweep has a plan
    full, smoke = tmp_path / "full", tmp_path / "smoke"
    levels_json(full)
    levels_json(smoke, "swe_interactive_levels_smoke", rounds=1)
    assert load_sweep(full).n_sessions == 72 and load_sweep(smoke).n_sessions == 24  # 3 settings x 2 tasks x rounds
    m = Monitor(full, load_sweep(full), prices={})
    assert list(m.cond)[:5] == ["single_baseline", "single_eta", "single_questions", "single_urgent",
                                "manager_background_baseline"]  # by setting, then level
    assert m.label("background_hinted_urgent")[0] == "All tools + background, hinted · L3 urgent bug"


def test_levels_waiting_page(tmp_path):
    """Before the first session: the header, the levels and the plan, no ladder."""
    lv = tmp_path / LEVELS_NAME
    levels_json(lv)
    page = build(lv, tmp_path / "o")[0].read_text()
    assert page.startswith("<title>Levels Sweep Monitor</title>") and "<h1>One kind of request at a time</h1>" in page
    assert "<b>0</b> of 72 graded" in page and "Waiting for the first session" in page
    assert "<h2>By level</h2>" not in page
    for s in ("sending the same 2 bugs at the same times at every level, one every 2 min",
              "L0 sends only the bugs (PR #13's baseline); L1 adds a status check at 3:00",
              "L2 adds 1 questions about the code after each bug; L3 makes the bug sent at 2:00 urgent",
              "which refuses a reply to a bug without <code>eta_min</code> or <code>done</code>",
              "Snapshots of the workspace give each bug's real fix time, graded after the run",
              "Every model is DeepSeek-V4.1-Flash, subagents too (up to 60 at once, 2000 steps each); the task "
              "containers run on Modal."):
        assert s in page
    table = page.split("<h2>What each level adds</h2>")[1].split("</table>")[0]
    assert [re.sub(r"<[^>]+>", "", r) for r in re.findall(r"<tr><td>(L\d.*?)</td>", table)] == [
        "L0 bugs only*_baseline", "L1 status checks*_eta", "L2 questions*_questions", "L3 urgent bug*_urgent"]
    assert "<code>status_every: 2</code>" in table and "<code>urgent_every: 1</code>" in table
    assert "+ the bug sent at 2:00 is urgent" in table and '12 <span class="muted">· rounds 1–2</span>' in table
    assert "rounds 1–2 run L0 and L1; rounds 1–4 run L2 and L3, 72 sessions in all" in page
    assert page.count('<div class="cond">') == 3 and "Controls (dashed cards)" not in page  # a card per setting
    assert "<span>Questions ready: demo not written yet · other not written yet</span>" in page
    smoke = tmp_path / "swe_interactive_levels_smoke"
    levels_json(smoke, "swe_interactive_levels_smoke", rounds=1)
    page = build(smoke, tmp_path / "s", page="smoke.html")[0].read_text()
    assert "<title>Levels Smoke Monitor</title>" in page and "<b>0</b> of 24 graded" in page
    assert "round 1 for every level, 24 sessions in all" in page and "main numbers" not in page


def test_levels_partial_run(tmp_path):
    """Sessions of a few conditions, one still live, one in round 3 (planned for L3 only): the ladder,
    the per-condition table in setting order, and the cards."""
    lv = tmp_path / LEVELS_NAME
    levels_json(lv)
    level_trial(lv, LEVELS_NAME, "single_baseline", 1)
    level_trial(lv, LEVELS_NAME, "single_urgent", 3)
    level_trial(lv, LEVELS_NAME, "manager_background_questions", 1, live=True)
    page = build(lv, tmp_path / "o", transcripts=False)[0].read_text()
    assert "<b>1</b> live" in page and "<b>2</b> of 72 graded" in page and "2 of 72 sessions graded" in page
    assert page.count('<details class="card') == 3
    rows = ladder_rows(page)
    assert list(rows) == ["Single agent", "Background manager", "All tools + background, hinted"]
    assert "· 2 of 24 sessions" in page and "· 1 of 24 sessions" in page and "· 0 of 24 sessions" in page
    l0, l1, l2, l3 = rows["Single agent"]
    # level, sessions, fixed, waited, first edit, answered, within 60 s, correct, ETA, done by ETA, status, urgent, cost
    assert l0[:3] == ["L0 bugs only", "1 of 4", "1/2"]
    assert l0[3:5] == ["2 s", "never"]  # bug 2 (the only one after the first) read after 2 s, never edited
    assert l0[5:8] == ["–", "–", "–"] and l0[10:12] == ["–", "–"]  # no questions, status checks or urgent bug
    assert l1 == ["L1 status checks", "0 of 4"] + ["–"] * 11
    assert l2[1] == "0 of 8" and l3[1] == "1 of 8"  # round 3 is planned for L3
    assert l3[5:7] == ["1/1", "100%"] and l3[10] == "1/1 · 10 s" and l3[11] == "0%"
    mgr = rows["Background manager"][2]
    assert mgr[1] == "1 of 8" and mgr[2] == "–" and mgr[5] == "1/1"  # live: no fix verdict yet
    by_cond = page.split("<h2>By condition</h2>")[1].split("</table>")[0]
    assert by_cond.index("Single agent · L3 urgent bug") < by_cond.index("Background manager · L0 bugs only")
    # a status check names the bugs sent before it; a level without questions has no question count
    assert "How long until each bug so far (#1, #3) is fixed?" in page
    card = page.split('<span class="who">Single agent · L0 bugs only</span>')[1].split("</summary>")[0]
    assert "questions" not in card and "estimates 1/2" in card


def test_levels_compared_with_the_baseline(tmp_path):
    """PR #13's baseline has the levels' L0 names: its rows stay on their side. Both runs send the same
    bugs in every round, so rounds are pooled: the levels' round 3 meets the baseline's rounds 1 and 2."""
    base = tmp_path / "swe_interactive_baseline_modal"
    levels_json(base, "swe_interactive_baseline_modal", rounds=2, levels=("baseline",))
    for rnd in (1, 2):
        level_trial(base, "swe_interactive_baseline_modal", "single_baseline", rnd)
    lv = tmp_path / LEVELS_NAME
    levels_json(lv)
    level_trial(lv, LEVELS_NAME, "single_baseline", 1)
    level_trial(lv, LEVELS_NAME, "single_urgent", 3)
    page = build(lv, tmp_path / "o", compare=base, transcripts=False)[0].read_text()
    assert "as the PR #13 baseline run, which sent 2 per session" in page and "rounds are pooled" in page
    assert "on demo, every round (2 shared bugs) · vs the last run&#x27;s single agent: the same design" in page

    def group(label):  # a group's rows: (fixed, sessions, bug sends) of this run's row, then of the last run's
        part = page.split(f"<b>{label}</b>")[1].split('<tr class="grp-head">')[0]
        rows = [[re.sub(r"<[^>]+>", "", c) for c in re.findall(r'<td class="num">(.*?)</td>', r)]
                for r in re.findall(r"<tr[^>]*><td>(?!<div).*?</tr>", part)]
        return [(r[0], r[-2], r[-1]) for r in rows]
    assert group("Single agent · L0 bugs only") == [("1/2 · 50%", "1", "2"), ("2/4 · 50%", "2", "4")]
    assert group("Single agent · L3 urgent bug") == [("1/2 · 50%", "1", "2"), ("2/4 · 50%", "2", "4")]
    assert '<td class="num">all</td>' in page and "pooled: a mark per session" in page
    head = page.split('<table class="grid">')[1].split("</thead>")[0]
    assert "L0<br>" in head and "L3<br>" in head and head.count('class="grp"') == 3  # a group per setting


def test_levels_smoke_sends_the_bugs_that_fit(tmp_path):
    """The smoke's 2-minute sessions fit 4 of max_hints' 6 bugs (one every 30 s): the comparison counts
    4; more questions written than a session asks reads as ready; the lede names only the rules on."""
    smoke = tmp_path / "swe_interactive_levels_smoke"
    levels_json(smoke, "swe_interactive_levels_smoke", rounds=1)
    doc = json.loads((smoke / "sweep.json").read_text())
    for c in doc["configs"].values():
        c["interactive"].update(wall_clock_sec=120, interval_sec=30, max_hints=6, snapshots=False)
    (smoke / "sweep.json").write_text(json.dumps(doc))
    assert load_sweep(smoke).n_bugs == 4
    (smoke / "hints").mkdir()
    for t in (TASK, TASK2):  # 8 asked per session (2 per bug at L2 and L3, 4 bugs); 12 written
        (smoke / "hints" / f"{t}.questions.json").write_text(json.dumps({"questions": [{}] * 12}))
    doc["configs"]["single_questions"]["interactive"]["questions_per_interval"] = 2
    (smoke / "sweep.json").write_text(json.dumps(doc))
    page = build(smoke, tmp_path / "o", page="smoke.html", transcripts=False)[0].read_text()
    assert "<span>Questions ready: demo 8/8 · other 8/8</span>" in page
    assert "it checks that every kind of message and the reply rule work." in page
    assert "sending the same 4 bugs at the same times at every level, one every 30 s" in page


def test_model_flag_sweep_reads_the_sweeps_own_questions(tmp_path):
    """A --model run's sweep.json names its hints dir: the monitor counts the questions there."""
    from subagents_bench import sweep
    from test_interactive import interactive_spec
    path = interactive_spec(tmp_path, overrides={"interactive": {
        "wall_clock_sec": 3600, "interval_sec": 300, "max_hints": 3, "replies": True, "questions_per_interval": 2}})
    spec = sweep.load_spec(path, model=Path(__file__).resolve().parents[1] / "configs/prime/models/gpt-5.6-luna.yaml")
    spec.jobs_dir.mkdir(parents=True)
    sweep._write_status(spec, "running", 0.0)
    hints = sweep.hints_dir(spec)
    hints.mkdir(parents=True)
    (hints / "task-1.questions.json").write_text(json.dumps({"questions": [{"id": "q1"}, {"id": "q2"}]}))
    sw = load_sweep(spec.jobs_dir)
    assert sw.name == "t-gpt-5.6-luna" and sw.hints == hints
    assert Monitor(spec.jobs_dir, sw, prices={}).questions_ready() == [("task-1", 2)]
