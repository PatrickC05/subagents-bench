"""The levels analysis (``analysis/levels.py``): setting and level from the condition name, the
question columns from ``answers.json`` (version 2 and older), each bug's start, preemption and the
earlier bugs open when it was sent, the urgency rows against twins, ``levels.csv`` with its
stratified cluster bootstrap, and the (sweep, condition) grouping. Fix times and estimates come from
monkeypatched ``fixes``/``estimates`` functions, so the tests do not depend on how they are computed."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from subagents_bench.analysis import estimates as estimates_mod
from subagents_bench.analysis import fixes as fixes_mod
from subagents_bench.analysis.__main__ import main as analyze
from subagents_bench.analysis.levels import (bug_rows, change_after, file_diffs, level_table, order_score,
                                             question_columns, setting_level, summary_text, urgency_rows, verdict,
                                             win)
from subagents_bench.analysis.load import Trial
from subagents_bench.analysis.stats import cluster_bootstrap

T0 = 1000.0
SUFFIX = {"L0": "baseline", "L1": "eta", "L2": "questions", "L3": "urgent"}


@pytest.mark.parametrize("condition, want", [
    ("single_baseline", ("single", "L0")), ("manager_background_eta", ("manager_background", "L1")),
    ("background_hinted_questions", ("background_hinted", "L2")), ("single_urgent", ("single", "L3")),
    ("single_requests", (None, None)), ("single_bugsonly", (None, None)), ("baseline", (None, None)),
    ("single_eta_x", (None, None)), ("", (None, None)), (None, (None, None)),
])
def test_setting_level(condition, want):
    assert setting_level(condition) == want


# --- stats.cluster_bootstrap -------------------------------------------------------------------------


def ratio(cl):
    den = sum(c[1] for c in cl)
    return sum(c[0] for c in cl) / den if den else math.nan


def test_cluster_bootstrap():
    cl, strata = [(1, 2), (0, 2), (2, 2), (1, 2)], ["a", "a", "b", "b"]
    est, lo, hi = cluster_bootstrap(cl, ratio, strata)
    assert est == 0.5 and 0.0 <= lo < est < hi <= 1.0 and cluster_bootstrap(cl, ratio, strata) == (est, lo, hi)
    # within a stratum: every resample keeps 2 clusters of a and 3 of b, a cluster drawn twice counts twice
    names, groups = ["a1", "a2", "b1", "b2", "b3"], ["a", "a", "b", "b", "b"]
    assert cluster_bootstrap(names, lambda c: sum(x.startswith("a") for x in c), groups) == (2, 2, 2)
    assert cluster_bootstrap(names, len, groups) == (5, 5, 5)
    # no stratum with two clusters: nothing to resample; one stratum without strata
    one_each = cluster_bootstrap(cl[:2], ratio, ["a", "b"])
    assert one_each[0] == 0.25 and math.isnan(one_each[1]) and math.isnan(one_each[2])
    assert not math.isnan(cluster_bootstrap(cl[:2], ratio)[1])
    assert all(math.isnan(x) for x in cluster_bootstrap([(0, 0), (0, 0)], ratio))  # NaN statistic
    # resamples with nothing in the denominator are left out: the rest all give 1
    assert cluster_bootstrap([(0, 0), (1, 1)], ratio) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        cluster_bootstrap(cl, ratio, ["a"])


# --- questions ---------------------------------------------------------------------------------------


def msg(mid, kind, idx, t, sub=None, urgent=False, delivery=2.0, files=None, f2p=None):
    return {"msg_id": mid, "kind": kind, "hint_index": idx, "t_ready": T0 + t,
            "t_delivered": None if delivery is None else T0 + t + delivery, "step": None, "subtask_id": sub,
            "urgent": urgent, "target_files": files if kind == "hint" else None,
            "f2p": (f2p or []) if kind == "hint" else None, "question_id": f"q{mid}" if kind == "question" else None,
            "question": "Q?" if kind == "question" else None, "reference": "R" if kind == "question" else None}


def rep(t, to, text="x", eta=None, done=False):
    return {"t": T0 + t, "to": to, "text": text, "eta_min": eta, "done": done}


def judged(t, v):
    return {"t": T0 + t, "verdict": v, "correct": None if v is None else v == "correct", "why": "w"}


def make_trial(d: Path, inbox, replies=(), answers=None, condition="single_questions", tools=(), dispatches=(),
               snapshots=None, replies_cfg=True, verdicts=None) -> Trial:
    (d / "agent").mkdir(parents=True)
    if answers is not None:
        (d / "agent" / "answers.json").write_text(json.dumps(answers))
    if verdicts is not None:
        (d / "verifier").mkdir()
        (d / "verifier" / "eval.json").write_text(json.dumps({"visible_ok": True, "subtasks": [
            {"subtask_id": k, "resolved": v} for k, v in verdicts.items()]}))
    ep = {"config": {"interactive": {"replies": replies_cfg, "interval_sec": 240}}, "session_t0": T0,
          "t_end": T0 + 1800, "exit_reason": "deadline", "topology": "single"}
    recs = {"inbox": list(inbox), "replies": list(replies), "tools": list(tools), "dispatches": list(dispatches),
            "steps": [], "spawns": [], "calls": []}
    if snapshots is not None:
        recs["snapshots"] = snapshots
    return Trial(d, "lv", 1, condition, None, ep, recs)


def test_question_columns_from_answers_v2(tmp_path):
    """The first correct reply answers a question; a wrong one before it is recorded; an unread
    verdict, or a reply not judged yet, leaves it unknown; never answered is known (never); a reply
    to another id citing the question is reported apart."""
    inbox = [msg(1, "hint", 0, 0, "b0", files=["a.py"]), msg(2, "question", 0, 80, delivery=10),
             msg(3, "question", 1, 160), msg(4, "question", 2, 400), msg(5, "question", 3, 500),
             msg(6, "question", 4, 600), msg(7, "question", 5, 700, delivery=None)]
    replies = [rep(100, 2, "wrong"), rep(150, 2, "right"), rep(170, 3, "on it"), rep(200, 3, "right"),
               rep(410, 4, "wrong"), rep(510, 5, "?"), rep(520, 5, "right"), rep(610, 6, "wrong"), rep(620, 6, "?"),
               rep(720, 1, "re #7: right", eta=3)]
    answers = {"version": 2, "judge_model": "j", "judge_temperature": 0.0, "graded_at": 0.0,
               "answers": {"2": {"correct": False, "verdict": "wrong"}, "3": {"correct": False, "verdict": "none"},
                           "4": {"correct": False, "verdict": "wrong"}, "5": {"correct": None, "verdict": None},
                           "6": {"correct": False, "verdict": "wrong"}},
               "replies": {"2": [judged(100, "wrong"), judged(150, "correct")],
                           "3": [judged(170, "none"), judged(200, "correct")], "4": [judged(410, "wrong")],
                           "5": [judged(510, None), judged(520, "correct")], "6": [judged(610, "wrong")]},
               "elsewhere": {"7": [{"t": T0 + 720, "to": 1, "verdict": "correct", "correct": True, "why": "w"}],
                             "4": [{"t": T0 + 430, "to": 1, "verdict": "wrong", "correct": False, "why": "w"}]}}
    cols = question_columns(make_trial(tmp_path / "t", inbox, replies, answers))
    assert sorted(cols) == [2, 3, 4, 5, 6, 7]
    q2, q3, q4, q5, q6, q7 = (cols[i] for i in range(2, 8))
    assert (q2["correct_sec"], q2["correct_known"], q2["first_verdict"], q2["first_reply_correct"]) == (
        70, True, "wrong", False)
    assert (q2["wrong_before_correct"], q2["n_judged"], q2["n_wrong"], q2["response_sec"]) == (True, 2, 1, 60)
    assert (q3["correct_sec"], q3["wrong_before_correct"], q3["first_verdict"]) == (40, False, "none")  # "on it"
    assert (q4["correct_sec"], q4["correct_known"], q4["wrong_before_correct"], q4["n_wrong"]) == (None, True, None, 1)
    assert (q4["n_elsewhere"], q4["elsewhere_correct"]) == (1, False)
    assert (q5["correct_sec"], q5["correct_known"], q5["first_reply_correct"]) == (None, False, None)  # unread first
    assert (q6["correct_sec"], q6["correct_known"]) == (None, False)  # its second reply was not judged
    assert (q7["correct_sec"], q7["correct_known"], q7["first_verdict"], q7["first_reply_correct"]) == (
        None, True, None, None)  # never answered (never delivered): known, never
    assert (q7["n_elsewhere"], q7["elsewhere_correct"], q7["elsewhere_correct_sec"], q7["response_sec"]) == (
        1, True, 20, None)
    assert q2["elsewhere_correct"] is None and q2["n_elsewhere"] == 0


def test_question_columns_from_older_answers(tmp_path):
    """An answers.json without a version judged only the first reply: correct there is the answer;
    not correct with later replies is unknown; no answers.json leaves every answered question
    unknown. A session without replies has no question columns."""
    inbox = [msg(1, "question", 0, 0), msg(2, "question", 1, 100), msg(3, "question", 2, 200), msg(4, "question", 3, 300)]
    replies = [rep(30, 1, "a"), rep(130, 2, "a"), rep(160, 2, "b"), rep(230, 3, "a")]
    v1 = {"judge_model": "j", "answers": {"1": {"correct": True}, "2": {"correct": False}, "3": {"correct": False}}}
    cols = question_columns(make_trial(tmp_path / "a", inbox, replies, v1))
    assert (cols[1]["correct_sec"], cols[1]["correct_known"], cols[1]["first_verdict"]) == (30, True, "correct")
    assert (cols[2]["correct_sec"], cols[2]["correct_known"]) == (None, False)
    assert (cols[3]["correct_sec"], cols[3]["correct_known"], cols[3]["n_wrong"]) == (None, True, 1)
    assert (cols[4]["correct_sec"], cols[4]["correct_known"]) == (None, True)  # never answered
    none = question_columns(make_trial(tmp_path / "b", inbox, replies))
    assert [none[i]["correct_known"] for i in (1, 2, 3, 4)] == [False, False, False, True]
    assert question_columns(make_trial(tmp_path / "c", inbox, replies, v1, replies_cfg=False)) == {}
    assert verdict({"correct": True}) == "correct" and verdict({"correct": False}) == "wrong"
    assert verdict({"verdict": "none", "correct": False}) == "none" and verdict({"correct": None}) is None
    assert verdict(None) is None and verdict({"verdict": "maybe"}) is None


# --- bugs --------------------------------------------------------------------------------------------


def diff(*files: str, body: str = "x") -> str:
    return "".join(f"diff --git a/{f} b/{f}\nindex 0..1 100644\n--- a/{f}\n+++ b/{f}\n@@ -1 +1 @@\n-a\n+{body}\n"
                   for f in files)


def test_file_diffs_and_change_after():
    p = diff("src/a.py", "src/b.py", body="1")
    assert set(file_diffs(p)) == {"src/a.py", "src/b.py"} and set(file_diffs(p, {"src/b.py"})) == {"src/b.py"}
    assert file_diffs("") == {} and "".join(file_diffs(p).values()) == p
    states = [(T0 + 1, {}), (T0 + 300, file_diffs(diff("src/a.py"))), (T0 + 700, file_diffs(diff("src/a.py", "src/b.py"))),
              (T0 + 760, file_diffs(diff("src/a.py", "src/b.py", body="2")))]
    assert change_after(states, T0, ["src/a.py"]) == 300  # before the first snapshot: the clean tree
    assert change_after(states, T0 + 240, ["src/b.py"]) == 460
    assert change_after(states, T0 + 300, ["src/a.py"]) == 460  # the snapshot at its send time is the reference
    assert change_after(states, T0 + 720, ["src/a.py", "src/c.py"]) == 40  # a.py changed again
    assert change_after(states, T0 + 720, ["src/c.py"]) is None


def tool(t, name, role="orchestrator", **args):
    return {"role": role, "name": name, "t_start": T0 + t, "t_end": T0 + t + 1, "dispatch_id": None, "arguments": args}


def urgent_trial(tmp_path, condition="single_urgent"):
    inbox = [msg(1, "hint", 0, 0, "b0", files=["src/a.py"]), msg(2, "hint", 1, 240, "b1", files=["src/b.py"]),
             msg(3, "hint", 2, 480, "b2", files=["src/c.py"]),
             msg(4, "hint", 3, 720, "b3", urgent=True, files=["src/d.py"], f2p=["tests/test_d.py::test_urgent_thing"]),
             msg(5, "question", 0, 800)]
    snaps = [{"seq": 0, "t_start": T0, "t_end": T0 + 1, "changed": True, "patch": "snapshots/0000.patch", "error": None},
             {"seq": 1, "t_start": T0 + 299, "t_end": T0 + 300, "changed": True, "patch": "snapshots/0001.patch",
              "error": None},
             {"seq": 2, "t_start": T0 + 400, "t_end": T0 + 401, "changed": False, "patch": None, "error": None},
             {"seq": 3, "t_start": T0 + 699, "t_end": T0 + 700, "changed": True, "patch": "snapshots/0003.patch",
              "error": None},
             {"seq": 4, "t_start": T0 + 740, "t_end": T0 + 745, "changed": None, "patch": None, "error": "timeout"},
             {"seq": 5, "t_start": T0 + 759, "t_end": T0 + 760, "changed": True, "patch": "snapshots/0005.patch",
              "error": None},
             {"seq": 6, "t_start": T0 + 769, "t_end": T0 + 770, "changed": True, "patch": "snapshots/0006.patch",
              "error": None}]  # its patch file is missing: left out
    tools = [tool(730, "read_file", path="src/d.py"), tool(740, "edit_file", path="src/d.py"),
             tool(750, "bash", role="subagent", command="cat src/a.py"), tool(760, "edit_file", path="src/b.py"),
             tool(765, "bash", command="ls"), tool(1000, "edit_file", path="src/d.py")]
    dispatches = [{"dispatch_id": "d1", "t_start": T0 + 725, "instruction": "Fix test_urgent_thing"}]
    t = make_trial(tmp_path, inbox, condition=condition, tools=tools, dispatches=dispatches, snapshots=snaps,
                   verdicts={"b0": True, "b3": False})
    sd = tmp_path / "agent" / "snapshots"
    sd.mkdir()
    for seq, patch in ((0, ""), (1, diff("src/a.py")), (3, diff("src/a.py", "src/b.py")),
                       (5, diff("src/a.py", "src/b.py", "src/d.py"))):
        (sd / f"{seq:04d}.patch").write_text(patch)
    return t


KEYS = {"sweep": "lv", "job": "lv__r1__single_urgent", "round": 1, "condition": "single_urgent", "task_name": "task-a",
        "trial_name": "t1"}


def test_bug_rows_start_preemption_and_open_bugs(tmp_path, monkeypatch):
    """start: the earlier of a dispatch naming the bug's test and the first snapshot changing its
    file; preemption over the 240 s after delivery; the earlier bugs failing at its send time
    (unknown counts as failing unless the bug was fixed by then)."""
    t = urgent_trial(tmp_path / "t")
    calls = []

    def state_at(trial, t_session):
        calls.append(t_session)
        return {"b0": "pass", "b1": "unknown", "b2": "unknown"}

    monkeypatch.setattr(fixes_mod, "state_at", state_at)
    fixes = {"b0": {"fixed_sec": 100.0, "timing_ok": True, "fixed_end": True},
             "b1": {"fixed_sec": None, "timing_ok": True, "fixed_end": False},
             "b2": {"fixed_sec": 50.0, "timing_ok": True, "fixed_end": False, "pass_before_sent": False},
             "b3": {"fixed_sec": 60.0, "timing_ok": True, "fixed_end": False}}
    rows = {r["subtask_id"]: r for r in bug_rows(t, KEYS, True, fixes)}
    assert list(rows) == ["b0", "b1", "b2", "b3"] and calls == [240, 480, 720]
    b0, b1, b2, b3 = (rows[s] for s in ("b0", "b1", "b2", "b3"))
    assert (b3["setting"], b3["level"], b3["urgent"], b3["t_ready_sec"], b3["delivery_sec"]) == ("single", "L3", True,
                                                                                                  720, 2)
    assert (b3["dispatch_sec"], b3["change_sec"], b3["start_sec"]) == (5, 40, 5)
    assert (b0["change_sec"], b0["start_sec"], b1["change_sec"], b2["change_sec"]) == (300, 300, 460, None)
    assert (b3["preempt_n_touch"], b3["preempt_n_touch_own"], b3["preempt_n_edit"], b3["preempt_n_edit_own"]) == (
        4, 2, 2, 1)
    assert "preempt_n_touch" not in b0  # no earlier bug
    # at 720 s: b0 passes, b1 unknown and never fixed (failing), b2 unknown but fixed at 530 s (not failing)
    assert (b3["open_before"], b2["open_before"], b1["open_before"], b0["open_before"]) == (["b1"], ["b1"], [], None)
    assert (b3["timed"], b3["fixed_sec"], b0["fixed_end"], b2["pass_before_sent"]) == (True, 60.0, True, False)
    # not reproducing the verifier: no timing at all, today's end verdict kept
    bad = {k: {**v, "fixed_sec": None, "timing_ok": False} for k, v in fixes.items()}
    rows = bug_rows(t, KEYS, True, bad)
    assert not any(r["timed"] for r in rows) and all(r["open_before"] is None for r in rows)
    assert rows[0]["fixed_end"] is True and len(calls) == 3
    # not replayed: from the verifier's verdicts (no verdict in a valid trial: unfixed)
    monkeypatch.setattr(fixes_mod, "fix_times", lambda trial: {})
    rows = bug_rows(t, KEYS, True)
    assert [r["fixed_end"] for r in rows] == [True, False, False, False] and not any(r["timed"] for r in rows)
    assert bug_rows(make_trial(tmp_path / "q", [msg(1, "question", 0, 0)]), KEYS, True) == []


# --- urgency -----------------------------------------------------------------------------------------


SENDS = (0, 240, 480, 720, 960)


def session(trial, level, fixed, open3, *, task="task-a", setting="single", sweep="lv", timed=True, valid=True,
            start=None, pre=None, send3=720):
    """Bugs b0..b4 sent every 240 s (b3 at ``send3``, urgent at L3); ``fixed``: subtask -> fixed_sec."""
    out = []
    for k, sid in enumerate(("b0", "b1", "b2", "b3", "b4")):
        cond = f"{setting}_{SUFFIX[level]}"
        b = {"sweep": sweep, "job": f"{sweep}__r1__{cond}", "round": 1, "condition": cond, "task_name": task,
             "trial_name": trial, "valid": valid, "setting": setting, "level": level, "msg_id": k + 1, "hint_index": k,
             "subtask_id": sid, "urgent": level == "L3" and sid == "b3",
             "t_ready_sec": float(send3 if k == 3 else SENDS[k]),
             "timed": timed, "fixed_sec": fixed.get(sid) if timed else None, "fixed_end": None, "dispatch_sec": None,
             "change_sec": None, "start_sec": None, "open_before": None}
        if sid == "b3":
            b.update(open_before=open3 if timed else None, start_sec=start, **(pre or {}))
        out.append(b)
    return out


def pre(n_touch, own_touch, n_edit, own_edit):
    return {"preempt_n_touch": n_touch, "preempt_n_touch_own": own_touch, "preempt_n_edit": n_edit,
            "preempt_n_edit_own": own_edit}


def urgency_bugs():
    """task-a, single: u1 (L3) against w1, w2 (L2, timed), w3 (L2, not timed) and z1 (L0); the rest
    are not twins (another task, setting, sweep, send time, or invalid)."""
    return [
        *session("u1", "L3", {"b0": 100, "b2": 400, "b3": 60, "b4": 50}, ["b1", "b2"], start=20, pre=pre(4, 3, 2, 2)),
        *session("w1", "L2", {"b0": 90, "b2": 100, "b3": 300, "b4": 100}, ["b1"], start=100, pre=pre(4, 1, 2, 0)),
        *session("w2", "L2", {"b1": 500, "b2": 330, "b3": 80}, ["b1", "b2"], pre=pre(0, 0, 0, 0)),
        *session("w3", "L2", {}, [], timed=False, start=50, pre=pre(2, 1, 0, 0)),
        *session("z1", "L0", {"b0": 10}, ["b1", "b2"]),
        *session("u2", "L3", {}, [], timed=False),
        *session("x1", "L2", {"b3": 10}, ["b1"], task="task-b"),
        *session("x2", "L2", {"b3": 10}, ["b1"], setting="manager_background"),
        *session("x3", "L2", {"b3": 10}, ["b1"], sweep="other"),
        *session("x4", "L2", {"b3": 10}, ["b1"], send3=900),
        *session("x5", "L2", {"b3": 10}, ["b1"], valid=False),
    ]


def test_win_and_order():
    assert (win(60, 200), win(200, 60), win(60, 80), win(80, 60), win(60, 90.5), win(None, None)) == (
        1.0, 0.0, 0.5, 0.5, 1.0, 0.5)
    assert (win(1500, None), win(None, 10)) == (1.0, 0.0)  # only one fixed: however late
    s = {b["subtask_id"]: b for b in session("s", "L3", {"b1": 300, "b2": 240, "b3": 0}, [])}
    # b3 at 720 s; b1 at 540 s (before it), b2 at 720 s (the same time), b4 never
    assert order_score(s, "b3", ["b1"]) == 0.0 and order_score(s, "b3", ["b2"]) == 0.5
    assert order_score(s, "b3", ["b4"]) == 1.0 and order_score(s, "b3", []) is None
    assert order_score(s, "b4", ["b0"]) == 0.5  # neither fixed


BROKEN_B0 = [(0, {"b0": "fail", "b3": "fail"}), (100, {"b0": "pass", "b3": "fail"}),
             (500, {"b0": ("fail", "fail"), "b3": "fail"}), (600, {"b0": "fail", "b3": "fail"})]


@pytest.mark.parametrize("later, again, want", [
    # b0 fixed again at 1000 s, after b3's fix at 800 s
    ([(800, {"b0": "fail", "b3": "pass"}), (1000, {"b0": "pass", "b3": "pass"})], 1000, 1.0),
    # b0 fixed again at 760 s and held to 1100 s (then broke for good): before b3's fix
    ([(760, {"b0": "pass", "b3": "fail"}), (800, {"b0": "pass", "b3": "pass"}), (1100, {"b0": "fail", "b3": "pass"}),
      (1200, {"b0": "fail", "b3": "pass"})], 760, 0.0),
])
def test_order_uses_an_open_earlier_bugs_fix_after_the_send_time(tmp_path, later, again, want):
    """Earlier bug b0 is fixed at 100 s (its fixed_sec: it held 400 s), breaks at 500 s (seen twice
    over 100 s) and is fixed again at ``again``; urgent b3 is sent at 720 s, when b0 fails, and fixed
    at 800 s. The order compares 800 s with b0's first fix after 720 s, not with 100 s."""
    from test_fixes import bug as fbug, make_trial as fix_trial, timeline
    inbox = [fbug(1, 0, 0, "b0"), fbug(2, 1, 720, "b3", urgent=True)]
    t = fix_trial(tmp_path, inbox, states_=timeline(BROKEN_B0 + later), job="lv__r1__single_urgent",
                  verdicts={"b0": later[-1][1]["b0"] == "pass", "b3": True})
    rows = {r["subtask_id"]: r for r in bug_rows(t, fixes_mod.keys(t), True, hints=[])}
    b0, b3 = rows["b0"], rows["b3"]
    assert (b0["fixed_sec"], b3["fixed_sec"], b3["urgent"], b3["level"]) == (100, 80, True, "L3")
    assert b3["open_before"] == ["b0"] and b3["open_fixed_session_sec"] == {"b0": again}
    assert b0["open_fixed_session_sec"] is None
    assert fixes_mod.fixed_after(t, 720) == {"b0": again, "b3": 800}
    assert order_score(rows, "b3", ["b0"]) == want
    # a session without the column (or a bug not open then) keeps its fixed_sec fix
    assert order_score({**rows, "b3": {**b3, "open_fixed_session_sec": None}}, "b3", ["b0"]) == 0.0


def test_urgency_rows_against_twins():
    rows = {(r["trial_name"], r["twin_level"]): r for r in urgency_rows(urgency_bugs())}
    assert sorted(rows) == [("u1", "L0"), ("u1", "L2"), ("u2", "L0"), ("u2", "L2")]
    u = rows[("u1", "L2")]
    assert (u["condition"], u["subtask_id"], u["t_ready_sec"], u["fixed_sec"], u["n_twins"], u["n_twins_timed"]) == (
        "single_urgent", "b3", 720, 60, 3, 2)
    # 60 s against 300 s (sooner) and 80 s (within 30 s: a tie)
    assert (u["win_rate"], u["n_pairs"], u["n_decisive"]) == (0.75, 2, 1)
    assert (u["fixed_240"], u["fixed_480"], u["twin_fixed_240"], u["twin_fixed_480"]) == (True, True, 0.5, 1.0)
    assert u["twin_fixed_sec_median"] == 190
    # order: vs w1 over b1 (b2 was fixed in w1 by then), vs w2 over b1 and b2; w2 fixed b1 first
    assert (u["order"], u["twin_order"], u["n_order_pairs"], u["n_order_bugs"]) == (1.0, 0.75, 2, 3)
    assert (u["start_sec"], u["twin_start_sec_median"]) == (20, 100)  # [100, never, 50]
    assert (u["preempt_n_touch"], u["preempt_touch_share"], u["preempt_edit_share"]) == (4, 0.75, 1.0)
    assert u["twin_preempt_touch_share"] == pytest.approx(2 / 6) and u["twin_preempt_edit_share"] == 0.0
    # cost: the other bugs fixed in u1 (b0 100, b1 never, b2 400, b4 50) and in the timed twins
    assert (u["others_fixed_frac"], u["others_fixed_sec_median"]) == (0.75, 250)
    assert u["twin_others_fixed_frac"] == pytest.approx(5 / 8)
    z = rows[("u1", "L0")]
    assert (z["n_twins"], z["win_rate"], z["n_pairs"], z["n_decisive"], z["fixed_240"], z["twin_fixed_240"]) == (
        1, 1.0, 1, 1, True, 0.0)
    assert (z["order"], z["twin_order"]) == (1.0, 0.5)  # z1: b3 never fixed, b1 and b2 neither
    untimed = rows[("u2", "L2")]
    assert (untimed["n_twins"], untimed["n_pairs"], untimed["fixed_240"]) == (3, 0, None)
    assert math.isnan(untimed["win_rate"]) and math.isnan(untimed["order"]) and math.isnan(untimed["others_fixed_frac"])


# --- levels.csv --------------------------------------------------------------------------------------


def frames(bugs):
    """The episodes table of the sessions of ``bugs``."""
    seen = {}
    for b in bugs:
        seen.setdefault((b["sweep"], b["job"], b["trial_name"]), b)
    eps = pd.DataFrame([{"sweep": b["sweep"], "job": b["job"], "condition": b["condition"], "task_name": b["task_name"],
                         "trial_name": b["trial_name"], "valid": b["valid"]} for b in seen.values()])
    return eps


def keyed(trial, cond, task="task-a", **kw):
    return {"sweep": "lv", "job": f"lv__r1__{cond}", "condition": cond, "task_name": task, "trial_name": trial,
            "valid": True, **kw}


def test_level_table_main_numbers():
    bugs = urgency_bugs() + session("u3", "L3", {"b3": 400}, [], task="task-b") + \
        session("w9", "L2", {"b3": 100}, [], task="task-b") + \
        session("e1", "L1", {}, []) + session("e2", "L1", {}, []) + session("e3", "L1", {}, [], task="task-b")
    eps = frames(bugs)
    def estimate(trial, sid, on_time, source, task="task-a", eligible=True, stale=False, via="structured"):
        return keyed(trial, "single_eta", task, subtask_id=sid, on_time=on_time, eligible=eligible, stale=stale,
                     via=via, source=source)

    est = pd.DataFrame([estimate("e1", "b1", True, "arrival"), estimate("e1", "b1", False, "check"),
                        estimate("e1", "b1", False, "update"), estimate("e1", "b2", True, "arrival"),
                        estimate("e1", "b4", None, "arrival", eligible=False),
                        estimate("e2", "b1", False, "check", stale=True),
                        estimate("e2", "b0", False, "check", via="text_status"),
                        estimate("e3", "b1", True, "arrival", "task-b"),
                        estimate("e3", "b2", False, "arrival", "task-b", eligible=False)])
    pairs = pd.DataFrame([keyed("e1", "single_eta", answered=True, right=True),
                          keyed("e2", "single_eta", answered=False, right=None)])

    def q(trial, sec, task="task-a", known=True, reply=None, first=True, delivery=5.0, wrong=False):
        return keyed(trial, "single_questions", task, kind="question", correct_sec=sec, correct_known=known,
                     reply_sec=reply if reply is not None else sec, first_reply_correct=first, delivery_sec=delivery,
                     response_sec=None if sec is None else sec - delivery, wrong_before_correct=wrong if sec else None,
                     elsewhere_correct=None)

    msgs = pd.DataFrame([q("w1", 20), q("w1", 50, reply=40, first=False, wrong=True), q("w1", None, reply=30, first=False),
                         q("w2", 100), q("w2", 250), q("w3", 61), q("w9", 30, "task-b"), q("w9", 400, "task-b"),
                         keyed("w1", "single_questions", kind="bug", reply_sec=1.0)])
    t = level_table(eps, msgs, est, pairs, bugs, pd.DataFrame(urgency_rows(bugs)))
    t = t.set_index(["sweep", "setting", "level"])
    assert ("lv", "single", "L0") in t.index and ("other", "single", "L2") in t.index
    assert ("lv", "manager_background", "L2") in t.index and ("lv", "single", "L1") in t.index
    e = t.loc[("lv", "single", "L1")]
    # each bug once: e1 b1 1/3, e1 b2 1, e3 b1 1 (ineligible, stale and text estimates left out)
    assert (e["n_sessions"], e["n_tasks"], e["eta_n_estimates"], e["eta_n_bugs"]) == (3, 2, 5, 3)
    assert e["eta_on_time"] == pytest.approx(7 / 9) and e["eta_on_time_lo"] <= e["eta_on_time"] <= e["eta_on_time_hi"]
    assert e["eta_on_time_any_via"] == pytest.approx((1 / 3 + 1 + 1 + 0) / 4) and e["eta_n_text"] == 1
    assert e["eta_on_time_arrival"] == 1.0 and e["eta_bugs_estimated_frac"] == pytest.approx(7 / 15)  # any estimate
    assert (e["status_n_pairs"], e["status_answered_frac"], e["status_right_frac"]) == (2, 0.5, 1.0)
    assert math.isnan(e["q_correct_60s"]) and math.isnan(e["urgent_win_rate"])
    qs = t.loc[("lv", "single", "L2")]
    # task-a: w1 20, 50, never; w2 100, 250; w3 61; task-b: w9 30, 400 (8 asked)
    assert qs["q_n"] == 8 and qs["n_sessions"] == 6 and qs["n_timed"] == 5  # w1, w2, w3 (not timed), w9, x1, x4
    assert (qs["q_correct_30s"], qs["q_correct_60s"], qs["q_correct_120s"], qs["q_correct_300s"]) == (
        2 / 8, 3 / 8, 5 / 8, 6 / 8)
    assert 0 <= qs["q_correct_60s_lo"] <= qs["q_correct_60s"] <= qs["q_correct_60s_hi"] <= 1
    assert qs["q_first_correct_60s"] == 2 / 8 and qs["q_wrong_before_correct_frac"] == pytest.approx(1 / 7)
    assert qs["q_delivery_median_sec"] == 5.0 and math.isnan(qs["eta_on_time"])
    u = t.loc[("lv", "single", "L3")]
    # task-a: u1 vs w1, w2 (1, 0.5); task-b: u3 (400 s) vs x1 (10 s) and w9 (100 s): 0, 0; u2 and w3 not timed
    assert u["urgent_win_rate"] == pytest.approx(1.5 / 4) and (u["urgent_n_pairs"], u["urgent_n_decisive"]) == (4, 3)
    assert 0 <= u["urgent_win_rate_lo"] <= u["urgent_win_rate"] <= u["urgent_win_rate_hi"] <= 1
    # urgent: u1 60 s, u3 400 s; each twin once: w1 300 s, w2 80 s, x1 10 s, w9 100 s
    assert (u["urgent_fixed_240"], u["twin_fixed_240"], u["urgent_fixed_480"], u["twin_fixed_480"]) == (
        0.5, 0.75, 1.0, 1.0)
    assert (u["urgent_order"], u["twin_order"]) == (1.0, 0.75)
    assert u["n_bugs"] == 15 and u["bugs_fixed_frac"] == pytest.approx(5 / 10)  # u1 and u3 timed
    # an answered question not graded yet: no shares
    unknown = level_table(eps, pd.DataFrame([q("w1", 20), q("w2", None, known=False, reply=30)]), est, pairs, bugs,
                          pd.DataFrame()).set_index(["sweep", "setting", "level"]).loc[("lv", "single", "L2")]
    assert math.isnan(unknown["q_correct_60s"]) and math.isnan(unknown["q_correct_30s"]) and unknown["q_n"] == 2
    only_l0 = level_table(eps[eps["condition"] == "single_baseline"], msgs, est, pairs, bugs, pd.DataFrame())
    assert list(only_l0.columns) == list(t.reset_index().columns)  # the same columns whatever the levels
    text = summary_text(t.reset_index())
    assert "eta_on_time" in text and "0.78 [" in text and "sweep" in text  # two sweeps
    assert level_table(eps.assign(condition="single_requests"), msgs, est, pairs, [], pd.DataFrame()).empty


# --- end to end: python -m subagents_bench.analysis ----------------------------------------------------


def write_trial(jobs: Path, sweep: str, rnd: int, cond: str, task: str, inbox, replies=(), answers=None) -> Path:
    d = jobs / f"{sweep}__r{rnd}__{cond}" / f"{task}__{cond}{rnd}"
    (d / "agent").mkdir(parents=True)
    (d / "verifier").mkdir()
    (d / "config.json").write_text(json.dumps({"task": {"path": task}, "trial_name": d.name}))
    (d / "result.json").write_text(json.dumps({
        "task_name": task, "trial_name": d.name, "exception_info": None,
        "verifier_result": {"rewards": {"reward": 0.25, "n_resolved": 1, "visible_ok": 1.0}},
        "verifier": {"started_at": "2026-10-16T00:00:00Z"}, "verifier_environment_mode": "separate"}))
    (d / "verifier" / "eval.json").write_text(json.dumps({"visible_ok": True, "subtasks": [
        {"subtask_id": "b0", "resolved": True}]}))
    icfg = {"replies": True, "interval_sec": 240, "wall_clock_sec": 1800}
    (d / "agent" / "episode.json").write_text(json.dumps({
        "condition": cond, "topology": "single", "strategy": None, "scope": None, "orchestrator_model": "m",
        "subagent_model": None, "task_name": task, "trial_name": d.name, "config": {"interactive": icfg, "tools": {}},
        "t_start": T0, "t_end": T0 + 1800, "wall_clock_sec": 1800.0, "session_t0": T0, "exit_reason": "deadline",
        "n_provider_errors": 0}))
    for kind, rows in (("inbox", inbox), ("replies", replies)):
        (d / "agent" / f"{kind}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if answers is not None:
        (d / "agent" / "answers.json").write_text(json.dumps(answers))
    return d


def test_analysis_writes_the_levels_tables(tmp_path, monkeypatch, capsys):
    """Two sweeps: the levels sweep (L0, L2 and L3 on two tasks) and an older baseline run of the
    same L0 condition, which never pools with it; fix times and estimates are monkeypatched."""
    bugs = [msg(1, "hint", 0, 0, "b0", files=["src/a.py"]), msg(2, "hint", 1, 240, "b1", files=["src/b.py"])]
    urgent = [bugs[0], msg(2, "hint", 1, 240, "b1", urgent=True, files=["src/b.py"])]
    qs = [msg(3, "question", 0, 80), msg(4, "question", 1, 320)]
    v2 = {"version": 2, "answers": {"3": {"verdict": "correct", "correct": True}},
          "replies": {"3": [judged(110, "correct")]}, "elsewhere": {}}
    jobs, old = tmp_path / "jobs" / "lv", tmp_path / "jobs" / "old"
    fixed = {}
    for rnd in (1, 2):
        for task in ("task-a", "task-b"):
            write_trial(jobs, "lv", rnd, "single_baseline", task, bugs)
            d = write_trial(jobs, "lv", rnd, "single_questions", task, bugs + qs, [rep(110, 3, "a")], v2)
            fixed[d.name] = 300.0
            d = write_trial(jobs, "lv", rnd, "single_urgent", task, urgent + qs, [rep(110, 3, "a")], v2)
            fixed[d.name] = 100.0 if rnd == 1 else 500.0
    write_trial(old, "old", 1, "single_baseline", "task-a", bugs)

    def fix_times(t):
        if t.dir.name not in fixed:
            return {}
        return {"b0": {"fixed_sec": 400.0, "timing_ok": True, "fixed_end": True},
                "b1": {"fixed_sec": fixed[t.dir.name], "timing_ok": True, "fixed_end": False}}

    monkeypatch.setattr(fixes_mod, "fix_times", fix_times)
    monkeypatch.setattr(fixes_mod, "state_at", lambda t, sec: {"b0": "unknown"})  # fixed at 400 s: failing at 240 s
    monkeypatch.setattr(fixes_mod, "bug_fix_rows", lambda t: [{"subtask_id": s, **f} for s, f in fix_times(t).items()])
    monkeypatch.setattr(estimates_mod, "estimate_rows", lambda t, f: [  # scored only with fix times
        {"subtask_id": "b1", "on_time": True if f else None, "scored": bool(f), "eligible": True, "stale": None,
         "via": "structured", "source": "arrival"}])
    monkeypatch.setattr(estimates_mod, "estimate_table", lambda rows, pairs: pd.DataFrame({"n_estimates": [len(rows)]}))
    monkeypatch.setattr(estimates_mod, "status_pair_rows", lambda t, f: [])
    out = tmp_path / "out"
    assert analyze([str(jobs), str(old), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "levels (main numbers" in printed and "urgent_win_rate" in printed
    for name in ("fixes", "estimates", "urgency", "levels"):  # status_pairs: no rows, an empty file
        assert (out / f"{name}.csv").is_file()
    conds = pd.read_csv(out / "conditions.csv")
    base = conds[conds["condition"] == "single_baseline"]
    assert list(base["sweep"]) == ["lv", "old"] and list(base["n"]) == [4, 1]  # never pooled, side by side
    assert set(pd.read_csv(out / "requests.csv")["sweep"]) == {"lv", "old"}
    inter = pd.read_csv(out / "interactivity.csv")
    assert len(inter[inter["condition"] == "single_baseline"]) == 2
    msgs = pd.read_csv(out / "messages.csv")
    q = msgs[(msgs["kind"] == "question") & (msgs["msg_id"] == 3)]
    assert len(q) == 8 and set(q["correct_sec"]) == {30.0} and q["correct_known"].all()
    assert len(pd.read_csv(out / "fixes.csv")) == 16 and len(pd.read_csv(out / "estimates.csv")) == 13
    assert pd.read_csv(out / "eta.csv")["n_estimates"].tolist() == [13]
    urg = pd.read_csv(out / "urgency.csv")
    assert len(urg) == 8 and set(urg["twin_level"]) == {"L2", "L0"}
    lv = pd.read_csv(out / "levels.csv").set_index(["sweep", "setting", "level"])
    assert list(lv.index) == [("lv", "single", "L0"), ("lv", "single", "L2"), ("lv", "single", "L3"),
                              ("old", "single", "L0")]
    # L3: per task, r1 (100 s) beats both L2 twins (300 s), r2 (500 s) loses to both
    l3 = lv.loc[("lv", "single", "L3")]
    assert (l3["urgent_win_rate"], l3["urgent_n_pairs"], l3["urgent_n_decisive"], l3["n_timed"]) == (0.5, 8, 8, 4)
    assert 0 <= l3["urgent_win_rate_lo"] < 0.5 < l3["urgent_win_rate_hi"] <= 1
    assert l3["q_correct_60s"] == 0.5 and l3["eta_on_time"] == 1.0
    assert math.isnan(lv.loc[("lv", "single", "L0"), "eta_on_time"]) and lv.loc[("lv", "single", "L0"), "n_timed"] == 0
    assert l3["urgent_order"] == 0.5 and l3["twin_order"] == 0.0  # b0 at 400 s: before r2's and the twins' b1
