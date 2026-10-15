"""Requests (``interactive.replies``): the message schedule with questions, status checks and urgent
bugs, the ``reply`` tool in a session with a fake model, the request analysis on synthetic records,
and the code questions (source files, generation, judging, ``subagents-bench questions`` and
``grade-answers``), all offline."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest

from fakes import ORCH, FakeFactory, Reply, assert_consistent, make_config, merge, tc
from subagents_bench import hints as hints_mod
from subagents_bench.analysis.__main__ import main as analyze
from subagents_bench.analysis.load import Trial
from subagents_bench.analysis.requests import request_rows, request_table
from subagents_bench.config import ConfigError
from subagents_bench.environment import LocalShellEnvironment
from subagents_bench.hints import (HintError, Question, fetch_source, generate_questions, judge_answer, load_questions,
                                   question_files, schedule, source_ref, write_questions)
from subagents_bench.interactive import InteractiveOrchestrator
from subagents_bench.prompts import interactive_prompt, request_text
from subagents_bench.records import EpisodeRecorder
from test_interactive import interactive_spec, make_task, trial_dir

REQ = {"replies": True, "questions_per_interval": 2, "status_every": 2, "urgent_every": 3}


def questions(n: int) -> list[Question]:
    return [Question(f"q{i + 1}", f"src/f{i}.py", f"What does f{i} return?", f"{i}") for i in range(n)]


def req_cfg(**interactive):
    return make_config(interactive={"wall_clock_sec": 2040, "interval_sec": 240, "max_hints": 6, **REQ,
                                    **interactive}).interactive


# --- config ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("over, msg", [
    ({"questions_per_interval": -1, "replies": True}, r"questions_per_interval: expected an integer >= 0, got -1"),
    ({"status_every": True, "replies": True}, r"status_every: expected an integer >= 0, got True"),
    ({"urgent_every": 1.5}, r"urgent_every: expected an integer >= 0, got 1.5"),
    ({"urgent_every": -2}, r"urgent_every: expected an integer >= 0"),
    ({"questions_per_interval": 2}, r"answered with `reply`: set replies: true"),
    ({"status_every": 1}, r"answered with `reply`: set replies: true"),
    ({"replies": "yes"}, r"replies: expected true or false, got 'yes'"),
    ({"replies": 1}, r"replies: expected true or false"),
])
def test_bad_request_configs(over, msg):
    with pytest.raises(ConfigError, match=msg):
        make_config(interactive=over)


def test_request_config_defaults_and_round_trip():
    i = make_config(interactive={}).interactive
    assert (i.replies, i.questions_per_interval, i.status_every, i.urgent_every) == (False, 0, 0, 0)
    cfg = make_config(interactive=REQ)
    assert {k: cfg.to_dict()["interactive"][k] for k in REQ} == REQ
    assert make_config(interactive={"replies": True}).interactive.replies is True  # replies alone: bugs with ids


@pytest.mark.xfail(reason="urgent_every without replies marks bugs urgent (inbox urgent: true) but request_text "
                          "drops the flag for a message without an id, so the agent is never told; the config "
                          "should reject it like questions and status checks, or the hint should say it")
def test_urgent_without_replies_is_rejected_or_told():
    try:
        cfg = make_config(interactive={"urgent_every": 1, "interval_sec": 0.5, "wall_clock_sec": 4.0})
    except ConfigError:
        return
    assert "urgent" in request_text("bug", None, "There is a bug.", urgent=True)
    assert cfg.interactive.urgent_every == 1


# --- schedule ----------------------------------------------------------------------------------------


def test_schedule_interleaves_questions_and_status_checks(tmp_path):
    """The requests spec's stream: 6 bugs every 240 s, 2 questions between bugs (80 and 160 s after
    each), a status check 120 s after bugs 1, 3 and 5, bug 3 urgent; ids 1, 2, ... in sending order."""
    task = make_task(tmp_path, 8)
    write_questions(tmp_path / "hints", "task-1", ORCH, questions(14))
    sched = schedule(req_cfg(), task, "task-1", tmp_path / "hints")
    expect = []
    for k in range(6):
        expect += [("bug", 240 * k), ("question", 240 * k + 80)]
        expect += [("status", 240 * k + 120)] if k % 2 == 1 else []
        expect += [("question", 240 * k + 160)]
    assert [(h.kind, h.t_offset) for h in sched] == expect and len(sched) == 21
    assert [h.msg_id for h in sched] == list(range(1, 22))
    by = {k: [h for h in sched if h.kind == k] for k in ("bug", "question", "status")}
    assert [h.index for h in by["bug"]] == list(range(6)) and [h.index for h in by["question"]] == list(range(12))
    assert [h.index for h in by["status"]] == [0, 1, 2]
    assert [h.urgent for h in by["bug"]] == [False, False, False, True, False, False]
    assert not any(h.urgent for h in by["question"] + by["status"])
    assert all(h.subtask is None and h.question is None and h.text == "" for h in by["status"])
    qs = [h.question for h in by["question"]]
    assert len({q.id for q in qs}) == 12 and all(h.subtask is None and h.text == h.question.question for h in by["question"])
    assert all(h.subtask is not None and h.question is None for h in by["bug"])
    assert schedule(req_cfg(), task, "task-1", tmp_path / "hints") == sched  # deterministic
    assert [h.subtask.id for h in by["bug"]] == [h.subtask.id for h in schedule(
        req_cfg(replies=False, questions_per_interval=0, status_every=0), task, "task-1")]  # the same bugs
    urgent_every_2 = schedule(req_cfg(urgent_every=2), task, "task-1", tmp_path / "hints")
    assert [h.index for h in urgent_every_2 if h.urgent] == [2, 4]


def test_schedule_ties_wall_clock_and_question_pool(tmp_path):
    task = make_task(tmp_path, 4)
    hd = tmp_path / "hints"
    write_questions(hd, "task-1", ORCH, questions(4))
    # one question halfway, where the status check is too: the bug first, then the status check, then the question
    sched = schedule(req_cfg(questions_per_interval=1, status_every=1, max_hints=2), task, "task-1", hd)
    assert [(h.msg_id, h.kind, h.t_offset) for h in sched] == [
        (1, "bug", 0), (2, "status", 120), (3, "question", 120), (4, "bug", 240), (5, "status", 360),
        (6, "question", 360)]
    # nothing at or past the wall clock: 3 bugs fit in 600 s, the status check and question at 600 do not
    short = schedule(req_cfg(wall_clock_sec=600, max_hints=None, questions_per_interval=1, status_every=1), task,
                     "task-1", hd)
    assert [(h.kind, h.t_offset) for h in short] == [("bug", 0), ("status", 120), ("question", 120), ("bug", 240),
                                                     ("status", 360), ("question", 360), ("bug", 480)]
    with pytest.raises(HintError, match=r"task-1: 12 question slots but 4 questions .*--per-task"):
        schedule(req_cfg(), make_task(tmp_path / "b", 8), "task-1", hd)
    with pytest.raises(HintError, match=r"no questions .*run `subagents-bench questions`"):
        schedule(req_cfg(), task, "task-2", hd)
    with pytest.raises(HintError, match="questions need a hints dir"):
        schedule(req_cfg(), task, "task-1", None)
    # replies alone or status checks only: no question file needed, every message has an id
    plain = schedule(req_cfg(questions_per_interval=0, status_every=0, urgent_every=0), task, "task-1", None)
    assert [(h.kind, h.msg_id) for h in plain] == [("bug", k + 1) for k in range(4)]
    status = schedule(req_cfg(questions_per_interval=0, status_every=3), task, "task-1", None)
    assert [h.kind for h in status] == ["bug", "bug", "bug", "status", "bug"] and status[3].t_offset == 600


def test_schedule_without_replies_is_unchanged(tmp_path):
    task = make_task(tmp_path, 14)
    cfg = make_config(interactive={}).interactive
    sched = schedule(cfg, task, "task-1")
    assert len(sched) == 12 and [h.t_offset for h in sched] == [300.0 * k for k in range(12)]
    assert all(h.kind == "bug" and h.msg_id is None and not h.urgent and h.question is None for h in sched)
    assert [h.index for h in sched] == list(range(12))


def test_request_texts():
    assert request_text("bug", None, "There is a bug.", first=True) == "Hint: There is a bug."
    assert request_text("bug", None, "B.").startswith("Another one, while you keep working") and "Hint: B." in request_text("bug", None, "B.")
    assert request_text("bug", 1, "B.", first=True) == "Request #1: a bug.\n\nHint: B."
    assert request_text("bug", 4, "B.") == "Request #4: another bug, while you keep working on the earlier ones.\n\nHint: B."
    assert request_text("bug", 7, "B.", urgent=True).startswith("Request #7: another bug, and this one is urgent")
    assert request_text("question", 2, "What is x?") == "Question #2: What is x?"
    assert request_text("status", 5, "").startswith("Status check #5: how long until each of your open bug requests")
    with_replies = interactive_prompt(make_config(interactive=REQ), "/w")
    assert "## Replying to the user" in with_replies and "- `reply`: send a message to the user" in with_replies
    without = interactive_prompt(make_config(interactive={}), "/w")
    assert "Replying to the user" not in without and "- `reply`" not in without


# --- the session -------------------------------------------------------------------------------------


FAST = {"interactive": {"wall_clock_sec": 4.0, "interval_sec": 0.5, "max_hints": 2, "replies": True,
                        "questions_per_interval": 1, "urgent_every": 1},
        "budget": {"orchestrator_steps": 40}, "topology": "single"}


def req_session(tmp_path, scripts, **overrides):
    """One session with replies in ``tmp_path/ep``: bugs #1 (0 s) and #3 (0.5 s, urgent), questions #2
    (0.25 s) and #4 (0.75 s) -> (exit, records, factory, hints)."""
    cfg = make_config(**merge(FAST, overrides))
    task = make_task(tmp_path, 2)
    write_questions(tmp_path / "hints", "task-1", ORCH, questions(4))
    hints = schedule(cfg.interactive, task, "task-1", tmp_path / "hints")
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True, exist_ok=True)
    f = FakeFactory(scripts)
    env = LocalShellEnvironment(os.path.realpath(work))
    rec = EpisodeRecorder(tmp_path / "ep", cfg, task_name="task-1")
    try:
        exit_reason = InteractiveOrchestrator(cfg, env, rec, env.workdir, hints, f).run("Fix every bug.")
    finally:
        env.close()
    return exit_reason, assert_consistent(tmp_path / "ep"), f, hints


def replying():
    """Step 1: replies to #1 (an estimate, then "#1" with a bad estimate, then a negative one) and to
    #2 before it arrived. Later: answers each question, says each new bug done, and once replies to
    #99; otherwise waits."""
    state = {"unknown": False}

    def orch(system, messages, tools):
        if sum(1 for m in messages if m.role == "assistant") == 0:
            return Reply("", [tc("reply", to=1, text="On it.", eta_min=5), tc("reply", to=2, text="too early"),
                              tc("reply", to="#1", text="Update.", eta_min="soon"),
                              tc("reply", to=1, text="Again.", eta_min=-3), tc("reply", to=1, text="  ")])
        last = messages[-1]
        calls = []
        if last.role == "user":
            for i in re.findall(r"Question #(\d+):", last.content):
                calls.append(tc("reply", to=int(i), text=f"Answer to #{i}."))
            for i in re.findall(r"Request #(\d+):", last.content):
                calls.append(tc("reply", to=int(i), text="Fixed.", eta_min=0.5, done="true"))
            if calls and not state["unknown"]:
                state["unknown"] = True
                calls.append(tc("reply", to=99, text="?"))
        return Reply("", calls or [tc("wait")])

    return {ORCH: orch}


def test_session_with_replies_records_replies_and_messages(tmp_path):
    exit_reason, r, f, hints = req_session(tmp_path, replying())
    assert exit_reason == "idle"
    first = f[ORCH].requests[0]
    assert first["messages"][0].content.endswith("every rule above still applies.\n\nRequest #1: a bug.\n\nHint: "
                                                 f"{hints[0].text}")
    assert "reply" in first["tools"] and "## Replying to the user" in first["system"]
    users = [m["content"] for m in r["transcript"] if m["role"] == "orchestrator" and m["msg_role"] == "user"]
    assert any(f"Message from the user:\nQuestion #2: {hints[1].question.question}" in u for u in users)
    assert any("Message from the user:\nRequest #3: another bug, and this one is urgent" in u for u in users)
    assert any(f"Question #4: {hints[3].question.question}" in u for u in users)

    inbox = [x for x in r["inbox"] if x["kind"] != "report"]
    assert [(x["msg_id"], x["kind"], x["hint_index"], x["urgent"]) for x in inbox] == [
        (1, "hint", 0, False), (2, "question", 0, False), (3, "hint", 1, True), (4, "question", 1, False)]
    assert all(x["t_delivered"] is not None for x in inbox)
    q = inbox[1]
    assert (q["subtask_id"], q["target_files"], q["f2p"]) == (None, None, None)
    assert (q["question_id"], q["question"], q["reference"]) == (hints[1].question.id, hints[1].question.question,
                                                                 hints[1].question.answer)
    assert inbox[0]["subtask_id"] == hints[0].subtask.id and inbox[0]["question_id"] is None

    reps = r["replies"]
    assert [(x["to"], x["kind"], x["eta_min"], x["done"]) for x in reps] == [
        (1, "bug", 5.0, False), (1, "bug", None, False), (1, "bug", None, False),  # "#1"; "soon" and -3 dropped
        (2, "question", None, False), (3, "bug", 0.5, True), (4, "question", None, False)]
    assert [x["text"] for x in reps] == ["On it.", "Update.", "Again.", "Answer to #2.", "Fixed.", "Answer to #4."]
    assert reps[0]["subtask_id"] == hints[0].subtask.id and reps[0]["question_id"] is None
    assert reps[3]["question_id"] == hints[1].question.id and reps[3]["subtask_id"] is None
    assert reps[4]["hint_index"] == 1 and all(x["step"] >= 1 for x in reps)
    assert reps == sorted(reps, key=lambda x: x["t"])

    results = [x for m in f[ORCH].requests[-1]["messages"] if m.role == "tool" for x in m.tool_results]
    early = next(x for x in results if "must be the id of a message you received" in x.content)
    assert early.is_error and "you received (#1)" in early.content  # #2 had not arrived yet
    assert any(x.is_error and "must be the id" in x.content and "(#1, #2" in x.content for x in results)  # #99
    assert any(x.is_error and "`text` must be a non-empty message" in x.content for x in results)
    assert sum(1 for x in results if x.content.startswith("[sent to the user, re #")) == 6
    assert sum(1 for x in r["tools"] if x["name"] == "reply") == 9
    ep = r["episode"]
    assert (ep["n_hints"], ep["n_hints_delivered"], ep["n_questions"], ep["n_replies"]) == (2, 2, 2, 6)


def test_reply_is_not_a_tool_without_replies(tmp_path):
    from test_interactive import session
    script = [Reply("", [tc("reply", to=1, text="x")]), Reply("", [tc("wait")])]
    _, r, f, _, _ = session(tmp_path, {ORCH: script}, n_hints=1, topology="single")
    assert "reply" not in f[ORCH].requests[0]["tools"] and r["replies"] == []
    res = f[ORCH].requests[1]["messages"][-1].tool_results[0]
    assert res.is_error and "unknown tool 'reply'" in res.content
    assert r["inbox"][0]["msg_id"] is None and r["episode"]["n_questions"] == 0


def test_undelivered_messages_cannot_be_replied_to_and_analysis_of_a_session(tmp_path):
    """The deadline cuts the session after #2: #3 and #4 are recorded undelivered; analyze writes
    messages.csv and requests.csv from the real records."""
    script = [Reply("", [tc("bash", command="sleep 0.35")]), Reply("", [tc("reply", to=2, text="Answer."),
                                                                       tc("reply", to=3, text="early")]),
              Reply("", [tc("bash", command="sleep 30")])]
    exit_reason, r, f, hints = req_session(tmp_path, {ORCH: script}, interactive={"wall_clock_sec": 0.8})
    assert exit_reason == "deadline"
    assert [(x["msg_id"], x["t_delivered"] is None) for x in r["inbox"]] == [(1, False), (2, False), (3, True),
                                                                             (4, True)]
    assert [x["to"] for x in r["replies"]] == [2]
    err = f[ORCH].requests[2]["messages"][-1].tool_results[1]
    assert err.is_error and "you received (#1, #2)" in err.content
    d = trial_dir(tmp_path / "jobs" / "rq", "rq__r1__solo", tmp_path / "ep", {hints[0].subtask.id: True,
                                                                               hints[2].subtask.id: False})
    (d / "agent" / "answers.json").write_text(json.dumps({"answers": {"2": {"correct": True, "why": "same"}}}))
    out = tmp_path / "analysis"
    assert analyze([str(tmp_path / "jobs" / "rq"), "--out", str(out)]) == 0
    msgs = pd.read_csv(out / "messages.csv")
    assert list(msgs["msg_id"]) == [1, 2, 3, 4] and list(msgs["kind"]) == ["bug", "question", "bug", "question"]
    assert msgs["reply_sec"].notna().tolist() == [False, True, False, False]
    assert list(msgs["correct"].iloc[[1, 3]]) == [True, False]  # the graded answer; #4 never answered
    assert list(pd.read_csv(out / "hints.csv")["urgent"]) == [False, True]  # bugs only, with the urgent flag
    req = pd.read_csv(out / "requests.csv").iloc[0]
    assert (req["condition"], req["n_questions"], req["n_bugs"], req["n_urgent"]) == ("solo", 2, 2, 1)
    assert (req["answered_frac"], req["correct_frac"], req["fixed_frac"]) == (0.5, 0.5, 0.5)
    assert pd.isna(req["eta_given_frac"]) or req["eta_given_frac"] == 0.0


# --- analysis ----------------------------------------------------------------------------------------


T0 = 1000.0


def msg(mid, kind, idx, t, sub=None, urgent=False, delivered=True, question=None):
    return {"msg_id": mid, "kind": kind, "hint_index": idx, "t_ready": T0 + t,
            "t_delivered": T0 + t + 1 if delivered else None, "subtask_id": sub, "urgent": urgent,
            "question_id": f"q{mid}" if kind == "question" else None, "question": question,
            "reference": "ref" if kind == "question" else None}


def rep(t, to, text="x", eta=None, done=False):
    return {"t": T0 + t, "to": to, "text": text, "eta_min": eta, "done": done}


def synthetic_trial(tmp_path: Path, inbox, replies, verdicts, answers=None, t_end=600.0, replies_cfg=True,
                    visible_ok=True) -> Trial:
    d = tmp_path / "trial"
    (d / "agent").mkdir(parents=True)
    (d / "verifier").mkdir()
    (d / "verifier" / "eval.json").write_text(json.dumps({"visible_ok": visible_ok, "subtasks": [
        {"subtask_id": k, "resolved": v} for k, v in verdicts.items()]}))
    if answers is not None:
        (d / "agent" / "answers.json").write_text(json.dumps({"judge_model": "j", "answers": answers}))
    ep = {"config": {"interactive": {"replies": replies_cfg}}, "session_t0": T0,
          "t_end": None if t_end is None else T0 + t_end}
    return Trial(d, "s", 1, "c", None, ep, {"inbox": inbox, "replies": replies})


def scenario(tmp_path, **kw):
    inbox = [msg(1, "hint", 0, 0, "b0"), msg(2, "question", 0, 80, question="Q2?"), msg(3, "hint", 1, 240, "b1"),
             msg(4, "status", 0, 360), msg(5, "question", 1, 400, question="Q5?"),
             msg(6, "hint", 2, 480, "b2", urgent=True), msg(7, "hint", 3, 500, "b3")]
    replies = [rep(10, 1, eta=2), rep(90, 2, "A2"), rep(200, 1, done=True), rep(250, 3, eta=1),
               rep(375, 3, eta=10), rep(490, 6, eta=1), rep(510, 7, eta=30), rep(520, 6, done=True)]
    return synthetic_trial(tmp_path, inbox, list(reversed(replies)),  # unsorted on disk: sorted by time
                           {"b0": True, "b1": False, "b2": True}, {"2": {"correct": True}}, **kw)


def test_request_rows(tmp_path):
    rows = {r["msg_id"]: r for r in request_rows(scenario(tmp_path), {"condition": "c", "trial_name": "t"}, True)}
    assert list(rows) == [1, 2, 3, 4, 5, 6, 7]
    assert [rows[i]["kind"] for i in rows] == ["bug", "question", "bug", "status", "question", "bug", "bug"]
    b0, q2, b1, s4, q5, b2, b3 = (rows[i] for i in range(1, 8))
    assert (b0["t_ready_sec"], b0["delivery_sec"], b0["reply_sec"], b0["n_replies"]) == (0, 1, 10, 2)
    # b0: estimate 2 min at 10 s (due 130 s), done at 200 s: 190 s for 120 predicted, late
    assert (b0["eta_sec"], b0["eta_min"], b0["done_sec"], b0["n_eta_updates"]) == (10, 2, 200, 1)
    assert b0["eta_log2_err"] == pytest.approx(math.log2(190 / 120)) and b0["on_time"] is False
    assert (b0["fixed"], b0["claim_ok"], b0["urgent_first"]) == (True, True, None)
    # questions: the first reply is the answer; #5 never answered is wrong, not ungraded
    assert (q2["reply_sec"], q2["answer"], q2["correct"], q2["question"], q2["reference"]) == (10, "A2", True, "Q2?", "ref")
    assert (q5["reply_sec"], q5["answer"], q5["correct"]) == (None, None, False)
    # b1: estimate 1 min at 250 s, never said done, the session ran past 310 s: late; not a claim
    assert (b1["eta_sec"], b1["eta_min"], b1["n_eta_updates"], b1["done_sec"]) == (10, 1, 2, None)
    assert (b1["on_time"], b1["eta_log2_err"], b1["fixed"], b1["claim_ok"]) == (False, None, False, None)
    # the status check: the first reply of any kind after it (#3's update at 375 s)
    assert s4["reply_sec"] == 15 and s4["kind"] == "status" and "eta_sec" not in s4
    # b2 urgent: done at 520 s, before b1 (open, never done); b0 was done before b2 arrived
    assert b2["urgent"] and b2["urgent_first"] == 1.0 and b2["on_time"] is True
    assert b2["eta_log2_err"] == pytest.approx(-1.0)  # 30 s for 60 predicted
    # b3: a 30 min estimate the session ended before: censored; no verdict in a valid trial: unfixed
    assert (b3["on_time"], b3["fixed"], b3["claim_ok"], b3["urgent"]) == (None, False, None, False)


def test_request_rows_urgency_censoring_and_validity(tmp_path):
    inbox = [msg(1, "hint", 0, 0, "b0"), msg(2, "hint", 1, 100, "b1", urgent=True),
             msg(3, "hint", 2, 200, "b2", urgent=True), msg(4, "hint", 3, 300, "b3", urgent=True)]
    replies = [rep(150, 2, done=True), rep(250, 1, eta=1), rep(260, 1, done=True), rep(280, 3, done=True),
               rep(300, 4, eta=0.5)]
    t = synthetic_trial(tmp_path, inbox, replies, {"b0": True}, t_end=None)
    rows = {r["msg_id"]: r for r in request_rows(t, {}, True)}
    assert rows[2]["urgent_first"] == 1.0  # done at 150 s, before b0 (260 s)
    assert rows[3]["urgent_first"] == 0.0  # done at 280 s, after b0; b1 was done before it was sent
    assert rows[4]["urgent_first"] is None  # nothing open when it was sent
    assert rows[1]["on_time"] is True and rows[1]["done_sec"] == 260 and rows[1]["eta_log2_err"] == -1.0  # 10 s: 30 s floor
    # no t_end: the session ends at the last event (300 s), before b3's due time (330 s): censored
    assert rows[4]["on_time"] is None
    late = synthetic_trial(tmp_path / "late", inbox, replies, {"b0": True}, t_end=331.0)
    assert {r["msg_id"]: r["on_time"] for r in request_rows(late, {}, True)}[4] is False
    assert rows[2]["claim_ok"] is False and rows[1]["claim_ok"] is True
    invalid = {r["msg_id"]: r for r in request_rows(t, {}, False)}
    assert invalid[3]["fixed"] is None and invalid[3]["claim_ok"] is None  # unknown in an invalid trial
    broken = synthetic_trial(tmp_path / "b", inbox, replies, {"b0": True}, visible_ok=False)
    assert {r["msg_id"]: r["fixed"] for r in request_rows(broken, {}, True)}[1] is False  # visible tests broke
    assert request_rows(synthetic_trial(tmp_path / "c", inbox, replies, {}, replies_cfg=False), {}, True) == []
    assert request_rows(synthetic_trial(tmp_path / "d", [], [], {}), {}, True) == []
    no_answers = synthetic_trial(tmp_path / "e", [msg(1, "question", 0, 0, question="Q?")], [rep(5, 1, "A")], {})
    assert request_rows(no_answers, {}, True)[0]["correct"] is None  # answered, not graded yet


def test_request_table(tmp_path):
    rows = pd.DataFrame(request_rows(scenario(tmp_path), {"condition": "c", "trial_name": "t"}, True))
    c = request_table(rows).iloc[0]
    assert (c["condition"], c["n_episodes"], c["n_questions"], c["n_bugs"], c["n_status"], c["n_urgent"]) == (
        "c", 1, 2, 4, 1, 1)
    assert (c["answered_frac"], c["answered_60s"], c["answered_180s"], c["correct_frac"]) == (0.5, 0.5, 0.5, 0.5)
    assert c["n_graded_answers"] == 2 and pd.isna(c["answer_median_sec"])  # [10, never]: infinite median
    assert (c["fixed_frac"], c["eta_given_frac"], c["done_claimed_frac"], c["claim_precision"]) == (0.5, 1.0, 0.5, 1.0)
    assert c["on_time_frac"] == pytest.approx(1 / 3)  # [late, late, on time]; b3 censored
    assert c["eta_abs_log2_median"] == pytest.approx((math.log2(190 / 120) + 1) / 2)
    assert c["eta_log2_median"] == pytest.approx((math.log2(190 / 120) - 1) / 2)
    assert (c["first_reply_median_sec"], c["eta_median_sec"], c["eta_min_median"]) == (10, 10, 1.5)
    assert (c["status_reply_median_sec"], c["urgent_first_frac"]) == (15, 1.0)
    rows["valid"] = False
    assert request_table(rows).empty and request_table(pd.DataFrame()).empty


# --- questions about the code ------------------------------------------------------------------------


def write(p: Path, size: int) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x" * size)


def test_question_files(tmp_path):
    src = tmp_path / "src"
    ok = ["pkg/core.py", "pkg/sub/types.go", "lib/util.rs", "cmd/main.c"]
    for f in ok:
        write(src / f, 2000)
    for f in ("pkg/target.py", "tests/conftest.py", "pkg/test_helpers.py", "pkg/tests/x.py", "docs/conf.py",
              "examples/demo.py", "vendor/dep.go", ".github/ci.py", "pkg/README.md", "pkg/core.pyc"):
        write(src / f, 2000)
    write(src / "pkg" / "tiny.py", 100)
    write(src / "pkg" / "huge.py", 50000)
    files = question_files(src, {"pkg/target.py"}, "task-1")
    assert sorted(files) == sorted(ok)
    assert question_files(src, {"pkg/target.py"}, "task-1") == files  # shuffled by the seed: reproducible
    assert {tuple(question_files(src, set(), s)) for s in ("a", "b", "c", "d", "e")} != {tuple(files)}


DOCKERFILE = """\
FROM python:3.11
RUN git clone --quiet --no-checkout https://github.com/pallets/click.git /src \\
 && git -C /src checkout --quiet 8.0.1 \\
 && find /src -name .git -prune -exec rm -rf {} +
WORKDIR /click
"""


def test_source_ref(tmp_path):
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "Dockerfile").write_text(DOCKERFILE)
    assert source_ref(tmp_path) == ("https://github.com/pallets/click.git", "8.0.1", "/click")
    (tmp_path / "environment" / "Dockerfile").write_text(
        "RUN git clone https://github.com/a/b /src && git -C /src checkout v1.2\n")
    assert source_ref(tmp_path) == ("https://github.com/a/b", "v1.2", "/")
    (tmp_path / "environment" / "Dockerfile").write_text("FROM scratch\nCOPY . /src\n")
    with pytest.raises(HintError, match="no `git clone URL /src`"):
        source_ref(tmp_path)


def _repo(tmp_path: Path) -> tuple[Path, str]:
    """A local repository with a tag and a commit after it -> (path, the tagged commit's sha)."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True)  # noqa: E731
    git("init", "-q")
    git("config", "user.email", "t@x")
    git("config", "user.name", "t")
    (repo / "a.py").write_text("v1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "one")
    git("tag", "v1")
    sha = git("rev-parse", "HEAD").stdout.strip()
    (repo / "a.py").write_text("v2\n")
    git("commit", "-q", "-am", "two")
    return repo, sha


def test_fetch_source_clones_the_tag_once(tmp_path, monkeypatch):
    repo, _ = _repo(tmp_path)
    monkeypatch.setattr(hints_mod, "source_ref", lambda d: (repo.as_uri(), "v1", "/w"))
    dest = fetch_source(tmp_path / "task", tmp_path / "src" / "task")
    assert (dest / "a.py").read_text() == "v1\n"
    monkeypatch.setattr(hints_mod, "source_ref", lambda d: pytest.fail("cloned twice"))
    assert fetch_source(tmp_path / "task", dest) == dest


@pytest.mark.xfail(reason="fetch_source clones with `git clone --branch <ref>`, which takes only a branch or tag; "
                          "4 SWE-sweep Dockerfiles check out a commit sha (e.g. 12f392c3d31e..., c9c20bf3683f)")
def test_fetch_source_at_a_commit_sha(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)
    monkeypatch.setattr(hints_mod, "source_ref", lambda d: (repo.as_uri(), sha, "/w"))
    assert (fetch_source(tmp_path / "task", tmp_path / "src" / "task") / "a.py").read_text() == "v1\n"


def test_generate_and_load_questions(tmp_path):
    task = make_task(tmp_path, 2)  # target files src/m0.py, src/m1.py: never asked about
    src = tmp_path / "repo"
    for f in ("src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/m0.py"):
        write(src / f, 2000)
    replies = {
        "src/b.py": 'Sure:\n[{"question": " What does b return? ", "answer": " 1 "}, {"question": "Qb2", "answer": "Ab2"},'
                    ' {"question": "Qb3", "answer": "Ab3"}]',
        "src/c.py": "I cannot read this file.",
        "src/d.py": '[{"question": "Qd1", "answer": ""}, {"question": "Qd2", "answer": "Ad2"}, "junk"]',
    }

    def model(system, messages, tools):
        rel = messages[0].content.split("\n", 1)[0].removeprefix("File: ")
        assert rel not in ("src/a.py", "src/m0.py") and tools == []
        return Reply(replies[rel])

    client = FakeFactory({ORCH: model})(make_config().orchestrator)
    kept = [Question("q1", "src/a.py", "What does a do?", "Nothing.")]
    qs = generate_questions(task, src, client, 4, kept)
    assert qs[0] == kept[0] and [q.id for q in qs] == ["q1", "q2", "q3", "q4"]
    assert sorted((q.file, q.question, q.answer) for q in qs[1:]) == [
        ("src/b.py", "Qb2", "Ab2"), ("src/b.py", "What does b return?", "1"), ("src/d.py", "Qd2", "Ad2")]
    assert "Write 2 questions" in client.requests[0]["system"]
    assert client.requests[0]["messages"][0].content.endswith("x" * 2000)
    write_questions(tmp_path / "hints", "task-1", ORCH, qs)
    assert load_questions(tmp_path / "hints", "task-1") == qs
    saved = json.loads((tmp_path / "hints" / "task-1.questions.json").read_text())
    assert (saved["task"], saved["model"], saved["questions"][0]) == ("task-1", ORCH, {
        "id": "q1", "file": "src/a.py", "question": "What does a do?", "answer": "Nothing."})
    with pytest.raises(HintError, match=r"only 4 questions of 5"):
        generate_questions(task, src, FakeFactory({ORCH: model})(make_config().orchestrator), 5, kept)
    (tmp_path / "hints" / "task-1.questions.json").write_text('{"questions": [{"id": 1}]}')
    with pytest.raises(HintError, match="no questions"):
        load_questions(tmp_path / "hints", "task-1")


def judge(text: str):
    return FakeFactory({ORCH: [Reply(text)]})(make_config().orchestrator)


def test_judge_answer():
    client = judge('{"correct": true, "why": "Same value."}')
    assert judge_answer(client, "What does f return?", "1", "It returns 1.") == (True, "Same value.")
    prompt = client.requests[0]["messages"][0].content
    assert prompt == "Question: What does f return?\n\nReference answer: 1\n\nAnswer to grade: It returns 1."
    assert client.requests[0]["tools"] == [] and "You grade answers" in client.requests[0]["system"]
    assert judge_answer(judge('Verdict:\n{"correct": false}'), "q", "r", "a") == (False, "")
    assert judge_answer(judge("I think it is right."), "q", "r", "a") == (None, "I think it is right.")
    assert judge_answer(judge('{"why": "no verdict"}'), "q", "r", "a")[0] is None
    assert judge_answer(judge("[1, 2]"), "q", "r", "a")[0] is None


@pytest.mark.xfail(reason='judge_answer takes bool(v["correct"]): a judge replying {"correct": "false"} is graded '
                          "correct (any non-empty string is truthy)")
def test_judge_answer_string_verdict():
    assert judge_answer(judge('{"correct": "false", "why": "Wrong value."}'), "q", "r", "a")[0] is False


# --- sweep: questions and grade-answers ----------------------------------------------------------------


def test_sweep_questions_and_grade_answers(tmp_path, monkeypatch, capsys):
    from subagents_bench import cli, sweep

    path = interactive_spec(tmp_path, overrides={"interactive": {
        "wall_clock_sec": 3600, "interval_sec": 300, "max_hints": 3, "replies": True, "questions_per_interval": 2}})
    spec = sweep.load_spec(path)
    assert sweep.questions_needed(spec) == 6
    assert "no questions" in sweep.check_interactive(spec)[0]
    src = tmp_path / "upstream"
    for i in range(4):
        write(src / f"pkg/f{i}.py", 2000)
    monkeypatch.setattr(sweep, "fetch_source", lambda task_dir, dest: src)
    counter = iter(range(100))

    def model(system, messages, tools):
        if system.startswith("You grade answers"):
            ok = "Answer to #2" in messages[0].content
            return Reply(json.dumps({"correct": ok, "why": "w"}) if "#6" not in messages[0].content else "unsure")
        return Reply(json.dumps([{"question": f"Q{next(counter)}?", "answer": "A"} for _ in range(2)]))

    monkeypatch.setattr(sweep, "SailClient", lambda m: FakeFactory({m.model: model})(m))
    assert cli.main(["questions", str(path)]) == 0
    assert "[OK] task-1: 6 questions (6 new)" in capsys.readouterr().out
    assert len(load_questions(sweep.hints_dir(spec), "task-1")) == 6 and sweep.check_interactive(spec) == []
    assert cli.main(["questions", str(path), "--per-task", "8"]) == 0
    assert "8 questions (2 new)" in capsys.readouterr().out  # the six kept, one more file asked
    assert cli.main(["questions", str(path), "--per-task", "20"]) == 1
    assert "[FAIL] task-1: HintError: " in capsys.readouterr().out

    trial = spec.jobs_dir / "t__r1__solo" / "task-1__x"
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text("{}")
    inbox = [msg(1, "hint", 0, 0, "b0"), msg(2, "question", 0, 80, question="Q2?"),
             msg(4, "question", 1, 160, question="Q4?"), msg(6, "question", 2, 240, question="Q6?")]
    replies = [rep(100, 2, "Answer to #2"), rep(110, 2, "A later answer to #2"), rep(170, 1, "eta", eta=3),
               rep(250, 6, "Answer to #6")]
    for name, rows in (("inbox", inbox), ("replies", list(reversed(replies)))):
        (trial / "agent" / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    assert cli.main(["grade-answers", str(path)]) == 1  # #6's verdict is unreadable
    out = capsys.readouterr().out
    assert "1 of 3 questions answered correctly (2 answered)" in out
    graded = json.loads((trial / "agent" / "answers.json").read_text())
    assert graded["answers"] == {"2": {"correct": True, "why": "w"}, "6": {"correct": None, "why": "unsure"}}
    assert cli.main(["grade-answers", str(path)]) == 0 and "questions answered" not in capsys.readouterr().out  # kept
    (trial / "result.json").unlink()
    assert cli.main(["grade-answers", str(path), "--force"]) == 0  # an unfinished trial is skipped
