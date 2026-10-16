"""``subagents-bench grade-answers``, version 2: the three-way judge (:func:`hints.judge_answer`,
which still reads the old ``{"correct": bool}`` shape), the per-reply grading of an episode
(:func:`hints.grade_replies`: every reply to a question in time order up to the first correct one,
and apart, replies to other ids that cite it after it was delivered), and the sweep command
(``answers.json`` version 2, the judge at temperature 0, which trials it grades again, exit codes),
with the request analysis reading what it writes. All offline: the judges are fakes that read a
tag in the reply they grade."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakes import ORCH, FakeFactory, Reply, make_config
from subagents_bench import cli, sweep
from subagents_bench.analysis.load import Trial
from subagents_bench.analysis.requests import request_rows
from subagents_bench.hints import ANSWERS_VERSION, QUESTION_JUDGE, VERDICTS, cites, grade_replies, judge_answer
from subagents_bench.llm import BillingError, ProviderError
from test_interactive import interactive_spec
from test_requests import T0, msg, rep

# What the fake judge answers to a reply that carries the tag.
TAGS = {
    "[ok]": '{"verdict": "correct", "why": "matches"}',
    "[bad]": '{"verdict": "wrong", "why": "contradicts"}',
    "[ack]": '{"verdict": "none", "why": "a promise"}',
    "[old]": '{"correct": true, "why": "old shape"}',
    "[??]": "I cannot tell.",
    "[down]": ProviderError("provider down"),
    "[bill]": BillingError("no credit"),
}


def graded_text(messages) -> str:
    return messages[0].content.split("Reply to grade: ", 1)[1]


def fake_judge(system, messages, tools):
    assert system == QUESTION_JUDGE and tools == []
    text = graded_text(messages)
    out = next((v for k, v in TAGS.items() if k in text), None)
    assert out is not None, f"untagged reply judged: {text!r}"
    return out if isinstance(out, BaseException) else Reply(out)


def judge_client(script=fake_judge):
    return FakeFactory({ORCH: script})(make_config().orchestrator)


def judge(text: str):
    return judge_client([Reply(text)])


# --- the judge ---------------------------------------------------------------------------------------


def test_the_judge_prompt_asks_for_three_verdicts():
    assert VERDICTS == ("correct", "wrong", "none") and ANSWERS_VERSION == 2
    assert '{"verdict": "correct" or "wrong" or "none"' in QUESTION_JUDGE
    for words in ("not an answer", "acknowledgement", "a promise or a plan", "only how to find out",
                  "grade only what it says about this question"):
        assert words in QUESTION_JUDGE


def test_judge_answer_prompt():
    client = judge('{"verdict": "none", "why": "Only a plan."}')
    assert judge_answer(client, "What does f return?", "1", "Let me check.") == ("none", "Only a plan.")
    (req,) = client.requests
    assert req["system"] == QUESTION_JUDGE and req["tools"] == []
    assert req["messages"][0].content == ("Question: What does f return?\n\nReference answer: 1\n\n"
                                          "Reply to grade: Let me check.")


@pytest.mark.parametrize("text, want", [
    ('{"verdict": "correct", "why": "w"}', ("correct", "w")),
    ('{"verdict": "wrong", "why": "w"}', ("wrong", "w")),
    ('{"verdict": "none", "why": "w"}', ("none", "w")),
    ('{"verdict": " Correct ", "why": "w"}', ("correct", "w")),
    ('{"verdict": "NONE"}', ("none", "")),
    ('Verdict:\n```json\n{"verdict": "wrong", "why": "w"}\n```', ("wrong", "w")),
    # a readable verdict wins over the old key; an unreadable one falls back to it
    ('{"verdict": "wrong", "correct": true, "why": "w"}', ("wrong", "w")),
    ('{"verdict": "partly", "correct": false, "why": "w"}', ("wrong", "w")),
    # the old shape
    ('{"correct": true, "why": "w"}', ("correct", "w")),
    ('{"correct": "false", "why": "w"}', ("wrong", "w")),
    # unreadable: no verdict word, not a string, not an object, no JSON
    ('{"verdict": "partly correct", "why": "w"}', None),
    ('{"verdict": "not an answer", "why": "w"}', None),
    ('{"verdict": null, "why": "w"}', None),
    ('{"verdict": true, "why": "w"}', None),
    ('{"verdict": ["correct"], "why": "w"}', None),
    ('{"correct": "yes", "why": "w"}', None),
    ('{"why": "no verdict"}', None),
    ('{"verdict": "partly correct", "why": "it quotes \\"verdict\\": \\"correct\\""}', None),
    ("correct", None),
    ('The "verdict": correct', None),
    # near-JSON whose verdict is plain to see: read by the fallback
    ('[{"verdict": "correct"}]', ("correct", '[{"verdict": "correct"}]')),
    ('{"verdict": "correct", "why": "ok"}\nNote: the reference mentions {x}.', ("correct", "ok")),
    ('{"verdict": "wrong", "why": "a"}\n{"verdict": "wrong", "why": "b"}', ("wrong", "a")),
    ('Verdict [correct]\n{"verdict": "correct", "why": "ok"}', ("correct", "ok")),
    ('{"verdict": "wrong", "why": "It says "Abort" but the reference says Retry"}',
     ("wrong", 'It says "Abort" but the reference says Retry')),
    ('{"verdict": "none", "why": "a plan,\nnot an answer"}', ("none", "a plan,\nnot an answer")),
    ('{"verdict": " Wrong ", "why": "w"} trailing }', ("wrong", "w")),
])
def test_judge_answer_verdicts(text, want):
    assert judge_answer(judge(text), "q", "r", "a") == (want if want is not None else (None, text))


def test_an_unreadable_verdict_keeps_the_start_of_the_reply():
    text = "x" * 1000
    assert judge_answer(judge(text), "q", "r", "a") == (None, "x" * 300)


@pytest.mark.parametrize("text, mid, want", [
    ("#3", 3, True), ("re #3: it returns None", 3, True), ("(#3)", 3, True), ("#3.", 3, True),
    ("**#3**", 3, True), ("#2, #3 and #4", 3, True), ("#2/#3", 3, True), ("line one\n#3\n", 3, True),
    ("#31", 3, False), ("#13", 3, False), ("##3", 3, False), ("a#3", 3, False), ("#3a", 3, False),
    ("#3_x", 3, False), ("# 3", 3, False), ("3", 3, False), ("no id", 3, False), ("#12", 12, True),
])
def test_cites(text, mid, want):
    assert cites(text, mid) is want


# --- grading one episode -------------------------------------------------------------------------------


def q(mid, t, delivered=True):
    return msg(mid, "question", 0, t, question=f"Q{mid}?", delivered=delivered)


def test_grade_replies_judges_each_reply_up_to_the_first_correct():
    """#2: a promise, a wrong answer, an unreadable verdict (does not stop), then correct; the reply
    after it is never judged. #3: correct at once. #4: the old judge shape. #5: never answered."""
    inbox = [msg(1, "hint", 0, 0, "b0"), q(2, 80), q(3, 160), q(4, 200), q(5, 220)]
    replies = [rep(90, 2, "on it [ack]"), rep(100, 2, "it returns 2 [bad]"), rep(110, 2, "hmm [??]"),
               rep(120, 2, "it returns 1 [ok]"), rep(130, 2, "and more [bad]"), rep(170, 3, "[ok]"),
               rep(210, 4, "[old]"), rep(30, 1, "fixing", eta=5)]
    client = judge_client()
    g = grade_replies(client, inbox, list(reversed(replies)))  # unsorted on disk: judged in time order
    assert set(g) == {"answers", "replies", "elsewhere"} and g["elsewhere"] == {}
    assert g["replies"]["2"] == [
        {"t": T0 + 90, "verdict": "none", "correct": False, "why": "a promise"},
        {"t": T0 + 100, "verdict": "wrong", "correct": False, "why": "contradicts"},
        {"t": T0 + 110, "verdict": None, "correct": None, "why": "I cannot tell."},
        {"t": T0 + 120, "verdict": "correct", "correct": True, "why": "matches"}]
    assert g["replies"]["3"] == [{"t": T0 + 170, "verdict": "correct", "correct": True, "why": "matches"}]
    assert g["replies"]["4"] == [{"t": T0 + 210, "verdict": "correct", "correct": True, "why": "old shape"}]
    # the first reply, as before version 2 (plus its verdict); unanswered questions have no key
    assert g["answers"] == {"2": {"correct": False, "verdict": "none", "why": "a promise"},
                            "3": {"correct": True, "verdict": "correct", "why": "matches"},
                            "4": {"correct": True, "verdict": "correct", "why": "old shape"}}
    assert [graded_text(r["messages"]) for r in client.requests] == [
        "on it [ack]", "it returns 2 [bad]", "hmm [??]", "it returns 1 [ok]", "[ok]", "[old]"]
    prompt = client.requests[0]["messages"][0].content
    assert prompt.startswith("Question: Q2?\n\nReference answer: ref\n\n")


def test_grade_replies_keeps_the_order_of_replies_sent_at_the_same_time():
    g = grade_replies(judge_client(), [q(2, 0)], [rep(10, 2, "first [bad]"), rep(10, 2, "second [ok]"),
                                                 rep(10, 2, "third [bad]")])
    assert [v["verdict"] for v in g["replies"]["2"]] == ["wrong", "correct"]


def test_grade_replies_judges_replies_elsewhere_apart():
    """Replies to other ids that cite a question, after it was delivered, are judged in time order up
    to the first correct one and reported apart; they never answer it."""
    inbox = [msg(1, "hint", 0, 0, "b0"), q(2, 80), q(3, 160), msg(4, "status", 0, 200), q(5, 300, delivered=False),
             q(12, 400)]
    replies = [
        rep(50, 1, "#2 is next [bad]"),  # before #2 was delivered (at 81): not about it
        rep(81, 1, "#2 delivered this second [bad]"),  # not after delivery
        rep(90, 1, "#2: it returns 2 [bad]"),  # a bug's reply citing #2
        rep(170, 3, "#3: X [ok]. And #2: it returns 1 [ok]"),  # #3's own reply also answers #2
        rep(180, 4, "re #2 again [bad]"),  # after the first correct one elsewhere: not judged
        rep(210, 1, "see #5 [ok]"),  # #5 was never delivered
        rep(420, 1, "#1 and #12 [ack]"),  # #12 cited; #1 is not a question
        rep(430, 1, "#120 [ok]"),  # not #12
        rep(440, 12, "about #12 [bad]"),  # its own id, not elsewhere
    ]
    g = grade_replies(judge_client(), inbox, replies)
    assert g["elsewhere"] == {
        "2": [{"t": T0 + 90, "to": 1, "verdict": "wrong", "correct": False, "why": "contradicts"},
              {"t": T0 + 170, "to": 3, "verdict": "correct", "correct": True, "why": "matches"}],
        "12": [{"t": T0 + 420, "to": 1, "verdict": "none", "correct": False, "why": "a promise"}]}
    # #2 had no reply to its own id: unanswered, whatever was said elsewhere
    assert g["answers"] == {"3": {"correct": True, "verdict": "correct", "why": "matches"},
                            "12": {"correct": False, "verdict": "wrong", "why": "contradicts"}}
    assert set(g["replies"]) == {"3", "12"}


def test_grade_replies_without_questions_or_replies():
    client = judge_client()
    assert grade_replies(client, [msg(1, "hint", 0, 0, "b0")], [rep(5, 1, "#1")]) == \
        {"answers": {}, "replies": {}, "elsewhere": {}}
    assert grade_replies(client, [q(1, 0)], []) == {"answers": {}, "replies": {}, "elsewhere": {}}
    assert client.requests == []


def test_grade_replies_lets_judge_errors_through():
    """A judge call that raises is not an unreadable verdict: the caller decides (grade-answers
    writes nothing for the trial)."""
    for tag, error in (("[down]", ProviderError), ("[bill]", BillingError)):
        with pytest.raises(error):
            grade_replies(judge_client(), [q(1, 0)], [rep(5, 1, "[bad]"), rep(6, 1, tag)][::-1])


# --- the sweep command ---------------------------------------------------------------------------------


@pytest.fixture
def graded_spec(tmp_path, monkeypatch):
    """A sweep with questions, and its judges: ``clients`` lists (model config, fake client) per
    SailClient made."""
    path = interactive_spec(tmp_path, overrides={"interactive": {
        "wall_clock_sec": 3600, "interval_sec": 300, "max_hints": 3, "replies": True, "questions_per_interval": 2}})
    spec = sweep.load_spec(path)
    clients = []

    def make(m):
        c = FakeFactory({m.model: fake_judge})(m)
        clients.append((m, c))
        return c

    monkeypatch.setattr(sweep, "SailClient", make)
    return path, spec, clients


def write_trial(spec, job, inbox, replies, name="task-1__x", result=True, answers=None) -> Path:
    d = spec.jobs_dir / job / name
    (d / "agent").mkdir(parents=True)
    if result:
        (d / "result.json").write_text("{}")
    for kind, rows in (("inbox", inbox), ("replies", replies)):
        if rows is not None:
            (d / "agent" / f"{kind}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    if answers is not None:
        (d / "agent" / "answers.json").write_text(answers if isinstance(answers, str) else json.dumps(answers))
    return d / "agent"


INBOX = [msg(1, "hint", 0, 0, "b0"), q(2, 80), q(3, 160), q(4, 240)]
REPLIES = [rep(90, 2, "on it [ack]"), rep(100, 2, "[ok]"), rep(170, 3, "[bad]"), rep(180, 1, "#3 is [ok]")]


def answers(agent: Path) -> dict:
    return json.loads((agent / "answers.json").read_text())


def test_grade_answers_writes_version_2_with_the_judge_at_temperature_0(graded_spec, capsys):
    path, spec, clients = graded_spec
    agent = write_trial(spec, "t__r1__solo", INBOX, REPLIES)
    assert cli.main(["grade-answers", str(path)]) == 0
    (m, client), = clients
    orch = next(iter(sweep.interactive(spec).values())).orchestrator
    assert orch.temperature != 0.0 and m.temperature == 0.0 and m.model == orch.model
    assert m.max_output_tokens == orch.max_output_tokens  # only the temperature changes
    a = answers(agent)
    assert list(a) == ["version", "judge_model", "judge_temperature", "graded_at", "answers", "replies", "elsewhere"]
    assert (a["version"], a["judge_model"], a["judge_temperature"]) == (2, orch.model, 0.0)
    assert isinstance(a["graded_at"], float)
    assert a["answers"] == {"2": {"correct": False, "verdict": "none", "why": "a promise"},
                            "3": {"correct": False, "verdict": "wrong", "why": "contradicts"}}
    assert [v["verdict"] for v in a["replies"]["2"]] == ["none", "correct"]
    assert a["elsewhere"] == {"3": [{"t": T0 + 180, "to": 1, "verdict": "correct", "correct": True, "why": "matches"}]}
    out = capsys.readouterr().out
    assert "[OK] jobs/t/t__r1__solo/task-1__x: 1 of 3 questions answered correctly (0 by the first reply; " \
           "2 answered; 4 replies judged, 1 of them to another id)" in out


def test_request_rows_read_a_version_2_file(graded_spec):
    """The request analysis (and the monitor, which uses it) reads ``answers``: the first reply's
    verdict; a first reply that is not an answer is not correct."""
    path, spec, _ = graded_spec
    agent = write_trial(spec, "t__r1__solo", INBOX, REPLIES)
    assert sweep.grade_answers(spec) == 0
    ep = {"config": {"interactive": {"replies": True}}, "session_t0": T0, "t_end": T0 + 600}
    t = Trial(agent.parent, "s", 1, "c", None, ep, {"inbox": INBOX, "replies": REPLIES})
    rows = {r["msg_id"]: r for r in request_rows(t, {}, True)}
    assert (rows[2]["correct"], rows[3]["correct"], rows[4]["correct"]) == (False, False, None)


def test_grade_answers_grades_again_what_is_not_version_2(graded_spec, capsys):
    path, spec, clients = graded_spec
    v1 = write_trial(spec, "t__r1__solo", INBOX, REPLIES, answers={"judge_model": "j", "graded_at": 1.0,
                                                                   "answers": {"2": {"correct": False, "why": "w"}}})
    v2 = write_trial(spec, "t__r2__solo", INBOX, REPLIES, answers={"version": 2, "answers": {}, "kept": True})
    cut = write_trial(spec, "t__r3__solo", INBOX, REPLIES, answers='{"version": 2, "answ')
    listed = write_trial(spec, "t__r4__solo", INBOX, REPLIES, answers="[2]")
    text = write_trial(spec, "t__r5__solo", INBOX, REPLIES, answers={"version": "2"})
    later = write_trial(spec, "t__r6__solo", INBOX, REPLIES, answers={"version": 3})  # a later format: kept
    assert cli.main(["grade-answers", str(path)]) == 0
    assert answers(v2) == {"version": 2, "answers": {}, "kept": True} and answers(later) == {"version": 3}
    for agent in (v1, cut, listed, text):
        assert answers(agent)["version"] == 2 and answers(agent)["replies"]["2"][-1]["verdict"] == "correct"
    assert capsys.readouterr().out.count("[OK]") == 4
    assert cli.main(["grade-answers", str(path)]) == 0 and "[OK]" not in capsys.readouterr().out  # all kept
    assert cli.main(["grade-answers", str(path), "--force"]) == 0
    assert capsys.readouterr().out.count("[OK]") == 6 and "kept" not in answers(v2) and answers(later)["version"] == 2
    assert sum(len(c.requests) for _, c in clients) == 10 * 4  # 4 judge calls per trial graded


def test_grade_answers_skips_what_it_cannot_grade(graded_spec, capsys):
    path, spec, clients = graded_spec
    skipped = [write_trial(spec, "_partial", INBOX, REPLIES, name="t__r1__solo.1"),  # under _partial
               write_trial(spec, ".hidden", INBOX, REPLIES),
               write_trial(spec, "t__r1__solo", INBOX, REPLIES, result=False),  # unfinished
               write_trial(spec, "t__r2__solo", [msg(1, "hint", 0, 0, "b0")], [rep(5, 1, "#1 [ok]")]),  # no questions
               write_trial(spec, "t__r3__solo", None, None)]  # no records at all
    no_replies = write_trial(spec, "t__r4__solo", INBOX, None)
    assert cli.main(["grade-answers", str(path)]) == 0
    assert not any((a / "answers.json").exists() for a in skipped)
    a = answers(no_replies)
    assert (a["answers"], a["replies"], a["elsewhere"]) == ({}, {}, {})
    assert "0 of 3 questions answered correctly (0 by the first reply; 0 answered; 0 replies judged)" in \
        capsys.readouterr().out
    assert sum(len(c.requests) for _, c in clients) == 0


def test_grade_answers_exits_1_on_an_unreadable_verdict(graded_spec, capsys):
    path, spec, _ = graded_spec
    agent = write_trial(spec, "t__r1__solo", INBOX, [rep(90, 2, "[??]"), rep(95, 2, "[ok]")])
    assert cli.main(["grade-answers", str(path)]) == 1
    assert "1 of 3 questions answered correctly (0 by the first reply; 1 answered; 2 replies judged; 1 verdicts " \
           "unreadable)" in capsys.readouterr().out
    assert answers(agent)["answers"] == {"2": {"correct": None, "verdict": None, "why": "I cannot tell."}}
    # an unread verdict is judged again on the next run (still unreadable here: exit 1 again)
    assert cli.main(["grade-answers", str(path)]) == 1
    assert "1 verdicts unreadable" in capsys.readouterr().out


def test_grade_answers_judges_an_unread_verdict_again(graded_spec, capsys, monkeypatch):
    """A version-2 file with an unread verdict is graded again: once the judge reads it, the
    verdict is filled in and the run exits 0; a file with every verdict read is kept."""
    path, spec, clients = graded_spec
    agent = write_trial(spec, "t__r1__solo", INBOX, [rep(90, 2, "[??] it returns 1"), rep(95, 2, "[bad]"),
                                                     rep(180, 1, "#3 is [??] x")])
    kept = write_trial(spec, "t__r2__solo", INBOX, REPLIES)
    assert cli.main(["grade-answers", str(path)]) == 1
    a = answers(agent)
    assert [v["verdict"] for v in a["replies"]["2"]] == [None, "wrong"]
    assert [v["verdict"] for v in a["elsewhere"]["3"]] == [None]
    kept_doc = answers(kept)
    capsys.readouterr()
    monkeypatch.setitem(TAGS, "[??]", '{"verdict": "correct", "why": "read now"}')  # the judge reads it now
    assert cli.main(["grade-answers", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.count("[OK]") == 1 and "t__r1__solo" in out and "unreadable" not in out
    a = answers(agent)
    assert a["version"] == 2 and a["answers"]["2"] == {"correct": True, "verdict": "correct", "why": "read now"}
    assert [v["verdict"] for v in a["replies"]["2"]] == ["correct"]  # up to the first correct one
    assert [v["verdict"] for v in a["elsewhere"]["3"]] == ["correct"]
    assert answers(kept) == kept_doc  # every verdict read: not judged again
    n = sum(len(c.requests) for _, c in clients)
    assert cli.main(["grade-answers", str(path)]) == 0 and "[OK]" not in capsys.readouterr().out
    assert sum(len(c.requests) for _, c in clients) == n


def test_grade_answers_keeps_a_later_format_with_an_unread_verdict(graded_spec):
    """Only a version-2 file is checked for unread verdicts: a later format is never downgraded;
    a version-2 file whose replies are not verdict objects is graded again."""
    path, spec, _ = graded_spec
    later = {"version": 3, "replies": {"2": [{"t": T0 + 90, "verdict": None}]}}
    agent = write_trial(spec, "t__r1__solo", INBOX, REPLIES, answers=later)
    broken = write_trial(spec, "t__r2__solo", INBOX, REPLIES, answers={"version": 2, "replies": {"2": ["x"]}})
    assert cli.main(["grade-answers", str(path)]) == 0
    assert answers(agent) == later
    assert answers(broken)["version"] == 2 and answers(broken)["replies"]["2"][-1]["verdict"] == "correct"


def test_grade_answers_a_failed_trial_writes_nothing_and_is_graded_next_time(graded_spec, capsys, monkeypatch):
    path, spec, _ = graded_spec
    failing =write_trial(spec, "t__r1__solo", INBOX, [rep(90, 2, "[bad]"), rep(95, 2, "[down]")])
    broken = write_trial(spec, "t__r2__solo", INBOX, REPLIES)
    (broken / "replies.jsonl").write_text('{"t": 1, "to"\n')  # cut mid-write
    fine = write_trial(spec, "t__r3__solo", INBOX, REPLIES)
    assert cli.main(["grade-answers", str(path)]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] jobs/t/t__r1__solo/task-1__x: ProviderError: provider down" in out
    assert "[FAIL] jobs/t/t__r2__solo/task-1__x: JSONDecodeError: " in out
    assert not (failing / "answers.json").exists() and not (broken / "answers.json").exists()
    assert answers(fine)["version"] == 2
    monkeypatch.setitem(TAGS, "[down]", '{"verdict": "correct", "why": "back"}')  # the provider is back
    assert cli.main(["grade-answers", str(path)]) == 1  # the broken records still fail
    assert [v["verdict"] for v in answers(failing)["replies"]["2"]] == ["wrong", "correct"]
    assert capsys.readouterr().out.count("[OK]") == 1


def test_grade_answers_stops_on_a_billing_error(graded_spec, capsys):
    path, spec, clients = graded_spec
    first = write_trial(spec, "t__r1__solo", INBOX, [rep(90, 2, "[bill]")])
    later = write_trial(spec, "t__r2__solo", INBOX, REPLIES)
    assert cli.main(["grade-answers", str(path)]) == 3
    assert "[FAIL] jobs/t/t__r1__solo/task-1__x: BillingError: no credit" in capsys.readouterr().out
    assert not (first / "answers.json").exists() and not (later / "answers.json").exists()
    assert len(clients[0][1].requests) == 1


def test_grade_answers_without_interactive_conditions(tmp_path, capsys):
    from test_sweep import make_spec
    path = make_spec(tmp_path)
    assert cli.main(["grade-answers", str(path)]) == 0
    assert "has no interactive condition: nothing to grade" in capsys.readouterr().out
