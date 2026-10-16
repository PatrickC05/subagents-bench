"""ETA scoring against replayed fix times and done claims (``analysis/estimates.py``): the text
fallback on replies from the first requests run, and estimates, status-check pairs and the
per-condition table on a synthetic trial."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from subagents_bench.analysis.estimates import (estimate_rows, estimate_table, parse_done, parse_estimates,
                                                status_pair_rows)
from subagents_bench.analysis.fixes import bug_fix_rows, fix_times
from test_fixes import END, bug, make_trial, rep, status, timeline

# -- the text fallback (replies of the first requests run) --------------------------------------


@pytest.mark.parametrize("text,ids,want", [
    ("Current ETAs: urgent #11 (registry.py / test_issue1066) ~8 min; #1 (pint_eval.py) ~12 min; #4 (numpy "
     "setitem masked) ~18 min; #8 (quantity.to_reduced_units) ~18 min. I'll update if estimates change.",
     {1, 4, 8, 11}, {11: 8, 1: 12, 4: 18, 8: 18}),
    ("Status update: bug #1 (required flag) — fixed and verified (full suite green, 488 passed / 0 failed). Bug #11 "
     "(urgent, @argument cls preserved) — fixed and verified. Bug #15 (symlink resolution) — fixed and verified. "
     "Bug #8 (envvar flag value) and bug #18 (flag default_map) — still being investigated; ETA ~10 min each.",
     {1, 4, 8, 11, 15, 18}, {8: 10, 18: 10}),
    ("Status: #1, #11, #4, #15 are fixed and verified (full suite green: 488 passed, 21 skipped, 1 xfailed, 0 failed). "
     "#8 (envvar flag value) and #18 (flag default_map) are still open — a subagent is now reproducing and fixing "
     "both in src/click/core.py; ETA ~10 min.", {1, 4, 8, 11, 15, 18}, {8: 10, 18: 10}),
    ("Final status: #18 (measurement.py default format) is FIXED and verified — the full pre-existing suite still "
     "passes (1257 passed, 18 skipped, 11 xfailed). #1, #4, #8, #11, #15 are still open: the git object store "
     "contained no pre-bug copies; a fix pass is running now (ETA ~10 min each, #11 first).",
     {1, 4, 8, 11, 15, 18}, {1: 10, 4: 10, 8: 10, 11: 10, 15: 10}),
    ("Progress: bug #18 (measurement.py default format) is fixed and verified (full suite still 1257 passed). Bug #1 "
     "(pint_eval.py): differential fuzzing found no mismatch, so I'm still hunting it. #4/#8/#11/#15: those are "
     "being pinned down by in-repo contract analysis. ETAs: #1 ~6 min, #4 ~8 min, #8 ~8 min, #11 ~6 min, #15 ~8 min.",
     {1, 4, 8, 11, 15, 18}, {1: 6, 4: 8, 8: 8, 11: 6, 15: 8}),
    ("Status: bug #1 (required flag) — fix applied and being verified now, ~3 min. Bug #4 (echo with no streams) — "
     "fix in progress, ~10 min.", {1, 4}, {1: 3, 4: 10}),
    ("Bug #11 (urgent, masked setitem): DONE — fixed and verified (full suite green). Bug #4 (numpy_func "
     "cross/dot): ETA ~5 min. Bug #8 (registry canonical-name parsing): ETA ~6 min.", {4, 8, 11}, {4: 5, 8: 6}),
    ("Status update: #1 (pint_eval.py) is fixed and verified — done. #4, #8, #11 are still open; I'm out of time, "
     "so I cannot complete them now. ETA unknown for those.", {1, 4, 8, 11}, {}),
    ("- #8 quantity.to_reduced_units: I fuzzed over ~140 unit pairs (m/cm/km/mm/s/ms) and 20k random expressions; "
     "1257 pre-existing tests still pass.", {8}, {}),
])
def test_parse_estimates_on_status_replies(text, ids, want):
    assert parse_estimates(text, ids) == want


@pytest.mark.parametrize("text,ids,want", [
    ("ETA ~10 min for #8 and #18.", {8, 18}, {8: 10, 18: 10}),  # the time first
    ("#8 and #18 are still open. ETA ~10 min.", {8, 18}, {8: 10, 18: 10}),  # carried to the next sentence
    ("#8 is open. A subagent is on it. ETA ~10 min.", {8}, {}),  # one sentence only
    ("#1 ~5 min; #4 ~8 min; #6 ~2 min", {1, 4}, {1: 5, 4: 8}),  # #6 is not a bug id
    ("#1 is fixed, #4 ~8 min", {1, 4}, {4: 8}),  # a done word drops the ids before it
    ("#1 will be fixed in about 5 minutes", {1}, {1: 5}),  # ... unless it is a future
    ("#1: ETA 1-2 hours; #2: 90 sec; #4 ~1.5h", {1, 2, 4}, {1: 120, 2: 1.5, 4: 90}),  # a range: its end
    ("#1 took 3 min to reproduce", {1}, {}),  # a time without an estimate cue
    ("#1 should be done in 4 min", {1}, {1: 4}),
    ("#1 ran in 4 min", {1}, {}),
    # with more times in a sentence, a time followed by "for #" is for the ids after it
    ("ETA ~10 min for #8, ~5 min for #2.", {2, 8}, {8: 10, 2: 5}),
    ("~10 min for #8; ~5 min for #2.", {2, 8}, {8: 10, 2: 5}),
    ("~10 min for bug #8 and #18, ~5 min on #2", {2, 8, 18}, {8: 10, 18: 10, 2: 5}),
    ("#4 ~3 min, then ~10 min for #8 and #18", {4, 8, 18}, {4: 3, 8: 10, 18: 10}),
])
def test_parse_estimates_rules(text, ids, want):
    assert parse_estimates(text, ids) == pytest.approx(want)


@pytest.mark.parametrize("text,ids,want", [
    ("Status: #1, #11, #4, #15 are fixed and verified (full suite green). #8 and #18 are still open.",
     {1, 4, 8, 11, 15, 18}, {1, 4, 11, 15}),
    ("Bug #11 (urgent, masked setitem): DONE — fixed and verified. Bug #4: ETA ~5 min.", {4, 11}, {11}),
    ("Bug #11 already fixed.", {11}, {11}),
    ("#1 is fixed and verified. #2: ETA ~5 min.", {1, 2}, {1}),
    ("#1 is fixed, #4 ~8 min", {1, 4}, {1}),
    ("#4 will be fixed in ~5 min", {4}, set()),  # a future
    ("#2 is not yet fixed", {2}, set()),  # negated
    ("#4 ~5 min, then it is done", {4}, set()),  # the time took #4
    ("#6 is fixed", {1}, set()),  # not a bug id
    ("", {1}, set()),
])
def test_parse_done(text, ids, want):
    assert parse_done(text, ids) == want


@pytest.mark.parametrize("text,want", [
    ("Got it — request #8 (pint/quantity.py, test_to_reduced_units) noted. ETA ~20 min. Still working on #1 "
     "(pint_eval) and #4 (numpy setitem).", 20),
    ("Got it — bug #8 (numpy setitem with masked arrays). ETA ~15 min.", 15),
    ("Got it — bug #4 is like this one. ETA ~15 min.", 15),  # a reply's time is for its own bug
    ("Got it, like #4 and #8. ETA ~15 min.", 15),
    ("Looking into it now (numpy_func cross/dot handling); ETA ~10 min.", 10),
    ("Still investigating: the envvar/flag-value path. This one will be in within ~6 minutes.", 6),
    ("ETA ~20 min; #1 and #4 come first.", 20),  # a reply's own bug keeps a time before other ids
    ("#1 ~5 min first, then this one ~20 min.", 20),
    ("Fixed: the parser no longer drops the unit. Full suite green (1257 passed, 18 skipped).", None),
    # a time left over for the reply's bug: the one after an ETA word, not an earlier approximate one
    ("Test suite takes ~2 min; ETA 10 min.", 10),
    ("Running the full suite (~2 min), then ETA ~8 min.", 8),
    ("Reproducing takes ~2 min. ETA ~12 min.", 12),
    ("Test suite takes ~2 min.", 2),  # no ETA word: the first time, as before
    ("~10 min for #8, ~5 min for #4.", 10),  # its own bug among the ids: "for #" works
])
def test_parse_estimates_of_a_bug_reply(text, want):
    assert parse_estimates(text, {1, 4, 8}, default=8).get(8) == want


# -- estimates and status pairs ------------------------------------------------------------------


def session(root, *, valid_states=True, job="lv__r1__single_eta", name="swe-sweep__click__abc"):
    """Bugs #1 b0 (0 s), #2 b1 (240 s), #4 b2 (480 s), #6 b3 (720 s); status checks #3 (360 s) and #7
    (840 s). Fixed: b0 at 100 s, b1 at 820 s, b2 at 1000 s; b3 never."""
    inbox = [bug(1, 0, 0, "b0"), bug(2, 1, 240, "b1"), status(3, 0, 360), bug(4, 2, 480, "b2"), bug(6, 3, 720, "b3"),
             status(7, 1, 840)]
    replies = [rep(10, 1, "On it.", eta=5), rep(250, 2, "Got it. ETA ~10 min."),
               rep(370, 3, "#1 ~2 min; #2 ~8 min"), rep(490, 4, "Working on it", eta=30),
               rep(500, 2, "Still on it", eta=4), rep(600, 1, "Fixed.", done=True), rep(730, 6, "On it", eta=10),
               rep(900, 6, "Fixed.", done=True), rep(905, 5, "an answer", eta=3)]
    sts = timeline([(0, dict.fromkeys(["b0", "b1", "b2", "b3"], "fail")),
                    (100, {"b0": "pass", "b1": "fail", "b2": "fail", "b3": "fail"}),
                    (820, {"b0": "pass", "b1": "pass", "b2": "fail", "b3": "fail"}),
                    (1000, {"b0": "pass", "b1": "pass", "b2": "pass", "b3": "fail"})])
    return make_trial(root, inbox, replies, states_=sts, matches=valid_states, job=job, name=name,
                      verdicts={"b0": True, "b1": True, "b2": True, "b3": False})


def by_key(rows):
    return {(round(r["est_session_sec"]), r["msg_id"]): r for r in rows}


def test_estimate_rows(tmp_path):
    t = session(tmp_path)
    rows = estimate_rows(t, fix_times(t))
    assert estimate_rows(t, bug_fix_rows(t)) == rows == estimate_rows(t)  # the fixes as rows, or computed
    r = by_key(rows)
    assert list(r) == [(10, 1), (250, 2), (370, 1), (370, 2), (490, 4), (500, 2), (730, 6)]  # not #5's reply
    assert [x["via"] for x in rows] == ["structured", "text_bug", "text_status", "text_status", "structured",
                                        "structured", "structured"]
    assert [x["source"] for x in rows] == ["arrival", "arrival", "check", "check", "arrival", "update", "arrival"]
    assert r[370, 2]["check_id"] == 3 and r[500, 2]["check_id"] is None and r[370, 2]["reply_to"] == 3
    # b0: 5 min at 10 s, fixed at 100 s (by 310 s + 75 s grace); its check estimate came after the fix
    b0 = r[10, 1]
    assert (b0["scored"], b0["on_time"], b0["on_time_strict"], b0["weight"]) == (True, True, True, 1.0)
    assert b0["err_min"] == pytest.approx(-3.5) and b0["log2_err"] == pytest.approx(math.log2(90 / 300))
    assert (b0["pred_session_sec"], b0["grace_sec"], b0["fixed_sec"]) == (310, 75, 100)
    assert (r[370, 1]["stale"], r[370, 1]["scored"], r[370, 1]["on_time"], r[370, 1]["weight"]) == (
        True, False, None, None)
    # b1 (sent at 240 s, fixed at 820 s): three estimates share its weight
    a, c, u = r[250, 2], r[370, 2], r[500, 2]
    assert (a["eta_min"], a["est_sec"], a["on_time"], a["on_time_strict"]) == (10, 10, True, True)
    assert a["log2_err"] == pytest.approx(math.log2(570 / 600)) and a["err_min"] == pytest.approx(-0.5)
    assert (c["eta_min"], c["on_time"], c["grace_sec"]) == (8, True, 120)
    # 4 min at 500 s: due 740 s, fixed 80 s late: late by the 60 s grace, within 2 and 5 minutes
    assert (u["on_time"], u["on_time_strict"], u["within_1m"], u["within_2m"], u["within_5m"]) == (
        False, False, False, True, True)
    assert u["err_min"] == pytest.approx(80 / 60) and u["fixed_sec"] == 580
    assert [a["weight"], c["weight"], u["weight"]] == pytest.approx([1 / 3] * 3)
    # b2: 30 min at 490 s cannot come due within the session: not eligible, reported as fixed by the deadline
    b2 = r[490, 4]
    assert (b2["eligible"], b2["scored"], b2["stale"], b2["fixed_by_deadline"], b2["on_time"]) == (
        False, False, False, True, None)
    assert (b2["claim_scored"], b2["claim_on_time"]) == (False, None)
    # b3: 10 min at 730 s, never fixed: late, the error a lower bound to the episode's end
    b3 = r[730, 6]
    assert (b3["scored"], b3["on_time"], b3["err_lower_bound"], b3["log2_err"]) == (True, False, True, math.inf)
    assert b3["err_min"] == pytest.approx((END - 1330) / 60)
    # against the done claims: b0 said done at 600 s (late for both estimates sent before it); b3 at 900 s
    assert (b0["claim_sec"], b0["claim_scored"], b0["claim_on_time"]) == (600, True, False)
    assert b0["claim_err_min"] == pytest.approx((600 - 310) / 60) and b0["claim_weight"] == 0.5
    assert (r[370, 1]["claim_scored"], r[370, 1]["claim_on_time"]) == (True, False)
    assert (b3["claim_on_time"], b3["claim_log2_err"]) == (True, pytest.approx(math.log2(170 / 600)))
    assert (a["claim_scored"], a["claim_on_time"], a["claim_log2_err"], a["claim_err_lower_bound"]) == (
        True, False, math.inf, True)  # b1 was never said done
    assert a["text"] == "Got it. ETA ~10 min." and b0["text"] is None


def test_estimates_after_a_claim_and_without_fix_times(tmp_path):
    t = session(tmp_path, valid_states=False)
    rows = by_key(estimate_rows(t))
    assert all(not x["timing_ok"] and not x["scored"] and x["stale"] is None and x["on_time"] is None
               for x in rows.values())
    assert rows[10, 1]["claim_on_time"] is False  # claims need no replay
    # an estimate after the done claim is kept (stale if after the fix) but not scored against the claim
    inbox = [bug(1, 0, 0, "b0")]
    late = make_trial(tmp_path / "x", inbox, [rep(10, 1, done=True), rep(20, 1, eta=2)],
                      states_=timeline([(0, {"b0": "fail"}), (5, {"b0": "pass"})]))
    (row,) = estimate_rows(late)
    assert (row["claim_sec"], row["claim_scored"], row["claim_on_time"], row["stale"], row["scored"]) == (
        10, False, None, True, False)
    # the session ends early (idle): the deadline still decides eligibility; the lower bound runs to it
    idle = make_trial(tmp_path / "z", inbox, [rep(10, 1, eta=2)], states_=timeline([(0, {"b0": "fail"})], end=300),
                      t_end=300)
    (row,) = estimate_rows(idle)
    assert (row["eligible"], row["on_time"], row["err_min"]) == (True, False, pytest.approx((1800 - 130) / 60))
    assert estimate_rows(make_trial(tmp_path / "y", [status(1, 0, 0)], [rep(5, 1, "#1 ~2 min")])) == []


def test_status_pairs(tmp_path):
    t = session(tmp_path)
    pairs = {(p["check_id"], p["msg_id"]): p for p in status_pair_rows(t, fix_times(t))}
    # check #3 lists #1 and #2; check #7 lists #2, #4 and #6 (#1 was said done at 600 s)
    assert list(pairs) == [(3, 1), (3, 2), (7, 2), (7, 4), (7, 6)]
    p = pairs[3, 1]  # an estimate 10 s after the check, on a bug fixed at 100 s
    assert (p["answered"], p["reply_sec"], p["kind"], p["via"], p["right"]) == (True, 10, "eta", "text_status", False)
    assert (p["fixed_at_check"], p["claimed"], p["done_owed"], p["check_session_sec"]) == (True, True, False, 360)
    assert (pairs[3, 2]["right"], pairs[3, 2]["fixed_at_check"]) == (True, False)
    p = pairs[7, 2]  # fixed at 820 s, never said done, no reply after the check
    assert (p["answered"], p["reply_sec"], p["kind"], p["right"], p["fixed_at_check"], p["done_owed"]) == (
        False, None, None, None, True, True)
    p = pairs[7, 6]  # a done claim 60 s after the check on a bug never fixed
    assert (p["answered"], p["kind"], p["via"], p["right"], p["done_owed"]) == (
        True, "done", "structured", False, False)
    assert pairs[7, 4]["done_owed"] is False  # fixed after the check
    bare = session(tmp_path / "x", valid_states=False)
    assert {p["right"] for p in status_pair_rows(bare)} == {None}
    assert {p["done_owed"] for p in status_pair_rows(bare)} == {None}
    assert status_pair_rows(make_trial(tmp_path / "y", [bug(1, 0, 0, "b0")], [rep(5, 1, eta=2)])) == []


def test_l0_arrival_ends_where_l1s_next_check_would_be(tmp_path):
    """Without status checks (L0), a bug's arrival window ends when L1's next check after it would
    be sent (half an interval after bugs 2, 4, 6, ...: 360, 840, 1320 s at 240 s), so L0 and L1
    arrival estimates share one window; a check that would fall at or after the deadline is none."""
    inbox = [bug(k + 1, k, 240 * k, f"b{k}") for k in range(6)]  # #1..#6 at 0, 240, ..., 1200 s
    replies = [rep(300, 1, eta=5), rep(400, 2, eta=5), rep(830, 3, eta=5), rep(1084, 4, "ETA ~5 min."),
               rep(1100, 4, eta=3), rep(1500, 6, eta=2)]
    t = make_trial(tmp_path / "a", inbox, replies, job="lv__r1__single_baseline")
    got = [(r["msg_id"], round(r["est_session_sec"]), r["source"]) for r in estimate_rows(t)]
    assert got == [(1, 300, "arrival"), (2, 400, "update"), (3, 830, "arrival"), (4, 1084, "update"),
                   (4, 1100, "update"), (6, 1500, "update")]
    # a 1300 s session: L1 would send no check after bug 5 (1320 s is past the deadline)
    short = make_trial(tmp_path / "b", inbox, [rep(1290, 5, eta=1)], job="lv__r1__single_baseline", wall=1300)
    assert [r["source"] for r in estimate_rows(short)] == ["arrival"]
    # five bugs: L1 has no check after the last one (only after bugs 2 and 4), so its window is open
    five = make_trial(tmp_path / "c", inbox[:5], [rep(1700, 5, eta=1)], job="lv__r1__single_baseline")
    assert [r["source"] for r in estimate_rows(five)] == ["arrival"]
    # with checks the window ends at the session's own next check, as before
    l1 = make_trial(tmp_path / "d", inbox[:2] + [status(7, 0, 360)], [rep(300, 1, eta=5), rep(400, 2, eta=5)])
    assert [r["source"] for r in estimate_rows(l1)] == ["arrival", "check"]


def test_a_done_written_in_a_status_check_reply_counts(tmp_path):
    """A reply to a status check's id that says a bug is done answers that bug's pair (via
    text_status) and counts as said done, from the earlier of it and the bug's done: true reply."""
    inbox = [bug(1, 0, 0, "b0"), bug(2, 1, 240, "b1"), status(3, 0, 360), bug(4, 2, 480, "b2"), status(5, 1, 840)]
    replies = [rep(380, 3, "#1 is fixed and verified. #2: ETA ~5 min."), rep(900, 1, "Fixed.", done=True)]
    sts = timeline([(0, dict.fromkeys(["b0", "b1", "b2"], "fail")), (100, {"b0": "pass", "b1": "fail", "b2": "fail"}),
                    (700, {"b0": "pass", "b1": "pass", "b2": "fail"})])
    t = make_trial(tmp_path, inbox, replies, states_=sts, verdicts={"b0": True, "b1": True, "b2": False})
    pairs = {(p["check_id"], p["msg_id"]): p for p in status_pair_rows(t)}
    # #1 was said done at 380 s (in text), before check #5, although its done: true came at 900 s
    assert list(pairs) == [(3, 1), (3, 2), (5, 2), (5, 4)]
    p = pairs[3, 1]
    assert (p["answered"], p["reply_sec"], p["kind"], p["via"], p["right"]) == (True, 20, "done", "text_status", True)
    assert (p["claimed"], p["done_owed"]) == (True, False)
    p = pairs[3, 2]
    assert (p["answered"], p["kind"], p["via"], p["claimed"]) == (True, "eta", "text_status", False)
    assert (pairs[5, 2]["answered"], pairs[5, 2]["done_owed"]) == (False, True)  # fixed at 700 s, never said done
    # the estimates and their claims are unchanged: the claim there is the done: true reply
    (e,) = [r for r in estimate_rows(t) if r["msg_id"] == 2]
    assert (e["via"], e["eta_min"], e["source"]) == ("text_status", 5, "check")
    assert parse_done("#1 is fixed and verified.", {1}) == {1}


def test_estimate_table(tmp_path):
    t = session(tmp_path)
    rows, pairs = pd.DataFrame(estimate_rows(t)), pd.DataFrame(status_pair_rows(t))
    (row,) = estimate_table(rows, pairs).to_dict("records")
    assert (row["sweep"], row["condition"], row["n_episodes_estimated"], row["n_estimates"]) == (
        "lv", "single_eta", 1, 7)
    assert (row["n_structured"], row["n_text_bug"], row["n_text_status"]) == (4, 1, 2)
    assert (row["n_bugs_estimated"], row["n_timed"], row["n_scored"], row["n_scored_bugs"]) == (4, 7, 5, 3)
    # each bug once: b0 on time, b1 2 of 3, b3 late (unweighted it would be 3 of 5)
    assert row["on_time_frac"] == pytest.approx((1 + 2 / 3 + 0) / 3)
    assert row["on_time_strict_frac"] == pytest.approx((1 + 2 / 3) / 3)
    assert (row["within_1m_frac"], row["within_2m_frac"]) == (pytest.approx((1 + 2 / 3) / 3), pytest.approx(2 / 3))
    assert (row["n_scored_arrival"], row["on_time_frac_arrival"]) == (3, pytest.approx(2 / 3))
    assert (row["on_time_frac_check"], row["on_time_frac_update"], row["n_scored_update"]) == (1.0, 0.0, 1)
    assert (row["on_time_frac_structured"], row["on_time_frac_text_bug"], row["on_time_frac_text_status"]) == (
        pytest.approx(1 / 3), 1.0, 1.0)
    fixed = sorted(abs(math.log2(x)) for x in (90 / 300, 570 / 600, 450 / 480, 320 / 240))
    assert row["abs_log2_median_fixed"] == pytest.approx((fixed[1] + fixed[2]) / 2)
    assert row["abs_log2_median"] == pytest.approx(fixed[2])  # b3's +inf counts
    assert row["stale_frac"] == pytest.approx(1 / 7)
    assert (row["n_ineligible"], row["ineligible_fixed_frac"]) == (1, 1.0)
    assert row["n_claim_scored"] == 6 and row["claim_on_time_frac"] == pytest.approx(1 / 3)  # b3 only
    assert (row["n_pairs"], row["coverage_120s"], row["right_frac"]) == (5, 0.6, pytest.approx(1 / 3))
    assert (row["done_owed_frac"], row["n_done_owed"]) == (0.5, 1)


def test_estimate_table_groups_by_sweep_and_drops_invalid_trials(tmp_path):
    a = session(tmp_path / "a")
    b = session(tmp_path / "b", job="other__r1__single_eta")
    c = session(tmp_path / "c", name="swe-sweep__click__def")
    rows = pd.DataFrame([{**r, "valid": True} for x in (a, b) for r in estimate_rows(x)]
                        + [{**r, "valid": False} for r in estimate_rows(c)])
    tab = estimate_table(rows)
    assert list(zip(tab["sweep"], tab["condition"])) == [("lv", "single_eta"), ("other", "single_eta")]
    assert list(tab["n_estimates"]) == [7, 7] and list(tab["n_pairs"]) == [0, 0]
    assert math.isnan(tab["coverage_120s"].iloc[0])
    assert estimate_table(pd.DataFrame()).empty and estimate_table(pd.DataFrame(), pd.DataFrame()).empty
    pairs_only = estimate_table(pd.DataFrame(), pd.DataFrame(status_pair_rows(a)))
    assert (pairs_only["n_estimates"].iloc[0], pairs_only["n_pairs"].iloc[0]) == (0, 5)
