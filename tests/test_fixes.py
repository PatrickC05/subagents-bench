"""Fix times from replayed workspace states (``analysis/fixes.py``) on synthetic trial dirs: a
harbor-like trial with ``agent/{episode.json, inbox.jsonl, replies.jsonl, snapshots.jsonl,
fix_states.json}`` and ``verifier/eval.json``, loaded with the analysis loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subagents_bench.analysis.fixes import (CONFIRM_SEC, HOLD_SEC, bug_fix_rows, fix_times, snapshot_records,
                                            state_at, states)
from subagents_bench.analysis.load import Trial, load_trial

T0 = 1_000_000.0
END = 1810.0  # the episode's end (the final state), seconds into the session


def bug(mid: int, idx: int, t: float, sid: str, urgent: bool = False) -> dict:
    return {"kind": "hint", "msg_id": mid, "hint_index": idx, "t_ready": T0 + t, "t_delivered": T0 + t + 1,
            "subtask_id": sid, "urgent": urgent, "target_files": [f"{sid}.py"], "f2p": []}


def status(mid: int, idx: int, t: float) -> dict:
    return {"kind": "status", "msg_id": mid, "hint_index": idx, "t_ready": T0 + t, "t_delivered": T0 + t + 1,
            "subtask_id": None, "urgent": False}


def rep(t: float, to: int, text: str = "x", eta: float | None = None, done: bool = False) -> dict:
    return {"t": T0 + t, "to": to, "text": text, "eta_min": eta, "done": done}


def timeline(points: list[tuple[float, dict]], end: float = END, final: dict | None = None,
             visible_ok: dict[float, bool] | None = None) -> list[dict]:
    """fix_states ``states``: snapshot k at each (t, {sid: status | (status, rerun)}), then the final
    state at ``end`` (by default the last snapshot's statuses)."""
    out = []
    for k, (t, bugs) in enumerate(points + [(end, final if final is not None else points[-1][1])]):
        last = k == len(points)
        out.append({"seq": None if last else k, "source": "final" if last else "snapshot", "t": T0 + t,
                    "patch": "final.patch" if last else f"snapshots/{k:04d}.patch", "applied": True,
                    "visible_ok": (visible_ok or {}).get(t, True), "error": None, "duration_s": 5.0,
                    "bugs": {sid: {"status": s[0] if isinstance(s, tuple) else s, "n_passed": 1, "n_failed": 0,
                                   "error": None, "rerun": s[1] if isinstance(s, tuple) else None}
                             for sid, s in bugs.items()}})
    return out


def snaps_for(states_: list[dict], heartbeats: list[float] = ()) -> list[dict]:
    """``snapshots.jsonl``: a changed snapshot per replayed snapshot state, plus unchanged ones at
    ``heartbeats`` (seconds into the session), numbered by time (the states' ``seq`` renumbered to
    match)."""
    marks = sorted([(s["t"] - T0, s) for s in states_ if s["source"] == "snapshot"]
                   + [(t, None) for t in heartbeats], key=lambda x: x[0])
    for k, (_, s) in enumerate(marks):
        if s is not None:
            s.update(seq=k, patch=f"snapshots/{k:04d}.patch")
    return [{"seq": k, "t_request": T0 + t, "t_start": T0 + t, "t_end": T0 + t + 1,
             "trigger": "heartbeat" if s is None else "tool", "tree": f"tree{k}", "changed": s is not None,
             "patch": None if s is None else s["patch"], "bytes": 0, "error": None}
            for k, (t, s) in enumerate(marks)]


def make_trial(root: Path, inbox: list[dict], replies: list[dict] = (), states_: list[dict] | None = None,
               snaps: list[dict] | None = None, verdicts: dict[str, bool] | None = None, matches=True,
               wall: float = 1800.0, t_end: float = END, job: str = "lv__r1__single_eta",
               name: str = "swe-sweep__click__abc", visible_ok: bool = True) -> Trial:
    d = root / job / name
    (d / "agent").mkdir(parents=True)
    (d / "verifier").mkdir()
    (d / "config.json").write_text("{}")
    ep = {"condition": job.rsplit("__", 1)[-1], "task_name": "swe-sweep__click", "session_t0": T0,
          "t_start": T0 - 5, "t_end": T0 + t_end, "exit_reason": "deadline",
          "config": {"interactive": {"replies": True, "wall_clock_sec": wall, "interval_sec": 240.0}}}
    (d / "agent" / "episode.json").write_text(json.dumps(ep))
    for kind, recs in (("inbox", inbox), ("replies", replies), ("snapshots", snaps)):
        if recs is not None:
            (d / "agent" / f"{kind}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    if states_ is not None:
        (d / "agent" / "fix_states.json").write_text(json.dumps({
            "version": 1, "task": "swe-sweep__click", "sent": [m["subtask_id"] for m in inbox if m["kind"] == "hint"],
            "states": states_, "trial": name, "graded_at": T0 + 9000, "image": "img",
            "final_matches_verifier": matches, "mismatches": []}))
    if verdicts is not None:
        (d / "verifier" / "eval.json").write_text(json.dumps({"visible_ok": visible_ok, "subtasks": [
            {"subtask_id": k, "resolved": v} for k, v in verdicts.items()]}))
    return load_trial(d)


def one_bug(tmp_path: Path, points, final=None, heartbeats=None, sent: float = 0.0, **kw) -> dict:
    sts = timeline(points, final=final)
    snaps = None if heartbeats is None else snaps_for(sts, heartbeats)
    t = make_trial(tmp_path, [bug(1, 0, sent, "b0")], states_=sts, snaps=snaps,
                   verdicts={"b0": (final or points[-1][1])["b0"] == "pass"}, **kw)
    return fix_times(t)["b0"]


def test_fix_times_simple_and_row_fields(tmp_path):
    inbox = [bug(1, 0, 0, "b0"), bug(3, 1, 240, "b1", urgent=True), status(2, 0, 120)]
    sts = timeline([(0, {"b0": "fail", "b1": "fail"}), (60, {"b0": "fail", "b1": "fail"}),
                    (100, {"b0": "pass", "b1": "fail"}), (400, {"b0": "pass", "b1": "pass"})],
                   visible_ok={100: False})
    t = make_trial(tmp_path, inbox, states_=sts, snaps=snaps_for(sts, [15, 30]),
                   verdicts={"b0": True, "b1": True})
    rows = bug_fix_rows(t)
    assert [r["subtask_id"] for r in rows] == ["b0", "b1"]  # by hint index; not the status check
    b0, b1 = rows
    assert (b0["fixed_sec"], b0["stable_sec"], b0["first_pass_sec"], b0["n_breaks"]) == (100, 100, 100, 0)
    assert b0["visible_ok_at_fix"] is False and b1["visible_ok_at_fix"] is True
    assert (b1["fixed_sec"], b1["fixed_session_sec"], b1["ready_session_sec"], b1["urgent"]) == (160, 400, 240, True)
    assert (b0["timing_ok"], b0["has_fix_states"], b0["final_matches_verifier"], b0["fixed_end"]) == (
        True, True, True, True)
    assert (b0["n_states"], b0["n_snapshots"], b0["n_snapshot_errors"], b0["final_status"]) == (5, 6, 0, "pass")
    assert (b0["pass_before_sent"], b0["state0_pass"], b0["n_unknown"], b0["msg_id"]) == (False, False, 0, 1)
    assert (b0["condition"], b0["round"], b0["sweep"], b0["trial_name"]) == (
        "single_eta", 1, "lv", "swe-sweep__click__abc")


def test_unknown_and_flaky_states_never_break_a_fix(tmp_path):
    r = one_bug(tmp_path / "u", [(0, {"b0": "fail"}), (100, {"b0": "pass"}), (130, {"b0": "unknown"}),
                                 (200, {"b0": "unknown"}), (210, {"b0": "pass"})], heartbeats=[150, 165, 180])
    assert (r["fixed_sec"], r["stable_sec"], r["n_breaks"], r["n_unknown"]) == (100, 100, 0, 2)
    # a fail whose rerun passed is a flaky test, even when seen long and often
    r = one_bug(tmp_path / "f", [(0, {"b0": "fail"}), (100, {"b0": "pass"}), (130, {"b0": ("fail", "pass")}),
                                 (400, {"b0": "pass"})], heartbeats=[150, 165, 180])
    assert (r["fixed_sec"], r["n_breaks"]) == (100, 0)


def test_a_failing_stretch_breaks_only_when_seen_twice_for_30_s(tmp_path):
    pts = [(0, {"b0": "fail"}), (100, {"b0": "pass"}), (150, {"b0": ("fail", "fail")}), (200, {"b0": "pass"})]
    # seen once (no snapshots.jsonl: each state counts once), though 50 s long: no break
    r = one_bug(tmp_path / "a", pts)
    assert (r["fixed_sec"], r["n_breaks"]) == (100, 0)
    # the heartbeat saw the same failing workspace again: confirmed; 100 -> 150 is under 120 s
    r = one_bug(tmp_path / "b", pts, heartbeats=[165])
    assert (r["fixed_sec"], r["stable_sec"], r["first_pass_sec"], r["n_breaks"]) == (200, 200, 100, 1)
    # two failing states but only 20 s until the next pass: no break
    short = [(0, {"b0": "fail"}), (100, {"b0": "pass"}), (150, {"b0": "fail"}), (160, {"b0": "fail"}),
             (150 + CONFIRM_SEC - 10, {"b0": "pass"})]
    assert one_bug(tmp_path / "c", short)["n_breaks"] == 0
    # unknown states inside a failing stretch are skipped, and do not count as observations
    mixed = [(0, {"b0": "fail"}), (100, {"b0": "pass"}), (150, {"b0": "fail"}), (160, {"b0": "unknown"}),
             (170, {"b0": "unknown"}), (190, {"b0": "pass"})]
    assert one_bug(tmp_path / "d", mixed)["n_breaks"] == 0
    mixed[3] = (160, {"b0": "fail"})
    assert one_bug(tmp_path / "e", mixed)["n_breaks"] == 1


def test_a_fix_must_hold_for_120_s(tmp_path):
    def run(name, gap):
        return one_bug(tmp_path / name, [(0, {"b0": "fail"}), (100, {"b0": "pass"}), (100 + gap, {"b0": "fail"}),
                                         (100 + gap + 60, {"b0": "fail"}), (600, {"b0": "pass"})])
    assert run("short", HOLD_SEC - 1)["fixed_sec"] == 600
    held = run("held", HOLD_SEC)
    assert (held["fixed_sec"], held["stable_sec"], held["n_breaks"]) == (100, 600, 1)


def test_the_end_cuts_a_streak_and_confirms_a_final_failure(tmp_path):
    # first passing 100 s before the end: "or to the end"
    r = one_bug(tmp_path / "a", [(0, {"b0": "fail"}), (END - 100, {"b0": "pass"})])
    assert (r["fixed_sec"], r["stable_sec"]) == (END - 100, END - 100)
    # broken in the final state: the end confirms it (it is what the user gets)
    r = one_bug(tmp_path / "b", [(0, {"b0": "fail"}), (END - 100, {"b0": "pass"})], final={"b0": ("fail", "fail")})
    assert (r["fixed_sec"], r["stable_sec"], r["first_pass_sec"], r["n_breaks"]) == (None, None, END - 100, 1)
    assert (r["final_status"], r["fixed_end"]) == ("fail", False)
    # held long before it broke at the end: fixed, but not stable
    r = one_bug(tmp_path / "c", [(0, {"b0": "fail"}), (100, {"b0": "pass"})], final={"b0": "fail"})
    assert (r["fixed_sec"], r["stable_sec"], r["n_breaks"]) == (100, None, 1)
    # never passing
    r = one_bug(tmp_path / "d", [(0, {"b0": "fail"}), (100, {"b0": "fail"})])
    assert (r["fixed_sec"], r["stable_sec"], r["first_pass_sec"], r["n_breaks"]) == (None, None, None, 0)


def test_a_pass_before_the_bug_was_sent_is_flagged_and_clipped(tmp_path):
    r = one_bug(tmp_path / "a", [(0, {"b0": "fail"}), (200, {"b0": "pass"})], sent=480)
    assert (r["fixed_sec"], r["stable_sec"], r["first_pass_sec"], r["pass_before_sent"]) == (0, 0, 0, True)
    assert r["fixed_session_sec"] == 480 and r["state0_pass"] is False
    r = one_bug(tmp_path / "b", [(0, {"b0": "pass"})])  # snapshot 0 (untouched workspace) passes
    assert r["state0_pass"] is True and r["pass_before_sent"] is False  # sent at the same time


def test_no_timing_without_a_verified_replay(tmp_path):
    pts = timeline([(0, {"b0": "fail"}), (100, {"b0": "pass"})])
    t = make_trial(tmp_path / "x", [bug(1, 0, 0, "b0")], states_=pts, verdicts={"b0": False}, matches=False)
    r = fix_times(t)["b0"]
    assert (r["timing_ok"], r["has_fix_states"], r["final_matches_verifier"]) == (False, True, False)
    assert (r["fixed_sec"], r["n_breaks"], r["n_states"], r["fixed_end"]) == (None, None, 3, False)
    assert state_at(t, 500) == {}
    t = make_trial(tmp_path / "y", [bug(1, 0, 0, "b0")], verdicts={"b0": True})  # never replayed
    r = fix_times(t)["b0"]
    assert (r["timing_ok"], r["has_fix_states"], r["final_matches_verifier"], r["fixed_end"]) == (
        False, False, None, True)
    t = make_trial(tmp_path / "z", [bug(1, 0, 0, "b0")])  # neither replayed nor graded
    assert fix_times(t)["b0"]["fixed_end"] is None
    t = make_trial(tmp_path / "w", [bug(1, 0, 0, "b0")], verdicts={"other": True}, visible_ok=True)
    assert fix_times(t)["b0"]["fixed_end"] is False  # graded, no verdict for it (a patch that broke the build)
    assert fix_times(make_trial(tmp_path / "v", [status(1, 0, 0)])) == {}  # no bugs


@pytest.mark.parametrize("matches,ok", [({"b0": True, "b1": True}, True), ({"b0": True, "b1": False}, False),
                                        ({}, False), (None, False), ("yes", False)])
def test_final_matches_verifier_shapes(tmp_path, matches, ok):
    t = make_trial(tmp_path, [bug(1, 0, 0, "b0")], states_=timeline([(0, {"b0": "pass"})]), matches=matches)
    assert fix_times(t)["b0"]["timing_ok"] is ok


def test_states_count_unchanged_snapshots_and_skip_errors(tmp_path):
    sts = timeline([(0, {"b0": "fail"}), (100, {"b0": "pass"})])
    snaps = snaps_for(sts, [15, 30, 115])
    snaps.append({**snaps[-1], "seq": len(snaps), "t_start": T0 + 130, "tree": None, "changed": False,
                  "error": "AgentStopped"})
    t = make_trial(tmp_path, [bug(1, 0, 0, "b0")], states_=sts, snaps=list(reversed(snaps)))
    assert [r["seq"] for r in snapshot_records(t)] == list(range(6))  # by seq
    got = states(json.loads((t.dir / "agent" / "fix_states.json").read_text()), snapshot_records(t))
    assert [(s.seq, s.obs, s.final) for s in got] == [(0, 3, False), (3, 2, False), (None, 1, True)]
    assert fix_times(t)["b0"]["n_snapshot_errors"] == 1
    # the loader may not know the record kind yet: the file is read directly
    t.records.pop("snapshots", None)
    assert len(snapshot_records(t)) == 6


def test_state_at_carries_the_last_known_status(tmp_path):
    inbox = [bug(1, 0, 0, "b0"), bug(2, 1, 240, "b1")]
    sts = timeline([(0, {"b0": "fail", "b1": "fail"}), (100, {"b0": "pass", "b1": "fail"}),
                    (300, {"b0": "unknown", "b1": "unknown"}), (400, {"b0": ("fail", "pass"), "b1": "pass"}),
                    (500, {"b0": "fail", "b1": "pass"})])
    t = make_trial(tmp_path, inbox, states_=sts, verdicts={"b0": False, "b1": True})
    assert state_at(t, -1) == {"b0": "unknown", "b1": "unknown"}
    assert state_at(t, 0) == {"b0": "fail", "b1": "fail"}
    assert state_at(t, 99.9) == {"b0": "fail", "b1": "fail"}
    assert state_at(t, 100) == {"b0": "pass", "b1": "fail"}
    assert state_at(t, 350) == {"b0": "pass", "b1": "fail"}  # the unknown state carries nothing
    assert state_at(t, 450) == {"b0": "pass", "b1": "pass"}  # a flaky fail stands for the pass before it
    assert state_at(t, 500) == {"b0": "fail", "b1": "pass"}
