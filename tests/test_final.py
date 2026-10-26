"""The final study's L0 table: shared sessions per setting against the isolated baseline."""

from __future__ import annotations

import types

import pandas as pd
import pytest

from subagents_bench.analysis import final
from subagents_bench.analysis.final import l0_table, main


def write(d, episodes, fixes):
    d.mkdir()
    pd.DataFrame(episodes).to_csv(d / "episodes.csv", index=False)
    pd.DataFrame(fixes).to_csv(d / "fixes.csv", index=False)


def ep(cond, trial, cost, task="a", valid=True):
    return {"condition": cond, "trial_name": trial, "task_name": task, "cost_total": cost, "valid": valid,
            "trial_dir": f"jobs/x/{trial}"}


def fx(cond, trial, sid, fixed_end, ready=0.0, task="a"):
    return {"condition": cond, "trial_name": trial, "task_name": task, "subtask_id": sid, "fixed_end": fixed_end,
            "ready_session_sec": ready, "valid": True}


@pytest.fixture
def trials(monkeypatch):
    """Fake trial dirs: every shared session has a 1800 s wall clock; ``states`` maps an isolated
    trial to its bug's replayed (time, status) list (a trial missing from it has no usable replay)."""
    states: dict[str, list[tuple[float, str]]] = {}
    asked: list[tuple[str, float]] = []
    monkeypatch.setattr(final, "_trial", lambda root, d: types.SimpleNamespace(
        name=d.rsplit("/", 1)[-1], episode={"config": {"interactive": {"wall_clock_sec": 1800.0}}}))

    def state_at(t, at):
        asked.append((t.name, at))
        if t.name not in states:
            return {}
        known = [s for when, s in states[t.name] if when <= at]
        return {"b0": known[-1] if known else "unknown", "b1": known[-1] if known else "unknown"}
    monkeypatch.setattr(final, "state_at", state_at)
    return states, asked


def test_l0_table(tmp_path, trials):
    states, asked = trials
    lv, iso = tmp_path / "levels", tmp_path / "isolated"
    # one shared L0 session of 2 bugs ($1.00, one fixed; bug 1 sent at 4:00), an L1 session the
    # table leaves out
    write(lv, [ep("manager_owner_baseline", "t1", 1.0), ep("manager_owner_eta", "t2", 9.0)],
          [fx("manager_owner_baseline", "t1", "b0", True), fx("manager_owner_baseline", "t1", "b1", False, 240.0),
           fx("manager_owner_eta", "t2", "b0", True)])
    # the same two bugs alone, both fixed at the end. Bug 0 passed at 5:00. Bug 1 (26:00 in the
    # shared sessions) passed at 16:40, broke at 19:10 and passed again only at 27:00: at its
    # cap it fails, though a pass held before it
    write(iso, [ep("single_bug0", "i0", 0.2), ep("single_bug1", "i1", 0.4)],
          [fx("single_bug0", "i0", "b0", True), fx("single_bug1", "i1", "b1", True)])
    states.update({"i0": [(300.0, "pass")], "i1": [(1000.0, "pass"), (1150.0, "fail"), (1620.0, "pass")]})
    t = l0_table(lv, iso).set_index("point")
    assert list(t.index) == ["manager_owner", "isolated", "isolated_capped"]
    assert (t.loc["manager_owner", "n_fixed"], t.loc["manager_owner", "usd_per_bug"]) == (1, 0.5)
    assert (t.loc["isolated", "n_fixed"], t.loc["isolated", "usd_per_bug"]) == (2, pytest.approx(0.3))
    assert t.loc["isolated_capped", "n_fixed"] == 1 and t.loc["isolated", "usd_per_repo"] == pytest.approx(0.6)
    assert sorted(asked) == [("i0", 1800.0), ("i1", 1560.0)]  # wall clock minus the send time there

    # no usable replay for bug 1's session: left out of the capped point only
    del states["i1"]
    t = l0_table(lv, iso).set_index("point")
    assert (t.loc["isolated", "n_bugs"], t.loc["isolated_capped", "n_bugs"], t.loc["isolated_capped", "n_fixed"]) == (2, 1, 1)

    out = tmp_path / "out"
    assert main(["--model", f"A={lv}:{iso}", "--model", f"B={lv}:{iso}", "--out", str(out)]) == 0
    both = pd.read_csv(out / "l0.csv")
    assert (out / "l0.png").is_file() and list(both["model"].unique()) == ["A", "B"] and len(both) == 6
    with pytest.raises(SystemExit):
        main(["--model", f"A={lv}", "--out", str(out)])


def test_points_cover_the_same_bugs_and_no_verdict_is_not_fixed(tmp_path, trials):
    states, _ = trials
    lv, iso = tmp_path / "levels", tmp_path / "isolated"
    # background_hinted has bugs b0 and b1 on task a; manager_owner's session on task a was invalid
    # (only task c, bug b0); b0 of background_hinted on task c has no verdict (NaN): not a fix
    write(lv, [ep("background_hinted_baseline", "h1", 1.0), ep("manager_owner_baseline", "o1", 1.0, valid=False),
               ep("manager_owner_baseline", "o2", 1.0, task="c"), ep("background_hinted_baseline", "h2", 1.0, task="c")],
          [fx("background_hinted_baseline", "h1", "b0", True), fx("background_hinted_baseline", "h1", "b1", None, 240.0),
           fx("background_hinted_baseline", "h2", "b0", None, task="c"),
           fx("manager_owner_baseline", "o2", "b0", True, task="c")])
    write(iso, [ep("single_bug0", "i0", 0.2), ep("single_bug0", "i2", 0.2, task="c")],
          [fx("single_bug0", "i0", "b0", True), fx("single_bug0", "i2", "b0", None, task="c")])
    states.update({"i2": [(0.0, "fail")]})
    t = l0_table(lv, iso).set_index("point")
    assert (t["n_bugs"] == 1).all() and (t["n_dropped"] == 2).all()  # only (c, b0) is in every point
    assert t.loc["background_hinted", "n_fixed"] == 0 and t.loc["manager_owner", "n_fixed"] == 1
    assert t.loc["isolated", "n_fixed"] == 0


def test_isolated_condition_names_are_checked(tmp_path, trials):
    lv, iso = tmp_path / "levels", tmp_path / "isolated"
    write(lv, [ep("x_baseline", "t", 1.0)], [fx("x_baseline", "t", "b0", True)])
    write(iso, [ep("single", "i", 1.0)], [fx("single", "i", "b0", True)])
    with pytest.raises(SystemExit, match="single_bug"):
        l0_table(lv, iso)


def test_a_provider_error_counts_as_a_failure(tmp_path, trials):
    """A session that ended in a provider error stays in, with its bugs not fixed (even one whose
    replayed state passed) and its cost counted; another invalid session is left out."""
    states, asked = trials
    lv, iso = tmp_path / "levels", tmp_path / "isolated"
    failed = {**ep("manager_owner_baseline", "t1", 1.0, valid=False), "invalid_reason": "provider_error"}
    crashed = {**ep("manager_owner_baseline", "t3", 1.0, task="c", valid=False), "invalid_reason": "agent_crash"}
    write(lv, [failed, crashed], [fx("manager_owner_baseline", "t1", "b0", True),
                                  fx("manager_owner_baseline", "t3", "b0", True, task="c")])
    iso_failed = {**ep("single_bug0", "i0", 0.5, valid=False), "invalid_reason": "provider_error"}
    write(iso, [iso_failed, ep("single_bug0", "i3", 0.2, task="c")],
          [fx("single_bug0", "i0", "b0", None), fx("single_bug0", "i3", "b0", True, task="c")])
    states.update({"i0": [(0.0, "pass")]})
    t = l0_table(lv, iso).set_index("point")
    assert (t["n_bugs"] == 1).all()  # (a, b0): task c's shared session crashed, so (c, b0) is not in every point
    assert (t.loc["manager_owner", "n_fixed"], t.loc["manager_owner", "usd_per_bug"]) == (0, 1.0)
    assert (t.loc["isolated", "n_fixed"], t.loc["isolated_capped", "n_fixed"]) == (0, 0)
    assert t.loc["isolated", "usd_total"] == 0.5 and not any(name == "i0" for name, _ in asked)
