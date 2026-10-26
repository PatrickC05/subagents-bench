"""The final study's L0 table: shared sessions per setting against the isolated baseline."""

from __future__ import annotations

import pandas as pd
import pytest

from subagents_bench.analysis.final import l0_table, main


def write(d, episodes, fixes):
    d.mkdir()
    pd.DataFrame(episodes).to_csv(d / "episodes.csv", index=False)
    pd.DataFrame(fixes).to_csv(d / "fixes.csv", index=False)


def test_l0_table(tmp_path):
    lv, iso = tmp_path / "levels", tmp_path / "isolated"
    # one shared L0 session of 2 bugs ($1.00, one fixed); an L1 session the table leaves out
    write(lv, [{"condition": "manager_owner_baseline", "trial_name": "t1", "task_name": "a", "cost_total": 1.0, "valid": True},
               {"condition": "manager_owner_eta", "trial_name": "t2", "task_name": "a", "cost_total": 9.0, "valid": True}],
          [{"condition": "manager_owner_baseline", "trial_name": "t1", "subtask_id": "b0", "fixed_end": True, "fixed_sec": 100, "valid": True},
           {"condition": "manager_owner_baseline", "trial_name": "t1", "subtask_id": "b1", "fixed_end": False, "fixed_sec": None, "valid": True},
           {"condition": "manager_owner_eta", "trial_name": "t2", "subtask_id": "b0", "fixed_end": True, "fixed_sec": 1, "valid": True}])
    # the same two bugs alone: both fixed at the end, bug 1 (sent at 4:00 in the shared sessions,
    # so 26:00 there) only after 27:00
    write(iso, [{"condition": "single_bug0", "trial_name": "i0", "task_name": "a", "cost_total": 0.2, "valid": True},
                {"condition": "single_bug1", "trial_name": "i1", "task_name": "a", "cost_total": 0.4, "valid": True}],
          [{"condition": "single_bug0", "trial_name": "i0", "subtask_id": "b0", "fixed_end": True, "fixed_sec": 300, "valid": True},
           {"condition": "single_bug1", "trial_name": "i1", "subtask_id": "b1", "fixed_end": True, "fixed_sec": 1620, "valid": True}])
    t = l0_table(lv, iso).set_index("point")
    assert list(t.index) == ["manager_owner", "isolated", "isolated_capped"]
    assert (t.loc["manager_owner", "n_fixed"], t.loc["manager_owner", "usd_per_bug"]) == (1, 0.5)
    assert (t.loc["isolated", "n_fixed"], t.loc["isolated", "usd_per_bug"]) == (2, pytest.approx(0.3))
    assert t.loc["isolated_capped", "n_fixed"] == 1 and t.loc["isolated", "usd_per_repo"] == pytest.approx(0.6)
    assert main([str(lv), str(iso), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "l0.png").is_file()


def test_isolated_condition_names_are_checked(tmp_path):
    lv, iso = tmp_path / "levels", tmp_path / "isolated"
    write(lv, [{"condition": "x_baseline", "trial_name": "t", "task_name": "a", "cost_total": 1.0}],
          [{"condition": "x_baseline", "trial_name": "t", "subtask_id": "b0", "fixed_end": True, "fixed_sec": 1}])
    write(iso, [{"condition": "single", "trial_name": "i", "task_name": "a", "cost_total": 1.0}],
          [{"condition": "single", "trial_name": "i", "subtask_id": "b0", "fixed_end": True, "fixed_sec": 1}])
    with pytest.raises(SystemExit, match="single_bug"):
        l0_table(lv, iso)
