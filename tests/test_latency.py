"""Performance and cost against latency: the join and the Pareto frontier."""

from __future__ import annotations

import pandas as pd

from subagents_bench.plots.latency import join, main, pareto


def test_pareto_keeps_points_nothing_beats():
    t = pd.DataFrame({"median_sec": [5, 6, 20, 30, 40], "fixed_frac": [0.6, 0.8, 0.7, 0.9, 0.9],
                      "usd_per_bug": [1.0, 0.1, 0.2, 0.05, 0.01]})
    assert pareto(t, "fixed_frac", higher=True) == [(5, 0.6), (6, 0.8), (30, 0.9)]
    assert pareto(t, "usd_per_bug", higher=False) == [(5, 1.0), (6, 0.1), (30, 0.05), (40, 0.01)]


def test_join_uses_l0_shared_settings(tmp_path):
    l0 = pd.DataFrame({"model": ["claude-opus-5.5"] * 3, "point": ["background_hinted", "isolated", "manager_owner"],
                       "fixed_frac": [0.9, 0.9, 0.95], "usd_per_bug": [0.1, 0.05, 0.3]})
    rt = pd.DataFrame({"model": ["Opus 5.5"] * 3, "setting": ["background_hinted", "manager_owner", "background_hinted"],
                       "level": ["L0", "L0", "L1"], "median_sec": [24.0, 6.0, 9.0], "replied_frac": [1.0, 1.0, 1.0]})
    t = join(l0, rt)
    assert sorted(zip(t["point"], t["median_sec"])) == [("background_hinted", 24.0), ("manager_owner", 6.0)]
    l0.to_csv(tmp_path / "l0.csv", index=False)
    rt.to_csv(tmp_path / "rt.csv", index=False)
    assert main(["--l0", str(tmp_path / "l0.csv"), "--response-time", str(tmp_path / "rt.csv"), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "latency.png").is_file() and (tmp_path / "latency_background_hinted.png").is_file()
