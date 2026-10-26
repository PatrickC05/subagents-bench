"""Experiment 2's cross-model figures (subagents_bench.plots.final)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from subagents_bench.plots.final import FIGURES, censored_median, main, split


def test_split_and_censored_median():
    assert split("manager_owner_questions") == ("manager_owner", "questions")
    assert split("background_hinted_baseline") == ("background_hinted", "baseline")
    assert censored_median(pd.Series([10.0, None, 30.0])) == 30.0
    assert math.isinf(censored_median(pd.Series([10.0, None, None])))
    assert censored_median(pd.Series([10.0, 20.0])) == 15.0


def write_model(d, slow: float):
    d.mkdir()
    msgs, eps, calls, tools, est = [], [], [], [], []
    for s in ("background_hinted", "manager_background", "manager_owner", "single"):  # single is left out
        for lv in ("baseline", "eta", "questions", "urgent"):
            cond, trial = f"{s}_{lv}", f"{s}_{lv}_t"
            eps.append({"condition": cond, "trial_name": trial, "wall_clock_sec": 1800.0, "valid": True})
            for k in range(3):
                msgs.append({"condition": cond, "trial_name": trial, "valid": True, "kind": "bug", "index": k,
                             "user_reply_sec": None if k == 2 else slow * (k + 1), "reply_sec": 5.0,
                             "fixed": k == 0})
            if lv in ("questions", "urgent"):
                msgs.append({"condition": cond, "trial_name": trial, "valid": True, "kind": "question", "index": 0,
                             "user_reply_sec": None, "reply_sec": 30.0, "fixed": None})
            calls.append({"condition": cond, "trial_name": trial, "valid": True, "role": "orchestrator",
                          "t_start": 0.0, "t_end": 900.0})
            calls.append({"condition": cond, "trial_name": trial, "valid": True, "role": "orchestrator",
                          "t_start": 600.0, "t_end": 1000.0})  # overlaps the first: 1000 s in all
            tools += [{"condition": cond, "trial_name": trial, "valid": True, "role": "orchestrator", "name": n,
                       "t_start": a, "t_end": b} for n, a, b in (("bash", 1000.0, 1180.0), ("wait", 1180.0, 1800.0))]
            est.append({"condition": cond, "trial_name": trial, "valid": True, "scored": True, "task_name": "a",
                        "subtask_id": "b0", "ready_session_sec": 0.0, "est_sec": 60.0, "eta_min": 5.0,
                        "fixed_sec": 360.0, "on_time": True, "weight": 1.0})
    for name, rows in (("messages", msgs), ("episodes", eps), ("calls", calls), ("tools", tools), ("estimates", est)):
        pd.DataFrame(rows).to_csv(d / f"{name}.csv", index=False)


def test_figures_and_tables(tmp_path):
    write_model(tmp_path / "a", 100.0)
    write_model(tmp_path / "b", 200.0)
    out = tmp_path / "fig"
    assert main(["--model", f"A={tmp_path / 'a'}", "--model", f"B={tmp_path / 'b'}", "--out", str(out)]) == 0
    for f in FIGURES:
        assert (out / f"{f.__name__}.png").is_file() and (out / f"{f.__name__}.csv").is_file()
    rt = pd.read_csv(out / "response_time.csv")
    assert set(rt["setting"]) == {"background_hinted", "manager_background", "manager_owner"}
    r = rt[(rt["model"] == "B") & (rt["setting"] == "manager_owner") & (rt["level"] == "eta")].iloc[0]
    assert (r["n"], r["replied_frac"]) == (2, 0.5) and math.isinf(r["median_sec"])  # bug 2 never replied
    ot = pd.read_csv(out / "orchestrator_time.csv").iloc[0]
    assert ot["llm_frac"] == pytest.approx(1000 / 1800) and ot["tool_frac"] == pytest.approx(180 / 1800)
    qb = pd.read_csv(out / "question_vs_bug.csv")
    assert set(qb["kind"]) == {"bug", "question"}
    assert qb[(qb["kind"] == "question")]["median_sec"].eq(30.0).all()
    eta = pd.read_csv(out / "eta.csv")
    assert eta["actual_min"].eq(5.0).all()  # fixed at 6:00, estimated at 1:00


def test_model_argument_is_checked(tmp_path):
    with pytest.raises(SystemExit):
        main(["--model", "no-equals-sign", "--out", str(tmp_path)])
