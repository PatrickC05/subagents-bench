"""End-to-end offline pipeline: REAL runtime output -> check-run -> completeness
-> aggregate -> plot, through the CLI functions.

:mod:`offline_pipeline` runs real episodes (``runner.run_task`` with the real
orchestrator, subagents, arbiter, tools and recorder on LocalShell sandboxes,
scripted models) for five pilot conditions -- single, parallel hard,
sequential medium, arbiter hard (copy isolation) and arbiter easy (with a
debate round) -- x 2 tasks x 2 rounds, laid out as harbor jobs by the sweep
launcher's own code. This test is the proof that what the runtime writes is
what the analysis expects: every record validates, every consistency check
passes, every field that these conditions should populate is populated, the
tables join harbor/manifest/task metadata correctly, the measured latency
replay reproduces the critical path, and every figure that has data is drawn.

No network, no Docker. The model behaviour is scripted (a test fixture), so
none of the numbers produced are measurements.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest

from offline_pipeline import CONDITIONS, SWEEP_NAME, TASKS, build_offline_sweep
from subagents_bench import cli
from subagents_bench import sweep as sw
from subagents_bench.analysis.validate import check_run, completeness
from subagents_bench.plots import FIGURE_NAMES

ROUNDS = 2
N_TRIALS = len(CONDITIONS) * len(TASKS) * ROUNDS


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline")
    built = build_offline_sweep(root, rounds=ROUNDS)
    analysis = root / "analysis"
    figures = root / "figures"
    return {"root": root, "sweep": built, "jobs": built.jobs_dir, "analysis": analysis, "figures": figures}


@pytest.fixture(scope="module")
def aggregated(offline):
    rc = cli.main(["aggregate", str(offline["jobs"]), "--out", str(offline["analysis"])])
    assert rc == 0
    a = offline["analysis"]
    return {name: pd.read_csv(a / f"{name}.csv") for name in (
        "episodes", "calls", "tools", "steps", "spawns", "dispatches", "arbitrations", "conditions",
        "conditions_by_difficulty", "paired", "calibration")} | {"summary": json.loads((a / "summary.json").read_text())}


# ---------------------------------------------------------------------------
# the runtime's output, as harbor jobs
# ---------------------------------------------------------------------------


def test_every_trial_ran_and_jobs_look_complete_to_the_launcher(offline):
    built = offline["sweep"]
    assert len(built.trials) == N_TRIALS
    assert {t["exit_reason"] for t in built.trials} == {"finished"}
    jobs = {t["job"] for t in built.trials}
    assert jobs == {sw.job_name(SWEEP_NAME, r, c) for r in range(1, ROUNDS + 1) for c in CONDITIONS}
    for job in jobs:  # the launcher's own completeness rule accepts the harbor-shaped job dirs
        assert sw.job_status(offline["jobs"] / job).complete
    # One scripted wrong answer: sequential medium, offline-sum, round 1.
    unsolved = [(t["condition"], t["task"], t["round"]) for t in built.trials if t["reward"] < 1.0]
    assert unsolved == [("gemma31b_sequential_medium", "offline-sum", 1)]


def test_check_run_is_clean(offline, capsys):
    assert cli.main(["check-run", str(offline["jobs"])]) == 0
    assert "check-run OK" in capsys.readouterr().out
    report = check_run([offline["jobs"]])
    assert report.n_episodes == N_TRIALS and report.n_trials_without_episode == 0
    assert report.problems == []  # neither errors nor warnings


def test_completeness_flags_nothing_unexpected(offline, tmp_path):
    out = tmp_path / "completeness.csv"
    assert cli.main(["completeness", str(offline["jobs"]), "--strict", "--out", str(out)]) == 0
    rows = {(r["file"], r["role"], r["field"]): r for r in completeness([offline["jobs"]])}
    unexpected = [k for k, r in rows.items() if r["never_populated"] and not r["often_empty"]]
    assert unexpected == []
    # Fields these conditions must populate (spot checks across files and roles).
    must = [
        ("calls.jsonl", "orchestrator", "ttft_sec"), ("calls.jsonl", "subagent", "decode_sec"),
        ("calls.jsonl", "orchestrator", "reasoning_tokens"), ("calls.jsonl", "orchestrator", "cached_tokens"),
        ("calls.jsonl", "subagent", "base_url"), ("calls.jsonl", "debate", "dispatch_id"),
        ("calls.jsonl", "arbiter", "arbitration_id"), ("tools.jsonl", "subagent", "cwd_after"),
        ("tools.jsonl", "subagent", "exec_sec"), ("dispatches.jsonl", "subagent", "files_modified"),
        ("arbitrations.jsonl", "arbiter", "applied_files"), ("arbitrations.jsonl", "arbiter", "isolation_sec"),
        ("arbitrations.jsonl", "arbiter", "debate_rounds_used"), ("spawns.jsonl", "orchestrator", "files_modified"),
        ("episode.json", "episode", "harness_exec_sec"), ("episode.json", "episode", "files_modified"),
        ("episode.json", "episode", "trial_name"), ("episode.json", "episode", "container_arch"),
        ("episode.json", "episode", "has_timeout_binary"), ("episode.json", "episode", "redispatches"),
    ]
    for key in must:
        assert not rows[key]["never_populated"], key
    assert out.is_file() and len(pd.read_csv(out)) == len(rows)


# ---------------------------------------------------------------------------
# aggregate: joins, totals, costs, latency
# ---------------------------------------------------------------------------


def test_episodes_join_harbor_manifest_and_task_metadata(offline, aggregated):
    ep = aggregated["episodes"]
    assert len(ep) == N_TRIALS and ep["valid"].all() and (ep["episode_status"] == "done").all()
    assert set(ep["sweep"]) == {SWEEP_NAME} and set(ep["round"]) == {1, 2}
    assert set(ep["condition"]) == set(CONDITIONS)
    by_trial = {t["trial_dir"].name: t for t in offline["sweep"].trials}
    for r in ep.itertuples():
        t = by_trial[r.trial_name]
        assert (r.condition, r.round, r.task_name, r.job_name) == (t["condition"], t["round"], t["task"], t["job"])
        assert r.reward == t["reward"] and r.resolved == int(t["reward"] == 1.0)
        assert r.episode_id == t["episode_id"]
        assert r.difficulty == TASKS[r.task_name].meta["difficulty"]
        assert r.expert_time_min == TASKS[r.task_name].meta["expert_time_min"]
        assert r.agent_exec_sec > 0 and r.trial_total_sec >= r.agent_exec_sec
    design = ep.set_index("condition")
    assert design.loc["single_5step", "topology"].unique().tolist() == ["single"]
    assert design.loc["gemma31b_arbiter_hard", "strategy"].unique().tolist() == ["arbiter"]
    assert design.loc["gemma31b_arbiter_hard", "n_candidates"].unique().tolist() == [3]
    assert design.loc["gemma31b_sequential_medium", "scope"].unique().tolist() == ["medium"]
    assert set(ep["subagent_model"].dropna()) == {"nvidia/Gemma-4-31B-IT-NVFP4"}
    assert set(ep["orchestrator_model"]) == {"deepseek-ai/DeepSeek-V4.1-Flash"}


def test_tokens_counts_and_costs_agree_with_the_records(aggregated):
    ep, calls = aggregated["episodes"], aggregated["calls"]
    roles = {"orchestrator": "orch", "subagent": "sub", "arbiter": "arb", "debate": "deb"}
    for r in ep.itertuples():
        mine = calls[calls["episode_id"] == r.episode_id]
        for role, s in roles.items():
            rc = mine[mine["role"] == role]
            assert getattr(r, f"{s}_calls") == len(rc)
            assert getattr(r, f"{s}_prompt_tokens") == rc["prompt_tokens"].sum()
            assert getattr(r, f"{s}_completion_tokens") == rc["completion_tokens"].sum()
        assert r.total_tokens == mine["prompt_tokens"].sum() + mine["completion_tokens"].sum()
        # base.yaml's run-time prices are Sail's balanced list prices, so the offline
        # recomputation under that scheme must reproduce the logged cost.
        assert r.cost_sail_balanced_total == pytest.approx(r.cost_runtime_total, rel=1e-9)
        assert r.n_retries == 1  # the scripted 429 on every episode's first call
    arb = ep[ep["strategy"] == "arbiter"]
    assert (arb["arbitrations"] == 1).all() and (arb["dispatches"] == 3).all()
    deb = ep[ep["condition"] == "gemma31b_arbiter_easy"]
    assert (deb["deb_calls"] == 3).all() and (deb["arb_calls"] == 2).all()
    par = ep[ep["condition"] == "gemma31b_parallel_hard"]
    assert (par["spawn_calls"] == 2).all() and (par["redispatches"] == 1).all()
    assert (ep.loc[ep["strategy"] == "arbiter", "instruction_following"] == 0.5).all()  # 1 of 2 granted
    single = ep[ep["topology"] == "single"]
    assert (single["dispatches"] == 0).all() and single["parallel_speedup"].isna().all()


def test_measured_replay_reproduces_the_critical_path_close_to_wall_clock(aggregated):
    ep = aggregated["episodes"]
    for r in ep.itertuples():
        assert r.sim_measured_wall_sec == pytest.approx(r.cp_total_sec, abs=1e-6)
        parts = r.cp_orch_llm_sec + r.cp_sub_llm_sec + r.cp_arb_llm_sec + r.cp_tool_sec
        assert parts == pytest.approx(r.cp_total_sec, abs=1e-6)
        # The chain model never exceeds the wall clock, and the gap is small...
        assert r.cp_total_sec <= r.wall_clock_sec + 1e-3
        assert abs(r.wall_clock_sec - r.cp_total_sec) <= 0.10 * r.wall_clock_sec + 0.5, r.trial_name
        # ...and measured: the harness's own measurement commands account for most of it.
        assert r.harness_exec_sec > 0 and r.overhead_sec <= r.harness_exec_sec + 0.15, r.trial_name
    # Subagents of a parallel group overlap; sequential ones do not. (For arbiter groups the
    # denominator also holds isolation, judging and apply time, so it is not a pure overlap ratio.)
    by = ep.groupby("condition")["parallel_speedup"].mean()
    assert by["gemma31b_parallel_hard"] > 1.1
    assert by["gemma31b_sequential_medium"] == pytest.approx(1.0, abs=0.05)
    arb_hard = ep[ep["condition"] == "gemma31b_arbiter_hard"]
    assert (arb_hard["cp_arb_llm_sec"] > 0).all() and (arb_hard["cp_tool_sec"] > 0).all()
    s = aggregated["summary"]["latency_checks"]
    assert s["measured_replay_max_abs_diff_sec"] < 1e-6
    for prof in ("fitted", "roofline_h100"):
        assert ep[f"sim_{prof}_wall_sec"].notna().all() and (ep[f"sim_{prof}_wall_sec"] > 0).all()


def test_record_tables_conditions_paired_and_calibration(offline, aggregated):
    ep = aggregated["episodes"]
    arbs = aggregated["arbitrations"]
    assert len(arbs) == 4 * ROUNDS
    hard = arbs[arbs["isolation"] == "copy"]
    assert len(hard) == 2 * ROUNDS and hard["applied"].all() and (hard["isolation_sec"] > 0).all()
    easy = arbs[arbs["isolation"] == "shared"]
    assert (easy["debate_rounds_used"] == 1).all() and (hard["debate_rounds_used"] == 0).all()
    disp = aggregated["dispatches"]
    assert len(disp) == ep["dispatches"].sum()
    assert set(disp["status"]) == {"submitted"}
    assert len(aggregated["spawns"]) == ep["spawn_calls"].sum()
    assert len(aggregated["steps"]) == ep["orchestrator_steps"].sum()
    n_tool_lines = sum(len(p.read_text().splitlines()) for p in offline["jobs"].glob("*/*/agent/tools.jsonl"))
    assert len(aggregated["tools"]) == n_tool_lines
    n_call_lines = sum(len(p.read_text().splitlines()) for p in offline["jobs"].glob("*/*/agent/calls.jsonl"))
    assert len(aggregated["calls"]) == n_call_lines

    cond = aggregated["conditions"].set_index("condition")
    assert set(cond.index) == set(CONDITIONS)
    assert cond.loc["single_5step", "resolve_rate"] == 1.0
    assert cond.loc["gemma31b_sequential_medium", "resolve_rate"] == pytest.approx(3 / 4)
    lo, hi = cond.loc["gemma31b_sequential_medium", ["resolve_ci_lo", "resolve_ci_hi"]]
    assert 0 < lo < 0.75 < hi <= 1
    assert (cond["n"] == len(TASKS) * ROUNDS).all()
    byd = aggregated["conditions_by_difficulty"]
    assert set(byd["difficulty"]) == {"easy", "medium"} and len(byd) == 2 * len(CONDITIONS)

    paired = aggregated["paired"]
    assert set(paired["baseline"]) == {"single_5step"}
    assert set(paired["condition"]) == set(CONDITIONS) - {"single_5step"}
    assert (paired["n_pairs"] == len(TASKS) * ROUNDS).all()
    seq = paired[paired["condition"] == "gemma31b_sequential_medium"].iloc[0]
    assert (seq["only_base"], seq["only_cond"], seq["delta_resolve"]) == (1, 0, pytest.approx(-0.25))

    cal = aggregated["calibration"].set_index("model")
    assert set(cal.index) == {"deepseek-ai/DeepSeek-V4.1-Flash", "nvidia/Gemma-4-31B-IT-NVFP4"}
    assert set(cal["provider"]) == {"sail"} and set(cal["completion_window"]) == {"balanced"}
    assert (cal["n_calls"] > 0).all() and cal["ttft_median"].notna().all()


# ---------------------------------------------------------------------------
# side directories of a real jobs dir are not double counted
# ---------------------------------------------------------------------------


def test_partial_and_oracle_dirs_are_ignored(offline, tmp_path):
    jobs = tmp_path / "jobs"
    shutil.copytree(offline["jobs"], jobs)
    some_job = sorted(p for p in jobs.iterdir() if p.name.startswith(SWEEP_NAME))[0]
    shutil.copytree(some_job, jobs / sw.PARTIAL_DIR / f"{some_job.name}.partial.1")
    shutil.copytree(some_job, jobs / sw.ORACLE_JOB / "oracle-job")
    assert check_run([jobs]).n_episodes == N_TRIALS
    out = tmp_path / "analysis"
    assert cli.main(["aggregate", str(jobs), "--out", str(out)]) == 0
    assert len(pd.read_csv(out / "episodes.csv")) == N_TRIALS


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def test_every_figure_with_data_is_drawn(offline, aggregated, capsys):
    figs = offline["figures"]
    assert cli.main(["plot", str(offline["analysis"]), "--out", str(figs)]) == 0
    out = capsys.readouterr().out
    assert "skipped" not in out, out
    index = (figs / "index.md").read_text()
    assert "Skipped figures" not in index
    for name in FIGURE_NAMES:
        if name == "timelines":
            continue
        for ext in ("png", "pdf"):
            assert (figs / f"{name}.{ext}").stat().st_size > 5000, name
        assert f"(`{name}`)" in index
    timelines = sorted((figs / "timelines").glob("*.png"))
    per_cond: dict[str, int] = {}
    for p in timelines:
        per_cond[p.name.split("__")[0]] = per_cond.get(p.name.split("__")[0], 0) + 1
    assert set(per_cond) == set(CONDITIONS) and max(per_cond.values()) <= 2
    assert not math.isnan(aggregated["episodes"]["cp_total_sec"].sum())


# ---------------------------------------------------------------------------
# docs/DATA.md documents exactly what this pipeline writes
# ---------------------------------------------------------------------------


def test_data_dictionary_is_complete_and_up_to_date(offline, aggregated):
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gen_data_docs", repo / "scripts" / "gen_data_docs.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    text, missing = gen.build(offline["jobs"], offline["analysis"], gen.OFFLINE_NOTE)
    assert missing == [], f"undocumented keys/columns: {missing}"
    assert (repo / "docs" / "DATA.md").read_text() == text + "\n", "docs/DATA.md is stale: run scripts/gen_data_docs.py"


def test_calibrate_fits_every_model_and_feeds_the_fitted_profile(offline, tmp_path):
    import yaml

    out = tmp_path / "serving_fitted.yaml"
    assert cli.main(["calibrate", str(offline["jobs"]), "--out", str(out), "--csv", str(tmp_path / "cal.csv")]) == 0
    doc = yaml.safe_load(out.read_text())
    params = {p["model"]: p for p in doc["params"]}
    assert set(params) == {"deepseek-ai/DeepSeek-V4.1-Flash", "nvidia/Gemma-4-31B-IT-NVFP4"}
    for p in params.values():
        assert (p["provider"], p["completion_window"]) == ("sail", "balanced")
        assert p["ttft_a_sec"] is not None and p["decode_tok_s"] > 0 and p["n_calls"] == p["ttft_fit_n"] > 0
    # A serving.yaml whose fitted profile points at that file simulates every episode.
    serving = tmp_path / "serving.yaml"
    serving.write_text(yaml.safe_dump({"profiles": {"fitted": {"kind": "fitted", "params_file": str(out)}}}))
    agg = tmp_path / "analysis"
    assert cli.main(["aggregate", str(offline["jobs"]), "--out", str(agg), "--serving", str(serving)]) == 0
    ep = pd.read_csv(agg / "episodes.csv")
    assert ep["sim_fitted_wall_sec"].notna().all() and (ep["sim_fitted_wall_sec"] > 0).all()
    assert json.loads((agg / "summary.json").read_text())["serving_profiles"]["fitted"]
