"""``aggregate`` over a synthetic sweep: every table, its columns and its values.

The sweep (tests/synth.py) has all 12 pilot conditions (both baselines, three
strategies x three scopes, the same-model control) x 2 tasks x 2 rounds, plus
edge cases in round 1: an environment failure without an agent dir, a crashed
``running`` episode, a missing result.json, a zero-dispatch episode and a
billing error. Expected values are recomputed from what the generator wrote.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import GEMMA, ORCH, C, EpisodeBuilder, T, make_sweep, write_job, write_trial  # noqa: E402

from subagents_bench import analysis  # noqa: E402
from subagents_bench.analysis.aggregate import aggregate  # noqa: E402
from subagents_bench.analysis.stats import mcnemar_exact, wilson_ci  # noqa: E402
from subagents_bench.analysis.tables import condition_table  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCHEMES = ["sail_balanced", "sail_asap", "sail_flex", "prime"]
PROFILES = ["measured", "fitted", "roofline_h100"]
ROLES = ["orch", "sub", "arb", "deb"]
TABLES = ["episodes", "calls", "tools", "dispatches", "spawns", "arbitrations", "steps", "conditions",
          "conditions_by_difficulty", "paired", "calibration"]


def _configs(tmp: Path) -> dict:
    """Copies of the repo configs (pricing without its reference include, so values are pinned)
    plus a fixture models.yaml (synthetic parameter counts)."""
    tmp.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "configs" / "pricing.yaml", tmp / "pricing.yaml")
    shutil.copy(REPO / "configs" / "serving.yaml", tmp / "serving.yaml")
    (tmp / "models.yaml").write_text(
        "models:\n"
        f"  {ORCH}: {{active_params_b: 16.0, active_params_prefill_b: 8.0, active_params_decode_b: 16.0, bytes_per_param: 0.8}}\n"
        f"  {GEMMA}: {{active_params_b: 30.0, bytes_per_param: 1.0}}\n"
    )
    return {"pricing_path": tmp / "pricing.yaml", "serving_path": tmp / "serving.yaml",
            "models_path": tmp / "models.yaml"}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweep")
    sweep = make_sweep(root / "jobs")
    res = aggregate([root / "jobs"], root / "analysis", **_configs(root / "cfg"))
    return sweep, res, root / "analysis"


def _row(eps: pd.DataFrame, trial) -> pd.Series:
    r = eps[(eps["condition"] == trial.condition) & (eps["trial_name"] == trial.trial_name)]
    assert len(r) == 1, (trial.condition, trial.trial_name)
    return r.iloc[0]


def test_every_table_written_with_expected_columns(run):
    sweep, res, out = run
    for name in TABLES:
        assert (out / f"{name}.csv").is_file(), name
        df = pd.read_csv(out / f"{name}.csv")
        assert len(df) == len(res.tables[name]) > 0, name
    eps = res.tables["episodes"]
    required = (
        ["sweep", "job_name", "round", "condition", "trial_name", "task_name", "episode_id",
         "topology", "strategy", "scope", "orchestrator_model", "subagent_model", "arbiter_model", "max_concurrent",
         "n_candidates", "orchestrator_steps_budget", "difficulty", "category", "expert_time_min", "junior_time_min",
         "reward", "resolved", "harbor_exception", "exit_reason", "valid",
         "env_setup_sec", "agent_setup_sec", "agent_exec_sec", "verifier_sec", "trial_total_sec",
         "wall_clock_sec", "orchestrator_steps", "spawn_calls", "dispatches", "redispatches", "arbitrations",
         "instruction_following", "dispatch_submitted_frac", "n_files_modified", "total_tokens",
         "cost_runtime_total", "flops_total", "backoff_sec", "n_retries", "n_command_timeouts",
         "cp_total_sec", "cp_orch_llm_sec", "cp_sub_llm_sec", "cp_arb_llm_sec", "cp_tool_sec", "overhead_sec",
         "sub_busy_sec", "parallel_speedup"]
        + [f"{r}_{k}" for r in ROLES for k in ("calls", "prompt_tokens", "cached_tokens", "uncached_prompt_tokens",
                                                "completion_tokens", "reasoning_tokens")]
        + [f"cost_{s}_{r}" for s in SCHEMES for r in ROLES + ["total"]]
        + [f"flops_{r}" for r in ROLES] + [f"llm_sec_{r}" for r in ROLES] + [f"tool_sec_{r}" for r in ROLES]
        + [f"sim_{p}_wall_sec" for p in PROFILES]
    )
    assert [c for c in required if c not in eps.columns] == []
    assert len(eps) == len(sweep.trials) == 53
    assert json.loads((out / "summary.json").read_text())["n_trials"] == 53
    # Bulky lists are dropped from calls.csv; chunk offsets are summarized.
    calls = res.tables["calls"]
    for c in ("msg_hashes", "msg_chars", "chunk_offsets_ms", "attempts", "response_headers", "usage_raw"):
        assert c not in calls.columns
    for c in ["chunk_offsets_n", "chunk_gap_ms_median", "attempt_statuses", "flops"] + [f"cost_{s}" for s in SCHEMES]:
        assert c in calls.columns
    for name in ("calls", "tools", "dispatches", "spawns", "arbitrations", "steps"):
        assert {"condition", "task_name", "round", "episode_id"} <= set(res.tables[name].columns)


def test_keys_validity_and_outcomes_match_generator(run):
    sweep, res, _ = run
    eps = res.tables["episodes"]
    for t in sweep.trials:
        r = _row(eps, t)
        assert r["round"] == t.round and r["job_name"] == t.job_name and r["task_name"] == t.task_name
        assert r["sweep"] == "synth"
        assert bool(r["valid"]) == t.expected_valid, t.edge
        if t.expected_resolved is None:
            assert pd.isna(r["resolved"])
        else:
            assert int(r["resolved"]) == t.expected_resolved, t.edge
    meta = eps.set_index("trial_name")
    fixgit = eps[eps["task_name"] == "fix-git"].iloc[0]
    assert (fixgit["difficulty"], fixgit["category"], fixgit["expert_time_min"]) == ("easy", "software-engineering", 5.0)
    assert set(eps.loc[eps["task_name"] == "crack-7z-hash", "difficulty"]) == {"medium"}
    assert len(meta) == len(eps)


def test_edge_cases(run):
    sweep, res, _ = run
    eps = res.tables["episodes"]
    by_edge = {t.edge: _row(eps, t) for t in sweep.trials if t.edge}

    env = by_edge["env_failure"]
    assert env["invalid_reason"] == "no_episode" and env["episode_status"] == "missing"
    assert env["harbor_exception"] == "EnvironmentStartTimeoutError"
    # Design comes from the manifest's resolved config; measurements are unknown, not 0.
    assert (env["topology"], env["strategy"], env["scope"], env["subagent_model"]) == ("subagents", "parallel", "easy", GEMMA)
    for c in ("total_tokens", "orch_calls", "cost_sail_balanced_total", "cp_total_sec", "wall_clock_sec",
              "agent_exec_sec", "env_setup_sec"):
        assert pd.isna(env[c]), c

    crash = by_edge["running"]
    assert crash["episode_status"] == "running" and bool(crash["valid"]) and crash["resolved"] == 0
    assert pd.isna(crash["exit_reason"]) and pd.isna(crash["wall_clock_sec"]) and pd.isna(crash["overhead_sec"])
    assert crash["harbor_exception"] == "RuntimeError"
    # Counts come from the records because the crashed episode.json has no aggregates.
    assert (crash["orchestrator_steps"], crash["spawn_calls"], crash["dispatches"]) == (1, 1, 2)
    assert crash["cp_total_sec"] == pytest.approx(2.0 + 1.0 + 0.3 + 1.5)  # sequential group: 1.3 + 1.5

    nores = by_edge["no_result"]
    assert nores["invalid_reason"] == "no_result" and pd.isna(nores["reward"]) and pd.isna(nores["trial_total_sec"])
    assert nores["task_name"] == "overfull-hbox"  # from episode.json

    zero = by_edge["zero_dispatch"]
    assert bool(zero["valid"]) and zero["resolved"] == 1
    assert (zero["dispatches"], zero["spawn_calls"], zero["sub_calls"]) == (0, 0, 0)
    assert zero["cost_sail_balanced_sub"] == 0.0 and zero["cp_group_sec"] == 0.0
    assert pd.isna(zero["parallel_speedup"]) and pd.isna(zero["dispatch_submitted_frac"])
    assert pd.isna(zero["instruction_following"])

    bill = by_edge["billing"]
    assert bill["invalid_reason"] == "billing_error" and bill["exit_reason"] == "billing_error"
    assert bill["n_failed_calls"] == 1


def test_tokens_costs_flops_and_time_per_episode(run):
    sweep, res, _ = run
    eps = res.tables["episodes"]
    t = sweep.get("gemma31b_arbiter_hard", "fix-git", 1)
    r = _row(eps, t)
    calls = t.builder.calls
    role = {"orch": "orchestrator", "sub": "subagent", "arb": "arbiter", "deb": "debate"}
    for short, long in role.items():
        cs = [c for c in calls if c["role"] == long]
        assert r[f"{short}_calls"] == len(cs)
        assert r[f"{short}_prompt_tokens"] == sum(c["prompt_tokens"] for c in cs)
        assert r[f"{short}_cached_tokens"] == sum(c["cached_tokens"] for c in cs)
        assert r[f"{short}_completion_tokens"] == sum(c["completion_tokens"] for c in cs)
        assert r[f"llm_sec_{short}"] == pytest.approx(sum(c["latency_sec"] for c in cs))
        flops = sum(2e9 * ((8.0 if c["model"] == ORCH else 30.0) * c["prompt_tokens"]
                           + (16.0 if c["model"] == ORCH else 30.0) * c["completion_tokens"]) for c in cs)
        assert r[f"flops_{short}"] == pytest.approx(flops)
    assert r["total_tokens"] == sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls)
    assert pd.isna(r["sub_reasoning_tokens"])  # Gemma reports none: unknown, not 0
    assert r["orch_reasoning_tokens"] == sum(c["reasoning_tokens"] for c in calls if c["role"] == "orchestrator")
    assert (r["arbitrations"], r["dispatches"], r["instruction_following"]) == (1, 3, 0.5)
    valid = eps[eps["valid"] & (eps["episode_status"] == "done")]
    # Runtime prices of the synthetic data are the sail_balanced list prices.
    np.testing.assert_allclose(valid["cost_runtime_total"], valid["cost_sail_balanced_total"], rtol=1e-12)
    # Gemma has no Prime price -> NaN totals for Gemma conditions; DeepSeek-only conditions are priced via alias.
    gem = eps[(eps["subagent_model"] == GEMMA) & (eps["sub_calls"] > 0)]
    assert gem["cost_prime_total"].isna().all() and gem["cost_prime_orch"].notna().all()
    single = eps[(eps["topology"] == "single") & (eps["episode_status"] == "done")]
    assert single["cost_prime_total"].notna().all() and (single["cost_prime_total"] > 0).all()


def test_critical_path_and_simulation_columns(run):
    _, res, _ = run
    eps = res.tables["episodes"]
    have = eps[eps["episode_status"] != "missing"]
    np.testing.assert_allclose(have["sim_measured_wall_sec"], have["cp_total_sec"], atol=1e-9)
    parts = have[["cp_orch_llm_sec", "cp_sub_llm_sec", "cp_arb_llm_sec", "cp_tool_sec"]].sum(axis=1)
    np.testing.assert_allclose(parts, have["cp_total_sec"], atol=1e-9)
    done = have[have["episode_status"] == "done"]
    np.testing.assert_allclose(done["overhead_sec"], done["wall_clock_sec"] - done["cp_total_sec"], atol=1e-9)
    assert (done["overhead_sec"] > 0).all()  # gaps and spawn overhead are not on the chain model
    assert have["sim_fitted_wall_sec"].notna().all() and have["sim_roofline_h100_wall_sec"].notna().all()
    # Sequential groups have speedup ~1 (only dispatch-internal gaps), parallel > 1.
    seq = done[(done["strategy"] == "sequential") & (done["dispatches"] > 0)]
    par = done[(done["strategy"] == "parallel") & (done["dispatches"] > 0)]
    np.testing.assert_allclose(seq["parallel_speedup"], 1.0, atol=1e-9)
    assert (par["parallel_speedup"] > 1.5).all()
    # harbor phase timings from result.json timestamps (synthetic layout: setup 7 s, verifier 5.6 s).
    np.testing.assert_allclose(done["env_setup_sec"].dropna(), 7.0, atol=1e-5)
    np.testing.assert_allclose(done["verifier_sec"].dropna(), 5.6, atol=1e-5)


def test_conditions_table(run):
    sweep, res, out = run
    cond = res.tables["conditions"].set_index("condition")
    assert list(cond.index[:2]) == ["single_5step", "single_matched"]  # baselines first
    assert len(cond) == 12
    for name in sweep.conditions:
        ts = [t for t in sweep.trials if t.condition == name]
        valid = [t for t in ts if t.expected_valid]
        k = sum(t.expected_resolved for t in valid)
        row = cond.loc[name]
        assert (row["n"], row["n_valid"], row["n_resolved"]) == (len(ts), len(valid), k)
        assert row["resolve_rate"] == pytest.approx(k / len(valid))
        lo, hi = wilson_ci(k, len(valid))
        assert (row["resolve_ci_lo"], row["resolve_ci_hi"]) == (pytest.approx(lo), pytest.approx(hi))
    eps = res.tables["episodes"]
    v = eps[eps["valid"] & (eps["condition"] == "gemma31b_parallel_hard")]
    row = cond.loc["gemma31b_parallel_hard"]
    assert row["wall_clock_sec_mean"] == pytest.approx(v["wall_clock_sec"].mean())
    assert row["wall_clock_sec_ci_lo"] <= row["wall_clock_sec_mean"] <= row["wall_clock_sec_ci_hi"]
    assert row["cost_per_solve_sail_balanced"] == pytest.approx(
        v["cost_sail_balanced_total"].sum() / v["resolved"].sum())
    assert row["time_per_solve_sec"] == pytest.approx(v["wall_clock_sec"].sum() / v["resolved"].sum())
    for m in (["wall_clock_sec", "total_tokens", "dispatches", "orch_prompt_tokens", "sub_completion_tokens",
               "flops_total", "cp_total_sec"] + [f"sim_{p}_wall_sec" for p in PROFILES]
              + [f"cost_{s}_total" for s in SCHEMES + ["runtime"]]):
        for suf in ("_mean", "_ci_lo", "_ci_hi"):
            assert m + suf in cond.columns, m + suf
    for s in SCHEMES + ["runtime"]:
        assert f"cost_per_solve_{s}" in cond.columns
    assert pd.isna(cond.loc["gemma31b_parallel_hard", "cost_prime_total_mean"])  # unpriced, never 0
    assert cond.loc["single_5step", "strategy"] is None or pd.isna(cond.loc["single_5step", "strategy"])
    # Bootstrap is seeded: recomputing the table gives identical CIs.
    again = condition_table(res.tables["episodes"], ["condition"], SCHEMES, ["single_5step", "single_matched"])
    pd.testing.assert_frame_equal(again, res.tables["conditions"])
    by_diff = pd.read_csv(out / "conditions_by_difficulty.csv")
    assert set(zip(by_diff["condition"], by_diff["difficulty"])) >= {(c, d) for c in sweep.conditions
                                                                      for d in ("easy", "medium")}
    assert by_diff.loc[(by_diff["condition"] == "single_5step") & (by_diff["difficulty"] == "easy"), "n"].item() == 2


def test_paired_table(run):
    sweep, res, _ = run
    paired = res.tables["paired"]
    assert set(paired["baseline"]) == {"single_5step", "single_matched"}
    # Every other condition vs each baseline; single_5step vs single_matched is the mirror of a listed pair.
    assert len(paired) == 11 + 10
    assert not ((paired["condition"] == "single_5step") & (paired["baseline"] == "single_matched")).any()
    for _, p in paired.iterrows():
        cond = {(t.task_name, t.round): t.expected_resolved for t in sweep.trials
                if t.condition == p["condition"] and t.expected_valid}
        base = {(t.task_name, t.round): t.expected_resolved for t in sweep.trials
                if t.condition == p["baseline"] and t.expected_valid}
        keys = set(cond) & set(base)
        only_c = sum(1 for k in keys if cond[k] and not base[k])
        only_b = sum(1 for k in keys if base[k] and not cond[k])
        assert p["n_pairs"] == len(keys) == 4
        assert (p["only_cond"], p["only_base"]) == (only_c, only_b)
        assert p["both"] == sum(1 for k in keys if cond[k] and base[k])
        assert p["delta_resolve"] == pytest.approx((only_c - only_b) / len(keys))
        assert p["mcnemar_p"] == pytest.approx(mcnemar_exact(only_c, only_b))


def test_calibration_table(run):
    sweep, res, _ = run
    cal = res.tables["calibration"].set_index("model")
    assert set(cal.index) == {ORCH, GEMMA}
    calls = [c for t in sweep.trials if t.builder for c in t.builder.calls if c["error"] is None]
    for model in (ORCH, GEMMA):
        cs = [c for c in calls if c["model"] == model]
        row = cal.loc[model]
        assert (row["provider"], row["completion_window"], row["n_calls"]) == ("sail", "balanced", len(cs))
        assert row["ttft_median"] == pytest.approx(np.median([c["ttft_sec"] for c in cs]))
        assert row["ttft_p90"] == pytest.approx(np.quantile([c["ttft_sec"] for c in cs], 0.9))
        dec = [c["completion_tokens"] / c["decode_sec"] for c in cs if c["decode_sec"] > 0 and c["completion_tokens"]]
        assert row["decode_tok_s_median"] == pytest.approx(np.median(dec))
        x = np.array([c["prompt_tokens"] / 1000 for c in cs])
        b, a = np.polyfit(x, np.array([c["ttft_sec"] for c in cs]), 1)
        assert (row["ttft_fit_a"], row["ttft_fit_b_per_1k_prompt_tokens"]) == (pytest.approx(a), pytest.approx(b))
        assert row["latency_median"] == pytest.approx(np.median([c["latency_sec"] for c in cs]))


def test_record_tables(run):
    sweep, res, _ = run
    builders = [t.builder for t in sweep.trials if t.builder]
    for name, attr in (("calls", "calls"), ("tools", "tools"), ("dispatches", "dispatches"), ("spawns", "spawns"),
                       ("arbitrations", "arbitrations"), ("steps", "steps")):
        assert len(res.tables[name]) == sum(len(getattr(b, attr)) for b in builders), name
    arbs = res.tables["arbitrations"]
    assert set(arbs["n_rounds"]) == {1, 2}
    two = arbs[arbs["n_rounds"] == 2].iloc[0]
    assert (two["round_winners"], two["round_confidences"], two["first_confidence"]) == ("0;0", "0.6;0.9", 0.6)
    assert two["n_debate_calls"] == 3
    calls = res.tables["calls"]
    np.testing.assert_allclose(calls.groupby("episode_id")["cost_sail_balanced"].sum().sort_index(),
                               res.tables["episodes"].dropna(subset=["episode_id"]).set_index("episode_id")
                               ["cost_sail_balanced_total"].sort_index(), rtol=1e-12)
    assert (calls["chunk_offsets_n"].isin([0, 6])).all()


def test_summary_json(run):
    _, res, out = run
    s = json.loads((out / "summary.json").read_text())
    assert s["invalid_reason_counts"] == {"billing_error": 1, "no_episode": 1, "no_result": 1}
    assert s["episode_status_counts"] == {"done": 51, "missing": 1, "running": 1}
    assert s["conditions"][:2] == ["single_5step", "single_matched"] and len(s["conditions"]) == 12
    assert s["rounds"] == [1, 2]
    assert s["pricing"]["models_without_price"]["prime"] == [GEMMA]
    assert "deepseek/deepseek-v4.1-flash" in s["pricing"]["schemes"]["prime"]["cached_price_missing_charged_at_input"]
    assert s["latency_checks"]["measured_replay_max_abs_diff_sec"] == pytest.approx(0.0, abs=1e-9)
    assert s["serving_profiles"]["roofline_h100"]["theoretical"] is True
    assert any("pricing_reference.yaml" in w for w in s["warnings"])
    assert s["tasks_meta"]["tasks_without_meta"] == []


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_no_manifest_falls_back_to_job_name(tmp_path):
    conds = {"single_5step": {"topology": "single", "steps_budget": 5},
             "gemma31b_parallel_easy": {"topology": "subagents", "strategy": "parallel", "scope": "easy",
                                        "sub_model": GEMMA}}
    sweep = make_sweep(tmp_path / "jobs", sweep="pilot", conditions=conds, rounds=(3,), with_manifest=False,
                       with_tasks_meta=False)
    res = aggregate([tmp_path / "jobs"], None, **_configs(tmp_path / "cfg"))
    eps = res.tables["episodes"]
    assert set(eps["round"]) == {3} and set(eps["sweep"]) == {"pilot"}
    assert set(eps["condition"]) == set(conds)
    env = eps[eps["episode_status"] == "missing"].iloc[0]
    # Design read from the sweep's materialized <jobs_dir>/configs/<condition>.yaml.
    assert env["condition"] == "gemma31b_parallel_easy"
    assert (env["topology"], env["strategy"], env["scope"]) == ("subagents", "parallel", "easy")
    assert eps["difficulty"].isna().all()  # no tasks_meta.json
    (tmp_path / "jobs" / "configs" / "gemma31b_parallel_easy.yaml").unlink()
    eps2 = aggregate([tmp_path / "jobs"], None, **_configs(tmp_path / "cfg")).tables["episodes"]
    assert pd.isna(eps2.loc[eps2["episode_status"] == "missing", "topology"].iloc[0])  # nothing to read: unknown
    cbd = res.tables["conditions_by_difficulty"]
    assert set(cbd["difficulty"]) == {"unknown"}
    assert len(sweep.trials) == len(eps)


def test_single_topology_only_and_missing_configs(tmp_path):
    conds = {"single_5step": {"topology": "single", "steps_budget": 5},
             "single_matched": {"topology": "single", "steps_budget": 40}}
    make_sweep(tmp_path / "jobs", conditions=conds, edge_cases=False)
    res = aggregate([tmp_path / "jobs"], tmp_path / "out", pricing_path=tmp_path / "none.yaml",
                    serving_path=None, models_path=None)
    eps = res.tables["episodes"]
    assert not any(c.startswith("cost_sail") for c in eps.columns)
    assert "cost_runtime_total" in eps.columns and eps["flops_total"].isna().all()
    assert [c for c in eps.columns if c.startswith("sim_")] == ["sim_measured_wall_sec"]
    assert eps["parallel_speedup"].isna().all() and (eps["dispatches"] == 0).all()
    for name in ("dispatches", "arbitrations", "spawns"):
        df = pd.read_csv(tmp_path / "out" / f"{name}.csv")
        assert len(df) == 0 and "episode_id" in df.columns
    assert len(res.tables["paired"]) == 1  # single_matched vs single_5step
    assert any("not found" in w for w in res.warnings)


def test_bare_episode_dir_and_empty_input(tmp_path):
    b = EpisodeBuilder(episode_id="local1", condition="local_cond", topology="single", t0=1000.0)
    b.step(C(1.0), [T("bash", 0.5)]).step(C(0.5), finish=True)
    b.finish()
    b.episode["reward"], b.episode["resolved"] = 1.0, True  # the local runner fills these
    b.write(tmp_path / "runs" / "local_cond" / "fix-git" / "local1")
    res = aggregate([tmp_path / "runs"], None, **_configs(tmp_path / "cfg"))
    r = res.tables["episodes"].iloc[0]
    assert (r["condition"], r["task_name"], r["resolved"], bool(r["valid"])) == ("local_cond", "fix-git", 1, True)
    assert pd.isna(r["job_name"]) and pd.isna(r["round"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no harbor trials"):
        aggregate([tmp_path / "empty"], None)
    with pytest.raises(FileNotFoundError):
        aggregate([tmp_path / "missing"], None)


def test_trial_without_result_or_agent(tmp_path):
    job = tmp_path / "jobs" / "x__r1__cond"
    write_job(job, 1)
    # A trial that started (config.json) but has neither agent/ nor result.json.
    write_trial(job, "fix-git", "fix-git__abc", None, reward=None, write_result=False)
    res = aggregate([tmp_path / "jobs"], None, **_configs(tmp_path / "cfg"))
    r = res.tables["episodes"].iloc[0]
    assert (r["condition"], r["round"], r["invalid_reason"], r["episode_status"]) == ("cond", 1, "no_episode", "missing")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    analysis.register_cli(sub)
    args = parser.parse_args(argv)
    return args.func(args)


def test_cli_aggregate_and_calibrate(tmp_path, capsys):
    make_sweep(tmp_path / "jobs", conditions={"gemma31b_parallel_hard": {
        "topology": "subagents", "strategy": "parallel", "scope": "hard", "sub_model": GEMMA}}, edge_cases=False)
    cfg = _configs(tmp_path / "cfg")
    rc = _cli(["aggregate", str(tmp_path / "jobs"), "--out", str(tmp_path / "out"), "--pricing",
               str(cfg["pricing_path"]), "--serving", str(cfg["serving_path"]), "--models", str(cfg["models_path"])])
    assert rc == 0 and (tmp_path / "out" / "episodes.csv").is_file()
    assert "aggregated 4 trials" in capsys.readouterr().out
    # calibrate -> YAML usable as the fitted profile's params_file, giving the same sims as the in-sample fit.
    assert _cli(["calibrate", str(tmp_path / "jobs"), "--out", str(tmp_path / "cfg" / "fit.yaml"),
                 "--csv", str(tmp_path / "cal.csv")]) == 0
    doc = yaml.safe_load((tmp_path / "cfg" / "fit.yaml").read_text())
    assert {p["model"] for p in doc["params"]} == {ORCH, GEMMA}
    assert all(p["ttft_a_sec"] is not None for p in doc["params"])
    serving = yaml.safe_load(cfg["serving_path"].read_text())
    serving["profiles"]["fitted"]["params_file"] = "fit.yaml"
    cfg["serving_path"].write_text(yaml.safe_dump(serving))
    res = aggregate([tmp_path / "jobs"], None, **cfg)
    before = pd.read_csv(tmp_path / "out" / "episodes.csv")
    np.testing.assert_allclose(res.tables["episodes"]["sim_fitted_wall_sec"], before["sim_fitted_wall_sec"], rtol=1e-9)
    assert res.summary["serving_profiles"]["fitted"]["source"].endswith("fit.yaml")
    assert math.isfinite(res.tables["episodes"]["sim_fitted_wall_sec"].iloc[0])
