"""Analysis, figures and scripts/analyze.sh on jobs dirs of real episodes: the lean runtime
runs with scripted fake models and each episode is wrapped in a harbor 0.24.0-like trial."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fakes import ORCH, SUB, FakeFactory, Reply, make_config, records, run_local, tc
from subagents_bench.analysis.__main__ import main as analyze
from subagents_bench.analysis.aggregate import condition_order
from subagents_bench.analysis.load import load_trial
from subagents_bench.analysis.stats import bootstrap_mean, wilson
from subagents_bench.analysis.validity import invalid_reason
from subagents_bench.environment import AgentStopped
from subagents_bench.llm import ProviderError
from subagents_bench.plots.__main__ import main as plot
from subagents_bench.plots.figures import FIGURES

ROOT = Path(__file__).resolve().parents[1]
PRICES = {ORCH: {"in": 1.0, "cached": 0.1, "out": 2.0}, SUB: {"in": 0.1, "cached": 0.01, "out": 0.2}}


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def orch_reply(*calls, text=""):
    return Reply(text, list(calls), prompt_tokens=1000, cached_tokens=600, completion_tokens=80)


def sub_reply(*calls, text=""):
    return Reply(text, list(calls), prompt_tokens=300, cached_tokens=100, completion_tokens=40)


def models(n_instructions=2, then=None, judges=(), sub_error=False):
    """Scripts: the orchestrator spawns once (unless n_instructions is 0), then ``then`` or
    finish; subagents run one command and submit; arbiter judges and debate as given."""
    judges = list(judges)

    def orch(system, messages, tools):
        if system.startswith("You are an impartial arbiter"):
            return judges.pop(0)
        if len(messages) == 1 and n_instructions:
            return orch_reply(tc("spawn_subagents", instructions=[f"part {i}" for i in range(n_instructions)]))
        if len(messages) == 1:
            return orch_reply(tc("bash", command="sleep 0.05; echo v1 > app.py"))
        return then or orch_reply(tc("finish", summary="ok"), text="done")

    def sub(system, messages, tools):
        if system.startswith("You are candidate"):
            return sub_reply(text="revised report")
        if sub_error:
            return ProviderError("sail sub-model: giving up after 5 attempts")
        if len(messages) == 1:
            return sub_reply(tc("bash", command="sleep 0.05; echo v2 > app.py"))
        return sub_reply(tc("submit", report="changed app.py"))

    return {ORCH: orch, SUB: sub}


def judgement(winner, confidence, agreement):
    return Reply("", [tc("submit_judgement", winner=winner, confidence=confidence, agreement=agreement,
                         rationale="r")], prompt_tokens=2000, completion_tokens=100)


def trial(jobs: Path, job: str, task: str, cfg, scripts, reward=None, exception=None, rewards=None) -> Path:
    """One real episode in ``<jobs>/<job>/<task>__x/agent`` plus harbor's config.json and result.json
    (``rewards``: the grader's other fields, as SWE-sweep's verifier writes them)."""
    d = jobs / job / f"{task}__x"
    work = jobs.parent / "work" / job / task
    work.mkdir(parents=True)
    (work / "app.py").write_text("v0\n")
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"task": {"path": task}, "trial_name": d.name}))
    t0 = time.time()
    exit_reason, rec = run_local(cfg, FakeFactory(scripts, latency=0.02), work, d / "agent", task_name=task,
                                 trial_name=d.name)
    if exit_reason == "cancelled":
        rec.update(cancel_kill={"ok": True, "n_killed": 1, "error": None})
    t1 = time.time()
    exc = {"exception_type": exception, "exception_message": "x", "exception_traceback": "", "occurred_at": iso(t1)}
    (d / "result.json").write_text(json.dumps({
        "task_name": task, "trial_name": d.name, "exception_info": exc if exception else None,
        "verifier_result": None if reward is None else {"rewards": {"reward": reward, **(rewards or {})}},
        "started_at": iso(t0 - 2), "finished_at": iso(t1 + 2),
        "environment_setup": {"started_at": iso(t0 - 2), "finished_at": iso(t0 - 1)},
        "agent_setup": {"started_at": iso(t0 - 1), "finished_at": iso(t0)},
        "agent_execution": {"started_at": iso(t0), "finished_at": iso(t1)},
        "verifier": None if reward is None else {"started_at": iso(t1 + 0.5), "finished_at": iso(t1 + 1)},
    }))
    return d


@pytest.fixture(scope="module")
def jobs(tmp_path_factory):
    """``smk``: single, parallel (+ an agent timeout), sequential (+ a provider failure) and
    arbiter with copy isolation (+ a debate); ``extra``: a condition that never delegates and a
    trial whose environment never started."""
    root = tmp_path_factory.mktemp("jobs")
    smk, extra = root / "jobs" / "smk", root / "jobs" / "extra"

    def cond(name, **kw):
        return make_config(name=name, **kw)

    single = cond("single_5step", topology="single")
    trial(smk, "smk__r1__single_5step", "task-a", single, models(0), reward=1.0)
    trial(smk, "smk__r1__single_5step", "task-b", single, models(0), reward=0.0)
    par = cond("gemma31b_parallel_hard")
    trial(smk, "smk__r1__gemma31b_parallel_hard", "task-a", par, models(), reward=1.0)
    trial(smk, "smk__r1__gemma31b_parallel_hard", "task-b", par, models(then=AgentStopped("harbor cancel")),
          reward=0.0, exception="AgentTimeoutError")
    seq = cond("gemma31b_sequential_medium", subagent={"strategy": "sequential", "scope": "medium"})
    trial(smk, "smk__r1__gemma31b_sequential_medium", "task-a", seq, models(), reward=1.0)
    trial(smk, "smk__r1__gemma31b_sequential_medium", "task-b", seq, models(sub_error=True),
          exception="ProviderUnavailableError")
    arb = cond("gemma31b_arbiter_hard", subagent={"strategy": "arbiter"})
    trial(smk, "smk__r1__gemma31b_arbiter_hard", "task-a", arb,
          models(1, judges=[judgement(0, 0.5, False), judgement(1, 0.9, True)]), reward=1.0)
    cut_off = Reply("", [], finish_reason="length", prompt_tokens=2000, completion_tokens=100)  # judge retry
    trial(smk, "smk__r1__gemma31b_arbiter_hard", "task-b", arb,
          models(1, judges=[cut_off, judgement(2, 0.95, True)]), reward=0.0)
    trial(smk, "_partial/smk__r1__single_5step.1", "task-a", single, models(0), reward=1.0)  # moved aside: skipped
    easy = cond("gemma31b_parallel_easy", subagent={"scope": "easy"})
    trial(extra, "extra__r1__gemma31b_parallel_easy", "task-a", easy, models(0), reward=0.0)
    no_env = extra / "extra__r1__gemma31b_parallel_easy" / "task-b__env"
    no_env.mkdir()
    (no_env / "config.json").write_text(json.dumps({"task": {"path": "task-b"}, "trial_name": no_env.name}))
    exc = {"exception_type": "EnvironmentStartTimeoutError", "occurred_at": iso(time.time())}
    (no_env / "result.json").write_text(json.dumps({"task_name": "task-b", "trial_name": no_env.name,
                                                    "exception_info": exc, "verifier_result": None}))
    pricing = root / "pricing.yaml"
    pricing.write_text(yaml.safe_dump({"fake": PRICES}))
    return root, smk, extra, pricing


@pytest.fixture(scope="module")
def analysis(jobs):
    root, smk, _, pricing = jobs
    out = root / "analysis" / "smk"
    assert analyze([str(smk), "--out", str(out), "--pricing", str(pricing), "--scheme", "fake"]) == 0
    return out, {n: pd.read_csv(out / f"{n}.csv") for n in ("episodes", "conditions", "calls", "dispatches")}


def call_cost(c):
    p = PRICES[c["model"]]
    return ((c["prompt_tokens"] - c["cached_tokens"]) * p["in"] + c["cached_tokens"] * p["cached"]
            + c["completion_tokens"] * p["out"]) / 1e6


def test_episode_rows_equal_the_records(analysis):
    out, t = analysis
    eps = t["episodes"].set_index("trial_name")
    assert len(eps) == 8 and set(eps["sweep"]) == {"smk"} and set(eps["round"]) == {1}
    for name, row in eps.iterrows():
        r = records(Path(row["trial_dir"]) / "agent")
        ep, calls = r["episode"], r["calls"]
        for role, short in (("orchestrator", "orch"), ("subagent", "sub"), ("arbiter", "arb"), ("debate", "deb")):
            for kind in ("prompt", "cached", "completion"):
                assert row[f"{short}_{kind}_tokens"] == ep["tokens_by_role"][role][kind]
            assert row[f"cost_{short}"] == pytest.approx(sum(call_cost(c) for c in calls if c["role"] == role))
        assert row["cost_total"] == pytest.approx(sum(call_cost(c) for c in calls))
        assert (row["steps"], row["spawns"], row["dispatches"], row["arbitrations"]) == (
            ep["orchestrator_steps"], ep["spawn_calls"], ep["dispatches"], ep["arbitrations"])
        parts = sum(row[f"cp_{p}_sec"] for p in ("orch_llm", "sub_llm", "judge_llm", "tool"))
        assert row["cp_total_sec"] == pytest.approx(parts) and 0 < row["cp_total_sec"] <= ep["wall_clock_sec"] + 1e-3
        assert row["overhead_sec"] == pytest.approx(ep["wall_clock_sec"] - row["cp_total_sec"])
    assert len(t["dispatches"]) == eps["dispatches"].sum()
    assert set(t["dispatches"]["status"]) == {"submitted", "error", "cancelled"}  # the provider failure stopped both
    calls = t["calls"]
    assert len(calls) == sum(len(records(Path(d) / "agent")["calls"]) for d in eps["trial_dir"])
    assert calls["cost"].sum() == pytest.approx(eps["cost_total"].sum())


def test_validity_of_timeout_and_provider_failure(analysis):
    eps = analysis[1]["episodes"].set_index(["condition", "task_name"])
    timeout = eps.loc[("gemma31b_parallel_hard", "task-b")]
    assert (timeout["valid"], timeout["resolved"], timeout["exit_reason"], timeout["harbor_exception"]) == (
        True, 0, "cancelled", "AgentTimeoutError")
    failed = eps.loc[("gemma31b_sequential_medium", "task-b")]
    assert (failed["valid"], failed["invalid_reason"], failed["exit_reason"]) == (False, "provider_error",
                                                                                  "provider_error")
    assert eps["valid"].sum() == 7 and eps.loc[("gemma31b_arbiter_hard", "task-a"), "resolved"] == 1


@pytest.mark.parametrize("episode, result, reason", [
    ({"exit_reason": "finished"}, None, "no_result"),
    (None, {"exception_info": {"exception_type": "EnvironmentStartTimeoutError"}}, "no_episode"),
    ({"exit_reason": "billing_error"}, {"exception_info": {"exception_type": "BillingError"}}, "billing_error"),
    ({"exit_reason": "cancelled"}, {"exception_info": {"exception_type": "CancelledError"},
                                    "agent_execution": {"started_at": "2026-10-12T16:00:00Z"}}, "harbor_exception"),
    ({"exit_reason": "finished"}, {"exception_info": {"exception_type": "VerifierTimeoutError", "occurred_at":
     "2026-10-12T15:00:00Z"}, "agent_execution": {"started_at": "2026-10-12T16:00:00Z"}}, "harbor_exception"),
    ({"exit_reason": "finished", "n_provider_errors": 1}, {"verifier_result": {"rewards": {"reward": 1}}},
     "provider_error"),
    ({"exit_reason": "error"}, {"exception_info": {"exception_type": "RuntimeError"},
                                "agent_execution": {"started_at": "2026-10-12T16:00:00Z"}}, "agent_crash"),
    ({"exit_reason": "cancelled"}, {"exception_info": {"exception_type": "AgentTimeoutError"},
                                    "agent_execution": {"started_at": "2026-10-12T16:00:00Z"},
                                    "verifier": {"started_at": "2026-10-12T16:30:00Z"}}, None),
    ({"exit_reason": "finished"}, {"verifier_result": {"rewards": {"reward": 0.0, "grader_failed": 1.0}},
                                   "verifier": {"started_at": "2026-10-12T16:30:00Z"}}, "grader_failed"),
    # after the verifier started: a separate verifier's failure is infrastructure, a shared one's an outcome
    ({"exit_reason": "finished"}, {"exception_info": {"exception_type": "EnvironmentStartTimeoutError"},
                                   "verifier_environment_mode": "separate",
                                   "verifier": {"started_at": "2026-10-12T16:30:00Z"}}, "verifier_failed"),
    ({"exit_reason": "finished"}, {"exception_info": {"exception_type": "RewardFileNotFoundError"},
                                   "verifier_environment_mode": "shared",
                                   "verifier": {"started_at": "2026-10-12T16:30:00Z"}}, None),
    ({"exit_reason": "cancelled"}, {"exception_info": {"exception_type": "AgentTimeoutError"},
                                    "verifier_environment_mode": "separate", "verifier_result": {"rewards": {"reward": 0}},
                                    "verifier": {"started_at": "2026-10-12T16:30:00Z"}}, None),
])
def test_invalid_reason(tmp_path, episode, result, reason):
    (tmp_path / "agent").mkdir()
    for path, doc in ((tmp_path / "agent" / "episode.json", episode), (tmp_path / "result.json", result)):
        if doc is not None:
            path.write_text(json.dumps(doc))
    assert invalid_reason(load_trial(tmp_path)) == reason


def test_conditions_rates_and_intervals(analysis):
    t = analysis[1]
    c, eps = t["conditions"].set_index("condition"), t["episodes"]
    assert list(c.index) == ["single_5step", "gemma31b_parallel_hard", "gemma31b_sequential_medium",
                             "gemma31b_arbiter_hard"]
    par = c.loc["gemma31b_parallel_hard"]
    assert (par["n"], par["n_valid"], par["n_resolved"], par["n_delegated"], par["delegated_frac"]) == (2, 2, 1, 2, 1.0)
    assert (par["resolve_ci_lo"], par["resolve_ci_hi"]) == pytest.approx((0.0945, 0.9055), abs=1e-4)
    seq = c.loc["gemma31b_sequential_medium"]
    assert (seq["n"], seq["n_valid"], seq["resolve_rate"]) == (2, 1, 1.0)
    assert (seq["resolve_ci_lo"], seq["resolve_ci_hi"]) == (pytest.approx(0.2065, abs=1e-4), 1.0)
    assert math.isnan(seq["cost_total_ci_lo"])  # one valid episode: a mean, no interval
    v = eps[eps["valid"] & (eps["condition"] == "gemma31b_arbiter_hard")]
    assert c.loc["gemma31b_arbiter_hard", "cost_total_mean"] == pytest.approx(v["cost_total"].mean())
    assert c.loc["gemma31b_arbiter_hard", "cost_per_solve"] == pytest.approx(v["cost_total"].sum())
    assert c.loc["single_5step", "delegated_frac"] == 0 and c.loc["single_5step", "resolve_rate"] == 0.5


def test_bugs_fixed_and_regressions_from_swe_sweep_rewards(tmp_path, capsys):
    """SWE-sweep's grader: reward = bugs fixed / bugs (rounded to 4 places), 0 on a regression; a
    grader failure is invalid. Bugs fixed are the grader's exact count, 0 on a regression. A task's
    bugs come from the sweep's tasks.json: a patch that breaks the build gets n_subtasks 0 from the
    grader, yet its task's bugs stay in the denominator. A valid trial without a reward (a verifier
    in the agent's container failed after it started) is reward 0 with no bug fixed."""
    jobs, job = tmp_path / "jobs" / "ss", "ss__r1__single_5step"
    single, g = make_config(name="single_5step", topology="single"), {"n_subtasks": 14.0, "visible_ok": 1.0}
    trial(jobs, job, "swe-sweep__goawk", single, models(0), reward=0.3571, rewards={**g, "n_resolved": 5.0})
    trial(jobs, job, "swe-sweep__echo", single, models(0), reward=0.0,
          rewards={**g, "n_subtasks": 6.0, "n_resolved": 2.0, "visible_ok": 0.0})
    trial(jobs, job, "swe-sweep__gin", single, models(0), reward=0.0, rewards={**g, "n_subtasks": 11.0, "grader_failed": 1.0})
    trial(jobs, job, "swe-sweep__pendulum", single, models(0), reward=0.0,
          rewards={"n_subtasks": 0.0, "visible_ok": 0.0, "agent_build_ok": 0.0, "grader_failed": 0.0})
    unscored = trial(jobs, job, "swe-sweep__click", single, models(0), exception="RewardFileNotFoundError")
    res = json.loads((unscored / "result.json").read_text())
    res.update(verifier={"started_at": res["finished_at"]}, verifier_environment_mode="shared")
    (unscored / "result.json").write_text(json.dumps(res))
    bugs = {"swe-sweep__goawk": 14, "swe-sweep__echo": 6, "swe-sweep__gin": 11, "swe-sweep__pendulum": 9,
            "swe-sweep__click": 13}
    (jobs / "tasks.json").write_text(json.dumps({"n_subtasks": bugs}))
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(yaml.safe_dump({"fake": PRICES}))
    assert analyze([str(jobs), "--out", str(tmp_path / "out"), "--pricing", str(pricing), "--scheme", "fake"]) == 0
    assert "bug_fix_rate" in capsys.readouterr().out
    eps = pd.read_csv(tmp_path / "out" / "episodes.csv").set_index("task_name")
    assert eps.loc["swe-sweep__goawk", ["n_subtasks", "bugs_fixed", "regressed", "resolved"]].tolist() == [14, 5, 0, 0]
    assert eps.loc["swe-sweep__echo", ["bugs_fixed", "regressed"]].tolist() == [0.0, 1]
    assert eps.loc["swe-sweep__pendulum", ["n_subtasks", "bugs_fixed", "regressed"]].tolist() == [9, 0.0, 1]
    assert eps.loc["swe-sweep__click", ["valid", "reward", "resolved", "n_subtasks", "bugs_fixed"]].tolist() == [
        True, 0, 0, 13, 0]
    assert (eps.loc["swe-sweep__gin", "valid"], eps.loc["swe-sweep__gin", "invalid_reason"]) == (False, "grader_failed")
    c = pd.read_csv(tmp_path / "out" / "conditions.csv").set_index("condition").loc["single_5step"]
    assert c[["n_valid", "n_resolved", "n_bugs", "bugs_fixed", "n_regressed"]].tolist() == [4, 0, 42, 5, 2]
    assert (c["bug_fix_rate"], c["reward_mean"]) == pytest.approx((5 / 42, 0.3571 / 4))


def test_stats():
    assert wilson(0, 0)[0] != wilson(0, 0)[0] and wilson(0, 5)[0] == 0 and wilson(5, 5)[1] == pytest.approx(1)
    mean, lo, hi = bootstrap_mean([1.0, 2.0, 3.0, 10.0])
    assert mean == 4.0 and 1.0 <= lo < mean < hi <= 10.0 and bootstrap_mean([1.0, 2.0, 3.0, 10.0]) == (mean, lo, hi)
    assert math.isnan(bootstrap_mean([1.0, float("nan")])[0]) and math.isnan(bootstrap_mean([2.0])[1])


def test_critical_path_by_strategy(analysis):
    eps = analysis[1]["episodes"].set_index(["condition", "task_name"])

    def chains(cond):
        """dispatch -> (subagent LLM s, chain s) from the records, and the episode's calls."""
        r = records(Path(eps.loc[(cond, "task-a"), "trial_dir"]) / "agent")
        by: dict[str, list[float]] = {}
        for x in r["calls"] + r["tools"]:
            if x["role"] == "subagent":
                llm, total = by.setdefault(x["dispatch_id"], [0.0, 0.0])
                d = x["t_end"] - x["t_start"]
                by[x["dispatch_id"]] = [llm + d * ("name" not in x), total + d]
        return by, r["calls"]

    par, _ = chains("gemma31b_parallel_hard")  # the slowest dispatch only
    assert len(par) == 2 and eps.loc[("gemma31b_parallel_hard", "task-a"), "cp_sub_llm_sec"] == pytest.approx(
        max(par.values(), key=lambda v: v[1])[0])
    seq, _ = chains("gemma31b_sequential_medium")  # every dispatch
    assert eps.loc[("gemma31b_sequential_medium", "task-a"), "cp_sub_llm_sec"] == pytest.approx(
        sum(v[0] for v in seq.values()))
    arb, calls = chains("gemma31b_arbiter_hard")  # the slowest candidate, the judges and the slowest debater
    row = eps.loc[("gemma31b_arbiter_hard", "task-a")]
    assert len(arb) == 3 and row["cp_sub_llm_sec"] == pytest.approx(max(arb.values(), key=lambda v: v[1])[0])
    (a,) = records(Path(row["trial_dir"]) / "agent")["arbitrations"]
    assert (a["isolation"], a["debate_rounds"], a["applied"]) == ("copy", 1, True)
    judge = sum(c["t_end"] - c["t_start"] for c in calls if c["role"] == "arbiter")
    debate = max(c["t_end"] - c["t_start"] for c in calls if c["role"] == "debate")
    assert row["cp_judge_llm_sec"] == pytest.approx(judge + debate)
    r = records(Path(eps.loc[("gemma31b_arbiter_hard", "task-b"), "trial_dir"]) / "agent")  # cut-off judge, retried
    judges = [c for c in r["calls"] if c["role"] == "arbiter"]
    assert len(judges) == 2 and r["arbitrations"][0]["rounds"][0]["judge_call_ids"] == [c["call_id"] for c in judges]
    assert eps.loc[("gemma31b_arbiter_hard", "task-b"), "cp_judge_llm_sec"] == pytest.approx(
        sum(c["t_end"] - c["t_start"] for c in judges))


def test_delegation_gate_fails_for_a_condition_that_never_delegated(jobs, tmp_path, capsys):
    _, smk, extra, pricing = jobs
    assert analyze([str(smk), str(extra), "--out", str(tmp_path), "--pricing", str(pricing), "--scheme", "fake"]) == 3
    assert "[FAIL] delegation gate: gemma31b_parallel_easy never delegated" in capsys.readouterr().out
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["delegation_gate"] == {"ok": False, "never_delegated": ["gemma31b_parallel_easy"], "no_valid_episodes": []}
    assert s["invalid_reasons"] == {"provider_error": 1, "no_episode": 1} and (s["n_trials"], s["n_valid"]) == (10, 8)
    assert s["critical_path"]["n_over_wall_clock"] == 0 and s["critical_path"]["n_valid_episodes"] == 8
    easy = pd.read_csv(tmp_path / "conditions.csv").set_index("condition").loc["gemma31b_parallel_easy"]
    assert (easy["n_valid"], easy["n_resolved"]) == (1, 0) and easy["cost_total_mean"] > 0
    assert math.isnan(easy["cost_per_solve"])  # no solve: no cost per solve


def test_condition_order():
    def row(cond, topology="subagents", strategy="parallel", scope="hard", tools="all", sub=SUB):
        return pd.Series({"condition": cond, "topology": topology, "strategy": strategy, "scope": scope,
                          "orchestrator_tools": tools, "orchestrator_model": ORCH, "subagent_model": sub})
    rows = [row("pending", topology=None), row("manager_parallel_hard", tools="read_file"),
            row("samemodel_parallel_hard", sub=ORCH), row("arbiter_easy", strategy="arbiter", scope="easy"),
            row("parallel_hard"), row("manager_arbiter_hard", strategy="arbiter", tools="read_file"),
            row("single_5step", topology="single", strategy=None, scope=None)]
    assert [r["condition"] for r in sorted(rows, key=condition_order)] == [
        "single_5step", "parallel_hard", "arbiter_easy", "samemodel_parallel_hard", "manager_parallel_hard",
        "manager_arbiter_hard", "pending"]


def test_pricing_config_matches_the_base_config():
    prices = yaml.safe_load((ROOT / "configs/pricing.yaml").read_text())["sail_balanced"]
    base = yaml.safe_load((ROOT / "configs/sail/base.yaml").read_text())
    for m in (base["orchestrator"], base["subagent"]["model"]):
        assert prices[m["model"]] == {"in": m["price_in"], "cached": m["price_cache_read"], "out": m["price_out"]}


def test_all_figures_render(analysis, tmp_path):
    out, _ = analysis
    assert plot([str(out), "--out", str(tmp_path)]) == 0
    for name in FIGURES:  # no interactive episode here: that figure only says so
        assert (tmp_path / f"{name}.png").stat().st_size > (1_000 if name == "interactivity" else 10_000)
    index = (tmp_path / "index.md").read_text()
    assert index.count("\n## ") == len(FIGURES) == 9 and all(f"({n}.png)" in index for n in FIGURES)


def test_analyze_script(jobs, tmp_path):
    script = str(ROOT / "scripts/analyze.sh")
    assert subprocess.run(["/bin/bash", "-n", script]).returncode == 0
    help_ = subprocess.run(["/bin/bash", script, "--help"], capture_output=True, text=True)
    assert help_.returncode == 0 and "JOBS_DIR" in help_.stdout
    _, smk, extra, _ = jobs
    env = {**os.environ, "PYTHON": sys.executable, "OUT": "both"}  # fake models are unpriced: NaN cost
    res = subprocess.run(["/bin/bash", script, str(smk), str(extra)], capture_output=True, text=True, env=env,
                         cwd=tmp_path)
    assert res.returncode == 3, res.stdout + res.stderr
    assert "[FAIL] delegation gate" in res.stdout and "figures:" in res.stdout
    assert sorted(p.stem for p in (tmp_path / "figures/both").glob("*.png")) == sorted(FIGURES)
    assert (tmp_path / "analysis/both/episodes.csv").is_file()
