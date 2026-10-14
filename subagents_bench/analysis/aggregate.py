"""The analysis tables: one row per trial (episodes), per record (calls, tools, dispatches,
arbitrations) and per condition, plus the summary and the delegation gate."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import SCOPES, STRATEGIES
from .fairplay import tool_flags
from .interactivity import condition_table as interactivity_table
from .interactivity import hint_rows, position_table
from .latency import PARTS, critical_path
from .load import Trial
from .pricing import call_cost
from .stats import bootstrap_mean, wilson
from .validity import outcome

ROLE_SHORT = {"orchestrator": "orch", "subagent": "sub", "arbiter": "arb", "debate": "deb"}
TOKEN_KINDS = ("prompt", "cached", "completion", "reasoning")
MEAN_METRICS = ("cost_total", "wall_clock_sec", "total_tokens")
CP_TOLERANCE_SEC = 1e-3


def _keys(t: Trial) -> dict[str, Any]:
    return {"sweep": t.sweep, "job": t.dir.parent.name, "round": t.round, "condition": t.condition,
            "task_name": t.task_name, "trial_name": t.dir.name}


def episode_row(t: Trial, prices: dict) -> dict[str, Any]:
    ep, recs = t.episode, t.records
    row = {**_keys(t), "episode_id": (ep or {}).get("episode_id"), "trial_dir": str(t.dir)}
    if ep is not None:
        tools = (ep["config"].get("tools") or {}).get("orchestrator_tools")
        row.update({k: ep.get(k) for k in ("topology", "strategy", "scope", "orchestrator_model", "subagent_model")},
                   orchestrator_tools="all" if tools is None else "+".join(tools))
    row.update(exit_reason=(ep or {}).get("exit_reason"), **outcome(t))
    if ep is None:
        return row
    row.update(t_start=ep["t_start"], wall_clock_sec=ep.get("wall_clock_sec"), steps=len(recs["steps"]),
               spawns=len(recs["spawns"]), dispatches=len(recs["dispatches"]), arbitrations=len(recs["arbitrations"]),
               delegated=int(len(recs["spawns"]) > 0), n_provider_errors=ep.get("n_provider_errors"),
               interactive=int(ep["config"].get("interactive") is not None), n_hints=ep.get("n_hints"),
               n_hints_delivered=ep.get("n_hints_delivered"),
               n_stream_repairs=sum(c["stream_repairs"] for c in recs["calls"]))
    for role, short in ROLE_SHORT.items():
        mine = [c for c in recs["calls"] if c["role"] == role]
        for k in TOKEN_KINDS:
            row[f"{short}_{k}_tokens"] = sum(c[f"{k}_tokens"] or 0 for c in mine)
        row[f"cost_{short}"] = sum(call_cost(c, prices) for c in mine)
    row["total_tokens"] = sum(row[f"{s}_{k}_tokens"] for s in ROLE_SHORT.values() for k in ("prompt", "completion"))
    row["cost_total"] = sum(row[f"cost_{s}"] for s in ROLE_SHORT.values())
    row.update(command_timing(recs))
    flags = [f for x in recs["tools"] for f in tool_flags(x)]
    row.update(n_fairplay_flags=len(flags), fairplay_kinds=";".join(sorted(set(flags))))
    cp = critical_path(recs)
    row.update({f"cp_{p}_sec": cp[p] for p in PARTS}, cp_total_sec=sum(cp.values()))
    row["overhead_sec"] = row["wall_clock_sec"] - row["cp_total_sec"] if row["wall_clock_sec"] is not None else None
    return row


def _median(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else math.nan


def command_timing(recs: dict[str, list[dict]]) -> dict[str, Any]:
    """How long the orchestrator's own bash commands and LLM calls took: what a message waits
    behind between its steps."""
    bash = [x["t_end"] - x["t_start"] for x in recs["tools"] if x["role"] == "orchestrator" and x["name"] == "bash"]
    llm = [c["t_end"] - c["t_start"] for c in recs["calls"] if c["role"] == "orchestrator" and c["error"] is None]
    return {"orch_bash_n": len(bash), "orch_bash_median_sec": _median(bash),
            "orch_bash_max_sec": max(bash) if bash else math.nan, "orch_llm_median_sec": _median(llm)}


def record_rows(t: Trial, valid: bool, prices: dict) -> dict[str, list[dict]]:
    key = {**_keys(t), "valid": valid}
    ep = t.episode or {}
    design = {"strategy": ep.get("strategy"), "scope": ep.get("scope")}
    r = t.records
    return {
        "calls": [{**key, **c, "cost": call_cost(c, prices)} for c in r["calls"]],
        "tools": [{**key, **{k: v for k, v in x.items() if k != "arguments"}, "fairplay": ";".join(tool_flags(x))}
                  for x in r["tools"]],
        "dispatches": [{**key, **design, **{k: v for k, v in d.items() if k not in ("files_modified", "instruction")},
                        "n_files_modified": len(d["files_modified"])} for d in r["dispatches"]],
        "arbitrations": [{**key, **{k: v for k, v in a.items() if k != "rounds"}, "n_rounds": len(a["rounds"]),
                          "candidate_dispatch_ids": ";".join(a["candidate_dispatch_ids"]),
                          "n_judge_calls": sum(len(x["judge_call_ids"]) for x in a["rounds"]),
                          "n_debate_calls": sum(len(x["debate_call_ids"]) for x in a["rounds"]),
                          "all_parsed": all(x["parse_ok"] for x in a["rounds"])} for a in r["arbitrations"]],
    }


def condition_order(row: pd.Series) -> tuple:
    """Single-agent baselines, strategy x scope, the controls (same model, then managers), and
    last the conditions without an episode yet (design unknown)."""
    if row["topology"] != "subagents":
        return (0 if row["topology"] == "single" else 3, 0, 0, 0, row["condition"])
    manager = row["orchestrator_tools"] != "all"
    control = manager or row["subagent_model"] == row["orchestrator_model"]
    return (1 + control, manager, STRATEGIES.index(row["strategy"]), SCOPES.index(row["scope"]), row["condition"])


def condition_table(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in episodes.groupby("condition", sort=False):
        design = g.dropna(subset=["topology"]).iloc[0] if g["topology"].notna().any() else g.iloc[0]
        v = g[g["valid"]]
        k, n = int(v["resolved"].sum()), len(v)
        lo, hi = wilson(k, n)
        n_del = int(v["delegated"].sum()) if n else 0
        bugs = float(v["n_subtasks"].sum()) if n else 0.0  # SWE-sweep: bugs in the valid trials' tasks
        row = {"condition": cond, **{c: design.get(c) for c in ("topology", "strategy", "scope", "orchestrator_tools",
                                                                 "orchestrator_model", "subagent_model")},
               "n": len(g), "n_valid": n, "n_resolved": k, "resolve_rate": k / n if n else math.nan,
               "resolve_ci_lo": lo, "resolve_ci_hi": hi, "n_delegated": n_del,
               "delegated_frac": n_del / n if n else math.nan,
               "n_bugs": bugs, "bugs_fixed": float(v["bugs_fixed"].sum()) if n else 0.0,
               "bug_fix_rate": float(v["bugs_fixed"].sum()) / bugs if bugs else math.nan,
               "n_regressed": int(v["regressed"].sum()) if n else 0,
               "n_fairplay_flagged": int((v["n_fairplay_flags"].fillna(0) > 0).sum()) if n and "n_fairplay_flags" in v else 0,
               "reward_mean": float(v["reward"].mean()) if n else math.nan}
        for m in MEAN_METRICS:
            row[f"{m}_mean"], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = bootstrap_mean(v[m] if n else [])
        row["cost_per_solve"] = float(v["cost_total"].sum(skipna=False)) / k if k else math.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.iloc[sorted(range(len(out)), key=lambda i: condition_order(out.iloc[i]))].reset_index(drop=True)


def build_tables(trials: list[Trial], prices: dict) -> dict[str, pd.DataFrame]:
    eps = [episode_row(t, prices) for t in trials]
    recs: dict[str, list[dict]] = {"calls": [], "tools": [], "dispatches": [], "arbitrations": []}
    hints: list[dict] = []
    for t, e in zip(trials, eps):
        for kind, rows in record_rows(t, e["valid"], prices).items():
            recs[kind] += rows
        hints += hint_rows(t, _keys(t), e["valid"])
    episodes = pd.DataFrame(eps)
    hint_table = pd.DataFrame(hints)
    return {"episodes": episodes, "conditions": condition_table(episodes),
            **{k: pd.DataFrame(v) for k, v in recs.items()},
            "hints": hint_table, "interactivity": interactivity_table(hint_table),
            "hint_positions": position_table(hint_table)}


def delegation_gate(conditions: pd.DataFrame) -> list[str]:
    """Subagent conditions with valid episodes of which none delegated."""
    c = conditions[(conditions["topology"] == "subagents") & (conditions["n_valid"] > 0)]
    return list(c.loc[c["n_delegated"] == 0, "condition"])


def _none(x: Any) -> Any:
    return None if isinstance(x, float) and not math.isfinite(x) else x


def summary(tables: dict[str, pd.DataFrame], jobs_dirs: list[str], pricing: dict[str, str]) -> dict[str, Any]:
    eps, conds = tables["episodes"], tables["conditions"]
    valid = eps[eps["valid"]]
    over = valid["cp_total_sec"] - valid["wall_clock_sec"]
    failed = delegation_gate(conds)
    flagged = eps[eps.get("n_fairplay_flags", pd.Series(dtype=float)).fillna(0) > 0] if len(eps) else eps
    return {
        "jobs_dirs": jobs_dirs, "pricing": pricing, "n_trials": len(eps), "n_valid": int(eps["valid"].sum()),
        "invalid_reasons": dict(Counter(eps["invalid_reason"].dropna())),
        "critical_path": {
            "n_valid_episodes": len(valid), "n_over_wall_clock": int((over > CP_TOLERANCE_SEC).sum()),
            "mean_overhead_sec": _none(float(valid["overhead_sec"].mean())),
            "mean_overhead_frac": _none(float((valid["overhead_sec"] / valid["wall_clock_sec"]).mean())),
        },
        "fairplay": {"n_flagged_episodes": len(flagged),
                     "trials": [str(x) for x in flagged.get("trial_dir", pd.Series(dtype=str))]},
        "delegation_gate": {"ok": not failed, "never_delegated": failed,
                            "no_valid_episodes": list(conds.loc[conds["n_valid"] == 0, "condition"])},
    }


def write(tables: dict[str, pd.DataFrame], summ: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
