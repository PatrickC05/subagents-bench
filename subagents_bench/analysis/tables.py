"""Build the aggregation tables of ``docs/DESIGN.md`` section 6 from loaded trials.

* :func:`episode_row` -- one ``episodes.csv`` row per trial (keys, design,
  task covariates, outcome/validity, harbor phases, counts, tokens by role,
  cost per pricing scheme, FLOPs, time, measured critical path, simulated wall
  time per serving profile);
* :func:`record_tables` -- ``calls/tools/dispatches/spawns/arbitrations/steps``
  flattened, bulky lists dropped or summarized, plus episode keys;
* :func:`condition_table` -- ``conditions.csv`` / ``conditions_by_difficulty.csv``
  (Wilson CI on resolve rate, task-balanced rate, delegation uptake and the
  per-protocol resolve rate, seeded bootstrap CIs on means, per-solve costs);
* :func:`paired_table` -- each condition vs each baseline on (sweep, task,
  round), exact McNemar;
* :func:`calibration_table` -- per (model, provider, completion_window) TTFT,
  decode rate, the least-squares TTFT fit and the joint call-time fit of the
  ``fitted`` serving profile.

Unknown values are NaN/None, never 0: a trial without ``episode.json`` has NaN
tokens, a model without a price has NaN cost, a model without parameters has
NaN FLOPs. Zero is used only when it was observed (e.g. a role that made no
calls costs exactly 0).

Validity (``invalid_reason``) removes trials that do not measure their
condition: no episode, a billing refusal, a harbor error outside the agent, a
provider failure anywhere in the episode, an exception from the agent itself
after which harbor skipped the verifier, or no ``result.json``. Several trials
of one (condition, sweep, task, round) -- re-runs, or the same sweep passed
twice -- are flagged ``duplicate`` except one, and only that one enters the
condition and paired tables.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..schema import ArbitrationRecord, CallRecord, DispatchRecord, SpawnRecord, StepRecord, ToolRecord
from ..telemetry import is_provider_error
from .latency import CATEGORIES, CriticalPath, EpisodeGraph, fit_target_sec, measured_call_sec
from .load import ROLE_SHORT, Trial, parse_ts, phase_seconds
from .pricing import ModelCatalog, PriceBook
from .stats import bootstrap_mean_ci, linear_fit, mcnemar_exact, median, nonneg_least_squares, quantile, wilson_ci

NAN = float("nan")
EPS = 1e-9

KEY_COLS = ["sweep", "job_name", "round", "condition", "trial_name", "task_name", "episode_id"]
DESIGN_COLS = [
    "topology", "strategy", "scope", "orchestrator_model", "subagent_model", "arbiter_model",
    "max_concurrent", "n_candidates", "orchestrator_steps_budget", "orchestrator_tools",
]
TASK_COLS = ["difficulty", "category", "expert_time_min", "junior_time_min"]
OUTCOME_COLS = ["reward", "resolved", "harbor_exception", "exit_reason", "valid", "invalid_reason", "episode_status",
                "duplicate"]
PHASE_COLS = ["env_setup_sec", "agent_setup_sec", "agent_exec_sec", "verifier_sec", "trial_total_sec"]
EPISODE_COLS = [
    "wall_clock_sec", "orchestrator_steps", "spawn_calls", "delegated", "dispatches", "redispatches", "arbitrations",
    "n_isolation_checked", "n_isolation_violations", "n_leaked_files",
    "instruction_following", "dispatch_submitted_frac", "n_files_modified",
]
TOKEN_KINDS = ["calls", "prompt_tokens", "cached_tokens", "uncached_prompt_tokens", "completion_tokens", "reasoning_tokens"]
TOKEN_COLS = [f"{s}_{k}" for s in ROLE_SHORT.values() for k in TOKEN_KINDS] + ["total_tokens"]
# Calls cut off at max_tokens (finish_reason 'length'), per role.
LENGTH_COLS = [f"{s}_length_finishes" for s in ROLE_SHORT.values()]
TIME_COLS = (
    [f"llm_sec_{s}" for s in ROLE_SHORT.values()]
    + [f"tool_sec_{s}" for s in ROLE_SHORT.values()]
    + ["backoff_sec", "n_retries", "n_command_timeouts"]
)
CP_COLS = [
    "cp_total_sec", "cp_orch_llm_sec", "cp_sub_llm_sec", "cp_arb_llm_sec", "cp_tool_sec", "cp_group_sec",
    "overhead_sec", "harness_exec_sec", "sub_busy_sec", "parallel_speedup",
]
EXTRA_COLS = ["n_failed_calls", "n_provider_errors", "n_usage_estimated", "container_arch", "host_machine"]

# harbor exceptions raised before/around the agent rather than by it.
OUTSIDE_AGENT_EXCEPTIONS = {
    "EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "SandboxBuildFailedError",
    "HealthcheckError", "CancelledError", "DockerBuildError",
}
# Agent-phase exceptions harbor catches before running the verifier anyway
# (harbor 0.24.0 SingleStepTrial._run_agent): the container is still graded.
GRADED_AGENT_EXCEPTIONS = {"AgentTimeoutError", "NonZeroAgentExitCodeError"}

INT_COLS = (
    ["round", "resolved", "max_concurrent", "n_candidates", "orchestrator_steps_budget", "orchestrator_steps",
     "spawn_calls", "delegated", "dispatches", "redispatches", "arbitrations", "n_isolation_checked",
     "n_isolation_violations", "n_leaked_files", "n_files_modified", "n_retries", "n_command_timeouts",
     "n_failed_calls", "n_provider_errors", "n_usage_estimated", "total_tokens"]
    + [c for c in TOKEN_COLS if c != "total_tokens"] + LENGTH_COLS
)


def cost_cols(schemes: Iterable[str]) -> list[str]:
    cols = []
    for s in schemes:
        cols += [f"cost_{s}_{r}" for r in ROLE_SHORT.values()] + [f"cost_{s}_total"]
    return cols + ["cost_runtime_total"]


def flops_cols() -> list[str]:
    return [f"flops_{r}" for r in ROLE_SHORT.values()] + ["flops_total"]


def sim_cols(profiles: Iterable[Any]) -> list[str]:
    return [f"sim_{p.name}_wall_sec" for p in profiles]


def episode_columns(schemes: Iterable[str], profiles: Iterable[Any]) -> list[str]:
    return (
        KEY_COLS + DESIGN_COLS + TASK_COLS + OUTCOME_COLS + PHASE_COLS + EPISODE_COLS + TOKEN_COLS + LENGTH_COLS
        + cost_cols(schemes) + flops_cols() + TIME_COLS + CP_COLS + sim_cols(profiles) + EXTRA_COLS
    )


# ---------------------------------------------------------------------------
# episodes.csv
# ---------------------------------------------------------------------------


def _nan_if_none(v: Any) -> Any:
    return NAN if v is None else v


def design_fields(ep: dict | None, cfg: dict | None) -> dict[str, Any]:
    """Design columns from episode.json, falling back to the resolved config dict."""
    ep, cfg = ep or {}, cfg or {}
    sub = cfg.get("subagent") or {}
    arb = sub.get("arbiter") or {}
    orch = cfg.get("orchestrator") or {}
    topology = ep.get("topology") or cfg.get("topology")
    is_sub = topology == "subagents"
    if ep:
        strategy, scope = ep.get("strategy"), ep.get("scope")
        orch_model, sub_model, arb_model = ep.get("orchestrator_model"), ep.get("subagent_model"), ep.get("arbiter_model")
    else:
        strategy = sub.get("strategy") if is_sub else None
        scope = sub.get("scope") if is_sub else None
        orch_model = orch.get("model")
        sub_model = ((sub.get("model") or {}).get("model") or orch_model) if is_sub else None
        arb_model = ((arb.get("model") or {}).get("model") or orch_model) if strategy == "arbiter" else None
    return {
        "topology": topology,
        "strategy": strategy,
        "scope": scope,
        "orchestrator_model": orch_model,
        "subagent_model": sub_model,
        "arbiter_model": arb_model,
        "max_concurrent": sub.get("max_concurrent") if is_sub else None,
        "n_candidates": arb.get("n_candidates") if strategy == "arbiter" else None,
        "orchestrator_steps_budget": (cfg.get("budget") or {}).get("orchestrator_steps"),
        "orchestrator_tools": orchestrator_tools_label(cfg),
    }


def orchestrator_tools_label(cfg: dict | None) -> str | None:
    """``all`` for the default tool set, else the orchestrator's own tools
    joined by ``+`` (``tools.orchestrator_tools``); None without a config."""
    if not cfg:
        return None
    tools = (cfg.get("tools") or {}).get("orchestrator_tools")
    return "all" if tools is None else "+".join(tools)


def harbor_reward(result: dict | None) -> float | None:
    if not result:
        return None
    vr = result.get("verifier_result")
    if not isinstance(vr, dict):
        return None
    rewards = vr.get("rewards")
    if not isinstance(rewards, dict) or not rewards:
        return None
    v = rewards.get("reward")
    if v is None and len(rewards) == 1:
        v = next(iter(rewards.values()))
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def exception_outside_agent(result: dict, exc: dict) -> bool:
    """True when harbor's exception happened outside the agent (env build/setup, job cancel)."""
    if exc.get("exception_type") in OUTSIDE_AGENT_EXCEPTIONS:
        return True
    ae = result.get("agent_execution")
    if not isinstance(ae, dict) or not ae.get("started_at"):
        return True  # the agent never started
    occurred, started = parse_ts(exc.get("occurred_at")), parse_ts(ae.get("started_at"))
    try:
        return bool(occurred and started and occurred < started)
    except TypeError:
        return False


def exception_skipped_verifier(result: dict, exc: dict) -> bool:
    """True when harbor's exception came from the agent phase and the verifier never ran.

    harbor runs the verifier after an agent timeout (``GRADED_AGENT_EXCEPTIONS``)
    but not after any other exception the agent raises (a harness bug, a
    crashed recorder): such a trial says nothing about the task. A verifier
    exception (the verifier phase started) leaves the trial graded-as-unsolved.
    """
    if exc.get("exception_type") in GRADED_AGENT_EXCEPTIONS or result.get("verifier_result") is not None:
        return False
    ver = result.get("verifier")
    return not (isinstance(ver, dict) and ver.get("started_at"))


def provider_failure(ep: dict, exc: dict | None, n_call_provider_errors: int) -> bool:
    """Whether a provider failure (a plain LLMError after all retries) touched the episode.

    The orchestrator could not be served (``exit_reason == 'provider_error'``,
    harbor exception ``ProviderUnavailableError``), or any call of any role
    failed that way (``n_provider_errors``, recounted from the calls so
    records written before that field existed count too). Records before
    schema_version 3 ended an orchestrator provider failure with
    ``exit_reason == 'error'`` and an ``LLMError:`` error text.
    """
    if ep.get("exit_reason") == "provider_error":
        return True
    if exc and exc.get("exception_type") == "ProviderUnavailableError":
        return True
    if (ep.get("n_provider_errors") or 0) > 0 or n_call_provider_errors > 0:
        return True
    return ep.get("exit_reason") == "error" and is_provider_error(ep.get("error"))


def _isolation_counts(arbitrations: list[dict]) -> dict[str, Any]:
    """Copy-isolation leaks over an episode's arbitrations; NaN for records that predate the check."""
    if arbitrations and not any("isolation_violated" in a for a in arbitrations):
        return {"n_isolation_checked": NAN, "n_isolation_violations": NAN, "n_leaked_files": NAN}
    return {
        "n_isolation_checked": sum(1 for a in arbitrations if a.get("isolation_violated") is not None),
        "n_isolation_violations": sum(1 for a in arbitrations if a.get("isolation_violated") is True),
        "n_leaked_files": sum(len(a.get("leaked_files") or []) for a in arbitrations),
    }


def _record_counts(data) -> dict[str, Any]:
    """Episode counters recomputed from the record files (used for unfinished episodes)."""
    orch_steps = {c.get("step") for c in data.calls if c.get("role") == "orchestrator"}
    spawns = sorted(data.spawns, key=lambda s: (s.get("t_start") or 0, s.get("orchestrator_step") or 0))
    first_group = spawns[0].get("group_id") if spawns else None
    ratios = [s["granted"] / s["requested"] for s in spawns if (s.get("requested") or 0) > 0 and s.get("granted") is not None]
    return {
        "orchestrator_steps": max(len(data.steps), len(orch_steps)),
        "spawn_calls": len(data.spawns),
        "dispatches": len(data.dispatches),
        "redispatches": sum(1 for d in data.dispatches if first_group is not None and d.get("group_id") != first_group),
        "arbitrations": len(data.arbitrations),
        "instruction_following": float(np.mean(ratios)) if ratios else None,
    }


def _sum(values: Iterable[Any]) -> float:
    total = 0.0
    for v in values:
        total += float(v) if v is not None else NAN
    return total


def episode_row(trial: Trial, pricebook: PriceBook, profiles: list[Any]) -> dict[str, Any]:
    data = trial.data
    ep = trial.episode
    res = trial.result
    row: dict[str, Any] = {c: NAN for c in episode_columns(pricebook.scheme_names, profiles)}

    row.update(
        sweep=trial.sweep, job_name=trial.job_name, round=trial.round, condition=trial.condition,
        trial_name=trial.trial_name, task_name=trial.task_name, episode_id=data.episode_id if data else None,
    )
    row.update(design_fields(ep, trial.config))
    meta = trial.task_meta
    for c in TASK_COLS:
        row[c] = meta.get(c)

    # --- outcome ---------------------------------------------------------------
    reward = harbor_reward(res)
    if reward is None and ep and ep.get("reward") is not None:
        reward = float(ep["reward"])  # local runner episodes carry their own reward
    exc = res.get("exception_info") if res else None
    exc = exc if isinstance(exc, dict) else None
    exit_reason = (ep or {}).get("exit_reason")
    n_provider_errors = sum(1 for c in data.calls if is_provider_error(c.get("error"))) if data else 0
    if ep is None:
        invalid = "no_episode"
    elif exit_reason == "billing_error" or (exc and exc.get("exception_type") == "BillingError"):
        invalid = "billing_error"
    elif exc and exception_outside_agent(res, exc):
        invalid = "harbor_exception_outside_agent"
    elif provider_failure(ep, exc, n_provider_errors):
        invalid = "provider_error"
    elif exc and exception_skipped_verifier(res, exc):
        invalid = "agent_exception"
    elif trial.ref.trial_dir is not None and res is None and reward is None:
        invalid = "no_result"
    else:
        invalid = None
    valid = invalid is None
    if reward is not None:
        resolved = 1 if reward >= 1.0 - 1e-9 else 0
    else:
        resolved = 0 if valid else None  # a valid run the verifier could not score is unsolved
    row.update(
        reward=reward, resolved=resolved, harbor_exception=exc.get("exception_type") if exc else None,
        exit_reason=exit_reason, valid=valid, invalid_reason=invalid,
        episode_status=data.status if data else "missing",
    )
    for col, phase in zip(PHASE_COLS, ["environment_setup", "agent_setup", "agent_execution", "verifier", None]):
        row[col] = _nan_if_none(phase_seconds(res, phase))

    if data is None:
        return row  # everything below is unknown for a trial without an episode dir

    # --- counts ----------------------------------------------------------------
    counts = _record_counts(data)
    done = ep is not None and ep.get("status") == "done"
    for k, v in counts.items():
        row[k] = ep.get(k, v) if done else v
        row[k] = _nan_if_none(row[k])
    row["delegated"] = int(row["spawn_calls"] > 0) if not pd.isna(row["spawn_calls"]) else NAN
    row.update(_isolation_counts(data.arbitrations))
    row["wall_clock_sec"] = _nan_if_none((ep or {}).get("wall_clock_sec"))
    row["n_files_modified"] = _nan_if_none((ep or {}).get("n_files_modified"))
    statuses = [d.get("status") for d in data.dispatches]
    row["dispatch_submitted_frac"] = statuses.count("submitted") / len(statuses) if statuses else NAN

    # --- tokens / cost / flops / llm time by role -------------------------------
    catalog: ModelCatalog = pricebook.catalog
    tot_tokens = 0
    for role, s in ROLE_SHORT.items():
        rc = [c for c in data.calls if c.get("role") == role]
        row[f"{s}_calls"] = len(rc)
        for k in ("prompt_tokens", "cached_tokens", "completion_tokens"):
            row[f"{s}_{k}"] = sum(int(c.get(k) or 0) for c in rc)
        row[f"{s}_uncached_prompt_tokens"] = sum(
            int(c["uncached_prompt_tokens"]) if c.get("uncached_prompt_tokens") is not None
            else max(int(c.get("prompt_tokens") or 0) - int(c.get("cached_tokens") or 0), 0)
            for c in rc
        )
        reasoning = [c["reasoning_tokens"] for c in rc if c.get("reasoning_tokens") is not None]
        row[f"{s}_reasoning_tokens"] = sum(reasoning) if reasoning else (0 if not rc else NAN)
        row[f"{s}_length_finishes"] = sum(1 for c in rc if c.get("finish_reason") == "length")
        tot_tokens += row[f"{s}_prompt_tokens"] + row[f"{s}_completion_tokens"]
        for scheme in pricebook.scheme_names:
            row[f"cost_{scheme}_{s}"] = _sum(pricebook.call_cost(scheme, c) for c in rc)
        row[f"flops_{s}"] = _sum(
            catalog.flops(c.get("model"), c.get("prompt_tokens"), c.get("completion_tokens")) for c in rc
        )
        row[f"llm_sec_{s}"] = _sum(measured_call_sec(c) for c in rc)
        row[f"tool_sec_{s}"] = _sum(t.get("duration_sec") for t in data.tools if t.get("role") == role)
    row["total_tokens"] = tot_tokens
    for scheme in pricebook.scheme_names:
        row[f"cost_{scheme}_total"] = _sum(row[f"cost_{scheme}_{s}"] for s in ROLE_SHORT.values())
    row["cost_runtime_total"] = _sum(c.get("cost_usd") for c in data.calls)
    row["flops_total"] = _sum(row[f"flops_{s}"] for s in ROLE_SHORT.values())
    row["backoff_sec"] = _sum(c.get("backoff_sec") or 0.0 for c in data.calls)
    row["n_retries"] = sum(max(int(c.get("n_attempts") or 1) - 1, 0) for c in data.calls)
    row["n_command_timeouts"] = sum(1 for t in data.tools if t.get("timed_out"))
    row["n_failed_calls"] = sum(1 for c in data.calls if c.get("error"))
    row["n_provider_errors"] = n_provider_errors
    row["n_usage_estimated"] = sum(1 for c in data.calls if c.get("usage_source") == "estimated")
    row["container_arch"] = (ep or {}).get("container_arch")
    row["host_machine"] = (ep or {}).get("host_machine")

    # --- critical path + simulations --------------------------------------------
    graph = EpisodeGraph(data)
    cp = CriticalPath(graph).episode()
    row["cp_total_sec"] = cp.total
    for cat in CATEGORIES:
        row[f"cp_{cat}_sec"] = cp.parts[cat]
    row["cp_group_sec"] = cp.group_sec
    row["overhead_sec"] = row["wall_clock_sec"] - cp.total
    row["harness_exec_sec"] = _nan_if_none((ep or {}).get("harness_exec_sec"))
    row["sub_busy_sec"] = _sum(d.get("duration_sec") for d in data.dispatches)
    # sub_busy / critical time inside groups. Undefined without dispatches, and NaN when no group
    # time was recorded (a crash mid-group) rather than a meaningless sub_busy / eps.
    row["parallel_speedup"] = (
        row["sub_busy_sec"] / max(cp.group_sec, EPS) if data.dispatches and cp.group_sec > EPS else NAN
    )
    for prof in profiles:
        row[f"sim_{prof.name}_wall_sec"] = CriticalPath(graph, prof.call_sec).episode().total
    return row


def episodes_frame(rows: list[dict], schemes: list[str], profiles: list[Any]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=episode_columns(schemes, profiles))
    df["valid"] = df["valid"].astype(bool)
    df["duplicate"] = duplicate_mask(df)
    return _int_cols(df, INT_COLS)


DUPLICATE_KEY = ["condition", "sweep", "task_name", "round"]


def duplicate_mask(episodes: pd.DataFrame) -> pd.Series:
    """True for every trial but one of each (condition, sweep, task, round) cell.

    A cell holds one trial per sweep design; more means a re-run, or the same
    sweep aggregated twice (two jobs dirs of one sweep name). The one kept is
    the valid trial, else the first by trial name. Trials without a round
    (bare local episodes, which repeat a task by design) are never duplicates.
    """
    mask = pd.Series(False, index=episodes.index)
    if episodes.empty:
        return mask
    keyed = episodes[episodes["round"].notna() & episodes["condition"].notna() & episodes["task_name"].notna()]
    order = keyed.assign(_invalid=~keyed["valid"].astype(bool)).sort_values(["_invalid", "trial_name"], kind="stable")
    dup = order.duplicated(DUPLICATE_KEY, keep="first")
    mask.loc[dup[dup].index] = True
    return mask


def _without_duplicates(episodes: pd.DataFrame) -> pd.DataFrame:
    return episodes[~episodes["duplicate"].astype(bool)] if "duplicate" in episodes.columns else episodes


def _int_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Nullable-int for count-like columns so CSVs show 3, not 3.0 (and blanks for unknown)."""
    for c in cols:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="raise").astype("Int64")
            except (TypeError, ValueError):
                pass
    return df


# ---------------------------------------------------------------------------
# Flattened record tables
# ---------------------------------------------------------------------------

RECORD_KEYS = ["sweep", "condition", "task_name", "round", "trial_name", "episode_id"]
TEXT_PREVIEW_CHARS = 300


def _join(v: Any) -> str | None:
    return None if v is None else ";".join("" if x is None else str(x) for x in v)


def _json(v: Any) -> str | None:
    return None if v is None else json.dumps(v, sort_keys=True, default=str)


def _preview(v: Any, n: int = TEXT_PREVIEW_CHARS) -> str | None:
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= n else s[:n] + "..."


def _chunk_summary(offsets: Any, _rec: dict) -> dict:
    o = [float(x) for x in offsets or []]
    gaps = np.diff(o) if len(o) > 1 else np.array([])
    return {
        "chunk_offsets_n": len(o),
        "chunk_offset_first_ms": o[0] if o else NAN,
        "chunk_offset_last_ms": o[-1] if o else NAN,
        "chunk_gap_ms_median": float(np.median(gaps)) if gaps.size else NAN,
        "chunk_gap_ms_max": float(gaps.max()) if gaps.size else NAN,
    }


def _attempts_summary(attempts: Any, _rec: dict) -> dict:
    a = attempts or []
    return {
        "attempt_statuses": _join([x.get("status") for x in a if isinstance(x, dict)]),
        "attempt_error_types": _join([x.get("error_type") for x in a if isinstance(x, dict) and x.get("error_type")]),
    }


def _rounds_summary(rounds: Any, _rec: dict) -> dict:
    r = sorted(rounds or [], key=lambda x: x.get("round") or 0)
    return {
        "n_rounds": len(r),
        "round_winners": _join([x.get("winner") for x in r]),
        "round_confidences": _join([x.get("confidence") for x in r]),
        "first_confidence": r[0].get("confidence") if r else None,
        "round_agreements": _join([x.get("agreement") for x in r]),
        "judge_call_ids": _join([x.get("judge_call_id") for x in r]),
        "n_debate_calls": sum(len(x.get("debate_call_ids") or []) for x in r),
        "all_parse_ok": all(bool(x.get("parse_ok")) for x in r) if r else None,
    }


Transform = Callable[[Any, dict], dict]


def _drop(_v: Any, _rec: dict) -> dict:
    return {}


def _rename(name: str, fn: Callable[[Any], Any]) -> Transform:
    return lambda v, _rec: {name: fn(v)}


def _keep_as(fn: Callable[[Any], Any]) -> Callable[[str], Transform]:
    return lambda name: (lambda v, _rec: {name: fn(v)})


RECORD_TABLES: dict[str, tuple[type, dict[str, Transform]]] = {
    "calls": (CallRecord, {
        "msg_hashes": _drop, "msg_chars": _drop, "response_headers": _drop, "usage_raw": _drop,
        "chunk_offsets_ms": _chunk_summary, "attempts": _attempts_summary,
        "tool_call_names": _keep_as(_join)("tool_call_names"), "extra_body": _keep_as(_json)("extra_body"),
    }),
    "tools": (ToolRecord, {
        "args": lambda v, _r: {"args_preview": _preview(v), "args_chars": len(_json(v) or "")},
    }),
    "steps": (StepRecord, {
        "tool_names": _keep_as(_join)("tool_names"),
        "spawned_group_ids": lambda v, _r: {"spawned_group_ids": _join(v), "n_spawned_groups": len(v or [])},
    }),
    "spawns": (SpawnRecord, {
        "instructions": _rename("n_instructions", lambda v: len(v or [])),
        "dispatch_ids": lambda v, _r: {"dispatch_ids": _join(v), "n_dispatches": len(v or [])},
        "arbitration_ids": lambda v, _r: {"arbitration_ids": _join(v), "n_arbitrations": len(v or [])},
        "files_modified": _drop,
    }),
    "dispatches": (DispatchRecord, {
        "instruction": lambda v, _r: {"instruction_preview": _preview(v), "instruction_chars": len(v or "")},
        "report": lambda v, _r: {"report_preview": _preview(v), "report_chars": len(v or "")},
        "tools": _keep_as(_join)("tools"), "tool_counts": _keep_as(_json)("tool_counts"), "files_modified": _drop,
    }),
    "arbitrations": (ArbitrationRecord, {
        "instruction": _rename("instruction_preview", _preview),
        "candidate_dispatch_ids": _keep_as(_join)("candidate_dispatch_ids"),
        "rounds": _rounds_summary,
        "applied_files": _rename("n_applied_files", lambda v: len(v or [])),
        # Absent (records before the leak check) stays empty, not 0.
        "leaked_files": lambda v, _r: {"leaked_files": _json(v),
                                       "n_leaked_files": len(v) if isinstance(v, list) else None},
    }),
}


def _flatten(cls: type, transforms: dict[str, Transform], rec: dict) -> dict:
    out: dict[str, Any] = {}
    for f in fields(cls):
        if f.name == "episode_id":
            continue  # carried in the key columns
        v = rec.get(f.name)
        if f.name in transforms:
            out.update(transforms[f.name](v, rec))
        elif isinstance(v, (list, dict)):
            out[f.name] = _json(v)
        else:
            out[f.name] = v
    return out


def record_columns(name: str, extra: list[str] | None = None) -> list[str]:
    cls, transforms = RECORD_TABLES[name]
    return RECORD_KEYS + list(_flatten(cls, transforms, {}).keys()) + (extra or [])


def record_tables(trials: list[Trial], pricebook: PriceBook) -> dict[str, pd.DataFrame]:
    """The six flattened per-record tables. calls.csv also gets cost_<scheme> and flops per call."""
    call_extra = [f"cost_{s}" for s in pricebook.scheme_names] + ["flops"]
    rows: dict[str, list[dict]] = {name: [] for name in RECORD_TABLES}
    for t in trials:
        if t.data is None:
            continue
        keys = {"sweep": t.sweep, "condition": t.condition, "task_name": t.task_name, "round": t.round,
                "trial_name": t.trial_name, "episode_id": t.data.episode_id}
        for name, (cls, transforms) in RECORD_TABLES.items():
            for rec in getattr(t.data, name):
                r = {**keys, **_flatten(cls, transforms, rec)}
                if name == "calls":
                    for s in pricebook.scheme_names:
                        r[f"cost_{s}"] = pricebook.call_cost(s, rec)
                    r["flops"] = pricebook.catalog.flops(
                        rec.get("model"), rec.get("prompt_tokens"), rec.get("completion_tokens")
                    )
                rows[name].append(r)
    out = {}
    for name in RECORD_TABLES:
        cols = record_columns(name, call_extra if name == "calls" else None)
        out[name] = _int_cols(pd.DataFrame(rows[name], columns=cols), ["round"])
    return out


# ---------------------------------------------------------------------------
# conditions.csv / conditions_by_difficulty.csv
# ---------------------------------------------------------------------------

CONDITION_FIXED_METRICS = (
    ["wall_clock_sec", "total_tokens", "flops_total"]
    + [f"{s}_{k}" for s in ROLE_SHORT.values() for k in TOKEN_KINDS]
    + ["dispatches", "spawn_calls", "redispatches", "orchestrator_steps", "instruction_following",
       "dispatch_submitted_frac", "cp_total_sec", "cp_orch_llm_sec", "cp_sub_llm_sec", "cp_arb_llm_sec",
       "cp_tool_sec", "cp_group_sec", "overhead_sec", "harness_exec_sec", "sub_busy_sec", "parallel_speedup",
       "agent_exec_sec", "trial_total_sec"]
)


def strict_nan_metric(m: str) -> bool:
    """Metrics whose NaN means "no price / no parameters for a model used", not "not applicable".

    For these, one NaN among a condition's valid episodes makes the condition's
    mean, CI and per-solve value NaN; dropping it would silently average over
    the few episodes that happened not to call the unpriced model. Other
    metrics (wall clock of a crashed episode, speedup of a zero-dispatch
    episode) drop NaN values.
    """
    return m.startswith(("cost_", "flops_", "sim_")) or m.endswith("_reasoning_tokens")


def condition_metrics(episodes: pd.DataFrame) -> list[str]:
    sims = [c for c in episodes.columns if c.startswith("sim_") and c.endswith("_wall_sec")]
    costs = [c for c in episodes.columns if c.startswith("cost_") and c.endswith("_total")]
    return ["wall_clock_sec"] + sims + costs + [m for m in CONDITION_FIXED_METRICS if m != "wall_clock_sec"]


def order_conditions(names: Iterable[str], baselines: Iterable[str] = ()) -> list[str]:
    names = [n for n in dict.fromkeys(names) if isinstance(n, str)]
    base = [b for b in baselines if b in names]
    return base + sorted(n for n in names if n not in base)


def _first(series: pd.Series) -> Any:
    s = series.dropna()
    return s.iloc[0] if len(s) else None


RESOLVE_EXTRA_COLS = [
    "n_tasks_valid", "resolve_rate_task_balanced",
    "n_delegated", "delegated_frac", "resolve_rate_delegated", "resolve_delegated_ci_lo", "resolve_delegated_ci_hi",
]
ISOLATION_COND_COLS = ["n_isolation_checked", "n_isolation_violations", "isolation_violated_frac", "n_leaked_files"]
LENGTH_FRAC_COLS = [f"length_finish_frac_{s}" for s in ROLE_SHORT.values()]


def condition_table(
    episodes: pd.DataFrame, by: list[str], schemes: list[str], baselines: Iterable[str] = ()
) -> pd.DataFrame:
    """One row per ``by`` group over its valid, non-duplicate episodes.

    The resolve rate pools trials (Wilson CI), so it describes these tasks
    with their trial-to-trial randomness; ``resolve_rate_task_balanced``
    weights every task with a valid trial equally, so a condition that lost
    trials of one task (invalid, e.g. an environment failure) is not shifted
    by the task mix. ``resolve_rate_delegated`` is the per-protocol rate:
    among valid episodes whose orchestrator spawned subagents at least once.
    """
    metrics = [m for m in condition_metrics(episodes) if m in episodes.columns]
    cols = list(by) + [c for c in DESIGN_COLS if c not in by] + [
        "n", "n_valid", "n_resolved", "resolve_rate", "resolve_ci_lo", "resolve_ci_hi"] + RESOLVE_EXTRA_COLS
    for m in metrics:
        cols += [f"{m}_mean", f"{m}_ci_lo", f"{m}_ci_hi"]
    cols += [f"cost_per_solve_{s}" for s in [*schemes, "runtime"]] + ["time_per_solve_sec"]
    cols += ISOLATION_COND_COLS + LENGTH_FRAC_COLS
    if episodes.empty:
        return pd.DataFrame(columns=cols)
    df = _without_duplicates(episodes).copy()
    if "difficulty" in by:
        df["difficulty"] = df["difficulty"].fillna("unknown")
    df = df[df["condition"].notna()]
    rows = []
    for key, g in df.groupby(by, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = dict(zip(by, key))
        for c in DESIGN_COLS:
            if c not in by:
                row[c] = _first(g[c])
        valid = g[g["valid"].astype(bool)]
        n_valid = len(valid)
        n_res = int(pd.to_numeric(valid["resolved"], errors="coerce").fillna(0).sum())
        lo, hi = wilson_ci(n_res, n_valid)
        row.update(n=len(g), n_valid=n_valid, n_resolved=n_res,
                   resolve_rate=n_res / n_valid if n_valid else NAN, resolve_ci_lo=lo, resolve_ci_hi=hi)
        row.update(_resolve_extras(valid))
        row.update(_isolation_and_length(valid))
        for m in metrics:
            vals = pd.to_numeric(valid[m], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            if strict_nan_metric(m) and np.isnan(vals).any():
                row[f"{m}_mean"] = row[f"{m}_ci_lo"] = row[f"{m}_ci_hi"] = NAN
            else:
                row[f"{m}_mean"], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = bootstrap_mean_ci(vals, 2000, 0)
        solved = pd.to_numeric(valid["resolved"], errors="coerce").fillna(0)
        for s in [*schemes, "runtime"]:
            row[f"cost_per_solve_{s}"] = _per_solve(valid.get(f"cost_{s}_total"), solved, strict=True)
        row["time_per_solve_sec"] = _per_solve(valid.get("wall_clock_sec"), solved, strict=False)
        rows.append(row)
    out = pd.DataFrame(rows, columns=cols)
    order = {c: i for i, c in enumerate(order_conditions(out["condition"], baselines))}
    sort_cols = ["_o"] + [c for c in by if c != "condition"]
    out = out.assign(_o=out["condition"].map(order)).sort_values(sort_cols, kind="stable").drop(columns="_o")
    return _int_cols(out.reset_index(drop=True), ["n", "n_valid", "n_resolved", "max_concurrent", "n_candidates",
                                                    "orchestrator_steps_budget", "n_tasks_valid", "n_delegated",
                                                    "n_isolation_checked", "n_isolation_violations",
                                                    "n_leaked_files"])


def _resolve_extras(valid: pd.DataFrame) -> dict[str, Any]:
    """Task-balanced resolve rate and delegation uptake over a group's valid episodes."""
    solved = pd.to_numeric(valid["resolved"], errors="coerce").fillna(0)
    per_task = solved.groupby(valid["task_name"]).mean() if len(valid) else pd.Series(dtype=float)
    # Without the column (an episodes frame from before it existed) uptake is unknown, not 0.
    deleg = (pd.to_numeric(valid["delegated"], errors="coerce") if "delegated" in valid
             else pd.Series(NAN, index=valid.index, dtype=float))
    known = deleg.notna()
    took = valid[deleg == 1]
    n_took = len(took)
    k_took = int(pd.to_numeric(took["resolved"], errors="coerce").fillna(0).sum())
    lo, hi = wilson_ci(k_took, n_took)
    return {
        "n_tasks_valid": len(per_task),
        "resolve_rate_task_balanced": float(per_task.mean()) if len(per_task) else NAN,
        "n_delegated": n_took,
        "delegated_frac": n_took / int(known.sum()) if known.any() else NAN,
        "resolve_rate_delegated": k_took / n_took if n_took else NAN,
        "resolve_delegated_ci_lo": lo, "resolve_delegated_ci_hi": hi,
    }


def _strict_sum(values: pd.Series) -> float:
    """Sum, or NaN if any value is unknown (records that predate the field)."""
    v = pd.to_numeric(values, errors="coerce")
    return float(v.sum()) if v.notna().all() else NAN


def _isolation_and_length(valid: pd.DataFrame) -> dict[str, Any]:
    """Copy-isolation leaks and per-role max_tokens truncation over a group's valid episodes."""
    out: dict[str, Any] = {c: NAN for c in ISOLATION_COND_COLS + LENGTH_FRAC_COLS}
    if "n_isolation_checked" in valid:
        checked, violated = _strict_sum(valid["n_isolation_checked"]), _strict_sum(valid["n_isolation_violations"])
        out.update(n_isolation_checked=checked, n_isolation_violations=violated,
                   isolation_violated_frac=violated / checked if checked > 0 else NAN,
                   n_leaked_files=_strict_sum(valid["n_leaked_files"]))
    for s in ROLE_SHORT.values():
        if f"{s}_length_finishes" in valid and f"{s}_calls" in valid:
            n_calls = pd.to_numeric(valid[f"{s}_calls"], errors="coerce").sum()
            n_len = pd.to_numeric(valid[f"{s}_length_finishes"], errors="coerce").sum()
            out[f"length_finish_frac_{s}"] = float(n_len / n_calls) if n_calls > 0 else NAN
    return out


def _per_solve(values: pd.Series | None, solved: pd.Series, strict: bool) -> float:
    """Total spent / solves. ``strict``: any unknown spend -> NaN; else over episodes where it is known."""
    if values is None:
        return NAN
    v = pd.to_numeric(values, errors="coerce")
    mask = v.notna()
    if strict and not mask.all():
        return NAN
    denom = float(solved[mask].sum())
    return float(v[mask].sum()) / denom if denom > 0 else NAN


# ---------------------------------------------------------------------------
# paired.csv
# ---------------------------------------------------------------------------

PAIRED_COLS = ["condition", "baseline", "n_pairs", "both", "only_cond", "only_base", "neither",
               "resolve_rate_cond", "resolve_rate_base", "delta_resolve", "mcnemar_p"]


def paired_table(episodes: pd.DataFrame, baselines: Iterable[str]) -> pd.DataFrame:
    """Each condition vs each baseline, matched on (sweep, task_name, round) over valid episodes.

    Rounds are numbered from 1 in every sweep, so the sweep is part of the
    key: aggregating several sweeps pairs trials within each sweep instead of
    dropping the cells whose (task, round) repeats. Trials flagged
    ``duplicate`` are left out (:func:`duplicate_mask`).
    """
    if episodes.empty:
        return pd.DataFrame(columns=PAIRED_COLS)
    eps = _without_duplicates(episodes)
    valid = eps[eps["valid"].astype(bool) & eps["condition"].notna()]
    # One outcome per (condition, sweep, task, round); any duplicate left keeps the first trial by name.
    valid = valid.sort_values("trial_name", kind="stable")
    outcome: dict[str, dict[tuple, int]] = {}
    for _, r in valid.iterrows():
        rnd = None if pd.isna(r["round"]) else int(r["round"])
        sweep = r.get("sweep")
        sweep = None if sweep is None or (isinstance(sweep, float) and math.isnan(sweep)) else str(sweep)
        outcome.setdefault(r["condition"], {}).setdefault((sweep, r["task_name"], rnd), int(r["resolved"]))
    rows = []
    baselines = list(baselines)
    for bi, base in enumerate(baselines):
        if base not in outcome:
            continue
        for cond in order_conditions(outcome, baselines):
            # Skip self, and the mirror image of a baseline-vs-baseline pair already listed.
            if cond == base or cond in baselines[:bi]:
                continue
            keys = sorted(set(outcome[cond]) & set(outcome[base]), key=str)
            c = [outcome[cond][k] for k in keys]
            b = [outcome[base][k] for k in keys]
            both = sum(1 for x, y in zip(c, b) if x and y)
            only_c = sum(1 for x, y in zip(c, b) if x and not y)
            only_b = sum(1 for x, y in zip(c, b) if y and not x)
            n = len(keys)
            rows.append({
                "condition": cond, "baseline": base, "n_pairs": n, "both": both, "only_cond": only_c,
                "only_base": only_b, "neither": n - both - only_c - only_b,
                "resolve_rate_cond": sum(c) / n if n else NAN, "resolve_rate_base": sum(b) / n if n else NAN,
                "delta_resolve": (only_c - only_b) / n if n else NAN,
                "mcnemar_p": mcnemar_exact(only_c, only_b) if n else NAN,
            })
    return pd.DataFrame(rows, columns=PAIRED_COLS)


# ---------------------------------------------------------------------------
# calibration.csv
# ---------------------------------------------------------------------------

CALIBRATION_COLS = [
    "model", "provider", "completion_window", "n_calls", "n_ttft", "ttft_median", "ttft_p90",
    "n_decode", "decode_tok_s_median", "ttft_fit_a", "ttft_fit_b_per_1k_prompt_tokens", "ttft_fit_n",
    "lat_fit_a", "lat_fit_b_per_1k_prompt", "lat_fit_c_per_completion_token", "lat_fit_n",
    "latency_median", "prompt_tokens_median", "completion_tokens_median", "n_retried",
]


def latency_fit(calls: Iterable[dict]) -> tuple[float, float, float, int]:
    """``(a, b, c, n)``: least squares of :func:`fit_target_sec` on
    ``a + b * prompt_tokens/1000 + c * completion_tokens`` with a, b, c >= 0."""
    pts = [c for c in calls if c.get("prompt_tokens") is not None and c.get("completion_tokens") is not None
           and math.isfinite(fit_target_sec(c))]
    if not pts:
        return (NAN, NAN, NAN, 0)
    coef, n = nonneg_least_squares(
        [(1.0, float(c["prompt_tokens"]) / 1000.0, float(c["completion_tokens"])) for c in pts],
        [fit_target_sec(c) for c in pts],
    )
    return (float(coef[0]), float(coef[1]), float(coef[2]), n)


def calibration_table(calls: Iterable[dict]) -> pd.DataFrame:
    """Provider latency per (model, provider, completion_window) over successful calls.

    ``lat_fit_*`` is what the ``fitted`` serving profile uses. The TTFT fit and
    the median decode rate are descriptive: a TTFT fit plus one decode rate
    has no per-call decode intercept, so it simulates short replies too fast
    and long ones too slow.
    """
    groups: dict[tuple, list[dict]] = {}
    for c in calls:
        if c.get("error"):
            continue
        groups.setdefault((c.get("model"), c.get("provider"), c.get("completion_window")), []).append(c)
    rows = []
    for (model, provider, window), cs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        ttft = [c.get("ttft_sec") for c in cs]
        decode = [
            c["completion_tokens"] / c["decode_sec"]
            for c in cs
            if c.get("decode_sec") is not None and c["decode_sec"] > 0 and (c.get("completion_tokens") or 0) > 0
        ]
        pts = [((c.get("prompt_tokens") or 0) / 1000.0, c["ttft_sec"]) for c in cs if c.get("ttft_sec") is not None]
        a, b, n_fit = linear_fit([p[0] for p in pts], [p[1] for p in pts])
        la, lb, lc, n_lat = latency_fit(cs)
        rows.append({
            "model": model, "provider": provider, "completion_window": window, "n_calls": len(cs),
            "n_ttft": sum(1 for x in ttft if x is not None), "ttft_median": median(ttft),
            "ttft_p90": quantile(ttft, 0.9), "n_decode": len(decode), "decode_tok_s_median": median(decode),
            "ttft_fit_a": a, "ttft_fit_b_per_1k_prompt_tokens": b, "ttft_fit_n": n_fit,
            "lat_fit_a": la, "lat_fit_b_per_1k_prompt": lb, "lat_fit_c_per_completion_token": lc, "lat_fit_n": n_lat,
            "latency_median": median(measured_call_sec(c) for c in cs),
            "prompt_tokens_median": median(c.get("prompt_tokens") for c in cs),
            "completion_tokens_median": median(c.get("completion_tokens") for c in cs),
            "n_retried": sum(1 for c in cs if (c.get("n_attempts") or 1) > 1),
        })
    return pd.DataFrame(rows, columns=CALIBRATION_COLS)


def nan_to_none(obj: Any) -> Any:
    """Recursively replace NaN/inf/NA with None (valid JSON) and numpy scalars with Python ones."""
    if isinstance(obj, dict):
        return {str(k): nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    if obj is pd.NA or obj is None:
        return None
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
