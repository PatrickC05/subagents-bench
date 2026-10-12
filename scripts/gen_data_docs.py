#!/usr/bin/env python
"""Generate ``docs/DATA.md``: the data dictionary of everything the study records.

Sources, so the document cannot drift from the code:

* the per-episode record files and every field (type and description) come
  from :mod:`subagents_bench.schema` (``RECORD_TYPES``, ``field_docs``);
* the sweep-level files (``sweep_manifest.jsonl``, ``sweep.json``,
  ``tasks_meta.json``) and the analysis tables (every CSV and every column,
  plus ``summary.json``) are introspected from REAL output: by default the
  script runs the offline pipeline of ``tests/offline_pipeline.py`` (the real
  runtime and the sweep launcher's own writers on local-shell sandboxes with
  scripted models; nothing touches the network or Docker) and aggregates it;
  ``--jobs-dir`` uses an existing jobs dir (e.g. a real sweep) instead.

Every key and column must have a description below (exact name or a
pattern). Anything undocumented is listed and the script exits 1, so new
columns cannot slip into the data unexplained.

Usage::

    .venv/bin/python scripts/gen_data_docs.py                  # writes docs/DATA.md
    .venv/bin/python scripts/gen_data_docs.py --jobs-dir jobs/smoke --out /tmp/DATA.md
    .venv/bin/python scripts/gen_data_docs.py --check          # exit 1 if docs/DATA.md is stale
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import pandas as pd  # noqa: E402

from subagents_bench import schema  # noqa: E402
from subagents_bench.types import Attempt  # noqa: E402

OFFLINE_NOTE = "the offline pipeline of `tests/offline_pipeline.py`: the real runtime and sweep writers with scripted models"
ROLE_SHORT = {"orch": "orchestrator", "sub": "subagent", "arb": "arbiter (judge)", "deb": "debate"}

# ---------------------------------------------------------------------------
# Nested structures inside the record files
# ---------------------------------------------------------------------------

ATTEMPT_DOCS = {
    "attempt": "1-based attempt number within the call.",
    "t_start": "Attempt start (epoch s).",
    "t_end": "Attempt end (epoch s).",
    "status": "HTTP status received; a stream that broke after the headers keeps it (e.g. 200); None only when no response arrived.",
    "error_type": "None on success; else RateLimited, ServerError, Billing, ContextLength, ReadError, ConnectError, TruncatedStream, StreamError, StreamParseError, ... (count transport failures by this, not by status).",
    "error": "Truncated error body or message.",
    "retry_after_sec": "Server-provided Retry-After (seconds), if any.",
    "backoff_sec": "Sleep taken after this attempt before the next one (0 on the last attempt).",
}
ROUND_DOCS = {
    "round": "Judge round (1 = first judgement; each later round follows one debate round).",
    "judge_call_id": "CallRecord of this round's judge call (role arbiter).",
    "debate_call_ids": "CallRecords of the debate calls that preceded this judgement (empty in round 1).",
    "winner": "0-based candidate index the judge picked (0 when unparseable).",
    "confidence": "Judge confidence in [0, 1]; None when unparseable.",
    "agreement": "Judge's verdict on whether the candidates agreed; None when unparseable.",
    "rationale": "Judge's rationale text.",
    "parse_ok": "Whether a complete judgement was parsed (tool call, or JSON in the text).",
    "t_start": "Round start (before its debate calls, if any).",
    "t_end": "Round end (judge call finished).",
}
TOKENS_DOCS = {
    "prompt": "Prompt tokens (cached included).",
    "completion": "Completion tokens (reasoning included).",
    "cached": "Cached prompt tokens.",
    "uncached_prompt": "prompt - cached.",
    "reasoning": "Reasoning tokens; None if the provider never broke them out.",
}
TRANSCRIPT_TOOL_CALL_DOCS = {"id": "Provider tool_call id.", "name": "Tool name.", "arguments": "Arguments object as parsed."}
TRANSCRIPT_TOOL_RESULT_DOCS = {"call_id": "tool_call id answered.", "content": "Text the model saw.", "is_error": "Flagged as an error to the model."}

FILE_INTRO = {
    "episode.json": "One per episode. Written at start with `status: running`, rewritten at the end with `status: done` and every aggregate computed from the JSONL files (so a crash leaves a `running` record).",
    "calls.jsonl": "One line per LLM call, every role (orchestrator steps, subagent steps, judge calls, debate calls), including calls that failed or were abandoned.",
    "tools.jsonl": "One line per tool invocation, including the runtime's own tools (finish, submit, spawn_subagents).",
    "steps.jsonl": "One line per orchestrator step (one LLM call plus the tools it requested).",
    "spawns.jsonl": "One line per spawn_subagents call (a group of dispatches).",
    "dispatches.jsonl": "One line per subagent run (arbiter candidates included).",
    "arbitrations.jsonl": "One line per arbitrated instruction (arbiter strategy).",
    "transcript.jsonl": "Every message every agent saw or produced, in order (system prompts, instructions, assistant turns with reasoning and tool calls, tool results as the model saw them).",
}

# ---------------------------------------------------------------------------
# Sweep-level files
# ---------------------------------------------------------------------------

MANIFEST_DOCS = {
    "sweep": "Sweep name.",
    "job": "Job name `<sweep>__r<round>__<condition>` (= harbor --job-name and the job dir).",
    "round": "Round number (1-based).",
    "condition": "Condition name.",
    "event": "run | skipped_complete | moved_partial | interrupted. Consumers take outcomes from the job's last `run` line.",
    "cmd": "The `harbor run` argv.",
    "config_path": "Materialized resolved config (`<jobs_dir>/configs/<cond>.yaml`).",
    "t_start": "Event start (epoch s).",
    "t_end": "Job end (epoch s); None for non-run events.",
    "duration_sec": "t_end - t_start.",
    "return_code": "harbor's exit code.",
    "n_trials": "Trials in the job (harbor n_total_trials).",
    "n_completed": "Finished trials (harbor counts errored trials as completed).",
    "n_errors": "Errored trials.",
    "complete": "Whether the job counts as complete (see DESIGN section 5).",
    "status_reason": "Why the job is complete or partial.",
    "n_episode_json": "Trial dirs with agent/episode.json.",
    "billing_error_trials": "Trials whose episode ended in a billing error (the sweep aborts).",
    "git_commit": "Harness commit that ran the job.",
    "moved_to": "Where a partial job dir was moved (`moved_partial` lines).",
}
SWEEP_JSON_DOCS = {
    "sweep_json_version": "Format version of this file.",
    "name": "Sweep name.",
    "status": "Outcome of the latest invocation: running | complete | incomplete | interrupted | aborted_billing | aborted_failures.",
    "jobs_total": "Jobs in the whole sweep (rounds x conditions).",
    "jobs_complete": "Of those, complete.",
    "spec_path": "Sweep spec file.",
    "spec_files": "sha256 of the spec and every file in its `extends` chain.",
    "base_config": "Base ExperimentConfig file.",
    "base_config_sha256": "Its sha256.",
    "spec": "The spec after `extends` resolution.",
    "dataset": "harbor dataset name@version.",
    "tasks": "Task names run.",
    "rounds": "Rounds.",
    "order_seed": "Seed of the per-round condition shuffle.",
    "condition_order": "round -> condition order used.",
    "conditions": "condition -> {config_path, config_sha256} of the materialized resolved config.",
    "provider_tiers": "Distinct (model, provider, completion_window) used.",
    "harbor": "harbor settings: env, n_concurrent, agent_timeout_multiplier, extra_args, agent import path, executable.",
    "jobs_dir": "Jobs dir.",
    "tasks_meta_path": "tasks_meta.json path.",
    "manifest_path": "sweep_manifest.jsonl path.",
    "provenance": "Provenance of the latest invocation: invocation_id, date (UTC), git_commit, git_dirty, python_version, python_executable, harbor_version, harbor_executable, lockfile, lockfile_sha256, host_platform, host_machine, docker (`docker info` summary: server_version, architecture, ncpu, mem_total_bytes), argv.",
    "mixed_git_commits": "True if the sweep's jobs span more than one git commit.",
    "invocations": "History: one provenance record (the provenance keys plus status, only, rounds, n_ran, n_skipped, n_failed) per invocation.",
}
TASK_META_DOCS = {
    "difficulty": "Terminal-Bench difficulty label (easy/medium/hard).",
    "category": "Task category.",
    "expert_time_min": "Expert time estimate (minutes).",
    "junior_time_min": "Junior time estimate (minutes).",
    "agent_timeout_sec": "task.toml agent timeout (harbor multiplies it by agent_timeout_multiplier).",
    "verifier_timeout_sec": "task.toml verifier timeout.",
    "docker_image": "Prebuilt task image, if any.",
    "tags": "task.toml tags.",
}
TASKS_META_TOP_DOCS = {
    "dataset": "harbor dataset name@version.",
    "source_dir": "Where the task definitions were read.",
    "generated_at": "When the file was written (UTC).",
    "tasks": "task name -> metadata (below), for the sweep's tasks only.",
}
PREFLIGHT_DOCS = {
    "spec": "Sweep spec checked.",
    "date": "When (UTC).",
    "dotenv": "{path, keys: {name: set|kept|overridden}}; never the values.",
    "docker": "`docker info` summary (local Docker only).",
    "harbor": "{executable, version}.",
    "requests": "One tiny streaming request per distinct (model, provider, tier): model, provider, completion_window, ok, latency_sec, ttft_sec, streamed, n_chunks, n_attempts, finish_reason, prompt/completion/cached/reasoning tokens, usage_source, text_chars (or error_type, error).",
    "oracle": "`--oracle`: {job_dir, reused, return_code, moved_partial_to, mean_reward, failing_tasks, trials}.",
    "checks": "[{name, ok, detail, required}] as printed.",
    "passed": "Every required check passed.",
}
HARBOR_TRIAL_DOCS = {
    "task_name": "Task.",
    "trial_name": "`<task>__<suffix>`.",
    "verifier_result.rewards.reward": "Verifier reward (1.0 = solved) -> episodes.csv reward/resolved.",
    "exception_info.exception_type / occurred_at": "harbor-level exception -> harbor_exception, validity.",
    "environment_setup / agent_setup / agent_execution / verifier .started_at/.finished_at": "Phase timestamps -> env_setup_sec, agent_setup_sec, agent_exec_sec, verifier_sec.",
    "started_at / finished_at": "Whole trial -> trial_total_sec.",
    "agent_result": "harbor's AgentContext as filled by the agent: n_input_tokens, n_cache_tokens, n_output_tokens, cost_usd, model_usage per model, metadata (the episode summary). Redundant with episode.json; not read by the analysis.",
}

# ---------------------------------------------------------------------------
# Analysis tables
# ---------------------------------------------------------------------------

TABLE_INTRO = {
    "episodes.csv": "One row per trial (harbor trial, or bare local episode): design, task covariates, outcome and validity, harbor phases, counts, tokens by role, cost under every pricing scheme, FLOPs, time, the measured critical path and simulated wall time under every serving profile.",
    "calls.csv": "calls.jsonl flattened, one row per LLM call, plus the episode keys, the call's cost under every pricing scheme and its FLOPs. Bulky lists are dropped or summarized.",
    "tools.csv": "tools.jsonl flattened, one row per tool invocation.",
    "steps.csv": "steps.jsonl flattened, one row per orchestrator step.",
    "spawns.csv": "spawns.jsonl flattened, one row per spawn_subagents call.",
    "dispatches.csv": "dispatches.jsonl flattened, one row per subagent run.",
    "arbitrations.csv": "arbitrations.jsonl flattened, one row per arbitrated instruction.",
    "conditions.csv": "One row per condition over its valid episodes: Wilson CI on the resolve rate, mean and bootstrap 95% CI (2000 resamples, seed 0) of every metric, per-solve cost and time. Cost, FLOPs, simulated-time and reasoning-token means are NaN if any valid episode is NaN (an unpriced model), never averaged over the priced subset.",
    "conditions_by_difficulty.csv": "conditions.csv split by Terminal-Bench difficulty (`unknown` when the task has no metadata).",
    "paired.csv": "Each condition vs each baseline condition (default single_5step, single_matched), matched on (task, round) over valid episodes.",
    "calibration.csv": "Provider latency per (model, provider, completion_window) over successful calls; feeds the `fitted` serving profile.",
}

KEY_DOCS = {
    "sweep": "Sweep name (from the job name, else sweep.json, else the jobs dir name).",
    "job_name": "harbor job (`<sweep>__r<round>__<condition>`); empty for bare episode dirs.",
    "round": "Round (from the manifest, else the job name).",
    "condition": "Condition name (manifest, else job name, else the episode's config name).",
    "trial_name": "harbor trial name `<task>__<suffix>`.",
    "task_name": "Task name.",
    "episode_id": "Episode id (episode.json).",
}
EPISODE_DOCS = {
    **KEY_DOCS,
    "topology": "single | subagents.",
    "strategy": "parallel | sequential | arbiter (empty for single).",
    "scope": "easy | medium | hard delegation scope (empty for single).",
    "orchestrator_model": "Orchestrator model id.",
    "subagent_model": "Subagent model id (empty for single).",
    "arbiter_model": "Judge model id (arbiter strategy only).",
    "max_concurrent": "Max subagents live at once.",
    "n_candidates": "Candidates per arbitrated instruction (arbiter only).",
    "orchestrator_steps_budget": "Orchestrator step budget.",
    "orchestrator_tools": "The orchestrator's own tools: `all` (default) or the `tools.orchestrator_tools` list joined by `+` (e.g. `read_file+list_dir+search` for a manager orchestrator that must delegate commands and edits).",
    "difficulty": "Terminal-Bench difficulty label (tasks_meta.json).",
    "category": "Task category (tasks_meta.json).",
    "expert_time_min": "Expert time estimate, minutes (tasks_meta.json).",
    "junior_time_min": "Junior time estimate, minutes (tasks_meta.json).",
    "reward": "Verifier reward (harbor result.json; local runs: episode.json).",
    "resolved": "1 if reward == 1, 0 otherwise; a valid trial the verifier could not score counts 0; empty for invalid trials without a reward.",
    "harbor_exception": "harbor exception type, if any.",
    "exit_reason": "Episode exit reason: finished | max_steps | no_tool_call | token_budget | timeout | cancelled | billing_error | error.",
    "valid": "False for no_episode, billing_error, harbor_exception_outside_agent and no_result (see invalid_reason); verifier exceptions stay valid.",
    "invalid_reason": "Why the trial is invalid (empty when valid).",
    "episode_status": "done | running (crashed before finalizing) | missing.",
    "env_setup_sec": "harbor environment_setup phase seconds.",
    "agent_setup_sec": "harbor agent_setup phase seconds.",
    "agent_exec_sec": "harbor agent_execution phase seconds.",
    "verifier_sec": "harbor verifier phase seconds.",
    "trial_total_sec": "Whole harbor trial seconds.",
    "wall_clock_sec": "Episode wall-clock (episode.json t_end - t_start).",
    "orchestrator_steps": "Orchestrator LLM steps taken.",
    "spawn_calls": "spawn_subagents calls.",
    "dispatches": "Subagent runs (arbiter candidates included).",
    "redispatches": "Dispatches from the 2nd and later spawn calls.",
    "arbitrations": "Arbitrations run.",
    "instruction_following": "Mean granted/requested over spawn calls (empty without spawns).",
    "dispatch_submitted_frac": "Fraction of dispatches that ended by calling submit.",
    "n_files_modified": "Files under the workdir modified during the episode (uncapped count).",
    "total_tokens": "Prompt + completion tokens over every call.",
    "cost_runtime_total": "Sum of the run-time cost_usd the calls logged (config price table); NaN if any call was unpriced.",
    "flops_total": "Sum of flops_<role>.",
    "backoff_sec": "Total retry backoff sleep.",
    "n_retries": "Extra HTTP attempts over all calls.",
    "n_command_timeouts": "Tool commands that hit their timeout.",
    "cp_total_sec": "Measured critical path: the orchestrator chain with every spawn replaced by its group's critical duration (DESIGN section 6).",
    "cp_orch_llm_sec": "Critical-path seconds in orchestrator LLM calls.",
    "cp_sub_llm_sec": "Critical-path seconds in subagent LLM calls.",
    "cp_arb_llm_sec": "Critical-path seconds in judge and debate calls.",
    "cp_tool_sec": "Critical-path seconds in tools (plus arbiter isolation, diff and apply time).",
    "cp_group_sec": "Critical-path seconds inside spawn groups.",
    "overhead_sec": "wall_clock_sec - cp_total_sec: time on no agent's chain.",
    "harness_exec_sec": "Seconds of the harness's own measurement commands (markers, modified-file scans, git diff --stat); explains most of overhead_sec (concurrent ones overlap).",
    "sub_busy_sec": "Sum of dispatch durations (total subagent work).",
    "parallel_speedup": "sub_busy_sec / cp_group_sec (NaN without dispatches or group time). For arbiter groups the denominator also holds isolation, judging and apply.",
    "n_failed_calls": "Calls that failed or were abandoned (error set).",
    "n_usage_estimated": "Calls whose usage was estimated (provider sent none).",
    "container_arch": "uname -m in the container (x86_64 on an arm64 host = emulated).",
    "host_machine": "Host machine (e.g. arm64).",
}
CONDITION_DOCS = {
    "condition": "Condition name.",
    "difficulty": "Terminal-Bench difficulty (conditions_by_difficulty.csv only).",
    "n": "Trials.",
    "n_valid": "Valid trials.",
    "n_resolved": "Resolved valid trials.",
    "resolve_rate": "n_resolved / n_valid.",
    "resolve_ci_lo": "Wilson 95% interval, lower bound.",
    "resolve_ci_hi": "Wilson 95% interval, upper bound.",
    "time_per_solve_sec": "Sum of wall_clock_sec over valid trials / n_resolved (NaN without solves).",
    "cost_per_solve_runtime": "Sum of cost_runtime_total / n_resolved (NaN if any valid trial is unpriced, or no solves).",
}
PAIRED_DOCS = {
    "condition": "Condition compared.",
    "baseline": "Baseline condition.",
    "n_pairs": "(task, round) cells where both have a valid trial.",
    "both": "Pairs both solved.",
    "only_cond": "Pairs only the condition solved.",
    "only_base": "Pairs only the baseline solved.",
    "neither": "Pairs neither solved.",
    "resolve_rate_cond": "Condition's resolve rate over the pairs.",
    "resolve_rate_base": "Baseline's resolve rate over the pairs.",
    "delta_resolve": "(only_cond - only_base) / n_pairs.",
    "mcnemar_p": "Exact two-sided McNemar p-value (binomial on the discordant pairs).",
}
CALIBRATION_DOCS = {
    "model": "Model id.",
    "provider": "Provider.",
    "completion_window": "Sail tier (empty for other providers).",
    "n_calls": "Successful calls.",
    "n_ttft": "Calls with a TTFT.",
    "ttft_median": "Median ttft_sec.",
    "ttft_p90": "90th percentile ttft_sec.",
    "n_decode": "Calls with decode_sec > 0 and completion tokens.",
    "decode_tok_s_median": "Median completion_tokens / decode_sec.",
    "ttft_fit_a": "Least-squares intercept of ttft_sec on prompt tokens / 1000 (seconds).",
    "ttft_fit_b_per_1k_prompt_tokens": "Least-squares slope (seconds per 1000 prompt tokens).",
    "ttft_fit_n": "Points in the fit.",
    "latency_median": "Median call latency_sec (retries included).",
    "prompt_tokens_median": "Median prompt tokens.",
    "completion_tokens_median": "Median completion tokens.",
    "n_retried": "Calls that needed more than one attempt.",
}
RECORD_DERIVED_DOCS = {
    # calls.csv
    "chunk_offsets_n": "Number of chunk arrival offsets logged (chunk_offsets_ms, capped at 4096).",
    "chunk_offset_first_ms": "First delta chunk's offset from the request (ms).",
    "chunk_offset_last_ms": "Last delta chunk's offset from the request (ms).",
    "chunk_gap_ms_median": "Median gap between consecutive delta chunks (ms).",
    "chunk_gap_ms_max": "Largest gap between consecutive delta chunks (ms).",
    "attempt_statuses": "HTTP status of each attempt, ';'-joined.",
    "attempt_error_types": "error_type of each failed attempt, ';'-joined.",
    "flops": "2 x active params x tokens for this call (prompt at the prefill, completion at the decode active-parameter count when models.yaml splits them); NaN for an unknown model.",
    # tools.csv
    "args_preview": "First 300 chars of the JSON arguments.",
    "args_chars": "Length of the JSON arguments.",
    # steps.csv
    "n_spawned_groups": "spawn groups created in the step.",
    # spawns.csv
    "n_instructions": "Instructions requested (= requested).",
    "n_dispatches": "Dispatches created.",
    "n_arbitrations": "Arbitrations created.",
    # dispatches.csv
    "instruction_preview": "First 300 chars of the instruction.",
    "instruction_chars": "Instruction length.",
    "report_preview": "First 300 chars of the report.",
    "report_chars": "Report length.",
    # arbitrations.csv
    "n_rounds": "Judge rounds.",
    "round_winners": "Winner per round, ';'-joined.",
    "round_confidences": "Confidence per round, ';'-joined.",
    "first_confidence": "Confidence of the first judgement.",
    "round_agreements": "Agreement verdict per round, ';'-joined.",
    "judge_call_ids": "Judge call ids, ';'-joined.",
    "n_debate_calls": "Debate calls over all rounds.",
    "all_parse_ok": "Every judgement parsed.",
    "n_applied_files": "Files written or deleted when applying the winner.",
}
LIST_JOINED_NOTE = " (list fields are ';'-joined, dict fields JSON-encoded in the CSV)"
SUMMARY_DOCS = {
    "generated_at": "When aggregate ran (UTC).",
    "harness_version": "subagents_bench version.",
    "inputs": "Jobs/trial/episode dirs aggregated.",
    "files": "pricing, serving and models files used (marked missing if absent).",
    "n_trials": "Trials found.",
    "n_with_episode": "Trials with episode.json.",
    "n_valid": "Valid trials.",
    "n_resolved": "Resolved valid trials.",
    "episode_status_counts": "episode_status -> count.",
    "invalid_reason_counts": "invalid_reason -> count.",
    "harbor_exception_counts": "harbor exception type -> count.",
    "exit_reason_counts": "exit_reason -> count.",
    "sweeps": "Sweep names.",
    "conditions": "Conditions, in table order.",
    "baselines": "Baselines present.",
    "rounds": "Rounds present.",
    "tasks": "Tasks present.",
    "tasks_meta": "datasets, n_tasks_with_meta, tasks_without_meta.",
    "pricing": "schemes (with provenance), aliases, models_without_price per scheme.",
    "models_without_active_params": "Models with no parameter count (their FLOPs and roofline times are NaN).",
    "serving_profiles": "Serving profiles and their parameters/source.",
    "latency_checks": "measured_replay_max_abs_diff_sec, overhead_frac_median, episodes_overhead_over_10pct, unexplained_overhead_frac_median, episodes_unexplained_overhead_over_10pct (overhead not explained by harness_exec_sec).",
    "table_rows": "Rows per table.",
    "warnings": "Loader warnings (missing files, unpriced models, malformed records).",
}


def _schemes_profiles(cols: list[str]) -> tuple[list[str], list[str]]:
    schemes = [m.group(1) for c in cols if (m := re.fullmatch(r"cost_(.+)_total", c)) and m.group(1) != "runtime"]
    profiles = [m.group(1) for c in cols if (m := re.fullmatch(r"sim_(.+)_wall_sec", c))]
    return schemes, profiles


def describe_episode_col(c: str, schemes: list[str], profiles: list[str]) -> str | None:
    if c in EPISODE_DOCS:
        return EPISODE_DOCS[c]
    for s, role in ROLE_SHORT.items():
        kinds = {
            "calls": f"LLM calls by the {role} role.",
            "prompt_tokens": f"Prompt tokens of {role} calls (cached included).",
            "cached_tokens": f"Cached prompt tokens of {role} calls.",
            "uncached_prompt_tokens": f"Uncached prompt tokens of {role} calls.",
            "completion_tokens": f"Completion tokens of {role} calls (reasoning included).",
            "reasoning_tokens": f"Reasoning tokens of {role} calls (NaN if the provider never reported them; 0 if the role made no calls).",
        }
        for k, d in kinds.items():
            if c == f"{s}_{k}":
                return d
        if c == f"flops_{s}":
            return f"FLOPs of {role} calls (2 x active params x tokens); NaN if a model's parameters are unknown."
        if c == f"llm_sec_{s}":
            return f"Summed latency of {role} calls."
        if c == f"tool_sec_{s}":
            return f"Summed duration of tools invoked by the {role} role" + (" (spawn_subagents blocking included)." if s == "orch" else ".")
        for sc in schemes:
            if c == f"cost_{sc}_{s}":
                return f"USD of {role} calls re-priced under scheme `{sc}` (configs/pricing.yaml); NaN if a model is unpriced in it."
    for sc in schemes:
        if c == f"cost_{sc}_total":
            return f"USD of all calls under scheme `{sc}`; NaN if any model is unpriced in it."
    for p in profiles:
        if c == f"sim_{p}_wall_sec":
            return (f"Simulated episode wall time under serving profile `{p}` (configs/serving.yaml): the critical path with every LLM call "
                    "re-timed by the profile and tools kept as measured" + ("; equals cp_total_sec." if p == "measured" else "."))
    return None


def describe_condition_col(c: str, schemes: list[str], profiles: list[str]) -> str | None:
    if c in CONDITION_DOCS:
        return CONDITION_DOCS[c]
    if c in EPISODE_DOCS and c not in KEY_DOCS:  # design columns
        return EPISODE_DOCS[c]
    for sc in schemes:
        if c == f"cost_per_solve_{sc}":
            return f"Sum of cost_{sc}_total over valid trials / n_resolved (NaN if any is unpriced, or no solves)."
    m = re.fullmatch(r"(.+)_(mean|ci_lo|ci_hi)", c)
    if m:
        metric, kind = m.groups()
        base = describe_episode_col(metric, schemes, profiles)
        if base is not None:
            what = {"mean": "Mean", "ci_lo": "Bootstrap 95% CI lower bound of the mean", "ci_hi": "Bootstrap 95% CI upper bound of the mean"}[kind]
            return f"{what} of episodes.csv `{metric}` over valid trials."
    return None


def describe_record_col(table: str, c: str, schemes: list[str], record_docs: dict[str, str]) -> str | None:
    if c in KEY_DOCS:
        return KEY_DOCS[c]
    if c in RECORD_DERIVED_DOCS:
        return RECORD_DERIVED_DOCS[c]
    if table == "calls.csv":
        for sc in schemes:
            if c == f"cost_{sc}":
                return f"This call's USD under scheme `{sc}`; NaN if its model is unpriced in it."
    if c in record_docs:
        return record_docs[c]
    return None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _md(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_md(x) for x in r) + " |" for r in rows]
    return out


def keys_of(path: Path) -> list[str]:
    obj = json.loads(path.read_text())
    return list(obj)


class Missing(list):
    def need(self, where: str, key: str, doc: str | None) -> str:
        if doc is None:
            self.append(f"{where}: {key}")
            return "**UNDOCUMENTED**"
        return doc


def build(jobs_dir: Path, analysis_dir: Path, source_note: str) -> tuple[str, list[str]]:
    missing = Missing()
    L: list[str] = []
    L += [
        "# Data dictionary",
        "",
        "<!-- Generated by scripts/gen_data_docs.py; do not edit by hand. -->",
        "",
        "Everything the study records, file by file and field by field: the per-episode records written by the",
        "runtime, the sweep-level files written by the launcher, the harbor files the analysis reads, and every",
        "table and column `subagents-bench aggregate` derives. Record fields and their descriptions come from",
        "`subagents_bench/schema.py`; the sweep files and analysis columns were introspected from real output",
        f"({source_note}).",
        "",
        "Conventions:",
        "",
        "- Timestamps are `time.time()` epoch seconds (harbor's own files use ISO 8601); durations are seconds.",
        "- Token counts: `prompt_tokens` includes `cached_tokens`; `completion_tokens` includes `reasoning_tokens`.",
        "- Unknown or not observable is `null` in the records and empty/NaN in the CSVs, never a made-up 0.",
        "  An observed zero (a role that made no calls) is 0.",
        "- Roles: `orchestrator`, `subagent`, `arbiter` (judge calls), `debate` (candidates revising their",
        "  reports). CSV columns use the short forms `orch`, `sub`, `arb`, `deb`.",
        "- Costs: the run-time `cost_usd` uses the price table in the run's config (a billed estimate). The",
        "  study's costs are recomputed offline under every scheme of `configs/pricing.yaml`.",
        "- Raw provider data is kept (`usage_raw`, `response_headers`, `attempts`, `chunk_offsets_ms`), so cost",
        "  and latency can be recomputed later under other assumptions.",
        "",
        "## Where everything lands",
        "",
        "```",
        "jobs/<sweep>/                          # one sweep (scripts/run_experiment.sh)",
        "  sweep.json                           # spec, resolved configs, provenance, invocation history",
        "  sweep_manifest.jsonl                 # one line per job event",
        "  tasks_meta.json                      # task covariates (difficulty, category, expert time, ...)",
        "  preflight.json                       # preflight checks and the per-model test requests",
        "  configs/<condition>.yaml             # the resolved ExperimentConfig of each condition",
        "  <sweep>__r<round>__<condition>/      # one harbor job",
        "    result.json, config.json, job.log  # harbor's",
        "    <task>__<suffix>/                  # one harbor trial",
        "      result.json, config.json, trial.log, verifier/   # harbor's (reward, phases, exception)",
        "      agent/                           # the episode dir: the runtime's records below",
        "        episode.json calls.jsonl tools.jsonl steps.jsonl spawns.jsonl",
        "        dispatches.jsonl arbitrations.jsonl transcript.jsonl",
        "  _partial/<job>.partial.<k>/          # partial jobs moved aside on resume (ignored by analysis)",
        "  _oracle/                             # preflight --oracle feasibility run (ignored by analysis)",
        "analysis/<sweep>/                      # scripts/analyze.sh: the CSVs below + summary.json",
        "figures/<sweep>/                       # PNG + PDF figures and index.md (captions)",
        "```",
        "",
        "## Per-episode records (`<trial>/agent/`)",
        "",
    ]
    for fname, cls in schema.RECORD_TYPES.items():
        L += [f"### `{fname}`", "", FILE_INTRO[fname], ""]
        L += table([(f"`{n}`", t, d) for n, t, d in schema.field_docs(cls)], ("field", "type", "description"))
        L.append("")
        if fname == "calls.jsonl":
            L += ["Each element of `attempts`:", ""]
            L += table([(f"`{f}`", "", ATTEMPT_DOCS[f]) for f in Attempt.__dataclass_fields__], ("key", "", "description"))
            L.append("")
        if fname == "arbitrations.jsonl":
            L += ["Each element of `rounds`:", ""]
            L += table([(f"`{k}`", "", d) for k, d in ROUND_DOCS.items()], ("key", "", "description"))
            L.append("")
        if fname == "episode.json":
            L += ["Each value of `tokens` (per role):", ""]
            L += table([(f"`{k}`", "", d) for k, d in TOKENS_DOCS.items()], ("key", "", "description"))
            L.append("")
        if fname == "transcript.jsonl":
            L += ["`tool_calls` elements: " + "; ".join(f"`{k}` {d}" for k, d in TRANSCRIPT_TOOL_CALL_DOCS.items()),
                  "",
                  "`tool_results` elements: " + "; ".join(f"`{k}` {d}" for k, d in TRANSCRIPT_TOOL_RESULT_DOCS.items()),
                  ""]

    # --- harbor's files ---------------------------------------------------------
    L += ["## harbor's files (read by the analysis)", "",
          "harbor 0.24.0 writes `result.json` per trial and per job. The analysis reads these trial fields:", ""]
    L += table([(f"`{k}`", d) for k, d in HARBOR_TRIAL_DOCS.items()], ("field", "use"))
    L += ["", "The sweep launcher reads the job `result.json` (`n_total_trials`, `stats.n_completed_trials`, "
          "`n_errored_trials`, `n_running_trials`, `n_pending_trials`, `finished_at`) to decide whether a job is complete.", ""]

    # --- sweep files --------------------------------------------------------------
    L += ["## Sweep-level files (`jobs/<sweep>/`)", ""]
    man = jobs_dir / "sweep_manifest.jsonl"
    man_keys: list[str] = []
    for line in man.read_text().splitlines():
        for k in json.loads(line):
            if k not in man_keys:
                man_keys.append(k)
    for k in MANIFEST_DOCS:  # keys only some events carry (moved_to, ...)
        if k not in man_keys:
            man_keys.append(k)
    L += ["### `sweep_manifest.jsonl`", "", "One line per job event, appended as the sweep runs.", ""]
    L += table([(f"`{k}`", missing.need("sweep_manifest.jsonl", k, MANIFEST_DOCS.get(k))) for k in man_keys], ("key", "description"))
    sj = json.loads((jobs_dir / "sweep.json").read_text())
    L += ["", "### `sweep.json`", "", "Rewritten at the start and end of every sweep invocation.", ""]
    L += table([(f"`{k}`", missing.need("sweep.json", k, SWEEP_JSON_DOCS.get(k))) for k in sj], ("key", "description"))
    tm = json.loads((jobs_dir / "tasks_meta.json").read_text())
    L += ["", "### `tasks_meta.json`", ""]
    L += table([(f"`{k}`", missing.need("tasks_meta.json", k, TASKS_META_TOP_DOCS.get(k))) for k in tm], ("key", "description"))
    task_keys = list(next(iter(tm["tasks"].values())))
    L += ["", "Per task:", ""]
    L += table([(f"`{k}`", missing.need("tasks_meta.json tasks", k, TASK_META_DOCS.get(k))) for k in task_keys], ("key", "description"))
    L += ["", "### `preflight.json`", "", "Written by `subagents-bench preflight` (not part of the offline introspection).", ""]
    L += table([(f"`{k}`", d) for k, d in PREFLIGHT_DOCS.items()], ("key", "description"))
    L += ["", "### `configs/<condition>.yaml`", "",
          "The fully resolved `ExperimentConfig` of each condition (every default explicit); see "
          "`docs/DESIGN.md` section 2 and `subagents_bench/config.py`. The same dict is stored in each "
          "`episode.json` under `config`, with its hash in `config_hash`.", ""]

    # --- analysis tables ------------------------------------------------------------
    L += ["## Analysis tables (`analysis/<sweep>/`)", "",
          "Written by `subagents-bench aggregate` (DESIGN section 6). Every table is written, header only when empty.", ""]
    ep_cols = list(pd.read_csv(analysis_dir / "episodes.csv", nrows=0).columns)
    schemes, profiles = _schemes_profiles(ep_cols)
    L += [f"Pricing schemes in this output: {', '.join(f'`{s}`' for s in schemes)}. Serving profiles: "
          f"{', '.join(f'`{p}`' for p in profiles)}. Column sets that repeat per scheme, profile or role follow "
          "the same pattern for any other scheme or profile configured.", ""]
    record_cls = {"calls.csv": schema.CallRecord, "tools.csv": schema.ToolRecord, "steps.csv": schema.StepRecord,
                  "spawns.csv": schema.SpawnRecord, "dispatches.csv": schema.DispatchRecord,
                  "arbitrations.csv": schema.ArbitrationRecord}
    order = ["episodes.csv", "calls.csv", "tools.csv", "steps.csv", "spawns.csv", "dispatches.csv",
             "arbitrations.csv", "conditions.csv", "conditions_by_difficulty.csv", "paired.csv", "calibration.csv"]
    found = sorted(p.name for p in analysis_dir.glob("*.csv"))
    for extra in found:
        if extra not in order:
            missing.append(f"table {extra}")
    for name in order:
        cols = list(pd.read_csv(analysis_dir / name, nrows=0).columns)
        L += [f"### `{name}`", "", TABLE_INTRO[name] + (LIST_JOINED_NOTE if name in record_cls else ""), ""]
        rows = []
        for c in cols:
            if name == "episodes.csv":
                d = describe_episode_col(c, schemes, profiles)
            elif name in record_cls:
                docs = {n: doc for n, _, doc in schema.field_docs(record_cls[name])}
                d = describe_record_col(name, c, schemes, docs)
            elif name.startswith("conditions"):
                d = describe_condition_col(c, schemes, profiles)
            elif name == "paired.csv":
                d = PAIRED_DOCS.get(c)
            else:
                d = CALIBRATION_DOCS.get(c)
            rows.append((f"`{c}`", missing.need(name, c, d)))
        L += table(rows, ("column", "description"))
        if name in record_cls:
            cls = record_cls[name]
            dropped = [n for n, _, _ in schema.field_docs(cls) if n not in cols and n != "episode_id"]
            if dropped:
                L += ["", "Not in the CSV (bulky; read the JSONL): " + ", ".join(f"`{d}`" for d in dropped) + "."]
        L.append("")
    summary = json.loads((analysis_dir / "summary.json").read_text())
    L += ["### `summary.json`", "", "Run-level summary written next to the tables.", ""]
    L += table([(f"`{k}`", missing.need("summary.json", k, SUMMARY_DOCS.get(k))) for k in summary], ("key", "description"))
    L += ["", "## Figures (`figures/<sweep>/`)", "",
          "`subagents-bench plot` renders the 14 figures of DESIGN section 7 from these tables; `index.md` in the "
          "figure dir captions each one and names the columns it uses.", ""]
    return "\n".join(L), list(missing)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs-dir", help="existing jobs dir to introspect (default: run the offline pipeline)")
    ap.add_argument("--out", default=str(REPO / "docs" / "DATA.md"))
    ap.add_argument("--check", action="store_true", help="exit 1 if --out differs from what would be generated")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="gen-data-docs-") as tmp:
        tmp_p = Path(tmp)
        if args.jobs_dir:
            jobs = Path(args.jobs_dir).resolve()
            note = f"`{args.jobs_dir}`"
        else:
            from offline_pipeline import build_offline_sweep  # noqa: PLC0415

            jobs = build_offline_sweep(tmp_p / "offline").jobs_dir
            note = OFFLINE_NOTE
        analysis = tmp_p / "analysis"
        rc = subprocess.run([sys.executable, "-m", "subagents_bench.cli", "aggregate", str(jobs), "--out", str(analysis)],
                            cwd=REPO, capture_output=True, text=True)
        if rc.returncode != 0:
            print(rc.stdout + rc.stderr, file=sys.stderr)
            return 1
        text, missing = build(jobs, analysis, note)
    if missing:
        print("undocumented keys/columns (add them to scripts/gen_data_docs.py):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
    out = Path(args.out)
    if args.check:
        same = out.is_file() and out.read_text() == text + "\n"
        print("docs/DATA.md is " + ("up to date" if same else "STALE: re-run scripts/gen_data_docs.py"))
        return 0 if same and not missing else 1
    out.write_text(text + "\n")
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
