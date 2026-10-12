# Data dictionary

Every file the harness writes, every record field and every table column.
Times are Unix epoch seconds (float) unless noted; token counts are as the
provider reported them. Ids are per episode: `call-<n>`, `dispatch-<n>`,
`group-<n>`, `arb-<n>`.

## Jobs dir

```
jobs/<sweep>/
  configs/<condition>.yaml            resolved ExperimentConfig the condition's job ran with
  sweep.json                          the sweep's status (below)
  sweep_manifest.jsonl                one line per job event (below)
  <sweep>__r<k>__<condition>/         one harbor job (round k); harbor's config.json, result.json, job.log
    <task>__<suffix>/                 one harbor trial: config.json, result.json, trial.log, verifier/
      agent/                          harbor's logs_dir: this harness's records (below)
  _partial/<job>.<k>/                 a job run moved aside before its rerun (never analysed)
  _oracle/                            preflight --oracle: harbor's oracle agent on every task
  tasks.json                          with a local `dataset_path`: the source dir, its git commit and dirtiness,
                                      and per task a sha256 of the dir harbor runs (`tasks`) and its bugs
                                      (`n_subtasks`: the subtasks in its tests/swesweep_eval.json, else null)
  tasks/<task>/                       with `platform` set: the sweep's tasks copied from `dataset_path` with
                                      their Dockerfiles pinned (the dirs harbor runs)
```

The analysis loads only job dirs whose name does not start with `_` or `.`,
and in them only trial dirs that hold harbor's `config.json`; each task's
bugs come from `tasks.json`.

### harbor's files (read, not written, by the harness)

| File | Fields used |
|---|---|
| job `result.json` | `n_total_trials`, `finished_at` (null while the job runs): a job is complete when both are set and every trial has finished |
| trial `result.json` | `task_name`, `trial_name`, `started_at`, `finished_at`, `exception_info.exception_type`, `exception_info.occurred_at`, `agent_execution.started_at`, `verifier.started_at`, `verifier_environment_mode` (`shared`, or `separate`: the verifier's own container), `verifier_result.rewards.reward`, and from SWE-sweep's grader `verifier_result.rewards.{n_resolved, visible_ok, grader_failed}` |
| trial `config.json` | `task.path` (its last component is the task name) |

harbor timestamps are ISO 8601 strings (`...Z`). The trial's `result.json`
also holds what the agent reported in `agent_result`: token counts and cost
per model (from the condition config's prices) and, in `metadata`, the
episode summary (`episode.json` without `config`).

## Episode records (`<trial>/agent/`)

One file per kind, appended as events happen; every JSONL line starts with
`episode_id`. Records that arrive after the episode is finalized (a thread
still unwinding after harbor's cancel) are dropped.

### `calls.jsonl`: one LLM call (failed ones included)

| Field | Meaning |
|---|---|
| `call_id` | `call-<n>` |
| `role` | `orchestrator`, `subagent`, `arbiter` (judge) or `debate` |
| `agent_id` | `orchestrator`, `sub-<n>` (from `dispatch-<n>`), `arbiter-arb-<n>`; a debate call has its candidate's id |
| `dispatch_id` | the subagent's dispatch (subagent and debate calls), else null |
| `group_id` | the spawn group (`group-<n>`) of subagent, judge and debate calls, else null |
| `arbitration_id` | the arbitration of candidate, judge and debate calls, else null |
| `step` | the agent's own step (1-based); judge and debate calls: the round (1, or 2 after a debate) |
| `model` | model name sent to Sail |
| `t_start` | before the first attempt |
| `t_first_chunk` | first stream chunk carrying content, reasoning or a tool call, in the successful attempt; null for a failed call |
| `t_end` | reply complete, or the call failed |
| `prompt_tokens` | prompt tokens, cached ones included (0 for a failed call) |
| `cached_tokens` | prompt tokens served from the provider's prefix cache |
| `completion_tokens` | output tokens, reasoning included |
| `reasoning_tokens` | reasoning tokens, null when the provider did not report them |
| `finish_reason` | `stop`, `tool_calls`, `length` (cut off at `max_output_tokens`), `unknown` (no finish reason in the stream); null for a failed call |
| `n_tool_calls` | tool calls in the reply after stream repair |
| `n_attempts` | HTTP attempts (1 = no retry); 0 when the call was abandoned on a stop |
| `backoff_sec` | seconds slept between attempts |
| `stream_repairs` | re-sent or clashing tool calls repaired in the stream (docs/DESIGN.md) |
| `error` | null; `cancelled` (the episode stopped during the call); else `<ErrorClass>: <message>` (`ProviderError`, `BillingError`, `ContextLengthError`, ...) |

### `tools.jsonl`: one tool call

| Field | Meaning |
|---|---|
| `role`, `agent_id`, `dispatch_id`, `arbitration_id` | the agent, as in calls |
| `group_id` | the spawn group of a subagent's tool, or of the orchestrator's `spawn_subagents` call |
| `step` | the agent's step |
| `call_id` | the LLM call that asked for the tool |
| `name` | `bash`, `edit_file`, `read_file`, `list_dir`, `search`, `finish`, `spawn_subagents`, `submit`, or an unknown name the model used |
| `t_start`, `t_end` | the tool's run; for `spawn_subagents` the whole spawn |
| `exit_code` | shell exit status of `bash`, `read_file`, `list_dir`, `search`; null for the others |
| `timed_out` | the command hit its timeout and was killed |
| `is_error` | the result was an error (bad arguments, non-zero exit of a read-only tool, unknown tool, ...) |
| `output_chars` | size of the full output before truncation (bytes for shell tools, characters otherwise) |

### `steps.jsonl`: one orchestrator step

| Field | Meaning |
|---|---|
| `step` | 1..`orchestrator_steps` |
| `call_id` | the step's LLM call (null if it failed) |
| `t_start`, `t_llm_end`, `t_end` | step start, LLM call end, end of its last tool |
| `tool_names` | tools the reply asked for, in order |
| `group_ids` | spawn groups started in this step |
| `finished` | `finish` was called |

### `spawns.jsonl`: one `spawn_subagents` call

| Field | Meaning |
|---|---|
| `group_id`, `step` | the group and the orchestrator step |
| `strategy`, `scope` | as configured |
| `requested` | instructions in the call |
| `granted` | instructions run: min(requested, per-call limit, dispatch budget left); the per-call limit is `max_concurrent` (3), or `max_concurrent // n_candidates` (1) for the arbiter, which spends `n_candidates` dispatches per instruction |
| `dispatch_ids` | every dispatch of the group (arbiter: every candidate) |
| `arbitration_ids` | the group's arbitrations (arbiter only) |
| `t_start`, `t_end` | the group's run |

### `dispatches.jsonl`: one subagent run

| Field | Meaning |
|---|---|
| `dispatch_id`, `group_id`, `agent_id` | ids |
| `arbitration_id`, `candidate_index` | the arbitration and the 0-based candidate (arbiter only) |
| `instruction_index` | 0-based position of its instruction in the spawn call |
| `status` | `submitted` (called `submit`), `no_tool_call` (answered without a tool; the text is the report), `max_steps`, `error` (a provider or harness failure), `cancelled` (the episode stopped) |
| `steps_used` | LLM steps taken |
| `t_start`, `t_end` | the run |
| `report_chars` | length of the report returned (`submit` argument, else its last text, else a status note) |
| `files_modified` | files under its start dir newer than a marker made when it started (harness state excluded), sorted, at most 200; empty for easy scope and when the episode stopped. Parallel subagents share the workdir, so each list includes its siblings' changes; arbiter candidates list paths in their private copy |
| `error` | the failure (`<ErrorClass>: ...` or a traceback tail) for status `error`, else null |

### `arbitrations.jsonl`: one arbitrated instruction

| Field | Meaning |
|---|---|
| `arbitration_id`, `group_id`, `instruction_index` | ids |
| `candidate_dispatch_ids` | the candidates' dispatches, in candidate order |
| `isolation` | `copy` (hard scope: private copies) or `shared` (easy, medium) |
| `rounds` | list of `{round, judge_call_ids, debate_call_ids, winner, confidence, agreement, parse_ok}`: judge calls of the round (2 when the first was cut off and retried), the debate calls before it (round 2), the parsed judgement (winner 0-based or null) and whether winner, confidence and agreement all parsed |
| `final_winner` | the candidate applied and reported: the last fully parsed judgement's winner, else the last parsed winner, else 0 |
| `final_confidence` | its confidence (null if unknown) |
| `debate_rounds` | debate rounds run (0 or 1): one runs unless agreement and confidence >= 0.8 |
| `applied` | the winner's copy was applied to the workdir (copy isolation) |
| `n_applied_files` | files the winner changed, added or deleted relative to the pristine workdir |
| `isolation_violated` | a candidate changed the real workdir anyway (restored before the apply); null when not checked (shared, or the episode stopped) |
| `n_leaked_files` | how many workdir files were restored |
| `t_start` | before the copies are made |
| `t_candidates_end` | all candidates done (null if the arbitration stopped earlier) |
| `t_end` | after apply and cleanup |

### `transcript.jsonl`: every message every agent sent or received

| Field | Meaning |
|---|---|
| `agent_id`, `role`, `dispatch_id` | the agent |
| `call_id` | the LLM call that produced an assistant message, else null |
| `t` | when it was recorded |
| `msg_role` | `system`, `user`, `assistant`, `tool` |
| `content`, `reasoning` | text, and the reasoning the model streamed (sent back on later turns) |
| `tool_calls` | `[{id, name, arguments}]` of an assistant message |
| `tool_results` | `[{call_id, content, is_error}]` of a tool message (the step note is appended to the last one) |

### `episode.json`: the episode

Written at the start and rewritten when the episode ends; the totals are
computed from the records, so they equal the JSONL sums.

| Field | Meaning |
|---|---|
| `episode_id` | 12 hex characters |
| `condition` | the config's `name` (the condition) |
| `topology`, `strategy`, `scope` | `single` or `subagents`; strategy and scope null for `single` |
| `orchestrator_model`, `subagent_model` | model names (subagent null for `single`) |
| `task_name`, `trial_name` | from harbor's trial dir |
| `git_commit` | commit of the checkout that ran (null outside git) |
| `config` | the resolved ExperimentConfig |
| `t_start`, `t_end`, `wall_clock_sec` | the agent's run (harbor's setup and verifier excluded) |
| `exit_reason` | `finished`, `max_steps`, `no_tool_call`, `cancelled` (harbor's agent timeout), `provider_error`, `billing_error`, `error` (e.g. the orchestrator's context overflowed) |
| `error` | the first provider or billing failure, the exception, or a traceback tail; null otherwise |
| `orchestrator_steps`, `spawn_calls`, `dispatches`, `arbitrations` | record counts |
| `tokens_by_role` | `{orchestrator, subagent, arbiter, debate}` -> `{prompt, cached, completion, reasoning}` |
| `n_calls` | LLM calls, failed ones included |
| `n_provider_errors` | calls whose error is a `ProviderError` |
| `n_stream_repairs` | sum of the calls' `stream_repairs` |
| `cancel_kill` | after harbor's cancel: `{ok, n_killed, error}` of killing the agent's leftover commands; else null |
| `reward`, `resolved` | null (the verifier's reward is in harbor's `result.json`) |

## Sweep files

`sweep.json` (rewritten at the start and end of every invocation):

| Field | Meaning |
|---|---|
| `name`, `spec` | the sweep and its spec (after `extends`): `dataset` or `dataset_path` (with `platform`), `tasks`, `rounds`, `harbor` (`n_concurrent`, `agent_timeout_multiplier`, `extra_args`), `overrides`, `conditions` |
| `status` | `running`, `complete`, `incomplete`, `aborted: billing error`, `aborted: provider outage`, `interrupted` |
| `git_commit`, `argv` | the code and command line of the last invocation |
| `t_start`, `t_update` | that invocation's start, last write |
| `configs` | condition -> resolved config |

`sweep_manifest.jsonl` (appended): every line has `t`, `sweep`, `event`, `job`,
`round`, `condition`; per event:

| `event` | Extra fields |
|---|---|
| `skipped` | (the job was complete) |
| `moved` | `state` (`unfinished`, `retry`), `why`, `to` (the `_partial/` dir) |
| `run` | `rc` (harbor's exit code), `state` (`complete`, `retry`, `unfinished`, `absent`) and `why` after the run, `duration_sec` |
| `abort` | `code` (3 billing, 4 provider outage), `reason` |
| `interrupted` | (Ctrl-C, SIGTERM or SIGHUP during the job) |

## Analysis tables (`analysis/<sweep>/`)

Written by `python -m subagents_bench.analysis JOBS_DIR... --out DIR`
(`scripts/analyze.sh`). Costs are USD from `configs/pricing.yaml`
(`--scheme`, default `sail_balanced`): (prompt - cached) x `in` + cached x
`cached` + completion x `out`, per 1M tokens; a model without a price costs
NaN, never 0.

Every per-trial and per-record table starts with the **keys**: `sweep`,
`job`, `round`, `condition` (parsed from the job name
`<sweep>__r<k>__<condition>`), `task_name`, `trial_name`.

### `episodes.csv`: one row per trial

| Columns | Meaning |
|---|---|
| keys, `episode_id`, `trial_dir` | ids and the trial's path |
| `topology`, `strategy`, `scope`, `orchestrator_model`, `subagent_model` | from the episode |
| `orchestrator_tools` | `all`, or the tools joined with `+` (a manager) |
| `exit_reason` | the episode's |
| `valid` | the trial measures its condition |
| `invalid_reason` | `no_episode`, `no_result`, `billing_error`, `provider_error` (an exit `provider_error`, harbor's `ProviderUnavailableError`, or any provider error in a call), `harbor_exception` (before the agent started, or a job cancel), `agent_crash` (an exception before the verifier), `grader_failed` (SWE-sweep's grader could not grade), `verifier_failed` (a harbor exception without a reward from a `separate` verifier: its container failed to build or start, it timed out or wrote no reward); empty when valid. Exactly the trials the sweep reruns are invalid, plus those without an episode or result; an agent timeout is valid |
| `harbor_exception` | harbor's exception type (`AgentTimeoutError` on a timeout), else empty |
| `reward`, `resolved` | the verifier's reward (Terminal-Bench: 1 or 0; SWE-sweep: the fraction of the task's bugs fixed, 0 on a regression); resolved = 1 if reward >= 1, else 0 (a valid unscored trial is 0, an invalid one empty) |
| `n_subtasks`, `bugs_fixed`, `regressed` | SWE-sweep only (else empty): the task's bugs (from `tasks.json`, so a trial whose patch breaks the build, which the grader reports as 0 subtasks, keeps them), the grader's `n_resolved` (0 if a visible test regressed, as SWE-sweep's scoring counts it; reward x n_subtasks would carry the reward's rounding to 4 places), and 1 if a visible test regressed (`visible_ok` 0) |
| `t_start`, `wall_clock_sec` | the episode's |
| `steps`, `spawns`, `dispatches`, `arbitrations` | record counts |
| `delegated` | 1 if the orchestrator spawned at least once |
| `n_provider_errors`, `n_stream_repairs` | from the episode and its calls |
| `{orch,sub,arb,deb}_{prompt,cached,completion,reasoning}_tokens` | token sums per role (`arb` = judge, `deb` = debate) |
| `cost_{orch,sub,arb,deb}`, `cost_total` | USD per role and in all |
| `total_tokens` | prompt + completion tokens over all roles |
| `cp_orch_llm_sec`, `cp_sub_llm_sec`, `cp_judge_llm_sec`, `cp_tool_sec` | the measured critical path split into orchestrator LLM, subagent LLM, judge + debate LLM and tool time (docs/DESIGN.md) |
| `cp_total_sec`, `overhead_sec` | their sum, and wall clock minus it |

A trial without an episode has only the keys, ids and outcome columns.

### Record tables

Each row is one record of a trial plus the keys and `valid`:

| Table | Columns after the keys and `valid` |
|---|---|
| `calls.csv` | every `calls.jsonl` field, `cost` |
| `tools.csv` | every `tools.jsonl` field |
| `dispatches.csv` | `strategy`, `scope`, every `dispatches.jsonl` field except `files_modified`, `n_files_modified` |
| `arbitrations.csv` | every `arbitrations.jsonl` field except `rounds` (`candidate_dispatch_ids` joined with `;`), `n_rounds`, `n_judge_calls`, `n_debate_calls`, `all_parsed` (every round parsed) |

### `conditions.csv`: one row per condition (valid trials only)

| Columns | Meaning |
|---|---|
| `condition`, `topology`, `strategy`, `scope`, `orchestrator_tools`, `orchestrator_model`, `subagent_model` | the design |
| `n`, `n_valid`, `n_resolved` | trials, valid trials, solved valid trials |
| `resolve_rate`, `resolve_ci_lo`, `resolve_ci_hi` | n_resolved / n_valid with its Wilson 95% interval |
| `n_delegated`, `delegated_frac` | valid episodes that spawned at least once |
| `n_bugs`, `bugs_fixed`, `bug_fix_rate`, `n_regressed`, `reward_mean` | over the valid trials: bugs in their tasks, bugs fixed, their ratio (SWE-sweep's micro-average; NaN without bugs), trials with a regression, and the mean reward |
| `{cost_total,wall_clock_sec,total_tokens}_{mean,ci_lo,ci_hi}` | mean per valid episode with a seeded percentile bootstrap 95% interval (2000 resamples; no interval for one episode; NaN if any value is NaN) |
| `cost_per_solve` | total cost of the valid episodes / n_resolved (NaN without a solve) |

Rows are ordered: single-agent baselines, the strategy x scope grid, the
same-model control, the managers, then conditions without an episode.

### `summary.json`

| Field | Meaning |
|---|---|
| `jobs_dirs`, `pricing` | inputs: the jobs dirs and `{file, scheme}` |
| `n_trials`, `n_valid`, `invalid_reasons` | trial counts and `{reason: n}` |
| `critical_path` | `{n_valid_episodes, n_over_wall_clock, mean_overhead_sec, mean_overhead_frac}`: valid episodes whose critical path exceeds the wall clock by more than 1 ms (should be 0) and the mean overhead |
| `delegation_gate` | `{ok, never_delegated, no_valid_episodes}`: subagent conditions with valid episodes of which none delegated (the analysis exits 3 when any), and conditions without a valid episode |

## Figures (`figures/<sweep>/`)

`python -m subagents_bench.plots ANALYSIS_DIR --out FIGDIR` draws eight PNGs
from the tables (valid trials only) and writes `index.md` with a caption for
each: `main_grid` (resolve rate with Wilson interval and delegated fraction
per condition), `pareto_cost` and `pareto_time` (resolve rate vs mean cost
and wall clock, with the Pareto front), `time_breakdown` (critical-path parts
and overhead per condition), `token_composition` (tokens and cost by role),
`delegation` (spawns and dispatch outcomes), `arbiter` (judge confidence,
debate rate, isolation violations) and `timeline` (Gantt of the calls and
tools of one delegating episode per strategy).
