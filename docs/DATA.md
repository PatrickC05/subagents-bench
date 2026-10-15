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
      verifier/eval.json              SWE-sweep's grader: per-subtask verdicts (read by the interactivity analysis)
  _partial/<job>.<k>/                 a job run moved aside before its rerun (never analysed)
  _oracle/                            preflight --oracle: harbor's oracle agent on every task
  tasks.json                          with a local `dataset_path`: the source dir, its git commit and dirtiness,
                                      and per task a sha256 of the dir harbor runs (`tasks`) and its bugs
                                      (`n_subtasks`: the subtasks in its tests/swesweep_eval.json, else null)
  tasks/<task>/                       with `platform` set: the sweep's tasks copied from `dataset_path` with
                                      their Dockerfiles pinned (the dirs harbor runs)
  hints/<task>.json                   interactive sweeps with `hint_detail: report`: {task, model, generated_at,
                                      reports: {subtask id: bug report}}, from `subagents-bench hints`
  hints/<task>.questions.json         interactive sweeps with `questions_per_interval`: {task, model, generated_at,
                                      questions: [{id (q1, q2, ...), file, question, answer}]}, from
                                      `subagents-bench questions` (`answer`: the reference answer)
  src/<task>/                         `subagents-bench questions`: a shallow clone of the task's repository at the
                                      ref its Dockerfile checks out (the source the questions are written from)
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
| `one_call_retry` | the reply came from a retry that asked for exactly one tool call, after a stream error once a tool call had started (Sail's Gemma-4 parsing of several tool calls in one reply; docs/DESIGN.md) |
| `error` | null; `cancelled` (the episode stopped during the call); else `<ErrorClass>: <message>` (`ProviderError`, `BillingError`, `ContextLengthError`, ...) |

### `tools.jsonl`: one tool call

| Field | Meaning |
|---|---|
| `role`, `agent_id`, `dispatch_id`, `arbitration_id` | the agent, as in calls |
| `group_id` | the spawn group of a subagent's tool, or of the orchestrator's `spawn_subagents` call |
| `step` | the agent's step |
| `call_id` | the LLM call that asked for the tool |
| `name` | `bash`, `edit_file`, `read_file`, `list_dir`, `search`, `finish`, `spawn_subagents`, `wait` (interactive; also written, with `arguments` `{"implicit": true}` and no `tool_call_id`, for the wait after a reply without a tool call), `submit`, or an unknown name the model used |
| `t_start`, `t_end` | the tool's run; for a blocking `spawn_subagents` the whole spawn, for a background one only starting it |
| `exit_code` | shell exit status of `bash`, `read_file`, `list_dir`, `search`; null for the others |
| `timed_out` | the command hit its timeout and was killed |
| `is_error` | the result was an error (bad arguments, non-zero exit of a read-only tool, unknown tool, ...) |
| `output_chars` | size of the full output before truncation (bytes for shell tools, characters otherwise) |
| `tool_call_id` | the model's id of the tool call (in the transcript's `tool_calls`) |
| `arguments` | the call's arguments, each string cut to 2000 characters (the transcript has them whole); the interactivity analysis matches its paths and commands |

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
| `spawn_mode` | `blocking` (the call returns with every report), or `background` (interactive episodes: it returns at once and each report is delivered through the inbox) |
| `requested` | instructions in the call |
| `granted` | instructions run: min(requested, per-call limit, dispatch budget left); the per-call limit is `max_concurrent` (3), or `max_concurrent // n_candidates` (1) for the arbiter, which spends `n_candidates` dispatches per instruction |
| `dispatch_ids` | every dispatch of the group (arbiter: every candidate) |
| `arbitration_ids` | the group's arbitrations (arbiter only) |
| `t_start`, `t_end` | the group's run (background: only starting it) |

Background spawns grant min(requested, free slots = `max_concurrent` minus
the subagents still running, dispatch budget left).

### `dispatches.jsonl`: one subagent run

| Field | Meaning |
|---|---|
| `dispatch_id`, `group_id`, `agent_id` | ids |
| `arbitration_id`, `candidate_index` | the arbitration and the 0-based candidate (arbiter only) |
| `instruction_index` | 0-based position of its instruction in the spawn call |
| `instruction` | its instruction, cut to 4000 characters |
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

### `inbox.jsonl`: one delivery to the orchestrator (interactive episodes)

Empty in batch episodes. One line per scheduled user message (a bug hint, and
with `interactive.replies` a question or a status check) and per background
subagent report, written when it is delivered, plus one per message or report
never delivered (the session ended first), written when the episode ends.

| Field | Meaning |
|---|---|
| `kind` | `hint` (a bug), `question`, `status` (a status check) or `report` |
| `t_ready` | a message: when it was sent (a bug: session start + `hint_index` x `interval_sec`; questions and status checks: docs/DESIGN.md, *Requests*); a report: when its subagent ended |
| `t_delivered` | when it entered the orchestrator's conversation (before a step's LLM call, never after the last step); null if never delivered |
| `step` | the orchestrator step whose LLM call saw it first (1 for hint 0, in the first message); null if never delivered |
| `hint_index` | a message's 0-based position among the messages of its kind (bugs, questions, status checks); null for a report |
| `subtask_id`, `target_files`, `f2p` | a bug's SWE-sweep subtask: id, the files the reference fix changes, the hidden tests that fail before it (null otherwise) |
| `dispatch_id`, `agent_id`, `status` | a report's subagent and its dispatch status (null for a hint) |
| `chars` | length of the hint or question text (0 for a status check) or of the full report |
| `msg_id` | with `interactive.replies`: the message's id (1, 2, ... in sending order; what `reply` names); else null, and null for a report |
| `urgent` | a bug marked urgent (`urgent_every`); false otherwise (absent in records written before it existed) |
| `question_id`, `question`, `reference` | a question's id in `questions.json`, its text and the reference answer; null otherwise |

### `replies.jsonl`: one `reply` call (interactive episodes with `interactive.replies`)

Only valid calls (`to` a delivered message's id, non-empty `text`) are
written; a rejected call is an errored tool record. Empty otherwise.

| Field | Meaning |
|---|---|
| `t` | when it was sent: the time the user got it |
| `step` | the orchestrator step |
| `to` | the `msg_id` it answers |
| `kind`, `hint_index` | that message's kind (`bug`, `question`, `status`: note `bug`, where the inbox says `hint`) and position |
| `subtask_id`, `question_id` | that message's subtask (a bug) or question, else null |
| `text` | the message to the user |
| `eta_min` | minutes from now until the bug is fixed, as a float; null when not given or not a number in [0, 10^6) |
| `done` | true when the call passed `done: true` (or `"true"`), else false |

### `answers.json`: verdicts on the answers (`subagents-bench grade-answers`)

Written after the run, for trials with questions; absent until then.
`{judge_model, graded_at, answers: {msg_id (string): {correct, why}}}`: per
question answered, the judge's verdict on its first reply against the
reference (`correct` null when the judge's reply could not be read) and its
one-sentence reason. Unanswered questions are not listed.

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
| `exit_reason` | `finished`, `max_steps`, `no_tool_call`, `cancelled` (harbor's agent timeout), `provider_error`, `billing_error`, `error` (e.g. the orchestrator's context overflowed); interactive episodes: `deadline` (the session's wall clock ran out), `idle` (every hint was sent, nothing ran and the orchestrator waited), `max_steps` (the safety cap) |
| `error` | the first provider or billing failure, the exception, or a traceback tail; null otherwise |
| `orchestrator_steps`, `spawn_calls`, `dispatches`, `arbitrations` | record counts |
| `tokens_by_role` | `{orchestrator, subagent, arbiter, debate}` -> `{prompt, cached, completion, reasoning}` |
| `n_calls` | LLM calls, failed ones included |
| `n_provider_errors` | calls whose error is a `ProviderError` |
| `n_stream_repairs` | sum of the calls' `stream_repairs` |
| `cancel_kill` | after harbor's cancel: `{ok, n_killed, error}` of killing the agent's leftover commands; else null |
| `reward`, `resolved` | null (the verifier's reward is in harbor's `result.json`) |

Interactive episodes add:

| Field | Meaning |
|---|---|
| `session_t0` | the session's start: hint k is sent at `session_t0` + k x `interval_sec` |
| `hint_order_seed` | the seed of the subtasks' order: `order_seed` + round - 1 with `vary_order_by_round` (harbor runs) |
| `n_hints`, `n_hints_delivered` | bug hints scheduled, and delivered before the session ended (questions and status checks not counted) |
| `n_questions`, `n_replies` | questions scheduled (inbox records of kind `question`), and `reply` calls written |
| `n_elided_results` | older tool outputs, subagent reports and reasoning replaced or dropped to stay within `max_context_chars` |
| `end_kill` | `{ok, n_killed, error}` of killing the commands left running when the session ended (harbor runs only) |

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
| `reward`, `resolved` | the verifier's reward (Terminal-Bench: 1 or 0; SWE-sweep: the fraction of the task's bugs fixed, 0 on a regression); resolved = 1 if reward >= 1, else 0. A valid trial without a reward (a verifier in the agent's container failed after it started) has reward 0 and resolved 0 (and on SWE-sweep bugs_fixed 0); an invalid one without a reward has both empty |
| `n_subtasks`, `bugs_fixed`, `regressed` | SWE-sweep only (else empty): the task's bugs (from `tasks.json`, so a trial whose patch breaks the build, which the grader reports as 0 subtasks, keeps them), the grader's `n_resolved` (0 if a visible test regressed, as SWE-sweep's scoring counts it; reward x n_subtasks would carry the reward's rounding to 4 places), and 1 if a visible test regressed (`visible_ok` 0) |
| `t_start`, `wall_clock_sec` | the episode's |
| `interactive`, `n_hints`, `n_hints_delivered` | 1 for an interactive session, else 0; its hints scheduled and delivered (empty in batch episodes) |
| `steps`, `spawns`, `dispatches`, `arbitrations` | record counts |
| `delegated` | 1 if the orchestrator spawned at least once |
| `orch_bash_n`, `orch_bash_median_sec`, `orch_bash_max_sec`, `orch_llm_median_sec` | the orchestrator's own bash commands (count, median and longest duration) and the median duration of its successful LLM calls: what a message waits behind between its steps |
| `n_fairplay_flags`, `fairplay_kinds` | tool calls of any agent matching a fairplay pattern, and their categories joined with `;` (`git_history`, `network`, `installed_copy`; analysis/fairplay.py): to review, not verdicts |
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
| `tools.csv` | every `tools.jsonl` field except `arguments`, `fairplay` (the fairplay categories its arguments match, `;`-joined) |
| `dispatches.csv` | `strategy`, `scope`, every `dispatches.jsonl` field except `files_modified` and `instruction`, `n_files_modified` |
| `arbitrations.csv` | every `arbitrations.jsonl` field except `rounds` (`candidate_dispatch_ids` joined with `;`), `n_rounds`, `n_judge_calls`, `n_debate_calls`, `all_parsed` (every round parsed) |

### `conditions.csv`: one row per condition (valid trials only)

| Columns | Meaning |
|---|---|
| `condition`, `topology`, `strategy`, `scope`, `orchestrator_tools`, `orchestrator_model`, `subagent_model` | the design |
| `n`, `n_valid`, `n_resolved` | trials, valid trials, solved valid trials |
| `resolve_rate`, `resolve_ci_lo`, `resolve_ci_hi` | n_resolved / n_valid with its Wilson 95% interval |
| `n_delegated`, `delegated_frac` | valid episodes that spawned at least once |
| `n_fairplay_flagged` | valid episodes with at least one fairplay flag |
| `n_bugs`, `bugs_fixed`, `bug_fix_rate`, `n_regressed`, `reward_mean` | over the valid trials, a trial without a reward counting as reward 0 with no bug fixed: bugs in their tasks, bugs fixed, their ratio (SWE-sweep's micro-average; NaN without bugs), trials with a regression, and the mean reward |
| `{cost_total,wall_clock_sec,total_tokens}_{mean,ci_lo,ci_hi}` | mean per valid episode with a seeded percentile bootstrap 95% interval (2000 resamples; no interval for one episode; NaN if any value is NaN) |
| `cost_per_solve` | total cost of the valid episodes / n_resolved (NaN without a solve) |

Rows are ordered: single-agent baselines, the strategy x scope grid, the
same-model control, the managers, then conditions without an episode.

### `hints.csv`: one row per hint of an interactive episode

Built by `analysis/interactivity.py` from `inbox.jsonl`, the steps, tools and
dispatches, and the grader's per-subtask verdicts in
`<trial>/verifier/eval.json`. Every latency is seconds after the hint was
sent (`t_ready`), empty when the event never happened. Path matching is a
heuristic (docs/DESIGN.md).

| Columns | Meaning |
|---|---|
| keys, `valid`, `topology`, `spawn_mode`, `hint_detail`, `interval_sec` | the trial and its design (`spawn_mode` empty for a single agent) |
| `hint_index`, `urgent`, `subtask_id`, `target_files` (`;`-joined), `n_target_files` | the hint (urgent: `urgent_every`) and its subtask |
| `scope` | `interactive.scope`: `reported` (the session fixes the bugs the user sends) or `all` |
| `shared_files` | one of its target files is also a target of an earlier hint's subtask (its touch and edit latencies can come from that subtask's work) |
| `shares_any` | one of its target files is a target of any other hint's subtask |
| `t_ready_sec` | when it was sent, after the session start |
| `delivered` | it reached the orchestrator before the session ended |
| `blocked_by`, `blocked_sec` | when `busy_at_arrival` is `llm`, `tool` or `spawn`: the call (`llm`) or tool it waited behind, and how long that still ran after the hint was sent (`delivery_sec` adds the rest of the step) |
| `busy_at_arrival` | what the orchestrator was doing when it was sent: `llm` (a call), `tool` (its own tool), `spawn` (a blocking spawn), `wait` (a `wait` call, or the implicit wait after a reply without a tool call), `idle` (between steps: delivery and bookkeeping between an LLM call and its tools), `start` (hint 0) |
| `delivery_sec` | until it entered the conversation |
| `response_sec` | until the first orchestrator LLM call that saw it returned |
| `dispatch_sec` | until a subagent started whose instruction names a target file or one of the hidden tests' names |
| `touch_sec`, `touch_by` | until any agent's tool call named a target file (`read_file`/`edit_file`/`list_dir`/`search` path, or a `bash` command), and that agent's role. A bare basename counts only when no other hinted target file shares it |
| `edit_sec` | until an edit of a target file: `edit_file` on it, or a `bash` command segment (split at unquoted `;`, `&&`, `\|\|`, `\|`, newlines; a command with a heredoc is one segment) naming it that writes (`sed -i`/`--in-place`, `perl -i`, a redirection or `tee` to it, `patch`, `git apply`, `gofmt -w`, `cp`/`mv`/`install` onto it, a Python write) |
| `pre_touched`, `pre_edited` | a target file was named, or edited, before the hint was sent (the agent got to the bug early; with `scope: reported` it was not asked to) |
| `status_at_next` | when the next hint was sent: `not_started` (no touch since this hint was sent), `continued` (touched before and again after), `stopped` (touched before, never after); empty for the last hint |
| `continued_after_next` | `status_at_next` is `continued` (empty when not started or the last hint) |
| `graded`, `resolved`, `partial`, `counted` | the grader reported a verdict for the subtask; its verdict; the grader's `partial` flag (some of the subtask's hidden tests pass, none of its other tests broke); resolved with no visible regression (what the reward counts). In a valid trial a subtask without a verdict (no `eval.json`, or a patch that broke the build, for which the grader lists no subtasks) is unfixed: false; in an invalid trial empty |

### `interactivity.csv`: one row per interactive condition (valid trials only)

Latency columns are over the hints after the first (hint 0 arrives with the
task, while the agent starts up; it has the `first_*` columns).

| Columns | Meaning |
|---|---|
| `condition`, `topology`, `spawn_mode`, `n_episodes`, `n_hints`, `n_later_hints` | the design and the counts (all hints, hints after the first) |
| `delivered_frac` | later hints delivered before the session ended |
| `busy_at_arrival_frac` | later hints that arrived while the orchestrator was in an LLM call, its own tool or a blocking spawn |
| `pre_edited_frac` | later hints whose files were edited before they were sent; left out of the touch and edit columns |
| `blocked_median_sec` | median `blocked_sec` over the later hints that arrived while the orchestrator was busy |
| `first_{response,touch,edit}_median_sec` | hint 0's medians (censored as below) |
| `{m}_frac`, `{m}_median_sec`, `{m}_mean_sec`, `{m}_mean_ci_lo`, `{m}_mean_ci_hi`, `{m}_before_next` | for m in `delivery`, `response`, `dispatch`, `touch`, `edit`: the share of hints where it happened; the median counting a missing one as infinitely late (NaN when more than half are missing); the mean (bootstrap 95% interval) over those where it happened; the share where it happened within one `interval_sec` (before the next hint) |
| `{touch,edit}_median_sec_unshared`, `{touch,edit}_before_next_unshared` | the same over hints without `shared_files` |
| `n_status_unshared`, `{not_started,continued,stopped}_at_next_frac` | over hints (hint 0 included, the last excluded) that share no file with any other hint: their `status_at_next` |
| `stopped_fixed_frac` | of those stopped, the share counted fixed in the end (finished, not dropped) |
| `n_graded`, `hinted_fixed_frac` | hints (all) with a grader verdict, and the share of all hints counted fixed (a hint without a verdict in a valid trial counts as unfixed) |

### `hint_positions.csv`: one row per interactive condition and hint position

`condition`, `hint_index`, `n`, `{delivery,response,dispatch,touch,edit}_median_sec`
(censored as above), `touch_before_next`, `hinted_fixed_frac`: whether the
reaction slows as requests pile up.

### `messages.csv`: one row per user message of an episode with `interactive.replies`

Built by `analysis/requests.py` from `inbox.jsonl`, `replies.jsonl`,
`answers.json` and the grader's verdicts. Every time is seconds after the
message was sent (`t_ready`), empty when it never happened. Columns after a
message's kind are filled only for that kind.

| Columns | Meaning |
|---|---|
| keys, `valid` | the trial |
| `msg_id`, `kind`, `index` | the message's id, `bug`, `question` or `status`, and its position among its kind |
| `urgent` | a bug marked urgent |
| `t_ready_sec`, `delivery_sec` | when it was sent after the session start; until it entered the conversation |
| `n_replies` | replies naming its id |
| `reply_sec` | a bug or question: until the first reply naming it (a question's answer); a status check: until the first reply of any kind sent after it |
| `question_id`, `question`, `reference`, `answer` | a question, its reference answer, and the text of its first reply |
| `correct` | the judge's verdict on that first reply (`answers.json`); false when never answered; empty when answered but not graded (or the verdict was unreadable) |
| `subtask_id`, `fixed` | a bug's subtask and whether the grader counts it fixed (resolved, no visible regression; false without a verdict in a valid trial, empty in an invalid one) |
| `eta_sec`, `eta_min` | the first reply with an estimate, and the estimate in minutes; it predicts done at `eta_sec + 60 x eta_min` |
| `n_eta_updates` | replies with an estimate, the first included |
| `done_sec` | the first reply with `done` |
| `eta_log2_err` | log2(actual / predicted time from the estimate to done), each floored at 30 s; > 0 late; only when done came at or after the estimate |
| `on_time` | done by the predicted time; false when never done and the session ran past it; empty when the session ended first, or without an estimate |
| `claim_ok` | a done claim the grader confirms (`fixed`); empty without a done claim |
| `urgent_first` | an urgent bug: of the earlier bugs still open (no done reply) when it was sent, the share it was said done before, never-done bugs counting as later; empty without an open earlier bug |

### `requests.csv`: one row per condition with replies (valid trials only)

| Columns | Meaning |
|---|---|
| `condition`, `n_episodes` | the condition and its valid episodes |
| `n_questions`, `answered_frac` | questions sent, and the share with a reply |
| `answer_median_sec`, `answered_60s`, `answered_180s` | `reply_sec` of questions: median counting no answer as infinitely late (NaN when more than half unanswered), and the share answered within 60 s and 180 s |
| `n_graded_answers`, `correct_frac` | questions with a `correct` value (graded answers plus unanswered ones, which count false), and the share of them correct |
| `n_bugs`, `fixed_frac` | bugs sent, and the share counted fixed |
| `first_reply_median_sec` | censored median `reply_sec` of the bugs after the first |
| `eta_given_frac`, `eta_median_sec`, `eta_min_median` | the share of bugs with an estimate; censored median `eta_sec` of the bugs after the first; median estimate in minutes |
| `done_claimed_frac`, `claim_precision` | the share of bugs said done; of those, the share the grader confirms |
| `eta_abs_log2_median`, `eta_log2_median` | median \|`eta_log2_err`\| (1: off by a factor of 2) and median signed error (> 0: late), over bugs said done after an estimate |
| `on_time_frac` | the share on time of the bugs with a known `on_time` |
| `n_status`, `status_reply_median_sec` | status checks sent, and the censored median of their `reply_sec` |
| `n_urgent`, `urgent_first_frac` | urgent bugs, and the mean `urgent_first` over those with an open earlier bug |

### `summary.json`

| Field | Meaning |
|---|---|
| `jobs_dirs`, `pricing` | inputs: the jobs dirs and `{file, scheme}` |
| `n_trials`, `n_valid`, `invalid_reasons` | trial counts and `{reason: n}` |
| `critical_path` | `{n_valid_episodes, n_over_wall_clock, mean_overhead_sec, mean_overhead_frac}`: valid episodes whose critical path exceeds the wall clock by more than 1 ms (should be 0) and the mean overhead |
| `fairplay` | `{n_flagged_episodes, trials}`: episodes with a fairplay flag (the analysis prints a `[WARN]`) |
| `delegation_gate` | `{ok, never_delegated, no_valid_episodes}`: subagent conditions with valid episodes of which none delegated (the analysis exits 3 when any), and conditions without a valid episode |

## Figures (`figures/<sweep>/`)

`python -m subagents_bench.plots ANALYSIS_DIR --out FIGDIR` draws nine PNGs
from the tables (valid trials only) and writes `index.md` with a caption for
each: `main_grid` (resolve rate with Wilson interval and delegated fraction
per condition), `pareto_cost` and `pareto_time` (resolve rate vs mean cost
and wall clock, with the Pareto front), `time_breakdown` (critical-path parts
and overhead per condition), `token_composition` (tokens and cost by role),
`delegation` (spawns and dispatch outcomes), `arbiter` (judge confidence,
debate rate, isolation violations) `timeline` (Gantt of the calls and
tools of one delegating episode per strategy) and `interactivity` (median
minutes from a hint to the orchestrator's answer, the first touch and the
first edit of its files per condition, over the hints after the first and,
for touch and edit, without pre-edited hints (`interactivity.csv`); and the
share touched before the next hint by hint position, hint 0 included; a
placeholder without interactive episodes).
