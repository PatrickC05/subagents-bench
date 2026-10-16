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
  .sweep.lock                         flock'd while a sweep or preflight --oracle runs: `pid N: <command line>`
  <sweep>__r<k>__<condition>/         one harbor job (round k); harbor's config.json, result.json, job.log
    <task>__<suffix>/                 one harbor trial: config.json, result.json, trial.log, verifier/
      agent/                          harbor's logs_dir: this harness's records (below), with
                                      `snapshots/<seq>.patch` (`interactive.snapshots`) and, written after
                                      the run, `answers.json` (`grade-answers`) and `fix_states.json` (`fix-times`)
      verifier/eval.json              SWE-sweep's grader: per-subtask verdicts (read by the interactivity analysis)
      artifacts/logs/artifacts/model.patch
                                      SWE-sweep's collect hook: the workspace's diff against `baseline` after
                                      the session, which the verifier graded (`fix-times`' final state)
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
                                      ref its Dockerfile checks out (the source the questions are written from;
                                      a full clone checked out at the commit when the ref is a sha)
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
| `t_first_chunk` | first stream chunk carrying content, reasoning or a tool call, in the successful attempt; null for a failed call (an interrupted call: in the aborted attempt, if any came) |
| `t_end` | reply complete, or the call failed |
| `prompt_tokens` | prompt tokens, cached ones included (0 for a failed call) |
| `cached_tokens` | prompt tokens served from the provider's prefix cache |
| `completion_tokens` | output tokens, reasoning included |
| `reasoning_tokens` | reasoning tokens, null when the provider did not report them |
| `finish_reason` | `stop`, `tool_calls`, `length` (cut off at `max_output_tokens`), `unknown` (no finish reason in the stream), `interrupted` (aborted by an interrupt); null for a failed call |
| `n_tool_calls` | tool calls in the reply after stream repair |
| `n_attempts` | HTTP attempts (1 = no retry); 0 when the call was abandoned on a stop |
| `backoff_sec` | seconds slept between attempts |
| `stream_repairs` | re-sent or clashing tool calls repaired in the stream (docs/DESIGN.md) |
| `one_call_retry` | the reply came from a retry that asked for exactly one tool call, after a stream error once a tool call had started (Sail's Gemma-4 parsing of several tool calls in one reply; docs/DESIGN.md) |
| `error` | null; `cancelled` (the episode stopped during the call); `interrupted` (`interactive.interrupt`: a user message aborted the orchestrator's call; not a failure); else `<ErrorClass>: <message>` (`ProviderError`, `BillingError`, `ContextLengthError`, ...) |
| `n_chunks` | stream chunks carrying content, reasoning or a tool call (about one token each); null for a failed call |
| `tokens_estimated` | the token counts are estimates: an interrupted call whose stream ended before the provider's usage chunk (docs/DESIGN.md, *Interrupts*); false otherwise |
| `partial_text_chars`, `partial_reasoning_chars`, `prompt_chars` | interrupted calls only: the characters of text and reasoning streamed before the abort (dropped from the conversation), and of the request (the prompt estimate's basis) |

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
| `interrupted` | the step's LLM call was interrupted (`interactive.interrupt`): no reply, no tools; the next step delivers the message and calls again |

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
| `to` | the `msg_id` it answers (the call may name it `3`, `3.0`, `"3"` or `"#3"`; a string of 10 or more digits is rejected) |
| `kind`, `hint_index` | that message's kind (`bug`, `question`, `status`: note `bug`, where the inbox says `hint`) and position |
| `subtask_id`, `question_id` | that message's subtask (a bug) or question, else null |
| `text` | the message to the user |
| `eta_min` | minutes from now until the bug is fixed, as a float (a numeric string is read); null when not given or not a number in [0, 10^6), in which case the tool result told the agent it was ignored |
| `done` | true when the call passed `done` as `true`, `1` or `"true"`/`"yes"`; false when absent, `false`, `0`, `"false"`/`"no"`, or anything else (which the tool result reports as ignored) |

With `interactive.require_eta`, a reply to a bug with no usable `eta_min` and
without `done: true` is refused (an errored tool record), so every bug reply
written has one or the other.

### `snapshots.jsonl`: one workspace snapshot (`interactive.snapshots`)

Empty without `interactive.snapshots`. One line per attempt of the
background recorder, which runs the collect hook's diff against `baseline`
privately (docs/DESIGN.md, *Workspace snapshots*).

| Field | Meaning |
|---|---|
| `seq` | 0, 1, ... in the order taken; 0 is the session's start, the untouched workspace |
| `t_request` | the earliest request the snapshot serves: the session start (seq 0), the end of the first `bash` or `edit_file` call that asked for it (several calls during one snapshot make one request), or the heartbeat's due time |
| `t_start`, `t_end` | the snapshot command's run (`t_start` is the state's time in the replay) |
| `trigger` | `start`, `tool` (after a `bash` or `edit_file` call of any agent) or `heartbeat` (`snapshot_every_sec` after the previous one ended, with no tool request) |
| `tree` | the git tree id of the workspace as the hook would stage it; null when the attempt failed before it |
| `changed` | the tree differs from the last successful snapshot's (true for seq 0, and for a return to an earlier state); false without a tree |
| `patch` | `snapshots/<seq:04d>.patch`, relative to `agent/`: the state's diff, written only for a changed state without an error; else null |
| `bytes` | the diff's size in bytes; null when no diff was computed (the tree was the previous one, or the attempt failed) |
| `error` | null; `too large` (a diff over 20 MB: `tree`, `changed` and `bytes` are kept, no file), `stopped` (the episode stopped during it; the recorder ends), `timed out after 120 s`, `exit <code>: <stderr tail>`, `no snapshot in the output`, `unreadable patch: ...`, or the exception |

`agent/snapshots/<seq>.patch`: `git diff --cached --binary baseline` of the
workspace at that moment, byte for byte what the collect hook would have
written then (an empty file for snapshot 0 of an untouched workspace).

### `answers.json`: verdicts on the answers (`subagents-bench grade-answers`)

Written after the run, for trials with questions; absent until then.
Version 2: `{version: 2, judge_model, judge_temperature (0.0), graded_at,
answers, replies, elsewhere}`. The last three are keyed by the question's
`msg_id` as a string, with no key for a question without such replies
(unanswered questions are not sent to the judge):

| Key | Per question |
|---|---|
| `answers` | `{correct, verdict, why}` of its first reply, the answer as the prompt calls it (version 1's shape plus `verdict`) |
| `replies` | `[{t, verdict, correct, why}]`: its replies to its own id in time order, up to and including the first `correct` one |
| `elsewhere` | `[{t, to, verdict, correct, why}]`: replies to another id (`to`) that cite `#<msg_id>` as a whole token and were sent after the question was delivered; judged the same way, never its answer |

`verdict` is `correct`, `wrong` or `none` (not an answer: an acknowledgement,
a promise, how to find out); null when the judge's reply could not be read,
`why` then being its start (else the judge's one-sentence reason). `correct`
is `verdict == "correct"`, null with the verdict. Version 1 (no `version`
key: `{judge_model, graded_at, answers: {msg_id: {correct, why}}}`) judged
only the first reply, correct or not ("not an answer" was not correct), at
the model's own temperature; the analysis reads its `correct` true as
correct and false as wrong, and `grade-answers` grades it again.

### `fix_states.json`: every recorded state graded (`subagents-bench fix-times`)

Written after the run, for trials with `snapshots.jsonl`; absent until then
(docs/DESIGN.md, *Fix times*).

| Field | Meaning |
|---|---|
| `version` | 1 |
| `trial`, `task` | `<job>/<trial>`, the task |
| `graded_at` | when it was written |
| `image` | `{dockerfile, tests_sha256, id}`: the verifier Dockerfile the sandbox was built from (relative to the repository), a hash of its `tests/` dir, Modal's image id |
| `final_matches_verifier` | the final state's per-bug `pass` equals `verifier/eval.json`'s `resolved` for every sent bug (a bug missing there is unresolved); null without a readable `eval.json` or when it holds an error |
| `mismatches` | `[{subtask, replay, verifier_resolved}]`: the bugs that differ (`replay`: the final state's status) |
| `sent` | the subtask ids sent in the session, in sending order: the bugs graded |
| `states` | one per state: every snapshot with a patch, by `seq`, then the final state |

Per state:

| Field | Meaning |
|---|---|
| `seq` | the snapshot's; null for the final state |
| `source` | `snapshot`, or `final` (the trial's `model.patch`) |
| `t` | the snapshot's `t_start`; for the final state the episode's `t_end` |
| `patch` | the patch, relative to `agent/` (`snapshots/0007.patch`, `../artifacts/logs/artifacts/model.patch`) |
| `applied` | the grader applied the patch; null when the evaluation failed |
| `visible_ok` | the visible suite passed; null when the evaluation failed or the patch did not apply |
| `error` | why none of the state's bugs could be graded (the evaluation's error with its stderr tail, `agent build failed: ...`, `agent patch did not apply: ...`, `patch file missing`); else null |
| `duration_s`, `rerun_duration_s` | seconds the state's grading and its rerun took (null without a rerun) |
| `bugs` | sent subtask id -> `{status, n_passed, n_failed, error, rerun}` |

Per bug: `status` `pass` (resolved), `fail` or `unknown` (the state says
nothing about the bug: docs/DESIGN.md); `n_passed`, `n_failed` the grader's
counts of its targeted tests; `error` why it is unknown (`evaluation
error`, `agent build failed`, `agent patch did not apply`, `hidden test
patch did not apply`, the grader's error for the targeted run, such as a
timeout, `targeted tests errored (collection or build failure): ...`, `no
targeted test ran`, `not graded`, `not in the task's bundle`), else null;
`rerun` the status when the state was graded again (the first `fail` after
a `pass`), else null.

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
| `n_interrupts` | calls with error `interrupted` (`interactive.interrupt`; 0 otherwise) |
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
`<sweep>__r<k>__<condition>`), `task_name`, `trial_name`. Several jobs dirs
are one data set, but the per-condition tables (`conditions.csv`,
`interactivity.csv`, `hint_positions.csv`, `requests.csv`, `eta.csv`) have
one row per (`sweep`, `condition`) and `levels.csv` one per (`sweep`,
`setting`, `level`): a condition run in two sweeps (the levels sweep's L0
and the earlier baseline run) is two rows, never pooled.

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
| `n_questions`, `n_replies` | an interactive session's questions scheduled and `reply` calls written (0 without `interactive.replies`; empty in batch episodes and in episodes written before they existed) |
| `steps`, `spawns`, `dispatches`, `arbitrations` | record counts |
| `delegated` | 1 if the orchestrator spawned at least once |
| `orch_bash_n`, `orch_bash_median_sec`, `orch_bash_max_sec`, `orch_llm_median_sec` | the orchestrator's own bash commands (count, median and longest duration) and the median duration of its successful LLM calls: what a message waits behind between its steps |
| `n_fairplay_flags`, `fairplay_kinds` | tool calls of any agent matching a fairplay pattern, and their categories joined with `;` (`git_history`, `network`, `installed_copy`; analysis/fairplay.py): to review, not verdicts |
| `n_provider_errors`, `n_stream_repairs` | from the episode and its calls |
| `n_interrupts` | the orchestrator's interrupted calls (`interactive.interrupt`) |
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

### `conditions.csv`: one row per (sweep, condition) (valid trials only)

| Columns | Meaning |
|---|---|
| `sweep`, `condition`, `topology`, `strategy`, `scope`, `orchestrator_tools`, `orchestrator_model`, `subagent_model` | the sweep and the design |
| `n`, `n_valid`, `n_resolved` | trials, valid trials, solved valid trials |
| `resolve_rate`, `resolve_ci_lo`, `resolve_ci_hi` | n_resolved / n_valid with its Wilson 95% interval |
| `n_delegated`, `delegated_frac` | valid episodes that spawned at least once |
| `n_fairplay_flagged` | valid episodes with at least one fairplay flag |
| `n_bugs`, `bugs_fixed`, `bug_fix_rate`, `n_regressed`, `reward_mean` | over the valid trials, a trial without a reward counting as reward 0 with no bug fixed: bugs in their tasks, bugs fixed, their ratio (SWE-sweep's micro-average; NaN without bugs), trials with a regression, and the mean reward |
| `{cost_total,wall_clock_sec,total_tokens}_{mean,ci_lo,ci_hi}` | mean per valid episode with a seeded percentile bootstrap 95% interval (2000 resamples; no interval for one episode; NaN if any value is NaN) |
| `cost_per_solve` | total cost of the valid episodes / n_resolved (NaN without a solve) |

Rows are ordered: single-agent baselines, the strategy x scope grid, the
same-model control, the managers, then conditions without an episode; a
condition's sweeps are side by side. The delegation gate names a condition
once, whichever of its sweeps failed it.

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

### `interactivity.csv`: one row per interactive (sweep, condition) (valid trials only)

Latency columns are over the hints after the first (hint 0 arrives with the
task, while the agent starts up; it has the `first_*` columns).

| Columns | Meaning |
|---|---|
| `sweep`, `condition`, `topology`, `spawn_mode`, `n_episodes`, `n_hints`, `n_later_hints` | the sweep, the design and the counts (all hints, hints after the first) |
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

### `hint_positions.csv`: one row per interactive (sweep, condition) and hint position

`sweep`, `condition`, `hint_index`, `n`, `{delivery,response,dispatch,touch,edit}_median_sec`
(censored as above), `touch_before_next`, `hinted_fixed_frac`: whether the
reaction slows as requests pile up.

### `messages.csv`: one row per user message of an episode with `interactive.replies`

Built by `analysis/requests.py` from `inbox.jsonl`, `replies.jsonl`,
`answers.json` and the grader's verdicts; the question timing columns, from
`correct_sec` on, by `analysis/levels.py`. Every time is seconds after the
message was sent (`t_ready`), empty when it never happened. Columns after a
message's kind are filled only for that kind.

| Columns | Meaning |
|---|---|
| keys, `valid` | the trial |
| `msg_id`, `kind`, `index` | the message's id, `bug`, `question` or `status`, and its position among its kind |
| `urgent` | a bug marked urgent |
| `t_ready_sec`, `delivery_sec` | when it was sent after the session start; until it entered the conversation |
| `n_replies` | replies naming its id |
| `reply_sec` | a bug: until the agent first acknowledged it, the first orchestrator turn after its delivery whose text, reasoning or tool calls name it (`#N`, request/bug N, or one of its tests), or a reply to it if earlier; a question: until the first reply naming it (the answer); a status check: until the first reply of any kind sent at or after its delivery (empty if never delivered) |
| `user_reply_sec` | a bug: until the first reply to it the user saw |
| `question_id`, `question`, `reference`, `answer` | a question, its reference answer, and the text of its first reply |
| `correct` | the judge's verdict on that first reply (`answers.json`); empty when never answered, and when answered but not graded (or the verdict was unreadable) |
| `subtask_id`, `fixed` | a bug's subtask and whether the grader counts it fixed (resolved, no visible regression; false without a verdict in a valid trial, empty in an invalid one) |
| `eta_sec`, `eta_min` | the first estimate: the first reply with `eta_min` sent before the done reply (not the done reply itself), and the estimate in minutes; it predicts done at `eta_sec + 60 x eta_min` |
| `n_estimates` | such replies (with `eta_min`, before the done reply), the first included |
| `done_sec` | the first reply with `done` |
| `eta_log2_err` | log2(actual / predicted time from the estimate to done), each floored at 30 s; > 0 late. A bug estimated and never said done: `inf` when the session ran past the predicted time or ended for another reason than its deadline (`exit_reason` not `deadline` or `cancelled`: the agent stopped, so it gave the bug up); empty when the deadline came first (censored). Empty without an estimate |
| `eta_total_log2_err` | the same prediction scored from when the bug was sent: log2(`done_sec` / (`eta_sec` + 60 x `eta_min`)), each floored at 30 s; a late estimate's short horizon makes `eta_log2_err` a ratio of a small part of the wait, this one is of the whole wait. `inf` and empty where `eta_log2_err` is |
| `on_time` | done by the predicted time; false where `eta_log2_err` is `inf`; empty where it is empty |
| `claim_ok` | a done claim the grader confirms (`fixed`); empty without a done claim |
| `urgent_first` | an urgent bug: of the earlier bugs still open (no done reply) when it was sent, the share it was said done before, never-done bugs counting as later; empty without an open earlier bug |
| `correct_sec` | a question: until the first reply the judge calls correct (`answers.json` version 2 judges the replies in time order up to the first correct one; version 1, only the first reply); empty when never answered correctly or not known |
| `correct_known` | `correct_sec` is known: every reply up to the first correct one (every reply, when none is) has a verdict; false while a verdict is unread or a reply not graded (version 1: a first reply not correct, with later replies) |
| `first_verdict`, `first_reply_correct` | the first reply's verdict (`correct`, `wrong`, `none`; version 1's false reads as `wrong`) and whether it is `correct` (`correct` above, from the same reply); empty when unanswered or unread |
| `wrong_before_correct` | a reply judged wrong came before the first correct one; empty when never answered correctly |
| `n_judged`, `n_wrong` | replies judged (up to the first correct one), and those judged wrong |
| `response_sec` | from delivery to the first correct reply (`delivery_sec` + `response_sec` = `correct_sec`) |
| `n_elsewhere`, `elsewhere_correct`, `elsewhere_correct_sec` | replies to another id citing the question (`elsewhere` in `answers.json`), whether one was judged correct (empty without any, or while one is unread and none is correct), and the first such reply's time; never counted as an answer |

### `requests.csv`: one row per (sweep, condition) with replies (valid trials only)

| Columns | Meaning |
|---|---|
| `sweep`, `condition`, `n_episodes` | the sweep, the condition and its valid episodes |
| `n_questions`, `answered_frac` | questions sent, and the share with a reply |
| `answer_median_sec`, `answered_60s`, `answered_180s` | `reply_sec` of questions: median counting no answer as infinitely late (NaN when more than half unanswered), and the share answered within 60 s and 180 s |
| `n_graded_answers`, `correct_of_answered`, `correct_of_asked` | questions with a verdict (answered and graded); the share of them correct; and correct answers over every question sent, unanswered ones counting as not correct (NaN while an answered question has no verdict, e.g. before `grade-answers`) |
| `n_bugs`, `fixed_frac` | bugs sent, and the share counted fixed |
| `first_reply_median_sec` | censored median `reply_sec` of the bugs after the first |
| `user_reply_median_sec` | the same for `user_reply_sec` |
| `eta_given_frac`, `eta_median_sec`, `eta_min_median` | the share of bugs with an estimate (before their done reply); censored median `eta_sec` of the bugs after the first; median estimate in minutes |
| `done_claimed_frac`, `claim_precision` | the share of bugs said done; of those, the share the grader confirms |
| `eta_abs_log2_median`, `eta_log2_median` | median \|`eta_log2_err`\| (1: off by a factor of 2) and median signed error (> 0: late), over bugs with a known error: those said done after an estimate and the never-done ones at `inf` (so either median can be `inf`) |
| `eta_total_abs_log2_median` | median \|`eta_total_log2_err`\| over the same bugs (can be `inf` too); NaN, like the two above, when no bug has a known error |
| `n_eta_never_done` | bugs estimated and never said done that count as late (`eta_log2_err` `inf`) |
| `on_time_frac` | the share on time of the bugs with a known `on_time` (never-done ones counting as late) |
| `n_status`, `status_reply_median_sec` | status checks sent, and the censored median of their `reply_sec` |
| `n_urgent`, `urgent_first_frac` | urgent bugs, and the mean `urgent_first` over those with an open earlier bug |

### Levels tables

Built by `analysis/fixes.py`, `estimates.py` and `levels.py` (definitions:
docs/DESIGN.md, *Levels analysis*). Times are seconds after the message was
sent (the bug, or the status check for `status_pairs.csv`) unless named
`*_session_sec` (after the session's start). Fix times come from
`fix_states.json`: a trial without it, or whose final state does not
reproduce the verifier, has no usable fix times (`timing_ok` false).
Censored medians count an event that never happened (a bug never fixed or
started, a question or check never answered) as infinitely late: NaN when
at least half never happened, since the median is then infinite.

#### `fixes.csv`: one row per bug sent in an interactive session

| Columns | Meaning |
|---|---|
| keys, `valid` | the trial |
| `has_fix_states` | the trial has `agent/fix_states.json` |
| `timing_ok` | it has, and the final state reproduces the verifier: the times are usable |
| `final_matches_verifier`, `n_states` | `fix_states.json`'s `final_matches_verifier` is true (false when it is false or null), and its states with a time; empty without the file |
| `n_snapshots`, `n_snapshot_errors` | snapshot records, and those with an error |
| `msg_id`, `hint_index`, `subtask_id`, `urgent` | the bug |
| `ready_session_sec` | when it was sent |
| `fixed_end` | the end-of-session verdict: resolved with the visible suite passing (`verifier/eval.json`); false for a bug without a verdict in a graded trial, empty without `eval.json` |
| `fixed_sec`, `fixed_session_sec` | the first `pass` from which the bug keeps passing, with no confirmed break, for 120 s or to the end; empty when never. Every column from here on is empty unless `timing_ok` |
| `stable_sec` | the start of the passing streak that lasts to the end; empty when the bug ends failing |
| `first_pass_sec` | the first `pass` |
| `n_breaks` | confirmed failing stretches after the first pass |
| `visible_ok_at_fix` | the visible suite passed in the `fixed_sec` state |
| `pass_before_sent` | the bug passed in a state before it was sent (its times are then clipped to 0) |
| `state0_pass` | it passed in snapshot 0, the untouched workspace (which should fail every bug) |
| `final_status` | its replayed status in the final state: `pass`, `fail` or `unknown` |
| `n_unknown` | states in which its status is `unknown` |

#### `estimates.csv`: one row per estimate of a bug

From `replies.jsonl`, the inbox and the fix times.

| Columns | Meaning |
|---|---|
| keys, `valid` | the trial |
| `msg_id`, `subtask_id`, `hint_index`, `urgent`, `ready_session_sec` | the bug estimated, and when it was sent |
| `via` | `structured` (a reply to the bug with `eta_min`), `text_bug` (read from the text of a bug reply without `eta_min`), `text_status` (read from a reply to a status check: one row per bug it gives a time for) |
| `source` | `arrival` (the bug's first estimate, sent before the next status check after it; without checks, before L1's next check time), `check` (within 120 s after a check that lists the bug), `update` |
| `reply_to`, `check_id` | the id replied to (the bug, or a status check); the check whose 120 s window the estimate is in, else empty |
| `est_session_sec`, `est_sec` | when the estimate was sent |
| `eta_min` | the estimate in minutes (text: hours and seconds converted, a range's end) |
| `pred_session_sec`, `grace_sec` | the predicted fix, sent + 60 x `eta_min`; the grace, max(60, 0.25 x 60 x `eta_min`) |
| `eligible` | prediction + grace falls within the session (start + `wall_clock_sec`) |
| `timing_ok`, `fixed_sec` | the trial has usable fix times; the bug's `fixed_sec` (empty when never fixed) |
| `stale` | sent at or after the bug's fix (not scored); empty without timing |
| `fixed_by_deadline` | the bug was fixed by the deadline; empty without timing |
| `scored` | `timing_ok`, eligible and not stale: scored against the fix |
| `on_time`, `on_time_strict`, `within_1m`, `within_2m`, `within_5m` | fixed by prediction + grace, by the prediction, by the prediction + 1, 2, 5 minutes; false for a bug never fixed; empty unless scored |
| `err_min`, `err_lower_bound` | (fix - prediction) / 60, > 0 late; for a bug never fixed, a lower bound to the later of the episode's end and the deadline (`err_lower_bound` true) |
| `log2_err` | log2(max(fix - sent, 30) / max(60 x `eta_min`, 30)), > 0 late; `inf` for a bug never fixed |
| `weight` | 1 / the bug's scored estimates, so each bug weighs 1; empty unless scored |
| `claim_sec` | the bug's first reply with `done: true` |
| `claim_scored` | eligible and sent before the claim (or the bug was never said done) |
| `claim_on_time`, `claim_on_time_strict`, `claim_within_{1,2,5}m`, `claim_err_min`, `claim_err_lower_bound`, `claim_log2_err`, `claim_weight` | the same scores against the done claim (never said done: late, `inf`); empty unless `claim_scored` |
| `text` | a text estimate's reply text (500 characters); empty for `structured` |

#### `status_pairs.csv`: one row per status check and bug it lists not said done before it

| Columns | Meaning |
|---|---|
| keys, `valid` | the trial |
| `check_id`, `check_session_sec` | the status check and when it was sent |
| `msg_id`, `subtask_id`, `hint_index`, `urgent` | a bug sent before the check and not said done before it (a reply with `done: true`, or a done written in a reply to a check's id) |
| `reply_sec` | until the first estimate (any via) or done for the bug sent at or after the check; empty when none |
| `answered` | that came within 120 s |
| `kind`, `via` | the answer: `eta` or `done`; `structured`, `text_bug` or `text_status` (a done written in a check reply is `text_status`); empty unless answered |
| `timing_ok` | the trial has usable fix times |
| `right` | a done for a bug fixed by the answer, or an estimate for one not fixed (by `fixed_sec`); empty unless answered with timing |
| `fixed_at_check` | the bug was fixed by the check's send time; empty without timing |
| `claimed` | the bug was said done at some point (here: at or after the check) |
| `done_owed` | fixed by the check and never said done in the session; empty without timing |

#### `eta.csv`: one row per (sweep, condition) with estimates or status pairs (valid trials only)

"Weighted" shares weigh each bug 1, its estimates in the subset sharing it.

| Columns | Meaning |
|---|---|
| `sweep`, `condition` | the run |
| `n_episodes_estimated` | sessions with an estimate |
| `n_estimates`, `n_structured`, `n_text_bug`, `n_text_status` | estimates, and by via |
| `n_bugs_estimated` | bugs with an estimate |
| `n_timed`, `n_scored`, `n_scored_bugs` | estimates in trials with usable fix times; those scored; their bugs |
| `on_time_frac`, `on_time_strict_frac`, `within_{1,2,5}m_frac` | weighted shares of the scored estimates, every via (`levels.csv`'s main number uses the structured ones) |
| `n_scored_<x>`, `on_time_frac_<x>` | for each source (`arrival`, `check`, `update`) and via (`structured`, `text_bug`, `text_status`): its scored estimates and their weighted on-time share |
| `abs_log2_median_fixed`, `abs_log2_median`, `log2_median` | median \|`log2_err`\| over the scored estimates of bugs fixed; over every scored estimate (`inf` for a bug never fixed, so it can be `inf`); the median signed error |
| `err_min_median_fixed` | median `err_min` over the scored estimates whose error is not a lower bound |
| `stale_frac` | of the estimates with timing, the share sent at or after the fix |
| `n_ineligible`, `ineligible_fixed_frac` | estimates with timing neither eligible nor stale, and the share whose bug was fixed by the deadline |
| `n_claim_scored`, `claim_on_time_frac`, `claim_abs_log2_median` | estimates scored against done claims, their weighted on-time share and median \|`claim_log2_err`\| |
| `n_pairs`, `coverage_120s`, `pair_reply_median_sec` | status pairs, the share answered within 120 s, and the median `reply_sec` counting no answer as infinitely late (NaN when at least half have none) |
| `right_frac` | of the pairs answered in trials with timing, the share answered right |
| `done_owed_frac`, `n_done_owed` | of the pairs with timing whose bug was fixed by the check, the share never said done; the bugs owed a done |

#### `urgency.csv`: one row per urgent bug of a valid L3 session and twin level

Its twins: the same subtask sent within 1 s in the valid sessions of the
same sweep, task and setting at `twin_level`. Only timed twins of a timed
urgent bug pair.

| Columns | Meaning |
|---|---|
| keys (valid sessions only, no `valid`), `setting`, `level` | the session (`level` L3) |
| `msg_id`, `subtask_id`, `t_ready_sec` | the urgent bug, and when it was sent (after the session's start) |
| `timed` | the session has usable fix times |
| `fixed_sec` | its `fixed_sec` |
| `dispatch_sec`, `change_sec`, `start_sec` | until a subagent started whose instruction names its target files or hidden tests (as in `hints.csv`); until the first snapshot after it was sent that changed its target files' diff (at that snapshot's `t_end`); the earlier of the two |
| `twin_level` | `L2` (the comparison) or `L0` (descriptive) |
| `n_twins`, `n_twins_timed` | its twins, and those with fix times |
| `twin_fixed_sec_median` | censored median `fixed_sec` of the timed twins |
| `fixed_240`, `fixed_480`, `twin_fixed_240`, `twin_fixed_480` | fixed within 240 s and 480 s (empty when not timed); the share of the timed twins |
| `win_rate`, `n_pairs`, `n_decisive` | the mean win score over its pairs with the timed twins (1: fixed more than 30 s sooner, or only it fixed; 0: the reverse; 0.5: within 30 s, or neither), the pairs, and those not 0.5 |
| `order`, `twin_order` | per pair, over the earlier bugs failing at its send time in both sessions: 1 if it was fixed before the earlier bug, 0 after, 0.5 if neither or at once (an earlier bug by its first fix after the send time); the mean over bugs, then over pairs; `twin_order` the same in the twins |
| `n_order_pairs`, `n_order_bugs` | pairs with at least one such earlier bug, and those bugs summed over pairs |
| `twin_start_sec_median` | censored median `start_sec` of all its twins |
| `preempt_n_touch`, `preempt_touch_share`, `twin_preempt_touch_share` | tool calls of any agent starting in the 240 s after delivery that touch its target files or an earlier bug's (as `touch_sec` matches them); the share touching its own (a call touching both counts as its own); the same pooled over its twins. Empty when it was never delivered or no bug came before it |
| `preempt_n_edit`, `preempt_edit_share`, `twin_preempt_edit_share` | the same for edits (as `edit_sec` matches them) |
| `others_fixed_frac`, `twin_others_fixed_frac` | the share of the session's other bugs fixed (by `fixed_sec`), and of the same bugs in the timed twins |
| `others_fixed_sec_median`, `twin_others_fixed_sec_median` | their censored median `fixed_sec` |

#### `levels.csv`: one row per sweep, setting and level (valid sessions only)

Only levels conditions (names ending `_baseline`, `_eta`, `_questions`,
`_urgent`). Every row has every column, NaN where its level has no such
message. `_lo` and `_hi`: the 95% interval of a cluster bootstrap over
sessions stratified by task (2000 resamples).

| Columns | Meaning |
|---|---|
| `sweep`, `setting`, `level`, `condition` | the design (`condition`: its condition names, `;`-joined) |
| `n_sessions`, `n_tasks`, `n_timed` | valid sessions, their tasks, and the sessions with usable fix times |
| `n_bugs`, `bugs_fixed_frac`, `bugs_fixed_sec_median`, `bugs_fixed_end_frac` | bugs sent; of the bugs in timed sessions, the share with a `fixed_sec` and its censored median; the share fixed at the end (`fixed_end`) |
| `eta_on_time`, `eta_on_time_lo`, `eta_on_time_hi` | main number: the on-time share of the structured scored estimates, each bug weighted once |
| `eta_n_estimates`, `eta_n_bugs` | those estimates, and their bugs |
| `eta_on_time_any_via`, `eta_n_text` | the on-time share with the text estimates too, and the scored text estimates |
| `eta_on_time_arrival` | the on-time share of the structured arrival estimates |
| `eta_bugs_estimated_frac` | of the bugs sent, the share with an estimate (any via, scored or not) |
| `status_n_pairs`, `status_answered_frac`, `status_right_frac` | status pairs, the share answered within 120 s, and of those with a verdict, the share answered right |
| `q_n` | questions asked |
| `q_correct_{30,60,120,300}s`, `q_correct_60s_lo`, `q_correct_60s_hi` | the share of the questions asked with `correct_sec` within each threshold (never answered correctly: never); NaN while any question's `correct_known` is false; 60 s is the main number |
| `q_first_correct_60s` | the share whose first reply was correct and came within 60 s; NaN while an answered question's first reply has no verdict |
| `q_wrong_before_correct_frac` | of the questions answered correctly, the share with a wrong reply first |
| `q_delivery_median_sec`, `q_response_median_sec` | censored medians of `delivery_sec` and `response_sec` |
| `q_n_elsewhere_correct` | questions with a correct reply to another id (not counted) |
| `urgent_win_rate`, `urgent_win_rate_lo`, `urgent_win_rate_hi` | main number (L3): the win rate over every pair of a timed urgent bug and a timed L2 twin in the sweep and setting; L3 and L2 sessions resampled within each level and task |
| `urgent_n_pairs`, `urgent_n_decisive` | those pairs, and those not 0.5 |
| `urgent_order`, `twin_order` | `urgency.csv`'s L2 rows' `order` and `twin_order`, weighted by their `n_order_pairs` |
| `urgent_fixed_240`, `urgent_fixed_480`, `twin_fixed_240`, `twin_fixed_480` | the share of the timed urgent bugs fixed within 240 s and 480 s, and of their timed L2 twins (each once) |

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

## Monitor page (`OUT_DIR/`)

Written by `python -m subagents_bench.monitor JOBS_DIR [--compare OTHER_JOBS_DIR] --out OUT_DIR`
from the jobs dirs (read only; the design from `JOBS_DIR/sweep.json` or `--spec`). Run it from a
checkout (or an editable install): costs come from the repo's `configs/pricing.yaml`, as for the
rest of `analysis`. The page names the jobs dir by its folder name only, never its local path.

| File | Contents |
|---|---|
| `index.html` | the page: per-condition requests table, controls against their twins, the comparison with `--compare` (when given), one card per session; costs from `configs/pricing.yaml`, `sail_balanced` |
| `transcripts/<job>__<trial>.json` | one session's transcript for the page's viewer (not with `--no-transcripts`): `v` (format version), `job`, `trial`, `t0` (the episode's `session_t0`, else its `t_start`, else the first transcript row's `t`), `sys` (the orchestrator's system prompt), `msgs` (its thread: `t` seconds after `t0`, `r` role, `c` content, `k` reasoning, `tc` tool calls `[id, name, args]`, `tr` results `[call id, content, is_error]`), `subs` (each subagent run: `agent`, `dispatch`, `instruction`, `status`, `steps`, `t_start`, `t_end`, `files` changed without git/test side effects, `noise` their count, `after` the index in `msgs` of the spawn it follows, `msgs`), `urgent` (urgent message ids), `live`; a text over 30000 characters is cut with a marker |
| `transcripts/manifest.json` | `files`: per transcript `file`, `job`, `trial`, `sig` (of its source records: an unchanged file is not rewritten), `bytes` |
