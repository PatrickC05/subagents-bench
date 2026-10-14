# Design

What the harness does, and why. Most rules below exist because a real run
on Sail broke without them; each section names the run and the numbers. The
runs are the smoke sweeps on `fix-git` and `prove-plus-comm` of 2026-10-11
(the all-tools conditions) and 2026-10-12 (all 15 conditions, over several
attempts), run with the earlier, larger version of this harness.

## Principles

- **Log raw, compute later.** Every LLM call is logged with its tokens and
  timestamps, every tool call with its timestamps, so cost and latency are
  computed offline from the records. Provider prices do not track model size,
  so a cost fixed at run time would be one arbitrary price list.
- **Budgets are steps.** The orchestrator has 5 steps, a subagent 8. An
  episode has no token budget or wall-clock cap of its own: harbor's agent
  timeout (x2, `agent_timeout_multiplier`: 30 min for most pilot tasks) is
  its only time limit, so provider latency never changes a trajectory unless
  it is extreme.
- **One provider.** Both models are on Sail's `balanced` tier, through one
  OpenAI-compatible client.
- **An invalid trial is rerun, never scored.** A trial that a provider or
  infrastructure failure shaped does not measure its condition: the runtime
  stops it at once, the sweep reruns its job and the analysis leaves it out.
- **Lean.** No context trimming, no token budgets, no schema versions, no
  provenance fingerprints or source snapshots, no cost or latency simulation
  at run time, no other providers or benchmarks. The records keep what the
  analysis reads.

## Configuration

`config.py` loads one condition from YAML into dataclasses; an unknown key
or a bad choice raises `ConfigError` (exit 2), so a typo never silently runs
a different condition. `configs/sail/base.yaml` is the base of every
condition, and a sweep spec deep-merges each condition's overrides onto it
(mappings merge; lists, scalars and null replace) and names it after its key.
The sweep writes each resolved config to `jobs/<sweep>/configs/`, and the
harbor agent reads it from `SUBAGENTS_BENCH_CONFIG`.

Sampling temperature is 0.7 for both models, so repeated rounds are
independent samples. DeepSeek's `max_output_tokens` is 32768: at 8192, the
orchestrator reached 7976 output tokens in the smoke runs and the judge (the
same model) hit the limit in 2 of 5 calls. Gemma keeps 4096.

## LLM client (`llm.py`)

One request per call: `model`, `messages`, `tools`, `temperature`,
`max_tokens`, `stream: true` with `stream_options.include_usage`, the header
`X-Sail-Completion-Window: balanced`, and the key from `SAIL_API_KEY`
(loaded from `.env`, never printed). An assistant message's reasoning goes
back as `reasoning_content` on later turns (Sail currently ignores it: the
prompt token counts do not change).

**Usage and timings.** Tokens come from the final usage chunk: prompt
(cached included), cached, completion (reasoning included) and reasoning.
`t_start` is before the first attempt, `t_first_chunk` the first chunk that
carries content, reasoning or a tool call, `t_end` the end of the stream.
Sail streams only 1-8 coarse chunks per reply and sometimes the whole reply
in one, so `t_first_chunk` can equal `t_end`.

**Tool-call re-send repair.** Sail streams each finished tool call twice:
first under index 3, then again in full under index 0 with the same id
(Gemma-4-31B, 2026-10-11). Keyed by index, as OpenAI's format suggests,
every command ran twice, and the duplicate id failed the next request.
The accumulator routes deltas by id, drops a full re-send of a call it
already has, and gives clashing ids unique names. Each repair counts in the
call's `stream_repairs`. The captured chunk sequences are test cases.

**Retries, fail-fast.** 408, 409, 425, 429 and 5xx answers, transport
errors and stream errors are retried with full-jitter backoff from 2 s,
capped at 15 s, at most 4 times, and no retry starts more than
`retry_budget_sec` (120 s) after the call's first attempt. In the smoke
runs, 4 of 314 calls needed a retry, Sail's failed attempts ended after
about 28 s (at most 60 s), one call recovered only after 380 s, and the two
that never recovered spent 407 s and 533 s on 9 attempts under the old limits (8
retries, 120 s backoff cap, 600 s read timeout). A failure now surfaces in
about two minutes, and the episode stops (below).

**Stall rule.** An attempt fails, retryably, once `request_timeout_sec`
(180 s) passes without a chunk carrying data; keep-alive bytes do not count.
httpx's read timeout counts any byte, and on 2026-10-12 single Sail
attempts got HTTP 200 and then no data for 468 s, 600 s and 2015 s, until
the connection was reset, while the host's network was failing. 180 s is
2.5 times the longest wait for data seen (73 s, a Gemma reply sent as one
chunk; DeepSeek's longest was 11 s). It is also httpx's connect and read
timeout.

**Errors.** `BillingError`: HTTP 402, a 4xx naming balance, billing, credit
or payment, or a 429 only with `insufficient_quota`/billing (any other 429
is a rate limit and retried). `ContextLengthError`: 400 or 413 with an
explicit context-length phrase, vLLM's "maximum model length" included.
Neither is retried. Anything else that survives the retries is a
`ProviderError`.

## Runtime

### Environment and tools (`environment.py`, `tools.py`)

An environment has one method, `exec(command, timeout)`.
`HarborEnvironment` calls harbor's async `environment.exec` from the agent
threads with `asyncio.run_coroutine_threadsafe` on harbor's loop;
`LocalShellEnvironment` runs the host's bash and exists for tests. Every
exec honours the episode's `stop_event`: once it is set, `exec` raises
`AgentStopped`, before starting or while waiting, so every agent thread
unwinds within a poll interval.

The five tools run through one shell script per call. The script stages the
command, runs it in a wrapper that restores and saves the agent's cwd and
exported variables (state in `/tmp/.sb/<agent_id>/`, so `cd` and `export`
persist per agent), runs the wrapper as its own process group (`set -m`)
next to a watchdog that kills the group at the timeout (default 180 s, up to
900 s on request), and prints the head and tail of the output with the exit
status. The script drops its own stderr: harbor's docker backend merges
stderr into stdout, and bash's job-control notices (`[1]- Done ...`) for
the wrapper and watchdog landed in every tool output, so `edit_file` read
the first notice as the file path and every replace failed. While a command
runs its pid is in `pid_<n>`, for harbor's cancel. `edit_file` replaces an
exact snippet that must occur once (or creates the file), `read_file`,
`list_dir` and `search` (grep -rE) are read-only.

### Orchestrator (`orchestrator.py`)

Up to `orchestrator_steps` steps, each one LLM call and then every tool it
asked for, in order. Its tools are its own (all five, or the manager's
`read_file`, `list_dir`, `search`), `finish(summary)`, and in subagent
conditions `spawn_subagents(instructions: list[str])`. After each step the
note `[step k/N complete; R remaining; dispatch budget left: D]` is appended
to the last tool result (not a separate message, which some APIs reject).
Tools after `finish` in the same reply do not run. Exit reasons: `finished`,
`max_steps`, `no_tool_call`, `cancelled`, `provider_error`, `billing_error`,
`error` (e.g. a `ContextLengthError`: the condition's own outcome).

**Cut-off replies.** A reply that stopped at the output limit
(`finish_reason` `length`) without a tool call does not end the run as
`no_tool_call`. The reply stays in the history (as `[reply cut off at the
output token limit]` when it had no visible text), a user note asks the agent
to reason more briefly and call a tool, and the run goes on; the step is
used. Subagents do the same. Checked live on Sail (2026-10-12): the cut-off
arrives as `length` with no tool call, DeepSeek's with no visible text, and
after the note both models called a tool.

**Spawning.** A call is granted min(requested, per-call limit, dispatches
left) instructions: the per-call limit is `max_concurrent` (3) for parallel
and sequential, and 1 for the arbiter, which spends `n_candidates` (3)
dispatches per instruction; an episode has 15 dispatches. Instructions past
the grant are not run, and the result says so. `parallel` runs the granted
subagents in threads at once, `sequential` one after another in order (each
sees the changes before it), `arbiter` as below. The reports come back as one
tool result, each capped (head and tail) to an equal share of 12000
characters, at least 2000.

### Subagents (`subagent.py`)

A subagent starts from a fresh context: its scope's system prompt and the
instruction, nothing of the orchestrator's conversation. It has `max_steps`
(8) steps, its scope's tools plus `submit(report)`, and a note on its last
step. Scopes: easy = `read_file`, `list_dir`, `search`; medium = + `bash`,
with the prompt rule not to modify project files and to report the exact
change instead (a prompt rule only: `bash` can still write, and
`files_modified` shows it); hard = + `edit_file`. The report is the `submit`
argument, else the last text. A provider failure ends the dispatch as
`error`; a harness bug in one dispatch fails that dispatch, not the episode.

`files_modified` lists files newer than a marker made when the subagent
started. In the shared workdir of parallel subagents it includes the
siblings' changes; for sequential subagents and arbiter copies it is exact.

### Prompts and delegation (`prompts.py`)

The prompts are short. The orchestrator's description of subagents never
names or characterizes their model: they are "separate model instances
running in the SAME container", so the same-model control reads the same
text as the Gemma conditions. It states the accounting plainly: "One
spawn_subagents call uses one of your steps", and the subagents' own steps
do not count. The default (all-tools) prompt never asks the orchestrator to
delegate. `subagent.delegation_hint: true` adds one paragraph that states the
trade-off: its steps are scarce, subagent steps do not count, so delegate as
much as the scope allows and keep its own steps for planning, combining
reports and checking.

**Delegation uptake and the manager conditions.** With every tool of its
own, the orchestrator spawned subagents in 0 of 18 Gemma-subagent episodes
on 2026-10-11 (and 0 of 2 same-model ones), and in 0 of 20 subagent-condition
episodes of the complete 2026-10-12 smoke sweep: it solved the tasks itself,
so those conditions measured only the orchestrator. With the delegation hint
it still delegated in 0 of 20 (the `smoke_hinted` sweep, issue #7): in 8 of
them it weighed delegating and judged the task too simple. On 3 medium tasks
it delegated in 1 of 6 hinted episodes and 0 of 6 neutral ones. The three
`manager_*_hard` conditions give it only the read-only tools and add one line
to its prompt: everything that runs a command or changes the container must
be delegated.
Only hard scope makes sense for a manager (otherwise nobody could edit). In
that sweep the managers delegated in 6 of 6 episodes, at about 25 times the
cost and 20 times the wall clock of an all-tools episode. The analysis
fails its delegation gate (exit 3) when a subagent condition never
delegated in any valid episode, so a sweep cannot quietly measure the wrong
thing.

### Interactive sessions (`interactive.py`, `hints.py`)

A batch episode gives the orchestrator one request and a step budget. An
interactive episode (`interactive:` in the config) is a timed session
that measures something else: how fast the agent turns to a new request
while earlier ones are still in progress, the way a user keeps adding
"and also look at X" to a running session. SWE-sweep fits this because one
task is one repository with 6-14 bugs (subtasks) in the sweeps' 10
repositories, each graded on its own hidden tests, so a request about bug k
is a new area of the same codebase rather than a fresh context.

**The schedule.** The harness reads the task dir's
`tests/swesweep_eval.json` on the host (the agent's container has no
`tests/`), shuffles the subtasks with a seed per task (`order_seed`, the same
order in every condition, so conditions are compared on the same sequence)
and sends subtask k as a hint at `k x interval_sec` (300 s), as many as fit
in `wall_clock_sec` (3600 s: 12) and the task has. With
`vary_order_by_round` (the default) round k of a sweep uses seed
`order_seed + k - 1`: the conditions of a round see the same order, and the
rounds sample different ones, so hint position is not tied to one subtask.
The first user message is the task instruction plus `Hint: <hint 0>`
(SWE-sweep's instruction already allows a `Hint:` section in the first
message); later ones say "Another one, while you keep working on the earlier
ones." SWE-sweep's instruction asks for every bug in the repository; with
`scope: reported` (the default) the first message adds that in this session
the user points out the bugs and the agent fixes those, the rules still
applying. Otherwise a bug the agent found and fixed before its hint would
look like a hint it never started (the analysis also records such work as
`pre_touched`/`pre_edited` rather than crediting it). A hint names the
files the reference fix changes (`files`) and by default the hidden tests'
ids (`tests`). Those ids describe the behaviour in most repositories
(`test_symlink_resolution`), not in goawk (`interp::TestInterp`) or libexpat
(`runtests`), which is why `swe_interactive.yaml` uses click, pint, isort
and echo. Some hidden tests extend a test the repository has (echo's
`TestContext_Scheme` gains cases), so the hint says "the version of it that
checks this is not in the repository" rather than that the test is new. `report` instead sends a one-to-three-sentence bug report written
beforehand from the hidden test patch by `subagents-bench hints` (the first
interactive condition's orchestrator model, told not to describe the fix),
read from `jobs/<sweep>/hints/<task>.json`. A scripted, timed user rather
than harbor's simulated user (an LLM that converses through a bridge): the
stimulus must be identical across conditions and arrive at fixed times,
whatever the agent does.

**Delivery.** Messages are delivered only between steps, as one user
message (hints and reports in arrival order, merged into the previous
message when that is a user message, so turns alternate). A hint that
arrives during the orchestrator's LLM call, its own tools or a blocking
`spawn_subagents` waits for the step to end. That queueing is the quantity
of interest: a single agent running a 10-minute test suite cannot hear the
user until it finishes, and neither can an orchestrator blocked on its
subagents. There is no `finish` (a call to it is an unknown tool), and the
note after each step is `[m:ss of 60:00 elapsed; subagents running: sub-3
(2:10), ...; dispatch budget left: D]`. `wait(timeout_sec?)` blocks until a
hint is due, a report is ready or the timeout passes; a reply without a
tool call waits the same way (the end of a chat turn), recorded as a `wait`
tool with `{"implicit": true}`. `timeout_sec` 0 or less checks once,
numeric strings are read, and anything longer than the session is capped.
The session ends at `wall_clock_sec` (a timer sets the recorder's stop
event, every agent unwinds as on harbor's cancel, exit `deadline`) or, with
`end_when_idle`, when every hint was sent, nothing runs and the orchestrator
waits without a timeout (exit `idle`): a timed `wait` runs its course, since
the agent may be waiting on its own background command, and a `wait` that
found nothing left does not end the session when a later tool of the same
step started a subagent. Nothing is delivered after the last step: what is
pending then is recorded undelivered. `budget.orchestrator_steps` is only a
safety cap (2000 in the sweep: every wait is a step). Afterwards the harbor agent runs the cancel's kill script, since
commands of stopped subagents keep running in the container, and records
`end_kill`.

**Background subagents.** With `subagent.spawn_mode: background` (parallel
strategy only), `spawn_subagents` starts each granted instruction on a pool
of `max_concurrent` threads and returns at once, naming the subagents;
granted = min(requested, free slots, dispatches left). Each report is
delivered like a hint when its subagent ends, capped like a blocking spawn's.
`blocking` keeps the batch behaviour, so the three interactive conditions
separate where a hint waits: behind the agent's own commands (single),
behind its subagents (blocking), or nowhere but the LLM call (background).
`delegation_hint` in an interactive episode needs background mode and says
that delegating keeps the orchestrator free to respond; the batch hint's
step argument does not apply without a step budget. For the parallel
strategy, the blocking and background prompts share their Subagents text
apart from when the call returns (the batch text's "one spawn_subagents call
uses one of your steps" would frame delegation differently). The dispatch
budget is 60 in the sweep (the batch study's 15 would run out within
minutes), and subagents get 20 steps: 8 rarely suffice to investigate, fix
and test one bug under emulation.

**Context.** A one-hour single agent can outgrow the model's context.
`max_context_chars` (400000 in the sweep, about 100k tokens) works oldest
message first once the conversation is over the limit, down to 75% of it so
the cached prompt prefix changes rarely: tool outputs and delivered
subagent reports become a marker and old reasoning is dropped (never the
task message or the newest message); `n_elided_results` counts them. The
orchestrator's own messages and tool-call arguments are kept, so the limit
is a target, not a hard bound.

**Checks.** `preflight` fails an interactive sweep when a task has no
subtasks, when `report` hints are missing, or when harbor's agent timeout
(the task's 43200 s x `agent_timeout_multiplier`) is below the session plus
120 s: SWE-sweep's 0.0833 gives 3599 s, which would cancel a 3600 s session,
so `swe_interactive.yaml` uses 0.0903 (3900 s). A spec with an interactive
condition must set `dataset_path`.

### Provider fail-fast (`records.py`, `orchestrator.py`)

The first provider failure after retries in a call of any role
(orchestrator, subagent, judge, debate) sets the episode's stop event. Every
agent unwinds (siblings in flight end `cancelled`), the episode ends with
`provider_error`, and the harbor agent raises `ProviderUnavailableError`
after finalizing, so harbor records an exception instead of grading a task
that was never properly attempted. A billing error does the same and is
re-raised as `BillingError`. On 2026-10-12, a 7-9 minute Sail outage
failed two Gemma calls of a `manager_arbiter_hard` episode after 9 attempts
each: that trial could not measure its condition, and what it spent after
the first failure was wasted. Such a trial is invalid and the sweep reruns
its job.

### Arbiter (`arbiter.py`)

Per granted instruction, `n_candidates` (3) candidate subagents run at the
same time.

**Copy isolation (hard scope).** Each candidate works in a private `cp -a`
copy of the workdir, `/tmp/.sb/arb/<id>/cand<k>`, next to a pristine `base`
copy. Its tool paths, and whole-path occurrences of the workdir in its bash
commands, are rewritten to its copy (a regex anchored on path boundaries,
so `/srv/app` or `$HOME/app` do not match `/app`), and its prompt says so.
After the candidates:

1. The real workdir's md5 manifest is compared with the one taken before.
   A changed file is a leak (a path built at run time, a background
   process): it is restored from `base` and recorded (`isolation_violated`,
   `n_leaked_files`).
2. In each copy, text files that name the copy's own path get the workdir
   path instead. Remapped commands write the copy path into files (a script
   that reads `/app/data.txt` was saved as one reading
   `/tmp/.sb/arb/arb-1/cand0/data.txt`), and once the copies were removed
   such files pointed at a deleted directory.
3. The judge sees each report plus a `diff -ruN` of the copy against `base`
   (not the live workdir, so a leak never shows as a change), capped at 4000
   characters.
4. The winner's changes relative to the pristine manifest are copied into
   the workdir, and files it deleted are deleted.

Every shell step checks each line's status: a copy script that reports
only its last line's status lets a failed copy go unnoticed, and the restore
would then treat the missing files as deleted. Apply and restore run in
batches of at most 100 kB, because a command reaches the container as one
argument, which Linux caps at 128 KiB: applying a winner that created a few
thousand files (a venv, a build tree) failed with "argument list too long".
Only the workdir is isolated: changes outside it (`/etc`, packages,
services, e.g. `nginx-request-logging`'s `/etc/nginx`) are made live by
every candidate. Easy and medium scopes share the workdir and make no copies.

**Judge.** The judge is the orchestrator's model. It must call
`submit_judgement(winner, confidence, agreement, rationale)`; a JSON object
in the text is the fallback. A reply cut off at the output limit without a
parsed judgement gets one retry with a note to reason briefly (2 of 5 judge
calls in the smoke runs used the whole 8192-token limit and gave no
verdict). Both calls are in the round and on the critical path.

**Confidence-aware debate** (arXiv 2602.12276). Unless the judge found the
candidates in agreement with confidence >= 0.8, each candidate makes one
tool-less call that reads the others' reports and revises its own, and the
judge decides again (one round). A failed or unparsed later round never
replaces an earlier parsed judgement. The orchestrator gets the winner's
original report, which is what its applied files reflect, followed by its
post-debate revision under a note, and the list of applied files.

### Harbor agent (`harness/harbor_agent.py`)

`OrchestratorAgent` is a harbor `BaseAgent` loaded with `-a
subagents_bench.harness.harbor_agent:OrchestratorAgent`. It reads the
condition config from `SUBAGENTS_BENCH_CONFIG`, finds the workdir with
`pwd`, runs the synchronous orchestrator in a worker thread, writes the
records to harbor's `logs_dir` (the trial's `agent/`) and fills harbor's
`AgentContext` with tokens and cost per model.

**Cancel.** On harbor's agent timeout (a task cancel) it sets the stop
event, waits up to 30 s for the thread, then kills every agent command still
running in the container: cancelling a `docker exec` client leaves the
process inside running, and the verifier grades the same container next. It
kills the process group in every `pid_<n>` file the tool wrapper left, and
removes `/tmp/.sb/arb`. The result goes to `episode.cancel_kill`, and the
episode ends `cancelled`. An agent timeout is an outcome: harbor still runs
the verifier.

### Records (`records.py`)

One JSONL file per kind (calls, tools, steps, spawns, dispatches,
arbitrations, inbox, transcript) plus `episode.json`, written as events happen by
a thread-safe recorder. Failed and abandoned LLM calls are recorded too.
`finalize` computes every episode total from the records written, so the
totals equal the JSONL sums, and closes the episode: records that arrive
later (a thread still unwinding after a cancel) are dropped. Fields:
docs/DATA.md.

## Sweep (`sweep.py`, `cli.py`)

A spec names the base config, dataset, tasks, rounds, `order_seed`, harbor
settings (`n_concurrent` 4, `agent_timeout_multiplier` 2.0) and conditions;
`extends:` merges it over another spec (smoke extends pilot). A job is one
(round, condition): `harbor run -d terminal-bench@2.0 -i <task>... -a
<agent> -e docker -n 4 -k 1 -o jobs/<sweep> --job-name <sweep>__r<k>__<cond>
--agent-timeout-multiplier 2.0 -y -q` with `SUBAGENTS_BENCH_CONFIG` set.
Each round shuffles the condition order with its own draw from the seed, so
drift over time (provider load, caches) does not line up with a condition,
and `--rounds`/`--only` never change an order.

**Resume.** A job whose dir has harbor's finished `result.json` and no trial
needing a rerun is skipped. An unfinished job, or one with a trial needing a
rerun (provider failure, billing error, a harbor exception before the
verifier: environment setup, or an agent crash; on SWE-sweep a grader or
verifier container that could not grade, see Validity), is moved to
`_partial/<job>.<k>` and rerun whole; nothing is deleted. Rerunning whole
jobs keeps every trial of a job from one harbor run; the cost is that its
valid trials run again (with a per-trial failure rate p, a 10-trial job
needs 1/(1-p)^10 runs, about 2.9 at p = 0.1).

**Live lines, probe and early stop.** While a job runs, the sweep checks its
dir every 10 s and prints one line per finished trial (reward, exit reason,
wall clock, provider errors, harbor exception), flushed, since its output
usually goes to a log. When a trial ended with a provider failure, it sends
one tiny request to every model (1 retry, 60 s stall limit, 30 s retry
budget). If one still fails, it stops the job with one SIGINT and exits 4,
naming the model and the error; if they answer, the failure was transient,
and the job is rerun next time. harbor resolves the dataset online at the
start of every job, so with no network it exits before creating the job
dir; that also triggers the probe. A billing error exits 3.

**Process handling.** harbor runs in its own process group and gets exactly
one SIGINT: a terminal Ctrl-C used to reach it twice (from the terminal and
from the sweep), and the second SIGINT cancels the teardown of harbor's
`asyncio.run`, which left task containers behind. harbor starts with SIGINT
at its default: a sweep launched with `&` from a script inherits SIGINT
ignored, and harbor would ignore the stop and be killed two minutes later,
its containers left running. SIGTERM and SIGHUP stop the sweep like Ctrl-C,
unless SIGHUP was already ignored (`nohup`). The sweep runs the `harbor`
next to its own Python (pinned in `uv.lock`) before one on PATH: a globally
installed harbor would otherwise run every job, with another version and
without this package importable. On macOS, `run_experiment.sh` runs the
sweep under `caffeinate -i`: a sleeping laptop drops the network and every
running call with it.

`preflight` checks Docker, harbor, `SAIL_API_KEY` and one request per model,
and with `--oracle` runs harbor's oracle agent on every task (reused while
it still solves them all), so a broken task image is found before the
sweep spends money on it.

**A second benchmark is a spec, not a code path.** Four spec keys cover
what differs: `dataset_path`, a local directory of harbor task dirs passed as
`harbor run -p` instead of a registry `dataset` (preflight checks that every
task has a `task.toml`); `platform`, below; `harbor.extra_args`, flags
appended to every harbor run (the oracle's too, which never gets the agent
timeout multiplier); and `overrides`, deep-merged onto the base config
before every condition, so a sweep-wide setting such as the step budget or
the command timeouts is set once for every condition that does not set its
own. A condition's own setting wins: `single_matched` keeps the 40 steps
(5 x 8 subagent steps) `pilot.yaml` gives it, so a larger step budget also
needs its matched budget under the spec's `conditions:`, which `extends`
deep-merges into the inherited conditions (`swe_sweep.yaml` shows both
edits in comments). `configs/sweeps/swe_sweep.yaml` uses all four keys.
SWE-sweep's tasks are a git clone; its tasks request 16 CPUs and 64 GB
(`--override-cpus 3 --override-memory-mb 7000`, which harbor
applies to the verifier container as well); and its test suites run for
minutes under emulation (`command_timeout_sec` 600,
`max_command_timeout_sec` 1800, agent timeout 43200 s x 0.0833 = 1 h). In
the user's runs of 2026-10-09 (mini-SWE-agent, 15-minute cap, emulated),
the first build of a small repository took 5-6 minutes, the verifier
2-5 minutes (goawk, symfony) and up to 13 minutes for a Maven project, which
is why the sweep's 10 repositories are Go, Python, JavaScript and C.

**Platform.** On Apple Silicon SWE-sweep's images must be `linux/amd64`:
the verifier image bundles an x86-64 Python. With `platform` set, the sweep
(and the oracle preflight) copies its tasks to `jobs/<sweep>/tasks/`, pins
every `FROM` of `environment/Dockerfile` and `tests/Dockerfile` that has no
platform to `FROM --platform=linux/amd64` (upper-case `FROM` only: a
lower-case `from distutils...` continuation line sits at column 0 in
pendulum's Dockerfile) and passes the copy to harbor; a rerun rewrites
identical files. Two simpler routes failed. A compose overlay with
`services.main.platform` (the user's 2026-10-09 route) does not reach
harbor's separate verifier environment, which harbor builds without the
overlays. A global `DOCKER_DEFAULT_PLATFORM` broke harbor 0.24.0 before
any trial: its egress-control probe runs
`docker run alpine:3.23.4@sha256:...`, a multi-arch digest whose arm64
variant this overlay2 image store already holds, so the amd64 run failed
with "cannot overwrite digest", the probe reported no kernel support, egress
control stayed off, and the tasks' `no-network` phases were rejected
(`ValueError: network_mode='no-network' is not supported`); the egress
sidecar image is built for arm64 too. With or without `platform`, a sweep
on a `dataset_path` writes `jobs/<sweep>/tasks.json`: the source dir, its git
commit and dirtiness, and per task a hash of the dir harbor runs and its bugs
(below).

## Analysis (`analysis/`, `plots/`)

**Validity.** A trial is invalid exactly when the sweep would rerun it
(`sweep.retry_reason`), or when it has no episode or no result. The reason
is `billing_error`, `provider_error`, `harbor_exception` (before the agent
started, or a job cancel), `agent_crash`, `no_episode` or `no_result`. An
agent timeout is valid and scored, and a valid trial without a reward
counts as reward 0, unsolved, with no bug fixed and its task's bugs in the
denominator, so `resolve_rate`, `reward_mean` and `bug_fix_rate` all count it
alike. A SWE-sweep trial whose grader reports `grader_failed`
(the verifier's own setup failed, not the patch) is `grader_failed`: invalid
and rerun, like a harbor exception. So is, as `verifier_failed`, any harbor
exception without a reward from a verifier that runs in its own container
(`verifier_environment_mode` `separate` in harbor's `result.json`, every
SWE-sweep task): harbor marks the verifier started before it builds and
starts that container, so a failed image build or start, a verifier timeout
(the task's limit covers the grader's own per-phase limits) or a missing
reward file would otherwise count as an unsolved trial that is never rerun.
The agent cannot break a fresh container, and the grader scores the patch's
own failures (it does not apply, or does not build) itself. On
Terminal-Bench the verifier shares the agent's container, which the agent
can leave broken, so an exception after the verifier started stays an
outcome there.

**Partial rewards.** SWE-sweep's reward is the fraction of the task's bugs
whose hidden tests pass, 0 when a visible test regressed, with the grader's
counts next to it in harbor's `result.json`. `episodes.csv` adds
`n_subtasks` (the task's bugs: the subtasks of its `tests/swesweep_eval.json`,
recorded in `tasks.json` by the sweep; the grader's own count is 0 when the
agent's patch breaks the build, which would drop those bugs from the
denominator and raise `bug_fix_rate`), `bugs_fixed` (the grader's
`n_resolved`, 0 on a regression, as SWE-sweep's own scoring counts it; the
reward is rounded to 4 places, so 5 of 14 bugs would give 0.3571 x 14 =
4.9994) and `regressed`;
`conditions.csv` adds `n_bugs`, `bugs_fixed`, `bug_fix_rate` (bugs fixed over
bugs in the valid trials, the micro-average SWE-sweep reports), `n_regressed`
and `reward_mean`. `resolved` stays reward >= 1, every bug of the task fixed,
which is rare there: in the SWE-sweep paper no model fully repairs any of the
100 repositories (Table 16), and the best agent, Sol 5.6 (xhigh), fixes 4.7%
of all bugs (Table 4) and 14.9% in the repositories with fewer than 25 bugs,
as all ten here are (Table 5). So the resolve-rate figures say little for
this benchmark; read `bug_fix_rate`. Terminal-Bench
trials leave the bug columns empty.

**Cost.** Tokens times per-model prices from `configs/pricing.yaml`, per role
(scheme `sail_balanced`, USD per 1M tokens: DeepSeek-V4.1-Flash in 0.07,
cached 0.005, out 0.22; Gemma-4-31B in 0.11, cached 0.06, out 0.32). Another
scheme in the file reprices every run without rerunning it.

**Latency.** The critical path of each episode, from the records: the
orchestrator's calls and tools in series; a `spawn_subagents` tool counts
as its group instead: parallel = the slowest dispatch, sequential = the sum,
arbiter = per arbitration the copy setup, the slowest candidate, per round
the slowest debate call plus the judge calls, and the rest (diffs, restore,
apply, cleanup) as tool time; a dispatch is its calls and tools in series.
It is split into orchestrator LLM, subagent LLM, judge + debate LLM and tool
time; overhead is wall clock minus critical path, and `summary.json` counts
episodes whose critical path exceeds their wall clock (a check that should
stay 0). With the test fakes, overhead is under 10% of wall clock (thread
start-up, the marker and `find` execs).

**Interactivity.** `analysis/interactivity.py` turns each hint of an
interactive episode into a row of latencies after the hint was sent:
delivery (into the conversation), response (the first orchestrator call that
saw it returned), dispatch (a subagent started whose instruction names a
target file or a hidden test's name), touch (any tool call naming a target
file) and edit (an `edit_file` on one, or a bash command segment naming one
that writes), plus what the orchestrator was doing when it arrived, what
had happened to it when the next hint arrived (not started, continued,
stopped; a stopped hint later counted fixed was finished, otherwise
dropped), and the grader's verdict for the subtask from
`verifier/eval.json` (a valid trial without a verdict, e.g. a patch that
broke the build, counts it unfixed, as the reward does). Summaries leave out
hint 0, which arrives with the task. Tool records
keep their arguments (strings cut to 2000 characters) and dispatches their
instruction for this. File matching is textual: a path or its last two
components, or a bare basename not inside another path (an agent that `cd`s
into the package), only when no other hinted target file has that basename
(pint has three `quantity.py`). It is a heuristic in two ways. Subtasks
share files: 2231 of SWE-sweep's 4068 subtasks share a target file with
another subtask of their task, so work on an earlier hint's bug can register
as touching a later one; the rows flag `shared_files` (an earlier hint) and
`shares_any` (any other), the summary reports touch and edit for unshared
hints too, and the status at the next hint only over hints that share no
file (echo's 6 bugs are in 6 files). And `search`
over the whole repository is not a touch. Missing events (never happened
before the session ended) count as infinitely late in the medians, and
`*_before_next` is the share within one interval, before the next request.
For a background spawn the critical path counts only starting it (its
subagents run off the orchestrator's chain), so it stays below the wall
clock; waits, explicit and implicit, are tool time.

**Statistics.** Resolve rates with Wilson 95% intervals; mean cost, wall
clock and tokens with a seeded percentile bootstrap (2000 resamples).

**Figures.** Nine PNGs (the ninth for interactive sessions) with captions in `index.md` (docs/DATA.md). Colour
is the strategy (parallel blue, sequential orange, arbiter aqua; single agent
grey); critical-path parts use orchestrator blue, subagent orange, judge
aqua, tools yellow and overhead grey.

## Tests

Offline by default: a scripted fake LLM client returns real `LLMResponse`
objects, the local shell is the environment, and a fake `harbor` executable
writes harbor-0.24.0-shaped job dirs. The analysis and figure tests run the
real runtime with fake models and wrap each episode in a harbor-like trial,
so no synthetic data generator stands in for the records. `RUN_DOCKER=1`
adds container tests (tools in Ubuntu and Debian images, arbiter isolation,
the cancel kill), `RUN_LIVE=1` one tool-calling request to each Sail model.
