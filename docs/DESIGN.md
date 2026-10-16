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
independent samples (`grade-answers` judges at 0: a verdict is not a
sample). DeepSeek's `max_output_tokens` is 32768: at 8192, the
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

**Several tool calls in one Gemma reply.** Sail's tool-call parsing for
Gemma-4-31B fails when a reply holds more than one tool call. It streams the
first call, sometimes under index 3, then an `upstream_error` chunk ("chat
completion failed"). The same request fails the same way on every retry, so
the provider fail-fast ended the session. In the interactive sweeps of
2026-10-14 this happened on 3 of Gemma's 4 two-call replies, and on all 12
replays of them; its 415 one-call replies never failed, and
`parallel_tool_calls: false` is ignored. When a stream error comes once a tool
call has started, the client retries with a note at the end of the system
prompt ("Reply with exactly one tool call"). The model then answers with one
call: 9 of 9 replays succeeded on that retry. The call records
`one_call_retry`. Every other request is sent unchanged.

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

**Interrupts.** With `interactive.interrupt`, a user message no longer waits
for the orchestrator's LLM call to end. Calls often reason for minutes (2-7
in the levels sessions, once 415 s up to the 32,768-token output limit),
and a message that arrives meanwhile waits that long to be read. The
recorder runs the orchestrator's call with a `CallCancel` (`llm.py`) and
checks the interrupt rule every 0.2 s (`records.POLL_SEC`). When the rule
holds, the stream stops at its next SSE line, which closes the HTTP
response, and `CallInterrupted` is raised. The reply streams a chunk every
few milliseconds, so the abort lands within about 0.3 s. A stream that
sends nothing for 2 s after the cancel (`ABORT_GRACE_SEC`) is abandoned
and closes on its next line. A cancel during a retry's backoff ends the
backoff, and nothing is retried after a cancel. The partial reply is
dropped: no assistant message, nothing in the transcript, no tool runs. The
step ends with `interrupted: true`. The next step delivers everything due
(the message, any later ones, ready reports) as usual and calls the model
again with them.

The rule (`InteractiveOrchestrator._interrupt_due`): a call is interrupted
when a user message of any kind (bug, question, status check) is due, and
- the call has run at least `interrupt_min_sec` (5 s). A message that comes
  due earlier interrupts at the 5 s mark. Every call gets a 5 s start, so
  dense messages cannot abort calls back to back faster than that;
- its reply has not started a tool call. The reasoning is over then, and
  the decided action is let through; the message comes with the result;
- fewer than `interrupt_max_in_row` (3) calls in a row were interrupted.
  The 4th call in a row runs to its end, as without interrupts, so a model
  that always reasons longer than the gap between messages still acts.
Subagent reports never interrupt. They are delivered with the message that
does, or at the step's end as before. Subagents, the arbiter and the
orchestrator's tools (a long `bash`, a blocking spawn) are never
interrupted. A message that arrives during a tool waits for it as before;
the `wait` tool already wakes on a message. The prompts are unchanged:
nothing tells the model about interrupts, so an interrupt condition differs
from its levels twin only in this mechanism.

The aborted call stays in `calls.jsonl` with `error` and `finish_reason`
`interrupted`. It is not a provider error (`n_provider_errors`, the retry
rules and validity ignore it), and `episode.json` counts it in
`n_interrupts`. Its cost still counts: the provider billed the prompt and
whatever it generated. Usage comes only in the stream's last chunk, so the
record usually holds estimates (`tokens_estimated`). Prompt tokens are the
request's characters (`prompt_chars`) at the ratio of the orchestrator's
last call that reported usage (4 characters a token before any). Cached
tokens are that call's prompt tokens (the prefix the provider has seen), at
most the estimate. Sail's cache covered the previous prompt in 256-token blocks in the
levels sessions. Completion tokens are the streamed chunks (`n_chunks`,
about one token each); reasoning tokens are the reasoning chunks. Every
call records `n_chunks`, so the chunk-per-token ratio can be checked on the
completed calls. In the interactivity analysis an interrupted call ends at
its abort: a hint's `busy_at_arrival` is `llm`, `blocked_sec` is what was
left of the call, about 0 to 5 s. `orch_llm_median_sec` counts only
completed calls.

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

**Session note.** `interactive.session_note` adds a paragraph to the
first message, before hint 0, the same in every condition; the slow-command
experiment uses it to make every agent test each fix, so its own commands
run for minutes.

**Fairplay flags.** SWE-sweep forbids learning a fix from the code's
"future" or the network. `analysis/fairplay.py` flags tool calls of any
agent that probe git objects or refs beyond the working tree, use the
network, or read other installed copies; the analysis warns and reports them
per tool, episode and condition. A flag is for review: the task images'
git holds only the `baseline` commit, so the first smoke's `git cat-file
--batch-all-objects` found nothing.

**Checks.** `preflight` fails an interactive sweep when a task has no
subtasks, when `report` hints are missing, or when harbor's agent timeout
(the task's 43200 s x `agent_timeout_multiplier`) is below the session plus
120 s: SWE-sweep's 0.0833 gives 3599 s, which would cancel a 3600 s session,
so `swe_interactive.yaml` uses 0.0903 (3900 s). A spec with an interactive
condition must set `dataset_path`.

#### Requests

The bug-only sessions measure when the agent turns to a request, read off
its tool calls. That leaves out what the user sees: an agent can keep working
on the bugs it knows, score well on those latencies and never tell the user
anything, and nothing rewards it for staying available for requests that have
not arrived. With `interactive.replies` the user also asks questions about the
code and checks on progress, and sees only what the agent sends with `reply`;
the questions are about the user's time (answered or not, how fast, right or
not), the bugs about fixes and about how well the agent predicts when each
will be done.

**The stream.** Bugs arrive at `k x interval_sec` as before.
`questions_per_interval` q puts a question at `k x interval_sec + j x
interval_sec / (q + 1)`, j = 1..q, after every bug including the last (below
`wall_clock_sec`); `status_every` n puts a status check at `(k + 0.5) x
interval_sec` after every bug k with `(k + 1) % n == 0`; `urgent_every` n
marks bugs n, 2n, ... urgent (never bug 0). Messages due at the same time go
bug, status check, question, and get ids 1, 2, ... in that order. The texts
are `Request #4: another bug, while you keep working on the earlier ones.`
(urgent: `... and this one is urgent: fix it before the earlier ones.`),
`Question #5: <question>` and `Status check #6: how long until each of your
bug requests so far (#1, #2, #4, #5) is fixed? Reply to each one's own id
with eta_min, or with done: true if it is fixed.` (the bugs sent before the
check; it replaced an earlier text, below, *Status checks that name the
bugs*); the first message ends with `Request #1: a bug.` and its hint. Questions are drawn
from the task's pool shuffled with `<seed>:<task>`, so the conditions of a
round get the same questions in the same order and rounds differ. Questions,
status checks or urgent bugs without `replies` are a config error (each comes
with message ids and `reply`); without `replies` nothing changes (no ids, the
earlier texts). With `end_when_idle` the session can end only after the last
message of any kind was sent.

**The sweep.** `swe_interactive_requests.yaml` runs 30-minute sessions (6
bugs every 4 minutes, the last at 20:00, then 10 minutes; harbor's timeout
0.05 x 43200 s = 2160 s: the 1800 s session, preflight's 120 s grace, and
240 s more for the background subagents' shutdown and the end kill) on click
and pint, 2 rounds. Three conditions get the requests (`single_requests`,
`manager_background_requests`, `background_hinted_requests`); two controls,
`single_bugsonly` and `manager_background_bugsonly`, set `replies: false`
and no questions, status checks or urgent bugs, so they get the same bugs at
the same times as plain hints. They are the baseline for what answering
costs in fixes and in how fast a new bug is picked up (`hints.csv`,
`interactivity.csv`), run on the same day and provider as the requests; the
arrival-rate sessions are a looser second reference (10 bugs, another day).
5 conditions x 2 tasks x 2 rounds = 20 sessions.

**The reply tool.** `reply(to, text, eta_min?, done?)` writes a line to
`replies.jsonl` and returns `[sent to the user, re #3]`; nothing is sent back.
`to` must be the id of a delivered message (`3`, `3.0`, `"3"` and `"#3"` are
read; any other value, `true` or a string of 10 or more digits included, is
an error that lists the delivered ids), `text` non-empty. `eta_min` is read as a number of minutes in [0,
10^6) (a numeric string too); `done` as `true`/`false`, `1`/`0` or the
strings `"true"`, `"false"`, `"yes"`, `"no"`. Anything else is not dropped
silently: the reply is still sent, without the estimate (null) or with
`done` false, and the result says so (`[sent to the user, re #3; eta_min
ignored: expected a number of minutes >= 0]`, `...; done ignored: expected
true or false`), so the agent can resend it. The first reply to a question
also says it was taken as the answer (`[sent to the user, re #5; taken as
your answer to this question]`); a later reply to the same question, or a
reply to a bug or status check, does not. So an agent that acknowledged a
question with "on it" learns, at that step, that the acknowledgement is what
gets graded. The system prompt adds a
*Replying to the user* section: the user reads only `reply`, not the agent's
other text or its tool calls; answer a question from the code, and the first
reply to a question is taken as the answer, so send it when you have it;
reply to a bug with `eta_min` (minutes from now) in the first response after
it arrives, again whenever the estimate changes, and with `done: true` once
it is fixed and checked; answer a status check with an updated `eta_min` for
each open bug; fix an urgent bug before the earlier ones. Asking for the
estimate in the first response (not "once planned") makes `eta_sec` a
latency the conditions share, and telling the agent the first reply is the
answer keeps an "on it" acknowledgement from being graded. One channel is what makes "when was the question answered"
a measurement: in free text an answer can sit in any assistant message, mixed
with reasoning or stated only implicitly, while here it is the first `reply`
to the question's id, at a known time, with a text to grade. The cost is that
an answer written as plain text, or sent with `to` naming the wrong message,
counts as no answer. The section is the same in every condition, appended
after the tools; the conditions still differ only in tools, spawn mode and the
delegation hint. With `interactive.require_eta` a reply to a bug must carry
an estimate or a done (below, *Estimates required*).

**Questions.** `subagents-bench questions SPEC`, once before the run, clones
each task's repository at the ref its Dockerfile checks out (`git clone URL
/src`, `git -C /src checkout REF`: the source in the image) shallowly to
`jobs/<sweep>/src/<task>` (`git clone --depth 1 --branch REF`; when REF is a
commit sha, which `--branch` cannot take and 4 SWE-sweep Dockerfiles check
out, the failed clone is removed and the whole history is cloned and the sha
checked out), lists its source files (`.py .go .js .ts .c .h .rs
.java .rb`, 1500-40000 bytes) minus the files any subtask touches, shuffles
them per task, and asks the orchestrator model of the first interactive
condition for two questions per file until it has the most one session asks
(`--per-task` overrides; 12 in `swe_interactive_requests.yaml`). Skipped:
a dot dir at the top; any directory in `SKIP_DIRS` (`doc(s)`,
`example(s)`, `vendor`, `node_modules`, `bench`, `benchmark(s)`,
`script(s)`, `tools`) or whose name starts or ends with `test` (`tests`,
`testing`, `unittest`); a file whose name has `test` or `tests` as a
whole `_`-separated word (`test_units.py`, `units_test.go`, `tests.py`) or
is `conftest.*` (a `test` inside a word, as in `latest.py` or
`testing.py`, is kept). The touched files (`touched_files`) are the
subtasks' target files plus every file named in a `diff --git` header of the
task's oracle and subtask patches (`tests/oracle/*.patch`,
`tests/subtasks/*/*.patch`): target files can name a later layout than the
image's source (pint's `pint/facets/numpy/numpy_func.py` is
`pint/numpy_func.py` in the image), so excluding them alone would let a
question land on a file a fix changes. Each question must be answerable from
that file alone in a sentence or two and name the file or function; the
reference answer is a value, a name, an exception type or a short
behaviour. Excluding the touched files keeps the questions off code a fix or
its tests change (where the reference could turn wrong mid-session, or the
question could hint at a bug); the agent can still edit a question's file
before it is asked. A failed call or a reply that is not JSON skips its file
(a `{"questions": [...]}` object is unwrapped); with too few questions the
error gives the eligible files and how many were skipped. A billing error is
not one file's failure: it is re-raised and stops the task's generation,
rather than being counted as a skip on every remaining file. Existing
questions are kept unless `--force`, except those about a file now touched
(written before a patch naming it was known): they are dropped, and their
file is not asked about again.
Written to `jobs/<sweep>/hints/<task>.questions.json`; `preflight` builds
every schedule and fails when a task has fewer questions than a session asks.
The questions and reference answers are the model's and are not checked by
hand, and the generator, the judge and the agents can be the same model.

**Grading answers.** `subagents-bench grade-answers SPEC`, after the run,
judges the replies to each question of every finished trial (a
`result.json`; `_`-prefixed job dirs skipped) against the reference answer,
with the same model as `questions` (version 2 of `answers.json`: every reply
up to the first correct one, three verdicts, temperature 0; below, *Grading
answers, version 2*). `requests.csv` reads the first reply's verdict
(`correct`): the user may act on it, and a later correction would let an
agent answer fast and fix the answer at leisure. Unanswered questions are
not sent to the judge and have no verdict (`correct` null): `requests.csv`
gives accuracy both over answered questions (`correct_of_answered`) and over
every question asked (`correct_of_asked`, an unanswered one counting as not
correct), so speed and accuracy are not mixed in one number. Rerun the
analysis after grading.

**Estimates.** Done is the first reply with `done: true` (`done_sec`). A
bug's estimate is its first reply with `eta_min` sent before that done reply
and not itself the done reply, sent at `eta_sec`; it predicts done at
`eta_sec + 60 x eta_min`. An `eta_min` on the done reply or after it is not
an estimate (nothing left to predict). `eta_log2_err = log2(actual /
predicted)`, with predicted `60 x eta_min` and actual the time from the
estimate to done, both floored at 30 s so an estimate of 0 or an immediate
done stays finite; > 0 is late, 1 is twice as long as predicted. Only the
first estimate is scored, as what the user is first told; it and the later
ones before done (status checks ask for them) are counted in
`n_estimates`. A bug estimated and never said done is late: its error is
+inf and `on_time` false, when the session ran past the predicted time or
ended for any reason but its deadline (`exit_reason` other than `deadline`
or `cancelled`: the agent stopped, e.g. `idle` or `finished`, so it gave the
bug up). Only when the deadline (or harbor's cancel) came before the
predicted time is it censored: error and `on_time` null, since it might
still have made it. A bug never estimated has neither. Medians of the error
keep the +inf values (`eta_abs_log2_median` can be +inf when most estimated
bugs were never said done) and `n_eta_never_done` counts them, so a
condition that promises fixes and never delivers does not look accurate by
dropping its misses. Read the error with `done_claimed_frac` and
`on_time_frac`.

`eta_log2_err` scores the estimate over its own horizon, from when it was
sent. A late estimate (given after most of the work) has a short horizon, so
its ratio says little about the wait the user had: a miss of a minute is a
factor of 2 on a 1-minute horizon, and a horizon under 30 s is floored to
an error near 0. `eta_total_log2_err = log2((done - sent) / (predicted
done - sent))`, both floored at 30 s, scores the same prediction from when the
bug was sent, against the user's whole wait: a bug sent at 0, estimated at
9:00 as 2 minutes and done at 10:00 is -1 on its own horizon (60 s for 120)
and log2(600 / 660) = -0.14 from the arrival; one estimated at 0:10 as 1
minute and done at 10:00 is log2(590 / 60) = 3.3 and log2(600 / 70) = 3.1.
The never-done and censored cases are those of `eta_log2_err` (+inf, null).
`eta_total_abs_log2_median` is its median absolute value. Neither error
charges for estimating late as such (an estimate given just before done is
close to right on both): that is `eta_sec`, so read the two errors with it.

**Done claims.** `claim_ok` is a done claim the grader confirms: the
subtask resolved with no visible regression, graded once at the end of the
session, not when the claim was made. A bug fixed and later broken by other
work, or any visible regression in the trial, makes its claim false.
`claim_precision` is over claimed bugs; `fixed_frac` over all bugs, claimed
or not.

**Urgent first.** For an urgent bug, the earlier bugs still open when it was
sent (no done reply by then); `urgent_first` is the share of them it was said
done before, a bug never said done counting as later (an urgent bug never
said done scores 0), and null when none was open. It uses done claims, not
the grader or edits: it measures the order the agent reports, which is what
the user sees.

**Status checks.** The agent is asked to reply to each open bug, so its
answers go to the bugs' ids; a status check's `reply_sec` is the time from
when it was sent to the first reply of any kind made after it was delivered
(a reply in the step that saw it, or later), which a reply the agent was
about to send anyway also satisfies; a reply made before the agent could see
the check does not count, and a check never delivered has none. Its
`eta_min` updates are not scored on their own (they add to `n_estimates`).

**Analysis.** `hints.csv` and `interactivity.csv` still cover only bugs
(inbox kind `hint`, now with `urgent`); `analysis/requests.py` builds
`messages.csv` (per message) and `requests.csv` (per condition, valid
trials). Medians of times count a missing reply as infinitely late (NaN when
more than half are missing); `first_reply_median_sec` and `eta_median_sec`
leave out bug 0, which arrives with the task. `episodes.csv` adds
`n_questions` and `n_replies` per episode. The bugs-only controls have no
`reply`, so they have no rows in `messages.csv` or `requests.csv`; compare
them with the request conditions in `hints.csv` and `interactivity.csv`.

##### Levels

The requests run sent every kind of message at once, so a difference
between conditions could come from any of them, and neither its done claims
nor its end-of-session grade said when a bug was fixed (below).
`swe_interactive_levels.yaml` adds the kinds one at a time on top of the
bugs-only baseline on Modal (`swe_interactive_baseline_modal.yaml`: 6 bugs
every 4 minutes in one fixed order, 30-minute sessions, every model
DeepSeek-V4.1-Flash, click and echo), in four cumulative levels, each in the
baseline's three settings (`single`, `manager_background`,
`background_hinted`), so 12 conditions named `<setting>_<suffix>`:

| Level | Suffix | Adds to the level below | Config |
|---|---|---|---|
| L0 | `_baseline` | nothing: bugs only | |
| L1 | `_eta` | a status check 2 minutes after bugs 1, 3 and 5 (6:00, 14:00, 22:00) | `status_every: 2` |
| L2 | `_questions` | 2 questions about the code after each bug (+80 s, +160 s; 12 per session) | `questions_per_interval: 2` |
| L3 | `_urgent` | bug 3 (12:00) is urgent | `urgent_every: 3` |

Bugs are numbered from 0, so bug 3 is the one sent at 12:00. Every level has
`replies: true`, which already puts the whole *Replying to the user* section
in the system prompt (estimate each bug, answer questions, fix urgent bugs
first), so the levels share one prompt and differ only in the messages they
receive. Bugs keep their send times and order at every level and in every
round (`vary_order_by_round: false`; `test_swe_interactive_levels_spec`
checks it on the tasks); message ids shift as checks and questions
interleave, so the analysis pairs bugs by subtask, never by `msg_id`. A bug's
*twin* is the same subtask sent at the same time in the same sweep, task and
setting, at another level. L2 - L1 then measures questions with status checks
present and L3 - L2 urgency with both (the urgent bug gets a question at
+80 s, a check at +120 s and another question at +160 s); L3 - L0 mixes all
three kinds and is only descriptive. There is one urgent bug, bug 3: an
urgent bug at 4:00 would have only bug 0 before it, which the single agent
often finishes first, so there would be nothing to jump ahead of, and a
second one would carry its effect into the first one's comparison. In both
tasks bug 3 (`click-4f1302f13075`, `echo-91d2b5077dbf`) is the only bug in
its file and has three bugs before it.

Every condition records workspace snapshots and has the `require_eta` rule
(the spec's `overrides`; both below). L0 runs again inside the sweep,
shuffled with L1-L3 within each round, rather than being taken from the
earlier baseline run on Modal (`swe_interactive_baseline_modal`, PR #13):
that run has no snapshots, so no per-bug fix times, and no `require_eta`,
and it ran on another day. It stays a looser
second baseline, kept apart from the in-sweep L0 in every table (*Levels
analysis*, below). Rounds 1-2 run all 12 conditions (24 jobs, 48 sessions);
rounds 3-4 only L2 and L3 (12 jobs, 24 sessions), because the urgency win
rate pairs urgent and twin sessions and 2 rounds give only 4 pairs per task
and setting. With the order fixed, the rounds of a condition are identical
runs, so their spread is the replicate noise; with 4 to 8 sessions per
setting and level, expect directions, not significance. Click's 12
questions are the requests run's, copied into the sweep's hints dir before
`subagents-bench questions` wrote echo's. The plan is PR #13's plan comment
with Jonathan's follow-up (`require_eta`, rounds 3-4, all jobs at once on
Modal).

##### Workspace snapshots (`snapshots.py`)

The grader runs once, at the end of a session, and nothing else in the
records says when a bug was fixed. Done claims are a poor stand-in: in the
requests run's 12 sessions with replies, 22 of 72 bugs were said done, 4 of
those claims were wrong (all on pint), and 11 of the 29 fixed bugs were never
claimed (9 of 25 on click). Edits are worse: an edit is not a passing fix,
fixes get reverted (in 5 of the run's 20 sessions, all
`manager_background`, a subagent ran `git checkout` or `git stash` over files
another agent had edited; in one, at 28:01, that wiped the work on two bugs,
which ended unresolved), and 3 of click's 6 baseline bugs are in `core.py`.
So with `interactive.snapshots` a `Snapshotter` thread records the workspace
while the session runs, and `fix-times` grades each recorded state
afterwards.

**The command.** One `env.exec` (120 s timeout) of the task's collect hook,
`git add -A && git diff --cached --binary baseline`, made private:
`GIT_INDEX_FILE` and `GIT_OBJECT_DIRECTORY` point under
`<state_root>/snap` (`/tmp/.sb/snap`; an index file per attempt, removed on
exit, so a killed attempt leaves no lock), with the repository's own object
dir as an alternate. The repository's `.git` (index, objects, refs) is never
written: a `git add -A` on the real index would stage the agents' work, and
their own `git diff` and `git status` would change under them. The patch is
byte for byte what the hook would write at that moment. The private index
starts as a copy of the repository's own (`cp -p`, mtime kept): from an
empty index, tracked files that `.gitignore` matches would show as deleted,
and the kept mtime lets git's racy-entry check trust the same file stats as
the hook. `git write-tree` gives the state's tree id; the same tree as the
last successful snapshot's skips the diff, and a new one's patch comes back
base64-encoded (a patch need not be UTF-8).

**Triggers and coalescing.** Snapshot 0 at the session's start (the
untouched workspace: an empty diff); after every `bash` or `edit_file` call
of any agent (a recorder tool listener, `EpisodeRecorder.add_tool_listener`,
which only sets a flag, so the agent's call never waits for a snapshot and
nothing it raises reaches the call); and, while nothing asks for one,
`snapshot_every_sec` (15 s) after the previous one ended (`heartbeat`: a
command still running or a background process changes files too). One
snapshot runs at a time; the requests made while it runs become exactly one
more, right after it, whose `t_request` is the earliest of them.

**Records.** One `snapshots` record per attempt (docs/DATA.md). A state is
`changed` when its tree differs from the previous successful snapshot's, so a
return to an earlier state is recorded again (a fix that is reverted and
redone is three states, not one). Only a changed state's patch is written, to
`agent/snapshots/<seq>.patch`; one over 20 MB is the error `too large`,
without a file. A failed attempt (a failed or timed-out exec, unreadable
output) is recorded with its error and the next trigger tries again; the
next snapshot is compared with the last one without an error. The deadline's
stop ends the thread, the session stops it when it ends (joined for up to
30 s), and nothing is written after that. There is no final snapshot: the
final state is the trial's own `model.patch`, which the collect hook writes
after the session.

**Overhead.** On Modal every snapshot is a remote exec, so the levels smoke
(2026-10-16, 24 sessions) checked that the recorder does not slow the agents
before the full run: 609 snapshots (24 at the start, 499 after a tool, 86
heartbeats), none failed; a snapshot took a median 0.09 s (90th percentile
0.18 s); and the agents' own `bash` calls took a median 0.084 s when they
overlapped a snapshot and 0.100 s when they did not.

##### Estimates required (`require_eta`)

In the requests run, 72 of 106 estimates were only in a reply's text, where
the requests analysis does not read them: 58 in replies to a status check's
own id, 14 in bug replies without `eta_min`. With `interactive.require_eta`
(it needs `replies`; on in every levels condition), `reply` to a bug with no
usable `eta_min` (absent, or not a number of minutes in [0, 10^6)) and
without `done: true` is refused: nothing is sent or recorded, and the result
is the error `[error] a reply to a bug (#4) needs eta_min (your estimate, in
minutes from now, of when it will be fixed) or done: true; nothing was
sent`, so the agent can resend it with a number. Replies to questions and
status checks are unaffected. The tool's description and the system prompt
do not change, so the levels share the baseline run's prompt and the rule
shows only in the error. It leaves no bug reply without a structured
estimate or done; estimates written in a reply to a status check are still
read by the regex (*Levels analysis*). In the levels smoke it refused 11 of
63 `reply` calls, in 7 of the 24 sessions.

##### Status checks that name the bugs

A status check lists the bugs sent before it and asks for one structured
reply per bug, to the bug's own id. In L1 the three checks of a session are
ids 3, 6 and 9 (the bugs are 1, 2, 4, 5, 7 and 8); the second reads:

> Status check #6: how long until each of your bug requests so far (#1, #2, #4, #5) is fixed? Reply to each one's own id with eta_min, or with done: true if it is fixed.

`hints.schedule` puts the ids in the check's text from the schedule alone
(every bug sent before the check, whatever the agent did with it), and
`prompts.request_text` renders them, so the message is the same in every
condition and round whatever the agent fixed or said done. This text
replaced `Status check #6: how long until each of your open bug requests is
fixed? Reply to each with an updated eta_min.` for every spec, after the
requests run (2026-10-15), which used the old one: "reply to each" left the
id open, and 58 of that run's 72 text-only estimates were replies to the
check's own id. "Open" also left the agent to decide which bugs to answer
for; now every listed bug is asked about, and the analysis pairs each one
not said done before the check with what came back (*Levels analysis*,
status pairs).

##### Fix times (`fix-times`)

`subagents-bench fix-times SPEC [--force] [--concurrency N] [--only-trial
SUBSTR] [--dry-run]` (`fixtimes.py`), after the run and with no LLM call,
grades every recorded state of every finished trial (harbor's `result.json`
with `finished_at`; `_`/`.` job dirs skipped) that has
`agent/snapshots.jsonl` and, unless `--force`, no `agent/fix_states.json`.

- **Sandbox.** One Modal sandbox per trial, from the task's verifier image:
  `tests/Dockerfile` of the task dirs the sweep ran (the platform-pinned copy
  under `jobs/<sweep>/tasks/`) with that `tests/` dir as context, the image
  harbor's separate verifier uses, built once per task. 3 CPUs, 7000 MB, no
  network (the verifier's `no-network` phase), 3 hours, app
  `subagents-bench-fixtimes`, Modal's 2025.06 image builder (as the sweep).
  Up to `--concurrency` (24) trials at once.
- **States.** Every snapshot with a patch (a changed state), in `seq` order,
  at its `t_start` (when the capture began), then the final state: the
  trial's `artifacts/logs/artifacts/model.patch`, what the verifier graded,
  at the episode's end. Every state is graded, with no bisection, because
  fixes do go backwards (above). The bugs graded are those sent in the
  session (the inbox's bugs ready by its end).
- **The grader, reused.** `replay_states.py` runs in the sandbox on the
  grader's own interpreter (`/opt/swesweep/python/bin/python3`, standard
  library only) and calls the task's grader, `/tests/shared/swesweep_grade.py`:
  its bundle (`load_bundle`) restricted to the sent bugs, and per state
  `run_evaluation`, exactly as the verifier grades a final patch: reset to
  `baseline`, apply the patch, run the visible suite, then each sent bug's
  hidden tests. A reimplementation could drift from the verifier; this one is
  checked against it on every trial's final state (below).
- **Statuses.** Per state and bug: `pass` (resolved); `unknown` when the
  state says nothing about the bug: the evaluation errored, the patch did not
  apply or did not build, the bug's hidden test patch did not apply, its
  targeted run timed out, produced no report or ran no test, or every failing
  targeted test is an `error` outcome (a pytest collection or import error, a
  Go `<build>` failure); `fail` otherwise. A collection error is unknown, not
  a failure: in click every test imports the whole package, so one
  half-finished edit elsewhere would fail all 6 bugs. The grader merges
  `failed` and `error` outcomes, so the replay wraps its `_run_targets` to
  keep each targeted run's outcomes; grading is unchanged.
- **Reruns.** For each bug and each `pass` followed by `fail` (consecutive
  known states), the failing state is graded once more for the bugs that
  need it (`rerun`): a failure that does not repeat is a flaky test, not a
  broken fix.
- **Timeouts.** The grader's own per-run limits (click: 3600 s for the visible
  suite, 900 s per bug) are capped at 120 s and 60 s
  (`SWESWEEP_VISIBLE_SUITE_TIMEOUT`, `SWESWEEP_TEST_TIMEOUT`). A state
  normally grades in seconds (in the verifiers' `eval.json` on Modal, the
  visible suite took at most 2.7 s and one bug's targeted run 1.9 s), and a
  mid-edit state that hangs would otherwise use up the trial's exec budget
  and lose the whole trial on every run; capped, it costs at most 120 s plus
  60 s per bug, and a bug whose run hit the cap is `unknown` in that state.
- **Final state against the verifier.** The host compares the final state's
  per-bug `pass` with `verifier/eval.json`'s `resolved` (a bug missing there
  is unresolved: its build failed) and records `final_matches_verifier` and
  the `mismatches`. A trial whose final state does not reproduce the
  verifier is left out of every timing metric (`timing_ok`, below).
- **Output and resume.** `agent/fix_states.json` (docs/DATA.md), written
  atomically. A trial that fails writes nothing and the next run retries it;
  exit 0 when every trial got its file, 1 otherwise, 2 without Modal, 130 on
  Ctrl-C (the running sandboxes are terminated). `--dry-run` lists the
  trials, their images and states, and touches nothing.
- **Cost.** The levels smoke's 24 trials (76 states) replayed in 40 s, and
  every final state reproduced `verifier/eval.json`.

##### Grading answers, version 2

Version 2 of `answers.json` changes how `grade-answers` judges the replies
to a question:

- **Three verdicts.** `correct` (states what the reference states; other
  wording, extra correct detail and a longer explanation are fine), `wrong`
  (an answer that contradicts the reference or misses its key fact) or
  `none`, not an answer: an acknowledgement, a promise or plan to answer,
  only how to find out, or not knowing. Version 1's two verdicts made "on it"
  and "let me check" wrong answers. The judge grades only what a reply says
  about the question, since a reply can also be about other things.
- **Temperature 0** (`judge_temperature` in the file): version 1 inherited
  the agents' 0.7, so a verdict was a sample.
- **Every reply up to the first correct one.** The replies to the
  question's own id are judged in time order, up to and including the first
  `correct` one (an unread verdict does not stop it), so a question first
  acknowledged and then answered gets the time of its answer
  (`correct_sec`). `answers` keeps the first reply's verdict, as before.
- **Replies elsewhere.** A reply to another id that cites the question
  (`#<id>` as a whole token: not `#31`, `##3` or `a#3`) and was sent after the
  question was delivered is judged the same way and reported apart
  (`elsewhere`); it never answers the question, since one channel is what
  makes the answer time a measurement.
- **Reading the verdict.** A JSON object with `verdict`; version 1's
  `{"correct": true|false}` reads as correct/wrong. Near-JSON whose verdict
  is plain to see (text with a brace after the object, two objects, a stray
  quote in `why`) reads as that verdict when every `"verdict": "<word>"` pair
  names the same one; pairs naming different verdicts (the format echoed
  before the object, a verdict taken back) leave it unread (`verdict` null,
  `why` the start of the reply).
- **Re-grading.** Trials without a version-2 `answers.json` (version 1, or
  unreadable) are graded again, and so are version-2 trials with an unread
  verdict (the judge's reply may be unreadable only this time); the others
  only with `--force`. A trial whose grading fails writes nothing. Exit 0
  when every reply judged got a verdict and no trial failed, 1 otherwise, 3
  on a billing error (it stops at once).

##### Levels analysis (`analysis/fixes.py`, `estimates.py`, `levels.py`)

All times are seconds after the message was sent, like `reply_sec`, unless
named `*_session_sec`.

**Fix times** (`fixes.csv`). From `fix_states.json`, the snapshot records
and the inbox. A state holds from its time to the next one, and is
*observed* once plus once per later successful snapshot that found the
workspace unchanged. Per bug, in state order:

- `unknown` carries no information: it never starts, breaks or ends a
  streak, and neither does a `fail` whose rerun passed (a flaky test). The
  final state is the exception: it reproduces the verifier, which calls
  anything but a pass unresolved, so there an unknown or flaky status is a
  `fail` and a streak never lasts "to the end" through it.
- A *failing stretch* runs from a `fail` through `fail` and unknown states
  to the next `pass`. It is *confirmed* when its fails were observed at
  least twice and it lasted at least 30 s until that pass, or when no pass
  follows (the end cuts it short, and the final state is what the user
  gets). Only a confirmed stretch breaks a passing streak (`n_breaks` counts
  those after the first pass): one half-finished edit or one flaky run must
  not undo a fix.
- `fixed_sec`: the first `pass` from which the bug keeps passing, with no
  confirmed break, for at least 120 s or to the end: a pass that a confirmed
  failure ends sooner is an intermediate state, not a fix. `stable_sec`, the
  strict variant: the start of the passing streak that lasts to the end
  (empty when the bug ends failing). `first_pass_sec`: the first `pass`.
  `visible_ok_at_fix`: whether the visible suite passed in the `fixed_sec`
  state, kept alongside.
- A pass before the bug was sent (the agent can fix a bug it was not told
  about) is flagged (`pass_before_sent`) and the times are clipped to 0;
  `state0_pass` flags a bug passing in snapshot 0, the untouched workspace,
  which should fail every bug.
- `timing_ok`: the trial has `fix_states.json` and its final state
  reproduces the verifier (`final_matches_verifier`); otherwise every time
  is empty and the trial is left out of the timing metrics. `fixed_end` is
  the end-of-session verdict, by the old rule (resolved, visible suite
  passing).

**Estimates** (`estimates.csv`, per (sweep, condition) `eta.csv`).

- *Sources.* Every reply to a bug's id with `eta_min` and without `done:
  true` (`via` `structured`). Estimates written only in text are read by a
  regex, no LLM: a bug reply without `eta_min` (`text_bug`; `require_eta`
  leaves none in the levels sweep), and a reply to a status check's id, once
  per bug it gives a time for (`text_status`: `#4 ~8 min; #5 ~6 min`,
  `#8 and #18 ... ETA ~10 min each`). A time counts when it is approximate
  (`~10 min`, `about 10 min`), follows an ETA word in its sentence, or is
  `in 10 min` after a future (`should be done in 6 min`); within a sentence it
  goes to the ids named since the previous time, and a done word (`#1 is
  fixed`) drops the ids before it. Every table reports the vias apart.
- *Source* (`source`). `arrival`: the bug's first estimate, sent before the
  next status check after the bug; L0 has no checks, so there the window ends
  where L1's next check would be sent (at 6:00, 14:00 or 22:00, by the same
  rule), and L0 and L1 arrival estimates share one window. `check`: sent
  within 120 s after a check that lists the bug. Otherwise `update`.
- *Scoring.* An estimate of m minutes sent at t predicts the fix at t + 60m,
  with a grace of max(60 s, 25% of 60m): 1 minute is the smallest lateness
  worth counting for estimates given in whole minutes, and 25% is about the
  spacing of the values agents give (88 of the requests run's 106 estimates
  were 5, 6, 8, 10, 12, 15 or 20 minutes, 20-33% apart). The strict version
  and the share fixed within +1, +2 and +5 minutes show the grace's effect.
- *Eligibility and censoring.* An estimate is `eligible` iff its prediction
  plus grace falls within the session (start + `wall_clock_sec`): censoring
  depends on the prediction only, never on the outcome. Ineligible estimates
  are reported only as the share whose bug was fixed by the deadline. An
  eligible estimate sent before the bug's `fixed_sec` is `scored`: `on_time`
  iff fixed by prediction + grace, `err_min` = (fix - prediction) / 60, and
  `log2_err` = log2(actual / predicted horizon) with the requests analysis's
  30 s floor on both. A scored estimate whose bug is never fixed is late
  whatever the exit: `log2_err` +inf and `err_min` a lower bound (to the
  later of the episode's end and the deadline), so medians stay valid.
- *Stale.* An estimate sent at or after the bug's fix is `stale`, not
  scored, and reported as the stale rate.
- *Against the done claim* (the bug's first reply with `done: true`, what the
  user is told): the same rules for eligible estimates sent before the claim,
  a bug never said done being late.
- *Weights.* Each bug counts once: its scored estimates share a weight of 1
  (`weight`, `claim_weight`), recomputed within each subset a table reports.

**Status pairs** (`status_pairs.csv`). One row per status check and bug it
lists that was not said done before the check (the user's view: a reply to
the bug with `done: true`, or a done written in a reply to a check's id,
`#1 is fixed and verified`). Its answer is the first estimate (any via) or
done for the bug after the check was sent; `answered` iff within 120 s. The
window is fixed rather than "before the next message" because in L2 and L3 a
question follows each check after 40 s. `right`: a done for a bug fixed by
then, an estimate for one not fixed (by `fixed_sec`); `done_owed`: fixed by
the check and never said done in the session.

**Question timing** (columns of `messages.csv`'s question rows, from
`answers.json` version 2). `correct_sec`: until the first reply judged
correct; `correct_known`: every reply up to it (every reply, when none is
correct) has a verdict (an unread verdict, or a reply not graded yet, leaves
it unknown). It splits into `delivery_sec` (waiting for the agent's step to
end) and `response_sec` (from delivery). Alongside: the first reply's verdict (`first_verdict`,
`first_reply_correct`: stricter, since the prompt says the first reply is
the answer), `wrong_before_correct`, and the replies elsewhere, never
counted. `levels.csv` reports the share of all questions asked with
`correct_sec` within 30, 60, 120 and 300 s; a question never answered
correctly, also one still open when the session ended idle, counts as never.
Every question is sent at least 440 s before the deadline (the last at
22:40), so no threshold needs censoring; the shares are NaN while any
question's `correct_known` is false. An older `answers.json` judged only the
first reply: a correct one gives `correct_sec`, and an incorrect one with
later replies leaves the question unknown.

**Urgency** (`urgency.csv`). One row per urgent bug of a valid L3 session
and twin level: L2, the comparison, and L0, descriptive. Twins are the same
subtask sent within 1 s in the valid sessions of the same sweep, task and
setting; a session without fix times (`timing_ok` false) pairs with nothing.
Against its twins:

- *Order*, the direct answer to "does the urgent bug go first": per urgent
  and twin session pair, over the earlier bugs failing at the urgent bug's
  send time in both (their last known status then; `unknown` counts as
  failing unless the bug's `fixed_sec` fix came by then), 1 if the urgent
  bug was fixed before the earlier bug, 0 after, 0.5 if neither was fixed or
  both at once. An earlier
  bug counts by its first fix after the send time, not by a fix that broke
  before then. The mean over those bugs, then over pairs; `twin_order` is the
  same score in the twin.
- *Time to fix*: fixed within 240 s (before the next bug) and 480 s, by
  `fixed_sec`. "Fixed by the end" alone would miss the effect: click's bug 3
  was fixed in 4 of the 5 requests-run sessions that sent it at 12:00.
- *Win rate* over the urgent x twin session pairs: 1 if the urgent bug was
  fixed more than 30 s sooner or only it was fixed, 0 for the reverse, 0.5
  within 30 s or if neither was; `n_decisive` counts the pairs that are not
  0.5.
- *Start*: the earlier of a dispatch naming the bug's files or hidden tests
  (`dispatch_sec`, as in `hints.csv`) and the first snapshot after it was
  sent whose diff of the bug's target files differs from the last snapshot
  before then (`change_sec`, at that snapshot's end). Not the first read: in the
  requests run, both the urgent bug and its control twin had their file read
  within seconds.
- *Preemption*: of the tool calls of any agent starting in the 240 s after
  delivery that touch the bug's files or an earlier bug's (as `touch_sec`
  and `edit_sec` match them), the share touching the bug's (a call touching
  both counts as the bug's), and the same for edits. This measures the single
  agent's and the hinted orchestrator's own work: the manager has no edit
  tools and cannot stop a subagent, so it can only dispatch sooner.
- *Cost*: the session's other bugs' fixed share and censored median
  `fixed_sec` (a bug never fixed counts as infinitely late), against the same
  bugs in the twins.

**Main numbers** (`levels.csv`, chosen before the run). One row per (sweep,
setting, level) of valid sessions:

- ETA, `eta_on_time`: the on-time share of the structured estimates scored
  against the fix times, each bug weighted once (per session, the sum of its
  bugs' on-time shares and the number of bugs, then a ratio of sums); L1 is
  the headline. With the text estimates too, `eta_on_time_any_via`; arrival
  estimates only (L0 against L1), `eta_on_time_arrival`.
- Questions, `q_correct_60s`: the share of all questions asked answered
  correctly within 60 s; L2 is the headline.
- Urgency, `urgent_win_rate` (L3 rows): the win rate over every urgent x L2
  twin session pair of the sweep and setting.

Each has a 95% percentile interval from a cluster bootstrap
(`stats.cluster_bootstrap`, 2000 resamples, seed 0) that resamples whole
sessions within each task, as many as the task has: answers and bugs cluster
by session, so resampling rows would understate the noise, and stratifying
keeps every resample's mix of click and echo. For the win rate, L3 and L2
sessions are resampled within each level and task. A resample whose
statistic is NaN is left out; with no stratum of two sessions there is no
interval. Everything else in the row is descriptive (bugs fixed by
`fixed_sec` and at the end, bugs estimated, status pairs answered and
right, the other question thresholds, order and the shares fixed within
240/480 s).

**(sweep, condition).** Several jobs dirs are one data set, but the same
condition run in two sweeps (the in-sweep L0 and the earlier baseline run)
must never pool: `conditions.csv`, `interactivity.csv`,
`hint_positions.csv`, `requests.csv` and `eta.csv` group by (sweep,
condition) and start with a `sweep` column, `levels.csv` groups by (sweep,
setting, level), and twins come from the same sweep. The printout shows the
sweep when there are several; the delegation gate names a condition once.

**Departures from the plan.** The analysis follows PR #13's plan comment
except:

- L0 is the baseline spec plus snapshots and `require_eta`, not the spec
  unchanged, so the earlier baseline run is a looser reference, not a
  replicate of L0 (above).
- `grade-answers` stops at a question's first correct reply; the plan judged
  every reply. No planned number reads the later ones.
- `start_sec` is measured from when the bug was sent, like every other time
  in the tables; the plan measured it from delivery (`delivery_sec` is the
  difference).
- The first-reply share is reported at 60 s only (`q_first_correct_60s`),
  and there is no step-curve figure yet: `levels.csv` has the four points of
  `correct_sec`.
- The plan asked a failure to repeat on a rerun and last 2 snapshots and
  30 s. Only the first failing state after a pass is rerun (a later fail of
  the same stretch is not); the 2 are observations (a state plus the
  unchanged snapshots after it); and a stretch that reaches the end of the
  session is confirmed however short, since the final state is what the
  user gets.
- A pass in snapshot 0 is flagged (`state0_pass`), not a reason to leave the
  trial out.

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
arbitrations, inbox, replies, snapshots, transcript) plus `episode.json`, written as events happen by
a thread-safe recorder. Failed and abandoned LLM calls are recorded too.
Tool listeners see each `tools` record right after it is written, in the
agent's thread; what a listener raises is swallowed, so it never changes the
tool call (the workspace snapshots are triggered this way).
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
and `--rounds`/`--only` never change an order. `--job ROUND:COND`
(repeatable) adds single jobs after the `--rounds`/`--only` selection, in the
order given and each once; alone it selects only those jobs. Its rounds and
conditions are checked like `--rounds`/`--only` (exit 2), and the shuffle
draws the same with or without it.

**One sweep per jobs dir.** A sweep holds an exclusive `flock` on
`<jobs_dir>/.sweep.lock` from before it writes anything (configs, the task
copies, `sweep.json`, moved-aside jobs) until it returns; the file names its
pid and command line. A second sweep on the same jobs dir exits 2 at once
with `another sweep is running on <jobs_dir> (pid N: ...)`: it would see
the first's running jobs as unfinished and move them aside, and rewrite
the task copies under its harbor runs. `preflight --oracle` holds the same
lock (its oracle job is moved aside and the tasks staged there too), so
`run_experiment.sh`'s preflight and sweep, two processes one after the
other, take it in turn. The kernel drops the lock when the process ends,
however it ends, so a crash or `kill -9` leaves no stale lock to clean up.
harbor does not inherit it: after a `kill -9` of the sweep (no chance to stop
harbor), its harbor runs go on unguarded until they end. A sweep started by
code from before the lock holds none, so it cannot be detected: a new sweep
starts as if the jobs dir were idle and moves its running jobs aside. A dry
run only reads: it neither takes nor checks the lock, so while a sweep runs,
its running jobs show as unfinished (`move aside, rerun`).

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

**Requests.** `analysis/requests.py` turns each message of an episode with
`interactive.replies` (bugs, questions, status checks) into a row of what the
user got back: the first reply, a question's verdict from `answers.json`, a
bug's first estimate, done claim and grader verdict, and for the urgent bug
the order of done claims (definitions under *Interactive sessions*,
*Requests*). There is no figure for it yet.

**Levels.** `analysis/fixes.py`, `estimates.py` and `levels.py` add the
replayed fix times (`fixes.csv`), every estimate scored against them and
against done claims (`estimates.csv`, per (sweep, condition) `eta.csv`),
the status-check pairs (`status_pairs.csv`), the question timing (columns of
`messages.csv`), the urgent bugs against their twins (`urgency.csv`) and
the main numbers per (sweep, setting, level) (`levels.csv`). Without what a
table reads (replies, `fix_states.json`, levels conditions) its times are
empty or it has no rows. Definitions under *Interactive sessions*,
*Requests*, *Levels analysis*.

**Statistics.** Resolve rates with Wilson 95% intervals; mean cost, wall
clock and tokens with a seeded percentile bootstrap (2000 resamples); the
levels' main numbers with a seeded cluster bootstrap that resamples whole
sessions within each task (`stats.cluster_bootstrap`).

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
