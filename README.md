# subagents-bench

Measuring the **cost and latency** of a frontier **orchestrator** directing
smaller **subagents** on executable coding tasks — the configuration
practitioners actually deploy, but one existing scaling studies don't primarily
measure.

Prior work (Kim et al. 2025, *Towards a Science of Scaling Agent Systems*)
assigns the **same** model to every agent, reports **accuracy** rather than
latency, and includes no repository-level coding with executable tests. This
project instead varies **subagent size** independently of the orchestrator, runs
on **Terminal-Bench 2.0** and **ExploitBench** (tasks with real, runnable
scoring), and reports resolve rate against **tokens** and **wall-clock time** as
Pareto curves.

This version runs Terminal-Bench 2.0 and [SWE-sweep](#swe-sweep); the
ExploitBench adapter is not part of it.

## The experiment

One episode is one Terminal-Bench 2.0 task, run by harbor 0.24.0 in the task's
own Docker image:

1. The **orchestrator**, `deepseek-ai/DeepSeek-V4.1-Flash` on Sail's `balanced`
   tier, gets the task instruction and **5 steps**. A step is one LLM call
   plus the tools it asks for: `bash`, `edit_file`, `read_file`, `list_dir`,
   `search`, `finish`.
2. In a subagent condition it can also call `spawn_subagents(instructions)`.
   **Subagents** (`nvidia/Gemma-4-31B-IT-NVFP4` on Sail, `balanced`) then work
   in the same container, each from a fresh context, for at most 8 steps, and
   report back. The **strategy** says how they run, the **scope** which tools
   they get.
3. harbor's verifier tests the container's final state (reward 1 = resolved).
   Every LLM call, tool call, step, spawn, dispatch and arbitration is logged
   with raw tokens and timestamps, so cost and latency are computed offline.

| Axis | Values |
|---|---|
| Strategy | `parallel` (up to 3 subagents at once), `sequential` (one at a time, in order), `arbiter` (3 candidates per instruction, an LLM judge, one confidence-aware debate round) |
| Scope | `easy`: `read_file`, `list_dir`, `search`; `medium`: + `bash` (told not to edit project files); `hard`: + `edit_file` |
| Orchestrator tools | all five, or `read_file`, `list_dir`, `search` only (a **manager**, which must delegate every command and edit) |

### Conditions

All 15 are in [`configs/sweeps/pilot.yaml`](configs/sweeps/pilot.yaml), each a
set of overrides on [`configs/sail/base.yaml`](configs/sail/base.yaml).

| Condition | What it is |
|---|---|
| `single_5step` | the orchestrator alone, 5 steps |
| `single_matched` | alone, 40 steps (5 orchestrator steps x 8 subagent steps) |
| `gemma31b_{parallel,sequential,arbiter}_{easy,medium,hard}` | Gemma-4-31B subagents, strategy x scope (9 conditions) |
| `samemodel_parallel_hard` | subagents run the orchestrator's own model: having subagents vs their size |
| `manager_{parallel,sequential,arbiter}_hard` | Gemma subagents, orchestrator limited to read-only tools |

The **pilot** is these 15 conditions on 10 tasks (4 easy, 6 short medium)
x 3 rounds = 45 harbor jobs, 450 episodes. The **smoke** sweep
([`configs/sweeps/smoke.yaml`](configs/sweeps/smoke.yaml)) is the same 15
conditions on `fix-git` and `prove-plus-comm`, 1 round (30 episodes). Each
round runs the conditions in its own shuffled order.

## Setup

You need macOS or Linux, [Docker](https://docs.docker.com/get-docker/) running
(4 task containers at once: give it about 6 CPUs and 12 GB), [uv](https://docs.astral.sh/uv/),
network access and a Sail API key.

```bash
scripts/setup.sh        # .venv from uv.lock (harbor 0.24.0 included), Docker check,
                        # Terminal-Bench 2.0 download, creates .env from .env.example
```

Put `SAIL_API_KEY=...` in `.env` and run it again; it ends with `Setup OK`.
`.env` never overrides a variable already set in your shell, and no script
prints a key. The task images are `linux/amd64`: on Apple Silicon they run
emulated, which slows tool calls (tool time is measured separately).

## The three scripts

```bash
scripts/run_experiment.sh configs/sweeps/smoke.yaml   # preflight, then the smoke sweep
scripts/run_experiment.sh                             # the pilot (configs/sweeps/pilot.yaml)
scripts/analyze.sh jobs/pilot                         # tables, figures, delegation gate
```

`run_experiment.sh [SPEC] [--dry-run] [--only COND...] [--rounds 1,2]
[--job ROUND:COND]...` runs `subagents-bench preflight SPEC --oracle` (Docker,
harbor, the key, one request per model, harbor's oracle agent on every task),
then `subagents-bench sweep SPEC`: one `harbor run` per (round, condition).
`--rounds` and `--only` narrow that to some rounds and conditions; `--job 1:single`
(repeatable) adds that one job after them, and without `--rounds` or `--only`
runs only the jobs it names, so new rounds and the reruns of earlier jobs are
one sweep. It prints one line per trial as harbor finishes it. `--dry-run`
prints the jobs and their harbor commands and runs nothing. On macOS the sweep
runs under `caffeinate -i` (turn off with `SB_NO_CAFFEINATE=1`). Exit codes:
0 done, 1 jobs failed or need a retry, 2 spec or config error or another sweep
running on the jobs dir, 3 billing error, 4 provider outage, 130 interrupted.

The sweep is resumable: run the same command again after any stop. Complete
jobs are skipped; an unfinished job, or one with a trial that did not measure
its condition (provider failure, billing error, harbor or agent crash before
the verifier, or on SWE-sweep a verifier that could not grade), is moved to
`jobs/<sweep>/_partial/` and rerun. An agent timeout is an outcome, not a
retry. On a provider failure the sweep sends one tiny request to every
model and stops with exit 4 if one still fails.
Ctrl-C gives harbor up to two minutes to remove its containers.

One sweep runs on a jobs dir at a time. A second one would take the first's
running jobs for unfinished ones and move them aside, so while a sweep (or
preflight's oracle job) holds `jobs/<sweep>/.sweep.lock`, another exits 2 at
once, before it writes anything: `another sweep is running on jobs/<sweep>
(pid N: <its command>)`. To add jobs, let the running sweep end (or stop it)
and start one sweep with every job, using `--job` for the odd ones. A sweep
started by code from before `.sweep.lock` holds no lock, so nothing can tell
it is running: a new sweep is not refused and moves its running jobs aside.
Let such a sweep end first (`pgrep -fl 'subagents-bench sweep'` lists the
sweeps on this machine). `--dry-run` neither takes nor checks the lock: while
a sweep runs, its running jobs show as `move aside, rerun`.

`analyze.sh [JOBS_DIR...]` aggregates the jobs (several dirs are one data
set), draws the figures, and exits 3 if a subagent condition never delegated
in any valid episode (tables and figures are written anyway).

The smoke sweep of 2026-10-12 took about an hour and $0.14. Expect the pilot
to take most of a day (roughly 6-10 h, mostly the manager jobs) and a few
dollars.

Other commands: `.venv/bin/subagents-bench validate CONFIG...` prints a
resolved config, and one harbor trial by hand is

```bash
SUBAGENTS_BENCH_CONFIG=configs/sail/base.yaml .venv/bin/harbor run -d terminal-bench@2.0 \
  -i fix-git -a subagents_bench.harness.harbor_agent:OrchestratorAgent -e docker -o jobs/manual -y
```

## SWE-sweep

[SWE-sweep](https://github.com/facebookresearch/swe-sweep) (Lieret et al.
2026) hands the agent a whole repository with no issue text: find and fix as
many bugs as you can without breaking a passing test. A separate verifier
container runs hidden tests per bug; the reward is the fraction of the task's
bugs fixed, 0 if a visible test regressed. It is not on the harbor registry:
clone it next to this repository (`git clone
https://github.com/facebookresearch/swe-sweep ../class-6-swe-sweep/swe-sweep`,
or set `dataset_path` in the spec to your clone's `tasks/`) and run

```bash
scripts/run_experiment.sh configs/sweeps/swe_sweep.yaml
```

[`configs/sweeps/swe_sweep.yaml`](configs/sweeps/swe_sweep.yaml) is the
pilot's 15 conditions on 10 small repositories (Go, Python, JavaScript, C;
6-14 bugs each, 106 in all) x 3 rounds. It copies the 10 tasks to
`jobs/swe_sweep/tasks/` with every Dockerfile `FROM` pinned to `linux/amd64`
(the verifier image ships an x86-64 Python; `jobs/swe_sweep/tasks.json`
records the clone's commit, a hash per task and its bugs) and passes that
copy to harbor with `-p`, caps each container at 3 CPUs and 7 GB (the tasks
ask for 16 and 64 GB), gives commands 600 s (test suites run emulated) and
the agent one hour. The orchestrator's step budget is 5, as in the pilot
(the spec's `overrides.budget.orchestrator_steps`), and `single_matched`
keeps the 40 (5 x 8 subagent steps) that `pilot.yaml` gives it: a
condition's own setting beats the overrides. A larger budget therefore takes
two edits, `orchestrator_steps` and `single_matched`'s matched budget under
`conditions:`, both shown in the spec's comments. The analysis adds the
bugs fixed (the grader's count, 0 on a regression) and regressions per
episode and the micro-averaged `bug_fix_rate` per condition, SWE-sweep's own
metric; `resolved` keeps meaning every bug of the task fixed. Expect 5-20
minutes and about ten times the tokens of a Terminal-Bench episode, and an
image build of a few minutes per task the first time (1-2.5 GB each).

The first agent run
([`configs/sweeps/swe_hint.yaml`](configs/sweeps/swe_hint.yaml),
2026-10-13: `gemma31b_parallel_hard` with the neutral and the hinted prompt
on goawk, click and svgo x 2 rounds, 12 of 12 trials valid, $0.45) is the
opposite of Terminal-Bench: the orchestrator delegated in every episode, 6
of 6 with each prompt, but no agent changed a source file and no bug was
fixed (0 of 152). The 82 Gemma subagents spent about two thirds of their
8 steps reading code (one tool call per step, 80 used all 8), and the
orchestrators delegated at step 3-5, with 0-2 steps left. There is no
single-agent baseline on these repositories yet
([issue #7](https://github.com/PatrickC05/subagents-bench/issues/7)).

### Interactivity: a new bug every 5 minutes

[`configs/sweeps/swe_interactive.yaml`](configs/sweeps/swe_interactive.yaml)
turns an episode into a 60-minute session. The first user message is the task
plus one bug as a hint; every 5 minutes the user sends another bug of the same
repository ("Another one, while you keep working on the earlier ones"), up to
12. The session is about the bugs the user sends (SWE-sweep's "fix every bug"
is set aside, `interactive.scope: reported`), each round sends them in its own
order, there is no step budget and no `finish`, and the agent can `wait` for
the next message. The question is how long each new request waits before work
on it starts, and whether the earlier ones keep going:

| Condition | Where a new request waits |
|---|---|
| `single_interactive` | the orchestrator alone: behind its own running commands |
| `blocking_interactive` | subagents, but `spawn_subagents` blocks until they all report |
| `background_interactive` | subagents run in the background and report as messages (`subagent.spawn_mode: background`) |
| `background_hinted` | the same, with the prompt saying that delegating keeps it free to respond |

Hints name the files the bug's reference fix changes and its hidden tests' ids
(`interactive.hint_detail: tests`); `report` uses bug reports written once per
subtask by `subagents-bench hints configs/sweeps/swe_interactive.yaml`. The
analysis adds `hints.csv` (per hint: delivery, response, dispatch, first touch
and first edit of its files, in seconds after it was sent, and whether the
grader counts it fixed; and whether it was not started, continued or stopped
when the next one arrived), `interactivity.csv` (per condition: censored
medians, the share started before the next hint, and continued/stopped/not
started at the next hint) and `hint_positions.csv`, plus the
`interactivity` figure. Design and caveats (shared files, path matching):
docs/DESIGN.md, *Interactive sessions*; columns: docs/DATA.md. 4 conditions x
4 repositories x 2 rounds = 32 one-hour episodes, about 8 hours at 4 at once.

```bash
scripts/run_experiment.sh configs/sweeps/swe_interactive.yaml
```

Two 15-minute smokes (echo and click, a bug every 3 minutes, 5 bugs, 1 round;
2026-10-14) came first:
- [`swe_interactive_smoke.yaml`](configs/sweeps/swe_interactive_smoke.yaml),
  the 4 conditions above: 8 of 8 trials valid, $0.22. The orchestrator fixed
  the bugs itself: 1 of the 6 sessions with subagents delegated, so the
  conditions mostly measured the same single agent.
- [`swe_interactive_managers_smoke.yaml`](configs/sweeps/swe_interactive_managers_smoke.yaml)
  adds managers (read-only orchestrators, so every edit and test goes through
  a subagent), blocking and background: 8 of 8 valid, $0.42. Both delegated
  in every session and background beat blocking, but the single agent was
  the most responsive (median 9 s to receive a new bug, against 25 s and
  53 s) and fixed the most (5 of 10 bugs sent, against 3 and 3). Its own
  commands never ran longer than 10 s, so it was never stuck behind them,
  while managing subagents made the orchestrator's own calls longer.

Next, two experiments look for where background subagents should pay off, the
single agent against a background manager in each:
- [`swe_interactive_slow.yaml`](configs/sweeps/swe_interactive_slow.yaml)
  (**slow commands**): both must test each fix before moving on
  (`interactive.session_note`), on pint and isort, whose suites run for
  minutes under emulation; 30-minute sessions, 10 bugs, 2 rounds (8
  sessions). Read `blocked_sec` and `delivery_sec` (hints.csv) against
  `orch_bash_median_sec` (episodes.csv).
- [`swe_interactive_rate.yaml`](configs/sweeps/swe_interactive_rate.yaml)
  (**faster arrivals**): the same 10 bugs every 60, 180 or 300 s on click and
  pint, each session 10 minutes past its last bug, 2 rounds (24 sessions).
  Read `edit_median_sec` by `hint_index` (hint_positions.csv) and bugs fixed.

First run of both (2026-10-14/15, one Mac, 2 sessions at a time; all 32
sessions valid, $4.56; tables and notes in PR #10's results comment):
- New bugs waited behind the orchestrator's own LLM calls, not its commands:
  86-98% of the bugs that arrived while it was busy arrived during a model
  call. The single agent's commands ran 0.3 s at the median and 25 s at most,
  even with the test-each-fix rule, so experiment 1 could not test slow
  commands (pint's and isort's suites run in 16-25 s).
- The single agent was more responsive (a new bug received after a median
  19-21 s, against 36-41 s for the background manager) and fixed as many bugs
  for a third of the cost: 36 of 156 at $0.08 a session, against 31 of 156 at
  $0.22 (experiment 2; in experiment 1 both fixed 1 of 48).
- Faster arrivals hurt the manager more: the single agent fixed 25% of the
  bugs sent at 60, 180 and 300 s alike, the manager 11%, 36% and 28%.
- The manager's subagents broke a passing test, which scores a session 0, in
  5 of its 16 sessions; the single agent in none. The single agent probed git
  history in 19 commands (nothing to find: the images' git holds only
  `baseline`).
- Sail's tool-call parsing for Gemma-4-31B fails on a reply with more than one
  tool call; the client now retries such a reply asking for one call
  (`one_call_retry`, docs/DESIGN.md).

The analysis also records what each hint waited behind (`blocked_by`,
`blocked_sec`), the orchestrator's own command and call times, and flags
commands that may break SWE-sweep's fairplay rules (git history, network,
installed copies: `fairplay` in tools.csv, `[WARN]` in the output); the first
smoke's stuck single agent ran `git cat-file --batch-all-objects`.

### Requests: answers, fixes and estimates

The runs above sent only bugs, and read a bug's latency off the agent's tool
calls (`touch_sec`, `edit_sec`): an agent that keeps working on the bugs it
knows looks responsive without ever telling the user anything, and nothing
asks it to stay available for requests it cannot see yet. A user wants three
things from a session: their questions answered, their bugs fixed, and to
know when each fix will be ready.
[`swe_interactive_requests.yaml`](configs/sweeps/swe_interactive_requests.yaml)
sends all three kinds of message in one 30-minute session
(`interactive.replies`):
- a bug every 4 minutes, 6 in all, with the hints above (`hint_detail:
  tests`); bug 3 (0-based, at 12:00) is urgent: "fix it before the earlier
  ones" (`urgent_every: 3`);
- two questions about the code between consecutive bugs, 80 s and 160 s after
  each (`questions_per_interval: 2`; 12 in all), such as what a function
  returns in a given case or a default value, written beforehand from files
  no subtask's fix or tests change;
- a status check 120 s after bugs 1, 3 and 5 asking how long each open bug
  will take (`status_every: 2`).

Every message has an id (`Request #4`, `Question #5`, `Status check #6`) and
the agent answers with a new tool, `reply(to, text, eta_min, done)`; the user
sees nothing else it writes. The prompt asks it to answer each question (its
first reply to a question is taken as the answer, and that reply's result says
so), to reply to a bug with an
estimate in minutes (`eta_min`) in its first response after the bug arrives
and again whenever the estimate changes, with `done: true` once it is fixed
and checked, and to fix the urgent bug before the earlier ones. So a question
is answered when its reply arrives, and a bug's replies say when the agent
expects it fixed and when it says it is. Three conditions get the requests,
the same prompt in each: `single_requests`, `manager_background_requests` (a
read-only orchestrator with background subagents) and
`background_hinted_requests` (all tools, background subagents, the delegation
hint). Two controls, `single_bugsonly` and `manager_background_bugsonly`, get
the same bugs at the same times and nothing else (no ids, no `reply`, no
questions, status checks or urgency): the in-sweep baseline for what the
requests cost in bugs fixed and in how fast a new bug is picked up
(`hints.csv`, `interactivity.csv`), on the same day and provider. The
arrival-rate runs above are a looser second reference. Click and pint, 2
rounds: 5 conditions, 20 sessions.

| The user's question | `requests.csv` columns (per condition) |
|---|---|
| Are my questions answered, and how fast? | `answered_frac`, `answer_median_sec`, `answered_60s`, `answered_180s` |
| Are the answers right? | `correct_of_answered`, `correct_of_asked` (an LLM judge against a reference answer, first reply only; unanswered questions count only in the second) |
| Are my bugs fixed? | `fixed_frac` |
| Are the estimates right? | `eta_given_frac`, `eta_abs_log2_median` (1: off by a factor of 2), `eta_log2_median` (> 0: late; a bug estimated and never said done counts as infinitely late unless the session's deadline came before the predicted time), `eta_total_abs_log2_median` (the same prediction scored from the bug's arrival, against the user's whole wait: a late estimate has a short horizon), `n_eta_never_done`, `on_time_frac` |
| Is a bug done when the agent says so? | `claim_precision` (done claims the grader confirms) |
| Does an urgent bug go first? | `urgent_first_frac` |
| Is a status check answered? | `status_reply_median_sec` |

`messages.csv` has one row per message. Questions are written once, before
the run, and answers are graded after it; rerun `analyze.sh` after grading
to fill `correct_of_answered` and `correct_of_asked`:

```bash
.venv/bin/subagents-bench questions configs/sweeps/swe_interactive_requests.yaml      # 12 per task, jobs/<sweep>/hints/
scripts/run_experiment.sh configs/sweeps/swe_interactive_requests.yaml
.venv/bin/subagents-bench grade-answers configs/sweeps/swe_interactive_requests.yaml  # <trial>/agent/answers.json
scripts/analyze.sh jobs/swe_interactive_requests
```

Design and caveats (question generation, the judge, the ETA metrics,
censoring): docs/DESIGN.md, *Requests*; columns: docs/DATA.md.

A monitor page follows the sweep while it runs and after (it only reads the
jobs dir, so rerun it for a new snapshot):
`.venv/bin/python -m subagents_bench.monitor jobs/swe_interactive_requests
--compare jobs/swe_interactive_rate --out OUT_DIR` writes `OUT_DIR/index.html`
and `OUT_DIR/transcripts/` (serve or publish the folder). It shows the
per-condition table above, each control against its twin, the same bugs
against the arrival-rate run (per condition, on the repository and round pairs
it has graded, and bug by bug), and one card per session (live, grading or
graded) with its timeline of messages, replies, estimates and subagent runs,
its messages table, and its full transcript on request. The design comes from
the jobs dir's `sweep.json` (`--spec` instead), so it works on any interactive
sweep; files: docs/DATA.md, *Monitor page*.

First run (2026-10-15, one Mac; all 20 sessions valid, $2.47; 4 sessions per
condition, so these are directions, not effects; jobs 7-10, round 2's hinted,
manager and both controls, ran two jobs at a time, `jobs_at_once` in the
manifest):

| Per condition (4 sessions) | Single agent | Background manager | All tools + background, hinted |
|---|---|---|---|
| Questions answered | 27 of 48 (56%) | 39 of 48 (81%) | 39 of 48 (81%) |
| Answer time, median (unanswered = never) | 293 s | 55 s | 61 s |
| Answered within 60 s | 27% | 52% | 50% |
| Answers correct, of those answered | 96% | 97% | 100% |
| Status check answered, median | 142 s | 42 s | 42 s |
| Requested bugs fixed | 9 of 24 | 9 of 24 | 11 of 24 |
| Same bugs, bugs-only control | 10 of 24 | 8 of 24 | – |
| Bugs given an estimate | 21% | 29% | 38% |
| Estimated bugs done on time | 0 of 5 | 4 of 6 | 1 of 9 |
| Done claims the grader confirms | 2 of 3 | 6 of 6 | 10 of 13 |
| Urgent bug done before the earlier open ones | 0% | 22% | 50% |
| Cost per session | $0.07 | $0.22 | $0.10 |

- The single agent was the least available: it left 44% of the questions
  unanswered and took a median 293 s to answer, five times the subagent
  conditions, and answered status checks three times slower. It was not
  stuck behind its own commands: the questions reached it a median 13 s
  after they were sent (it was in a model call for 24 of the 25 questions it
  left unanswered or answered after 180 s; LLM calls took 82-92% of each
  session, its own tools 7-18%). It then kept working on its bugs: for 17 of
  those 25 its reasoning says it will answer ("Let me answer #9 quickly"),
  often with the answer worked out, and it ran 10-57 more steps without
  calling `reply`. In one session its last reply came at 2:47 of 30:00.
  The earlier bugs-only runs could not show this: it picked up new bugs as
  fast as the others (a median 10-12 s to receive one here too).
- The requests cost few fixes in total: 9 against 10 bugs fixed for the
  single agent, 9 against 8 for the manager, on the same bugs at the same
  times. Per repository and round the differences are larger (r2 pint,
  single: 1 against 3), so this is noise-level.
- With questions to answer, the all-tools orchestrator delegated in 3 of 4
  sessions, each a single burst (one `spawn_subagents` call of 1-3
  subagents); in the first bugs-only smoke 1 of 6 all-tools sessions did. It
  answered as fast as the manager, fixed the most and claimed done most
  often, at half the manager's cost.
- The agents mostly did not estimate, although the prompt asks for an
  estimate in the first response after a bug arrives (21-38% of bugs; median
  estimate 8-9 minutes). Only the manager's estimates were usable: off by a
  factor of about 3 at the median, mostly early (4 of the 5 bugs it said done
  beat the estimate). The single agent never said done any bug it estimated;
  the hinted orchestrator 4 of 9.
- Fairplay: 11 sessions probed git history (the images hold only the
  `baseline` commit, so there was nothing to find); two bugs-only sessions
  looked for an installed copy and one also tried `pip download` (no network,
  it failed). No command obtained a fix (`fairplay` in tools.csv).

### Baseline, and running on Modal

[`swe_interactive_baseline.yaml`](configs/sweeps/swe_interactive_baseline.yaml)
is the bugs-only baseline: 30-minute sessions, 6 bugs every 4 minutes in one
fixed order for every round and condition, the reply tool on, every model
DeepSeek-V4.1-Flash, for the orchestrator alone, the orchestrator with
background subagents (hinted), and a read-only orchestrator whose work all
goes to subagents, on click and echo.

`harbor.environment: modal` in a spec runs the task containers (the agent's
sandbox and the separate verifier) in Modal sandboxes, built from the tasks'
Dockerfiles on x86-64 machines, instead of the local Docker: no emulation, and
sessions don't share one machine, so more run at once. The agent loop stays
here and calls Sail. It needs the `modal` extra (`scripts/setup.sh` installs
it) and a Modal token (`modal token new`); the sweep sets Modal's 2025.06 image
builder, since the legacy one fails on images without pip (echo's).
[`swe_interactive_baseline_modal.yaml`](configs/sweeps/swe_interactive_baseline_modal.yaml)
is the baseline on Modal, and `swe_interactive_baseline_modal_smoke.yaml` a
2-minute version of it for a quick check:

```bash
scripts/run_experiment.sh configs/sweeps/swe_interactive_baseline_modal_smoke.yaml --jobs-at-once 3
scripts/run_experiment.sh configs/sweeps/swe_interactive_baseline_modal.yaml --jobs-at-once 3
```

## Where data and figures land

```
jobs/<sweep>/                       the sweep (git-ignored, like analysis/ and figures/)
  configs/<condition>.yaml          resolved config of each condition
  sweep.json, sweep_manifest.jsonl  status and one line per job event
  .sweep.lock                       locked while a sweep runs: its pid and command line
  <sweep>__r<k>__<condition>/       one harbor job
    <task>__<id>/result.json        harbor's trial result (reward, exceptions, timings)
    <task>__<id>/agent/             this harness's records: episode.json, calls.jsonl,
                                    tools.jsonl, steps.jsonl, spawns.jsonl,
                                    dispatches.jsonl, arbitrations.jsonl, transcript.jsonl
  _partial/, _oracle/               moved-aside job runs, preflight's oracle job
analysis/<sweep>/                   episodes, calls, tools, dispatches, arbitrations and
                                    conditions CSVs, summary.json
figures/<sweep>/                    8 PNG figures and index.md with their captions
```

Every record field and table column is described in
[`docs/DATA.md`](docs/DATA.md); the design decisions, and the measurements
behind them, in [`docs/DESIGN.md`](docs/DESIGN.md).

## Findings so far

Four smoke sweeps (2 easy tasks x 1 round each, so no resolve-rate claims yet).
The first two ran on the earlier, larger harness; the third and fourth
(2026-10-12) on this code. The third ran all 15 conditions: 30 of 30 trials
valid, 21 solved, no reruns, $0.11, about 45 min. The fourth (`smoke_hinted`)
ran the 10 all-tools subagent conditions with the delegation hint: 20 of 20
valid, 13 solved, no reruns, $0.019, about 15 min.

- **An orchestrator with every tool does not delegate.** It spawned subagents
  in 0 of 18 Gemma-subagent episodes (and 0 of 2 same-model ones) in the first
  sweep, and in 0 of 20 in each of the next two. It solved the tasks itself,
  so those conditions measured only the orchestrator: 12 of 20 solved in the
  second and third sweeps, about 30 s and $0.0008 per episode, like the
  single-agent baselines (`single_5step` 1 of 2, then 2 of 2;
  `single_matched` 2 of 2 both times).
- **Telling it to delegate did not help.** With `subagent.delegation_hint`
  (its steps are scarce, subagent steps do not count) it still delegated in
  0 of 20 episodes. It read the hint: in 8 episodes it weighed delegating and
  each time judged the task too simple to need it (issue #7). On 3 medium
  tasks (`hint_medium`, 2 rounds) it weighed delegating in 6 of 6 hinted
  episodes but delegated in 1, which failed (0 of 6 without the hint).
- **The manager conditions do delegate**: 6 of 6 episodes in both sweeps.
  They solved 3 of 6 on the earlier harness and 5 of 6 on this one
  (parallel 2 of 2, sequential 2 of 2, arbiter 1 of 2), at about 7-9 min and
  $0.016-0.02 per episode, 20-25 times the cost of an all-tools episode.
  They make up most of a sweep's cost and time.
- **Sail is the main source of failed trials**: a 7-9 minute outage and a
  dropped network invalidated trials, and attempts hung for up to 2015 s on
  keep-alive bytes; the client and the sweep now fail fast (docs/DESIGN.md).
  Sail re-sends Gemma's tool calls in about 85% of its streamed replies; the
  client repairs them (186 repairs in 220 Gemma calls in the third sweep).

## Study notes

Baselines:
- no subagents

Higher level strategies:
- Adaptive parellization:
-   maximizing parallel subagents with max of n concurrent agents
-   verifiers
-   debate/ensemble
-   swarms

Variables:
- task difficulty
-   defined by (summarization, + analysis, + implementation)
- size of task


┌────────────────────┬────────────────────────────────────────────────────┐
│      Category      │               Example subagent task                │
├────────────────────┼────────────────────────────────────────────────────┤
│ Locate / retrieve  │ "Find where the budget is enforced"                │
├────────────────────┼────────────────────────────────────────────────────┤
│ Summarize          │ "Describe what orchestrator.py does"               │
├────────────────────┼────────────────────────────────────────────────────┤
│ Execute / observe  │ "Run the tests and report failures"                │
├────────────────────┼────────────────────────────────────────────────────┤
│ Analyze / diagnose │ "Why does test X fail?"                            │
├────────────────────┼────────────────────────────────────────────────────┤
│ Plan / decompose   │ "Propose the steps to fix this"                    │
├────────────────────┼────────────────────────────────────────────────────┤
│ Implement / edit   │ "Apply this patch to file Y"                       │
├────────────────────┼────────────────────────────────────────────────────┤
│ Verify / test      │ "Write a test that checks the fix"                 │
├────────────────────┼────────────────────────────────────────────────────┤
│ Judge / critique   │ "Which of these 3 patches is best?" (your arbiter) │
└────────────────────┴────────────────────────────────────────────────────┘

Task Difficuluties:
- Easy: Locate/retrieve + Summarize
- Medium: Easy + Execute / Observe + Analyze/Diagnose
- Hard: Medium + plan/decompose+implement/edit + verify/test

Research Question: What is the best user experience for agentic coding with subagents? (user experience being represented by wallclock time and subagent calls)

Experiments:
- To start, only 1 model
- dsv4.1 no subagents
- subagent strategies:
-   [n parallel agents, no parallelization, arbiter] x [easy/medium/hard task difficulty]
-     LLM arbiter{dsv4.1} looking over debates https://arxiv.org/pdf/2602.12276

- Later on: give cost structure for spawning different model calls

Subagent costs:
- ~ to real costs/tokens
- changing cost structures/allocation based on proxies for probability of success

To keep track of:
- final score on benchmark
- number of subagents spawned & distribution of tokens used in each (and tool calls by each) & $ inference costs
- orchestrator # of tokens & inference costs
- wallclock time 
- 

## Layout

```
subagents_bench/
  config.py           ExperimentConfig: one condition, loaded strictly from YAML
  llm.py              Sail client: streamed SSE, tool-call repair, fail-fast retries
  environment.py      harbor's container or the host shell, with a shared stop event
  tools.py            bash, edit_file, read_file, list_dir, search via a per-agent shell
  prompts.py          orchestrator, subagent, judge and debate prompts
  orchestrator.py     the step loop and spawn_subagents
  interactive.py      timed sessions: hints as user messages, background subagents, wait, reply
  hints.py            SWE-sweep subtasks -> the message schedule; generated bug reports and code questions
  subagent.py         one subagent run (a dispatch)
  arbiter.py          candidates, judge, debate, copy isolation and apply
  records.py          the episode records
  harness/harbor_agent.py   harbor's external agent (OrchestratorAgent)
  sweep.py, cli.py    sweep specs, preflight, harbor jobs; `subagents-bench`
  analysis/           validity, cost, critical path, tables, delegation gate; interactivity.py (hints),
                      requests.py (replies: answers, estimates, done claims)
  monitor/            the HTML monitor of an interactive sweep (python -m subagents_bench.monitor)
  plots/              the nine figures
configs/              sail/base.yaml, sweeps/{pilot,smoke,swe_sweep,swe_interactive*}.yaml, pricing.yaml
scripts/              setup.sh, run_experiment.sh, analyze.sh
tests/                offline tests (fake models, fake harbor); RUN_DOCKER=1, RUN_LIVE=1 opt in
```

Tests: `.venv/bin/python -m pytest -q` (offline, under a minute).
`RUN_DOCKER=1` adds the container tests, `RUN_LIVE=1` one tool-calling request
to each Sail model.
