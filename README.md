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

This version runs Terminal-Bench 2.0 only; the ExploitBench adapter is not part of it.

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

`run_experiment.sh [SPEC] [--dry-run] [--only COND...] [--rounds 1,2]` runs
`subagents-bench preflight SPEC --oracle` (Docker, harbor, the key, one request
per model, harbor's oracle agent on every task), then `subagents-bench sweep
SPEC`: one `harbor run` per (round, condition). It prints one line per trial
as harbor finishes it. `--dry-run` prints the jobs and their harbor commands
and runs nothing. On macOS the sweep runs under `caffeinate -i` (turn off with
`SB_NO_CAFFEINATE=1`). Exit codes: 0 done, 1 jobs failed or need a retry,
2 spec or config error, 3 billing error, 4 provider outage, 130 interrupted.

The sweep is resumable: run the same command again after any stop. Complete
jobs are skipped; an unfinished job, or one with a trial that did not measure
its condition (provider failure, billing error, harbor or agent crash before
the verifier), is moved to `jobs/<sweep>/_partial/` and rerun. An agent
timeout is an outcome, not a retry. On a provider failure the sweep sends one
tiny request to every model and stops with exit 4 if one still fails.
Ctrl-C gives harbor up to two minutes to remove its containers.

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

## Where data and figures land

```
jobs/<sweep>/                       the sweep (git-ignored, like analysis/ and figures/)
  configs/<condition>.yaml          resolved config of each condition
  sweep.json, sweep_manifest.jsonl  status and one line per job event
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
  each time judged the task too simple to need it (issue #7).
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
  subagent.py         one subagent run (a dispatch)
  arbiter.py          candidates, judge, debate, copy isolation and apply
  records.py          the episode records
  harness/harbor_agent.py   harbor's external agent (OrchestratorAgent)
  sweep.py, cli.py    sweep specs, preflight, harbor jobs; `subagents-bench`
  analysis/           validity, cost, critical path, tables, delegation gate
  plots/              the eight figures
configs/              sail/base.yaml, sweeps/{pilot,smoke}.yaml, pricing.yaml
scripts/              setup.sh, run_experiment.sh, analyze.sh
tests/                offline tests (fake models, fake harbor); RUN_DOCKER=1, RUN_LIVE=1 opt in
```

Tests: `.venv/bin/python -m pytest -q` (offline, under a minute).
`RUN_DOCKER=1` adds the container tests, `RUN_LIVE=1` one tool-calling request
to each Sail model.
