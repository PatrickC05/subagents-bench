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

## The experiment

One episode is one Terminal-Bench 2.0 task, run by **harbor 0.24.0** in the
task's official Docker image:

1. A large **orchestrator** (`deepseek-ai/DeepSeek-V4.1-Flash` on Sail's
   `balanced` tier) gets the task instruction and a fixed budget of **5
   steps**. Each step is one LLM call followed by the tools it requested:
   `bash`, `edit_file`, `read_file`, `list_dir`, `search`, `finish`.
2. With `topology: subagents` it can also call `spawn_subagents(instructions)`.
   Small **subagent** models (default `nvidia/Gemma-4-31B-IT-NVFP4` on Sail)
   then work in the same container, with a tool set set by the delegation
   **scope** and scheduled by the **strategy** (below).
3. harbor's verifier grades the container's final state (reward 1 =
   resolved). The harness logs every LLM call, tool call, step, dispatch and
   arbitration with raw tokens and timestamps, so cost and latency are
   recomputed offline under other price schemes and serving assumptions.

Budgets are steps and tokens; wall-clock limits are only loose safety caps,
so provider latency never changes a trajectory. The full contract is
[`docs/DESIGN.md`](docs/DESIGN.md).

### Axes

| Axis | Values | Config field |
|------|--------|--------------|
| Topology | `single` (orchestrator alone) / `subagents` | `topology` |
| Strategy | `parallel` (up to `max_concurrent` subagents at once) / `sequential` (one at a time) / `arbiter` (3 candidates per instruction, an LLM judge, one confidence-aware debate round) | `subagent.strategy` |
| Scope (delegation difficulty) | `easy` = `read_file, list_dir, search` / `medium` = easy + `bash` / `hard` = medium + `edit_file` | `subagent.scope` |
| Subagent model | Gemma-4-31B (default), the orchestrator itself (same-model control), Qwen3.5-{0.8,2,4,9}B on Prime (size sweep) | `subagent.model` |
| Task difficulty | Terminal-Bench label, category, expert-time estimate (covariates) | `tasks_meta.json` |

The pilot sweep ([`configs/sweeps/pilot.yaml`](configs/sweeps/pilot.yaml), on
[`configs/sail/base.yaml`](configs/sail/base.yaml)) is 15 conditions × 10
tasks × 3 rounds = 450 episodes. The smoke sweep
([`configs/sweeps/smoke.yaml`](configs/sweeps/smoke.yaml)) is the same 15
conditions on 2 tasks × 1 round. The optional size sweep
([`configs/sweeps/prime_size.yaml`](configs/sweeps/prime_size.yaml)) runs the
four Qwen3.5 subagent arms on Prime next to two anchors (`single_5step`,
`gemma31b_parallel_hard`) in the same rounds, 180 episodes.

### Baselines

- `single_5step`: the orchestrator alone, same 5 steps.
- `single_matched`: alone, 40 steps (step-matched: 40 = 5 orchestrator
  steps × 8 subagent `max_steps`, one subagent's steps per orchestrator step
  along the serial path; it matches steps, not FLOPs, tokens or the total
  work of parallel subagents).
- `samemodel_parallel_hard`: subagents running the orchestrator's own model,
  which separates *having* subagents from subagent *size*.

All three are conditions in [`configs/sweeps/pilot.yaml`](configs/sweeps/pilot.yaml).

### Manager orchestrator

With every tool of its own, the orchestrator delegated in 0 of 18
subagent-condition episodes of the first smoke sweep: it simply solved the
task itself. The `manager_{parallel,sequential,arbiter}_hard` conditions set
`tools.orchestrator_tools: [read_file, list_dir, search]`, so the
orchestrator can read and plan but every command, edit and test must go
through subagents. In the follow-up smoke run, `manager_arbiter_hard`
delegated in both of its episodes (11 dispatches, 3 arbitrations); one
solved its task, the other is invalid because a provider outage failed two
subagent calls.
`subagents-bench completeness --strict` (step 2 of `scripts/analyze.sh`) now
fails a sweep in which a subagent condition never delegated.
The earlier Anthropic-orchestrator configs
([`configs/baselines/`](configs/baselines/), [`configs/unified.yaml`](configs/unified.yaml),
[`configs/exploitbench.yaml`](configs/exploitbench.yaml)) still load but are
not part of the Sail sweeps.

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


## Metrics

Every episode writes eight record files into its harbor trial dir
(`<trial>/agent/`, schema in
[`subagents_bench/schema.py`](subagents_bench/schema.py)); `subagents-bench
aggregate` turns them into tables. Every file, field and column is described
in [`docs/DATA.md`](docs/DATA.md). The main measurements:

- **resolve rate**: harbor's verifier reward, with Wilson 95% intervals and
  paired McNemar tests against the baselines;
- **tokens** per role (orchestrator, subagent, judge, debate) and kind
  (uncached input, cached input, reasoning, visible output), per call;
- **dollars** under every price scheme of
  [`configs/pricing.yaml`](configs/pricing.yaml) (Sail tiers, Prime, an
  OpenRouter reference median), recomputed offline; a model without a price
  is NaN, never 0;
- **FLOPs** (2 × active parameters × tokens);
- **wall-clock time**: measured, decomposed along the critical path
  (orchestrator LLM, subagent LLM, judge LLM, tools, harness overhead), and
  simulated under other serving profiles
  ([`configs/serving.yaml`](configs/serving.yaml): replay, a fit to the
  measured latencies, a theoretical H100 roofline);
- **delegation behaviour**: subagents spawned, re-dispatches,
  instruction-following (granted / requested), per-dispatch tokens, tool
  calls, steps, status and files modified; arbiter agreement, confidence,
  debate rounds and winners;
- **provider behaviour**: TTFT, decode speed, retries and backoff, chunk
  timings, cached-token counts.

## Layout

```
subagents_bench/
  config.py           ExperimentConfig (YAML): topology, strategy, scope, budgets, models
  types.py            messages, tool specs, LLMResponse, usage, timings, errors
  schema.py           every on-disk record (field docs) + validate_record
  llm.py              Sail / Prime / vLLM / OpenAI (httpx SSE) and Anthropic clients
  credentials.py      .env loading, Prime CLI credentials (never printed)
  environment.py      harbor / local Docker / local shell sandboxes
  tools.py            bash, read_file, list_dir, search, edit_file (per-agent shell state)
  workspace.py        markers, modified-file scans, copies, manifests, diffs, apply
  prompts.py          orchestrator, subagent, judge and debate prompts
  orchestrator.py     the step loop and spawn_subagents
  subagent.py         bounded subagent loop, context guard
  arbiter.py          candidates, judge, confidence-aware debate, apply
  telemetry.py        EpisodeRecorder: writes the record files
  provenance.py       source fingerprint and git state of the code that ran
  runner.py           local runs + `validate` / `run`
  sweep.py            sweep specs, preflight, harbor jobs, `tasks`
  analysis/           check-run, completeness, aggregate, calibrate
  plots/              the figures and index.md
  harness/
    harbor_agent.py        harbor BaseAgent (OrchestratorAgent)
    exploitbench_agent.py  ExploitBench adapter
configs/
  sail/base.yaml      base config of the Sail sweeps
  sweeps/             pilot, smoke, prime_size sweep specs
  pricing.yaml, pricing_reference.yaml, models.yaml, serving.yaml, provider_models.json
  baselines/, unified.yaml, exploitbench.yaml   earlier Anthropic configs
scripts/
  setup.sh, run_experiment.sh, analyze.sh       the three reproduction scripts
  gen_data_docs.py    regenerates docs/DATA.md
  bench_latency.py    standalone provider latency micro-benchmark (not part of the sweeps)
docs/
  REPRODUCE.md        runbook: setup, the three scripts, cost/runtime, outputs, resuming
  DESIGN.md           implementation contract
  DATA.md             data dictionary: every file, field and table column
  PRICING_SOURCES.md  where every price and parameter count comes from
tests/                offline unit tests, the offline pipeline test, opt-in live/docker tests
```

## Running

Three scripts reproduce the study from a fresh clone; the runbook, with
prerequisites, credentials, expected cost and runtime, outputs, resuming and
troubleshooting, is [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

```bash
scripts/setup.sh                                        # install from uv.lock, check Docker + credentials, offline tests, dataset
scripts/run_experiment.sh configs/sweeps/smoke.yaml     # preflight, then the smoke sweep (30 episodes)
scripts/run_experiment.sh                               # the pilot sweep (450 episodes); resumable
scripts/analyze.sh jobs/pilot                           # check-run, completeness, aggregate, plot
```

`scripts/run_experiment.sh SPEC --dry-run` prints the plan, every `harbor
run` command and a rough cost estimate without running anything.
Credentials go in `.env` (created from [`.env.example`](.env.example)):
`SAIL_API_KEY`, and for the size sweep `PRIME_API_KEY` + `PRIME_TEAM_ID` or a
logged-in `prime` CLI.

Other commands (`.venv/bin/subagents-bench --help`):

```bash
subagents-bench validate configs/sweeps/pilot.yaml      # any config, sweep spec or pricing/models/serving YAML
subagents-bench tasks terminal-bench@2.0                # task metadata
subagents-bench run CONFIG --task-id demo --instruction "..."   # one local episode (sandbox local_docker / local_shell)
.venv/bin/python -m pytest -q                           # offline tests (RUN_LIVE=1 / RUN_DOCKER=1 for opt-in ones)
```

A single harbor job by hand:

```bash
SUBAGENTS_BENCH_CONFIG=jobs/smoke/configs/single_5step.yaml \
  .venv/bin/harbor run -d terminal-bench@2.0 -i fix-git \
  -a subagents_bench.harness.harbor_agent:OrchestratorAgent -e docker -n 1 -k 1 -o jobs/manual --job-name demo -y
```

**ExploitBench**: call
`subagents_bench.harness.exploitbench_agent.run_exploitbench_trial(config,
task_id, instruction, exec_fn, grade_fn)` from an ExploitBench runner;
`exec_fn` runs a command in the task container and `grade_fn` returns the
capability bitmap.

## Notes

- Subagents share the orchestrator's container, so their edits are visible to
  everyone. The exception is arbiter + hard scope: each candidate edits a
  private copy of the workdir, and only the judge's winner is applied to it
  (paths naming the workdir are redirected to the copy, and stray writes to
  the real workdir are undone and recorded). Only the workdir is copied:
  changes outside it (`/etc`, packages, services) are shared and made live by
  every candidate.
- Provider prices do not track model size, so the study logs raw tokens and
  timings and recomputes cost and latency offline; the run-time `cost_usd` is
  only an estimate from the config's prices (null when a model has none).
- The Terminal-Bench images are `linux/amd64`; on Apple Silicon they run
  emulated, which slows tool calls but not LLM calls. Tool time is measured
  separately and `container_arch` is recorded, so the latency analysis can
  account for it.
- `google/gemma-4-12B-it` is served by Sail with a 16384-token context; a
  condition that uses it needs `context_window: 16384` (the commented
  condition in `configs/sweeps/pilot.yaml` has it).
- `subtask_type` only swaps a legacy orchestrator prompt prefix
  ([`prompts.py`](subagents_bench/prompts.py)); the Sail sweeps use `none`.
