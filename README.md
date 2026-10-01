# subagents-bench

Measuring the **cost and latency** of a frontier **orchestrator** directing
smaller **subagents** on executable coding tasks — the configuration
practitioners actually deploy, but one existing scaling studies don't primarily
measure.

Prior work (Kim et al. 2025, *Towards a Science of Scaling Agent Systems*)
assigns the **same** model to every agent, reports **accuracy** rather than
latency, and includes no repository-level coding with executable tests. This
project instead varies **subagent size** independently of the orchestrator, runs
on **Terminal-Bench 4.0** and **ExploitBench** (tasks with real, runnable
scoring), and reports resolve rate against **tokens** and **wall-clock time** as
Pareto curves.

## The experiment

Each episode runs for a fixed **5 orchestrator steps**:

1. A frontier orchestrator (Anthropic API) receives an issue and the repository
   at its base commit inside the official benchmark Docker image, with `bash`,
   `edit_file`, and `search` tools.
2. It may spawn **subagents** — independent workers with the same tools in the
   *same* container, running a (usually smaller) model served by **vLLM**. A
   `subtask_type` prefix optionally steers *what* it delegates.
3. The harness scores the patch and logs telemetry.

### Axes (one `ExperimentConfig` per point)

| Axis | Field | Values |
|------|-------|--------|
| Orchestrator model | `orchestrator.model` | any Anthropic / OpenAI-compatible model |
| Subagent model | `subagent.model` | any model, incl. vLLM-served (**size axis**) |
| # subagents | `subagent.max_concurrent`, `max_dispatches` | int |
| Subtask type | `subtask_type` | `none`, `parallel_exploration`, `localize_then_fix`, `verify`, `decompose` |
| Benchmark subset | `benchmark`, `task_ids`, `difficulties` | TB / EB; difficulty = **task-size axis** |
| Timeout / budget | `budget.*` | steps, tokens, seconds |
| Sandbox | `sandbox.kind` | `harbor`, `exploitbench`, `local_docker`, `local_shell` |

### Baselines

- **Single agent** (`topology: single`) — orchestrator alone, same tools/budget.
  Validate against the model's published SWE-bench Verified result.
  → [`configs/baselines/single_agent.yaml`](configs/baselines/single_agent.yaml)
- **Same-model subagents** (`subagent.model: null`) — isolates *having*
  subagents from subagent *size*.
  → [`configs/baselines/same_model_subagents.yaml`](configs/baselines/same_model_subagents.yaml)

## Metrics

Per trial ([`telemetry.py`](subagents_bench/telemetry.py)), written to
`runs/<experiment>/<task>/telemetry.json`:

- **resolve rate** (official SWE-bench / ExploitBench scoring, in a clean container);
- **total tokens**, split orchestrator vs. subagent, in/out/cache;
- **total generation cost** (USD, from per-model prices);
- **subagent dispatches and re-dispatches**;
- **wall-clock time**, plus every API call timestamped in `calls.jsonl` for
  latency reconstruction;
- **instruction-following**: granted-vs-requested subagents per spawn call.

## Layout

```
subagents_bench/
  config.py         Unified ExperimentConfig (YAML) for both benchmarks
  llm.py            Anthropic + OpenAI-compatible (vLLM) clients, usage/cost
  environment.py    Environment protocol; harbor / local_docker / shell backends
  tools.py          bash / edit_file / search
  prompts.py        Orchestrator & subagent prompts + subtask prefixes
  subagent.py       Bounded tool-using worker
  orchestrator.py   5-step loop, spawn_subagents / finish, budget enforcement
  telemetry.py      The reported metrics
  analysis.py       Per-config tables + Pareto points
  cli.py            validate / run / aggregate
  harness/
    harbor_agent.py        BaseAgent wrapper for Terminal-Bench 4.0
    exploitbench_agent.py  ExploitBench adapter
configs/            unified.yaml, exploitbench.yaml, baselines/
tests/              scripted-LLM end-to-end tests (no network/Docker)
```

## Running

Install:

```bash
pip install -e ".[all,dev]"      # anthropic + openai + pytest
```

Set keys. Subagents run on hosted OpenAI-compatible endpoints — **Prime
Intellect** (`provider: prime`) and **Sail** (`provider: sail`) — which fill in
their base URL and key env var automatically and report USD cost in the usage
object (so `price_in/out: 0` still gives real cost). A local vLLM server works
too (`provider: vllm`, `base_url: http://localhost:8000/v1`).

```bash
export ANTHROPIC_API_KEY=...     # frontier orchestrator (Anthropic API)
export PRIME_API_KEY=...         # Prime Intellect subagents (provider: prime)
export SAIL_API_KEY=...          # Sail subagents           (provider: sail)
```

| provider | base URL | key env |
|----------|----------|---------|
| `prime` / `prime_intellect` | `https://api.pinference.ai/api/v1` | `PRIME_API_KEY` |
| `sail` | `https://api.sailresearch.com/v1` | `SAIL_API_KEY` |
| `vllm` / `openai` | from `base_url` | from `api_key_env` |

Sail is also Anthropic-compatible, so the orchestrator can point at it too
(`provider: anthropic`, `base_url: https://api.sailresearch.com/v1`).

**Terminal-Bench 4.0** (via harbor): point harbor at the agent wrapper, and the
config via env var so a sweep launches one job per config without code changes.

```bash
SUBAGENTS_BENCH_CONFIG=configs/unified.yaml \
  harbor run -d "terminal-bench@head" \
  --agent-import-path subagents_bench.harness.harbor_agent:OrchestratorAgent \
  --n-concurrent 3
```

Terminal-Bench 4.0 = 3 GPU jobs of 66 tasks; stratify by the task difficulty
label (`difficulties:`), which is the task-size axis.

**ExploitBench**: call
`subagents_bench.harness.exploitbench_agent.run_exploitbench_trial(config,
task_id, instruction, exec_fn, grade_fn)` from an ExploitBench runner once it has
a live task handle. `exec_fn` runs a command in the task container; `grade_fn`
returns the 16-flag capability bitmap.

**Local smoke test** (no benchmark harness — your own container or host shell):

```bash
subagents-bench validate configs/unified.yaml
subagents-bench run configs/unified.yaml --task-id demo --issue "Fix the failing test in app/"
```
(set `sandbox.kind: local_docker` + `sandbox.image`, or `local_shell` for offline).

**Aggregate** into tables + Pareto points:

```bash
subagents-bench aggregate runs --csv summary.csv --pareto mean_total_tokens
```

## Notes

- The orchestrator is Anthropic-native with prompt caching on the system block
  so long repository context is cheap across the 5 steps.
- Subagents share the orchestrator's container, so their file edits are visible
  to the orchestrator and to each other — they are workers, not sandboxes.
- vLLM subagents on fixed hardware have no API price; set `price_in/out: 0` to
  report tokens only, or an amortised $/Mtok estimate for a cost comparison.
- `subtask_type` currently only swaps the orchestrator prompt *prefix*
  ([`prompts.py`](subagents_bench/prompts.py)); the wordings are intentionally
  thin and meant to be iterated on.
