# Reproducing the subagents-bench study

A large orchestrator (DeepSeek-V4.1-Flash on Sail) directs small subagent
models on Terminal-Bench 2.0 tasks, run by harbor 0.24.0 in local Docker. Every
LLM call and tool call is logged with raw tokens and timestamps, so cost and
latency can be recomputed offline under other price and serving assumptions.
The design is in [`DESIGN.md`](DESIGN.md); this page is the runbook.

From a fresh clone it takes three scripts:

```bash
scripts/setup.sh                                        # 1. install + verify (idempotent)
scripts/run_experiment.sh configs/sweeps/smoke.yaml     # 2a. small end-to-end check (24 episodes)
scripts/run_experiment.sh                               # 2b. the pilot sweep (360 episodes)
scripts/analyze.sh jobs/pilot                           # 3. tables + figures
```

All three work from any directory (they `cd` to the repo root), are safe to
re-run, and end with a PASS/FAIL checklist.

## 1. Prerequisites

| Need | Notes |
|---|---|
| macOS (Apple Silicon or Intel) or Linux | Windows is untested. |
| [Docker](https://docs.docker.com/get-docker/), running | Docker Desktop on macOS, Docker Engine on Linux. The sweeps run 4 task containers at once; the pilot tasks request 1 CPU / 2 GB each (`overfull-hbox`: 2 CPUs / 4 GB), so give Docker at least ~6 CPUs and ~12 GB of memory. |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh`. uv also downloads Python 3.13 if you don't have it. |
| git, bash, network access | Network reaches Sail's API, Docker Hub (task images) and GitHub (task definitions). |
| Disk | About 0.5 GB for `.venv`, under 50 MB for the task definitions, plus the task images: these are `linux/amd64` images (for example `alexgshaw/fix-git:20251031` is 310 MB), one per task, so expect a few GB for the 10 pilot tasks. Job logs are small (MBs per job). |

## 2. Credentials

`scripts/setup.sh` creates `.env` from [`.env.example`](../.env.example) the
first time it runs (gitignored, `chmod 600`). Fill it in and run setup again.

- **Sail** (required for every sweep): `SAIL_API_KEY=...` in `.env`.
- **Prime Intellect** (only for `configs/sweeps/prime_size.yaml`): either run
  `prime login` once (the CLI stores `api_key` and `team_id` in
  `~/.prime/config.json`, which the harness reads), or set `PRIME_API_KEY` and
  `PRIME_TEAM_ID` in `.env`. The team id is sent as `X-Prime-Team-ID`;
  without it Prime bills your personal balance instead of the team's.

`.env` never overrides a variable that is already set in your shell. No
script prints a credential: the checks report only `set` / `missing`.

## 3. The three commands

### `scripts/setup.sh`

1. Checks that uv is installed and that Docker is installed and running.
2. Creates `.venv` with Python 3.13 (`uv venv`) and installs the package
   with the exact dependency versions in `uv.lock` (`uv sync --locked
   --all-extras`; harbor is pinned to `harbor==0.24.0`).
3. Creates `.env` if it is missing and checks the credentials.
4. Runs the offline unit tests (`pytest -m "not live and not docker"`; no
   network, no Docker, no API spend).
5. Downloads the Terminal-Bench 2.0 task definitions to
   `.cache/datasets/terminal-bench@2.0/` (`harbor dataset download`).

Missing Prime credentials are a WARN, not a FAIL: they are only needed for
the size sweep. `SKIP_TESTS=1 scripts/setup.sh` skips step 4.

### `scripts/run_experiment.sh [SPEC] [sweep options]`

`SPEC` defaults to `configs/sweeps/pilot.yaml`. The script runs:

1. **`subagents-bench preflight SPEC --oracle`**, which checks that:
   - Docker is up;
   - harbor 0.24.0 is installed and the dataset downloads;
   - every task exists, and writes `<jobs_dir>/tasks_meta.json`;
   - every condition's config loads;
   - the API keys resolve;
   - one tiny streaming request per distinct (model, provider, tier)
     succeeds;
   - harbor's *oracle* agent (the reference solution) solves every task on
     this machine (`<jobs_dir>/_oracle`).

   The oracle job runs once and is reused on later runs. Delete
   `<jobs_dir>/_oracle` to redo it. Results are in `<jobs_dir>/preflight.json`.
2. **`subagents-bench sweep SPEC`**: one `harbor run` job per (round,
   condition). Conditions run in a per-round shuffled order, so drift in
   provider latency is spread across conditions.

Extra options go to `sweep`, for example:

```bash
scripts/run_experiment.sh configs/sweeps/pilot.yaml --only single_5step gemma31b_parallel_hard --rounds 1
```

To see the full plan, every `harbor run` command and a rough cost estimate
without running anything (no preflight, no containers, no API requests):

```bash
scripts/run_experiment.sh configs/sweeps/pilot.yaml --dry-run
```

`.venv/bin/subagents-bench validate configs/sweeps/pilot.yaml` checks that
every resolved condition loads.

`SKIP_PREFLIGHT=1` skips preflight (for example, when you resume a sweep that
already passed it).

### `scripts/analyze.sh [JOBS_DIR] [OUT]`

This runs `check-run` (validates every record against `schema.py` and checks
internal consistency), then `completeness` (which fields are never
populated), then `aggregate` (CSV tables), then `plot` (figures).

- `JOBS_DIR` defaults to `jobs/pilot`.
- `OUT` defaults to the basename of `JOBS_DIR`.

Tables go to `analysis/<OUT>/` and figures to `figures/<OUT>/`. The figures
come with `index.md`, which has one caption per figure. Each step's log is
saved in `analysis/<OUT>/_logs/`. You can analyze a partial sweep.

## 4. Expected cost and runtime (estimates)

`sweep --dry-run` prints these numbers. They come from an **assumed** token
and latency model (`subagents_bench.sweep.RoughModel`), not from a
measurement, so treat them as orders of magnitude. After the smoke sweep,
`analysis/smoke/episodes.csv` has the real per-episode tokens, cost and time.

| Sweep | Jobs | Episodes | API cost, typical / all budgets exhausted | Wall-clock (sequential jobs, 4 trials at once) |
|---|---|---|---|---|
| smoke | 12 | 24 | ~$0.8 / ~$2 | ~2–3 h |
| pilot | 36 | 360 | ~$12 / ~$32 | ~1 day |
| prime_size | 12 | 120 | ~$4 / ~$12 | ~6 h |

Runtime is dominated by `single_matched` (40 orchestrator steps) and by
emulated amd64 containers on Apple Silicon (see caveats). The first run of a
sweep also pulls the task images. Preflight's oracle job takes a few minutes
per sweep.

## 5. Outputs

```
jobs/<sweep>/
  configs/<condition>.yaml          resolved ExperimentConfig per condition (what the agent ran)
  tasks_meta.json                   difficulty, category, expert/junior minutes, timeouts, image per task
  preflight.json                    preflight checks, per-model test requests, oracle results
  sweep.json                        provenance: spec, git commit + dirty flag, Python/harbor versions,
                                    uv.lock sha256, dataset, condition order + seed, provider tiers,
                                    date, and every invocation of the sweep
  sweep_manifest.jsonl              one line per job event: job, round, condition, cmd, t_start, t_end,
                                    return_code, n_trials, n_errors (events: run, skipped_complete,
                                    moved_partial, interrupted)
  _oracle/                          the oracle feasibility job
  _partial/<job>.partial.<k>/       interrupted jobs that were moved aside and rerun (kept, not analyzed)
  <sweep>__r<round>__<condition>/   one harbor job
    result.json, config.json, job.log          harbor's job files
    <task>__<id>/                               one trial (= one episode)
      result.json                               harbor: reward, exception, phase timings
      agent/episode.json, calls.jsonl, tools.jsonl, steps.jsonl, spawns.jsonl,
            dispatches.jsonl, arbitrations.jsonl, transcript.jsonl   (schema: subagents_bench/schema.py)
      verifier/                                 test output and reward
analysis/<sweep>/                   episodes.csv, calls.csv, conditions.csv, paired.csv, ... (DESIGN section 6)
figures/<sweep>/                    PNG + PDF per figure, index.md with captions (DESIGN section 7)
```

Every file, field and table column is described in [`DATA.md`](DATA.md).
`jobs/`, `analysis/`, `figures/` and `.cache/` are gitignored. Archive
`jobs/<sweep>/` to share raw results: the analysis can be rebuilt from it alone.

## 6. Resuming

Re-run the same command. For each job:

- if `result.json` shows every trial finished, the job is **skipped**;
- if the job dir exists but is incomplete (interrupted), it is **moved aside** to
  `_partial/<job>.partial.<k>` and **rerun** from scratch. Nothing is deleted,
  and the move is recorded in the manifest.

Other things that stop a sweep:

- **Billing error.** After every job the sweep checks each episode for
  `exit_reason == "billing_error"` and stops (exit code 3) if it finds one.
  Top up the account and re-run.
- **Repeated failures.** Three failed jobs in a row (Docker or harbor
  trouble) stop the sweep.
- **Changed condition config.** A resumed sweep refuses to run if a
  condition's resolved config changed since its first jobs ran, because that
  would mix two configs under one name. Use a new `jobs_dir` instead.
- **Changed code.** If the git commit changes between invocations,
  `sweep.json` records it and the sweep warns about it.

Ctrl-C gives harbor up to two minutes to cancel its trials and remove their
containers. If you kill it harder, check `docker ps` for leftover task
containers.

## 7. Running on a cloud sandbox instead of local Docker

harbor supports remote environments (`daytona`, `modal`, `e2b`, `runloop`,
`prime`, ...; see `harbor run --help`). To use one:

1. Put the provider's credentials in `.env` (see `.env.example`).
2. Copy the sweep spec and change `harbor.env`, typically raising
   `n_concurrent`:

   ```yaml
   extends: configs/sweeps/pilot.yaml
   name: pilot_daytona
   harbor: {env: daytona, n_concurrent: 16, agent_timeout_multiplier: 4.0, extra_args: []}
   jobs_dir: jobs/pilot_daytona
   ```

3. Run `scripts/run_experiment.sh configs/sweeps/pilot_daytona.yaml`.
   Preflight skips the local Docker check for non-docker environments.

The agent itself still runs on your machine; only the task containers move,
and the agent's tool calls go to the sandbox. Expect different tool latency:
it is logged per call, so results stay comparable.

## 8. Caveats

- **Emulation on Apple Silicon.** The Terminal-Bench task images are
  `linux/amd64`, so on an arm64 Mac they run under emulation, and commands
  inside the container (compiling, tests) are slower than on x86 hardware.
  That is why tool time is measured separately from LLM time: each tool call
  has `duration_sec` and `exec_sec`, and `episode.json` records
  `container_arch` and `host_machine`. The offline latency simulation can
  then replace or rescale tool time. Budgets are in steps and tokens, not
  seconds, so emulation never changes a trajectory; it only changes measured
  wall-clock time.
- **Provider latency drifts** over a day-long sweep. Rounds interleave the
  conditions (shuffled per round, seed recorded) so drift does not
  confound any one condition. Calibration plots show latency over sweep time.
- **Tokens and dollars.** The run-time `cost_usd` uses the prices in the
  config (null when a model has no price). Reported costs are recomputed
  offline from raw tokens using `configs/pricing.yaml`. Sail does not report
  cost; Prime's usage carried no cost field in our test requests.
- **Harness overhead.** Wall-clock time also includes the harness's own
  measurement commands (workdir markers, modified-file scans, `git diff
  --stat`), which belong to no agent. They are logged as `harness_exec_sec`
  and, with thread start-up, make up `overhead_sec = wall_clock_sec -
  cp_total_sec`. `analysis/<sweep>/summary.json` (`latency_checks`) lists
  the episodes whose overhead exceeds 10% of wall-clock, with and without
  the part `harness_exec_sec` explains. Under amd64 emulation each such
  command costs a few tenths of a second.
- **Sampling.** Every model samples at temperature 0.7, so rounds are
  independent samples, not replays. Reasoning is left at each provider's
  default (DeepSeek thinks; the Gemma models do not), and it is recorded per
  call.
- **Timeouts are safety caps.** harbor's agent timeout is the task's
  `[agent] timeout_sec` × 4 (`agent_timeout_multiplier`), and the episode cap
  is 3600 s. Hitting a cap shows up as `exit_reason` `cancelled` or
  `timeout`.

## 9. What each condition means

All conditions share the orchestrator (DeepSeek-V4.1-Flash, Sail `balanced`
tier, 5 orchestrator steps unless noted) and the tasks. See
[`configs/sail/base.yaml`](../configs/sail/base.yaml) for every setting and
[`configs/sweeps/pilot.yaml`](../configs/sweeps/pilot.yaml) for the overrides.

| Condition | Meaning |
|---|---|
| `single_5step` | Baseline: the orchestrator works alone (no subagents) for 5 steps. |
| `single_matched` | Compute-matched baseline: alone, 40 steps. |
| `gemma31b_parallel_{easy,medium,hard}` | Up to 3 Gemma-4-31B subagents run **concurrently** per `spawn_subagents` call. |
| `gemma31b_sequential_{easy,medium,hard}` | The same grants, but subagents run **one at a time**. |
| `gemma31b_arbiter_{easy,medium,hard}` | Each instruction goes to 3 candidate subagents; an LLM judge (the orchestrator model) picks a winner, with one confidence-aware debate round when the judge is not confident (≥ 0.8) or the candidates disagree. |
| `samemodel_parallel_hard` | Same-model control: parallel/hard, but the subagents run the orchestrator's own model. |
| `qwen{0.8,2,4,9}b_parallel_hard` (`prime_size.yaml`) | Size sweep: Qwen3.5 subagents of 0.8B to 9B parameters on Prime, parallel/hard, with the same orchestrator. |

Scope is the delegation difficulty, and it sets the subagent's tools:

- **easy** = locate/retrieve + summarize: `read_file`, `list_dir`, `search`
- **medium** = easy + execute/observe + analyze: adds `bash`
- **hard** = medium + implement/edit + verify: adds `edit_file`

Subagent budgets: at most 3 live at once, 15 dispatches per episode (arbiter
candidates count individually), and 8 steps each.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `setup.sh`: docker FAIL | Start Docker Desktop / `dockerd`, re-run. |
| `setup.sh`: dependencies FAIL with "lockfile needs to be updated" | `pyproject.toml` changed without `uv lock`; run `uv lock`, commit `uv.lock`. |
| preflight: request FAIL with a 402/balance message | Top up the provider account (Sail) or fix the Prime team id. |
| preflight: oracle FAIL on a task | The task does not pass with its reference solution on this machine (often an image or emulation problem); see `<jobs_dir>/_oracle/<task>__*/verifier/`. Drop the task from the spec or fix Docker resources. |
| sweep refuses: "resolved config ... differs" | You edited a config mid-sweep; use a new `jobs_dir`, or `--allow-config-change` if you really mean to mix. |
| `harbor` not found | Run `scripts/setup.sh`; the sweep uses `.venv/bin/harbor`. |
