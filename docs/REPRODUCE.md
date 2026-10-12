# Reproducing the subagents-bench study

A large orchestrator (DeepSeek-V4.1-Flash on Sail) directs small subagent
models on Terminal-Bench 2.0 tasks, run by harbor 0.24.0 in local Docker. Every
LLM call and tool call is logged with raw tokens and timestamps, so cost and
latency can be recomputed offline under other price and serving assumptions.
The design is in [`DESIGN.md`](DESIGN.md); this page is the runbook.

From a fresh clone it takes three scripts:

```bash
scripts/setup.sh                                        # 1. install + verify (idempotent)
scripts/run_experiment.sh configs/sweeps/smoke.yaml     # 2a. small end-to-end check (30 episodes)
scripts/run_experiment.sh                               # 2b. the pilot sweep (450 episodes)
scripts/analyze.sh jobs/pilot                           # 3. tables + figures
```

The optional size sweep uses the same two commands (section 3, "Size sweep").

All three work from any directory (they `cd` to the repo root; a relative
path you pass is looked up from your current directory first, then from the
repo root), are safe to re-run, end with a PASS/FAIL checklist, and print
their usage with `-h`/`--help`. They run under macOS's `/bin/bash` 3.2 as
well as a current Linux bash.

## 1. Prerequisites

| Need | Notes |
|---|---|
| macOS (Apple Silicon or Intel) or Linux | Windows is untested. |
| [Docker](https://docs.docker.com/get-docker/), running | Docker Desktop on macOS, Docker Engine on Linux. The sweeps run 4 task containers at once; the pilot tasks request 1 CPU / 2 GB each (`overfull-hbox`: 2 CPUs / 4 GB), so give Docker at least ~6 CPUs and ~12 GB of memory. |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh`. uv also downloads Python 3.13 if you don't have it. |
| git, bash, network access | Network reaches Sail's API, Docker Hub (task images) and GitHub (task definitions). |
| Disk | About 0.5 GB for `.venv`, under 50 MB for the task definitions, plus the task images: these are `linux/amd64` images (for example `alexgshaw/fix-git:20251031` is 310 MB), one per task, so expect a few GB for the 10 pilot tasks. Job logs are small (MBs per job). |

A git clone is recommended, since every run records its commit and saves
uncommitted changes as a patch. A ZIP or tarball copy works too (tests and
analysis included): the source fingerprint `source_sha256` then identifies
the code.

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
the size sweep. `SKIP_TESTS=1 scripts/setup.sh` skips step 4. If the dataset
download fails, setup prints harbor's own error (network, proxy, GitHub rate
limit, missing git) above the checklist; set `SUBAGENTS_BENCH_DATASET_CACHE`
to an existing export to use that instead. setup takes no arguments.

### `scripts/run_experiment.sh [SPEC] [sweep options]`

`SPEC` defaults to `configs/sweeps/pilot.yaml`, also when you pass only
options (`scripts/run_experiment.sh --dry-run`). The script runs:

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

   The oracle job is reused on later runs while it passes on every task. A
   failed or unfinished one is moved aside to `<jobs_dir>/_partial/` and
   rerun, so a transient failure (an image pull timeout, too little memory)
   is not cached; `subagents-bench preflight SPEC --redo-oracle` reruns even
   a passing one. Results are in `<jobs_dir>/preflight.json`.
2. **`subagents-bench sweep SPEC`**: one `harbor run` job per (round,
   condition). Conditions run in a per-round shuffled order, so drift in
   provider latency is spread across conditions.

Extra options go to `sweep`, for example:

```bash
scripts/run_experiment.sh configs/sweeps/pilot.yaml --only single_5step gemma31b_parallel_hard --rounds 1
```

`--env-file FILE` (default `<repo>/.env`) is given to preflight as well, so
both steps check and use the same credentials. `--allow-task-change` and
`--allow-code-change` let a resume mix task lists, harbor arguments or code
versions (see section 6).

To see the full plan (which jobs would be skipped, run, moved aside or
retried, and why), every `harbor run` command, the resume checks and a rough
cost estimate without running anything (no preflight, no containers, no API
requests, no files written; it exits 2 when a config problem or a resume
check would make a real run refuse, while another sweep holding the jobs dir
is only a NOTE, so you can dry-run a sweep that is running):

```bash
scripts/run_experiment.sh configs/sweeps/pilot.yaml --dry-run
```

`.venv/bin/subagents-bench validate configs/sweeps/pilot.yaml` checks that
every resolved condition loads.

`SKIP_PREFLIGHT=1` skips preflight (for example, when you resume a sweep that
already passed it).

Exit codes: 0 done; 1 a preflight check or a job failed, or a job still
needs a retry (re-run); 2 a spec or config problem, a refused resume, or
another sweep running on the same jobs dir; 3 a billing error (top up, then
re-run); 4 a provider outage: a trial hit a provider failure and the models
still fail a probe request (re-run later);
130 interrupted (re-run). Section 6 explains each.

### `scripts/analyze.sh [JOBS_DIR...]`

This runs `check-run` (validates every record against `schema.py` and checks
internal consistency), then `completeness` (which fields are never
populated), then `aggregate` (CSV tables), then `plot` (figures).

- `JOBS_DIR` defaults to `jobs/pilot`. Several dirs are analyzed as one data
  set.
- The output name comes from the `OUT` environment variable and defaults to
  the basename of the last `JOBS_DIR`: `OUT=both scripts/analyze.sh jobs/a jobs/b`.

Tables go to `analysis/<OUT>/` and figures to `figures/<OUT>/`. The figures
come with `index.md`, which has one caption per figure. Each step's log is
saved in `analysis/<OUT>/_logs/`. You can analyze a partial sweep.

Two checks guard the data before it reaches a table:

- **Treatment coverage.** `completeness --strict` fails (and `check-run` and
  `aggregate` warn) when a subagent condition never delegated in any of its
  episodes, or an arbiter condition never ran an arbitration: its resolve
  rate then measures the orchestrator alone. `episodes.csv` has `delegated`
  per episode; `conditions.csv` has `delegated_frac` and
  `resolve_rate_delegated` (the rate among the episodes that delegated)
  next to the intention-to-treat `resolve_rate`. The first smoke sweep's
  all-tools data fails this check on all 10 subagent conditions, as it
  should.
- **Validity.** A trial is invalid (`episodes.csv` `valid`,
  `invalid_reason`) when it did not measure its condition: no episode, a
  billing error, a harbor error outside the agent, a provider failure (a
  model call of any role failed after all retries), or an agent exception
  before the verifier ran (a harness bug). Invalid trials are left out of
  every condition statistic and figure, with their dispatches and
  arbitrations; the sweep reruns them (section 6).

### Size sweep (optional)

`configs/sweeps/prime_size.yaml` swaps the subagent for Qwen3.5 0.8B, 2B, 4B
and 9B on Prime (parallel/hard, same orchestrator, tasks and rounds as the
pilot). It needs the Prime credentials of section 2. Two anchor conditions,
`single_5step` and `gemma31b_parallel_hard` (Gemma-4-31B on Sail), run in
the same shuffled rounds, so size effects are measured within the sweep and
not against a pilot run hours or days earlier:

```bash
scripts/run_experiment.sh configs/sweeps/prime_size.yaml
scripts/analyze.sh jobs/prime_size                       # anchors + Qwen sizes; paired.csv vs single_5step
OUT=size scripts/analyze.sh jobs/pilot jobs/prime_size   # next to every pilot condition
```

The first is the comparison to report. In the second, conditions with the
same name in both sweeps (the two anchors) are pooled across the sweeps
(`episodes.csv` keeps the `sweep` column to tell them apart), so use it to
place the Qwen points next to the pilot's other conditions, not for the
anchors. The subagent provider differs between the Gemma anchor (Sail) and
the Qwen arms (Prime): treat provider as a covariate of that gap.

## 4. Expected cost and runtime (estimates)

`sweep --dry-run` prints these numbers. They come from a token and latency
model (`subagents_bench.sweep.RoughModel`) calibrated from the smoke sweep of
2026-10-11/12 on Sail's balanced tier: 24 episodes of the all-tools
conditions (about $0.001 and 70 s per trial, about 90 s per 2-trial job) and
the first two `manager_arbiter_hard` episodes (about $0.015–0.025 and 6–17
min each, with Gemma's time to first token reaching 12–73 s). Two easy tasks
are a thin base: the numbers are still rough, orders of magnitude rather
than a quote. A larger sweep's `analysis/<sweep>/episodes.csv` has the real
per-episode tokens, cost and time.

| Sweep | Jobs | Episodes | API cost, typical / all budgets exhausted | Wall-clock (sequential jobs, 4 trials at once) |
|---|---|---|---|---|
| smoke | 15 | 30 | ~$0.13 / ~$1.5 | ~0.7 h |
| pilot | 45 | 450 | ~$2 / ~$22 | ~6 h |
| prime_size | 18 | 180 | ~$0.16 / ~$8 | ~1 h |

"Typical" assumes what the smoke sweep showed: an orchestrator that keeps
all its tools does not delegate (0 of 18 episodes), so those subagent
conditions cost about as much as `single_5step`, while the `manager_*`
conditions must delegate and dominate cost and time (sequential is the
slowest). If the orchestrator does delegate, costs move toward the upper
column. Emulated amd64 containers on Apple Silicon slow tool calls (see
caveats). The first run of a sweep also pulls the task images. Preflight's
oracle job takes a few minutes per sweep.

## 5. Outputs

```
jobs/<sweep>/
  configs/<condition>.yaml          resolved ExperimentConfig per condition (what the agent ran)
  tasks_meta.json                   difficulty, category, expert/junior minutes, timeouts, image per task
  preflight.json                    preflight checks, per-model test requests, oracle results
  sweep.json                        provenance: spec, source fingerprint (source_sha256), git commit +
                                    dirty flag, Python/harbor versions, uv.lock sha256, dataset, condition
                                    order + seed, provider tiers, date, every invocation of the sweep, and
                                    whether its jobs ran more than one code version (mixed_source)
  sweep_manifest.jsonl              one line per job event: job, round, condition, cmd, t_start, t_end,
                                    return_code, n_trials, n_errors; run lines add the tasks, harbor
                                    settings, source fingerprint and the trials to rerun (events: run,
                                    skipped_complete, moved_partial, retried, duplicate_trials, interrupted)
  provenance/                       source_<sha16>.json (per-file hashes of the code each fingerprint
                                    names) and <invocation>.patch (uncommitted source: `git apply` it on a
                                    checkout of the recorded commit to get the code that ran)
  .sweep.lock                       held while a sweep runs (pid, host, invocation)
  _oracle/                          the oracle feasibility job
  _partial/<job>.partial.<k>/       interrupted jobs that were moved aside and rerun (kept, not analyzed)
  _partial/<job>.retry.<k>/         finished jobs rerun because a trial did not measure its condition
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

- if `result.json` shows every trial finished and every trial measured its
  condition, the job is **skipped**;
- if the job dir exists but is incomplete (interrupted), it is **moved aside** to
  `_partial/<job>.partial.<k>` and **rerun** from scratch;
- if the job finished but a trial did not measure the condition, the whole
  job is moved aside to `_partial/<job>.retry.<k>` and **rerun**. That covers a
  billing error, a provider failure (a model call of any role that failed
  after all retries: an outage, a rejected key), an infrastructure error
  outside the agent (environment start, image pull or build), and an agent
  that crashed before the verifier ran (a harness bug). An agent timeout is
  an outcome, not a retry. The manifest's `retried` line lists each trial's
  reason.

The whole job is rerun, not just the failed trial, so every trial of a job
comes from one harbor run and one code version. The price is that a job's
valid trials are rerun too and a job is complete only when all its trials
are clean in one run: with a 10% per-trial failure rate a 10-trial job takes
about 2.9 runs on average. Provider failures are the likely cause (in the
smoke run, one of two `manager_arbiter_hard` trials lost two Gemma calls to
a 7-9 minute outage), so watch `n_provider_error_trials` on the manifest's
`run` lines.

Nothing is deleted, and every move is recorded in the manifest. `--dry-run`
shows which jobs would be skipped, moved aside or retried, and why.

Other things that stop a sweep:

- **Billing error.** After every job the sweep checks each episode for
  `exit_reason == "billing_error"` and stops (exit code 3) if it finds one.
  Top up the account and re-run: the job is rerun.
- **Provider outage (fail fast).** A model call gives up after at most
  ~2 minutes of retries (`retry_budget_sec`), and the first such failure, in
  any role, ends its episode at once (the trial is invalid anyway). The
  sweep prints each trial's outcome as it finishes; when a trial hits a
  provider failure it sends one tiny request to every model, and if any
  still fails it stops the running job and exits with code 4, naming the
  model and the error, typically within a few minutes of the outage. Re-run
  later (after fixing the key, if a 401/403 caused it): the job is rerun. If
  the models answer again, the failure was transient: the sweep goes on,
  that job fails (exit code 1 at the end), and the next run retries it.
  The sweep needs the network throughout: harbor looks the dataset up
  online at the start of every job, and a job harbor cannot start is also
  followed by the probe (no network: exit 4). Keep the machine online; on
  macOS `run_experiment.sh` runs the sweep under `caffeinate -i` so it does
  not idle-sleep (closing the lid still sleeps it; `SB_NO_CAFFEINATE=1`
  turns this off).
- **Repeated failures.** Three failed jobs in a row (Docker, harbor or
  provider trouble) stop the sweep.
- **Another sweep on the same jobs dir.** A sweep holds
  `<jobs_dir>/.sweep.lock` while it runs (so does `preflight --oracle`). A
  second invocation refuses to start (exit code 2) and names the holder (pid,
  host, start time), because it would move the first one's running job aside
  and both would write trials into one job. The lock goes away with the
  process, however it ends. A job dir that already holds duplicate trials is
  reported and not run: move it out of the jobs dir and re-run.
- **Changed condition config.** A resumed sweep refuses to run if a
  condition's resolved config changed since its first jobs ran, because that
  would mix two configs under one name. Use a new `jobs_dir` instead.
- **Changed task list or harbor settings.** A resume refuses (exit code 2)
  when the task list or the `harbor run` arguments (`n_concurrent`,
  `agent_timeout_multiplier`, `extra_args`, `env`) differ from those of the
  sweep's complete jobs, since conditions would be compared on different
  tasks or settings. `--allow-task-change` runs anyway; `tasks_meta.json`
  keeps the metadata of tasks dropped from the spec.
- **Changed code.** Every invocation, and every job just before it starts,
  fingerprints the harness source (`source_sha256`: a hash of
  `subagents_bench/`, `configs/`, `scripts/`, `pyproject.toml` and `uv.lock`,
  committed or not). A resume refuses (exit code 2) when the code that
  episodes run differs from what the complete jobs ran, naming the files
  that changed, and a sweep stops when that code changes under it mid-run.
  Edits to files no episode imports or reads (`subagents_bench/analysis/`,
  `subagents_bench/plots/`, `sweep.py`, `cli.py`, `configs/`, `scripts/`)
  only print a NOTE or WARNING, so you can work on the analysis in the same
  checkout while a sweep runs (condition configs are checked separately,
  above). `--allow-code-change` mixes versions anyway; `sweep.json` then
  sets `mixed_source`. When the source differs from the git commit, the
  sweep saves the difference as `<jobs_dir>/provenance/<invocation>.patch`,
  so the exact code can be restored (`git checkout <git_commit> && git apply
  <patch>`).

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
- **Arbiter isolation covers the workdir only.** With arbiter + hard scope
  each candidate edits a private copy of the task's workdir: paths naming
  the workdir are redirected into the copy, and stray writes to the real
  workdir are undone and recorded (`arbitrations.csv` `isolation_violated`,
  `leaked_files`; `conditions.csv` `isolation_violated_frac`). Changes
  outside the workdir (`/etc`, installed packages, services) are shared by
  all candidates and take effect at once.
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
| `single_matched` | Step-matched baseline: alone, 40 steps (= 5 orchestrator steps × 8 subagent `max_steps`; matches steps on the serial path, not FLOPs or tokens). |
| `gemma31b_parallel_{easy,medium,hard}` | Up to 3 Gemma-4-31B subagents run **concurrently** per `spawn_subagents` call. |
| `gemma31b_sequential_{easy,medium,hard}` | The same grants, but subagents run **one at a time**. |
| `gemma31b_arbiter_{easy,medium,hard}` | Each instruction goes to 3 candidate subagents; an LLM judge (the orchestrator model) picks a winner, with one confidence-aware debate round when the judge is not confident (≥ 0.8) or the candidates disagree. |
| `samemodel_parallel_hard` | Same-model control: parallel/hard, but the subagents run the orchestrator's own model. |
| `manager_{parallel,sequential,arbiter}_hard` | Manager orchestrator: it keeps only `read_file`, `list_dir`, `search` (`tools.orchestrator_tools`), so every command, edit and test must be delegated to Gemma-4-31B subagents (hard scope). Added because the orchestrator with all tools delegated in 0 of 18 subagent-condition episodes of the first smoke sweep. In the follow-up smoke run, `manager_arbiter_hard` delegated in both of its episodes (11 dispatches, 3 arbitrations); the valid one solved `prove-plus-comm`, the other lost Gemma calls to a provider outage and is invalid. |
| `qwen{0.8,2,4,9}b_parallel_hard` (`prime_size.yaml`) | Size sweep: Qwen3.5 subagents of 0.8B to 9B parameters on Prime, parallel/hard, with the same orchestrator. The sweep also runs `single_5step` and `gemma31b_parallel_hard` as anchors in the same rounds. |

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
| preflight: oracle FAIL on a task | The task does not pass with its reference solution on this machine (often an image or emulation problem); see `<jobs_dir>/_oracle/<task>__*/verifier/`. Drop the task from the spec or fix Docker resources, then re-run: a failed oracle job is rerun automatically (`preflight SPEC --redo-oracle` forces a rerun of a passing one). |
| sweep refuses: "resolved config ... differs" | You edited a config mid-sweep; use a new `jobs_dir`, or `--allow-config-change` if you really mean to mix. |
| sweep refuses: "another sweep holds .../.sweep.lock" | A sweep (or `preflight --oracle`) is still running on that jobs dir, e.g. in another terminal or tmux session: wait for it or stop it. |
| sweep refuses: "the harness source changed since ... complete job(s) ran" | Code that episodes run differs from what the finished jobs ran (the message lists the files). Revert the change, use a new `jobs_dir`, or pass `--allow-code-change` to mix versions. A change confined to analysis, plots, the launcher, scripts or configs is only a WARNING. |
| `analyze.sh`: completeness FAIL, "treatment not exercised" | A subagent condition never delegated (or an arbiter condition never arbitrated), so it measured the orchestrator alone. Expected for the all-tools smoke data; for the pilot, compare with the `manager_*` conditions and see `delegated_frac` in `conditions.csv`. |
| sweep refuses: "the task list differs" / "the harbor arguments differ" | The spec's tasks or `harbor` settings changed since jobs finished. Restore them, use a new `jobs_dir`, or pass `--allow-task-change`. |
| sweep exits 4: "provider failures in N/M trials" | The provider failed most model calls of a job (outage, or a 401/403 key problem: see the trials' `agent/calls.jsonl` errors). Fix or wait, then re-run; the job is rerun. |
| a job reports "duplicate trials" | Two harbor runs wrote into one job dir. Move the job dir out of `<jobs_dir>` (keep it for inspection) and re-run to rerun the job. |
| `harbor` not found | Run `scripts/setup.sh`; the sweep uses `.venv/bin/harbor`. |
