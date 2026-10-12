"""An offline, harbor-shaped sweep produced by the REAL runtime.

:func:`build_offline_sweep` runs real episodes -- :func:`subagents_bench.runner.run_task`
with the real orchestrator, subagents, arbiter, tools and recorder -- on
:class:`~subagents_bench.environment.LocalShellEnvironment` sandboxes, with
the scripted :class:`fakes.FakeClient` standing in for the providers. It lays
the output out exactly as a real sweep does, using the sweep launcher's own
code for everything except launching harbor:

* the conditions are the pilot spec's own overrides
  (``configs/sweeps/pilot.yaml`` on top of ``configs/sail/base.yaml``), with
  only the sandbox switched to ``local_shell``;
* ``<jobs_dir>/configs/<cond>.yaml``, ``tasks_meta.json``,
  ``sweep_manifest.jsonl`` and ``sweep.json`` are written by
  :mod:`subagents_bench.sweep` (``materialize_configs``,
  ``write_tasks_meta``, ``append_manifest``, ``write_sweep_json``);
* each job is ``<jobs_dir>/<sweep>__r<k>__<cond>/`` with harbor 0.24.0's
  ``result.json``/``config.json`` shapes, and each trial is
  ``<task>__<suffix>/`` holding ``agent/`` (the episode dir) and a
  ``result.json`` whose reward comes from a verifier that inspects the
  sandbox's final state; phase timestamps are the real instants of this run.

The two tasks and the scripted model behaviour are test fixtures: nothing here
is a measurement of a real model or a real Terminal-Bench task.

Used by ``tests/test_pipeline_offline.py`` and ``scripts/gen_data_docs.py``.
"""

from __future__ import annotations

import json
import random
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from fakes import FakeFactory, Reply, tc  # noqa: E402

from subagents_bench import sweep as sw  # noqa: E402
from subagents_bench.config import ExperimentConfig  # noqa: E402
from subagents_bench.environment import LocalShellEnvironment  # noqa: E402
from subagents_bench.llm import resolve_model_config  # noqa: E402
from subagents_bench.runner import run_task  # noqa: E402

REPO_ROOT = TESTS_DIR.parent
PILOT_SPEC = REPO_ROOT / "configs" / "sweeps" / "pilot.yaml"
SWEEP_NAME = "offline"
DATASET = "offline-fixture@0"

# Pilot conditions exercised offline: a baseline and every strategy, with
# copy isolation (arbiter hard) and a debate round (arbiter easy).
CONDITIONS = (
    "single_5step",
    "gemma31b_parallel_hard",
    "gemma31b_sequential_medium",
    "gemma31b_arbiter_hard",
    "gemma31b_arbiter_easy",
)

# Scripted per-call latency (seconds) of the fake providers. Large enough that
# LLM time dominates the harness's own local-shell round-trips.
ORCH_LATENCY = 0.12
SUB_LATENCY = 0.05


@dataclass
class OfflineTask:
    """A tiny task: initial files, an instruction and a verifier."""

    name: str
    files: dict[str, str]
    instruction: str
    target: str  # file the verifier checks (relative to the workdir)
    expected: str  # its required content
    source: str  # file an investigator reads
    search: str  # pattern an investigator searches for
    edit: dict[str, str]  # edit_file arguments that solve it
    shell_fix: str  # bash command that solves it
    shell_wrong: str  # bash command that gets it wrong
    meta: dict[str, Any]  # tasks_meta.json entry (fixture values)

    def verify(self, workdir: Path) -> float:
        p = workdir / self.target
        return 1.0 if p.is_file() and p.read_text() == self.expected else 0.0


TASKS: dict[str, OfflineTask] = {
    t.name: t
    for t in (
        OfflineTask(
            name="offline-fix-config",
            files={"app/config.txt": "mode=broken\nlevel=3\n", "app/README.md": "The service needs mode=fixed.\n"},
            instruction="In app/config.txt change the line mode=broken to mode=fixed; keep every other line.",
            target="app/config.txt",
            expected="mode=fixed\nlevel=3\n",
            source="app/config.txt",
            search="mode=",
            edit={"path": "app/config.txt", "old": "mode=broken", "new": "mode=fixed"},
            shell_fix="printf 'mode=fixed\\nlevel=3\\n' > app/config.txt && cat app/config.txt",
            shell_wrong="printf 'mode=fixed\\n' > app/config.txt",
            meta={"difficulty": "easy", "category": "software-engineering", "expert_time_min": 2.0,
                  "junior_time_min": 5.0, "agent_timeout_sec": 900.0, "verifier_timeout_sec": 60.0,
                  "docker_image": None, "tags": ["fixture"]},
        ),
        OfflineTask(
            name="offline-sum",
            files={"data/numbers.txt": "1\n2\n3\n4\n5\n"},
            instruction="Write the sum of the integers in data/numbers.txt to answer.txt (only the number).",
            target="answer.txt",
            expected="15\n",
            source="data/numbers.txt",
            search="[0-9]",
            edit={"path": "answer.txt", "old": "", "new": "15\n"},
            shell_fix="awk '{s+=$1} END {print s}' data/numbers.txt > answer.txt && cat answer.txt",
            shell_wrong="echo 14 > answer.txt",
            meta={"difficulty": "medium", "category": "data-processing", "expert_time_min": 3.0,
                  "junior_time_min": 10.0, "agent_timeout_sec": 900.0, "verifier_timeout_sec": 60.0,
                  "docker_image": None, "tags": ["fixture"]},
        ),
    )
}


# ---------------------------------------------------------------------------
# scripted models
# ---------------------------------------------------------------------------


def _n_assistant(messages: list) -> int:
    return sum(1 for m in messages if m.role == "assistant")


def orchestrator_script(cfg: ExperimentConfig, task: OfflineTask):
    """The orchestrator (and, for arbiter conditions, the judge: same model).

    single: inspect, edit, check, finish. subagents: inspect, spawn two
    instructions, (parallel: spawn a verifier too), fix it itself if the
    subagents could only read, check, finish. Calls after the first report
    cached prompt tokens and every call reasons (as DeepSeek does)."""
    subagents = cfg.topology.value == "subagents"
    strategy = cfg.subagent.strategy.value
    scope = cfg.subagent.scope.value
    judgements = (
        [{"winner": 0, "confidence": 0.55, "agreement": False, "rationale": "the reports differ in detail"},
         {"winner": 2, "confidence": 0.9, "agreement": True, "rationale": "after revision they agree"}]
        if scope == "easy" else
        [{"winner": 1, "confidence": 0.92, "agreement": True, "rationale": "all candidates made the same fix"}]
    )

    def reply(text: str, calls: list, step: int, **kw: Any) -> Reply:
        return Reply(text, calls, reasoning=f"step {step}: deciding the next action for {task.name}.",
                     reasoning_tokens=24 + 4 * step, cached_tokens=0 if step == 1 else 256, **kw)

    def script(system, messages, tools, client):
        names = [t.name for t in tools]
        if "submit_judgement" in names:
            j = judgements.pop(0) if len(judgements) > 1 else judgements[0]
            return Reply(f"Candidate {j['winner']} is best.", [tc("submit_judgement", **j)],
                         reasoning="comparing candidates", reasoning_tokens=30)
        step = _n_assistant(messages) + 1
        if step == 1:
            # One retried attempt (a scripted 429) on the first call of every episode.
            return reply("Inspecting the working directory.", [tc("bash", command="ls -la && find . -type f | sort")],
                         step, failed_attempts=1)
        if not subagents:
            if step == 2:
                return reply("Applying the fix.", [tc("edit_file", **task.edit)], step)
            if step == 3:
                return reply("Checking the result.", [tc("read_file", path=task.target)], step)
            return reply("Done.", [tc("finish", summary=f"{task.target} is correct.")], step)
        instr = [
            f"[{task.name}] {task.instruction} Report exactly what you changed.",
            f"[{task.name}] Inspect {task.source} and report its contents and anything unusual.",
        ]
        if step == 2:
            return reply("Delegating.", [tc("spawn_subagents", instructions=instr)], step)
        if step == 3 and strategy == "parallel":
            return reply("Asking for verification.", [tc("spawn_subagents", instructions=[
                f"[{task.name}] Verify that {task.target} is correct and report its exact contents."])], step)
        if step <= 3 or (step == 4 and strategy == "parallel"):
            calls = [tc("bash", command=task.shell_fix)] if scope == "easy" else []
            calls.append(tc("search", pattern=task.search, path="."))
            return reply("Checking the result.", calls, step)
        return reply("Done.", [tc("finish", summary="The subagents' work is verified.")], step)

    return script


def subagent_script(task_for: dict[str, OfflineTask], *, wrong: bool = False):
    """Subagents by scope: hard edits, medium runs a command, easy reads.
    A debate call (no tools) returns a revised report."""

    def script(system, messages, tools, client):
        names = [t.name for t in tools]
        if not names:
            return Reply("Revised report: having read the other reports, my result stands.")
        first = messages[0].content or ""
        task = next((t for n, t in task_for.items() if f"[{n}]" in first), next(iter(task_for.values())))
        turn = _n_assistant(messages) + 1
        investigate = "Inspect" in first or "Verify" in first
        if "edit_file" in names and not investigate:
            if turn == 1:
                return Reply("Editing the file.", [tc("edit_file", **task.edit)])
            return Reply("", [tc("submit", report=f"Edited {task.edit['path']} as instructed.")])
        if "bash" in names and not investigate:
            if turn == 1:
                return Reply("Running the fix.", [tc("bash", command=task.shell_wrong if wrong else task.shell_fix)])
            return Reply("", [tc("submit", report=f"Ran the command; {task.target} written.")])
        if turn == 1:
            return Reply("Looking around.", [tc("list_dir", path=".", depth=2), tc("search", pattern=task.search, path=".")])
        if turn == 2:
            return Reply("", [tc("read_file", path=task.source)])
        return Reply("", [tc("submit", report=f"{task.source} looks as described; nothing unusual.")])

    return script


class ResolvingFactory(FakeFactory):
    """Gives each fake client the provider-resolved model config, exactly as
    :func:`subagents_bench.llm.make_client` does (base_url, api_key_env)."""

    def __call__(self, cfg: Any) -> Any:
        return super().__call__(resolve_model_config(cfg))


def make_factory(cfg: ExperimentConfig, task: OfflineTask, *, wrong: bool = False) -> FakeFactory:
    orch = cfg.orchestrator.model
    sub = cfg.resolved_subagent_model().model
    scripts: dict[str, Any] = {orch: orchestrator_script(cfg, task)}
    lat: dict[str, float] = {orch: ORCH_LATENCY}
    if sub != orch:
        scripts[sub] = subagent_script(TASKS, wrong=wrong)
        lat[sub] = SUB_LATENCY
    return ResolvingFactory(scripts, latency=lat)


# ---------------------------------------------------------------------------
# harbor-shaped output
# ---------------------------------------------------------------------------


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _suffix(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(7))


def trial_result(task: str, trial: str, job_dir: Path, reward: float, phases: dict[str, tuple[float, float]]) -> dict:
    """A trial ``result.json`` in harbor 0.24.0's shape (fields the analysis reads)."""
    out: dict[str, Any] = {
        "id": f"offline-{trial}",
        "task_name": task,
        "trial_name": trial,
        "trial_uri": job_dir.joinpath(trial).as_uri(),
        "source": "offline-fixture",
        "config": {"task": {"path": task, "name": None}, "trial_name": trial, "trials_dir": str(job_dir),
                   "agent": {"name": None, "import_path": sw.AGENT_IMPORT_PATH}},
        "agent_info": {"name": "subagents-orchestrator", "version": "offline", "model_info": None},
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": None,
        "started_at": _iso(phases["environment_setup"][0]),
        "finished_at": _iso(phases["verifier"][1]),
    }
    for name, (t0, t1) in phases.items():
        out[name] = {"started_at": _iso(t0), "finished_at": _iso(t1)}
    return out


def job_result(n_trials: int, t0: float, t1: float, rewards: dict[str, float]) -> dict:
    """A job ``result.json`` in harbor 0.24.0's shape (what sweep.job_status reads)."""
    by_reward: dict[str, list[str]] = {}
    for trial, r in rewards.items():
        by_reward.setdefault(str(r), []).append(trial)
    return {
        "id": f"offline-job-{int(t0)}",
        "started_at": _iso(t0)[:-1],
        "updated_at": _iso(t1)[:-1],
        "finished_at": _iso(t1)[:-1],
        "n_total_trials": n_trials,
        "stats": {
            "n_completed_trials": n_trials, "n_errored_trials": 0, "n_running_trials": 0, "n_pending_trials": 0,
            "n_cancelled_trials": 0, "n_retries": 0,
            "evals": {"offline": {"n_trials": n_trials, "n_errors": 0, "reward_stats": {"reward": by_reward}}},
        },
    }


@dataclass
class OfflineSweep:
    jobs_dir: Path
    spec: Any  # subagents_bench.sweep.SweepSpec
    trials: list[dict[str, Any]] = field(default_factory=list)  # per trial: job, condition, round, task, dirs, reward


def offline_spec(root: Path, rounds: int, conditions: tuple[str, ...] = CONDITIONS) -> Path:
    """Write a sweep spec whose conditions are the pilot spec's own overrides
    (on the pilot's base config), with the sandbox switched to local_shell."""
    pilot, _ = sw.load_spec_dict(PILOT_SPEC, REPO_ROOT)
    conds = {}
    for c in conditions:
        conds[c] = sw.deep_merge(pilot["conditions"][c] or {}, {"sandbox": {"kind": "local_shell", "workdir": "auto"}})
    spec = {
        "name": SWEEP_NAME,
        "description": "offline fixture: real runtime + scripted models on local-shell sandboxes",
        "dataset": DATASET,
        "tasks": list(TASKS),
        "rounds": rounds,
        "order_seed": 0,
        "base": str(REPO_ROOT / pilot["base"]),
        "conditions": conds,
        "harbor": {"env": "local_shell", "n_concurrent": len(TASKS)},
        "jobs_dir": str(root / "jobs" / SWEEP_NAME),
    }
    import yaml  # noqa: PLC0415

    path = root / "offline_sweep.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def _run_trial(root: Path, j: Any, cfg: ExperimentConfig, task: OfflineTask, trial: str) -> dict[str, Any]:
    """One harbor-shaped trial: sandbox setup, the real episode, the verifier, result.json."""
    trial_dir = j.job_dir / trial
    trial_dir.mkdir()
    (trial_dir / "config.json").write_text(json.dumps({"task": {"path": task.name}, "trial_name": trial}))
    t_env0 = time.time()
    work = root / "work" / j.job / trial
    for rel, text in task.files.items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(text)
    env = LocalShellEnvironment(str(work))
    t_env1 = time.time()
    wrong = cfg.name == "gemma31b_sequential_medium" and task.name == "offline-sum" and j.round == 1
    try:
        res = run_task(cfg, task.name, task.instruction, env, client_factory=make_factory(cfg, task, wrong=wrong),
                       episode_dir=trial_dir / "agent", trial_name=trial)
    finally:
        env.close()
    t_agent1 = time.time()
    reward = task.verify(work)
    t_ver1 = time.time()
    phases = {
        "environment_setup": (t_env0, t_env1),
        "agent_setup": (t_env1, t_env1),
        "agent_execution": (t_env1, t_agent1),
        "verifier": (t_agent1, t_ver1),
    }
    (trial_dir / "result.json").write_text(json.dumps(trial_result(task.name, trial, j.job_dir, reward, phases), indent=2))
    return {"job": j.job, "condition": j.condition, "round": j.round, "task": task.name, "trial_dir": trial_dir,
            "workdir": work, "reward": reward, "exit_reason": res.exit_reason, "episode_id": res.episode_id}


def build_offline_sweep(root: str | Path, *, rounds: int = 2, seed: int = 0) -> OfflineSweep:
    """Run every (round, condition, task) episode and write a harbor-shaped
    jobs dir under ``<root>/jobs/offline``. One condition/task/round cell
    (sequential medium, offline-sum, round 1) has its subagent run a wrong
    command, so not every trial is solved."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = sw.load_spec(offline_spec(root, rounds), root=REPO_ROOT)
    resolved = sw.resolve_conditions(spec)
    full = {}
    for c, cfg in resolved.items():
        problems, f = sw.validate_condition(cfg)
        if problems or f is None:
            raise AssertionError(f"condition {c} does not load: {problems}")
        full[c] = f
    spec.jobs_dir.mkdir(parents=True, exist_ok=True)
    config_paths = sw.materialize_configs(spec, full)
    tasks = list(TASKS)
    sw.write_tasks_meta(spec, {"dataset": DATASET, "source_dir": "tests/offline_pipeline.py",
                               "generated_at": _iso(time.time()),
                               "tasks": {n: t.meta for n, t in TASKS.items()}}, tasks)
    prov = sw.provenance(REPO_ROOT, None, with_docker=False)
    sw.write_sweep_json(spec, tasks=tasks, config_paths=config_paths, resolved=full, prov=prov, status="running",
                        harbor=None)
    rng = random.Random(seed)
    out = OfflineSweep(jobs_dir=spec.jobs_dir, spec=spec)
    manifest = spec.jobs_dir / sw.MANIFEST_NAME
    for j in sw.plan_jobs(spec, tasks):
        cfg = ExperimentConfig.from_yaml(j.config_path)
        j.job_dir.mkdir(parents=True)
        (j.job_dir / "config.json").write_text(json.dumps({"job_name": j.job,
                                                            "n_concurrent_trials": spec.harbor.n_concurrent}))
        t_job = time.time()
        rewards: dict[str, float] = {}
        # Trials of a job run concurrently in this process, as harbor runs them (-n).
        names = {t: f"{t}__{_suffix(rng)}" for t in tasks}
        with ThreadPoolExecutor(max_workers=spec.harbor.n_concurrent) as pool:
            futures = [pool.submit(_run_trial, root, j, cfg, TASKS[t], names[t]) for t in tasks]
            results = [f.result() for f in futures]
        for r in results:
            rewards[r["trial_dir"].name] = r["reward"]
            out.trials.append(r)
        t_job1 = time.time()
        (j.job_dir / "result.json").write_text(json.dumps(job_result(len(tasks), t_job, t_job1, rewards), indent=2))
        st = sw.job_status(j.job_dir)
        sw.append_manifest(manifest, sw._manifest_record(  # noqa: SLF001 - the launcher's own record shape
            spec, j, "run", t_start=t_job, t_end=t_job1, duration_sec=t_job1 - t_job, return_code=0,
            n_trials=st.n_total, n_completed=st.n_completed, n_errors=st.n_errored, complete=st.complete,
            status_reason=st.reason, n_episode_json=len(list(j.job_dir.glob("*/agent/episode.json"))),
            billing_error_trials=sw.scan_billing_errors(j.job_dir), git_commit=prov.get("git_commit"),
        ))
    sw.write_sweep_json(spec, tasks=tasks, config_paths=config_paths, resolved=full, prov=prov, status="complete",
                        harbor=None)
    return out


if __name__ == "__main__":  # python tests/offline_pipeline.py OUT_DIR
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "offline-sweep")
    sweep = build_offline_sweep(target)
    print(f"wrote {len(sweep.trials)} trials under {sweep.jobs_dir}")
