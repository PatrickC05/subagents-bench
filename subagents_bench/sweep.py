"""Sweep launcher: one ``harbor run`` job per (round, condition) of a sweep spec, resumable.

A spec (``configs/sweeps/*.yaml``) names a base config, the dataset (a harbor registry name, or
``dataset_path``: a local directory of task dirs, passed as ``harbor run -p``, with ``platform``
to pin its Dockerfiles' ``FROM`` lines in a copy under the jobs dir) and tasks, the rounds, the
harbor settings (``n_concurrent``, ``agent_timeout_multiplier``, ``extra_args`` for every harbor
run), sweep-wide ``overrides`` (deep-merged onto the base before the conditions, e.g. the step
budget) and the conditions (overrides deep-merged onto that); ``extends: <spec>`` merges it over
another spec. Relative paths resolve against the repository root (the spec's nearest ancestor
holding a pyproject.toml). Exit codes: 0 done, 1 jobs failed or need a retry, 2 spec or config
error, 3 billing error, 4 provider outage, 130 interrupted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .config import ConfigError, ExperimentConfig, ModelConfig, from_dict
from .hints import (HINTS_DIR_ENV, HintError, fetch_source, generate_questions, generate_reports, judge_answer,
                    load_questions, load_reports, load_subtasks, n_slots, schedule, write_questions, write_reports)
from .llm import BillingError, Message, SailClient
from .records import git_commit

AGENT = "subagents_bench.harness.harbor_agent:OrchestratorAgent"
SPEC_KEYS = {"extends", "name", "base", "dataset", "dataset_path", "platform", "tasks", "rounds", "order_seed",
             "harbor", "overrides", "conditions", "jobs_dir"}
HARBOR_KEYS = {"n_concurrent", "agent_timeout_multiplier", "extra_args"}
FROM_RE = re.compile(r"^FROM (?!--platform)", re.M)  # a Dockerfile FROM without a platform (instructions in upper case)
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")  # job names are <sweep>__r<k>__<condition>
POLL_SEC = 10.0  # live trial lines
STOP_GRACE_SEC = 120.0  # on SIGINT harbor cancels its trials and removes their containers
SESSION_GRACE_SEC = 120.0  # an interactive session's shutdown and kill, before harbor's agent timeout
STATUS = {0: "complete", 1: "incomplete", 3: "aborted: billing error", 4: "aborted: provider outage",
          130: "interrupted"}


def say(line: str) -> None:
    """Print at once: the sweep's output usually goes to a log file."""
    print(line, flush=True)


@dataclass
class Spec:
    name: str
    root: Path
    dataset: str | None  # harbor registry name@version (``-d``), or None with a dataset_path
    dataset_path: Path | None  # a local directory of task dirs (``-p``)
    platform: str | None  # pin the tasks' Dockerfiles to this platform, in a copy (stage_tasks)
    tasks: list[str]
    rounds: int
    order_seed: int
    n_concurrent: int
    timeout_multiplier: float
    extra_args: list[str]  # appended to every harbor run (the oracle's too)
    jobs_dir: Path
    configs: dict[str, ExperimentConfig]  # condition -> resolved config
    raw: dict[str, Any]  # the spec after `extends`


def deep_merge(base: Any, over: Any) -> Any:
    """Mappings merge recursively; any other value (list, scalar, null) replaces."""
    if isinstance(base, dict) and isinstance(over, dict):
        return {**base, **{k: deep_merge(base.get(k), v) for k, v in over.items()}}
    return over


def _yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    return data


def _read_spec(path: Path, root: Path) -> dict[str, Any]:
    raw = _yaml(path)
    parent = raw.pop("extends", None)
    return deep_merge(_read_spec(root / parent, root), raw) if parent else raw


def load_spec(path: str | os.PathLike) -> Spec:
    """Read and check a sweep spec; every condition must load as an ExperimentConfig."""
    path = Path(path).resolve()
    root = next((d for d in path.parents if (d / "pyproject.toml").is_file()), Path.cwd())
    raw = _read_spec(path, root)
    harbor = raw.get("harbor") or {}
    unknown = sorted(set(raw) - SPEC_KEYS) + sorted(f"harbor.{k}" for k in set(harbor) - HARBOR_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s): {', '.join(unknown)}")
    try:
        name, conditions, tasks, rounds = raw["name"], raw["conditions"], raw["tasks"], raw["rounds"]
        base = deep_merge(_yaml(root / raw["base"]), raw.get("overrides") or {})
        dataset, dataset_path, platform = raw.get("dataset"), raw.get("dataset_path"), raw.get("platform")
        extra_args = harbor.get("extra_args") or []
        spec = Spec(name, root, dataset, root / dataset_path if dataset_path else None, platform, tasks, rounds,
                    raw["order_seed"], harbor["n_concurrent"], harbor["agent_timeout_multiplier"], extra_args,
                    root / raw.get("jobs_dir", f"jobs/{name}"), {}, raw)
    except KeyError as e:
        raise ConfigError(f"{path}: missing key {e}") from None
    if not (isinstance(tasks, list) and tasks and isinstance(conditions, dict) and conditions
            and isinstance(rounds, int) and rounds >= 1):
        raise ConfigError(f"{path}: tasks and conditions must be non-empty, rounds an integer >= 1")
    if bool(dataset) == bool(dataset_path):
        raise ConfigError(f"{path}: set exactly one of dataset (a harbor registry name) and dataset_path (a directory)")
    if platform is not None and not (isinstance(platform, str) and platform and dataset_path):
        raise ConfigError(f"{path}: platform must be a string like linux/amd64 and needs dataset_path (images built here)")
    if not (isinstance(extra_args, list) and all(isinstance(a, str) for a in extra_args)):
        raise ConfigError(f"{path}: harbor.extra_args must be a list of strings")
    bad = [n for n in (name, *conditions) if not NAME_RE.fullmatch(str(n)) or "__" in str(n)]
    if bad:
        raise ConfigError(f"{path}: invalid name(s) {bad}: use letters, digits, '_', '.', '-' but not '__'")
    for cond, over in conditions.items():
        try:
            spec.configs[cond] = from_dict({**deep_merge(base, over or {}), "name": cond})
        except ConfigError as e:
            raise ConfigError(f"{path}: condition {cond}: {e}") from None
    if interactive(spec) and not dataset_path:
        raise ConfigError(f"{path}: interactive conditions take their hints from SWE-sweep task dirs: set dataset_path")
    return spec


def interactive(spec: Spec) -> dict[str, ExperimentConfig]:
    """The conditions whose episodes are interactive sessions."""
    return {c: cfg for c, cfg in spec.configs.items() if cfg.interactive is not None}


def hints_dir(spec: Spec) -> Path:
    """Generated bug reports for ``hint_detail: report`` (``subagents-bench hints``)."""
    return spec.jobs_dir / "hints"


def check_interactive(spec: Spec) -> list[str]:
    """Problems that would fail or cut short an interactive condition's trials: a task without
    SWE-sweep subtasks or (``report``) without a generated report per scheduled subtask, and a
    harbor agent timeout (the task's ``[agent] timeout_sec`` x the multiplier) that would cancel
    the session before its own deadline."""
    conds = interactive(spec)
    if not conds:
        return []
    problems = []
    longest = max(cfg.interactive.wall_clock_sec for cfg in conds.values())
    for t in spec.tasks:
        task = spec.dataset_path / t
        try:
            load_subtasks(task)
            for c, cfg in conds.items():
                if cfg.interactive.hint_detail == "report" or cfg.interactive.replies:
                    schedule(cfg.interactive, task, t, hints_dir(spec))
        except HintError as e:
            problems.append(f"{t}: {e}")
            continue
        try:
            timeout = tomllib.loads((task / "task.toml").read_text())["agent"]["timeout_sec"] * spec.timeout_multiplier
        except (OSError, ValueError, KeyError, TypeError):
            continue  # harbor applies its own default; preflight's dataset check reports a missing task.toml
        if timeout < longest + SESSION_GRACE_SEC:
            problems.append(f"{t}: harbor's agent timeout {timeout:.0f}s ({spec.timeout_multiplier} x the task's) "
                            f"would cancel a {longest:.0f}s session: raise harbor.agent_timeout_multiplier")
    return problems


def questions_needed(spec: Spec) -> int:
    """The most questions one session of the sweep asks (every round draws from the same pool)."""
    n = 0
    for cfg in interactive(spec).values():
        i = cfg.interactive
        if i.replies and i.questions_per_interval:
            bugs = min(n_slots(i), i.max_hints or n_slots(i))
            n = max(n, bugs * i.questions_per_interval)
    return n


def make_questions(spec: Spec, per_task: int | None = None, force: bool = False) -> int:
    """Write questions about each task's source (``questions_per_interval``) with the orchestrator's
    model of the first interactive condition, from a clone of the task's repository at its ref in
    ``<jobs_dir>/src``. Existing questions are kept unless ``force``. 0 if every task has enough."""
    need = per_task or questions_needed(spec)
    if not need:
        say(f"sweep {spec.name!r} asks no questions (interactive.questions_per_interval): nothing to generate")
        return 0
    model = next(iter(interactive(spec).values())).orchestrator
    client, failed = SailClient(model), 0
    try:
        for t in spec.tasks:
            try:
                existing = [] if force else load_questions(hints_dir(spec), t)
            except HintError:
                existing = []
            try:
                src = fetch_source(spec.dataset_path / t, spec.jobs_dir / "src" / t)
                qs = generate_questions(spec.dataset_path / t, src, client, need, existing)
            except BillingError as e:  # every later task would fail the same way
                say(f"  [FAIL] {t}: {type(e).__name__}: {str(e)[:300]}")
                return 3
            except Exception as e:  # one task's failure does not lose the others'
                say(f"  [FAIL] {t}: {type(e).__name__}: {str(e)[:300]}")
                failed += 1
                continue
            path = write_questions(hints_dir(spec), t, model.model, qs)
            new = len({q.id for q in qs} - {q.id for q in existing})
            say(f"  [OK] {t}: {len(qs)} questions ({new} new) -> {_rel(path, spec.root)}")
    finally:
        client.close()
    return 1 if failed else 0


def grade_answers(spec: Spec, force: bool = False) -> int:
    """Judge the first reply to every question of every finished trial of the sweep against its
    reference answer, with the orchestrator's model of the first interactive condition; writes
    ``<trial>/agent/answers.json``. 0 when every answer got a verdict, else 1."""
    conds = interactive(spec)
    if not conds:
        say(f"sweep {spec.name!r} has no interactive condition: nothing to grade")
        return 0
    model = next(iter(conds.values())).orchestrator
    client, unread = SailClient(model), 0
    try:
        for trial in sorted(spec.jobs_dir.glob("*/*/agent")):
            if trial.parent.parent.name.startswith(("_", ".")) or not (trial.parent / "result.json").is_file():
                continue
            out = trial / "answers.json"
            if out.is_file() and not force:
                continue
            inbox = [json.loads(x) for x in (trial / "inbox.jsonl").read_text().splitlines() if x.strip()] \
                if (trial / "inbox.jsonl").is_file() else []
            questions = {r["msg_id"]: r for r in inbox if r.get("kind") == "question"}
            if not questions:
                continue
            replies = [json.loads(x) for x in (trial / "replies.jsonl").read_text().splitlines() if x.strip()]
            first: dict[int, str] = {}
            for r in sorted(replies, key=lambda r: r["t"]):
                first.setdefault(r["to"], r["text"])
            answers = {}
            for mid, q in questions.items():
                if mid not in first:
                    continue
                correct, why = judge_answer(client, q["question"], q["reference"], first[mid])
                unread += correct is None
                answers[str(mid)] = {"correct": correct, "why": why}
            out.write_text(json.dumps({"judge_model": model.model, "graded_at": time.time(),
                                       "answers": answers}, indent=2) + "\n")
            n_ok = sum(1 for a in answers.values() if a["correct"])
            say(f"  [OK] {_rel(trial.parent, spec.root)}: {n_ok} of {len(questions)} questions "
                f"answered correctly ({len(answers)} answered)")
    finally:
        client.close()
    return 1 if unread else 0


def make_hints(spec: Spec, force: bool = False) -> int:
    """Write a bug report per subtask of every task (``hint_detail: report``) with the
    orchestrator's model of the first interactive condition; existing reports are kept unless
    ``force``. 0 if every task has its reports, else 1."""
    conds = interactive(spec)
    if not conds:
        say(f"sweep {spec.name!r} has no interactive condition: nothing to generate")
        return 0
    model = next(iter(conds.values())).orchestrator
    client, failed = SailClient(model), 0
    try:
        for t in spec.tasks:
            try:
                existing = {} if force else load_reports(hints_dir(spec), t)
            except HintError:
                existing = {}
            try:
                reports = generate_reports(spec.dataset_path / t, client, existing)
            except Exception as e:  # one task's failure does not lose the others'
                say(f"  [FAIL] {t}: {type(e).__name__}: {str(e)[:300]}")
                failed += 1
                continue
            path = write_reports(hints_dir(spec), t, model.model, reports)
            say(f"  [OK] {t}: {len(reports)} reports ({len(reports) - len(existing)} new) -> {_rel(path, spec.root)}")
    finally:
        client.close()
    return 1 if failed else 0


def plan(spec: Spec, only: list[str] | None = None, rounds: list[int] | None = None) -> list[tuple[int, str]]:
    """(round, condition) jobs in run order; each round shuffles the conditions with order_seed."""
    bad = [*sorted(set(only or ()) - set(spec.configs)), *(f"round {r}" for r in rounds or () if not 1 <= r <= spec.rounds)]
    if bad:
        raise ConfigError(f"not in sweep {spec.name!r}: {', '.join(bad)}")
    rng = random.Random(spec.order_seed)
    jobs = []
    for r in range(1, spec.rounds + 1):
        order = list(spec.configs)
        rng.shuffle(order)  # every round draws, so an order never depends on --rounds/--only
        jobs += [(r, c) for c in order if (not rounds or r in rounds) and (not only or c in only)]
    return jobs


def job_name(spec: Spec, round_: int, cond: str) -> str:
    return f"{spec.name}__r{round_}__{cond}"


def harbor_bin() -> str | None:
    """The ``harbor`` next to this Python (the venv's, pinned in uv.lock), else one on PATH."""
    return shutil.which("harbor", path=str(Path(sys.executable).parent)) or shutil.which("harbor")


def _rel(p: Path, root: Path) -> str:
    return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)


def task_root(spec: Spec) -> Path | None:
    """The task dirs harbor runs: ``dataset_path``, or its pinned copy when ``platform`` is set."""
    return spec.jobs_dir / "tasks" if spec.platform else spec.dataset_path


def harbor_cmd(spec: Spec, job: str, agent: str = AGENT) -> list[str]:
    source = ["-p", _rel(task_root(spec), spec.root)] if spec.dataset_path else ["-d", spec.dataset]
    cmd = [harbor_bin() or "harbor", "run", *source]
    for t in spec.tasks:
        cmd += ["-i", t]
    cmd += ["-a", agent, "-e", "docker", "-n", str(spec.n_concurrent), "-k", "1",
            "-o", _rel(spec.jobs_dir, spec.root), "--job-name", job]
    if agent == AGENT:
        cmd += ["--agent-timeout-multiplier", str(spec.timeout_multiplier)]
    return cmd + spec.extra_args + ["-y", "-q"]


def pin_platform(dockerfile: str, platform: str) -> str:
    """``FROM --platform=<platform> ...`` on every FROM that has no platform (idempotent)."""
    return FROM_RE.sub(f"FROM --platform={platform} ", dockerfile)


def _tree_hash(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        h.update(f"{p.relative_to(d)}\0{p.stat().st_size}\0".encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def n_subtasks(task: Path) -> int | None:
    """A SWE-sweep task's bugs: the subtasks of ``tests/swesweep_eval.json``, all of which its
    grader grades (it reports none when the agent's patch breaks the build); else None."""
    subtasks = _json(task / "tests" / "swesweep_eval.json").get("subtasks")
    return len(subtasks) if isinstance(subtasks, list) else None


def stage_tasks(spec: Spec) -> None:
    """For a local dataset, write ``<jobs_dir>/tasks.json``: source dir, its git commit and
    dirtiness, and per task a hash of the dir harbor runs and its bugs (:func:`n_subtasks`, the
    analysis's denominator). With ``platform`` set, first copy the tasks to ``<jobs_dir>/tasks/``
    with every ``FROM`` of their ``environment/Dockerfile`` and ``tests/Dockerfile`` pinned to it.
    A rerun rewrites identical files. Pinning in the Dockerfiles is the one way that reaches
    harbor's separate verifier image: a compose overlay does not, and a global
    DOCKER_DEFAULT_PLATFORM breaks harbor's egress-control probe (``docker run`` of a multi-arch
    digest fails when the other variant is in the image store), which rejects the tasks'
    ``no-network`` phases."""
    if not spec.dataset_path:
        return
    dest, src = task_root(spec), spec.dataset_path
    commit = _output(["git", "-C", str(src), "rev-parse", "HEAD"])
    manifest = {"source": str(src), "git_commit": commit, "platform": spec.platform, "tasks": {}, "n_subtasks": {},
                "git_dirty": bool(_output(["git", "-C", str(src), "status", "--porcelain", "."])) if commit else None}
    for t in spec.tasks:
        if spec.platform:
            if (dest / t).exists():
                shutil.rmtree(dest / t)
            shutil.copytree(src / t, dest / t)
            for df in (dest / t / "environment" / "Dockerfile", dest / t / "tests" / "Dockerfile"):
                if df.is_file():
                    df.write_text(pin_platform(df.read_text(), spec.platform))
        manifest["tasks"][t], manifest["n_subtasks"][t] = _tree_hash(dest / t), n_subtasks(dest / t)
    (spec.jobs_dir / "tasks.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):  # missing, or harbor is writing it
        return {}
    return data if isinstance(data, dict) else {}


def _trials(job_dir: Path) -> list[Path]:
    return sorted(d for d in job_dir.iterdir() if d.is_dir()) if job_dir.is_dir() else []


def retry_reason(trial: Path) -> str | None:
    """Why a finished trial does not measure its condition (so its job is rerun), or None.
    An agent timeout is an outcome: harbor still runs the verifier after it. So is an exception
    after the verifier started in the agent's container (Terminal-Bench): the agent can leave it
    broken. A verifier in its own fresh container (harbor's ``verifier_environment_mode``
    ``separate``, SWE-sweep) the agent cannot break: an exception there without a reward (its
    image failed to build or start, it timed out, or it wrote no reward file) is infrastructure."""
    res, ep = _json(trial / "result.json"), _json(trial / "agent" / "episode.json")
    exc = (res.get("exception_info") or {}).get("exception_type")
    if ep.get("exit_reason") == "billing_error" or exc == "BillingError":
        return "billing"
    if ep.get("exit_reason") == "provider_error" or ep.get("n_provider_errors") or exc == "ProviderUnavailableError":
        return "provider"
    if exc and not (res.get("verifier") or {}).get("started_at"):
        return f"harbor {exc}"  # environment/setup failure, or the agent crashed before the verifier
    if rewards(res).get("grader_failed"):
        return "grader"  # SWE-sweep's verifier could not grade (its own setup failed): not a 0
    if exc and _reward(res) is None and res.get("verifier_environment_mode") == "separate":
        return "verifier"  # SWE-sweep's separate verifier failed before it graded: not a 0
    return None


def job_state(job_dir: Path) -> tuple[str, str]:
    """``absent``, ``unfinished``, ``retry`` (a trial needs a rerun) or ``complete``, and why."""
    if not job_dir.is_dir():
        return "absent", "no job dir"
    res = _json(job_dir / "result.json")
    done = [t for t in _trials(job_dir) if _json(t / "result.json").get("finished_at")]
    n = res.get("n_total_trials")
    if not res.get("finished_at") or not n or len(done) < n:
        return "unfinished", f"{len(done)}/{n or '?'} trials finished"
    retry = {t.name: r for t in done if (r := retry_reason(t))}
    if retry:
        return "retry", "to rerun: " + ", ".join(f"{t} ({r})" for t, r in retry.items())
    return "complete", f"{n} trials"


def move_aside(job_dir: Path, jobs_dir: Path) -> Path:
    """Move a job dir to ``<jobs_dir>/_partial/<job>.<k>`` (k = 1, 2, ...); nothing is deleted."""
    k = 1
    while (dest := jobs_dir / "_partial" / f"{job_dir.name}.{k}").exists():
        k += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    return job_dir.rename(dest)


def sweep_models(spec: Spec) -> list[ModelConfig]:
    """Every distinct model the conditions call (the judge runs the orchestrator's model)."""
    out: dict[tuple[str, str], ModelConfig] = {}
    for cfg in spec.configs.values():
        for m in [cfg.orchestrator] + ([cfg.subagent_model] if cfg.topology == "subagents" else []):
            out.setdefault((m.model, m.completion_window), m)
    return list(out.values())


def ping(m: ModelConfig) -> str | None:
    """One tiny request (1 retry, 60 s stall limit, 30 s retry budget); the error, or None."""
    client = SailClient(dataclasses.replace(m, max_output_tokens=256, max_retries=1, request_timeout_sec=60.0,
                                            retry_budget_sec=30.0))
    try:
        client.complete("You are a connectivity check.", [Message("user", "Reply with the single word OK.")], [])
    except Exception as e:  # any failure means the model cannot serve the sweep now
        return f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        client.close()
    return None


def probe(models: list[ModelConfig]) -> list[str]:
    """``model: error`` for every model that does not answer."""
    return [f"{m.model}: {err}" for m in models if (err := ping(m))]


def run_job(cmd: list[str], env: dict[str, str], cwd: Path, watch: Callable[[], bool] | None = None) -> int:
    """Run harbor with inherited output; call ``watch`` every POLL_SEC and stop the job when it says so.
    Its own process group: a terminal Ctrl-C reaches harbor once, from :func:`_stop` (a second
    SIGINT makes asyncio abort harbor's container teardown). SIGINT is reset to its default in
    harbor: a sweep started with ``&`` from a script inherits it ignored, and so would harbor."""
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, process_group=0,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    try:
        while True:
            try:
                return proc.wait(timeout=POLL_SEC)
            except subprocess.TimeoutExpired:
                if watch is not None and watch():
                    return _stop(proc)
    except KeyboardInterrupt:
        _stop(proc)
        raise


def _stop(proc: subprocess.Popen) -> int:
    """SIGINT harbor like a Ctrl-C (it cancels its trials and removes containers), kill it if it hangs."""
    proc.send_signal(signal.SIGINT)
    try:
        return proc.wait(timeout=STOP_GRACE_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()


class Watcher:
    """Prints a line per trial as harbor finishes it. On the first provider failure it probes the
    models; if one still fails, :meth:`poll` returns True (stop the job) and ``abort`` says why."""

    def __init__(self, job_dir: Path, models: list[ModelConfig]):
        self.job_dir, self.models = job_dir, models
        self.seen: set[str] = set()
        self.probed = False
        self.abort: str | None = None

    def poll(self) -> bool:
        for trial in _trials(self.job_dir):
            res = _json(trial / "result.json")
            if trial.name in self.seen or not res.get("finished_at"):
                continue
            self.seen.add(trial.name)
            say(f"    trial {trial.name}: {trial_line(trial, res)}")
            if retry_reason(trial) == "provider" and not self.probed:
                self.probed = True
                say("    provider failure: probing every model ...")
                failing = probe(self.models)
                if failing:
                    self.abort = f"provider failure in {self.job_dir.name}/{trial.name}; still failing: {'; '.join(failing)}"
                    return True
                say("    every model answers: going on (the job is retried on the next run)")
        return False


def trial_line(trial: Path, res: dict[str, Any]) -> str:
    ep = _json(trial / "agent" / "episode.json")
    exc = (res.get("exception_info") or {}).get("exception_type")
    parts = [f"reward {_reward(res)}", f"exit {ep.get('exit_reason')}"]
    if ep.get("wall_clock_sec") is not None:
        parts.append(f"{ep['wall_clock_sec']:.0f}s")
    if ep.get("n_provider_errors"):
        parts.append(f"{ep['n_provider_errors']} provider error(s): {str(ep.get('error'))[:160]}")
    if exc:
        parts.append(f"harbor {exc}")
    return ", ".join(parts)


def _log(spec: Spec, event: str, **fields: Any) -> None:
    with (spec.jobs_dir / "sweep_manifest.jsonl").open("a") as f:
        f.write(json.dumps({"t": time.time(), "sweep": spec.name, "event": event, **fields}) + "\n")


def _write_status(spec: Spec, status: str, t_start: float) -> None:
    doc = {"name": spec.name, "status": status, "git_commit": git_commit(), "argv": sys.argv, "t_start": t_start,
           "t_update": time.time(), "spec": spec.raw, "configs": {c: cfg.to_dict() for c, cfg in spec.configs.items()}}
    (spec.jobs_dir / "sweep.json").write_text(json.dumps(doc, indent=2) + "\n")


def run(spec: Spec, jobs: list[tuple[int, str]], dry_run: bool = False) -> int:
    """Run the planned jobs in order (``dry_run``: print them); returns the exit code."""
    configs_dir = spec.jobs_dir / "configs"
    say(f"Sweep {spec.name!r}: {len(jobs)} jobs x {len(spec.tasks)} tasks, jobs dir {_rel(spec.jobs_dir, spec.root)}")
    if dry_run:
        for i, (r, cond) in enumerate(jobs, 1):
            job = job_name(spec, r, cond)
            state, why = job_state(spec.jobs_dir / job)
            say(f"  [{i}] {job}: {({'complete': 'skip', 'absent': 'run'}).get(state, 'move aside, rerun')} ({why})")
            hints_env = f"{HINTS_DIR_ENV}={_rel(hints_dir(spec), spec.root)} " if interactive(spec) else ""
            say(f"      $ SUBAGENTS_BENCH_CONFIG={_rel(configs_dir / f'{cond}.yaml', spec.root)} {hints_env}"
                f"{shlex.join(harbor_cmd(spec, job))}")
        return 0
    if harbor_bin() is None:
        say("harbor not found: run scripts/setup.sh")
        return 2
    configs_dir.mkdir(parents=True, exist_ok=True)
    for cond, cfg in spec.configs.items():
        (configs_dir / f"{cond}.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    stage_tasks(spec)
    t_start = time.time()
    _write_status(spec, "running", t_start)
    models = sweep_models(spec)
    code, n_failed = None, 0
    for i, (r, cond) in enumerate(jobs, 1):
        job = job_name(spec, r, cond)
        job_dir, ids, prefix = spec.jobs_dir / job, {"job": job, "round": r, "condition": cond}, f"[{i}/{len(jobs)}] {job}"
        state, why = job_state(job_dir)
        if state == "complete":
            say(f"{prefix}: complete ({why}) -> skip")
            _log(spec, "skipped", **ids)
            continue
        if state != "absent":
            moved = _rel(move_aside(job_dir, spec.jobs_dir), spec.root)
            say(f"{prefix}: {state} ({why}) -> moved to {moved}")
            _log(spec, "moved", **ids, state=state, why=why, to=moved)
        say(f"{prefix}: running {len(spec.tasks)} trials")
        watcher, t0 = Watcher(job_dir, models), time.time()
        env = {**os.environ, "SUBAGENTS_BENCH_CONFIG": str(configs_dir / f"{cond}.yaml"),
               HINTS_DIR_ENV: str(hints_dir(spec))}
        try:
            rc = run_job(harbor_cmd(spec, job), env, spec.root, watcher.poll)
        except KeyboardInterrupt:
            say(f"{prefix}: interrupted; re-run the same command to resume")
            _log(spec, "interrupted", **ids)
            code = 130
            break
        watcher.poll()  # trials that finished after the last poll
        state, why = job_state(job_dir)
        say(f"{prefix}: harbor exit {rc}, {state} ({why}), {time.time() - t0:.0f}s")
        _log(spec, "run", **ids, rc=rc, state=state, why=why, duration_sec=time.time() - t0)
        if state == "absent" and rc != 0 and watcher.abort is None:
            # harbor resolves the dataset online first: a dropped network kills it before the job dir exists
            say(f"{prefix}: harbor exited before creating the job: probing every model ...")
            failing = probe(models)
            if failing:
                watcher.abort = f"harbor could not start {job}; models failing too: {'; '.join(failing)}"
        if any(retry_reason(t) == "billing" for t in _trials(job_dir)):
            code, reason = 3, f"billing error in {job}: top up the Sail account"
        elif watcher.abort is not None:
            code, reason = 4, watcher.abort
        if code is not None:
            say(f"ABORT: {reason}. Re-run the same command later to resume (this job is rerun).")
            _log(spec, "abort", **ids, code=code, reason=reason)
            break
        n_failed += state != "complete"
    if code is None:
        code = 1 if n_failed else 0
    _write_status(spec, STATUS[code], t_start)
    say(f"Sweep {spec.name!r} {STATUS[code]}" + (f": {n_failed} job(s) failed or need a retry; re-run to retry"
                                                  if n_failed and code == 1 else ""))
    if code == 0:
        say(f"Next: scripts/analyze.sh {_rel(spec.jobs_dir, spec.root)}")
    return code


def preflight(spec: Spec, oracle: bool = False) -> int:
    """Docker, harbor, SAIL_API_KEY, one request per model and (``oracle``) harbor's oracle agent
    on every task, reusing an earlier oracle job that solved them all. 0 if all passed, else 1."""
    results: list[bool] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append(ok)
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    say(f"Preflight for sweep {spec.name!r}:")
    docker = _output(["docker", "info", "--format", "{{.ServerVersion}}"])
    check("docker", docker is not None, f"server {docker}" if docker else "not running: start Docker")
    harbor = harbor_bin()
    check("harbor", harbor is not None, f"{harbor} {_output([harbor, '--version'])}" if harbor
          else "not found: run scripts/setup.sh")
    check("SAIL_API_KEY", bool(os.environ.get("SAIL_API_KEY")), "set" if os.environ.get("SAIL_API_KEY")
          else "missing: add it to .env")
    if spec.dataset_path is not None:  # a registry dataset is resolved by harbor itself
        missing = [t for t in spec.tasks if not (spec.dataset_path / t / "task.toml").is_file()]
        check("dataset_path", not missing, f"{len(spec.tasks)} tasks under {spec.dataset_path}" if not missing
              else f"no task.toml under {spec.dataset_path} for: {', '.join(missing)}")
    if interactive(spec):
        problems = check_interactive(spec)
        check("interactive sessions", not problems, "; ".join(problems) if problems else
              f"{len(interactive(spec))} condition(s): subtasks, hints and agent timeout fit every task")
    for m in sweep_models(spec):
        t0 = time.time()
        err = ping(m)
        check(f"request {m.model}", err is None, err or f"answered in {time.time() - t0:.1f}s")
    if oracle and all(results):
        job_dir = spec.jobs_dir / "_oracle"
        if job_state(job_dir)[0] == "complete" and not _unsolved(spec, job_dir):
            say("  oracle: reusing the earlier _oracle job")
        else:
            if job_dir.exists():
                say(f"  oracle: earlier job moved to {_rel(move_aside(job_dir, spec.jobs_dir), spec.root)}")
            spec.jobs_dir.mkdir(parents=True, exist_ok=True)
            stage_tasks(spec)
            cmd = harbor_cmd(spec, "_oracle", agent="oracle")
            say(f"  oracle: $ {shlex.join(cmd)}")
            run_job(cmd, dict(os.environ), spec.root)
        unsolved = _unsolved(spec, job_dir)
        check("oracle solves every task", not unsolved, f"unsolved: {unsolved}" if unsolved else f"{len(spec.tasks)} tasks")
    say(f"  -> {'ALL PASSED' if all(results) else 'FAILED'}")
    return 0 if all(results) else 1


def _unsolved(spec: Spec, job_dir: Path) -> list[str]:
    results = [_json(t / "result.json") for t in _trials(job_dir)]
    solved = {r.get("task_name") for r in results if _reward(r) == 1.0}
    return [t for t in spec.tasks if t not in solved]


def rewards(res: dict[str, Any]) -> dict[str, Any]:
    """The verifier's reward file: ``reward`` (Terminal-Bench: 1 or 0; SWE-sweep: the fraction of
    bugs fixed, 0 on a regression) and, from SWE-sweep's grader, ``n_resolved``, ``visible_ok``,
    ``grader_failed`` among others."""
    return (res.get("verifier_result") or {}).get("rewards") or {}


def _reward(res: dict[str, Any]) -> Any:
    return rewards(res).get("reward")


def _output(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None
