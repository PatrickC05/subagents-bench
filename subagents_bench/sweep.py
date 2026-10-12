"""Sweep launcher: one ``harbor run`` job per (round, condition) of a sweep spec, resumable.

A spec (``configs/sweeps/*.yaml``) names a base config, the dataset and tasks, the rounds, the
harbor settings and the conditions (overrides deep-merged onto the base); ``extends: <spec>``
merges it over another spec. Relative paths resolve against the repository root (the spec's
nearest ancestor holding a pyproject.toml). Exit codes: 0 done, 1 jobs failed or need a retry,
2 spec or config error, 3 billing error, 4 provider outage, 130 interrupted.
"""

from __future__ import annotations

import dataclasses
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .config import ConfigError, ExperimentConfig, ModelConfig, from_dict
from .llm import Message, SailClient
from .records import git_commit

AGENT = "subagents_bench.harness.harbor_agent:OrchestratorAgent"
SPEC_KEYS = {"extends", "name", "base", "dataset", "tasks", "rounds", "order_seed", "harbor", "conditions",
             "jobs_dir"}
HARBOR_KEYS = {"n_concurrent", "agent_timeout_multiplier"}
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")  # job names are <sweep>__r<k>__<condition>
POLL_SEC = 10.0  # live trial lines
STOP_GRACE_SEC = 120.0  # on SIGINT harbor cancels its trials and removes their containers
STATUS = {0: "complete", 1: "incomplete", 3: "aborted: billing error", 4: "aborted: provider outage",
          130: "interrupted"}


def say(line: str) -> None:
    """Print at once: the sweep's output usually goes to a log file."""
    print(line, flush=True)


@dataclass
class Spec:
    name: str
    root: Path
    dataset: str
    tasks: list[str]
    rounds: int
    order_seed: int
    n_concurrent: int
    timeout_multiplier: float
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
        base = _yaml(root / raw["base"])
        spec = Spec(name, root, raw["dataset"], tasks, rounds, raw["order_seed"], harbor["n_concurrent"],
                    harbor["agent_timeout_multiplier"], root / raw.get("jobs_dir", f"jobs/{name}"), {}, raw)
    except KeyError as e:
        raise ConfigError(f"{path}: missing key {e}") from None
    if not (isinstance(tasks, list) and tasks and isinstance(conditions, dict) and conditions
            and isinstance(rounds, int) and rounds >= 1):
        raise ConfigError(f"{path}: tasks and conditions must be non-empty, rounds an integer >= 1")
    bad = [n for n in (name, *conditions) if not NAME_RE.fullmatch(str(n)) or "__" in str(n)]
    if bad:
        raise ConfigError(f"{path}: invalid name(s) {bad}: use letters, digits, '_', '.', '-' but not '__'")
    for cond, over in conditions.items():
        try:
            spec.configs[cond] = from_dict({**deep_merge(base, over or {}), "name": cond})
        except ConfigError as e:
            raise ConfigError(f"{path}: condition {cond}: {e}") from None
    return spec


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


def harbor_cmd(spec: Spec, job: str, agent: str = AGENT) -> list[str]:
    cmd = [harbor_bin() or "harbor", "run", "-d", spec.dataset]
    for t in spec.tasks:
        cmd += ["-i", t]
    cmd += ["-a", agent, "-e", "docker", "-n", str(spec.n_concurrent), "-k", "1",
            "-o", _rel(spec.jobs_dir, spec.root), "--job-name", job]
    if agent == AGENT:
        cmd += ["--agent-timeout-multiplier", str(spec.timeout_multiplier)]
    return cmd + ["-y", "-q"]


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
    An agent timeout is an outcome: harbor still runs the verifier after it."""
    res, ep = _json(trial / "result.json"), _json(trial / "agent" / "episode.json")
    exc = (res.get("exception_info") or {}).get("exception_type")
    if ep.get("exit_reason") == "billing_error" or exc == "BillingError":
        return "billing"
    if ep.get("exit_reason") == "provider_error" or ep.get("n_provider_errors") or exc == "ProviderUnavailableError":
        return "provider"
    if exc and not (res.get("verifier") or {}).get("started_at"):
        return f"harbor {exc}"  # environment/setup failure, or the agent crashed before the verifier
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
            say(f"      $ SUBAGENTS_BENCH_CONFIG={_rel(configs_dir / f'{cond}.yaml', spec.root)} "
                f"{shlex.join(harbor_cmd(spec, job))}")
        return 0
    if harbor_bin() is None:
        say("harbor not found: run scripts/setup.sh")
        return 2
    configs_dir.mkdir(parents=True, exist_ok=True)
    for cond, cfg in spec.configs.items():
        (configs_dir / f"{cond}.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
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
        env = {**os.environ, "SUBAGENTS_BENCH_CONFIG": str(configs_dir / f"{cond}.yaml")}
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


def _reward(res: dict[str, Any]) -> Any:
    return ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward")


def _output(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None
