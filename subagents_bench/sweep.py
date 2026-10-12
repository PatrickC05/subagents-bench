"""Sweep launcher: preflight checks, harbor job orchestration, task metadata.

A *sweep spec* (``configs/sweeps/*.yaml``) names a dataset, a task list, a
number of rounds, a base :class:`~subagents_bench.config.ExperimentConfig`
YAML and a set of *conditions* (deep-merged overrides of the base). The
launcher turns it into one ``harbor run`` job per (round, condition):

* every condition's resolved config is materialized to
  ``<jobs_dir>/configs/<cond>.yaml`` and handed to the harbor agent through
  ``SUBAGENTS_BENCH_CONFIG``;
* within a round the conditions run in an order shuffled with ``order_seed``
  (interleaving, so provider drift over the sweep is spread across conditions);
* a job whose ``result.json`` shows every trial finished is skipped, so a sweep
  is resumable; a partial job dir is moved aside to
  ``<jobs_dir>/_partial/<job>.partial.<k>`` and rerun, and so is a finished
  job with a trial that did not measure the condition (billing or provider
  failure, an infrastructure error, an agent crash before the verifier), to
  ``<job>.retry.<k>`` (:func:`trial_retry_reasons`);
* after every job, the sweep aborts on a billing error (exit 3) or when
  provider failures hit more than half the job's trials (exit 4);
* one sweep at a time per jobs dir: ``<jobs_dir>/.sweep.lock`` (``flock``);
* every job event is appended to ``<jobs_dir>/sweep_manifest.jsonl`` and the
  sweep's provenance (source fingerprint, git commit, versions, lockfile
  hash, spec, resolved configs, order seed, date) is kept in
  ``<jobs_dir>/sweep.json``, with uncommitted source changes saved under
  ``<jobs_dir>/provenance/``; a resume refuses to mix code versions, task
  lists or harbor settings with the jobs already complete unless told to.

Subcommands (``register_cli``): ``preflight SPEC [--oracle] [--redo-oracle]``,
``sweep SPEC [--dry-run] [--only COND ...] [--rounds 1,2]`` and
``tasks [DATASET]``. See ``docs/DESIGN.md`` section 5 and
``docs/REPRODUCE.md``.

Spec keys beyond DESIGN section 5: ``extends: <spec>`` deep-merges the spec
over a parent spec (``smoke.yaml`` reuses the pilot's conditions this way), and
``task_filter`` also accepts ``category``. Relative paths inside a spec
(``base``, ``extends``, ``jobs_dir``) resolve against the repository root: the
nearest ancestor of the spec file that holds a ``pyproject.toml`` (else the
current directory).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from . import provenance as source_prov

try:  # POSIX only; without it the sweep runs unlocked (Windows is untested anyway)
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

AGENT_IMPORT_PATH = "subagents_bench.harness.harbor_agent:OrchestratorAgent"
PINNED_HARBOR_VERSION = "0.24.0"
DEFAULT_DATASET = "terminal-bench@2.0"
ORACLE_JOB = "_oracle"
PARTIAL_DIR = "_partial"
CONFIGS_DIR = "configs"
PROVENANCE_DIR = "provenance"
MANIFEST_NAME = "sweep_manifest.jsonl"
SWEEP_JSON_NAME = "sweep.json"
TASKS_META_NAME = "tasks_meta.json"
PREFLIGHT_JSON_NAME = "preflight.json"
LOCK_NAME = ".sweep.lock"
LOCKFILE_NAMES = ("uv.lock", "requirements.lock")
# Job names are <sweep>__r<round>__<condition>; "__" is the separator.
JOB_SEP = "__"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PRIME_PROVIDERS = frozenset({"prime", "prime_intellect", "pinference"})
# max_tokens for the preflight ping. Reasoning models may spend all of it
# thinking (finish_reason "length"); any completed stream counts as a pass.
PING_MAX_TOKENS = 256

SPEC_KEYS = {
    "name", "description", "extends", "dataset", "tasks", "task_filter", "rounds",
    "base", "conditions", "harbor", "jobs_dir", "order_seed",
}
HARBOR_KEYS = {"env", "n_concurrent", "agent_timeout_multiplier", "extra_args"}
TASK_FILTER_KEYS = {"difficulty", "category", "max_expert_min"}

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2
EXIT_BILLING = 3
EXIT_PROVIDER = 4
EXIT_INTERRUPTED = 130

# Why a finished trial does not measure its condition, so its job is rerun
# (trial_retry_reasons). Checked in this order; a trial gets the first match.
RETRY_BILLING = "billing"  # BillingError: the account ran out
RETRY_PROVIDER = "provider"  # a model call failed after all retries (outage, 401/403, missing key)
RETRY_INFRA = "infrastructure"  # harbor failed outside the agent (environment start, image pull, build)
RETRY_AGENT_EXCEPTION = "agent_exception"  # the agent raised and the verifier never ran (harness bug)
# harbor exceptions raised before/around the agent rather than by it. Kept
# equal to analysis.tables.OUTSIDE_AGENT_EXCEPTIONS (a test checks), without
# importing the analysis stack here.
INFRA_EXCEPTIONS = frozenset({
    "EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "SandboxBuildFailedError",
    "HealthcheckError", "CancelledError", "DockerBuildError",
})
# telemetry.PROVIDER_ERROR_PREFIX: how a provider failure starts in calls.jsonl
# `error` (and in episode.json `error` before schema 3); a test checks it.
PROVIDER_ERROR_PREFIX = "LLMError:"


class SpecError(ValueError):
    """The sweep spec (or something it references) is invalid."""


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


@dataclass
class HarborSettings:
    env: str = "docker"
    n_concurrent: int = 4
    agent_timeout_multiplier: float = 1.0
    extra_args: list[str] = field(default_factory=list)


@dataclass
class SweepSpec:
    name: str
    dataset: str
    tasks: list[str] | None
    task_filter: dict[str, Any] | None
    rounds: int
    base_path: Path | None
    base: dict[str, Any]
    conditions: dict[str, dict[str, Any]]
    harbor: HarborSettings
    jobs_dir: Path
    order_seed: int
    spec_path: Path
    root: Path
    # The spec after `extends` resolution (what sweep.json records).
    raw: dict[str, Any] = field(default_factory=dict)
    # Every spec file read (the spec plus its `extends` chain), for hashing.
    source_files: list[Path] = field(default_factory=list)


def deep_merge(base: Any, override: Any) -> Any:
    """Merge ``override`` onto ``base`` without mutating either.

    Mappings merge recursively; any other override value (scalar, list or
    None) replaces the base value outright.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        out = copy.deepcopy(base)
        for k, v in override.items():
            out[k] = deep_merge(out[k], v) if k in out else copy.deepcopy(v)
        return out
    return copy.deepcopy(override)


def find_root(start: Path) -> Path:
    """Nearest ancestor of ``start`` holding a pyproject.toml, else the cwd."""
    start = start.resolve()
    for d in [start, *start.parents]:
        if (d / "pyproject.toml").is_file():
            return d
    return Path.cwd().resolve()


def default_root() -> Path:
    """The repository this package was installed from (editable install),
    else the project around the current directory."""
    pkg_parent = Path(__file__).resolve().parents[1]
    if (pkg_parent / "pyproject.toml").is_file():
        return pkg_parent
    return find_root(Path.cwd())


def _resolve(root: Path, p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (root / p)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except OSError as e:
        raise SpecError(f"cannot read {path}: {e}") from e
    except yaml.YAMLError as e:
        raise SpecError(f"invalid YAML in {path}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SpecError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def load_spec_dict(path: Path, root: Path, _chain: tuple[Path, ...] = ()) -> tuple[dict[str, Any], list[Path]]:
    """Read a spec and resolve its ``extends`` chain (child keys win)."""
    path = path.resolve()
    if path in _chain:
        raise SpecError(f"`extends` cycle: {' -> '.join(str(p) for p in (*_chain, path))}")
    raw = _load_yaml_mapping(path)
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw, [path]
    if not isinstance(parent_ref, str):
        raise SpecError(f"{path}: `extends` must be a path string")
    parent_path = _resolve(root, parent_ref)
    if not parent_path.is_file():
        raise SpecError(f"{path}: `extends` target not found: {parent_path}")
    parent, files = load_spec_dict(parent_path, root, (*_chain, path))
    return deep_merge(parent, raw), [*files, path]


def _check_name(kind: str, name: Any) -> str:
    if not isinstance(name, str) or not NAME_RE.match(name) or JOB_SEP in name:
        raise SpecError(
            f"invalid {kind} name {name!r}: use letters, digits, '_', '.', '-' "
            f"(no '{JOB_SEP}', which separates the parts of a job name)"
        )
    return name


def parse_spec(raw: dict[str, Any], spec_path: Path, root: Path, source_files: list[Path] | None = None) -> SweepSpec:
    """Validate an (extends-resolved) spec mapping into a :class:`SweepSpec`."""
    unknown = sorted(set(raw) - SPEC_KEYS)
    if unknown:
        raise SpecError(f"unknown spec keys: {unknown} (allowed: {sorted(SPEC_KEYS)})")
    name = _check_name("sweep", raw.get("name"))

    dataset = raw.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise SpecError("`dataset` is required (e.g. terminal-bench@2.0)")

    tasks = raw.get("tasks")
    task_filter = raw.get("task_filter")
    if tasks is not None and task_filter is not None:
        raise SpecError("give either `tasks` or `task_filter`, not both")
    if tasks is None and task_filter is None:
        raise SpecError("`tasks` (explicit list) or `task_filter` is required")
    if tasks is not None:
        if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) and t for t in tasks):
            raise SpecError("`tasks` must be a non-empty list of task names")
        if len(set(tasks)) != len(tasks):
            raise SpecError("`tasks` has duplicates")
    if task_filter is not None:
        if not isinstance(task_filter, dict):
            raise SpecError("`task_filter` must be a mapping")
        bad = sorted(set(task_filter) - TASK_FILTER_KEYS)
        if bad:
            raise SpecError(f"unknown task_filter keys: {bad} (allowed: {sorted(TASK_FILTER_KEYS)})")

    rounds = raw.get("rounds", 1)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise SpecError("`rounds` must be an integer >= 1")
    order_seed = raw.get("order_seed", 0)
    if not isinstance(order_seed, int) or isinstance(order_seed, bool):
        raise SpecError("`order_seed` must be an integer")

    base_ref = raw.get("base")
    base_path: Path | None = None
    base: dict[str, Any] = {}
    if base_ref is not None:
        if not isinstance(base_ref, str):
            raise SpecError("`base` must be a path string")
        base_path = _resolve(root, base_ref)
        if not base_path.is_file():
            raise SpecError(f"base config not found: {base_path}")
        base = _load_yaml_mapping(base_path)

    conds_raw = raw.get("conditions")
    if not isinstance(conds_raw, dict) or not conds_raw:
        raise SpecError("`conditions` must be a non-empty mapping of name -> overrides")
    conditions: dict[str, dict[str, Any]] = {}
    for cname, override in conds_raw.items():
        _check_name("condition", cname)
        if override is None:
            override = {}
        if not isinstance(override, dict):
            raise SpecError(f"condition {cname!r}: overrides must be a mapping")
        conditions[cname] = override

    h = raw.get("harbor") or {}
    if not isinstance(h, dict):
        raise SpecError("`harbor` must be a mapping")
    bad = sorted(set(h) - HARBOR_KEYS)
    if bad:
        raise SpecError(f"unknown harbor keys: {bad} (allowed: {sorted(HARBOR_KEYS)})")
    harbor = HarborSettings(
        env=h.get("env", "docker"),
        n_concurrent=h.get("n_concurrent", 4),
        agent_timeout_multiplier=h.get("agent_timeout_multiplier", 1.0),
        extra_args=list(h.get("extra_args") or []),
    )
    if not isinstance(harbor.env, str) or not harbor.env:
        raise SpecError("harbor.env must be a non-empty string")
    if not isinstance(harbor.n_concurrent, int) or isinstance(harbor.n_concurrent, bool) or harbor.n_concurrent < 1:
        raise SpecError("harbor.n_concurrent must be an integer >= 1")
    m = harbor.agent_timeout_multiplier
    if not isinstance(m, (int, float)) or isinstance(m, bool) or m <= 0:
        raise SpecError("harbor.agent_timeout_multiplier must be a number > 0")
    if not all(isinstance(a, str) for a in harbor.extra_args):
        raise SpecError("harbor.extra_args must be a list of strings")

    jobs_dir_ref = raw.get("jobs_dir", f"jobs/{name}")
    if not isinstance(jobs_dir_ref, str) or not jobs_dir_ref:
        raise SpecError("`jobs_dir` must be a path string")

    return SweepSpec(
        name=name,
        dataset=dataset,
        tasks=list(tasks) if tasks is not None else None,
        task_filter=dict(task_filter) if task_filter is not None else None,
        rounds=rounds,
        base_path=base_path,
        base=base,
        conditions=conditions,
        harbor=harbor,
        jobs_dir=_resolve(root, jobs_dir_ref),
        order_seed=order_seed,
        spec_path=spec_path.resolve(),
        root=root,
        raw=raw,
        source_files=list(source_files or [spec_path.resolve()]),
    )


def load_spec(path: str | Path, root: Path | None = None) -> SweepSpec:
    """Load, ``extends``-resolve and validate a sweep spec file."""
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"spec not found: {path}")
    root = root or find_root(path.parent)
    raw, files = load_spec_dict(path, root)
    return parse_spec(raw, path, root, files)


# ---------------------------------------------------------------------------
# Conditions: resolution, validation, materialization, order
# ---------------------------------------------------------------------------


def resolve_conditions(spec: SweepSpec) -> dict[str, dict[str, Any]]:
    """Condition name -> full config dict (base deep-merged with overrides)."""
    out = {}
    for cname, override in spec.conditions.items():
        merged = deep_merge(spec.base, override)
        merged["name"] = cname
        out[cname] = merged
    return out


def _dropped_keys(merged: Any, roundtrip: Any, prefix: str = "") -> list[str]:
    """Keys set in ``merged`` that ``roundtrip`` lost or changed.

    A None in ``merged`` may be normalized to a default by the loader, so only
    non-None values must survive unchanged.
    """
    problems: list[str] = []
    if isinstance(merged, dict):
        if not isinstance(roundtrip, dict):
            if any(v is not None for v in merged.values()):
                problems.append(f"{prefix or '<root>'}: mapping became {type(roundtrip).__name__}")
            return problems
        for k, v in merged.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if k not in roundtrip:
                problems.append(f"{key}: dropped by ExperimentConfig")
            else:
                problems.extend(_dropped_keys(v, roundtrip[k], key))
        return problems
    if merged is None:
        return problems
    if merged != roundtrip:
        problems.append(f"{prefix}: {merged!r} became {roundtrip!r}")
    return problems


def load_experiment_config(cfg: dict[str, Any]) -> Any:
    """``ExperimentConfig.from_dict`` (imported lazily so specs can be
    inspected even while the runtime package is being changed)."""
    from .config import ExperimentConfig

    return ExperimentConfig.from_dict(cfg)


def validate_condition(cfg: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None]:
    """Check a resolved condition loads into ExperimentConfig losslessly.

    Returns ``(problems, full_dict)`` where ``full_dict`` is
    ``ExperimentConfig.from_dict(cfg).to_dict()`` (every default explicit)
    when loading worked, else None.
    """
    try:
        exp = load_experiment_config(cfg)
        full = exp.to_dict()
    except Exception as e:  # any loader error is a config problem to report
        return [f"ExperimentConfig.from_dict failed: {type(e).__name__}: {e}"], None
    problems = _dropped_keys(cfg, full)
    try:
        yaml.safe_dump(full)
    except yaml.YAMLError as e:
        problems.append(f"to_dict() is not YAML-serializable: {e}")
    return problems, full


def condition_order(conditions: list[str], rounds: int, seed: int) -> dict[int, list[str]]:
    """Per-round shuffled condition order, deterministic in (conditions, seed).

    One ``random.Random(seed)`` shuffles a fresh copy of the spec's condition
    list for round 1, 2, ... in turn, so a round's order never depends on which
    rounds or conditions a particular invocation selects.
    """
    rng = random.Random(seed)
    out = {}
    for r in range(1, rounds + 1):
        order = list(conditions)
        rng.shuffle(order)
        out[r] = order
    return out


def job_name(sweep: str, round_: int, cond: str) -> str:
    return f"{sweep}{JOB_SEP}r{round_}{JOB_SEP}{cond}"


def parse_job_name(job: str) -> tuple[str, int, str] | None:
    """Inverse of :func:`job_name`; None if ``job`` is not a sweep job name."""
    parts = job.split(JOB_SEP)
    if len(parts) != 3 or not re.fullmatch(r"r\d+", parts[1]):
        return None
    return parts[0], int(parts[1][1:]), parts[2]


def _dump_config_yaml(cfg: dict[str, Any], header: str) -> str:
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    return "".join(f"# {line}\n" for line in header.splitlines()) + body


def materialize_configs(
    spec: SweepSpec,
    full_configs: dict[str, dict[str, Any]],
    *,
    allow_change: bool = False,
) -> dict[str, Path]:
    """Write ``<jobs_dir>/configs/<cond>.yaml`` for each condition.

    Refuses (SpecError) to change an existing materialized config of a
    condition that already has job dirs, unless ``allow_change``: a resumed
    sweep must not mix two configs under one condition name.
    """
    out_dir = spec.jobs_dir / CONFIGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cname, cfg in full_configs.items():
        path = out_dir / f"{cname}.yaml"
        if path.is_file() and not allow_change:
            try:
                old = yaml.safe_load(path.read_text())
            except yaml.YAMLError:
                old = None
            if old != cfg and _condition_has_jobs(spec, cname):
                raise SpecError(
                    f"resolved config of condition {cname!r} differs from {path}, and jobs for it "
                    "already exist. Use a new jobs_dir (or pass --allow-config-change to mix configs)."
                )
        header = (
            f"Resolved ExperimentConfig for condition {cname!r} of sweep {spec.name!r}.\n"
            f"Generated by `subagents-bench sweep` from {_rel(spec.spec_path, spec.root)}; do not edit."
        )
        path.write_text(_dump_config_yaml(cfg, header))
        paths[cname] = path
    return paths


def _condition_has_jobs(spec: SweepSpec, cname: str) -> bool:
    if not spec.jobs_dir.is_dir():
        return False
    for d in spec.jobs_dir.iterdir():
        parsed = parse_job_name(d.name)
        if d.is_dir() and parsed and parsed[0] == spec.name and parsed[2] == cname:
            return True
    return False


def models_used(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Model dicts a resolved condition calls: orchestrator, then subagent
    (topology subagents), then arbiter judge (strategy arbiter)."""
    orch = cfg.get("orchestrator") or {}
    out = [orch]
    if cfg.get("topology", "subagents") == "subagents":
        sub = cfg.get("subagent") or {}
        out.append(sub.get("model") or orch)
        if sub.get("strategy") == "arbiter":
            out.append(((sub.get("arbiter") or {}).get("model")) or orch)
    return out


def model_key(m: dict[str, Any]) -> tuple[str, str, str | None]:
    return (str(m.get("model")), str(m.get("provider", "anthropic")), m.get("completion_window"))


# ---------------------------------------------------------------------------
# Harbor command, jobs and their status
# ---------------------------------------------------------------------------


def harbor_executable() -> str | None:
    """The venv's ``harbor`` (sibling of sys.executable), else one on PATH."""
    sibling = Path(sys.executable).parent / "harbor"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which("harbor")


def build_harbor_command(
    *,
    harbor: str,
    dataset: str,
    tasks: Iterable[str],
    agent: str,
    env: str,
    n_concurrent: int,
    jobs_dir: str,
    job: str,
    agent_timeout_multiplier: float | None = None,
    extra_args: Iterable[str] = (),
) -> list[str]:
    """``harbor run`` argv (flags verified against harbor 0.24.0)."""
    cmd = [harbor, "run", "-d", dataset]
    for t in tasks:
        cmd += ["-i", t]
    cmd += ["-a", agent, "-e", env, "-n", str(n_concurrent), "-k", "1", "-o", jobs_dir, "--job-name", job]
    if agent_timeout_multiplier is not None:
        cmd += ["--agent-timeout-multiplier", str(agent_timeout_multiplier)]
    cmd += ["-y", "-q"]
    cmd += list(extra_args)
    return cmd


@dataclass
class JobStatus:
    """State of a harbor job dir: ``absent``, ``complete``, ``partial``,
    ``needs_retry`` (finished, but some trial did not measure the condition:
    see ``retry_trials``) or ``duplicate`` (more trial dirs than trials, or a
    task run more often than its attempts: two harbor runs wrote into it)."""

    state: str
    reason: str
    n_total: int | None = None
    # harbor 0.24.0 counts EVERY finished trial in n_completed_trials, errored
    # ones included (JobStats.increment), so n_errored is a subset of it.
    n_completed: int | None = None
    n_errored: int | None = None
    n_running: int | None = None
    n_pending: int | None = None
    n_trial_results: int | None = None
    n_trial_dirs: int | None = None
    # trial name -> reasons (RETRY_*), for every trial that needs a rerun.
    retry_trials: dict[str, list[str]] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.state == "complete"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _trial_dirs(job_dir: Path) -> list[Path]:
    """Every trial dir of a job (harbor writes only trial dirs as sub-dirs)."""
    return sorted(d for d in job_dir.iterdir() if d.is_dir())


def _trial_task(trial_dir: Path) -> str:
    """Task of a trial: its result.json ``task_name``, else ``<task>__<suffix>``."""
    res = _read_json(trial_dir / "result.json")
    name = res.get("task_name") if isinstance(res, dict) else None
    return name or trial_dir.name.rsplit(JOB_SEP, 1)[0]


def _duplicate_trials(job_dir: Path, trial_dirs: list[Path], n_total: int) -> str | None:
    """Why ``job_dir`` holds more trials than one harbor run makes, else None.

    Two runs into one job dir (two sweeps, or a job relaunched while the old
    harbor still ran) leave extra trial dirs that analysis would count twice.
    """
    if len(trial_dirs) > n_total:
        return f"{len(trial_dirs)} trial dirs for {n_total} trials"
    cfg = _read_json(job_dir / "config.json")
    attempts = cfg.get("n_attempts", 1) if isinstance(cfg, dict) else 1
    attempts = attempts if isinstance(attempts, int) and attempts > 0 else 1
    counts: dict[str, int] = {}
    for d in trial_dirs:
        t = _trial_task(d)
        counts[t] = counts.get(t, 0) + 1
    dup = sorted(t for t, n in counts.items() if n > attempts)
    if dup:
        return f"task(s) run more than {attempts}x: {dup}"
    return None


def job_status(job_dir: Path) -> JobStatus:
    """Classify a job dir. Finished means: ``result.json`` parses, the job has
    ``finished_at``, no trial is running or pending, every trial finished
    (``n_completed_trials >= n_total_trials``; harbor counts errored trials
    as completed) and every trial dir has its own ``result.json``. A finished
    job is ``complete`` unless a trial needs a rerun (``needs_retry``, see
    :func:`trial_retry_reasons`). Extra trials make it ``duplicate``."""
    if not job_dir.exists():
        return JobStatus("absent", "no job dir")
    rp = job_dir / "result.json"
    if not rp.is_file():
        return JobStatus("partial", "job dir has no result.json")
    try:
        data = json.loads(rp.read_text())
    except (OSError, ValueError) as e:
        return JobStatus("partial", f"unreadable result.json: {e}")
    stats = data.get("stats") or {}
    st = JobStatus(
        "partial",
        "",
        n_total=data.get("n_total_trials"),
        n_completed=stats.get("n_completed_trials"),
        n_errored=stats.get("n_errored_trials"),
        n_running=stats.get("n_running_trials"),
        n_pending=stats.get("n_pending_trials"),
    )
    trials = _trial_dirs(job_dir)
    st.n_trial_dirs = len(trials)
    st.n_trial_results = sum(1 for d in trials if (d / "result.json").is_file())
    dup = _duplicate_trials(job_dir, trials, st.n_total) if isinstance(st.n_total, int) and st.n_total > 0 else None
    if not isinstance(st.n_total, int) or st.n_total <= 0:
        st.reason = "n_total_trials missing or 0"
    elif dup:
        st.state, st.reason = "duplicate", dup
    elif st.n_running or st.n_pending:
        st.reason = f"{st.n_running or 0} running / {st.n_pending or 0} pending trials"
    elif not isinstance(st.n_completed, int) or st.n_completed < st.n_total:
        st.reason = f"{st.n_completed}/{st.n_total} trials finished"
    elif data.get("finished_at") is None:
        st.reason = "job has no finished_at"
    elif st.n_trial_results < st.n_total:
        st.reason = f"only {st.n_trial_results}/{st.n_total} trial result.json files"
    else:
        done = f"{st.n_completed}/{st.n_total} trials finished ({st.n_errored or 0} errored)"
        st.retry_trials = trial_retry_reasons(job_dir)
        if st.retry_trials:
            st.state, st.reason = "needs_retry", f"{done}; {_retry_summary(st.retry_trials)}"
        else:
            st.state, st.reason = "complete", done
    return st


def _retry_summary(retry: dict[str, list[str]]) -> str:
    counts: dict[str, int] = {}
    for reasons in retry.values():
        for r in reasons:
            counts[r] = counts.get(r, 0) + 1
    return f"{len(retry)} to rerun (" + ", ".join(f"{r} {n}" for r, n in sorted(counts.items())) + ")"


def _calls_have_provider_error(calls_path: Path) -> bool:
    try:
        with calls_path.open() as f:
            for line in f:
                if PROVIDER_ERROR_PREFIX in line:
                    try:
                        err = json.loads(line).get("error")
                    except ValueError:
                        continue
                    if isinstance(err, str) and err.startswith(PROVIDER_ERROR_PREFIX):
                        return True
    except OSError:
        return False
    return False


def _provider_failed(ep: dict[str, Any], agent_dir: Path) -> bool:
    """Whether a provider failure touched the episode: exit_reason
    ``provider_error``, or ``n_provider_errors > 0``. Records from before
    schema 3 have neither, so their ``error`` and calls.jsonl are checked."""
    if ep.get("exit_reason") == "provider_error":
        return True
    n = ep.get("n_provider_errors")
    if isinstance(n, int):
        return n > 0
    err = ep.get("error")
    if ep.get("exit_reason") == "error" and isinstance(err, str) and err.startswith(PROVIDER_ERROR_PREFIX):
        return True
    return _calls_have_provider_error(agent_dir / "calls.jsonl")


def _exception_outside_agent(res: dict[str, Any], exc: dict[str, Any]) -> bool:
    """harbor's exception happened outside the agent (environment build or
    start, setup, cancel): the rule of analysis.tables.exception_outside_agent."""
    if exc.get("exception_type") in INFRA_EXCEPTIONS:
        return True
    ae = res.get("agent_execution")
    started = ae.get("started_at") if isinstance(ae, dict) else None
    if not started:
        return True  # the agent never started
    occurred, start = _iso_ts(exc.get("occurred_at")), _iso_ts(started)
    return bool(occurred is not None and start is not None and occurred < start)


def _iso_ts(v: Any) -> float | None:
    """Epoch seconds of one of harbor's ISO timestamps (naive ones are UTC)."""
    if not isinstance(v, str) or not v:
        return None
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def trial_retry_reasons(job_dir: Path) -> dict[str, list[str]]:
    """trial name -> why the trial must be rerun, for every trial whose
    outcome does not measure its condition:

    * ``billing``: episode exit_reason ``billing_error`` or harbor exception
      ``BillingError``;
    * ``provider``: exit_reason ``provider_error``, ``n_provider_errors > 0``
      (any role's call failed after all retries) or harbor exception
      ``ProviderUnavailableError``;
    * ``infrastructure``: harbor's exception happened outside the agent
      (:data:`INFRA_EXCEPTIONS`, or before the agent started);
    * ``agent_exception``: the agent raised and the verifier never ran.

    An agent timeout is an outcome, not a retry reason: harbor still runs
    the verifier after it. Trials without ``result.json`` (still running)
    count only through their episode.json.
    """
    out: dict[str, list[str]] = {}
    if not job_dir.is_dir():
        return out
    for trial in _trial_dirs(job_dir):
        res = _read_json(trial / "result.json")
        res = res if isinstance(res, dict) else {}
        ep = _read_json(trial / "agent" / "episode.json")
        ep = ep if isinstance(ep, dict) else {}
        exc = res.get("exception_info")
        exc = exc if isinstance(exc, dict) else None
        etype = exc.get("exception_type") if exc else None
        if ep.get("exit_reason") == "billing_error" or etype == "BillingError":
            reason = RETRY_BILLING
        elif etype == "ProviderUnavailableError" or (ep and _provider_failed(ep, trial / "agent")):
            reason = RETRY_PROVIDER
        elif exc and res and _exception_outside_agent(res, exc):
            reason = RETRY_INFRA
        elif exc and res and not (isinstance(res.get("verifier"), dict) and res["verifier"].get("started_at")):
            reason = RETRY_AGENT_EXCEPTION
        else:
            continue
        out[trial.name] = [reason]
    return out


def move_aside(job_dir: Path, jobs_dir: Path, kind: str = "partial") -> Path:
    """Move a job dir to ``<jobs_dir>/_partial/<job>.<kind>.<k>`` (k = 1, 2,
    ...) and return the new path. Nothing is deleted. ``kind`` is
    ``partial`` (unfinished) or ``retry`` (finished, rerun)."""
    dest_root = jobs_dir / PARTIAL_DIR
    dest_root.mkdir(parents=True, exist_ok=True)
    k = 1
    while (dest_root / f"{job_dir.name}.{kind}.{k}").exists():
        k += 1
    dest = dest_root / f"{job_dir.name}.{kind}.{k}"
    shutil.move(str(job_dir), str(dest))
    return dest


def scan_billing_errors(job_dir: Path) -> list[str]:
    """Trial names whose episode ended in a billing error (from
    ``<trial>/agent/episode.json``, or harbor's exception type as a backup)."""
    hits = set()
    if not job_dir.is_dir():
        return []
    for ep in job_dir.glob("*/agent/episode.json"):
        try:
            if json.loads(ep.read_text()).get("exit_reason") == "billing_error":
                hits.add(ep.parent.parent.name)
        except (OSError, ValueError):
            continue
    for res in job_dir.glob("*/result.json"):
        try:
            exc = json.loads(res.read_text()).get("exception_info") or {}
        except (OSError, ValueError):
            continue
        if isinstance(exc, dict) and exc.get("exception_type") == "BillingError":
            hits.add(res.parent.name)
    return sorted(hits)


def trial_rewards(job_dir: Path) -> dict[str, dict[str, Any]]:
    """trial name -> {task, reward, exception} from each trial's result.json."""
    out = {}
    if not job_dir.is_dir():
        return out
    for res in sorted(job_dir.glob("*/result.json")):
        try:
            data = json.loads(res.read_text())
        except (OSError, ValueError):
            continue
        rewards = (data.get("verifier_result") or {}).get("rewards") or {}
        exc = data.get("exception_info") or {}
        out[res.parent.name] = {
            "task": data.get("task_name") or res.parent.name.split(JOB_SEP)[0],
            "reward": rewards.get("reward"),
            "exception": exc.get("exception_type") if isinstance(exc, dict) else None,
        }
    return out


# ---------------------------------------------------------------------------
# Subprocesses, environment, provenance
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, timeout: float | None = 60, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a short command, capturing output. Missing binaries and timeouts
    become a CompletedProcess with return code 127 / 124 instead of raising."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env, check=False)
    except FileNotFoundError as e:
        return subprocess.CompletedProcess(cmd, 127, "", str(e))
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(cmd, 124, "", f"timed out after {e.timeout}s")


def run_job_process(
    cmd: list[str], *, env: dict[str, str], cwd: Path, grace_sec: float = 120.0, pass_fds: tuple[int, ...] = ()
) -> int:
    """Run a harbor job with inherited stdout/stderr; return its exit code.

    On Ctrl-C harbor (same process group) receives SIGINT too; we give it
    ``grace_sec`` to cancel trials and remove containers before terminating
    it, then re-raise KeyboardInterrupt. ``pass_fds`` stay open in harbor
    (the sweep lock).
    """
    kw: dict[str, Any] = {"pass_fds": pass_fds} if pass_fds else {}
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, **kw)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=grace_sec)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise


def load_env_files(dotenv_path: Path | None) -> dict[str, str]:
    """Load ``.env`` into ``os.environ`` without overriding set variables.

    Uses :func:`subagents_bench.credentials.load_dotenv`; falls back to a
    minimal KEY=VALUE reader if that module is unavailable. Returns
    ``{key: status}`` (never values).
    """
    if dotenv_path is None or not dotenv_path.is_file():
        return {}
    try:
        from .credentials import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        return dict(load_dotenv(dotenv_path, override=False) or {})
    status = {}
    for line in dotenv_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k in os.environ:
            status[k] = "kept"
        else:
            os.environ[k] = v
            status[k] = "set"
    return status


def prime_credentials() -> tuple[str | None, str | None]:
    """(api_key, team_id) via subagents_bench.credentials, (None, None) if
    that module is unavailable."""
    try:
        from .credentials import prime_credentials as _pc
    except ImportError:
        return None, None
    return _pc()


def build_subprocess_env(
    config_path: Path | None,
    *,
    base_env: dict[str, str] | None = None,
    need_prime: bool = False,
) -> dict[str, str]:
    """Environment for a harbor subprocess: the current env (``.env`` already
    loaded), Prime credentials when a condition uses Prime, and
    ``SUBAGENTS_BENCH_CONFIG``. A stale ``SUBAGENTS_BENCH_CONFIG_JSON`` is
    removed so it cannot shadow the file."""
    env = dict(os.environ if base_env is None else base_env)
    if need_prime:
        key, team = prime_credentials()
        if key and not env.get("PRIME_API_KEY"):
            env["PRIME_API_KEY"] = key
        if team and not env.get("PRIME_TEAM_ID"):
            env["PRIME_TEAM_ID"] = team
    env.pop("SUBAGENTS_BENCH_CONFIG_JSON", None)
    if config_path is not None:
        env["SUBAGENTS_BENCH_CONFIG"] = str(config_path)
    return env


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def code_root() -> Path:
    """Root of the harness code that harbor jobs import (the repository this
    package runs from): what the source fingerprint and git state describe."""
    return source_prov.repo_root()


def source_state(root: Path) -> dict[str, Any]:
    """``{source_sha256, git_commit, git_dirty, untracked_source_files}`` of
    the code at ``root`` (:mod:`subagents_bench.provenance`; never raises).
    ``source_sha256`` identifies the code even when it is uncommitted;
    ``git_dirty`` covers tracked source files only."""
    return {"source_sha256": source_prov.source_fingerprint(root), **source_prov.git_info(root)}


def source_file_hashes(root: Path) -> dict[str, str]:
    """relative path -> sha256 of every source file the fingerprint covers."""
    out = {}
    for rel in source_prov.source_files(root):
        h = sha256_file(root / rel)
        if h is not None:
            out[rel] = h
    return out


def write_source_record(jobs_dir: Path, root: Path, state: dict[str, Any], name: str) -> dict[str, str | None]:
    """Save what is needed to recover the code that ran, under
    ``<jobs_dir>/provenance/``, and return the paths written (relative to
    ``jobs_dir``):

    * ``source_<sha16>.json``: per-file sha256 of the fingerprinted source,
      once per fingerprint (lets a refusal name the files that changed);
    * ``<name>.patch``, only when tracked source differs from HEAD or there
      are untracked source files: ``git diff HEAD`` of the source paths plus
      every untracked source file as a new-file diff. ``git apply`` on a
      checkout of ``git_commit`` restores the tree. Nothing outside git.
    """
    out: dict[str, str | None] = {"files": None, "patch": None}
    pdir = jobs_dir / PROVENANCE_DIR
    fp = state.get("source_sha256")
    if fp:
        files_path = pdir / f"source_{fp[:16]}.json"
        if not files_path.is_file():
            write_json_atomic(files_path, {"source_sha256": fp, "root": str(root), "files": source_file_hashes(root)})
        out["files"] = _rel(files_path, jobs_dir)
    untracked = state.get("untracked_source_files") or []
    if state.get("git_commit") and (state.get("git_dirty") or untracked):
        parts = [
            f"# Uncommitted harness source of {name}: apply with `git apply` on a checkout of\n"
            f"# {state['git_commit']} (source_sha256 {fp}).\n"
        ]
        diff = _run(["git", "-C", str(root), "diff", "--binary", "HEAD", "--", *source_prov.SOURCE_PATHS], timeout=60)
        parts.append(diff.stdout if diff.returncode == 0 else f"# git diff HEAD failed: {diff.stderr.strip()[:300]}\n")
        for rel in untracked:
            # Exit code 1 means "differs", which a new file always does.
            new_file = _run(["git", "diff", "--no-index", "--binary", "--", "/dev/null", rel], timeout=60, cwd=root)
            parts.append(new_file.stdout if new_file.returncode in (0, 1) else f"# cannot diff {rel}\n")
        patch_path = pdir / f"{name}.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text("".join(parts))
        out["patch"] = _rel(patch_path, jobs_dir)
    return out


def _source_record(jobs_dir: Path, sha: str) -> dict[str, str] | None:
    """The per-file hashes recorded for fingerprint ``sha``
    (``provenance/source_<sha16>.json``), or None."""
    rec = _read_json(jobs_dir / PROVENANCE_DIR / f"source_{sha[:16]}.json")
    if not isinstance(rec, dict) or not isinstance(rec.get("files"), dict) or rec.get("source_sha256") != sha:
        return None
    return rec["files"]


def source_changes(jobs_dir: Path, old_sha: str, root: Path) -> list[str] | None:
    """Every file that differs between the source recorded for ``old_sha``
    and the code at ``root`` now; None when that record does not exist."""
    before = _source_record(jobs_dir, old_sha)
    if before is None:
        return None
    now = source_file_hashes(root)
    return sorted(f for f in set(before) | set(now) if before.get(f) != now.get(f))


def _listing(files: list[str], limit: int = 12) -> list[str]:
    """``files`` cut to ``limit`` names for a message."""
    return files[:limit] + ([f"... {len(files) - limit} more"] if len(files) > limit else [])


def outside_runtime(changed: list[str] | None) -> bool:
    """True when a source change is known and touches only files no episode
    imports or reads (analysis, plots, the launcher, scripts, configs:
    :data:`subagents_bench.provenance.NON_RUNTIME_PATHS`), so the code under
    test is unchanged."""
    return bool(changed) and not any(source_prov.affects_runtime(f) for f in changed)


def runtime_versions(jobs_dir: Path, shas: Iterable[str]) -> set[str]:
    """The distinct versions of the code under test among source
    fingerprints ``shas``: each one's runtime fingerprint from its provenance
    record, or the fingerprint itself when it has no record (then it counts
    as different)."""
    out = set()
    for sha in shas:
        files = _source_record(jobs_dir, sha)
        out.add(sha if files is None else "runtime:" + source_prov.runtime_fingerprint(files))
    return out


def harbor_version(harbor: str | None = None) -> str | None:
    try:
        import importlib.metadata as md

        return md.version("harbor")
    except Exception:
        pass
    if harbor:
        r = _run([harbor, "--version"], timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().split()[-1]
    return None


def lockfile_info(root: Path) -> tuple[str | None, str | None]:
    for name in LOCKFILE_NAMES:
        p = root / name
        if p.is_file():
            return name, sha256_file(p)
    return None, None


def docker_info() -> dict[str, Any] | None:
    r = _run(["docker", "info", "--format", "{{.ServerVersion}}|{{.Architecture}}|{{.NCPU}}|{{.MemTotal}}"], timeout=30)
    if r.returncode != 0:
        return None
    parts = (r.stdout.strip().split("|") + [None] * 4)[:4]
    return {
        "server_version": parts[0] or None,
        "architecture": parts[1] or None,
        "ncpu": int(parts[2]) if parts[2] and parts[2].isdigit() else None,
        "mem_total_bytes": int(parts[3]) if parts[3] and parts[3].isdigit() else None,
    }


def provenance(root: Path, harbor: str | None, *, with_docker: bool) -> dict[str, Any]:
    """One invocation's provenance; ``root`` is the harness code's root
    (:func:`code_root`)."""
    state = source_state(root)
    lock_name, lock_sha = lockfile_info(root)
    return {
        "invocation_id": uuid.uuid4().hex,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **state,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "harbor_version": harbor_version(harbor),
        "harbor_executable": harbor,
        "lockfile": lock_name,
        "lockfile_sha256": lock_sha,
        "host_platform": platform.platform(),
        "host_machine": platform.machine(),
        "docker": docker_info() if with_docker else None,
        "argv": sys.argv,
    }


def _rel(p: Path, root: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# One sweep at a time per jobs dir
# ---------------------------------------------------------------------------


class SweepLocked(RuntimeError):
    """Another process holds the jobs dir's sweep lock."""


class SweepLock:
    """Exclusive, non-blocking ``flock`` on ``<jobs_dir>/.sweep.lock``, held
    while a sweep (or ``preflight --oracle``) changes the jobs dir.

    Without it a second invocation would see the first one's running job as
    partial, move it aside and relaunch it while the first harbor keeps
    writing trials into the new dir. The kernel drops the lock when the
    holder exits, however it exits, so a stale file never blocks anything;
    harbor jobs inherit the descriptor (``fd`` for ``pass_fds``), so the
    lock also outlives a sweep process killed while its harbor job runs.
    The file names the holder (pid, host, invocation id, command, start).
    """

    def __init__(self, jobs_dir: Path, *, invocation_id: str | None = None, what: str = "sweep"):
        self.path = jobs_dir / LOCK_NAME
        self.info = {
            "pid": os.getpid(),
            "host": platform.node(),
            "invocation_id": invocation_id,
            "what": what,
            "argv": sys.argv,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.fd: int | None = None

    def acquire(self) -> SweepLock:
        if fcntl is None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = _read_lock_info(fd)
            os.close(fd)
            raise SweepLocked(
                f"another {holder.get('what') or 'sweep'} holds {self.path}: {describe_lock_holder(holder)}. "
                "Wait for it to finish or stop it: two sweeps on one jobs dir move each other's "
                "running jobs aside and write duplicate trials."
            ) from None
        os.ftruncate(fd, 0)
        os.write(fd, (json.dumps(self.info, default=str) + "\n").encode())
        self.fd = fd
        return self

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            os.ftruncate(self.fd, 0)  # still held: nobody reads a half-cleared file
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> SweepLock:
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _read_lock_info(fd: int) -> dict[str, Any]:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        data = json.loads(os.read(fd, 65536).decode() or "{}")
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def describe_lock_holder(info: dict[str, Any]) -> str:
    if not info:
        return "holder unknown"
    cmd = " ".join(str(a) for a in info.get("argv") or [])
    return (
        f"pid {info.get('pid')} on {info.get('host')}, invocation {info.get('invocation_id')}, "
        f"since {info.get('started_at')}" + (f", command: {cmd}" if cmd else "")
    )


def lock_holder(jobs_dir: Path) -> dict[str, Any] | None:
    """Who holds ``<jobs_dir>/.sweep.lock`` right now (its info, possibly
    empty), or None if nobody does. Creates nothing."""
    path = jobs_dir / LOCK_NAME
    if fcntl is None or not path.is_file():
        return None
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return _read_lock_info(fd)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Dataset and task metadata
# ---------------------------------------------------------------------------


def dataset_cache_dir(root: Path, dataset: str) -> Path:
    """Where ``harbor dataset download`` exports a dataset:
    ``$SUBAGENTS_BENCH_DATASET_CACHE`` (or ``<root>/.cache/datasets``) /
    ``<dataset@version>``. Tasks land in ``<that>/<dataset-name>/<task>/``."""
    base = os.environ.get("SUBAGENTS_BENCH_DATASET_CACHE")
    base_dir = Path(base).expanduser() if base else root / ".cache" / "datasets"
    return base_dir / dataset


def dataset_task_root(cache_dir: Path, dataset: str) -> Path | None:
    """``<cache_dir>/<dataset-name>`` if it holds task dirs, else the first
    sub-directory that does, else None."""
    preferred = cache_dir / dataset.split("@")[0]
    if any(preferred.glob("*/task.toml")):
        return preferred
    if cache_dir.is_dir():
        for d in sorted(cache_dir.iterdir()):
            if d.is_dir() and any(d.glob("*/task.toml")):
                return d
    return None


def download_dataset(dataset: str, cache_dir: Path, harbor: str | None) -> Path:
    """``harbor dataset download <dataset> -o <cache_dir>`` (export layout).
    Idempotent: harbor skips tasks already present. Returns the task root."""
    if harbor is None:
        raise RuntimeError("harbor executable not found (run scripts/setup.sh)")
    cache_dir.mkdir(parents=True, exist_ok=True)
    r = _run([harbor, "dataset", "download", dataset, "-o", str(cache_dir)], timeout=1800)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip()[-800:]
        raise RuntimeError(f"`harbor dataset download {dataset}` failed (exit {r.returncode}): {tail}")
    task_root = dataset_task_root(cache_dir, dataset)
    if task_root is None:
        raise RuntimeError(f"download of {dataset} produced no */task.toml under {cache_dir}")
    return task_root


def parse_task_toml(path: Path) -> dict[str, Any]:
    """The task metadata fields we use; anything missing is None."""
    data = tomllib.loads(path.read_text())
    md = data.get("metadata") or {}
    tags = md.get("tags")
    return {
        "difficulty": md.get("difficulty"),
        "category": md.get("category"),
        "expert_time_min": md.get("expert_time_estimate_min"),
        "junior_time_min": md.get("junior_time_estimate_min"),
        "agent_timeout_sec": (data.get("agent") or {}).get("timeout_sec"),
        "verifier_timeout_sec": (data.get("verifier") or {}).get("timeout_sec"),
        "docker_image": (data.get("environment") or {}).get("docker_image"),
        "tags": list(tags) if isinstance(tags, list) else [],
    }


def build_tasks_meta(task_root: Path, dataset: str, tasks: Iterable[str] | None = None) -> dict[str, Any]:
    """``{"dataset", "tasks": {name: {...}}}`` from ``<task_root>/*/task.toml``.
    Restricted to ``tasks`` when given (missing ones raise KeyError)."""
    found = {p.parent.name: p for p in sorted(task_root.glob("*/task.toml"))}
    names = sorted(found) if tasks is None else list(tasks)
    missing = [t for t in names if t not in found]
    if missing:
        raise KeyError(f"tasks not in {dataset} ({task_root}): {missing}")
    return {
        "dataset": dataset,
        "source_dir": str(task_root),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks": {t: parse_task_toml(found[t]) for t in names},
    }


def filter_tasks(meta: dict[str, Any], task_filter: dict[str, Any]) -> list[str]:
    """Task names matching ``{difficulty: [...], category: [...], max_expert_min: N}``."""
    diffs = task_filter.get("difficulty")
    cats = task_filter.get("category")
    max_min = task_filter.get("max_expert_min")
    diffs = [diffs] if isinstance(diffs, str) else diffs
    cats = [cats] if isinstance(cats, str) else cats
    out = []
    for name, t in sorted(meta["tasks"].items()):
        if diffs and t.get("difficulty") not in diffs:
            continue
        if cats and t.get("category") not in cats:
            continue
        if max_min is not None and (t.get("expert_time_min") is None or t["expert_time_min"] > max_min):
            continue
        out.append(name)
    return out


def load_dataset_meta(spec: SweepSpec, harbor: str | None, *, download: bool = True) -> dict[str, Any]:
    """Metadata of every task in the spec's dataset, downloading if needed
    (falls back to an existing local export when the download fails)."""
    cache = dataset_cache_dir(spec.root, spec.dataset)
    task_root = None
    err = None
    if download:
        try:
            task_root = download_dataset(spec.dataset, cache, harbor)
        except RuntimeError as e:
            err = e
    if task_root is None:
        task_root = dataset_task_root(cache, spec.dataset)
    if task_root is None:
        raise RuntimeError(str(err) if err else f"dataset {spec.dataset} not downloaded to {cache}")
    return build_tasks_meta(task_root, spec.dataset)


def select_tasks(spec: SweepSpec, dataset_meta: dict[str, Any] | None) -> list[str]:
    if spec.tasks is not None:
        return list(spec.tasks)
    if dataset_meta is None:
        raise SpecError("`task_filter` needs task metadata: run `subagents-bench preflight SPEC` first")
    tasks = filter_tasks(dataset_meta, spec.task_filter or {})
    if not tasks:
        raise SpecError(f"task_filter {spec.task_filter} matched no task in {spec.dataset}")
    return tasks


def write_tasks_meta(spec: SweepSpec, dataset_meta: dict[str, Any], tasks: list[str]) -> Path:
    """Write ``<jobs_dir>/tasks_meta.json`` for ``tasks``, keeping the entries
    of tasks an earlier invocation wrote: their trials stay in the jobs dir
    and still need their difficulty and category."""
    missing = [t for t in tasks if t not in dataset_meta["tasks"]]
    if missing:
        raise SpecError(f"tasks not in {spec.dataset}: {missing}")
    meta = dict(dataset_meta)
    earlier = (read_tasks_meta(spec) or {}).get("tasks")
    meta["tasks"] = {**(earlier if isinstance(earlier, dict) else {}), **{t: dataset_meta["tasks"][t] for t in tasks}}
    path = spec.jobs_dir / TASKS_META_NAME
    write_json_atomic(path, meta)
    return path


def read_tasks_meta(spec: SweepSpec) -> dict[str, Any] | None:
    p = spec.jobs_dir / TASKS_META_NAME
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Rough cost / runtime estimate (dry-run only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoughModel:
    """Token and latency model behind ``sweep --dry-run``'s planning numbers.

    ROUGH, calibrated from the smoke sweep of 2026-10-11/12 (Sail balanced
    tier; fix-git and prove-plus-comm; 1 round): the 24 episodes of the
    all-tools conditions (``jobs/smoke_v1_alltools``: 133 orchestrator
    calls) and the first 2 ``manager_arbiter_hard`` episodes (``jobs/smoke``,
    one cancelled by its timeout: 63 subagent, 5 judge, 6 debate calls). Two
    easy tasks and two manager episodes are a thin base, so quote orders of
    magnitude only, and re-fit from a larger sweep's ``calls.csv`` and
    ``episodes.csv`` before quoting a budget. No all-tools orchestrator
    delegated (0 of 18 subagent-condition episodes), so the subagent, judge
    and debate numbers come from the manager episodes alone.

    Token model, per agent chain of n calls: the prompt of call k (0-based)
    is ``base + k * growth`` tokens (history grows by one step of tool output
    per call), a ``*cached_frac`` share of the orchestrator's and subagents'
    prompt tokens is billed at the cached price (judge and debate prompts are
    not cached), and every call emits a fixed number of completion tokens.
    "typical" uses the measured usage below; "upper" assumes every budget is
    exhausted (all orchestrator steps, all dispatches, all subagent steps,
    every debate round). A *manager* orchestrator (``tools.orchestrator_tools``
    without ``bash`` and ``edit_file``) must delegate; one with all its tools
    is assumed not to (``typical_dispatch_frac``).
    """

    # Orchestrator (all-tools episodes; prompt fitted per step).
    orch_prompt_base: int = 2400
    orch_prompt_growth: int = 700
    orch_completion: int = 450
    manager_orch_completion: int = 1400  # a manager writes the instructions
    # Subagent (Gemma-4-31B) chains, fitted within each dispatch.
    sub_prompt_base: int = 2000
    sub_prompt_growth: int = 500
    sub_completion: int = 300
    # Judge (the orchestrator model, reasoning: long completions) and debate.
    judge_prompt_base: int = 2000
    judge_prompt_per_candidate: int = 1200
    judge_completion: int = 4800
    debate_prompt_base: int = 1500
    debate_prompt_per_candidate: int = 1000
    debate_completion: int = 1500
    cached_frac: float = 0.75
    sub_cached_frac: float = 0.6
    # Usage. Steps: 5-step budgets ran out; single_matched finished after 12
    # of its 40 steps. Dispatches: all-tools 0 of 15; manager 5-6 of 15, with
    # ~5.7 of 8 steps per subagent; 2 of 3 arbitrations debated.
    typical_steps_frac: float = 1.0
    typical_steps_max: int = 12
    typical_dispatch_frac: float = 0.0
    manager_dispatch_frac: float = 0.4
    typical_sub_steps_frac: float = 0.7
    typical_debate_frac: float = 0.67
    # Mean seconds per call (Gemma on Sail is heavy-tailed: median 3 s, time
    # to first token up to 73 s), per tool call (own tools of the
    # orchestrator, and each subagent step's), and harbor's per-trial
    # overhead (environment start with the image present, agent setup and
    # verifier: trial time minus agent wall-clock).
    orch_call_sec: float = 6.0
    manager_orch_call_sec: float = 15.0
    sub_call_sec: float = 16.5
    judge_call_sec: float = 53.0
    debate_call_sec: float = 42.0
    tool_sec: float = 1.0
    trial_overhead_sec: float = 33.0


ROUGH_MODEL_SOURCE = "calibrated from the smoke sweep of 2026-10-11/12 (2 easy tasks), not a measurement of this sweep"


def is_manager(cfg: dict[str, Any]) -> bool:
    """A subagents condition whose orchestrator has neither ``bash`` nor
    ``edit_file`` (``tools.orchestrator_tools``), so it must delegate."""
    if cfg.get("topology", "subagents") != "subagents":
        return False
    tools = (cfg.get("tools") or {}).get("orchestrator_tools")
    return isinstance(tools, list) and not ({"bash", "edit_file"} & set(tools))


def _chain_prompt(n: float, base: int, growth: int) -> float:
    return n * base + growth * n * (n - 1) / 2 if n > 0 else 0.0


def _price(m: dict[str, Any], key: str) -> float | None:
    v = m.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _cost(m: dict[str, Any], prompt: float, cached: float, completion: float) -> float | None:
    p_in, p_out = _price(m, "price_in"), _price(m, "price_out")
    if p_in is None or p_out is None:
        return None
    p_cache = _price(m, "price_cache_read")
    p_cache = p_in if p_cache is None else p_cache
    return ((prompt - cached) * p_in + cached * p_cache + completion * p_out) / 1e6


@dataclass
class EpisodeEstimate:
    calls: dict[str, float]
    tokens: dict[str, float]
    cost: dict[str, float | None]
    seconds: float

    @property
    def cost_total(self) -> float | None:
        vals = list(self.cost.values())
        return None if any(v is None for v in vals) else sum(vals)  # type: ignore[arg-type]


def estimate_episode(cfg: dict[str, Any], rm: RoughModel = RoughModel(), *, upper: bool = False) -> EpisodeEstimate:
    """Rough per-episode calls/tokens/cost/seconds for one resolved condition."""
    orch = cfg.get("orchestrator") or {}
    budget = cfg.get("budget") or {}
    steps = int(budget.get("orchestrator_steps", 5))
    sub = cfg.get("subagent") or {}
    manager = is_manager(cfg)
    f_steps = 1.0 if upper else rm.typical_steps_frac
    f_disp = 1.0 if upper else (rm.manager_dispatch_frac if manager else rm.typical_dispatch_frac)
    f_sub = 1.0 if upper else rm.typical_sub_steps_frac
    f_deb = 1.0 if upper else rm.typical_debate_frac

    n_orch = steps * f_steps if upper else min(steps, rm.typical_steps_max) * f_steps
    p_orch = _chain_prompt(n_orch, rm.orch_prompt_base, rm.orch_prompt_growth)
    c_orch = n_orch * (rm.manager_orch_completion if manager else rm.orch_completion)
    calls = {"orch": n_orch, "sub": 0.0, "arb": 0.0, "deb": 0.0}
    tokens = {"orch": p_orch + c_orch, "sub": 0.0, "arb": 0.0, "deb": 0.0}
    cost: dict[str, float | None] = {"orch": _cost(orch, p_orch, rm.cached_frac * p_orch, c_orch)}
    seconds = rm.trial_overhead_sec + n_orch * ((rm.manager_orch_call_sec if manager else rm.orch_call_sec) + rm.tool_sec)

    if cfg.get("topology", "subagents") == "subagents":
        sub_model = sub.get("model") or orch
        max_conc = int(sub.get("max_concurrent", 3))
        max_disp = sub.get("max_dispatches")
        max_disp = int(max_disp) if max_disp is not None else max_conc * steps
        dispatches = max_disp * f_disp
        per_disp_calls = math.ceil(int(sub.get("max_steps", 8)) * f_sub)
        p_sub = dispatches * _chain_prompt(per_disp_calls, rm.sub_prompt_base, rm.sub_prompt_growth)
        c_sub = dispatches * per_disp_calls * rm.sub_completion
        calls["sub"] = dispatches * per_disp_calls
        tokens["sub"] = p_sub + c_sub
        cost["sub"] = _cost(sub_model, p_sub, rm.sub_cached_frac * p_sub, c_sub)
        chain_sec = per_disp_calls * (rm.sub_call_sec + rm.tool_sec)
        strategy = sub.get("strategy", "parallel")
        if strategy == "arbiter":
            arb = sub.get("arbiter") or {}
            n_cand = int(arb.get("n_candidates", 3))
            rounds = int(arb.get("debate_rounds", 1))
            arbitrations = dispatches / max(n_cand, 1)
            n_judge = arbitrations * (1 + f_deb * rounds)
            n_deb = arbitrations * f_deb * rounds * n_cand
            p_judge = n_judge * (rm.judge_prompt_base + rm.judge_prompt_per_candidate * n_cand)
            p_deb = n_deb * (rm.debate_prompt_base + rm.debate_prompt_per_candidate * n_cand)
            calls["arb"], calls["deb"] = n_judge, n_deb
            tokens["arb"] = p_judge + n_judge * rm.judge_completion
            tokens["deb"] = p_deb + n_deb * rm.debate_completion
            cost["arb"] = _cost(arb.get("model") or orch, p_judge, 0.0, n_judge * rm.judge_completion)
            cost["deb"] = _cost(sub_model, p_deb, 0.0, n_deb * rm.debate_completion)
            per_group = max(1, max_conc // max(n_cand, 1))
            n_groups = math.ceil(arbitrations / per_group) if arbitrations else 0
            seconds += n_groups * (chain_sec + (1 + f_deb * rounds) * rm.judge_call_sec + f_deb * rounds * rm.debate_call_sec)
        elif strategy == "sequential":
            seconds += dispatches * chain_sec
        else:
            seconds += math.ceil(dispatches / max(max_conc, 1)) * chain_sec
    return EpisodeEstimate(calls=calls, tokens=tokens, cost=cost, seconds=seconds)


# ---------------------------------------------------------------------------
# Planning and running
# ---------------------------------------------------------------------------


@dataclass
class PlannedJob:
    index: int
    round: int
    condition: str
    job: str
    job_dir: Path
    status: JobStatus
    cmd: list[str]
    config_path: Path

    @property
    def action(self) -> str:
        return {
            "complete": "skip", "partial": "move-aside+run", "needs_retry": "retry", "absent": "run",
            "duplicate": "ERROR",
        }[self.status.state]


def parse_rounds_arg(s: str) -> list[int]:
    try:
        vals = sorted({int(x) for x in s.split(",") if x.strip()})
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--rounds expects e.g. 1,2 (got {s!r})") from e
    if not vals or vals[0] < 1:
        raise argparse.ArgumentTypeError("--rounds values must be >= 1")
    return vals


def plan_jobs(
    spec: SweepSpec,
    tasks: list[str],
    *,
    only: list[str] | None = None,
    rounds: list[int] | None = None,
    harbor: str | None = None,
) -> list[PlannedJob]:
    """Every (round, condition) job in execution order, with its status."""
    names = list(spec.conditions)
    if only:
        unknown = [c for c in only if c not in spec.conditions]
        if unknown:
            raise SpecError(f"--only: unknown conditions {unknown} (known: {names})")
    sel_rounds = rounds or list(range(1, spec.rounds + 1))
    bad = [r for r in sel_rounds if r > spec.rounds]
    if bad:
        raise SpecError(f"--rounds {bad} exceed the spec's rounds ({spec.rounds})")
    order = condition_order(names, spec.rounds, spec.order_seed)
    jobs_dir_arg = _rel(spec.jobs_dir, spec.root)
    harbor = harbor or "harbor"
    plan = []
    for r in sel_rounds:
        for cname in order[r]:
            if only and cname not in only:
                continue
            job = job_name(spec.name, r, cname)
            cmd = build_harbor_command(
                harbor=harbor,
                dataset=spec.dataset,
                tasks=tasks,
                agent=AGENT_IMPORT_PATH,
                env=spec.harbor.env,
                n_concurrent=spec.harbor.n_concurrent,
                jobs_dir=jobs_dir_arg,
                job=job,
                agent_timeout_multiplier=spec.harbor.agent_timeout_multiplier,
                extra_args=spec.harbor.extra_args,
            )
            plan.append(
                PlannedJob(
                    index=len(plan) + 1,
                    round=r,
                    condition=cname,
                    job=job,
                    job_dir=spec.jobs_dir / job,
                    status=job_status(spec.jobs_dir / job),
                    cmd=cmd,
                    config_path=spec.jobs_dir / CONFIGS_DIR / f"{cname}.yaml",
                )
            )
    return plan


def read_manifest_runs(jobs_dir: Path) -> dict[str, dict[str, Any]]:
    """job name -> its last ``run`` line in ``sweep_manifest.jsonl``."""
    out: dict[str, dict[str, Any]] = {}
    try:
        lines = (jobs_dir / MANIFEST_NAME).read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("event") == "run" and rec.get("job"):
            out[rec["job"]] = rec
    return out


def cmd_settings(cmd: list[str]) -> tuple[set[str], list[str]]:
    """(task set, the other arguments) of a ``harbor run`` argv: what decides
    which trials a job ran and how. Left out: the executable, and where the
    job dir is (``-o``, ``--job-name``), so a moved jobs dir still resumes."""
    tasks: set[str] = set()
    rest: list[str] = []
    i = 1
    while i < len(cmd):
        if cmd[i] in ("-i", "-o", "--job-name") and i + 1 < len(cmd):
            if cmd[i] == "-i":
                tasks.add(cmd[i + 1])
            i += 2
            continue
        rest.append(cmd[i])
        i += 1
    return tasks, rest


def job_ran_settings(job_dir: Path, run_line: dict[str, Any] | None) -> tuple[set[str] | None, list[str] | None]:
    """(tasks, other harbor args) a finished job ran with: from its last
    manifest ``run`` line's ``cmd``, else the tasks in harbor's job
    ``config.json`` or its trials (args unknown then)."""
    if run_line and isinstance(run_line.get("cmd"), list):
        return cmd_settings([str(a) for a in run_line["cmd"]])
    cfg = _read_json(job_dir / "config.json")
    names = set()
    if isinstance(cfg, dict):
        for ds in cfg.get("datasets") or []:
            if isinstance(ds, dict):
                names.update(t for t in ds.get("task_names") or [] if isinstance(t, str))
    if not names and job_dir.is_dir():
        names = {_trial_task(d) for d in _trial_dirs(job_dir)}
    return (names or None), None


def job_source_shas(job_dir: Path, run_line: dict[str, Any] | None) -> set[str]:
    """Source fingerprints a job ran with: its manifest ``run`` line's and
    every episode's ``source_sha256`` (absent in records before schema 3)."""
    shas = set()
    if run_line and isinstance(run_line.get("source_sha256"), str):
        shas.add(run_line["source_sha256"])
    for ep in job_dir.glob("*/agent/episode.json"):
        d = _read_json(ep)
        if isinstance(d, dict) and isinstance(d.get("source_sha256"), str):
            shas.add(d["source_sha256"])
    return shas


@dataclass
class ResumeCheck:
    """What a resume would mix with the sweep's complete jobs."""

    task_problems: list[str] = field(default_factory=list)  # task list or harbor args differ
    code_problems: list[str] = field(default_factory=list)  # harness source differs
    warnings: list[str] = field(default_factory=list)
    complete_shas: set[str] = field(default_factory=set)


def _jobs_listing(jobs: list[str], limit: int = 3) -> str:
    return ", ".join(jobs[:limit]) + (f" and {len(jobs) - limit} more" if len(jobs) > limit else "")


def resume_check(
    spec: SweepSpec,
    tasks: list[str],
    plan: list[PlannedJob],
    *,
    harbor: str | None,
    current_sha: str | None,
    root: Path,
) -> ResumeCheck:
    """Compare every complete job of the sweep (all rounds and conditions,
    not only the planned ones) with what this invocation would run: the task
    set and the other ``harbor run`` arguments (from the job's last manifest
    ``run`` line), and the harness source fingerprint (manifest line and
    episodes). Differences mean the conditions would end up compared on
    different tasks, settings or code."""
    check = ResumeCheck()
    planned = {j.job: j for j in plan}
    runs = read_manifest_runs(spec.jobs_dir)
    task_diffs: dict[str, list[str]] = {}
    arg_diffs: dict[str, list[str]] = {}
    sha_jobs: dict[str, list[str]] = {}
    unknown_sha: list[str] = []
    for r in range(1, spec.rounds + 1):
        for cname in spec.conditions:
            job = job_name(spec.name, r, cname)
            job_dir = spec.jobs_dir / job
            st = planned[job].status if job in planned else job_status(job_dir)
            if not st.complete:
                continue
            new_tasks, new_rest = cmd_settings(build_harbor_command(
                harbor=harbor or "harbor", dataset=spec.dataset, tasks=tasks, agent=AGENT_IMPORT_PATH,
                env=spec.harbor.env, n_concurrent=spec.harbor.n_concurrent, jobs_dir=_rel(spec.jobs_dir, spec.root),
                job=job, agent_timeout_multiplier=spec.harbor.agent_timeout_multiplier, extra_args=spec.harbor.extra_args,
            ))
            old_tasks, old_rest = job_ran_settings(job_dir, runs.get(job))
            if old_tasks is not None and old_tasks != new_tasks:
                added, removed = sorted(new_tasks - old_tasks), sorted(old_tasks - new_tasks)
                task_diffs.setdefault(f"added {added}, removed {removed}", []).append(job)
            if old_rest is not None and old_rest != new_rest:
                arg_diffs.setdefault(f"{shlex.join(old_rest)}  ->  {shlex.join(new_rest)}", []).append(job)
            shas = job_source_shas(job_dir, runs.get(job))
            if not shas:
                unknown_sha.append(job)
            for sha in shas:
                sha_jobs.setdefault(sha, []).append(job)
    for diff, jobs in task_diffs.items():
        check.task_problems.append(f"the task list differs from {len(jobs)} complete job(s) ({_jobs_listing(jobs)}): {diff}")
    for diff, jobs in arg_diffs.items():
        check.task_problems.append(
            f"the harbor arguments differ from {len(jobs)} complete job(s) ({_jobs_listing(jobs)}): {diff}")
    check.complete_shas = set(sha_jobs)
    if current_sha:
        for sha, jobs in sha_jobs.items():
            if sha == current_sha:
                continue
            changed = source_changes(spec.jobs_dir, sha, root)
            if outside_runtime(changed):
                check.warnings.append(
                    f"the source changed since {len(jobs)} complete job(s) ran ({_jobs_listing(jobs)}), but only in "
                    f"files episodes do not use ({', '.join(_listing(changed))}): the code under test is the same"
                )
                continue
            files = (f"; changed files: {', '.join(_listing(changed))}" if changed
                     else ("; no change found" if changed == [] else ""))
            check.code_problems.append(
                f"the harness source changed since {len(jobs)} complete job(s) ran ({_jobs_listing(jobs)}): "
                f"source_sha256 {sha[:12]} -> {current_sha[:12]}{files}"
            )
    if unknown_sha:
        check.warnings.append(
            f"{len(unknown_sha)} complete job(s) ({_jobs_listing(unknown_sha)}) record no source fingerprint "
            "(an older launcher and runtime): cannot check that they ran the current code"
        )
    return check


def _fmt_usd(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.2f}" if x >= 1 else f"${x:.3f}"


def _fmt_hours(sec: float) -> str:
    return f"{sec / 3600:.1f} h"


def print_plan(
    spec: SweepSpec,
    tasks: list[str],
    plan: list[PlannedJob],
    resolved: dict[str, dict[str, Any]],
    problems: dict[str, list[str]],
    *,
    harbor: str | None,
    check: ResumeCheck | None = None,
    holder: dict[str, Any] | None = None,
    out: Callable[[str], None] = print,
) -> None:
    """The --dry-run report: settings, validation, resume checks, jobs +
    commands, estimate."""
    n_tasks = len(tasks)
    to_run = [j for j in plan if j.status.state not in ("complete", "duplicate")]
    out(f"Sweep {spec.name!r}  (spec {_rel(spec.spec_path, spec.root)})")
    out(f"  dataset     {spec.dataset}")
    out(f"  tasks       {n_tasks}: {', '.join(tasks)}")
    out(f"  conditions  {len(spec.conditions)}; rounds {spec.rounds}; order_seed {spec.order_seed}")
    out(f"  jobs_dir    {_rel(spec.jobs_dir, spec.root)}")
    out(
        f"  harbor      env={spec.harbor.env} n_concurrent={spec.harbor.n_concurrent} "
        f"agent_timeout_multiplier={spec.harbor.agent_timeout_multiplier} executable={harbor or 'NOT FOUND'}"
    )
    if holder is not None:
        out(f"  NOTE        a sweep holds {_rel(spec.jobs_dir / LOCK_NAME, spec.root)} ({describe_lock_holder(holder)}); "
            "a real run now is refused")
    out("")
    bad = {c: p for c, p in problems.items() if p}
    if bad:
        out(f"Config validation: {len(bad)} condition(s) with problems")
        for c, ps in bad.items():
            for p in ps:
                out(f"  {c}: {p}")
    else:
        out(f"Config validation: OK ({len(resolved)} conditions load into ExperimentConfig losslessly)")
    if check is not None:
        issues = (
            [f"REFUSE (unless --allow-task-change): {p}" for p in check.task_problems]
            + [f"REFUSE (unless --allow-code-change): {p}" for p in check.code_problems]
            + [f"WARNING: {w}" for w in check.warnings]
        )
        out("Resume check: " + ("OK" if not issues else f"{len(issues)} issue(s)"))
        for line in issues:
            out(f"  {line}")
    out("")
    n_skip = sum(1 for j in plan if j.status.state == "complete")
    n_part = sum(1 for j in plan if j.status.state == "partial")
    n_retry = sum(1 for j in plan if j.status.state == "needs_retry")
    n_dup = sum(1 for j in plan if j.status.state == "duplicate")
    out(
        f"Plan: {len(plan)} jobs ({len(to_run)} to run, {n_skip} complete -> skip, "
        f"{n_part} partial -> move aside and rerun, {n_retry} to retry -> move aside and rerun"
        + (f", {n_dup} with duplicate trials -> not run (ERROR)" if n_dup else "")
        + f"); {len(to_run) * n_tasks} episodes to run"
    )
    for j in plan:
        out(f"  [{j.index:>3}] r{j.round} {j.condition:<28} {j.action:<15} {j.job}  ({j.status.reason})")
        for trial, reasons in list(j.status.retry_trials.items())[:5]:
            out(f"        rerun: {trial}: {', '.join(reasons)}")
        env_prefix = f"SUBAGENTS_BENCH_CONFIG={shlex.quote(_rel(j.config_path, spec.root))}"
        out(f"        $ {env_prefix} {shlex.join(j.cmd)}")
    out("")
    out(f"ROUGH estimate -- token/latency model {ROUGH_MODEL_SOURCE} (subagents_bench.sweep.RoughModel):")
    header = f"  {'condition':<28} {'jobs':>4} {'episodes':>8} {'$/ep typ':>9} {'$/ep upper':>10} {'$ typ':>9} {'$ upper':>9} {'min/ep typ':>10}"
    out(header)
    tot_typ: float | None = 0.0
    tot_up: float | None = 0.0
    tot_sec = 0.0
    waves = math.ceil(n_tasks / max(spec.harbor.n_concurrent, 1))
    for cname in spec.conditions:
        jobs = [j for j in to_run if j.condition == cname]
        if not jobs:
            continue
        eps = len(jobs) * n_tasks
        typ = estimate_episode(resolved[cname])
        up = estimate_episode(resolved[cname], upper=True)
        ct, cu = typ.cost_total, up.cost_total
        tot_typ = None if (tot_typ is None or ct is None) else tot_typ + ct * eps
        tot_up = None if (tot_up is None or cu is None) else tot_up + cu * eps
        tot_sec += len(jobs) * waves * typ.seconds
        out(
            f"  {cname:<28} {len(jobs):>4} {eps:>8} {_fmt_usd(ct):>9} {_fmt_usd(cu):>10} "
            f"{_fmt_usd(None if ct is None else ct * eps):>9} {_fmt_usd(None if cu is None else cu * eps):>9} "
            f"{typ.seconds / 60:>10.1f}"
        )
    out(f"  TOTAL: {_fmt_usd(tot_typ)} typical, {_fmt_usd(tot_up)} upper (all budgets exhausted)")
    out(
        f"  Runtime: ~{_fmt_hours(tot_sec)} typical, sequential jobs, each ~{waves} wave(s) of "
        f"{spec.harbor.n_concurrent} concurrent trials (first run also pulls task images)."
    )


def write_sweep_json(
    spec: SweepSpec,
    *,
    tasks: list[str],
    config_paths: dict[str, Path],
    resolved: dict[str, dict[str, Any]],
    prov: dict[str, Any],
    status: str,
    harbor: str | None,
    extra_invocation: dict[str, Any] | None = None,
) -> Path:
    """Write (or update) ``<jobs_dir>/sweep.json``. Earlier invocations of a
    resumed sweep are kept under ``invocations``; the newest provenance is
    also at top level. ``source_sha256s`` lists the harness source
    fingerprints the sweep's jobs ran with (manifest ``run`` lines and
    episodes); ``mixed_source`` is set when they hold more than one version
    of the code under test (:func:`runtime_versions`), so an analysis-only
    edit between jobs adds a fingerprint but does not set it."""
    path = spec.jobs_dir / SWEEP_JSON_NAME
    old: dict[str, Any] = {}
    if path.is_file():
        try:
            old = json.loads(path.read_text())
        except (OSError, ValueError):
            old = {}
    invocations = list(old.get("invocations") or [])
    inv = dict(prov)
    inv["status"] = status
    inv.update(extra_invocation or {})
    if invocations and invocations[-1].get("invocation_id") == prov.get("invocation_id"):
        invocations[-1] = inv  # same invocation, status update
    else:
        invocations.append(inv)
    commits = sorted({i.get("git_commit") for i in invocations if i.get("git_commit")})
    tiers = sorted(
        {model_key(m) for cfg in resolved.values() for m in models_used(cfg)},
        key=lambda k: tuple(str(x) for x in k),
    )
    all_jobs = [job_name(spec.name, r, c) for r in range(1, spec.rounds + 1) for c in spec.conditions]
    runs = read_manifest_runs(spec.jobs_dir)
    shas = sorted({sha for j in all_jobs if (spec.jobs_dir / j).is_dir() for sha in job_source_shas(spec.jobs_dir / j, runs.get(j))})
    obj = {
        "sweep_json_version": 1,
        "name": spec.name,
        # Outcome of the latest invocation (which may have been an --only /
        # --rounds subset); jobs_complete / jobs_total is the whole sweep.
        "status": status,
        "jobs_total": len(all_jobs),
        "jobs_complete": sum(job_status(spec.jobs_dir / j).complete for j in all_jobs),
        "spec_path": _rel(spec.spec_path, spec.root),
        "spec_files": {_rel(p, spec.root): sha256_file(p) for p in spec.source_files},
        "base_config": _rel(spec.base_path, spec.root) if spec.base_path else None,
        "base_config_sha256": sha256_file(spec.base_path) if spec.base_path else None,
        "spec": spec.raw,
        "dataset": spec.dataset,
        "tasks": tasks,
        "rounds": spec.rounds,
        "order_seed": spec.order_seed,
        "condition_order": {str(r): o for r, o in condition_order(list(spec.conditions), spec.rounds, spec.order_seed).items()},
        "conditions": {
            c: {"config_path": _rel(p, spec.root), "config_sha256": sha256_file(p)} for c, p in config_paths.items()
        },
        "provider_tiers": [{"model": k[0], "provider": k[1], "completion_window": k[2]} for k in tiers],
        "harbor": {**dataclasses.asdict(spec.harbor), "agent": AGENT_IMPORT_PATH, "executable": harbor},
        "jobs_dir": _rel(spec.jobs_dir, spec.root),
        "tasks_meta_path": _rel(spec.jobs_dir / TASKS_META_NAME, spec.root),
        "manifest_path": _rel(spec.jobs_dir / MANIFEST_NAME, spec.root),
        "provenance": prov,
        "source_sha256s": shas,
        "mixed_source": len(runtime_versions(spec.jobs_dir, shas)) > 1,
        "mixed_git_commits": len(commits) > 1,
        "invocations": invocations,
    }
    write_json_atomic(path, obj)
    return path


def _manifest_record(spec: SweepSpec, j: PlannedJob, event: str, **kw: Any) -> dict[str, Any]:
    rec = {
        "sweep": spec.name,
        "job": j.job,
        "round": j.round,
        "condition": j.condition,
        "event": event,
        "cmd": j.cmd,
        "config_path": _rel(j.config_path, spec.root),
        "t_start": None,
        "t_end": None,
        "duration_sec": None,
        "return_code": None,
        "n_trials": j.status.n_total,
        "n_completed": j.status.n_completed,
        "n_errors": j.status.n_errored,
        "complete": j.status.complete,
    }
    rec.update(kw)
    return rec


def run_sweep(
    spec: SweepSpec,
    *,
    only: list[str] | None = None,
    rounds: list[int] | None = None,
    dry_run: bool = False,
    allow_config_change: bool = False,
    allow_task_change: bool = False,
    allow_code_change: bool = False,
    max_consecutive_failures: int = 3,
    dotenv_path: Path | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """Run (or with ``dry_run`` only print) every planned job. Exit codes:
    0 all selected jobs complete, 1 some failed or still need a retry,
    2 config/spec error, a refused resume (changed task list, harbor
    arguments or code) or another sweep holding the jobs dir, 3 aborted on a
    billing error, 4 aborted on provider failures in over half a job's
    trials, 130 interrupted."""
    load_env_files(dotenv_path)
    harbor = harbor_executable()
    resolved = resolve_conditions(spec)
    problems: dict[str, list[str]] = {}
    full: dict[str, dict[str, Any]] = {}
    for c, cfg in resolved.items():
        problems[c], f = validate_condition(cfg)
        full[c] = f if (f is not None and not problems[c]) else cfg

    dataset_meta = read_tasks_meta(spec) if spec.tasks is None else None
    if spec.tasks is None and dataset_meta is None and not dry_run:
        dataset_meta = load_dataset_meta(spec, harbor)
    tasks = select_tasks(spec, dataset_meta)
    root = code_root()

    if dry_run:
        plan = plan_jobs(spec, tasks, only=only, rounds=rounds, harbor=harbor)
        check = resume_check(spec, tasks, plan, harbor=harbor, current_sha=source_prov.source_fingerprint(root), root=root)
        print_plan(spec, tasks, plan, resolved, problems, harbor=harbor, check=check,
                   holder=lock_holder(spec.jobs_dir), out=out)
        refused = (check.task_problems and not allow_task_change) or (check.code_problems and not allow_code_change)
        return EXIT_CONFIG if (any(problems.values()) or refused) else EXIT_OK

    if any(problems.values()):
        for c, ps in problems.items():
            for p in ps:
                out(f"config problem: {c}: {p}")
        out("Refusing to run: fix the configs above (see `subagents-bench sweep SPEC --dry-run`).")
        return EXIT_CONFIG
    if harbor is None:
        out("harbor executable not found; run scripts/setup.sh")
        return EXIT_CONFIG

    prov = provenance(root, harbor, with_docker=spec.harbor.env == "docker")
    try:
        lock = SweepLock(spec.jobs_dir, invocation_id=prov["invocation_id"]).acquire()
    except SweepLocked as e:
        out(f"Refusing to run: {e}")
        return EXIT_CONFIG
    try:
        return _run_sweep_locked(
            spec, tasks, dataset_meta, full, harbor, prov, root, lock,
            only=only, rounds=rounds, allow_config_change=allow_config_change, allow_task_change=allow_task_change,
            allow_code_change=allow_code_change, max_consecutive_failures=max_consecutive_failures, out=out,
        )
    finally:
        lock.release()


def _run_sweep_locked(
    spec: SweepSpec,
    tasks: list[str],
    dataset_meta: dict[str, Any] | None,
    full: dict[str, dict[str, Any]],
    harbor: str,
    prov: dict[str, Any],
    root: Path,
    lock: SweepLock,
    *,
    only: list[str] | None,
    rounds: list[int] | None,
    allow_config_change: bool,
    allow_task_change: bool,
    allow_code_change: bool,
    max_consecutive_failures: int,
    out: Callable[[str], None],
) -> int:
    """:func:`run_sweep` once the jobs dir's lock is held: plan (job states
    cannot change under us now), check the resume, then run job by job."""
    plan = plan_jobs(spec, tasks, only=only, rounds=rounds, harbor=harbor)
    check = resume_check(spec, tasks, plan, harbor=harbor, current_sha=prov.get("source_sha256"), root=root)
    for w in check.warnings:
        out(f"WARNING: {w}")
    refusals = [*([] if allow_task_change else check.task_problems), *([] if allow_code_change else check.code_problems)]
    if refusals:
        for r in refusals:
            out(f"resume refused: {r}")
        out(
            "Refusing to run: these jobs would be compared with jobs that ran differently. Restore what changed, "
            "use a new jobs_dir, or pass --allow-task-change / --allow-code-change to mix them anyway "
            "(sweep.json then records it)."
        )
        return EXIT_CONFIG
    for p in [*check.task_problems, *check.code_problems]:
        out(f"WARNING (allowed): {p}")

    config_paths = materialize_configs(spec, full, allow_change=allow_config_change)

    meta_path = spec.jobs_dir / TASKS_META_NAME
    existing_meta = read_tasks_meta(spec)
    if not (existing_meta and all(t in existing_meta.get("tasks", {}) for t in tasks)):
        dataset_meta = dataset_meta or load_dataset_meta(spec, harbor)
        meta_path = write_tasks_meta(spec, dataset_meta, tasks)

    need_prime = any(model_key(m)[1] in PRIME_PROVIDERS for cfg in full.values() for m in models_used(cfg))
    inv_id = prov["invocation_id"]
    record = write_source_record(spec.jobs_dir, root, prov, inv_id)
    prov.update(source_files_record=record["files"], source_patch=record["patch"])
    harbor_settings = dataclasses.asdict(spec.harbor)
    inv_extra: dict[str, Any] = {"only": only, "rounds": rounds, "tasks": tasks, "harbor": harbor_settings}
    sweep_json = write_sweep_json(
        spec, tasks=tasks, config_paths=config_paths, resolved=full, prov=prov, status="running",
        harbor=harbor, extra_invocation=inv_extra,
    )
    if json.loads(sweep_json.read_text()).get("mixed_source"):
        out("WARNING: this sweep's jobs ran more than one version of the harness code (sweep.json source_sha256s).")
    if record["patch"]:
        out(f"Uncommitted harness source saved to {_rel(spec.jobs_dir / record['patch'], spec.root)}")
    manifest = spec.jobs_dir / MANIFEST_NAME
    out(f"Sweep {spec.name!r}: {len(plan)} jobs, tasks metadata {_rel(meta_path, spec.root)}")

    n_ran = n_skipped = n_failed = consecutive = 0
    current_sha = prov.get("source_sha256")
    final = "complete"
    code = EXIT_OK
    for j in plan:
        prefix = f"[{j.index}/{len(plan)}] {j.job}"
        if j.status.complete:
            out(f"{prefix}: complete ({j.status.reason}) -> skip")
            append_manifest(manifest, _manifest_record(spec, j, "skipped_complete", t_start=time.time()))
            n_skipped += 1
            continue
        if j.status.state == "duplicate":
            out(f"{prefix}: ERROR: {j.status.reason} (two harbor runs wrote into it?). Not run: inspect it, "
                f"move it out of {_rel(spec.jobs_dir, spec.root)} and re-run to rerun the job.")
            append_manifest(manifest, _manifest_record(spec, j, "duplicate_trials", t_start=time.time(),
                                                       status_reason=j.status.reason))
            n_failed += 1
            continue
        # The code harbor imports is whatever is on disk when the job starts.
        state = source_state(root)
        sha = state.get("source_sha256")
        if sha and current_sha and sha != current_sha:
            changed = source_changes(spec.jobs_dir, current_sha, root)
            what = (f"source_sha256 {current_sha[:12]} -> {sha[:12]}"
                    + (f"; changed files: {', '.join(_listing(changed))}" if changed else ""))
            if outside_runtime(changed):
                out(f"NOTE: only files episodes do not use changed during the sweep ({what}); going on.")
            elif not allow_code_change:
                out(f"ABORT: the harness source changed during the sweep ({what}). Finish or revert the edit, "
                    "then re-run to resume (or pass --allow-code-change to mix code versions).")
                final, code = "aborted_code_change", EXIT_CONFIG
                break
            else:
                out(f"WARNING (allowed): the harness source changed during the sweep ({what}).")
            write_source_record(spec.jobs_dir, root, state, f"{inv_id}.{j.job}")
            current_sha = sha
        if j.status.state in ("partial", "needs_retry"):
            kind = "partial" if j.status.state == "partial" else "retry"
            moved = move_aside(j.job_dir, spec.jobs_dir, kind)
            if kind == "partial":
                out(f"{prefix}: partial ({j.status.reason}) -> moved to {_rel(moved, spec.root)}, rerunning")
                append_manifest(manifest, _manifest_record(
                    spec, j, "moved_partial", t_start=time.time(), moved_to=_rel(moved, spec.root)))
            else:
                out(f"{prefix}: needs a retry ({j.status.reason}) -> moved to {_rel(moved, spec.root)}, rerunning")
                append_manifest(manifest, _manifest_record(
                    spec, j, "retried", t_start=time.time(), moved_to=_rel(moved, spec.root),
                    retry_trials=j.status.retry_trials, status_reason=j.status.reason))
        env = build_subprocess_env(j.config_path, need_prime=need_prime)
        out(f"{prefix}: running ({len(tasks)} trials)")
        t0 = time.time()
        try:
            rc = run_job_process(j.cmd, env=env, cwd=spec.root, pass_fds=(lock.fd,) if lock.fd is not None else ())
        except KeyboardInterrupt:
            st = job_status(j.job_dir)
            append_manifest(
                manifest,
                _manifest_record(
                    spec, j, "interrupted", t_start=t0, t_end=time.time(), duration_sec=time.time() - t0,
                    n_trials=st.n_total, n_completed=st.n_completed, n_errors=st.n_errored, complete=st.complete,
                ),
            )
            out(f"{prefix}: interrupted; re-run the same command to resume.")
            final, code = "interrupted", EXIT_INTERRUPTED
            break
        t1 = time.time()
        st = job_status(j.job_dir)
        billing = scan_billing_errors(j.job_dir)
        retry = trial_retry_reasons(j.job_dir)
        n_provider = sum(1 for reasons in retry.values() if RETRY_PROVIDER in reasons)
        n_episodes = len(list(j.job_dir.glob("*/agent/episode.json"))) if j.job_dir.is_dir() else 0
        append_manifest(
            manifest,
            _manifest_record(
                spec, j, "run", t_start=t0, t_end=t1, duration_sec=t1 - t0, return_code=rc,
                n_trials=st.n_total, n_completed=st.n_completed, n_errors=st.n_errored, complete=st.complete,
                state=st.state, status_reason=st.reason, n_episode_json=n_episodes, billing_error_trials=billing,
                retry_trials=retry, n_provider_error_trials=n_provider, invocation_id=inv_id, tasks=tasks,
                harbor=harbor_settings,
                source_sha256=sha, git_commit=state.get("git_commit"), git_dirty=state.get("git_dirty"),
            ),
        )
        n_ran += 1
        out(f"{prefix}: exit {rc}, {st.state} ({st.reason}), {t1 - t0:.0f}s")
        if billing:
            out(f"ABORT: billing error in {j.job}: {billing}. Top up the provider account, then re-run to resume "
                "(the job is rerun).")
            final, code = "aborted_billing", EXIT_BILLING
            break
        n_total = st.n_total or len(tasks)
        if n_provider * 2 > n_total:
            out(f"ABORT: provider failures in {n_provider}/{n_total} trials of {j.job}: likely an outage (or a "
                "rejected key: 401/403, see the trials' calls.jsonl errors). Re-run later to resume (the job is rerun).")
            final, code = "aborted_provider", EXIT_PROVIDER
            break
        if st.complete and rc == 0:
            consecutive = 0
        else:
            if st.state == "needs_retry":
                out(f"{prefix}: {_retry_summary(st.retry_trials)}; re-run the sweep to retry this job.")
            n_failed += 1
            consecutive += 1
            if consecutive >= max_consecutive_failures:
                out(f"ABORT: {consecutive} consecutive jobs failed or need a retry; check Docker, harbor and the "
                    f"providers (see {j.job_dir}/job.log and the reasons above).")
                final, code = "aborted_failures", EXIT_FAIL
                break
    else:
        if n_failed:
            final, code = "incomplete", EXIT_FAIL

    write_sweep_json(
        spec, tasks=tasks, config_paths=config_paths, resolved=full, prov=prov, status=final,
        harbor=harbor, extra_invocation={**inv_extra, "n_ran": n_ran, "n_skipped": n_skipped, "n_failed": n_failed},
    )
    out(f"Sweep {spec.name!r} {final}: ran {n_ran}, skipped {n_skipped} complete, failed {n_failed}.")
    if code == EXIT_OK:
        out(f"Next: scripts/analyze.sh {_rel(spec.jobs_dir, spec.root)}")
    elif final == "incomplete":
        out("Re-run the same command to rerun the jobs that failed or need a retry.")
    return code


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    required: bool = True


def _distinct_models(resolved: dict[str, dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    out: dict[tuple, dict[str, Any]] = {}
    for cfg in resolved.values():
        for m in models_used(cfg):
            out.setdefault(model_key(m), m)
    return out


def _model_configs(resolved: dict[str, dict[str, Any]]) -> dict[tuple, Any]:
    """Distinct ModelConfig objects (via ExperimentConfig) keyed by model_key."""
    out: dict[tuple, Any] = {}
    for cfg in resolved.values():
        exp = load_experiment_config(cfg)
        cands = [exp.orchestrator]
        if cfg.get("topology", "subagents") == "subagents":
            cands.append(exp.resolved_subagent_model())
            arb = getattr(exp.subagent, "arbiter", None)
            if (cfg.get("subagent") or {}).get("strategy") == "arbiter":
                cands.append(getattr(arb, "model", None) or exp.orchestrator)
        for m in cands:
            key = (m.model, m.provider, getattr(m, "completion_window", None))
            out.setdefault(key, m)
    return out


def check_credentials(models: dict[tuple, dict[str, Any]]) -> list[Check]:
    """One check per provider in use; reports set/missing, never values."""
    checks = []
    providers = sorted({k[1] for k in models})
    for prov in providers:
        if prov in PRIME_PROVIDERS:
            key, team = prime_credentials()
            ok = bool(key) and bool(team)
            detail = f"PRIME_API_KEY {'set' if key else 'missing'}, PRIME_TEAM_ID {'set' if team else 'missing'} (env or ~/.prime/config.json)"
            checks.append(Check(f"credentials: {prov}", ok, detail))
            continue
        envs = {m.get("api_key_env") for k, m in models.items() if k[1] == prov and m.get("api_key_env")}
        if not envs:
            default = {"sail": "SAIL_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(prov)
            envs = {default} if default else set()
        if not envs:
            checks.append(Check(f"credentials: {prov}", True, "no API key variable configured (e.g. local vLLM)", required=False))
            continue
        missing = sorted(e for e in envs if not os.environ.get(e))
        checks.append(
            Check(
                f"credentials: {prov}",
                not missing,
                ", ".join(f"{e} {'missing' if e in missing else 'set'}" for e in sorted(envs)),
            )
        )
    return checks


def ping_model(model_cfg: Any) -> dict[str, Any]:
    """One tiny streaming request through ``llm.make_client``. Returns a
    JSON-able record; credentials never appear in it."""
    from .llm import make_client
    from .types import Message

    changes = {"max_output_tokens": PING_MAX_TOKENS}
    if hasattr(model_cfg, "max_retries"):
        changes["max_retries"] = 2
    cfg = dataclasses.replace(model_cfg, **changes)
    rec: dict[str, Any] = {
        "model": cfg.model,
        "provider": cfg.provider,
        "completion_window": getattr(cfg, "completion_window", None),
        "ok": False,
    }
    t0 = time.time()
    try:
        client = make_client(cfg)
        resp = client.complete(
            "You are a connectivity check.",
            [Message(role="user", content="Reply with the single word OK.")],
            [],
        )
    except Exception as e:  # report any client failure as a failed check
        rec.update(error_type=type(e).__name__, error=str(e)[:500], latency_sec=time.time() - t0)
        return rec
    timing = getattr(resp, "timing", None)
    usage = getattr(resp, "usage", None)
    t_req = getattr(timing, "t_request_sent", None)
    t_first = getattr(timing, "t_first_chunk", None)
    rec.update(
        ok=True,
        latency_sec=time.time() - t0,
        ttft_sec=(t_first - t_req) if (t_first is not None and t_req is not None) else None,
        streamed=getattr(timing, "streamed", None),
        n_chunks=getattr(timing, "n_chunks", None),
        n_attempts=len(getattr(resp, "attempts", []) or []) or None,
        finish_reason=getattr(resp, "finish_reason", None) or getattr(resp, "stop_reason", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        cached_tokens=getattr(usage, "cached_tokens", None),
        reasoning_tokens=getattr(usage, "reasoning_tokens", None),
        usage_source=getattr(resp, "usage_source", None),
        text_chars=len(getattr(resp, "text", "") or ""),
    )
    return rec


def _oracle_rewards(job_dir: Path, tasks: list[str]) -> tuple[dict[str, dict[str, Any]], list[str], float | None]:
    """(trial rewards, tasks without a 1.0 reward on every trial, mean reward)."""
    rewards = trial_rewards(job_dir)
    by_task: dict[str, list[float]] = {t: [] for t in tasks}
    for r in rewards.values():
        if r["task"] in by_task:
            by_task[r["task"]].append(float(r["reward"]) if isinstance(r["reward"], (int, float)) else 0.0)
    failing = sorted(t for t, rs in by_task.items() if not rs or min(rs) < 1.0)
    all_r = [x for rs in by_task.values() for x in rs]
    return rewards, failing, (sum(all_r) / len(all_r) if all_r else None)


def run_oracle(
    spec: SweepSpec, tasks: list[str], harbor: str, out: Callable[[str], None] = print, *, redo: bool = False,
    pass_fds: tuple[int, ...] = (),
) -> tuple[bool, dict[str, Any]]:
    """``harbor run -a oracle`` on every task into ``<jobs_dir>/_oracle``.
    An earlier oracle job is reused only if it finished and passed on every
    task; otherwise (and always with ``redo``) it is moved aside and rerun,
    so a transient failure (an image pull timeout, too little memory) is not
    cached. Passes iff every task's reward is 1.0."""
    job_dir = spec.jobs_dir / ORACLE_JOB
    st = job_status(job_dir)
    moved = None
    _, old_failing, _ = _oracle_rewards(job_dir, tasks)
    if st.state != "absent" and (redo or not st.complete or old_failing):
        # Unfinished, failed (maybe transiently), run for an older task list
        # (harbor will not reuse a job dir whose config changed), or --redo-oracle:
        # keep it aside and rerun.
        why = "--redo-oracle" if redo else (f"failing: {old_failing}" if st.complete else st.reason)
        moved = move_aside(job_dir, spec.jobs_dir, "partial" if st.state == "partial" else "retry")
        out(f"  oracle: earlier job ({why}) moved to {_rel(moved, spec.root)}")
        st = job_status(job_dir)
    reused = st.complete
    rc = None
    if not reused:
        cmd = build_harbor_command(
            harbor=harbor, dataset=spec.dataset, tasks=tasks, agent="oracle", env=spec.harbor.env,
            n_concurrent=spec.harbor.n_concurrent, jobs_dir=_rel(spec.jobs_dir, spec.root), job=ORACLE_JOB,
            extra_args=spec.harbor.extra_args,
        )
        out(f"  oracle: $ {shlex.join(cmd)}")
        rc = run_job_process(cmd, env=build_subprocess_env(None), cwd=spec.root, pass_fds=pass_fds)
    rewards, failing, mean = _oracle_rewards(job_dir, tasks)
    info = {
        "job_dir": _rel(job_dir, spec.root),
        "reused": reused,
        "return_code": rc,
        "moved_partial_to": _rel(moved, spec.root) if moved else None,
        "mean_reward": mean,
        "failing_tasks": failing,
        "trials": rewards,
    }
    return (not failing and mean == 1.0), info


def run_preflight(
    spec: SweepSpec,
    *,
    oracle: bool = False,
    redo_oracle: bool = False,
    skip_llm: bool = False,
    dotenv_path: Path | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """Run every check, write ``<jobs_dir>/preflight.json``, print a
    PASS/FAIL checklist. Returns 0 iff every required check passed, 2 if
    the oracle job is needed and another sweep holds the jobs dir.
    ``redo_oracle`` implies ``oracle``."""
    oracle = oracle or redo_oracle
    lock = None
    if oracle:
        # The oracle job moves and writes <jobs_dir>/_oracle: one process at a time.
        try:
            lock = SweepLock(spec.jobs_dir, what="preflight --oracle").acquire()
        except SweepLocked as e:
            out(f"Refusing to run preflight --oracle: {e}")
            return EXIT_CONFIG
    try:
        return _run_preflight(spec, oracle=oracle, redo_oracle=redo_oracle, skip_llm=skip_llm,
                              dotenv_path=dotenv_path, lock=lock, out=out)
    finally:
        if lock is not None:
            lock.release()


def _run_preflight(
    spec: SweepSpec,
    *,
    oracle: bool,
    redo_oracle: bool,
    skip_llm: bool,
    dotenv_path: Path | None,
    lock: SweepLock | None,
    out: Callable[[str], None],
) -> int:
    checks: list[Check] = []
    record: dict[str, Any] = {"spec": _rel(spec.spec_path, spec.root), "date": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    env_status = load_env_files(dotenv_path)
    record["dotenv"] = {"path": str(dotenv_path) if dotenv_path else None, "keys": env_status}

    # Docker (only for the local docker environment).
    if spec.harbor.env == "docker":
        di = docker_info()
        record["docker"] = di
        checks.append(Check("docker running", di is not None, (f"server {di['server_version']} ({di['architecture']})" if di else "`docker info` failed: start Docker")))
    else:
        checks.append(Check("docker running", True, f"skipped (harbor env {spec.harbor.env!r})", required=False))

    # harbor
    harbor = harbor_executable()
    hv = harbor_version(harbor)
    record["harbor"] = {"executable": harbor, "version": hv}
    checks.append(Check("harbor installed", harbor is not None and hv is not None, f"{harbor} version {hv}"))
    if hv is not None and hv != PINNED_HARBOR_VERSION:
        checks.append(Check("harbor version pinned", False, f"found {hv}, study pins {PINNED_HARBOR_VERSION} (re-run scripts/setup.sh)"))

    # Configs
    resolved = resolve_conditions(spec)
    bad = {}
    for c, cfg in resolved.items():
        ps, _ = validate_condition(cfg)
        if ps:
            bad[c] = ps
    checks.append(Check("configs load", not bad, f"{len(resolved)} conditions" if not bad else f"problems: {bad}"))

    # Dataset + tasks
    tasks: list[str] = list(spec.tasks or [])
    try:
        dataset_meta = load_dataset_meta(spec, harbor)
        tasks = select_tasks(spec, dataset_meta)
        meta_path = write_tasks_meta(spec, dataset_meta, tasks)
        checks.append(Check("dataset + tasks", True, f"{spec.dataset}: {len(tasks)} tasks -> {_rel(meta_path, spec.root)}"))
    except (RuntimeError, SpecError, KeyError) as e:
        checks.append(Check("dataset + tasks", False, str(e)))

    # Credentials
    models = _distinct_models(resolved)
    checks.extend(check_credentials(models))

    # One tiny streaming request per distinct (model, provider, tier)
    pings = []
    if skip_llm:
        checks.append(Check("model requests", True, "skipped (--skip-llm)", required=False))
    elif bad:
        checks.append(Check("model requests", False, "skipped: configs do not load"))
    else:
        try:
            mcfgs = _model_configs(resolved)
        except Exception as e:  # config loader problems are reported above
            mcfgs = {}
            checks.append(Check("model requests", False, f"cannot build model configs: {e}"))
        for key, mcfg in mcfgs.items():
            rec = ping_model(mcfg)
            pings.append(rec)
            label = f"request {key[0]} @ {key[1]}" + (f"/{key[2]}" if key[2] else "")
            detail = (
                f"{rec.get('latency_sec', 0):.1f}s, finish={rec.get('finish_reason')}, usage={rec.get('usage_source')}"
                if rec["ok"]
                else f"{rec.get('error_type')}: {rec.get('error')}"
            )
            checks.append(Check(label, rec["ok"], detail))
    record["requests"] = pings

    # Oracle feasibility
    if oracle:
        if harbor is None or not tasks:
            checks.append(Check("oracle solves all tasks", False, "needs harbor and the task list"))
        else:
            fds = (lock.fd,) if lock is not None and lock.fd is not None else ()
            ok, info = run_oracle(spec, tasks, harbor, out=out, redo=redo_oracle, pass_fds=fds)
            record["oracle"] = info
            detail = f"mean reward {info['mean_reward']}" + (" (reused earlier _oracle job)" if info["reused"] else "")
            if info["failing_tasks"]:
                detail += f"; failing: {info['failing_tasks']}"
            checks.append(Check("oracle solves all tasks", ok, detail))

    record["checks"] = [dataclasses.asdict(c) for c in checks]
    passed = all(c.ok for c in checks if c.required)
    record["passed"] = passed
    write_json_atomic(spec.jobs_dir / PREFLIGHT_JSON_NAME, record)

    out("")
    out(f"Preflight for {spec.name!r}:")
    for c in checks:
        tag = "PASS" if c.ok else ("FAIL" if c.required else "WARN")
        out(f"  [{tag}] {c.name}: {c.detail}")
    out(f"  -> {'ALL PASSED' if passed else 'FAILED'}; details in {_rel(spec.jobs_dir / PREFLIGHT_JSON_NAME, spec.root)}")
    return EXIT_OK if passed else EXIT_FAIL


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_dotenv(spec: SweepSpec, arg: str | None) -> Path:
    return Path(arg) if arg else spec.root / ".env"


def _cmd_preflight(args: argparse.Namespace) -> int:
    try:
        spec = load_spec(args.spec)
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    return run_preflight(spec, oracle=args.oracle, redo_oracle=args.redo_oracle, skip_llm=args.skip_llm,
                         dotenv_path=_default_dotenv(spec, args.env_file))


def _cmd_sweep(args: argparse.Namespace) -> int:
    try:
        spec = load_spec(args.spec)
        return run_sweep(
            spec,
            only=args.only,
            rounds=args.rounds,
            dry_run=args.dry_run,
            allow_config_change=args.allow_config_change,
            allow_task_change=args.allow_task_change,
            allow_code_change=args.allow_code_change,
            max_consecutive_failures=args.max_consecutive_failures,
            dotenv_path=_default_dotenv(spec, args.env_file),
        )
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return EXIT_CONFIG


def _cmd_tasks(args: argparse.Namespace) -> int:
    root = default_root()
    load_env_files(root / ".env")  # may set SUBAGENTS_BENCH_DATASET_CACHE
    cache = Path(args.cache_dir) / args.dataset if args.cache_dir else dataset_cache_dir(root, args.dataset)
    task_root = None
    if not args.offline:
        try:
            task_root = download_dataset(args.dataset, cache, harbor_executable())
        except RuntimeError as e:
            print(f"download failed ({e}); trying the local copy", file=sys.stderr)
    task_root = task_root or dataset_task_root(cache, args.dataset)
    if task_root is None:
        print(f"dataset {args.dataset} not available under {cache}", file=sys.stderr)
        return EXIT_FAIL
    try:
        meta = build_tasks_meta(task_root, args.dataset, args.tasks)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return EXIT_FAIL
    if args.out:
        write_json_atomic(Path(args.out), meta)
    if args.json:
        print(json.dumps(meta, indent=2))
        return EXIT_OK
    print(f"{args.dataset}: {len(meta['tasks'])} tasks ({task_root})")
    print(f"{'task':<36} {'difficulty':<10} {'category':<24} {'expert':>6} {'junior':>6} {'agent_s':>7} {'verif_s':>7}  docker_image")
    for name, t in meta["tasks"].items():
        def num(x: Any) -> str:
            return "-" if x is None else f"{x:g}"

        print(
            f"{name:<36} {str(t['difficulty']):<10} {str(t['category']):<24} {num(t['expert_time_min']):>6} "
            f"{num(t['junior_time_min']):>6} {num(t['agent_timeout_sec']):>7} {num(t['verifier_timeout_sec']):>7}  {t['docker_image']}"
        )
    return EXIT_OK


def register_cli(subparsers: Any) -> None:
    p = subparsers.add_parser("preflight", help="check Docker, harbor, dataset, credentials and models before a sweep")
    p.add_argument("spec", help="sweep spec YAML (configs/sweeps/*.yaml)")
    p.add_argument("--oracle", action="store_true",
                   help="also run harbor's oracle agent on every task (reuses an earlier _oracle job only if it passed)")
    p.add_argument("--redo-oracle", action="store_true", help="rerun the oracle job even if an earlier one passed (implies --oracle)")
    p.add_argument("--skip-llm", action="store_true", help="skip the one-request-per-model check")
    p.add_argument("--env-file", default=None, help="dotenv file to load (default: <repo>/.env)")
    p.set_defaults(func=_cmd_preflight)

    s = subparsers.add_parser("sweep", help="run (or --dry-run) every round x condition harbor job of a sweep spec")
    s.add_argument("spec", help="sweep spec YAML (configs/sweeps/*.yaml)")
    s.add_argument("--dry-run", action="store_true", help="print the plan, commands and a rough cost estimate; run nothing")
    s.add_argument("--only", nargs="+", metavar="COND", help="run only these conditions")
    s.add_argument("--rounds", type=parse_rounds_arg, help="comma-separated rounds to run, e.g. 1,2")
    s.add_argument("--allow-config-change", action="store_true", help="allow a condition's resolved config to change on resume")
    s.add_argument("--allow-task-change", action="store_true",
                   help="allow the task list or harbor arguments to differ from the sweep's complete jobs")
    s.add_argument("--allow-code-change", action="store_true",
                   help="allow the harness source to differ from the code the sweep's complete jobs ran")
    s.add_argument("--max-consecutive-failures", type=int, default=3, help="abort after this many failed jobs in a row (default 3)")
    s.add_argument("--env-file", default=None, help="dotenv file to load (default: <repo>/.env)")
    s.set_defaults(func=_cmd_sweep)

    t = subparsers.add_parser("tasks", help="print Terminal-Bench task metadata (difficulty, category, times, image)")
    t.add_argument("dataset", nargs="?", default=DEFAULT_DATASET, help=f"dataset name@version (default {DEFAULT_DATASET})")
    t.add_argument("--tasks", nargs="+", metavar="TASK", help="only these tasks")
    t.add_argument("--json", action="store_true", help="print the tasks_meta.json document")
    t.add_argument("--out", default=None, help="also write the metadata JSON here")
    t.add_argument("--cache-dir", default=None, help="dataset export dir (default <repo>/.cache/datasets)")
    t.add_argument("--offline", action="store_true", help="use the local export only, no download")
    t.set_defaults(func=_cmd_tasks)


def main(argv: list[str] | None = None) -> int:
    """``python -m subagents_bench.sweep {preflight,sweep,tasks} ...``: the
    same subcommands without importing the rest of the CLI."""
    parser = argparse.ArgumentParser(prog="python -m subagents_bench.sweep")
    register_cli(parser.add_subparsers(dest="cmd", required=True))
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
