"""Find harbor trials / episode directories on disk and load their records.

Layouts understood (see ``docs/DESIGN.md`` sections 4-6):

* a **jobs dir** (sweep root): ``<jobs_dir>/{sweep_manifest.jsonl,
  tasks_meta.json, sweep.json, <job_name>/...}``;
* a **job dir**: ``<job_name>/{result.json, config.json, job.log,
  <trial_name>/...}``;
* a **trial dir**: ``<trial_name>/{result.json, config.json, agent/...}``
  where ``agent/`` is the episode dir (``episode.json`` + ``*.jsonl``);
* a bare **episode dir** (the local runner's
  ``<output_dir>/<name>/<task_id>/<episode_id>/``).

Everything is loaded leniently: a missing ``result.json``, a trial whose
environment never came up (no ``agent/``), an episode that crashed while
``status == "running"`` or a JSONL file whose last line was cut off all load
as far as the data goes, with the gaps recorded rather than papered over.

Joins (per trial): harbor ``result.json`` (reward, exception, phase timings),
``tasks_meta.json`` (difficulty, category, expert time) and
``sweep_manifest.jsonl`` (round, condition). When the manifest is missing the
job name ``<sweep>__r<round>__<condition>`` is parsed instead; when that fails
too, the condition falls back to the episode's ``experiment`` (config name).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

# Long and short role names; the short names prefix episodes.csv columns.
ROLE_SHORT = {"orchestrator": "orch", "subagent": "sub", "arbiter": "arb", "debate": "deb"}
ROLES = tuple(ROLE_SHORT)

RECORD_FILES = {
    "calls": "calls.jsonl",
    "tools": "tools.jsonl",
    "steps": "steps.jsonl",
    "spawns": "spawns.jsonl",
    "dispatches": "dispatches.jsonl",
    "arbitrations": "arbitrations.jsonl",
}

JOB_NAME_RE = re.compile(r"^(?P<sweep>.+?)__r(?P<round>\d+)__(?P<condition>.+)$")


def parse_job_name(name: str | None) -> tuple[str | None, int | None, str | None]:
    """``<sweep>__r<round>__<condition>`` -> (sweep, round, condition); Nones if no match."""
    if not name:
        return (None, None, None)
    m = JOB_NAME_RE.match(name)
    if not m:
        return (None, None, None)
    return (m.group("sweep"), int(m.group("round")), m.group("condition"))


def parse_ts(value: Any) -> datetime | None:
    """Parse harbor's ISO timestamps ('2026-10-12T00:52:53.700275Z'); None if absent/bad."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def phase_seconds(result: dict | None, phase: str | None) -> float | None:
    """Duration of a harbor phase ({started_at, finished_at}); ``phase=None`` -> whole trial."""
    if not result:
        return None
    obj = result if phase is None else result.get(phase)
    if not isinstance(obj, dict):
        return None
    t0, t1 = parse_ts(obj.get("started_at")), parse_ts(obj.get("finished_at"))
    if t0 is None or t1 is None:
        return None
    # Mixed naive/aware timestamps cannot be subtracted; treat as unknown.
    try:
        return (t1 - t0).total_seconds()
    except TypeError:
        return None


def read_json(path: Path) -> tuple[Any, str | None]:
    """(data, problem). Missing file -> (None, None); unreadable -> (None, message)."""
    if not path.exists():
        return (None, None)
    try:
        return (json.loads(path.read_text()), None)
    except (OSError, ValueError) as e:
        return (None, f"{path.name}: unreadable JSON ({e})")


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """Records of a JSONL file plus problems (bad lines). A missing file is empty.

    A crashed writer can leave a partial last line; it is skipped and reported.
    """
    records: list[dict] = []
    problems: list[str] = []
    if not path.exists():
        return records, problems
    try:
        lines = path.read_text().splitlines()
    except OSError as e:
        return records, [f"{path.name}: unreadable ({e})"]
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            where = "last line (truncated write?)" if i == len(lines) else f"line {i}"
            problems.append(f"{path.name}: invalid JSON on {where}")
            continue
        if not isinstance(obj, dict):
            problems.append(f"{path.name}:{i}: not a JSON object")
            continue
        records.append(obj)
    return records, problems


@dataclass
class EpisodeData:
    """Everything one episode dir holds (transcripts excluded unless asked)."""

    dir: Path
    episode: dict | None
    calls: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    spawns: list[dict] = field(default_factory=list)
    dispatches: list[dict] = field(default_factory=list)
    arbitrations: list[dict] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.episode is None:
            return "missing"
        return str(self.episode.get("status") or "unknown")

    @property
    def episode_id(self) -> str | None:
        if self.episode and self.episode.get("episode_id"):
            return self.episode["episode_id"]
        for recs in (self.calls, self.tools):
            if recs:
                return recs[0].get("episode_id")
        return None


def load_episode(episode_dir: str | Path, with_transcript: bool = False) -> EpisodeData:
    d = Path(episode_dir)
    episode, problem = read_json(d / "episode.json")
    data = EpisodeData(dir=d, episode=episode if isinstance(episode, dict) else None)
    if problem:
        data.problems.append(problem)
    for attr, fname in RECORD_FILES.items():
        recs, probs = read_jsonl(d / fname)
        setattr(data, attr, recs)
        data.problems.extend(probs)
    if with_transcript:
        data.transcript, probs = read_jsonl(d / "transcript.jsonl")
        data.problems.extend(probs)
    return data


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass
class TrialRef:
    """Where one trial (or bare episode) lives. ``trial_dir`` is None for bare episodes."""

    trial_dir: Path | None
    episode_dir: Path | None

    @property
    def job_dir(self) -> Path | None:
        return self.trial_dir.parent if self.trial_dir is not None else None

    @property
    def jobs_root(self) -> Path | None:
        return self.trial_dir.parent.parent if self.trial_dir is not None else None


def _json_keys(path: Path) -> set[str]:
    data, _ = read_json(path)
    return set(data) if isinstance(data, dict) else set()


def is_trial_dir(d: Path) -> bool:
    """A harbor trial dir: has ``agent/episode.json``, or a trial-shaped result/config.json."""
    if (d / "agent" / "episode.json").is_file():
        return True
    if "trial_name" in _json_keys(d / "result.json"):
        return True
    keys = _json_keys(d / "config.json")
    return "trial_name" in keys and "task" in keys


def _is_bare_episode_dir(d: Path) -> bool:
    return (d / "episode.json").is_file() and not (d.name == "agent" and is_trial_dir(d.parent))


def discover(paths: Iterable[str | Path]) -> list[TrialRef]:
    """All trials / bare episodes under the given jobs, job, trial or episode dirs."""
    found: dict[Path, TrialRef] = {}

    def add_trial(t: Path) -> None:
        t = t.resolve()
        ep = t / "agent"
        found.setdefault(t, TrialRef(trial_dir=t, episode_dir=ep if (ep / "episode.json").is_file() else None))

    def add_episode(e: Path) -> None:
        e = e.resolve()
        found.setdefault(e, TrialRef(trial_dir=None, episode_dir=e))

    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"no such directory: {p}")
        if p.name == "agent" and (p / "episode.json").is_file() and is_trial_dir(p.parent):
            add_trial(p.parent)
            continue
        if is_trial_dir(p):
            add_trial(p)
            continue
        if _is_bare_episode_dir(p):
            add_episode(p)
            continue
        for root, dirs, _files in os.walk(p):
            r = Path(root)
            # Skip hidden dirs and the sweep launcher's side dirs (``_partial/``
            # holds moved-aside partial jobs, ``_oracle/`` the feasibility run):
            # counting them would double-count trials.
            dirs[:] = sorted(x for x in dirs if not x.startswith((".", "_")))
            if r == p:
                continue
            if is_trial_dir(r):
                add_trial(r)
                dirs[:] = []
            elif _is_bare_episode_dir(r):
                add_episode(r)
                dirs[:] = []
    return sorted(found.values(), key=lambda t: str(t.trial_dir or t.episode_dir))


# ---------------------------------------------------------------------------
# Sweep-level context (manifest, task metadata)
# ---------------------------------------------------------------------------


@dataclass
class SweepContext:
    """Per-jobs-dir files written by the sweep launcher (all optional)."""

    root: Path | None
    manifest: dict[str, dict] = field(default_factory=dict)  # job name -> its last `run` manifest line
    tasks_meta: dict[str, dict] = field(default_factory=dict)  # task_name -> metadata
    dataset: str | None = None
    sweep_json: dict | None = None
    problems: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path | None, tasks_meta_path: Path | None = None) -> "SweepContext":
        ctx = cls(root=root)
        if root is not None:
            recs, probs = read_jsonl(root / "sweep_manifest.jsonl")
            ctx.problems.extend(probs)
            # One line per job event (``event``: run | skipped_complete |
            # moved_partial | interrupted); round/condition/config_path agree
            # across a job's lines. The last ``run`` line wins (a resumed
            # sweep can re-run a job), else the last line of any event.
            ran: dict[str, dict] = {}
            for r in recs:
                job = r.get("job") or r.get("job_name")
                if not job:
                    continue
                ctx.manifest[job] = r
                if r.get("event", "run") == "run":
                    ran[job] = r
            ctx.manifest.update(ran)
            sj, prob = read_json(root / "sweep.json")
            ctx.sweep_json = sj if isinstance(sj, dict) else None
            if prob:
                ctx.problems.append(prob)
        meta_path = tasks_meta_path or (root / "tasks_meta.json" if root is not None else None)
        if meta_path is not None:
            meta, prob = read_json(Path(meta_path))
            if prob:
                ctx.problems.append(prob)
            if isinstance(meta, dict):
                ctx.dataset = meta.get("dataset")
                tasks = meta.get("tasks")
                if isinstance(tasks, dict):
                    ctx.tasks_meta = {k: v for k, v in tasks.items() if isinstance(v, dict)}
        return ctx

    def sweep_name(self) -> str | None:
        if self.sweep_json:
            spec = self.sweep_json.get("spec")
            if isinstance(spec, dict) and spec.get("name"):
                return str(spec["name"])
            if self.sweep_json.get("name"):
                return str(self.sweep_json["name"])
        return None


# ---------------------------------------------------------------------------
# Trials: one joined view per trial
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    ref: TrialRef
    result: dict | None  # harbor trial result.json
    data: EpisodeData | None  # None when there is no episode dir at all
    ctx: SweepContext
    problems: list[str] = field(default_factory=list)

    @property
    def episode(self) -> dict | None:
        return self.data.episode if self.data is not None else None

    @property
    def job_name(self) -> str | None:
        return self.ref.job_dir.name if self.ref.job_dir is not None else None

    @property
    def manifest(self) -> dict | None:
        return self.ctx.manifest.get(self.job_name or "")

    @property
    def trial_name(self) -> str | None:
        if self.result and self.result.get("trial_name"):
            return self.result["trial_name"]
        if self.ref.trial_dir is not None:
            return self.ref.trial_dir.name
        return (self.episode or {}).get("trial_name")

    @property
    def task_name(self) -> str | None:
        if self.result and self.result.get("task_name"):
            return self.result["task_name"]
        if self.episode and self.episode.get("task_name"):
            return self.episode["task_name"]
        tn = self.trial_name
        return tn.split("__")[0] if tn else None

    @property
    def sweep(self) -> str | None:
        parsed = parse_job_name(self.job_name)[0]
        if parsed:
            return parsed
        name = self.ctx.sweep_name()
        if name:
            return name
        return self.ctx.root.name if self.ctx.root is not None else None

    @property
    def round(self) -> int | None:
        m = self.manifest
        if m and m.get("round") is not None:
            try:
                return int(m["round"])
            except (TypeError, ValueError):
                pass
        return parse_job_name(self.job_name)[1]

    @property
    def condition(self) -> str | None:
        m = self.manifest
        if m and m.get("condition"):
            return str(m["condition"])
        parsed = parse_job_name(self.job_name)[2]
        if parsed:
            return parsed
        ep = self.episode or {}
        if ep.get("experiment"):
            return str(ep["experiment"])
        cfg = self.config
        return cfg.get("name") if cfg else None

    @property
    def task_meta(self) -> dict:
        return self.ctx.tasks_meta.get(self.task_name or "", {})

    @property
    def config(self) -> dict | None:
        """The resolved ExperimentConfig: from episode.json, else the sweep's materialized config
        (manifest ``config_path``, or ``<jobs_dir>/configs/<condition>.yaml``)."""
        ep = self.episode
        if ep and isinstance(ep.get("config"), dict):
            return ep["config"]
        candidates: list[Path] = []
        m = self.manifest
        if m and m.get("config_path"):
            p = Path(m["config_path"])
            candidates += [p] if p.is_absolute() else [p, Path.cwd() / p]
            if self.ctx.root is not None:
                candidates.append(self.ctx.root / "configs" / p.name)
        # Not self.condition: its last fallback reads this config.
        cond = (m or {}).get("condition") or parse_job_name(self.job_name)[2]
        if cond and self.ctx.root is not None:
            candidates.append(self.ctx.root / "configs" / f"{cond}.yaml")
        for c in candidates:
            if c.is_file():
                try:
                    cfg = yaml.safe_load(c.read_text())
                except (OSError, yaml.YAMLError):
                    return None
                return cfg if isinstance(cfg, dict) else None
        return None


def load_trials(paths: Iterable[str | Path], tasks_meta_path: str | Path | None = None) -> list[Trial]:
    """Discover and load every trial under ``paths`` (records, result.json, sweep context)."""
    contexts: dict[Path | None, SweepContext] = {}
    meta = Path(tasks_meta_path) if tasks_meta_path else None
    trials = []
    for ref in discover(paths):
        root = ref.jobs_root
        if root not in contexts:
            contexts[root] = SweepContext.load(root, meta)
        ctx = contexts[root]
        result, problems = None, []
        if ref.trial_dir is not None:
            res, prob = read_json(ref.trial_dir / "result.json")
            result = res if isinstance(res, dict) else None
            if prob:
                problems.append(prob)
        data = load_episode(ref.episode_dir) if ref.episode_dir is not None else None
        trials.append(Trial(ref=ref, result=result, data=data, ctx=ctx, problems=problems))
    return trials
