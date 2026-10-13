"""Find the harbor trials of sweep jobs dirs and load their ``result.json`` and episode records.

Layout: ``<jobs_dir>/<sweep>__r<round>__<condition>/<trial>/{config.json, result.json, agent/}``.
Dirs starting with ``_`` (moved-aside partial jobs, the oracle run) or ``.`` are skipped. A task's
bugs (SWE-sweep) come from ``<jobs_dir>/tasks.json``, which the sweep writes from the task dirs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..records import KINDS

JOB_RE = re.compile(r"^(?P<sweep>.+)__r(?P<round>\d+)__(?P<condition>.+)$")
RECORD_KINDS = tuple(k for k in KINDS if k != "transcript")


@dataclass
class Trial:
    dir: Path
    sweep: str
    round: int | None
    condition: str | None
    result: dict | None  # harbor's result.json; None while the trial runs
    episode: dict | None  # agent/episode.json; None if the agent never started
    records: dict[str, list[dict]]  # kind -> records (empty lists without an episode)
    n_subtasks: int | None = None  # the task's bugs (SWE-sweep), from the sweep's tasks.json

    @property
    def task_name(self) -> str:
        r, ep = self.result or {}, self.episode or {}
        return r.get("task_name") or ep.get("task_name") or self.dir.name.rsplit("__", 1)[0]


def _json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def load_trial(trial_dir: Path) -> Trial:
    job = trial_dir.parent
    m = JOB_RE.match(job.name)
    episode = _json(trial_dir / "agent" / "episode.json")
    t = Trial(
        dir=trial_dir,
        sweep=m["sweep"] if m else job.parent.name,
        round=int(m["round"]) if m else None,
        condition=m["condition"] if m else (episode or {}).get("condition", job.name),
        result=_json(trial_dir / "result.json"),
        episode=episode,
        records={k: _jsonl(trial_dir / "agent" / f"{k}.jsonl") for k in RECORD_KINDS},
    )
    t.n_subtasks = ((_json(job.parent / "tasks.json") or {}).get("n_subtasks") or {}).get(t.task_name)
    return t


def load_trials(jobs_dir: str | Path) -> list[Trial]:
    """Every trial dir (a dir with harbor's ``config.json``) of every job under ``jobs_dir``."""
    return [load_trial(t)
            for job in sorted(Path(jobs_dir).iterdir()) if job.is_dir() and not job.name.startswith(("_", "."))
            for t in sorted(job.iterdir()) if t.is_dir() and (t / "config.json").is_file()]
