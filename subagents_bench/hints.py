"""SWE-sweep subtasks as the timed user messages of an interactive episode.

A SWE-sweep task dir holds ``tests/swesweep_eval.json``: its subtasks (bugs), each with an id, the
hidden tests that fail before the fix (``f2p``) and the files the reference fix changes
(``target_files``). The harness reads it on the host (the agent's container has no ``tests/``),
orders the subtasks by a per-task seed (the same order in every condition) and turns subtask k
into the hint sent at ``k * interval_sec``:

- ``files``: the files the fix changes;
- ``tests``: the files, plus the hidden tests' ids, which name the behaviour (most repositories'
  test names are descriptive; goawk's ``interp::TestInterp`` and libexpat's ``runtests`` are not);
- ``report``: a short bug report written beforehand from the hidden test patch by
  ``subagents-bench hints`` (:func:`generate_reports`), read from ``<hints_dir>/<task>.json``.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import InteractiveConfig
from .llm import Message

EVAL_JSON = Path("tests") / "swesweep_eval.json"
HINTS_DIR_ENV = "SUBAGENTS_BENCH_HINTS_DIR"  # where harbor's agent finds the generated reports (the sweep sets it)
MAX_TESTS_SHOWN = 3
PATCH_CHARS = 20000  # of a hidden test patch shown to the report writer


class HintError(ValueError):
    """A task without usable subtasks, or a missing generated report."""


@dataclass(frozen=True)
class Subtask:
    id: str
    f2p: tuple[str, ...]
    target_files: tuple[str, ...]


@dataclass(frozen=True)
class Hint:
    index: int  # 0-based position in the schedule
    t_offset: float  # seconds after the episode start
    subtask: Subtask
    text: str


def load_subtasks(task_dir: str | Path) -> list[Subtask]:
    """The subtasks of a SWE-sweep task dir, in the file's order."""
    path = Path(task_dir) / EVAL_JSON
    try:
        raw = json.loads(path.read_text())["subtasks"]
        subtasks = [Subtask(str(s["id"]), tuple(map(str, s.get("f2p") or ())), tuple(map(str, s.get("target_files") or ())))
                    for s in raw]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise HintError(f"{path}: no SWE-sweep subtasks ({type(e).__name__}: {e})") from None
    if not subtasks or len({s.id for s in subtasks}) != len(subtasks):
        raise HintError(f"{path}: expected a non-empty list of subtasks with distinct ids")
    return subtasks


def ordered(subtasks: list[Subtask], task_name: str, seed: int) -> list[Subtask]:
    """A shuffle of the subtasks (by id) that depends only on the task and the seed."""
    out = sorted(subtasks, key=lambda s: s.id)
    random.Random(f"{seed}:{task_name}").shuffle(out)
    return out


def test_label(test_id: str) -> str:
    """A test id without its parameters: ``a::test_x[p0]`` -> ``a::test_x``."""
    return re.sub(r"\[.*\]$", "", test_id).strip()


def _code_list(items: list[str]) -> str:
    quoted = [f"`{x}`" for x in items]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def hint_text(subtask: Subtask, detail: str, report: str | None = None) -> str:
    if detail == "report":
        if not report:
            raise HintError(f"subtask {subtask.id}: no generated report (run `subagents-bench hints`)")
        return report.strip()
    files = list(subtask.target_files)
    where = (f"There is a bug in {_code_list(files)}." if len(files) == 1
             else f"There is a bug whose fix touches {_code_list(files)}." if files else "There is a bug in the repository.")
    if detail == "files":
        return where
    labels = list(dict.fromkeys(test_label(t) for t in subtask.f2p if test_label(t)))
    if not labels:
        return where
    more = f" (and {len(labels) - MAX_TESTS_SHOWN} more)" if len(labels) > MAX_TESTS_SHOWN else ""
    tests = _code_list(labels[:MAX_TESTS_SHOWN]) + more
    if len(labels) == 1:
        return (f"{where} Once it is fixed, the test {tests} should pass; it is not in the repository yet, so its "
                "name only describes the expected behaviour.")
    return (f"{where} Once it is fixed, the tests {tests} should pass; they are not in the repository yet, so their "
            "names only describe the expected behaviour.")


def n_slots(cfg: InteractiveConfig) -> int:
    """Hints that fit in the session: offsets 0, interval, ... below the wall clock."""
    return max(1, math.ceil(cfg.wall_clock_sec / cfg.interval_sec - 1e-9))


def schedule(cfg: InteractiveConfig, task_dir: str | Path, task_name: str, hints_dir: str | Path | None = None,
             ) -> list[Hint]:
    """The episode's hints: the ordered subtasks, as many as fit (and ``max_hints``)."""
    subtasks = ordered(load_subtasks(task_dir), task_name, cfg.order_seed)
    n = min(len(subtasks), n_slots(cfg), cfg.max_hints or len(subtasks))
    reports = load_reports(hints_dir, task_name) if cfg.hint_detail == "report" else {}
    return [Hint(k, k * cfg.interval_sec, s, hint_text(s, cfg.hint_detail, reports.get(s.id)))
            for k, s in enumerate(subtasks[:n])]


def reports_path(hints_dir: str | Path, task_name: str) -> Path:
    return Path(hints_dir) / f"{task_name}.json"


def load_reports(hints_dir: str | Path | None, task_name: str) -> dict[str, str]:
    if hints_dir is None:
        raise HintError("hint_detail report needs a hints dir (SUBAGENTS_BENCH_HINTS_DIR; the sweep sets it)")
    path = reports_path(hints_dir, task_name)
    try:
        reports = json.loads(path.read_text())["reports"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise HintError(f"{path}: no generated reports ({type(e).__name__}: {e}); run `subagents-bench hints`") from None
    if not isinstance(reports, dict):
        raise HintError(f"{path}: `reports` must map subtask ids to text")
    return {str(k): str(v) for k, v in reports.items()}


REPORT_SYSTEM = """\
You write bug reports. You are given the files a bug fix changed and the new tests that \
check the fix (a diff that adds them). Write the report a user of the project would file \
for this bug: one to three sentences on the observable wrong behaviour (what they did, what \
they expected, what happened instead) and the module or file involved. Do not describe the \
fix, do not quote the tests and do not mention that tests exist. Reply with the report only.
"""


def report_prompt(subtask: Subtask, test_patch: str) -> str:
    if len(test_patch) > PATCH_CHARS:
        test_patch = test_patch[:PATCH_CHARS] + "\n[... truncated ...]"
    return (f"Files the fix changed: {', '.join(subtask.target_files) or '(not recorded)'}\n"
            f"Tests that fail before the fix: {', '.join(subtask.f2p) or '(not recorded)'}\n\n"
            f"Diff adding the tests:\n{test_patch}")


def generate_reports(task_dir: str | Path, client: Any, existing: dict[str, str] | None = None) -> dict[str, str]:
    """A bug report per subtask of ``task_dir`` (those in ``existing`` are kept), written by ``client``."""
    out = dict(existing or {})
    for s in load_subtasks(task_dir):
        if out.get(s.id):
            continue
        patch_path = Path(task_dir) / "tests" / "subtasks" / s.id / "test.patch"
        patch = patch_path.read_text(errors="replace") if patch_path.is_file() else "(no test patch)"
        r = client.complete(REPORT_SYSTEM, [Message("user", report_prompt(s, patch))], [])
        text = r.text.strip()
        if not text:
            raise HintError(f"subtask {s.id}: the model returned an empty report (finish_reason {r.finish_reason})")
        out[s.id] = text
    return out


def write_reports(hints_dir: str | Path, task_name: str, model: str, reports: dict[str, str]) -> Path:
    path = reports_path(hints_dir, task_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task": task_name, "model": model, "generated_at": time.time(),
                                "reports": reports}, indent=2) + "\n")
    return path
