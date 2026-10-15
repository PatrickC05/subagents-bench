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

With ``interactive.replies`` the schedule also holds questions about the code (``questions_per_interval``
between consecutive bugs, from ``<hints_dir>/<task>.questions.json``, written by ``subagents-bench
questions`` from the task's source tree) and status checks (``status_every``); every message gets an
id (``msg_id``, 1, 2, ... in sending order) and ``urgent_every`` marks some bugs urgent.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import re
import subprocess
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
class Question:
    id: str
    file: str  # the source file it is about (relative to the repository root)
    question: str
    answer: str  # the reference answer the grader compares with


@dataclass(frozen=True)
class Hint:
    """One scheduled user message: a bug (``subtask``), a ``question`` or a ``status`` check."""

    index: int  # 0-based position among the messages of its kind
    t_offset: float  # seconds after the episode start
    subtask: Subtask | None
    text: str
    kind: str = "bug"
    msg_id: int | None = None  # the id the agent replies to (with interactive.replies)
    urgent: bool = False
    question: Question | None = None


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
    # Some hidden tests extend a test the repository already has (echo's TestContext_Scheme gains
    # cases): "the version that checks it" holds for both.
    if len(labels) == 1:
        return (f"{where} Once it is fixed, the test {tests} should pass; the version of it that checks this is not "
                "in the repository, so its name only describes the expected behaviour.")
    return (f"{where} Once it is fixed, the tests {tests} should pass; the versions of them that check this are not "
            "in the repository, so their names only describe the expected behaviour.")


def n_slots(cfg: InteractiveConfig) -> int:
    """Hints that fit in the session: offsets 0, interval, ... below the wall clock."""
    return max(1, math.ceil(cfg.wall_clock_sec / cfg.interval_sec - 1e-9))


def order_seed(cfg: InteractiveConfig, round_: int | None) -> int:
    """The seed of a sweep round's order: ``order_seed + round - 1`` with ``vary_order_by_round``, so
    the conditions of a round share an order and rounds sample different ones."""
    return cfg.order_seed + (round_ - 1 if cfg.vary_order_by_round and round_ else 0)


def schedule(cfg: InteractiveConfig, task_dir: str | Path, task_name: str, hints_dir: str | Path | None = None,
             round_: int | None = None) -> list[Hint]:
    """The episode's messages in sending order: the ordered subtasks, as many as fit (and
    ``max_hints``), sent every ``interval_sec``; with ``replies``, the questions and status checks
    between them and an id per message."""
    seed = order_seed(cfg, round_)
    subtasks = ordered(load_subtasks(task_dir), task_name, seed)
    n = min(len(subtasks), n_slots(cfg), cfg.max_hints or len(subtasks))
    reports = load_reports(hints_dir, task_name) if cfg.hint_detail == "report" else {}
    out = [Hint(k, k * cfg.interval_sec, s, hint_text(s, cfg.hint_detail, reports.get(s.id)),
                urgent=bool(cfg.urgent_every) and k > 0 and k % cfg.urgent_every == 0)
           for k, s in enumerate(subtasks[:n])]
    if not cfg.replies:
        return out
    q, extra = cfg.questions_per_interval, []
    slots = [k * cfg.interval_sec + j * cfg.interval_sec / (q + 1) for k in range(n) for j in range(1, q + 1)]
    slots = [t for t in slots if t < cfg.wall_clock_sec]
    if slots:
        questions = load_questions(hints_dir, task_name)
        random.Random(f"{seed}:{task_name}:questions").shuffle(questions)
        if len(questions) < len(slots):
            raise HintError(f"{task_name}: {len(slots)} question slots but {len(questions)} questions "
                            "(run `subagents-bench questions` with a larger --per-task)")
        extra += [Hint(j, t, None, x.question, kind="question", question=x) for j, (t, x) in enumerate(zip(slots, questions))]
    if cfg.status_every:
        checks = [(k + 0.5) * cfg.interval_sec for k in range(n) if (k + 1) % cfg.status_every == 0]
        extra += [Hint(j, t, None, "", kind="status") for j, t in enumerate(c for c in checks if c < cfg.wall_clock_sec)]
    kinds = {"bug": 0, "status": 1, "question": 2}  # at the same time: the bug first
    out = sorted(out + extra, key=lambda h: (h.t_offset, kinds[h.kind], h.index))
    return [dataclasses.replace(h, msg_id=i + 1) for i, h in enumerate(out)]


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


# -- questions about the code (interactive.questions_per_interval) -----------------------------

QUESTION_SUFFIXES = (".py", ".go", ".js", ".ts", ".c", ".h", ".rs", ".java", ".rb")
QUESTION_FILE_CHARS = (1500, 40000)  # files outside this size give trivial or unreadable questions
QUESTIONS_PER_FILE = 2

QUESTION_SYSTEM = """\
You write questions a developer new to a repository asks a colleague who has it open. You are \
given one source file. Write {k} questions about it, each answerable from this file alone in a \
sentence or two by someone who reads the relevant function: what a function returns or raises \
in a given case, a default value, which function or class handles something, what an argument \
controls. Each question names the file or the function so it can be found. The answer must be \
exact and unambiguous: a value, a name, an exception type, a short behaviour. Ask about what \
the code does, not about style or history. Reply with only a JSON list of objects with keys \
"question" and "answer"."""

QUESTION_JUDGE = """\
You grade answers to questions about a code repository. You get the question, a reference \
answer and the answer to grade. The answer is correct if it states what the reference states \
(other wording, extra correct detail and a longer explanation are fine); it is wrong if it \
contradicts the reference, misses its key fact, or only says how to find out. Reply with only a \
JSON object {"correct": true or false, "why": "<one sentence>"}."""


def questions_path(hints_dir: str | Path, task_name: str) -> Path:
    return Path(hints_dir) / f"{task_name}.questions.json"


def load_questions(hints_dir: str | Path | None, task_name: str) -> list[Question]:
    if hints_dir is None:
        raise HintError("questions need a hints dir (SUBAGENTS_BENCH_HINTS_DIR; the sweep sets it)")
    path = questions_path(hints_dir, task_name)
    try:
        raw = json.loads(path.read_text())["questions"]
        return [Question(str(x["id"]), str(x["file"]), str(x["question"]), str(x["answer"])) for x in raw]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise HintError(f"{path}: no questions ({type(e).__name__}: {e}); run `subagents-bench questions`") from None


def source_ref(task_dir: str | Path) -> tuple[str, str, str]:
    """(git url, ref, workdir in the image) of a SWE-sweep task: its Dockerfile clones the
    repository at a tag and copies the tree to the image's working directory."""
    text = (Path(task_dir) / "environment" / "Dockerfile").read_text()
    url = re.search(r"git clone\b[^\n]*?\s(https?://\S+)\s+/src\b", text)
    ref = re.search(r"git -C /src checkout(?:\s+--quiet)?\s+(\S+)", text)
    workdir = re.search(r"^WORKDIR\s+(\S+)", text, re.M)
    if not (url and ref):
        raise HintError(f"{task_dir}: no `git clone URL /src` and checkout in the Dockerfile")
    return url.group(1), ref.group(1), workdir.group(1) if workdir else "/"


def fetch_source(task_dir: str | Path, dest: str | Path) -> Path:
    """The task's source tree at its ref, cloned to ``dest`` once."""
    dest = Path(dest)
    if (dest / ".git").is_dir():
        return dest
    url, ref, _ = source_ref(task_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", ref, url, str(dest)], check=True,
                   capture_output=True, timeout=600)
    return dest


SKIP_DIRS = ("doc", "docs", "example", "examples", "vendor", "node_modules", "bench", "benchmark", "benchmarks",
             "script", "scripts", "tools")


def touched_files(task_dir: str | Path) -> set[str]:
    """Every file a subtask's fix or tests change: the target files and the files of the task's
    oracle and test patches (target files can name paths of a later layout: pint's
    ``pint/facets/numpy/numpy_func.py`` is ``pint/numpy_func.py`` in the image)."""
    out = {f for s in load_subtasks(task_dir) for f in s.target_files}
    tests = Path(task_dir) / "tests"
    for patch in [*tests.glob("oracle/*.patch"), *tests.glob("subtasks/*/*.patch")]:
        out |= set(re.findall(r"^diff --git a/(\S+) b/", patch.read_text(errors="replace"), re.M))
    return out


def question_files(src: str | Path, exclude: set[str], seed: str) -> list[str]:
    """Source files to ask about (not tests, benchmarks or docs, not ``exclude``: the files the
    subtasks' fixes and tests change), shuffled by ``seed``."""
    src = Path(src)
    out = []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src).as_posix()
        parts = rel.lower().split("/")
        if (not p.is_file() or p.suffix not in QUESTION_SUFFIXES or rel in exclude or parts[0].startswith(".")
                or any(x in SKIP_DIRS or x.startswith("test") or x.endswith("test") for x in parts[:-1])
                or re.search(r"(^|_)tests?(_|\.|$)|^conftest\.", parts[-1])):
            continue
        if QUESTION_FILE_CHARS[0] <= p.stat().st_size <= QUESTION_FILE_CHARS[1]:
            out.append(rel)
    random.Random(seed).shuffle(out)
    return out


def _json_value(text: str) -> Any:
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        raise ValueError("no JSON in the reply")
    return json.loads(m.group(1))


def generate_questions(task_dir: str | Path, src: str | Path, client: Any, n: int,
                       existing: list[Question] | None = None) -> list[Question]:
    """At least ``n`` questions about the task's source files (``existing`` kept), two per file."""
    out = list(existing or [])
    done = {q.file for q in out}
    files = question_files(src, touched_files(task_dir), Path(task_dir).name)
    skipped = 0
    for rel in files:
        if len(out) >= n:
            break
        if rel in done:
            continue
        body = (Path(src) / rel).read_text(errors="replace")
        try:
            r = client.complete(QUESTION_SYSTEM.format(k=QUESTIONS_PER_FILE),
                                [Message("user", f"File: {rel}\n\n{body}")], [])
            items = _json_value(r.text)
        except Exception:  # one failed call or unreadable reply skips its file
            skipped += 1
            continue
        if isinstance(items, dict) and len(items) == 1:  # {"questions": [...]}
            items = next(iter(items.values()))
        for x in items[:QUESTIONS_PER_FILE] if isinstance(items, list) else []:
            if isinstance(x, dict) and str(x.get("question", "")).strip() and str(x.get("answer", "")).strip():
                out.append(Question(f"q{len(out) + 1}", rel, str(x["question"]).strip(), str(x["answer"]).strip()))
    if len(out) < n:
        raise HintError(f"{task_dir}: only {len(out)} questions of {n} ({len(files)} eligible files, "
                        f"{skipped} skipped after a failed call or unreadable reply)")
    return out


def write_questions(hints_dir: str | Path, task_name: str, model: str, questions: list[Question]) -> Path:
    path = questions_path(hints_dir, task_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task": task_name, "model": model, "generated_at": time.time(),
                                "questions": [dataclasses.asdict(q) for q in questions]}, indent=2) + "\n")
    return path


def judge_answer(client: Any, question: str, reference: str, answer: str) -> tuple[bool | None, str]:
    """(correct, why) of one answer; None when the judge's reply is unreadable."""
    r = client.complete(QUESTION_JUDGE, [Message("user", f"Question: {question}\n\nReference answer: {reference}\n\n"
                                                         f"Answer to grade: {answer}")], [])
    try:
        v = _json_value(r.text)
        c = v["correct"]
        if isinstance(c, str) and c.strip().lower() in ("true", "false"):
            c = c.strip().lower() == "true"
        if isinstance(c, bool):
            return c, str(v.get("why", ""))
    except (ValueError, KeyError, TypeError):
        pass
    return None, r.text[:300]
