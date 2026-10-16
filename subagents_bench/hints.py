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
  ``subagents-bench hints`` (:func:`generate_reports`), read from ``<hints_dir>/<task>.json``;
- ``symptom``: a report of the observable behaviour only, as an end user of the library would write
  it (what they ran, what they expected, what happened), written by ``subagents-bench hints`` from
  the hidden test patch and the reference fix and read from ``<hints_dir>/<task>.symptom.json``. It
  names no source file, module path, private function or test (:func:`leaks` checks each one as it
  is written; a report that leaks is rewritten).

With ``bundle: n`` one user message reports n consecutive bugs of the order as a numbered list
(the message's ``Hint`` has them all in ``items``), one message every ``interval_sec``.

With ``interactive.replies`` the schedule also holds questions about the code (``questions_per_interval``
between consecutive bugs, from ``<hints_dir>/<task>.questions.json``, written by ``subagents-bench
questions`` from the task's source tree) and status checks (``status_every``); every message gets an
id (``msg_id``, 1, 2, ... in sending order) and ``urgent_every`` marks some bugs urgent. A
``change_of_plan`` entry adds a ``cancel`` or ``undo`` message about an earlier bug: its ``subtask``
is that bug's and ``target`` its msg_id.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import InteractiveConfig
from .llm import BillingError, LLMError, Message

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
    """One scheduled user message: a bug (``subtask``), a ``question``, a ``status`` check, or a
    ``cancel`` or ``undo`` of the bug ``target`` (its msg_id; ``subtask`` is that bug's)."""

    index: int  # 0-based position among the messages of its kind
    t_offset: float  # seconds after the episode start
    subtask: Subtask | None  # a bug message's first (with ``bundle``: of its items)
    text: str
    kind: str = "bug"
    msg_id: int | None = None  # the id the agent replies to (with interactive.replies)
    urgent: bool = False
    question: Question | None = None
    target: int | None = None
    # A bug message's subtasks (item 1, 2, ... of its list; one without ``bundle``) and each one's
    # 0-based position among every bug sent (``hint_index`` in the records)
    items: tuple[Subtask, ...] = ()
    positions: tuple[int, ...] = ()


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
    if detail in GENERATED:
        if not report:
            raise HintError(f"subtask {subtask.id}: no generated {detail} (run `subagents-bench hints`)")
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


def bundle_text(texts: list[str]) -> str:
    """Several bugs' texts as one numbered list (a single text as it is); an item's later lines are
    indented under its number."""
    if len(texts) == 1:
        return texts[0]
    return "\n\n".join(f"{k}. " + "\n".join(("   " + x) if x.strip() and i else x.rstrip() if i else x
                                           for i, x in enumerate(t.strip().splitlines()))
                       for k, t in enumerate(texts, 1))


def n_slots(cfg: InteractiveConfig) -> int:
    """Bug messages that fit in the session: offsets 0, interval, ... below the wall clock."""
    return max(1, math.ceil(cfg.wall_clock_sec / cfg.interval_sec - 1e-9))


def order_seed(cfg: InteractiveConfig, round_: int | None) -> int:
    """The seed of a sweep round's order: ``order_seed + round - 1`` with ``vary_order_by_round``, so
    the conditions of a round share an order and rounds sample different ones."""
    return cfg.order_seed + (round_ - 1 if cfg.vary_order_by_round and round_ else 0)


def schedule(cfg: InteractiveConfig, task_dir: str | Path, task_name: str, hints_dir: str | Path | None = None,
             round_: int | None = None) -> list[Hint]:
    """The episode's messages in sending order: the ordered subtasks, as many as fit (and
    ``max_hints``), sent every ``interval_sec``; with ``replies``, the questions and status checks
    between them, the changes of plan and an id per message. A status check's text is the ids of the
    bugs sent before it (``#1, #2, #4``), a change's the id of its bug (``#4``), which
    ``prompts.request_text`` puts in its message."""
    seed = order_seed(cfg, round_)
    subtasks = ordered(load_subtasks(task_dir), task_name, seed)
    b = cfg.bundle
    n = min(len(subtasks), n_slots(cfg) * b, cfg.max_hints or len(subtasks))  # bugs
    reports = load_reports(hints_dir, task_name, cfg.hint_detail) if cfg.hint_detail in GENERATED else {}
    texts = [hint_text(s, cfg.hint_detail, reports.get(s.id)) for s in subtasks[:n]]
    out = [Hint(k, k * cfg.interval_sec, subtasks[i], bundle_text(texts[i:i + b]),
                urgent=bool(cfg.urgent_every) and k > 0 and k % cfg.urgent_every == 0,
                items=tuple(subtasks[i:i + b]), positions=tuple(range(i, min(i + b, n))))
           for k, i in enumerate(range(0, n, b))]
    n = len(out)  # bug messages
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
    if cfg.change_of_plan and cfg.bundle > 1:
        raise HintError("interactive.change_of_plan takes back one bug message; it does not work with bundle > 1")
    for j, c in enumerate(cfg.change_of_plan):
        if c["bug"] >= n:
            raise HintError(f"{task_name}: change_of_plan[{j}] takes back bug {c['bug']}, but only {n} bugs are sent")
        extra.append(Hint(j, float(c["at_sec"]), out[c["bug"]].subtask, "", kind=c["kind"]))
    kinds = {"bug": 0, "status": 1, "question": 2, "cancel": 3, "undo": 3}  # at the same time: the bug first
    out = sorted(out + extra, key=lambda h: (h.t_offset, kinds[h.kind], h.index))
    out = [dataclasses.replace(h, msg_id=i + 1) for i, h in enumerate(out)]
    bug_ids = {h.subtask.id: h.msg_id for h in out if h.kind == "bug"}
    # a status check's text: the ids of the bugs sent before it ("#1, #2, #4"), whatever happened to them
    return [dataclasses.replace(h, text=", ".join(f"#{b.msg_id}" for b in out if b.kind == "bug"
                                                  and b.t_offset < h.t_offset)) if h.kind == "status" else
            dataclasses.replace(h, target=bug_ids[h.subtask.id], text=f"#{bug_ids[h.subtask.id]}")
            if h.kind in ("cancel", "undo") else h
            for h in out]


GENERATED = ("report", "symptom")  # hint details written beforehand by `subagents-bench hints`


def reports_path(hints_dir: str | Path, task_name: str, detail: str = "report") -> Path:
    return Path(hints_dir) / (f"{task_name}.json" if detail == "report" else f"{task_name}.{detail}.json")


def load_reports(hints_dir: str | Path | None, task_name: str, detail: str = "report") -> dict[str, str]:
    if hints_dir is None:
        raise HintError(f"hint_detail {detail} needs a hints dir (SUBAGENTS_BENCH_HINTS_DIR; the sweep sets it)")
    path = reports_path(hints_dir, task_name, detail)
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


SYMPTOM_SYSTEM = """\
You write the bug report that an end user of a software project sends its maintainers. You are \
given the diff of the fix and the new tests that check it, so you know what goes wrong; the user \
does not: they only saw the behaviour. Write what the user did (a short code snippet or the \
command they ran is fine), what they expected, and what happened instead (the wrong output, the \
exception or the error message they saw). Write it as that user, in the first person and in \
plain words, using only the project's public interface as its documentation shows it. Never \
name a source file, a module or package path inside the project, or an internal (private) \
function, method, class or variable; never explain the cause or the fix; never quote, name or \
mention the tests. Two to five sentences, plus a snippet of a few lines if it helps. Reply with \
the report only: no title, no headings.
"""
SYMPTOM_TRIES = 4  # a report that leaks (``leaks``) is rewritten, up to this many writes in all
FIX_CHARS = 12000  # of the reference fix shown to the symptom writer


def symptom_prompt(subtask: Subtask, test_patch: str, fix_patch: str, banned: list[str]) -> str:
    if len(test_patch) > PATCH_CHARS:
        test_patch = test_patch[:PATCH_CHARS] + "\n[... truncated ...]"
    if len(fix_patch) > FIX_CHARS:
        fix_patch = fix_patch[:FIX_CHARS] + "\n[... truncated ...]"
    return (f"Words the report must not contain: {', '.join(banned)}\n\n"
            f"Diff of the fix (for your understanding only):\n{fix_patch}\n\n"
            f"Diff adding the tests that fail before the fix:\n{test_patch}")


# Python ``def name`` and Go ``func name`` / ``func (r *T) name`` in a diff (hunk headers included)
DEF_RE = re.compile(r"(?:\bdef\s+|\bfunc\s+(?:\([^)]*\)\s*)?)([A-Za-z_]\w*)\s*[(\[]")


def leak_terms(subtask: Subtask, fix_patch: str = "") -> tuple[list[str], list[str]]:
    """(text, word) terms a ``symptom`` report must not contain: text terms (matched as substrings,
    ignoring case) are the target files, their basenames and module paths (``src/click/core.py``:
    ``click/core``, ``click.core``) and the hidden tests' names and test files (``test_types``,
    ``TestGzipEmpty``); word terms (matched as whole identifiers) are the private functions the fix
    changes (Python ``_name``, Go lower-case names), from the hunks of ``fix_patch``."""
    text: list[str] = []
    for f in subtask.target_files:
        parts = Path(f).with_suffix("").parts
        parts = parts[1:] if parts and parts[0] in ("src", "lib") else parts
        text += [f, Path(f).name] + [sep.join(parts[i:]) for i in range(len(parts) - 1) for sep in ("/", ".")]
    for t in subtask.f2p:
        for part in re.split(r"::|/|\.", test_label(t)):
            if re.match(r"(?i)test", part) and part.lower() not in ("test", "tests", "testing"):
                text.append(part)
    words = []
    for section in re.split(r"^(?=diff --git )", fix_patch, flags=re.M):
        go = re.match(r"diff --git a/\S+\.go\b", section) is not None
        for name in DEF_RE.findall(section):
            if name.startswith("_") and not name.endswith("__"):  # Python's private names
                words.append(name)
            elif go and name[:1].islower() and re.search(r"[A-Z_]", name):  # Go's unexported (not a plain word)
                words.append(name)
    return list(dict.fromkeys(text)), list(dict.fromkeys(words))


def leaks(report: str, terms: tuple[list[str], list[str]]) -> list[str]:
    """The terms of :func:`leak_terms` that ``report`` contains."""
    text, words = terms
    low = report.lower()
    return ([x for x in text if x.lower() in low]
            + [w for w in words if re.search(rf"(?<![\w.]){re.escape(w)}(?!\w)", report)])


def _read(path: Path, missing: str) -> str:
    return path.read_text(errors="replace") if path.is_file() else missing


def generate_reports(task_dir: str | Path, client: Any, existing: dict[str, str] | None = None,
                     detail: str = "report", log: Any = None) -> dict[str, str]:
    """A ``detail`` report (``report`` or ``symptom``) per subtask of ``task_dir`` (those in
    ``existing`` are kept), written by ``client``. A symptom that :func:`leaks` is sent back to be
    rewritten (``SYMPTOM_TRIES`` writes in all; then a HintError); ``log`` gets a line per rewrite."""
    out = dict(existing or {})
    tests = Path(task_dir) / "tests"
    for s in load_subtasks(task_dir):
        if out.get(s.id):
            continue
        patch = _read(tests / "subtasks" / s.id / "test.patch", "(no test patch)")
        if detail == "report":
            r = client.complete(REPORT_SYSTEM, [Message("user", report_prompt(s, patch))], [])
            text = r.text.strip()
            if not text:
                raise HintError(f"subtask {s.id}: the model returned an empty report (finish_reason {r.finish_reason})")
            out[s.id] = text
            continue
        fix = _read(tests / "oracle" / f"{s.id}.patch", "(no fix diff)")
        terms = leak_terms(s, fix)
        messages = [Message("user", symptom_prompt(s, patch, fix, terms[0] + terms[1]))]
        for attempt in range(1, SYMPTOM_TRIES + 1):
            r = client.complete(SYMPTOM_SYSTEM, messages, [])
            text = r.text.strip()
            found = leaks(text, terms) if text else ["(empty reply)"]
            if not found:
                out[s.id] = text
                break
            if log:
                log(f"    {s.id}: write {attempt} names {', '.join(found)}; rewriting")
            messages += [Message("assistant", text or "(empty)"),
                         Message("user", f"Your report names {', '.join(found)}, which the user would not know or "
                                         "which points at the code to change. Rewrite it without them, describing "
                                         "only what the user did and saw. Reply with the report only.")]
        else:
            raise HintError(f"subtask {s.id}: every symptom report of {SYMPTOM_TRIES} named {', '.join(found)}")
    return out


def write_reports(hints_dir: str | Path, task_name: str, model: str, reports: dict[str, str],
                  detail: str = "report") -> Path:
    path = reports_path(hints_dir, task_name, detail)
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
answer and a reply to grade, which may also be about other things: grade only what it says \
about this question. The verdict is "correct" if the reply states what the reference states \
(other wording, extra correct detail and a longer explanation are fine); "wrong" if it gives an \
answer that contradicts the reference or misses its key fact; "none" if it is not an answer: an \
acknowledgement, a promise or a plan to answer, only how to find out, or saying it does not \
know. Reply with only a JSON object {"verdict": "correct" or "wrong" or "none", "why": "<one \
sentence>"}."""
VERDICTS = ("correct", "wrong", "none")  # QUESTION_JUDGE's; none: not an answer
ANSWERS_VERSION = 2  # agent/answers.json: per-reply verdicts, three-way, at temperature 0 (grade_replies)


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
    try:  # a tag or branch: one shallow clone
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", ref, url, str(dest)], check=True,
                       capture_output=True, timeout=600)
    except subprocess.CalledProcessError:  # a commit sha (4 SWE-sweep Dockerfiles): the whole history
        shutil.rmtree(dest, ignore_errors=True)
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", url, str(dest)], check=True, capture_output=True,
                       timeout=1800)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", ref], check=True, capture_output=True,
                       timeout=600)
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
    touched = touched_files(task_dir)
    out = [q for q in existing or [] if q.file not in touched]  # written before a file was known to be touched
    done = {q.file for q in out}
    files = question_files(src, touched, Path(task_dir).name)
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
        except BillingError:
            raise
        except (LLMError, ValueError):  # one failed call or unreadable reply skips its file
            skipped += 1
            continue
        if isinstance(items, dict) and len(items) == 1:  # {"questions": [...]}
            items = next(iter(items.values()))
        for x in items[:QUESTIONS_PER_FILE] if isinstance(items, list) else []:
            if isinstance(x, dict) and str(x.get("question", "")).strip() and str(x.get("answer", "")).strip():
                out.append(Question(_next_id(out, existing), rel, str(x["question"]).strip(), str(x["answer"]).strip()))
    if len(out) < n:
        raise HintError(f"{task_dir}: only {len(out)} questions of {n} ({len(files)} eligible files, "
                        f"{skipped} skipped after a failed call or unreadable reply)")
    return out


def _next_id(out: list[Question], existing: list[Question] | None) -> str:
    """``q<n>`` above every id used so far, dropped questions' included (ids stay unique)."""
    used = [int(q.id[1:]) for q in [*out, *(existing or [])] if re.fullmatch(r"q\d+", q.id)]
    return f"q{max(used, default=0) + 1}"


def write_questions(hints_dir: str | Path, task_name: str, model: str, questions: list[Question]) -> Path:
    path = questions_path(hints_dir, task_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task": task_name, "model": model, "generated_at": time.time(),
                                "questions": [dataclasses.asdict(q) for q in questions]}, indent=2) + "\n")
    return path


def judge_answer(client: Any, question: str, reference: str, answer: str) -> tuple[str | None, str]:
    """(verdict, why) of one reply: ``correct``, ``wrong`` or ``none`` (not an answer); None, with
    the start of the judge's reply as ``why``, when that reply is unreadable. The judge's sampling
    is the client's (``grade-answers`` runs it at temperature 0). The pre-version-2 shape
    ``{"correct": true|false}`` (a boolean or the string "true"/"false") reads as correct/wrong; a
    readable ``verdict`` wins over it. A reply that reads neither way but has ``"verdict":
    "<verdict word>"`` pairs that all name one verdict (near-JSON: text with a brace after the
    object, two objects, a stray quote in ``why``) reads as that verdict, with its ``"why"`` string
    when one can be found; pairs naming different verdicts leave it unreadable."""
    r = client.complete(QUESTION_JUDGE, [Message("user", f"Question: {question}\n\nReference answer: {reference}\n\n"
                                                         f"Reply to grade: {answer}")], [])
    try:
        v = _json_value(r.text)
        verdict = v.get("verdict")
        if isinstance(verdict, str) and verdict.strip().lower() in VERDICTS:
            return verdict.strip().lower(), str(v.get("why", ""))
        c = v["correct"]
        if isinstance(c, str) and c.strip().lower() in ("true", "false"):
            c = c.strip().lower() == "true"
        if isinstance(c, bool):
            return "correct" if c else "wrong", str(v.get("why", ""))
    except (ValueError, KeyError, TypeError, AttributeError):  # AttributeError: a JSON list
        pass
    # near-JSON whose verdict is plain to see: trailing text with a brace, two objects, a "[" before
    # the object, a stray quote or a raw newline inside ``why``; never one of two different verdicts
    # (the prompt's format echoed before the object, a verdict taken back): that stays unread
    found = {m.lower() for m in re.findall(r'"verdict"\s*:\s*"\s*(correct|wrong|none)\s*"', r.text, re.I)}
    if len(found) == 1:
        why = re.search(r'"why"\s*:\s*"(.*?)"\s*[,}]', r.text, re.S)
        return found.pop(), why.group(1) if why else r.text[:300]
    return None, r.text[:300]


def cites(text: str, msg_id: int) -> bool:
    """``text`` names message ``#<msg_id>`` as a whole token (``#3`` is not in ``#31``, ``##3`` or ``a#3``)."""
    return re.search(rf"(?<![\w#])#{msg_id}(?!\w)", text) is not None


def grade_replies(client: Any, inbox: list[dict], replies: list[dict]) -> dict[str, dict[str, Any]]:
    """The judge's verdicts (:func:`judge_answer`) on an episode's replies to its questions, from its
    ``inbox`` and ``replies`` records: ``{"answers": ..., "replies": ..., "elsewhere": ...}``, each
    keyed by the question's msg_id as a string (no key for a question without such replies):

    - ``replies``: the replies to the question's own id in time order (records of one step keep
      their order), judged one by one up to and including the first correct one (an unreadable
      verdict does not stop it): ``[{"t", "verdict", "correct", "why"}, ...]``;
    - ``answers``: the first of them, ``{"correct", "verdict", "why"}`` (the reply the prompt calls
      the answer; the shape before version 2, which had no ``verdict``);
    - ``elsewhere``: replies to other ids that cite the question (:func:`cites`) and were sent after
      it was delivered (none if it never was), judged the same way, ``[{"t", "to", "verdict",
      "correct", "why"}, ...]``. Reported apart: only a reply to its own id answers a question.

    ``correct`` is ``verdict == "correct"``, None when the verdict is unreadable. Calls the judge
    once per judged reply; its errors propagate."""
    ordered = sorted(replies, key=lambda r: r["t"])
    out: dict[str, dict[str, Any]] = {"answers": {}, "replies": {}, "elsewhere": {}}
    for q in sorted((m for m in inbox if m.get("kind") == "question"), key=lambda m: m["msg_id"]):
        mid, delivered = q["msg_id"], q.get("t_delivered")
        own = [r for r in ordered if r["to"] == mid]
        other = [r for r in ordered if r["to"] != mid and delivered is not None and r["t"] > delivered
                 and cites(r["text"], mid)]
        for key, rs in (("replies", own), ("elsewhere", other)):
            got = []
            for r in rs:
                verdict, why = judge_answer(client, q["question"], q["reference"], r["text"])
                got.append({"t": r["t"], **({"to": r["to"]} if key == "elsewhere" else {}), "verdict": verdict,
                            "correct": None if verdict is None else verdict == "correct", "why": why})
                if verdict == "correct":
                    break
            if got:
                out[key][str(mid)] = got
        if own:
            first = out["replies"][str(mid)][0]
            out["answers"][str(mid)] = {"correct": first["correct"], "verdict": first["verdict"], "why": first["why"]}
    return out
