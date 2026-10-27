"""Grade a session's workspace states with the task's own SWE-sweep grader: which sent bugs pass in each.

Runs inside the task's verifier image (``tests/Dockerfile``: the repository at its ``baseline`` tag,
the grader under ``/tests``) on the grader's interpreter, ``/opt/swesweep/python/bin/python3``
(Python 3.12), so it uses the standard library only. ``subagents-bench fix-times``
(:mod:`subagents_bench.fixtimes`) uploads it with the trial's states, one sandbox per trial::

    python3 replay_states.py --tests /tests --input input.tar.gz --out fix_states.json

The input (a directory or a ``.tar.gz`` of one) holds ``manifest.json``, ``{"task", "sent": [subtask
id, ...], "states": [{"seq", "source", "t", "patch", "file"}, ...]}``, and each state's patch at
``file`` (relative to the input; null when the host had no file). Every state is graded the way the
verifier grades a trial's final patch, with the grader's ``run_evaluation`` on its bundle
(``load_bundle``) restricted to the sent subtasks: reset to ``baseline``, apply the patch (an empty
one is ``b""``), the visible suite, then each sent bug's hidden tests. A bug's status in a state:

- ``pass``: resolved (its targeted tests ran, none failed);
- ``unknown``: the state says nothing about the bug: the evaluation returned an error or raised, the
  patch file is missing, the agent patch did not apply or did not build, the bug's hidden test patch
  did not apply, its targeted run timed out or produced no report, no targeted test ran, or every
  failing targeted test is an ``error`` outcome (a pytest collection or import error, a Go
  ``<build>`` failure: click's tests all import the whole package, so one half-finished edit
  elsewhere would otherwise fail every bug);
- ``fail``: otherwise.

The ``error`` outcomes are not in the grader's result (its ``failed`` merges ``failed`` and
``error``): :func:`record_outcomes` wraps the grader's ``_run_targets`` to keep each targeted run's
outcomes, unfiltered by the grader except for its id spelling, in the subtask result.

Then, for each bug and each pass -> fail transition between consecutive known (pass or fail)
states, the failing state is graded once more (only the bugs that need it) and the bug's ``rerun``
status recorded: a failure that does not repeat is noise, not a broken fix. Output (``--out``)::

    {"version": 1, "task": str, "sent": [sid, ...], "states": [
       {"seq": int|None, "source": "snapshot"|"final", "t": float, "patch": str, "applied": bool|None,
        "visible_ok": bool|None, "error": str|None, "duration_s": float, "rerun_duration_s": float|None,
        "bugs": {sid: {"status": "pass"|"fail"|"unknown", "n_passed": int, "n_failed": int,
                       "error": str|None, "rerun": "pass"|"fail"|"unknown"|None}}}, ...]}

``applied`` and ``visible_ok`` are the grader's (None when the evaluation failed; ``visible_ok`` also
when the patch did not apply, since the visible run then saw the unpatched tree); a state's ``error``
says why none of its bugs could be graded. A progress line per state goes to stdout.
"""

from __future__ import annotations

import argparse
import ast
import functools
import importlib.util
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

VERSION = 1
ERROR_CHARS = 1500  # of an evaluation's error and traceback


def load_grader(tests_dir: Path) -> Any:
    """The task's grader module, ``<tests>/shared/swesweep_grade.py``. A docstring of it holds a lone
    surrogate (``\\ud83d``), which Python 3.13+ cannot compile (it encodes docstrings to clean them):
    there the docstrings' surrogates are replaced first. The verifier image's 3.12 needs no change."""
    path = tests_dir / "shared" / "swesweep_grade.py"
    spec = importlib.util.spec_from_file_location("swesweep_grade", path)
    if spec is None or spec.loader is None or not path.is_file():
        raise FileNotFoundError(f"no grader at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["swesweep_grade"] = module
    try:
        spec.loader.exec_module(module)
    except UnicodeEncodeError:  # raised while compiling, before any of the module ran
        tree = ast.parse(path.read_bytes(), str(path))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body
                    and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body[0].value.value = body[0].value.value.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
        exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


class _OutcomeSpy:
    """A suite whose ``run_selected`` keeps the outcomes it returns; everything else is the suite's."""

    def __init__(self, suite: Any):
        self._suite = suite
        self.outcomes: dict[str, str] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._suite, name)

    def run_selected(self, *args: Any, **kwargs: Any) -> Any:
        got = self._suite.run_selected(*args, **kwargs)
        if isinstance(got, tuple) and got and isinstance(got[0], dict):
            self.outcomes = dict(got[0])
        return got


def record_outcomes(grader: Any) -> None:
    """Make the grader keep every targeted run's outcomes in its subtask result (``outcomes``, test
    id as the grader spells it -> passed/failed/error/skipped). Grading is unchanged; a grader
    without ``_run_targets`` is left alone (then only a Go ``::<build>`` id reveals a build error)."""
    run_targets = getattr(grader, "_run_targets", None)
    if run_targets is None or getattr(run_targets, "records_outcomes", False):
        return
    normalize = getattr(grader, "normalize_test_id", lambda x: x)

    @functools.wraps(run_targets)
    def recording(env: Any, suite: Any, config: Any, files: Any, names: Any, capture: Any, jobs: Any,
                  result: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        spy = _OutcomeSpy(suite)
        try:
            return run_targets(env, spy, config, files, names, capture, jobs, result, *args, **kwargs)
        finally:
            if spy.outcomes is not None:
                result["outcomes"] = {normalize(k): v for k, v in spy.outcomes.items()}

    recording.records_outcomes = True  # type: ignore[attr-defined]
    grader._run_targets = recording


def evaluate(grader: Any, bundle: dict[str, Any], tests_dir: Path, patch: bytes | None) -> tuple[dict[str, Any], float]:
    """One ``run_evaluation`` -> (verdict, seconds). Never the oracle: a missing patch is an error."""
    if patch is None:
        return {"error": "patch file missing"}, 0.0
    t0 = time.monotonic()
    try:
        verdict = grader.run_evaluation(bundle, tests_dir, patch)
    except Exception as e:  # run_evaluation reports its own failures as an error; anything else is one too
        verdict = {"error": f"evaluation raised {type(e).__name__}: {e}"}
    if not isinstance(verdict, dict):
        verdict = {"error": f"evaluation returned {type(verdict).__name__}"}
    return verdict, round(time.monotonic() - t0, 3)


def state_error(verdict: dict[str, Any]) -> str | None:
    """Why none of the state's bugs could be graded, or None."""
    detail = verdict.get("detail") or {}
    if verdict.get("error"):
        tail = str(verdict.get("stderr_tail") or "").strip()
        return (str(verdict["error"]) + (f" | {tail[-ERROR_CHARS:]}" if tail else ""))[:2 * ERROR_CHARS]
    if detail.get("agent_build_ok") is False:
        return ("agent build failed: " + str(detail.get("build_error") or "").strip()[-ERROR_CHARS:]).strip()
    if detail.get("agent_patch_applied") is False:
        return ("agent patch did not apply: " + str(detail.get("agent_patch_error") or "").strip()).strip()
    return None


def classify(verdict: dict[str, Any], sid: str) -> tuple[str, int, int, str | None]:
    """A bug's (status, n_passed, n_failed, error) in an evaluation's verdict (module docstring)."""
    detail = verdict.get("detail") or {}
    if verdict.get("error"):
        return "unknown", 0, 0, "evaluation error"
    if detail.get("agent_build_ok") is False:
        return "unknown", 0, 0, "agent build failed"
    if detail.get("agent_patch_applied") is False:
        return "unknown", 0, 0, "agent patch did not apply"
    r = (verdict.get("subtasks") or {}).get(sid)
    if not isinstance(r, dict):
        return "unknown", 0, 0, "not graded"
    n_passed, n_failed = int(r.get("n_passed") or 0), int(r.get("n_failed") or 0)
    if r.get("resolved"):
        return "pass", n_passed, n_failed, None
    error = (r.get("detail") or {}).get("error")
    if error:  # hidden test patch did not apply, targeted tests timed out / produced no report, no targets
        return "unknown", n_passed, n_failed, str(error)
    if r.get("test_applied") is False:
        return "unknown", n_passed, n_failed, "hidden test patch did not apply"
    failed = list(r.get("failed") or [])
    outcomes = r.get("outcomes")
    if failed and all(outcomes.get(x) == "error" if isinstance(outcomes, dict) else x.endswith("::<build>")
                      for x in failed):
        shown = ", ".join(failed[:3]) + (" ..." if len(failed) > 3 else "")
        return "unknown", n_passed, n_failed, f"targeted tests errored (collection or build failure): {shown}"
    if not failed and not n_passed:
        return "unknown", n_passed, n_failed, "no targeted test ran"
    return "fail", n_passed, n_failed, None


def transitions(states: list[dict[str, Any]], sent: list[str]) -> dict[int, list[str]]:
    """State index -> the bugs that fail there right after passing in the previous known state."""
    out: dict[int, list[str]] = {}
    for sid in sent:
        prev = None
        for i, st in enumerate(states):
            status = st["bugs"][sid]["status"]
            if status == "unknown":
                continue
            if prev == "pass" and status == "fail":
                out.setdefault(i, []).append(sid)
            prev = status
    return out


def _patch(input_dir: Path, state: dict[str, Any]) -> bytes | None:
    f = state.get("file")
    path = input_dir / f if f else None
    return path.read_bytes() if path is not None and path.is_file() else None


def replay(grader: Any, tests_dir: Path, manifest: dict[str, Any], input_dir: Path,
           log: Callable[[str], None] = print) -> dict[str, Any]:
    """Grade every state of ``manifest`` (module docstring) -> the output document."""
    record_outcomes(grader)
    bundle = grader.load_bundle(tests_dir)
    sent = list(dict.fromkeys(manifest["sent"]))
    known = {s["id"] for s in bundle["subtasks"]}
    graded = {**bundle, "subtasks": [s for s in bundle["subtasks"] if s["id"] in sent]}
    missing = [s for s in sent if s not in known]
    if missing:
        log(f"not in the task's bundle (unknown in every state): {', '.join(missing)}")
    states: list[dict[str, Any]] = []
    todo = manifest["states"]
    for k, st in enumerate(todo):
        verdict, seconds = evaluate(grader, graded, tests_dir, _patch(input_dir, st))
        detail = verdict.get("detail") or {}
        error = state_error(verdict)
        applied = None if verdict.get("error") else detail.get("agent_patch_applied")
        bugs = {}
        for sid in sent:
            status, n_passed, n_failed, why = classify(verdict, sid) if sid in known else (
                "unknown", 0, 0, "not in the task's bundle")
            bugs[sid] = {"status": status, "n_passed": n_passed, "n_failed": n_failed, "error": why, "rerun": None}
        states.append({
            "seq": st.get("seq"), "source": st.get("source"), "t": st.get("t"), "patch": st.get("patch"),
            "applied": applied, "visible_ok": None if verdict.get("error") or applied is False else verdict.get("visible_ok"),
            "error": error, "duration_s": seconds, "rerun_duration_s": None, "bugs": bugs,
        })
        log(f"state {k + 1}/{len(todo)} {st.get('source')} seq={st.get('seq')} {seconds:.1f}s"
            + (f" error: {error[:200]}" if error else "") + ": "
            + " ".join(f"{sid}={b['status']}" for sid, b in bugs.items()))
    for i, sids in sorted(transitions(states, sent).items()):
        again = {**bundle, "subtasks": [s for s in graded["subtasks"] if s["id"] in sids]}
        verdict, seconds = evaluate(grader, again, tests_dir, _patch(input_dir, todo[i]))
        states[i]["rerun_duration_s"] = seconds
        for sid in sids:
            states[i]["bugs"][sid]["rerun"] = classify(verdict, sid)[0]
        log(f"rerun state {i + 1} seq={states[i]['seq']} {seconds:.1f}s: "
            + " ".join(f"{sid}={states[i]['bugs'][sid]['rerun']}" for sid in sids))
    return {"version": VERSION, "task": manifest.get("task") or bundle.get("task"), "sent": sent, "states": states}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade a session's workspace states with the task's grader.")
    parser.add_argument("--tests", default="/tests", help="the task's tests dir (the grader is under shared/)")
    parser.add_argument("--input", required=True, help="a directory, or a .tar.gz of one, holding manifest.json")
    parser.add_argument("--out", required=True, help="where to write the output JSON")
    args = parser.parse_args(argv)
    tests_dir, out = Path(args.tests), Path(args.out)
    grader = load_grader(tests_dir)
    with tempfile.TemporaryDirectory(prefix="replay-") as tmp:
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            with tarfile.open(input_dir) as tar:
                tar.extractall(tmp, filter="data")
            input_dir = Path(tmp)
        manifest = json.loads((input_dir / "manifest.json").read_text())
        doc = replay(grader, tests_dir, manifest, input_dir, log=lambda line: print(line, flush=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    partial.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(partial, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
