"""``subagents-bench fix-times SPEC``: when each sent bug was really fixed, from the session's snapshots.

The verifier grades a session once, at its end. With ``interactive.snapshots`` a session also
records its workspace while it runs: ``agent/snapshots.jsonl``, and for every changed state its
patch against ``baseline`` at ``agent/snapshots/<seq>.patch``. After the sweep this command grades
each of those states with the task's own grader, on Modal, with no LLM call: one sandbox per
finished trial (harbor's ``result.json`` with ``finished_at``; job dirs starting with ``_`` or ``.``
are skipped) that has ``agent/snapshots.jsonl`` and, unless ``force``, no ``agent/fix_states.json``.

- The sandbox runs the task's verifier image, built from ``<task>/tests/Dockerfile`` of the task
  dirs the sweep ran (``sweep.task_root``: the copy under ``<jobs_dir>/tasks/`` when the spec pins a
  platform) with that ``tests/`` dir as context: the image harbor's separate verifier uses. It is
  built once per task. 3 CPUs, 7000 MB, no network (the verifier's ``no-network``), 3 hours, app
  ``subagents-bench-fixtimes``, Modal's image builder ``sweep.MODAL_IMAGE_BUILDER`` unless
  MODAL_IMAGE_BUILDER_VERSION is set.
- The states: every snapshot record with a patch (a changed state) in ``seq`` order, at its
  ``t_start`` (when the capture began), then the state the verifier graded: the trial's
  ``artifacts/logs/artifacts/model.patch`` (``source`` ``final``) at the episode's ``t_end``. The
  sent bugs: the inbox's bug (``hint``) records ready by ``t_end``, in sending order.
- :mod:`subagents_bench.replay_states` grades them in the sandbox; its docstring has the statuses,
  the reruns and the output. This side adds ``trial`` (``<job>/<trial>``), ``graded_at``, ``image``
  (the Dockerfile, a hash of the ``tests/`` dir, Modal's image id) and ``final_matches_verifier``:
  whether the final state's per-bug resolved (status ``pass``) equals ``verifier/eval.json``'s for
  every sent bug, None without a readable eval.json or when the verifier errored; ``mismatches``
  lists the bugs that differ. It writes ``<trial>/agent/fix_states.json``; its ``patch`` paths are
  relative to ``agent/``, like the snapshot records'.

Up to ``concurrency`` trials run at once. A trial that fails writes nothing and is replayed by the
next run. Exit 0 when every trial got its file, 1 otherwise, 2 without Modal, 130 on Ctrl-C (the
running sandboxes are terminated). :class:`ModalRunner` is the whole Modal side; :func:`fix_times`
takes any object with its methods (the tests use a fake).
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import sweep
from .replay_states import VERSION
from .sweep import Spec, _json, _rel, say

APP_NAME = "subagents-bench-fixtimes"
CPUS, MEMORY_MB = 3, 7000  # the sweeps' task containers (harbor --override-cpus/--override-memory-mb)
SANDBOX_TIMEOUT_SEC = 3 * 3600
EXEC_TIMEOUT_SEC = SANDBOX_TIMEOUT_SEC - 600  # the replay's own budget: the download still fits after it
GRADER_PYTHON = "/opt/swesweep/python/bin/python3"  # the verifier image's interpreter (tests/test.sh)
TESTS = "/tests"  # the task's tests/ dir in the verifier image (COPY . /tests)
REMOTE = "/fixtimes"  # the replay's files in the sandbox
FIX_STATES = "fix_states.json"
FINAL_PATCH = Path("artifacts") / "logs" / "artifacts" / "model.patch"  # the collect hook's, as harbor stores it
SCRIPT = Path(__file__).with_name("replay_states.py")
# The open-files limit raised as tests/test.sh does; the grader's chatter goes to a log, whose tail is returned.
RUN_SH = (f'ulimit -n "$(ulimit -Hn)" 2>/dev/null || true; "$@" > {REMOTE}/replay.log 2>&1; rc=$?; '
          f"tail -c 4000 {REMOTE}/replay.log; exit $rc")


class FixTimesError(RuntimeError):
    """A trial that could not be replayed this time (the next run retries it)."""


@dataclass
class State:
    seq: int | None  # the snapshot's; None for the final state
    source: str  # snapshot | final
    t: float | None  # epoch seconds
    patch: str  # relative to the trial's agent/ dir
    file: Path | None  # the patch on this machine; None when it is missing


@dataclass
class TrialPlan:
    dir: Path
    name: str  # <job>/<trial>
    task: str
    tests_dir: Path
    t0: float | None  # session start, for printing
    sent: list[str]  # subtask ids, in sending order
    states: list[State]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines() if path.is_file() else []:
        try:
            rec = json.loads(line)
        except ValueError:  # blank, or cut short by a kill
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _first(rec: dict[str, Any], *keys: str) -> Any:
    return next((rec[k] for k in keys if rec.get(k) is not None), None)


def trial_plan(spec: Spec, trial: Path) -> TrialPlan:
    """A trial's sent bugs and states (module docstring)."""
    agent = trial / "agent"
    res, ep = _json(trial / "result.json"), _json(agent / "episode.json")
    task = res.get("task_name") or ep.get("task_name") or trial.name.rsplit("__", 1)[0]
    t_end = ep.get("t_end")
    bugs = sorted((r for r in _jsonl(agent / "inbox.jsonl") if r.get("kind") == "hint" and r.get("subtask_id")),
                  key=lambda r: r.get("t_ready") or 0.0)
    sent = list(dict.fromkeys(r["subtask_id"] for r in bugs if t_end is None or (r.get("t_ready") or 0.0) <= t_end))
    snaps = sorted((r for r in _jsonl(agent / "snapshots.jsonl") if r.get("patch")), key=lambda r: r.get("seq") or 0)
    states = [State(r.get("seq"), "snapshot", _first(r, "t_start", "t_end", "t_request"), r["patch"],
                    agent / r["patch"] if (agent / r["patch"]).is_file() else None) for r in snaps]
    if t_end is None:
        t_end = max((s.t for s in states if s.t is not None), default=None)
    final = trial / FINAL_PATCH
    states.append(State(None, "final", t_end, str(".." / FINAL_PATCH), final if final.is_file() else None))
    return TrialPlan(trial, f"{trial.parent.name}/{trial.name}", task, sweep.task_root(spec) / task / "tests",
                     _first(ep, "session_t0", "t_start"), sent, states)


def plan_trials(spec: Spec, force: bool = False, only: str | None = None) -> tuple[list[TrialPlan], dict[str, int]]:
    """The trials to replay (``only``: whose ``<job>/<trial>`` contains it), and how many were skipped and why."""
    plans: list[TrialPlan] = []
    skipped = {"unfinished": 0, "without snapshots": 0, "already replayed": 0}
    for agent in sorted(spec.jobs_dir.glob("*/*/agent")):
        trial = agent.parent  # not a trial without harbor's result.json (the staged tasks/<task>/agent/)
        if (trial.parent.name.startswith(("_", ".")) or not (trial / "result.json").is_file()
                or (only and only not in f"{trial.parent.name}/{trial.name}")):
            continue
        if not _json(trial / "result.json").get("finished_at"):
            skipped["unfinished"] += 1
        elif not (agent / "snapshots.jsonl").is_file():
            skipped["without snapshots"] += 1
        elif (agent / FIX_STATES).is_file() and not force:
            skipped["already replayed"] += 1
        else:
            plans.append(trial_plan(spec, trial))
    return plans, skipped


def pack(plan: TrialPlan, archive: Path) -> None:
    """The replay's input: ``manifest.json`` and each state's patch as ``states/<k>.patch``."""
    states = []
    with tarfile.open(archive, "w:gz") as tar:
        for k, s in enumerate(plan.states):
            name = f"states/{k:04d}.patch" if s.file is not None else None
            if s.file is not None:
                tar.add(s.file, arcname=name, recursive=False)
            states.append({"seq": s.seq, "source": s.source, "t": s.t, "patch": s.patch, "file": name})
        data = json.dumps({"task": plan.task, "sent": plan.sent, "states": states}, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size, info.mtime = len(data), int(time.time())
        tar.addfile(info, io.BytesIO(data))


def verifier_check(plan: TrialPlan, doc: dict[str, Any]) -> tuple[bool | None, list[dict[str, Any]]]:
    """Whether the final state's per-bug resolved equals ``verifier/eval.json``'s, and the bugs that differ.
    A bug missing from eval.json's subtasks is unresolved there (its build failed: no subtask graded)."""
    ev = _json(plan.dir / "verifier" / "eval.json")
    final = next((s for s in reversed(doc["states"]) if s.get("source") == "final"), None)
    if not ev or ev.get("error") or final is None:
        return None, []
    resolved = {s.get("subtask_id"): bool(s.get("resolved")) for s in ev.get("subtasks") or [] if isinstance(s, dict)}
    mismatches = [{"subtask": sid, "replay": final["bugs"][sid]["status"], "verifier_resolved": resolved.get(sid, False)}
                  for sid in doc["sent"] if (final["bugs"][sid]["status"] == "pass") != resolved.get(sid, False)]
    return not mismatches, mismatches


def _check(doc: Any, plan: TrialPlan) -> None:
    if not isinstance(doc, dict) or doc.get("version") != VERSION:
        raise FixTimesError(f"replay output is not version {VERSION}")
    states = doc.get("states")
    if not isinstance(states, list) or len(states) != len(plan.states) or doc.get("sent") != plan.sent:
        raise FixTimesError(f"replay output has {len(states) if isinstance(states, list) else 'no'} states for "
                            f"{len(plan.states)}, sent {doc.get('sent')} for {plan.sent}")


class _Images:
    """One image per task, built on first use; a failed build fails that task's trials of this run."""

    def __init__(self, runner: Any, root: Path):
        self.runner, self.root = runner, root
        self.lock = threading.Lock()
        self.task_locks: dict[str, threading.Lock] = {}
        self.built: dict[str, tuple[Any, dict[str, Any]] | Exception] = {}

    def get(self, task: str, tests_dir: Path) -> tuple[Any, dict[str, Any]]:
        with self.lock:
            lock = self.task_locks.setdefault(task, threading.Lock())
        with lock:
            if task not in self.built:
                dockerfile = tests_dir / "Dockerfile"
                try:
                    if not dockerfile.is_file():
                        raise FixTimesError(f"no verifier Dockerfile at {dockerfile}")
                    image = self.runner.build(dockerfile, tests_dir)
                    self.built[task] = (image, {"dockerfile": _rel(dockerfile, self.root),
                                                "tests_sha256": sweep._tree_hash(tests_dir),
                                                "id": getattr(image, "object_id", None)})
                except Exception as e:  # every trial of the task reports it
                    self.built[task] = e
            got = self.built[task]
        if isinstance(got, Exception):
            raise FixTimesError(str(got) if isinstance(got, FixTimesError)
                                else f"image of {task}: {type(got).__name__}: {got}")
        return got


class _Live:
    """The sandboxes running now, to terminate on Ctrl-C; one created after that is refused."""

    def __init__(self, runner: Any):
        self.runner, self.lock, self.boxes, self.stopped = runner, threading.Lock(), {}, False

    def add(self, box: Any) -> bool:
        with self.lock:
            if not self.stopped:
                self.boxes[id(box)] = box
            return not self.stopped

    def discard(self, box: Any) -> None:
        with self.lock:
            self.boxes.pop(id(box), None)

    def terminate_all(self) -> None:
        with self.lock:
            boxes, self.boxes, self.stopped = list(self.boxes.values()), {}, True
        for box in boxes:
            _terminate(self.runner, box)


def _terminate(runner: Any, box: Any) -> None:
    try:
        runner.terminate(box)
    except Exception:  # its own timeout ends it anyway
        pass


def replay_trial(spec: Spec, plan: TrialPlan, runner: Any, images: _Images, live: _Live) -> dict[str, Any]:
    """Replay one trial in its own sandbox and write its ``fix_states.json`` -> the document written."""
    image, info = images.get(plan.task, plan.tests_dir)
    with tempfile.TemporaryDirectory(prefix="fixtimes-") as tmp:
        archive, got = Path(tmp) / "input.tar.gz", Path(tmp) / "out.json"
        pack(plan, archive)
        box = runner.create(image)
        if not live.add(box):
            _terminate(runner, box)
            raise FixTimesError("interrupted")
        try:
            runner.upload(box, SCRIPT, f"{REMOTE}/replay_states.py")
            runner.upload(box, archive, f"{REMOTE}/input.tar.gz")
            rc, tail = runner.exec(box, [GRADER_PYTHON, f"{REMOTE}/replay_states.py", "--tests", TESTS,
                                         "--input", f"{REMOTE}/input.tar.gz", "--out", f"{REMOTE}/out.json"],
                                   TESTS, EXEC_TIMEOUT_SEC)
            if rc != 0:
                raise FixTimesError(f"the replay exited {rc}: ...{(tail or '').strip()[-1500:]}")
            runner.download(box, f"{REMOTE}/out.json", got)
        finally:
            live.discard(box)
            _terminate(runner, box)
        try:
            doc = json.loads(got.read_text())
        except (OSError, ValueError) as e:
            raise FixTimesError(f"unreadable replay output: {e}") from None
    _check(doc, plan)
    matches, mismatches = verifier_check(plan, doc)
    doc = {"version": doc["version"], "trial": plan.name, "task": doc.get("task"), "graded_at": time.time(),
           "image": info, "final_matches_verifier": matches, "mismatches": mismatches,
           **{k: v for k, v in doc.items() if k not in ("version", "task")}}
    out = plan.dir / "agent" / FIX_STATES
    partial = out.with_name(FIX_STATES + ".partial")
    partial.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(partial, out)
    return doc


def _summary(doc: dict[str, Any]) -> str:
    final = doc["states"][-1]
    n_pass = sum(b["status"] == "pass" for b in final["bugs"].values())
    verdict = {True: "as in verifier/eval.json", None: "no verifier/eval.json to compare"}.get(
        doc["final_matches_verifier"], "UNLIKE verifier/eval.json: " + ", ".join(
            f"{m['subtask']} (replay {m['replay']}, verifier {'resolved' if m['verifier_resolved'] else 'unresolved'})"
            for m in doc["mismatches"]))
    return f"{len(doc['states'])} states; final state: {n_pass} of {len(doc['sent'])} bugs pass, {verdict}"


def _describe(plan: TrialPlan) -> str:
    def at(s: State) -> str:
        t = "?" if s.t is None or plan.t0 is None else f"{s.t - plan.t0:.0f}s"
        return f"{s.seq if s.source == 'snapshot' else 'final'}@{t}" + ("" if s.file is not None else " (no patch file)")
    return ", ".join(at(s) for s in plan.states)


def fix_times(spec: Spec, force: bool = False, concurrency: int = 24, only: str | None = None,
              dry_run: bool = False, runner: Any = None) -> int:
    """Replay the sweep's trials (module docstring); ``dry_run`` lists the trials and states and
    touches nothing. ``runner`` defaults to a :class:`ModalRunner`. Returns the exit code."""
    if not sweep.interactive(spec):
        say(f"sweep {spec.name!r} has no interactive condition: nothing to replay")
        return 0
    plans, skipped = plan_trials(spec, force, only)
    why = ", ".join(f"{n} {k}" for k, n in skipped.items() if n)
    say(f"fix-times {spec.name!r}: {len(plans)} trial(s) to replay" + (f" (skipped: {why})" if why else ""))
    if dry_run:
        for i, p in enumerate(plans, 1):
            dockerfile = p.tests_dir / "Dockerfile"
            say(f"  [{i}] {p.name}: {p.task}, {len(p.states) - 1} snapshot state(s) + final, "
                f"{len(p.sent)} bug(s) sent: {', '.join(p.sent) or '-'}")
            say(f"      image: {_rel(dockerfile, spec.root)}" + ("" if dockerfile.is_file() else " (MISSING)"))
            say(f"      states (seq@seconds into the session): {_describe(p)}")
        return 0
    if not plans:
        return 0
    if runner is None:
        ok, detail = sweep.modal_ready()
        if not ok:
            say(f"[FAIL] modal: {detail}")
            return 2
        runner = ModalRunner()
    images, live, failed, t0 = _Images(runner, spec.root), _Live(runner), 0, time.time()
    pool = ThreadPoolExecutor(concurrency, thread_name_prefix="fixtimes")
    futures = {pool.submit(replay_trial, spec, p, runner, images, live): p for p in plans}
    try:
        for f in as_completed(futures):
            p = futures[f]
            try:
                doc = f.result()
            except Exception as e:  # one trial's failure does not lose the others'
                failed += 1
                say(f"  [FAIL] {p.name}: {e if isinstance(e, FixTimesError) else f'{type(e).__name__}: {e}'}"[:2000])
                continue
            say(f"  [OK] {p.name}: {_summary(doc)}")
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        live.terminate_all()
        say("interrupted: the running sandboxes were terminated; re-run the same command to resume")
        return 130
    pool.shutdown(wait=True)
    say(f"fix-times {spec.name!r}: {len(plans) - failed} of {len(plans)} trial(s) replayed in {time.time() - t0:.0f}s"
        + (f"; {failed} failed: re-run to retry them" if failed else ""))
    return 1 if failed else 0


class ModalRunner:
    """The Modal side of :func:`fix_times`, on the modal client's blocking API (safe from threads):
    ``build`` an image, ``create`` a sandbox from it, ``upload``/``download`` a file, ``exec`` an argv
    (-> exit code, the tail of its output), ``terminate``."""

    def __init__(self, app_name: str = APP_NAME):
        os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", sweep.MODAL_IMAGE_BUILDER)
        import modal

        self.modal = modal
        self.app = modal.App.lookup(app_name, create_if_missing=True)

    def build(self, dockerfile: Path, context_dir: Path) -> Any:
        """Built now, so that the task's sandboxes share one build."""
        return self.modal.Image.from_dockerfile(str(dockerfile), context_dir=str(context_dir)).build(self.app)

    def create(self, image: Any) -> Any:
        return self.modal.Sandbox.create("sleep", "infinity", app=self.app, image=image, cpu=CPUS, memory=MEMORY_MB,
                                         block_network=True, timeout=SANDBOX_TIMEOUT_SEC)

    def upload(self, box: Any, local: Path, remote: str) -> None:
        box.filesystem.copy_from_local(str(local), remote)

    def exec(self, box: Any, argv: list[str], cwd: str, timeout: int) -> tuple[int, str]:
        proc = box.exec("bash", "-c", RUN_SH, "bash", *argv, workdir=cwd, timeout=timeout)
        output = proc.stdout.read()
        return proc.wait(), output

    def download(self, box: Any, remote: str, local: Path) -> None:
        box.filesystem.copy_to_local(remote, str(local))

    def terminate(self, box: Any) -> None:
        box.terminate()
