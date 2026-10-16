"""fix-times: the in-sandbox replay against a stub of the SWE-sweep grader (scripted verdicts), and the
host flow (trials, states, Modal sandboxes, fix_states.json, the CLI) against fake runners; one
check runs the replay's outcome recorder on the real grader's ``_run_targets`` when its task dir is here."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from subagents_bench import cli, fixtimes, replay_states, sweep
from subagents_bench.replay_states import classify, record_outcomes, replay, transitions
from test_interactive import interactive_spec

REPO = Path(__file__).resolve().parents[1]
CLICK_TESTS = next((p / "class-6-swe-sweep/swe-sweep/tasks/swe-sweep__click/tests" for p in REPO.parents
                    if (p / "class-6-swe-sweep/swe-sweep/tasks/swe-sweep__click/tests/shared").is_dir()), None)
T0 = 1_000_000.0

# A stub of tests/shared/swesweep_grade.py: run_evaluation reads a JSON script from the patch. Per bug,
# "pass" | "fail" | "collect" (pytest collection error) | "build" (Go build failure) | "mixed" (a failed
# test and an errored one) | "noreport" | "noapply" (hidden test patch), or "a,b,..." for successive
# evaluations of the same patch (a rerun); an absent bug fails. "_error", "_raise", "_build" and
# "_unapplied" script the whole evaluation. Every call is logged to $FAKE_GRADER_CALLS. The targeted
# runs go through a module-level _run_targets(env, suite, ...) -> suite.run_selected, as in the real one.
FAKE_GRADER = r'''
import json, os
from pathlib import Path

_seen = {}
OUTCOMES = {"pass": {"t::{sid}": "passed"}, "fail": {"t::{sid}": "failed", "t::{sid}_other": "passed"},
            "collect": {"tests.test_m": "error"}, "build": {".::<build>": "error"},
            "mixed": {"t::{sid}": "failed", "tests.test_x": "error"}, "noreport": None}


def load_bundle(tests_dir):
    return json.loads((Path(tests_dir) / "swesweep_eval.json").read_text())


def normalize_test_id(test_id):
    return test_id.strip()


class Suite:
    def __init__(self, outcomes):
        self.outcomes, self.no_targets = outcomes, "no targets"

    def run_selected(self, env, repo, files, names, timeout, capture, jobs):
        return (None if self.outcomes is None else {f" {k}": v for k, v in self.outcomes.items()}), False, {}


def _run_targets(env, suite, config, files, names, capture, jobs, result, f2p=frozenset()):
    outcomes, timed_out, output = suite.run_selected(env, "/repo", files, names, 10, capture, jobs)
    if timed_out or outcomes is None:
        result["detail"]["error"] = "targeted tests produced no report"
        return output
    outcomes = {normalize_test_id(k): v for k, v in outcomes.items()}
    failed = sorted(k for k, v in outcomes.items() if v in ("failed", "error"))
    passed = sorted(k for k, v in outcomes.items() if v == "passed")
    result.update(n_passed=len(passed), n_failed=len(failed), passed=passed, failed=failed,
                  resolved=not failed and bool(passed))
    return output


def run_evaluation(bundle, tests_dir, agent_patch):
    assert agent_patch is not None, "never the oracle"
    if os.environ.get("FAKE_GRADER_CALLS"):
        with open(os.environ["FAKE_GRADER_CALLS"], "a") as f:
            f.write(json.dumps({"patch": agent_patch.decode(), "subtasks": [s["id"] for s in bundle["subtasks"]]}) + "\n")
    script = json.loads(agent_patch) if agent_patch.strip() else {}
    if "_raise" in script:
        raise RuntimeError(script["_raise"])
    if "_error" in script:
        return {"error": script["_error"], "stderr_tail": "Traceback: deep", "rc": None}
    out = {"visible_ok": script.get("_visible", True), "regressions": [], "subtasks": {}, "timings": {},
           "detail": {"agent_patch_applied": not script.get("_unapplied")}}
    if script.get("_build"):
        out["visible_ok"] = False
        out["detail"].update(agent_build_ok=False, build_error="cc: error")
        return out
    for s in bundle["subtasks"]:
        sid = s["id"]
        steps = script.get(sid, "fail").split(",")
        k = _seen[(agent_patch, sid)] = _seen.get((agent_patch, sid), -1) + 1
        how = steps[min(k, len(steps) - 1)]
        result = {"resolved": False, "test_applied": True, "targeted": [], "n_passed": 0, "n_failed": 0,
                  "passed": [], "failed": [], "detail": {}}
        if how == "noapply":
            result["test_applied"] = False
            result["detail"]["error"] = "hidden test patch did not apply: corrupt"
        else:
            outcomes = OUTCOMES[how] and {k.format(sid=sid): v for k, v in OUTCOMES[how].items()}
            _run_targets(None, Suite(outcomes), None, [], [], None, "4", result)
        out["subtasks"][sid] = result
    return out
'''


def make_tests_dir(d: Path, ids=("b0", "b1", "b2", "b3")) -> Path:
    """A task's tests/ dir with the stub grader, a bundle of ``ids`` and a Dockerfile."""
    (d / "shared").mkdir(parents=True, exist_ok=True)
    (d / "shared" / "swesweep_grade.py").write_text(FAKE_GRADER)
    (d / "swesweep_eval.json").write_text(json.dumps({"task": "task-1", "subtasks": [{"id": i} for i in ids]}))
    (d / "Dockerfile").write_text("FROM scratch\nCOPY . /tests\n")
    return d


@pytest.fixture
def grader(tmp_path, monkeypatch):
    """The stub grader, loaded the way the replay loads it, with its calls logged."""
    tests = make_tests_dir(tmp_path / "tests")
    monkeypatch.setenv("FAKE_GRADER_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.delitem(sys.modules, "swesweep_grade", raising=False)
    g = replay_states.load_grader(tests)
    yield SimpleNamespace(module=g, tests=tests, calls=lambda: [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
                          if (tmp_path / "calls.jsonl").is_file() else [])
    sys.modules.pop("swesweep_grade", None)


def states_input(d: Path, scripts: list, sent: list[str]) -> dict:
    """An input dir: one state per script (a dict -> its patch; "" -> an empty patch; None -> no file)."""
    d.mkdir(parents=True, exist_ok=True)
    states = []
    for k, s in enumerate(scripts):
        name = None if s is None else f"states/{k:04d}.patch"
        if name:
            (d / "states").mkdir(exist_ok=True)
            (d / name).write_text(json.dumps(s) if isinstance(s, dict) else s)
        final = k == len(scripts) - 1
        states.append({"seq": None if final else k, "source": "final" if final else "snapshot", "t": T0 + 10 * k,
                       "patch": "../artifacts/logs/artifacts/model.patch" if final else f"snapshots/{k:04d}.patch",
                       "file": name})
    manifest = {"task": "task-1", "sent": sent, "states": states}
    (d / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def statuses(doc: dict, sid: str, key: str = "status") -> list:
    return [s["bugs"][sid][key] for s in doc["states"]]


# --- replay_states: statuses, reruns, the sent bugs -----------------------------------------------


def test_replay_grades_every_state_and_reruns_pass_to_fail(grader, tmp_path):
    scripts = [
        "",                                                   # 0: snapshot 0, the baseline: every bug fails
        {"b0": "pass"},                                       # 1
        {"b0": "fail,pass", "b1": "collect"},                 # 2: b0 pass->fail, its rerun passes; b1 unknown
        {"b0": "pass", "b1": "pass"},                         # 3
        {"_error": "evaluation failed: boom"},                # 4: nothing known
        {"b0": "collect", "b1": "pass"},                      # 5: b0 unknown, b1 still passes
        {"b0": "fail", "b1": "fail,fail"},                    # 6: both fail after a pass (b0's across unknowns)
        {"b0": "pass", "b1": "pass", "b2": "pass"},           # final (b2 was never sent)
    ]
    manifest = states_input(tmp_path / "in", scripts, sent=["b1", "b0", "b9"])
    lines: list[str] = []
    doc = replay(grader.module, grader.tests, manifest, tmp_path / "in", log=lines.append)
    assert doc["version"] == 1 and doc["task"] == "task-1" and doc["sent"] == ["b1", "b0", "b9"]
    assert statuses(doc, "b0") == ["fail", "pass", "fail", "pass", "unknown", "unknown", "fail", "pass"]
    assert statuses(doc, "b0", "rerun") == [None, None, "pass", None, None, None, "fail", None]
    assert statuses(doc, "b1") == ["fail", "fail", "unknown", "pass", "unknown", "pass", "fail", "pass"]
    assert statuses(doc, "b1", "rerun") == [None] * 6 + ["fail", None]
    assert statuses(doc, "b9") == ["unknown"] * 8 and statuses(doc, "b9", "error")[0] == "not in the task's bundle"
    assert doc["states"][2]["bugs"]["b1"]["error"].startswith("targeted tests errored (collection or build failure): ")
    assert doc["states"][1]["bugs"]["b0"] == {"status": "pass", "n_passed": 1, "n_failed": 0, "error": None, "rerun": None}
    assert doc["states"][0]["bugs"]["b0"] == {"status": "fail", "n_passed": 1, "n_failed": 1, "error": None, "rerun": None}
    s4 = doc["states"][4]
    assert (s4["applied"], s4["visible_ok"], s4["error"]) == (None, None, "evaluation failed: boom | Traceback: deep")
    assert statuses(doc, "b1", "error")[4] == "evaluation error"
    s0, final = doc["states"][0], doc["states"][-1]
    assert (s0["seq"], s0["source"], s0["t"], s0["patch"], s0["applied"], s0["visible_ok"], s0["error"]) == (
        0, "snapshot", T0, "snapshots/0000.patch", True, True, None)
    assert (final["seq"], final["source"], final["patch"]) == (None, "final", "../artifacts/logs/artifacts/model.patch")
    assert all(isinstance(s["duration_s"], float) for s in doc["states"])
    assert [s["rerun_duration_s"] is not None for s in doc["states"]] == [False, False, True] + [False] * 3 + [True, False]
    calls = grader.calls()
    assert calls[0] == {"patch": "", "subtasks": ["b0", "b1"]}  # the empty patch is b"", the bundle cut to the sent bugs
    assert [c["subtasks"] for c in calls[8:]] == [["b0"], ["b0", "b1"]]  # the reruns: only the bugs that need one
    assert len(calls) == 10 and len(lines) == 11 and lines[0].startswith("not in the task's bundle")


def test_replay_unknown_reasons_and_a_missing_patch(grader, tmp_path):
    scripts = [
        {"b0": "build", "b1": "mixed", "b2": "noreport", "b3": "noapply"},
        {"_build": True},
        {"_unapplied": True, "b0": "pass"},
        {"_raise": "segfault"},
        None,  # the host had no patch file: not graded at all
        {"b0": "pass"},
    ]
    manifest = states_input(tmp_path / "in", scripts, sent=["b0", "b1", "b2", "b3"])
    doc = replay(grader.module, grader.tests, manifest, tmp_path / "in", log=lambda _: None)
    first = doc["states"][0]["bugs"]
    assert {sid: b["status"] for sid, b in first.items()} == {"b0": "unknown", "b1": "fail", "b2": "unknown", "b3": "unknown"}
    assert first["b0"]["error"] == "targeted tests errored (collection or build failure): .::<build>"
    assert (first["b2"]["error"], first["b3"]["error"]) == ("targeted tests produced no report",
                                                            "hidden test patch did not apply: corrupt")
    build, unapplied, raised, missing = doc["states"][1:5]
    assert (build["error"], build["visible_ok"], build["bugs"]["b0"]["error"]) == (
        "agent build failed: cc: error", False, "agent build failed")
    assert (unapplied["applied"], unapplied["visible_ok"], unapplied["error"], unapplied["bugs"]["b0"]["status"]) == (
        False, None, "agent patch did not apply:", "unknown")
    assert raised["error"] == "evaluation raised RuntimeError: segfault" and raised["applied"] is None
    assert (missing["error"], missing["duration_s"], missing["bugs"]["b0"]["status"]) == ("patch file missing", 0.0, "unknown")
    assert len(grader.calls()) == 5 and all(c["subtasks"] == ["b0", "b1", "b2", "b3"] for c in grader.calls())
    assert statuses(doc, "b0", "rerun") == [None] * 6  # no pass -> fail anywhere


def test_classify_and_transitions_edge_cases():
    ok = {"detail": {"agent_patch_applied": True}}
    sub = lambda **r: {**ok, "subtasks": {"b": {"test_applied": True, "detail": {}, **r}}}  # noqa: E731
    assert classify(sub(resolved=True, n_passed=2), "b") == ("pass", 2, 0, None)
    assert classify(sub(failed=["x::<build>"], n_failed=1), "b")[0] == "unknown"  # no recorded outcomes: Go's id
    assert classify(sub(failed=["tests.test_m"], n_failed=1), "b")[0] == "fail"  # without outcomes, a pytest error is not told apart
    assert classify(sub(failed=["a"], n_failed=1, outcomes={"a": "error"}, n_passed=3), "b")[0] == "unknown"
    assert classify(sub(), "b") == ("unknown", 0, 0, "no targeted test ran")
    assert classify(sub(test_applied=False), "b") == ("unknown", 0, 0, "hidden test patch did not apply")
    assert classify(ok, "b") == ("unknown", 0, 0, "not graded")
    states = [{"bugs": {"b": {"status": s}}} for s in ("fail", "pass", "unknown", "unknown", "fail", "fail", "pass", "fail")]
    assert transitions(states, ["b"]) == {4: ["b"], 7: ["b"]}


def test_replay_script_runs_standalone_on_a_tarball(grader, tmp_path):
    """As in the sandbox: a script run by path (stdlib only), the input packed, the output written."""
    states_input(tmp_path / "in", ["", {"b0": "pass"}], sent=["b0"])
    with tarfile.open(tmp_path / "input.tar.gz", "w:gz") as tar:
        tar.add(tmp_path / "in", arcname=".")
    out = tmp_path / "out" / "fix.json"
    r = subprocess.run([sys.executable, "-I", str(replay_states.__file__), "--tests", str(grader.tests),
                        "--input", str(tmp_path / "input.tar.gz"), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert statuses(json.loads(out.read_text()), "b0") == ["fail", "pass"] and "state 2/2 final" in r.stdout


def test_replay_script_is_stdlib_only_python_312():
    """It runs on the verifier image's standalone Python 3.12, which has nothing else installed."""
    tree = ast.parse(Path(replay_states.__file__).read_text(), feature_version=(3, 12))
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
    assert mods and mods <= set(sys.stdlib_module_names) | {"__future__"}
    assert not any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree))


@pytest.mark.skipif(CLICK_TESTS is None, reason="no SWE-sweep task dir next to the repository")
def test_record_outcomes_on_the_real_graders_run_targets(monkeypatch):
    """The real grader's ``_run_targets`` (signature, suite call, id spelling) with the recorder."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # no __pycache__ in the task dir (a verifier image's context)
    monkeypatch.delitem(sys.modules, "swesweep_grade", raising=False)
    monkeypatch.syspath_prepend(str(CLICK_TESTS / "shared"))
    g = replay_states.load_grader(CLICK_TESTS)
    try:
        record_outcomes(g)
        record_outcomes(g)  # idempotent: one wrapper

        class Suite:
            def __init__(self, outcomes):
                self.outcomes = outcomes

            def run_selected(self, env, repo, files, names, timeout, capture, jobs):
                return self.outcomes, False, {"text": ""}

        def run(outcomes):
            result = {**g._new_subtask_result(), "test_applied": True}  # as _run_hidden_tests sets it first
            g._run_targets(None, Suite(outcomes), SimpleNamespace(repo="/click", test_timeout=5), ["tests/test_m.py"],
                           ["test_x"], None, "4", result, f2p=frozenset())
            return classify({"detail": {"agent_patch_applied": True}, "subtasks": {"b": result}}, "b"), result

        (status, *_), result = run({"tests.test_options": "error"})  # pytest's collection error
        assert status == "unknown" and result["outcomes"] == {"tests.test_options": "error"}
        assert run({".::<build>": "error"})[0][0] == "unknown"  # Go's build failure
        assert run({"tests.t::a": "failed", "tests.t::b": "error"})[0][0] == "fail"
        assert run({"tests.t::a": "passed"})[0][0] == "pass"
        assert run(None)[0][:1] == ("unknown",) and "produced no report" in run(None)[0][3]
        assert getattr(g._run_targets, "records_outcomes", False)
        assert "_run_targets" in g._run_hidden_tests.__code__.co_names  # looked up when called: the wrapper is used
    finally:
        sys.modules.pop("swesweep_grade", None)


# --- the host: trials, states, the runner, fix_states.json ---------------------------------------


@pytest.fixture
def spec(tmp_path):
    """An interactive sweep whose task-1 (bugs b0-b2) carries the stub grader and a Dockerfile."""
    s = sweep.load_spec(interactive_spec(tmp_path))
    make_tests_dir(sweep.task_root(s) / "task-1" / "tests", ids=("b0", "b1", "b2"))
    return s


def snap(seq: int, t: float, patch: bool = True, error: str | None = None) -> dict:
    return {"seq": seq, "t_request": T0 + t - 0.5, "t_start": T0 + t, "t_end": T0 + t + 0.3,
            "trigger": "start" if seq == 0 else "tool", "tree": None if error else f"tree{seq}",
            "changed": patch, "patch": f"snapshots/{seq:04d}.patch" if patch else None,
            "bytes": 10 if patch else None, "error": error}


def make_trial(spec, job: str = "t__r1__solo", name: str = "task-1__a", scripts: dict | None = None,
               final: dict | str | None = "", resolved: dict | None = None, finished: bool = True,
               snapshots: bool = True) -> Path:
    """A finished trial: bugs b0, b1 sent (b2 ready after the end), snapshots seq -> script (None: an
    unchanged or failed attempt), the final model.patch and verifier/eval.json."""
    d = spec.jobs_dir / job / name
    agent = d / "agent"
    (agent / "snapshots").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"task_name": "task-1", "finished_at": "2026-10-16T00:00:00Z" if finished else None}))
    (agent / "episode.json").write_text(json.dumps({"task_name": "task-1", "session_t0": T0, "t_end": T0 + 1800.0}))
    inbox = [{"kind": "hint", "msg_id": 2, "t_ready": T0 + 240, "subtask_id": "b1"},
             {"kind": "question", "msg_id": 3, "t_ready": T0 + 300, "subtask_id": None},
             {"kind": "hint", "msg_id": 1, "t_ready": T0, "subtask_id": "b0"},
             {"kind": "hint", "msg_id": 9, "t_ready": T0 + 2400, "subtask_id": "b2"}]
    (agent / "inbox.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inbox))
    scripts = {0: "", 1: None, 3: {"b0": "pass"}, 2: None, 5: {"b0": "pass", "b1": "pass"}} if scripts is None else scripts
    if snapshots:
        recs = [snap(seq, 15.0 * seq, s is not None, error="timeout" if s is None and seq == 2 else None)
                for seq, s in scripts.items()]
        (agent / "snapshots.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs) + "{cut short")
        for seq, s in scripts.items():
            if s is not None:
                (agent / "snapshots" / f"{seq:04d}.patch").write_text(json.dumps(s) if isinstance(s, dict) else s)
    if final is not None:
        (d / fixtimes.FINAL_PATCH).parent.mkdir(parents=True)
        (d / fixtimes.FINAL_PATCH).write_text(json.dumps(final) if isinstance(final, dict) else final)
    if resolved is not None:
        (d / "verifier").mkdir()
        (d / "verifier" / "eval.json").write_text(json.dumps(
            {"error": None, "subtasks": [{"subtask_id": k, "resolved": v} for k, v in resolved.items()]}))
    return d


class LocalRunner:
    """A fake Modal: a sandbox is a dir, files are copied, and the replay runs here on the stub grader
    (the image's ``/tests`` is the dir it was built from; its python is this one)."""

    def __init__(self, tmp: Path):
        self.tmp, self.log, self.lock, self.n_boxes = tmp, [], threading.Lock(), 0
        self.calls = tmp / "grader_calls.jsonl"

    def _note(self, *event) -> None:
        with self.lock:
            self.log.append(event)

    def build(self, dockerfile: Path, context_dir: Path):
        self._note("build", dockerfile, context_dir)
        return SimpleNamespace(object_id="im-123", tests=context_dir)

    def create(self, image):
        with self.lock:
            self.n_boxes += 1
            root = self.tmp / f"box{self.n_boxes}"
            self.log.append(("create", root.name))
        root.mkdir()
        return SimpleNamespace(root=root, image=image)

    def _path(self, box, remote: str) -> Path:
        return box.image.tests if remote == fixtimes.TESTS else box.root / remote.lstrip("/")

    def upload(self, box, local: Path, remote: str) -> None:
        self._note("upload", box.root.name, remote)
        self._path(box, remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, self._path(box, remote))

    def exec(self, box, argv: list[str], cwd: str, timeout: int):
        self._note("exec", box.root.name, timeout)
        assert argv[0] == fixtimes.GRADER_PYTHON and cwd == fixtimes.TESTS and timeout == fixtimes.EXEC_TIMEOUT_SEC
        args = [sys.executable, "-I", "-B"] + [str(self._path(box, a)) if a.startswith("/") else a for a in argv[1:]]
        r = subprocess.run(args, cwd=box.image.tests, capture_output=True, text=True,
                           env={**os.environ, "FAKE_GRADER_CALLS": str(self.calls)})
        return r.returncode, (r.stdout + r.stderr)[-4000:]

    def download(self, box, remote: str, local: Path) -> None:
        self._note("download", box.root.name, remote)
        shutil.copy(self._path(box, remote), local)

    def terminate(self, box) -> None:
        self._note("terminate", box.root.name)


def fix(spec, path=None, *args, runner=None, monkeypatch=None) -> int:
    """``subagents-bench fix-times`` through the CLI, with ``runner`` standing in for Modal."""
    monkeypatch.setattr(sweep, "modal_ready", lambda: (True, "fake"))
    monkeypatch.setattr(fixtimes, "ModalRunner", lambda: runner)
    return cli.main(["fix-times", str(path or spec.root / "spec.yaml"), *args])


def test_trial_plan_states_and_sent_bugs(spec):
    d = make_trial(spec)
    (d / "agent" / "snapshots" / "0005.patch").unlink()  # a patch the host lost
    p = fixtimes.trial_plan(spec, d)
    assert (p.name, p.task, p.tests_dir, p.t0) == ("t__r1__solo/task-1__a", "task-1", sweep.task_root(spec) / "task-1" / "tests", T0)
    assert p.sent == ["b0", "b1"]  # in sending order; b2 was ready after the session ended, the question is no bug
    assert [(s.seq, s.source, s.t, s.patch) for s in p.states] == [
        (0, "snapshot", T0, "snapshots/0000.patch"), (3, "snapshot", T0 + 45, "snapshots/0003.patch"),
        (5, "snapshot", T0 + 75, "snapshots/0005.patch"), (None, "final", T0 + 1800, "../artifacts/logs/artifacts/model.patch")]
    assert [s.file for s in p.states] == [d / "agent/snapshots/0000.patch", d / "agent/snapshots/0003.patch", None,
                                          d / "artifacts/logs/artifacts/model.patch"]
    fixtimes.pack(p, spec.root / "in.tar.gz")
    with tarfile.open(spec.root / "in.tar.gz") as tar:
        names = sorted(tar.getnames())
        manifest = json.loads(tar.extractfile("manifest.json").read())
    assert names == ["manifest.json", "states/0000.patch", "states/0001.patch", "states/0003.patch"]
    assert [s["file"] for s in manifest["states"]] == ["states/0000.patch", "states/0001.patch", None, "states/0003.patch"]
    assert (manifest["task"], manifest["sent"]) == ("task-1", ["b0", "b1"])


def test_fix_times_end_to_end(spec, tmp_path, monkeypatch, capsys):
    a = make_trial(spec, resolved={"b0": True, "b1": True, "b2": False},
                   final={"b0": "pass", "b1": "pass", "b2": "pass"})
    b = make_trial(spec, job="t__r2__solo", name="task-1__b", resolved={"b0": True, "b1": True},  # b1 differs
                   scripts={0: "", 1: {"b0": "pass", "b1": "pass"}, 2: {"b0": "fail", "b1": "pass"}}, final={"b0": "pass"})
    make_trial(spec, job="t__r1__solo", name="task-1__c", finished=False)
    make_trial(spec, job="t__r1__solo", name="task-1__d", snapshots=False)
    make_trial(spec, job="_partial", name="task-1__e")
    (spec.jobs_dir / "tasks" / "task-1" / "agent").mkdir(parents=True)  # a staged task dir: no trial
    runner = LocalRunner(tmp_path)
    assert fix(spec, None, "--concurrency", "2", runner=runner, monkeypatch=monkeypatch) == 0
    out = capsys.readouterr().out
    assert "2 trial(s) to replay (skipped: 1 unfinished, 1 without snapshots)" in out
    assert "[OK] t__r1__solo/task-1__a: 4 states; final state: 2 of 2 bugs pass, as in verifier/eval.json" in out
    assert ("[OK] t__r2__solo/task-1__b: 4 states; final state: 1 of 2 bugs pass, UNLIKE verifier/eval.json: "
            "b1 (replay fail, verifier resolved)" in out) and "2 of 2 trial(s) replayed" in out
    doc = json.loads((a / "agent" / "fix_states.json").read_text())
    assert list(doc)[:7] == ["version", "trial", "task", "graded_at", "image", "final_matches_verifier", "mismatches"]
    assert (doc["version"], doc["trial"], doc["task"], doc["sent"]) == (1, "t__r1__solo/task-1__a", "task-1", ["b0", "b1"])
    assert doc["image"]["dockerfile"] == "tasks/task-1/tests/Dockerfile" and doc["image"]["id"] == "im-123"
    assert doc["image"]["tests_sha256"] == sweep._tree_hash(sweep.task_root(spec) / "task-1" / "tests")
    assert (doc["final_matches_verifier"], doc["mismatches"]) == (True, [])
    assert [(s["seq"], s["source"], s["t"] - T0) for s in doc["states"]] == [(0, "snapshot", 0), (3, "snapshot", 45),
                                                                           (5, "snapshot", 75), (None, "final", 1800)]
    assert statuses(doc, "b0") == ["fail", "pass", "pass", "pass"] and statuses(doc, "b1") == ["fail", "fail", "pass", "pass"]
    db = json.loads((b / "agent" / "fix_states.json").read_text())
    assert db["final_matches_verifier"] is False
    assert db["mismatches"] == [{"subtask": "b1", "replay": "fail", "verifier_resolved": True}]
    assert statuses(db, "b0") == ["fail", "pass", "fail", "pass"] and statuses(db, "b0", "rerun") == [None, None, "fail", None]
    assert statuses(db, "b1") == ["fail", "pass", "pass", "fail"] and statuses(db, "b1", "rerun") == [None, None, None, "fail"]
    assert sum(e[0] == "build" for e in runner.log) == 1  # one image per task, shared by its trials
    boxes = {e[1] for e in runner.log if e[0] == "create"}
    for box in boxes:  # each sandbox: the script and the input up, one run, the output down, terminated
        assert [e[0] for e in runner.log if len(e) > 1 and e[1] == box] == ["create", "upload", "upload", "exec",
                                                                             "download", "terminate"]
    assert {e[2] for e in runner.log if e[0] == "upload"} == {"/fixtimes/replay_states.py", "/fixtimes/input.tar.gz"}
    calls = [json.loads(x) for x in runner.calls.read_text().splitlines()]
    # 4 + 4 states, then b's reruns: b0 at seq 2, b1 in the final state
    assert sorted(tuple(c["subtasks"]) for c in calls) == [("b0",)] + [("b0", "b1")] * 8 + [("b1",)]
    # resumable: done trials are kept; --force replays them; --only-trial picks some
    runner.log.clear()
    assert fix(spec, None, runner=runner, monkeypatch=monkeypatch) == 0
    assert "0 trial(s) to replay (skipped: 1 unfinished, 1 without snapshots, 2 already replayed)" in capsys.readouterr().out
    assert runner.log == []
    assert fix(spec, None, "--force", "--only-trial", "r2__solo/task-1__b", runner=runner, monkeypatch=monkeypatch) == 0
    assert "1 trial(s) to replay" in capsys.readouterr().out and sum(e[0] == "exec" for e in runner.log) == 1
    (a / "verifier" / "eval.json").unlink()
    assert fix(spec, None, "--force", "--only-trial", "task-1__a", runner=runner, monkeypatch=monkeypatch) == 0
    assert "final state: 2 of 2 bugs pass, no verifier/eval.json to compare" in capsys.readouterr().out
    assert json.loads((a / "agent" / "fix_states.json").read_text())["final_matches_verifier"] is None


def test_fix_times_dry_run_touches_nothing(spec, monkeypatch, capsys):
    d = make_trial(spec)
    (d / "agent" / "snapshots" / "0003.patch").unlink()
    monkeypatch.setattr(sweep, "modal_ready", lambda: pytest.fail("no Modal on a dry run"))
    monkeypatch.setattr(fixtimes, "ModalRunner", lambda: pytest.fail("no Modal on a dry run"))
    assert cli.main(["fix-times", str(spec.root / "spec.yaml"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[1] t__r1__solo/task-1__a: task-1, 3 snapshot state(s) + final, 2 bug(s) sent: b0, b1" in out
    assert "image: tasks/task-1/tests/Dockerfile\n" in out
    assert "states (seq@seconds into the session): 0@0s, 3@45s (no patch file), 5@75s, final@1800s" in out
    assert not (d / "agent" / "fix_states.json").exists()
    (sweep.task_root(spec) / "task-1" / "tests" / "Dockerfile").unlink()
    assert cli.main(["fix-times", str(spec.root / "spec.yaml"), "--dry-run"]) == 0
    assert "image: tasks/task-1/tests/Dockerfile (MISSING)" in capsys.readouterr().out


class ScriptedRunner:
    """A fake Modal whose steps can fail: ``exec`` -> (rc, tail), and an output to download."""

    def __init__(self, rc: int = 0, tail: str = "", output: dict | str | None = None, fail: dict | None = None):
        self.rc, self.tail, self.output, self.fail = rc, tail, output, fail or {}
        self.log: list[tuple] = []  # every step tried
        self.boxes: list[str] = []  # the sandboxes created
        self.lock = threading.Lock()

    def _step(self, name: str, *args):
        self.log.append((name, *args))
        if name in self.fail:
            raise self.fail[name]

    def build(self, dockerfile, context_dir):
        self._step("build")
        return object()  # no object_id: an image id is optional

    def create(self, image):
        self._step("create")
        with self.lock:
            self.boxes.append(f"box{len(self.boxes) + 1}")
            return self.boxes[-1]

    def upload(self, box, local, remote):
        self._step("upload", box)

    def exec(self, box, argv, cwd, timeout):
        self._step("exec", box)
        return self.rc, self.tail

    def download(self, box, remote, local):
        self._step("download", box)
        Path(local).write_text(self.output if isinstance(self.output, str) else json.dumps(self.output))

    def terminate(self, box):
        self.log.append(("terminate", box))


def ok_output(n_states: int, sent=("b0", "b1")) -> dict:
    bugs = {s: {"status": "fail", "n_passed": 0, "n_failed": 1, "error": None, "rerun": None} for s in sent}
    states = [{"seq": k, "source": "snapshot", "t": T0, "patch": "p", "applied": True, "visible_ok": True, "error": None,
               "duration_s": 1.0, "rerun_duration_s": None, "bugs": bugs} for k in range(n_states)]
    states[-1]["source"] = "final"
    return {"version": 1, "task": "task-1", "sent": list(sent), "states": states}


@pytest.mark.parametrize("runner, why", [
    (ScriptedRunner(rc=1, tail="Traceback ...\nFixtureError: missing base_tests.json.gz\n"),
     "the replay exited 1: ...Traceback ...\nFixtureError: missing base_tests.json.gz"),
    (ScriptedRunner(output=ok_output(2)), "replay output has 2 states for 4, sent ['b0', 'b1'] for ['b0', 'b1']"),
    (ScriptedRunner(output={**ok_output(4), "version": 2}), "replay output is not version 1"),
    (ScriptedRunner(output="{not json"), "unreadable replay output: "),
    (ScriptedRunner(fail={"create": RuntimeError("quota")}), "RuntimeError: quota"),
    (ScriptedRunner(fail={"download": FileNotFoundError("out.json")}), "FileNotFoundError: out.json"),
    (ScriptedRunner(fail={"build": RuntimeError("build step 3 failed")}),
     "image of task-1: RuntimeError: build step 3 failed"),
])
def test_fix_times_failures_write_nothing(spec, monkeypatch, capsys, runner, why):
    trials = [make_trial(spec, name=f"task-1__{x}", resolved={"b0": False, "b1": False}) for x in "ab"]
    assert fix(spec, None, runner=runner, monkeypatch=monkeypatch) == 1
    out = capsys.readouterr().out
    assert f"[FAIL] t__r1__solo/task-1__a: {why}" in out and "0 of 2 trial(s) replayed" in out
    assert not any((t / "agent" / "fix_states.json").exists() for t in trials)
    assert sum(e[0] == "build" for e in runner.log) == 1  # a failed build is not retried within the run
    assert sorted(runner.boxes) == sorted(e[1] for e in runner.log if e[0] == "terminate")  # every sandbox ends
    runner2 = ScriptedRunner(output=ok_output(4))  # the next run replays them
    assert fix(spec, None, runner=runner2, monkeypatch=monkeypatch) == 0
    assert all((t / "agent" / "fix_states.json").is_file() for t in trials)


def test_fix_times_without_a_dockerfile_or_modal(spec, monkeypatch, capsys):
    make_trial(spec)
    (sweep.task_root(spec) / "task-1" / "tests" / "Dockerfile").unlink()
    runner = ScriptedRunner(output=ok_output(4))
    assert fix(spec, None, runner=runner, monkeypatch=monkeypatch) == 1
    assert f"no verifier Dockerfile at {sweep.task_root(spec) / 'task-1' / 'tests' / 'Dockerfile'}" in capsys.readouterr().out
    assert runner.log == []
    monkeypatch.setattr(sweep, "modal_ready", lambda: (False, "no Modal token"))
    monkeypatch.setattr(fixtimes, "ModalRunner", lambda: pytest.fail("not without a token"))
    assert cli.main(["fix-times", str(spec.root / "spec.yaml")]) == 2
    assert "[FAIL] modal: no Modal token" in capsys.readouterr().out
    assert cli.main(["fix-times", str(spec.root / "spec.yaml"), "--concurrency", "0"]) == 2


def test_fix_times_ctrl_c_terminates_running_sandboxes(spec, monkeypatch, capsys):
    """A Ctrl-C while trials run: every live sandbox is terminated, nothing is written, exit 130."""
    trials = [make_trial(spec, name=f"task-1__{x}") for x in "ab"]
    both, killed = threading.Barrier(2, timeout=10), threading.Event()

    class Runner(ScriptedRunner):
        def exec(self, box, argv, cwd, timeout):
            both.wait()
            if box == "box1":
                raise KeyboardInterrupt
            assert killed.wait(10)
            return 137, "terminated"

        def terminate(self, box):
            super().terminate(box)
            if box == "box2":
                killed.set()

    runner = Runner()
    assert fix(spec, None, runner=runner, monkeypatch=monkeypatch) == 130
    assert "interrupted: the running sandboxes were terminated" in capsys.readouterr().out
    assert ("terminate", "box2") in runner.log and ("terminate", "box1") in runner.log
    assert not any((t / "agent" / "fix_states.json").exists() for t in trials)
    # a sandbox created after the Ctrl-C (its create call was under way) is terminated at once
    live, late = fixtimes._Live(ScriptedRunner()), ScriptedRunner()
    live.terminate_all()
    with pytest.raises(fixtimes.FixTimesError, match="interrupted"):
        fixtimes.replay_trial(spec, fixtimes.trial_plan(spec, trials[0]), late,
                              fixtimes._Images(late, spec.root), live)
    assert [e[0] for e in late.log] == ["build", "create", "terminate"]


def test_modal_runner_calls(monkeypatch):
    """The Modal side against a stand-in ``modal`` module: resources, no network, the bash wrapper."""
    seen: dict = {}

    class Proc:
        stdout = SimpleNamespace(read=lambda: "tail ☃".encode()[:-1])  # bytes: tail -c cut a character

        def wait(self):
            return 0

    class Box:
        filesystem = SimpleNamespace(copy_from_local=lambda src, dst: seen.setdefault("up", (src, dst)),
                                     copy_to_local=lambda src, dst: seen.setdefault("down", (src, dst)))

        def exec(self, *args, **kw):
            seen["exec"] = (args, kw)
            return Proc()

        def terminate(self):
            seen["terminated"] = True

    class Image:
        @staticmethod
        def from_dockerfile(path, context_dir):
            seen["dockerfile"] = (path, context_dir, os.environ.get("MODAL_IMAGE_BUILDER_VERSION"))
            return SimpleNamespace(build=lambda app: ("built", app))

    fake = SimpleNamespace(App=SimpleNamespace(lookup=lambda name, create_if_missing: f"app:{name}:{create_if_missing}"),
                           Image=Image, Sandbox=SimpleNamespace(create=lambda *a, **kw: seen.setdefault("create", (a, kw)) and Box()))
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.delenv("MODAL_IMAGE_BUILDER_VERSION", raising=False)
    r = fixtimes.ModalRunner()
    assert r.app == "app:subagents-bench-fixtimes:True"
    image = r.build(Path("/t/tests/Dockerfile"), Path("/t/tests"))
    assert image == ("built", r.app) and seen["dockerfile"] == ("/t/tests/Dockerfile", "/t/tests", sweep.MODAL_IMAGE_BUILDER)
    box = r.create(image)
    args, kw = seen["create"]
    assert args == ("sleep", "infinity") and kw == {"app": r.app, "image": image, "cpu": 3, "memory": 7000,
                                                    "block_network": True, "timeout": 3 * 3600}
    r.upload(box, Path("/x/in.tar.gz"), "/fixtimes/input.tar.gz")
    assert r.exec(box, ["py", "s.py", "--out", "o"], "/tests", 99) == (0, "tail \ufffd")  # decoded, not raised
    (cmd, *argv), kw = seen["exec"]
    assert cmd == "bash" and argv[:2] == ["-c", fixtimes.RUN_SH] and argv[2:] == ["bash", "py", "s.py", "--out", "o"]
    assert kw == {"workdir": "/tests", "timeout": 99, "text": False} and "ulimit -n" in fixtimes.RUN_SH
    r.download(box, "/fixtimes/out.json", Path("/x/out.json"))
    r.terminate(box)
    assert seen["up"] == ("/x/in.tar.gz", "/fixtimes/input.tar.gz") and seen["down"] == ("/fixtimes/out.json", "/x/out.json")
    assert seen["terminated"]
    monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", "2024.10")  # an explicit setting wins
    fixtimes.ModalRunner().build(Path("/d"), Path("/c"))
    assert seen["dockerfile"][2] == "2024.10"


def test_bash_wrapper_returns_the_exit_code_and_the_log_tail(tmp_path):
    """RUN_SH as the sandbox runs it (the log under REMOTE, here a temp dir)."""
    sh = fixtimes.RUN_SH.replace(fixtimes.REMOTE, str(tmp_path))
    r = subprocess.run(["bash", "-c", sh, "bash", sys.executable, "-c", "import sys; print('x' * 5000); sys.exit(3)"],
                       capture_output=True, text=True)
    assert r.returncode == 3 and r.stdout == ("x" * 5000 + "\n")[-4000:] and (tmp_path / "replay.log").stat().st_size == 5001


def test_bash_wrapper_caps_the_graders_run_timeouts(tmp_path):
    """The grader reads its per-run timeouts from the environment: the wrapper caps them far below
    its defaults (3600 s visible, 900 s per bug), so a state that hangs cannot use up the exec budget."""
    sh = fixtimes.RUN_SH.replace(fixtimes.REMOTE, str(tmp_path))
    code = "import os; print(os.environ['SWESWEEP_VISIBLE_SUITE_TIMEOUT'], os.environ['SWESWEEP_TEST_TIMEOUT'])"
    r = subprocess.run(["bash", "-c", sh, "bash", sys.executable, "-c", code], capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if not k.startswith("SWESWEEP_")})
    assert r.returncode == 0 and r.stdout == "120 60\n"
    assert fixtimes.RUN_TIMEOUTS == {"SWESWEEP_VISIBLE_SUITE_TIMEOUT": "120", "SWESWEEP_TEST_TIMEOUT": "60"}
    worst = int(fixtimes.RUN_TIMEOUTS["SWESWEEP_VISIBLE_SUITE_TIMEOUT"]) + 13 * int(fixtimes.RUN_TIMEOUTS["SWESWEEP_TEST_TIMEOUT"])
    assert worst * 2 < fixtimes.EXEC_TIMEOUT_SEC / 4  # a hung state and its rerun, 13 bugs: well within budget


def test_bash_wrapper_tail_cut_through_a_character_is_read_leniently(tmp_path):
    """``tail -c 4000`` cuts by bytes: the cut can fall inside a multi-byte character. The tail is
    read as bytes and decoded with replacement (modal's text reader is strict UTF-8 and would raise)."""
    sh = fixtimes.RUN_SH.replace(fixtimes.REMOTE, str(tmp_path))
    code = "import sys; sys.stdout.buffer.write('林'.encode() + b'x' * 3998)"  # the cut: 2 of 3 bytes of 林
    r = subprocess.run(["bash", "-c", sh, "bash", sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0 and r.stdout[:2] == "林".encode()[1:]
    with pytest.raises(UnicodeDecodeError):
        r.stdout.decode("utf-8")

    class Box:
        def exec(self, *args, **kw):
            assert kw["text"] is False
            return SimpleNamespace(stdout=SimpleNamespace(read=lambda: r.stdout), wait=lambda: 0)

    runner = fixtimes.ModalRunner.__new__(fixtimes.ModalRunner)  # no modal needed for exec
    rc, tail = runner.exec(Box(), ["true"], "/tests", 10)
    assert rc == 0 and tail == "\ufffd\ufffd" + "x" * 3998
