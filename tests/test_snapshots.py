"""Workspace snapshots: the real git command on a temp repository (the patch is byte for byte the
collect hook's, the repo's .git is never written), and the thread's triggers, coalescing, errors and
stop, with a fake environment."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from fakes import make_config, records
from subagents_bench.environment import AgentStopped, ExecResult, LocalShellEnvironment
from subagents_bench.records import KINDS, EpisodeRecorder
from subagents_bench.snapshots import MAX_PATCH_BYTES, Snapshotter, parse_output, snapshot_command

COLLECT = "git add -A && git diff --cached --binary baseline"  # the SWE-sweep task's collect hook
TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's empty tree


def isolate_git(monkeypatch) -> None:
    """No user or system git config, a fixed identity: the same git for the snapshot and the hook."""
    for k, v in {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t",
                 "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(k, v)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout


def write(repo: Path, files: dict[str, bytes | None]) -> None:
    for name, data in files.items():
        p = repo / name
        if data is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)


BASE = {
    ".gitignore": b"*.log\nbuild/\n",
    "src/a.py": b"def a():\n    return 1\n",
    "src/gone.py": b"x = 1\n",
    "src/staged.py": b"s = 1\n",
    "data.bin": bytes(range(256)) * 4,
    "latin1.txt": b"caf\xe9\n",
    "run.sh": b"echo hi\n",
    "kept.log": b"tracked, though .gitignore matches it\n",
}


def init_repo(repo: Path, files: dict[str, bytes] = BASE) -> Path:
    """A repository at ``baseline`` (a tag, as in the SWE-sweep images), with a tracked ignored file."""
    repo.mkdir(parents=True, exist_ok=True)
    write(repo, files)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    if "kept.log" in files:
        git(repo, "add", "-f", "kept.log")
    git(repo, "commit", "-qm", "base")
    git(repo, "tag", "baseline")
    return repo


def tree_state(d: Path) -> dict[str, str]:
    """Every file under ``d`` -> the sha256 of its bytes (symlinks by target)."""
    out = {}
    for p in sorted(d.rglob("*")):
        if p.is_symlink():
            out[str(p.relative_to(d))] = "-> " + os.readlink(p)
        elif p.is_file():
            out[str(p.relative_to(d))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def collect_on_copy(repo: Path, dest: Path) -> tuple[bytes, str]:
    """The collect hook on a copy of ``repo`` -> (patch, its tree)."""
    shutil.copytree(repo, dest, symlinks=True)
    patch = subprocess.run(["bash", "-c", COLLECT], cwd=dest, check=True, capture_output=True).stdout
    return patch, git(dest, "write-tree").decode().strip()


def recorder(tmp_path: Path) -> EpisodeRecorder:
    return EpisodeRecorder(tmp_path / "ep", make_config(interactive={"snapshots": True}), task_name="task-1")


def wait_for(cond, timeout: float = 20.0) -> None:
    t_end = time.time() + timeout
    while not cond():
        assert time.time() < t_end, "timed out"
        time.sleep(0.02)


def snaps(rec: EpisodeRecorder) -> list[dict]:
    return list(rec.records["snapshots"])


# --- the git command, for real ------------------------------------------------------------------


def test_snapshot_is_the_collect_hooks_patch_and_leaves_git_alone(tmp_path, monkeypatch):
    """Modified, deleted, new untracked, binary, staged, non-UTF-8, mode-changed, symlinked and
    tracked-but-ignored files: the patch equals the hook's on a copy byte for byte, ignored files
    stay out, and the repo's .git (index, objects, refs) is byte for byte the same afterwards."""
    isolate_git(monkeypatch)
    repo = init_repo(tmp_path / "repo")
    env = LocalShellEnvironment(str(repo / "src"))  # a subdirectory: the snapshot covers the whole repo
    rec = recorder(tmp_path)
    snap = Snapshotter(env, rec, env.workdir, every_sec=60)
    try:
        snap.start()
        wait_for(lambda: len(snaps(rec)) == 1)
        s0 = snaps(rec)[0]
        assert (s0["seq"], s0["trigger"], s0["changed"], s0["error"], s0["bytes"]) == (0, "start", True, None, 0)
        assert s0["tree"] == git(repo, "rev-parse", "baseline^{tree}").decode().strip()
        assert s0["patch"] == "snapshots/0000.patch" and (rec.dir / s0["patch"]).read_bytes() == b""
        assert s0["t_request"] <= s0["t_start"] <= s0["t_end"]

        write(repo, {"src/a.py": b"def a():\n    return 2\n", "src/gone.py": None, "src/new.py": b"n = 1\n",
                     "data.bin": bytes(range(256))[::-1] * 4 + b"\0", "latin1.txt": b"caf\xe9 cr\xe8me\n",
                     "kept.log": b"changed\n", "x.log": b"ignored\n", "build/out.o": b"\0\1ignored"})
        (repo / "run.sh").chmod(0o755)
        (repo / "link").symlink_to("src/a.py")
        write(repo, {"src/staged.py": b"s = 2\n"})
        git(repo, "add", "src/staged.py")  # the agent staged this one in the repo's own index
        write(repo, {"src/staged.py": b"s = 3\n"})  # and changed it again
        before = tree_state(repo / ".git")
        snap.on_tool({"name": "bash"})
        wait_for(lambda: len(snaps(rec)) == 2)
        assert tree_state(repo / ".git") == before
    finally:
        snap.stop()
        env.close()
    s1 = snaps(rec)[1]
    patch, tree = collect_on_copy(repo, tmp_path / "copy")
    assert (s1["seq"], s1["trigger"], s1["changed"], s1["error"], s1["tree"]) == (1, "tool", True, None, tree)
    assert (rec.dir / "snapshots" / "0001.patch").read_bytes() == patch and s1["bytes"] == len(patch)
    for name in ("src/a.py", "src/gone.py", "src/new.py", "data.bin", "latin1.txt", "kept.log", "run.sh", "link",
                 "src/staged.py"):
        assert f"diff --git a/{name} b/{name}".encode() in patch, name
    assert b"GIT binary patch" in patch and b"caf\xe9 cr\xe8me" in patch and b"s = 3" in patch
    assert b"x.log" not in patch and b"build/out.o" not in patch and b"deleted file mode" in patch
    assert b"new mode 100755" in patch
    assert sorted(p.name for p in (rec.dir / "snapshots").iterdir()) == ["0000.patch", "0001.patch"]
    assert records(rec.dir)["snapshots"] == snaps(rec)  # the JSONL has them too
    assert "snapshots" in KINDS


def test_a_return_to_an_earlier_tree_is_a_change_and_the_same_tree_is_not(tmp_path, monkeypatch):
    isolate_git(monkeypatch)
    repo = init_repo(tmp_path / "repo", {"a.py": b"1\n"})
    env = LocalShellEnvironment(str(repo))
    rec = recorder(tmp_path)
    snap = Snapshotter(env, rec, env.workdir, every_sec=60)
    try:
        snap.start()
        wait_for(lambda: len(snaps(rec)) == 1)
        for k, content in enumerate([b"2\n", b"1\n", b"1\n"], start=1):  # a change, back to baseline, nothing new
            write(repo, {"a.py": content})
            snap.request()
            wait_for(lambda: len(snaps(rec)) == k + 1)
    finally:
        snap.stop()
        env.close()
    s = snaps(rec)
    assert [(x["seq"], x["changed"], x["error"]) for x in s] == [(0, True, None), (1, True, None), (2, True, None),
                                                                  (3, False, None)]
    assert s[2]["tree"] == s[0]["tree"] != s[1]["tree"] and s[3]["tree"] == s[0]["tree"]
    assert [x["patch"] for x in s] == ["snapshots/0000.patch", "snapshots/0001.patch", "snapshots/0002.patch", None]
    assert (rec.dir / s[2]["patch"]).read_bytes() == b"" and b"+2" in (rec.dir / s[1]["patch"]).read_bytes()
    assert s[3]["bytes"] is None  # the same tree: no diff computed
    assert not (rec.dir / "snapshots" / "0003.patch").exists()


def test_a_same_size_rewrite_in_the_index_second_is_seen(tmp_path, monkeypatch):
    """git's racy-entry check: a file rewritten at the same size within the mtime second of the
    index's last write has the stat recorded in the index, and git compares its content only
    because the entry is not older than the index file. The private copy keeps the index's mtime,
    so the snapshot sees the change, like the hook. (core.checkStat minimal makes the case
    deterministic: only the mtime second and the size are compared.)"""
    isolate_git(monkeypatch)
    repo = init_repo(tmp_path / "repo", {"a.py": b"a = 0\n"})
    git(repo, "config", "core.checkStat", "minimal")
    m = int(time.time()) - 100
    os.utime(repo / "a.py", (m, m))
    git(repo, "update-index", "--refresh")  # the index records mtime m
    os.utime(repo / ".git" / "index", (m, m))  # as if written in that second
    write(repo, {"a.py": b"a = 1\n"})
    os.utime(repo / "a.py", (m, m))
    env = LocalShellEnvironment(str(repo))
    rec = recorder(tmp_path)
    snap = Snapshotter(env, rec, env.workdir, every_sec=60)
    try:
        snap.start()
        wait_for(lambda: len(snaps(rec)) == 1)
    finally:
        snap.stop()
        env.close()
    patch, tree = collect_on_copy(repo, tmp_path / "copy")  # copytree keeps the mtimes
    assert b"+a = 1" in patch and snaps(rec)[0]["tree"] == tree
    assert (rec.dir / "snapshots" / "0000.patch").read_bytes() == patch


def test_too_large_patch_and_git_errors(tmp_path, monkeypatch):
    """Over the cap: the error ``too large`` with its size and tree, no file, and the next snapshot is
    still compared with the last successful one. Outside a repository or without ``baseline``: an
    error from git."""
    isolate_git(monkeypatch)
    repo = init_repo(tmp_path / "repo", {"a.py": b"1\n"})
    env = LocalShellEnvironment(str(repo))
    rec = recorder(tmp_path)
    snap = Snapshotter(env, rec, env.workdir, every_sec=60, max_bytes=100)
    try:
        snap.start()
        wait_for(lambda: len(snaps(rec)) == 1)
        write(repo, {"a.py": b"x" * 500 + b"\n"})
        snap.request()
        wait_for(lambda: len(snaps(rec)) == 2)
        snap.request()
        wait_for(lambda: len(snaps(rec)) == 3)
    finally:
        snap.stop()
        env.close()
    s0, s1, s2 = snaps(rec)
    assert s0["error"] is None and s0["bytes"] == 0
    for s in (s1, s2):
        assert (s["error"], s["changed"], s["patch"]) == ("too large", True, None) and s["bytes"] > 100
        assert s["tree"] != s0["tree"]
    assert snap.tree == s0["tree"] and sorted(p.name for p in (rec.dir / "snapshots").iterdir()) == ["0000.patch"]

    plain = tmp_path / "plain"
    plain.mkdir()
    no_tag = tmp_path / "no_tag"
    no_tag.mkdir()
    git(no_tag, "init", "-q")
    write(no_tag, {"a": b"1\n"})
    git(no_tag, "add", "-A")
    git(no_tag, "commit", "-qm", "x")
    for d, want in ((plain, "not a git repository"), (no_tag, "baseline")):
        env = LocalShellEnvironment(str(d))
        rec = EpisodeRecorder(tmp_path / f"ep-{d.name}", make_config(), task_name="t")
        snap = Snapshotter(env, rec, env.workdir, every_sec=60)
        try:
            snap.start()
            wait_for(lambda: len(snaps(rec)) == 1)
        finally:
            snap.stop()
            env.close()
        (s,) = snaps(rec)
        assert s["error"].startswith("exit 128: fatal: ") and want in s["error"], s
        assert (s["tree"], s["changed"], s["patch"]) == (None, False, None) and snap.tree is None


def test_snapshot_command_and_output():
    cmd = snapshot_command("/my repo", "/tmp/.sb", None)
    assert "git -C '/my repo' rev-parse --show-toplevel" in cmd and "S=/tmp/.sb/snap;" in cmd
    assert '[ "$T" = none ]' in cmd and f'[ "$N" -le {MAX_PATCH_BYTES} ]' in cmd
    assert "git diff --cached --binary baseline" in cmd and 'GIT_INDEX_FILE="$X"' in cmd
    assert f'[ "$T" = {TREE} ]' in snapshot_command("/w", "/s", TREE, 10)
    assert '[ "$N" -le 10 ]' in snapshot_command("/w", "/s", None, 10)
    assert parse_output(f"noise\n__SB_TREE {TREE} same\n__SB_END\n") == (TREE, None, None, None)
    assert parse_output(f"__SB_TREE {TREE} 3\r\nYWL/\r\n__SB_END\r\n") == (TREE, 3, b"ab\xff", None)
    assert parse_output(f"__SB_TREE {TREE} 0\n__SB_END\n") == (TREE, 0, b"", None)
    assert parse_output(f"__SB_TREE {TREE} 30\n__SB_END\n", max_bytes=10) == (TREE, 30, None, "too large")
    for bad, msg in ((f"__SB_TREE {TREE} 4\nYWL/\n__SB_END\n", "patch of 3 bytes, expected 4"),
                     (f"__SB_TREE {TREE} 3\nYW*/\n__SB_END\n", "unreadable patch"),
                     (f"__SB_TREE {TREE} 3\nYWL/\n", "no snapshot"), ("__SB_TREE xyz same\n__SB_END\n", "no snapshot")):
        with pytest.raises(ValueError, match=msg):
            parse_output(bad)


# --- the thread, with a fake environment ---------------------------------------------------------


class FakeEnv:
    """``exec`` answers a snapshot of tree ``tree`` (a changed one gets a 1-byte patch); ``gate``
    blocks it until set; ``fail`` (a list) gives exceptions or ExecResults to return first."""

    def __init__(self, gate: bool = True):
        self.state_root = "/tmp/.sb"
        self.stop_event = threading.Event()
        self.gate = threading.Event()
        if gate:
            self.gate.set()
        self.calls: list[float] = []
        self.tree = TREE
        self.fail: list = []
        self.lock = threading.Lock()

    def exec(self, command: str, timeout: float) -> ExecResult:
        with self.lock:
            self.calls.append(time.time())
        while not self.gate.wait(0.05):
            if self.stop_event.is_set():
                raise AgentStopped("stopped during exec")
        if self.fail:
            x = self.fail.pop(0)
            if isinstance(x, BaseException):
                raise x
            return x
        if f'[ "$T" = {self.tree} ]' in command:
            return ExecResult(f"__SB_TREE {self.tree} same\n__SB_END\n", "", 0)
        return ExecResult(f"__SB_TREE {self.tree} 1\nCg==\n__SB_END\n", "", 0)


def started(tmp_path, env, every_sec=60.0) -> tuple[Snapshotter, EpisodeRecorder]:
    rec = recorder(tmp_path)
    snap = Snapshotter(env, rec, "/w", every_sec)
    snap.start()
    return snap, rec


def test_requests_during_a_snapshot_are_coalesced_into_one_more(tmp_path):
    env = FakeEnv(gate=False)
    snap, rec = started(tmp_path, env)
    try:
        wait_for(lambda: len(env.calls) == 1)  # snapshot 0 runs (blocked)
        t0 = time.time()
        for _ in range(5):
            snap.on_tool({"name": "edit_file"})
        snap.on_tool({"name": "read_file"})  # not a trigger
        assert time.time() - t0 < 0.1  # the agent never waits on a snapshot
        env.gate.set()
        wait_for(lambda: len(snaps(rec)) == 2)
        time.sleep(0.3)
        assert len(env.calls) == 2  # exactly one more
        s0, s1 = snaps(rec)
        assert (s0["trigger"], s1["trigger"]) == ("start", "tool") and t0 <= s1["t_request"] <= t0 + 0.1
        assert s1["t_request"] <= s1["t_start"] and (s0["changed"], s1["changed"]) == (True, False)
        snap.on_tool({"name": "bash"})
        wait_for(lambda: len(snaps(rec)) == 3)
        assert [s["seq"] for s in snaps(rec)] == [0, 1, 2]
    finally:
        snap.stop()
    snap.on_tool({"name": "bash"})  # after stop: nothing
    time.sleep(0.2)
    assert len(env.calls) == 3 and not snap.thread.is_alive()


def test_heartbeats_follow_the_previous_snapshot(tmp_path):
    env = FakeEnv()
    snap, rec = started(tmp_path, env, every_sec=0.15)
    try:
        wait_for(lambda: len(snaps(rec)) >= 4)
    finally:
        snap.stop()
    s = snaps(rec)
    assert s[0]["trigger"] == "start" and {x["trigger"] for x in s[1:]} == {"heartbeat"}
    for prev, x in zip(s, s[1:]):
        assert x["t_start"] - prev["t_end"] >= 0.14 and x["t_request"] == pytest.approx(prev["t_end"] + 0.15, abs=1e-6)
    assert [x["changed"] for x in s] == [True] + [False] * (len(s) - 1)


def test_errors_are_recorded_and_the_next_trigger_tries_again(tmp_path):
    env = FakeEnv()
    env.fail = [RuntimeError("sandbox gone"), ExecResult("", "fatal: oops", 128), ExecResult("", "", None, True),
                ExecResult("garbage", "", 0)]
    snap, rec = started(tmp_path, env)
    try:
        for k in range(1, 5):
            wait_for(lambda: len(snaps(rec)) == k)
            snap.request()
        wait_for(lambda: len(snaps(rec)) == 5)
    finally:
        snap.stop()
    s = snaps(rec)
    assert [x["error"] for x in s] == ["RuntimeError: sandbox gone", "exit 128: fatal: oops", "timed out after 120 s",
                                       "no snapshot in the output", None]
    assert all(x["tree"] is None and not x["changed"] and x["patch"] is None for x in s[:4])
    assert (s[4]["tree"], s[4]["changed"], s[4]["patch"]) == (TREE, True, "snapshots/0004.patch")  # the first success
    assert (rec.dir / "snapshots" / "0004.patch").read_bytes() == b"\n"


def test_the_episode_stop_ends_the_thread(tmp_path):
    """AgentStopped (the deadline) during a snapshot: recorded as ``stopped``; the thread ends and
    later triggers start nothing."""
    env = FakeEnv(gate=False)
    snap, rec = started(tmp_path, env)
    wait_for(lambda: len(env.calls) == 1)
    env.stop_event.set()
    wait_for(lambda: not snap.thread.is_alive())
    snap.on_tool({"name": "bash"})
    snap.stop()
    assert [(x["error"], x["changed"]) for x in snaps(rec)] == [("stopped", False)] and len(env.calls) == 1


def test_stop_waits_with_a_timeout_and_nothing_is_written_after_it(tmp_path):
    env = FakeEnv(gate=False)
    snap, rec = started(tmp_path, env)
    wait_for(lambda: len(env.calls) == 1)
    t0 = time.time()
    snap.stop(timeout=0.2)
    assert 0.15 < time.time() - t0 < 2 and snap.thread.is_alive()  # still in its exec
    env.gate.set()
    wait_for(lambda: not snap.thread.is_alive())
    assert snaps(rec) == [] and not (rec.dir / "snapshots").exists()
    # stopped before it ever started, or twice: no harm
    idle = Snapshotter(FakeEnv(), recorder(tmp_path / "b"), "/w", 60)
    idle.stop()
    idle.stop()


def test_a_snapshot_in_flight_at_stop_is_recorded(tmp_path):
    env = FakeEnv(gate=False)
    snap, rec = started(tmp_path, env)
    wait_for(lambda: len(env.calls) == 1)
    threading.Timer(0.2, env.gate.set).start()
    snap.stop()
    assert [x["seq"] for x in snaps(rec)] == [0] and not snap.thread.is_alive()
