"""Workspace helpers on LocalShellEnvironment: probes, change detection,
manifests, copy/diff/apply round trips (with spaces in names), git stats,
and the never-raise failure convention."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import subprocess
import threading
import time

import pytest

from subagents_bench import workspace
from subagents_bench.environment import ExecResult, LocalShellEnvironment
from subagents_bench.types import AgentStopped
from subagents_bench.workspace import (
    apply_copy,
    container_arch,
    copy_tree,
    diff_summary,
    dir_size_mb,
    files_modified_since,
    git_diff_stat,
    has_timeout_binary,
    make_marker,
    manifest,
    resolve_workdir,
    sb_root,
)


@pytest.fixture
def env(tmp_path):
    wd = tmp_path / "work"
    wd.mkdir()
    e = LocalShellEnvironment(str(wd))
    yield e
    e.close()


def build_tree(root):
    """A small tree with spaces, nesting, an executable and a symlink."""
    (root / "src" / "sub dir").mkdir(parents=True)
    (root / "src" / "a.py").write_text("print('a')\n")
    (root / "src" / "sub dir" / "file one.txt").write_text("one\n")
    (root / "src" / "gone dir").mkdir()
    (root / "src" / "gone dir" / "x.txt").write_text("x\n")
    (root / "run.sh").write_text("#!/bin/sh\necho run\n")
    (root / "run.sh").chmod(0o755)
    (root / "plain.sh").write_text("#!/bin/sh\n")
    (root / "keep.txt").write_text("unchanged\n")
    (root / "old.txt").write_text("old line\n")
    os.symlink("keep.txt", root / "link")


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def test_resolve_workdir_and_arch(env, tmp_path):
    assert resolve_workdir(env) == str(tmp_path / "work")
    assert container_arch(env) == platform.machine()


def test_has_timeout_binary_matches_path_and_caches(env):
    expected = shutil.which("timeout") is not None
    assert has_timeout_binary(env) is expected
    calls = []

    def runner(cmd, t):
        calls.append(cmd)
        return ExecResult("__SB_YES\n", "", 0)

    assert has_timeout_binary(env, runner=runner) is expected  # cached: runner unused
    assert calls == []
    assert has_timeout_binary(env, refresh=True, runner=runner) is True
    workspace.set_timeout_binary(env, None)
    assert has_timeout_binary(env, runner=lambda c, t: ExecResult("", "", 1)) is False  # failed probe
    assert has_timeout_binary(env, runner=runner) is True  # ...was not cached


def test_sb_root(env):
    assert sb_root(env) == env.state_root

    class Bare:
        pass

    assert sb_root(Bare()) == "/tmp/.sb"


# ---------------------------------------------------------------------------
# change detection
# ---------------------------------------------------------------------------


def _age(root, seconds=100):
    past = time.time() - seconds
    for dirpath, dirnames, filenames in os.walk(root):
        for n in filenames:
            p = os.path.join(dirpath, n)
            if not os.path.islink(p):
                os.utime(p, (past, past))


def test_make_marker_and_files_modified_since(tmp_path):
    wd = tmp_path / "work"
    wd.mkdir()
    # state root INSIDE the scanned tree, to check it is excluded
    e = LocalShellEnvironment(str(wd), state_root=str(wd / ".sb"))
    try:
        build_tree(wd)
        _age(wd)
        marker = make_marker(e, "group/1 a")
        assert marker == f"{wd}/.sb/markers/group_1_a"
        assert os.path.exists(marker)
        (wd / "src" / "a.py").write_text("print('changed')\n")
        (wd / "new file.txt").write_text("new\n")
        (wd / ".sb" / "sub-1").mkdir(parents=True)
        (wd / ".sb" / "sub-1" / "cwd").write_text("/x\n")  # harness state: excluded
        res = files_modified_since(e, marker, str(wd))
        assert res == ([f"{wd}/new file.txt", f"{wd}/src/a.py"], 2)
        # relative root resolves to absolute paths too
        assert files_modified_since(e, marker, ".") == res
        for i in range(5):
            (wd / f"n{i}").write_text("x")
        paths, total = files_modified_since(e, marker, str(wd), cap=3)
        assert total == 7 and len(paths) == 3 and paths == sorted(paths)
    finally:
        e.close()


def test_files_modified_since_failures(env, tmp_path):
    marker = make_marker(env, "m")
    assert files_modified_since(env, marker, str(tmp_path / "nope")) is None
    assert files_modified_since(env, str(tmp_path / "no-marker"), str(tmp_path)) is None


def test_dir_size_mb(env, tmp_path):
    d = tmp_path / "work" / "blob"
    d.mkdir()
    (d / "f").write_bytes(os.urandom(2 * 1024 * 1024))
    size = dir_size_mb(env, str(d))
    assert 1.9 <= size <= 2.6
    assert dir_size_mb(env, str(tmp_path / "missing")) is None


# ---------------------------------------------------------------------------
# manifest / copy / diff / apply
# ---------------------------------------------------------------------------


def test_manifest_values(env, tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    build_tree(root)
    m = manifest(env, str(root))
    assert m["src/sub dir/file one.txt"] == hashlib.md5(b"one\n").hexdigest()
    assert m["run.sh"] == hashlib.md5(b"#!/bin/sh\necho run\n").hexdigest()
    assert m["link"] == "symlink:keep.txt"
    assert len(m) == 8
    mm = manifest(env, str(root), include_mode=True)
    assert mm["run.sh"].endswith(" +x") and not mm["plain.sh"].endswith(" +x")
    assert manifest(env, str(tmp_path / "missing")) is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert manifest(env, str(empty)) == {}


def test_manifest_handles_backslash_and_newline_names(env, tmp_path):
    root = tmp_path / "weird"
    root.mkdir()
    (root / "back\\slash").write_text("b")
    (root / "new\nline").write_text("n")
    m = manifest(env, str(root))
    assert m == {"back\\slash": hashlib.md5(b"b").hexdigest(), "new\nline": hashlib.md5(b"n").hexdigest()}


def test_copy_tree(env, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    build_tree(src)
    (src / ".hidden").write_text("h")
    dst = tmp_path / "copies" / "cand 0"
    assert copy_tree(env, str(src), str(dst)) is True
    assert manifest(env, str(src), include_mode=True) == manifest(env, str(dst), include_mode=True)
    assert os.path.islink(dst / "link")
    assert copy_tree(env, str(tmp_path / "missing"), str(tmp_path / "x")) is False


def _mutate(copy):
    (copy / "src" / "a.py").write_text("print('A')\n")  # modified
    (copy / "src" / "sub dir" / "brand new.txt").write_text("new\n")  # added
    (copy / "deep" / "er").mkdir(parents=True)
    (copy / "deep" / "er" / "z.txt").write_text("z\n")  # added in new dirs
    (copy / "old.txt").unlink()  # deleted
    shutil.rmtree(copy / "src" / "gone dir")  # deleted, dir should disappear
    (copy / "plain.sh").chmod(0o755)  # mode-only change
    os.remove(copy / "link")
    os.symlink("old-target", copy / "link")  # retargeted symlink


def test_diff_summary(env, tmp_path):
    orig = tmp_path / "orig"
    orig.mkdir()
    build_tree(orig)
    copy = tmp_path / "copy"
    copy_tree(env, str(orig), str(copy))
    assert diff_summary(env, str(orig), str(copy)) == "no file changes"
    _mutate(copy)
    s = diff_summary(env, str(orig), str(copy))
    lines = s.splitlines()
    assert lines[0] == "changed files: 3 modified, 2 added, 2 deleted"
    assert "M src/a.py" in lines and "M plain.sh" in lines and "M link" in lines
    assert "A src/sub dir/brand new.txt" in lines and "A deep/er/z.txt" in lines
    assert "D old.txt" in lines and "D src/gone dir/x.txt" in lines
    assert "--- diff -ruN (excerpt) ---" in s
    assert "-print('a')" in s and "+print('A')" in s
    short = diff_summary(env, str(orig), str(copy), cap_chars=120)
    assert len(short) <= 120 and short.endswith("[... diff summary truncated]")


def test_diff_summary_without_diff_binary(tmp_path):
    orig, copy = tmp_path / "o", tmp_path / "c"
    orig.mkdir()
    copy.mkdir()
    (orig / "f").write_text("1")
    (copy / "f").write_text("2")
    # a PATH with the hashing tools but no diff
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ("md5sum", "md5", "find", "xargs", "readlink", "cat", "head"):
        p = shutil.which(tool)
        if p:
            os.symlink(p, bindir / tool)
    e = LocalShellEnvironment(str(tmp_path), env={"PATH": str(bindir)})
    try:
        s = diff_summary(e, str(orig), str(copy))
        assert s.splitlines()[:2] == ["changed files: 1 modified, 0 added, 0 deleted", "M f"]
        assert "(no diff binary in the sandbox)" in s
    finally:
        e.close()


def test_apply_copy_round_trip(env, tmp_path):
    dst = tmp_path / "dst"
    dst.mkdir()
    build_tree(dst)
    copy = tmp_path / "arb" / "cand 1"
    assert copy_tree(env, str(dst), str(copy))
    _mutate(copy)
    applied, err = apply_copy(env, str(copy), str(dst))
    assert err is None
    expected = sorted(
        f"{dst}/{p}"
        for p in [
            "src/a.py",
            "src/sub dir/brand new.txt",
            "deep/er/z.txt",
            "old.txt",
            "src/gone dir/x.txt",
            "plain.sh",
            "link",
        ]
    )
    assert applied == expected
    assert manifest(env, str(dst), include_mode=True) == manifest(env, str(copy), include_mode=True)
    assert not (dst / "src" / "gone dir").exists()  # emptied dir removed
    assert (dst / "src").is_dir()
    assert os.readlink(dst / "link") == "old-target"
    assert stat.S_IMODE((dst / "plain.sh").stat().st_mode) & 0o100
    # untouched file not rewritten
    assert (dst / "keep.txt").read_text() == "unchanged\n"
    # idempotent
    assert apply_copy(env, str(copy), str(dst)) == ([], None)
    # cap
    for i in range(5):
        (copy / f"extra{i}").write_text(str(i))
    applied, err = apply_copy(env, str(copy), str(dst), cap=2)
    assert err is None and len(applied) == 2
    assert all((dst / f"extra{i}").read_text() == str(i) for i in range(5))
    # no leftover temp files from the atomic replace
    assert not [p for p in os.listdir(dst) if ".sb_apply_" in p]


def test_apply_copy_errors(env, tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    applied, err = apply_copy(env, str(tmp_path / "missing"), str(d))
    assert applied == [] and "could not hash" in err
    applied, err = apply_copy(env, str(d), str(tmp_path / "missing"))
    assert applied == [] and "could not hash" in err
    # a file in the copy where dst has a directory cannot be applied
    c = tmp_path / "c"
    c.mkdir()
    (c / "thing").write_text("file now")
    (d / "thing").mkdir()
    (d / "thing" / "inner").write_text("x")
    applied, err = apply_copy(env, str(c), str(d))
    assert err is not None and "thing" in err


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_diff_stat(env, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    assert git_diff_stat(env, str(repo)) is None  # not a repo
    g = ["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t"]
    subprocess.run([*g, "init", "-q"], check=True)
    (repo / "a.txt").write_text("1\n")
    assert git_diff_stat(env, str(repo)) == ""  # no HEAD yet, nothing tracked
    subprocess.run([*g, "add", "a.txt"], check=True)
    subprocess.run([*g, "commit", "-qm", "init"], check=True)
    assert git_diff_stat(env, str(repo)) == ""
    (repo / "a.txt").write_text("1\n2\n")
    stat_out = git_diff_stat(env, str(repo))
    assert "a.txt | 1 +" in stat_out and "1 file changed, 1 insertion(+)" in stat_out
    assert git_diff_stat(env, str(tmp_path / "nope")) is None


# ---------------------------------------------------------------------------
# failure convention
# ---------------------------------------------------------------------------


class _Exploding:
    workdir = "/"
    stop_event = None
    state_root = "/tmp/.sb"

    def exec(self, command, timeout_sec=None, cwd=None):
        raise RuntimeError("container gone")


class _TimingOut(_Exploding):
    def exec(self, command, timeout_sec=None, cwd=None):
        return ExecResult("", "", None, 1.0, True)


@pytest.mark.parametrize("bad", [_Exploding(), _TimingOut()])
def test_helpers_never_raise(bad):
    assert resolve_workdir(bad) is None
    assert container_arch(bad) is None
    assert has_timeout_binary(bad) is False
    assert make_marker(bad, "m") is None
    assert files_modified_since(bad, "/m", "/") is None
    assert dir_size_mb(bad, "/") is None
    assert copy_tree(bad, "/a", "/b") is False
    assert manifest(bad, "/") is None
    assert diff_summary(bad, "/a", "/b").startswith("(diff summary unavailable")
    assert apply_copy(bad, "/a", "/b") == ([], "could not hash files under /b")
    assert git_diff_stat(bad, "/") is None


def test_agent_stopped_propagates(env):
    env.stop_event = threading.Event()
    env.stop_event.set()
    for fn, args in [
        (resolve_workdir, ()),
        (make_marker, ("m",)),
        (manifest, ("/",)),
        (apply_copy, ("/a", "/b")),
        (diff_summary, ("/a", "/b")),
        (files_modified_since, ("/m", "/")),
    ]:
        with pytest.raises(AgentStopped):
            fn(env, *args)
