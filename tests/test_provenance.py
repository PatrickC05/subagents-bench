"""Source fingerprint and git state (subagents_bench.provenance), with and
without a git work tree. Trees are built in tmp dirs; the git cases make their
own throwaway repositories and are skipped when git is not installed."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from subagents_bench import provenance as prov

HAS_GIT = shutil.which("git") is not None
needs_git = pytest.mark.skipif(not HAS_GIT, reason="git is not installed")


def make_tree(root: Path) -> Path:
    files = {
        "subagents_bench/__init__.py": "__version__ = '0'\n",
        "subagents_bench/orchestrator.py": "STEPS = 5\n",
        "subagents_bench/analysis/tables.py": "X = 1\n",
        "configs/base.yaml": "name: base\n",
        "scripts/run.sh": "echo run\n",
        "pyproject.toml": "[project]\nname = 'x'\n",
        "uv.lock": "version = 1\n",
        "README.md": "not source\n",
        "tests/test_x.py": "def test(): pass\n",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def git(root: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    res = subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, env=env, check=True,
    )
    return res.stdout


def test_fingerprint_without_git(tmp_path):
    root = make_tree(tmp_path / "copy")
    fp = prov.source_fingerprint(root)
    assert isinstance(fp, str) and len(fp) == 64 and fp == prov.source_fingerprint(root)
    assert prov.source_files(root) == [
        "configs/base.yaml", "pyproject.toml", "scripts/run.sh", "subagents_bench/__init__.py",
        "subagents_bench/analysis/tables.py", "subagents_bench/orchestrator.py", "uv.lock",
    ]
    # Interpreter caches and files outside the source paths do not count.
    (root / "subagents_bench/__pycache__").mkdir()
    (root / "subagents_bench/__pycache__/orchestrator.cpython-313.pyc").write_bytes(b"\0")
    (root / "subagents_bench/stale.pyc").write_bytes(b"\0")
    (root / "notes.md").write_text("scratch")
    (root / "tests/test_x.py").write_text("changed")
    assert prov.source_fingerprint(root) == fp
    # Any source change does: an edit, a new file, a config.
    (root / "subagents_bench/orchestrator.py").write_text("STEPS = 6\n")
    fp2 = prov.source_fingerprint(root)
    assert fp2 != fp
    (root / "subagents_bench/new_module.py").write_text("")
    assert prov.source_fingerprint(root) not in (fp, fp2)
    # Outside a git work tree there is no commit, but the fingerprint still identifies the code.
    assert prov.git_info(root) == {"git_commit": None, "git_dirty": None, "untracked_source_files": None}


def test_fingerprint_of_missing_source_is_none(tmp_path):
    assert prov.source_fingerprint(tmp_path) is None


@needs_git
def test_git_info_and_fingerprint_in_a_work_tree(tmp_path):
    root = make_tree(tmp_path / "repo")
    (root / ".gitignore").write_text("*.log\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    head = git(root, "rev-parse", "HEAD").strip()
    clean = prov.source_fingerprint(root)
    # A clean checkout and an export of it (no .git) fingerprint the same.
    export = tmp_path / "export"
    shutil.copytree(root, export, ignore=shutil.ignore_patterns(".git"))
    assert prov.source_fingerprint(export) == clean
    assert prov.git_info(root) == {"git_commit": head, "git_dirty": False, "untracked_source_files": []}

    # Stray files outside the source paths, and ignored files, change nothing.
    (root / "notes.md").write_text("thoughts")
    (root / "scripts/run.log").write_text("log")
    assert prov.git_info(root)["git_dirty"] is False and prov.source_fingerprint(root) == clean

    # An untracked source file: not dirty (git cannot describe it), but listed and fingerprinted.
    (root / "subagents_bench/extra.py").write_text("Y = 2\n")
    info = prov.git_info(root)
    assert info["git_dirty"] is False and info["untracked_source_files"] == ["subagents_bench/extra.py"]
    with_untracked = prov.source_fingerprint(root)
    assert with_untracked != clean

    # An uncommitted edit: dirty, and the same commit now names different code.
    (root / "configs/base.yaml").write_text("name: edited\n")
    info = prov.git_info(root)
    assert info["git_commit"] == head and info["git_dirty"] is True
    assert prov.source_fingerprint(root) not in (clean, with_untracked)

    # A deleted tracked file drops out of the fingerprint (ls-files -c still lists it).
    (root / "uv.lock").unlink()
    assert "uv.lock" not in prov.source_files(root)


@needs_git
def test_git_listing_falls_back_when_the_package_is_not_in_the_repo(tmp_path):
    # e.g. an install inside an ignored directory of some other repository.
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / ".gitignore").write_text("venv/\n")
    git(outer, "init", "-q")
    root = make_tree(outer / "venv" / "site")
    assert prov._git_listed(root) is None  # git lists nothing under an ignored dir
    assert prov.source_files(root) == sorted(prov._walk(root)) and len(prov.source_files(root)) == 7
    assert prov.source_fingerprint(root) == prov.source_fingerprint(make_tree(tmp_path / "plain"))


def test_repo_root_holds_the_package():
    assert (prov.repo_root() / "subagents_bench" / "provenance.py").is_file()


def test_runtime_paths_and_runtime_fingerprint():
    assert prov.affects_runtime("subagents_bench/orchestrator.py")
    assert prov.affects_runtime("subagents_bench/harness/harbor_agent.py")
    assert prov.affects_runtime("pyproject.toml") and prov.affects_runtime("uv.lock")
    for rel in ("subagents_bench/analysis/tables.py", "subagents_bench/plots/figures.py", "subagents_bench/sweep.py",
                "subagents_bench/cli.py", "configs/sweeps/pilot.yaml", "configs/pricing.yaml", "scripts/analyze.sh"):
        assert not prov.affects_runtime(rel), rel
    files = {"subagents_bench/orchestrator.py": "a" * 64, "subagents_bench/analysis/tables.py": "b" * 64}
    fp = prov.runtime_fingerprint(files)
    assert prov.runtime_fingerprint({**files, "subagents_bench/analysis/tables.py": "c" * 64}) == fp
    assert prov.runtime_fingerprint({**files, "configs/new.yaml": "d" * 64}) == fp
    assert prov.runtime_fingerprint({**files, "subagents_bench/orchestrator.py": "e" * 64}) != fp


def test_what_a_harbor_job_imports_is_runtime_source():
    """The harbor agent's import closure stays outside NON_RUNTIME_PATHS, so
    an edit there cannot change an episode."""
    code = (
        "import sys, subagents_bench.harness.harbor_agent\n"
        "for name, m in sorted(sys.modules.items()):\n"
        "    if name.startswith('subagents_bench') and getattr(m, '__file__', None):\n"
        "        print(m.__file__)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=prov.repo_root())
    root = prov.repo_root()
    loaded = [Path(f).resolve().relative_to(root).as_posix() for f in res.stdout.split()]
    assert {"subagents_bench/orchestrator.py", "subagents_bench/arbiter.py", "subagents_bench/provenance.py"} <= set(loaded)
    assert [f for f in loaded if not prov.affects_runtime(f)] == []


@needs_git
def test_a_copy_inside_an_unrelated_repository_has_no_commit(tmp_path):
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "other.txt").write_text("someone else's project\n")
    git(outer, "init", "-q")
    git(outer, "add", "-A")
    git(outer, "commit", "-q", "-m", "init")
    root = make_tree(outer / "unpacked")  # not ignored, not tracked
    assert prov.git_info(root) == {"git_commit": None, "git_dirty": None, "untracked_source_files": None}
    assert prov.source_fingerprint(root) == prov.source_fingerprint(make_tree(tmp_path / "plain"))
