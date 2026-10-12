"""Which harness code ran: a content fingerprint of the source, plus git state.

A git commit alone does not identify the code: sweeps run from a working
tree with uncommitted edits, and each harbor job re-imports whatever is on
disk. :func:`source_fingerprint` therefore hashes the source files
themselves, so two runs with equal fingerprints ran identical code whether or
not it was committed, and a ZIP or tarball copy (no ``.git``) still gets one.
:func:`git_info` adds the commit and whether tracked source files differ from
it.

The source is :data:`SOURCE_PATHS` under the repository root: the package,
the configs, the scripts, and the project/lock files. Files are listed with
``git ls-files -co --exclude-standard`` (tracked plus untracked, minus
ignored) when the root is a git work tree, else by walking the paths; both
skip ``__pycache__`` and compiled Python, so a clean checkout and an export
of it give the same fingerprint.

Both functions take the root explicitly and never raise; the sweep launcher
and the episode recorder call them with the repository root.

Not every source file changes what an episode runs: :data:`NON_RUNTIME_PATHS`
(analysis, plots, the sweep launcher and CLI, scripts, configs) are never
imported or read by a harbor job. :func:`affects_runtime` and
:func:`runtime_fingerprint` let the sweep tell an analysis-only edit from a
change to the code under test.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

# Paths (relative to the repository root) whose contents define the harness.
SOURCE_PATHS = ("subagents_bench", "configs", "scripts", "pyproject.toml", "uv.lock")
# Never part of the fingerprint: caches the interpreter writes as it runs.
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
_SKIP_SUFFIXES = (".pyc", ".pyo")
_SKIP_NAMES = frozenset({".DS_Store"})
GIT_TIMEOUT_SEC = 15
# Source paths an episode never imports or reads. A harbor job imports the
# package minus these (tests/test_provenance.py checks the harbor agent's
# import closure) and reads its condition config from the jobs dir, where the
# sweep materializes it and refuses a changed one on resume, so configs/ is
# not read either. Prefixes of paths relative to the repository root.
NON_RUNTIME_PATHS = (
    "subagents_bench/analysis/",
    "subagents_bench/plots/",
    "subagents_bench/sweep.py",
    "subagents_bench/cli.py",
    "configs/",
    "scripts/",
)


def repo_root() -> Path:
    """The directory holding the ``subagents_bench`` package (the repo root
    for a checkout or an editable install)."""
    return Path(__file__).resolve().parent.parent


def _git(root: Path, *args: str) -> str | None:
    """stdout of ``git -C root ...``; None if git is missing or fails."""
    try:
        res = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=GIT_TIMEOUT_SEC
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout if res.returncode == 0 else None


def _keep(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in _SKIP_DIRS for p in parts[:-1]):
        return False
    return parts[-1] not in _SKIP_NAMES and not parts[-1].endswith(_SKIP_SUFFIXES)


def _walk(root: Path) -> list[str]:
    out: list[str] = []
    for top in SOURCE_PATHS:
        p = root / top
        if p.is_file():
            out.append(top)
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            out.extend(f"{rel_dir}/{name}" for name in filenames)
    return out


def _git_listed(root: Path) -> list[str] | None:
    """Tracked + untracked, non-ignored source files per git; None outside a
    work tree, or when git lists none of the package (e.g. an install inside
    an ignored directory of some other repository)."""
    out = _git(root, "ls-files", "-z", "-co", "--exclude-standard", "--", *SOURCE_PATHS)
    if out is None:
        return None
    files = [f for f in out.split("\0") if f]
    if not any(f.startswith(f"{SOURCE_PATHS[0]}/") for f in files):
        return None
    return files


def source_files(root: str | os.PathLike | None = None) -> list[str]:
    """Sorted relative paths of the existing source files under ``root``."""
    base = Path(root) if root is not None else repo_root()
    listed = _git_listed(base)
    files = listed if listed is not None else _walk(base)
    # ls-files -c still lists tracked files deleted from the working tree.
    return sorted({f for f in files if _keep(f) and (base / f).is_file()})


def source_fingerprint(root: str | os.PathLike | None = None) -> str | None:
    """sha256 over the sorted ``(relative path, sha256 of contents)`` of every
    source file under ``root`` (hex). None if nothing could be read."""
    base = Path(root) if root is not None else repo_root()
    try:
        files = source_files(base)
        h = hashlib.sha256()
        for rel in files:
            digest = hashlib.sha256((base / rel).read_bytes()).hexdigest()
            h.update(f"{rel}\0{digest}\n".encode())
    except OSError:
        return None
    return h.hexdigest() if files else None


def affects_runtime(rel: str) -> bool:
    """Whether source file ``rel`` (relative to the root) can change what an
    episode runs, i.e. is not under :data:`NON_RUNTIME_PATHS`."""
    return not rel.startswith(NON_RUNTIME_PATHS)


def runtime_fingerprint(files: dict[str, str]) -> str:
    """sha256 over the runtime entries (:func:`affects_runtime`) of a
    ``{relative path: sha256 of contents}`` map, such as a sweep's
    ``provenance/source_<sha16>.json`` records. Two source fingerprints with
    equal runtime fingerprints ran the same code under test."""
    h = hashlib.sha256()
    for rel in sorted(files):
        if affects_runtime(rel):
            h.update(f"{rel}\0{files[rel]}\n".encode())
    return h.hexdigest()


def git_info(root: str | os.PathLike | None = None) -> dict[str, Any]:
    """``{git_commit, git_dirty, untracked_source_files}`` for ``root``.

    ``git_commit``: HEAD, or None outside a git work tree or when the work
    tree tracks none of the package (a copy unpacked inside some unrelated
    repository, whose HEAD says nothing about this code); then the other two
    are None as well. ``git_dirty``: tracked source files differ from HEAD
    (``git status --porcelain --untracked-files=no -- SOURCE_PATHS``), so a
    stray note at the repo root does not make every run dirty.
    ``untracked_source_files``: untracked, non-ignored files under the source
    paths (they are in the fingerprint, not in the commit).
    """
    base = Path(root) if root is not None else repo_root()
    head = _git(base, "rev-parse", "HEAD")
    tracked = _git(base, "ls-files", "-z", "-c", "--", SOURCE_PATHS[0]) if head is not None else None
    if head is None or not (tracked or "").strip("\0"):
        return {"git_commit": None, "git_dirty": None, "untracked_source_files": None}
    status = _git(base, "status", "--porcelain", "--untracked-files=no", "--", *SOURCE_PATHS)
    untracked = _git(base, "ls-files", "-z", "-o", "--exclude-standard", "--", *SOURCE_PATHS)
    return {
        "git_commit": head.strip(),
        "git_dirty": None if status is None else bool(status.strip()),
        "untracked_source_files": None if untracked is None else sorted(f for f in untracked.split("\0") if f and _keep(f)),
    }


__all__ = [
    "NON_RUNTIME_PATHS",
    "SOURCE_PATHS",
    "affects_runtime",
    "git_info",
    "repo_root",
    "runtime_fingerprint",
    "source_files",
    "source_fingerprint",
]
