"""Workspace helpers: probing and copying the sandbox filesystem from the host.

Every helper runs plain shell commands through ``env.exec`` using only
coreutils/findutils (plus ``diff``/``git`` where named, with fallbacks), so it
works in any Terminal-Bench image and on the macOS host used by the unit
tests. Paths are always shell-quoted, and file lists travel NUL-separated, so
names with spaces (or newlines) are safe.

Failure convention: a helper that cannot do its job returns ``None`` (or
``False``/an explanatory string where the signature says so) instead of
raising -- these feed measurements, and a failed measurement must never kill
an episode. The one exception is :class:`~subagents_bench.types.AgentStopped`,
which always propagates so worker threads unwind.

Path convention: :func:`files_modified_since` returns **absolute** paths
(the root is resolved with ``pwd`` first); :func:`apply_copy` reports applied
files as absolute paths under the destination; :func:`manifest` keys are
paths **relative** to its root (they are compared across two roots).

Everything the harness itself writes lives under :func:`sb_root` (default
``/tmp/.sb``), which file-change scans exclude.
"""

from __future__ import annotations

import logging
import posixpath
import re
import shlex
import threading
import uuid
import weakref
from typing import Any, Callable, Iterable

from .environment import DEFAULT_STATE_ROOT, ExecResult, check_stopped, plan_b64_upload
from .types import AgentStopped

log = logging.getLogger(__name__)

# Host-side timeouts for helper commands (they are not model-controlled).
PROBE_TIMEOUT_SEC = 60
SCAN_TIMEOUT_SEC = 300
COPY_TIMEOUT_SEC = 900

# Max bytes of NUL-separated paths fetched by files_modified_since.
_MAX_LIST_BYTES = 2_000_000

Runner = Callable[[str, float], ExecResult]


def sb_root(env: Any) -> str:
    """Root of harness-owned state in this sandbox (``/tmp/.sb`` normally)."""
    return getattr(env, "state_root", None) or DEFAULT_STATE_ROOT


def _run(env: Any, command: str, timeout: float, runner: Runner | None = None) -> ExecResult | None:
    """exec that never raises (except AgentStopped); None on error/timeout."""
    check_stopped(env)
    try:
        res = runner(command, timeout) if runner else env.exec(command, timeout_sec=timeout)
    except AgentStopped:
        raise
    except Exception as e:  # noqa: BLE001 - measurement helpers must not raise
        log.debug("workspace exec failed: %s: %s", type(e).__name__, e)
        return None
    if res.timed_out:
        log.debug("workspace exec timed out after %ss", timeout)
        return None
    return res


def _ok(res: ExecResult | None) -> bool:
    return res is not None and res.return_code == 0


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def resolve_workdir(env: Any) -> str | None:
    """``pwd`` with the backend's default cwd (harbor: the task's WORKDIR)."""
    res = _run(env, "pwd", PROBE_TIMEOUT_SEC)
    if not _ok(res):
        return None
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    path = lines[-1].strip() if lines else ""
    return path if path.startswith("/") else None


def container_arch(env: Any) -> str | None:
    """``uname -m`` inside the sandbox (x86_64 on an arm64 host = emulated)."""
    res = _run(env, "uname -m", PROBE_TIMEOUT_SEC)
    if not _ok(res):
        return None
    arch = res.stdout.strip()
    return arch or None


# Per-environment cache of the timeout-binary probe.
_timeout_cache: "weakref.WeakKeyDictionary[Any, bool]" = weakref.WeakKeyDictionary()
_timeout_cache_by_id: dict[int, bool] = {}
_timeout_lock = threading.Lock()


def _cache_get(env: Any) -> bool | None:
    with _timeout_lock:
        try:
            return _timeout_cache.get(env)
        except TypeError:  # not weak-referenceable
            return _timeout_cache_by_id.get(id(env))


def set_timeout_binary(env: Any, value: bool | None) -> None:
    """Override (or with None, forget) the cached probe result for ``env``."""
    with _timeout_lock:
        try:
            if value is None:
                _timeout_cache.pop(env, None)
            else:
                _timeout_cache[env] = value
        except TypeError:
            if value is None:
                _timeout_cache_by_id.pop(id(env), None)
            else:
                _timeout_cache_by_id[id(env)] = value


def has_timeout_binary(env: Any, *, refresh: bool = False, runner: Runner | None = None) -> bool:
    """Whether a coreutils-compatible ``timeout`` (``-k`` supported) exists.

    Probed once per environment and cached. A failed probe returns False
    without caching. ``runner(command, timeout)`` lets the tools layer count
    the probe as one of its exec round-trips.
    """
    if not refresh:
        cached = _cache_get(env)
        if cached is not None:
            return cached
    res = _run(
        env,
        "if command -v timeout >/dev/null 2>&1 && timeout -k 1 5 true >/dev/null 2>&1; "
        "then echo __SB_YES; else echo __SB_NO; fi",
        PROBE_TIMEOUT_SEC,
        runner,
    )
    if res is None or "__SB_" not in res.stdout:
        return False
    value = "__SB_YES" in res.stdout
    set_timeout_binary(env, value)
    return value


# ---------------------------------------------------------------------------
# change detection
# ---------------------------------------------------------------------------


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "_"


def make_marker(env: Any, name: str) -> str | None:
    """Touch ``<sb_root>/markers/<name>`` and return its path."""
    mdir = f"{sb_root(env)}/markers"
    path = f"{mdir}/{_safe_name(name)}"
    res = _run(env, f"mkdir -p -- {shlex.quote(mdir)} && touch -- {shlex.quote(path)}", PROBE_TIMEOUT_SEC)
    return path if _ok(res) else None


def files_modified_since(
    env: Any, marker: str, root: str, cap: int = 200
) -> tuple[list[str], int] | None:
    """Regular files under ``root`` with mtime newer than ``marker``.

    Returns ``(sorted absolute paths, capped at cap; uncapped total)``.
    ``sb_root(env)`` (and ``/proc``, ``/sys``, ``/dev``) are pruned. None if
    ``root`` or ``marker`` is missing or the scan fails.
    """
    sb = sb_root(env)
    tmp = f"{sb}/tmp_fms_{uuid.uuid4().hex}"
    q, qm, qsb, qt = shlex.quote(root), shlex.quote(marker), shlex.quote(sb), shlex.quote(tmp)
    script = (
        f"cd -- {q} 2>/dev/null || exit 3\n"
        f"[ -e {qm} ] || exit 4\n"
        f"mkdir -p -- {qsb} || exit 5\n"
        f'find "$PWD" \\( -path {qsb} -o -path /proc -o -path /sys -o -path /dev \\) -prune '
        f"-o -type f -newer {qm} -print0 2>/dev/null > {qt}\n"
        f"__n=$(tr -cd '\\000' < {qt} | wc -c)\n"
        f'printf "__SB_N %s\\n" $((__n + 0))\n'
        f"head -c {_MAX_LIST_BYTES} {qt}\n"
        f"rm -f {qt}\n"
    )
    res = _run(env, script, SCAN_TIMEOUT_SEC)
    if not _ok(res):
        return None
    head, sep, body = res.stdout.partition("\n")
    if not sep or not head.startswith("__SB_N "):
        return None
    try:
        total = int(head.split()[1])
    except (IndexError, ValueError):
        return None
    items = body.split("\0")
    # The last item is "" after a trailing NUL, or a partial path if the
    # listing hit _MAX_LIST_BYTES; drop it either way.
    paths = sorted(p for p in items[:-1] if p)
    return paths[:cap], total


def dir_size_mb(env: Any, path: str) -> float | None:
    """Disk usage of ``path`` in MiB (``du -sk``)."""
    res = _run(env, f"du -sk -- {shlex.quote(path)}", SCAN_TIMEOUT_SEC)
    if not _ok(res):
        return None
    try:
        return int(res.stdout.split()[0]) / 1024.0
    except (IndexError, ValueError):
        return None


def copy_tree(env: Any, src: str, dst: str) -> bool:
    """``cp -a src/. dst/`` (dst created if needed). True on success."""
    qs, qd = shlex.quote(src), shlex.quote(dst)
    res = _run(env, f"[ -d {qs} ] && mkdir -p -- {qd} && cp -a -- {qs}/. {qd}/", COPY_TIMEOUT_SEC)
    return _ok(res)


# ---------------------------------------------------------------------------
# manifests, diffs, apply
# ---------------------------------------------------------------------------


def _unescape_md5sum_name(name: str) -> str:
    # GNU md5sum escapes "\\", "\n" (and in newer coreutils "\r") in names
    # and marks such lines with a leading backslash.
    out, i = [], 0
    while i < len(name):
        c = name[i]
        if c == "\\" and i + 1 < len(name):
            nxt = name[i + 1]
            out.append({"n": "\n", "r": "\r", "\\": "\\"}.get(nxt, "\\" + nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def manifest(env: Any, root: str, *, include_mode: bool = False) -> dict[str, str] | None:
    """``relpath -> md5`` for every regular file under ``root``.

    Uses ``find -print0 | xargs -0 md5sum`` (``md5 -r`` on macOS hosts
    without md5sum). Symlinks map to ``"symlink:<target>"``. With
    ``include_mode``, executable files get a ``" +x"`` suffix so mode-only
    changes compare unequal. None if any file cannot be hashed.
    """
    nonce = uuid.uuid4().hex[:10]
    q = shlex.quote(root)
    script = (
        # pipefail: a partial listing (unreadable dir) must fail the manifest,
        # or apply_copy would treat the missing files as deleted.
        "set -o pipefail\n"
        f"cd -- {q} 2>/dev/null || exit 3\n"
        "if command -v md5sum >/dev/null 2>&1; then __h=md5sum; "
        "elif command -v md5 >/dev/null 2>&1; then __h='md5 -r'; else exit 5; fi\n"
        "__nl='\n'\n"
        f"echo __SB_FILES_{nonce}\n"
        # md5sum output is line-based; names containing a newline (not
        # escaped by BSD md5sum) are hashed one by one, NUL-delimited.
        'find . -type f ! -name "*${__nl}*" -print0 | xargs -0 -r $__h || exit 6\n'
        f"echo __SB_NLFILES_{nonce}\n"
        'find . -type f -name "*${__nl}*" -print0 | while IFS= read -r -d \'\' __f; do '
        '__d=$($__h < "$__f") || exit 6; printf \'%s\\0%s\\0\' "$__f" "${__d%% *}"; done || exit 6\n'
        f"echo __SB_LINKS_{nonce}\n"
        "find . -type l -print0 | while IFS= read -r -d '' __l; do "
        "printf '%s\\0%s\\0' \"$__l\" \"$(readlink -- \"$__l\")\"; done\n"
        f"echo __SB_EXEC_{nonce}\n"
        "find . -type f -perm -100 -print0\n"
    )
    res = _run(env, script, SCAN_TIMEOUT_SEC)
    if not _ok(res):
        return None
    out = res.stdout
    try:
        files_part = out.split(f"__SB_FILES_{nonce}\n", 1)[1]
        files_part, nl_part = files_part.split(f"__SB_NLFILES_{nonce}\n", 1)
        nl_part, links_part = nl_part.split(f"__SB_LINKS_{nonce}\n", 1)
        links_part, exec_part = links_part.split(f"__SB_EXEC_{nonce}\n", 1)
    except (IndexError, ValueError):
        return None
    result: dict[str, str] = {}
    for line in files_part.split("\n"):
        if not line:
            continue
        escaped = line.startswith("\\")
        if escaped:
            line = line[1:]
        digest, _, name = line.partition(" ")
        name = name.lstrip(" *")
        if escaped:
            name = _unescape_md5sum_name(name)
        if not name.startswith("./") or len(digest) != 32:
            continue
        result[name[2:]] = digest
    toks = nl_part.split("\0")
    for i in range(0, len(toks) - 1, 2):
        if toks[i].startswith("./") and len(toks[i + 1]) == 32:
            result[toks[i][2:]] = toks[i + 1]
    toks = links_part.split("\0")
    for i in range(0, len(toks) - 1, 2):
        if toks[i].startswith("./"):
            result[toks[i][2:]] = "symlink:" + toks[i + 1]
    if include_mode:
        for p in exec_part.split("\0"):
            if p.startswith("./") and p[2:] in result:
                result[p[2:]] += " +x"
    return result


def _changes(orig: dict[str, str], new: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    added = sorted(set(new) - set(orig))
    deleted = sorted(set(orig) - set(new))
    modified = sorted(p for p in set(orig) & set(new) if orig[p] != new[p])
    return modified, added, deleted


def changed_paths(orig: dict[str, str], new: dict[str, str]) -> list[str]:
    """Sorted relpaths modified, added or deleted between two manifests."""
    modified, added, deleted = _changes(orig, new)
    return sorted(modified + added + deleted)


def diff_summary(env: Any, orig_root: str, copy_root: str, cap_chars: int = 4000) -> str:
    """Changed files plus a ``diff -ruN`` excerpt, at most ``cap_chars``.

    Changes under ``.git/`` are counted but not listed or diffed. Falls back
    to the file list alone when there is no ``diff`` binary. Never raises
    (except AgentStopped); problems are described in the returned text.
    """
    m_o = manifest(env, orig_root, include_mode=True)
    m_c = manifest(env, copy_root, include_mode=True)
    if m_o is None or m_c is None:
        return "(diff summary unavailable: could not hash files)"
    modified, added, deleted = _changes(m_o, m_c)
    entries = [("M", p) for p in modified] + [("A", p) for p in added] + [("D", p) for p in deleted]
    if not entries:
        return "no file changes"
    is_git = [p == ".git" or p.startswith(".git/") for _, p in entries]
    git = [e for e, g in zip(entries, is_git) if g]
    visible = sorted((e for e, g in zip(entries, is_git) if not g), key=lambda e: e[1])
    lines = [f"changed files: {len(modified)} modified, {len(added)} added, {len(deleted)} deleted"]
    lines += [f"{k} {p}" for k, p in visible[:50]]
    if len(visible) > 50:
        lines.append(f"... and {len(visible) - 50} more")
    if git:
        lines.append(f"(+ {len(git)} changes under .git/)")
    text = "\n".join(lines)
    if visible:
        qo, qc = shlex.quote(orig_root), shlex.quote(copy_root)
        res = _run(
            env,
            f"command -v diff >/dev/null 2>&1 || exit 127\n"
            f"diff -ruN -x .git -- {qo} {qc} 2>&1 | head -c {max(cap_chars, 0) * 2 + 1024}",
            SCAN_TIMEOUT_SEC,
        )
        if res is None:
            text += "\n(diff failed)"
        elif res.return_code == 127:
            text += "\n(no diff binary in the sandbox)"
        elif res.stdout.strip():
            text += "\n--- diff -ruN (excerpt) ---\n" + res.stdout
    if len(text) > cap_chars:
        note = "\n[... diff summary truncated]"
        text = text[: max(cap_chars - len(note), 0)] + note
    return text


def apply_copy(
    env: Any,
    copy_root: str,
    dst_root: str,
    cap: int | None = 200,
    *,
    base: dict[str, str] | None = None,
    only: Iterable[str] | None = None,
    relative: bool = False,
) -> tuple[list[str], str | None]:
    """Make ``dst_root`` match ``copy_root`` (the arbiter applying a winner).

    Compares md5 manifests (with mode) and copies added and modified files
    with ``cp -a``, deletes files the copy no longer has, and removes
    directories left empty by those deletions. Untouched files are not
    rewritten. Returns ``(applied absolute dst paths, capped at cap (None:
    all); error or None)``; with ``relative``, paths relative to
    ``dst_root``. Empty directories created in the copy are not propagated.

    ``base`` (a ``manifest(..., include_mode=True)`` of the copy taken when it
    was created) restricts the apply to what changed in the copy since then:
    files changed in ``dst_root`` meanwhile by someone else (e.g. a
    concurrent arbitration applying its own winner) are left alone instead of
    being reverted or deleted. ``only`` (relpaths) restricts it to those
    files: with ``copy_root`` a pristine copy, that restores just them.
    """
    m_dst = manifest(env, dst_root, include_mode=True)
    m_copy = manifest(env, copy_root, include_mode=True)
    if m_dst is None:
        return [], f"could not hash files under {dst_root}"
    if m_copy is None:
        return [], f"could not hash files under {copy_root}"
    if base is None:
        writes = sorted(p for p in m_copy if m_dst.get(p) != m_copy[p])
        deletes = sorted(p for p in m_dst if p not in m_copy)
    else:
        writes = sorted(p for p in m_copy if base.get(p) != m_copy[p] and m_dst.get(p) != m_copy[p])
        deletes = sorted(p for p in base if p not in m_copy and p in m_dst)
    if only is not None:
        keep = set(only)
        writes = [p for p in writes if p in keep]
        deletes = [p for p in deletes if p in keep]
    if not writes and not deletes:
        return [], None

    sb = sb_root(env)
    xfer = f"{sb}/xfer"
    tag = uuid.uuid4().hex
    wlist, dlist = f"{xfer}/apply_{tag}_w", f"{xfer}/apply_{tag}_d"
    staged: list[str] = []
    finals: list[str] = []
    for path, items in ((wlist, writes), (dlist, deletes)):
        data = b"".join(p.encode("utf-8", errors="surrogateescape") + b"\0" for p in items)
        pre, final = plan_b64_upload(data, path, xfer, mkdirs=True)
        staged += pre
        finals.append(final)
    for cmd in staged:
        if not _ok(_run(env, cmd, SCAN_TIMEOUT_SEC)):
            return [], "failed to upload the change list"
    qc, qd, qw, ql = shlex.quote(copy_root), shlex.quote(dst_root), shlex.quote(wlist), shlex.quote(dlist)
    script = (
        f"{finals[0]} && {finals[1]} || exit 7\n"
        f"cd -- {qd} || exit 3\n"
        "printf '%s\\0' \"$PWD\"\n"
        "while IFS= read -r -d '' p; do\n"
        '  mkdir -p -- "$(dirname -- "$p")" 2>/dev/null\n'
        # Copy to a temp name, then rename over the target: never writes
        # through an existing symlink, and a failed copy leaves dst intact.
        '  if [ -L "$p" ]; then rm -f -- "$p"; fi\n'
        '  if [ -d "$p" ]; then printf "E\\0%s\\0" "$p"; continue; fi\n'
        f'  t="$p.sb_apply_{tag[:8]}"\n'
        f'  if cp -a -- {qc}/"$p" "$t" 2>/dev/null && mv -f -- "$t" "$p" 2>/dev/null; then printf "W\\0%s\\0" "$p"; '
        'else rm -f -- "$t"; printf "E\\0%s\\0" "$p"; fi\n'
        f"done < {qw}\n"
        "while IFS= read -r -d '' p; do\n"
        '  if rm -f -- "$p" 2>/dev/null; then printf "D\\0%s\\0" "$p"; else printf "E\\0%s\\0" "$p"; fi\n'
        '  d=$(dirname -- "$p")\n'
        f'  while [ "$d" != "." ] && [ ! -d {qc}/"$d" ]; do rmdir -- "$d" 2>/dev/null || break; d=$(dirname -- "$d"); done\n'
        f"done < {ql}\n"
        f"rm -f -- {qw} {ql}\n"
    )
    res = _run(env, script, COPY_TIMEOUT_SEC)
    if res is None or res.return_code != 0:
        rc = None if res is None else res.return_code
        return [], f"apply script failed (exit {rc})"
    toks = res.stdout.split("\0")
    dst_abs = toks[0]
    applied: list[str] = []
    errors: list[str] = []
    for i in range(1, len(toks) - 1, 2):
        kind, rel = toks[i], toks[i + 1]
        if kind in ("W", "D"):
            applied.append(rel if relative else posixpath.join(dst_abs, rel))
        elif kind == "E":
            errors.append(rel)
    err = None
    if errors:
        err = f"{len(errors)} file(s) failed to apply: " + ", ".join(errors[:10])
    elif len(applied) != len(writes) + len(deletes):
        err = f"applied {len(applied)} of {len(writes) + len(deletes)} changes"
    return sorted(applied)[:cap], err


def _sed_replace_expr(old: str, new: str) -> str | None:
    """A ``sed`` command replacing every literal ``old`` by ``new`` (None if
    no delimiter fits or either string spans lines)."""
    delim = next((c for c in "|#@,%!~:" if c not in old and c not in new), None)
    if delim is None or not old or "\n" in old or "\n" in new:
        return None
    pattern = re.sub(r"([\\.*\[\]^$])", r"\\\1", old)
    replacement = re.sub(r"([\\&])", r"\\\1", new)
    return f"s{delim}{pattern}{delim}{replacement}{delim}g"


def replace_in_tree(env: Any, root: str, old: str, new: str, cap: int = 200) -> list[str] | None:
    """Replace the literal string ``old`` by ``new`` in every text file and
    every symlink target under ``root`` (the arbiter mapping a candidate's
    copy path back to the real workdir path inside its copy).

    Text files are found with ``grep -rlI --null -F`` (binary files are left
    alone) and rewritten in place with ``sed`` (``cat`` back over the
    original, so inode, mode and owner are kept); symlinks are re-pointed
    with ``ln -sfn`` (first occurrence in the target). Returns the sorted relpaths changed (capped at cap), or
    None if the rewrite could not run. A file that could not be rewritten is
    skipped (logged).
    """
    expr = _sed_replace_expr(old, new)
    if expr is None:
        return None
    tag = uuid.uuid4().hex[:8]
    script = (
        f"cd -- {shlex.quote(root)} 2>/dev/null || exit 3\n"
        f"__o={shlex.quote(old)}; __n={shlex.quote(new)}\n"
        "grep -rlI --null -F -e \"$__o\" . 2>/dev/null | while IFS= read -r -d '' f; do\n"
        f'  t="$f.sb_rw_{tag}"\n'
        f'  if sed {shlex.quote(expr)} "$f" > "$t" 2>/dev/null && cat "$t" > "$f"; then printf "W\\0%s\\0" "$f"; '
        'else printf "E\\0%s\\0" "$f"; fi\n'
        '  rm -f -- "$t"\n'
        "done\n"
        "find . -type l -print0 | while IFS= read -r -d '' l; do\n"
        '  t=$(readlink -- "$l")\n'
        '  case "$t" in *"$__o"*) ;; *) continue ;; esac\n'
        # First occurrence only (prefix/suffix expansion: portable across bash
        # versions, unlike ${t//old/new}, whose replacement quoting differs).
        '  if ln -sfn -- "${t%%"$__o"*}$__n${t#*"$__o"}" "$l" 2>/dev/null; then printf "L\\0%s\\0" "$l"; '
        'else printf "E\\0%s\\0" "$l"; fi\n'
        "done\n"
        "exit 0\n"
    )
    res = _run(env, script, SCAN_TIMEOUT_SEC)
    if not _ok(res):
        return None
    toks = res.stdout.split("\0")
    changed, errors = [], []
    for i in range(0, len(toks) - 1, 2):
        kind, rel = toks[i], toks[i + 1]
        rel = rel[2:] if rel.startswith("./") else rel
        (errors if kind == "E" else changed).append(rel)
    if errors:
        log.debug("replace_in_tree: %d file(s) under %s not rewritten: %s", len(errors), root, errors[:10])
    return sorted(changed)[:cap]


def git_diff_stat(env: Any, workdir: str) -> str | None:
    """``git diff --stat HEAD`` (staged + unstaged vs HEAD) in ``workdir``.

    Returns "" for a clean repo, None if ``workdir`` is not a git work tree,
    git is missing, or the command fails. ``safe.directory=*`` avoids git's
    ownership refusal for repos created by another user in the image.
    """
    q = shlex.quote(workdir)
    g = "git -c safe.directory='*' -c core.quotepath=off"
    script = (
        f"cd -- {q} 2>/dev/null || exit 3\n"
        "command -v git >/dev/null 2>&1 || exit 4\n"
        f"{g} rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 5\n"
        f"if {g} rev-parse --verify -q HEAD >/dev/null 2>&1; then {g} diff --stat HEAD; else {g} diff --stat; fi\n"
    )
    res = _run(env, script, SCAN_TIMEOUT_SEC)
    if not _ok(res):
        return None
    return res.stdout.rstrip("\n")
