"""Workspace snapshots: the collect hook's diff against ``baseline``, recorded during a session.

SWE-sweep grades a trial once, at the end, on the patch its collect hook writes (``git add -A &&
git diff --cached --binary baseline``). To learn when each bug was really fixed, ``fix-times``
replays the states recorded here through the grader after the run (``interactive.snapshots``).

A :class:`Snapshotter` thread takes that diff in the agent's container:

- **When.** Snapshot 0 at the session's start (the baseline state: an empty diff); after every
  ``bash`` or ``edit_file`` tool record of any agent (a recorder tool listener); and, while nothing
  else asks for one, ``every_sec`` after the previous snapshot ended (``heartbeat``: commands still
  running and background processes change files too). One snapshot runs at a time: the requests
  made while it runs are coalesced into exactly one more, right after it.
- **How.** One ``env.exec`` of :func:`snapshot_command`. It stages the work tree into a PRIVATE
  index and a private object dir under ``<state_root>/snap`` (the repo's objects as alternates), so
  the repo's ``.git`` (index, objects, refs) is never written, and the patch is byte for byte what
  the hook would write at that moment. The private index is a fresh copy of the repo's own index
  for each snapshot, mtime included: tracked files that ``.gitignore`` matches stay tracked (an
  empty index would show them as deleted) and git trusts the same file stats as in the hook. The
  tree id tells whether the state changed; only a changed state's diff is computed and sent back,
  base64-encoded (a patch need not be UTF-8).
- **Records.** One ``snapshots`` record per attempt::

      {"seq": k, "t_request": float, "t_start": float, "t_end": float,
       "trigger": "start" | "tool" | "heartbeat", "tree": str | None, "changed": bool,
       "patch": "snapshots/0007.patch" | None, "bytes": int | None, "error": str | None}

  ``t_request`` is the earliest request the snapshot serves; times are epoch seconds. ``changed``:
  the tree differs from the previous successful snapshot's (snapshot 0 is changed, and so is a
  return to an earlier state). Only a changed state's patch is written, to ``<seq:04d>.patch``
  under ``<episode dir>/snapshots/``; ``bytes`` is its size (None when no diff was computed). A
  patch over :data:`MAX_PATCH_BYTES` is the error ``too large``, without a file. An error leaves no
  tree to compare with: the next snapshot is compared with the last one without error.
- **Never in the agent's way.** A listener only sets a flag; an error is recorded and the next
  trigger tries again; AgentStopped (the deadline) ends the thread; ``stop`` joins it with a timeout
  and nothing is written after it returns. The final state for the replay is the trial's own
  ``model.patch`` (the hook runs after the session), so there is no final snapshot.
"""

from __future__ import annotations

import base64
import binascii
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Any

from .environment import AgentStopped
from .records import EpisodeRecorder

EXEC_TIMEOUT_SEC = 120
MAX_PATCH_BYTES = 20 * 2 ** 20  # 20 MB
JOIN_SEC = 30.0  # for a snapshot still running when the session ends
TRIGGERS = ("bash", "edit_file")  # the tools after which a snapshot is taken (any agent's)
PATCH_DIR = "snapshots"

# {prev}: the last successful snapshot's tree: the same tree again skips the diff. Per-attempt index
# and patch files ($$), so an attempt killed mid-way (timeout, stop) leaves no lock behind. `cp -p`:
# the copy keeps the index's mtime, which git's racy-entry check compares file mtimes with (a file
# rewritten in the same second, at the same size, would otherwise look unchanged).
SNAPSHOT_SCRIPT = """set -e
R=$(git -C {workdir} rev-parse --show-toplevel); S={snap}; mkdir -p "$S/objects"
X="$S/index.$$"; P="$S/patch.$$"; trap 'rm -f "$X" "$X.lock" "$P"' EXIT
cd "$R"; O=$(cd "$(git rev-parse --git-path objects)" && pwd -P); I=$(git rev-parse --git-path index)
if [ -f "$I" ]; then cp -p "$I" "$X"; fi
export GIT_INDEX_FILE="$X" GIT_OBJECT_DIRECTORY="$S/objects" GIT_ALTERNATE_OBJECT_DIRECTORIES="$O"
if ! E=$(git add -A 2>&1 >/dev/null); then printf '%s\\n' "$E" | tail -c 1000 >&2; exit 1; fi
T=$(git write-tree)
if [ "$T" = {prev} ]; then echo "__SB_TREE $T same"; echo __SB_END; exit 0; fi
git diff --cached --binary baseline > "$P"
N=$(wc -c < "$P" | tr -d ' ')
echo "__SB_TREE $T $N"
if [ "$N" -le {max_bytes} ]; then base64 < "$P"; fi
echo __SB_END
"""
OUTPUT_RE = re.compile(r"^__SB_TREE ([0-9a-f]{40}|[0-9a-f]{64}) (same|\d+)\r?\n(.*?)^__SB_END\r?$", re.S | re.M)


def snapshot_command(workdir: str, state_root: str, prev_tree: str | None, max_bytes: int = MAX_PATCH_BYTES) -> str:
    """The shell command of one snapshot of the repository that contains ``workdir``."""
    return SNAPSHOT_SCRIPT.format(workdir=shlex.quote(workdir), snap=shlex.quote(f"{state_root}/snap"),
                                  prev=shlex.quote(prev_tree or "none"), max_bytes=int(max_bytes))


def parse_output(stdout: str, max_bytes: int = MAX_PATCH_BYTES) -> tuple[str, int | None, bytes | None, str | None]:
    """A successful command's output -> (tree, patch size, patch, error); size and patch are None
    when the tree was the previous one; raises ValueError on output it cannot read."""
    m = OUTPUT_RE.search(stdout)
    if m is None:
        raise ValueError("no snapshot in the output")
    tree, size, body = m[1], m[2], m[3]
    if size == "same":
        return tree, None, None, None
    n = int(size)
    if n > max_bytes:
        return tree, n, None, "too large"
    try:
        data = base64.b64decode("".join(body.split()), validate=True)
    except binascii.Error as e:
        raise ValueError(f"unreadable patch: {e}") from None
    if len(data) != n:
        raise ValueError(f"patch of {len(data)} bytes, expected {n}")
    return tree, n, data, None


class Snapshotter:
    """Background snapshots of the repository that contains ``workdir`` (see the module docstring)."""

    def __init__(self, env: Any, rec: EpisodeRecorder, workdir: str, every_sec: float,
                 max_bytes: int = MAX_PATCH_BYTES):
        self.env, self.rec, self.workdir, self.every_sec, self.max_bytes = env, rec, workdir, every_sec, max_bytes
        self.cv = threading.Condition()
        self.pending: float | None = None  # the earliest tool request not served yet
        self.stopping = False
        self.closed = False  # set by stop(): nothing is written after it
        self._write_lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.seq = 0
        self.tree: str | None = None  # the last successful snapshot's tree
        self.last_end = time.time()

    def start(self) -> None:
        """Take snapshot 0 now, then follow the triggers until :meth:`stop`."""
        t_request = time.time()
        self.rec.add_tool_listener(self.on_tool)
        self.thread = threading.Thread(target=self._run, args=(t_request,), name="snapshots", daemon=True)
        self.thread.start()

    def on_tool(self, record: dict[str, Any]) -> None:
        """The recorder's tool listener: called in the agent's thread, returns at once."""
        if record.get("name") in TRIGGERS:
            self.request()

    def request(self) -> None:
        with self.cv:
            if self.pending is None:
                self.pending = time.time()
            self.cv.notify_all()

    def stop(self, timeout: float = JOIN_SEC) -> None:
        """No new snapshot; wait up to ``timeout`` for the running one; nothing is written afterwards."""
        self.rec.remove_tool_listener(self.on_tool)
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout)
        with self._write_lock:
            self.closed = True

    # -- the thread ------------------------------------------------------------------------

    def _run(self, t_request: float) -> None:
        trigger: str | None = "start"
        while trigger is not None:
            try:
                if not self._take(trigger, t_request):
                    return
            except Exception:  # a harness bug or a failed record write: the next trigger tries again
                pass
            trigger, t_request = self._next()

    def _stopped(self) -> bool:
        stop_event = getattr(self.env, "stop_event", None)
        return self.stopping or (stop_event is not None and stop_event.is_set())

    def _next(self) -> tuple[str | None, float]:
        """Block until the next snapshot is due -> (trigger, t_request), or (None, 0) to end."""
        with self.cv:
            while not self._stopped():
                if self.pending is not None:
                    t, self.pending = self.pending, None
                    return "tool", t
                due = self.last_end + self.every_sec
                now = time.time()
                if now >= due:
                    return "heartbeat", due
                self.cv.wait(min(due - now, 1.0))  # also notices the env's stop event
        return None, 0.0

    def _take(self, trigger: str, t_request: float) -> bool:
        """One attempt and its record; False when the episode stopped (the thread ends)."""
        seq, self.seq = self.seq, self.seq + 1
        out: dict[str, Any] = {"seq": seq, "t_request": t_request, "t_start": time.time(), "t_end": None,
                               "trigger": trigger, "tree": None, "changed": False, "patch": None, "bytes": None,
                               "error": None}
        data, go_on = None, True
        try:
            res = self.env.exec(snapshot_command(self.workdir, self.env.state_root, self.tree, self.max_bytes),
                                EXEC_TIMEOUT_SEC)
        except AgentStopped:
            out["error"], go_on = "stopped", False
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"[:500]
        else:
            if res.timed_out:
                out["error"] = f"timed out after {EXEC_TIMEOUT_SEC} s"
            elif res.return_code != 0:
                out["error"] = f"exit {res.return_code}: {(res.stderr or res.stdout).strip()[-500:]}"
            else:
                try:
                    out["tree"], out["bytes"], data, out["error"] = parse_output(res.stdout, self.max_bytes)
                except ValueError as e:
                    out["error"] = str(e)[:500]
        out["t_end"] = self.last_end = time.time()
        out["changed"] = out["tree"] is not None and out["tree"] != self.tree
        with self._write_lock:
            if self.closed:
                return False
            if out["error"] is None:
                if out["changed"]:
                    try:
                        out["patch"] = self._write_patch(seq, data or b"")
                    except OSError as e:
                        out["error"] = f"{type(e).__name__}: {e}"[:500]
                if out["error"] is None:
                    self.tree = out["tree"]
            self.rec.write("snapshots", out)
        return go_on

    def _write_patch(self, seq: int, data: bytes) -> str:
        rel = f"{PATCH_DIR}/{seq:04d}.patch"
        path = Path(self.rec.dir) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return rel
