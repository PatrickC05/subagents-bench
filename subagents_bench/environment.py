"""Execution environments: one synchronous ``exec`` primitive over every sandbox.

Every tool the agents use ultimately runs a shell command inside a sandbox. The
:class:`Environment` protocol is deliberately small::

    exec(command, timeout_sec=None, cwd=None)
        -> ExecResult(stdout, stderr, return_code, duration_sec, timed_out)

plus byte-exact file helpers (``read_file``/``write_file`` and their
``*_bytes`` variants) built on top of ``exec`` with base64 transfer, so every
backend gets them for free (:class:`_ShellFileMixin`).

Backends:

* :class:`HarborEnvironment` -- wraps harbor's async ``BaseEnvironment`` (the
  Terminal-Bench container). Worker threads submit ``harbor_env.exec`` to
  harbor's own event loop with :func:`asyncio.run_coroutine_threadsafe`.
* :class:`LocalDockerEnvironment` -- a container we start ourselves (local
  smoke runs and the ``docker``-marked tests).
* :class:`LocalShellEnvironment` -- the host shell in a directory. Tests only:
  agent commands would run unsandboxed on the host.

ExploitBench's ``CallableEnvironment`` (in
:mod:`subagents_bench.harness.exploitbench_agent`) subclasses
:class:`_ShellFileMixin` and builds :class:`ExecResult` positionally from
``(stdout, stderr, return_code)``; the extra fields therefore have defaults.

Every environment exposes ``stop_event`` (a :class:`threading.Event` or None).
Once it is set, ``exec`` raises :class:`~subagents_bench.types.AgentStopped`
-- before starting a command, and while waiting on a running one -- so worker
threads unwind promptly when harbor cancels the agent.

Timeouts here are *host-side*: they stop waiting (and kill the local client
process) but cannot reach processes inside a container. The in-container kill
(``timeout -k 5 T bash ...``) is the tools layer's job (:mod:`.tools`).
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import math
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .types import AgentStopped

# Root of all harness-owned state inside the sandbox (per-agent shell state,
# markers, staged uploads, arbiter copies). Excluded from file-change scans.
DEFAULT_STATE_ROOT = "/tmp/.sb"

# Linux limits a single argv string to 128 KiB (MAX_ARG_STRLEN), and every
# command reaches the container as one ``bash -c <command>`` argument, so
# base64 payloads above this size are uploaded in chunks.
B64_CHUNK_CHARS = 96_000

# Host-side timeout for the file helpers.
FILE_OP_TIMEOUT_SEC = 300

# How often blocked waits wake up to check the stop event / deadline.
_POLL_SEC = 0.2

# After a local process exits, how long to keep reading its pipes. A
# background job that inherited stdout can hold the pipe open indefinitely;
# we return what we have instead of waiting for it.
READER_GRACE_SEC = 2.0


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    # None when no exit status was observed (killed host-side on timeout).
    return_code: int | None
    # Host-measured wall time of the exec; None if the backend did not measure.
    duration_sec: float | None = None
    # True if the host-side timeout fired (the backend gave up waiting).
    timed_out: bool = False

    @property
    def output(self) -> str:
        """Combined, trimmed output as agents should see it."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(self.stderr if not self.stdout else f"[stderr]\n{self.stderr}")
        text = "\n".join(parts).rstrip()
        return text if text else "(no output)"


class Environment(Protocol):
    # Default working directory for exec (None: the backend's default).
    workdir: str | None
    stop_event: threading.Event | None
    # Root for harness state inside the sandbox (DEFAULT_STATE_ROOT normally).
    state_root: str

    def exec(self, command: str, timeout_sec: float | None = None, cwd: str | None = None) -> ExecResult: ...

    def read_file(self, path: str) -> str: ...

    def write_file(self, path: str, content: str | bytes) -> None: ...


def check_stopped(env: Any) -> None:
    """Raise AgentStopped if ``env``'s stop event is set (envs without one never stop)."""
    ev = getattr(env, "stop_event", None)
    if ev is not None and ev.is_set():
        raise AgentStopped("episode stopped")


def plan_b64_upload(
    data: bytes, dest: str, staging_dir: str, *, mkdirs: bool = False
) -> tuple[list[str], str]:
    """Shell commands that land ``data`` byte-exactly at ``dest``.

    Returns ``(pre, final)``: ``pre`` are commands to run first, one exec each
    (empty unless the payload exceeds :data:`B64_CHUNK_CHARS`); ``final`` is a
    snippet that completes the write and can be embedded in a larger script
    (its exit status is the write's). ``dest`` is truncated in place, so an
    existing file keeps its mode and owner. Nothing is appended -- no trailing
    newline, unlike a heredoc.
    """
    b64 = base64.b64encode(data).decode("ascii")
    qd = shlex.quote(dest)
    mk = f'mkdir -p -- "$(dirname -- {qd})" && ' if mkdirs else ""
    if len(b64) <= B64_CHUNK_CHARS:
        return [], f"{{ {mk}printf '%s' {shlex.quote(b64)} | base64 -d > {qd}; }}"
    staged = shlex.quote(f"{staging_dir}/{uuid.uuid4().hex}.b64")
    pre = []
    for i in range(0, len(b64), B64_CHUNK_CHARS):
        chunk = shlex.quote(b64[i : i + B64_CHUNK_CHARS])
        if i == 0:
            pre.append(f"mkdir -p -- {shlex.quote(staging_dir)} && printf '%s' {chunk} > {staged}")
        else:
            pre.append(f"printf '%s' {chunk} >> {staged}")
    final = f"{{ {mk}base64 -d < {staged} > {qd}; __sb_w=$?; rm -f {staged}; [ $__sb_w -eq 0 ]; }}"
    return pre, final


class _ShellFileMixin:
    """File helpers implemented via ``exec`` so every backend gets them free.

    Relative paths resolve against the backend's default cwd. Defaults are
    class attributes so subclasses that skip ``__init__`` (ExploitBench's
    ``CallableEnvironment``) still work.
    """

    workdir: str | None = None
    stop_event: threading.Event | None = None
    state_root: str = DEFAULT_STATE_ROOT

    def exec(self, command: str, timeout_sec: float | None = None, cwd: str | None = None) -> ExecResult:  # pragma: no cover - overridden
        raise NotImplementedError

    def check_stopped(self) -> None:
        check_stopped(self)

    def _exec_checked(self, command: str, what: str) -> ExecResult:
        check_stopped(self)
        res = self.exec(command, timeout_sec=FILE_OP_TIMEOUT_SEC)
        if res.timed_out:
            raise TimeoutError(f"{what} timed out after {FILE_OP_TIMEOUT_SEC}s")
        return res

    def read_bytes(self, path: str) -> bytes:
        q = shlex.quote(path)
        res = self._exec_checked(
            f"if [ -d {q} ]; then exit 45; fi; if [ ! -e {q} ]; then exit 44; fi; base64 < {q}",
            f"reading {path}",
        )
        if res.return_code == 44:
            raise FileNotFoundError(path)
        if res.return_code == 45:
            raise IsADirectoryError(path)
        if res.return_code != 0:
            raise OSError(f"failed to read {path}: {res.output}")
        # b64decode discards non-alphabet characters (the line wrapping).
        return base64.b64decode(res.stdout)

    def read_file(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8", errors="replace")

    def write_bytes(self, path: str, data: bytes, *, mkdirs: bool = True) -> None:
        """Write ``data`` to ``path`` exactly (creating parent dirs if ``mkdirs``)."""
        pre, final = plan_b64_upload(data, path, f"{self.state_root}/xfer", mkdirs=mkdirs)
        for cmd in pre:
            res = self._exec_checked(cmd, f"uploading {path}")
            if res.return_code != 0:
                raise OSError(f"failed to stage upload for {path}: {res.output}")
        res = self._exec_checked(final, f"writing {path}")
        if res.return_code != 0:
            raise OSError(f"failed to write {path}: {res.output}")

    def write_file(self, path: str, content: str | bytes) -> None:
        data = content.encode("utf-8") if isinstance(content, str) else content
        self.write_bytes(path, data)


# ---------------------------------------------------------------------------
# harbor
# ---------------------------------------------------------------------------


def _is_timeout_error(e: BaseException) -> bool:
    # harbor's docker backend raises RuntimeError("Command timed out after N
    # seconds"); other backends may raise TimeoutError/asyncio.TimeoutError.
    if isinstance(e, TimeoutError):
        return True
    return isinstance(e, RuntimeError) and "timed out" in str(e).lower()


class HarborEnvironment(_ShellFileMixin):
    """Bridge harbor's async ``BaseEnvironment`` to the synchronous protocol.

    ``loop`` is harbor's running event loop (captured with
    ``asyncio.get_running_loop()`` in the agent's ``run``); ``exec`` must then
    be called from worker threads (``asyncio.to_thread``), never from the loop
    thread itself, which would deadlock. If ``loop`` is None a private loop
    thread is started instead (the pre-v2 behavior).

    ``workdir=None`` lets harbor use the task's WORKDIR; resolve it once with
    :func:`subagents_bench.workspace.resolve_workdir` and assign it.

    Timeouts: ``timeout_sec`` is passed to harbor (which kills its local
    ``docker exec`` client and raises); as a safety net for backends that
    ignore it, we stop waiting ``HOST_SLACK_SEC`` later. Either way the result
    has ``timed_out=True`` and ``return_code=None``.
    """

    HOST_SLACK_SEC = 10.0

    def __init__(self, harbor_env: Any, loop: asyncio.AbstractEventLoop | None = None, workdir: str | None = None):
        self._env = harbor_env
        self.workdir = workdir
        self.stop_event: threading.Event | None = None
        self.state_root = DEFAULT_STATE_ROOT
        self._thread: threading.Thread | None = None
        if loop is None:
            loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=loop.run_forever, name="harbor-env-loop", daemon=True)
            self._thread.start()
        self._loop = loop

    def exec(self, command: str, timeout_sec: float | None = None, cwd: str | None = None) -> ExecResult:
        check_stopped(self)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            raise RuntimeError(
                "HarborEnvironment.exec called on harbor's event-loop thread; "
                "run the agent in a worker thread (asyncio.to_thread)"
            )
        t0 = time.monotonic()
        harbor_timeout = None if timeout_sec is None else max(1, int(math.ceil(timeout_sec)))
        fut = asyncio.run_coroutine_threadsafe(
            self._env.exec(command, cwd=cwd if cwd is not None else self.workdir, timeout_sec=harbor_timeout),
            self._loop,
        )
        deadline = None if timeout_sec is None else t0 + timeout_sec + self.HOST_SLACK_SEC
        while True:
            wait = _POLL_SEC if deadline is None else max(0.0, min(_POLL_SEC, deadline - time.monotonic()))
            done, _ = concurrent.futures.wait([fut], timeout=wait)
            if done:
                break
            if self.stop_event is not None and self.stop_event.is_set():
                fut.cancel()
                raise AgentStopped("episode stopped during exec")
            if deadline is not None and time.monotonic() >= deadline:
                fut.cancel()
                return ExecResult("", f"[host-side timeout after {timeout_sec}s]", None, time.monotonic() - t0, True)
        dur = time.monotonic() - t0
        if fut.cancelled():
            raise AgentStopped("harbor exec was cancelled")
        exc = fut.exception()
        if exc is not None:
            if _is_timeout_error(exc):
                return ExecResult("", str(exc), None, dur, True)
            raise exc
        res = fut.result()
        return ExecResult(
            stdout=res.stdout or "",
            stderr=res.stderr or "",
            return_code=res.return_code,
            duration_sec=dur,
        )

    def close(self) -> None:
        # Only stop a loop we own; harbor's loop is not ours to stop.
        if self._thread is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._thread = None


# ---------------------------------------------------------------------------
# local processes (docker client / host shell)
# ---------------------------------------------------------------------------


def _start_reader(stream: Any, sink: list[bytes]) -> threading.Thread:
    def pump() -> None:
        try:
            while True:
                chunk = stream.read1(65536)
                if not chunk:
                    break
                sink.append(chunk)
        except (OSError, ValueError):
            pass

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return t


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_process(
    argv: list[str],
    *,
    timeout_sec: float | None,
    stop_event: threading.Event | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> ExecResult:
    """Run ``argv`` in its own session; kill the whole process group on
    timeout or stop. Output is decoded as UTF-8 (invalid bytes replaced)."""
    t0 = time.monotonic()
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    out: list[bytes] = []
    err: list[bytes] = []
    readers = [_start_reader(proc.stdout, out), _start_reader(proc.stderr, err)]
    deadline = None if timeout_sec is None else t0 + timeout_sec
    timed_out = False
    try:
        while True:
            wait = _POLL_SEC if deadline is None else max(0.0, min(_POLL_SEC, deadline - time.monotonic()))
            try:
                proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                pass
            if stop_event is not None and stop_event.is_set():
                _kill_group(proc)
                raise AgentStopped("episode stopped during exec")
            if deadline is not None and time.monotonic() >= deadline:
                _kill_group(proc)
                timed_out = True
                break
    except BaseException:
        if proc.poll() is None:
            _kill_group(proc)
        raise
    for r in readers:
        r.join(timeout=READER_GRACE_SEC)
    return ExecResult(
        stdout=b"".join(out).decode("utf-8", errors="replace"),
        stderr=b"".join(err).decode("utf-8", errors="replace"),
        return_code=None if timed_out else proc.returncode,
        duration_sec=time.monotonic() - t0,
        timed_out=timed_out,
    )


class LocalDockerEnvironment(_ShellFileMixin):
    """A container we start ourselves (``docker run -d --init ... sleep infinity``).

    ``--init`` reaps zombies left by killed commands (PID 1 would otherwise
    be ``sleep``). ``workdir=None`` keeps the image's WORKDIR. ``platform``
    (e.g. ``linux/amd64``) is passed to ``docker run`` when given.
    """

    def __init__(
        self,
        image: str,
        workdir: str | None = "/app",
        network: bool = False,
        *,
        platform: str | None = None,
        state_root: str = DEFAULT_STATE_ROOT,
    ):
        self.image = image
        self.workdir = workdir
        self.stop_event: threading.Event | None = None
        self.state_root = state_root
        self._cid: str | None = None
        argv = ["docker", "run", "-d", "--rm", "--init", "--network", "bridge" if network else "none"]
        if platform:
            argv += ["--platform", platform]
        if workdir:
            argv += ["-w", workdir]
        argv += ["--entrypoint", "sleep", image, "infinity"]
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"docker run {image} failed: {proc.stderr.strip()}")
        self._cid = proc.stdout.strip()

    @property
    def container_id(self) -> str | None:
        return self._cid

    def exec(self, command: str, timeout_sec: float | None = None, cwd: str | None = None) -> ExecResult:
        check_stopped(self)
        if self._cid is None:
            raise RuntimeError("container is closed")
        argv = ["docker", "exec"]
        wd = cwd if cwd is not None else self.workdir
        if wd:
            argv += ["-w", wd]
        argv += [self._cid, "bash", "-c", command]
        return run_process(argv, timeout_sec=timeout_sec, stop_event=self.stop_event)

    def close(self) -> None:
        if self._cid:
            subprocess.run(["docker", "rm", "-f", self._cid], capture_output=True)
            self._cid = None

    def __enter__(self) -> "LocalDockerEnvironment":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class LocalShellEnvironment(_ShellFileMixin):
    """Runs commands with the host's bash inside a working directory.

    Only for unit tests and offline demos -- NEVER use for real benchmark runs,
    since agent-generated commands would execute unsandboxed on the host.

    Harness state goes to a private temporary ``state_root`` (removed by
    :meth:`close`) rather than the host's ``/tmp/.sb``, so concurrent test
    runs cannot see each other's shell state. ``env`` entries override the
    inherited environment (e.g. a restricted ``PATH``); ``bash`` picks the
    interpreter (default: ``bash`` on the host PATH).
    """

    def __init__(
        self,
        workdir: str,
        *,
        state_root: str | None = None,
        env: dict[str, str] | None = None,
        bash: str | None = None,
    ):
        self.workdir = workdir
        self.stop_event: threading.Event | None = None
        self._own_state = state_root is None
        self.state_root = os.path.realpath(state_root or tempfile.mkdtemp(prefix="sb-state-"))
        self._env = {**os.environ, **env} if env else None
        self._bash = bash or shutil.which("bash") or "/bin/bash"

    def exec(self, command: str, timeout_sec: float | None = None, cwd: str | None = None) -> ExecResult:
        check_stopped(self)
        return run_process(
            [self._bash, "-c", command],
            timeout_sec=timeout_sec,
            stop_event=self.stop_event,
            cwd=cwd if cwd is not None else self.workdir,
            env=self._env,
        )

    def close(self) -> None:
        if self._own_state:
            shutil.rmtree(self.state_root, ignore_errors=True)
