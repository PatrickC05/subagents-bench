"""Sandboxes with one ``exec(command, timeout)``: harbor's container, or the host shell (tests).

Every exec honours the environment's ``stop_event``: once it is set, ``exec`` raises
:class:`AgentStopped` (before starting, or while waiting) so agent threads unwind.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

POLL_SEC = 0.2


class AgentStopped(Exception):
    """The episode was stopped (harbor's cancel, or a provider failure)."""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    return_code: int | None
    timed_out: bool = False


class HarborEnvironment:
    """harbor's async ``environment.exec``, called from agent threads on harbor's loop."""

    state_root = "/tmp/.sb"  # harness state in the container (shell state, arbiter copies)

    def __init__(self, harbor_env: Any, loop: asyncio.AbstractEventLoop):
        self._env = harbor_env
        self._loop = loop
        self.stop_event = threading.Event()

    def exec(self, command: str, timeout: float) -> ExecResult:
        if self.stop_event.is_set():
            raise AgentStopped("episode stopped")
        coro = self._env.exec(command, timeout_sec=max(1, math.ceil(timeout)))
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        while not concurrent.futures.wait([fut], timeout=POLL_SEC).done:
            if self.stop_event.is_set():
                fut.cancel()
                raise AgentStopped("episode stopped during exec")
        try:
            res = fut.result()
        except Exception as e:  # harbor's docker backend: RuntimeError("Command timed out after N seconds")
            if isinstance(e, TimeoutError) or "timed out" in str(e).lower():
                return ExecResult("", str(e), None, True)
            raise
        return ExecResult(res.stdout or "", res.stderr or "", res.return_code)


def run_process(argv: list[str], timeout: float, stop_event: threading.Event, cwd: str | None = None) -> ExecResult:
    """Run ``argv`` in its own session; SIGKILL its process group on timeout or stop."""
    if stop_event.is_set():
        raise AgentStopped("episode stopped")
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        deadline = time.monotonic() + timeout
        timed_out = False
        while True:
            try:
                proc.wait(timeout=POLL_SEC)
                break
            except subprocess.TimeoutExpired:
                pass
            if stop_event.is_set() or time.monotonic() >= deadline:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                if stop_event.is_set():
                    raise AgentStopped("episode stopped during exec")
                timed_out = True
                break
        out.seek(0)
        err.seek(0)
        return ExecResult(
            out.read().decode("utf-8", "replace"),
            err.read().decode("utf-8", "replace"),
            None if timed_out else proc.returncode,
            timed_out,
        )


class LocalShellEnvironment:
    """The host's bash in ``workdir``; state under a private temp dir. Tests only: unsandboxed."""

    def __init__(self, workdir: str):
        self.workdir = workdir
        self.state_root = os.path.realpath(tempfile.mkdtemp(prefix="sb-state-"))
        self.stop_event = threading.Event()

    def exec(self, command: str, timeout: float) -> ExecResult:
        return run_process(["bash", "-c", command], timeout, self.stop_event, cwd=self.workdir)

    def close(self) -> None:
        shutil.rmtree(self.state_root, ignore_errors=True)
