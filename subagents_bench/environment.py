"""Execution environment abstraction.

Every tool the agents use ultimately runs a shell command inside a sandbox. We
define one small synchronous protocol, :class:`Environment`, with a single
``exec`` primitive plus file read/write helpers built on top of it. Concrete
sandboxes:

* :class:`HarborEnvironment` -- wraps a harbor ``BaseEnvironment`` (the live
  SWE-bench / Terminal-Bench container the harness hands us). ``exec`` is async
  there, so we bridge it onto a private event loop.
* :class:`LocalDockerEnvironment` -- the standalone runner's own container,
  started from an image, used for local smoke tests.

ExploitBench is driven through its MCP tool interface rather than a raw shell;
its adapter lives in :mod:`subagents_bench.harness.exploitbench_agent` and
presents the same :class:`Environment` surface.
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    return_code: int

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
    workdir: str

    def exec(self, command: str, timeout_sec: int | None = None) -> ExecResult: ...

    def read_file(self, path: str) -> str: ...

    def write_file(self, path: str, content: str) -> None: ...


class _ShellFileMixin:
    """read/write_file implemented via exec so every backend gets them free."""

    def exec(self, command: str, timeout_sec: int | None = None) -> ExecResult:  # pragma: no cover - overridden
        raise NotImplementedError

    def read_file(self, path: str) -> str:
        res = self.exec(f"cat {shlex.quote(path)}")
        if res.return_code != 0:
            raise FileNotFoundError(res.stderr or path)
        return res.stdout

    def write_file(self, path: str, content: str) -> None:
        marker = f"EOF_{uuid.uuid4().hex}"
        # Use a quoted heredoc so content is written verbatim (no expansion).
        script = f"cat > {shlex.quote(path)} <<'{marker}'\n{content}\n{marker}\n"
        res = self.exec(script)
        if res.return_code != 0:
            raise OSError(res.stderr or f"failed to write {path}")


class HarborEnvironment(_ShellFileMixin):
    """Bridge a harbor async ``BaseEnvironment`` to our sync protocol."""

    def __init__(self, harbor_env: Any, workdir: str = "/app"):
        self._env = harbor_env
        self.workdir = workdir
        # A dedicated loop + thread so sync callers (the agent loop) can drive
        # the async harbor env regardless of any ambient loop.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def exec(self, command: str, timeout_sec: int | None = None) -> ExecResult:
        res = self._run(
            self._env.exec(command, cwd=self.workdir, timeout_sec=timeout_sec)
        )
        return ExecResult(
            stdout=res.stdout or "",
            stderr=res.stderr or "",
            return_code=res.return_code,
        )

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)


class LocalDockerEnvironment(_ShellFileMixin):
    """A container we start ourselves, for local development and smoke tests."""

    def __init__(self, image: str, workdir: str = "/app", network: bool = False):
        self.image = image
        self.workdir = workdir
        self._cid: str | None = None
        net = "bridge" if network else "none"
        cid = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "--network", net,
                "-w", workdir,
                image, "sleep", "infinity",
            ],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self._cid = cid

    def exec(self, command: str, timeout_sec: int | None = None) -> ExecResult:
        assert self._cid is not None
        try:
            proc = subprocess.run(
                ["docker", "exec", "-w", self.workdir, self._cid, "bash", "-lc", command],
                capture_output=True, text=True, timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(stdout=e.stdout or "", stderr=f"timed out after {timeout_sec}s", return_code=124)
        return ExecResult(stdout=proc.stdout, stderr=proc.stderr, return_code=proc.returncode)

    def close(self) -> None:
        if self._cid:
            subprocess.run(["docker", "kill", self._cid], capture_output=True)
            self._cid = None


class LocalShellEnvironment(_ShellFileMixin):
    """Runs commands on the host inside a working directory.

    Only for unit tests and offline demos -- NEVER use for real benchmark runs,
    since agent-generated commands would execute unsandboxed on the host.
    """

    def __init__(self, workdir: str):
        self.workdir = workdir

    def exec(self, command: str, timeout_sec: int | None = None) -> ExecResult:
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                capture_output=True, text=True, cwd=self.workdir, timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(stdout=e.stdout or "", stderr=f"timed out after {timeout_sec}s", return_code=124)
        return ExecResult(stdout=proc.stdout, stderr=proc.stderr, return_code=proc.returncode)

    def close(self) -> None:
        pass
