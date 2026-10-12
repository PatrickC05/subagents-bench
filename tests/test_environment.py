"""Environment backends: exec semantics, timeouts, stop events, byte-exact
file transfer, and the harbor bridge (against a fake async harbor env)."""

from __future__ import annotations

import asyncio
import os
import stat
import threading
import time

import pytest

from subagents_bench import environment
from subagents_bench.environment import (
    ExecResult,
    HarborEnvironment,
    LocalShellEnvironment,
    _ShellFileMixin,
    check_stopped,
    plan_b64_upload,
    run_process,
)
from subagents_bench.types import AgentStopped


@pytest.fixture
def env(tmp_path):
    e = LocalShellEnvironment(str(tmp_path))
    yield e
    e.close()


# ---------------------------------------------------------------------------
# ExecResult / helpers
# ---------------------------------------------------------------------------


def test_exec_result_positional_construction_keeps_defaults():
    # exploitbench's CallableEnvironment builds it from three fields.
    r = ExecResult("out", "err", 3)
    assert r.duration_sec is None
    assert r.timed_out is False
    assert r.output == "out\n[stderr]\nerr"
    assert ExecResult("", "", 0).output == "(no output)"
    assert ExecResult("", "boom\n", 1).output == "boom"


def test_check_stopped_handles_envs_without_stop_event():
    class Bare:
        pass

    check_stopped(Bare())  # no attribute -> never stopped
    b = Bare()
    b.stop_event = threading.Event()
    check_stopped(b)
    b.stop_event.set()
    with pytest.raises(AgentStopped):
        check_stopped(b)


def test_plan_b64_upload_small_and_chunked(monkeypatch):
    pre, final = plan_b64_upload(b"hello", "/x/y", "/stage")
    assert pre == []
    assert "base64 -d > /x/y" in final
    monkeypatch.setattr(environment, "B64_CHUNK_CHARS", 8)
    pre, final = plan_b64_upload(b"0123456789" * 3, "/x/y", "/stage", mkdirs=True)
    # 30 bytes -> 40 base64 chars -> 5 chunks of 8
    assert len(pre) == 5
    assert pre[0].startswith("mkdir -p -- /stage && printf")
    assert all(">>" in c for c in pre[1:])
    assert final.startswith('{ mkdir -p -- "$(dirname -- /x/y)" && base64 -d <')


# ---------------------------------------------------------------------------
# LocalShellEnvironment exec
# ---------------------------------------------------------------------------


def test_exec_basic_fields_and_cwd(env, tmp_path):
    (tmp_path / "sub").mkdir()
    r = env.exec("echo out; echo err >&2; pwd; exit 5")
    assert r.stdout.splitlines() == ["out", str(tmp_path)]
    assert r.stderr == "err\n"
    assert r.return_code == 5
    assert r.timed_out is False
    assert r.duration_sec is not None and r.duration_sec >= 0
    assert env.exec("pwd", cwd=str(tmp_path / "sub")).stdout.strip() == str(tmp_path / "sub")


def test_exec_timeout_kills_process_group(env, tmp_path):
    flag = tmp_path / "flag"
    t0 = time.monotonic()
    r = env.exec(f"echo early; (sleep 2; touch {flag}) & sleep 30", timeout_sec=0.5)
    elapsed = time.monotonic() - t0
    assert r.timed_out is True
    assert r.return_code is None
    assert elapsed < 4
    assert r.stdout == "early\n"
    time.sleep(2.5)
    assert not flag.exists(), "background child survived the host-side kill"


def test_exec_does_not_hang_on_background_job_holding_stdout(env, monkeypatch, tmp_path):
    monkeypatch.setattr(environment, "READER_GRACE_SEC", 0.3)
    pidfile = tmp_path / "bg.pid"
    t0 = time.monotonic()
    r = env.exec(f"sleep 20 & echo $! > {pidfile}; echo started")
    try:
        assert time.monotonic() - t0 < 3
        assert r.return_code == 0
        assert r.stdout == "started\n"
    finally:
        os.kill(int(pidfile.read_text()), 9)


def test_stop_event_before_and_during_exec(env):
    env.stop_event = threading.Event()
    env.stop_event.set()
    with pytest.raises(AgentStopped):
        env.exec("echo never")
    env.stop_event = threading.Event()
    threading.Timer(0.3, env.stop_event.set).start()
    t0 = time.monotonic()
    with pytest.raises(AgentStopped):
        env.exec("sleep 10")
    assert time.monotonic() - t0 < 3


def test_run_process_reports_nonzero_and_unicode(tmp_path):
    r = run_process(["bash", "-c", "printf 'h\\xc3\\xa9\\xff'; exit 2"], timeout_sec=10)
    assert r.return_code == 2
    assert r.stdout == "hé�"


def test_close_removes_only_own_state_root(tmp_path):
    e = LocalShellEnvironment(str(tmp_path))
    root = e.state_root
    assert os.path.isdir(root)
    e.close()
    assert not os.path.exists(root)
    keep = tmp_path / "keep"
    keep.mkdir()
    e2 = LocalShellEnvironment(str(tmp_path), state_root=str(keep))
    e2.close()
    assert keep.is_dir()


# ---------------------------------------------------------------------------
# byte-exact file helpers
# ---------------------------------------------------------------------------

TRICKY = [
    "no trailing newline",
    "one\n",
    "two newlines\n\n",
    "",
    "EOF\nEOF_marker\n'single' \"double\" $HOME `id` \\backslash\\\n",
    "unicode: héllo wörld ✓ 日本\n",
    "\ttabs\r\nand CRLF\r\n",
]


@pytest.mark.parametrize("content", TRICKY)
def test_write_read_roundtrip_exact(env, tmp_path, content):
    env.write_file("t.txt", content)
    assert (tmp_path / "t.txt").read_bytes() == content.encode()
    assert env.read_file("t.txt") == content


def test_write_read_bytes_binary_and_chunked(env, tmp_path, monkeypatch):
    data = bytes(range(256)) * 3
    monkeypatch.setattr(environment, "B64_CHUNK_CHARS", 100)  # force ~11 chunks
    env.write_bytes(str(tmp_path / "bin.dat"), data)
    assert (tmp_path / "bin.dat").read_bytes() == data
    assert env.read_bytes(str(tmp_path / "bin.dat")) == data
    # staged chunk files are cleaned up
    assert os.listdir(os.path.join(env.state_root, "xfer")) == []


def test_write_large_payload_real_chunking(env, tmp_path):
    data = os.urandom(300_000)  # > B64_CHUNK_CHARS after encoding
    env.write_bytes("big.bin", data)
    assert (tmp_path / "big.bin").read_bytes() == data


def test_write_creates_parents_and_keeps_mode(env, tmp_path):
    env.write_file("a/b c/d.txt", "x")
    assert (tmp_path / "a" / "b c" / "d.txt").read_text() == "x"
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\necho old\n")
    script.chmod(0o755)
    env.write_file(str(script), "#!/bin/sh\necho new\n")
    assert script.read_text() == "#!/bin/sh\necho new\n"
    assert stat.S_IMODE(script.stat().st_mode) == 0o755


def test_read_errors(env, tmp_path):
    with pytest.raises(FileNotFoundError):
        env.read_file("missing.txt")
    (tmp_path / "d").mkdir()
    with pytest.raises(IsADirectoryError):
        env.read_file("d")


def test_legacy_mixin_subclass_without_init_or_cwd(tmp_path):
    """The shape of exploitbench's CallableEnvironment: exec(command, timeout_sec)
    only, no super().__init__, ExecResult built positionally."""
    import subprocess

    class Legacy(_ShellFileMixin):
        def __init__(self, workdir):
            self.workdir = workdir

        def exec(self, command, timeout_sec=None):
            p = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=self.workdir)
            return ExecResult(p.stdout, p.stderr, p.returncode)

    e = Legacy(str(tmp_path))
    assert e.stop_event is None
    assert e.state_root == "/tmp/.sb"
    e.write_file("f.txt", "abc")
    assert e.read_file("f.txt") == "abc"
    e.check_stopped()


def test_real_callable_environment_still_works(tmp_path):
    try:
        from subagents_bench.harness.exploitbench_agent import CallableEnvironment
    except ImportError as e:  # its other imports may be mid-rewrite
        pytest.skip(f"exploitbench_agent not importable: {e}")
    import subprocess

    def exec_fn(command, timeout_sec):
        p = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=str(tmp_path))
        return p.stdout, p.stderr, p.returncode

    e = CallableEnvironment(exec_fn, workdir=str(tmp_path))
    e.write_file("g.txt", "no newline")
    assert (tmp_path / "g.txt").read_text() == "no newline"
    r = e.exec("echo hi")
    assert (r.stdout, r.return_code, r.timed_out) == ("hi\n", 0, False)


# ---------------------------------------------------------------------------
# HarborEnvironment against a fake async harbor env
# ---------------------------------------------------------------------------


class _HarborResult:
    def __init__(self, stdout, stderr, return_code):
        self.stdout, self.stderr, self.return_code = stdout, stderr, return_code


class FakeHarborEnv:
    """Mimics harbor BaseEnvironment.exec: async, keyword cwd/timeout_sec."""

    def __init__(self, behavior="ok"):
        self.behavior = behavior
        self.calls = []
        self.cancelled = threading.Event()

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.calls.append({"command": command, "cwd": cwd, "timeout_sec": timeout_sec, "thread": threading.current_thread().name})
        if self.behavior == "ok":
            return _HarborResult(f"ran: {command}", None, 0)
        if self.behavior == "harbor_timeout":
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        if self.behavior == "boom":
            raise RuntimeError("docker compose exploded")
        if self.behavior == "hang":
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        raise AssertionError(self.behavior)


@pytest.fixture
def harbor_loop():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="harbor-loop", daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)
    loop.close()


def test_harbor_exec_maps_result_and_passes_cwd_timeout(harbor_loop):
    fake = FakeHarborEnv()
    env = HarborEnvironment(fake, harbor_loop, workdir="/app")
    r = env.exec("echo hi", timeout_sec=2.5)
    assert (r.stdout, r.stderr, r.return_code, r.timed_out) == ("ran: echo hi", "", 0, False)
    assert r.duration_sec is not None
    assert fake.calls[0]["cwd"] == "/app"
    assert fake.calls[0]["timeout_sec"] == 3  # ceil, harbor takes int seconds
    assert fake.calls[0]["thread"] == "harbor-loop"  # ran on harbor's loop
    env.exec("pwd", cwd="/tmp")
    assert fake.calls[1]["cwd"] == "/tmp"
    assert fake.calls[1]["timeout_sec"] is None


def test_harbor_workdir_none_uses_harbor_default(harbor_loop):
    fake = FakeHarborEnv()
    env = HarborEnvironment(fake, harbor_loop)
    env.exec("pwd")
    assert fake.calls[0]["cwd"] is None


def test_harbor_timeout_error_becomes_timed_out(harbor_loop):
    env = HarborEnvironment(FakeHarborEnv("harbor_timeout"), harbor_loop)
    r = env.exec("sleep 99", timeout_sec=5)
    assert r.timed_out is True
    assert r.return_code is None
    assert "timed out" in r.stderr


def test_harbor_other_errors_propagate(harbor_loop):
    env = HarborEnvironment(FakeHarborEnv("boom"), harbor_loop)
    with pytest.raises(RuntimeError, match="exploded"):
        env.exec("x", timeout_sec=5)


def test_harbor_host_side_safety_timeout_cancels(harbor_loop, monkeypatch):
    fake = FakeHarborEnv("hang")
    env = HarborEnvironment(fake, harbor_loop)
    monkeypatch.setattr(HarborEnvironment, "HOST_SLACK_SEC", 0.3)
    t0 = time.monotonic()
    r = env.exec("sleep 60", timeout_sec=0.5)
    assert time.monotonic() - t0 < 3
    assert r.timed_out is True and r.return_code is None
    assert fake.cancelled.wait(2), "the harbor coroutine was not cancelled"


def test_harbor_stop_event_interrupts_wait(harbor_loop):
    fake = FakeHarborEnv("hang")
    env = HarborEnvironment(fake, harbor_loop)
    env.stop_event = threading.Event()
    threading.Timer(0.3, env.stop_event.set).start()
    t0 = time.monotonic()
    with pytest.raises(AgentStopped):
        env.exec("sleep 60", timeout_sec=100)
    assert time.monotonic() - t0 < 3
    assert fake.cancelled.wait(2)
    # once stopped, nothing new is submitted
    n = len(fake.calls)
    with pytest.raises(AgentStopped):
        env.exec("echo again")
    assert len(fake.calls) == n


def test_harbor_exec_on_loop_thread_refuses_instead_of_deadlocking(harbor_loop):
    env = HarborEnvironment(FakeHarborEnv(), harbor_loop)

    async def call_from_loop():
        return env.exec("echo hi")

    fut = asyncio.run_coroutine_threadsafe(call_from_loop(), harbor_loop)
    with pytest.raises(RuntimeError, match="event-loop thread"):
        fut.result(timeout=5)


def test_harbor_private_loop_fallback():
    fake = FakeHarborEnv()
    env = HarborEnvironment(fake, workdir="/w")
    try:
        assert env.exec("echo x").stdout == "ran: echo x"
        assert fake.calls[0]["thread"] == "harbor-env-loop"
    finally:
        env.close()
    assert env._thread is None


def test_harbor_file_helpers_go_through_exec(harbor_loop):
    """write_file sends a base64 command; read_file decodes base64 stdout."""

    class B64Fake(FakeHarborEnv):
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            self.calls.append(command)
            import subprocess

            p = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=self.root)
            return _HarborResult(p.stdout or None, p.stderr or None, p.returncode)

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        fake = B64Fake()
        fake.root = d
        env = HarborEnvironment(fake, harbor_loop, workdir=d)
        env.write_file("x.txt", "a\nb")
        assert open(os.path.join(d, "x.txt"), "rb").read() == b"a\nb"
        assert env.read_file("x.txt") == "a\nb"
        assert any("base64 -d" in c for c in fake.calls)
