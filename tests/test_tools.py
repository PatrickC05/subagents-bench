"""Tools through the per-agent shell, on the host (LocalShellEnvironment) and in a Linux container."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import pytest

from subagents_bench.environment import AgentStopped, LocalShellEnvironment, run_process
from subagents_bench.tools import ToolContext, remap_args, run_tool


@pytest.fixture
def env(tmp_path):
    e = LocalShellEnvironment(os.path.realpath(tmp_path))
    yield e
    e.close()


def ctx_for(env, agent="a1", **kw) -> ToolContext:
    return ToolContext(env, agent, env.workdir, **{"max_output_chars": 400, "command_timeout_sec": 10,
                                                    "max_command_timeout_sec": 20, **kw})


def bash(ctx, command, **kw):
    return run_tool(ctx, "bash", {"command": command, **kw})


def test_cwd_and_exports_persist_per_agent(env):
    a, b = ctx_for(env, "a1"), ctx_for(env, "b2")
    assert bash(a, "mkdir -p sub && cd sub && export FOO=bar && echo hi").text == "hi"
    assert bash(a, "pwd; echo $FOO").text == f"{env.workdir}/sub\nbar"
    assert bash(b, "pwd; echo ${FOO:-unset}").text == f"{env.workdir}\nunset"
    out = bash(a, "echo out; echo err >&2; exit 3")
    assert out.text == "out\nerr\n[exit code: 3]" and out.exit_code == 3 and not out.is_error
    bash(a, "cd ..; PATH=/nowhere; ls")
    assert bash(a, "pwd").text == env.workdir  # saved even though the command broke PATH


def test_heredocs_quotes_and_closed_stdin(env):
    c = ctx_for(env)
    assert bash(c, "cat <<'EOF'\n$HOME 'q' \"d\"\nEOF").text == "$HOME 'q' \"d\""
    assert bash(c, "read x; echo rc=$?").text == "rc=1"


def test_long_output_keeps_head_and_tail(env):
    out = bash(ctx_for(env), "seq 1 2000")
    assert out.output_chars == len("\n".join(map(str, range(1, 2001)))) + 1
    assert out.text.startswith("1\n2\n") and out.text.endswith("1999\n2000") and "bytes elided" in out.text
    assert len(out.text) < 500


def test_timeout_kills_the_whole_process_group(env, tmp_path):
    c = ctx_for(env, command_timeout_sec=1)
    t0 = time.time()
    out = bash(c, "(sleep 2; touch late) & echo started; sleep 30")
    assert out.timed_out and out.text.startswith("started") and "timed out after 1s" in out.text
    assert time.time() - t0 < 8
    time.sleep(2.5)
    assert not (tmp_path / "late").exists()
    assert bash(c, "echo again").text == "again"  # the shell still works


def test_timeout_is_capped_at_the_maximum(env):
    out = bash(ctx_for(env, max_command_timeout_sec=1), "sleep 5", timeout_sec=600)
    assert out.timed_out and "after 1s" in out.text


def test_edit_file(env, tmp_path):
    c = ctx_for(env)
    bash(c, "mkdir -p d && cd d")
    assert run_tool(c, "edit_file", {"path": "f.txt", "old": "", "new": "a\nb\na\n"}).text.startswith("Wrote")
    assert (tmp_path / "d/f.txt").read_text() == "a\nb\na\n"  # relative to the agent's cwd
    assert "2 times" in run_tool(c, "edit_file", {"path": "f.txt", "old": "a", "new": "x"}).text
    assert "0 times" in run_tool(c, "edit_file", {"path": "f.txt", "old": "zzz", "new": "x"}).text
    assert run_tool(c, "edit_file", {"path": "f.txt", "old": "b\n", "new": "B\n"}).text.startswith("Edited")
    assert (tmp_path / "d/f.txt").read_text() == "a\nB\na\n"
    assert "no such file" in run_tool(c, "edit_file", {"path": "nope", "old": "x", "new": "y"}).text
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfeX\x00")
    run_tool(c, "edit_file", {"path": str(tmp_path / "bin.dat"), "old": "X", "new": "Y"})
    assert (tmp_path / "bin.dat").read_bytes() == b"\xff\xfeY\x00"


def test_read_list_search(env, tmp_path):
    c = ctx_for(env)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/m.py").write_text("one\ntwo\nthree\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/HEAD").write_text("ref")
    assert run_tool(c, "read_file", {"path": "pkg/m.py", "start_line": 2, "end_line": 3}).text == "     2\ttwo\n     3\tthree"
    missing = run_tool(c, "read_file", {"path": "nope.txt"})
    assert missing.is_error and missing.text == "[error] no such file: nope.txt"
    assert run_tool(c, "list_dir", {}).text == ".git/\npkg/\npkg/m.py"
    assert run_tool(c, "search", {"pattern": "t(wo|hree)"}).text == "./pkg/m.py:2:two\n./pkg/m.py:3:three"
    assert run_tool(c, "search", {"pattern": "absent"}).text == "No matches."


def test_bad_calls_become_error_outputs(env):
    c = ctx_for(env)
    assert run_tool(c, "rm_rf", {}).is_error and run_tool(c, "bash", "ls").is_error
    assert run_tool(c, "bash", {"command": " "}).is_error


def test_stop_event_raises_agent_stopped(env):
    c = ctx_for(env)
    threading.Timer(0.5, env.stop_event.set).start()
    t0 = time.time()
    with pytest.raises(AgentStopped):
        bash(c, "sleep 5")
    assert time.time() - t0 < 2
    with pytest.raises(AgentStopped):
        bash(c, "echo never")


@pytest.mark.parametrize("command,expected", [
    ("cd /app && ls /app/src", "cd /c && ls /c/src"),
    ("cat '/app/a b' \"/app\";ls /app", "cat '/c/a b' \"/c\";ls /c"),
    ("ls /apple /srv/app ~/app $HOME/app /app_data", "ls /apple /srv/app ~/app $HOME/app /app_data"),
    ("ls /c/x /app:/app", "ls /c/x /c:/c"),
])
def test_remap_rewrites_whole_workdir_paths_in_commands(command, expected):
    assert remap_args("bash", {"command": command}, "/app", "/c")["command"] == expected


def test_remap_paths_and_run_tool_note(env, tmp_path):
    assert remap_args("read_file", {"path": "/app/x"}, "/app", "/c") == {"path": "/c/x"}
    assert remap_args("read_file", {"path": "/apps/x"}, "/app", "/c") == {"path": "/apps/x"}
    assert remap_args("list_dir", {"path": "rel"}, "/app", "/c") == {"path": "rel"}
    work, copy = tmp_path / "work", tmp_path / "copy"
    work.mkdir(), copy.mkdir()
    c = ctx_for(env, remap=(str(work), str(copy)))
    out = bash(c, f"echo hi > {work}/f.txt")
    assert (copy / "f.txt").exists() and not (work / "f.txt").exists() and "redirected" in out.text


class DockerEnv:
    """docker exec with stderr merged into stdout, as harbor's docker backend runs commands."""

    def __init__(self, image: str):
        self.cid = subprocess.run(["docker", "run", "-d", "--rm", "--init", image, "sleep", "infinity"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        self.state_root, self.workdir, self.stop_event = "/tmp/.sb", "/root", threading.Event()

    def exec(self, command, timeout):
        return run_process(["docker", "exec", self.cid, "bash", "-c", "exec 2>&1\n" + command], timeout, self.stop_event)

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.cid], capture_output=True)


@pytest.mark.docker
@pytest.mark.parametrize("image", ["ubuntu:24.04", "debian:bookworm-slim"])
def test_shell_in_a_linux_container(image):
    if shutil.which("docker") is None:
        pytest.skip("no docker")
    d = DockerEnv(image)
    try:
        c = ctx_for(d, command_timeout_sec=2)
        assert bash(c, "cd /tmp && export X=1").exit_code == 0
        assert bash(c, "echo $PWD $X").text == "/tmp 1"
        out = bash(c, "(sleep 4; touch /tmp/late) & sleep 30")
        assert out.timed_out
        time.sleep(4)
        left = d.exec("ls /tmp/late 2>&1; for p in /proc/[0-9]*; do tr '\\0' ' ' < $p/cmdline; echo; done | grep -c '^sleep [0-9]'", 10)
        assert left.stdout.split()[-1] == "0" and "No such file" in left.stdout
        assert run_tool(c, "edit_file", {"path": "f", "old": "", "new": "x"}).text == "Wrote /tmp/f."
    finally:
        d.close()
