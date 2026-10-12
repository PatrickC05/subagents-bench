"""Tools on LocalShellEnvironment: per-agent persistent shell state, timeouts
(host-side kill and in-sandbox `timeout`), truncation, and every tool's
success and failure paths."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import textwrap
import threading
import time

import pytest

from subagents_bench import tools, workspace
from subagents_bench.environment import ExecResult, LocalShellEnvironment
from subagents_bench.tools import (
    ORCHESTRATOR_TOOLS,
    SCOPE_TOOLS,
    TOOL_FNS,
    TOOL_SPECS,
    run_tool,
    specs_for,
    truncate_output,
)
from subagents_bench.types import AgentStopped, ToolContext, ToolOutput, ToolSpec


def make_ctx(env, workdir, agent_id="sub-1", max_output_chars=4000, command_timeout_sec=20, max_command_timeout_sec=60):
    return ToolContext(
        env=env,
        agent_id=agent_id,
        initial_cwd=str(workdir),
        max_output_chars=max_output_chars,
        command_timeout_sec=command_timeout_sec,
        max_command_timeout_sec=max_command_timeout_sec,
    )


@pytest.fixture
def env(tmp_path):
    e = LocalShellEnvironment(str(tmp_path))
    yield e
    e.close()


@pytest.fixture
def ctx(env, tmp_path):
    return make_ctx(env, tmp_path)


def bash(ctx, command, **kw):
    return run_tool(ctx, "bash", {"command": command, **kw})


FAKE_TIMEOUT = f"""#!{sys.executable}
# GNU-compatible `timeout [-k K] T CMD...` for hosts without coreutils.
import os, signal, subprocess, sys
a = sys.argv[1:]
k = None
if a and a[0] == "-k":
    k = float(a[1]); a = a[2:]
t = float(a[0]); cmd = a[1:]
p = subprocess.Popen(cmd, start_new_session=True)
try:
    sys.exit(p.wait(timeout=t))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=k if k is not None else 1e9); sys.exit(124)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait(); sys.exit(137)
"""


@pytest.fixture
def timeout_env(tmp_path):
    """Env whose PATH has a `timeout`: the real one if the host has it, else a fake."""
    if shutil.which("timeout"):
        e = LocalShellEnvironment(str(tmp_path))
    else:
        fakebin = tmp_path / "_fakebin"
        fakebin.mkdir()
        (fakebin / "timeout").write_text(FAKE_TIMEOUT)
        (fakebin / "timeout").chmod(0o755)
        e = LocalShellEnvironment(str(tmp_path), env={"PATH": f"{fakebin}:{os.environ['PATH']}"})
    yield e
    e.close()


# ---------------------------------------------------------------------------
# registry and specs
# ---------------------------------------------------------------------------


def test_registry_shapes():
    assert ORCHESTRATOR_TOOLS == ["bash", "edit_file", "read_file", "list_dir", "search"]
    assert SCOPE_TOOLS == {
        "easy": ["read_file", "list_dir", "search"],
        "medium": ["read_file", "list_dir", "search", "bash"],
        "hard": ["read_file", "list_dir", "search", "bash", "edit_file"],
    }
    assert set(TOOL_FNS) == set(TOOL_SPECS) == set(ORCHESTRATOR_TOOLS)
    for name, spec in TOOL_SPECS.items():
        assert isinstance(spec, ToolSpec) and spec.name == name
        params = spec.parameters
        assert params["type"] == "object"
        assert set(params["required"]) <= set(params["properties"])
        for prop in params["properties"].values():
            assert prop["type"] in ("string", "integer")
    assert TOOL_SPECS["bash"].parameters["required"] == ["command"]
    assert TOOL_SPECS["edit_file"].parameters["required"] == ["path", "old", "new"]
    assert TOOL_SPECS["list_dir"].parameters["required"] == []


def test_bash_description_is_honest_and_specs_for_fills_timeouts():
    d = TOOL_SPECS["bash"].description
    for phrase in ("working directory and exported environment variables persist", "no TTY", "nohup", "killed"):
        assert phrase in d
    s = specs_for(["read_file", "bash"], command_timeout_sec=180, max_command_timeout_sec=900)
    assert [x.name for x in s] == ["read_file", "bash"]
    assert "default 180s" in s[1].description and "up to 900s" in s[1].description


# ---------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------


def test_truncate_output_exact_counts():
    assert truncate_output("abc", 10) == ("abc", False)
    text = "H" * 600 + "M" * 3000 + "T" * 400
    out, truncated = truncate_output(text, 1000)
    assert truncated
    assert out == "H" * 500 + "\n[... 3000 characters elided ...]\n" + "M" * 100 + "T" * 400


def test_bash_truncation_marker_and_raw_count(env, tmp_path):
    c = make_ctx(env, tmp_path, max_output_chars=1000)
    o = bash(c, "head -c 50000 /dev/zero | tr '\\0' a")
    assert o.output_chars_raw == 50000
    assert o.truncated is True
    marker = "\n[... 49000 characters elided ...]\n"
    assert marker in o.text
    assert o.text.replace(marker, "") == "a" * 1000


def test_bash_huge_output_elided_in_sandbox_counts_stay_exact(env, tmp_path):
    c = make_ctx(env, tmp_path, max_output_chars=1000)
    o = bash(c, "head -c 3000000 /dev/zero | tr '\\0' b")
    assert o.output_chars_raw == 3_000_000
    assert "[... 2999000 characters elided ...]" in o.text
    assert o.text.count("b") == 1000


# ---------------------------------------------------------------------------
# persistent per-agent shell
# ---------------------------------------------------------------------------


def _bash_interpreters():
    found = []
    for b in (shutil.which("bash"), "/bin/bash"):
        if b and os.path.exists(b) and os.path.realpath(b) not in [os.path.realpath(x) for x in found]:
            found.append(b)
    return found


@pytest.mark.parametrize("bash_bin", _bash_interpreters())
def test_cwd_and_exports_persist(tmp_path, bash_bin):
    """Also run with macOS's /bin/bash 3.2 when present (portability)."""
    env = LocalShellEnvironment(str(tmp_path), bash=bash_bin)
    try:
        c = make_ctx(env, tmp_path)
        o1 = bash(c, "mkdir -p 'my dir/sub' && cd 'my dir/sub' && export FOO='bar baz' && LOCALVAR=1")
        assert o1.exit_code == 0 and not o1.is_error
        assert o1.cwd_before == str(tmp_path)
        assert o1.cwd_after == str(tmp_path / "my dir" / "sub")
        o2 = bash(c, 'pwd; echo "FOO=$FOO"; echo "LOCAL=${LOCALVAR:-unset}"')
        assert o2.text.splitlines() == [str(tmp_path / "my dir" / "sub"), "FOO=bar baz", "LOCAL=unset"]
        assert o2.cwd_before == o2.cwd_after == str(tmp_path / "my dir" / "sub")
    finally:
        env.close()


def test_agents_have_independent_state(env, tmp_path):
    a = make_ctx(env, tmp_path, agent_id="sub-1")
    b = make_ctx(env, tmp_path, agent_id="sub-2")
    (tmp_path / "x").mkdir()
    bash(a, "cd x && export WHO=a")
    ob = bash(b, 'pwd; echo "WHO=${WHO:-none}"')
    assert ob.text.splitlines() == [str(tmp_path), "WHO=none"]
    oa = bash(a, 'pwd; echo "WHO=$WHO"')
    assert oa.text.splitlines() == [str(tmp_path / "x"), "WHO=a"]
    assert os.path.isdir(os.path.join(env.state_root, "sub-1"))
    assert os.path.isdir(os.path.join(env.state_root, "sub-2"))


def test_initial_cwd_and_missing_cwd_fallback(env, tmp_path):
    (tmp_path / "start").mkdir()
    c = make_ctx(env, tmp_path / "start", agent_id="cand0")
    assert bash(c, "pwd").text == str(tmp_path / "start")
    bash(c, "mkdir gone && cd gone")
    shutil.rmtree(tmp_path / "start" / "gone")
    o = bash(c, "pwd")
    assert "does not exist" in o.text
    assert o.text.splitlines()[-1] == str(tmp_path / "start")


def test_exit_codes_and_state_saved_on_exit(env, tmp_path, ctx):
    (tmp_path / "d").mkdir()
    o = bash(ctx, "cd d; echo hi; exit 3")
    assert o.exit_code == 3
    assert o.text == "hi\n[exit code: 3]"
    assert o.is_error is False and o.timed_out is False
    assert bash(ctx, "pwd").text == str(tmp_path / "d")
    o = bash(ctx, "false")
    assert o.exit_code == 1 and o.text == "(no output)\n[exit code: 1]"
    o = bash(ctx, "echo ok")
    assert o.exit_code == 0 and o.text == "ok"


def test_heredoc_quotes_and_stderr_interleaved(ctx):
    cmd = textwrap.dedent(
        """\
        cat <<'EOF'
        literal $HOME `id` "dq" 'sq' \\n
        EOF
        echo to-stderr >&2
        echo after"""
    )
    o = bash(ctx, cmd)
    assert o.text == "literal $HOME `id` \"dq\" 'sq' \\n\nto-stderr\nafter"


def test_stdin_is_closed(ctx):
    o = bash(ctx, "cat; echo rc=$?", timeout_sec=5)
    assert o.timed_out is False
    assert o.text == "rc=0"


def test_background_job_does_not_hang_the_call(ctx, tmp_path):
    t0 = time.monotonic()
    o = bash(ctx, f"sleep 20 & echo $! > {tmp_path}/bg.pid; echo started")
    try:
        assert time.monotonic() - t0 < 5
        assert o.text == "started"
    finally:
        os.kill(int((tmp_path / "bg.pid").read_text()), 9)


def test_broken_path_does_not_break_state_saving(ctx, tmp_path):
    (tmp_path / "keep").mkdir()
    o = bash(ctx, "cd keep && export PATH=/nonexistent")
    assert o.cwd_after == str(tmp_path / "keep")
    o = bash(ctx, "echo $PATH; pwd")
    assert o.text.splitlines() == ["/nonexistent", str(tmp_path / "keep")]


def test_shell_options_in_command_do_not_break_saving(ctx, tmp_path):
    (tmp_path / "opt").mkdir()
    o = bash(ctx, "set -eu; cd opt; false; echo unreachable")
    assert o.exit_code == 1 and "unreachable" not in o.text
    assert o.cwd_after == str(tmp_path / "opt")
    o = bash(ctx, "set -x; cd ..; echo traced")
    assert o.exit_code == 0 and o.cwd_after == str(tmp_path)
    assert o.text == "++ cd ..\n++ echo traced\ntraced"  # the save trap is not traced


def test_concurrent_agents_on_one_env(env, tmp_path):
    results = {}

    def work(i):
        c = make_ctx(env, tmp_path, agent_id=f"sub-{i}")
        bash(c, f"mkdir -p d{i} && cd d{i} && export ME={i}")
        results[i] = bash(c, 'echo "$(basename "$PWD") $ME"').text

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert results == {i: f"d{i} {i}" for i in range(6)}


def test_large_command_is_uploaded_in_chunks(ctx, monkeypatch):
    from subagents_bench import environment

    monkeypatch.setattr(environment, "B64_CHUNK_CHARS", 500)
    payload = "x" * 3000
    o = bash(ctx, f"echo {payload} | wc -c")
    assert o.text.strip() == "3001"
    assert o.n_exec > 2  # chunk uploads counted


# ---------------------------------------------------------------------------
# timeouts
# ---------------------------------------------------------------------------


def test_timeout_without_timeout_binary_host_kill(env, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "HOST_TIMEOUT_GRACE_SEC", 1)
    workspace.set_timeout_binary(env, False)
    c = make_ctx(env, tmp_path)
    flag = tmp_path / "flag"
    t0 = time.monotonic()
    o = bash(c, f"echo before; (sleep 4; touch {flag}) & sleep 30", timeout_sec=1)
    elapsed = time.monotonic() - t0
    assert elapsed < 5  # T + grace + follow-up
    assert o.timed_out is True
    assert o.exit_code is None  # no exit status observed
    assert o.timeout_sec == 1
    assert o.text == "before\n[command timed out after 1s; it was killed]"
    assert o.n_exec == 2  # main exec + follow-up (probe was pre-set)
    assert o.cwd_after == str(tmp_path)
    time.sleep(4.5)
    assert not flag.exists(), "child process survived the timeout"


def test_timeout_with_timeout_binary_kills_in_sandbox(timeout_env, tmp_path, monkeypatch):
    assert workspace.has_timeout_binary(timeout_env) is True
    c = make_ctx(timeout_env, tmp_path)
    flag = tmp_path / "flag2"
    t0 = time.monotonic()
    o = bash(c, f"cd /; echo before; (sleep 3; touch {flag}) & sleep 30", timeout_sec=1)
    elapsed = time.monotonic() - t0
    assert elapsed < 4
    assert o.timed_out is True
    assert o.exit_code in (124, 137)
    assert o.text == "before\n[command timed out after 1s; it was killed]"
    assert o.n_exec == 1
    # the EXIT trap ran on SIGTERM, so the cd was persisted
    assert o.cwd_after == "/"
    assert bash(c, "pwd").text == "/"
    time.sleep(3.5)
    assert not flag.exists()


def test_quick_exit_124_is_not_a_timeout(timeout_env, tmp_path):
    c = make_ctx(timeout_env, tmp_path)
    o = bash(c, "exit 124", timeout_sec=30)
    assert o.exit_code == 124
    assert o.timed_out is False
    assert "[exit code: 124]" in o.text


def test_timeout_clamping_and_validation(ctx):
    o = bash(ctx, "echo hi", timeout_sec=10_000)
    assert o.timeout_sec == 60 and "[timeout_sec capped at 60s]" in o.text
    o = bash(ctx, "echo hi", timeout_sec=0)
    assert o.timeout_sec == 1
    o = bash(ctx, "echo hi")
    assert o.timeout_sec == 20
    o = bash(ctx, "echo hi", timeout_sec="7")
    assert o.timeout_sec == 7 and o.text == "hi"
    o = bash(ctx, "echo hi", timeout_sec="soon")
    assert o.is_error and "timeout_sec" in o.text and o.n_exec == 0


def test_timeout_probe_is_cached_and_counted(tmp_path):
    env = LocalShellEnvironment(str(tmp_path))
    calls = []
    orig = env.exec

    def counting(command, timeout_sec=None, cwd=None):
        calls.append(command)
        return orig(command, timeout_sec=timeout_sec, cwd=cwd)

    env.exec = counting
    c = make_ctx(env, tmp_path)
    o1 = bash(c, "echo 1")
    o2 = bash(c, "echo 2")
    assert o1.n_exec == 2 and o2.n_exec == 1
    assert sum("command -v timeout" in x for x in calls) == 1
    assert o1.exec_sec > 0
    env.close()


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_numbered_and_ranges(ctx, tmp_path):
    (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    o = run_tool(ctx, "read_file", {"path": "f.txt"})
    assert not o.is_error and o.exit_code == 0
    assert o.text.splitlines()[0] == "     1\tline1"
    assert o.text.splitlines()[-1] == "    10\tline10"
    assert "[" not in o.text  # whole file, no notes
    o = run_tool(ctx, "read_file", {"path": "f.txt", "start_line": 3, "end_line": 5})
    assert o.text.splitlines() == ["     3\tline3", "     4\tline4", "     5\tline5", "[lines 3-5 of 10]"]
    o = run_tool(ctx, "read_file", {"path": "f.txt", "start_line": 9})
    assert o.text.splitlines() == ["     9\tline9", "    10\tline10", "[lines 9-10 of 10]"]
    o = run_tool(ctx, "read_file", {"path": "f.txt", "start_line": 50})
    assert o.text == f"[{tmp_path}/f.txt has 10 lines; start_line 50 is past the end]"
    o = run_tool(ctx, "read_file", {"path": "f.txt", "start_line": 5, "end_line": 2})
    assert o.is_error and "before start_line" in o.text


def test_read_file_last_line_without_newline_and_empty(ctx, tmp_path):
    (tmp_path / "n.txt").write_text("a\nb")
    o = run_tool(ctx, "read_file", {"path": "n.txt"})
    assert o.text == "     1\ta\n     2\tb"
    (tmp_path / "e.txt").write_text("")
    o = run_tool(ctx, "read_file", {"path": "e.txt"})
    assert o.text == f"({tmp_path}/e.txt is empty)" and not o.is_error


def test_read_file_caps_and_continuation_hint(env, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "READ_MAX_LINES", 50)
    (tmp_path / "big.txt").write_text("".join(f"{i}\n" for i in range(1, 201)))
    c = make_ctx(env, tmp_path)
    o = run_tool(c, "read_file", {"path": "big.txt"})
    assert o.truncated
    assert o.text.splitlines()[-1] == "[showing lines 1-50 of 200 (line limit); call read_file with start_line=51 to continue]"
    c2 = make_ctx(env, tmp_path, max_output_chars=100)
    o = run_tool(c2, "read_file", {"path": "big.txt"})
    assert "(output size limit)" in o.text and o.truncated
    body = o.text.rsplit("\n", 1)[0]
    assert len(body) <= 100


def test_read_file_errors_and_binary(ctx, tmp_path):
    o = run_tool(ctx, "read_file", {"path": "nope.txt"})
    assert o.is_error and o.text == f"[error] file not found: {tmp_path}/nope.txt"
    (tmp_path / "dir").mkdir()
    o = run_tool(ctx, "read_file", {"path": "dir"})
    assert o.is_error and "is a directory" in o.text
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02abc")
    o = run_tool(ctx, "read_file", {"path": "b.bin"})
    assert not o.is_error and "looks binary (6 bytes)" in o.text
    o = run_tool(ctx, "read_file", {})
    assert o.is_error and "missing required argument 'path'" in o.text


def test_read_file_relative_to_agent_cwd(ctx, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "here.txt").write_text("in sub\n")
    bash(ctx, "cd sub")
    o = run_tool(ctx, "read_file", {"path": "here.txt"})
    assert o.text == "     1\tin sub"
    # read-only tools do not move the persisted cwd
    assert o.cwd_before == o.cwd_after == str(tmp_path / "sub")


def test_read_file_long_line_clipped(ctx, tmp_path):
    (tmp_path / "long.txt").write_text("x" * 5000 + "\n")
    o = run_tool(ctx, "read_file", {"path": "long.txt"})
    assert "[... line truncated, 5000 chars total]" in o.text


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


def test_list_dir_depth_and_markers(ctx, tmp_path):
    (tmp_path / "a b" / "deep" / "deeper").mkdir(parents=True)
    (tmp_path / "a b" / "file 1.txt").write_text("x")
    (tmp_path / "a b" / "deep" / "deeper" / "z.txt").write_text("x")
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / ".git" / "HEAD").write_text("ref")
    (tmp_path / "top.py").write_text("x")
    o = run_tool(ctx, "list_dir", {})
    lines = o.text.splitlines()
    assert lines[0] == f"{tmp_path} (depth 2; 5 entries)"
    # sorted; .git listed but not expanded; nothing below depth 2
    assert lines[1:-1] == [".git/", "a b/", "a b/deep/", "a b/file 1.txt", "top.py"]
    assert lines[-1] == "[contents of .git, node_modules, __pycache__ are not expanded]"
    o = run_tool(ctx, "list_dir", {"path": "a b", "depth": 1})
    assert o.text.splitlines()[1:3] == ["deep/", "file 1.txt"]
    o = run_tool(ctx, "list_dir", {"path": ".", "depth": 99})
    assert "(depth 5;" in o.text and "a b/deep/deeper/z.txt" in o.text


def test_list_dir_errors_and_empty(ctx, tmp_path):
    o = run_tool(ctx, "list_dir", {"path": "missing"})
    assert o.is_error and "no such directory" in o.text
    (tmp_path / "f").write_text("x")
    o = run_tool(ctx, "list_dir", {"path": "f"})
    assert o.is_error and "not a directory" in o.text
    (tmp_path / "empty").mkdir()
    o = run_tool(ctx, "list_dir", {"path": "empty"})
    assert o.text.endswith("(empty directory)")


def test_list_dir_entry_cap(env, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LIST_MAX_ENTRIES", 5)
    for i in range(12):
        (tmp_path / f"f{i:02d}").write_text("")
    o = run_tool(make_ctx(env, tmp_path), "list_dir", {"depth": 1})
    assert o.truncated
    assert "(depth 1; 12 entries, showing 5)" in o.text
    assert "[... 7 more entries not shown" in o.text


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.fixture
def grep_ctx(tmp_path):
    """PATH without ripgrep, so the grep fallback is exercised."""
    env = LocalShellEnvironment(str(tmp_path), env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"})
    if env.exec("command -v rg").return_code == 0:
        pytest.skip("rg is on /usr/bin; cannot exercise the grep fallback")
    yield make_ctx(env, tmp_path)
    env.close()


def test_search_grep_fallback(grep_ctx, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "src" / "b.txt").write_text("foo bar\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x").write_text("foo in git\n")
    o = run_tool(grep_ctx, "search", {"pattern": "fo+"})
    assert not o.is_error and o.exit_code == 0
    assert sorted(o.text.splitlines()) == ["./src/a.py:1:def foo():", "./src/b.txt:1:foo bar"]
    o = run_tool(grep_ctx, "search", {"pattern": "foo", "glob": "*.py"})
    assert o.text == "./src/a.py:1:def foo():"
    o = run_tool(grep_ctx, "search", {"pattern": "foo", "path": "src/b.txt"})
    assert o.text == "src/b.txt:1:foo bar"


def test_search_limits_and_no_match_and_errors(grep_ctx, tmp_path):
    (tmp_path / "many.txt").write_text("hit\n" * 300)
    o = run_tool(grep_ctx, "search", {"pattern": "hit"})
    assert o.truncated
    assert len([ln for ln in o.text.splitlines() if ln.startswith("./many.txt:")]) == 200
    assert "[showing the first 200 of 300 matching lines" in o.text
    o = run_tool(grep_ctx, "search", {"pattern": "zzz_nothing"})
    assert o.text == "No matches found." and not o.is_error and o.exit_code == 1
    o = run_tool(grep_ctx, "search", {"pattern": "(unclosed"})
    assert o.is_error and o.exit_code == 2
    o = run_tool(grep_ctx, "search", {"pattern": "x", "path": "nowhere"})
    assert o.is_error and "no such file or directory" in o.text
    o = run_tool(grep_ctx, "search", {"pattern": ""})
    assert o.is_error


def test_search_prefers_rg_when_present(tmp_path):
    fakebin = tmp_path / "_bin"
    fakebin.mkdir()
    (fakebin / "rg").write_text('#!/bin/sh\necho "RG-ARGS: $*"\n')
    (fakebin / "rg").chmod(0o755)
    env = LocalShellEnvironment(str(tmp_path), env={"PATH": f"{fakebin}:/usr/bin:/bin"})
    try:
        o = run_tool(make_ctx(env, tmp_path), "search", {"pattern": "needle", "glob": "*.c"})
        assert o.text == "RG-ARGS: -n -H --no-heading --color never -g *.c -e needle -- ."
    finally:
        env.close()


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


def test_edit_unique_replacement_exact_bytes(ctx, tmp_path):
    p = tmp_path / "code.py"
    p.write_bytes(b"def f():\n    return 1\n\n\ndef g():\n    return 2")  # no trailing newline
    p.chmod(0o755)
    workspace.has_timeout_binary(ctx.env)  # warm the probe cache: n_exec counts only the edit
    o = run_tool(ctx, "edit_file", {"path": "code.py", "old": "    return 1\n", "new": "    return 10\n    # changed\n"})
    assert not o.is_error, o.text
    assert p.read_bytes() == b"def f():\n    return 10\n    # changed\n\n\ndef g():\n    return 2"
    assert o.text == f"Edited {p}: replaced 1 line(s) at line 2 with 2 line(s)."
    assert o.files_written == [str(p)]
    assert o.n_exec == 2 and o.exit_code is None
    assert stat.S_IMODE(p.stat().st_mode) == 0o755


def test_edit_ambiguous_missing_and_not_found(ctx, tmp_path):
    p = tmp_path / "dup.txt"
    p.write_text("x = 1\ny = 2\nx = 1\n")
    o = run_tool(ctx, "edit_file", {"path": "dup.txt", "old": "x = 1", "new": "x = 3"})
    assert o.is_error and "occurs 2 times" in o.text and "(at lines 1, 3)" in o.text
    assert p.read_text() == "x = 1\ny = 2\nx = 1\n"  # untouched
    o = run_tool(ctx, "edit_file", {"path": "dup.txt", "old": "z = 9", "new": "q"})
    assert o.is_error and "not found" in o.text
    o = run_tool(ctx, "edit_file", {"path": "missing.txt", "old": "a", "new": "b"})
    assert o.is_error and o.text == f"[error] file not found: {tmp_path}/missing.txt (to create it, pass an empty 'old')"
    assert o.files_written == []
    o = run_tool(ctx, "edit_file", {"path": "dup.txt", "new": "b"})
    assert o.is_error and "missing required argument 'old'" in o.text


def test_edit_crlf_hint(ctx, tmp_path):
    (tmp_path / "w.txt").write_bytes(b"a\r\nb\r\n")
    o = run_tool(ctx, "edit_file", {"path": "w.txt", "old": "a\nb", "new": "c"})
    assert o.is_error and "CRLF" in o.text


def test_edit_create_and_overwrite(ctx, tmp_path):
    o = run_tool(ctx, "edit_file", {"path": "new dir/new.txt", "old": "", "new": "hello\nworld"})
    p = tmp_path / "new dir" / "new.txt"
    assert not o.is_error
    assert p.read_bytes() == b"hello\nworld"
    assert o.text == f"Created {p} (11 bytes, 2 lines)."
    assert o.files_written == [str(p)]
    o = run_tool(ctx, "edit_file", {"path": "new dir/new.txt", "old": "", "new": "short\n"})
    assert o.text == f"Overwrote existing file {p} (11 -> 6 bytes, 1 lines)."
    assert p.read_text() == "short\n"
    (tmp_path / "adir").mkdir()
    o = run_tool(ctx, "edit_file", {"path": "adir", "old": "", "new": "x"})
    assert o.is_error and "is a directory" in o.text


def test_edit_relative_path_resolves_against_agent_cwd(ctx, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("A = 1\n")
    (tmp_path / "m.py").write_text("A = 1\n")
    bash(ctx, "cd pkg")
    o = run_tool(ctx, "edit_file", {"path": "m.py", "old": "A = 1", "new": "A = 2"})
    assert o.files_written == [str(tmp_path / "pkg" / "m.py")]
    assert (tmp_path / "pkg" / "m.py").read_text() == "A = 2\n"
    assert (tmp_path / "m.py").read_text() == "A = 1\n"
    o = run_tool(ctx, "edit_file", {"path": "../m.py", "old": "A = 1", "new": "A = 3"})
    assert o.files_written == [str(tmp_path / "m.py")]  # normalized
    assert (tmp_path / "m.py").read_text() == "A = 3\n"


def test_edit_non_utf8_file_preserved(ctx, tmp_path):
    p = tmp_path / "latin1.txt"
    p.write_bytes(b"caf\xe9\nkeep \xff\xfe\n")
    o = run_tool(ctx, "edit_file", {"path": "latin1.txt", "old": "keep", "new": "kept"})
    assert not o.is_error
    assert p.read_bytes() == b"caf\xe9\nkept \xff\xfe\n"


# ---------------------------------------------------------------------------
# run_tool error handling and stop
# ---------------------------------------------------------------------------


def test_run_tool_unknown_and_bad_args(ctx):
    o = run_tool(ctx, "teleport", {})
    assert o.is_error and "unknown tool 'teleport'" in o.text
    o = run_tool(ctx, "bash", {})
    assert o.is_error and o.error == "missing required argument 'command'"
    o = run_tool(ctx, "bash", "not a dict")  # type: ignore[arg-type]
    assert o.is_error and "JSON object" in o.text
    o = run_tool(ctx, "bash", {"command": "   "})
    assert o.is_error and "empty" in o.text
    o = run_tool(ctx, "read_file", {"path": ["a"]})
    assert o.is_error and "must be a string" in o.text


def test_run_tool_converts_env_exceptions(tmp_path):
    class Broken:
        workdir = str(tmp_path)
        stop_event = None
        state_root = str(tmp_path / ".sb")

        def exec(self, command, timeout_sec=None, cwd=None):
            raise RuntimeError("docker went away")

    env = Broken()
    workspace.set_timeout_binary(env, True)
    c = make_ctx(env, tmp_path)
    for name, args in [
        ("bash", {"command": "ls"}),
        ("read_file", {"path": "x"}),
        ("list_dir", {}),
        ("search", {"pattern": "x"}),
        ("edit_file", {"path": "x", "old": "", "new": "y"}),
    ]:
        o = run_tool(c, name, args)
        assert isinstance(o, ToolOutput)
        assert o.is_error and "RuntimeError: docker went away" in o.text, name
        assert o.n_exec == 1


def test_wrapper_failure_reported_as_error(tmp_path):
    class NoMeta:
        workdir = str(tmp_path)
        stop_event = None
        state_root = str(tmp_path / ".sb")

        def exec(self, command, timeout_sec=None, cwd=None):
            return ExecResult("[sb-error] cannot create state dir", "", 97)

    env = NoMeta()
    workspace.set_timeout_binary(env, False)
    o = run_tool(make_ctx(env, tmp_path), "bash", {"command": "ls"})
    assert o.is_error and "cannot create state dir" in o.text and o.exit_code == 97


def test_agent_stopped_propagates(env, ctx):
    env.stop_event = threading.Event()
    env.stop_event.set()
    for name, args in [("bash", {"command": "ls"}), ("read_file", {"path": "x"}), ("edit_file", {"path": "x", "old": "", "new": "y"})]:
        with pytest.raises(AgentStopped):
            run_tool(ctx, name, args)


def test_agent_stopped_during_long_command(env, ctx):
    workspace.set_timeout_binary(env, False)
    env.stop_event = threading.Event()
    threading.Timer(0.5, env.stop_event.set).start()
    t0 = time.monotonic()
    with pytest.raises(AgentStopped):
        bash(ctx, "sleep 30")
    assert time.monotonic() - t0 < 3
