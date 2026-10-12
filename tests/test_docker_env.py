"""Docker tests (opt-in: RUN_DOCKER=1): the per-agent wrapper, in-container
timeout kills, workspace helpers and byte-exact transfer inside real Linux
containers, plus a tool-availability probe of a Terminal-Bench image.

    RUN_DOCKER=1 .venv/bin/python -m pytest tests/test_docker_env.py -q -s
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time

import pytest

from subagents_bench import tools, workspace
from subagents_bench.environment import LocalDockerEnvironment
from subagents_bench.tools import run_tool
from subagents_bench.types import ToolContext

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(os.environ.get("RUN_DOCKER") != "1", reason="set RUN_DOCKER=1 to run Docker tests"),
]

IMAGE = "ubuntu:24.04"
TB_IMAGE = "alexgshaw/fix-git:20251031"


def _docker_ok() -> bool:
    return shutil.which("docker") is not None and subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _have_image(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def denv():
    if not _docker_ok():
        pytest.skip("docker daemon not available")
    if not _have_image(IMAGE):
        subprocess.run(["docker", "pull", IMAGE], check=True, capture_output=True)
    with LocalDockerEnvironment(IMAGE, workdir="/app") as e:
        yield e


def make_ctx(env, agent_id="sub-1", initial_cwd="/app", max_output_chars=4000):
    return ToolContext(
        env=env,
        agent_id=agent_id,
        initial_cwd=initial_cwd,
        max_output_chars=max_output_chars,
        command_timeout_sec=60,
        max_command_timeout_sec=120,
    )


def alive(env, cmdline: str) -> bool:
    """Is a process with exactly this command line running? (Exact match, so
    the probe's own `bash -c` line, which contains the text, never matches.)"""
    r = env.exec(
        "for p in /proc/[0-9]*; do c=$(tr '\\000' ' ' < $p/cmdline 2>/dev/null); "
        f'[ "$c" = "{cmdline} " ] && echo FOUND; done; true'
    )
    return "FOUND" in r.stdout


# ---------------------------------------------------------------------------


def test_probes(denv):
    assert workspace.has_timeout_binary(denv) is True
    assert workspace.container_arch(denv) in ("aarch64", "x86_64")
    assert workspace.resolve_workdir(denv) == "/app"


def test_wrapper_persists_cwd_and_exports_per_agent(denv):
    a, b = make_ctx(denv, "sub-a"), make_ctx(denv, "sub-b")
    o = run_tool(a, "bash", {"command": "mkdir -p '/tmp/w dir' && cd '/tmp/w dir' && export TOKEN='x y'"})
    assert o.exit_code == 0 and o.cwd_before == "/app" and o.cwd_after == "/tmp/w dir"
    o = run_tool(a, "bash", {"command": 'pwd; echo "T=$TOKEN"'})
    assert o.text.splitlines() == ["/tmp/w dir", "T=x y"]
    o = run_tool(b, "bash", {"command": 'pwd; echo "T=${TOKEN:-none}"'})
    assert o.text.splitlines() == ["/app", "T=none"]
    r = denv.exec("ls /tmp/.sb")
    assert {"sub-a", "sub-b"} <= set(r.stdout.split())


def test_in_container_timeout_kills_process_tree(denv):
    c = make_ctx(denv, "sub-timeout")
    t0 = time.monotonic()
    o = run_tool(c, "bash", {"command": "cd /tmp; echo started; sleep 3171 & sleep 3172", "timeout_sec": 2})
    elapsed = time.monotonic() - t0
    assert o.timed_out is True
    assert o.exit_code in (124, 137)
    assert o.text == "started\n[command timed out after 2s; it was killed]"
    assert elapsed < 2 + tools.KILL_AFTER_SEC + 5
    assert o.cwd_after == "/tmp"  # EXIT trap ran on SIGTERM
    time.sleep(0.5)
    assert not alive(denv, "sleep 3171"), "background child survived"
    assert not alive(denv, "sleep 3172"), "foreground command survived"


def test_host_timeout_fallback_kills_via_proc(denv, monkeypatch):
    """No `timeout` binary: the host gives up after T+grace and the follow-up
    exec kills the wrapper's process tree inside the container."""
    monkeypatch.setattr(tools, "HOST_TIMEOUT_GRACE_SEC", 2)
    workspace.set_timeout_binary(denv, False)
    try:
        c = make_ctx(denv, "sub-hostkill")
        t0 = time.monotonic()
        o = run_tool(c, "bash", {"command": "echo partial; sleep 3173 & sleep 3174", "timeout_sec": 1})
        assert time.monotonic() - t0 < 10
        assert o.timed_out is True and o.exit_code is None
        assert o.text == "partial\n[command timed out after 1s; it was killed]"
        time.sleep(0.5)
        assert not alive(denv, "sleep 3173")
        assert not alive(denv, "sleep 3174")
    finally:
        workspace.set_timeout_binary(denv, None)


def test_background_server_survives_and_call_returns(denv):
    c = make_ctx(denv, "sub-bg")
    t0 = time.monotonic()
    o = run_tool(c, "bash", {"command": "nohup sleep 3175 > /tmp/srv.log 2>&1 & echo launched"})
    assert time.monotonic() - t0 < 10 and o.text == "launched"
    # without redirection the job holds the output file, not the exec pipe
    t0 = time.monotonic()
    o = run_tool(c, "bash", {"command": "sleep 3176 & echo also"})
    assert time.monotonic() - t0 < 10 and o.text == "also"
    assert alive(denv, "sleep 3175") and alive(denv, "sleep 3176")
    denv.exec("pkill -f 'sleep 317[56]' || true")


def test_large_payloads_cross_argv_limit(denv):
    data = os.urandom(400_000)  # base64 ~533 KB > 128 KiB MAX_ARG_STRLEN
    denv.write_bytes("/tmp/big.bin", data)
    r = denv.exec("md5sum /tmp/big.bin")
    assert r.stdout.split()[0] == hashlib.md5(data).hexdigest()
    assert denv.read_bytes("/tmp/big.bin") == data
    c = make_ctx(denv, "sub-big")
    body = "y" * 200_000
    o = run_tool(c, "bash", {"command": f"cat <<'EOF' | wc -c\n{body}\nEOF"})
    assert o.text.strip() == "200001"
    assert o.n_exec > 1


def test_tools_in_container(denv):
    c = make_ctx(denv, "sub-tools")
    run_tool(c, "bash", {"command": "mkdir -p /app/proj && cd /app/proj"})
    o = run_tool(c, "edit_file", {"path": "m.py", "old": "", "new": "def f():\n    return 1\n"})
    assert o.files_written == ["/app/proj/m.py"]
    o = run_tool(c, "edit_file", {"path": "m.py", "old": "return 1", "new": "return 2"})
    assert not o.is_error
    assert denv.read_file("/app/proj/m.py") == "def f():\n    return 2\n"
    o = run_tool(c, "read_file", {"path": "m.py", "start_line": 2})
    assert o.text.splitlines()[0] == "     2\t    return 2"
    o = run_tool(c, "list_dir", {"path": "/app"})
    assert "proj/" in o.text.splitlines() and "proj/m.py" in o.text.splitlines()
    o = run_tool(c, "search", {"pattern": "return [0-9]", "path": "/app"})
    assert o.text == "/app/proj/m.py:2:    return 2"


def test_workspace_helpers_in_container(denv):
    setup = r"""
set -e
rm -rf /app/ws && mkdir -p '/app/ws/src/sub dir' '/app/ws/src/gone dir'
printf 'a\n' > /app/ws/src/a.py
printf 'one\n' > '/app/ws/src/sub dir/file one.txt'
printf 'x\n' > '/app/ws/src/gone dir/x.txt'
printf '#!/bin/sh\n' > /app/ws/plain.sh
printf 'keep\n' > /app/ws/keep.txt
touch -d '2001-01-01' /app/ws/src/a.py '/app/ws/src/sub dir/file one.txt' '/app/ws/src/gone dir/x.txt' /app/ws/plain.sh /app/ws/keep.txt
"""
    assert denv.exec(setup).return_code == 0
    marker = workspace.make_marker(denv, "grp-1")
    assert marker == "/tmp/.sb/markers/grp-1"
    denv.exec("printf 'A\\n' > /app/ws/src/a.py && printf n > '/app/ws/new file'")
    assert workspace.files_modified_since(denv, marker, "/app/ws") == (["/app/ws/new file", "/app/ws/src/a.py"], 2)
    # a scan rooted at / prunes /tmp/.sb (where the marker itself lives)
    res = workspace.files_modified_since(denv, marker, "/")
    assert res is not None and not any(p.startswith("/tmp/.sb") for p in res[0])

    cand = "/tmp/.sb/arb/arb-1/cand0"
    assert workspace.copy_tree(denv, "/app/ws", cand)
    assert workspace.manifest(denv, "/app/ws", include_mode=True) == workspace.manifest(denv, cand, include_mode=True)
    assert workspace.dir_size_mb(denv, "/app/ws") is not None
    mutate = f"""
set -e
cd {cand}
printf 'AA\\n' > src/a.py
printf 'brand\\n' > 'src/sub dir/brand new.txt'
rm -rf 'src/gone dir'
chmod +x plain.sh
"""
    assert denv.exec(mutate).return_code == 0
    summary = workspace.diff_summary(denv, "/app/ws", cand)
    assert summary.splitlines()[0] == "changed files: 2 modified, 1 added, 1 deleted"
    assert "+AA" in summary
    applied, err = workspace.apply_copy(denv, cand, "/app/ws")
    assert err is None
    assert applied == sorted(
        ["/app/ws/src/a.py", "/app/ws/src/sub dir/brand new.txt", "/app/ws/src/gone dir/x.txt", "/app/ws/plain.sh"]
    )
    assert workspace.manifest(denv, "/app/ws", include_mode=True) == workspace.manifest(denv, cand, include_mode=True)
    assert denv.exec("test -e '/app/ws/src/gone dir'").return_code == 1
    if denv.exec("command -v git").return_code != 0:
        assert workspace.git_diff_stat(denv, "/app/ws") is None


def test_terminal_bench_image_tool_availability():
    """Probe a real Terminal-Bench image (amd64; emulated on Apple Silicon)."""
    if not _docker_ok():
        pytest.skip("docker daemon not available")
    if not _have_image(TB_IMAGE):
        pytest.skip(f"{TB_IMAGE} not pulled")
    with LocalDockerEnvironment(TB_IMAGE, workdir=None, platform="linux/amd64") as e:
        bins = "base64 timeout find xargs md5sum diff git rg grep tr head tail wc du cp readlink ps".split()
        r = e.exec("for b in " + " ".join(bins) + '; do if command -v $b >/dev/null 2>&1; then echo "$b yes"; else echo "$b no"; fi; done')
        avail = dict(line.split() for line in r.stdout.splitlines())
        print(f"\n{TB_IMAGE}: arch={workspace.container_arch(e)} workdir={workspace.resolve_workdir(e)} {avail}")
        for required in ("base64", "find", "xargs", "md5sum", "tr", "head", "tail", "wc", "cp"):
            assert avail[required] == "yes", required
        assert workspace.has_timeout_binary(e) == (avail["timeout"] == "yes")
        wd = workspace.resolve_workdir(e)
        c = make_ctx(e, "probe", initial_cwd=wd)
        o = run_tool(c, "bash", {"command": "cd / && echo ok"})
        assert o.text == "ok" and o.cwd_after == "/"
        o = run_tool(c, "bash", {"command": "sleep 30", "timeout_sec": 2})
        assert o.timed_out
