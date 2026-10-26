"""The shell scripts parse under /bin/bash (3.2 on macOS), print their help, and run_experiment.sh
calls preflight then sweep with the documented arguments."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ["setup.sh", "run_experiment.sh", "run_final.sh", "run_all_final.sh", "run_final_patrick.sh", "run_final_jonathan.sh"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses_and_prints_help(name):
    path = REPO / "scripts" / name
    assert subprocess.run(["/bin/bash", "-n", str(path)]).returncode == 0
    r = subprocess.run(["/bin/bash", str(path), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and f"scripts/{name}" in r.stdout and "set -euo" not in r.stdout


def run_experiment(tmp_path: Path, *args: str, **env: str) -> tuple[int, list[str]]:
    """Run a copy of run_experiment.sh whose subagents-bench and caffeinate only log their argv."""
    root, bin_dir, log = tmp_path / "repo", tmp_path / "bin", tmp_path / "calls.log"
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / ".venv/bin").mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(exist_ok=True)
    shutil.copy(REPO / "scripts/run_experiment.sh", root / "scripts")
    fakes = {root / ".venv/bin/subagents-bench":
             f'echo "subagents-bench $*" >> {log}; [ "$1" != preflight ] || exit "${{PREFLIGHT_RC:-0}}"',
             bin_dir / "caffeinate": f'echo "caffeinate $1" >> {log}; shift; exec "$@"'}
    for path, body in fakes.items():
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    log.write_text("")
    r = subprocess.run(["/bin/bash", str(root / "scripts/run_experiment.sh"), *args], cwd=tmp_path, text=True,
                       env={**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", **env})
    return r.returncode, [line.replace(str(root), "ROOT") for line in log.read_text().splitlines()]


def test_run_experiment_runs_preflight_then_sweep(tmp_path):
    assert run_experiment(tmp_path) == (0, [
        "subagents-bench preflight ROOT/configs/sweeps/pilot.yaml --oracle",
        "caffeinate -i",
        "subagents-bench sweep ROOT/configs/sweeps/pilot.yaml"])
    assert run_experiment(tmp_path, "configs/sweeps/smoke.yaml", "--only", "a", "b", SB_NO_CAFFEINATE="1") == (0, [
        "subagents-bench preflight ROOT/configs/sweeps/smoke.yaml --oracle",
        "subagents-bench sweep ROOT/configs/sweeps/smoke.yaml --only a b"])
    assert run_experiment(tmp_path, "--dry-run") == (0, ["subagents-bench sweep ROOT/configs/sweeps/pilot.yaml --dry-run"])
    assert run_experiment(tmp_path, "/abs/spec.yaml", "--rounds", "2", SKIP_PREFLIGHT="1") == (0, [
        "caffeinate -i", "subagents-bench sweep /abs/spec.yaml --rounds 2"])
    assert run_experiment(tmp_path, "--rounds", "3,4", "--only", "a", "b", "--job", "1:c", "--job", "2:a",
                          SB_NO_CAFFEINATE="1") == (0, [
        "subagents-bench preflight ROOT/configs/sweeps/pilot.yaml --oracle",
        "subagents-bench sweep ROOT/configs/sweeps/pilot.yaml --rounds 3,4 --only a b --job 1:c --job 2:a"])
    assert run_experiment(tmp_path, PREFLIGHT_RC="1") == (1, ["subagents-bench preflight ROOT/configs/sweeps/pilot.yaml --oracle"])


def test_parallel_forwards_ctrl_c_to_every_command(tmp_path):
    """scripts/parallel.sh: each command in its own process group, a Ctrl-C forwarded to all."""
    import signal
    import time
    child = tmp_path / "child.sh"
    child.write_text('trap \'echo "$1 got INT"; exit 130\' INT\necho "$1 running"\n'
                     'for i in $(seq 1 300); do sleep 0.1; done\necho "$1 done"\n')
    main = tmp_path / "main.sh"
    main.write_text(f"set -euo pipefail\n. {REPO / 'scripts' / 'parallel.sh'}\n"
                    f"par_start {tmp_path}/a.log bash {child} a\npar_start {tmp_path}/b.log bash {child} b\n"
                    "rc=0; par_wait || rc=$?\necho \"rc=$rc\"\n")
    p = subprocess.Popen(["/bin/bash", str(main)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL), start_new_session=True)
    for _ in range(100):  # until both children are running
        if all((tmp_path / f"{x}.log").exists() and "running" in (tmp_path / f"{x}.log").read_text() for x in "ab"):
            break
        time.sleep(0.1)
    p.send_signal(signal.SIGINT)  # the terminal's Ctrl-C reaches the foreground script only
    out, _ = p.communicate(timeout=20)
    assert "rc=130" in out and out.count("[FAIL] exit 130") == 2
    assert all("got INT" in (tmp_path / f"{x}.log").read_text() for x in "ab")
