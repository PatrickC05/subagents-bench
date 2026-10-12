"""Stand-in for the ``harbor`` executable in the sweep tests: writes harbor-0.24-shaped job dirs.

Each call is appended to ``$FAKE_HARBOR_LOG``; ``$FAKE_HARBOR_MODES`` (JSON ``{job: mode}``) picks
how a job goes (default ``ok``). The first trial of a job gets the mode, the others are ``ok``:
ok, timeout (agent timeout: an outcome), provider, billing, crash, infra, unsolved (reward 0).
Whole-job modes: absent (exit 1 before creating the job dir), slow (1 s between trials),
outage (a provider failure, then hang until SIGINT), hang (no trial, hang until SIGINT).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

T0, T1 = "2026-10-12T17:39:20Z", "2026-10-12T17:40:05Z"


def write_trial(job_dir: Path, task: str, i: int, mode: str = "ok") -> Path:
    trial = job_dir / f"{task}__t{i}"
    ep = {"exit_reason": "finished", "wall_clock_sec": 12.0, "n_provider_errors": 0, "error": None}
    exc, verified, reward = None, True, 1.0
    if mode == "timeout":
        ep["exit_reason"], exc, reward = "cancelled", "AgentTimeoutError", 0.0
    elif mode in ("provider", "outage"):
        ep.update(exit_reason="provider_error", n_provider_errors=1, error="ProviderError: giving up after 5 attempts")
        exc, verified = "ProviderUnavailableError", False
    elif mode == "billing":
        ep.update(exit_reason="billing_error", error="BillingError: HTTP 402: insufficient balance")
        exc, verified = "BillingError", False
    elif mode == "crash":
        ep["exit_reason"], exc, verified = "error", "RuntimeError", False
    elif mode == "infra":
        ep, exc, verified = None, "EnvironmentStartTimeoutError", False
    elif mode == "unsolved":
        reward = 0.0
    (trial / "agent").mkdir(parents=True)
    if ep is not None:
        (trial / "agent" / "episode.json").write_text(json.dumps(ep))
    res = {"task_name": task, "trial_name": trial.name, "started_at": T0, "finished_at": T1,
           "exception_info": {"exception_type": exc, "exception_message": "x", "occurred_at": T1} if exc else None,
           "verifier_result": {"rewards": {"reward": reward}} if verified else None,
           "verifier": {"started_at": T0, "finished_at": T1} if verified else None}
    (trial / "result.json").write_text(json.dumps(res))
    return trial


def write_job(job_dir: Path, n_total: int, finished: bool) -> None:
    (job_dir / "result.json").write_text(json.dumps({"n_total_trials": n_total, "finished_at": T1 if finished else None}))


def main(args: list[str]) -> int:
    if args == ["--version"]:
        print("0.24.0")
        return 0
    with open(os.environ["FAKE_HARBOR_LOG"], "a") as f:
        f.write(json.dumps({"argv": args, "config": os.environ.get("SUBAGENTS_BENCH_CONFIG"), "cwd": os.getcwd(),
                            "pgid": os.getpgid(0)}) + "\n")
    tasks = [args[i + 1] for i, a in enumerate(args) if a == "-i"]
    job = args[args.index("--job-name") + 1]
    mode = json.loads(Path(os.environ["FAKE_HARBOR_MODES"]).read_text()).get(job, "ok")
    if mode == "absent":
        print("could not resolve dataset terminal-bench@2.0", file=sys.stderr)
        return 1
    job_dir = Path(args[args.index("-o") + 1]) / job
    job_dir.mkdir(parents=True)  # fails if the sweep reuses a job dir instead of moving it aside
    write_job(job_dir, len(tasks), finished=False)
    try:
        if mode == "hang":
            time.sleep(30)
        for i, task in enumerate(tasks):
            write_trial(job_dir, task, i, mode if i == 0 else "ok")
            if mode in ("slow", "outage"):
                time.sleep(30 if mode == "outage" else 1)
    except KeyboardInterrupt:
        (job_dir / "sigint").touch()
        return 130
    write_job(job_dir, len(tasks), finished=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
