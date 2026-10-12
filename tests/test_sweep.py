"""Sweep launcher against a fake ``harbor`` executable (tests/fake_harbor.py): resume, move-aside
retries, live trial lines, provider probe and early stop, billing abort, interrupt, preflight, CLI."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from fake_harbor import write_job, write_trial
from subagents_bench import cli, sweep
from subagents_bench.config import ConfigError, load_config

REPO = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).with_name("fake_harbor.py")
ORCH_MODEL, SUB_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash", "nvidia/Gemma-4-31B-IT-NVFP4"
TASKS = ["fix-git", "prove-plus-comm"]


class Fake:
    """The fake harbor's controls and call log, plus a stub for ``sweep.ping`` (no network)."""

    def __init__(self, tmp: Path):
        self.log, self.modes = tmp / "harbor_calls.jsonl", tmp / "harbor_modes.json"
        self.failing: dict[str, str] = {}  # model -> error the probe reports
        self.pinged: list[str] = []
        self.set()

    def set(self, **modes: str) -> None:
        self.modes.write_text(json.dumps(modes))

    def calls(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def jobs(self) -> list[str]:
        return [c["argv"][c["argv"].index("--job-name") + 1] for c in self.calls()]

    def ping(self, m) -> str | None:
        self.pinged.append(m.model)
        return self.failing.get(m.model)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {"harbor": f'exec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE))} "$@"',
                       "docker": "echo 28.3.2"}.items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    f = Fake(tmp_path)
    monkeypatch.setenv("FAKE_HARBOR_LOG", str(f.log))
    monkeypatch.setenv("FAKE_HARBOR_MODES", str(f.modes))
    monkeypatch.setattr(sweep, "POLL_SEC", 0.05)
    monkeypatch.setattr(sweep, "ping", f.ping)
    monkeypatch.setattr(sweep, "harbor_bin", lambda: str(bin_dir / "harbor"))  # never the venv's real harbor
    return f


def make_spec(tmp_path: Path, **over) -> Path:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("")
    shutil.copy(REPO / "configs/sail/base.yaml", root / "configs/base.yaml")
    spec = {"name": "t", "base": "configs/base.yaml", "dataset": "terminal-bench@2.0", "tasks": TASKS, "rounds": 1,
            "order_seed": 0, "harbor": {"n_concurrent": 4, "agent_timeout_multiplier": 4.0},
            "conditions": {"single": {"topology": "single"}, "par": {"subagent": {"strategy": "parallel"}}}, **over}
    (root / "spec.yaml").write_text(yaml.safe_dump(spec))
    return root / "spec.yaml"


@pytest.fixture
def spec(tmp_path):
    return sweep.load_spec(make_spec(tmp_path))


def go(spec, **kw) -> int:
    return sweep.run(spec, sweep.plan(spec, **kw))


def manifest(spec) -> list[dict]:
    return [json.loads(line) for line in (spec.jobs_dir / "sweep_manifest.jsonl").read_text().splitlines()]


def first_job(spec) -> str:
    return sweep.job_name(spec, *sweep.plan(spec)[0])


# --- the study's specs ---------------------------------------------------------------------------


def test_pilot_and_smoke_specs(capsys):
    pilot, smoke = sweep.load_spec(REPO / "configs/sweeps/pilot.yaml"), sweep.load_spec(REPO / "configs/sweeps/smoke.yaml")
    assert (len(pilot.tasks), pilot.rounds, len(sweep.plan(pilot))) == (10, 3, 45)
    assert (smoke.tasks, smoke.rounds, len(sweep.plan(smoke))) == (TASKS, 1, 15)
    assert list(smoke.configs) == list(pilot.configs) and len(pilot.configs) == 15
    assert smoke.jobs_dir == REPO / "jobs/smoke" and smoke.root == REPO
    c = pilot.configs
    assert c["single_matched"].topology == "single" and c["single_matched"].budget.orchestrator_steps == 40
    assert c["samemodel_parallel_hard"].subagent_model.model == ORCH_MODEL
    assert c["gemma31b_arbiter_medium"].subagent_model.model == SUB_MODEL
    assert (c["gemma31b_arbiter_medium"].subagent.strategy, c["gemma31b_arbiter_medium"].subagent.scope) == ("arbiter", "medium")
    assert c["manager_sequential_hard"].orchestrator_tools == ["read_file", "list_dir", "search"]
    assert c["gemma31b_parallel_hard"].orchestrator_tools == ["bash", "edit_file", "read_file", "list_dir", "search"]
    orders = [[cond for r, cond in sweep.plan(pilot) if r == k] for k in (1, 2, 3)]
    assert orders[0] != orders[1] and all(sorted(o) == sorted(c) for o in orders)
    assert sweep.plan(pilot, rounds=[2]) == [(2, cond) for cond in orders[1]]  # a round's order is fixed
    assert sweep.harbor_cmd(smoke, "smoke__r1__single_5step")[1:] == [
        "run", "-d", "terminal-bench@2.0", "-i", "fix-git", "-i", "prove-plus-comm", "-a", sweep.AGENT, "-e", "docker",
        "-n", "4", "-k", "1", "-o", "jobs/smoke", "--job-name", "smoke__r1__single_5step",
        "--agent-timeout-multiplier", "4.0", "-y", "-q"]
    for s in (pilot, smoke):
        assert sweep.run(s, sweep.plan(s), dry_run=True) == 0
        out = capsys.readouterr().out
        assert out.count(f"$ SUBAGENTS_BENCH_CONFIG=jobs/{s.name}/configs/") == len(sweep.plan(s))


def test_dry_run_runs_and_writes_nothing(fake, spec, capsys):
    assert sweep.run(spec, sweep.plan(spec, only=["par"]), dry_run=True) == 0
    out = capsys.readouterr().out
    assert "t__r1__par: run (no job dir)" in out and "t__r1__single" not in out
    assert fake.calls() == [] and not spec.jobs_dir.exists()


# --- running and resuming ------------------------------------------------------------------------


def test_sweep_runs_every_job_then_resume_skips_them(fake, spec):
    assert go(spec) == 0
    jobs = [sweep.job_name(spec, r, c) for r, c in sweep.plan(spec)]
    assert fake.jobs() == jobs
    for call, (_, cond) in zip(fake.calls(), sweep.plan(spec)):
        assert call["config"] == str(spec.jobs_dir / "configs" / f"{cond}.yaml") and call["cwd"] == str(spec.root)
        assert call["argv"][call["argv"].index("-o") + 1] == "jobs/t"
        assert load_config(call["config"]).to_dict() == spec.configs[cond].to_dict()
    status = json.loads((spec.jobs_dir / "sweep.json").read_text())
    assert status["status"] == "complete" and set(status["configs"]) == {"single", "par"} and "git_commit" in status
    assert status["spec"]["tasks"] == TASKS
    assert [m["event"] for m in manifest(spec)] == ["run", "run"]
    assert go(spec) == 0
    assert fake.jobs() == jobs  # nothing rerun
    assert [m["event"] for m in manifest(spec)][2:] == ["skipped", "skipped"]


def test_job_state_and_retry_reasons(tmp_path):
    job = tmp_path / "job"
    assert sweep.job_state(job) == ("absent", "no job dir")
    job.mkdir()
    write_job(job, 2, finished=False)
    write_trial(job, "a", 0)
    assert sweep.job_state(job) == ("unfinished", "1/2 trials finished")
    write_job(job, 2, finished=True)  # a finished job missing a trial result
    assert sweep.job_state(job)[0] == "unfinished"
    write_trial(job, "b", 1)
    assert sweep.job_state(job) == ("complete", "2 trials")
    expected = {"ok": None, "timeout": None, "unsolved": None, "provider": "provider", "billing": "billing",
                "crash": "harbor RuntimeError", "infra": "harbor EnvironmentStartTimeoutError"}
    for i, (mode, reason) in enumerate(expected.items()):
        assert sweep.retry_reason(write_trial(tmp_path / "trials", "x", i, mode)) == reason, mode
    ep = job / "a__t0" / "agent" / "episode.json"
    ep.write_text(json.dumps({"exit_reason": "finished", "n_provider_errors": 1}))  # a subagent's call failed
    assert sweep.job_state(job) == ("retry", "to rerun: a__t0 (provider)")


def test_unfinished_and_retry_jobs_are_moved_aside_and_rerun(fake, spec, capsys):
    unfinished = spec.jobs_dir / first_job(spec)
    unfinished.mkdir(parents=True)
    write_job(unfinished, 2, finished=False)
    write_trial(unfinished, "fix-git", 0)
    other = [j for j in (sweep.job_name(spec, 1, c) for c in ("single", "par")) if j != unfinished.name][0]
    fake.set(**{other: "crash"})
    assert go(spec) == 1  # the crashed trial's job needs a retry
    partial = spec.jobs_dir / "_partial"
    assert (partial / f"{unfinished.name}.1" / "fix-git__t0").is_dir()
    assert sweep.job_state(spec.jobs_dir / other)[0] == "retry"
    fake.set(**{other: "provider"})  # a transient provider failure: the models answer the probe
    assert go(spec) == 1
    out = capsys.readouterr().out
    assert "probing every model" in out and "going on" in out and sorted(fake.pinged) == [ORCH_MODEL, SUB_MODEL]
    fake.set()
    assert go(spec) == 0
    assert sorted(p.name for p in partial.iterdir()) == sorted([f"{unfinished.name}.1", f"{other}.1", f"{other}.2"])
    assert fake.jobs() == [unfinished.name, other, other, other]
    assert [m["event"] for m in manifest(spec) if m["job"] == other] == ["run", "moved", "run", "moved", "run"]


def test_agent_timeout_is_an_outcome(fake, spec):
    fake.set(**{first_job(spec): "timeout"})
    assert go(spec) == 0 and len(fake.calls()) == 2


def test_live_trial_lines_while_the_job_runs(fake, spec, monkeypatch):
    lines: list[tuple[float, str]] = []
    monkeypatch.setattr(sweep, "say", lambda line: lines.append((time.monotonic(), line)))
    fake.set(**{first_job(spec): "slow"})
    assert go(spec, only=[sweep.plan(spec)[0][1]]) == 0
    t_trial = next(t for t, line in lines if "trial fix-git__t0: reward 1.0, exit finished, 12s" in line)
    t_exit = next(t for t, line in lines if "harbor exit 0" in line)
    assert t_exit - t_trial > 0.5  # printed while harbor was still running
    assert sum("trial prove-plus-comm__t1" in line for _, line in lines) == 1


# --- aborts ---------------------------------------------------------------------------------------


def test_provider_outage_stops_the_job_and_exits_4(fake, spec, capsys):
    job = first_job(spec)
    fake.set(**{job: "outage"})
    fake.failing = {SUB_MODEL: "ProviderError: HTTP 503: no healthy upstream"}
    t0 = time.monotonic()
    saved = signal.signal(signal.SIGINT, signal.SIG_IGN)  # as for `sweep ... &` from a script: harbor inherits it
    try:
        assert go(spec) == 4
    finally:
        signal.signal(signal.SIGINT, saved)
    assert time.monotonic() - t0 < 20 and (spec.jobs_dir / job / "sigint").exists()  # harbor got SIGINT
    out = capsys.readouterr().out
    assert "trial fix-git__t0: reward None, exit provider_error" in out and "harbor ProviderUnavailableError" in out
    assert f"ABORT: provider failure in {job}/fix-git__t0; still failing: {SUB_MODEL}: ProviderError: HTTP 503" in out
    assert fake.jobs() == [job] and manifest(spec)[-1]["event"] == "abort" and manifest(spec)[-1]["code"] == 4
    assert json.loads((spec.jobs_dir / "sweep.json").read_text())["status"] == "aborted: provider outage"
    fake.set()
    fake.failing = {}
    assert go(spec) == 0 and (spec.jobs_dir / "_partial" / f"{job}.1").is_dir()


def test_harbor_exiting_before_the_job_dir_triggers_the_probe(fake, spec, capsys):
    job = first_job(spec)
    fake.set(**{job: "absent"})
    fake.failing = {ORCH_MODEL: "ProviderError: ConnectError: nodename nor servname provided"}
    assert go(spec) == 4
    assert f"ABORT: harbor could not start {job}; models failing too: {ORCH_MODEL}" in capsys.readouterr().out
    assert len(fake.calls()) == 1
    fake.failing = {}  # the models answer: the job counts as failed, the sweep goes on
    assert go(spec) == 1 and len(fake.calls()) == 3


def test_billing_error_exits_3(fake, spec, capsys):
    fake.set(**{first_job(spec): "billing"})
    assert go(spec) == 3
    assert "ABORT: billing error" in capsys.readouterr().out and len(fake.calls()) == 1 and fake.pinged == []


def test_interrupt_stops_harbor_and_exits_130(fake, spec):
    job = first_job(spec)
    fake.set(**{job: "hang"})
    saved = signal.signal(signal.SIGINT, signal.default_int_handler)  # also when pytest runs with it ignored
    threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGINT)).start()
    try:
        assert go(spec) == 130
    finally:
        signal.signal(signal.SIGINT, saved)
    assert (spec.jobs_dir / job / "sigint").exists() and manifest(spec)[-1]["event"] == "interrupted"
    assert fake.calls()[0]["pgid"] != os.getpgid(0)  # a terminal's Ctrl-C reaches harbor only via the sweep
    assert json.loads((spec.jobs_dir / "sweep.json").read_text())["status"] == "interrupted"


def test_sigterm_and_sighup_stop_the_sweep_like_ctrl_c(fake, tmp_path):
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)  # as under nohup
        assert cli.main(["sweep", str(make_spec(tmp_path)), "--dry-run"]) == 0
        assert signal.getsignal(signal.SIGTERM) is signal.default_int_handler  # harbor is not in our group
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)


# --- preflight, probe, CLI ------------------------------------------------------------------------


def test_preflight_checks_and_oracle(fake, spec, monkeypatch, capsys):
    monkeypatch.delenv("SAIL_API_KEY", raising=False)
    assert sweep.preflight(spec, oracle=True) == 1 and fake.calls() == []  # no oracle after a failed check
    assert "[FAIL] SAIL_API_KEY: missing" in capsys.readouterr().out
    monkeypatch.setenv("SAIL_API_KEY", "test-key")
    fake.set(_oracle="unsolved")
    assert sweep.preflight(spec, oracle=True) == 1
    out = capsys.readouterr().out
    assert "[PASS] docker: server 28.3.2" in out and "[PASS] harbor:" in out and "0.24.0" in out
    assert f"[PASS] request {ORCH_MODEL}" in out and f"[PASS] request {SUB_MODEL}" in out
    assert "[FAIL] oracle solves every task: unsolved: ['fix-git']" in out
    argv = fake.calls()[0]["argv"]
    assert argv[argv.index("-a") + 1] == "oracle" and "--agent-timeout-multiplier" not in argv
    fake.set()
    assert sweep.preflight(spec, oracle=True) == 0  # the failed oracle job is moved aside and rerun
    assert sweep.preflight(spec, oracle=True) == 0  # a passing one is reused
    assert len(fake.calls()) == 2 and (spec.jobs_dir / "_partial" / "_oracle.1").is_dir()
    fake.failing = {SUB_MODEL: "ProviderError: HTTP 401"}
    assert sweep.preflight(spec) == 1
    assert f"[FAIL] request {SUB_MODEL}: ProviderError: HTTP 401" in capsys.readouterr().out


def test_harbor_bin_prefers_the_venv(tmp_path, monkeypatch):
    venv, other = tmp_path / "venv", tmp_path / "other"
    for d in (venv, other):
        d.mkdir()
        (d / "harbor").write_text("#!/bin/sh\n")
        (d / "harbor").chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(venv / "python"))
    monkeypatch.setenv("PATH", str(other))
    assert sweep.harbor_bin() == str(venv / "harbor")  # not a global harbor that cannot import the agent
    (venv / "harbor").unlink()
    assert sweep.harbor_bin() == str(other / "harbor")


def test_probe_sends_one_short_request_per_model(spec, monkeypatch):
    sent = []

    class Client:
        def __init__(self, cfg):
            self.cfg = cfg

        def complete(self, system, messages, tools):
            sent.append(self.cfg)
            if self.cfg.model == SUB_MODEL:
                raise RuntimeError("boom")

        def close(self):
            pass

    monkeypatch.setattr(sweep, "SailClient", Client)
    assert sweep.probe(sweep.sweep_models(spec)) == [f"{SUB_MODEL}: RuntimeError: boom"]
    assert [(c.model, c.max_retries, c.request_timeout_sec, c.retry_budget_sec) for c in sent] == [
        (ORCH_MODEL, 1, 60.0, 30.0), (SUB_MODEL, 1, 60.0, 30.0)]


@pytest.mark.parametrize("over, msg", [
    ({"bogus": 1}, "unknown key(s): bogus"),
    ({"harbor": {"n_concurrent": 4, "agent_timeout_multiplier": 4.0, "env": "modal"}}, "harbor.env"),
    ({"rounds": 0}, "rounds an integer >= 1"),
    ({"conditions": {"a__b": {}}}, "invalid name(s) ['a__b']"),
    ({"conditions": {"x": {"subagent": {"strategy": "swarm"}}}}, "condition x: subagent.strategy"),
])
def test_bad_specs_are_config_errors(tmp_path, over, msg):
    with pytest.raises(ConfigError) as e:
        sweep.load_spec(make_spec(tmp_path, **over))
    assert msg in str(e.value)


def test_cli_sweep_and_preflight(fake, tmp_path, monkeypatch, capsys):
    path = str(make_spec(tmp_path))
    assert cli.main(["sweep", path, "--only", "nope"]) == 2
    assert cli.main(["sweep", path, "--rounds", "1,2"]) == 2
    assert "not in sweep 't': nope" in capsys.readouterr().err
    assert cli.main(["sweep", str(tmp_path / "missing.yaml")]) == 2
    assert cli.main(["sweep", path, "--dry-run", "--only", "par"]) == 0 and fake.calls() == []
    assert cli.main(["sweep", path, "--rounds", "1"]) == 0 and len(fake.calls()) == 2
    monkeypatch.setenv("SAIL_API_KEY", "test-key")
    assert cli.main(["preflight", path]) == 0
    r = subprocess.run([sys.executable, "-m", "subagents_bench.cli", "sweep", "--help"], capture_output=True,
                       text=True, cwd=REPO)
    assert r.returncode == 0 and "--dry-run" in r.stdout and "--only" in r.stdout and "--rounds" in r.stdout
