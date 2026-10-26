"""Sweep launcher against a fake ``harbor`` executable (tests/fake_harbor.py): resume, move-aside
retries, live trial lines, provider probe and early stop, billing abort, interrupt, preflight, CLI."""

from __future__ import annotations

import json
import os
import random
import re
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
from subagents_bench.hints import n_slots

REPO = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).with_name("fake_harbor.py")
ORCH_MODEL, SUB_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash", "nvidia/Gemma-4-31B-IT-NVFP4"
TASKS = ["fix-git", "prove-plus-comm"]


class Fake:
    """The fake harbor's controls and call log, plus a stub for ``sweep.ping`` (no network)."""

    def __init__(self, tmp: Path):
        self.log, self.modes = tmp / "harbor_calls.jsonl", tmp / "harbor_modes.json"
        self.times = tmp / "harbor_times.jsonl"  # start/end of each harbor call
        self.failing: dict[str, str] = {}  # model -> error the probe reports
        self.pinged: list[str] = []
        self.set()

    def set(self, **modes: str) -> None:
        self.modes.write_text(json.dumps(modes))

    def calls(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def jobs(self) -> list[str]:
        return [c["argv"][c["argv"].index("--job-name") + 1] for c in self.calls()]

    def most_at_once(self) -> int:
        """The most harbor processes alive at the same time."""
        events = sorted((e["t"], e["event"] == "start") for e in map(json.loads, self.times.read_text().splitlines()))
        n = most = 0
        for _, start in events:  # an end sorts before a start at the same time
            n += 1 if start else -1
            most = max(most, n)
        return most

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
    monkeypatch.setenv("FAKE_HARBOR_TIMES", str(f.times))
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
        "--agent-timeout-multiplier", "2.0", "-y", "-q"]
    for s in (pilot, smoke):
        assert sweep.run(s, sweep.plan(s), dry_run=True) == 0
        out = capsys.readouterr().out
        assert out.count(f"$ SUBAGENTS_BENCH_CONFIG=jobs/{s.name}/configs/") == len(sweep.plan(s))


def test_hint_specs_match_the_pilot_but_for_the_hint():
    pilot = sweep.load_spec(REPO / "configs/sweeps/pilot.yaml")
    hinted, medium = (sweep.load_spec(REPO / f"configs/sweeps/{n}.yaml") for n in ("smoke_hinted", "hint_medium"))
    assert (hinted.tasks, hinted.rounds, len(sweep.plan(hinted))) == (TASKS, 1, 10)
    assert set(medium.tasks) < set(pilot.tasks) and (medium.rounds, len(sweep.plan(medium))) == (2, 4)
    for s in (hinted, medium):
        assert (s.n_concurrent, s.timeout_multiplier) == (pilot.n_concurrent, pilot.timeout_multiplier)
        for cond, cfg in s.configs.items():
            twin = cond.replace("hinted_", "gemma31b_", 1) if "samemodel" not in cond else "samemodel_parallel_hard"
            assert cfg.subagent.delegation_hint == cond.startswith("hinted_") and not pilot.configs[twin].subagent.delegation_hint
            cfg.subagent.delegation_hint = False
            assert {**cfg.to_dict(), "name": twin} == pilot.configs[twin].to_dict()


def test_swe_sweep_spec(capsys):
    s, pilot = sweep.load_spec(REPO / "configs/sweeps/swe_sweep.yaml"), sweep.load_spec(REPO / "configs/sweeps/pilot.yaml")
    assert s.dataset is None and s.dataset_path == REPO / "../class-6-swe-sweep/swe-sweep/tasks"
    assert len(s.tasks) == 10 and all(t.startswith("swe-sweep__") for t in s.tasks) and len(sweep.plan(s)) == 45
    assert list(s.configs) == list(pilot.configs)
    for cond, cfg in s.configs.items():
        assert cfg.budget.orchestrator_steps == (40 if cond == "single_matched" else 5), cond
        assert (cfg.budget.command_timeout_sec, cfg.budget.max_command_timeout_sec) == (600, 1800)
    assert s.platform == "linux/amd64" and s.timeout_multiplier * 43200 == pytest.approx(3600, abs=5)
    cmd = sweep.harbor_cmd(s, "swe_sweep__r1__single_5step")
    assert cmd[1:4] == ["run", "-p", "jobs/swe_sweep/tasks"] and cmd[-2:] == ["-y", "-q"]  # the pinned copy
    assert cmd[cmd.index("--override-cpus") + 1] == "3" and cmd[cmd.index("--override-memory-mb") + 1] == "7000"
    oracle = sweep.harbor_cmd(s, "_oracle", agent="oracle")
    assert "--override-cpus" in oracle and "--agent-timeout-multiplier" not in oracle
    assert sweep.run(s, sweep.plan(s), dry_run=True) == 0
    assert capsys.readouterr().out.count("$ SUBAGENTS_BENCH_CONFIG=jobs/swe_sweep/configs/") == 45


def test_swe_hint_spec_matches_swe_sweep_but_for_the_hint():
    s, full = sweep.load_spec(REPO / "configs/sweeps/swe_hint.yaml"), sweep.load_spec(REPO / "configs/sweeps/swe_sweep.yaml")
    assert set(s.tasks) < set(full.tasks) and (s.rounds, len(sweep.plan(s))) == (2, 4)
    assert (s.dataset_path, s.platform, s.n_concurrent, s.timeout_multiplier, s.extra_args) == (
        full.dataset_path, full.platform, full.n_concurrent, full.timeout_multiplier, full.extra_args)
    for cond, cfg in s.configs.items():
        twin = cond.replace("hinted_", "gemma31b_", 1)
        assert cfg.subagent.delegation_hint == cond.startswith("hinted_") and not full.configs[twin].subagent.delegation_hint
        cfg.subagent.delegation_hint = False
        assert {**cfg.to_dict(), "name": twin} == full.configs[twin].to_dict()
    text = (REPO / "configs/sweeps/swe_hint.yaml").read_text()
    if s.dataset_path.is_dir():
        assert {t: int(re.search(rf"- {t} +# \w+, (\d+)\n", text)[1]) for t in s.tasks} == {
            t: sweep.n_subtasks(s.dataset_path / t) for t in s.tasks}


def test_swe_interactive_smoke_is_the_full_spec_shortened():
    """swe_interactive_smoke.yaml: swe_interactive.yaml's 4 conditions on 2 of its repositories x 1
    round, in 15-minute sessions with a hint every 3 minutes (5 hints), under a timeout that fits."""
    s, full = (sweep.load_spec(REPO / f"configs/sweeps/{n}.yaml") for n in ("swe_interactive_smoke", "swe_interactive"))
    assert set(s.tasks) < set(full.tasks) and (s.rounds, len(sweep.plan(s))) == (1, 4)
    assert (s.dataset_path, s.platform, s.n_concurrent, s.extra_args) == (
        full.dataset_path, full.platform, full.n_concurrent, full.extra_args)
    assert 900 + sweep.SESSION_GRACE_SEC <= 43200 * s.timeout_multiplier < 43200 * full.timeout_multiplier
    if s.dataset_path.is_dir():
        assert sweep.check_interactive(s) == []
    assert set(s.configs) == set(full.configs)
    for cond, cfg in s.configs.items():
        i, f = cfg.interactive, full.configs[cond].interactive
        assert (i.wall_clock_sec, i.interval_sec, n_slots(i)) == (900, 180, 5)
        i.wall_clock_sec, i.interval_sec = f.wall_clock_sec, f.interval_sec
        assert cfg.to_dict() == full.configs[cond].to_dict()


def test_swe_interactive_managers_smoke_differs_only_in_the_manager_conditions():
    """The managers have read-only tools and differ from each other only in the spawn mode; the
    parent's conditions are unchanged."""
    s, smoke = (sweep.load_spec(REPO / f"configs/sweeps/{n}.yaml") for n in ("swe_interactive_managers_smoke", "swe_interactive_smoke"))
    assert (s.tasks, s.rounds) == (smoke.tasks, smoke.rounds)
    assert set(s.configs) == set(smoke.configs) | {"manager_blocking", "manager_background"}
    for cond, cfg in smoke.configs.items():
        assert {**s.configs[cond].to_dict(), "name": cond} == cfg.to_dict()
    blocking, background = s.configs["manager_blocking"], s.configs["manager_background"]
    assert blocking.orchestrator_tools == background.orchestrator_tools == ["read_file", "list_dir", "search"]
    assert (blocking.subagent.spawn_mode, background.subagent.spawn_mode) == ("blocking", "background")
    blocking.subagent.spawn_mode = "background"
    assert {**blocking.to_dict(), "name": "manager_background"} == background.to_dict()
    twin = s.configs["background_interactive"]
    twin.tools.orchestrator_tools = background.tools.orchestrator_tools
    assert {**twin.to_dict(), "name": "manager_background"} == background.to_dict()


def _same_as_full_spec(s, full, keep):
    """``s`` runs swe_interactive.yaml's settings but for its tasks, rounds, session (``keep``: the
    interactive fields it sets itself) and conditions: each condition equals the full spec's twin."""
    assert (s.dataset_path, s.platform, s.n_concurrent, s.extra_args) == (
        full.dataset_path, full.platform, full.n_concurrent, full.extra_args)
    assert set(s.tasks) <= set(full.tasks) | {"swe-sweep__pint", "swe-sweep__isort", "swe-sweep__click"}
    if s.dataset_path.is_dir():
        assert sweep.check_interactive(s) == []
    single, background = full.configs["single_interactive"], full.configs["background_interactive"]
    for cond, cfg in s.configs.items():
        twin = single if cfg.topology == "single" else background
        assert cfg.subagent.spawn_mode == twin.subagent.spawn_mode
        d, t = cfg.to_dict(), twin.to_dict()
        for k in ("name", "tools"):
            d.pop(k), t.pop(k)
        for k in keep:
            d["interactive"].pop(k), t["interactive"].pop(k)
        assert d == t, cond


def test_swe_interactive_slow_spec():
    """Experiment 1: the single agent and a background manager, both told to test each fix, on two
    slow-suite repositories in 30-minute sessions (10 bugs, one every 3 minutes)."""
    s, full = (sweep.load_spec(REPO / f"configs/sweeps/{n}.yaml") for n in ("swe_interactive_slow", "swe_interactive"))
    assert (s.tasks, s.rounds, len(sweep.plan(s))) == (["swe-sweep__pint", "swe-sweep__isort"], 2, 4)
    assert 1800 + sweep.SESSION_GRACE_SEC <= 43200 * s.timeout_multiplier
    _same_as_full_spec(s, full, keep=("wall_clock_sec", "interval_sec", "session_note"))
    single, manager = s.configs["single_interactive"], s.configs["manager_background"]
    assert manager.orchestrator_tools == ["read_file", "list_dir", "search"] and manager.subagent.spawn_mode == "background"
    for cfg in (single, manager):
        i = cfg.interactive
        assert (i.wall_clock_sec, i.interval_sec, n_slots(i)) == (1800, 180, 10)
        assert "run the existing tests that cover the files you changed" in i.session_note
    assert single.interactive == manager.interactive


def test_swe_interactive_rate_spec():
    """Experiment 2: the same 10 bugs every 60, 180 or 300 s, each session 10 minutes past its last
    bug, for the single agent and a background manager; nothing else differs between intervals."""
    s, full = (sweep.load_spec(REPO / f"configs/sweeps/{n}.yaml") for n in ("swe_interactive_rate", "swe_interactive"))
    assert (s.tasks, s.rounds, len(sweep.plan(s))) == (["swe-sweep__click", "swe-sweep__pint"], 2, 12)
    assert 3600 + sweep.SESSION_GRACE_SEC <= 43200 * s.timeout_multiplier
    _same_as_full_spec(s, full, keep=("wall_clock_sec", "interval_sec", "max_hints"))
    for cond, cfg in s.configs.items():
        i = cfg.interactive
        every = int(cond.rsplit("every", 1)[1])
        assert (i.interval_sec, i.wall_clock_sec, i.max_hints) == (every, 10 * every + 600, 10)
        assert cfg.topology == ("single" if cond.startswith("single") else "subagents")
        if cfg.topology == "subagents":
            assert cfg.orchestrator_tools == ["read_file", "list_dir", "search"]
    for mode in ("single", "manager_background"):
        dicts = [s.configs[f"{mode}_every{k}"].to_dict() for k in (60, 180, 300)]
        for d in dicts:
            d.pop("name")
            d["interactive"].pop("interval_sec"), d["interactive"].pop("wall_clock_sec")
        assert dicts[0] == dicts[1] == dicts[2]


def test_swe_interactive_baseline_spec():
    """Baseline: the requests sweep's three request conditions (single, background manager, hinted
    background) with only the bugs (reply tool on; no questions, status checks or urgent bug), on
    click and echo, with one fixed bug order for every round and condition: the requests sweep's
    round 1 order. Every model is the orchestrator's (subagent.model: null), and subagents have no practical
    limit on how many run at once or how many steps each takes."""
    from subagents_bench.hints import schedule
    s, req = (sweep.load_spec(REPO / f"configs/sweeps/{n}.yaml") for n in ("swe_interactive_baseline", "swe_interactive_requests"))
    assert (s.tasks, s.rounds, len(sweep.plan(s))) == (["swe-sweep__click", "swe-sweep__echo"], 2, 6)
    assert (s.dataset_path, s.platform, s.extra_args, s.n_concurrent, s.timeout_multiplier) == (
        req.dataset_path, req.platform, req.extra_args, req.n_concurrent, req.timeout_multiplier)
    twins = {"single_baseline": "single_requests", "manager_background_baseline": "manager_background_requests",
             "background_hinted_baseline": "background_hinted_requests"}
    assert set(s.configs) == set(twins)
    for cond, twin in twins.items():
        d, t = s.configs[cond].to_dict(), req.configs[twin].to_dict()
        assert (d["interactive"].pop("vary_order_by_round"), t["interactive"].pop("vary_order_by_round")) == (False, True)
        for k, v in {"questions_per_interval": (0, 2), "status_every": (0, 2), "urgent_every": (0, 3)}.items():
            assert (d["interactive"].pop(k), t["interactive"].pop(k)) == v, k
        assert d["subagent"].pop("model") is None and t["subagent"].pop("model")["model"] == "nvidia/Gemma-4-31B-IT-NVFP4"
        assert (d["subagent"].pop("max_concurrent"), t["subagent"].pop("max_concurrent")) == (60, 3)
        assert (d["subagent"].pop("max_steps"), t["subagent"].pop("max_steps")) == (2000, 20)
        assert s.configs[cond].subagent_model == s.configs[cond].orchestrator  # all DeepSeek-V4.1-Flash
        assert {**d, "name": twin} == t, cond
        i = s.configs[cond].interactive
        assert (i.wall_clock_sec, i.interval_sec, i.max_hints, i.replies) == (1800, 240, 6, True)
    if s.dataset_path.is_dir():
        for t in s.tasks:
            orders = {tuple(h.subtask.id for h in schedule(cfg.interactive, s.dataset_path / t, t, round_=r))
                      for cfg in s.configs.values() for r in (1, 2)}
            assert len(orders) == 1 and len(next(iter(orders))) == 6, t
        click = [h.subtask.id for h in schedule(s.configs["single_baseline"].interactive, s.dataset_path / "swe-sweep__click",
                                                "swe-sweep__click", round_=2)]
        assert click == [h.subtask.id for h in schedule(req.configs["single_bugsonly"].interactive,
                                                        req.dataset_path / "swe-sweep__click", "swe-sweep__click", round_=1)]


def test_swe_interactive_requests_spec(tmp_path):
    """Requests: swe_interactive.yaml's settings but for the tasks, rounds, a 30-minute session with 6
    bugs every 4 minutes, the request stream (replies, 2 questions per interval, status every 2 bugs,
    bug 3 urgent) and 3 at once; its 3 request conditions are the full spec's single agent, a
    background manager, and the hinted background condition; 2 bugs-only controls are the single
    and manager request conditions without the requests."""
    from subagents_bench.hints import Question, write_questions
    from test_interactive import make_task
    s, full = (sweep.load_spec(REPO / f"configs/sweeps/{n}.yaml") for n in ("swe_interactive_requests", "swe_interactive"))
    assert (s.tasks, s.rounds, len(sweep.plan(s))) == (["swe-sweep__click", "swe-sweep__pint"], 2, 10)
    assert s.timeout_multiplier == 0.05
    assert 1800 + sweep.SESSION_GRACE_SEC <= 43200 * s.timeout_multiplier
    assert (s.dataset_path, s.platform, s.extra_args, s.n_concurrent) == (full.dataset_path, full.platform,
                                                                         full.extra_args, 3)
    requests = ("replies", "questions_per_interval", "status_every", "urgent_every")
    keep = ("wall_clock_sec", "interval_sec", "max_hints", *requests)
    twins = {"single_requests": "single_interactive", "manager_background_requests": "background_interactive",
             "background_hinted_requests": "background_hinted"}
    controls = {"single_bugsonly": "single_requests", "manager_background_bugsonly": "manager_background_requests"}
    assert set(s.configs) == set(twins) | set(controls)
    for cond in twins:
        cfg = s.configs[cond]
        i = cfg.interactive
        assert (i.wall_clock_sec, i.interval_sec, i.max_hints, i.replies, i.questions_per_interval, i.status_every,
                i.urgent_every) == (1800, 240, 6, True, 2, 2, 3)
        d, t = cfg.to_dict(), full.configs[twins[cond]].to_dict()
        for k in ("name", "tools"):
            d.pop(k), t.pop(k)
        for k in keep:
            d["interactive"].pop(k), t["interactive"].pop(k)
        assert d == t, cond
    # the controls: the same session as their request twin but for the request fields, all off
    for cond, twin in controls.items():
        i = s.configs[cond].interactive
        assert (i.wall_clock_sec, i.interval_sec, i.max_hints) == (1800, 240, 6)
        assert (i.replies, i.questions_per_interval, i.status_every, i.urgent_every) == (False, 0, 0, 0), cond
        d, t = s.configs[cond].to_dict(), s.configs[twin].to_dict()
        assert (d.pop("name"), t.pop("name")) == (cond, twin)
        for k in requests:
            d["interactive"].pop(k), t["interactive"].pop(k)
        assert d == t, cond
    assert s.configs["manager_background_requests"].orchestrator_tools == ["read_file", "list_dir", "search"]
    assert s.configs["manager_background_bugsonly"].orchestrator_tools == ["read_file", "list_dir", "search"]
    assert (s.configs["single_requests"].orchestrator_tools == s.configs["background_hinted_requests"].orchestrator_tools
            == full.configs["background_hinted"].orchestrator_tools)
    assert s.configs["background_hinted_requests"].subagent.delegation_hint
    a, b, c = (s.configs[cond].interactive for cond in twins)
    assert a == b == c  # the same stream for every request condition
    assert s.configs["single_bugsonly"].interactive == s.configs["manager_background_bugsonly"].interactive
    assert sweep.questions_needed(s) == 12
    # the stream its comments describe, on a 13-bug task
    task = make_task(tmp_path, 13)
    write_questions(tmp_path / "hints", "task-1", "m", [Question(f"q{k}", "f.py", f"Q{k}?", "A") for k in range(12)])
    sched = sweep.schedule(s.configs["single_requests"].interactive, task, "task-1", tmp_path / "hints")
    bugs = [h for h in sched if h.kind == "bug"]
    assert [h.t_offset for h in bugs] == [240 * k for k in range(6)] and [h.index for h in bugs if h.urgent] == [3]
    assert bugs[-1].t_offset == 1200 and 1800 - bugs[-1].t_offset == 600  # the last at 20:00, then 10 minutes
    assert [h.t_offset for h in sched if h.kind == "status"] == [360, 840, 1320]
    assert [h.t_offset % 240 for h in sched if h.kind == "question"] == [80, 160] * 6
    # the controls: the same bugs at the same times, nothing else, no ids or urgency (no hints dir needed)
    for cond in controls:
        plain = sweep.schedule(s.configs[cond].interactive, task, "task-1", None)
        assert [(h.kind, h.t_offset, h.subtask.id) for h in plain] == [("bug", h.t_offset, h.subtask.id) for h in bugs]
        assert all(h.msg_id is None and not h.urgent for h in plain), cond
    if s.dataset_path.is_dir() and all(sweep.hints_dir(s).joinpath(f"{t}.questions.json").is_file() for t in s.tasks):
        assert sweep.check_interactive(s) == []


def test_swe_sweep_larger_budget_hint(tmp_path, monkeypatch):
    """swe_sweep.yaml's larger step budget, made as its comments say (orchestrator_steps and the
    commented conditions block): every condition gets it, single_matched it x 8 subagent steps."""
    text = (REPO / "configs/sweeps/swe_sweep.yaml").read_text()
    assert text.count("\n    orchestrator_steps: 5 ") == 1
    text = text.replace("\n    orchestrator_steps: 5 ", "\n    orchestrator_steps: 30 ")
    text = text.replace("\n# conditions:", "\nconditions:").replace("\n#   single_matched:", "\n  single_matched:")
    (tmp_path / "b30.yaml").write_text(text)
    monkeypatch.chdir(REPO)  # no pyproject.toml above tmp_path: the spec's paths resolve against the cwd
    s = sweep.load_spec(tmp_path / "b30.yaml")
    steps = {c: cfg.budget.orchestrator_steps for c, cfg in s.configs.items()}
    assert steps.pop("single_matched") == 30 * s.configs["gemma31b_parallel_hard"].subagent.max_steps
    assert set(steps.values()) == {30} and len(steps) == 14


def test_swe_sweep_bug_counts_in_the_docs():
    """README's bug counts are the spec's per-task comments, which are the tasks' own (if cloned)."""
    s = sweep.load_spec(REPO / "configs/sweeps/swe_sweep.yaml")
    text = (REPO / "configs/sweeps/swe_sweep.yaml").read_text()
    bugs = {t: int(re.search(rf"- {t} +# \w+, (\d+)\n", text)[1]) for t in s.tasks}
    readme = " ".join((REPO / "README.md").read_text().split())
    assert f"{min(bugs.values())}-{max(bugs.values())} bugs each, {sum(bugs.values())} in all" in readme
    if s.dataset_path.is_dir():
        assert bugs == {t: sweep.n_subtasks(s.dataset_path / t) for t in s.tasks}


DOCKERFILE = "FROM docker.io/library/alpine:3.20 AS src\nRUN true\nFROM --platform=$BUILDPLATFORM golang:1.20\n"


def make_tasks(root: Path) -> None:
    """TASKS as local task dirs, the first with SWE-sweep's tests/swesweep_eval.json (3 bugs)."""
    for t in TASKS:
        for d in ("environment", "tests"):
            (root / t / d).mkdir(parents=True)
            (root / t / d / "Dockerfile").write_text(DOCKERFILE)
        (root / t / "task.toml").write_text("")
        (root / t / "tests" / "from.py").write_text("FROM x\n")  # not a Dockerfile: untouched
    (root / TASKS[0] / "tests" / "swesweep_eval.json").write_text(json.dumps({"subtasks": [{"id": i} for i in "abc"]}))


def test_local_dataset_platform_and_overrides(fake, tmp_path, monkeypatch, capsys):
    harbor = {"n_concurrent": 2, "agent_timeout_multiplier": 0.1, "extra_args": ["--override-cpus", "3"]}
    path = make_spec(tmp_path, dataset=None, dataset_path="tasks", platform="linux/amd64", harbor=harbor,
                     overrides={"budget": {"orchestrator_steps": 7}})
    make_tasks(path.parent / "tasks")
    spec = sweep.load_spec(path)
    assert spec.dataset is None and spec.dataset_path == path.parent / "tasks" and spec.platform == "linux/amd64"
    assert {c.budget.orchestrator_steps for c in spec.configs.values()} == {7}
    assert sweep.run(spec, sweep.plan(spec), dry_run=True) == 0 and not spec.jobs_dir.exists()
    assert "$ SUBAGENTS_BENCH_CONFIG=jobs/t/configs/" in capsys.readouterr().out
    assert go(spec) == 0
    for call in fake.calls():
        argv = call["argv"]
        assert argv[:3] == ["run", "-p", "jobs/t/tasks"] and "-d" not in argv
        assert argv[argv.index("--override-cpus") + 1] == "3" and argv[-2:] == ["-y", "-q"]
    staged = spec.jobs_dir / "tasks" / TASKS[0]
    pinned = ("FROM --platform=linux/amd64 docker.io/library/alpine:3.20 AS src\nRUN true\n"
              "FROM --platform=$BUILDPLATFORM golang:1.20\n")
    assert (staged / "environment" / "Dockerfile").read_text() == (staged / "tests" / "Dockerfile").read_text() == pinned
    assert (staged / "tests" / "from.py").read_text() == "FROM x\n" and (staged / "task.toml").is_file()
    manifest = json.loads((spec.jobs_dir / "tasks.json").read_text())
    assert manifest["platform"] == "linux/amd64" and manifest["source"] == str(path.parent / "tasks")
    assert (manifest["git_commit"], manifest["git_dirty"]) == (None, None) and sorted(manifest["tasks"]) == sorted(TASKS)
    assert manifest["n_subtasks"] == {TASKS[0]: 3, TASKS[1]: None}
    assert sweep.pin_platform(pinned, "linux/amd64") == pinned
    assert go(spec) == 0 and json.loads((spec.jobs_dir / "tasks.json").read_text()) == manifest  # a rerun: identical
    monkeypatch.setenv("SAIL_API_KEY", "test-key")
    shutil.rmtree(spec.jobs_dir / "tasks")
    assert sweep.preflight(spec, oracle=True) == 0  # the oracle stages the copy too
    out = capsys.readouterr().out
    assert f"[PASS] dataset_path: 2 tasks under {path.parent / 'tasks'}" in out and "oracle: $ " in out
    assert "--override-cpus" in fake.calls()[-1]["argv"] and (staged / "tests" / "Dockerfile").read_text() == pinned
    (path.parent / "tasks" / TASKS[1] / "task.toml").unlink()
    assert sweep.preflight(spec) == 1 and "[FAIL] dataset_path: no task.toml under" in capsys.readouterr().out


def test_local_dataset_without_platform_records_its_tasks(fake, tmp_path):
    """No copy: harbor runs the source task dirs, and tasks.json still records them and their bugs."""
    path = make_spec(tmp_path, dataset=None, dataset_path="tasks")
    make_tasks(path.parent / "tasks")
    spec = sweep.load_spec(path)
    assert go(spec) == 0 and not (spec.jobs_dir / "tasks").exists()
    assert all(c["argv"][:3] == ["run", "-p", "tasks"] for c in fake.calls())
    manifest = json.loads((spec.jobs_dir / "tasks.json").read_text())
    assert (manifest["source"], manifest["platform"]) == (str(path.parent / "tasks"), None)
    assert sorted(manifest["tasks"]) == sorted(TASKS) and manifest["n_subtasks"] == {TASKS[0]: 3, TASKS[1]: None}


def test_dry_run_runs_and_writes_nothing(fake, spec, capsys):
    assert sweep.run(spec, sweep.plan(spec, only=["par"]), dry_run=True) == 0
    out = capsys.readouterr().out
    assert "t__r1__par: run (no job dir)" in out and "t__r1__single" not in out
    assert fake.calls() == [] and not spec.jobs_dir.exists()


# --- running and resuming ------------------------------------------------------------------------


def test_sweep_runs_every_job_then_resume_skips_them(fake, spec, capsys):
    assert go(spec) == 0
    assert capsys.readouterr().out.endswith("Sweep 't' complete\nNext: scripts/analyze.sh jobs/t\n")
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
                "crash": "harbor RuntimeError", "infra": "harbor EnvironmentStartTimeoutError",
                "partial": None, "regressed": None, "grader": "grader", "verifier": "verifier", "shared_verifier": None}
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
    sweep.lock_jobs_dir(spec).close()  # the interrupted sweep released the jobs dir


# --- several jobs at once (--jobs-at-once) --------------------------------------------------------


@pytest.fixture
def spec4(tmp_path):
    """2 conditions x 2 rounds: 4 jobs, so with 2 at once two are queued behind the first two."""
    return sweep.load_spec(make_spec(tmp_path, rounds=2))


def names(spec) -> list[str]:
    return [sweep.job_name(spec, r, c) for r, c in sweep.plan(spec)]


def go2(spec, **kw) -> int:
    return sweep.run(spec, sweep.plan(spec, **kw), jobs_at_once=2)


def test_two_jobs_at_once_run_every_job_then_resume_skips_them(fake, spec4):
    jobs = names(spec4)
    fake.set(**{j: "slow" for j in jobs})  # ~2 s each: they overlap
    assert go2(spec4) == 0
    assert sorted(fake.jobs()) == sorted(jobs) and fake.most_at_once() == 2
    assert all(sweep.job_state(spec4.jobs_dir / j)[0] == "complete" for j in jobs)
    runs = manifest(spec4)
    assert [m["event"] for m in runs] == ["run"] * 4 and all(m["jobs_at_once"] == 2 for m in runs)
    assert sorted(m["job"] for m in runs) == sorted(jobs)
    assert json.loads((spec4.jobs_dir / "sweep.json").read_text())["status"] == "complete"
    assert go2(spec4) == 0
    assert len(fake.calls()) == 4 and [m["event"] for m in manifest(spec4)][4:] == ["skipped"] * 4


def test_sequential_run_records_one_job_at_once(fake, spec):
    assert go(spec) == 0 and fake.most_at_once() == 1
    assert [m["jobs_at_once"] for m in manifest(spec)] == [1, 1]


def test_provider_abort_in_one_job_stops_the_other_and_the_queue(fake, spec4, capsys):
    a, b, *queued = names(spec4)
    fake.set(**{a: "outage", b: "hang"})
    fake.failing = {SUB_MODEL: "ProviderError: HTTP 503: no healthy upstream"}
    t0 = time.monotonic()
    assert go2(spec4) == 4
    assert time.monotonic() - t0 < 20
    assert sorted(fake.jobs()) == sorted([a, b])  # the queued jobs never started
    assert (spec4.jobs_dir / a / "sigint").exists() and (spec4.jobs_dir / b / "sigint").exists()
    m = manifest(spec4)
    assert [(e["job"], e["code"]) for e in m if e["event"] == "abort"] == [(a, 4)]
    assert {e["job"] for e in m if e["event"] == "run"} == {a, b}
    assert not any(e["job"] in queued for e in m)
    assert json.loads((spec4.jobs_dir / "sweep.json").read_text())["status"] == "aborted: provider outage"
    out = capsys.readouterr().out
    assert f"ABORT: provider failure in {a}/fix-git__t0" in out and out.count("ABORT:") == 1
    fake.set()
    fake.failing = {}
    assert go2(spec4) == 0
    assert sorted(fake.jobs()) == sorted([a, b, *names(spec4)])  # the stopped two rerun, the queued ones run
    assert {p.name for p in (spec4.jobs_dir / "_partial").iterdir()} == {f"{a}.1", f"{b}.1"}
    assert all(sweep.job_state(spec4.jobs_dir / j)[0] == "complete" for j in names(spec4))


def test_billing_error_with_two_jobs_at_once_exits_3(fake, spec4, capsys):
    a, b, *_ = names(spec4)
    fake.set(**{a: "billing", b: "hang"})
    t0 = time.monotonic()
    assert go2(spec4) == 3
    assert time.monotonic() - t0 < 20
    assert sorted(fake.jobs()) == sorted([a, b]) and fake.pinged == []
    assert (spec4.jobs_dir / b / "sigint").exists()  # the other running job was stopped
    assert [(e["job"], e["code"]) for e in manifest(spec4) if e["event"] == "abort"] == [(a, 3)]
    assert json.loads((spec4.jobs_dir / "sweep.json").read_text())["status"] == "aborted: billing error"
    assert "ABORT: billing error" in capsys.readouterr().out


def test_interrupt_with_two_jobs_at_once_stops_both_harbors_and_exits_130(fake, spec4):
    a, b, *queued = names(spec4)
    fake.set(**{a: "hang", b: "hang"})

    def ctrl_c_once_both_run():
        deadline = time.monotonic() + 15
        while len(fake.calls()) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.3)  # both are in their poll loop
        os.kill(os.getpid(), signal.SIGINT)

    saved = signal.signal(signal.SIGINT, signal.default_int_handler)  # also when pytest runs with it ignored
    threading.Thread(target=ctrl_c_once_both_run, daemon=True).start()
    t0 = time.monotonic()
    try:
        assert go2(spec4) == 130
    finally:
        signal.signal(signal.SIGINT, saved)
    assert time.monotonic() - t0 < 20
    assert sorted(fake.jobs()) == sorted([a, b])
    assert (spec4.jobs_dir / a / "sigint").exists() and (spec4.jobs_dir / b / "sigint").exists()
    m = manifest(spec4)
    assert sorted(e["job"] for e in m if e["event"] == "interrupted") == sorted([a, b])
    assert not any(e["event"] in ("run", "abort") for e in m) and not any(e["job"] in queued for e in m)
    assert all(c["pgid"] != os.getpgid(0) for c in fake.calls())
    assert json.loads((spec4.jobs_dir / "sweep.json").read_text())["status"] == "interrupted"
    fake.set()
    assert go2(spec4) == 0  # resume: both are moved aside and rerun
    assert {p.name for p in (spec4.jobs_dir / "_partial").iterdir()} == {f"{a}.1", f"{b}.1"}


def test_cli_jobs_at_once(fake, tmp_path, capsys):
    path = str(make_spec(tmp_path))
    for bad in ("0", "-1"):
        assert cli.main(["sweep", path, "--jobs-at-once", bad]) == 2
        assert "--jobs-at-once must be at least 1" in capsys.readouterr().err
    assert fake.calls() == []
    assert cli.main(["sweep", path, "--jobs-at-once", "2"]) == 0 and len(fake.calls()) == 2
    spec = sweep.load_spec(Path(path))
    assert [m["jobs_at_once"] for m in manifest(spec)] == [2, 2]


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


# --- one sweep per jobs dir (.sweep.lock) ---------------------------------------------------------


def test_a_second_sweep_on_the_jobs_dir_exits_2_and_touches_nothing(fake, spec, monkeypatch, capsys):
    running = spec.jobs_dir / first_job(spec)  # the first sweep's job: a second sweep would move it aside
    running.mkdir(parents=True)
    write_job(running, 2, finished=False)
    held = sweep.lock_jobs_dir(spec)
    assert go(spec) == 2
    assert f"another sweep is running on jobs/t (pid {os.getpid()}: {shlex.join(sys.argv)})" in capsys.readouterr().out
    monkeypatch.setenv("SAIL_API_KEY", "test-key")
    assert sweep.preflight(spec, oracle=True) == 2  # its oracle job writes there too: refused before any check
    assert "another sweep is running on jobs/t" in capsys.readouterr().out
    assert sweep.preflight(spec) == 0  # without the oracle it only reads
    assert sweep.run(spec, sweep.plan(spec), dry_run=True) == 0  # so does a dry run
    assert fake.calls() == [] and sorted(p.name for p in spec.jobs_dir.iterdir()) == [".sweep.lock", running.name]
    held.close()
    assert go(spec) == 0 and len(fake.calls()) == 2  # released: the next sweep runs
    assert (spec.jobs_dir / "_partial" / f"{running.name}.1").is_dir()
    assert (spec.jobs_dir / ".sweep.lock").read_text() == f"pid {os.getpid()}: {shlex.join(sys.argv)}\n"
    sweep.lock_jobs_dir(spec).close()  # and it released the jobs dir when it returned


HOLD = ("import sys; from subagents_bench import sweep; lock = sweep.lock_jobs_dir(sweep.load_spec(sys.argv[1])); "
        "print('locked', flush=True); sys.stdin.read()")


def test_the_lock_holds_across_processes_until_the_holder_dies(fake, tmp_path, capsys):
    path = str(make_spec(tmp_path))
    holder = subprocess.Popen([sys.executable, "-c", HOLD, path], cwd=REPO, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline() == "locked\n"
        assert cli.main(["sweep", path]) == 2
        assert cli.main(["preflight", path, "--oracle"]) == 2
        out = capsys.readouterr().out
        assert out.count(f"another sweep is running on jobs/t (pid {holder.pid}: {shlex.join(['-c', path])})") == 2
        assert fake.calls() == []
    finally:
        holder.kill()  # SIGKILL: no cleanup runs in the holder, the OS releases its lock
        holder.wait()
    assert cli.main(["sweep", path]) == 0 and len(fake.calls()) == 2


def test_preflight_then_sweep_take_the_lock_in_turn(fake, tmp_path, monkeypatch):
    """run_experiment.sh runs `preflight --oracle`, then the sweep: each holds the jobs dir while it runs."""
    path = str(make_spec(tmp_path))
    monkeypatch.setenv("SAIL_API_KEY", "test-key")
    assert cli.main(["preflight", path, "--oracle"]) == 0 and cli.main(["sweep", path]) == 0
    assert fake.jobs()[0] == "_oracle" and len(fake.jobs()) == 3


# --- job selection (--rounds, --only, --job) ------------------------------------------------------


def plan_before_job(spec, only=None, rounds=None) -> list[tuple[int, str]]:
    """plan() as it was before --job, verbatim."""
    rng = random.Random(spec.order_seed)
    jobs = []
    for r in range(1, spec.rounds + 1):
        order = list(spec.configs)
        rng.shuffle(order)
        jobs += [(r, c) for c in order if (not rounds or r in rounds) and (not only or c in only)]
    return jobs


def test_plans_without_job_are_unchanged():
    for path in sorted((REPO / "configs/sweeps").glob("*.yaml")):
        s = sweep.load_spec(path)
        conds, rounds = list(s.configs), list(range(1, s.rounds + 1))
        for only, rs in [(None, None), *((None, [r]) for r in rounds), *(([c], None) for c in conds),
                         (conds[:2], [1]), (conds[-3:], rounds[-2:])]:
            assert sweep.plan(s, only, rs) == sweep.plan(s, only, rs, []) == plan_before_job(s, only, rs), (path, only, rs)


def test_job_selects_single_jobs(tmp_path):
    s = sweep.load_spec(make_spec(tmp_path, rounds=3))
    assert sweep.plan(s, jobs=[(3, "par")]) == [(3, "par")]  # alone: only these jobs
    assert sweep.plan(s, jobs=[(2, "par"), (1, "single"), (2, "par")]) == [(2, "par"), (1, "single")]  # in order, once
    r1 = sweep.plan(s, rounds=[1])
    assert sweep.plan(s, rounds=[1], jobs=[(3, "par"), r1[1], (3, "par")]) == r1 + [(3, "par")]  # the union
    single = sweep.plan(s, only=["single"])
    assert sweep.plan(s, only=["single"], jobs=[(2, "par"), (2, "single")]) == single + [(2, "par")]
    par23 = sweep.plan(s, only=["par"], rounds=[2, 3])
    assert sweep.plan(s, only=["par"], rounds=[2, 3], jobs=[(1, "single")]) == par23 + [(1, "single")]
    levels = sweep.load_spec(REPO / "configs/sweeps/swe_interactive_levels.yaml")
    l23 = [c for c in levels.configs if c.endswith(("_questions", "_urgent"))]
    reruns = [(1, "manager_background_baseline"), (2, "manager_background_questions")]
    assert sweep.plan(levels, l23, [3, 4], reruns) == sweep.plan(levels, l23, [3, 4]) + reruns


def test_bad_job_values(fake, tmp_path, capsys):
    path = str(make_spec(tmp_path))
    with pytest.raises(ConfigError, match="not in sweep 't': job 2:single, job 1:nope"):
        sweep.plan(sweep.load_spec(Path(path)), jobs=[(1, "par"), (2, "single"), (1, "nope")])
    for bad in ("2:single", "0:par", "1:nope", "1:Par"):
        assert cli.main(["sweep", path, "--job", bad]) == 2
        assert f"not in sweep 't': job {bad}" in capsys.readouterr().err
    assert cli.main(["sweep", path, "--only", "nope", "--job", "1:par"]) == 2  # --only is checked as before
    assert "not in sweep 't': nope" in capsys.readouterr().err
    for bad in ("1", ":par", "x:par", "1:", "1,2:par"):
        with pytest.raises(SystemExit) as e:
            cli.main(["sweep", path, "--job", bad])
        assert e.value.code == 2 and f"expected ROUND:CONDITION like 1:single (got {bad!r})" in capsys.readouterr().err
    assert fake.calls() == [] and not (tmp_path / "proj/jobs").exists()


def test_cli_job(fake, tmp_path, capsys):
    path = str(make_spec(tmp_path, rounds=2))
    s = sweep.load_spec(Path(path))
    r1 = [sweep.job_name(s, *j) for j in sweep.plan(s, rounds=[1])]
    assert cli.main(["sweep", path, "--dry-run", "--rounds", "1", "--job", "2:par", "--job", "1:par"]) == 0
    out = capsys.readouterr().out
    assert "Sweep 't': 3 jobs" in out and re.findall(r"\] (t__\S+):", out) == [*r1, "t__r2__par"]
    assert cli.main(["sweep", path, "--job", "2:par", "--job", "1:single"]) == 0
    assert fake.jobs() == ["t__r2__par", "t__r1__single"]
    assert cli.main(["sweep", path, "--only", "single", "--job", "2:par"]) == 0  # 1:single and 2:par: skipped
    assert fake.jobs() == ["t__r2__par", "t__r1__single", "t__r2__single"]
    r = subprocess.run([sys.executable, "-m", "subagents_bench.cli", "sweep", "--help"], capture_output=True,
                       text=True, cwd=REPO)
    assert "--job ROUND:CONDITION" in r.stdout


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

    monkeypatch.setattr(sweep, "ChatClient", Client)
    assert sweep.probe(sweep.sweep_models(spec)) == [f"{SUB_MODEL}: RuntimeError: boom"]
    assert [(c.model, c.max_retries, c.request_timeout_sec, c.retry_budget_sec) for c in sent] == [
        (ORCH_MODEL, 1, 60.0, 30.0), (SUB_MODEL, 1, 60.0, 30.0)]


@pytest.mark.parametrize("over, msg", [
    ({"bogus": 1}, "unknown key(s): bogus"),
    ({"harbor": {"n_concurrent": 4, "agent_timeout_multiplier": 4.0, "env": "modal"}}, "harbor.env"),
    ({"rounds": 0}, "rounds an integer >= 1"),
    ({"conditions": {"a__b": {}}}, "invalid name(s) ['a__b']"),
    ({"conditions": {"x": {"subagent": {"strategy": "swarm"}}}}, "condition x: subagent.strategy"),
    ({"dataset_path": "tasks"}, "exactly one of dataset"),
    ({"dataset": None}, "exactly one of dataset"),
    ({"overrides": {"budget": {"steps": 1}}}, "unknown key(s): budget.steps"),
    ({"platform": "linux/amd64"}, "platform must be a string like linux/amd64 and needs dataset_path"),
    ({"harbor": {"n_concurrent": 4, "agent_timeout_multiplier": 4.0, "extra_args": "-y"}}, "harbor.extra_args must be"),
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


def test_a_job_threads_error_stops_the_other_harbor_and_the_queue(fake, spec4, monkeypatch):
    """With two jobs at once, an ordinary error in one job's thread (here the job's first status
    line) stops the other running harbor, starts no queued job, records the error and re-raises."""
    a, b, *queued = names(spec4)
    fake.set(**{a: "hang", b: "hang"})
    real_say = sweep.say

    def say(msg):
        if f"] {b}: running" in msg:
            deadline = time.monotonic() + 15  # once a's harbor runs
            while not fake.calls() and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(0.3)
            raise BrokenPipeError("the log pipe closed")
        real_say(msg)

    monkeypatch.setattr(sweep, "say", say)
    t0 = time.monotonic()
    with pytest.raises(BrokenPipeError):
        go2(spec4)
    assert time.monotonic() - t0 < 20
    assert fake.jobs() == [a] and (spec4.jobs_dir / a / "sigint").exists()
    m = manifest(spec4)
    assert [e["event"] for e in m if e.get("job") in queued] == [] and m[-1]["event"] == "error"
    assert json.loads((spec4.jobs_dir / "sweep.json").read_text())["status"] == "error"
    sweep.lock_jobs_dir(spec4).close()  # released though run() raised


LEVELS = {"baseline": {}, "eta": {"status_every": 2}, "questions": {"status_every": 2, "questions_per_interval": 2},
          "urgent": {"status_every": 2, "questions_per_interval": 2, "urgent_every": 3}}
SETTINGS = ("single", "manager_background", "background_hinted")


def test_swe_interactive_levels_spec(tmp_path):
    """Levels: the baseline on Modal plus the request kinds added one at a time. Every condition is
    its setting's baseline condition with snapshots and require_eta on and only its level's request
    settings changed, and every condition, level and round sends the same bugs at the same times."""
    from subagents_bench.hints import Question, schedule, write_questions
    s = sweep.load_spec(REPO / "configs/sweeps/swe_interactive_levels.yaml")
    base = sweep.load_spec(REPO / "configs/sweeps/swe_interactive_baseline_modal.yaml")
    assert (s.environment, s.tasks, s.rounds) == ("modal", base.tasks, 4)
    assert (s.dataset_path, s.platform, s.extra_args, s.n_concurrent, s.timeout_multiplier) == (
        base.dataset_path, base.platform, base.extra_args, base.n_concurrent, base.timeout_multiplier)
    assert set(s.configs) == {f"{st}_{lv}" for st in SETTINGS for lv in LEVELS}
    for st in SETTINGS:
        twin = base.configs[f"{st}_baseline"].to_dict()
        for lv, knobs in LEVELS.items():
            d = s.configs[f"{st}_{lv}"].to_dict()
            assert d["interactive"].pop("snapshots") is True and d["interactive"].pop("require_eta") is True
            twin["interactive"].pop("snapshots", None), twin["interactive"].pop("require_eta", None)
            for k in ("questions_per_interval", "status_every", "urgent_every"):
                assert d["interactive"].pop(k) == knobs.get(k, 0), (st, lv, k)
                twin["interactive"].pop(k, None)
            assert {**d, "name": twin["name"]} == twin, (st, lv)
    rounds12, l23 = sweep.plan(s, rounds=[1, 2]), [c for c in s.configs if c.endswith(("_questions", "_urgent"))]
    assert len(rounds12) == 24 and len(sweep.plan(s, only=l23, rounds=[3, 4])) == 12
    if s.dataset_path.is_dir():
        for t in s.tasks:
            write_questions(tmp_path, t, "m", [Question(f"q{i}", "x.py", f"Q{i}?", "a") for i in range(12)])
            sends = {tuple((h.subtask.id, h.t_offset) for h in schedule(cfg.interactive, s.dataset_path / t, t, tmp_path, r)
                           if h.kind == "bug")
                     for cfg in s.configs.values() for r in (1, 2, 3, 4)}
            assert len(sends) == 1 and [o for _, o in next(iter(sends))] == [k * 240 for k in range(6)], t
            urgent = [h.index for h in schedule(s.configs["single_urgent"].interactive, s.dataset_path / t, t, tmp_path, 1)
                      if h.urgent]
            assert urgent == [3], t
    smoke = sweep.load_spec(REPO / "configs/sweeps/swe_interactive_levels_smoke.yaml")
    i = smoke.configs["single_urgent"].interactive
    assert (smoke.environment, smoke.rounds, len(smoke.configs), i.wall_clock_sec, i.interval_sec) == ("modal", 1, 12, 120, 30)
    assert 120 + sweep.SESSION_GRACE_SEC <= 43200 * smoke.timeout_multiplier


def test_harbor_environment_modal(monkeypatch):
    """`harbor.environment` picks harbor's -e (docker by default); on Modal the sweep sets Modal's
    image builder (an explicit setting wins) and preflight checks for a Modal token, not Docker."""
    base = sweep.load_spec(REPO / "configs/sweeps/swe_interactive_baseline.yaml")
    m = sweep.load_spec(REPO / "configs/sweeps/swe_interactive_baseline_modal.yaml")
    smoke = sweep.load_spec(REPO / "configs/sweeps/swe_interactive_baseline_modal_smoke.yaml")
    assert (base.environment, m.environment, smoke.environment) == ("docker", "modal", "modal")
    cmd = sweep.harbor_cmd(m, "j")
    assert cmd[cmd.index("-e") + 1] == "modal" and sweep.harbor_cmd(base, "j")[cmd.index("-e") + 1] == "docker"
    for c, cfg in m.configs.items():  # the same conditions as the Docker baseline
        assert cfg.to_dict() == base.configs[c].to_dict()
    monkeypatch.delenv("MODAL_IMAGE_BUILDER_VERSION", raising=False)
    assert sweep.harbor_env(m, X="1")["MODAL_IMAGE_BUILDER_VERSION"] == sweep.MODAL_IMAGE_BUILDER
    assert "MODAL_IMAGE_BUILDER_VERSION" not in sweep.harbor_env(base)
    monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", "2024.10")
    assert sweep.harbor_env(m)["MODAL_IMAGE_BUILDER_VERSION"] == "2024.10"
    i = smoke.configs["single_baseline"].interactive
    assert (smoke.rounds, i.wall_clock_sec, i.interval_sec) == (1, 120, 30)
    assert 120 + sweep.SESSION_GRACE_SEC <= 43200 * smoke.timeout_multiplier
    if smoke.dataset_path.is_dir():
        assert sweep.check_interactive(smoke) == []


def test_harbor_environment_is_checked(tmp_path, monkeypatch):
    spec = (REPO / "configs/sweeps/swe_interactive_baseline.yaml").read_text().replace(
        "  n_concurrent: 3\n", "  n_concurrent: 3\n  environment: kubernetes\n")
    (tmp_path / "bad.yaml").write_text(spec)
    monkeypatch.chdir(REPO)
    with pytest.raises(ConfigError, match="harbor.environment"):
        sweep.load_spec(tmp_path / "bad.yaml")


# --- --model ------------------------------------------------------------------------------------

LUNA = REPO / "configs/prime/models/gpt-5.6-luna.yaml"


def test_model_flag_runs_every_orchestrator_on_the_model(tmp_path, fake, monkeypatch, capsys):
    path = make_spec(tmp_path, conditions={"single": {"topology": "single"}, "same": {"subagent": {"model": None}},
                                           "par": {}})
    own, spec = sweep.load_spec(path), sweep.load_spec(path, model=LUNA)
    assert spec.name == "t-gpt-5.6-luna" and spec.jobs_dir == own.jobs_dir.with_name("t-gpt-5.6-luna")
    assert spec.model == LUNA and sweep.hints_dir(spec) == own.jobs_dir / "hints"
    for cfg in spec.configs.values():  # the file replaces the orchestrator: no Sail keys left over
        assert (cfg.orchestrator.model, cfg.orchestrator.provider, cfg.orchestrator.completion_window) == (
            "openai/gpt-5.6-luna", "prime", None)
    assert spec.configs["same"].subagent_model.model == "openai/gpt-5.6-luna"
    assert spec.configs["par"].subagent_model.model == SUB_MODEL  # a sweep's own subagent model stays
    assert sweep.job_name(spec, 1, "single") == "t-gpt-5.6-luna__r1__single"
    assert cli.main(["sweep", str(path), "--model", str(LUNA), "--dry-run"]) == 0
    assert "t-gpt-5.6-luna__r1__single" in capsys.readouterr().out
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.setenv("SAIL_API_KEY", "test-key")
    assert sweep.preflight(spec) == 1
    out = capsys.readouterr().out
    assert "[FAIL] PRIME_API_KEY: missing" in out and "[PASS] SAIL_API_KEY: set" in out  # Gemma's still Sail
    assert "[PASS] request openai/gpt-5.6-luna" in out and f"request {ORCH_MODEL}" not in out


def test_model_flag_errors_and_relative_path(tmp_path, monkeypatch):
    path = make_spec(tmp_path)
    bad = tmp_path / "proj/configs/bad.yaml"
    bad.write_text("model: x/y\nprovider: openrouter\n")
    with pytest.raises(ConfigError, match=r"^--model .*bad\.yaml: orchestrator\.provider: 'openrouter' is not one of"):
        sweep.load_spec(path, model=bad)
    with pytest.raises(ConfigError, match="^--model: .*missing.yaml"):
        sweep.load_spec(path, model=tmp_path / "missing.yaml")
    shutil.copy(LUNA, tmp_path / "proj/configs/luna.yaml")
    monkeypatch.chdir(tmp_path)  # not under the repository root: resolved against it, as the spec in run_experiment.sh
    assert sweep.load_spec(path, model="configs/luna.yaml").name == "t-luna"


@pytest.mark.parametrize("model", sorted((REPO / "configs/prime/models").glob("*.yaml")), ids=lambda p: p.stem)
def test_prime_model_files_load_and_match_the_prices(model):
    spec = sweep.load_spec(REPO / "configs/sweeps/swe_interactive_levels.yaml", model=model)
    m = spec.configs["single_baseline"].orchestrator
    assert m.provider == "prime" and m.max_output_tokens == 32768 and spec.configs["single_eta"].subagent_model == m
    prices = yaml.safe_load((REPO / "configs/pricing.yaml").read_text())
    assert prices["prime"][m.model] == prices["default"][m.model] == {
        "in": m.price_in, "cached": m.price_cache_read, "out": m.price_out}


@pytest.mark.parametrize("over, msg", [
    ({"provider": "openrouter"}, "orchestrator.provider: 'openrouter' is not one of prime, sail"),
    ({"reasoning_effort": ""}, "orchestrator.reasoning_effort: expected null or a level"),
    ({"cache_control": "yes"}, "orchestrator.cache_control: expected true or false"),
])
def test_bad_model_configs(tmp_path, over, msg):
    with pytest.raises(ConfigError, match=msg):
        sweep.load_spec(make_spec(tmp_path, overrides={"orchestrator": over}))
