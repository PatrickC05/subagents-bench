"""Tests for the sweep launcher (subagents_bench.sweep) and the sweep configs.

No network and no Docker: ``subprocess.run`` / ``subprocess.Popen`` are
replaced by fakes that mimic harbor's on-disk job layout (taken from a real
harbor 0.24.0 oracle job), and the LLM client is a fake module.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from subagents_bench import sweep

REPO = Path(__file__).resolve().parents[1]
SWEEPS = REPO / "configs" / "sweeps"

PILOT_CONDITIONS = [
    "single_5step",
    "single_matched",
    *(f"gemma31b_{s}_{sc}" for s in ("parallel", "sequential", "arbiter") for sc in ("easy", "medium", "hard")),
    "samemodel_parallel_hard",
    *(f"manager_{s}_hard" for s in ("parallel", "sequential", "arbiter")),
]
PILOT_TASKS = [
    "fix-git", "prove-plus-comm", "cobol-modernization", "overfull-hbox",
    "crack-7z-hash", "raman-fitting", "constraints-scheduling", "nginx-request-logging",
    "openssl-selfsigned-cert", "polyglot-c-py",
]


# ---------------------------------------------------------------------------
# helpers / fakes
# ---------------------------------------------------------------------------


def write_yaml(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def make_project(tmp_path: Path, **spec_overrides) -> Path:
    """A minimal project (pyproject.toml marks the root) with a base config
    and a 2-condition spec; returns the spec path."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    write_yaml(
        tmp_path / "configs/base.yaml",
        {
            "name": "base",
            "topology": "subagents",
            "orchestrator": {"model": "orch-m", "provider": "sail", "completion_window": "balanced",
                             "price_in": 1.0, "price_out": 2.0, "price_cache_read": 0.5},
            "subagent": {"model": {"model": "sub-m", "provider": "sail", "completion_window": "balanced",
                                   "price_in": 0.1, "price_out": 0.2},
                         "strategy": "parallel", "scope": "hard", "max_concurrent": 3,
                         "max_dispatches": 15, "max_steps": 8},
            "budget": {"orchestrator_steps": 5},
        },
    )
    spec = {
        "name": "t",
        "dataset": "terminal-bench@2.0",
        "tasks": ["fix-git", "prove-plus-comm"],
        "rounds": 2,
        "order_seed": 0,
        "base": "configs/base.yaml",
        "conditions": {"single": {"topology": "single"}, "par": {"subagent": {"strategy": "parallel"}}},
        "harbor": {"env": "docker", "n_concurrent": 4, "agent_timeout_multiplier": 4.0, "extra_args": []},
        "jobs_dir": "jobs/t",
    }
    spec.update(spec_overrides)
    return write_yaml(tmp_path / "configs/sweeps/t.yaml", spec)


class StubExperimentConfig:
    """Lossless stand-in for ExperimentConfig so sweep tests do not depend on
    config.py (owned and changed by the runtime)."""

    def __init__(self, d):
        self.d = copy.deepcopy(d)
        self.orchestrator = StubModel.of(d["orchestrator"])
        sub = d.get("subagent") or {}
        self.subagent = types.SimpleNamespace(
            arbiter=types.SimpleNamespace(model=StubModel.of((sub.get("arbiter") or {}).get("model")))
        )
        self._sub_model = StubModel.of(sub.get("model"))

    def resolved_subagent_model(self):
        return self._sub_model or self.orchestrator

    def to_dict(self):
        return copy.deepcopy(self.d)


@dataclass
class StubModel:
    model: str
    provider: str
    completion_window: str | None = None
    max_output_tokens: int = 4096

    @classmethod
    def of(cls, d):
        if not d:
            return None
        return cls(d["model"], d.get("provider", "anthropic"), d.get("completion_window"))


@pytest.fixture
def stub_config(monkeypatch):
    monkeypatch.setattr(sweep, "load_experiment_config", StubExperimentConfig)


def harbor_result(n_total, n_completed, n_errored=0, running=0, pending=0, finished=True):
    """A job-level result.json in harbor 0.24.0's shape."""
    return {
        "id": "x",
        "started_at": "2026-10-11T20:52:53",
        "finished_at": "2026-10-11T20:53:39" if finished else None,
        "n_total_trials": n_total,
        "stats": {
            "n_completed_trials": n_completed,
            "n_errored_trials": n_errored,
            "n_running_trials": running,
            "n_pending_trials": pending,
            "n_cancelled_trials": 0,
        },
    }


def make_job(job_dir: Path, tasks, *, mode="complete", reward=1.0, exit_reason="finished"):
    """Create a harbor-like job dir. mode: complete | partial | billing | errored."""
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "config.json").write_text("{}")
    n_err = 0
    for i, t in enumerate(tasks):
        trial = job_dir / f"{t}__abc{i}"
        (trial / "agent").mkdir(parents=True)
        er = exit_reason
        exc = None
        if mode == "billing" and i == 0:
            er = "billing_error"
            exc = {"exception_type": "BillingError", "exception_message": "402"}
        if mode == "errored" and i == 0:
            exc = {"exception_type": "AgentTimeoutError", "exception_message": "t"}
        if exc:
            n_err += 1
        (trial / "agent" / "episode.json").write_text(json.dumps({"exit_reason": er}))
        if mode == "partial" and i > 0:
            continue  # trial still running: no trial result.json
        (trial / "result.json").write_text(
            json.dumps({"task_name": t, "trial_name": trial.name, "exception_info": exc,
                        "verifier_result": {"rewards": {"reward": reward}}})
        )
    n = len(tasks)
    if mode == "partial":
        res = harbor_result(n, 1, running=n - 1, finished=False)
    else:
        res = harbor_result(n, n, n_errored=n_err)
    (job_dir / "result.json").write_text(json.dumps(res))


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class FakeHarbor:
    """Fakes subprocess.Popen (harbor run) and subprocess.run (git, docker,
    harbor --version, harbor dataset download)."""

    def __init__(self, modes=None, rc=0):
        self.modes = modes or {}  # job name -> make_job mode
        self.rc = rc
        self.popen_calls = []  # (cmd, env, cwd)
        self.run_calls = []

    def popen(self, cmd, env=None, cwd=None, **kw):
        self.popen_calls.append((list(cmd), dict(env or {}), Path(cwd)))
        job = _arg(cmd, "--job-name")
        tasks = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-i"]
        make_job(Path(cwd) / _arg(cmd, "-o") / job, tasks, mode=self.modes.get(job, "complete"))
        rc = self.rc

        class P:
            returncode = rc

            def wait(self, timeout=None):
                return rc

        return P()

    def run(self, cmd, **kw):
        self.run_calls.append(list(cmd))
        if cmd[0] == "git" and "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "deadbeef" * 5 + "\n", "")
        if cmd[0] == "git" and "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, " M foo.py\n", "")
        if cmd[0] == "docker":
            return subprocess.CompletedProcess(cmd, 0, "28.3.2|aarch64|12|33601363968\n", "")
        if len(cmd) > 2 and cmd[1:3] == ["dataset", "download"]:
            make_task_tree(Path(_arg(cmd, "-o")) / "terminal-bench", PILOT_TASKS)
            return subprocess.CompletedProcess(cmd, 0, "Successfully downloaded", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def fake_harbor(monkeypatch):
    fh = FakeHarbor()
    monkeypatch.setattr(sweep.subprocess, "Popen", fh.popen)
    monkeypatch.setattr(sweep.subprocess, "run", fh.run)
    monkeypatch.setattr(sweep, "harbor_executable", lambda: "/fake/bin/harbor")
    monkeypatch.delenv("SUBAGENTS_BENCH_DATASET_CACHE", raising=False)
    return fh


TASK_TOML = """version = "1.0"

[metadata]
difficulty = "{difficulty}"
category = "software-engineering"
tags = [ "coding", "version-control",]
expert_time_estimate_min = {expert}
junior_time_estimate_min = 20.0

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 600.0

[environment]
build_timeout_sec = 600.0
docker_image = "alexgshaw/{name}:20251031"
"""


def make_task_tree(root: Path, names, difficulty="easy", expert=5.0):
    for n in names:
        d = root / n
        d.mkdir(parents=True, exist_ok=True)
        (d / "task.toml").write_text(TASK_TOML.format(name=n, difficulty=difficulty, expert=expert))


def read_manifest(spec):
    p = spec.jobs_dir / sweep.MANIFEST_NAME
    return [json.loads(line) for line in p.read_text().splitlines()]


# ---------------------------------------------------------------------------
# spec parsing and merging
# ---------------------------------------------------------------------------


def test_deep_merge_semantics_and_no_mutation():
    base = {"a": {"b": 1, "c": [1, 2], "d": {"e": 1}}, "m": {"x": 1}, "s": 1}
    over = {"a": {"c": [3], "d": {"f": 2}}, "m": None, "n": {"y": 2}}
    base_copy, over_copy = copy.deepcopy(base), copy.deepcopy(over)
    out = sweep.deep_merge(base, over)
    assert out == {"a": {"b": 1, "c": [3], "d": {"e": 1, "f": 2}}, "m": None, "s": 1, "n": {"y": 2}}
    assert base == base_copy and over == over_copy
    out["a"]["d"]["e"] = 99
    assert base["a"]["d"]["e"] == 1  # no shared references


def test_pilot_spec_matches_design():
    spec = sweep.load_spec(SWEEPS / "pilot.yaml")
    assert spec.name == "pilot"
    assert spec.dataset == "terminal-bench@2.0"
    assert spec.tasks == PILOT_TASKS
    assert spec.rounds == 3
    assert spec.order_seed == 0
    assert list(spec.conditions) == PILOT_CONDITIONS
    assert spec.harbor == sweep.HarborSettings(env="docker", n_concurrent=4, agent_timeout_multiplier=4.0, extra_args=[])
    assert spec.jobs_dir == REPO / "jobs" / "pilot"
    assert spec.base_path == REPO / "configs" / "sail" / "base.yaml"
    assert spec.root == REPO


def test_smoke_spec_extends_pilot():
    smoke = sweep.load_spec(SWEEPS / "smoke.yaml")
    pilot = sweep.load_spec(SWEEPS / "pilot.yaml")
    assert smoke.name == "smoke"
    assert smoke.tasks == ["fix-git", "prove-plus-comm"]
    assert smoke.rounds == 1
    assert smoke.jobs_dir == REPO / "jobs" / "smoke"
    assert smoke.harbor == pilot.harbor
    assert smoke.order_seed == 0
    assert smoke.conditions == pilot.conditions
    assert sweep.resolve_conditions(smoke) == sweep.resolve_conditions(pilot)
    assert [p.name for p in smoke.source_files] == ["pilot.yaml", "smoke.yaml"]
    assert "extends" not in smoke.raw


def test_base_config_values():
    base = yaml.safe_load((REPO / "configs/sail/base.yaml").read_text())
    o = base["orchestrator"]
    assert (o["model"], o["provider"], o["completion_window"]) == ("deepseek-ai/DeepSeek-V4.1-Flash", "sail", "balanced")
    assert (o["temperature"], o["max_output_tokens"], o["context_window"]) == (0.7, 8192, None)
    assert (o["price_in"], o["price_cache_read"], o["price_out"]) == (0.07, 0.005, 0.22)
    s = base["subagent"]
    m = s["model"]
    assert (m["model"], m["provider"], m["completion_window"]) == ("nvidia/Gemma-4-31B-IT-NVFP4", "sail", "balanced")
    assert (m["temperature"], m["max_output_tokens"], m["context_window"]) == (0.7, 4096, 262144)
    assert (m["price_in"], m["price_cache_read"], m["price_out"]) == (0.11, 0.06, 0.32)
    assert (s["strategy"], s["scope"], s["max_concurrent"], s["max_dispatches"], s["max_steps"]) == ("parallel", "hard", 3, 15, 8)
    assert s["arbiter"] == {"n_candidates": 3, "debate_rounds": 1, "confidence_threshold": 0.8, "model": None, "max_copy_mb": 500}
    assert base["budget"] == {"orchestrator_steps": 5, "total_token_budget": 5000000, "timeout_sec": 3600,
                              "command_timeout_sec": 180, "max_command_timeout_sec": 900}
    assert base["sandbox"]["kind"] == "harbor" and base["sandbox"]["workdir"] == "auto"


def test_pilot_resolved_conditions():
    spec = sweep.load_spec(SWEEPS / "pilot.yaml")
    res = sweep.resolve_conditions(spec)
    for name, cfg in res.items():
        assert cfg["name"] == name
        assert cfg["orchestrator"]["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
        # Every Sail model in the sweep uses the same tier.
        for m in sweep.models_used(cfg):
            assert m["provider"] == "sail" and m["completion_window"] == "balanced"
            assert m["temperature"] == 0.7
    assert res["single_5step"]["topology"] == "single"
    assert res["single_5step"]["budget"]["orchestrator_steps"] == 5
    assert res["single_matched"]["topology"] == "single"
    assert res["single_matched"]["budget"]["orchestrator_steps"] == 40
    assert res["single_matched"]["budget"]["total_token_budget"] == 5000000  # untouched by the override
    for strat in ("parallel", "sequential", "arbiter"):
        for scope in ("easy", "medium", "hard"):
            c = res[f"gemma31b_{strat}_{scope}"]
            assert c["topology"] == "subagents"
            assert (c["subagent"]["strategy"], c["subagent"]["scope"]) == (strat, scope)
            assert c["subagent"]["model"]["model"] == "nvidia/Gemma-4-31B-IT-NVFP4"
            assert (c["subagent"]["max_concurrent"], c["subagent"]["max_dispatches"], c["subagent"]["max_steps"]) == (3, 15, 8)
    same = res["samemodel_parallel_hard"]["subagent"]
    assert same["model"] is None and (same["strategy"], same["scope"]) == ("parallel", "hard")
    assert sweep.models_used(res["samemodel_parallel_hard"])[1]["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    # Arbiter judge = orchestrator model.
    arb = sweep.models_used(res["gemma31b_arbiter_hard"])
    assert [m["model"] for m in arb] == ["deepseek-ai/DeepSeek-V4.1-Flash", "nvidia/Gemma-4-31B-IT-NVFP4", "deepseek-ai/DeepSeek-V4.1-Flash"]


def test_prime_size_spec():
    spec = sweep.load_spec(SWEEPS / "prime_size.yaml")
    res = sweep.resolve_conditions(spec)
    assert list(res) == ["qwen0.8b_parallel_hard", "qwen2b_parallel_hard", "qwen4b_parallel_hard", "qwen9b_parallel_hard"]
    assert spec.tasks == PILOT_TASKS and spec.rounds == 3 and spec.jobs_dir == REPO / "jobs/prime_size"
    sizes = []
    for cfg in res.values():
        o, s = cfg["orchestrator"], cfg["subagent"]
        assert (o["model"], o["provider"], o["completion_window"]) == ("deepseek-ai/DeepSeek-V4.1-Flash", "sail", "balanced")
        assert (s["strategy"], s["scope"]) == ("parallel", "hard")
        assert s["model"]["provider"] == "prime" and s["model"]["completion_window"] is None
        assert s["model"]["context_window"] is None
        sizes.append(s["model"]["model"])
    assert sizes == ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]


@pytest.mark.parametrize("spec_file", ["smoke.yaml", "pilot.yaml", "prime_size.yaml"])
def test_every_resolved_condition_loads_into_experiment_config(spec_file):
    """Needs config.py to implement DESIGN section 2 (runtime owner)."""
    from subagents_bench.config import ExperimentConfig

    spec = sweep.load_spec(SWEEPS / spec_file)
    for name, cfg in sweep.resolve_conditions(spec).items():
        exp = ExperimentConfig.from_dict(cfg)
        assert exp.name == name
        problems, full = sweep.validate_condition(cfg)
        assert problems == [], f"{name}: {problems}"
        assert ExperimentConfig.from_dict(full).to_dict() == full


@pytest.mark.parametrize(
    "path",
    ["configs/unified.yaml", "configs/exploitbench.yaml", "configs/baselines/single_agent.yaml",
     "configs/baselines/same_model_subagents.yaml", "configs/sail/base.yaml"],
)
def test_legacy_and_base_configs_load(path):
    """Needs config.py to implement DESIGN section 2 (runtime owner)."""
    from subagents_bench.config import ExperimentConfig

    raw = yaml.safe_load((REPO / path).read_text())
    cfg = ExperimentConfig.from_yaml(REPO / path)
    assert cfg.name == raw["name"]
    assert sweep._dropped_keys(raw, cfg.to_dict()) == []


@pytest.mark.parametrize(
    "override, msg",
    [
        ({"bogus": 1}, "unknown spec keys"),
        ({"name": "a__b"}, "invalid sweep name"),
        ({"conditions": {"x__y": {}}}, "invalid condition name"),
        ({"task_filter": {"difficulty": ["easy"]}}, "not both"),
        ({"tasks": []}, "non-empty list"),
        ({"tasks": ["a", "a"]}, "duplicates"),
        ({"rounds": 0}, "rounds"),
        ({"base": "configs/missing.yaml"}, "base config not found"),
        ({"harbor": {"n_concurrent": 0}}, "n_concurrent"),
        ({"harbor": {"envv": "docker"}}, "unknown harbor keys"),
        ({"conditions": {}}, "conditions"),
        ({"conditions": {"x": [1]}}, "must be a mapping"),
    ],
)
def test_spec_validation_errors(tmp_path, override, msg):
    path = make_project(tmp_path, **override)
    with pytest.raises(sweep.SpecError, match=msg):
        sweep.load_spec(path)


def test_spec_extends_cycle_and_defaults(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    write_yaml(tmp_path / "a.yaml", {"extends": "b.yaml", "name": "a"})
    write_yaml(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(sweep.SpecError, match="cycle"):
        sweep.load_spec(tmp_path / "a.yaml")
    # Defaults: jobs_dir jobs/<name>, order_seed 0, rounds 1, harbor docker.
    write_yaml(tmp_path / "c.yaml", {"name": "c", "dataset": "d@1", "tasks": ["t"], "conditions": {"x": None}})
    spec = sweep.load_spec(tmp_path / "c.yaml")
    assert spec.jobs_dir == tmp_path / "jobs" / "c"
    assert (spec.order_seed, spec.rounds, spec.harbor.env, spec.base) == (0, 1, "docker", {})
    assert sweep.resolve_conditions(spec) == {"x": {"name": "x"}}


# ---------------------------------------------------------------------------
# order, job names, commands, plan
# ---------------------------------------------------------------------------


def test_condition_order_deterministic_per_seed():
    conds = list(PILOT_CONDITIONS)
    a = sweep.condition_order(conds, 3, 0)
    b = sweep.condition_order(conds, 3, 0)
    assert a == b
    assert set(a) == {1, 2, 3}
    for r in a:
        assert sorted(a[r]) == sorted(conds)  # a permutation
    assert len({tuple(o) for o in a.values()}) == 3  # rounds differ
    assert sweep.condition_order(conds, 3, 1) != a
    # Golden values: Python's Mersenne Twister + shuffle are stable, and
    # sweep.json records the order anyway.
    assert sweep.condition_order(["a", "b", "c", "d", "e"], 2, 0) == {1: ["c", "b", "a", "e", "d"], 2: ["a", "c", "b", "d", "e"]}
    # A round's order does not depend on how many rounds are planned.
    assert sweep.condition_order(conds, 1, 0)[1] == a[1]


def test_job_name_roundtrip():
    assert sweep.job_name("pilot", 2, "gemma31b_parallel_hard") == "pilot__r2__gemma31b_parallel_hard"
    assert sweep.parse_job_name("pilot__r2__gemma31b_parallel_hard") == ("pilot", 2, "gemma31b_parallel_hard")
    assert sweep.parse_job_name("prime_size__r10__qwen0.8b_parallel_hard") == ("prime_size", 10, "qwen0.8b_parallel_hard")
    assert sweep.parse_job_name("_oracle") is None
    assert sweep.parse_job_name("pilot__x2__c") is None


def test_build_harbor_command_exact():
    cmd = sweep.build_harbor_command(
        harbor="/v/bin/harbor", dataset="terminal-bench@2.0", tasks=["fix-git", "prove-plus-comm"],
        agent=sweep.AGENT_IMPORT_PATH, env="docker", n_concurrent=4, jobs_dir="jobs/pilot",
        job="pilot__r1__single_5step", agent_timeout_multiplier=4.0, extra_args=["--debug"],
    )
    assert cmd == [
        "/v/bin/harbor", "run", "-d", "terminal-bench@2.0", "-i", "fix-git", "-i", "prove-plus-comm",
        "-a", "subagents_bench.harness.harbor_agent:OrchestratorAgent", "-e", "docker", "-n", "4", "-k", "1",
        "-o", "jobs/pilot", "--job-name", "pilot__r1__single_5step", "--agent-timeout-multiplier", "4.0",
        "-y", "-q", "--debug",
    ]
    oracle = sweep.build_harbor_command(harbor="h", dataset="d@1", tasks=["t"], agent="oracle", env="daytona",
                                        n_concurrent=2, jobs_dir="j", job="_oracle")
    assert "--agent-timeout-multiplier" not in oracle
    assert oracle[oracle.index("-a") + 1] == "oracle" and oracle[oracle.index("-e") + 1] == "daytona"


def test_harbor_executable_prefers_venv_sibling(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python").write_text("")
    h = fake_bin / "harbor"
    h.write_text("#!/bin/sh\n")
    h.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(fake_bin / "python"))
    assert sweep.harbor_executable() == str(h)
    h.unlink()
    monkeypatch.setattr(sweep.shutil, "which", lambda name: "/usr/local/bin/harbor" if name == "harbor" else None)
    assert sweep.harbor_executable() == "/usr/local/bin/harbor"


def test_plan_jobs_order_filters_and_status(tmp_path):
    spec = sweep.load_spec(make_project(tmp_path))
    make_job(spec.jobs_dir / "t__r1__par", ["fix-git", "prove-plus-comm"])
    make_job(spec.jobs_dir / "t__r2__single", ["fix-git", "prove-plus-comm"], mode="partial")
    plan = sweep.plan_jobs(spec, spec.tasks, harbor="/h")
    order = sweep.condition_order(["single", "par"], 2, 0)
    assert [(j.round, j.condition) for j in plan] == [(1, c) for c in order[1]] + [(2, c) for c in order[2]]
    by = {j.job: j for j in plan}
    assert by["t__r1__par"].action == "skip"
    assert by["t__r2__single"].action == "move-aside+run"
    assert by["t__r1__single"].action == "run"
    j = by["t__r1__single"]
    assert j.cmd[:2] == ["/h", "run"] and j.cmd[j.cmd.index("-o") + 1] == "jobs/t"
    assert j.config_path == spec.jobs_dir / "configs" / "single.yaml"
    only = sweep.plan_jobs(spec, spec.tasks, only=["par"], rounds=[2])
    assert [j.job for j in only] == ["t__r2__par"]
    with pytest.raises(sweep.SpecError, match="unknown conditions"):
        sweep.plan_jobs(spec, spec.tasks, only=["nope"])
    with pytest.raises(sweep.SpecError, match="exceed"):
        sweep.plan_jobs(spec, spec.tasks, rounds=[3])


def test_parse_rounds_arg():
    assert sweep.parse_rounds_arg("2,1,2") == [1, 2]
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        sweep.parse_rounds_arg("0")
    with pytest.raises(argparse.ArgumentTypeError):
        sweep.parse_rounds_arg("a,b")


# ---------------------------------------------------------------------------
# job status, resume, billing scan
# ---------------------------------------------------------------------------


def test_job_status_cases(tmp_path):
    assert sweep.job_status(tmp_path / "nope").state == "absent"
    d = tmp_path / "j"
    d.mkdir()
    st = sweep.job_status(d)
    assert (st.state, st.reason) == ("partial", "job dir has no result.json")
    (d / "result.json").write_text("{not json")
    assert sweep.job_status(d).reason.startswith("unreadable result.json")

    make_job(tmp_path / "ok", ["a", "b"])
    st = sweep.job_status(tmp_path / "ok")
    assert st.complete and (st.n_total, st.n_completed, st.n_errored, st.n_trial_results) == (2, 2, 0, 2)

    # harbor 0.24.0 counts errored trials inside n_completed_trials: complete.
    make_job(tmp_path / "err", ["a", "b"], mode="errored")
    st = sweep.job_status(tmp_path / "err")
    assert st.complete and (st.n_completed, st.n_errored) == (2, 1)

    make_job(tmp_path / "part", ["a", "b", "c"], mode="partial")
    st = sweep.job_status(tmp_path / "part")
    assert st.state == "partial" and st.reason == "2 running / 0 pending trials"

    # Stats say done but a trial dir lacks its result.json.
    make_job(tmp_path / "miss", ["a", "b"])
    (tmp_path / "miss" / "b__abc1" / "result.json").unlink()
    assert sweep.job_status(tmp_path / "miss").reason == "only 1/2 trial result.json files"

    # No finished_at -> partial even if counts look done.
    make_job(tmp_path / "nf", ["a"])
    (tmp_path / "nf" / "result.json").write_text(json.dumps(harbor_result(1, 1, finished=False)))
    assert sweep.job_status(tmp_path / "nf").reason == "job has no finished_at"

    (tmp_path / "nf" / "result.json").write_text(json.dumps(harbor_result(2, 1)))
    assert sweep.job_status(tmp_path / "nf").reason == "1/2 trials finished"


def test_job_status_on_real_oracle_job_layout():
    job = REPO / "jobs" / "oracle-smoke"
    if not (job / "result.json").is_file():
        pytest.skip("no local oracle-smoke job")
    st = sweep.job_status(job)
    assert st.complete and st.n_total == 2
    assert {r["task"] for r in sweep.trial_rewards(job).values()} == {"fix-git", "prove-plus-comm"}


def test_move_partial_increments(tmp_path):
    jobs = tmp_path / "jobs"
    for k in (1, 2):
        (jobs / "t__r1__x").mkdir(parents=True)
        (jobs / "t__r1__x" / "marker").write_text(str(k))
        dest = sweep.move_partial(jobs / "t__r1__x", jobs)
        assert dest == jobs / "_partial" / f"t__r1__x.partial.{k}"
        assert (dest / "marker").read_text() == str(k)
        assert not (jobs / "t__r1__x").exists()


def test_scan_billing_errors(tmp_path):
    make_job(tmp_path / "ok", ["a", "b"])
    assert sweep.scan_billing_errors(tmp_path / "ok") == []
    make_job(tmp_path / "bill", ["a", "b"], mode="billing")
    assert sweep.scan_billing_errors(tmp_path / "bill") == ["a__abc0"]
    # Backup signal: harbor's exception type even without episode.json.
    (tmp_path / "bill" / "a__abc0" / "agent" / "episode.json").unlink()
    assert sweep.scan_billing_errors(tmp_path / "bill") == ["a__abc0"]
    assert sweep.scan_billing_errors(tmp_path / "missing") == []


# ---------------------------------------------------------------------------
# task metadata
# ---------------------------------------------------------------------------


def test_parse_task_toml_and_tasks_meta(tmp_path):
    root = tmp_path / "terminal-bench"
    make_task_tree(root, ["fix-git", "prove-plus-comm"])
    make_task_tree(root, ["crack-7z-hash"], difficulty="medium", expert=15.0)
    (root / "bare").mkdir()
    (root / "bare" / "task.toml").write_text('version = "1.0"\n')
    meta = sweep.build_tasks_meta(root, "terminal-bench@2.0")
    assert meta["dataset"] == "terminal-bench@2.0"
    assert set(meta["tasks"]) == {"fix-git", "prove-plus-comm", "crack-7z-hash", "bare"}
    assert meta["tasks"]["fix-git"] == {
        "difficulty": "easy", "category": "software-engineering", "expert_time_min": 5.0, "junior_time_min": 20.0,
        "agent_timeout_sec": 600.0, "verifier_timeout_sec": 900.0, "docker_image": "alexgshaw/fix-git:20251031",
        "tags": ["coding", "version-control"],
    }
    assert meta["tasks"]["bare"] == {k: None for k in meta["tasks"]["fix-git"] if k != "tags"} | {"tags": []}
    assert sweep.filter_tasks(meta, {"difficulty": ["medium"]}) == ["crack-7z-hash"]
    assert sweep.filter_tasks(meta, {"difficulty": "easy", "max_expert_min": 10}) == ["fix-git", "prove-plus-comm"]
    assert sweep.filter_tasks(meta, {"max_expert_min": 10}) == ["fix-git", "prove-plus-comm"]  # bare has no estimate
    sub = sweep.build_tasks_meta(root, "terminal-bench@2.0", ["prove-plus-comm"])
    assert list(sub["tasks"]) == ["prove-plus-comm"]
    with pytest.raises(KeyError, match="not-a-task"):
        sweep.build_tasks_meta(root, "terminal-bench@2.0", ["not-a-task"])
    assert sweep.dataset_task_root(tmp_path, "terminal-bench@2.0") == root
    assert sweep.dataset_task_root(tmp_path / "empty", "terminal-bench@2.0") is None


def test_download_dataset_command_and_failure(tmp_path, fake_harbor, monkeypatch):
    root = sweep.download_dataset("terminal-bench@2.0", tmp_path / "cache", "/h")
    assert root == tmp_path / "cache" / "terminal-bench"
    assert fake_harbor.run_calls[-1] == ["/h", "dataset", "download", "terminal-bench@2.0", "-o", str(tmp_path / "cache")]
    monkeypatch.setattr(sweep.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "network down"))
    with pytest.raises(RuntimeError, match="network down"):
        sweep.download_dataset("terminal-bench@2.0", tmp_path / "other", "/h")
    with pytest.raises(RuntimeError, match="harbor executable not found"):
        sweep.download_dataset("terminal-bench@2.0", tmp_path / "other", None)


def test_tasks_cli_prints_table(tmp_path, fake_harbor, capsys):
    import argparse

    parser = argparse.ArgumentParser()
    sweep.register_cli(parser.add_subparsers())
    out_json = tmp_path / "meta.json"
    args = parser.parse_args(["tasks", "terminal-bench@2.0", "--tasks", "fix-git", "--cache-dir", str(tmp_path / "c"),
                              "--out", str(out_json)])
    assert args.func(args) == 0
    text = capsys.readouterr().out
    assert "terminal-bench@2.0: 1 tasks" in text
    assert "fix-git" in text and "alexgshaw/fix-git:20251031" in text
    assert list(json.loads(out_json.read_text())["tasks"]) == ["fix-git"]
    args = parser.parse_args(["tasks", "--tasks", "nope", "--cache-dir", str(tmp_path / "c"), "--offline"])
    assert args.func(args) == 1


# ---------------------------------------------------------------------------
# subprocess environment
# ---------------------------------------------------------------------------


def test_build_subprocess_env(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "prime_credentials", lambda: ("pk-from-cli", "team-from-cli"))
    base = {"PATH": "/bin", "SUBAGENTS_BENCH_CONFIG_JSON": "{}", "PRIME_TEAM_ID": "team-env"}
    env = sweep.build_subprocess_env(tmp_path / "c.yaml", base_env=base, need_prime=True)
    assert env["SUBAGENTS_BENCH_CONFIG"] == str(tmp_path / "c.yaml")
    assert "SUBAGENTS_BENCH_CONFIG_JSON" not in env
    assert env["PRIME_API_KEY"] == "pk-from-cli"
    assert env["PRIME_TEAM_ID"] == "team-env"  # existing env wins
    assert base == {"PATH": "/bin", "SUBAGENTS_BENCH_CONFIG_JSON": "{}", "PRIME_TEAM_ID": "team-env"}
    env2 = sweep.build_subprocess_env(None, base_env={"PATH": "/bin"}, need_prime=False)
    assert env2 == {"PATH": "/bin"}


def test_load_env_files_does_not_override(monkeypatch, tmp_path):
    # setenv + delenv: the variable is absent now and removed again on teardown.
    monkeypatch.setenv("SB_TEST_NEW", "placeholder")
    monkeypatch.delenv("SB_TEST_NEW")
    monkeypatch.setenv("SB_TEST_KEEP", "shell-value")
    dotenv = tmp_path / ".env"
    dotenv.write_text("SB_TEST_NEW=from-file\nSB_TEST_KEEP=file-value\n# comment\n")
    status = sweep.load_env_files(dotenv)
    assert status == {"SB_TEST_NEW": "set", "SB_TEST_KEEP": "kept"}
    import os

    assert os.environ["SB_TEST_NEW"] == "from-file"
    assert os.environ["SB_TEST_KEEP"] == "shell-value"
    assert sweep.load_env_files(tmp_path / "missing.env") == {}


# ---------------------------------------------------------------------------
# run_sweep end to end (fake harbor)
# ---------------------------------------------------------------------------


def test_run_sweep_runs_resumes_and_records(tmp_path, fake_harbor, stub_config):
    spec = sweep.load_spec(make_project(tmp_path))
    lines: list[str] = []
    assert sweep.run_sweep(spec, out=lines.append) == 0

    # 2 rounds x 2 conditions, in the shuffled order, each with its config.
    order = sweep.condition_order(["single", "par"], 2, 0)
    expected = [f"t__r{r}__{c}" for r in (1, 2) for c in order[r]]
    assert [_arg(c, "--job-name") for c, _, _ in fake_harbor.popen_calls] == expected
    for cmd, env, cwd in fake_harbor.popen_calls:
        cond = _arg(cmd, "--job-name").split("__")[2]
        assert env["SUBAGENTS_BENCH_CONFIG"] == str(spec.jobs_dir / "configs" / f"{cond}.yaml")
        assert cwd == tmp_path
        assert cmd[cmd.index("-a") + 1] == sweep.AGENT_IMPORT_PATH
        assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "-i"] == ["fix-git", "prove-plus-comm"]

    # Materialized configs are the resolved condition configs.
    single = yaml.safe_load((spec.jobs_dir / "configs" / "single.yaml").read_text())
    assert single["name"] == "single" and single["topology"] == "single"
    assert single["orchestrator"]["model"] == "orch-m"

    man = read_manifest(spec)
    assert [m["job"] for m in man] == expected
    m0 = man[0]
    for key in ("job", "round", "condition", "cmd", "t_start", "t_end", "return_code", "n_trials", "n_errors"):
        assert key in m0
    assert (m0["event"], m0["return_code"], m0["n_trials"], m0["n_errors"], m0["complete"]) == ("run", 0, 2, 0, True)
    assert m0["t_end"] >= m0["t_start"] and m0["billing_error_trials"] == []
    assert m0["config_path"] == f"jobs/t/configs/{m0['condition']}.yaml"

    sj = json.loads((spec.jobs_dir / "sweep.json").read_text())
    assert sj["name"] == "t" and sj["status"] == "complete"
    assert (sj["jobs_total"], sj["jobs_complete"]) == (4, 4)
    assert sj["dataset"] == "terminal-bench@2.0" and sj["tasks"] == ["fix-git", "prove-plus-comm"]
    assert sj["order_seed"] == 0 and sj["rounds"] == 2
    assert sj["condition_order"] == {"1": order[1], "2": order[2]}
    assert sj["conditions"]["par"]["config_path"] == "jobs/t/configs/par.yaml"
    assert len(sj["conditions"]["par"]["config_sha256"]) == 64
    assert sj["spec"]["name"] == "t" and sj["spec_path"] == "configs/sweeps/t.yaml"
    assert sj["provenance"]["git_commit"] == "deadbeef" * 5 and sj["provenance"]["git_dirty"] is True
    assert sj["provenance"]["python_version"] and "harbor_version" in sj["provenance"]
    assert "lockfile_sha256" in sj["provenance"] and sj["provenance"]["date"]
    assert sj["provenance"]["docker"]["architecture"] == "aarch64"
    assert {"model": "orch-m", "provider": "sail", "completion_window": "balanced"} in sj["provider_tiers"]
    assert sj["tasks_meta_path"] == "jobs/t/tasks_meta.json"
    meta = json.loads((spec.jobs_dir / "tasks_meta.json").read_text())
    assert list(meta["tasks"]) == ["fix-git", "prove-plus-comm"]
    assert len(sj["invocations"]) == 1 and sj["invocations"][0]["n_ran"] == 4

    # Second invocation: everything complete -> skipped, nothing launched.
    fake_harbor.popen_calls.clear()
    assert sweep.run_sweep(spec, out=lines.append) == 0
    assert fake_harbor.popen_calls == []
    man = read_manifest(spec)
    assert [m["event"] for m in man[4:]] == ["skipped_complete"] * 4
    assert len(json.loads((spec.jobs_dir / "sweep.json").read_text())["invocations"]) == 2


def test_run_sweep_moves_partial_job_aside(tmp_path, fake_harbor, stub_config):
    spec = sweep.load_spec(make_project(tmp_path))
    make_job(spec.jobs_dir / "t__r1__par", spec.tasks, mode="partial")
    assert sweep.run_sweep(spec, only=["par"], rounds=[1], out=lambda s: None) == 0
    sj = json.loads((spec.jobs_dir / "sweep.json").read_text())
    assert (sj["status"], sj["jobs_total"], sj["jobs_complete"]) == ("complete", 4, 1)
    moved = spec.jobs_dir / "_partial" / "t__r1__par.partial.1"
    assert json.loads((moved / "result.json").read_text())["finished_at"] is None  # the old partial job
    assert sweep.job_status(spec.jobs_dir / "t__r1__par").complete
    man = read_manifest(spec)
    assert [(m["event"], m["job"]) for m in man] == [("moved_partial", "t__r1__par"), ("run", "t__r1__par")]
    assert man[0]["moved_to"] == "jobs/t/_partial/t__r1__par.partial.1"


def test_run_sweep_aborts_on_billing_error(tmp_path, fake_harbor, stub_config):
    spec = sweep.load_spec(make_project(tmp_path))
    order = sweep.condition_order(["single", "par"], 2, 0)
    first = f"t__r1__{order[1][0]}"
    fake_harbor.modes[first] = "billing"
    lines: list[str] = []
    assert sweep.run_sweep(spec, out=lines.append) == sweep.EXIT_BILLING
    assert len(fake_harbor.popen_calls) == 1  # nothing after the billing error
    man = read_manifest(spec)
    assert man[-1]["billing_error_trials"] == ["fix-git__abc0"]
    assert json.loads((spec.jobs_dir / "sweep.json").read_text())["status"] == "aborted_billing"
    assert any("ABORT: billing error" in s for s in lines)


def test_run_sweep_stops_after_consecutive_failures(tmp_path, fake_harbor, stub_config):
    spec = sweep.load_spec(make_project(tmp_path))
    fake_harbor.rc = 1
    for r in (1, 2):
        for c in ("single", "par"):
            fake_harbor.modes[f"t__r{r}__{c}"] = "partial"
    assert sweep.run_sweep(spec, max_consecutive_failures=2, out=lambda s: None) == sweep.EXIT_FAIL
    assert len(fake_harbor.popen_calls) == 2
    assert json.loads((spec.jobs_dir / "sweep.json").read_text())["status"] == "aborted_failures"


def test_run_sweep_refuses_bad_configs_and_config_changes(tmp_path, fake_harbor, monkeypatch):
    spec = sweep.load_spec(make_project(tmp_path))

    def broken(cfg):
        raise TypeError("unexpected keyword argument 'scope'")

    monkeypatch.setattr(sweep, "load_experiment_config", broken)
    lines: list[str] = []
    assert sweep.run_sweep(spec, out=lines.append) == sweep.EXIT_CONFIG
    assert fake_harbor.popen_calls == [] and not spec.jobs_dir.exists()
    assert any("unexpected keyword argument 'scope'" in s for s in lines)

    # A loader that silently drops a key is caught too.
    class Lossy(StubExperimentConfig):
        def to_dict(self):
            d = super().to_dict()
            d["subagent"].pop("scope", None)
            return d

    monkeypatch.setattr(sweep, "load_experiment_config", Lossy)
    lines.clear()
    assert sweep.run_sweep(spec, out=lines.append) == sweep.EXIT_CONFIG
    assert any("subagent.scope: dropped by ExperimentConfig" in s for s in lines)

    # Resume with a changed condition config is refused.
    monkeypatch.setattr(sweep, "load_experiment_config", StubExperimentConfig)
    assert sweep.run_sweep(spec, only=["par"], rounds=[1], out=lambda s: None) == 0
    spec.conditions["par"] = {"subagent": {"strategy": "sequential"}}
    with pytest.raises(sweep.SpecError, match="differs"):
        sweep.run_sweep(spec, out=lambda s: None)
    assert sweep.run_sweep(spec, only=["par"], rounds=[1], allow_config_change=True, out=lambda s: None) == 0


def test_dry_run_prints_plan_and_writes_nothing(tmp_path, fake_harbor, stub_config):
    spec = sweep.load_spec(make_project(tmp_path))
    make_job(spec.jobs_dir / "t__r1__single", spec.tasks)
    lines: list[str] = []
    assert sweep.run_sweep(spec, dry_run=True, out=lines.append) == 0
    text = "\n".join(lines)
    assert fake_harbor.popen_calls == []
    assert sorted(p.name for p in spec.jobs_dir.iterdir()) == ["t__r1__single"]  # nothing written
    assert "Plan: 4 jobs (3 to run, 1 complete -> skip, 0 partial -> move aside and rerun); 6 episodes to run" in text
    assert "Config validation: OK (2 conditions" in text
    assert "SUBAGENTS_BENCH_CONFIG=jobs/t/configs/par.yaml /fake/bin/harbor run -d terminal-bench@2.0 -i fix-git -i prove-plus-comm" in text
    assert "--job-name t__r2__par --agent-timeout-multiplier 4.0 -y -q" in text
    assert "ROUGH estimate" in text and "NOT a measurement" in text
    assert "TOTAL:" in text and "Runtime:" in text


def test_dry_run_reports_config_problems(tmp_path, fake_harbor, monkeypatch):
    spec = sweep.load_spec(make_project(tmp_path))
    monkeypatch.setattr(sweep, "load_experiment_config", lambda cfg: (_ for _ in ()).throw(TypeError("boom")))
    lines: list[str] = []
    assert sweep.run_sweep(spec, dry_run=True, out=lines.append) == sweep.EXIT_CONFIG
    assert any("single: ExperimentConfig.from_dict failed: TypeError: boom" in s for s in lines)


def test_real_pilot_dry_run_counts(fake_harbor, stub_config, monkeypatch):
    spec = sweep.load_spec(SWEEPS / "pilot.yaml")
    monkeypatch.setattr(spec, "jobs_dir", Path("/nonexistent/jobs/pilot"))
    lines: list[str] = []
    sweep.run_sweep(spec, dry_run=True, out=lines.append)
    assert "Plan: 45 jobs (45 to run, 0 complete -> skip, 0 partial -> move aside and rerun); 450 episodes to run" in "\n".join(lines)


# ---------------------------------------------------------------------------
# rough estimate
# ---------------------------------------------------------------------------


def test_estimate_episode_shapes():
    spec = sweep.load_spec(SWEEPS / "pilot.yaml")
    res = sweep.resolve_conditions(spec)
    single = sweep.estimate_episode(res["single_5step"])
    assert single.calls == {"orch": 5, "sub": 0.0, "arb": 0.0, "deb": 0.0}
    rm = sweep.RoughModel()
    prompt = 5 * rm.orch_prompt_base + rm.orch_prompt_growth * 10
    expect = ((1 - rm.cached_frac) * prompt * 0.07 + rm.cached_frac * prompt * 0.005 + 5 * rm.orch_completion * 0.22) / 1e6
    assert single.cost["orch"] == pytest.approx(expect)
    par = sweep.estimate_episode(res["gemma31b_parallel_hard"])
    seq = sweep.estimate_episode(res["gemma31b_sequential_hard"])
    arb = sweep.estimate_episode(res["gemma31b_arbiter_hard"])
    up = sweep.estimate_episode(res["gemma31b_parallel_hard"], upper=True)
    assert par.calls["sub"] == 7.5 * 6 and up.calls["sub"] == 15 * 8
    assert up.cost_total > par.cost_total > single.cost_total
    assert seq.seconds > par.seconds  # sequential runs dispatches one at a time
    assert arb.calls["arb"] > 0 and arb.calls["deb"] > 0
    nop = copy.deepcopy(res["single_5step"])
    nop["orchestrator"]["price_in"] = None
    assert sweep.estimate_episode(nop).cost_total is None  # unknown price -> unknown cost


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def _fake_llm_module(fail_models=()):
    mod = types.ModuleType("subagents_bench.llm")
    seen = []

    class Client:
        def __init__(self, cfg):
            self.cfg = cfg

        def complete(self, system, messages, tools):
            seen.append((self.cfg.model, self.cfg.max_output_tokens, messages[0].content, tools))
            if self.cfg.model in fail_models:
                raise RuntimeError("HTTP 402: insufficient balance")
            return types.SimpleNamespace(
                text="OK",
                finish_reason="stop",
                usage=types.SimpleNamespace(prompt_tokens=12, completion_tokens=3, cached_tokens=0, reasoning_tokens=None),
                timing=types.SimpleNamespace(t_request_sent=100.0, t_first_chunk=100.5, streamed=True, n_chunks=4),
                attempts=[object()],
                usage_source="provider",
            )

    mod.make_client = Client
    mod.seen = seen
    return mod


def test_preflight_all_checks(tmp_path, fake_harbor, stub_config, monkeypatch):
    spec = sweep.load_spec(make_project(tmp_path))
    monkeypatch.setenv("SAIL_API_KEY", "sk-secret-value")
    llm = _fake_llm_module()
    monkeypatch.setitem(sys.modules, "subagents_bench.llm", llm)
    monkeypatch.setattr(sweep, "harbor_version", lambda h=None: sweep.PINNED_HARBOR_VERSION)
    lines: list[str] = []
    assert sweep.run_preflight(spec, out=lines.append) == 0
    pf = json.loads((spec.jobs_dir / "preflight.json").read_text())
    assert pf["passed"] is True
    names = [c["name"] for c in pf["checks"]]
    assert names[:4] == ["docker running", "harbor installed", "configs load", "dataset + tasks"]
    assert "credentials: sail" in names
    # One request per distinct (model, provider, tier): orch-m and sub-m.
    assert sorted((r["model"], r["provider"], r["completion_window"]) for r in pf["requests"]) == [
        ("orch-m", "sail", "balanced"), ("sub-m", "sail", "balanced")]
    assert all(r["ok"] and r["ttft_sec"] == 0.5 and r["prompt_tokens"] == 12 for r in pf["requests"])
    assert all(max_tok == sweep.PING_MAX_TOKENS and tools == [] for _, max_tok, _, tools in llm.seen)
    meta = json.loads((spec.jobs_dir / "tasks_meta.json").read_text())
    assert list(meta["tasks"]) == ["fix-git", "prove-plus-comm"]
    text = "\n".join(lines) + (spec.jobs_dir / "preflight.json").read_text()
    assert "sk-secret-value" not in text  # credentials are never printed or stored
    assert "[PASS] credentials: sail: SAIL_API_KEY set" in "\n".join(lines)


def test_preflight_failures_exit_nonzero(tmp_path, fake_harbor, stub_config, monkeypatch):
    spec = sweep.load_spec(make_project(tmp_path, tasks=["fix-git", "no-such-task"]))
    monkeypatch.setenv("SAIL_API_KEY", "")
    monkeypatch.setitem(sys.modules, "subagents_bench.llm", _fake_llm_module(fail_models=("sub-m",)))
    monkeypatch.setattr(sweep, "harbor_version", lambda h=None: "0.23.0")
    fake_harbor_docker_down = fake_harbor.run

    def run(cmd, **kw):
        if cmd[0] == "docker":
            return subprocess.CompletedProcess(cmd, 1, "", "Cannot connect to the Docker daemon")
        return fake_harbor_docker_down(cmd, **kw)

    monkeypatch.setattr(sweep.subprocess, "run", run)
    lines: list[str] = []
    assert sweep.run_preflight(spec, out=lines.append) == 1
    checks = {c["name"]: c for c in json.loads((spec.jobs_dir / "preflight.json").read_text())["checks"]}
    assert checks["docker running"]["ok"] is False
    assert checks["harbor version pinned"]["ok"] is False
    assert checks["dataset + tasks"]["ok"] is False and "no-such-task" in checks["dataset + tasks"]["detail"]
    assert checks["credentials: sail"]["ok"] is False and checks["credentials: sail"]["detail"] == "SAIL_API_KEY missing"
    assert checks["request orch-m @ sail/balanced"]["ok"] is True
    assert checks["request sub-m @ sail/balanced"]["ok"] is False
    assert "insufficient balance" in checks["request sub-m @ sail/balanced"]["detail"]
    assert any(s.startswith("  [FAIL] docker running") for s in lines)


def test_preflight_oracle_reports_failing_tasks_and_reuses_job(tmp_path, fake_harbor, stub_config, monkeypatch):
    spec = sweep.load_spec(make_project(tmp_path))
    monkeypatch.setenv("SAIL_API_KEY", "x")
    monkeypatch.setattr(sweep, "harbor_version", lambda h=None: sweep.PINNED_HARBOR_VERSION)

    def oracle_popen(cmd, env=None, cwd=None, **kw):
        fake_harbor.popen_calls.append((list(cmd), env, cwd))
        job_dir = Path(cwd) / _arg(cmd, "-o") / _arg(cmd, "--job-name")
        make_job(job_dir, ["fix-git", "prove-plus-comm"])
        res = job_dir / "prove-plus-comm__abc1" / "result.json"
        d = json.loads(res.read_text())
        d["verifier_result"]["rewards"]["reward"] = 0.0
        res.write_text(json.dumps(d))
        return types.SimpleNamespace(wait=lambda timeout=None: 0)

    monkeypatch.setattr(sweep.subprocess, "Popen", oracle_popen)
    lines: list[str] = []
    assert sweep.run_preflight(spec, oracle=True, skip_llm=True, out=lines.append) == 1
    cmd = fake_harbor.popen_calls[0][0]
    assert cmd[cmd.index("-a") + 1] == "oracle" and _arg(cmd, "--job-name") == "_oracle"
    pf = json.loads((spec.jobs_dir / "preflight.json").read_text())
    assert pf["oracle"]["failing_tasks"] == ["prove-plus-comm"]
    assert pf["oracle"]["mean_reward"] == 0.5
    # Second preflight reuses the complete _oracle job instead of rerunning.
    sweep.run_preflight(spec, oracle=True, skip_llm=True, out=lines.append)
    assert len(fake_harbor.popen_calls) == 1
    assert json.loads((spec.jobs_dir / "preflight.json").read_text())["oracle"]["reused"] is True
    # A task added to the spec makes the old oracle job stale: moved aside, rerun.
    spec.tasks = ["fix-git", "prove-plus-comm", "crack-7z-hash"]
    ok, info = sweep.run_oracle(spec, spec.tasks, "/fake/bin/harbor", out=lines.append)
    assert len(fake_harbor.popen_calls) == 2 and info["reused"] is False
    assert info["moved_partial_to"] == "jobs/t/_partial/_oracle.partial.1"


def test_register_cli_parses_subcommands():
    import argparse

    parser = argparse.ArgumentParser()
    sweep.register_cli(parser.add_subparsers(dest="cmd"))
    a = parser.parse_args(["sweep", "configs/sweeps/pilot.yaml", "--dry-run", "--only", "single_5step", "samemodel_parallel_hard", "--rounds", "1,3"])
    assert (a.cmd, a.dry_run, a.only, a.rounds, a.max_consecutive_failures) == ("sweep", True, ["single_5step", "samemodel_parallel_hard"], [1, 3], 3)
    p = parser.parse_args(["preflight", "configs/sweeps/smoke.yaml", "--oracle"])
    assert (p.cmd, p.oracle, p.skip_llm) == ("preflight", True, False)
    t = parser.parse_args(["tasks"])
    assert t.dataset == "terminal-bench@2.0"


# ---------------------------------------------------------------------------
# opt-in: real provider / real harbor + Docker
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_ping_model_live_sail():
    """One tiny real request through llm.make_client (RUN_LIVE=1)."""
    from subagents_bench.config import ModelConfig

    sweep.load_env_files(REPO / ".env")
    rec = sweep.ping_model(ModelConfig(model="deepseek-ai/DeepSeek-V4.1-Flash", provider="sail",
                                       completion_window="balanced", temperature=0.7))
    assert rec["ok"] is True, rec
    assert rec["streamed"] is True and rec["prompt_tokens"] > 0 and rec["finish_reason"]


@pytest.mark.docker
def test_oracle_job_through_real_harbor(tmp_path):
    """Real `harbor run -a oracle` on fix-git via run_oracle (RUN_DOCKER=1;
    pulls the task image on first use)."""
    spec = sweep.load_spec(make_project(tmp_path, tasks=["fix-git"], harbor={"env": "docker", "n_concurrent": 1}))
    harbor = sweep.harbor_executable()
    assert harbor is not None
    ok, info = sweep.run_oracle(spec, ["fix-git"], harbor, out=lambda s: None)
    assert ok is True, info
    assert info["mean_reward"] == 1.0 and info["failing_tasks"] == []
    st = sweep.job_status(spec.jobs_dir / sweep.ORACLE_JOB)
    assert st.complete and (st.n_total, st.n_completed, st.n_errored) == (1, 1, 0)
