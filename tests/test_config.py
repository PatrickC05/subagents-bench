"""ExperimentConfig: DESIGN section 2 defaults, YAML compatibility, full
round trip, enum coercion and clear validation errors."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from subagents_bench.config import (
    ArbiterConfig,
    BudgetConfig,
    ConfigError,
    ExperimentConfig,
    ModelConfig,
    SandboxConfig,
    Scope,
    Strategy,
    SubagentConfig,
    ToolsConfig,
    Topology,
)

REPO = Path(__file__).resolve().parent.parent
YAMLS = [
    "configs/unified.yaml",
    "configs/exploitbench.yaml",
    "configs/baselines/single_agent.yaml",
    "configs/baselines/same_model_subagents.yaml",
    "configs/sail/base.yaml",
]


def test_defaults_match_design():
    s = SubagentConfig()
    assert (s.strategy, s.scope) == (Strategy.PARALLEL, Scope.HARD)
    assert (s.max_concurrent, s.max_dispatches, s.max_steps, s.token_budget, s.max_output_chars) == (3, None, 8, None, 12000)
    a = ArbiterConfig()
    assert (a.n_candidates, a.debate_rounds, a.confidence_threshold, a.model, a.max_copy_mb) == (3, 1, 0.8, None, 500)
    b = BudgetConfig()
    assert (b.orchestrator_steps, b.total_token_budget, b.timeout_sec, b.command_timeout_sec, b.max_command_timeout_sec) == (
        5, 5_000_000, 3600, 180, 900,
    )
    assert ToolsConfig().max_output_chars == 12000
    assert SandboxConfig().workdir == "auto"
    m = ModelConfig(model="x")
    assert m.retry_budget_sec == 120.0
    assert (m.reasoning_passback, m.max_retries, m.request_timeout_sec, m.stream, m.context_window) == (True, 4, 180.0, True, None)


@pytest.mark.parametrize("path", YAMLS)
def test_repo_yaml_loads_and_round_trips(path):
    raw = yaml.safe_load((REPO / path).read_text())
    cfg = ExperimentConfig.from_yaml(REPO / path)
    assert cfg.name == raw["name"]
    full = cfg.to_dict()
    again = ExperimentConfig.from_dict(full)
    assert again.to_dict() == full
    assert again == cfg
    # Every key the YAML sets survives with its value (None may become a default).
    def check(r, f, where):
        for k, v in r.items():
            assert k in f, f"{where}{k} dropped"
            if isinstance(v, dict):
                check(v, f[k], f"{where}{k}.")
            elif v is not None:
                assert f[k] == v, f"{where}{k}: {v!r} -> {f[k]!r}"
    check(raw, full, "")


def test_base_yaml_values():
    cfg = ExperimentConfig.from_yaml(REPO / "configs/sail/base.yaml")
    assert cfg.topology is Topology.SUBAGENTS
    assert cfg.subagent.strategy is Strategy.PARALLEL and cfg.subagent.scope is Scope.HARD
    assert cfg.subagent.max_dispatches == 15 and cfg.resolved_max_dispatches() == 15
    assert cfg.subagent.model.context_window == 262144
    assert cfg.orchestrator.completion_window == "balanced"
    assert cfg.resolved_arbiter_model() is cfg.orchestrator
    assert cfg.sandbox.workdir == "auto"


def test_nested_arbiter_model_round_trip():
    d = {
        "name": "arb",
        "subagent": {
            "strategy": "arbiter",
            "scope": "medium",
            "arbiter": {"n_candidates": 2, "model": {"model": "judge-m", "provider": "sail", "price_in": 0.5}},
        },
    }
    cfg = ExperimentConfig.from_dict(d)
    assert cfg.subagent.strategy is Strategy.ARBITER and cfg.subagent.scope is Scope.MEDIUM
    assert cfg.resolved_arbiter_model().model == "judge-m"
    full = cfg.to_dict()
    assert full["subagent"]["arbiter"]["model"]["model"] == "judge-m"
    assert full["subagent"]["arbiter"]["model"]["price_in"] == 0.5
    assert full["subagent"]["strategy"] == "arbiter"  # enums serialize as values
    assert ExperimentConfig.from_dict(full).resolved_arbiter_model() == cfg.resolved_arbiter_model()
    yaml.safe_dump(full)  # plain YAML types only


def test_resolved_models_and_dispatch_budget():
    cfg = ExperimentConfig(name="x")
    assert cfg.resolved_subagent_model() is cfg.orchestrator
    assert cfg.resolved_arbiter_model() is cfg.orchestrator
    assert cfg.is_same_model_baseline()
    assert cfg.resolved_max_dispatches() == 3 * 5
    cfg.subagent.max_dispatches = 7
    assert cfg.resolved_max_dispatches() == 7
    cfg.subagent.model = ModelConfig(model="small")
    assert cfg.resolved_subagent_model().model == "small" and not cfg.is_same_model_baseline()


def test_string_enums_are_coerced_in_constructors():
    s = SubagentConfig(strategy="sequential", scope="easy")
    assert s.strategy is Strategy.SEQUENTIAL and s.scope is Scope.EASY
    cfg = ExperimentConfig(name="x", topology="single")
    assert cfg.topology is Topology.SINGLE


@pytest.mark.parametrize(
    "override, needle",
    [
        ({"subagent": {"strategy": "paralel"}}, "subagent.strategy: 'paralel' is not one of parallel, sequential, arbiter"),
        ({"subagent": {"scope": "extreme"}}, "subagent.scope: 'extreme' is not one of easy, medium, hard"),
        ({"topology": "mesh"}, "topology: 'mesh' is not one of single, subagents"),
        ({"sandbox": {"kind": "k8s"}}, "sandbox.kind: 'k8s' is not one of harbor, local_docker, local_shell, exploitbench"),
        ({"subagent": {"max_concurent": 3}}, "unknown key(s) subagent.max_concurent"),
        ({"budget": {"steps": 5}}, "unknown key(s) budget.steps"),
        ({"orchestrator": {"model": "m", "temp": 1}}, "unknown key(s) orchestrator.temp"),
        ({"subagent": {"arbiter": {"rounds": 2}}}, "unknown key(s) subagent.arbiter.rounds"),
        ({"bogus": 1}, "unknown key(s) bogus"),
        ({"subagent": {"arbiter": {"confidence_threshold": 1.5}}}, "confidence_threshold must be in [0, 1]"),
        ({"budget": {"command_timeout_sec": 1000}}, "command_timeout_sec must be <= budget.max_command_timeout_sec"),
        ({"subagent": {"max_concurrent": 0}}, "subagent.max_concurrent must be >= 1"),
        ({"budget": {"orchestrator_steps": 0}}, "budget.orchestrator_steps must be >= 1"),
        ({"orchestrator": {"provider": "sail"}}, "orchestrator.model: required"),
        ({"subagent": "parallel"}, "subagent: expected a mapping"),
    ],
)
def test_bad_values_raise_clear_errors(override, needle):
    d = {"name": "x", **override}
    with pytest.raises(ConfigError) as ei:
        ExperimentConfig.from_dict(d)
    assert needle in str(ei.value)


def test_missing_name_and_non_mapping():
    with pytest.raises(ConfigError, match="name: required"):
        ExperimentConfig.from_dict({"topology": "single"})
    with pytest.raises(ConfigError, match="must be a mapping"):
        ExperimentConfig.from_dict(["x"])  # type: ignore[arg-type]


def test_from_yaml_error_names_the_file(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nsubagent: {scope: huge}\n")
    with pytest.raises(ConfigError) as ei:
        ExperimentConfig.from_yaml(p)
    assert str(p) in str(ei.value) and "subagent.scope" in str(ei.value)


def test_null_means_default_for_non_optional_fields():
    cfg = ExperimentConfig.from_dict(
        {"name": "x", "budget": {"timeout_sec": None}, "subagent": {"max_dispatches": None, "strategy": None}}
    )
    assert cfg.budget.timeout_sec == 3600
    assert cfg.subagent.max_dispatches is None
    assert cfg.subagent.strategy is Strategy.PARALLEL


def test_config_hash_is_stable_and_sensitive():
    a = ExperimentConfig.from_dict({"name": "x"})
    b = ExperimentConfig.from_dict({"name": "x"})
    assert a.config_hash() == b.config_hash() and len(a.config_hash()) == 12
    b.subagent.scope = Scope.EASY
    assert a.config_hash() != b.config_hash()


def test_cli_validate(tmp_path, capsys):
    from subagents_bench.cli import main

    assert main(["validate", str(REPO / "configs/sail/base.yaml"), "--print"]) == 0
    out = capsys.readouterr().out
    assert "[OK]" in out and "subagents: parallel/hard, model nvidia/Gemma-4-31B-IT-NVFP4" in out
    assert "max_dispatches 15" in out and "strategy: parallel" in out  # --print dumps the resolved YAML
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nsubagent: {strategy: paralel}\n")
    assert main(["validate", str(bad)]) == 1
    out = capsys.readouterr().out
    assert out.count(str(bad)) == 1 and "subagent.strategy: 'paralel'" in out


def test_orchestrator_tools_validation_and_resolution():
    from subagents_bench.config import ConfigError, ExperimentConfig

    base = {"name": "x", "orchestrator": {"model": "m", "provider": "sail"}}
    cfg = ExperimentConfig.from_dict(base)
    assert cfg.tools.orchestrator_tools is None
    assert cfg.resolved_orchestrator_tools() == ["bash", "edit_file", "read_file", "list_dir", "search"]
    assert cfg.validate() == []
    mgr = ExperimentConfig.from_dict({**base, "tools": {"orchestrator_tools": ["search", "read_file"]}})
    assert mgr.resolved_orchestrator_tools() == ["read_file", "search"]  # canonical order
    assert mgr.validate() == []
    assert ExperimentConfig.from_dict(mgr.to_dict()).tools.orchestrator_tools == ["search", "read_file"]
    assert mgr.config_hash() != cfg.config_hash()
    with pytest.raises(ConfigError, match=r"unknown tool\(s\) \['spawn_subagents'\]"):
        ExperimentConfig.from_dict({**base, "tools": {"orchestrator_tools": ["bash", "spawn_subagents"]}})
    with pytest.raises(ConfigError, match="cannot change the container"):
        ExperimentConfig.from_dict({**base, "topology": "single", "tools": {"orchestrator_tools": ["read_file"]}})
