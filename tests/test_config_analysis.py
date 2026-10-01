from __future__ import annotations

import json
from pathlib import Path

from subagents_bench.config import ExperimentConfig, Topology
from subagents_bench.analysis import summarize, pareto_points, load_trials


def test_config_yaml_roundtrip(tmp_path):
    src = Path("configs/unified.yaml")
    cfg = ExperimentConfig.from_yaml(src)
    assert cfg.topology is Topology.SUBAGENTS
    assert cfg.resolved_subagent_model().model == "Qwen/Qwen3-8B"
    # round-trip through dict
    cfg2 = ExperimentConfig.from_dict(cfg.to_dict())
    assert cfg2.name == cfg.name
    assert cfg2.budget.orchestrator_steps == 5


def test_same_model_baseline_detection():
    cfg = ExperimentConfig.from_yaml("configs/baselines/same_model_subagents.yaml")
    assert cfg.is_same_model_baseline() is True
    single = ExperimentConfig.from_yaml("configs/baselines/single_agent.yaml")
    assert single.is_same_model_baseline() is False


def test_analysis_summarize_and_pareto(tmp_path):
    # Fabricate two configs, 2 trials each.
    def trial(exp, resolved, tokens, wall, disp):
        return {
            "experiment": exp,
            "benchmark": "terminal_bench",
            "orchestrator_model": "orch",
            "subagent_model": "sub" if exp == "B" else "orch",
            "topology": "subagents",
            "subtask_type": "none",
            "resolved": resolved,
            "total_tokens": tokens,
            "cost_usd": tokens / 1e6,
            "wall_clock_sec": wall,
            "subagent_dispatches": disp,
            "subagent_redispatches": max(disp - 1, 0),
            "instruction_following": 1.0,
        }

    for i, t in enumerate(
        [
            trial("A", True, 1000, 10, 0),
            trial("A", False, 1200, 12, 0),
            trial("B", True, 3000, 20, 2),
            trial("B", True, 3200, 22, 3),
        ]
    ):
        d = tmp_path / t["experiment"] / f"task{i}"
        d.mkdir(parents=True)
        (d / "telemetry.json").write_text(json.dumps(t))

    trials = load_trials(tmp_path)
    assert len(trials) == 4
    summaries = {s.experiment: s for s in summarize(trials)}
    assert summaries["A"].resolve_rate == 0.5
    assert summaries["B"].resolve_rate == 1.0

    pts = pareto_points(summarize(trials), x="mean_total_tokens")
    # Sorted ascending by tokens: A (1100) before B (3100).
    assert pts[0]["experiment"] == "A"
    assert pts[0]["mean_total_tokens"] < pts[1]["mean_total_tokens"]
