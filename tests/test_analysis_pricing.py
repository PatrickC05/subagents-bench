"""Price schemes, aliases, includes and FLOPs; hand-computed costs for one episode."""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import GEMMA, ORCH, C, EpisodeBuilder, Spawn  # noqa: E402

from subagents_bench.analysis.load import load_trials  # noqa: E402
from subagents_bench.analysis.pricing import ModelCatalog, load_models, load_pricing  # noqa: E402
from subagents_bench.analysis.tables import episode_row  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PRICING = REPO / "configs" / "pricing.yaml"


@pytest.fixture
def pricing_copy(tmp_path):
    """configs/pricing.yaml alone in a temp dir (its reference include is then absent)."""
    dst = tmp_path / "pricing.yaml"
    shutil.copy(PRICING, dst)
    return dst


@pytest.fixture
def models_yaml(tmp_path):
    # Fixture values for the FLOPs arithmetic; not real parameter counts.
    p = tmp_path / "models.yaml"
    p.write_text(
        "models:\n"
        f"  {ORCH}: {{active_params_b: 16.0, active_params_prefill_b: 8.0, active_params_decode_b: 16.0}}\n"
        f"  {GEMMA}: {{active_params_b: 30.0}}\n"
        "aliases:\n"
        f"  gemma-alias: {GEMMA}\n"
    )
    return p


def _episode(tmp_path):
    b = EpisodeBuilder(t0=1000.0, strategy="parallel")
    b.step(C(2.0, prompt=10_000, cached=2_560, completion=1_000, reasoning=400),
           [Spawn(members=[[C(1.0, prompt=4_000, cached=1_024, completion=500)]])])
    b.step(C(1.0, prompt=0, completion=0), finish=True)
    b.finish()
    b.write(tmp_path / "ep")
    (trial,) = load_trials([tmp_path / "ep"])  # a bare episode dir (local-runner layout)
    assert trial.ref.trial_dir is None and trial.condition == "cond"
    return trial


def test_pricing_yaml_schemes_and_provenance(pricing_copy):
    pb = load_pricing(pricing_copy)
    assert pb.scheme_names == ["sail_balanced", "sail_asap", "sail_flex", "prime"]
    assert any("pricing_reference.yaml" in w for w in pb.warnings)  # missing include tolerated
    sb = pb.schemes["sail_balanced"]
    assert sb.meta["retrieved"] == "2026-10-11" and sb.meta["source"] == "provider pricing pages"
    p = pb.price("sail_balanced", "google/gemma-4-12B-it")
    assert (p.in_, p.cached, p.out) == (0.10, 0.07, 2.00)
    p = pb.price("sail_asap", "zai-org/GLM-5.3-Flash")
    assert (p.in_, p.cached, p.out) == (0.11, 0.02, 0.35)
    p = pb.price("sail_flex", GEMMA)
    assert (p.in_, p.cached, p.out) == (0.07, 0.04, 0.20)
    assert pb.price("sail_flex", "google/gemma-4-31B-it") is None  # the BF16 checkpoint is not in the scheme
    p = pb.price("prime", "Qwen/Qwen3.5-2B")
    assert (p.in_, p.cached, p.out) == (0.06, 0.054, 0.18)
    assert pb.schemes["prime"].cached_fallback == ["deepseek/deepseek-v4.1-flash"]
    # Alias from pricing.yaml: Sail's DeepSeek id prices at Prime's DeepSeek entry.
    p = pb.price("prime", ORCH)
    assert (p.in_, p.cached, p.out) == (0.30, None, 1.20)


def test_cost_per_scheme_hand_computed(tmp_path, pricing_copy):
    data = _episode(tmp_path)
    pb = load_pricing(pricing_copy)
    row = episode_row(data, pb, [])
    # orchestrator: 10000 prompt (2560 cached, 7440 uncached), 1000 completion; plus a 0-token call
    # subagent: 4000 prompt (1024 cached, 2976 uncached), 500 completion
    assert row["cost_sail_balanced_orch"] == pytest.approx((7440 * 0.07 + 2560 * 0.005 + 1000 * 0.22) / 1e6)  # 753.6e-6
    assert row["cost_sail_balanced_sub"] == pytest.approx((2976 * 0.11 + 1024 * 0.06 + 500 * 0.32) / 1e6)  # 548.8e-6
    assert row["cost_sail_balanced_total"] == pytest.approx(1302.4e-6)
    assert row["cost_sail_asap_orch"] == pytest.approx((7440 * 0.09 + 2560 * 0.006 + 1000 * 0.30) / 1e6)
    assert row["cost_sail_asap_sub"] == pytest.approx((2976 * 0.14 + 1024 * 0.07 + 500 * 0.40) / 1e6)
    assert row["cost_sail_flex_orch"] == pytest.approx((7440 * 0.05 + 2560 * 0.004 + 1000 * 0.15) / 1e6)
    # Prime: cached price not published -> cached tokens at the input price; DeepSeek via alias.
    assert row["cost_prime_orch"] == pytest.approx((10_000 * 0.30 + 1000 * 1.20) / 1e6)  # 4.2e-3
    assert row["cost_sail_flex_sub"] == pytest.approx((2976 * 0.07 + 1024 * 0.04 + 500 * 0.20) / 1e6)
    # Gemma 31B has no price under prime: NaN, never 0, and the total is NaN too.
    assert math.isnan(row["cost_prime_sub"]) and math.isnan(row["cost_prime_total"])
    # Roles without calls cost exactly 0 (an observation, not a fill-in).
    assert row["cost_sail_balanced_arb"] == 0.0 and row["cost_sail_balanced_deb"] == 0.0
    # The runtime price table of the synthetic data equals sail_balanced.
    assert row["cost_runtime_total"] == pytest.approx(row["cost_sail_balanced_total"])


def test_flops_split_and_missing_models(tmp_path, models_yaml, pricing_copy):
    data = _episode(tmp_path)
    catalog, warnings = load_models(models_yaml)
    assert warnings == []
    row = episode_row(data, load_pricing(pricing_copy, catalog), [])
    # DeepSeek: 8B per prompt token, 16B per completion token; Gemma: 30B for both.
    assert row["flops_orch"] == pytest.approx(2e9 * (8.0 * 10_000 + 16.0 * 1_000))
    assert row["flops_sub"] == pytest.approx(2e9 * 30.0 * 4_500)
    assert row["flops_total"] == pytest.approx(row["flops_orch"] + row["flops_sub"])
    assert row["flops_arb"] == 0.0
    # No models.yaml -> FLOPs unknown (NaN), with a warning.
    catalog, warnings = load_models(tmp_path / "nope.yaml")
    assert warnings and "not found" in warnings[0]
    row = episode_row(data, load_pricing(pricing_copy, catalog), [])
    assert math.isnan(row["flops_orch"]) and math.isnan(row["flops_total"])


def test_aliases_both_directions_and_case(models_yaml):
    catalog, _ = load_models(models_yaml)
    table = {GEMMA: "g"}
    assert catalog.lookup(table, "gemma-alias") == "g"
    assert catalog.lookup({"gemma-alias": "a"}, GEMMA) == "a"
    assert catalog.lookup(table, GEMMA.lower()) == "g"  # case-insensitive fallback
    assert catalog.lookup(table, "other/model") is None
    assert catalog.active_params_split_b("gemma-alias") == (30.0, 30.0)
    assert ModelCatalog().active_params_b(ORCH) is None


def test_include_flat_form_and_incomplete_entries(tmp_path):
    (tmp_path / "ref.yaml").write_text(
        "scheme: reference\nretrieved: '2026-10-11'\nmethod: median\n"
        f"models:\n  {ORCH}: {{in: 0.21, cached: 0.007, out: 1.188, n_providers: 29, source: x}}\n"
        f"  {GEMMA}: {{in: null, cached: null, out: null}}\n"
    )
    (tmp_path / "pricing.yaml").write_text(
        "include: [ref.yaml]\n"
        f"schemes:\n  flat: {{{ORCH}: {{in: 1.0, cached: null, out: 2.0}}}}\n"
    )
    pb = load_pricing(tmp_path / "pricing.yaml")
    assert pb.warnings == []
    assert pb.scheme_names == ["flat", "reference"]
    assert pb.schemes["reference"].meta["method"] == "median"
    assert pb.price("reference", ORCH).out == 1.188
    assert pb.price("reference", GEMMA) is None  # all-null entry = no price
    call = {"model": ORCH, "prompt_tokens": 1000, "cached_tokens": 500, "uncached_prompt_tokens": 500,
            "completion_tokens": 100}
    assert pb.call_cost("flat", call) == pytest.approx((1000 * 1.0 + 100 * 2.0) / 1e6)
    assert math.isnan(pb.call_cost("reference", {**call, "model": GEMMA}))


def test_missing_pricing_file_gives_no_schemes(tmp_path):
    pb = load_pricing(tmp_path / "absent.yaml")
    assert pb.scheme_names == [] and "not found" in pb.warnings[0]


def test_bad_price_entry_raises(tmp_path):
    (tmp_path / "pricing.yaml").write_text(f"schemes:\n  s: {{{ORCH}: 0.5}}\n")
    with pytest.raises(ValueError, match="expected a mapping"):
        load_pricing(tmp_path / "pricing.yaml")
