from __future__ import annotations

from types import SimpleNamespace

from subagents_bench.config import ModelConfig
from subagents_bench.llm import make_client, _reported_cost, _cost
from subagents_bench.telemetry import Usage


def test_prime_provider_fills_defaults():
    c = make_client(ModelConfig(model="Qwen/Qwen3-8B", provider="prime"))
    assert c.cfg.base_url == "https://api.pinference.ai/api/v1"
    assert c.cfg.api_key_env == "PRIME_API_KEY"


def test_sail_provider_fills_defaults():
    c = make_client(ModelConfig(model="gpt-oss-120b", provider="sail"))
    assert c.cfg.base_url == "https://api.sailresearch.com/v1"
    assert c.cfg.api_key_env == "SAIL_API_KEY"


def test_explicit_base_url_wins():
    c = make_client(
        ModelConfig(model="m", provider="prime", base_url="http://localhost:9/v1")
    )
    assert c.cfg.base_url == "http://localhost:9/v1"


def test_reported_cost_variants():
    assert _reported_cost(SimpleNamespace(cost=0.0123)) == 0.0123
    assert _reported_cost(SimpleNamespace(cost=SimpleNamespace(total_usd=0.5))) == 0.5
    assert _reported_cost(SimpleNamespace(cost={"usd": 0.25})) == 0.25
    assert _reported_cost(SimpleNamespace()) is None
    assert _reported_cost(None) is None


def test_config_price_fallback():
    cfg = ModelConfig(model="m", provider="openai", price_in=1.0, price_out=2.0)
    cost = _cost(cfg, Usage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert cost == 3.0
