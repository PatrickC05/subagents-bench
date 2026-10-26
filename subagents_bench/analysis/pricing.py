"""Per-call cost from token counts and per-model prices (``configs/pricing.yaml``)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICING = REPO_ROOT / "configs" / "pricing.yaml"


def load_prices(path: str | Path = DEFAULT_PRICING, scheme: str = "default") -> dict[str, dict]:
    """``{model: {in, cached, out}}`` (USD per 1M tokens) of one scheme."""
    return yaml.safe_load(Path(path).read_text())[scheme]


def call_cost(call: dict, prices: dict[str, dict[str, float]]) -> float:
    """USD of one call; NaN if its model has no price."""
    p = prices.get(call["model"])
    if p is None:
        return float("nan")
    prompt, cached, completion = call["prompt_tokens"], call["cached_tokens"], call["completion_tokens"]
    return ((prompt - cached) * p["in"] + cached * p["cached"] + completion * p["out"]) / 1e6
