"""Offline price schemes and model metadata (cost and FLOPs recomputation).

Costs are recomputed from logged token counts under every scheme in
``configs/pricing.yaml``; the runtime's own ``cost_usd`` is kept only as
``cost_runtime_*``. File formats (``docs/DESIGN.md`` section 6):

``configs/pricing.yaml`` (owned by analysis)::

    include: [pricing_reference.yaml]      # optional, relative to this file
    aliases: {deepseek/deepseek-v4.1-flash: deepseek-ai/DeepSeek-V4.1-Flash}
    schemes:
      sail_balanced:                       # nested form, with provenance
        retrieved: "2026-10-11"
        source: provider pricing pages
        models: {deepseek-ai/DeepSeek-V4.1-Flash: {in: 0.07, cached: 0.005, out: 0.22}}
      other: {some/model: {in: 1.0, cached: null, out: 2.0}}   # flat form also accepted

Included files use the research step's format (``scheme: <name>``, ``models:
{...}``). An included file that does not exist is tolerated (a warning).

``configs/models.yaml`` (owned by research)::

    models: {<id>: {active_params_b, total_params_b, context_window, architecture, sources}}
    aliases: {<alias id>: <canonical id>}

Prices are USD per 1M tokens. ``cached: null`` means no cached-input price is
published: cached tokens are then charged at the input price (and the scheme
lists the model under ``cached_fallback``). A model with no price in a scheme
costs NaN, never 0. FLOPs are ``2 x active_params x tokens``, using
``active_params_prefill_b`` for prompt tokens and ``active_params_decode_b``
for completion tokens when models.yaml gives them (else ``active_params_b``
for both); unknown active params give NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

NAN = float("nan")


def _num(v: Any) -> float | None:
    """YAML number (or numeric string such as '989e12') -> float; None/blank -> None."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, bool):
        raise ValueError(f"expected a number, got {v!r}")
    f = float(v)
    return f if math.isfinite(f) else None


@dataclass
class Price:
    """USD per 1M tokens. ``cached`` None -> cached tokens are charged at ``in_``."""

    in_: float
    out: float
    cached: float | None = None

    def cost(self, uncached: float, cached: float, completion: float) -> float:
        cached_rate = self.cached if self.cached is not None else self.in_
        return (uncached * self.in_ + cached * cached_rate + completion * self.out) / 1e6


@dataclass
class PriceScheme:
    name: str
    prices: dict[str, Price] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)  # retrieved, source, notes, ...

    @property
    def cached_fallback(self) -> list[str]:
        return sorted(m for m, p in self.prices.items() if p.cached is None)


class ModelCatalog:
    """Model metadata plus alias resolution (alias <-> canonical, both directions)."""

    def __init__(self, models: dict[str, dict] | None = None, aliases: dict[str, str] | None = None):
        self.models: dict[str, dict] = dict(models or {})
        self.aliases: dict[str, str] = {}
        self._groups: dict[str, set[str]] = {}
        for a, c in (aliases or {}).items():
            self.add_alias(a, c)

    def add_alias(self, alias: str, canonical: str) -> None:
        self.aliases[alias] = canonical
        group = self._groups.get(alias, {alias}) | self._groups.get(canonical, {canonical})
        for name in group:
            self._groups[name] = group

    def equivalents(self, model: str) -> list[str]:
        """``model`` first, then every name linked to it by aliases (sorted)."""
        return [model] + sorted(self._groups.get(model, {model}) - {model})

    def lookup(self, table: dict[str, Any], model: str | None) -> Any:
        """Value for ``model`` in ``table``, trying aliases and then a case-insensitive match."""
        if not model:
            return None
        for name in self.equivalents(model):
            if name in table:
                return table[name]
        folded = {k.lower(): k for k in table}
        for name in self.equivalents(model):
            if name.lower() in folded:
                return table[folded[name.lower()]]
        return None

    def _field(self, model: str | None, key: str) -> float | None:
        info = self.lookup(self.models, model)
        if not isinstance(info, dict):
            return None
        try:
            return _num(info.get(key))
        except (TypeError, ValueError):
            return None

    def active_params_b(self, model: str | None) -> float | None:
        return self._field(model, "active_params_b")

    def active_params_split_b(self, model: str | None) -> tuple[float, float] | None:
        """(prefill, decode) active params in billions.

        Models whose prompt and generated tokens activate different parameter
        counts (models.yaml ``active_params_prefill_b`` / ``active_params_decode_b``)
        use them; otherwise both equal ``active_params_b``.
        """
        base = self.active_params_b(model)
        pre = self._field(model, "active_params_prefill_b") or base
        dec = self._field(model, "active_params_decode_b") or base
        return (pre, dec) if pre is not None and dec is not None else None

    def bytes_per_param(self, model: str | None) -> float | None:
        """Effective weight bytes read per parameter at decode (models.yaml), if known."""
        return self._field(model, "bytes_per_param")

    def flops(self, model: str | None, prompt_tokens: float, completion_tokens: float) -> float:
        """2 x active params x tokens (prefill/decode split when known); NaN if params unknown."""
        split = self.active_params_split_b(model)
        if split is None:
            return NAN
        pre, dec = split
        return 2.0 * 1e9 * (pre * float(prompt_tokens or 0) + dec * float(completion_tokens or 0))


def load_models(path: str | Path | None) -> tuple[ModelCatalog, list[str]]:
    """(catalog, warnings). A missing file gives an empty catalog (FLOPs -> NaN)."""
    if path is None:
        return ModelCatalog(), ["no models.yaml given: FLOPs and roofline latency are NaN"]
    p = Path(path)
    if not p.is_file():
        return ModelCatalog(), [f"{p} not found: FLOPs and roofline latency are NaN"]
    raw = yaml.safe_load(p.read_text()) or {}
    models = raw.get("models") or {}
    aliases = raw.get("aliases") or {}
    if not isinstance(models, dict) or not isinstance(aliases, dict):
        raise ValueError(f"{p}: 'models' and 'aliases' must be mappings")
    return ModelCatalog({str(k): (v or {}) for k, v in models.items()}, {str(k): str(v) for k, v in aliases.items()}), []


def _parse_price(scheme: str, model: str, entry: Any) -> Price | None:
    if not isinstance(entry, dict):
        raise ValueError(f"pricing scheme {scheme!r}, model {model!r}: expected a mapping, got {entry!r}")
    pin, pout, pc = _num(entry.get("in")), _num(entry.get("out")), _num(entry.get("cached"))
    if pin is None or pout is None:
        return None  # incomplete entry -> treated as "no price" (NaN cost)
    return Price(in_=pin, out=pout, cached=pc)


def _parse_scheme(name: str, body: Any, extra_meta: dict | None = None) -> PriceScheme:
    if not isinstance(body, dict):
        raise ValueError(f"pricing scheme {name!r} must be a mapping")
    if isinstance(body.get("models"), dict):
        models = body["models"]
        meta = {k: v for k, v in body.items() if k != "models"}
    else:
        models, meta = body, {}
    meta.update(extra_meta or {})
    scheme = PriceScheme(name=name, meta=meta)
    for model, entry in models.items():
        price = _parse_price(name, str(model), entry)
        if price is not None:
            scheme.prices[str(model)] = price
    return scheme


@dataclass
class PriceBook:
    schemes: dict[str, PriceScheme]
    catalog: ModelCatalog
    warnings: list[str] = field(default_factory=list)

    @property
    def scheme_names(self) -> list[str]:
        return list(self.schemes)

    def price(self, scheme: str, model: str | None) -> Price | None:
        return self.catalog.lookup(self.schemes[scheme].prices, model)

    def call_cost(self, scheme: str, call: dict) -> float:
        """Cost of one CallRecord under ``scheme``; NaN when the model has no price there."""
        price = self.price(scheme, call.get("model"))
        if price is None:
            return NAN
        prompt = call.get("prompt_tokens") or 0
        cached = call.get("cached_tokens") or 0
        uncached = call.get("uncached_prompt_tokens")
        if uncached is None:
            uncached = max(prompt - cached, 0)
        return price.cost(uncached, cached, call.get("completion_tokens") or 0)

    def describe(self) -> dict[str, Any]:
        """Provenance summary for summary.json: every price used (USD per 1M tokens), so the
        cost columns can be traced to a price book even after pricing.yaml changes."""
        return {
            name: {
                "meta": s.meta,
                "models": sorted(s.prices),
                "prices": {m: {"in": p.in_, "cached": p.cached, "out": p.out} for m, p in sorted(s.prices.items())},
                "cached_price_missing_charged_at_input": s.cached_fallback,
            }
            for name, s in self.schemes.items()
        }

    def files(self) -> list[str]:
        """Files the schemes came from besides pricing.yaml itself (its includes)."""
        return sorted({str(s.meta["file"]) for s in self.schemes.values() if s.meta.get("file")})


def load_pricing(path: str | Path | None, catalog: ModelCatalog | None = None) -> PriceBook:
    """Load pricing.yaml (+ includes). Aliases in pricing.yaml are added to ``catalog``."""
    catalog = catalog or ModelCatalog()
    if path is None:
        return PriceBook({}, catalog, ["no pricing.yaml given: no cost_<scheme> columns"])
    p = Path(path)
    if not p.is_file():
        return PriceBook({}, catalog, [f"{p} not found: no cost_<scheme> columns"])
    raw = yaml.safe_load(p.read_text()) or {}
    warnings: list[str] = []
    for a, c in (raw.get("aliases") or {}).items():
        catalog.add_alias(str(a), str(c))
    schemes: dict[str, PriceScheme] = {}
    for name, body in (raw.get("schemes") or {}).items():
        schemes[str(name)] = _parse_scheme(str(name), body)
    for inc in raw.get("include") or []:
        ip = (p.parent / inc) if not Path(inc).is_absolute() else Path(inc)
        if not ip.is_file():
            warnings.append(f"pricing include {ip} not found: its scheme is skipped")
            continue
        inc_raw = yaml.safe_load(ip.read_text()) or {}
        name = str(inc_raw.get("scheme") or ip.stem)
        meta = {k: v for k, v in inc_raw.items() if k not in ("scheme", "models")}
        meta["file"] = str(ip)
        if name in schemes:
            warnings.append(f"pricing include {ip} redefines scheme {name!r}; the include wins")
        schemes[name] = _parse_scheme(name, {"models": inc_raw.get("models") or {}}, meta)
    return PriceBook(schemes, catalog, warnings)
