"""Loading the analysis tables for the figures, defensively.

The aggregator (``subagents-bench aggregate``) writes the CSVs of
``docs/DESIGN.md`` section 6 into an analysis directory. This module reads
them for :mod:`subagents_bench.plots` and never assumes more than it checks:

* a missing or empty table, or a missing column, raises :class:`Skip` with a
  human-readable reason, which the caller turns into "figure/panel skipped"
  plus a log line -- never a crash of the whole run;
* per-condition statistics come from ``conditions.csv`` (``<metric>_mean``,
  ``_ci_lo``, ``_ci_hi``; Wilson ``resolve_ci_*``) when present, and are
  otherwise recomputed from ``episodes.csv`` with the same definitions
  (bootstrap 95% CI, 2000 resamples, seed 0; Wilson 95% for rates);
* condition metadata (topology, strategy, scope, subagent model) comes from
  ``conditions.csv``, else ``episodes.csv``, else the condition name;
* ``episodes.csv`` rows flagged ``duplicate`` (a re-run of a trial already
  counted) are dropped, as the aggregator drops them from its tables, and
  per-event rows (dispatches, spawns, arbitrations) of invalid or duplicate
  episodes are dropped with them (:meth:`AnalysisData.event_rows`);
* "unknown" stays NaN; nothing is filled with 0.
"""

from __future__ import annotations

import fnmatch
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

log = logging.getLogger("subagents_bench.plots")

STRATEGIES = ("parallel", "sequential", "arbiter")
SCOPES = ("easy", "medium", "hard")
DIFFICULTIES = ("easy", "medium", "hard")
ROLES = ("orch", "sub", "arb", "deb")
ROLE_NAMES = {"orch": "orchestrator", "sub": "subagent", "arb": "arbiter (judge)", "deb": "debate"}
# Known pricing schemes / serving profiles first, in DESIGN order; others follow alphabetically.
KNOWN_SCHEMES = ("sail_balanced", "sail_asap", "prime", "reference")
KNOWN_PROFILES = ("measured", "fitted", "roofline_h100")

N_BOOT = 2000
BOOT_SEED = 0
Z95 = 1.959963984540054


class Skip(Exception):
    """A figure or panel cannot be drawn; the message says why."""


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def wilson(k: float, n: float, z: float = Z95) -> tuple[float, float, float]:
    """Wilson score interval: (p, lo, hi); NaNs when n == 0."""
    if not n:
        return math.nan, math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean(values: Iterable[float], *, strict_nan: bool = False,
                   n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap 95% CI.

    With ``strict_nan`` any NaN makes the result NaN (e.g. a cost that is
    unknown for some episodes must not be averaged as if it were known);
    otherwise NaNs are dropped. Fewer than 2 values give a NaN interval.
    """
    v = np.asarray(list(values) if not isinstance(values, (np.ndarray, pd.Series)) else values, dtype=float)
    if strict_nan and np.isnan(v).any():
        return math.nan, math.nan, math.nan
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return math.nan, math.nan, math.nan
    mean = float(v.mean())
    if len(v) < 2:
        return mean, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_ratio(num: Iterable[float], den: Iterable[float], *,
                    n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """sum(num)/sum(den) with a paired bootstrap 95% CI. NaN in either input
    makes everything NaN; a zero total denominator has no point estimate.

    A resample with a zero denominator (no solve drawn) has an infinite ratio
    and is kept, not dropped: dropping it would give the interval conditional
    on at least one solve, which is tightest exactly when solves are rare.
    When more than 2.5% of the resamples draw no solve the upper bound is
    therefore +inf ("unbounded"). The bounds are the outward order statistics
    (``lower``/``higher``), so an infinite tail gives inf, never NaN.
    """
    a = np.asarray(list(num), dtype=float)
    b = np.asarray(list(den), dtype=float)
    if len(a) == 0 or np.isnan(a).any() or np.isnan(b).any() or b.sum() == 0:
        return math.nan, math.nan, math.nan
    point = float(a.sum() / b.sum())
    if len(a) < 2:
        return point, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    sb = b[idx].sum(axis=1)
    ratios = np.where(sb > 0, a[idx].sum(axis=1) / np.where(sb > 0, sb, 1.0), math.inf)
    return (point, float(np.quantile(ratios, 0.025, method="lower")),
            float(np.quantile(ratios, 0.975, method="higher")))


def as_bool(s: pd.Series) -> pd.Series:
    """Map True/False/1/0/"true"/"False"/... to 1.0/0.0, anything else to NaN."""
    def conv(x: Any) -> float:
        if isinstance(x, (bool, np.bool_)):
            return float(x)
        if isinstance(x, (int, float, np.integer, np.floating)):
            return math.nan if (isinstance(x, float) and math.isnan(x)) else float(bool(x))
        if isinstance(x, str):
            t = x.strip().lower()
            if t in ("true", "1", "1.0", "yes"):
                return 1.0
            if t in ("false", "0", "0.0", "no"):
                return 0.0
        return math.nan

    return s.map(conv).astype(float)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _clean(x: Any) -> str | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    t = str(x).strip()
    return t if t and t.lower() not in ("nan", "none", "null") else None


def _ordered(found: Iterable[str], known: tuple[str, ...]) -> list[str]:
    found = set(found)
    return [k for k in known if k in found] + sorted(found - set(known))


# ---------------------------------------------------------------------------
# condition metadata
# ---------------------------------------------------------------------------


@dataclass
class CondInfo:
    name: str
    topology: str | None
    strategy: str | None
    scope: str | None
    subagent_model: str | None
    orchestrator_model: str | None
    steps_budget: float | None
    # tools.orchestrator_tools label ("all" = default); restricted = control.
    orchestrator_tools: str | None = None
    baseline: bool = False
    # Not part of the strategy x scope grid of the primary subagent model
    # (e.g. the same-model control, a size sweep, or unknown strategy/scope).
    control: bool = False
    order: int = 0
    baseline_index: int = -1

    @property
    def manager(self) -> bool:
        """The orchestrator's own tools are restricted (a manager control that must delegate)."""
        return self.orchestrator_tools not in (None, "all")

    @property
    def label(self) -> str:
        """Display name: the condition, tagged ``[manager]`` for a manager control."""
        return f"{self.name} [manager]" if self.manager else self.name

    @property
    def kind(self) -> str:
        """Legend wording for an off-grid condition."""
        return f"manager control, orchestrator tools {self.orchestrator_tools}" if self.manager else "control"


def _parse_name(name: str) -> dict[str, str | None]:
    toks = re.split(r"[_\-]", name.lower())
    return {
        "strategy": next((t for t in toks if t in STRATEGIES), None),
        "scope": next((t for t in toks if t in SCOPES), None),
        "topology": "single" if toks and toks[0] == "single" else None,
    }


# ---------------------------------------------------------------------------
# the analysis directory
# ---------------------------------------------------------------------------


class AnalysisData:
    """Lazy, cached, condition-filtered access to one analysis directory."""

    def __init__(self, root: str | Path, conditions: list[str] | None = None):
        self.root = Path(root)
        self._raw: dict[str, pd.DataFrame | None] = {}
        self._why: dict[str, str] = {}
        self.patterns = [p for p in (conditions or []) if p]
        self._meta: dict[str, CondInfo] | None = None
        self._selected: list[str] | None = None

    # -- tables ---------------------------------------------------------------

    def raw(self, name: str) -> pd.DataFrame | None:
        """The whole table, or None (reason in :meth:`why`)."""
        if name not in self._raw:
            path = self.root / f"{name}.csv"
            df = None
            if not path.exists():
                self._why[name] = f"{name}.csv not found"
            else:
                try:
                    df = pd.read_csv(path, low_memory=False)
                except pd.errors.EmptyDataError:
                    self._why[name] = f"{name}.csv is empty"
                except Exception as e:  # unreadable file: skip, never crash
                    self._why[name] = f"{name}.csv unreadable ({type(e).__name__}: {e})"
                    log.warning("could not read %s: %s", path, e)
                if df is not None and df.empty:
                    self._why[name] = f"{name}.csv has no rows"
                    df = None
            self._raw[name] = df
        return self._raw[name]

    def why(self, name: str) -> str:
        return self._why.get(name, f"{name}.csv unavailable")

    def table(self, name: str) -> pd.DataFrame | None:
        """The table restricted to the selected conditions (if it has a condition column)."""
        df = self.raw(name)
        if df is None:
            return None
        if "condition" in df.columns:
            df = df[df["condition"].astype(str).isin(set(self.selected))]
            if df.empty:
                self._why[name] = f"{name}.csv has no rows for the selected conditions"
                return None
        return df

    def require(self, name: str, cols: Iterable[str] = ()) -> pd.DataFrame:
        df = self.table(name)
        if df is None:
            raise Skip(self.why(name))
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise Skip(f"{name}.csv lacks column(s) {', '.join(missing)}")
        return df

    def has(self, name: str, cols: Iterable[str] = ()) -> bool:
        df = self.table(name)
        return df is not None and all(c in df.columns for c in cols)

    def episodes(self, cols: Iterable[str] = (), valid_only: bool = True) -> pd.DataFrame:
        """episodes.csv rows (valid ones by default) with a float ``resolved`` column."""
        eps = self.require("episodes", ["condition", *cols]).copy()
        if "resolved" in eps.columns:
            eps["resolved"] = as_bool(eps["resolved"])
        elif "reward" in eps.columns:
            r = num(eps["reward"])
            eps["resolved"] = np.where(r.isna(), np.nan, (r >= 1.0).astype(float))
        else:
            eps["resolved"] = np.nan
        if "duplicate" in eps.columns:
            eps = eps[as_bool(eps["duplicate"]) != 1.0]
        if valid_only and "valid" in eps.columns:
            eps = eps[as_bool(eps["valid"]) == 1.0]
        if eps.empty:
            raise Skip(f"episodes.csv has no {'valid ' if valid_only else ''}episodes for the selected conditions")
        return eps

    def event_rows(self, name: str, cols: Iterable[str] = ()) -> pd.DataFrame:
        """Rows of a per-event table (``dispatches``, ``spawns``,
        ``arbitrations``) whose episode counts in the per-episode figures:
        valid and not a duplicate in episodes.csv. An invalid trial's events
        are real, but an outage or harness failure shaped them (a provider
        retrying for minutes inflates a dispatch's LLM seconds), so they are
        left out like the trial. ``.attrs['n_dropped']`` counts the rows left
        out; every row is kept when episodes.csv cannot tell."""
        df = self.require(name, cols)
        eps = self.raw("episodes")
        if eps is None or "episode_id" not in df.columns or not {"episode_id", "valid"} <= set(eps.columns):
            df = df.copy()
            df.attrs["n_dropped"] = 0
            return df
        keep = as_bool(eps["valid"]) == 1.0
        if "duplicate" in eps.columns:
            keep &= as_bool(eps["duplicate"]) != 1.0
        mask = df["episode_id"].astype(str).isin(set(eps.loc[keep, "episode_id"].astype(str)))
        out = df[mask].copy()
        out.attrs["n_dropped"] = int((~mask).sum())
        if out.empty:
            raise Skip(f"{name}.csv has no rows from valid episodes")
        return out

    # -- conditions -------------------------------------------------------------

    @property
    def all_conditions(self) -> list[str]:
        names: list[str] = []
        for t in ("conditions", "episodes"):
            df = self.raw(t)
            if df is not None and "condition" in df.columns:
                names += [str(c) for c in df["condition"].dropna().unique()]
        return list(dict.fromkeys(names))

    @property
    def selected(self) -> list[str]:
        if self._selected is None:
            names = self.all_conditions
            if self.patterns:
                keep = [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in self.patterns)]
                for p in self.patterns:
                    if not any(fnmatch.fnmatchcase(n, p) for n in names):
                        log.warning("--conditions pattern %r matches no condition", p)
                names = keep
            self._selected = names
        return self._selected

    @property
    def meta(self) -> dict[str, CondInfo]:
        if self._meta is None:
            self._meta = self._build_meta()
        return self._meta

    def _attr_source(self, col: str) -> dict[str, Any]:
        """condition -> value of ``col`` from conditions.csv, then episodes.csv (mode)."""
        out: dict[str, Any] = {}
        for t in ("episodes", "conditions"):  # conditions.csv wins (written last)
            df = self.raw(t)
            if df is None or col not in df.columns or "condition" not in df.columns:
                continue
            for cond, g in df.groupby("condition", sort=False):
                vals = [v for v in (_clean(x) for x in g[col]) if v is not None]
                if vals:
                    out[str(cond)] = max(set(vals), key=vals.count)
        return out

    def _build_meta(self) -> dict[str, CondInfo]:
        names = self.all_conditions
        src = {c: self._attr_source(c) for c in
               ("topology", "strategy", "scope", "subagent_model", "orchestrator_model", "orchestrator_steps_budget",
                "orchestrator_tools")}
        infos: list[CondInfo] = []
        for n in names:
            parsed = _parse_name(n)
            topo = src["topology"].get(n) or parsed["topology"]
            strat = src["strategy"].get(n) or (None if topo == "single" else parsed["strategy"])
            scope = src["scope"].get(n) or (None if topo == "single" else parsed["scope"])
            budget = src["orchestrator_steps_budget"].get(n)
            try:
                budget = float(budget) if budget is not None else None
            except ValueError:
                budget = None
            info = CondInfo(n, topo, strat, scope, src["subagent_model"].get(n),
                            src["orchestrator_model"].get(n), budget, src["orchestrator_tools"].get(n))
            info.baseline = topo == "single" or (topo is None and strat is None and n.lower().startswith("single"))
            infos.append(info)
        models = [i.subagent_model for i in infos if not i.baseline and i.subagent_model]
        primary = max(set(models), key=models.count) if models else None
        taken: set[tuple[str, str]] = set()
        # Deterministic grid assignment: primary-model conditions by name.
        for i in sorted((i for i in infos if not i.baseline), key=lambda i: i.name):
            cell = (i.strategy, i.scope)
            if (i.strategy not in STRATEGIES or i.scope not in SCOPES or cell in taken
                    or (primary is not None and i.subagent_model not in (None, primary))
                    or i.orchestrator_tools not in (None, "all")):
                i.control = True
            else:
                taken.add(cell)  # type: ignore[arg-type]
        def key(i: CondInfo) -> tuple:
            if i.baseline:
                return (0, i.steps_budget if i.steps_budget is not None else math.inf, i.name)
            if not i.control:
                return (1, STRATEGIES.index(i.strategy), SCOPES.index(i.scope), i.name)  # type: ignore[arg-type]
            return (2, i.name)
        out: dict[str, CondInfo] = {}
        b = 0
        for k, i in enumerate(sorted(infos, key=key)):
            i.order = k
            if i.baseline:
                i.baseline_index = b
                b += 1
            out[i.name] = i
        return out

    def order(self, names: Iterable[str]) -> list[str]:
        meta = self.meta
        return sorted(dict.fromkeys(str(n) for n in names if str(n) in meta), key=lambda n: meta[n].order)

    @property
    def baselines(self) -> list[str]:
        return [n for n in self.order(self.selected) if self.meta[n].baseline]

    @property
    def primary_subagent_model(self) -> str | None:
        models = [i.subagent_model for i in self.meta.values() if not i.baseline and not i.control and i.subagent_model]
        return models[0] if models else None

    # -- schemes / profiles -----------------------------------------------------

    def _columns(self) -> list[str]:
        cols: list[str] = []
        for t in ("episodes", "conditions"):
            df = self.raw(t)
            if df is not None:
                cols += list(df.columns)
        return cols

    def schemes(self) -> list[str]:
        found = set()
        for c in self._columns():
            m = re.fullmatch(r"cost_(.+)_total(?:_mean)?", c) or re.fullmatch(r"cost_per_solve_(.+)", c)
            if m and m.group(1) != "runtime" and not m.group(1).endswith(("_ci_lo", "_ci_hi")):
                found.add(m.group(1))
        return _ordered(found, KNOWN_SCHEMES)

    def profiles(self) -> list[str]:
        found = set()
        for c in self._columns():
            m = re.fullmatch(r"sim_(.+)_wall_sec(?:_mean)?", c)
            if m:
                found.add(m.group(1))
        return _ordered(found, KNOWN_PROFILES)

    # -- per-condition statistics -------------------------------------------------

    def cond_metric(self, metric: str, *, strict_nan: bool = False) -> pd.DataFrame:
        """Per selected condition: mean, lo, hi of an episode metric.

        From ``conditions.csv`` ``<metric>_mean/_ci_lo/_ci_hi`` if present,
        else bootstrapped from ``episodes.csv``. ``.attrs['source']`` says which.
        """
        conds = self.table("conditions")
        if conds is not None and f"{metric}_mean" in conds.columns:
            df = conds.drop_duplicates("condition").set_index("condition")
            out = pd.DataFrame({
                "mean": num(df[f"{metric}_mean"]),
                "lo": num(df[f"{metric}_ci_lo"]) if f"{metric}_ci_lo" in df.columns else np.nan,
                "hi": num(df[f"{metric}_ci_hi"]) if f"{metric}_ci_hi" in df.columns else np.nan,
            })
            out.index = out.index.astype(str)
            out = out.loc[self.order(out.index)]
            out.index.name = "condition"
            out.attrs["source"] = f"`conditions.csv` ({metric}_mean, {metric}_ci_lo, {metric}_ci_hi)"
            return out
        eps = self.table("episodes")
        if eps is None or metric not in eps.columns:
            raise Skip(f"no {metric}_mean in conditions.csv and no {metric} column in episodes.csv")
        eps = self.episodes([metric])
        rows = {}
        for cond, g in eps.groupby("condition", sort=False):
            rows[str(cond)] = bootstrap_mean(num(g[metric]).to_numpy(), strict_nan=strict_nan)
        out = pd.DataFrame.from_dict(rows, orient="index", columns=["mean", "lo", "hi"])
        out = out.loc[self.order(out.index)]
        out.index.name = "condition"
        out.attrs["source"] = f"`episodes.csv` ({metric}; bootstrap 95% CI recomputed here)"
        return out

    def _episode_uptake(self, keys: list[str]) -> tuple[pd.DataFrame, str]:
        """Treatment uptake per group of the valid episodes.csv rows with a
        known outcome, indexed by ``keys`` (as strings): n_delegated,
        delegated_frac, rate_delegated (from ``delegated``, else
        ``spawn_calls > 0``; NaN when episodes.csv has neither) and
        n_tasks_valid; plus the data-source note."""
        eps = self.episodes(keys[1:])
        eps = eps[eps["resolved"].notna()]
        col = next((c for c in ("delegated", "spawn_calls") if c in eps.columns), None)
        if col == "delegated":
            deleg = num(eps[col])
        elif col == "spawn_calls":
            deleg = (num(eps[col]) > 0).astype(float).where(num(eps[col]).notna())
        else:
            deleg = pd.Series(np.nan, eps.index)
        eps = eps.assign(_deleg=deleg)
        rows = []
        for key, g in eps.groupby(keys, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            took = g[g["_deleg"] == 1.0]
            known = int(g["_deleg"].notna().sum())
            rows.append({**dict(zip(keys, map(str, key))),
                         "n_delegated": float(len(took)) if known else np.nan,
                         "delegated_frac": len(took) / known if known else np.nan,
                         "rate_delegated": float(took["resolved"].mean()) if len(took) else np.nan,
                         "n_tasks_valid": float(g["task_name"].nunique()) if "task_name" in g.columns else np.nan})
        frame = pd.DataFrame(rows, columns=[*keys, "n_delegated", "delegated_frac", "rate_delegated",
                                            "n_tasks_valid"]).set_index(keys)
        used = [c for c in (col, "task_name" if "task_name" in eps.columns else None) if c]
        return frame, f"`episodes.csv` ({', '.join(used) or 'no uptake columns'})"

    def resolve_stats(self, by_difficulty: bool = False) -> pd.DataFrame:
        """Resolve rate + Wilson CI per condition (or per condition x difficulty).

        Columns: condition, [difficulty,] n_valid, n_resolved, rate, lo, hi,
        plus the treatment uptake n_delegated, delegated_frac (valid episodes
        whose orchestrator spawned subagents), rate_delegated (the
        per-protocol resolve rate among them) and n_tasks_valid (tasks with a
        valid trial); these are NaN when the inputs lack them.

        Rows without a rate (no valid trial) are dropped from the frame; their
        keys are listed in ``.attrs['no_valid']`` (condition names, or
        (condition, difficulty) pairs) so figures can say what is missing.
        """
        tname = "conditions_by_difficulty" if by_difficulty else "conditions"
        keys = ["condition", "difficulty"] if by_difficulty else ["condition"]
        need = ["resolve_rate", "resolve_ci_lo", "resolve_ci_hi"]
        df = self.table(tname)
        if df is not None and all(c in df.columns for c in keys + need):
            out = pd.DataFrame({k: df[k].astype(str) for k in keys})
            for col, src in (("n_valid", "n_valid"), ("n_resolved", "n_resolved"), ("rate", "resolve_rate"),
                             ("lo", "resolve_ci_lo"), ("hi", "resolve_ci_hi"), ("n_delegated", "n_delegated"),
                             ("delegated_frac", "delegated_frac"), ("rate_delegated", "resolve_rate_delegated"),
                             ("n_tasks_valid", "n_tasks_valid")):
                out[col] = num(df[src]) if src in df.columns else np.nan
            source = f"`{tname}.csv` (resolve_rate, resolve_ci_lo, resolve_ci_hi, n_valid, n_resolved)"
            uptake = f"`{tname}.csv` (n_delegated, resolve_rate_delegated, n_tasks_valid)"
            if "n_delegated" not in df.columns:
                # Written before the uptake columns existed: count them from episodes.csv.
                try:
                    filled, src = self._episode_uptake(keys)
                except Skip:
                    filled = None
                if filled is not None and filled["n_delegated"].notna().any():
                    cols = ["n_delegated", "delegated_frac", "rate_delegated"]
                    cols += [] if "n_tasks_valid" in df.columns else ["n_tasks_valid"]
                    out = out.drop(columns=cols).join(filled[cols], on=keys)
                    uptake = src
        else:
            every = self.episodes(keys[1:], valid_only=False)
            eps = self.episodes(keys[1:])
            eps = eps[eps["resolved"].notna()]
            rows = []
            for key, g in eps.groupby(keys, sort=False):
                key = key if isinstance(key, tuple) else (key,)
                k, n = float(g["resolved"].sum()), float(len(g))
                p, lo, hi = wilson(k, n)
                rows.append({**dict(zip(keys, map(str, key))), "n_valid": n, "n_resolved": k, "rate": p, "lo": lo,
                             "hi": hi})
            # Groups with trials but none valid with a known outcome: listed, never silently lost.
            have = {tuple(r[k] for k in keys) for r in rows}
            for key, _g in every.groupby(keys, sort=False):
                key = tuple(map(str, key if isinstance(key, tuple) else (key,)))
                if key not in have:
                    rows.append({**dict(zip(keys, key)), "n_valid": 0.0, "n_resolved": 0.0, "rate": np.nan,
                                 "lo": np.nan, "hi": np.nan})
            out = pd.DataFrame(rows).reindex(columns=keys + ["n_valid", "n_resolved", "rate", "lo", "hi"])
            source = "`episodes.csv` (resolved, valid; Wilson 95% CI recomputed here)"
            filled, uptake = self._episode_uptake(keys)
            out = out.join(filled, on=keys)
        out = out[out["condition"].isin(set(self.selected))]
        empty = out[out["rate"].isna()]
        no_valid = list(empty["condition"]) if not by_difficulty else list(zip(empty["condition"], empty["difficulty"]))
        out = out[out["rate"].notna()]
        if out.empty:
            raise Skip(f"no resolve rates in {tname}.csv / episodes.csv"
                       + (f" (no valid trials: {', '.join(map(str, no_valid))})" if no_valid else ""))
        order = {n: self.meta[n].order for n in out["condition"] if n in self.meta}
        out = out.assign(_o=out["condition"].map(order)).sort_values(["_o"] + keys[1:]).drop(columns="_o")
        out = out.reset_index(drop=True)
        out.attrs["source"] = source
        out.attrs["uptake_source"] = uptake

        def rank(k: Any) -> tuple:
            c = k if isinstance(k, str) else k[0]
            return (self.meta[c].order if c in self.meta else math.inf, str(k))
        out.attrs["no_valid"] = sorted(no_valid, key=rank)
        return out
