"""``subagents-bench aggregate``: jobs dirs -> analysis tables + summary.json.

Pipeline: discover and load trials (:mod:`.load`), load model metadata and
price schemes (:mod:`.pricing`), build ``calibration.csv`` from every
successful call (it feeds the ``fitted`` serving profile), load the serving
profiles (:mod:`.latency`), then build and write:

``episodes.csv``, ``calls.csv``, ``tools.csv``, ``dispatches.csv``,
``spawns.csv``, ``arbitrations.csv``, ``steps.csv``, ``conditions.csv``,
``conditions_by_difficulty.csv``, ``paired.csv``, ``calibration.csv`` and
``summary.json`` (inputs, counts, the prices and the sha256 of every input
file used, the analysis code's provenance, treatment coverage, latency
checks, warnings).

Every table is written even when empty (header only), so downstream plotting
can rely on the files existing.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .. import provenance
from .latency import FittedProfile, fitted_vs_measured, load_serving
from .load import load_trials
from .pricing import load_models, load_pricing
from .tables import (
    calibration_table,
    condition_table,
    episode_row,
    episodes_frame,
    nan_to_none,
    paired_table,
    record_tables,
)
from .validate import treatment_coverage_rows

# Tolerance (seconds) below which overhead_sec - harness_exec_sec < 0 is clock jitter.
OVER_EXPLAINED_TOL_SEC = 0.05

DEFAULT_BASELINES = ("single_5step", "single_matched")
REPO_ROOT = Path(__file__).resolve().parents[2]


def default_config_path(name: str) -> Path:
    """``configs/<name>`` under the CWD if present, else under the repo this package lives in."""
    local = Path("configs") / name
    return local if local.exists() else REPO_ROOT / "configs" / name


@dataclass
class AggregateResult:
    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    out_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)


def aggregate(
    paths: Iterable[str | Path],
    out_dir: str | Path | None,
    pricing_path: str | Path | None = None,
    serving_path: str | Path | None = None,
    models_path: str | Path | None = None,
    tasks_meta_path: str | Path | None = None,
    baselines: Iterable[str] = DEFAULT_BASELINES,
) -> AggregateResult:
    """Build every table; write them to ``out_dir`` unless it is None."""
    paths = [Path(p) for p in paths]
    baselines = list(baselines)
    trials = load_trials(paths, tasks_meta_path)
    if not trials:
        raise ValueError(f"no harbor trials or episode dirs found under {', '.join(map(str, paths))}")

    warnings: list[str] = []
    catalog, w = load_models(models_path)
    warnings += w
    pricebook = load_pricing(pricing_path, catalog)
    warnings += pricebook.warnings

    all_calls = [c for t in trials if t.data is not None for c in t.data.calls]
    calibration = calibration_table(all_calls)
    profiles, w = load_serving(serving_path, catalog, calibration.to_dict("records"))
    warnings += w

    schemes = pricebook.scheme_names
    episodes = episodes_frame([episode_row(t, pricebook, profiles) for t in trials], schemes, profiles)
    tables: dict[str, pd.DataFrame] = {"episodes": episodes}
    tables.update(record_tables(trials, pricebook))
    tables["conditions"] = condition_table(episodes, ["condition"], schemes, baselines)
    tables["conditions_by_difficulty"] = condition_table(episodes, ["condition", "difficulty"], schemes, baselines)
    tables["paired"] = paired_table(episodes, baselines)
    tables["calibration"] = calibration

    for t in trials:
        for p in t.problems + (t.data.problems if t.data else []):
            warnings.append(f"{t.ref.trial_dir or t.ref.episode_dir}: {p}")
    for ctx in {id(t.ctx): t.ctx for t in trials}.values():
        warnings += [f"{ctx.root}: {p}" for p in ctx.problems]
    for scheme, models in _unpriced(tables["calls"], pricebook).items():
        if models:
            warnings.append(f"pricing scheme {scheme!r} has no price for {', '.join(models)}: "
                            f"cost_{scheme}_* of their calls, and every total including them, are NaN")
    dups = episodes.loc[episodes["duplicate"].astype(bool), "trial_name"].dropna().tolist()
    if dups:
        warnings.append(f"{len(dups)} duplicate trial(s) of an already-present (condition, sweep, task, round) "
                        f"left out of conditions/paired (the valid one, else the first by name, is kept): "
                        f"{', '.join(dups[:20])}{' ...' if len(dups) > 20 else ''}")
    coverage = treatment_coverage_rows(nan_to_none(
        episodes.loc[episodes["episode_status"] != "missing",
                     ["condition", "topology", "strategy", "spawn_calls", "dispatches", "arbitrations"]]
        .to_dict("records")))
    for row in coverage:
        if row["problem"]:
            warnings.append(f"condition {row['condition']}: treatment not exercised: {row['problem']}")

    summary = _summary(trials, tables, pricebook, profiles, catalog, paths, warnings, baselines,
                       pricing_path, serving_path, models_path, coverage, dups)
    result = AggregateResult(tables=tables, summary=summary, warnings=warnings)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(out / f"{name}.csv", index=False)
        (out / "summary.json").write_text(json.dumps(nan_to_none(summary), indent=2, sort_keys=False) + "\n")
        result.out_dir = out
    return result


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.fillna("none").value_counts().sort_index().items()}


def _unpriced(calls: pd.DataFrame, pricebook) -> dict[str, list[str]]:
    """scheme -> models that made calls but have no price in it."""
    models_seen = sorted({m for m in calls["model"].dropna().unique()}) if len(calls) else []
    return {s: [m for m in models_seen if pricebook.price(s, m) is None] for s in pricebook.scheme_names}


def _sha256(path: Any) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def _input_hashes(pricing_path, serving_path, models_path, pricebook, profiles) -> dict[str, str | None]:
    """sha256 of every config file the tables were computed from (None if missing)."""
    paths = [pricing_path, *pricebook.files(), serving_path, models_path]
    paths += [p.source for p in profiles if isinstance(p, FittedProfile) and Path(p.source).is_file()]
    return {str(p): _sha256(p) for p in dict.fromkeys(paths) if p is not None}


def _summary(trials, tables, pricebook, profiles, catalog, paths, warnings, baselines,
             pricing_path, serving_path, models_path, coverage, duplicates) -> dict[str, Any]:
    from .. import __version__

    ep = tables["episodes"]
    calls = tables["calls"]
    valid = ep[ep["valid"]]
    models_seen = sorted({m for m in calls["model"].dropna().unique()}) if len(calls) else []
    unpriced = _unpriced(calls, pricebook)
    no_params = [m for m in models_seen if catalog.active_params_b(m) is None]
    task_names = sorted({t for t in ep["task_name"].dropna().unique()})
    tasks_meta = {k: v for t in trials for k, v in t.ctx.tasks_meta.items()}
    datasets = sorted({t.ctx.dataset for t in trials if t.ctx.dataset})
    root = provenance.repo_root()
    code_by_cond: dict[str, set] = {}
    for t in trials:
        if t.episode is not None and t.condition is not None:
            code_by_cond.setdefault(str(t.condition), set()).add(t.episode.get("source_sha256") or "unknown")

    # measured replay vs critical path, and the overhead check of DESIGN section 9. Overhead the
    # harness's own measurement commands account for (harness_exec_sec) counts as explained.
    # Unexplained overhead is not clipped at 0: a negative value means harness_exec_sec
    # over-explains it (the summed execs of records before schema_version 3, or execs that
    # overlapped a chain), which is reported rather than hidden.
    diff = (ep["sim_measured_wall_sec"] - ep["cp_total_sec"]).abs() if "sim_measured_wall_sec" in ep else pd.Series(dtype=float)
    frac = (ep["overhead_sec"] / ep["wall_clock_sec"]).where(ep["wall_clock_sec"] > 0)
    over = ep.loc[frac > 0.10, "trial_name"].dropna().tolist()
    unexplained = ep["overhead_sec"] - ep["harness_exec_sec"].fillna(0.0)
    ufrac = (unexplained / ep["wall_clock_sec"]).where(ep["wall_clock_sec"] > 0)
    u_over = ep.loc[ufrac > 0.10, "trial_name"].dropna().tolist()
    over_explained = ep.loc[unexplained < -OVER_EXPLAINED_TOL_SEC, "trial_name"].dropna().tolist()
    all_calls = [c for t in trials if t.data is not None for c in t.data.calls]
    fitted = {p.name: fitted_vs_measured(p, all_calls) for p in profiles if isinstance(p, FittedProfile)}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_version": __version__,
        "inputs": [str(p) for p in paths],
        "files": {"pricing": _p(pricing_path), "serving": _p(serving_path), "models": _p(models_path)},
        "input_sha256": _input_hashes(pricing_path, serving_path, models_path, pricebook, profiles),
        # The analysis code that produced these tables (not the runtime that recorded the episodes).
        "analysis_source": {"source_sha256": provenance.source_fingerprint(root), **provenance.git_info(root)},
        # The runtime code that recorded the episodes, per condition (EpisodeRecord.source_sha256).
        "episode_source_sha256": {c: sorted(v) for c, v in sorted(code_by_cond.items())},
        "n_trials": len(ep),
        "n_with_episode": int((ep["episode_status"] != "missing").sum()),
        "n_valid": len(valid),
        "n_resolved": int(pd.to_numeric(valid["resolved"], errors="coerce").fillna(0).sum()),
        "episode_status_counts": _counts(ep["episode_status"]),
        "invalid_reason_counts": _counts(ep.loc[~ep["valid"], "invalid_reason"]),
        "harbor_exception_counts": _counts(ep["harbor_exception"].dropna()),
        "exit_reason_counts": _counts(ep["exit_reason"]),
        "sweeps": sorted({str(s) for s in ep["sweep"].dropna().unique()}),
        "conditions": tables["conditions"]["condition"].tolist(),
        "baselines": [b for b in baselines if b in set(ep["condition"].dropna())],
        "rounds": sorted({int(r) for r in ep["round"].dropna().unique()}),
        "tasks": task_names,
        "tasks_meta": {
            "datasets": datasets,
            "n_tasks_with_meta": sum(1 for t in task_names if t in tasks_meta),
            "tasks_without_meta": [t for t in task_names if t not in tasks_meta],
        },
        "pricing": {
            "schemes": pricebook.describe(),
            "aliases": catalog.aliases,
            "models_without_price": unpriced,
        },
        "models_without_active_params": no_params,
        "serving_profiles": {p.name: p.describe() for p in profiles},
        "latency_checks": {
            "measured_replay_max_abs_diff_sec": float(diff.max()) if diff.notna().any() else None,
            "overhead_frac_median": float(frac.median()) if frac.notna().any() else None,
            "episodes_overhead_over_10pct": over,
            "unexplained_overhead_frac_median": float(ufrac.median()) if ufrac.notna().any() else None,
            "episodes_unexplained_overhead_over_10pct": u_over,
            "unexplained_overhead_sec_min": float(unexplained.min()) if unexplained.notna().any() else None,
            "episodes_overhead_over_explained": over_explained,
            "fitted_vs_measured": fitted,
        },
        "treatment_coverage": coverage,
        "duplicate_trials": duplicates,
        "table_rows": {k: int(len(v)) for k, v in tables.items()},
        "warnings": warnings,
    }


def _p(path: Any) -> str | None:
    if path is None:
        return None
    p = Path(path)
    return f"{p} (missing)" if not p.exists() else str(p)


def fmt_float(x: Any, nd: int = 3) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if math.isnan(f) else f"{f:.{nd}f}"
