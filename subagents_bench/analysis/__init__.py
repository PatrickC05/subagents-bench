"""Offline analysis of recorded episodes (``docs/DESIGN.md`` section 6).

Subcommands registered by :func:`register_cli`:

* ``aggregate JOBS_DIR... --out DIR`` -- every table (episodes, calls, tools,
  dispatches, spawns, arbitrations, steps, conditions,
  conditions_by_difficulty, paired, calibration) plus ``summary.json``;
  costs are recomputed per pricing scheme and wall time is simulated per
  serving profile;
* ``check-run DIR...`` -- schema and internal-consistency checks; exit 1 on problems;
  also prints each subagents condition's delegation rate;
* ``completeness DIR...`` -- per-field population report, flags never-populated
  fields; ``--strict`` also fails when a subagents condition never delegated;
* ``calibrate JOBS_DIR... --out configs/serving_fitted.yaml`` -- writes the
  per-model call-time fit used by the ``fitted`` serving profile.

Modules: :mod:`.load` (discovery and joins), :mod:`.pricing` (price schemes,
model catalog, FLOPs), :mod:`.latency` (critical path, serving profiles),
:mod:`.stats`, :mod:`.tables`, :mod:`.aggregate`, :mod:`.validate`. Heavy
imports (pandas) happen inside the command functions so ``subagents-bench
--help`` stays fast.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("aggregate", help="build analysis tables from harbor jobs dirs")
    p.add_argument("jobs_dirs", nargs="+", help="jobs dirs (sweep roots), job dirs, trial dirs or episode dirs")
    p.add_argument("--out", required=True, help="output directory, e.g. analysis/pilot")
    p.add_argument("--pricing", default=None, help="pricing schemes (default: configs/pricing.yaml)")
    p.add_argument("--serving", default=None, help="serving profiles (default: configs/serving.yaml)")
    p.add_argument("--models", default=None, help="model metadata + aliases (default: configs/models.yaml)")
    p.add_argument("--tasks-meta", default=None, help="tasks_meta.json (default: <jobs_dir>/tasks_meta.json)")
    p.add_argument("--baselines", nargs="*", default=None,
                   help="conditions used as paired baselines (default: single_5step single_matched)")
    p.set_defaults(func=_cmd_aggregate)

    p = subparsers.add_parser("check-run", help="validate episode records (schema + consistency); exit 1 on problems")
    p.add_argument("dirs", nargs="+", help="jobs dirs, job dirs, trial dirs or episode dirs")
    p.add_argument("--quiet", action="store_true", help="print only errors and the summary line")
    p.set_defaults(func=_cmd_check_run)

    p = subparsers.add_parser("completeness", help="report how often each schema field is None/empty/zero")
    p.add_argument("dirs", nargs="+", help="jobs dirs, job dirs, trial dirs or episode dirs")
    p.add_argument("--out", default=None, help="also write the full report as CSV")
    p.add_argument("--all", action="store_true", help="print every field, not only never-populated ones")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if a field outside the expected-empty list is never populated, or a subagents "
                        "condition never delegated (an arbiter condition never arbitrated)")
    p.set_defaults(func=_cmd_completeness)

    p = subparsers.add_parser("calibrate", help="fit per-model TTFT/decode latency for the 'fitted' serving profile")
    p.add_argument("jobs_dirs", nargs="+")
    p.add_argument("--out", default="configs/serving_fitted.yaml", help="YAML to write")
    p.add_argument("--csv", default=None, help="also write calibration.csv here")
    p.set_defaults(func=_cmd_calibrate)


def _cfg(arg: str | None, name: str) -> Path:
    from .aggregate import default_config_path

    return Path(arg) if arg else default_config_path(name)


def _cmd_aggregate(args: argparse.Namespace) -> int:
    from .aggregate import DEFAULT_BASELINES, aggregate

    res = aggregate(
        args.jobs_dirs,
        args.out,
        pricing_path=_cfg(args.pricing, "pricing.yaml"),
        serving_path=_cfg(args.serving, "serving.yaml"),
        models_path=_cfg(args.models, "models.yaml"),
        tasks_meta_path=args.tasks_meta,
        baselines=args.baselines if args.baselines is not None else DEFAULT_BASELINES,
    )
    s = res.summary
    print(f"aggregated {s['n_trials']} trials ({s['n_with_episode']} with episodes, {s['n_valid']} valid, "
          f"{s['n_resolved']} resolved) over {len(s['conditions'])} conditions -> {res.out_dir}")
    for name, n in s["table_rows"].items():
        print(f"  {name}.csv: {n} rows")
    for w in res.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def _cmd_check_run(args: argparse.Namespace) -> int:
    from .validate import check_run, coverage_lines

    report = check_run(args.dirs)
    for p in report.problems:
        if p.level == "error" or not args.quiet:
            print(p)
    for line in coverage_lines(report.coverage):
        print(line)
    status = "OK" if report.ok else "FAILED"
    print(f"check-run {status}: {report.n_episodes} episodes, {report.n_trials_without_episode} trials without "
          f"episode.json, {len(report.errors)} errors, {len(report.warnings)} warnings")
    if report.n_episodes == 0:
        print("check-run: no episodes found", file=sys.stderr)
    return 0 if report.ok else 1


def _cmd_completeness(args: argparse.Namespace) -> int:
    import pandas as pd

    from .validate import completeness, coverage_lines, treatment_coverage

    rows = completeness(args.dirs)
    if not rows:
        print("completeness: no episodes found", file=sys.stderr)
        return 1
    coverage = treatment_coverage(args.dirs)
    df = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
    shown = df if args.all else df[df["never_populated"]]
    if len(shown):
        print(f"{'file':<20} {'role':<13} {'field':<26} {'n':>6} {'none':>6} {'empty':>6} {'zero':>6}  note")
        for r in shown.itertuples():
            note = "NEVER POPULATED" if r.never_populated else ""
            if r.never_populated and r.often_empty:
                note += " (often legitimately empty)"
            print(f"{r.file:<20} {r.role:<13} {r.field:<26} {r.n_records:>6} {r.frac_none:>6.2f} "
                  f"{r.frac_empty:>6.2f} {r.frac_zero:>6.2f}  {note}")
    flagged = df[df["never_populated"] & ~df["often_empty"]]
    print(f"completeness: {len(df)} (file, role, field) rows; {int(df['never_populated'].sum())} never populated, "
          f"{len(flagged)} of them unexpected")
    # A treatment that never ran leaves no records, so no field above can flag it.
    for line in coverage_lines(coverage):
        print(line)
    untreated = [r["condition"] for r in coverage if r["problem"]]
    if untreated:
        print(f"completeness: treatment not exercised in {len(untreated)} subagents condition(s): "
              f"{', '.join(untreated)}")
    return 1 if (args.strict and (len(flagged) or untreated)) else 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import yaml

    from .load import load_trials
    from .tables import calibration_table, nan_to_none

    trials = load_trials(args.jobs_dirs)
    calls = [c for t in trials if t.data is not None for c in t.data.calls]
    table = calibration_table(calls)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
    params = []
    for r in table.to_dict("records"):
        params.append(nan_to_none({
            "model": r["model"], "provider": r["provider"], "completion_window": r["completion_window"],
            "lat_fit_a": r["lat_fit_a"], "lat_fit_b_per_1k_prompt": r["lat_fit_b_per_1k_prompt"],
            "lat_fit_c_per_completion_token": r["lat_fit_c_per_completion_token"], "lat_fit_n": r["lat_fit_n"],
            "n_calls": r["n_calls"],
            # Descriptive only (the profile uses lat_fit_*).
            "ttft_a_sec": r["ttft_fit_a"], "ttft_b_sec_per_1k_prompt_tokens": r["ttft_fit_b_per_1k_prompt_tokens"],
            "decode_tok_s": r["decode_tok_s_median"], "ttft_fit_n": r["ttft_fit_n"],
        }))
    doc = {
        "profile": "fitted",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [str(p) for p in args.jobs_dirs],
        "model": "call_sec = lat_fit_a + lat_fit_b_per_1k_prompt * prompt/1000 + lat_fit_c_per_completion_token * "
                 "completion (non-negative least squares of ttft_sec + decode_sec)",
        "params": params,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Written by `subagents-bench calibrate`; referenced from configs/serving.yaml\n"
              "# (profiles.fitted.params_file). Entries with null fit values simulate to NaN.\n")
    out.write_text(header + yaml.safe_dump(doc, sort_keys=False))
    n_ok = sum(1 for p in params if p["lat_fit_a"] is not None)
    print(f"calibrate: {len(calls)} calls, {len(params)} (model, provider, window) groups, {n_ok} fitted -> {out}")
    return 0
