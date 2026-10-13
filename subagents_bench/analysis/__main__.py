"""``python -m subagents_bench.analysis JOBS_DIR... --out DIR``: tables, summary.json and the
delegation gate.

Exit 3 if a subagent condition never delegated in any of its valid episodes (in the first
smoke run the all-tools orchestrator delegated in 0 of 18 episodes), 2 if there is no episode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .aggregate import build_tables, summary, write
from .load import load_trials
from .pricing import DEFAULT_PRICING, load_prices

SHOW = ["condition", "n", "n_valid", "n_resolved", "resolve_rate", "resolve_ci_lo", "resolve_ci_hi", "n_delegated",
        "cost_total_mean", "wall_clock_sec_mean"]
SHOW_BUGS = ["n_bugs", "bugs_fixed", "bug_fix_rate", "n_regressed"]  # when a task reported its bugs (SWE-sweep)
SHOW_INTERACTIVE = ["condition", "n_episodes", "n_hints", "busy_at_arrival_frac", "delivery_median_sec",
                    "response_median_sec", "dispatch_median_sec", "touch_median_sec", "edit_median_sec",
                    "touch_before_next", "hinted_fixed_frac"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m subagents_bench.analysis", description=__doc__.split("\n\n")[0])
    p.add_argument("jobs_dirs", nargs="+", type=Path, help="sweep jobs dir(s), analyzed as one data set")
    p.add_argument("--out", required=True, type=Path, help="output dir for the CSV tables and summary.json")
    p.add_argument("--pricing", type=Path, default=DEFAULT_PRICING, help="price file (default configs/pricing.yaml)")
    p.add_argument("--scheme", default="sail_balanced", help="price scheme in the price file")
    args = p.parse_args(argv)
    missing = [str(d) for d in args.jobs_dirs if not d.is_dir()]
    if missing:
        print(f"[FAIL] jobs dir not found: {' '.join(missing)}", flush=True)
        return 2
    trials = [t for d in args.jobs_dirs for t in load_trials(d)]
    if not any(t.episode for t in trials):
        print(f"[FAIL] no episode under {' '.join(map(str, args.jobs_dirs))}", flush=True)
        return 2
    tables = build_tables(trials, load_prices(args.pricing, args.scheme))
    summ = summary(tables, [str(d) for d in args.jobs_dirs], {"file": str(args.pricing), "scheme": args.scheme})
    write(tables, summ, args.out)
    conds = tables["conditions"]
    print(conds[SHOW + (SHOW_BUGS if conds["n_bugs"].sum() else [])].to_string(index=False,
                                                                                float_format=lambda x: f"{x:.3g}"))
    inter = tables["interactivity"]
    if len(inter):
        print("\ninteractivity (seconds after each hint was sent; medians count a missing event as never):")
        print(inter[SHOW_INTERACTIVE].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\n{summ['n_valid']} of {summ['n_trials']} trials valid; invalid: {summ['invalid_reasons'] or 'none'}")
    cp = summ["critical_path"]
    overhead = "n/a" if cp["mean_overhead_sec"] is None else f"{cp['mean_overhead_sec']:.3g} s"
    print(f"critical path > wall clock in {cp['n_over_wall_clock']} of {cp['n_valid_episodes']} valid episodes; "
          f"mean overhead {overhead}")
    print(f"tables: {args.out}/")
    gate = summ["delegation_gate"]
    for cond in gate["never_delegated"]:
        print(f"[FAIL] delegation gate: {cond} never delegated in any valid episode")
    if gate["ok"]:
        print("[PASS] delegation gate: every subagent condition delegated in at least one valid episode")
    return 0 if gate["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
