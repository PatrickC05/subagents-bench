"""Command-line entrypoint: ``subagents-bench ...``.

Subcommands:

* ``validate CONFIG`` -- load and echo a resolved config (dry run).
* ``run CONFIG`` -- run one hand-written task locally (smoke test), using the
  local_docker/local_shell sandbox in the config.
* ``aggregate RUNS_DIR`` -- roll per-trial telemetry into the per-config table
  and Pareto points.

Real Terminal-Bench / ExploitBench runs are launched through their own
harnesses pointed at :mod:`subagents_bench.harness`; see the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import load_trials, summarize, pareto_points, write_csv
from .config import ExperimentConfig
from .runner import run_local


def _cmd_validate(args) -> int:
    cfg = ExperimentConfig.from_yaml(args.config)
    print(json.dumps(cfg.to_dict(), indent=2))
    print(f"\nresolved subagent model: {cfg.resolved_subagent_model().model}", file=sys.stderr)
    print(f"same-model baseline: {cfg.is_same_model_baseline()}", file=sys.stderr)
    return 0


def _cmd_run(args) -> int:
    cfg = ExperimentConfig.from_yaml(args.config)
    instruction = Path(args.issue_file).read_text() if args.issue_file else args.issue
    if not instruction:
        print("provide --issue or --issue-file", file=sys.stderr)
        return 2
    telem = run_local(cfg, args.task_id, instruction)
    print(json.dumps(telem.summary(), indent=2))
    return 0


def _cmd_aggregate(args) -> int:
    trials = load_trials(args.runs_dir)
    if not trials:
        print(f"no telemetry.json found under {args.runs_dir}", file=sys.stderr)
        return 1
    summaries = summarize(trials)
    rows = [s.row() for s in summaries]
    print(json.dumps(rows, indent=2))
    if args.csv:
        write_csv(summaries, args.csv)
        print(f"wrote {args.csv}", file=sys.stderr)
    if args.pareto:
        pts = pareto_points(summaries, x=args.pareto)
        print(f"\nPareto ({args.pareto} vs resolve_rate):", file=sys.stderr)
        for p in pts:
            print(
                f"  {p['experiment']:<24} {p[args.pareto]!s:>14}  resolve={p['resolve_rate']}",
                file=sys.stderr,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="subagents-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="load and print a config")
    p_val.add_argument("config")
    p_val.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="run one task locally (smoke test)")
    p_run.add_argument("config")
    p_run.add_argument("--task-id", default="adhoc")
    p_run.add_argument("--issue", default=None)
    p_run.add_argument("--issue-file", default=None)
    p_run.set_defaults(func=_cmd_run)

    p_agg = sub.add_parser("aggregate", help="summarize telemetry into tables/Pareto")
    p_agg.add_argument("runs_dir")
    p_agg.add_argument("--csv", default=None)
    p_agg.add_argument(
        "--pareto",
        default=None,
        choices=["mean_total_tokens", "mean_cost_usd", "mean_wall_clock_sec"],
    )
    p_agg.set_defaults(func=_cmd_aggregate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
