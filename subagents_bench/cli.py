"""Command line: ``subagents-bench validate|preflight|sweep`` (exit codes in :mod:`subagents_bench.sweep`)."""

from __future__ import annotations

import argparse
import sys

import yaml

from . import sweep
from .config import ConfigError, load_config, load_dotenv


def _rounds(text: str) -> list[int]:
    try:
        return sorted({int(x) for x in text.split(",")})
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected rounds like 1,2 (got {text!r})") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="subagents-bench", description="Orchestrator + subagents on Terminal-Bench 2.0.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate", help="load experiment config YAML(s) and print the resolved config")
    p.add_argument("configs", nargs="+")
    p = sub.add_parser("preflight", help="check Docker, harbor, SAIL_API_KEY and every model before a sweep")
    p.add_argument("spec", help="sweep spec (configs/sweeps/*.yaml)")
    p.add_argument("--oracle", action="store_true", help="also run harbor's oracle agent on every task")
    p = sub.add_parser("sweep", help="run every (round, condition) harbor job of a sweep spec; re-run to resume")
    p.add_argument("spec", help="sweep spec (configs/sweeps/*.yaml)")
    p.add_argument("--dry-run", action="store_true", help="print the jobs and their harbor commands; run nothing")
    p.add_argument("--only", nargs="+", metavar="COND", help="run only these conditions")
    p.add_argument("--rounds", type=_rounds, help="comma-separated rounds to run, e.g. 1,2")
    return parser


def _validate(paths: list[str]) -> int:
    rc = 0
    for path in paths:
        try:
            print(yaml.safe_dump(load_config(path).to_dict(), sort_keys=False))
        except (ConfigError, OSError, yaml.YAMLError) as e:
            print(f"[FAIL] {e}", file=sys.stderr)
            rc = 2
    return rc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "validate":
        return _validate(args.configs)
    try:
        spec = sweep.load_spec(args.spec)
        jobs = sweep.plan(spec, args.only, args.rounds) if args.cmd == "sweep" else []
    except (ConfigError, OSError, yaml.YAMLError) as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 2
    load_dotenv(spec.root / ".env")
    try:
        if args.cmd == "preflight":
            return sweep.preflight(spec, oracle=args.oracle)
        return sweep.run(spec, jobs, dry_run=args.dry_run)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
