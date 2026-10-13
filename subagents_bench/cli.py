"""Command line: ``subagents-bench validate CONFIG...`` (the sweep commands come with sweep.py)."""

from __future__ import annotations

import argparse
import sys

import yaml

from .config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="subagents-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate", help="load experiment config YAML(s) and print the resolved config")
    p.add_argument("configs", nargs="+")
    args = parser.parse_args(argv)
    rc = 0
    for path in args.configs:
        try:
            print(yaml.safe_dump(load_config(path).to_dict(), sort_keys=False))
        except (ConfigError, OSError, yaml.YAMLError) as e:
            print(f"[FAIL] {e}", file=sys.stderr)
            rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
