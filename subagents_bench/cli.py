"""Command-line entrypoint: ``subagents-bench ...``.

Each module registers its own subcommands via ``register_cli(subparsers)``;
every registered subparser sets ``func`` (args -> exit code):

* :mod:`subagents_bench.runner` -- ``validate``, ``run`` (local smoke test)
* :mod:`subagents_bench.sweep` -- ``preflight``, ``sweep``, ``tasks``
* :mod:`subagents_bench.analysis` -- ``aggregate``, ``check-run``,
  ``completeness``, ``calibrate``
* :mod:`subagents_bench.plots` -- ``plot``

See ``docs/DESIGN.md`` section 8 and ``docs/REPRODUCE.md``.
"""

from __future__ import annotations

import argparse
import importlib
import sys

# Modules that register subcommands, in --help order.
MODULES = ("runner", "sweep", "analysis", "plots")


def build_parser() -> argparse.ArgumentParser:
    """The parser with every module's subcommands. A module that fails to
    import or register is skipped with a warning on stderr, so one broken
    module (e.g. a missing optional dependency) does not disable the rest."""
    parser = argparse.ArgumentParser(prog="subagents-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in MODULES:
        try:
            module = importlib.import_module(f"{__package__}.{name}")
            module.register_cli(sub)
        except Exception as e:  # noqa: BLE001 - reported, the other modules still load
            print(f"subagents-bench: warning: the {name!r} subcommands are unavailable "
                  f"({type(e).__name__}: {e})", file=sys.stderr)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
