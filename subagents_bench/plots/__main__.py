"""``python -m subagents_bench.plots ANALYSIS_DIR --out FIGDIR``: the figures as PNG plus
``index.md`` with a caption for each."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .figures import FIGURES

TABLES = ("episodes", "conditions", "calls", "tools", "dispatches", "arbitrations", "interactivity",
          "hint_positions")


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):  # a table without rows, or from an older analysis
        return pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m subagents_bench.plots", description=__doc__.split("\n\n")[0])
    p.add_argument("analysis_dir", type=Path, help="output dir of python -m subagents_bench.analysis")
    p.add_argument("--out", required=True, type=Path, help="figure dir")
    args = p.parse_args(argv)
    tables = {name: _read(args.analysis_dir / f"{name}.csv") for name in TABLES}
    args.out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Figures of {args.analysis_dir.name}", "",
             "Drawn from the tables in the analysis dir; only valid trials enter a figure. Colour is the "
             "subagent strategy (grey: single agent) unless the legend says otherwise.", ""]
    for i, (name, draw) in enumerate(FIGURES.items(), 1):
        title, caption = draw(tables, args.out / f"{name}.png")
        lines += [f"## {i}. {title} (`{name}`)", "", f"![{title}]({name}.png)", "", caption, ""]
        print(f"figure: {args.out / name}.png", flush=True)
    (args.out / "index.md").write_text("\n".join(lines))
    print(f"figures: {args.out}/ (captions in index.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
