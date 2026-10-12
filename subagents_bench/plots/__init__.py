"""Figures for the study: ``subagents-bench plot ANALYSIS_DIR --out FIGDIR``.

Reads the CSV tables that ``subagents-bench aggregate`` writes
(``docs/DESIGN.md`` section 6) and renders the 14 figures of section 7 as PNG
(dpi 200) and PDF, plus ``index.md`` with one captioned entry per figure and
a list of skipped figures with the reason.

Design rules (see :mod:`.style` for the visual vocabulary):

* matplotlib only, on bare ``Figure`` objects (Agg; no pyplot state);
* every figure and panel checks its inputs: a missing table or column skips
  that figure/panel with a log line, never the whole run;
* statistics come from the CSV CI columns when present; when a table lacks
  them they are recomputed from ``episodes.csv`` with the aggregator's
  definitions, and the caption says so;
* unknown values (e.g. an unpriced model's cost) stay NaN and are left out or
  labelled, never drawn as 0.

Modules: :mod:`.data` (loading, condition metadata, statistics),
:mod:`.style`, :mod:`.figures` (figures 1-10, 12-14), :mod:`.timelines`
(figure 11).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from matplotlib import rc_context

from .data import AnalysisData, Skip
from .figures import (
    Rendered,
    fig_arbiter,
    fig_calibration,
    fig_cost_effectiveness,
    fig_main_grid,
    fig_orchestrator_tokens,
    fig_parallelism,
    fig_pareto_cost,
    fig_pareto_time,
    fig_per_subagent,
    fig_spawn_counts,
    fig_task_heatmap,
    fig_time_breakdown,
    fig_token_composition,
    pareto_frontier,
)
from .style import RC
from .timelines import TIMELINES_INTRO, fig_timelines

log = logging.getLogger("subagents_bench.plots")

# DESIGN.md section 7 order (the index numbers follow it).
FIGURES: list[tuple[str, Callable[[AnalysisData], Iterator[Rendered]]]] = [
    ("main_grid", fig_main_grid),
    ("pareto_cost", fig_pareto_cost),
    ("pareto_time", fig_pareto_time),
    ("time_breakdown", fig_time_breakdown),
    ("token_composition", fig_token_composition),
    ("orchestrator_tokens", fig_orchestrator_tokens),
    ("per_subagent", fig_per_subagent),
    ("spawn_counts", fig_spawn_counts),
    ("parallelism", fig_parallelism),
    ("task_heatmap", fig_task_heatmap),
    ("timelines", fig_timelines),
    ("calibration", fig_calibration),
    ("cost_effectiveness", fig_cost_effectiveness),
    ("arbiter", fig_arbiter),
]
FIGURE_NAMES = [n for n, _ in FIGURES]
FORMATS = ("png", "pdf", "svg")

__all__ = ["FIGURES", "FIGURE_NAMES", "PlotReport", "WrittenFigure", "make_plots", "pareto_frontier", "register_cli"]


@dataclass
class WrittenFigure:
    figure: str  # entry in FIGURES this belongs to
    name: str  # output stem relative to the figure dir
    title: str
    caption: str
    uses: list[str]
    notes: list[str]
    files: list[Path]


@dataclass
class PlotReport:
    out_dir: Path
    index: Path | None = None
    written: list[WrittenFigure] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def figures_written(self) -> list[str]:
        return list(dict.fromkeys(w.figure for w in self.written))


def _save(r: Rendered, figure: str, out_dir: Path, formats: Iterable[str], dpi: int) -> WrittenFigure:
    files = []
    for fmt in formats:
        path = out_dir / f"{r.name}.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Strip timestamps so re-running on the same tables gives identical files.
        meta = {"CreationDate": None} if fmt == "pdf" else ({"Software": None} if fmt == "png" else {"Date": None})
        r.fig.savefig(path, format=fmt, dpi=dpi, metadata=meta, bbox_inches="tight", pad_inches=0.08)
        files.append(path)
    return WrittenFigure(figure, r.name, r.title, r.caption, list(r.uses), list(r.notes), files)


def make_plots(
    analysis_dir: str | Path,
    out_dir: str | Path,
    *,
    formats: Iterable[str] = ("png", "pdf"),
    conditions: list[str] | None = None,
    only: Iterable[str] | None = None,
    dpi: int = 200,
) -> PlotReport:
    """Render every figure that its inputs allow and write ``index.md``.

    ``conditions`` are condition names or shell globs; ``only`` restricts the
    figures by name (see :data:`FIGURE_NAMES`). Raises ``FileNotFoundError``
    for a missing analysis dir and ``ValueError`` for bad arguments.
    """
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.is_dir():
        raise FileNotFoundError(f"analysis dir not found: {analysis_dir}")
    fmts = [f.strip().lower().lstrip(".") for f in formats if f and f.strip()]
    bad = [f for f in fmts if f not in FORMATS]
    if bad or not fmts:
        raise ValueError(f"unsupported format(s) {bad or fmts}; choose from {', '.join(FORMATS)}")
    only_set = set(only) if only else None
    if only_set and not only_set <= set(FIGURE_NAMES):
        raise ValueError(f"unknown figure(s) {sorted(only_set - set(FIGURE_NAMES))}; choose from {FIGURE_NAMES}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = AnalysisData(analysis_dir, conditions)
    if conditions and data.all_conditions and not data.selected:
        raise ValueError(f"--conditions {conditions} matches none of {data.all_conditions}")
    report = PlotReport(out_dir)
    if only_set is None or "timelines" in only_set:
        # Episode selection can change between runs; drop stale timeline images first.
        for old in (out_dir / "timelines").glob("*"):
            if old.suffix.lstrip(".") in FORMATS and old.is_file():
                old.unlink()
    with rc_context(RC):
        for name, fn in FIGURES:
            if only_set is not None and name not in only_set:
                continue
            n_before = len(report.written)
            try:
                for r in fn(data):
                    try:
                        report.written.append(_save(r, name, out_dir, fmts, dpi))
                    finally:
                        r.fig.clear()
            except Skip as e:
                done = len(report.written) - n_before
                reason = str(e) + (f" (after {done} figure(s))" if done else "")
                log.info("skipped figure %s: %s", name, reason)
                report.skipped.append((name, reason))
                continue
            except Exception as e:  # one broken figure must not stop the others
                log.exception("figure %s failed", name)
                report.skipped.append((name, f"error: {type(e).__name__}: {e}"))
                continue
            if len(report.written) == n_before:
                report.skipped.append((name, "no figure produced"))
                log.info("skipped figure %s: no figure produced", name)
            else:
                log.info("wrote %s (%d file(s))", name, (len(report.written) - n_before) * len(fmts))
    report.index = write_index(report, analysis_dir, data, fmts, dpi)
    return report


def _link_format(fmts: list[str]) -> str:
    return "png" if "png" in fmts else ("svg" if "svg" in fmts else fmts[0])


def write_index(report: PlotReport, analysis_dir: Path, data: AnalysisData, fmts: list[str], dpi: int) -> Path:
    out = report.out_dir
    img = _link_format(fmts)
    lines = ["# Figures", ""]
    summary = analysis_dir / "summary.json"
    sweep = None
    if summary.exists():
        try:
            s = json.loads(summary.read_text())
            sweep = s.get("sweep") if isinstance(s, dict) else None
            if isinstance(s, dict) and s.get("synthetic"):
                lines += ["**Synthetic data.** These tables were generated by the test fixture "
                          "`tests/synth_tables.py`; none of the numbers below are measurements.", ""]
        except (OSError, ValueError):
            pass
    lines += [
        f"Rendered by `subagents-bench plot` from the tables in `{analysis_dir}`"
        + (f" (sweep `{sweep}`)" if sweep else "")
        + f". Each figure is saved as {', '.join(f.upper() for f in fmts)}"
        + (f" (PNG at {dpi} dpi)" if "png" in fmts else "")
        + ". Error bars are 95% intervals: Wilson for resolve rates and bootstrap (2000 resamples, seed 0) for "
        "means, taken from the `*_ci_lo`/`*_ci_hi` columns unless a caption says otherwise. An interval that "
        "cannot be computed (e.g. fewer than 2 valid values) gets no whisker rather than a zero-length one, and "
        "the figure's notes name those conditions. Colour encodes the "
        "strategy (blue parallel, orange sequential, aqua arbiter), lightness or marker shape the delegation scope "
        "(lighter / o = easy, s = medium, ^ = hard), grey the single-agent baselines; hatched bars and hollow "
        "markers are controls off the main strategy x scope grid, and `[manager]` tags a manager control (the "
        "orchestrator lacks the editing tools, so it must delegate). Invalid trials and trials flagged "
        "`duplicate` in `episodes.csv` are left out everywhere, with their dispatches, spawn calls and "
        "arbitrations; only the provider calibration uses every call.",
        "",
    ]
    sel = data.order(data.selected)
    if sel:
        lines += ["Conditions: " + ", ".join(f"`{c}`" for c in sel) + ".", ""]
    try:
        no_valid = data.resolve_stats().attrs.get("no_valid", [])
    except Skip:
        no_valid = []
    if no_valid:
        lines += ["Conditions with no valid trial (no resolve rate; left out of the per-episode figures, labelled in "
                  "main_grid and cost_effectiveness): " + ", ".join(f"`{c}`" for c in no_valid) + ".", ""]
    by_fig: dict[str, list[WrittenFigure]] = {}
    for w in report.written:
        by_fig.setdefault(w.figure, []).append(w)
    for idx, (name, _) in enumerate(FIGURES, start=1):
        ws = by_fig.get(name)
        if not ws:
            continue
        if name == "timelines":
            lines += [f"## {idx}. Episode timelines (`timelines/`)", "", TIMELINES_INTRO, ""]
            for w in ws:
                lines += [f"### {w.title}", "", f"![{w.title}]({w.name}.{img})", "", w.caption + " " + _files(w, out), ""]
            continue
        w = ws[0]
        text = w.caption
        if w.uses:
            text += " Data: " + "; ".join(dict.fromkeys(w.uses)) + "."
        if w.notes:
            text += " Notes: " + "; ".join(w.notes) + "."
        lines += [f"## {idx}. {w.title} (`{name}`)", "", f"![{w.title}]({w.name}.{img})", "", text, "", _files(w, out), ""]
    if report.skipped:
        lines += ["## Skipped figures", ""]
        lines += [f"- `{n}`: {why}" for n, why in report.skipped]
        lines.append("")
    path = out / "index.md"
    path.write_text("\n".join(lines))
    return path


def _files(w: WrittenFigure, out: Path) -> str:
    return "Files: " + ", ".join(f"[{p.suffix.lstrip('.')}]({p.relative_to(out).as_posix()})" for p in w.files) + "."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "plot",
        help="render the study's figures and index.md from an analysis dir",
        description="Render the figures of docs/DESIGN.md section 7 from the CSVs written by `aggregate`.",
    )
    p.add_argument("analysis_dir", help="directory written by `subagents-bench aggregate` (episodes.csv, ...)")
    p.add_argument("--out", help="figure directory (default: figures/<analysis dir name>)")
    p.add_argument("--formats", default="png,pdf", help="comma-separated formats: png, pdf, svg (default: png,pdf)")
    p.add_argument("--conditions", nargs="+", metavar="COND",
                   help="only these conditions (names or shell globs, space- or comma-separated); "
                        "baselines are drawn only if selected")
    p.add_argument("--only", nargs="+", metavar="FIGURE", choices=FIGURE_NAMES,
                   help=f"only these figures ({', '.join(FIGURE_NAMES)})")
    p.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200)")
    p.set_defaults(func=_cmd_plot)


def _cmd_plot(args: argparse.Namespace) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("fontTools", "matplotlib"):  # PDF font subsetting logs every glyph at INFO
        logging.getLogger(noisy).setLevel(logging.WARNING)
    conditions = [c.strip() for item in (args.conditions or []) for c in item.split(",") if c.strip()]
    out = Path(args.out) if args.out else Path("figures") / Path(args.analysis_dir).resolve().name
    try:
        report = make_plots(args.analysis_dir, out, formats=args.formats.split(","), conditions=conditions or None,
                            only=args.only, dpi=args.dpi)
    except (FileNotFoundError, ValueError) as e:
        print(f"plot: {e}", file=sys.stderr)
        return 2
    print(f"plot: wrote {len(report.written)} figure(s) to {out}; index: {report.index}")
    for name, why in report.skipped:
        print(f"plot: skipped {name}: {why}")
    return 0 if report.written else 1
