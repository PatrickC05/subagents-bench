"""The figures of ``docs/DESIGN.md`` section 7 (all but the timelines).

Each ``fig_*`` function takes an :class:`~.data.AnalysisData` and yields
:class:`Rendered` figures (normally one). It raises :class:`~.data.Skip` when
its inputs are missing or empty; a missing *optional* input only drops that
panel, logs it, and is mentioned in the caption's notes. The caption says
what is plotted, how to read it, and which CSV columns it uses; the driver
(:mod:`subagents_bench.plots`) writes it into ``index.md``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .data import (
    DIFFICULTIES,
    ROLE_NAMES,
    ROLES,
    SCOPES,
    STRATEGIES,
    AnalysisData,
    CondInfo,
    Skip,
    as_bool,
    bootstrap_mean,
    bootstrap_ratio,
    num,
    wilson,
)
from .style import (
    AXIS,
    BASELINE_STYLES,
    INK,
    INK2,
    KIND_COLORS,
    MUTED,
    OVERHEAD_COLOR,
    ROLE_COLORS,
    ROLE_STACK,
    ROUND_RAMP,
    SCOPE_TINT,
    SEQ_CMAP,
    STRATEGY_COLORS,
    SURFACE,
    TOOL_COLOR,
    bar_style,
    base_color,
    ci_known,
    condition_legend,
    errs,
    fill_color,
    hgrid,
    log_scale_if_wide,
    new_figure,
    outside_legend,
    panel_axes,
    point_style,
    tint,
    vgrid,
    wrap,
)

log = logging.getLogger("subagents_bench.plots")


@dataclass
class Rendered:
    """One figure ready to save, with its index.md entry."""

    name: str  # output stem relative to the figure dir ("main_grid", "timelines/<...>")
    fig: Figure
    title: str
    caption: str
    uses: list[str]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def human(x: float, _pos: object = None) -> str:
    """Compact number: 1.2k, 3.4M, 5G ..."""
    if not np.isfinite(x):
        return ""
    a = abs(x)
    for div, suf in ((1e18, "E"), (1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            return f"{x / div:.3g}{suf}"
    return f"{x:.3g}"


def usd(x: float, _pos: object = None) -> str:
    if not np.isfinite(x):
        return ""
    return f"${x:.2g}" if abs(x) < 0.01 and x != 0 else f"${x:.3g}"


HUMAN = FuncFormatter(human)
USD = FuncFormatter(usd)


def pareto_frontier(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points when lower x and higher y are better
    (the "upper-left" frontier). Non-finite points are never on it."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.zeros(len(x), bool)
    ok = np.isfinite(x) & np.isfinite(y)
    best = -np.inf
    for i in np.lexsort((-y, x)):  # by x ascending, ties: higher y first
        if ok[i] and y[i] > best:
            keep[i] = True
            best = y[i]
    return keep


def _infos(data: AnalysisData, names) -> list[CondInfo]:
    return [data.meta[n] for n in names if n in data.meta]


def _baseline_handles(infos: list[CondInfo], *, band: bool = True, empty: frozenset[str] = frozenset()) -> list:
    """Legend lines for the baselines; one in ``empty`` (no valid trials) is labelled so, not drawn."""
    out = []
    for i in infos:
        if i.baseline:
            c, ls, _ = BASELINE_STYLES[i.baseline_index % len(BASELINE_STYLES)]
            what = "; no valid trials, not drawn" if i.name in empty else ("; band = 95% CI" if band else "")
            out.append(Line2D([], [], color=c, linestyle=ls, linewidth=1.4, alpha=0.4 if i.name in empty else 1.0,
                              label=f"{i.name} (baseline{what})"))
    return out


def _bands(ax, data: AnalysisData, stats: pd.DataFrame, *, orient: str = "h",
           mean: str = "rate", lo: str = "lo", hi: str = "hi") -> list[str]:
    """Baselines as reference bands (line = estimate, band = CI). Returns names drawn."""
    drawn = []
    for n in stats.index:
        info = data.meta.get(n)
        if info is None or not info.baseline:
            continue
        c, ls, _ = BASELINE_STYLES[info.baseline_index % len(BASELINE_STYLES)]
        m, l, h = (float(stats.loc[n, k]) for k in (mean, lo, hi))
        if not np.isfinite(m):
            continue
        span, line = (ax.axhspan, ax.axhline) if orient == "h" else (ax.axvspan, ax.axvline)
        if np.isfinite(l) and np.isfinite(h):
            span(l, h, color=c, alpha=0.10, linewidth=0, zorder=0)
        line(m, color=c, linestyle=ls, linewidth=1.3, zorder=1)
        drawn.append(n)
    return drawn


def _block_separators(ax, data: AnalysisData, names: list[str], orient: str = "h") -> None:
    """Hairlines between the baseline / grid / control blocks of an ordered list."""
    def block(n: str) -> int:
        i = data.meta[n]
        return 0 if i.baseline else (2 if i.control else 1)
    for k in range(1, len(names)):
        if block(names[k]) != block(names[k - 1]):
            (ax.axhline if orient == "h" else ax.axvline)(k - 0.5, color=AXIS, linewidth=0.7, zorder=0)


def _labels(data: AnalysisData, names) -> list[str]:
    """Tick labels: condition names, with manager controls tagged."""
    return [data.meta[n].label if n in data.meta else n for n in names]


def _ci_gaps(names, mean, lo, hi) -> tuple[list[str], list[str]]:
    """(unknown, unbounded): conditions with a value whose 95% interval is
    unknown (a NaN bound, or zero width), or known below but unbounded above
    (hi = +inf)."""
    mean, lo, hi = (np.asarray(v, float) for v in (mean, lo, hi))
    has = np.isfinite(mean)
    unbounded = has & np.isfinite(lo) & np.isposinf(hi)
    unknown = has & ~ci_known(lo, hi) & ~unbounded
    return [n for n, u in zip(names, unknown) if u], [n for n, u in zip(names, unbounded) if u]


def _ci_note(unknown: list[str], where: str = "") -> list[str]:
    """Notes line for conditions drawn without a whisker because their interval is unknown.

    ``where`` completes "CI unavailable ...", e.g. "for the wall-clock" or "in every panel".
    """
    if not unknown:
        return []
    return [f"CI unavailable{f' {where}' if where else ''} (fewer than 2 valid values, an unknown one, or all "
            f"equal; no whisker drawn): {', '.join(unknown)}"]


def _panels_phrase(titles: list[str], n_panels: int) -> str:
    """'in every panel' or "in 'a', 'b'" for a note shared by several panels."""
    return "in every panel" if len(titles) == n_panels > 1 else "in " + ", ".join(titles)


UNBOUNDED_HANDLE_LABEL = "95% interval unbounded above (arrow to the panel edge)"


def _unbounded_handle() -> Line2D:
    return Line2D([], [], color=INK, linewidth=0.9, marker=">", markersize=4, label=UNBOUNDED_HANDLE_LABEL)


def _hbars(ax, data: AnalysisData, names: list[str], mean, lo, hi, *,
           missing: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
    """Horizontal bars per condition with CI whiskers; missing values get a text note.

    Whiskers are drawn only for known intervals. One unbounded above runs from
    its lower bound to the right edge of the axes, with an arrow head.
    Returns (conditions without a CI, conditions with an unbounded CI).
    """
    mean = np.asarray(mean, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    y = np.arange(len(names))
    for k, n in enumerate(names):
        if np.isfinite(mean[k]):
            ax.barh(k, mean[k], height=0.62, **bar_style(data.meta[n]))
        else:
            ax.text(0, k, "  " + (missing or {}).get(n, "n/a"), va="center", ha="left", fontsize=7, color=MUTED)
    ok = np.isfinite(mean)
    if ok.any():
        ax.errorbar(mean[ok], y[ok], xerr=errs(mean[ok], lo[ok], hi[ok]), fmt="none", ecolor=INK,
                    elinewidth=0.9, capsize=2, zorder=3)
    unknown, unbounded = _ci_gaps(names, mean, lo, hi)
    for n in unbounded:
        k = names.index(n)
        # Lower cap as a marker (it also extends the data limits), then an arrow to the axes edge.
        ax.plot([lo[k]], [k], marker="|", markersize=4, markeredgewidth=0.9, color=INK, zorder=3)
        ax.annotate("", xy=(1.0, k), xycoords=ax.get_yaxis_transform(), xytext=(lo[k], k), textcoords="data",
                    arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 0.9, "shrinkA": 0, "shrinkB": 0,
                                "mutation_scale": 7}, zorder=3)
    ax.set_yticks(y, _labels(data, names))
    ax.set_ylim(len(names) - 0.5, -0.5)
    hgrid(ax)
    _block_separators(ax, data, names)
    return unknown, unbounded


def _src(df: pd.DataFrame, default: str) -> str:
    return df.attrs.get("source", default)


def _cols(table: str, cols: list[str]) -> str:
    return f"`{table}.csv` ({', '.join(cols)})"


# ---------------------------------------------------------------------------
# 1. main_grid
# ---------------------------------------------------------------------------


DELEGATED_PAD = 13  # points between the axis and the x tick labels, room for the delegated counts


def _rate_bars(ax, data: AnalysisData, st: pd.DataFrame, empty: list[str] = ()) -> dict[str, Any]:
    """Resolve-rate bars of one main_grid panel.

    ``empty`` are conditions without a valid trial in this panel: their slot
    is kept and labelled instead of silently closing up. Under each subagent
    bar, the delegated count (valid episodes that spawned subagents / valid
    episodes); a hollow circle marks the per-protocol rate among them.
    Returns the names shown plus what the notes need.
    """
    st = st.set_index("condition")
    meta = data.meta
    empty = [n for n in empty if n in meta and n not in st.index]
    names = data.order(list(st.index) + empty)
    grid = [n for n in names if not meta[n].baseline and not meta[n].control]
    controls = [n for n in names if meta[n].control and not meta[n].baseline]
    bases = [n for n in names if meta[n].baseline]
    strategies = [s for s in STRATEGIES if any(meta[n].strategy == s for n in grid)]
    scopes = [s for s in SCOPES if any(meta[n].scope == s for n in grid)]
    w = 0.8 / max(len(strategies), 1)
    xs: list[float] = []
    rows: list[str] = []
    widths: list[float] = []
    ticks: list[float] = []
    labels: list[str] = []
    for i, sc in enumerate(scopes):
        for j, sg in enumerate(strategies):
            hit = [n for n in grid if meta[n].scope == sc and meta[n].strategy == sg]
            if hit:
                xs.append(i + (j - (len(strategies) - 1) / 2) * w)
                rows.append(hit[0])
                widths.append(w * 0.9)
        ticks.append(i)
        labels.append(f"{sc} scope")
    # Baselines become bars only when there is nothing else to compare against.
    as_bars = bases if not grid and not controls else []
    x0 = len(scopes) + (0.15 if scopes else 0)
    for k, n in enumerate(controls + as_bars):
        x = x0 + k * 0.8
        xs.append(x)
        rows.append(n)
        widths.append(0.55)
        ticks.append(x)
        labels.append(wrap(meta[n].label, 14))
    out: dict[str, Any] = {"names": names, "no_ci": [], "never_delegated": [], "per_protocol": []}
    deleg_drawn = False
    for x, n, bw in zip(xs, rows, widths):
        if n not in st.index:
            ax.text(x, 0.03, "no valid trials", rotation=90, ha="center", va="bottom", fontsize=6.3, color=MUTED,
                    zorder=4)
            continue
        r = st.loc[n]
        ax.bar(x, r["rate"], width=bw, **bar_style(meta[n]), zorder=2)
        ax.errorbar(x, r["rate"], yerr=errs([r["rate"]], [r["lo"]], [r["hi"]]), fmt="none", ecolor=INK,
                    elinewidth=1.0, capsize=2.5, zorder=3)
        out["no_ci"] += _ci_gaps([n], [r["rate"]], [r["lo"]], [r["hi"]])[0]
        top = r["hi"] if np.isfinite(r["hi"]) else r["rate"]
        if np.isfinite(r.get("n_valid", np.nan)) and np.isfinite(r.get("n_resolved", np.nan)):
            ax.text(x, top + 0.015, f"{int(r['n_resolved'])}/{int(r['n_valid'])}", ha="center", va="bottom",
                    fontsize=6.3, color=INK2, zorder=4)
        nd, nv = r.get("n_delegated", np.nan), r.get("n_valid", np.nan)
        if meta[n].baseline or not (np.isfinite(nd) and np.isfinite(nv)):
            continue
        ax.annotate(f"{int(nd)}/{int(nv)}", xy=(x, 0), xycoords=("data", "axes fraction"), xytext=(0, -4),
                    textcoords="offset points", ha="center", va="top", fontsize=6.0,
                    color=ROLE_COLORS["arb"] if nd == 0 else INK2)
        deleg_drawn = True
        if nd == 0:
            out["never_delegated"].append(n)
        rd = r.get("rate_delegated", np.nan)
        if nd > 0 and np.isfinite(rd):
            ax.plot(x, rd, marker="o", markersize=4.4, markerfacecolor=SURFACE, markeredgecolor=INK,
                    markeredgewidth=1.0, linestyle="none", zorder=5)
            out["per_protocol"].append(n)
    if not as_bars:
        _bands(ax, data, st.loc[[b for b in bases if b in st.index]])
    ax.set_xticks(ticks, labels)
    ax.tick_params(axis="x", length=0, pad=DELEGATED_PAD if deleg_drawn else 3.5)
    out["delegated_row"] = deleg_drawn
    if xs:
        ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    vgrid(ax)
    return out


def _task_mix_note(st: pd.DataFrame) -> list[str]:
    """Note the conditions whose valid trials cover fewer tasks than the others'."""
    if "n_tasks_valid" not in st.columns:
        return []
    nt = num(st.set_index("condition")["n_tasks_valid"]).dropna()
    if nt.empty or nt.nunique() < 2:
        return []
    full = int(nt.max())
    short = [f"{n} ({int(v)} of {full})" for n, v in nt.items() if v < full]
    return [f"valid trials cover fewer tasks for {', '.join(short)}: their pooled rate also reflects which tasks "
            "are missing (conditions.csv resolve_rate_task_balanced weights tasks equally; paired.csv compares "
            "matched trials only)"]


def fig_main_grid(data: AnalysisData) -> Iterator[Rendered]:
    overall = data.resolve_stats()
    source = _src(overall, "conditions.csv")
    panels = [("all tasks", overall, list(overall.attrs.get("no_valid", [])))]
    notes: list[str] = []
    uses = [source]
    if np.isfinite(overall["n_delegated"]).any():
        uses.append(overall.attrs["uptake_source"])
    try:
        byd = data.resolve_stats(by_difficulty=True)
        byd_empty = list(byd.attrs.get("no_valid", []))
        diffs = [d for d in DIFFICULTIES if d in set(byd["difficulty"])] + \
            sorted(set(byd["difficulty"]) - set(DIFFICULTIES))
        if len(diffs) > 1 or (diffs and len(byd) != len(overall)):
            for d in diffs:
                panels.append((f"{d} tasks (Terminal-Bench label)", byd[byd["difficulty"] == d],
                               [c for c, dd in byd_empty if dd == d]))
            uses.append(_src(byd, "conditions_by_difficulty.csv"))
    except Skip as e:
        notes.append(f"per-difficulty panels not drawn ({e})")
        log.info("main_grid: per-difficulty panels skipped: %s", e)
    n_units = max(len(SCOPES) + 0.8 * sum(1 for i in data.meta.values() if i.control), 2.0)
    fig = new_figure(max(4.2, 0.6 + 1.25 * n_units) * len(panels) * 0.92, 4.6)
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    shown: set[str] = set()
    per_protocol = deleg_row = False
    for k, (ax, (title, st, empty)) in enumerate(zip(axes, panels)):
        res = _rate_bars(ax, data, st, empty)
        shown |= set(res["names"])
        per_protocol |= bool(res["per_protocol"])
        deleg_row |= bool(res["delegated_row"])
        ax.set_title(title)
        if empty:
            notes.append(f"no valid trials{'' if k == 0 else f' ({title})'}, slot labelled, nothing drawn: "
                         + ", ".join(empty))
        notes += _ci_note(res["no_ci"], f"in '{title}'")
        if k == 0:
            if res["never_delegated"]:
                notes.append("never delegated (no valid episode spawned subagents, so the bar measures the "
                             "orchestrator alone): " + ", ".join(data.order(res["never_delegated"])))
            notes += _task_mix_note(st)
            if res["delegated_row"]:
                ax.annotate("delegated", xy=(0, 0), xycoords="axes fraction", xytext=(-4, -4),
                            textcoords="offset points", ha="right", va="top", fontsize=6.0, color=INK2)
    axes[0].set_ylabel("resolve rate (share of valid trials, Wilson 95% CI)")
    infos = _infos(data, data.order(shown))
    handles = condition_legend([i for i in infos if not i.baseline], kind="patch", scope_legend=True)
    handles += _baseline_handles(infos, empty=frozenset(panels[0][2]))
    if per_protocol:
        handles.append(Line2D([], [], marker="o", markersize=4.4, markerfacecolor=SURFACE, markeredgecolor=INK,
                              markeredgewidth=1.0, linestyle="none",
                              label="resolve rate of the episodes that delegated (per protocol)"))
    outside_legend(fig, handles, ncols=min(len(handles), 4))
    fig.suptitle("Resolve rate by subagent strategy and delegation scope")
    per_diff = len(panels) > 1
    if not deleg_row and any(not data.meta[n].baseline for n in shown if n in data.meta):
        notes.append("no 'delegated' row: neither the resolve-rate table (n_delegated) nor episodes.csv "
                     "(delegated, spawn_calls) has the treatment uptake (re-run aggregate to add it)")
    caption = (
        "Headline result. Each bar is the share of valid trials whose verifier reward was 1, for one subagent "
        "condition; bars are grouped by delegation scope on the x axis and coloured by strategy (blue parallel, "
        "orange sequential, aqua arbiter), with a lighter fill for easier scopes. Whiskers are Wilson 95% "
        "intervals and the k/n label above each bar is solved/valid trials. Grey horizontal lines are the "
        "single-agent baselines (topology single) and the shaded band around each is its Wilson 95% interval. "
        "The intervals pool every valid trial of a condition, so they describe these tasks and their "
        "trial-to-trial randomness, not Terminal-Bench as a whole; to compare a condition with a baseline use "
        "paired.csv (McNemar test on matched task and round pairs), not whether the intervals overlap. "
        + ("The small k/n under each subagent bar (row 'delegated') counts the valid episodes whose orchestrator "
           "spawned subagents at least once, out of all valid episodes (red when none did: that condition never "
           "received its treatment), and the hollow circle is the resolve rate among those episodes only (per "
           "protocol; the bar is the intention-to-treat rate over all of them). " if deleg_row else "")
        + "Conditions off the strategy x scope grid of "
        "the main subagent model (for example the same-model control, or a manager control whose orchestrator "
        "lacks the editing tools, tagged [manager]) are hatched bars on the right. A condition with no valid "
        "trial keeps its slot, labelled 'no valid trials', and is listed in the notes"
        + (". The other panels repeat the comparison per Terminal-Bench difficulty label." if per_diff else ".")
    )
    yield Rendered("main_grid", fig, "Resolve rate: strategy x scope vs baselines", caption, uses, notes)


# ---------------------------------------------------------------------------
# 2-3. Pareto plots
# ---------------------------------------------------------------------------


def _pareto(data: AnalysisData, name: str, suptitle: str, specs: list[tuple[str, str, str, bool, object]],
            heading: str, caption: str, extra_notes: list[str] | None = None) -> Iterator[Rendered]:
    stats = data.resolve_stats()
    rs = stats.set_index("condition")[["rate", "lo", "hi"]]
    rs.columns = ["y", "ylo", "yhi"]
    panels = []
    notes: list[str] = list(extra_notes or [])
    x_gaps: dict[tuple[str, ...], list[str]] = {}  # conditions without an x interval -> panels
    if stats.attrs.get("no_valid"):
        notes.append("no valid trials (no resolve rate, not drawn): " + ", ".join(stats.attrs["no_valid"]))
    uses = [_src(stats, "conditions.csv")]
    for title, metric, xlabel, strict, fmt in specs:
        try:
            m = data.cond_metric(metric, strict_nan=strict)
        except Skip as e:
            notes.append(f"panel '{title}' not drawn: {e}")
            log.info("%s: panel %s skipped: %s", name, title, e)
            continue
        j = m.join(rs, how="inner")
        ok = j[np.isfinite(j["mean"]) & np.isfinite(j["y"])]
        dropped = [n for n in j.index if n not in ok.index]
        if ok.empty:
            notes.append(f"panel '{title}' not drawn: no finite {metric} values (e.g. models unpriced in this scheme)")
            log.info("%s: panel %s skipped: no finite %s", name, title, metric)
            continue
        if dropped:
            notes.append(f"'{title}': no value (NaN) for {', '.join(dropped)}")
        no_ci = _ci_gaps(list(ok.index), ok["mean"], ok["lo"], ok["hi"])[0]
        if no_ci:
            x_gaps.setdefault(tuple(no_ci), []).append(f"'{title}'")
        panels.append((title, xlabel, ok, dropped, fmt))
        uses.append(_src(m, metric))
    if not panels:
        raise Skip("; ".join(notes) or "no metrics available")
    for which, titles in x_gaps.items():
        notes += _ci_note(list(which), "for x " + _panels_phrase(titles, len(panels)))
    notes += _ci_note(_ci_gaps(list(rs.index), rs["y"], rs["ylo"], rs["yhi"])[0], "for the resolve rate")
    ncols = min(len(panels), 3)
    nrows = math.ceil(len(panels) / ncols)
    fig = new_figure(4.1 * ncols + 0.4, 3.5 * nrows + 1.0)
    axes = panel_axes(fig, len(panels), max_cols=3, sharey=True)
    shown: set[str] = set()
    for ax, (title, xlabel, df, dropped, fmt) in zip(axes, panels):
        for n, r in df.iterrows():
            info = data.meta[n]
            ax.errorbar(r["mean"], r["y"], xerr=errs([r["mean"]], [r["lo"]], [r["hi"]]),
                        yerr=errs([r["y"]], [r["ylo"]], [r["yhi"]]), fmt="none",
                        ecolor=tint(base_color(info), 0.35), elinewidth=0.9, capsize=1.8, zorder=2)
            ax.plot(r["mean"], r["y"], zorder=4, **point_style(info))
            shown.add(n)
        front = df[pareto_frontier(df["mean"].to_numpy(), df["y"].to_numpy())].sort_values("mean")
        if len(front):
            ax.step(front["mean"], front["y"], where="post", color=INK, linewidth=1.0, alpha=0.75, zorder=3)
            for k, (n, r) in enumerate(front.iterrows()):
                # Alternate above/below, but never above a point at the top (it would hit the panel title).
                above = r["y"] < 0.1 or (r["y"] <= 0.9 and k % 2 == 0)
                ax.annotate(wrap(data.meta[n].label, 20), (r["mean"], r["y"]), xytext=(6, 7 if above else -8),
                            textcoords="offset points", fontsize=6.3, color=INK2,
                            va="bottom" if above else "top", zorder=2.5)  # under the markers: data stays visible
        if not log_scale_if_wide(ax, list(df["mean"]) + list(df["lo"]) + list(df["hi"]), "x", 15):
            ax.xaxis.set_major_locator(MaxNLocator(5))
        if fmt is not None:
            ax.xaxis.set_major_formatter(fmt)
        ax.set_xlabel(xlabel)
        ax.set_title(title + (f"\n({len(dropped)} condition(s) without a value)" if dropped else ""), fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.margins(x=0.12)
    for ax in axes:
        ax.set_ylabel("resolve rate (Wilson 95% CI)")
        ax.yaxis.set_tick_params(labelleft=True)
    handles = condition_legend(_infos(data, data.order(shown)), kind="marker")
    handles.append(Line2D([], [], color=INK, linewidth=1.0, drawstyle="steps-post", label="Pareto frontier"))
    outside_legend(fig, handles, ncols=min(len(handles), 5))
    fig.suptitle(suptitle)
    yield Rendered(name, fig, heading, caption, uses, notes)


def _has_finite(data: AnalysisData, metric: str) -> bool:
    """Whether conditions.csv (<metric>_mean) or episodes.csv (<metric>) has any finite value."""
    for table, col in (("conditions", f"{metric}_mean"), ("episodes", metric)):
        df = data.table(table)
        if df is not None and col in df.columns and np.isfinite(num(df[col])).any():
            return True
    return False


RUNTIME_NOTE = ("no pricing scheme has a finite cost (was `aggregate` run without configs/pricing.yaml?); "
                "showing the cost logged at run time from the config price table instead")


def fig_pareto_cost(data: AnalysisData) -> Iterator[Rendered]:
    specs = [(f"{s} prices", f"cost_{s}_total", f"mean cost per episode (USD, {s})", True, USD)
             for s in data.schemes()]
    extra: list[str] = []
    if not any(_has_finite(data, f"cost_{s}_total") for s in data.schemes()) and _has_finite(data, "cost_runtime_total"):
        log.info("pareto_cost: %s", RUNTIME_NOTE)
        extra.append(RUNTIME_NOTE)
        specs.append(("run-time price table (logged)", "cost_runtime_total",
                      "mean cost per episode (USD, logged at run time)", True, USD))
    specs.append(("compute (FLOPs)", "flops_total", "mean FLOPs per episode (2 x active params x tokens)", True, HUMAN))
    caption = (
        "Cost-effectiveness frontier. Each point is a condition: y is its resolve rate (Wilson 95% interval), x the "
        "mean cost of one episode (bootstrap 95% interval) under one pricing scheme, recomputed offline from the "
        "logged tokens; the last panel uses FLOPs (2 x active parameters x tokens) instead of dollars. Colour is "
        "strategy, marker shape is scope (o easy, s medium, ^ hard, lighter fill = easier), grey stars/crosses "
        "are the single-agent baselines and hollow markers are controls (a manager control, whose orchestrator "
        "lacks the editing tools, says so in the legend). The black staircase is the Pareto frontier: "
        "conditions that no other condition beats on both axes at once (cheaper and at least as accurate); points "
        "up and to the left are better, and only frontier points are labelled. A condition with no price for one "
        "of its models in a scheme has an unknown (NaN) cost and is left out of that panel, never drawn at 0. "
        "An interval that cannot be computed (fewer than 2 valid episodes) gets no whisker, and a condition with "
        "no valid trial has no point; the notes list both. x is log-scaled when values span more than 15x."
    )
    yield from _pareto(data, "pareto_cost", "Resolve rate vs cost per episode", specs,
                       "Pareto: resolve rate vs cost (per pricing scheme) and FLOPs", caption, extra)


def fig_pareto_time(data: AnalysisData) -> Iterator[Rendered]:
    specs = [("measured wall-clock", "wall_clock_sec", "mean wall-clock per episode (s)", False, None)]
    specs += [(f"simulated: {p}", f"sim_{p}_wall_sec", f"mean simulated wall-clock (s), profile {p}", True, None)
              for p in data.profiles() if p != "measured"]
    caption = (
        "Latency frontier. Same encoding as pareto_cost, with x the mean wall-clock time of one episode "
        "(bootstrap 95% interval). The first panel is the measured agent wall-clock; each further panel replays the "
        "same trajectories under a simulated serving profile (critical path recomputed with that profile's TTFT and "
        "decode speed; tool time kept as measured), so the frontier can be compared across serving assumptions. "
        "The `measured` replay profile is omitted here because it reproduces the measured critical path (see "
        "time_breakdown). Up and to the left is better."
    )
    yield from _pareto(data, "pareto_time", "Resolve rate vs wall-clock per episode", specs,
                       "Pareto: resolve rate vs wall-clock (measured and simulated)", caption)


# ---------------------------------------------------------------------------
# 4. time_breakdown
# ---------------------------------------------------------------------------

CP_COMPONENTS = [
    ("cp_orch_llm_sec", "orchestrator LLM", ROLE_COLORS["orch"]),
    ("cp_arb_llm_sec", "arbiter LLM (judge + debate)", ROLE_COLORS["arb"]),
    ("cp_sub_llm_sec", "subagent LLM", ROLE_COLORS["sub"]),
    ("cp_tool_sec", "tools", TOOL_COLOR),
    ("overhead_sec", "overhead (wall-clock - critical path)", OVERHEAD_COLOR),
]


def fig_time_breakdown(data: AnalysisData) -> Iterator[Rendered]:
    eps = data.episodes()
    comps = [c for c in CP_COMPONENTS if c[0] in eps.columns]
    missing = [c[0] for c in CP_COMPONENTS if c[0] not in eps.columns]
    if len(comps) < 2:
        raise Skip("episodes.csv lacks the critical-path columns (cp_*_sec, overhead_sec)")
    notes = [f"components not available: {', '.join(missing)}"] if missing else []
    cols = [c[0] for c in comps]
    means = eps[["condition"]].join(eps[cols].apply(num)).groupby("condition").mean()
    names = data.order(means.index)
    fig = new_figure(9.0, 1.6 + 0.40 * len(names))
    ax = fig.subplots()
    y = np.arange(len(names))
    left = np.zeros(len(names))
    # Spawn groups' segments start after the orchestrator's: the arbiter's judge and debate calls run inside them.
    group_left = np.zeros(len(names))
    neg = []
    for col, label, color in comps:
        v = means.loc[names, col].to_numpy(float)
        if col == "overhead_sec" and (v < 0).any():
            neg = [n for n, x in zip(names, v) if x < 0]
        v = np.nan_to_num(np.clip(v, 0, None))
        ax.barh(y, v, left=left, height=0.56, color=color, edgecolor=SURFACE, linewidth=1.0, label=label, zorder=2)
        left += v
        if col == "cp_orch_llm_sec":
            group_left = left.copy()
    if neg:
        notes.append(f"negative mean overhead clipped to 0 for {', '.join(neg)}")
    uses = [_cols("episodes", cols + ["valid"])]
    handles = [Patch(facecolor=c, edgecolor=c, label=l) for _, l, c in comps]
    try:
        wc = data.cond_metric("wall_clock_sec").reindex(names)
        m = wc["mean"].to_numpy(float)
        ok = np.isfinite(m)
        # The mean always gets its tick; the whisker only where the interval is known.
        ax.errorbar(m[ok], y[ok], xerr=errs(m[ok], wc["lo"][ok], wc["hi"][ok]),
                    fmt="|", color=INK, markersize=8, elinewidth=1.0, capsize=2.5, zorder=4)
        handles.append(Line2D([], [], color=INK, marker="|", markersize=8, label="mean wall-clock, 95% CI"))
        uses.append(_src(wc, "wall_clock_sec"))
        notes += _ci_note(_ci_gaps(names, m, wc["lo"], wc["hi"])[0], "for the wall-clock")
    except Skip as e:
        notes.append(f"no wall-clock CI ({e})")
    if "sub_busy_sec" in eps.columns:
        busy = eps[["condition"]].join(num(eps["sub_busy_sec"])).groupby("condition").mean().reindex(names)["sub_busy_sec"]
        b = busy.to_numpy(float)
        ok = np.isfinite(b) & (b > 0)
        yb = y - 0.37  # just above each bar
        if "cp_group_sec" in eps.columns:
            grp = eps[["condition"]].join(num(eps["cp_group_sec"])).groupby("condition").mean()
            g = grp.reindex(names)["cp_group_sec"].to_numpy(float)
            okg = ok & np.isfinite(g)
            hidden_drawn = False
            for k in np.flatnonzero(okg):
                x0, x1, xb = group_left[k], group_left[k] + g[k], group_left[k] + b[k]
                ax.plot([x0, x0, x1, x1], [y[k] - 0.29, yb[k], yb[k], y[k] - 0.29], color=INK, linewidth=0.9,
                        solid_joinstyle="miter", zorder=5)
                if xb > x1:
                    ax.plot([x1, xb], [yb[k], yb[k]], color=INK, linestyle=":", linewidth=1.0, zorder=5)
                    hidden_drawn = True
            if okg.any():
                handles.append(Line2D([], [], color=INK, linewidth=0.9, marker="|", markersize=5,
                                      label="critical-path time inside spawn groups (cp_group_sec)"))
            if hidden_drawn:  # legend only for what is drawn
                handles.append(Line2D([], [], color=INK, linewidth=1.0, linestyle=":",
                                      label="work hidden by parallelism (sub_busy_sec - cp_group_sec)"))
            uses.append(_cols("episodes", ["cp_group_sec"]))
        else:
            notes.append("no spawn-group bracket: episodes.csv lacks cp_group_sec")
        if ok.any():
            ax.plot((group_left + b)[ok], yb[ok], linestyle="none", marker="D", markersize=5, markerfacecolor=SURFACE,
                    markeredgecolor=INK, markeredgewidth=1.1, zorder=6)
            handles.append(Line2D([], [], linestyle="none", marker="D", markersize=5, markerfacecolor=SURFACE,
                                  markeredgecolor=INK, label="total subagent work (sub_busy_sec), from the same start"))
        else:
            notes.append("no condition has subagent work (sub_busy_sec is 0 or unknown everywhere): no bracket "
                         "or diamond drawn")
        uses.append(_cols("episodes", ["sub_busy_sec"]))
    ax.set_yticks(y, _labels(data, names))
    ax.set_ylim(len(names) - 0.5, -0.5)
    _block_separators(ax, data, names)
    hgrid(ax)
    ax.set_xlabel("seconds per episode (mean over valid trials)")
    ax.set_title("Where the wall-clock goes: mean critical-path composition per condition")
    outside_legend(fig, handles, ncols=3)
    caption = (
        "Each bar decomposes the mean wall-clock of one condition into the measured critical path: orchestrator LLM "
        "calls, arbiter judge/debate calls, subagent LLM calls and tool execution on the longest chain (for a "
        "parallel spawn group only the slowest subagent counts; for a sequential group all of them), plus overhead "
        "= wall-clock minus critical path (harness and scheduling time not attributed to any event). The bar's "
        "length is the mean wall-clock, with its bootstrap 95% interval as the black whisker (no whisker when the "
        "interval is unknown; the notes say for which). Above each bar of a condition that spawned subagents, a "
        "bracket starts where the spawn groups' segments start (the end of the orchestrator LLM segment; the "
        "arbiter's judge and debate calls run inside the groups) and spans the mean critical-path time inside "
        "spawn groups (cp_group_sec). The white diamond, measured from the same start, is the mean total subagent "
        "work (sub_busy_sec, the sum of all dispatch durations). The dotted gap from the bracket's end to the "
        "diamond is work that parallelism took off the critical path (sub_busy_sec - cp_group_sec; their ratio is "
        "the parallel speedup). A diamond at or inside the bracket's end means nothing was hidden: the groups "
        "cost at least as much critical time as the subagents worked, as in sequential groups or when judging "
        "dominates."
    )
    yield Rendered("time_breakdown", fig, "Critical-path time breakdown", caption, uses, notes)


# ---------------------------------------------------------------------------
# 5. token_composition
# ---------------------------------------------------------------------------

KINDS = ["uncached input", "cached input", "reasoning output", "visible output", "output (reasoning not reported)"]


def _role_kinds(eps: pd.DataFrame, r: str) -> pd.DataFrame | None:
    p, c = f"{r}_prompt_tokens", f"{r}_completion_tokens"
    if p not in eps.columns or c not in eps.columns:
        return None
    prompt = num(eps[p])
    comp = num(eps[c])
    cached = num(eps[f"{r}_cached_tokens"]) if f"{r}_cached_tokens" in eps.columns else pd.Series(np.nan, eps.index)
    reas = num(eps[f"{r}_reasoning_tokens"]) if f"{r}_reasoning_tokens" in eps.columns else pd.Series(np.nan, eps.index)
    known_c = cached.notna()
    known_r = reas.notna()
    out = pd.DataFrame({
        "condition": eps["condition"],
        # Unknown cache split: all input counted as uncached, flagged via a note by the caller.
        "uncached input": np.where(known_c, prompt - cached, prompt),
        "cached input": np.where(known_c, cached, 0.0),
        "reasoning output": np.where(known_r, reas, 0.0),
        "visible output": np.where(known_r, comp - reas, 0.0),
        "output (reasoning not reported)": np.where(known_r, 0.0, comp),
    }, index=eps.index)
    out.attrs["cache_unknown"] = bool((~known_c & (prompt > 0)).any())
    return out


def fig_token_composition(data: AnalysisData) -> Iterator[Rendered]:
    eps = data.episodes()
    names_all = data.order(eps["condition"].unique())
    role_means: dict[str, pd.DataFrame] = {}
    notes: list[str] = []
    for r in ROLES:
        k = _role_kinds(eps, r)
        if k is None:
            notes.append(f"{ROLE_NAMES[r]} tokens not available")
            continue
        if k.attrs.get("cache_unknown"):
            notes.append(f"{ROLE_NAMES[r]}: cached-token split unknown for some episodes (counted as uncached)")
        m = k.groupby("condition").mean().reindex(names_all)
        if np.nansum(m.to_numpy(float)) > 0:
            role_means[r] = m
    if not role_means:
        raise Skip("episodes.csv has no {role}_prompt_tokens / {role}_completion_tokens columns with data")
    # Cost share by role: the reference scheme unless another scheme prices more conditions.
    coverage: dict[str, int] = {}
    for s in data.schemes():
        if f"cost_{s}_total" in eps.columns:
            tot = num(eps[f"cost_{s}_total"])
            coverage[s] = int(eps.loc[np.isfinite(tot), "condition"].nunique())
    coverage = {s: c for s, c in coverage.items() if c > 0}
    share_scheme = None
    if coverage:
        best = max(coverage.values())
        share_scheme = "reference" if coverage.get("reference") == best else \
            next(s for s in data.schemes() if coverage.get(s) == best)
    if share_scheme is None:
        notes.append("cost-share panel not drawn: no pricing scheme with finite cost_{S}_total")
    elif share_scheme != "reference":
        notes.append(f"cost shares use the {share_scheme} scheme, which prices {coverage[share_scheme]} conditions "
                     f"(reference prices {coverage.get('reference', 0)})")
    roles = [r for r in ROLES if r in role_means]
    n_panels = len(roles) + (1 if share_scheme else 0)
    fig = new_figure(1.9 + 2.55 * n_panels, 1.9 + 0.32 * len(names_all))
    axes = fig.subplots(1, n_panels, sharey=True, squeeze=False)[0]
    y = np.arange(len(names_all))
    kinds_used: list[str] = []
    for ax, r in zip(axes, roles):
        m = role_means[r]
        left = np.zeros(len(names_all))
        for kind in KINDS:
            v = np.nan_to_num(m[kind].to_numpy(float))
            if not (v > 0).any():
                continue
            ax.barh(y, v, left=left, height=0.62, color=KIND_COLORS[kind], edgecolor=SURFACE, linewidth=0.8, zorder=2)
            left += v
            if kind not in kinds_used:
                kinds_used.append(kind)
        ax.set_title(f"{ROLE_NAMES[r]} tokens")
        ax.xaxis.set_major_formatter(HUMAN)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel("mean tokens per episode")
        hgrid(ax)
    uses = [_cols("episodes", ["{role}_uncached_prompt_tokens", "{role}_cached_tokens", "{role}_prompt_tokens",
                               "{role}_completion_tokens", "{role}_reasoning_tokens"])]
    handles = [Patch(facecolor=KIND_COLORS[k], edgecolor=KIND_COLORS[k], label=k) for k in KINDS if k in kinds_used]
    if share_scheme:
        ax = axes[-1]
        rows = []
        for n in names_all:
            g = eps[eps["condition"] == n]
            tot = num(g[f"cost_{share_scheme}_total"])
            ok = tot.notna()
            denom = tot[ok].sum()
            shares = {}
            for r in ROLE_STACK:
                col = f"cost_{share_scheme}_{r}"
                shares[r] = (num(g.loc[ok, col]).sum() / denom) if (col in g.columns and ok.any() and denom > 0) else np.nan
            rows.append(shares)
        sh = pd.DataFrame(rows, index=names_all)
        left = np.zeros(len(names_all))
        for r in ROLE_STACK:
            v = sh[r].to_numpy(float)
            if not np.isfinite(v).any() or not (np.nan_to_num(v) > 0).any():
                continue
            ax.barh(y, np.nan_to_num(v), left=left, height=0.62, color=ROLE_COLORS[r], edgecolor=SURFACE,
                    linewidth=0.8, zorder=2)
            left += np.nan_to_num(v)
            handles.append(Patch(facecolor=ROLE_COLORS[r], edgecolor=ROLE_COLORS[r], label=f"{ROLE_NAMES[r]} cost"))
        for k, n in enumerate(names_all):
            if not np.isfinite(sh.loc[n].to_numpy(float)).any():
                ax.text(0.02, k, "unpriced (NaN)", va="center", fontsize=7, color=MUTED)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
        ax.set_title(f"cost share by role ({share_scheme})")
        ax.set_xlabel("share of episode cost")
        hgrid(ax)
        uses.append(_cols("episodes", [f"cost_{share_scheme}_{{role}}", f"cost_{share_scheme}_total"]))
    axes[0].set_yticks(y, _labels(data, names_all))
    axes[0].set_ylim(len(names_all) - 0.5, -0.5)
    for ax in axes:
        _block_separators(ax, data, names_all)
    outside_legend(fig, handles, ncols=min(len(handles), 5))
    fig.suptitle("Token composition by role and kind")
    caption = (
        "Mean tokens per episode, one panel per role (orchestrator, subagent, arbiter judge, debate) and one bar per "
        "condition, split into uncached input, cached input (provider prefix cache), reasoning output and visible "
        "output (prompt tokens include cached; completion tokens include reasoning). When a provider does not break "
        "out reasoning tokens, that output is shown as 'output (reasoning not reported)' rather than guessed. Each "
        "token panel has its own x scale (read the axis before comparing bar lengths across roles). The last panel "
        "gives each role's share of the episode cost under the reference pricing scheme (or, if another scheme "
        "prices more conditions, that one; the notes say which); conditions with an unpriced model are marked "
        "unpriced rather than drawn."
    )
    yield Rendered("token_composition", fig, "Token composition by role and kind", caption, uses, notes)


# ---------------------------------------------------------------------------
# 6. orchestrator_tokens
# ---------------------------------------------------------------------------


def _box_strip(ax, data: AnalysisData, names: list[str], values: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    y = np.arange(len(names))
    for k, n in enumerate(names):
        v = values[n][np.isfinite(values[n])]
        if len(v) == 0:
            ax.text(0.01, k, "no data", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=MUTED)
            continue
        info = data.meta[n]
        c = base_color(info)
        if len(v) >= 2:
            ax.boxplot([v], positions=[k], orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
                       boxprops={"facecolor": tint(c, 0.75), "edgecolor": c, "linewidth": 0.9},
                       medianprops={"color": c, "linewidth": 1.8}, whiskerprops={"color": c, "linewidth": 0.9},
                       capprops={"color": c, "linewidth": 0.9}, manage_ticks=False)
        ax.plot(v, k + rng.uniform(-0.17, 0.17, len(v)), alpha=0.9, zorder=3,
                **{**point_style(info, 3.6 / (1.5 if info.baseline else 1.0)), "markeredgewidth": 0.6})
    ax.set_yticks(y, _labels(data, names))
    ax.set_ylim(len(names) - 0.5, -0.5)
    hgrid(ax)
    _block_separators(ax, data, names)


def fig_orchestrator_tokens(data: AnalysisData) -> Iterator[Rendered]:
    eps = data.episodes(["orch_prompt_tokens", "orch_completion_tokens"])
    eps = eps.assign(_orch=num(eps["orch_prompt_tokens"]) + num(eps["orch_completion_tokens"]))
    names = data.order(eps["condition"].unique())
    panels = [("orchestrator-role tokens per episode", "_orch")]
    notes: list[str] = []
    uses = [_cols("episodes", ["orch_prompt_tokens", "orch_completion_tokens"])]
    if "orchestrator_model" in eps.columns:
        big = eps["_orch"].copy()
        om = eps["orchestrator_model"].astype(str)
        for r, mcol in (("arb", "arbiter_model"), ("sub", "subagent_model"), ("deb", "subagent_model")):
            p, c = f"{r}_prompt_tokens", f"{r}_completion_tokens"
            if p not in eps.columns or c not in eps.columns:
                continue
            t = (num(eps[p]) + num(eps[c])).fillna(0.0)
            if mcol in eps.columns:
                same = eps[mcol].astype(str) == om
                big = big + np.where(same, t, 0.0)
            else:
                # Model of this role unknown: big-model tokens unknown wherever the role used tokens.
                big = big.where(t == 0, np.nan)
                notes.append(f"{mcol} column missing: big-model tokens unknown where {r} tokens > 0")
        eps = eps.assign(_big=big)
        if not np.allclose(np.nan_to_num(eps["_big"].to_numpy(float)), np.nan_to_num(eps["_orch"].to_numpy(float))):
            panels.append(("all tokens processed by the orchestrator's model", "_big"))
            uses.append(_cols("episodes", ["orchestrator_model", "subagent_model", "arbiter_model",
                                           "{role}_prompt_tokens", "{role}_completion_tokens"]))
    fig = new_figure(1.9 + 3.6 * len(panels), 1.8 + 0.34 * len(names))
    axes = fig.subplots(1, len(panels), sharey=True, sharex=True, squeeze=False)[0]
    rng = np.random.default_rng(0)
    ref = "single_5step" if "single_5step" in names else next((n for n in names if data.meta[n].baseline), None)
    handles = []
    for ax, (title, col) in zip(axes, panels):
        vals = {n: eps.loc[eps["condition"] == n, col].to_numpy(float) for n in names}
        _box_strip(ax, data, names, vals, rng)
        if ref is not None and np.isfinite(vals[ref]).any():
            med = float(np.nanmedian(vals[ref]))
            ax.axvline(med, color=INK, linestyle=":", linewidth=1.2, zorder=1)
            if not handles:
                handles.append(Line2D([], [], color=INK, linestyle=":", linewidth=1.2,
                                      label=f"{ref} median (orchestrator-role panel: {human(med)})"))
        ax.set_title(title)
        ax.set_xlabel("tokens per episode (prompt + completion)")
        ax.xaxis.set_major_formatter(HUMAN)
    all_v = np.concatenate([eps[c].to_numpy(float) for _, c in panels])
    if log_scale_if_wide(axes[0], all_v, "x", 20):
        axes[0].xaxis.set_major_formatter(HUMAN)
    if ref is None:
        notes.append("no baseline condition selected: no reference line")
    handles = condition_legend(_infos(data, names), kind="marker") + handles
    outside_legend(fig, handles, ncols=min(len(handles), 5))
    fig.suptitle("Does delegation offload the big model?")
    caption = (
        "Distribution over episodes of the tokens (prompt + completion) handled by the orchestrator, one row per "
        "condition: the box spans the inter-quartile range with the median as a thick line, whiskers reach 1.5 IQR, "
        "and every episode is a dot. The dotted vertical line is the median of the single_5step baseline (orchestrator "
        "role). Rows left of that line mean delegation reduced the big model's load. The second panel, when present, "
        "counts every token processed by the orchestrator's model in any role (the arbiter judge, and subagents in the "
        "same-model control), which is what the big model is actually billed for. x is log-scaled when values span "
        "more than 20x."
    )
    yield Rendered("orchestrator_tokens", fig, "Orchestrator (big-model) tokens per episode", caption, uses, notes)


# ---------------------------------------------------------------------------
# 7. per_subagent
# ---------------------------------------------------------------------------


def _violins(ax, groups: list[tuple[float, str, str, np.ndarray]], logy: bool, rng: np.random.Generator) -> None:
    for x, strategy, scope, v in groups:
        v = v[np.isfinite(v)]
        if logy:
            v = np.log10(v[v > 0])
        if len(v) == 0:
            continue
        c = STRATEGY_COLORS.get(strategy, INK2)
        if len(v) >= 3 and np.ptp(v) > 0:
            parts = ax.violinplot([v], positions=[x], widths=0.24, showextrema=False, showmedians=False)
            for b in parts["bodies"]:
                b.set_facecolor(tint(c, SCOPE_TINT.get(scope, 0.0) * 0.6 + 0.3))
                b.set_edgecolor(c)
                b.set_linewidth(0.8)
                b.set_alpha(1.0)
        ax.plot(x + rng.uniform(-0.05, 0.05, len(v)), v, linestyle="none", marker=".", markersize=2.2, color=c,
                alpha=0.55, zorder=3)
        ax.plot([x - 0.07, x + 0.07], [np.median(v)] * 2, color=INK, linewidth=1.6, zorder=4, solid_capstyle="butt")
    if logy:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _p: human(10 ** t)))


def _dropped_note(df: pd.DataFrame, what: str) -> list[str]:
    n = df.attrs.get("n_dropped", 0)
    return [f"{n} {what} of invalid or duplicate trials left out"] if n else []


def fig_per_subagent(data: AnalysisData) -> Iterator[Rendered]:
    d = data.event_rows("dispatches", ["strategy", "scope"])
    notes: list[str] = _dropped_note(d, "dispatch(es)")
    primary = data.primary_subagent_model
    if primary and "model" in d.columns:
        other = sorted(set(d.loc[d["model"].astype(str) != primary, "condition"].astype(str))) \
            if "condition" in d.columns else []
        d = d[d["model"].astype(str) == primary]
        if other:
            notes.append(f"dispatches of other subagent models excluded ({', '.join(other)})")
    d = d[d["strategy"].astype(str).isin(STRATEGIES) & d["scope"].astype(str).isin(SCOPES)]
    if d.empty:
        raise Skip("dispatches.csv has no rows with a known strategy and scope")
    strategies = [s for s in STRATEGIES if s in set(d["strategy"])]
    scopes = [s for s in SCOPES if s in set(d["scope"])]
    metrics: list[tuple[str, list[str], Callable[[pd.DataFrame], pd.Series], bool]] = [
        ("tokens per dispatch (prompt + completion)", ["prompt_tokens", "completion_tokens"],
         lambda g: num(g["prompt_tokens"]) + num(g["completion_tokens"]), True),
        ("tool calls per dispatch", ["n_tool_calls"], lambda g: num(g["n_tool_calls"]), False),
        ("LLM steps used", ["steps_used"], lambda g: num(g["steps_used"]), False),
        ("LLM seconds per dispatch", ["llm_sec"], lambda g: num(g["llm_sec"]), True),
        ("dispatch wall-clock (s)", ["duration_sec"], lambda g: num(g["duration_sec"]), True),
    ]
    avail = [m for m in metrics if all(c in d.columns for c in m[1])]
    for m in metrics:
        if m not in avail:
            notes.append(f"panel '{m[0]}' not drawn: dispatches.csv lacks {', '.join(c for c in m[1] if c not in d.columns)}")
    has_status = "status" in d.columns
    if not has_status:
        notes.append("submitted-fraction panel not drawn: dispatches.csv lacks status")
    n_panels = len(avail) + int(has_status)
    if n_panels == 0:
        raise Skip("dispatches.csv has none of the per-dispatch metric columns")
    fig = new_figure(3.6 * min(n_panels, 3) + 0.3, 3.0 * math.ceil(n_panels / 3) + 0.9)
    axes = panel_axes(fig, n_panels, max_cols=3)
    rng = np.random.default_rng(0)
    off = {s: (j - (len(strategies) - 1) / 2) * 0.27 for j, s in enumerate(strategies)}
    for ax, (title, cols, fn, logy) in zip(axes, avail):
        vals = fn(d)
        lg = logy and (vals > 0).sum() > 1 and vals[vals > 0].max() / vals[vals > 0].min() >= 30
        groups = []
        for i, sc in enumerate(scopes):
            for s in strategies:
                m = (d["scope"] == sc) & (d["strategy"] == s)
                if m.any():
                    groups.append((i + off[s], s, sc, vals[m].to_numpy(float)))
        _violins(ax, groups, lg, rng)
        ax.set_title(title + (" (log)" if lg else ""))
        ax.set_xticks(range(len(scopes)), [f"{s} scope" for s in scopes])
        ax.tick_params(axis="x", length=0)
        vgrid(ax)
    if has_status:
        ax = axes[-1]
        sub = as_bool(d["status"].astype(str).eq("submitted"))
        for i, sc in enumerate(scopes):
            for s in strategies:
                m = (d["scope"] == sc) & (d["strategy"] == s)
                if not m.any():
                    continue
                p, lo, hi = wilson(float(sub[m].sum()), float(m.sum()))
                c = STRATEGY_COLORS[s]
                ax.bar(i + off[s], p, width=0.24, color=tint(c, SCOPE_TINT[sc]), edgecolor=c, linewidth=0.8, zorder=2)
                ax.errorbar(i + off[s], p, yerr=errs([p], [lo], [hi]), fmt="none", ecolor=INK, elinewidth=0.9,
                            capsize=2, zorder=3)
        ax.set_ylim(0, 1.05)
        ax.set_title("share of dispatches that submitted a report")
        ax.set_xticks(range(len(scopes)), [f"{s} scope" for s in scopes])
        ax.tick_params(axis="x", length=0)
        vgrid(ax)
    handles = [Patch(facecolor=tint(STRATEGY_COLORS[s], 0.45), edgecolor=STRATEGY_COLORS[s], label=s) for s in strategies]
    handles.append(Line2D([], [], color=INK, linewidth=1.6, label="median"))
    outside_legend(fig, handles, ncols=len(handles))
    fig.suptitle(f"Per-subagent behaviour by scope and strategy{f' ({primary})' if primary else ''}")
    caption = (
        "One observation per subagent dispatch of a valid trial (arbiter candidates count individually). Panels "
        "show, by delegation scope (x groups) and strategy (colour), the distribution of tokens, tool calls "
        "(excluding submit), LLM steps, "
        "LLM seconds and wall-clock per dispatch as violins with every dispatch as a dot and the median as a black "
        "tick (fewer than 3 distinct values: dots only). Metrics spanning more than 30x use a log axis. The last "
        "panel is the share of dispatches that ended by calling submit (status 'submitted') with a Wilson 95% "
        "interval; the rest hit max_steps, answered without a tool call, overflowed the context or errored."
    )
    uses = [_cols("dispatches", ["strategy", "scope", "model", "prompt_tokens", "completion_tokens", "n_tool_calls",
                                 "steps_used", "llm_sec", "duration_sec", "status"])]
    yield Rendered("per_subagent", fig, "Per-subagent work by scope and strategy", caption, uses, notes)


# ---------------------------------------------------------------------------
# 8. spawn_counts
# ---------------------------------------------------------------------------

CAP_COLORS = {"not capped": "#e1e0d9", "max_concurrent": "#898781", "max_dispatches": "#3d3c3a", "other": "#b5b3ac"}


def _uptake(data: AnalysisData, eps: pd.DataFrame, names: list[str]) -> tuple[dict, dict, str] | None:
    """Per condition: (valid episodes that delegated, valid episodes, source).

    From ``conditions.csv`` (n_delegated, n_valid; delegated_frac is their
    ratio), else from the valid ``episodes.csv`` rows (delegated, or
    spawn_calls > 0). None when neither table has it.
    """
    conds = data.table("conditions")
    if conds is not None and {"n_delegated", "n_valid"} <= set(conds.columns):
        c = conds.drop_duplicates("condition")
        c = c.set_index(c["condition"].astype(str))
        k, n = num(c["n_delegated"]), num(c["n_valid"])
        if k.reindex(names).notna().any():
            return k.to_dict(), n.to_dict(), _cols("conditions", ["n_delegated", "n_valid", "delegated_frac"])
    col = next((c for c in ("delegated", "spawn_calls") if c in eps.columns), None)
    if col is None:
        return None
    d = num(eps[col])
    g = (d > 0).astype(float).where(d.notna()).groupby(eps["condition"].astype(str))
    return g.sum().to_dict(), g.count().to_dict(), _cols("episodes", [col, "valid"])


def fig_spawn_counts(data: AnalysisData) -> Iterator[Rendered]:
    eps = data.episodes()
    notes: list[str] = []
    uses: list[str] = []
    panels: list[str] = []
    subs = [n for n in data.order(eps["condition"].unique()) if not data.meta[n].baseline]
    if not subs:
        raise Skip("no subagent condition selected (baselines spawn nothing)")
    uptake = _uptake(data, eps, subs)
    if uptake is not None:
        panels.append("deleg")
    else:
        notes.append("delegation panel not drawn: neither conditions.csv delegated_frac nor episodes.csv "
                     "delegated/spawn_calls")
    if "dispatches" in eps.columns:
        panels.append("hist")
    else:
        notes.append("dispatch-count panel not drawn: episodes.csv lacks dispatches")
    if "instruction_following" in eps.columns:
        panels.append("follow")
    else:
        notes.append("instruction-following panel not drawn: episodes.csv lacks instruction_following")
    try:
        sp = data.event_rows("spawns", ["cap_reason", "condition"])
        panels.append("caps")
        notes += _dropped_note(sp, "spawn call(s)")
    except Skip as e:
        sp = None
        notes.append(f"cap-reason panel not drawn: {e}")
    if not panels:
        raise Skip("; ".join(notes))
    widths = {"deleg": 0.75, "hist": 1.3}
    fig = new_figure(2.0 + 3.2 * sum(widths.get(p, 1.0) for p in panels), max(3.4, 1.7 + 0.34 * len(subs)))
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False,
                        width_ratios=[widths.get(p, 1.0) for p in panels])[0]
    ax_by = dict(zip(panels, axes))
    y = np.arange(len(subs))
    handles: list = []
    if "deleg" in ax_by:
        ax = ax_by["deleg"]
        k_del, n_val, src = uptake
        never: list[str] = []
        for k, n in enumerate(subs):
            kd, nv = k_del.get(n, np.nan), n_val.get(n, np.nan)
            if not (np.isfinite(kd) and np.isfinite(nv) and nv > 0):
                ax.text(0.02, k, "unknown", va="center", fontsize=7, color=MUTED)
                continue
            p, lo, hi = wilson(kd, nv)
            ax.barh(k, p, height=0.62, **bar_style(data.meta[n]), zorder=2)
            ax.errorbar(p, k, xerr=errs([p], [lo], [hi]), fmt="none", ecolor=INK, elinewidth=0.9, capsize=2, zorder=3)
            ax.text(1.03, k, f"{int(kd)}/{int(nv)}", va="center", fontsize=6.5,
                    color=ROLE_COLORS["arb"] if kd == 0 else INK2, transform=ax.get_yaxis_transform())
            never += [n] if kd == 0 else []
        if never:
            notes.append("never delegated (no valid episode spawned subagents): " + ", ".join(never))
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
        ax.set_xlabel("share of valid episodes with a spawn call\n(Wilson 95% CI; k/n at right)")
        ax.set_title("episodes that delegated")
        hgrid(ax)
        uses.append(src)
    if "hist" in ax_by:
        ax = ax_by["hist"]
        for k, n in enumerate(subs):
            v = num(eps.loc[eps["condition"] == n, "dispatches"]).dropna()
            if v.empty:
                ax.text(0.01, k, "no data", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=MUTED)
                continue
            info = data.meta[n]
            frac = v.value_counts(normalize=True).sort_index()
            ax.scatter(frac.index, [k] * len(frac), s=260 * frac.to_numpy(), color=fill_color(info),
                       edgecolor=base_color(info), linewidth=0.8, hatch="////" if info.control else None, zorder=3)
            ax.plot([v.mean()] * 2, [k - 0.36, k + 0.36], color=INK, linewidth=1.6, zorder=4)
        ax.set_xlabel("subagent dispatches per episode (arbiter: each candidate)")
        ax.set_title("dispatches per episode")
        vmax = float(np.nanmax(num(eps.loc[eps["condition"].isin(subs), "dispatches"]).to_numpy(float), initial=0.0))
        ax.set_xlim(-0.6, max(vmax, 1.0) + 0.6)  # never a +-0.05 axis when nothing was dispatched
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        hgrid(ax)
        handles += [Line2D([], [], linestyle="none", marker="o", markersize=math.sqrt(260 * f), color=MUTED,
                           label=f"{f:.0%} of episodes") for f in (0.25, 1.0)]
        handles.append(Line2D([], [], color=INK, linewidth=1.6, label="mean"))
        uses.append(_cols("episodes", ["dispatches"]))
    if "follow" in ax_by:
        ax = ax_by["follow"]
        try:
            m = data.cond_metric("instruction_following")
            uses.append(_src(m, "instruction_following"))
            m = m.reindex(subs)
            no_ci, _ = _hbars(ax, data, subs, m["mean"], m["lo"], m["hi"], missing={n: "no spawns" for n in subs})
            notes += _ci_note(no_ci, "for instruction following")
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("granted / requested instructions (mean, 95% CI)")
            ax.set_title("instruction following")
        except Skip as e:
            ax.set_visible(False)
            notes.append(f"instruction-following panel not drawn: {e}")
    if "caps" in ax_by:
        ax = ax_by["caps"]
        sp = sp[sp["condition"].astype(str).isin(subs)]
        reason = sp["cap_reason"].astype(object).where(sp["cap_reason"].notna(), "not capped").astype(str)
        reason = reason.where(reason.isin(list(CAP_COLORS)), "other")
        frac = pd.crosstab(sp["condition"].astype(str), reason, normalize="index").reindex(subs)
        cnt = sp.groupby(sp["condition"].astype(str)).size().reindex(subs)
        left = np.zeros(len(subs))
        for r, c in CAP_COLORS.items():
            if r in frac.columns:
                v = np.nan_to_num(frac[r].to_numpy(float))
                ax.barh(y, v, left=left, height=0.62, color=c, edgecolor=SURFACE, linewidth=0.8, zorder=2)
                left += v
                handles.append(Patch(facecolor=c, edgecolor=AXIS,
                                     label="spawn call not capped" if r == "not capped" else f"capped by {r}"))
        for k, n in enumerate(subs):
            if np.isfinite(cnt.get(n, np.nan)):
                ax.text(1.02, k, f"n={int(cnt[n])}", va="center", fontsize=6.5, color=INK2,
                        transform=ax.get_yaxis_transform())
            else:
                ax.text(0.02, k, "no spawn calls", va="center", fontsize=7, color=MUTED)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
        ax.set_xlabel("share of spawn_subagents calls")
        ax.set_title("why requests were cut")
        hgrid(ax)
        uses.append(_cols("spawns", ["cap_reason", "condition"]))
    axes[0].set_yticks(y, _labels(data, subs))
    axes[0].set_ylim(len(subs) - 0.5, -0.5)
    for ax in axes:
        _block_separators(ax, data, subs)
    controls = [i for i in _infos(data, subs) if i.control]
    if controls:
        handles += condition_legend(controls, kind="patch", scope_legend=False)[-len(controls):]
    outside_legend(fig, handles, ncols=min(len(handles), 6))
    fig.suptitle("How many subagents the orchestrator spawned, and how much of each request it got")
    caption = (
        "One row per subagent condition (baselines spawn nothing and are omitted; controls are hatched, a manager "
        "control whose orchestrator lacks the editing tools is tagged [manager]). First panel: the share of valid "
        "episodes whose orchestrator spawned subagents at least once, with a Wilson 95% interval and the k/n count "
        "(red when no episode delegated: the condition never received its treatment, and its resolve rate "
        "measures the orchestrator alone). Next: the distribution over episodes of the number of subagent "
        "dispatches (arbiter candidates count individually), as a histogram drawn with dots: dot area is the "
        "share of episodes with that count, the black tick is the mean; dots further right mean more subagents. "
        "Then instruction following: the mean over episodes of each episode's mean granted / requested "
        "instructions per spawn call (1 = every request fully granted; an episode with many spawn calls counts "
        "once, like one with a single call), with a bootstrap 95% interval over episodes; episodes without a "
        "spawn call have no value. Last: the share of the valid trials' spawn_subagents calls that were cut by "
        "the concurrency cap or by the dispatch budget, with the number of calls."
    )
    yield Rendered("spawn_counts", fig, "Subagents spawned and instruction following", caption, uses, notes)


# ---------------------------------------------------------------------------
# 9. parallelism
# ---------------------------------------------------------------------------


def fig_parallelism(data: AnalysisData) -> Iterator[Rendered]:
    eps = data.episodes(["parallel_speedup"])
    eps = eps.assign(_s=num(eps["parallel_speedup"]))
    eps = eps[np.isfinite(eps["_s"]) & ~eps["condition"].map(lambda n: data.meta[n].baseline)]
    if eps.empty:
        raise Skip("no finite parallel_speedup values (no episode spawned subagents)")
    eps = eps.assign(_strategy=eps["condition"].map(lambda n: data.meta[n].strategy or "unknown"),
                     _scope=eps["condition"].map(lambda n: data.meta[n].scope or ""))
    strategies = [s for s in STRATEGIES if s in set(eps["_strategy"])] + \
        sorted(set(eps["_strategy"]) - set(STRATEGIES))
    notes: list[str] = []
    has_b = "sub_busy_sec" in eps.columns
    if not has_b:
        notes.append("scatter panel not drawn: episodes.csv lacks sub_busy_sec")
    conc = None
    if "max_concurrent" in eps.columns and num(eps["max_concurrent"]).notna().any():
        conc = float(num(eps["max_concurrent"]).mode().iloc[0])
    fig = new_figure(5.0 + (5.2 if has_b else 0), 4.4)
    axes = fig.subplots(1, 2 if has_b else 1, squeeze=False)[0]
    ax = axes[0]
    rng = np.random.default_rng(0)
    for k, s in enumerate(strategies):
        g = eps[eps["_strategy"] == s]
        v = g["_s"].to_numpy(float)
        c = STRATEGY_COLORS.get(s, INK2)
        if len(v) >= 2:
            ax.boxplot([v], positions=[k], widths=0.5, patch_artist=True, showfliers=False, manage_ticks=False,
                       boxprops={"facecolor": tint(c, 0.8), "edgecolor": c}, medianprops={"color": c, "linewidth": 1.8},
                       whiskerprops={"color": c}, capprops={"color": c})
        for n, gg in g.groupby("condition"):
            info = data.meta[n]
            st = point_style(info, 4.2)
            ax.plot(k + rng.uniform(-0.18, 0.18, len(gg)), gg["_s"], alpha=0.85, zorder=3,
                    **{**st, "markeredgewidth": 0.7})
    ax.axhline(1.0, color=INK2, linestyle=":", linewidth=1.0)
    ref_handles = [Line2D([], [], color=INK2, linestyle=":", linewidth=1.0, label="no overlap (speedup 1, y = x)")]
    if conc:
        ax.axhline(conc, color=INK2, linestyle="--", linewidth=0.9)
        ref_handles.append(Line2D([], [], color=INK2, linestyle="--", linewidth=0.9,
                                  label=f"perfect {conc:g}-way overlap (max_concurrent, y = x/{conc:g})"))
    ax.set_xticks(range(len(strategies)), strategies)
    ax.set_xlim(-0.6, len(strategies) - 0.4)
    ax.set_ylabel("parallel speedup = total subagent work / group critical path")
    ax.set_title("parallel speedup per episode")
    vgrid(ax)
    uses = [_cols("episodes", ["parallel_speedup", "max_concurrent"])]
    if has_b:
        ax = axes[1]
        busy = num(eps["sub_busy_sec"])
        cp = busy / eps["_s"]
        for n, idx in eps.groupby("condition").groups.items():
            info = data.meta[n]
            ax.plot(busy[idx], cp[idx], alpha=0.85, zorder=3, **{**point_style(info, 5.0), "markeredgewidth": 0.8})
        hi = float(np.nanmax(busy)) * 1.05 if np.isfinite(busy).any() else 1.0
        xs = np.array([0, hi])
        ax.plot(xs, xs, color=INK2, linestyle=":", linewidth=1.0)
        if conc:
            ax.plot(xs, xs / conc, color=INK2, linestyle="--", linewidth=0.9)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, None)
        ax.set_xlabel("total subagent work per episode, sub_busy_sec (s)")
        ax.set_ylabel("critical-path time inside spawn groups (s)")
        ax.set_title("subagent work vs time it cost on the critical path")
        uses.append(_cols("episodes", ["sub_busy_sec", "parallel_speedup"]))
    handles = condition_legend(_infos(data, data.order(eps["condition"].unique())), kind="marker") + ref_handles
    outside_legend(fig, handles, ncols=min(len(handles), 5))
    fig.suptitle("How much subagent work ran in parallel")
    caption = (
        "Left: per-episode parallel speedup, the total duration of all subagent dispatches (sub_busy_sec) divided by "
        "the critical-path time spent inside spawn groups, pooled by strategy (box = inter-quartile range, thick "
        "line = median, dots = episodes with marker shape by scope). Sequential groups sit at 1 by construction; "
        "parallel and arbiter groups can reach max_concurrent (dashed). Right: the same episodes as a scatter of "
        "total subagent work (x) against the critical-path time it cost (y = x / speedup); points on the dotted "
        "diagonal gained nothing from concurrency, points on the dashed line overlapped perfectly. Episodes without "
        "any spawn group are omitted."
    )
    yield Rendered("parallelism", fig, "Parallelism achieved by spawn groups", caption, uses, notes)


# ---------------------------------------------------------------------------
# 10. task_heatmap
# ---------------------------------------------------------------------------


def fig_task_heatmap(data: AnalysisData) -> Iterator[Rendered]:
    eps = data.episodes(["task_name"])
    eps = eps[eps["resolved"].notna()]
    if eps.empty:
        raise Skip("no valid episodes with a known outcome")
    g = eps.groupby(["task_name", "condition"])["resolved"].agg(["sum", "count"])
    tasks = list(dict.fromkeys(eps["task_name"].astype(str)))
    info = {}
    for t in tasks:
        row = eps[eps["task_name"] == t].iloc[0]
        diff = str(row["difficulty"]) if "difficulty" in eps.columns and pd.notna(row.get("difficulty")) else None
        exp = float(row["expert_time_min"]) if "expert_time_min" in eps.columns and pd.notna(row.get("expert_time_min")) else math.nan
        info[t] = (diff, exp)
    rank = {d: i for i, d in enumerate(DIFFICULTIES)}
    tasks.sort(key=lambda t: (rank.get(info[t][0], len(rank)), info[t][1] if np.isfinite(info[t][1]) else math.inf, t))
    conds = data.order(eps["condition"].unique())
    frac = np.full((len(tasks), len(conds)), np.nan)
    lab = [["" for _ in conds] for _ in tasks]
    for i, t in enumerate(tasks):
        for j, c in enumerate(conds):
            if (t, c) in g.index:
                k, n = g.loc[(t, c)]
                frac[i, j] = k / n
                lab[i][j] = f"{int(k)}/{int(n)}"
    fig = new_figure(2.6 + 0.62 * len(conds), 1.9 + 0.42 * len(tasks) + 0.09 * max(len(c) for c in conds))
    ax = fig.subplots()
    cmap = SEQ_CMAP.with_extremes(bad="#f4f3f0")
    im = ax.imshow(np.ma.masked_invalid(frac), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(tasks)):
        for j in range(len(conds)):
            v = frac[i, j]
            ax.text(j, i, lab[i][j] if np.isfinite(v) else "-", ha="center", va="center", fontsize=6.8,
                    color=SURFACE if (np.isfinite(v) and v > 0.55) else INK)
    ax.set_xticks(range(len(conds)), _labels(data, conds), rotation=40, ha="right", rotation_mode="anchor")
    ylabels = []
    for t in tasks:
        d, e = info[t]
        extra = ", ".join(x for x in (d, f"{e:g} min" if np.isfinite(e) else None) if x)
        ylabels.append(f"{t} ({extra})" if extra else t)
    ax.set_yticks(range(len(tasks)), ylabels)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for j in range(len(conds) + 1):
        ax.axvline(j - 0.5, color=SURFACE, linewidth=1.5)
    for i in range(len(tasks) + 1):
        ax.axhline(i - 0.5, color=SURFACE, linewidth=1.5)
    # Block separators (baselines | grid | controls; difficulty groups) on top of the gutters.
    for j in range(1, len(conds)):
        a, b = data.meta[conds[j - 1]], data.meta[conds[j]]
        if (a.baseline, a.control) != (b.baseline, b.control):
            ax.axvline(j - 0.5, color=INK2, linewidth=1.6)
    for i in range(1, len(tasks)):
        if info[tasks[i]][0] != info[tasks[i - 1]][0]:
            ax.axhline(i - 0.5, color=INK2, linewidth=1.6)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("solve fraction over rounds")
    cb.outline.set_visible(False)
    ax.set_title("Per-task solve fraction (cells: solved/valid trials)")
    caption = (
        "Rows are tasks, sorted by Terminal-Bench difficulty label and then by the expert time estimate (both in the "
        "row label); columns are conditions in the usual order (baselines, strategy x scope grid, controls), "
        "with dark rules between those blocks and between difficulty groups. Each cell is the share of that task's valid rounds the condition solved, darker = "
        "more often, with the solved/valid count printed; '-' means no valid trial. Reading along a row shows which "
        "conditions crack a given task; a column that is dark only on easy rows adds nothing on hard tasks."
    )
    uses = [_cols("episodes", ["task_name", "condition", "resolved", "valid", "difficulty", "expert_time_min"])]
    yield Rendered("task_heatmap", fig, "Per-task solve fraction", caption, uses, [])


# ---------------------------------------------------------------------------
# 12. calibration
# ---------------------------------------------------------------------------


def fig_calibration(data: AnalysisData) -> Iterator[Rendered]:
    calls = data.require("calls", ["model"]).copy()
    notes: list[str] = []
    if "ttft_sec" in calls.columns:
        calls["_ttft"] = num(calls["ttft_sec"])
    elif {"t_first_chunk", "t_request_sent"} <= set(calls.columns):
        calls["_ttft"] = num(calls["t_first_chunk"]) - num(calls["t_request_sent"])
    else:
        calls["_ttft"] = np.nan
    dec = num(calls["decode_sec"]) if "decode_sec" in calls.columns else (
        num(calls["t_last_chunk"]) - num(calls["t_first_chunk"]) if {"t_last_chunk", "t_first_chunk"} <= set(calls.columns)
        else pd.Series(np.nan, calls.index))
    comp = num(calls["completion_tokens"]) if "completion_tokens" in calls.columns else pd.Series(np.nan, calls.index)
    calls["_rate"] = np.where(dec > 0, comp / dec, np.nan)
    calls["_lat"] = num(calls["latency_sec"]) if "latency_sec" in calls.columns else (
        num(calls["t_end"]) - num(calls["t_start"]) if {"t_end", "t_start"} <= set(calls.columns) else np.nan)
    calls["_prompt"] = num(calls["prompt_tokens"]) if "prompt_tokens" in calls.columns else np.nan
    cols = []
    if np.isfinite(calls["_ttft"]).any() and np.isfinite(calls["_prompt"]).any():
        cols.append("ttft")
    else:
        notes.append("TTFT column not drawn: calls.csv lacks ttft_sec/prompt_tokens")
    if np.isfinite(calls["_rate"]).any():
        cols.append("decode")
    else:
        notes.append("decode-rate column not drawn: calls.csv lacks decode_sec/completion_tokens")
    if "t_start" in calls.columns and np.isfinite(calls["_lat"]).any():
        cols.append("drift")
    else:
        notes.append("drift column not drawn: calls.csv lacks t_start/latency_sec")
    if not cols:
        raise Skip("calls.csv has no timing columns")
    keys = [k for k in ("model", "provider", "completion_window") if k in calls.columns]
    groups = sorted(calls.groupby(keys, dropna=False), key=lambda kv: -len(kv[1]))
    if len(groups) > 6:
        notes.append(f"only the 6 most-called of {len(groups)} model/provider/tier groups shown")
        groups = groups[:6]
    cal = data.raw("calibration")
    t0 = float(num(calls["t_start"]).min()) if "t_start" in calls.columns else 0.0
    fig = new_figure(3.6 * len(cols) + 0.4, 2.7 * len(groups) + 1.0)
    axes = fig.subplots(len(groups), len(cols), squeeze=False)
    model_colors = [ROLE_COLORS["orch"], ROLE_COLORS["sub"], ROLE_COLORS["deb"], TOOL_COLOR, ROLE_COLORS["arb"], INK2]
    orch_models = set(data.table("episodes")["orchestrator_model"].dropna().astype(str)) \
        if data.has("episodes", ["orchestrator_model"]) else set()
    rounds = sorted(num(calls["round"]).dropna().unique()) if "round" in calls.columns else []
    uses = [_cols("calls", ["model", "provider", "completion_window", "prompt_tokens", "ttft_sec", "completion_tokens",
                            "decode_sec", "latency_sec", "t_start", "round"])]
    fit_from_csv = False
    k_other = 1
    for row, (key, g) in enumerate(groups):
        key = key if isinstance(key, tuple) else (key,)
        kd = dict(zip(keys, key))
        model = str(kd.get("model"))
        if model in orch_models:
            mc = model_colors[0]
        else:
            mc = model_colors[k_other % len(model_colors)]
            k_other += 1
        label = " / ".join(str(v) for v in key if pd.notna(v))
        for col, ax in zip(cols, axes[row]):
            if col == "ttft":
                ok = np.isfinite(g["_ttft"]) & np.isfinite(g["_prompt"])
                x, yv = g.loc[ok, "_prompt"] / 1000.0, g.loc[ok, "_ttft"]
                if not ok.any():
                    ax.text(0.5, 0.5, "no streamed timings", transform=ax.transAxes, ha="center", color=MUTED)
                    continue
                ax.scatter(x, yv, s=5, color=mc, alpha=0.35, linewidths=0, rasterized=True)
                a = b = None
                if cal is not None and {"ttft_fit_a", "ttft_fit_b_per_1k_prompt_tokens"} <= set(cal.columns):
                    m = pd.Series(True, cal.index)
                    for kk, vv in kd.items():
                        if kk in cal.columns:
                            m &= (cal[kk].astype(str) == str(vv)) if pd.notna(vv) else cal[kk].isna()
                    if m.any():
                        a = float(num(cal.loc[m, "ttft_fit_a"]).iloc[0])
                        b = float(num(cal.loc[m, "ttft_fit_b_per_1k_prompt_tokens"]).iloc[0])
                        fit_from_csv = True
                if (a is None or not np.isfinite(a)) and ok.sum() >= 2:
                    b, a = np.polyfit(x.to_numpy(float), yv.to_numpy(float), 1)
                    notes.append(f"{label}: TTFT fit computed here (not found in calibration.csv)")
                if a is not None and np.isfinite(a):
                    xs = np.linspace(0, float(x.max()), 50)
                    ax.plot(xs, a + b * xs, color=INK, linewidth=1.3)
                    ax.text(0.02, 0.97, f"TTFT = {a:.2f} s + {b:.3f} s per 1k prompt tokens", transform=ax.transAxes,
                            va="top", fontsize=6.5, color=INK)
                ax.set_xlabel("prompt tokens (thousands)")
                ax.set_ylabel("TTFT (s)")
                # Clip only extreme outliers of a large sample, and say so; a small one shows every point.
                top = float(np.nanpercentile(yv, 99.5) if len(yv) >= 200 else np.nanmax(yv)) * 1.15
                ax.set_ylim(0, top)
                if (yv > top).any():
                    notes.append(f"{label}: {int((yv > top).sum())} TTFT point(s) above the y range not shown")
            elif col == "decode":
                r = g["_rate"].to_numpy(float)
                r = r[np.isfinite(r)]
                if len(r) == 0:
                    ax.text(0.5, 0.5, "no decode timings", transform=ax.transAxes, ha="center", color=MUTED)
                    continue
                ax.hist(r, bins=30, color=tint(mc, 0.2), edgecolor=SURFACE, linewidth=0.5)
                med = float(np.median(r))
                ax.axvline(med, color=INK, linewidth=1.2)
                ax.text(med, ax.get_ylim()[1] * 0.97, f" median {med:.0f} tok/s", fontsize=6.5, color=INK, va="top")
                ax.set_xlabel("decode rate (completion tokens / decode s)")
                ax.set_ylabel("calls")
            else:
                t = (num(g["t_start"]) - t0) / 3600.0
                lat = g["_lat"]
                if rounds:
                    for i, rv in enumerate(rounds):
                        m = num(g["round"]) == rv
                        ax.scatter(t[m], lat[m], s=5, color=ROUND_RAMP[i % len(ROUND_RAMP)], alpha=0.5, linewidths=0,
                                   rasterized=True, label=f"round {rv:g}")
                else:
                    ax.scatter(t, lat, s=5, color=mc, alpha=0.5, linewidths=0, rasterized=True)
                log_scale_if_wide(ax, lat, "y", 30)
                ax.set_xlabel("hours since first call of the sweep")
                ax.set_ylabel("call latency (s)")
            vgrid(ax)
        axes[row][0].set_title(label, loc="left", fontsize=8.5)
    handles = [Line2D([], [], color=INK, linewidth=1.3, label="least-squares TTFT fit")]
    if "drift" in cols and rounds:
        handles += [Line2D([], [], color=ROUND_RAMP[i % len(ROUND_RAMP)], marker="o", linestyle="none",
                           label=f"round {rv:g}") for i, rv in enumerate(rounds)]
    outside_legend(fig, handles, ncols=len(handles))
    fig.suptitle("Provider calibration: TTFT, decode speed and drift per model")
    if fit_from_csv:
        uses.append(_cols("calibration", ["model", "provider", "completion_window", "ttft_fit_a",
                                          "ttft_fit_b_per_1k_prompt_tokens"]))
    caption = (
        "One row per (model, provider, tier) group, most-called first. Left: time to first token of every call "
        "against its prompt size, with the least-squares line TTFT = a + b x prompt/1000 (calibration.csv ttft_fit_*, "
        "descriptive; the scatter around it is provider queueing and noise; the 'fitted' serving profile instead fits "
        "whole call time, a + b x prompt/1000 + c x completion tokens, calibration.csv lat_fit_*). The y axis shows "
        "every point unless a group has 200 calls or more, where the top 0.5% may be cut (the notes count them). "
        "Middle: histogram of decode speed, completion "
        "tokens divided by the time between first and last streamed chunk, with the median; providers that stream "
        "few coarse chunks make this a rough estimate. Right: call latency over the sweep's wall-clock time coloured "
        "by round, a drift check: a trend or a shift between rounds means provider speed changed during the sweep, "
        "which the round-interleaved design spreads evenly over conditions."
    )
    yield Rendered("calibration", fig, "Provider calibration and drift", caption, uses, notes)


# ---------------------------------------------------------------------------
# 13. cost_effectiveness
# ---------------------------------------------------------------------------


def fig_cost_effectiveness(data: AnalysisData) -> Iterator[Rendered]:
    conds = data.table("conditions")
    try:
        eps_v = data.episodes()
    except Skip:
        eps_v = None
    stats = data.resolve_stats()
    rs = stats.set_index("condition")
    no_valid = [n for n in stats.attrs.get("no_valid", []) if n in data.meta]
    names = data.order(list(rs.index) + no_valid)
    specs = [(f"cost per solve, {s}", f"cost_per_solve_{s}", f"cost_{s}_total", USD, "USD per solved task")
             for s in data.schemes()]
    notes: list[str] = []
    if no_valid:
        notes.append("no valid trials (rows labelled, nothing drawn): " + ", ".join(no_valid))
    if not any(_has_finite(data, f"cost_{s}_total") for s in data.schemes()) and _has_finite(data, "cost_runtime_total"):
        log.info("cost_effectiveness: %s", RUNTIME_NOTE)
        notes.append(RUNTIME_NOTE)
        specs.append(("cost per solve, run-time prices (logged)", "cost_per_solve_runtime", "cost_runtime_total", USD,
                      "USD per solved task"))
    specs.append(("time per solve", "time_per_solve_sec", "wall_clock_sec", None, "agent wall-clock seconds per solved task"))
    panels = []
    uses: list[str] = []
    for title, col, numer, fmt, xlabel in specs:
        point = lo = hi = None
        src = []
        if conds is not None and col in conds.columns:
            c = conds.drop_duplicates("condition").set_index(conds.drop_duplicates("condition")["condition"].astype(str))
            point = num(c[col]).reindex(names)
            src.append(f"`conditions.csv` ({col})")
            if f"{col}_ci_lo" in c.columns and f"{col}_ci_hi" in c.columns:
                lo, hi = num(c[f"{col}_ci_lo"]).reindex(names), num(c[f"{col}_ci_hi"]).reindex(names)
                src[-1] = f"`conditions.csv` ({col}, {col}_ci_lo, {col}_ci_hi)"
        if (lo is None or point is None) and eps_v is not None and numer in eps_v.columns:
            boot = {}
            for n in names:
                g = eps_v[(eps_v["condition"] == n) & eps_v["resolved"].notna()]
                if numer == "wall_clock_sec":
                    # Same subset as time_per_solve_sec: episodes with a known wall-clock (costs are strict: a NaN
                    # cost makes both the CSV value and the ratio NaN).
                    g = g[num(g[numer]).notna()]
                boot[n] = bootstrap_ratio(num(g[numer]), g["resolved"])
            b = pd.DataFrame.from_dict(boot, orient="index", columns=["p", "lo", "hi"]).reindex(names)
            if point is None:
                point = b["p"]
            lo, hi = b["lo"], b["hi"]
            src.append(f"`episodes.csv` ({numer}, resolved; bootstrap 95% CI of the ratio recomputed here)")
        if point is None:
            notes.append(f"panel '{title}' not drawn: no {col} in conditions.csv and no {numer} in episodes.csv")
            continue
        if not np.isfinite(point.to_numpy(float)).any():
            notes.append(f"panel '{title}' not drawn: no finite values (unpriced models or no solves)")
            log.info("cost_effectiveness: panel %s skipped: no finite values", title)
            continue
        panels.append((title, point, lo if lo is not None else point * np.nan, hi if hi is not None else point * np.nan,
                       fmt, xlabel))
        uses += src
    if not panels:
        raise Skip("; ".join(notes) or "no cost/time per solve available")
    nres = rs["n_resolved"].reindex(names) if "n_resolved" in rs.columns else pd.Series(np.nan, names)
    fig = new_figure(2.0 + 2.9 * len(panels), 1.8 + 0.32 * len(names))
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    # (kind, conditions) -> panels, so a gap shared by every panel is noted once.
    gaps: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for ax, (title, p, lo, hi, fmt, xlabel) in zip(axes, panels):
        missing = {n: ("no valid trials" if n in no_valid else "no solves" if nres.get(n) == 0 else "unknown (NaN)")
                   for n in names}
        no_ci, unbounded = _hbars(ax, data, names, p.to_numpy(float), lo.to_numpy(float), hi.to_numpy(float),
                                  missing=missing)
        for kind, which in (("no_ci", no_ci), ("unbounded", unbounded)):
            if which:
                gaps.setdefault((kind, tuple(which)), []).append(f"'{title}'")
        if fmt is not None:
            ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(title)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_xlim(0, None)
    for (kind, which), titles in gaps.items():
        where = _panels_phrase(titles, len(panels))
        if kind == "no_ci":
            notes += _ci_note(list(which), where)
        else:
            notes.append(f"95% interval unbounded above {where} (more than 2.5% of the bootstrap resamples "
                         f"contain no solve; the whisker runs to the panel edge): {', '.join(which)}")
    handles = condition_legend(_infos(data, names), kind="patch")
    if any(kind == "unbounded" for kind, _ in gaps):
        handles.append(_unbounded_handle())
    outside_legend(fig, handles, ncols=min(len(handles), 5))
    fig.suptitle("What one solved task costs")
    caption = (
        "Cost and time per solved task by condition: total cost (or agent wall-clock) over all valid trials divided "
        "by the number solved, i.e. mean cost per episode / resolve rate, one panel per pricing scheme and one for "
        "time (over the valid trials with a known wall-clock). Shorter is better. Whiskers are 95% intervals: from "
        "the CSV when it has them, otherwise a paired bootstrap over episodes of the same ratio (2000 resamples, "
        "seed 0). A resample that draws no solved episode has an undefined (infinite) ratio and is kept, so when "
        "solves are rare the upper bound is unbounded: that whisker runs from its lower bound to the right edge of "
        "the panel with an arrow head, and the notes list it. A bar without a whisker has no computable interval "
        "(fewer than 2 valid episodes), also listed in the notes. Bars use the strategy colour with scope lightness "
        "(grey = baselines, hatched = controls, [manager] = a manager control); 'no solves' marks conditions that "
        "solved nothing (the ratio is undefined), 'unknown (NaN)' marks conditions with an unpriced model in that "
        "scheme, and 'no valid trials' conditions whose every trial was invalid. Schemes with no finite value at "
        "all are not drawn."
    )
    yield Rendered("cost_effectiveness", fig, "Cost and time per solved task", caption, uses, notes)


# ---------------------------------------------------------------------------
# 14. arbiter
# ---------------------------------------------------------------------------


def fig_arbiter(data: AnalysisData) -> Iterator[Rendered]:
    a = data.event_rows("arbitrations")
    if "condition" not in a.columns:
        raise Skip("arbitrations.csv lacks the condition column")
    a["condition"] = a["condition"].astype(str)
    names = data.order(a["condition"].unique())
    if not names:
        raise Skip("arbitrations.csv has no rows for known conditions")
    notes: list[str] = _dropped_note(a, "arbitration(s)")
    uses: list[str] = []
    panels: list[str] = []
    for p, cols in (("agree", ["agreement"]), ("conf", ["final_confidence"]), ("debate", ["debate_rounds_used"]),
                    ("resolved", ["final_confidence", "episode_id"]), ("winner", ["final_winner"])):
        if all(c in a.columns for c in cols):
            panels.append(p)
        else:
            notes.append(f"panel '{p}' not drawn: arbitrations.csv lacks {', '.join(c for c in cols if c not in a.columns)}")
    eps = None
    if "resolved" in panels:
        try:
            eps = data.episodes(["episode_id"])
        except Skip as e:
            panels.remove("resolved")
            notes.append(f"confidence-vs-resolved panel not drawn: {e}")
    if not panels:
        raise Skip("; ".join(notes))
    ncols = min(len(panels), 3)
    fig = new_figure(3.7 * ncols + 0.3, 3.2 * math.ceil(len(panels) / 3) + 1.1)
    axes = dict(zip(panels, panel_axes(fig, len(panels), max_cols=3)))
    x = np.arange(len(names))
    wn = 0.8 / len(names)
    labels = [wrap(data.meta[n].label, 14) for n in names]
    ref_handles: list = []
    if "agree" in axes:
        ax = axes["agree"]
        ag = as_bool(a["agreement"])
        for k, n in enumerate(names):
            m = (a["condition"] == n) & ag.notna()
            p, lo, hi = wilson(float(ag[m].sum()), float(m.sum()))
            if np.isfinite(p):
                ax.bar(k, p, width=0.6, **bar_style(data.meta[n]), zorder=2)
                ax.errorbar(k, p, yerr=errs([p], [lo], [hi]), fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
                ax.text(k, (hi if np.isfinite(hi) else p) + 0.02, f"n={int(m.sum())}", ha="center", fontsize=6.3, color=INK2)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("share of arbitrations")
        ax.set_title("judge says candidates agreed")
        vgrid(ax)
        uses.append(_cols("arbitrations", ["agreement"]))
    if "conf" in axes:
        ax = axes["conf"]
        vals = [num(a.loc[a["condition"] == n, "final_confidence"]).dropna().to_numpy() for n in names]
        if any(len(v) for v in vals):
            ax.hist(vals, bins=np.linspace(0, 1, 11), stacked=True, color=[fill_color(data.meta[n]) for n in names],
                    edgecolor=SURFACE, linewidth=0.8)
        nan_n = int(num(a["final_confidence"]).isna().sum())
        ax.set_xlabel("final judge confidence")
        ax.set_ylabel("arbitrations")
        ax.set_title("judge confidence" + (f" ({nan_n} unparsed)" if nan_n else ""))
        vgrid(ax)
        uses.append(_cols("arbitrations", ["final_confidence"]))
    if "debate" in axes:
        ax = axes["debate"]
        dr = num(a["debate_rounds_used"])
        levels = sorted(int(v) for v in dr.dropna().unique())
        for k, n in enumerate(names):
            m = a["condition"] == n
            tot = float(dr[m].notna().sum())
            fr = [float((dr[m] == lv).sum()) / tot if tot else np.nan for lv in levels]
            ax.bar(np.arange(len(levels)) + (k - (len(names) - 1) / 2) * wn, fr, width=wn * 0.92,
                   **bar_style(data.meta[n]), zorder=2)
        ax.set_xticks(range(len(levels)), [str(v) for v in levels])
        ax.set_xlabel("debate rounds used")
        ax.set_ylabel("share of arbitrations")
        ax.set_ylim(0, 1.05)
        ax.set_title("confidence-aware debate rounds")
        vgrid(ax)
        uses.append(_cols("arbitrations", ["debate_rounds_used"]))
    if "resolved" in axes and eps is not None:
        ax = axes["resolved"]
        conf = a.assign(_c=num(a["final_confidence"])).groupby("episode_id")["_c"].mean()
        j = eps.set_index("episode_id")[["condition", "resolved"]].join(conf.rename("conf"), how="inner")
        j = j[j["conf"].notna() & j["resolved"].notna()]
        rng = np.random.default_rng(0)
        no_ci: list[str] = []
        for k, n in enumerate(names):
            for outcome in (0.0, 1.0):
                v = j.loc[(j["condition"] == n) & (j["resolved"] == outcome), "conf"].to_numpy(float)
                if len(v) == 0:
                    continue
                xpos = outcome + (k - (len(names) - 1) / 2) * 0.16
                ax.plot(xpos + rng.uniform(-0.04, 0.04, len(v)), v, alpha=0.75,
                        **{**point_style(data.meta[n], 4.0), "markeredgewidth": 0.7})
                mm, lo, hi = bootstrap_mean(v)
                ax.errorbar(xpos + 0.06, mm, yerr=errs([mm], [lo], [hi]), fmt="_", color=INK, markersize=8,
                            elinewidth=1.0, capsize=2)
                if _ci_gaps([n], [mm], [lo], [hi])[0]:
                    no_ci.append(f"{n} ({'resolved' if outcome else 'not resolved'})")
        notes += _ci_note(no_ci, "for the mean confidence")
        ax.set_xticks([0, 1], ["not resolved", "resolved"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mean final confidence per episode")
        ax.set_title("confidence vs episode outcome")
        vgrid(ax)
        uses.append(_cols("episodes", ["episode_id", "resolved"]))
    if "winner" in axes:
        ax = axes["winner"]
        w = num(a["final_winner"])
        levels = sorted(int(v) for v in w.dropna().unique())
        for k, n in enumerate(names):
            m = a["condition"] == n
            tot = float(w[m].notna().sum())
            fr = [float((w[m] == lv).sum()) / tot if tot else np.nan for lv in levels]
            ax.bar(np.arange(len(levels)) + (k - (len(names) - 1) / 2) * wn, fr, width=wn * 0.92,
                   **bar_style(data.meta[n]), zorder=2)
        if "n_candidates" in a.columns and num(a["n_candidates"]).notna().any():
            nc = float(num(a["n_candidates"]).mode().iloc[0])
            ax.axhline(1 / nc, color=INK2, linestyle=":", linewidth=1.0)
            ref_handles.append(Line2D([], [], color=INK2, linestyle=":", linewidth=1.0,
                                      label=f"uniform winner rate (1/{nc:g})"))
        ax.set_xticks(range(len(levels)), [f"candidate {v}" for v in levels])
        ax.set_ylabel("share of arbitrations")
        ax.set_title("winning candidate index")
        vgrid(ax)
        uses.append(_cols("arbitrations", ["final_winner", "n_candidates"]))
    handles = condition_legend(_infos(data, names), kind="patch") + ref_handles
    if "resolved" in axes:
        handles.append(Line2D([], [], color=INK, marker="_", markersize=8, label="mean, bootstrap 95% CI"))
    outside_legend(fig, handles, ncols=min(len(handles), 5))
    fig.suptitle("Arbiter: agreement, confidence, debate and winners")
    caption = (
        "One observation per arbitrated instruction of a valid trial (N candidate subagents + an LLM judge), "
        "coloured by condition (arbiter strategy hue, lighter = easier scope). Panels, where the columns exist: the "
        "share of arbitrations "
        "whose judge said the candidates agreed (Wilson 95% interval); the distribution of the final judge "
        "confidence (stacked by condition; unparsed judgements have no confidence and are counted in the title); how "
        "many confidence-aware debate rounds ran (0 = the first judgement was confident and agreeing); the episode's "
        "mean final confidence split by whether the episode was resolved (dots = episodes, black tick = mean with "
        "bootstrap 95% interval), a check of whether judge confidence predicts success; and how often each "
        "candidate index won, against the uniform rate (a skew toward candidate 0 hints at position bias or parse "
        "fallbacks, which default to 0)."
    )
    yield Rendered("arbiter", fig, "Arbiter behaviour", caption, uses, notes)
