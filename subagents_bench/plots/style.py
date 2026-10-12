"""Visual vocabulary shared by every figure.

One encoding everywhere, so a reader learns it once:

* **strategy -> hue**: parallel blue, sequential orange, arbiter aqua (the
  first three slots of a validated categorical palette; they pass the
  all-pairs colour-vision checks, so they also work in scatter plots);
* **scope -> lightness** (easy lightest, hard full strength) for fills, and
  **scope -> marker shape** for points (easy o, medium s, hard ^);
* **baselines** (topology single) are neutral grey: horizontal reference bands
  (line = estimate, band = 95% CI) where the y axis is a rate, grey marks
  elsewhere;
* **controls** (conditions off the primary strategy x scope grid, e.g. the
  same-model control) use their strategy/scope style but hollow markers or a
  hatched fill, and are always named; a manager control (orchestrator tools
  restricted, so it must delegate) is also tagged ``[manager]`` in labels
  and legends;
* **roles** have their own hues (orchestrator violet, arbiter red, subagent
  yellow, debate magenta, tools green), stacked in an order that keeps
  adjacent segments distinguishable.

Figures are built on bare :class:`matplotlib.figure.Figure` objects (Agg, no
pyplot global state) under :data:`RC`.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .data import SCOPES, STRATEGIES, CondInfo

# -- ink and chrome ------------------------------------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

STRATEGY_COLORS = {"parallel": "#2a78d6", "sequential": "#eb6834", "arbiter": "#1baf7a"}
UNKNOWN_COLOR = "#3d3c3a"
# Fraction mixed toward white per scope (lightness encodes scope).
SCOPE_TINT = {"easy": 0.55, "medium": 0.28, "hard": 0.0}
SCOPE_MARKERS = {"easy": "o", "medium": "s", "hard": "^"}
SCOPE_LINESTYLES = {"easy": ":", "medium": "--", "hard": "-"}
# Baselines: (colour, line style, marker) by baseline index.
BASELINE_STYLES = [("#3d3c3a", "-", "*"), ("#898781", "--", "X"), ("#5f5e5a", ":", "P"), ("#a8a69e", "-.", "h")]

ROLE_COLORS = {"orch": "#6250d6", "arb": "#e34948", "sub": "#eda100", "deb": "#e87ba4"}
# Stack order chosen so adjacent role segments pass the CVD/normal-vision checks.
ROLE_STACK = ("orch", "arb", "sub", "deb")
TOOL_COLOR = "#008300"
OVERHEAD_COLOR = "#c3c2b7"
RETRY_COLOR = "#898781"

# Token kinds: input in neutral greys (context), output in greens.
KIND_COLORS = {
    "uncached input": "#52514e",
    "cached input": "#c3c2b7",
    "reasoning output": "#006b00",
    "visible output": "#86c786",
    "output (reasoning not reported)": "#3f9a3f",
}

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "sb_blues", ["#f4f3f0", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ROUND_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b", "#5598e7", "#256abf"]

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "font.size": 8.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "legend.title_fontsize": 8,
    "figure.titlesize": 11,
    "figure.titleweight": "bold",
    "errorbar.capsize": 2.5,
    "lines.linewidth": 1.4,
    "hatch.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.constrained_layout.h_pad": 0.06,
    "figure.constrained_layout.w_pad": 0.06,
}


def tint(color: str, frac: float) -> tuple[float, float, float]:
    """Mix ``color`` toward white by ``frac`` (0 = unchanged)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def base_color(info: CondInfo) -> str:
    if info.baseline:
        return BASELINE_STYLES[info.baseline_index % len(BASELINE_STYLES)][0]
    return STRATEGY_COLORS.get(info.strategy or "", UNKNOWN_COLOR)


def fill_color(info: CondInfo) -> tuple[float, float, float]:
    """Bar fill: strategy hue at the scope's lightness (baselines: grey)."""
    c = base_color(info)
    return to_rgb(c) if info.baseline else tint(c, SCOPE_TINT.get(info.scope or "", 0.0))


def marker(info: CondInfo) -> str:
    if info.baseline:
        return BASELINE_STYLES[info.baseline_index % len(BASELINE_STYLES)][2]
    return SCOPE_MARKERS.get(info.scope or "", "D")


def point_style(info: CondInfo, size: float = 6.5) -> dict:
    """Keyword args for ``ax.plot``/``ax.errorbar`` markers of a condition."""
    c = base_color(info)
    face = SURFACE if info.control else fill_color(info)
    ms = size * (1.5 if info.baseline else 1.0)
    return {"marker": marker(info), "markersize": ms, "markerfacecolor": face, "markeredgecolor": c,
            "markeredgewidth": 1.2, "color": c, "linestyle": "none"}


def bar_style(info: CondInfo) -> dict:
    """Keyword args for ``ax.bar``/``ax.barh`` of a condition."""
    c = base_color(info)
    st = {"color": fill_color(info), "edgecolor": c, "linewidth": 0.8}
    if info.control:
        # White hatch: visible on full-strength and tinted fills alike.
        st.update(hatch="////", hatchcolor=SURFACE)
    return st


def wrap(text: str, width: int = 18) -> str:
    """Wrap a condition name at underscores so tick labels never collide."""
    lines: list[str] = []
    cur = ""
    for part in text.split("_"):
        cand = f"{cur}_{part}" if cur else part
        if cur and len(cand) > width:
            lines.append(cur + "_")
            cur = part
        else:
            cur = cand
    lines.append(cur)
    return "\n".join(lines)


def ci_known(lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    """Boolean mask: both CI bounds are finite and the interval has a width.

    A zero-width interval (e.g. a bootstrap over identical values) is
    degenerate, not exact, and counts as unknown.
    """
    lo_, hi_ = np.asarray(lo, float), np.asarray(hi, float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(lo_) & np.isfinite(hi_) & (hi_ > lo_)


def errs(mean: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    """2xN error-bar array from CI bounds.

    An interval with a non-finite bound (unknown, or unbounded above) or no
    width gives NaN on both sides, for which matplotlib draws neither whisker
    nor caps: a zero-length capped whisker would look like an exact estimate.
    Callers say so in the figure notes (and draw unbounded intervals
    themselves).
    """
    m = np.asarray(mean, float)
    lo_ = np.asarray(lo, float)
    hi_ = np.asarray(hi, float)
    ok = ci_known(lo_, hi_) & np.isfinite(m)
    with np.errstate(invalid="ignore"):
        down = np.where(ok, np.clip(m - lo_, 0, None), np.nan)
        up = np.where(ok, np.clip(hi_ - m, 0, None), np.nan)
    return np.vstack([down, up])


def new_figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), layout="constrained")


def panel_axes(fig: Figure, n: int, *, max_cols: int = 3, sharex: bool = False, sharey: bool = False) -> list:
    """``n`` axes in rows of at most ``max_cols``; an incomplete last row is centred."""
    if n <= 0:
        return []
    ncols = min(n, max_cols)
    nrows = math.ceil(n / ncols)
    labels = [chr(ord("A") + i) if i < 26 else f"P{i}" for i in range(n)]
    mosaic = []
    k = 0
    for _ in range(nrows):
        row_n = min(ncols, n - k)
        pad = ncols - row_n
        row = ["."] * pad
        for j in range(row_n):
            row += [labels[k + j]] * 2
        row += ["."] * pad
        mosaic.append(row)
        k += row_n
    axd = fig.subplot_mosaic(mosaic, sharex=sharex, sharey=sharey, empty_sentinel=".")
    return [axd[lab] for lab in labels]


def log_scale_if_wide(ax, values: Iterable[float], axis: str = "x", ratio: float = 20.0) -> bool:
    """Log-scale the axis when the positive values span ``ratio`` or more.

    Ticks go at 1, 2 and 5 per decade with plain labels (callers may set
    their own major formatter); minor ticks stay unlabelled, so a range of
    less than a decade still gets labels that never collide.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x) and x > 0], float)
    if len(v) >= 2 and v.max() / v.min() >= ratio:
        (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
        a = ax.xaxis if axis == "x" else ax.yaxis
        a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        a.set_major_formatter(FuncFormatter(lambda t, _p: f"{t:g}"))
        a.set_minor_formatter(NullFormatter())
        return True
    return False


def hgrid(ax) -> None:
    """Gridlines along x only (for horizontal bars/boxes)."""
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)


def vgrid(ax) -> None:
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)


def condition_legend(infos: Iterable[CondInfo], *, kind: str = "marker", scope_legend: bool = True):
    """Legend handles: strategies, scopes, each baseline, each control."""
    infos = list(infos)
    handles: list = []
    strategies = [s for s in STRATEGIES if any(i.strategy == s and not i.baseline for i in infos)]
    scopes = [s for s in SCOPES if any(i.scope == s and not i.baseline for i in infos)]
    for s in strategies:
        c = STRATEGY_COLORS[s]
        if kind == "marker":
            handles.append(Line2D([], [], color=c, marker="o", linestyle="none", markersize=7, label=f"{s}"))
        elif kind == "line":
            handles.append(Line2D([], [], color=c, linewidth=2, label=s))
        else:
            handles.append(Patch(facecolor=c, edgecolor=c, label=s))
    if scope_legend:
        for s in scopes:
            if kind == "marker":
                handles.append(Line2D([], [], color=INK2, marker=SCOPE_MARKERS[s], linestyle="none",
                                      markerfacecolor=tint(INK2, SCOPE_TINT[s]), markersize=7, label=f"{s} scope"))
            elif kind == "line":
                handles.append(Line2D([], [], color=INK2, linestyle=SCOPE_LINESTYLES[s], linewidth=1.5, label=f"{s} scope"))
            else:
                handles.append(Patch(facecolor=tint(INK2, SCOPE_TINT[s]), edgecolor=INK2, label=f"{s} scope"))
    for i in infos:
        if i.baseline:
            c, ls, m = BASELINE_STYLES[i.baseline_index % len(BASELINE_STYLES)]
            if kind == "marker":
                handles.append(Line2D([], [], color=c, marker=m, linestyle="none", markersize=9, label=f"{i.name} (baseline)"))
            elif kind == "line":
                handles.append(Line2D([], [], color=c, linestyle=ls, linewidth=1.5, label=f"{i.name} (baseline)"))
            else:
                handles.append(Patch(facecolor=c, edgecolor=c, label=f"{i.name} (baseline)"))
    for i in infos:
        if i.control and not i.baseline:
            if kind == "marker":
                handles.append(Line2D([], [], **{**point_style(i), "label": f"{i.name} ({i.kind}; hollow)"}))
            elif kind == "line":
                handles.append(Line2D([], [], color=base_color(i), linestyle=SCOPE_LINESTYLES.get(i.scope or "", "-"),
                                      linewidth=1.0, marker=".", label=f"{i.name} ({i.kind})"))
            else:
                st = bar_style(i)
                handles.append(Patch(facecolor=st["color"], edgecolor=st["edgecolor"], hatch=st["hatch"],
                                     hatchcolor=st["hatchcolor"], label=f"{i.name} ({i.kind}; hatched)"))
    return handles


def outside_legend(fig: Figure, handles: list, *, ncols: int | None = None, loc: str = "outside lower center",
                   title: str | None = None) -> None:
    if not handles:
        return
    n = len(handles)
    fig.legend(handles=handles, loc=loc, ncols=ncols or min(n, 6), title=title, handlelength=1.6,
               columnspacing=1.4, borderaxespad=0.2)
