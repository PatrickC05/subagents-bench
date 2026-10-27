"""Experiment 2 of the final study (issue #23): the three settings at L0-L2, compared across models.

    python -m subagents_bench.plots.final --model DeepSeek=analysis/final_levels \
        --model "Opus 5.5"=analysis/final_levels_opus ... --out figures/final

Each ``--model NAME=DIR`` is one model's levels sweep, as ``scripts/analyze.sh`` writes it
(``messages.csv``, ``estimates.csv``, ``calls.csv``, ``tools.csv``, ``episodes.csv``). Writes six
figures, each with the table it draws (``<name>.csv``):

1. ``response_time``: median seconds from a bug being sent to the first reply the user saw
   (``user_reply_sec``; not the session's first bug), per model, level and setting. A bug never
   replied to counts as never (``interactivity._median_censored``: NaN when the median is never):
   such a bar is drawn hatched at the session length.
2. ``bugs_fixed``: share of the bugs sent that pass at the end (the verifier, ``fixed``).
3. ``eta``: each scored estimate (``estimates.csv``) against the minutes the fix actually took from
   then (replayed ``fixed_sec``): the overestimation factor per model and setting (``eta.png``) and
   estimate against actual (``eta_scatter.png``); a bug never fixed counts at the session's end, as
   a lower bound. The on-time share is weighted, one weight per bug.
4. ``response_cdf``: share of bugs replied to within t, per setting, one panel per model. A curve
   that stays below 1 is bugs never replied to.
5. ``orchestrator_time``: the share of each session the orchestrator spends in model calls, in its
   own tool calls (``wait`` excluded), and neither (waiting for the user or subagents), per model
   and setting.
6. ``question_vs_bug``: median time to the first reply to a question (``reply_sec``) and to a bug
   (``user_reply_sec``) at L2, censored as in 1, with the median seconds above each bar.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.estimates import _flag
from ..analysis.final import counted
from ..analysis.interactivity import _median_censored
from ..analysis.levels import setting_level

SETTINGS = ["background_hinted", "manager_background", "manager_owner"]
SETTING_LABEL = {"background_hinted": "Orchestrator + subagents", "manager_background": "Subagents only",
                 "manager_owner": "Subagents only (owner)"}
LEVELS = {"L0": "L0 bugs", "L1": "L1 + ETA", "L2": "L2 + questions"}  # levels.setting_level's; no urgent tier (L3)
SETTING_SHORT = {"background_hinted": "Orch + sub", "manager_background": "Sub only", "manager_owner": "Owner"}
MARKERS = {"background_hinted": "o", "manager_background": "s", "manager_owner": "D", "isolated": "*",
           "isolated_capped": "*"}  # experiment 1's points (analysis.final)
# the columns each table is read with (calls.csv and tools.csv have hundreds of thousands of rows)
COLUMNS = {"messages": ["condition", "trial_name", "valid", "kind", "index", "user_reply_sec", "reply_sec", "fixed"],
           "estimates": ["condition", "trial_name", "valid", "scored", "task_name", "subtask_id", "ready_session_sec",
                         "est_sec", "eta_min", "fixed_sec", "on_time", "weight"],
           "calls": ["condition", "trial_name", "valid", "role", "t_start", "t_end"],
           "tools": ["condition", "trial_name", "valid", "role", "name", "t_start", "t_end"],
           "episodes": ["condition", "trial_name", "valid", "invalid_reason", "wall_clock_sec"]}
COLORS = {"background_hinted": "#3a6fb0", "manager_background": "#c4571f", "manager_owner": "#d9a634"}
WALL = 1800.0


def _read(d: Path, name: str, keep: set[str] | None, required: bool = True) -> pd.DataFrame:
    """A table's rows of the counted trials (``keep``; None: every row) in the plotted cells."""
    path = d / f"{name}.csv"
    if not path.is_file():
        if required:
            raise SystemExit(f"[FAIL] {path} not found: run scripts/analyze.sh on that sweep first")
        return pd.DataFrame(columns=COLUMNS[name])
    df = pd.read_csv(path, usecols=lambda c: c in COLUMNS[name], low_memory=False)
    if keep is not None:
        df = df[df["trial_name"].isin(keep)]
    sl = df["condition"].map(setting_level)
    df = df.assign(setting=sl.str[0], level=sl.str[1])
    return df[df["setting"].isin(SETTINGS) & df["level"].isin(LEVELS)]


def load(models: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Every model's tables, concatenated with a ``model`` column. The valid sessions and those that
    failed (``analysis.final.counted``: a provider error or an agent crash), which count as failures: their bugs are
    not fixed and the requests they never answered count as never answered."""
    out: dict[str, list[pd.DataFrame]] = {}
    for model, d in models.items():
        ep = _read(d, "episodes", None)
        keep, failed = counted(ep)
        out.setdefault("episodes", []).append(ep[ep["trial_name"].isin(keep)].assign(model=model))
        for name, required in (("messages", True), ("estimates", False), ("calls", False), ("tools", False)):
            df = _read(d, name, keep, required).assign(model=model)
            if name == "messages":
                df["fixed"] = _flag(df, "fixed") & ~df["trial_name"].isin(failed)
                df["lost"] = df["trial_name"].isin(failed)
            out.setdefault(name, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def _never(median: float, n: int) -> bool:
    """``interactivity._median_censored`` is NaN when the censored median is never (or no rows)."""
    return n > 0 and math.isnan(median)


def pyplot():
    """matplotlib.pyplot on the Agg backend (no display needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def named_dirs(n: int):
    """An argparse type for ``NAME=DIR`` (n = 1) or ``NAME=DIR:DIR`` (n = 2): (name, dirs)."""
    def parse(text: str) -> tuple[str, Path | tuple[Path, ...]]:
        name, sep, rest = text.partition("=")
        dirs = rest.split(":") if rest else []
        if not sep or not name or len(dirs) != n or not all(dirs):
            raise argparse.ArgumentTypeError(f"expected NAME={':'.join(['DIR'] * n)}, got {text!r}")
        return name, Path(dirs[0]) if n == 1 else tuple(Path(d) for d in dirs)
    return parse


def _log_limits(values, pad: float = 1.6, floor: float = 1e-3) -> tuple[float, float]:
    """Log-axis limits around the finite positive values, padded by a factor."""
    v = [x for x in values if x is not None and math.isfinite(x) and x > 0]
    if not v:
        return 1.0, 10.0
    return max(floor, min(v) / pad), max(v) * pad


def _bars(ax, table: pd.DataFrame, value: str, models: list[str], ylabel: str, log: bool = False,
          extra: str | None = None) -> None:
    """Grouped bars: x = model x the levels it ran, one bar per setting. With ``log``, a log axis
    fitted to the data, and a median that is never (NaN with rows) drawn hatched up to the top. With
    ``extra``, a dashed outline of that column's height stacked on each bar."""
    width, x0 = 0.8 / len(SETTINGS), 0
    ticks, labels = [], []
    top = None
    if log:
        lo, hi = _log_limits(table[value])
        top = hi * 2.5  # room above the tallest bar for the never bars
        ax.set_yscale("log")
    for m in models:
        levels = [lv for lv in LEVELS if ((table["model"] == m) & (table["level"] == lv)).any()]
        first = len(ticks)
        for lv in levels:
            for j, s in enumerate(SETTINGS):
                r = table[(table["model"] == m) & (table["level"] == lv) & (table["setting"] == s)]
                if r.empty:
                    continue
                v = float(r[value].iloc[0])
                never = log and _never(v, int(r["n"].iloc[0]))
                x = x0 + j * width
                if never:
                    ax.bar(x, top - lo, width, bottom=lo, color="white", edgecolor=COLORS[s], hatch="//")
                    ax.annotate("never", (x, top), ha="center", va="top", fontsize=5, rotation=90)
                else:
                    ax.bar(x, v - (lo if log else 0), width, bottom=lo if log else 0, color=COLORS[s])
                    if extra and float(r[extra].iloc[0]) > 0:
                        ax.bar(x, float(r[extra].iloc[0]), width, bottom=v, fill=False, edgecolor=COLORS[s],
                               linestyle="--", linewidth=1)
            ticks.append(x0 + width * (len(SETTINGS) - 1) / 2)
            labels.append(lv)
            x0 += 1
        if len(ticks) > first:
            ax.text((ticks[-1] + ticks[first]) / 2, -0.12, m, transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=8, fontweight="bold")
        x0 += 0.5
    ax.set_xticks(ticks, labels, fontsize=7)
    if log:
        ax.set_ylim(lo, top)
    ax.set_ylabel(ylabel)
    _legend(ax, extra=extra is not None and bool((table[extra] > 0).any()))


def _legend(ax, extra: bool = False) -> None:
    from matplotlib.patches import Patch
    handles = [_patch(s) for s in SETTINGS]
    if extra:
        handles.append(Patch(fill=False, edgecolor="dimgrey", linestyle="--", label="bugs in lost sessions"))
    ax.legend(handles=handles, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=len(handles), frameon=False)


def _patch(s: str):
    from matplotlib.patches import Patch
    return Patch(color=COLORS[s], label=SETTING_LABEL[s])


def response_time(t: dict[str, pd.DataFrame], models: list[str], out: Path) -> pd.DataFrame:
    m = t["messages"]
    bugs = m[(m["kind"] == "bug") & (m["index"] > 0)]
    tab = bugs.groupby(["model", "setting", "level"])["user_reply_sec"].agg(
        median_sec=_median_censored, replied_frac=lambda v: v.notna().mean(), n="size").reset_index()
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(max(6, 3 * len(models)), 4))
    _bars(ax, tab, "median_sec", models, "median s to first reply (log; hatched: never)", log=True)
    fig.suptitle("Time to the first reply the user saw, bugs (L0 bugs, L1 + ETA, L2 + questions)", fontsize=9)
    _save(fig, out / "response_time.png")
    return tab


def bugs_fixed(t: dict[str, pd.DataFrame], models: list[str], out: Path) -> pd.DataFrame:
    m = t["messages"]
    bugs = m[m["kind"] == "bug"].assign(fixed=lambda d: _flag(d, "fixed"))  # no verdict: not fixed
    bugs = bugs.assign(lost=bugs["lost"].fillna(False).astype(bool))
    tab = bugs.groupby(["model", "setting", "level"]).agg(
        fixed_frac=("fixed", "mean"), n_fixed=("fixed", "sum"), n=("fixed", "size"),
        lost_frac=("lost", "mean")).reset_index()
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(max(6, 3 * len(models)), 4))
    _bars(ax, tab, "fixed_frac", models, "share of bugs fixed", extra="lost_frac")
    ax.set_ylim(max(0.0, float(tab["fixed_frac"].min()) - 0.1), max(1.02, float((tab["fixed_frac"] + tab["lost_frac"]).max()) + 0.02))
    fig.suptitle("Bugs fixed at the end of the session", fontsize=9)
    _save(fig, out / "bugs_fixed.png")
    return tab


def eta(t: dict[str, pd.DataFrame], models: list[str], out: Path) -> pd.DataFrame:
    """How far each estimate is off: ``eta.png`` shows estimate / actual time to fix (log; 1 = exact,
    above = overestimated) per model and setting, for the bugs that were fixed, with each model's
    median factor and on-time share; ``eta_scatter.png`` shows estimate against actual, each axis
    fitted to its own data (bugs never fixed as triangles at the session's end, a lower bound)."""
    e = t["estimates"]
    if e.empty:
        return e
    e = e[_flag(e, "scored")].copy()
    start = e["ready_session_sec"] + e["est_sec"]  # seconds into the session the estimate was given
    fixed = pd.to_numeric(e["fixed_sec"], errors="coerce")
    e["predicted_min"] = e["eta_min"]
    e["actual_min"] = ((fixed - e["est_sec"]).where(fixed.notna(), WALL - start)) / 60
    e["never_fixed"] = fixed.isna()
    e["ratio"] = (e["predicted_min"] / e["actual_min"].clip(lower=1 / 60)).where(~e["never_fixed"])
    models = [m for m in models if (e["model"] == m).any()]
    if not models:
        return e
    plt = pyplot()
    rng = np.random.default_rng(0)

    # 1. the overestimation factor
    fig, ax = plt.subplots(figsize=(max(6, 2.6 * len(models)), 4.2))
    width = 0.8 / len(SETTINGS)
    ratios = e["ratio"].dropna()
    lo, hi = _log_limits(list(ratios) + [1.0], pad=1.8)
    for i, mname in enumerate(models):
        d = e[e["model"] == mname]
        for j, s in enumerate(SETTINGS):
            r = d.loc[(d["setting"] == s), "ratio"].dropna()
            if r.empty:
                continue
            x = i + (j - (len(SETTINGS) - 1) / 2) * width
            ax.boxplot(r, positions=[x], widths=width * 0.8, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": COLORS[s], "alpha": 0.35, "edgecolor": COLORS[s]},
                       medianprops={"color": COLORS[s], "lw": 1.5}, whiskerprops={"color": COLORS[s]},
                       capprops={"color": COLORS[s]})
            ax.scatter(x + rng.uniform(-width * 0.3, width * 0.3, len(r)), r, s=6, color=COLORS[s], alpha=0.6, zorder=3)
        med = float(d["ratio"].median())
        w = d["weight"].fillna(0)
        on_time = float((_flag(d, "on_time") * w).sum() / w.sum()) if w.sum() else math.nan
        n_never = int(d["never_fixed"].sum())
        ax.annotate(f"median ×{med:.1f}\non time {on_time:.0%}\n{n_never} never fixed", (i, hi), ha="center", va="top",
                    fontsize=7)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.annotate("exact", (len(models) - 0.5, 1.0), ha="right", va="bottom", fontsize=7, color="grey")
    ax.set_yscale("log")
    ax.set_ylim(lo, hi * 4)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_xticks(range(len(models)), models, fontsize=8)
    ax.set_ylabel("estimate / actual time to fix (log; >1: overestimated)")
    _legend(ax)
    fig.suptitle("How far the ETAs are off (bugs fixed in the session)", fontsize=9)
    _save(fig, out / "eta.png")

    # 2. estimate against actual
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.8), squeeze=False)
    for ax, mname in zip(axes[0], models):
        d = e[e["model"] == mname]
        xlo, xhi = 0.0, float(d["predicted_min"].max()) * 1.1 + 0.5  # estimates span a few minutes: linear
        ylo, yhi = _log_limits(d["actual_min"], pad=1.5, floor=1 / 60)
        for s in SETTINGS:
            x = d[d["setting"] == s]
            for never, marker in ((False, "o"), (True, "^")):
                y = x[x["never_fixed"] == never]
                ax.scatter(y["predicted_min"], y["actual_min"].clip(lower=ylo), s=14, marker=marker, color=COLORS[s],
                           alpha=0.7, facecolors="none" if never else None)
        line = np.linspace(max(xlo, ylo, 1e-3), xhi, 100)
        ax.plot(line, line, color="grey", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_title(mname, fontsize=9)
        ax.set_xlabel("estimate (min)")
    axes[0][0].set_ylabel("actual minutes to fix (log)\n(triangles: never fixed, at session end)")
    fig.legend(handles=[_patch(s) for s in SETTINGS], fontsize=7, loc="lower center", ncol=len(SETTINGS), frameon=False)
    fig.subplots_adjust(bottom=0.22, wspace=0.35)
    _save(fig, out / "eta_scatter.png", tight=False)
    eta_vs_fix(e, models, out)
    return e[["model", "setting", "level", "task_name", "subtask_id", "predicted_min", "actual_min", "ratio",
              "never_fixed", "on_time", "weight"]]


def eta_vs_fix(e: pd.DataFrame, models: list[str], out: Path) -> None:
    """``eta_scatter_all.png``: each estimate's predicted fix (when it was given + its ETA) against when the
    bug was fixed, both in minutes after the bug was sent, every setting and level pooled; one panel per
    model (log axes, the same on every panel), bugs never fixed in a band at the top."""
    from matplotlib.lines import Line2D
    plt = pyplot()
    rng = np.random.default_rng(0)
    pred = e["est_sec"] / 60 + e["eta_min"]
    fixed = pd.to_numeric(e["fixed_sec"], errors="coerce") / 60
    d = e.assign(pred=pred, fix=fixed)
    lo, hi, never_y = 0.2, 40.0, 70.0
    ncol = 4
    nrow = math.ceil((len(models) + 1) / ncol)  # one cell left for the legend
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.3 * nrow), sharex=True, sharey=True, squeeze=False)
    late_c, ok_c = COLORS["manager_background"], COLORS["background_hinted"]
    for ax, mname in zip(axes.flat, models):
        g = d[d["model"] == mname]
        nv = g[g["never_fixed"]]
        ok = g[~g["never_fixed"] & (g["fix"] <= g["pred"])]
        late = g[~g["never_fixed"] & (g["fix"] > g["pred"])]
        ax.axhspan(hi, never_y * 1.4, color="0.94", zorder=0)
        ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--", zorder=1)
        ax.scatter(ok["pred"], ok["fix"].clip(lower=lo), s=16, color=ok_c, alpha=0.6, linewidths=0, zorder=2)
        ax.scatter(late["pred"], late["fix"].clip(lower=lo), s=22, color=late_c, alpha=0.9, linewidths=0, zorder=3)
        ax.scatter(nv["pred"], never_y * np.exp(rng.uniform(-0.15, 0.15, len(nv))), s=16, facecolors="none",
                   edgecolors="grey", linewidths=0.8, zorder=2)
        n = len(g) - len(nv)
        ax.set_title(f"{mname}\n{len(ok)}/{n} fixed by their ETA" if n else mname, fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo * 5, hi)
        ax.set_ylim(lo, never_y * 1.4)
        ticks = [1, 2, 5, 10, 20]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_yticks([0.5] + ticks + [never_y], ["0.5"] + [str(t) for t in ticks] + ["never"])
        ax.minorticks_off()
        ax.tick_params(labelsize=10)
        ax.grid(True, color="0.9", lw=0.6)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    for i, ax in enumerate(axes.flat[:len(models)]):
        if i + ncol >= len(models):  # nothing below it: label its x axis
            ax.set_xlabel("predicted time (min)", fontsize=11)
            ax.xaxis.set_tick_params(labelbottom=True)
    for row in axes:
        row[0].set_ylabel("actual time (min)", fontsize=11)
    handles = [Line2D([], [], color=ok_c, marker="o", ls="", markeredgewidth=0, label="fixed by its ETA"),
               Line2D([], [], color=late_c, marker="o", ls="", markeredgewidth=0, label="fixed after its ETA"),
               Line2D([], [], color="grey", marker="o", ls="", markerfacecolor="none", label="never fixed"),
               Line2D([], [], color="grey", ls="--", label="ETA exact")]
    spare = axes.flat[len(models)] if len(models) < axes.size else None
    if spare is not None:
        spare.legend(handles=handles, fontsize=11, loc="center", frameon=False)
    else:
        fig.legend(handles=handles, fontsize=11, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Predicted vs actual fix time for each estimate", fontsize=14)
    _save(fig, out / "eta_scatter_all.png")


def response_cdf(t: dict[str, pd.DataFrame], models: list[str], out: Path) -> pd.DataFrame:
    m = t["messages"]
    bugs = m[(m["kind"] == "bug") & (m["index"] > 0)]
    replied = pd.to_numeric(bugs["user_reply_sec"], errors="coerce").dropna()
    lo, hi = _log_limits(list(replied), pad=1.3, floor=0.5)
    hi = min(hi, WALL)
    grid = np.geomspace(lo, hi, 200)
    rows = []
    plt = pyplot()
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5), squeeze=False, sharey=True)
    for ax, mname in zip(axes[0], models):
        for s in SETTINGS:
            v = pd.to_numeric(bugs[(bugs["model"] == mname) & (bugs["setting"] == s)]["user_reply_sec"], errors="coerce")
            if v.empty:
                continue
            cdf = [(v <= g).sum() / len(v) for g in grid]
            ax.step(grid, cdf, where="post", color=COLORS[s], label=SETTING_LABEL[s])
            rows += [{"model": mname, "setting": s, "t_sec": g, "replied_frac": c} for g, c in zip(grid, cdf)]
        ax.set_title(mname, fontsize=9)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("seconds after the bug was sent (log)")
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("share of bugs replied to")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    _save(fig, out / "response_cdf.png")
    return pd.DataFrame(rows)


def _busy_sec(df: pd.DataFrame) -> pd.Series:
    """Per trial: seconds covered by the orchestrator's intervals (overlaps merged)."""
    def merged(g: pd.DataFrame) -> float:
        total, end = 0.0, -math.inf
        for a, b in sorted(zip(g["t_start"], g["t_end"])):
            if b <= end:
                continue
            total += b - max(a, end)
            end = b
        return total
    if "role" not in df:  # no calls.csv / tools.csv
        return pd.Series(dtype=float)
    o = df[df["role"] == "orchestrator"]
    return o.groupby("trial_name")[["t_start", "t_end"]].apply(merged) if len(o) else pd.Series(dtype=float)


def orchestrator_time(t: dict[str, pd.DataFrame], models: list[str], out: Path) -> pd.DataFrame:
    ep = t["episodes"][["model", "setting", "level", "trial_name", "wall_clock_sec"]].copy()
    tools = t["tools"]
    if len(tools):
        tools = tools[tools["name"] != "wait"]  # `wait` blocks until a message comes: waiting, not working
    ep["llm_sec"] = ep["trial_name"].map(_busy_sec(t["calls"])).fillna(0)
    ep["tool_sec"] = ep["trial_name"].map(_busy_sec(tools)).fillna(0)
    for c in ("llm", "tool"):
        ep[f"{c}_frac"] = (ep[f"{c}_sec"] / ep["wall_clock_sec"]).clip(0, 1)
    ep["other_frac"] = (1 - ep["llm_frac"] - ep["tool_frac"]).clip(lower=0)
    tab = ep.groupby(["model", "setting"])[["llm_frac", "tool_frac", "other_frac"]].mean().reset_index()
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(models)), 4))
    x, ticks, labels = 0, [], []
    for mname in models:
        for s in SETTINGS:
            r = tab[(tab["model"] == mname) & (tab["setting"] == s)]
            if r.empty:
                continue
            bottom = 0.0
            for col, alpha, hatch in (("llm_frac", 1.0, None), ("tool_frac", 0.55, None), ("other_frac", 0.15, None)):
                v = float(r[col].iloc[0])
                ax.bar(x, v, 0.8, bottom=bottom, color=COLORS[s], alpha=alpha, hatch=hatch, edgecolor=COLORS[s])
                bottom += v
            ticks.append(x)
            labels.append(SETTING_SHORT[s])
            x += 1
        ax.text(x - 1 - (len(SETTINGS) - 1) / 2, -0.14, mname, transform=ax.get_xaxis_transform(), ha="center",
                fontsize=8, fontweight="bold")
        x += 0.5
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_ylabel("share of the session")
    ax.set_ylim(0, 1)
    from matplotlib.patches import Patch
    handles = [Patch(color="dimgrey", alpha=a, label=l) for a, l in
               ((1.0, "in model calls"), (0.55, "in its own tools"), (0.15, "neither (waiting)"))]
    fig.suptitle("Where the orchestrator's time goes", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    _save(fig, out / "orchestrator_time.png", tight=False)
    return tab


def question_vs_bug(t: dict[str, pd.DataFrame], models: list[str], out: Path) -> pd.DataFrame:
    m = t["messages"]
    m = m[m["level"] == "L2"]
    q = m[m["kind"] == "question"].assign(kind="question", sec=lambda d: d["reply_sec"])
    b = m[(m["kind"] == "bug") & (m["index"] > 0)].assign(kind="bug", sec=lambda d: d["user_reply_sec"])
    both = pd.concat([q, b])
    tab = both.groupby(["model", "setting", "kind"])["sec"].agg(
        median_sec=_median_censored, mean_replied_sec="mean", replied_frac=lambda v: v.notna().mean(), n="size").reset_index()
    plt = pyplot()
    models = [mn for mn in models if (tab["model"] == mn).any()]  # the models that ran L2
    if not models:
        return tab
    lo, hi = _log_limits(tab["median_sec"])
    top = hi * 2.5
    fig, axes = plt.subplots(1, len(models), figsize=(max(6.5, 4 * len(models)), 4.2), squeeze=False, sharey=True)
    for ax, mname in zip(axes[0], models):
        ax.set_yscale("log")
        for j, s in enumerate(SETTINGS):
            for k, (kind, alpha) in enumerate((("bug", 1.0), ("question", 0.5))):
                r = tab[(tab["model"] == mname) & (tab["setting"] == s) & (tab["kind"] == kind)]
                if r.empty:
                    continue
                v, frac = float(r["median_sec"].iloc[0]), float(r["replied_frac"].iloc[0])
                never = _never(v, int(r["n"].iloc[0]))
                xpos = j + (k - 0.5) * 0.38
                height = top if never else v
                ax.bar(xpos, height - lo, 0.38, bottom=lo, color="white" if never else COLORS[s], alpha=alpha,
                       edgecolor=COLORS[s], hatch="//" if never else None)
                label = "never" if never else f"{v:.0f}s" if v >= 10 else f"{v:.1f}s"
                if frac < 1 and not never:
                    label += f"\n{frac:.0%}"
                ax.annotate(label, (xpos, height), ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(SETTINGS)), [SETTING_SHORT[s] for s in SETTINGS], fontsize=7)
        ax.set_title(mname, fontsize=9)
        ax.set_ylim(lo, top * 1.6)
    axes[0][0].set_ylabel("median s to first reply (log)")
    from matplotlib.patches import Patch
    handles = [Patch(color="dimgrey", label="bug"), Patch(color="dimgrey", alpha=0.5, label="question")]
    if any(_never(float(r["median_sec"]), int(r["n"])) for _, r in tab.iterrows()):
        handles.append(Patch(facecolor="white", edgecolor="dimgrey", hatch="//", label="never replied"))
    fig.suptitle("L2: median time to first reply", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    _save(fig, out / "question_vs_bug.png", tight=False)
    return tab


FIGURES = [response_time, bugs_fixed, eta, response_cdf, orchestrator_time, question_vs_bug]


def _save(fig, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    pyplot().close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=named_dirs(1), action="append", required=True, metavar="NAME=DIR",
                   help="a model's levels-sweep analysis dir; repeat per model, in plotting order")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    models = dict(a.model)
    tables = load(models)
    a.out.mkdir(parents=True, exist_ok=True)
    for fig in FIGURES:
        fig(tables, list(models), a.out).to_csv(a.out / f"{fig.__name__}.csv", index=False)
        print(f"[OK] {a.out / fig.__name__}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
