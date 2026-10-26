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
3. ``eta``: each scored estimate (``estimates.csv``): the minutes it gave against the minutes the
   fix actually took from then (replayed ``fixed_sec``); a bug never fixed is drawn at the session's
   end, as a lower bound. The title gives the on-time share (weighted, one weight per bug).
4. ``response_cdf``: share of bugs replied to within t, per setting, one panel per model. A curve
   that stays below 1 is bugs never replied to.
5. ``orchestrator_time``: the share of each session the orchestrator spends in model calls, in its
   own tool calls (``wait`` excluded), and neither (waiting for the user or subagents), per model
   and setting.
6. ``question_vs_bug``: median time to the first reply to a question (``reply_sec``) and to a bug
   (``user_reply_sec``) at L2, censored as in 1, with the share replied to above each bar.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.estimates import _flag, _valid
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
           "episodes": ["condition", "trial_name", "valid", "wall_clock_sec"]}
COLORS = {"background_hinted": "#3a6fb0", "manager_background": "#c4571f", "manager_owner": "#d9a634"}
WALL = 1800.0


def _read(d: Path, name: str, required: bool = True) -> pd.DataFrame:
    path = d / f"{name}.csv"
    if not path.is_file():
        if required:
            raise SystemExit(f"[FAIL] {path} not found: run scripts/analyze.sh on that sweep first")
        return pd.DataFrame(columns=COLUMNS[name])
    df = _valid(pd.read_csv(path, usecols=lambda c: c in COLUMNS[name], low_memory=False))
    sl = df["condition"].map(setting_level)
    df = df.assign(setting=sl.str[0], level=sl.str[1])
    return df[df["setting"].isin(SETTINGS) & df["level"].isin(LEVELS)]


def load(models: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Every model's tables, concatenated with a ``model`` column."""
    out: dict[str, list[pd.DataFrame]] = {}
    for model, d in models.items():
        for name, required in (("messages", True), ("estimates", False), ("calls", False), ("tools", False),
                               ("episodes", True)):
            out.setdefault(name, []).append(_read(d, name, required).assign(model=model))
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


def _bars(ax, table: pd.DataFrame, value: str, models: list[str], ylabel: str, cap: float | None = None) -> None:
    """Grouped bars: x = model x level, one bar per setting; inf drawn hatched at ``cap``."""
    width, x0 = 0.8 / len(SETTINGS), 0
    ticks, labels = [], []
    for m in models:
        for lv in LEVELS:
            for j, s in enumerate(SETTINGS):
                r = table[(table["model"] == m) & (table["level"] == lv) & (table["setting"] == s)]
                if r.empty:
                    continue
                v = float(r[value].iloc[0])
                never = cap is not None and _never(v, int(r["n"].iloc[0]))
                ax.bar(x0 + j * width, cap if never else v, width, color="white" if never else COLORS[s],
                       edgecolor=COLORS[s], hatch="//" if never else None)
            ticks.append(x0 + width * (len(SETTINGS) - 1) / 2)
            labels.append(lv)
            x0 += 1
        ax.text((ticks[-1] + ticks[-len(LEVELS)]) / 2, -0.12, m, transform=ax.get_xaxis_transform(), ha="center",
                fontsize=8, fontweight="bold")
        x0 += 0.5
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_ylabel(ylabel)
    _legend(ax)


def _legend(ax) -> None:
    ax.legend(handles=[_patch(s) for s in SETTINGS], fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=len(SETTINGS), frameon=False)


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
    _bars(ax, tab, "median_sec", models, "median s to first reply (hatched: never)", cap=WALL)
    fig.suptitle("Time to the first reply the user saw, bugs (L0 bugs, L1 + ETA, L2 + questions)", fontsize=9)
    _save(fig, out / "response_time.png")
    return tab


def bugs_fixed(t: dict[str, pd.DataFrame], models: list[str], out: Path) -> pd.DataFrame:
    m = t["messages"]
    bugs = m[m["kind"] == "bug"].assign(fixed=lambda d: _flag(d, "fixed"))  # no verdict: not fixed
    tab = bugs.groupby(["model", "setting", "level"])["fixed"].agg(
        fixed_frac="mean", n_fixed="sum", n="size").reset_index()
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(max(6, 3 * len(models)), 4))
    _bars(ax, tab, "fixed_frac", models, "share of bugs fixed")
    ax.set_ylim(0, 1)
    fig.suptitle("Bugs fixed at the end of the session", fontsize=9)
    _save(fig, out / "bugs_fixed.png")
    return tab


def eta(t: dict[str, pd.DataFrame], models: list[str], out: Path) -> pd.DataFrame:
    e = t["estimates"]
    if e.empty:
        return e
    e = e[_flag(e, "scored")].copy()
    start = e["ready_session_sec"] + e["est_sec"]  # seconds into the session the estimate was given
    fixed = pd.to_numeric(e["fixed_sec"], errors="coerce")
    e["predicted_min"] = e["eta_min"]
    e["actual_min"] = ((fixed - e["est_sec"]).where(fixed.notna(), WALL - start)) / 60
    e["never_fixed"] = fixed.isna()
    plt = pyplot()
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    for ax, mname in zip(axes[0], models):
        d = e[e["model"] == mname]
        for s in SETTINGS:
            x = d[d["setting"] == s]
            for never, marker in ((False, "o"), (True, "^")):
                y = x[x["never_fixed"] == never]
                ax.scatter(y["predicted_min"].clip(lower=0.1), y["actual_min"].clip(lower=0.1), s=14, marker=marker,
                           color=COLORS[s], alpha=0.7, facecolors="none" if never else None)
        lim = [0.1, 1e5]
        ax.plot(lim, lim, color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, max(100, float(d["predicted_min"].max() or 1) * 2) if len(d) else 100)
        ax.set_ylim(0.1, 100)
        w = d["weight"].fillna(0)
        share = float((_flag(d, "on_time") * w).sum() / w.sum()) if w.sum() else math.nan
        ax.set_title(f"{mname}: on time {share:.0%}" if not math.isnan(share) else mname, fontsize=9)
        ax.set_xlabel("estimate (min)")
    axes[0][0].set_ylabel("actual minutes to fix\n(triangles: never fixed, at session end)")
    fig.legend(handles=[_patch(s) for s in SETTINGS], fontsize=7, loc="lower center", ncol=len(SETTINGS), frameon=False)
    fig.subplots_adjust(bottom=0.2)
    _save(fig, out / "eta.png", tight=False)
    return e[["model", "setting", "level", "task_name", "subtask_id", "predicted_min", "actual_min", "never_fixed",
              "on_time", "weight"]]


def response_cdf(t: dict[str, pd.DataFrame], models: list[str], out: Path) -> pd.DataFrame:
    m = t["messages"]
    bugs = m[(m["kind"] == "bug") & (m["index"] > 0)]
    grid = np.arange(0, WALL + 1, 15.0)
    rows = []
    plt = pyplot()
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5), squeeze=False, sharey=True)
    for ax, mname in zip(axes[0], models):
        for s in SETTINGS:
            v = pd.to_numeric(bugs[(bugs["model"] == mname) & (bugs["setting"] == s)]["user_reply_sec"], errors="coerce")
            if v.empty:
                continue
            cdf = [(v <= g).sum() / len(v) for g in grid]
            ax.step(grid / 60, cdf, where="post", color=COLORS[s], label=SETTING_LABEL[s])
            rows += [{"model": mname, "setting": s, "t_sec": g, "replied_frac": c} for g, c in zip(grid, cdf)]
        ax.set_title(mname, fontsize=9)
        ax.set_xlabel("minutes after the bug was sent")
        ax.set_ylim(0, 1)
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
        ax.text(x - 1 - (len(SETTINGS) - 1) / 2, -0.1, mname, transform=ax.get_xaxis_transform(), ha="center",
                fontsize=8, fontweight="bold")
        x += 0.5
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_ylabel("share of the session")
    ax.set_ylim(0, 1)
    fig.suptitle("Orchestrator: in model calls (solid), own tools (light), neither (faint)", fontsize=9)
    _save(fig, out / "orchestrator_time.png")
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
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.8), squeeze=False, sharey=True)
    for ax, mname in zip(axes[0], models):
        for j, s in enumerate(SETTINGS):
            for k, (kind, alpha) in enumerate((("bug", 1.0), ("question", 0.5))):
                r = tab[(tab["model"] == mname) & (tab["setting"] == s) & (tab["kind"] == kind)]
                if r.empty:
                    continue
                v, frac = float(r["median_sec"].iloc[0]), float(r["replied_frac"].iloc[0])
                never = _never(v, int(r["n"].iloc[0]))
                xpos = j + (k - 0.5) * 0.38
                ax.bar(xpos, WALL / 60 if never else v / 60, 0.38, color="white" if never else COLORS[s], alpha=alpha,
                       edgecolor=COLORS[s], hatch="//" if never else None)
                ax.annotate(f"{frac:.0%}", (xpos, (WALL if never else v) / 60), ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(SETTINGS)), [SETTING_SHORT[s] for s in SETTINGS], fontsize=7)
        ax.set_title(f"{mname} (dark: bugs, light: questions)", fontsize=8)
    axes[0][0].set_ylabel("median minutes to first reply (L2; % replied)")
    _save(fig, out / "question_vs_bug.png")
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
