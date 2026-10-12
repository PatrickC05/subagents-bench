"""The eight figures, drawn from the analysis tables of valid trials.

Each figure function takes the tables and a PNG path, saves the figure and returns its
(title, caption) for ``index.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..analysis.aggregate import condition_order
from ..analysis.latency import CATEGORY
from ..analysis.stats import wilson

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STRATEGY_COLOR = {"parallel": SERIES[0], "sequential": SERIES[1], "arbiter": SERIES[2]}
SCOPE_MARKER = {"easy": "o", "medium": "s", "hard": "^"}
PARTS = {"orch_llm": ("orchestrator LLM", SERIES[0]), "sub_llm": ("subagent LLM", SERIES[1]),
         "judge_llm": ("judge + debate LLM", SERIES[2]), "tool": ("tools", SERIES[3])}
STATUS = {"submitted": "#0ca30c", "max_steps": "#fab219", "no_tool_call": "#ec835a", "error": "#d03b3b",
          "cancelled": AXIS}
Tables = dict[str, pd.DataFrame]

plt.rcParams.update({
    "font.size": 8.5, "font.family": "sans-serif", "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
    "ytick.major.size": 0,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 160,
})


def _color(row: pd.Series) -> str:
    return STRATEGY_COLOR.get(row["strategy"], MUTED)


def _control(row: pd.Series) -> bool:
    return condition_order(row)[0] == 2


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["valid"].astype(bool)] if len(df) else df


def _rows(ax: plt.Axes, conds: pd.DataFrame) -> np.ndarray:
    """Conditions as rows, top to bottom, with a rule between baselines, the grid and controls."""
    y = np.arange(len(conds))
    ax.set_yticks(y, conds["condition"])
    ax.set_ylim(len(conds) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    groups = [condition_order(r)[0] for _, r in conds.iterrows()]
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color=AXIS, linewidth=0.8)
    return y


def _strategy_legend() -> list:
    return [Patch(color=c, label=s) for s, c in STRATEGY_COLOR.items()] + [Patch(color=MUTED, label="single agent")]


def _empty(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=INK2)
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _height(n: int) -> float:
    return 1.6 + 0.32 * max(n, 1)


def main_grid(t: Tables, path: Path) -> tuple[str, str]:
    c = t["conditions"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, _height(len(c))), sharey=True,
                               gridspec_kw={"width_ratios": [3, 1.3], "wspace": 0.08})
    y = _rows(a, c)
    colors = [_color(r) for _, r in c.iterrows()]
    rate = c["resolve_rate"]
    a.barh(y, rate.fillna(0), color=colors, height=0.62)
    a.errorbar(rate, y, xerr=[rate - c["resolve_ci_lo"], c["resolve_ci_hi"] - rate], fmt="none", ecolor=INK,
               elinewidth=0.9, capsize=2)
    for yi, (_, r) in zip(y, c.iterrows()):
        a.text(1.02, yi, f"{r['n_resolved']}/{r['n_valid']}" if r["n_valid"] else "no valid trials",
               va="center", color=INK2, fontsize=7.5)
        if r["topology"] == "single":
            b.text(0.02, yi, "n/a (single agent)", va="center", color=MUTED, fontsize=7.5)
        elif r["n_valid"]:
            b.barh(yi, r["delegated_frac"], color=colors[yi], height=0.62)
            label = f"{r['n_delegated']}/{r['n_valid']}" + ("  never delegated" if r["n_delegated"] == 0 else "")
            b.text(r["delegated_frac"] + 0.03, yi, label, va="center", color=INK if r["n_delegated"] == 0 else INK2,
                   fontsize=7.5, fontweight="bold" if r["n_delegated"] == 0 else "normal")
    a.set_xlim(0, 1.12)
    a.set_xlabel("resolve rate (share of valid trials, Wilson 95% CI)")
    b.set_xlim(0, 1.6)
    b.set_xticks([0, 0.5, 1])
    b.set_xlabel("delegated (share of valid episodes)")
    a.set_title("Resolve rate per condition")
    b.set_title("Delegation uptake")
    b.legend(handles=_strategy_legend(), loc="upper left", bbox_to_anchor=(1.01, 1))
    _save(fig, path)
    return ("Resolve rate and delegation per condition",
            "Share of valid trials the verifier solved (k/n, whiskers: Wilson 95% interval), with baselines on top, "
            "the strategy x scope grid in the middle and the controls (same model, then manager) below. The right "
            "panel is the share of valid episodes in which the orchestrator spawned subagents at least once; a "
            "subagent condition that never delegated measured only its orchestrator.")


def _frontier(xs: np.ndarray, ys: np.ndarray) -> tuple[list[float], list[float]]:
    """Points no other point beats on both axes (lower x, higher y), as a staircase."""
    fx, fy, best = [], [], -1.0
    for x, y in sorted(zip(xs, ys), key=lambda p: (p[0], -p[1])):
        if y > best:
            fx += [x, x] if fx else [x]
            fy += [best, y] if fy else [y]
            best = y
    return fx, fy


def _labels(ax: plt.Axes, pts: pd.DataFrame, xcol: str) -> None:
    """Condition names beside the points, stacked downwards where they would overlap."""
    placed: list[tuple[float, float, float]] = []  # display x, y and width of each label
    ax.get_xlim()  # settles the autoscaled limits that transData uses
    for _, r in pts.sort_values(["resolve_rate", xcol], ascending=[False, True]).iterrows():
        px, py = ax.transData.transform((r[xcol], r["resolve_rate"]))
        w, shift = 5.0 * len(r["condition"]), 0
        while any(px < qx + qw and qx < px + w and abs(py - shift - qy) < 9 for qx, qy, qw in placed):
            shift += 9
        placed.append((px, py - shift, w))
        ax.annotate(r["condition"], (r[xcol], r["resolve_rate"]), xytext=(5, 3 - shift), textcoords="offset points",
                    fontsize=6.5, color=INK2)


def _pareto(t: Tables, path: Path, metric: str, xlabel: str, title: str) -> None:
    c = t["conditions"]
    pts = c[c[f"{metric}_mean"].notna() & c["resolve_rate"].notna()]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if pts.empty:
        _empty(ax, f"no condition has a known mean {xlabel}")
        _save(fig, path)
        return
    for _, r in pts.iterrows():
        x, y, color, control = r[f"{metric}_mean"], r["resolve_rate"], _color(r), _control(r)
        xerr = [[x - r[f"{metric}_ci_lo"]], [r[f"{metric}_ci_hi"] - x]] if pd.notna(r[f"{metric}_ci_lo"]) else None
        ax.errorbar(x, y, xerr=xerr, yerr=[[y - r["resolve_ci_lo"]], [r["resolve_ci_hi"] - y]], fmt="none",
                    ecolor=AXIS, elinewidth=0.8, zorder=1)
        marker = SCOPE_MARKER.get(r["scope"], "*") if r["topology"] == "subagents" else "*"
        ax.scatter(x, y, marker=marker, s=90 if marker == "*" else 48, zorder=3, linewidth=1.4 if control else 1,
                   facecolor=SURFACE if control else color, edgecolor=color if control else SURFACE)
    fx, fy = _frontier(pts[f"{metric}_mean"].to_numpy(), pts["resolve_rate"].to_numpy())
    ax.plot(fx, fy, color=INK2, linewidth=0.9, zorder=2)
    xs = pts[f"{metric}_mean"]
    if xs.min() > 0 and xs.max() / xs.min() > 20:
        ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    _labels(ax, pts, f"{metric}_mean")
    ax.set_xlabel(f"{xlabel} (mean per episode, bootstrap 95% CI)")
    ax.set_ylabel("resolve rate (Wilson 95% CI)")
    ax.set_title(title)
    gray = {"color": MUTED, "linestyle": "none"}
    handles = [Line2D([], [], marker="o", color=c, linestyle="none", label=s) for s, c in STRATEGY_COLOR.items()]
    handles += [Line2D([], [], marker=m, label=f"{s} scope", **gray) for s, m in SCOPE_MARKER.items()]
    handles += [Line2D([], [], marker="*", markersize=9, label="single agent", **gray),
                Line2D([], [], marker="o", markerfacecolor=SURFACE, label="control (manager, same model)", **gray),
                Line2D([], [], color=INK2, linewidth=0.9, label="Pareto frontier")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7.5)
    _save(fig, path)


def pareto_cost(t: Tables, path: Path) -> tuple[str, str]:
    _pareto(t, path, "cost_total", "cost in USD", "Resolve rate vs cost")
    return ("Resolve rate vs cost per episode",
            "Each point is a condition: mean cost of an episode (all roles, offline prices from configs/pricing.yaml) "
            "against its resolve rate; colour is the strategy, shape the scope, hollow marks the controls. The line "
            "links the conditions no other condition beats on both cost and resolve rate.")


def pareto_time(t: Tables, path: Path) -> tuple[str, str]:
    _pareto(t, path, "wall_clock_sec", "wall clock in s", "Resolve rate vs wall clock")
    return ("Resolve rate vs wall clock per episode",
            "As pareto_cost, with the mean measured wall clock of the agent's episode on the x axis. Up and to the "
            "left is better.")


def time_breakdown(t: Tables, path: Path) -> tuple[str, str]:
    c, eps = t["conditions"], _valid(t["episodes"])
    fig, ax = plt.subplots(figsize=(9, _height(len(c))))
    y = _rows(ax, c)
    for yi, cond in zip(y, c["condition"]):
        e = eps[eps["condition"] == cond]
        if e.empty:
            continue
        left = 0.0
        for col, color in [*((f"cp_{p}_sec", col) for p, (_, col) in PARTS.items()), ("overhead_sec", AXIS)]:
            w = e[col].mean()
            ax.barh(yi, w, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=1)
            left += w
        ax.text(left, yi, f"  {left:.3g} s", va="center", color=INK2, fontsize=7.5)
    ax.set_xlabel("seconds per episode (mean over valid trials)")
    ax.set_title("Where the wall clock goes: the measured critical path")
    handles = [Patch(color=col, label=lab) for lab, col in PARTS.values()]
    ax.legend(handles=handles + [Patch(color=AXIS, label="overhead (wall clock - critical path)")],
              loc="upper left", bbox_to_anchor=(1.01, 1))
    _save(fig, path)
    return ("Critical-path time breakdown",
            "Mean wall clock per condition split along the measured critical path: orchestrator steps in series; a "
            "parallel spawn counts its slowest subagent, a sequential one all of them, an arbiter its slowest "
            "candidate plus the judge and debate calls and the copy/apply steps (counted as tool time). Overhead is "
            "wall clock minus critical path (harness bookkeeping not attributed to any call or tool).")


def token_composition(t: Tables, path: Path) -> tuple[str, str]:
    c, eps = t["conditions"], _valid(t["episodes"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, _height(len(c))), sharey=True, gridspec_kw={"wspace": 0.08})
    y = _rows(a, c)
    roles = {"orch_llm": ["orch"], "sub_llm": ["sub"], "judge_llm": ["arb", "deb"]}
    for yi, cond in zip(y, c["condition"]):
        e = eps[eps["condition"] == cond]
        if e.empty:
            continue
        for ax, value in ((a, lambda s: e[f"{s}_prompt_tokens"] + e[f"{s}_completion_tokens"]),
                          (b, lambda s: e[f"cost_{s}"])):
            left = 0.0
            for part, shorts in roles.items():
                w = sum(value(s) for s in shorts).mean()
                ax.barh(yi, w, left=left, color=PARTS[part][1], height=0.62, edgecolor=SURFACE, linewidth=1)
                left += w if pd.notna(w) else 0.0
            if ax is b and pd.isna(e["cost_total"]).any():
                ax.text(0, yi, " unpriced model", va="center", color=MUTED, fontsize=7.5)
    a.set_xlabel("tokens per episode (prompt incl. cached + completion; mean)")
    b.set_xlim(left=0)
    b.set_xlabel("USD per episode (mean)")
    a.set_title("Tokens by role")
    b.set_title("Cost by role")
    b.legend(handles=[Patch(color=PARTS[p][1], label=PARTS[p][0].replace(" LLM", "")) for p in roles],
             loc="upper left", bbox_to_anchor=(1.01, 1))
    _save(fig, path)
    return ("Tokens and cost by role",
            "Mean tokens (left) and cost (right) of a valid episode per condition, split into the orchestrator, the "
            "subagents and the arbiter's judge plus debate calls. Prompt tokens include cached ones, which are "
            "priced lower; a model without a price leaves the cost unknown.")


def delegation(t: Tables, path: Path) -> tuple[str, str]:
    c = t["conditions"][t["conditions"]["topology"] == "subagents"].reset_index(drop=True)
    eps, ds = _valid(t["episodes"]), _valid(t["dispatches"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, _height(len(c))), sharey=True, gridspec_kw={"wspace": 0.08})
    if c.empty:
        _empty(a, "no subagent conditions")
        _save(fig, path)
        return ("Delegation", "No subagent conditions in these jobs.")
    y = _rows(a, c)
    for yi, (_, r) in zip(y, c.iterrows()):
        e = eps[eps["condition"] == r["condition"]]
        if e.empty:
            continue
        spawns = e["spawns"].mean()
        a.barh(yi, spawns, color=_color(r), height=0.62)
        a.text(spawns, yi, f"  {spawns:.2f} ({e['dispatches'].mean():.1f} dispatches)", va="center", color=INK2,
               fontsize=7.5)
        d = ds[ds["condition"] == r["condition"]] if len(ds) else ds
        left = 0
        for status, color in STATUS.items():
            n = int((d["status"] == status).sum()) if len(d) else 0
            b.barh(yi, n, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=1)
            left += n
    a.set_xlim(0, 1.7 * max(eps["spawns"].max() if len(eps) else 0, 1))
    a.set_xlabel("spawn_subagents calls per valid episode (mean)")
    b.set_xlabel("subagent dispatches of valid episodes, by how they ended")
    a.set_title("Spawns")
    b.set_title("Dispatch outcomes")
    b.legend(handles=[Patch(color=col, label=s) for s, col in STATUS.items()], loc="upper left",
             bbox_to_anchor=(1.01, 1))
    _save(fig, path)
    return ("Spawns and dispatch outcomes",
            "Left: mean spawn_subagents calls per valid episode of each subagent condition (with mean dispatches; an "
            "arbiter instruction runs one dispatch per candidate). Right: all dispatches by how they ended: "
            "submitted a report, ran out of steps, answered without a tool call, errored, or were cancelled.")


def arbiter(t: Tables, path: Path) -> tuple[str, str]:
    caption = ("Arbiter conditions: the judge's final confidence per arbitration (dots, median tick), the share of "
               "arbitrations that went to a debate round (unless the candidates agreed with confidence >= 0.8), and "
               "the share of copy-isolated arbitrations in which a candidate changed the real workdir (restored "
               "before applying the winner); k/n labels, Wilson 95% whiskers.")
    arbs = _valid(t["arbitrations"])
    c = t["conditions"][t["conditions"]["strategy"] == "arbiter"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, _height(len(c))), sharey=True, gridspec_kw={"wspace": 0.08})
    if arbs.empty or c.empty:
        for ax in axes:
            _empty(ax, "no arbitrations in valid episodes")
        _save(fig, path)
        return "Arbiter: judge confidence, debates, isolation", caption
    y = _rows(axes[0], c)
    rng = np.random.default_rng(0)
    for yi, (_, r) in zip(y, c.iterrows()):
        g = arbs[arbs["condition"] == r["condition"]]
        if g.empty:
            axes[1].text(0.02, yi, "no arbitrations", va="center", color=MUTED, fontsize=7.5)
            continue
        conf = g["final_confidence"].dropna()
        axes[0].scatter(conf, yi + rng.uniform(-0.18, 0.18, len(conf)), s=14, color=_color(r), zorder=3)
        if len(conf):
            axes[0].plot([conf.median()] * 2, [yi - 0.3, yi + 0.3], color=INK, linewidth=1.5)
        checked = g[(g["isolation"] == "copy") & g["isolation_violated"].notna()]
        for ax, k, n, note in ((axes[1], int((g["debate_rounds"] > 0).sum()), len(g), ""),
                               (axes[2], int(checked["isolation_violated"].eq(True).sum()), len(checked),
                                "shared workdir (no copies)")):
            if n == 0:
                ax.text(0.02, yi, note, va="center", color=MUTED, fontsize=7.5)
                continue
            lo, hi = wilson(k, n)
            ax.barh(yi, k / n, color=_color(r), height=0.62)
            ax.errorbar(k / n, yi, xerr=[[k / n - lo], [hi - k / n]], fmt="none", ecolor=INK, elinewidth=0.9,
                        capsize=2)
            ax.text(1.03, yi, f"{k}/{n}", va="center", color=INK2, fontsize=7.5)
    labels = ("final judge confidence", "share of arbitrations", "share of copy arbitrations")
    for ax, xlabel, title in zip(axes, labels, ("Judge confidence", "Debate rate", "Isolation violations")):
        ax.set_xlim(-0.02, 1.15)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    _save(fig, path)
    return "Arbiter: judge confidence, debates, isolation", caption


def _example(eps: pd.DataFrame, strategy: str) -> pd.Series | None:
    """The delegating valid episode of a strategy with the median wall clock."""
    e = eps[(eps["strategy"] == strategy) & (eps["delegated"] == 1)].sort_values("wall_clock_sec")
    return e.iloc[len(e) // 2] if len(e) else None


def _agent_key(agent: str) -> tuple:
    kind, _, n = agent.rpartition("-")
    return ({"": 0, "sub": 1}.get(kind, 2), int(n) if n.isdigit() else 0, agent)


def timeline(t: Tables, path: Path) -> tuple[str, str]:
    eps = _valid(t["episodes"])
    caption = ("Gantt chart of the LLM calls (tall bars, coloured by role) and tool calls (short bars) of each agent "
               "in one valid, delegating episode per strategy (the one with the median wall clock). Grey spans: the "
               "orchestrator's spawn_subagents calls waiting for the subagents, and each arbitration from its "
               "workdir copies to applying the winner.")
    if eps.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        _empty(ax, "no valid episodes")
        _save(fig, path)
        return "Timeline of one episode per strategy", caption
    examples = [e for s in STRATEGY_COLOR if (e := _example(eps, s)) is not None] or [eps.iloc[0]]
    calls, tools = t["calls"], t["tools"]
    agents = [sorted(set(calls.loc[calls["episode_id"] == e["episode_id"], "agent_id"]), key=_agent_key)
              for e in examples]
    fig, axes = plt.subplots(len(examples), 1, figsize=(10, sum(1.0 + 0.3 * len(a) for a in agents)), squeeze=False,
                             gridspec_kw={"height_ratios": [1.0 + 0.3 * len(a) for a in agents], "hspace": 0.55})
    for ax, e, names in zip(axes[:, 0], examples, agents):
        row = {a: i for i, a in enumerate(names)}
        mine_t = tools[(tools["episode_id"] == e["episode_id"]) & tools["agent_id"].isin(names)]
        for _, x in mine_t.iterrows():
            spawn = x["name"] == "spawn_subagents"
            ax.barh(row[x["agent_id"]], x["t_end"] - x["t_start"], left=x["t_start"] - e["t_start"],
                    height=0.8 if spawn else 0.36, color=GRID if spawn else PARTS["tool"][1], zorder=1 if spawn else 3)
        arbs = t["arbitrations"]
        for _, x in (arbs[arbs["episode_id"] == e["episode_id"]] if len(arbs) else arbs).iterrows():
            if f"arbiter-{x['arbitration_id']}" in row:  # copies, candidates, diffs, judging and apply
                ax.barh(row[f"arbiter-{x['arbitration_id']}"], x["t_end"] - x["t_start"], height=0.8,
                        left=x["t_start"] - e["t_start"], color=GRID, zorder=1)
        for _, x in calls[calls["episode_id"] == e["episode_id"]].iterrows():
            ax.barh(row[x["agent_id"]], x["t_end"] - x["t_start"], left=x["t_start"] - e["t_start"], height=0.62,
                    color=PARTS[CATEGORY[x["role"]]][1], zorder=2)
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, e["wall_clock_sec"] * 1.01)
        ax.set_title(f"{e['strategy'] or 'single agent'}: {e['condition']} on {e['task_name']} "
                     f"({'solved' if e['resolved'] == 1 else 'unsolved'}, {e['wall_clock_sec']:.3g} s)", loc="left")
    axes[-1, 0].set_xlabel("seconds since the episode started")
    handles = [Patch(color=col, label=lab) for lab, col in PARTS.values()]
    axes[0, 0].legend(handles=handles + [Patch(color=GRID, label="spawn / arbitration span")], loc="lower left",
                      bbox_to_anchor=(0, 1.12), ncol=5)
    _save(fig, path)
    return "Timeline of one episode per strategy", caption


FIGURES: dict[str, Callable[[Tables, Path], tuple[str, str]]] = {
    "main_grid": main_grid, "pareto_cost": pareto_cost, "pareto_time": pareto_time,
    "time_breakdown": time_breakdown, "token_composition": token_composition, "delegation": delegation,
    "arbiter": arbiter, "timeline": timeline,
}
