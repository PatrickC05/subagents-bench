"""Gantt timelines of representative episodes (``docs/DESIGN.md`` section 7, figure 11).

For every condition at most two episodes are drawn: the one with the median
wall-clock and the fastest solved one (if it is a different episode). Each
figure has one lane per agent -- the orchestrator, then each subagent in
start order, then each arbiter judge -- built only from ``calls.csv`` and
``tools.csv`` (plus ``spawns.csv``/``dispatches.csv`` when present):

* an LLM call is a bar split at its streaming instants: retry/backoff time
  (``t_start`` -> ``t_request_sent``, hatched grey), waiting for the first
  token (``t_request_sent`` -> ``t_first_chunk``, light role colour) and
  decoding (``t_first_chunk`` -> ``t_end``, full role colour); a call without
  streaming instants is one full-colour bar;
* a tool call is a thinner green bar in its agent's lane, with a red x when
  the command timed out;
* a ``spawn_subagents`` call is an outlined bar in the orchestrator lane, and
  its group is bracketed by a box (strategy colour) around the lanes of its
  members;
* orchestrator steps are marked on the top axis.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Iterator

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .data import AnalysisData, Skip, as_bool, num
from .figures import Rendered
from .style import (
    INK,
    INK2,
    MUTED,
    RETRY_COLOR,
    ROLE_COLORS,
    STRATEGY_COLORS,
    SURFACE,
    TOOL_COLOR,
    new_figure,
    outside_legend,
    tint,
)

log = logging.getLogger("subagents_bench.plots")

ROLE_KEYS = {"orchestrator": "orch", "subagent": "sub", "arbiter": "arb", "debate": "deb"}
ROLE_LABELS = {"orch": "orchestrator LLM", "sub": "subagent LLM", "arb": "arbiter judge LLM", "deb": "debate revision LLM"}
FIG_WIDTH = 11.0
LABEL_IN = 1.5  # inches reserved for lane labels (for text-fit estimates)

TIMELINES_INTRO = (
    "Gantt charts of representative episodes: for each condition the episode with the median wall-clock and, if "
    "different, the fastest solved episode (at most two per condition). Lanes are agents: the orchestrator on top, "
    "then each subagent in start order, then each arbiter judge. Every LLM call is split into hatched grey "
    "retry/backoff time (if any), a light part waiting for the first streamed token (TTFT: queueing + prefill) and a "
    "dark part decoding the rest; colours are by role (violet orchestrator, yellow subagent, red arbiter judge, "
    "magenta debate revision). Thinner green bars are tool calls (a red x marks a command that timed out). An "
    "outlined bar in the orchestrator lane is a spawn_subagents call, and the box in the strategy colour brackets "
    "the lanes of the subagents that group ran, so concurrency (parallel, arbiter) and serial chains (sequential) "
    "are visible directly. Ticks on the top axis mark orchestrator steps. Data: `calls.csv` (episode_id, agent_id, "
    "role, step, t_start, t_request_sent, t_first_chunk, t_end, group_id), `tools.csv` (agent_id, name, t_start, "
    "t_end, timed_out, group_id), `spawns.csv` (group_id, strategy, requested, granted, t_start, t_end), "
    "`dispatches.csv` (agent_id, candidate_index, status) and `episodes.csv` (wall_clock_sec, resolved) for the "
    "selection."
)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-") or "x"


def select_episodes(data: AnalysisData, eps: pd.DataFrame, have_events: set[str]) -> list[tuple[str, str, pd.Series]]:
    """(condition, kind, episode row): median wall-clock + fastest solved, max 2 per condition."""
    out = []
    for cond in data.order(eps["condition"].unique()):
        g = eps[(eps["condition"] == cond) & eps["episode_id"].astype(str).isin(have_events)]
        g = g[np.isfinite(g["_wall"])]
        if g.empty:
            continue
        s = g.sort_values(["_wall", "episode_id"])
        med = s.iloc[(len(s) - 1) // 2]
        out.append((cond, "median wall-clock", med))
        solved = s[s["resolved"] == 1.0]
        if len(solved) and solved.iloc[0]["episode_id"] != med["episode_id"]:
            out.append((cond, "fastest solved", solved.iloc[0]))
    return out


def _lane_order(agents: pd.DataFrame) -> list[str]:
    """orchestrator, then sub-* by first start, then arbiter-*, then anything else."""
    first = agents.groupby("agent_id")["t_start"].min()

    def key(a: str) -> tuple:
        group = 0 if a == "orchestrator" else 1 if a.startswith("sub") else 2 if a.startswith("arbiter") else 3
        return (group, first.get(a, math.inf), a)

    return sorted(first.index.astype(str), key=key)


def _fits(text: str, width_sec: float, sec_per_pt: float, fontsize: float = 6.3) -> bool:
    return width_sec >= (0.58 * fontsize * len(text) + 6) * sec_per_pt


def _draw(data: AnalysisData, cond: str, kind: str, ep: pd.Series, calls: pd.DataFrame, tools: pd.DataFrame | None,
          spawns: pd.DataFrame | None, disp: pd.DataFrame | None) -> Rendered:
    eid = str(ep["episode_id"])
    ev = [calls[["agent_id", "t_start", "t_end"]]]
    if tools is not None:
        ev.append(tools[["agent_id", "t_start", "t_end"]])
    events = pd.concat(ev)
    events["agent_id"] = events["agent_id"].astype(str)
    t0 = float(events["t_start"].min())
    t_max = float(events["t_end"].max()) - t0
    lanes = _lane_order(events)
    lane = {a: i for i, a in enumerate(lanes)}
    fig = new_figure(FIG_WIDTH, 1.9 + 0.27 * len(lanes) + 0.5)
    ax = fig.subplots()
    sec_per_pt = max(t_max, 1e-6) * 1.02 / ((FIG_WIDTH - LABEL_IN - 0.4) * 72)
    roles_seen: list[str] = []
    batches: dict[tuple, list[tuple[float, float, float]]] = {}

    def add(y: float, a: float, b: float, color, height: float = 0.64, hatch: str | None = None) -> None:
        if b > a:
            batches.setdefault((color, height, hatch), []).append((y, a - t0, b - a))

    for _, c in calls.iterrows():
        y = lane[str(c["agent_id"])]
        rk = ROLE_KEYS.get(str(c.get("role")), "orch")
        if rk not in roles_seen:
            roles_seen.append(rk)
        col = ROLE_COLORS[rk]
        ts, te = float(c["t_start"]), float(c["t_end"])
        tr = float(c["t_request_sent"]) if "t_request_sent" in c and pd.notna(c.get("t_request_sent")) else ts
        tr = min(max(tr, ts), te)
        tf = float(c["t_first_chunk"]) if "t_first_chunk" in c and pd.notna(c.get("t_first_chunk")) else math.nan
        if tr - ts > 1e-3:
            add(y, ts, tr, tint(RETRY_COLOR, 0.55), hatch="////")
        if np.isfinite(tf) and tr <= tf <= te:
            add(y, tr, tf, tint(col, 0.55))
            add(y, tf, te, col)
        else:
            add(y, tr, te, col)
    timeouts = []
    spawn_bars = []
    if tools is not None:
        for _, t in tools.iterrows():
            y = lane[str(t["agent_id"])]
            ts, te = float(t["t_start"]), float(t["t_end"])
            if str(t.get("name")) == "spawn_subagents":
                spawn_bars.append((y, ts, te, t.get("group_id")))
                continue
            add(y, ts, te, TOOL_COLOR, height=0.34)
            if "timed_out" in t and as_bool(pd.Series([t["timed_out"]])).iloc[0] == 1.0:
                timeouts.append((te - t0, y))
    for (color, height, hatch), segs in batches.items():
        ys, lefts, widths = zip(*segs)
        ax.barh(ys, widths, left=lefts, height=height, color=color, edgecolor=SURFACE if not hatch else MUTED,
                linewidth=0.3 if not hatch else 0.0, hatch=hatch, zorder=3)
    if timeouts:
        xs, ys = zip(*timeouts)
        ax.plot(xs, ys, linestyle="none", marker="x", color="#d03b3b", markersize=6, markeredgewidth=1.6, zorder=5)
    # Spawn groups: members are the non-orchestrator agents whose events carry the group id.
    group_info: dict[str, dict] = {}
    if spawns is not None and "group_id" in spawns.columns:
        for _, s in spawns.iterrows():
            group_info[str(s["group_id"])] = s.to_dict()
    gcol = pd.concat([calls[["agent_id", "group_id"]]] + ([tools[["agent_id", "group_id"]]] if tools is not None
                                                          and "group_id" in tools.columns else []))
    gcol = gcol[gcol["group_id"].notna() & (gcol["agent_id"].astype(str) != "orchestrator")] \
        if "group_id" in gcol.columns else gcol.iloc[0:0]
    strategies_seen: list[str] = []
    for k, (y, ts, te, gid) in enumerate(spawn_bars):
        gid = str(gid) if pd.notna(gid) else None
        info = group_info.get(gid or "", {})
        strategy = info.get("strategy")
        if not isinstance(strategy, str) or not strategy:
            strategy = data.meta[cond].strategy if cond in data.meta else None
        strategy = strategy or "group"
        sc = STRATEGY_COLORS.get(strategy, INK2)
        if strategy in STRATEGY_COLORS and strategy not in strategies_seen:
            strategies_seen.append(strategy)
        ax.barh(y, te - ts, left=ts - t0, height=0.34, facecolor=tint(sc, 0.85), edgecolor=sc, linewidth=0.9, zorder=3)
        members = sorted({lane[str(a)] for a in gcol.loc[gcol["group_id"].astype(str) == gid, "agent_id"]
                          if str(a) in lane}) if gid else []
        gs = float(info.get("t_start", ts)) if pd.notna(info.get("t_start", np.nan)) else ts
        ge = float(info.get("t_end", te)) if pd.notna(info.get("t_end", np.nan)) else te
        if members:
            lo_, hi_ = min(members) - 0.46, max(members) + 0.46
            ax.add_patch(Rectangle((gs - t0, lo_), ge - gs, hi_ - lo_, facecolor=tint(sc, 0.93), edgecolor=sc,
                                   linewidth=0.9, zorder=1))
            ax.plot([gs - t0, gs - t0], [y + 0.17, lo_], color=sc, linewidth=0.8, linestyle=":", zorder=2)
        req, gr = info.get("requested"), info.get("granted")
        label = f"{gid or 'group'}: {strategy}" + (f", {int(gr)}/{int(req)} granted" if pd.notna(req) and pd.notna(gr) else "")
        short = f"{strategy[:3]} {int(gr)}/{int(req)}" if pd.notna(req) and pd.notna(gr) else strategy[:3]
        for text in (label, short):
            if _fits(text, te - ts, sec_per_pt):
                ax.text((ts + te) / 2 - t0, y, text, ha="center", va="center", fontsize=6.3, color=INK, zorder=4)
                break
    # Lane labels, with candidate index and non-submitted status for subagents.
    status: dict[str, str] = {}
    cand: dict[str, str] = {}
    if disp is not None and "agent_id" in disp.columns:
        for _, d in disp.iterrows():
            a = str(d["agent_id"])
            if "status" in d and pd.notna(d.get("status")) and str(d["status"]) != "submitted":
                status[a] = str(d["status"])
            if "candidate_index" in d and pd.notna(d.get("candidate_index")):
                cand[a] = f"cand {int(float(d['candidate_index']))}"
    labels = []
    for a in lanes:
        extra = ", ".join(x for x in (cand.get(a), status.get(a)) if x)
        name = a if len(a) <= 18 else a[:17] + "~"
        labels.append(f"{name} ({extra})" if extra else name)
    ax.set_yticks(range(len(lanes)), labels)
    ax.set_ylim(len(lanes) - 0.4, -0.6)
    ax.set_xlim(0, t_max * 1.02 if t_max > 0 else 1)
    ax.yaxis.grid(False)
    ax.set_xlabel("seconds since the episode's first event")
    # Orchestrator steps on the top axis.
    oc = calls[calls["agent_id"].astype(str) == "orchestrator"]
    if len(oc) and "step" in oc.columns:
        st = oc.groupby(num(oc["step"]))["t_start"].min().sort_index()
        every = max(1, math.ceil(len(st) / 14))
        top = ax.secondary_xaxis("top")
        top.set_xticks((st.to_numpy(float) - t0)[::every], [f"s{int(s)}" for s in st.index[::every]])
        top.tick_params(labelsize=6.5, colors=INK2, length=3)
        top.spines["top"].set_visible(False)
    resolved = ep.get("resolved")
    outcome = "resolved" if resolved == 1.0 else "not resolved" if resolved == 0.0 else "outcome unknown"
    wall = ep.get("_wall")
    task = ep.get("task_name", "?")
    rnd = ep.get("round")
    rtxt = f"round {int(float(rnd))}" if rnd is not None and pd.notna(rnd) else "round ?"
    ax.set_title(f"{cond}  |  {task}  |  {rtxt}  |  {kind}\n"
                 f"{outcome}, wall-clock {wall:.0f} s, {len(lanes)} lanes, episode {eid}", fontsize=9)
    handles = [Patch(facecolor=ROLE_COLORS[r], edgecolor=ROLE_COLORS[r], label=ROLE_LABELS[r]) for r in
               ("orch", "sub", "arb", "deb") if r in roles_seen]
    handles += [Patch(facecolor=tint(INK2, 0.6), edgecolor=tint(INK2, 0.6), label="light part: waiting for first token"),
                Patch(facecolor=INK2, edgecolor=INK2, label="dark part: decoding"),
                Patch(facecolor=TOOL_COLOR, edgecolor=TOOL_COLOR, label="tool call")]
    if any(h == "////" for (_, _, h) in batches):
        handles.append(Patch(facecolor=tint(RETRY_COLOR, 0.55), edgecolor=MUTED, hatch="////", label="retry / backoff"))
    if timeouts:
        handles.append(Line2D([], [], linestyle="none", marker="x", color="#d03b3b", label="command timed out"))
    for s in strategies_seen:
        handles.append(Patch(facecolor=tint(STRATEGY_COLORS[s], 0.9), edgecolor=STRATEGY_COLORS[s],
                             label=f"spawn group ({s})"))
    outside_legend(fig, handles, ncols=min(len(handles), 5))
    stem = f"timelines/{_safe(cond)}__{_safe(task)}__{_safe(rtxt.replace('round ', 'r'))}__" \
           f"{'median' if kind.startswith('median') else 'fastest-solved'}"
    caption = (
        f"{cond} on {task} ({rtxt}), selected as the condition's {kind} episode: {outcome}, wall-clock "
        f"{wall:.0f} s, {len(calls)} LLM calls and {0 if tools is None else len(tools)} tool calls over "
        f"{len(lanes)} lanes."
    )
    return Rendered(stem, fig, f"{cond} | {task} | {rtxt} | {kind}", caption, [], [])


def fig_timelines(data: AnalysisData) -> Iterator[Rendered]:
    calls = data.require("calls", ["episode_id", "agent_id", "t_start", "t_end"])
    tools = data.table("tools")
    if tools is not None and not {"episode_id", "agent_id", "t_start", "t_end"} <= set(tools.columns):
        log.info("timelines: tools.csv lacks episode_id/agent_id/t_start/t_end; tools not drawn")
        tools = None
    spawns = data.table("spawns")
    disp = data.table("dispatches")
    eps = data.episodes(["episode_id"])
    eps = eps.assign(episode_id=eps["episode_id"].astype(str))
    by_calls = {str(k): g for k, g in calls.groupby(calls["episode_id"].astype(str))}
    if "wall_clock_sec" in eps.columns:
        eps["_wall"] = num(eps["wall_clock_sec"])
    else:
        span = calls.groupby(calls["episode_id"].astype(str)).agg(a=("t_start", "min"), b=("t_end", "max"))
        eps["_wall"] = eps["episode_id"].map(span["b"] - span["a"])
    picks = select_episodes(data, eps, set(by_calls))
    if not picks:
        raise Skip("no valid episode has events in calls.csv")

    def per_ep(df: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
        if df is None or "episode_id" not in df.columns:
            return {}
        return {str(k): g for k, g in df.groupby(df["episode_id"].astype(str))}

    by_tools, by_spawns, by_disp = per_ep(tools), per_ep(spawns), per_ep(disp)
    for cond, kind, ep in picks:
        eid = str(ep["episode_id"])
        yield _draw(data, cond, kind, ep, by_calls[eid], by_tools.get(eid), by_spawns.get(eid), by_disp.get(eid))
