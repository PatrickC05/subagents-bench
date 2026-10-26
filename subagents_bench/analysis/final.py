"""Experiment 1 of the final study (issue #23): cost against bugs fixed at L0, each setting's shared
sessions next to the isolated baseline (each bug alone in a fresh session).

    python -m subagents_bench.analysis.final \\
        --model Opus=analysis/final_levels-claude-opus-5.5:analysis/final_isolated-claude-opus-5.5 \\
        --model Luna=... --out analysis/final_l0

Each ``--model NAME=LEVELS:ISOLATED`` is one model's two sweeps' tables (``scripts/analyze.sh``
writes them): ``episodes.csv`` for cost and the trial dirs, ``fixes.csv`` for each bug's verdict
and send time. The trial dirs (``trial_dir``, relative to ``--root``, default the current dir)
give the shared sessions' wall clock and the isolated sessions' replayed snapshots. Writes
``l0.csv`` (one row per model and point) and ``l0.png`` (dollars per bug, log scale, against the
share fixed: a color per model, a marker per point).

A session that failed counts as a failure, not as missing data: its bugs are not fixed, its cost
counts. Failed means ``invalid_reason`` ``provider_error`` (the provider failed past every retry)
or ``agent_crash`` (harbor failed once the agent had started, e.g. the sandbox ran out of memory
under the agents' own commands). Other invalid sessions (no episode, a grader or verifier failure,
a harbor exception before the agent ran) are left out.

Every point of a model covers the same bugs: the (task, subtask) pairs that every L0 setting and
the isolated sessions have a valid verdict for (``n_bugs``; ``n_dropped`` the bugs left out). The
points:

- a setting at L0 (``*_baseline``): fixed means passing at the end (``fixed_end``); a bug's cost is
  its session's cost over the bugs the session was sent;
- ``isolated``: the same bugs, each in its own session (``single_bug<k>``), fixed at its end;
- ``isolated_capped``: the same sessions, graded on the replayed state at the time the bug had in
  the shared sessions, their wall clock minus its send time there (``fixes.state_at``), as the
  shared sessions are graded at their end. Trials whose replay is unusable (``timing_ok`` false)
  are left out of this point only (``n_bugs`` is then smaller). Its cost is the whole session's,
  an upper bound on what the capped run would have cost.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .estimates import _flag
from .fixes import state_at
from .levels import setting_level
from .load import load_trial

ISOLATED = re.compile(r"single_bug(\d+)$")
FAILURES = ("provider_error", "agent_crash")  # invalid sessions that count as failures
BUG = ["task_name", "subtask_id"]


def _read(d: Path, name: str) -> pd.DataFrame:
    path = d / name
    if not path.is_file():
        raise SystemExit(f"[FAIL] {path} not found: run scripts/analyze.sh on that sweep first")
    return pd.read_csv(path, low_memory=False)


def counted(episodes: pd.DataFrame) -> tuple[set[str], set[str]]:
    """(trials the analysis counts, those of them that are failures): the valid trials and those
    that failed (FAILURES), which count as failures."""
    valid = _flag(episodes, "valid")
    failed = episodes["invalid_reason"].isin(FAILURES) if "invalid_reason" in episodes else valid & False
    return set(episodes.loc[valid | failed, "trial_name"]), set(episodes.loc[failed & ~valid, "trial_name"])


def _load(d: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(episodes, fixes) of the counted trials; a failed trial's bugs are not fixed."""
    ep, fx = _read(d, "episodes.csv"), _read(d, "fixes.csv")
    keep, failed = counted(ep)
    ep, fx = ep[ep["trial_name"].isin(keep)], fx[fx["trial_name"].isin(keep)].copy()
    fx["fixed"] = _flag(fx, "fixed_end") & ~fx["trial_name"].isin(failed)
    fx["failed"] = fx["trial_name"].isin(failed)
    return ep, fx


def _trial(root: Path, trial_dir: str):
    path = root / trial_dir
    if not path.is_dir():
        raise SystemExit(f"[FAIL] {path} not found: the analysis needs the sweep's jobs dir (run it from the repo root)")
    return load_trial(path)


def _row(point: str, fixed: pd.Series, cost_per_bug: pd.Series, n_tasks: int, n_dropped: int) -> dict:
    n = len(fixed)
    return {"point": point, "n_bugs": n, "n_fixed": int(fixed.sum()), "fixed_frac": fixed.mean() if n else float("nan"),
            "usd_per_bug": cost_per_bug.mean() if n else float("nan"), "usd_total": cost_per_bug.sum(),
            "usd_per_repo": cost_per_bug.sum() / n_tasks if n_tasks else float("nan"), "n_dropped": n_dropped}


def _bugs(d: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, d[BUG].to_numpy()))


def l0_table(levels: Path, isolated: Path, root: Path = Path(".")) -> pd.DataFrame:
    ep, fx = _load(levels)
    shared_ep = ep[ep["condition"].str.endswith("_baseline")]
    shared = fx.merge(shared_ep[["trial_name", "cost_total", "trial_dir"]], on="trial_name")
    shared["usd"] = shared["cost_total"] / shared.groupby("trial_name")["subtask_id"].transform("count")
    iep, ifx = _load(isolated)
    iso = ifx.merge(iep[["trial_name", "cost_total", "trial_dir"]], on="trial_name")
    bad = sorted(c for c in iso["condition"].unique() if not ISOLATED.search(c))
    if bad:
        raise SystemExit(f"[FAIL] {isolated}: conditions {bad} are not single_bug<k>")

    # the bugs every point has: a verdict in each L0 setting and in the isolated sessions
    sets = [_bugs(g) for _, g in shared.groupby("condition")] + [_bugs(iso)]
    common, n_all = set.intersection(*sets), len(set.union(*sets))
    shared = shared[[b in common for b in map(tuple, shared[BUG].to_numpy())]]
    iso = iso[[b in common for b in map(tuple, iso[BUG].to_numpy())]]
    n_tasks, dropped = len({t for t, _ in common}), n_all - len(common)
    rows = [_row(cond.removesuffix("_baseline"), g["fixed"], g["usd"], n_tasks, dropped)
            for cond, g in shared.groupby("condition")]
    rows.append(_row("isolated", iso["fixed"], iso["cost_total"], n_tasks, dropped))

    # each bug's time in the shared sessions: their configured wall clock minus its send time there
    walls = {d: float((_trial(root, d).episode or {})["config"]["interactive"]["wall_clock_sec"])
             for d in shared["trial_dir"].unique()}
    had = (shared.assign(had=shared["trial_dir"].map(walls) - shared["ready_session_sec"])
           .groupby(BUG)["had"].median())
    capped = []
    for _, r in iso.iterrows():
        if r["failed"]:  # a failed session: a failure at the cap too
            capped.append({"fixed": False, "usd": r["cost_total"]})
            continue
        status = state_at(_trial(root, r["trial_dir"]), float(had[(r["task_name"], r["subtask_id"])])).get(r["subtask_id"])
        if status is not None:  # {} when the replay is unusable: left out of this point
            capped.append({"fixed": status == "pass", "usd": r["cost_total"]})
    c = pd.DataFrame(capped, columns=["fixed", "usd"])
    rows.append(_row("isolated_capped", c["fixed"].astype(bool), c["usd"], n_tasks, dropped))
    return pd.DataFrame(rows)


POINT_LABEL = {"background_hinted": "same session: orch + sub", "manager_background": "same session: sub only",
               "manager_owner": "same session: owner", "isolated": "independent sessions"}  # legend names
TITLE = {"background_hinted": "Cost Performance Frontier", "manager_background": "Cost Performance Frontier: subagents only",
         "manager_owner": "Cost Performance Frontier: subagents with owner"}
SHORT = {"claude-opus-5.5": "Opus 5.5", "claude-sonnet-5.5": "Sonnet 5.5", "gpt-5.6-sol": "GPT-5.6 Sol",
         "gpt-5.6-terra": "GPT-5.6 Terra", "gpt-5.6-luna": "GPT-5.6 Luna", "kimi-k3": "Kimi K3",
         "deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash"}  # display names for the legend
TAG = {"claude-opus-5.5": "Opus", "claude-sonnet-5.5": "Sonnet", "gpt-5.6-sol": "Sol", "gpt-5.6-terra": "Terra",
       "gpt-5.6-luna": "Luna", "kimi-k3": "Kimi", "deepseek-v4.1-flash-1017": "DeepSeek"}  # and on the points


def levels_table(levels: Path) -> pd.DataFrame:
    """Cost against bugs fixed per setting and level (L0-L2) of a levels sweep, from its
    ``messages.csv`` (each bug's verdict) and ``episodes.csv`` (cost): one row per setting and level,
    on the bugs that every setting and level has (a session that failed counts, as in ``l0_table``).
    A bug's cost is its session's cost over the bugs the session was sent (questions excluded)."""
    ep, m = _read(levels, "episodes.csv"), _read(levels, "messages.csv")
    keep, failed = counted(ep)
    m = m[(m["kind"] == "bug") & m["trial_name"].isin(keep)].copy()
    m["fixed"] = _flag(m, "fixed") & ~m["trial_name"].isin(failed)
    m = m.merge(ep[["trial_name", "cost_total"]], on="trial_name")
    m["usd"] = m["cost_total"] / m.groupby("trial_name")["subtask_id"].transform("count")
    common = set.intersection(*[_bugs(g) for _, g in m.groupby("condition")])
    m = m[[b in common for b in map(tuple, m[BUG].to_numpy())]]
    n_tasks = len({t for t, _ in common})
    rows = []
    for cond, g in m.groupby("condition"):
        setting, level = setting_level(cond)
        if setting is not None:
            rows.append({**_row(setting, g["fixed"], g["usd"], n_tasks, 0), "level": level})
    return pd.DataFrame(rows)


def _place_labels(fig, ax, labels, table, colors) -> None:
    """Label each model's isolated point at the nearby spot (a ring of candidate offsets) that overlaps the
    fewest earlier labels and markers and stays inside the axes, nearest first; a label set further off
    gets a thin leader line to its point."""
    import math

    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inside = ax.get_window_extent(rend)
    pts = ax.transData.transform(table[["usd_per_bug", "fixed_frac"]].to_numpy(dtype=float))
    marks = [Bbox.from_extents(px - 8, py - 8, px + 8, py + 8) for px, py in pts]
    scale = fig.dpi / 72  # offset points -> pixels
    cands = [(r * math.cos(a), r * math.sin(a)) for r in (12, 20, 30, 42, 56)
             for a in [math.radians(d) for d in (90, 270, 30, 150, 210, 330, 0, 180, 60, 120, 240, 300)]]
    stars = [(m, *ax.transData.transform((x, y))) for m, x, y in labels]
    taken = []
    for m, x, y in sorted(labels, key=lambda t: -t[2]):
        best = None
        sx, sy = ax.transData.transform((x, y))
        for i, (dx, dy) in enumerate(cands):
            ha = "left" if dx > 3 else "right" if dx < -3 else "center"
            va = "bottom" if dy > 3 else "top" if dy < -3 else "center"
            t = ax.annotate(TAG.get(m, m), (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=6.5, color=colors[m], zorder=3)
            bb = t.get_window_extent(rend).padded(4 * scale)
            def gap(st):  # pixels from the label's box to a star
                return math.hypot(max(bb.x0 - st[1], 0, st[1] - bb.x1), max(bb.y0 - st[2], 0, st[2] - bb.y1))
            own = gap(next(st for st in stars if st[0] == m))
            near = m if all(gap(st) > own + 12 for st in stars if st[0] != m) else None  # unambiguous?
            outside = not (inside.x0 <= bb.x0 and bb.x1 <= inside.x1 and inside.y0 <= bb.y0 and bb.y1 <= inside.y1)
            score = (100 * sum(bb.overlaps(o) for o in taken)
                     + 30 * sum(bb.overlaps(mk) for mk, (px, py) in zip(marks, pts)
                                if math.hypot(px - sx, py - sy) > 2)  # its own star(s) don't count
                     + 1000 * outside + 150 * (near != m) + i * 0.5)
            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, t, bb, math.hypot(dx, dy))
            else:
                t.remove()
        _, t, bb, dist = best
        if dist > 14:
            t.arrow_patch = None
            t.set_annotation_clip(False)
            ax.annotate("", (x, y), xytext=t.get_position(), textcoords="offset points",
                        arrowprops={"arrowstyle": "-", "color": colors[m], "lw": 0.6, "alpha": 0.7,
                                    "shrinkA": 0, "shrinkB": 5}, zorder=2)
        taken.append(bb)


LEVEL_STYLE = {"L1": {"alpha": 0.5}, "L2": {"facecolor": "white"}}  # L0 is the filled point


def pareto(points: pd.DataFrame) -> list[tuple[float, float]]:
    """(USD per bug, share fixed) of the points no other point beats on both lower cost and more
    bugs fixed, cheapest first."""
    front, best = [], None
    for _, r in points.sort_values(["usd_per_bug", "fixed_frac"], ascending=[True, False]).iterrows():
        if best is None or r["fixed_frac"] > best:
            front.append((float(r["usd_per_bug"]), float(r["fixed_frac"])))
            best = float(r["fixed_frac"])
    return front


def plot(table: pd.DataFrame, path: Path, title: str = "", levels: pd.DataFrame | None = None,
         frontiers: bool = False) -> None:
    """``levels`` (``levels_table`` rows with a ``model`` column): each model's L1 and L2 points for the
    settings in ``table``, joined to its L0 point by a dotted line. With ``frontiers``, the Pareto
    frontier of the independent sessions (before) and of the same-session L0 points (after)."""
    from matplotlib.lines import Line2D

    from ..plots.final import MARKERS, pyplot
    plt = pyplot()
    table = table[table["point"] != "isolated_capped"]  # the capped point is in l0.csv; the figure shows the frontier
    fig, ax = plt.subplots(figsize=(7, 4.5))
    models = list(dict.fromkeys(table["model"]))
    colors = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(models)}
    labels = []  # (model, x, y) of each isolated point, labelled once the axes are final
    for m, g in table.groupby("model", sort=False):  # each shared setting joined to the model's isolated point
        iso = g[g["point"] == "isolated"]
        if iso.empty:
            continue
        x0, y0 = float(iso["usd_per_bug"].iloc[0]), float(iso["fixed_frac"].iloc[0])
        for _, r in g[~g["point"].str.startswith("isolated")].iterrows():
            ax.plot([x0, r["usd_per_bug"]], [y0, r["fixed_frac"]], color=colors[m], lw=0.9, alpha=0.45, zorder=1)
        labels.append((m, x0, y0))
    for _, r in table.iterrows():
        ax.scatter(r["usd_per_bug"], r["fixed_frac"], marker=MARKERS.get(r["point"], "x"),
                   s=130 if r["point"] == "isolated" else 45, color=colors[r["model"]], linewidths=0, zorder=2)
    lv = pd.DataFrame() if levels is None else levels[levels["model"].isin(models) & levels["point"].isin(set(table["point"]))
                                                      & levels["level"].isin(LEVEL_STYLE)]
    for (m, pt), g in lv.groupby(["model", "point"], sort=False):
        l0 = table[(table["model"] == m) & (table["point"] == pt)]
        path_pts = l0[["usd_per_bug", "fixed_frac"]].values.tolist() + g.sort_values("level")[["usd_per_bug", "fixed_frac"]].values.tolist()
        ax.plot(*zip(*path_pts), color=colors[m], lw=0.8, ls=":", alpha=0.8, zorder=1)
        for _, r in g.iterrows():
            style = LEVEL_STYLE[r["level"]]
            ax.scatter(r["usd_per_bug"], r["fixed_frac"], marker=MARKERS[pt], s=45, zorder=2,
                       color=colors[m], edgecolors=colors[m], linewidths=1.2 if "facecolor" in style else 0,
                       **({"facecolors": "white"} if "facecolor" in style else {"alpha": style["alpha"]}))
    every = pd.concat([table, lv], ignore_index=True)
    right = float(every["usd_per_bug"].max()) * 1.4
    fronts = []
    if frontiers:
        for pts, color, label in ((table[table["point"] == "isolated"], "dimgrey", "frontier: independent sessions"),
                                  (table[~table["point"].str.startswith("isolated")], "black", "frontier: same session")):
            if pts.empty:
                continue
            fx, fy = zip(*pareto(pts))
            ax.step(list(fx) + [right], list(fy) + [fy[-1]], where="post", color=color, lw=1.1,
                    ls=(0, (2, 2)) if color == "dimgrey" else "--", zorder=0)
            fronts.append(Line2D([], [], color=color, lw=1.1, ls=(0, (2, 2)) if color == "dimgrey" else "--", label=label))
    ax.set_xscale("log")
    ax.set_xlabel("USD per bug (log)")
    ax.set_ylabel("share of bugs fixed")
    ax.set_ylim(max(0.0, float(every["fixed_frac"].min()) - 0.08), min(1.0, float(every["fixed_frac"].max()) + 0.05))
    lo, hi = float(every["usd_per_bug"].min()), float(every["usd_per_bug"].max())
    ax.set_xlim(lo / 2.6, right)  # room on the left for the model labels
    ax.set_title(title or "Cost Performance Frontier")
    _place_labels(fig, ax, labels, every, colors)
    handles = [Line2D([], [], color=colors[m], marker="o", ls="", label=SHORT.get(m, m)) for m in models]
    points = set(table["point"])
    shared = [p for p in POINT_LABEL if p in points and p != "isolated"]
    for p in shared + ["isolated"]:
        label = "same session" if p != "isolated" and len(shared) == 1 else POINT_LABEL[p]
        handles.append(Line2D([], [], color="grey", marker=MARKERS[p], ls="", markeredgewidth=0,
                              markersize=9 if p == "isolated" else 6, label=label))
    if not lv.empty:
        mk = MARKERS[shared[0]] if len(shared) == 1 else "o"
        handles += [Line2D([], [], color="grey", marker=mk, ls="", markeredgewidth=0, markersize=6, label="L0"),
                    Line2D([], [], color="grey", marker=mk, ls=":", alpha=0.5, markeredgewidth=0, markersize=6,
                           label="L1 (+ ETA requests)"),
                    Line2D([], [], color="grey", marker=mk, ls=":", markerfacecolor="white", markersize=6,
                           label="L2 (+ questions)")]
    handles += fronts
    ax.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    from ..plots.final import named_dirs
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=named_dirs(2), action="append", default=[], metavar="NAME=LEVELS:ISOLATED",
                   help="a model's levels and isolated analysis dirs; repeat per model")
    p.add_argument("--table", type=named_dirs(1), action="append", default=[], metavar="NAME=L0_CSV",
                   help="a model's l0.csv computed elsewhere (e.g. on the machine that ran it, whose jobs dirs "
                        "the capped point needs); repeat per model")
    p.add_argument("--levels", type=named_dirs(1), action="append", default=[], metavar="NAME=LEVELS",
                   help="a model's levels analysis dir whose L1 and L2 points are drawn next to its L0 ones; "
                        "repeat per model")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("."), help="the dir the tables' trial_dir paths are relative to")
    a = p.parse_args(argv)
    if not a.model and not a.table:
        p.error("give at least one --model or --table")
    parts = [l0_table(lv, iso, a.root).assign(model=name) for name, (lv, iso) in a.model]
    parts += [pd.read_csv(path).drop(columns="model", errors="ignore").assign(model=name) for name, path in a.table]
    table = pd.concat(parts, ignore_index=True)
    table = table[["model"] + [c for c in table.columns if c != "model"]]
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out / "l0.csv", index=False)
    levels = (pd.concat([levels_table(d).assign(model=name) for name, d in a.levels], ignore_index=True)
              if a.levels else None)
    if levels is not None:
        levels.to_csv(a.out / "levels_cost.csv", index=False)
    plot(table, a.out / "l0.png", title="Cost Performance Frontier: all settings", levels=levels)
    # the same plot with one shared setting at a time against the isolated points: l0_<setting>.png
    shared = [pt for pt in dict.fromkeys(table["point"]) if not pt.startswith("isolated")]
    for pt in shared:
        sub = table[table["point"].isin([pt, "isolated", "isolated_capped"])]
        plot(sub, a.out / f"l0_{pt}.png", title=TITLE.get(pt, f"Cost Performance Frontier: {pt}"), levels=levels,
             frontiers=pt == "background_hinted")
    print(table.to_string(index=False))
    print(f"[OK] {a.out / 'l0.csv'}, {a.out / 'l0.png'} and l0_<setting>.png for {', '.join(shared)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
