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


def plot(table: pd.DataFrame, path: Path, title: str = "") -> None:
    from matplotlib.lines import Line2D

    from ..plots.final import MARKERS, pyplot
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    models = list(dict.fromkeys(table["model"]))
    colors = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(models)}
    for m, g in table.groupby("model", sort=False):  # each shared setting joined to the model's isolated point
        iso = g[g["point"] == "isolated"]
        if iso.empty:
            continue
        x0, y0 = float(iso["usd_per_bug"].iloc[0]), float(iso["fixed_frac"].iloc[0])
        for _, r in g[~g["point"].str.startswith("isolated")].iterrows():
            ax.plot([x0, r["usd_per_bug"]], [y0, r["fixed_frac"]], color=colors[m], lw=0.9, alpha=0.45, zorder=1)
        ax.annotate(m, (x0, y0), xytext=(-6, 7), textcoords="offset points", ha="right", fontsize=6.5, color=colors[m])
    for _, r in table.iterrows():
        capped = r["point"] == "isolated_capped"  # a hollow ring around the isolated star, visible when they coincide
        ax.scatter(r["usd_per_bug"], r["fixed_frac"], marker=MARKERS.get(r["point"], "x"),
                   s=260 if capped else 110 if r["point"] == "isolated" else 45, color=colors[r["model"]],
                   facecolors="none" if capped else colors[r["model"]], linewidths=1.1 if capped else None, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("USD per bug (log)")
    ax.set_ylabel("share of bugs fixed")
    ax.set_ylim(max(0.0, float(table["fixed_frac"].min()) - 0.08), min(1.0, float(table["fixed_frac"].max()) + 0.05))
    lo, hi = float(table["usd_per_bug"].min()), float(table["usd_per_bug"].max())
    ax.set_xlim(lo / 2.6, hi * 1.4)  # room on the left for the model labels
    ax.set_title(title or "L0: cost vs bugs fixed")
    handles = [Line2D([], [], color=colors[m], marker="o", ls="", label=m) for m in models]
    handles += [Line2D([], [], color="grey", marker=MARKERS[p], ls="", label=p,
                       markerfacecolor="none" if p == "isolated_capped" else "grey") for p in MARKERS]
    handles.append(Line2D([], [], color="grey", lw=0.9, alpha=0.45, label="setting to its isolated point"))
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
    plot(table, a.out / "l0.png")
    print(table.to_string(index=False))
    print(f"[OK] {a.out / 'l0.csv'}, {a.out / 'l0.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
