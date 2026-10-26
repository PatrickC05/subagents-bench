"""Experiment 1 of the final study (issue #23): cost against bugs fixed at L0, each setting's shared
sessions next to the isolated baseline (each bug alone in a fresh session).

    python -m subagents_bench.analysis.final \
        --model Opus=analysis/final_levels-claude-opus-5.5:analysis/final_isolated-claude-opus-5.5 \
        --model Luna=... --out analysis/final_l0 [--interval 240] [--wall 1800]

Each ``--model NAME=LEVELS:ISOLATED`` is one model's two sweeps' tables (``scripts/analyze.sh``
writes them): ``episodes.csv`` for cost and ``fixes.csv`` for each bug's replayed fix time. Writes
``l0.csv`` (one row per model and point) and ``l0.png`` (dollars per bug, log scale, against the
share fixed: a color per model, a marker per point). The points:

- a setting at L0 (``*_baseline``): every bug sent in its sessions; fixed means passing at the end
  (``fixed_end``); a bug's cost is its session's cost over the bugs the session was sent;
- ``isolated``: the same bugs, each in its own session (``single_bug<k>``), fixed at its end;
- ``isolated_capped``: the same sessions, fixed only if the fix came within the time the bug had in
  the shared sessions, the wall clock minus its send time there (``wall - k * interval``), so the
  gap to the settings is not only the isolated agent's extra time. Its cost is the whole session's,
  an upper bound on what the capped run would have cost.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ISOLATED = re.compile(r"single_bug(\d+)$")


def _load(d: Path, name: str) -> pd.DataFrame:
    path = d / name
    if not path.is_file():
        raise SystemExit(f"[FAIL] {path} not found: run scripts/analyze.sh on that sweep first")
    df = pd.read_csv(path)
    return df[df["valid"].astype(bool)] if "valid" in df else df


def _row(point: str, fixed: pd.Series, cost_per_bug: pd.Series, n_sessions: int, n_tasks: int) -> dict:
    n = len(fixed)
    return {"point": point, "n_bugs": n, "n_fixed": int(fixed.sum()), "fixed_frac": fixed.mean() if n else float("nan"),
            "usd_per_bug": cost_per_bug.mean() if n else float("nan"), "usd_total": cost_per_bug.sum(),
            "usd_per_repo": cost_per_bug.sum() / n_tasks if n_tasks else float("nan"), "n_sessions": n_sessions}


def l0_table(levels: Path, isolated: Path, interval: float = 240.0, wall: float = 1800.0) -> pd.DataFrame:
    ep, fx = _load(levels, "episodes.csv"), _load(levels, "fixes.csv")
    rows = []
    for cond, e in ep[ep["condition"].str.endswith("_baseline")].groupby("condition"):
        f = fx[fx["condition"] == cond].merge(e[["trial_name", "cost_total"]], on="trial_name")
        per_bug = f["cost_total"] / f.groupby("trial_name")["subtask_id"].transform("count")
        rows.append(_row(cond.removesuffix("_baseline"), f["fixed_end"].astype(bool), per_bug, len(e),
                         e["task_name"].nunique()))
    ep, fx = _load(isolated, "episodes.csv"), _load(isolated, "fixes.csv")
    f = fx.merge(ep[["trial_name", "cost_total"]], on="trial_name")
    k = f["condition"].str.extract(ISOLATED, expand=False)
    if k.isna().any():
        bad = sorted(f.loc[k.isna(), "condition"].unique())
        raise SystemExit(f"[FAIL] {isolated}: conditions {bad} are not single_bug<k>")
    cap = wall - k.astype(int) * interval
    fixed_sec = pd.to_numeric(f["fixed_sec"], errors="coerce")
    n_tasks = ep["task_name"].nunique()
    rows.append(_row("isolated", f["fixed_end"].astype(bool), f["cost_total"], len(ep), n_tasks))
    # fixed_sec: the first pass the bug then keeps (fixes.py), so a fix by the cap counts even if the
    # isolated agent went on working after it
    rows.append(_row("isolated_capped", fixed_sec.le(cap), f["cost_total"], len(ep), n_tasks))
    return pd.DataFrame(rows)


MARKERS = {"background_hinted": "o", "manager_background": "s", "manager_owner": "D", "isolated": "*",
           "isolated_capped": "*"}


def plot(table: pd.DataFrame, path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7, 4.5))
    models = list(dict.fromkeys(table["model"]))
    colors = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(models)}
    for _, r in table.iterrows():
        capped = r["point"] == "isolated_capped"
        ax.scatter(r["usd_per_bug"], r["fixed_frac"], marker=MARKERS.get(r["point"], "x"),
                   s=110 if r["point"].startswith("isolated") else 45, color=colors[r["model"]],
                   facecolors="none" if capped else colors[r["model"]])
    ax.set_xscale("log")
    ax.set_xlabel("USD per bug (log)")
    ax.set_ylabel("share of bugs fixed")
    ax.set_ylim(0, 1)
    ax.set_title(title or "L0: cost vs bugs fixed")
    handles = [Line2D([], [], color=colors[m], marker="o", ls="", label=m) for m in models]
    handles += [Line2D([], [], color="grey", marker=MARKERS[p], ls="", label=p,
                       markerfacecolor="none" if p == "isolated_capped" else "grey") for p in MARKERS]
    ax.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _model_arg(text: str) -> tuple[str, Path, Path]:
    name, sep, dirs = text.partition("=")
    levels, sep2, isolated = dirs.partition(":")
    if not (sep and sep2 and name and levels and isolated):
        raise argparse.ArgumentTypeError(f"expected NAME=LEVELS_DIR:ISOLATED_DIR, got {text!r}")
    return name, Path(levels), Path(isolated)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=_model_arg, action="append", required=True, metavar="NAME=LEVELS:ISOLATED",
                   help="a model's levels and isolated analysis dirs; repeat per model")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--interval", type=float, default=240.0, help="seconds between bugs in the shared sessions")
    p.add_argument("--wall", type=float, default=1800.0, help="the shared sessions' wall clock in seconds")
    a = p.parse_args(argv)
    table = pd.concat([l0_table(lv, iso, a.interval, a.wall).assign(model=name) for name, lv, iso in a.model],
                      ignore_index=True)
    table = table[["model"] + [c for c in table.columns if c != "model"]]
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out / "l0.csv", index=False)
    plot(table, a.out / "l0.png")
    print(table.to_string(index=False))
    print(f"[OK] {a.out / 'l0.csv'}, {a.out / 'l0.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
