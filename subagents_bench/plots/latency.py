"""Performance and cost against latency at L0, with the Pareto frontier of each.

Joins the L0 table (``analysis.final``: share of bugs fixed and USD per bug per model and setting)
with the L0 median time to the first reply (``plots.final``'s ``response_time.csv``) and writes:

1. ``latency_background_hinted.png``: orchestrator + subagents only.
2. ``latency.png``: all three same-session settings, each model's points joined by a line.
3. ``latency.csv``: the joined table.

Each figure has two panels: share of bugs fixed against latency, and USD per bug against latency
(log axes). The dashed step line is the Pareto frontier: the points no other point beats on both
latency and share fixed (or on both latency and cost). The isolated runs have no reply latency to
compare, so they are left out.

    python -m subagents_bench.plots.latency --l0 figures/final_all/l0.csv \\
        --response-time figures/final_all/response_time.csv --out figures/final_all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..analysis.final import POINT_LABEL, SHORT, TAG
from .final import MARKERS, pyplot

SHARED = ["background_hinted", "manager_background", "manager_owner"]


def join(l0: pd.DataFrame, response_time: pd.DataFrame) -> pd.DataFrame:
    """The shared-session rows of the L0 table with their L0 median time to first reply."""
    rt = response_time[response_time["level"] == "L0"]
    rt = rt.assign(model=rt["model"].map({v: k for k, v in SHORT.items()}).fillna(rt["model"]))
    rt = rt.rename(columns={"setting": "point"})[["model", "point", "median_sec", "replied_frac"]]
    return l0[l0["point"].isin(SHARED)].merge(rt, on=["model", "point"])


def pareto(table: pd.DataFrame, col: str, higher: bool) -> list[tuple[float, float]]:
    """(latency, value) of the points no other point beats on both lower latency and ``col``
    (higher or lower is better), in order of latency."""
    front, best = [], None
    for _, r in table.sort_values(["median_sec", col], ascending=[True, not higher]).iterrows():
        v = float(r[col])
        if best is None or (v > best if higher else v < best):
            front.append((float(r["median_sec"]), v))
            best = v
    return front


def plot(table: pd.DataFrame, models: list[str], points: list[str], path: Path, title: str) -> None:
    from matplotlib.lines import Line2D
    plt = pyplot()
    colors = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(models)}  # as in analysis.final
    g = table[table["point"].isin(points)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, ylabel, higher in ((axes[0], "fixed_frac", "share of bugs fixed", True),
                                    (axes[1], "usd_per_bug", "USD per bug (log)", False)):
        for m, mg in g.groupby("model", sort=False):
            mg = mg.sort_values("median_sec")
            if len(points) > 1:
                ax.plot(mg["median_sec"], mg[col], color=colors[m], lw=0.9, alpha=0.4, zorder=1)
            for _, r in mg.iterrows():
                ax.scatter(r["median_sec"], r[col], marker=MARKERS[r["point"]], s=50, color=colors[m],
                           linewidths=0, zorder=2)
            r = mg.loc[mg[col].idxmax()]
            ax.annotate(TAG.get(m, m), (r["median_sec"], r[col]), xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color=colors[m])
        fx, fy = zip(*pareto(g, col, higher))
        ax.step(list(fx) + [g["median_sec"].max() * 1.3], list(fy) + [fy[-1]], where="post", color="black",
                ls="--", lw=1, zorder=0)
        ax.set_xscale("log")
        if not higher:
            ax.set_yscale("log")
        ax.set_xlabel("median time to first reply, s (log)")
        ax.set_ylabel(ylabel)
    axes[0].set_title("Performance vs latency")
    axes[1].set_title("Cost vs latency")
    handles = [Line2D([], [], color=colors[m], marker="o", ls="", label=SHORT.get(m, m)) for m in models]
    handles.append(Line2D([], [], color="black", ls="--", lw=1, label="Pareto frontier"))
    if len(points) > 1:
        handles += [Line2D([], [], color="grey", marker=MARKERS[p], ls="", markeredgewidth=0, label=POINT_LABEL[p])
                    for p in points]
    axes[1].legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--l0", type=Path, required=True, help="l0.csv from analysis.final")
    p.add_argument("--response-time", type=Path, required=True, help="response_time.csv from plots.final")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    l0 = pd.read_csv(a.l0)
    table = join(l0, pd.read_csv(a.response_time))
    if table.empty:
        raise SystemExit("[FAIL] no model and setting is in both tables")
    models = list(dict.fromkeys(l0["model"]))
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out / "latency.csv", index=False)
    for points, name, title in ((["background_hinted"], "latency_background_hinted", "L0, orchestrator + subagents"),
                                (SHARED, "latency", "L0, all same-session settings")):
        plot(table, models, points, a.out / f"{name}.png", title)
        print(f"[OK] {a.out / name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
