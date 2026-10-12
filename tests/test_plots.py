"""Tests for the ``plot`` subcommand (subagents_bench.plots).

The full run renders every figure from the synthetic analysis directory of
``tests/synth_tables.py`` through the real CLI parser; the other tests pin the
defensive behaviour (minimal inputs, missing columns, NaN costs, condition
filters) and the statistics helpers. No network, no Docker.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from matplotlib import rc_context
from matplotlib.container import ErrorbarContainer
from matplotlib.text import Annotation
from matplotlib.ticker import NullFormatter

from subagents_bench import plots
from subagents_bench.plots import figures
from subagents_bench.plots.data import AnalysisData, bootstrap_mean, bootstrap_ratio, num, wilson
from subagents_bench.plots.figures import pareto_frontier
from subagents_bench.plots.style import RC, errs
from subagents_bench.plots.timelines import select_episodes

# The fixture module lives next to this file; make it importable under any pytest import mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_tables import TASKS, default_conditions, write_synthetic_analysis  # noqa: E402

SINGLE_FIGURES = [n for n in plots.FIGURE_NAMES if n != "timelines"]


def _run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="subagents-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    plots.register_cli(sub)
    args = parser.parse_args(argv)
    return args.func(args)


@pytest.fixture(scope="module")
def synth(tmp_path_factory) -> Path:
    return write_synthetic_analysis(tmp_path_factory.mktemp("synth") / "analysis")


@pytest.fixture(scope="module")
def full_run(synth, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("figs") / "figures"
    assert _run_cli(["plot", str(synth), "--out", str(out), "--formats", "png,pdf"]) == 0
    return out


def _copy(src: Path, dst: Path, tables: list[str]) -> Path:
    dst.mkdir(parents=True)
    for t in tables:
        shutil.copy(src / f"{t}.csv", dst / f"{t}.csv")
    return dst


def _render(fn, data: AnalysisData) -> list:
    """Run one figure function without saving (figures stay inspectable)."""
    with rc_context(RC):
        return list(fn(data))


def _zero_length_whiskers(fig) -> list[tuple[str, np.ndarray]]:
    """Every error-bar segment of zero length in a figure (a capped whisker that looks exact)."""
    bad = []
    for ax in fig.axes:
        for c in ax.containers:
            if not isinstance(c, ErrorbarContainer):
                continue
            for coll in c.lines[2]:
                for seg in coll.get_segments():
                    if len(seg) == 2 and np.all(np.isfinite(seg)) and np.allclose(seg[0], seg[1]):
                        bad.append((ax.get_title(), seg))
    return bad


def _legend_texts(fig) -> list[str]:
    return [t.get_text() for lg in fig.legends for t in lg.get_texts()]


# ---------------------------------------------------------------------------
# full run
# ---------------------------------------------------------------------------


def test_full_run_writes_every_figure(full_run):
    for name in SINGLE_FIGURES:
        png, pdf = full_run / f"{name}.png", full_run / f"{name}.pdf"
        assert png.exists(), name
        assert png.stat().st_size > 5_000, f"{name}.png is suspiciously small"
        assert pdf.exists() and pdf.stat().st_size > 5_000, name
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert pdf.read_bytes()[:5] == b"%PDF-"


def test_full_run_png_resolution_is_200_dpi(full_run):
    import matplotlib.image as mpimg

    h, w = mpimg.imread(full_run / "time_breakdown.png").shape[:2]
    # time_breakdown is 9 in wide before the tight bbox trims margins: ~1800 px at dpi 200 (not ~900 at 100).
    assert 1500 < w < 1900, w


def test_timelines_at_most_two_per_condition(full_run):
    pngs = sorted((full_run / "timelines").glob("*.png"))
    assert pngs, "no timelines written"
    per_cond: dict[str, list[str]] = {}
    for p in pngs:
        assert p.stat().st_size > 5_000, p.name
        assert p.with_suffix(".pdf").exists()
        cond, _task, _rnd, kind = p.stem.split("__")
        assert kind in ("median", "fastest-solved")
        per_cond.setdefault(cond, []).append(kind)
    expected = {c.name for c in default_conditions()}
    assert set(per_cond) == expected
    for cond, kinds in per_cond.items():
        assert 1 <= len(kinds) <= 2, cond
        assert kinds.count("median") == 1, cond


def test_index_has_heading_link_and_caption_for_every_figure(full_run):
    index = (full_run / "index.md").read_text()
    assert "**Synthetic data.**" in index
    for i, name in enumerate(plots.FIGURE_NAMES, start=1):
        assert re.search(rf"^## {i}\. .*", index, re.M), name
    for name in SINGLE_FIGURES:
        assert f"]({name}.png)" in index, name
        # heading, blank line, image, blank, caption paragraph with the CSV columns it uses
        block = index.split(f"(`{name}`)", 1)[1].split("\n## ", 1)[0]
        caption = block.strip().split("\n\n")[1]
        assert len(caption) > 200 and "Data: `" in caption, name
    # every image written is linked from the index
    for png in full_run.rglob("*.png"):
        assert f"]({png.relative_to(full_run).as_posix()})" in index, png.name
    assert "## Skipped figures" not in index


def test_unpriced_schemes_are_not_drawn_as_zero(full_run):
    index = (full_run / "index.md").read_text()
    pareto = index.split("(`pareto_cost`)", 1)[1].split("\n## ", 1)[0]
    # prime prices no model in the synthetic data -> whole panel dropped, reference lacks the Gemma price.
    assert "panel 'prime prices' not drawn: no finite cost_prime_total values" in pareto
    assert "'reference prices': no value (NaN) for gemma31b_parallel_easy" in pareto
    ce = index.split("(`cost_effectiveness`)", 1)[1].split("\n## ", 1)[0]
    assert "panel 'cost per solve, prime' not drawn" in ce


# ---------------------------------------------------------------------------
# graceful degradation
# ---------------------------------------------------------------------------


def test_minimal_dir_only_episodes_single_baseline(synth, tmp_path, caplog):
    eps = pd.read_csv(synth / "episodes.csv")
    d = tmp_path / "minimal"
    d.mkdir()
    eps[eps["condition"] == "single_5step"].to_csv(d / "episodes.csv", index=False)
    caplog.set_level("INFO", logger="subagents_bench.plots")
    report = plots.make_plots(d, tmp_path / "out", formats=["png"])
    written = set(report.figures_written)
    skipped = dict(report.skipped)
    assert written == {"main_grid", "pareto_cost", "pareto_time", "time_breakdown", "token_composition",
                       "orchestrator_tokens", "task_heatmap", "cost_effectiveness"}
    assert set(skipped) == {"per_subagent", "spawn_counts", "parallelism", "timelines", "calibration", "arbiter"}
    assert not any(why.startswith("error") for why in skipped.values()), skipped
    assert skipped["timelines"] == "calls.csv not found"
    assert skipped["per_subagent"] == "dispatches.csv not found"
    assert skipped["arbiter"] == "arbitrations.csv not found"
    assert "no subagent condition" in skipped["spawn_counts"]
    for name in written:
        assert (tmp_path / "out" / f"{name}.png").stat().st_size > 5_000
    index = (tmp_path / "out" / "index.md").read_text()
    assert "## Skipped figures" in index
    assert "- `timelines`: calls.csv not found" in index
    # every skip leaves a log line
    logged = " ".join(r.getMessage() for r in caplog.records)
    for name in skipped:
        assert f"skipped figure {name}" in logged


def test_missing_optional_columns_skip_panels_not_figures(synth, tmp_path):
    d = _copy(synth, tmp_path / "partial", ["episodes", "conditions", "arbitrations", "dispatches"])
    arb = pd.read_csv(d / "arbitrations.csv").drop(columns=["final_confidence"])
    arb.to_csv(d / "arbitrations.csv", index=False)
    disp = pd.read_csv(d / "dispatches.csv").drop(columns=["llm_sec", "status"])
    disp.to_csv(d / "dispatches.csv", index=False)
    eps = pd.read_csv(d / "episodes.csv").drop(columns=[c for c in pd.read_csv(d / "episodes.csv").columns
                                                         if c.startswith("cp_") or c == "overhead_sec"])
    eps.to_csv(d / "episodes.csv", index=False)
    report = plots.make_plots(d, tmp_path / "out", formats=["png"],
                              only=["arbiter", "per_subagent", "time_breakdown"])
    by = {w.figure: w for w in report.written}
    assert set(by) == {"arbiter", "per_subagent"}
    assert "panel 'conf' not drawn: arbitrations.csv lacks final_confidence" in by["arbiter"].notes
    assert any("confidence-vs-resolved" in n or "panel 'resolved' not drawn" in n for n in by["arbiter"].notes)
    assert "panel 'LLM seconds per dispatch' not drawn: dispatches.csv lacks llm_sec" in by["per_subagent"].notes
    assert "submitted-fraction panel not drawn: dispatches.csv lacks status" in by["per_subagent"].notes
    assert dict(report.skipped)["time_breakdown"].startswith("episodes.csv lacks the critical-path columns")


def test_condition_metadata_inferred_from_names(synth, tmp_path):
    d = tmp_path / "bare"
    d.mkdir()
    design = ["topology", "strategy", "scope", "subagent_model", "orchestrator_model", "arbiter_model"]
    pd.read_csv(synth / "episodes.csv").drop(columns=design).to_csv(d / "episodes.csv", index=False)
    pd.read_csv(synth / "conditions.csv").drop(columns=[c for c in design if c != "arbiter_model"]) \
        .to_csv(d / "conditions.csv", index=False)
    data = AnalysisData(d)
    meta = data.meta
    assert meta["single_5step"].baseline and meta["single_matched"].baseline
    assert (meta["gemma31b_arbiter_hard"].strategy, meta["gemma31b_arbiter_hard"].scope) == ("arbiter", "hard")
    # Same strategy x scope cell as gemma31b_parallel_hard -> the later name becomes a control.
    assert not meta["gemma31b_parallel_hard"].control
    assert meta["samemodel_parallel_hard"].control
    order = data.order(meta)
    assert order[:2] == ["single_5step", "single_matched"]
    assert order[2:5] == ["gemma31b_parallel_easy", "gemma31b_parallel_medium", "gemma31b_parallel_hard"]
    assert order[-1] == "samemodel_parallel_hard"


def test_condition_filter_globs_and_errors(synth, tmp_path):
    report = plots.make_plots(synth, tmp_path / "out", formats=["png"], only=["main_grid"],
                              conditions=["single_*", "gemma31b_parallel_*"])
    assert report.figures_written == ["main_grid"]
    index = (tmp_path / "out" / "index.md").read_text()
    line = next(l for l in index.splitlines() if l.startswith("Conditions:"))
    assert re.findall(r"`([^`]+)`", line) == ["single_5step", "single_matched", "gemma31b_parallel_easy",
                                              "gemma31b_parallel_medium", "gemma31b_parallel_hard"]
    with pytest.raises(ValueError, match="matches none"):
        plots.make_plots(synth, tmp_path / "out2", conditions=["nope_*"])
    with pytest.raises(ValueError, match="unsupported format"):
        plots.make_plots(synth, tmp_path / "out3", formats=["gif"])


def test_cli_exit_codes(tmp_path, synth, capsys):
    assert _run_cli(["plot", str(tmp_path / "missing"), "--out", str(tmp_path / "o")]) == 2
    assert "analysis dir not found" in capsys.readouterr().err
    assert _run_cli(["plot", str(synth), "--out", str(tmp_path / "o"), "--formats", "bmp"]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run_cli(["plot", str(empty), "--out", str(tmp_path / "o2")]) == 1  # nothing to plot
    assert (tmp_path / "o2" / "index.md").exists()
    assert _run_cli(["plot", str(synth), "--out", str(tmp_path / "o3"), "--only", "task_heatmap",
                     "--conditions", "single_5step,gemma31b_arbiter_*"]) == 0
    assert (tmp_path / "o3" / "task_heatmap.png").exists()
    assert not (tmp_path / "o3" / "main_grid.png").exists()


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------


def test_pareto_frontier_upper_left():
    x = np.array([1.0, 2.0, 3.0, 4.0, 2.0, np.nan, 0.5])
    y = np.array([0.2, 0.5, 0.4, 0.9, 0.5, 1.0, np.nan])
    assert pareto_frontier(x, y).tolist() == [True, True, False, True, False, False, False]
    assert pareto_frontier(np.array([]), np.array([])).tolist() == []


def test_wilson_and_bootstrap():
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    p, lo, hi = wilson(0, 9)
    assert p == 0.0 and lo == 0.0 and hi == pytest.approx(0.2991, abs=1e-4)
    assert all(math.isnan(v) for v in wilson(0, 0))
    m, lo, hi = bootstrap_mean([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and 1.0 <= lo < 2.5 < hi <= 4.0
    assert bootstrap_mean([1.0, 2.0, 3.0, 4.0]) == (m, lo, hi)  # seeded: reproducible
    assert all(math.isnan(v) for v in bootstrap_mean([1.0, np.nan], strict_nan=True))
    assert bootstrap_mean([1.0, np.nan, 3.0])[0] == 2.0
    m1 = bootstrap_mean([7.0])
    assert m1[0] == 7.0 and math.isnan(m1[1]) and math.isnan(m1[2])
    r, lo, hi = bootstrap_ratio([1.0, 2.0, 3.0], [1.0, 0.0, 1.0])
    assert r == 3.0 and lo <= r <= hi
    assert all(math.isnan(v) for v in bootstrap_ratio([1.0, 2.0], [0.0, 0.0]))  # no solves
    assert all(math.isnan(v) for v in bootstrap_ratio([np.nan, 2.0], [1.0, 1.0]))  # unpriced


def test_fallback_statistics_match_conditions_csv(synth, tmp_path):
    with_csv = AnalysisData(synth)
    d = _copy(synth, tmp_path / "no_conditions", ["episodes"])
    without = AnalysisData(d)
    a = with_csv.cond_metric("wall_clock_sec")
    b = without.cond_metric("wall_clock_sec")
    assert "conditions.csv" in a.attrs["source"] and "episodes.csv" in b.attrs["source"]
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index(), check_exact=False, rtol=1e-9)
    ra = with_csv.resolve_stats().set_index("condition")
    rb = without.resolve_stats().set_index("condition")
    pd.testing.assert_frame_equal(ra[["n_valid", "n_resolved", "rate", "lo", "hi"]],
                                  rb[["n_valid", "n_resolved", "rate", "lo", "hi"]], rtol=1e-9)
    # The invalid (environment-build failure) trial is excluded from the denominator.
    assert ra.loc["gemma31b_sequential_medium", "n_valid"] == 8


def test_timeline_selection(tmp_path):
    d = tmp_path / "sel"
    d.mkdir()
    eps = pd.DataFrame({
        "condition": ["c"] * 4 + ["k"] * 3,
        "episode_id": ["a", "b", "x", "y", "p", "q", "r"],
        "wall_clock_sec": [40, 10, 20, 30, 5, 6, 7],
        "resolved": [1, 0, 0, 0, 0, 1, 0],
        "valid": [True] * 7,
    })
    eps.to_csv(d / "episodes.csv", index=False)
    data = AnalysisData(d)
    e = data.episodes().assign(_wall=lambda f: f["wall_clock_sec"].astype(float))
    picks = select_episodes(data, e, set(eps["episode_id"]))
    got = [(c, k, r["episode_id"]) for c, k, r in picks]
    # c: walls 10,20,30,40 -> median = lower middle (20, "x"); fastest solved = "a" (40 s, the only solve).
    # k: median is "q" (6 s), which is also the fastest solve -> only one pick.
    assert sorted(got) == [("c", "fastest solved", "a"), ("c", "median wall-clock", "x"), ("k", "median wall-clock", "q")]
    # Episodes without events in calls.csv are never picked.
    only_b = [(c, k, r["episode_id"]) for c, k, r in select_episodes(data, e, {"b"})]
    assert only_b == [("c", "median wall-clock", "b")]


def test_runtime_cost_fallback_when_no_scheme_is_priced(synth, tmp_path):
    d = _copy(synth, tmp_path / "unpriced", ["episodes", "conditions"])
    scheme_cost = re.compile(r"cost_(?!runtime)(.+)_(orch|sub|arb|deb|total)|cost_per_solve_(?!runtime)")
    for t in ("episodes", "conditions"):
        df = pd.read_csv(d / f"{t}.csv")
        df.drop(columns=[c for c in df.columns if scheme_cost.match(c)]).to_csv(d / f"{t}.csv", index=False)
    assert AnalysisData(d).schemes() == []
    report = plots.make_plots(d, tmp_path / "out", formats=["png"], only=["pareto_cost", "cost_effectiveness"])
    by = {w.figure: w for w in report.written}
    assert set(by) == {"pareto_cost", "cost_effectiveness"}
    for w in by.values():
        assert any("logged at run time" in n for n in w.notes), w.notes
    assert any("cost_runtime_total" in u for u in by["pareto_cost"].uses)


def test_plots_from_real_aggregator_output(tmp_path):
    """Contract check: the analysis package's tables (built from tests/synth.py) plot without errors."""
    try:
        from synth import make_sweep
        from subagents_bench.analysis.aggregate import aggregate
    except (ImportError, AttributeError) as e:  # analysis package not available in this checkout
        pytest.skip(f"aggregator unavailable: {e}")
    configs = Path(__file__).resolve().parents[1] / "configs"
    jobs = tmp_path / "jobs"
    make_sweep(jobs, rounds=(1, 2))
    aggregate([jobs], tmp_path / "analysis", pricing_path=configs / "pricing.yaml",
              serving_path=configs / "serving.yaml", models_path=configs / "models.yaml")
    report = plots.make_plots(tmp_path / "analysis", tmp_path / "figures", formats=["png"])
    errors = [(n, why) for n, why in report.skipped if why.startswith("error")]
    assert not errors, errors
    assert set(report.figures_written) == set(plots.FIGURE_NAMES), report.skipped


# ---------------------------------------------------------------------------
# intervals: unbounded, unknown, degenerate
# ---------------------------------------------------------------------------


def test_bootstrap_ratio_keeps_resamples_without_a_solve():
    rng = np.random.default_rng(1)
    cost = rng.uniform(0.003, 0.006, 30)
    for k in (1, 2, 3):
        solved = np.zeros(30)
        solved[:k] = 1.0
        p, lo, hi = bootstrap_ratio(cost, solved)
        # More than 2.5% of resamples draw no solve: the ratio is unbounded above, never a tight
        # interval conditional on a solve (which gave hi ~ 1.06 x the point for one solve).
        assert p == pytest.approx(cost.sum() / k)
        assert math.isinf(hi) and hi > 0, (k, hi)
        assert np.isfinite(lo) and lo <= p, (k, lo)
    solved = (np.arange(30) < 15).astype(float)
    p, lo, hi = bootstrap_ratio(cost, solved)
    assert np.isfinite(hi) and lo < p < hi
    # Same resamples as a direct computation (no resample dropped when every one has a solve).
    idx = np.random.default_rng(0).integers(0, 30, size=(2000, 30))
    ratios = cost[idx].sum(axis=1) / solved[idx].sum(axis=1)
    assert lo == pytest.approx(np.quantile(ratios, 0.025, method="lower"))
    assert hi == pytest.approx(np.quantile(ratios, 0.975, method="higher"))


def test_errs_draw_no_whisker_for_unknown_unbounded_or_zero_width_intervals():
    e = errs([1.0, 2.0, 3.0, 4.0, 5.0], [np.nan, 1.5, 3.0, 4.0, 4.0], [1.5, 2.5, np.inf, 4.0, np.nan])
    assert np.isnan(e[:, 0]).all()  # unknown lower bound
    assert e[:, 1].tolist() == [0.5, 0.5]
    assert np.isnan(e[:, 2]).all()  # unbounded: the caller draws the arrow
    assert np.isnan(e[:, 3]).all()  # zero width: degenerate, not exact
    assert np.isnan(e[:, 4]).all()


@pytest.fixture(scope="module")
def sparse(tmp_path_factory) -> Path:
    """A sweep where gemma31b_parallel_easy has one valid episode and gemma31b_sequential_medium
    one solve in nine (so its cost and time per solve are unbounded above)."""
    keep = ("raman-fitting", 3)
    invalid = [("gemma31b_parallel_easy", t["name"], r) for t in TASKS for r in (1, 2, 3) if (t["name"], r) != keep]
    return write_synthetic_analysis(tmp_path_factory.mktemp("sparse") / "analysis", invalid=invalid)


def test_unknown_intervals_are_noted_never_drawn_as_zero_length_whiskers(synth, sparse):
    for d in (synth, sparse):
        data = AnalysisData(d)
        for name, fn in plots.FIGURES:
            if name == "timelines":
                continue
            for r in _render(fn, data):
                assert not _zero_length_whiskers(r.fig), (d.name, r.name, _zero_length_whiskers(r.fig))
    data = AnalysisData(sparse)
    (tb,) = _render(figures.fig_time_breakdown, data)
    assert any(n.startswith("CI unavailable for the wall-clock") and "gemma31b_parallel_easy" in n for n in tb.notes)
    (pt,) = _render(figures.fig_pareto_time, data)
    assert any(n.startswith("CI unavailable for x in ") and "gemma31b_parallel_easy" in n for n in pt.notes), pt.notes


def test_arbiter_confidence_mean_of_one_episode_is_noted(synth, tmp_path):
    d = _copy(synth, tmp_path / "one_arb", ["episodes", "arbitrations"])
    arb = pd.read_csv(d / "arbitrations.csv")
    keep = arb.loc[(arb["condition"] == "gemma31b_arbiter_easy") & arb["final_confidence"].notna(), "episode_id"].iloc[0]
    eps = pd.read_csv(d / "episodes.csv")
    eps.loc[(eps["condition"] == "gemma31b_arbiter_easy") & (eps["episode_id"] != keep), "valid"] = False
    eps.to_csv(d / "episodes.csv", index=False)
    (r,) = _render(figures.fig_arbiter, AnalysisData(d))
    assert not _zero_length_whiskers(r.fig)
    note = next(n for n in r.notes if n.startswith("CI unavailable for the mean confidence"))
    # Every (condition, outcome) group whose mean has no interval is listed, and only those.
    conf = arb.groupby("episode_id")["final_confidence"].mean().rename("conf")
    j = eps[eps["valid"].astype(bool)].set_index("episode_id")[["condition", "resolved"]].join(conf, how="inner")
    j = j[j["conf"].notna()]
    want = sorted(f"{c} ({'resolved' if o else 'not resolved'})" for (c, o), g in j.groupby(["condition", "resolved"])
                  if len(g) < 2 or g["conf"].nunique() < 2)
    assert f"gemma31b_arbiter_easy ({'resolved' if eps.set_index('episode_id').loc[keep, 'resolved'] else 'not resolved'})" in want
    assert sorted(note.split(": ", 1)[1].split(", ")) == want


def test_cost_per_solve_with_rare_solves_is_unbounded_and_noted(sparse):
    data = AnalysisData(sparse)
    eps = data.episodes()
    solves = eps.groupby("condition")["resolved"].agg(["sum", "count"])
    # P(a resample draws no solve) = (1 - k/n)^n, well above 2.5% for these (time per solve has no unpriced model).
    rare = [c for c, r in solves.iterrows() if 0 < r["sum"] and (1 - r["sum"] / r["count"]) ** r["count"] > 0.06]
    common = [c for c, r in solves.iterrows() if (1 - r["sum"] / r["count"]) ** r["count"] < 0.001]
    assert rare and common, solves
    (ce,) = _render(figures.fig_cost_effectiveness, data)
    notes = [n for n in ce.notes if n.startswith("95% interval unbounded above in")]
    time_note = " ".join(n for n in notes if "every panel" in n or "'time per solve'" in n)
    for c in rare:
        assert c in time_note, (c, notes)
    for c in common:
        assert c not in " ".join(notes), (c, notes)
    assert figures.UNBOUNDED_HANDLE_LABEL in _legend_texts(ce.fig)
    # Each unbounded whisker is an arrow ending on the right edge of its panel.
    ax = next(a for a in ce.fig.axes if a.get_title() == "time per solve")
    arrows = [t for t in ax.texts if getattr(t, "arrow_patch", None) is not None]
    assert arrows and all(t.xy[0] == 1.0 for t in arrows)
    assert "resample that draws no solved episode" in ce.caption


def test_time_per_solve_ci_uses_the_episodes_of_its_point(synth, tmp_path):
    d = _copy(synth, tmp_path / "nowall", ["episodes", "conditions"])
    eps = pd.read_csv(d / "episodes.csv")
    row = eps.index[(eps["condition"] == "single_matched") & eps["valid"].astype(bool)][0]
    eps.loc[row, "wall_clock_sec"] = np.nan  # e.g. a still-running episode
    eps.to_csv(d / "episodes.csv", index=False)
    (ce,) = _render(figures.fig_cost_effectiveness, AnalysisData(d))
    ax = next(a for a in ce.fig.axes if a.get_title() == "time per solve")
    k = [t.get_text() for t in ce.fig.axes[0].get_yticklabels()].index("single_matched")
    segs = [s for c in ax.containers if isinstance(c, ErrorbarContainer) for coll in c.lines[2]
            for s in coll.get_segments() if len(s) and np.isclose(s[0][1], k)]
    arrows = [t for t in ax.texts if getattr(t, "arrow_patch", None) is not None and np.isclose(t.xy[1], k)]
    # The CSV point skips the unknown wall-clock; the bootstrap must too, or its CI is NaN (no whisker at all).
    assert segs or arrows, "the time-per-solve whisker of single_matched must still be drawn"
    assert not any("CI unavailable" in n and "single_matched" in n for n in ce.notes), ce.notes


# ---------------------------------------------------------------------------
# time_breakdown: hidden work measured from the spawn groups' start
# ---------------------------------------------------------------------------


def test_time_breakdown_marks_hidden_work_from_the_group_start(synth):
    data = AnalysisData(synth)
    (r,) = _render(figures.fig_time_breakdown, data)
    ax = r.fig.axes[0]
    eps = data.episodes()
    m = eps.groupby("condition")[["cp_orch_llm_sec", "sub_busy_sec", "cp_group_sec"]].mean()
    names = data.order(m.index)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    diamonds = [ln for ln in ax.lines if ln.get_marker() == "D"]
    assert len(diamonds) == 1
    dx, dy = (np.asarray(v, float) for v in diamonds[0].get_data())
    busy = [n for n in names if m.loc[n, "sub_busy_sec"] > 0]
    assert "gemma31b_sequential_easy" not in busy and "single_5step" not in busy
    for n in busy:
        k = labels.index(data.meta[n].label)
        x = dx[np.isclose(dy, k - 0.37)]
        assert x == pytest.approx(m.loc[n, "cp_orch_llm_sec"] + m.loc[n, "sub_busy_sec"]), n
    brackets = [ln for ln in ax.lines if len(ln.get_xdata()) == 4]
    assert len(brackets) == len(busy)
    ends = sorted(float(ln.get_xdata()[2]) for ln in brackets)
    assert ends == pytest.approx(sorted(m.loc[n, "cp_orch_llm_sec"] + m.loc[n, "cp_group_sec"] for n in busy))
    # Parallel groups hide work (diamond right of the bracket); sequential ones do not.
    hidden = m["sub_busy_sec"] - m["cp_group_sec"]
    assert hidden["gemma31b_parallel_hard"] > 1.0 and abs(hidden["gemma31b_sequential_hard"]) < 1.0
    assert "sub_busy_sec - cp_group_sec" in r.caption and "far right of the subagent segment" not in r.caption
    hidden_label = "work hidden by parallelism (sub_busy_sec - cp_group_sec)"
    assert hidden_label in _legend_texts(r.fig)


def test_time_breakdown_legend_has_no_hidden_work_entry_when_none_is_drawn(synth, tmp_path):
    # Judging dominates everywhere (as in the real manager_arbiter_hard smoke episode): no dotted gap.
    d = _copy(synth, tmp_path / "judging", ["episodes", "conditions"])
    eps = pd.read_csv(d / "episodes.csv")
    eps["sub_busy_sec"] = eps["cp_group_sec"] * 0.5
    eps.to_csv(d / "episodes.csv", index=False)
    (r,) = _render(figures.fig_time_breakdown, AnalysisData(d))
    ax = r.fig.axes[0]
    assert not [ln for ln in ax.lines if ln.get_linestyle() == ":"]
    texts = _legend_texts(r.fig)
    assert "critical-path time inside spawn groups (cp_group_sec)" in texts
    assert not any(t.startswith("work hidden by parallelism") for t in texts), texts


# ---------------------------------------------------------------------------
# treatment uptake, estimand, manager controls, zero-valid conditions, duplicates
# ---------------------------------------------------------------------------


def test_main_grid_shows_uptake_estimand_and_manager_tag(synth):
    data = AnalysisData(synth)
    (r,) = _render(figures.fig_main_grid, data)
    ax = r.fig.axes[0]
    conds = pd.read_csv(synth / "conditions.csv").set_index("condition")
    subs = conds[conds["topology"] == "subagents"]
    # One delegated k/n under each subagent bar, red when the condition never delegated.
    counts = [t for t in ax.texts if t.get_text() in {f"{a}/{b}" for a, b in zip(subs["n_delegated"], subs["n_valid"])}
              and isinstance(t, Annotation) and t.xy[1] == 0]
    assert len(counts) == len(subs)
    never = [t for t in counts if t.get_text() == "0/9"]
    assert len(never) == 1 and never[0].get_color() == figures.ROLE_COLORS["arb"]
    assert any(n.startswith("never delegated") and "gemma31b_sequential_easy" in n for n in r.notes)
    # A hollow per-protocol marker for every condition with delegated episodes.
    hollow = [ln for ln in ax.lines if ln.get_marker() == "o" and ln.get_markerfacecolor() == figures.SURFACE]
    rates = sorted(float(ln.get_ydata()[0]) for ln in hollow)
    assert rates == pytest.approx(sorted(subs.loc[subs["n_delegated"] > 0, "resolve_rate_delegated"]))
    assert "resolve rate of the episodes that delegated (per protocol)" in _legend_texts(r.fig)
    assert "these tasks" in r.caption and "paired.csv" in r.caption and "beats that baseline" not in r.caption
    assert any("[manager]" in t.get_text() for t in ax.get_xticklabels())
    assert any("manager_arbiter_hard (manager control, orchestrator tools read_file+list_dir+search" in t
               for t in _legend_texts(r.fig))


def test_main_grid_caption_describes_the_delegated_row_only_when_drawn(synth, tmp_path):
    # An analysis dir written before the uptake columns existed: the row is counted from episodes.csv ...
    d = _copy(synth, tmp_path / "old", ["episodes", "conditions", "conditions_by_difficulty"])
    uptake = ["n_delegated", "delegated_frac", "resolve_rate_delegated", "resolve_delegated_ci_lo",
              "resolve_delegated_ci_hi"]
    for t in ("conditions", "conditions_by_difficulty"):
        pd.read_csv(d / f"{t}.csv").drop(columns=uptake).to_csv(d / f"{t}.csv", index=False)
    for by_difficulty in (False, True):
        cols = ["condition", *(["difficulty"] if by_difficulty else []), "n_delegated", "delegated_frac",
                "rate_delegated"]
        got = AnalysisData(d).resolve_stats(by_difficulty)
        want = AnalysisData(synth).resolve_stats(by_difficulty)
        pd.testing.assert_frame_equal(got[cols], want[cols], check_dtype=False)
        assert got.attrs["uptake_source"].startswith("`episodes.csv` (delegated")
    (counted,) = _render(figures.fig_main_grid, AnalysisData(d))
    assert "row 'delegated'" in counted.caption
    assert any(t.get_text() == "delegated" for t in counted.fig.axes[0].texts)
    # ... and with no uptake anywhere there is no row, so the caption must not describe one.
    eps = pd.read_csv(d / "episodes.csv")
    eps.drop(columns=[c for c in ("delegated", "spawn_calls") if c in eps.columns]).to_csv(d / "episodes.csv",
                                                                                      index=False)
    (old,) = _render(figures.fig_main_grid, AnalysisData(d))
    assert "row 'delegated'" not in old.caption
    assert not any(t.get_text() == "delegated" for t in old.fig.axes[0].texts)
    assert any(n.startswith("no 'delegated' row") for n in old.notes), old.notes
    (new,) = _render(figures.fig_main_grid, AnalysisData(synth))
    assert "row 'delegated'" in new.caption and not any(n.startswith("no 'delegated' row") for n in new.notes)


def test_spawn_counts_shows_uptake_and_the_computed_definition(synth):
    data = AnalysisData(synth)
    (r,) = _render(figures.fig_spawn_counts, data)
    titles = [a.get_title() for a in r.fig.axes]
    assert titles[0] == "episodes that delegated"
    assert any(n.startswith("never delegated") and "gemma31b_sequential_easy" in n for n in r.notes)
    assert "mean over episodes of each episode's mean granted / requested" in r.caption
    assert "mean over spawn calls" not in r.caption
    assert "manager_arbiter_hard [manager]" in [t.get_text() for t in r.fig.axes[0].get_yticklabels()]
    (p,) = _render(figures.fig_pareto_cost, data)
    assert any(t.startswith("manager_arbiter_hard (manager control") for t in _legend_texts(p.fig))


@pytest.mark.parametrize("with_conditions_csv", [True, False])
def test_zero_valid_conditions_are_listed_not_dropped(tmp_path, with_conditions_csv):
    invalid = [("gemma31b_arbiter_easy", None, None), ("single_matched", None, None),
               ("gemma31b_parallel_medium", "crack-7z-hash", None)]
    d = write_synthetic_analysis(tmp_path / "analysis", invalid=invalid)
    if not with_conditions_csv:
        for t in ("conditions", "conditions_by_difficulty"):
            (d / f"{t}.csv").unlink()
    data = AnalysisData(d)
    assert data.resolve_stats().attrs["no_valid"] == ["single_matched", "gemma31b_arbiter_easy"]
    (mg,) = _render(figures.fig_main_grid, data)
    note = next(n for n in mg.notes if n.startswith("no valid trials, slot labelled"))
    assert note.endswith("single_matched, gemma31b_arbiter_easy")
    assert any(t.get_text() == "no valid trials" for t in mg.fig.axes[0].texts)
    assert "single_matched (baseline; no valid trials, not drawn)" in _legend_texts(mg.fig)
    mix = next(n for n in mg.notes if n.startswith("valid trials cover fewer tasks"))
    assert "gemma31b_parallel_medium (2 of 3)" in mix
    (pc,) = _render(figures.fig_pareto_cost, data)
    assert "no valid trials (no resolve rate, not drawn): single_matched, gemma31b_arbiter_easy" in pc.notes
    (ce,) = _render(figures.fig_cost_effectiveness, data)
    assert "gemma31b_arbiter_easy" in [t.get_text() for t in ce.fig.axes[0].get_yticklabels()]
    assert any(n.startswith("no valid trials") and "gemma31b_arbiter_easy" in n for n in ce.notes)
    report = plots.make_plots(d, tmp_path / "out", formats=["png"], only=["main_grid"])
    assert report.figures_written == ["main_grid"]
    assert "Conditions with no valid trial" in (tmp_path / "out" / "index.md").read_text()


def test_events_of_invalid_trials_are_left_out(synth, tmp_path):
    """Dispatches, spawn calls and arbitrations of an invalid trial (e.g. a
    provider outage that kept harbor's trial alive) are dropped with it."""
    d = _copy(synth, tmp_path / "invalid_events", ["episodes", "dispatches", "spawns", "arbitrations", "conditions"])
    eps = pd.read_csv(d / "episodes.csv")
    arb = pd.read_csv(d / "arbitrations.csv")
    gone = arb.loc[arb["condition"] == "gemma31b_arbiter_hard", "episode_id"].drop_duplicates().iloc[:2].tolist()
    eps.loc[eps["episode_id"].isin(gone), ["valid", "invalid_reason"]] = [False, "provider_error"]
    eps.to_csv(d / "episodes.csv", index=False)
    disp, sp = pd.read_csv(d / "dispatches.csv"), pd.read_csv(d / "spawns.csv")
    n_disp, n_sp, n_arb = (int(t["episode_id"].isin(gone).sum()) for t in (disp, sp, arb))
    assert min(n_disp, n_sp, n_arb) > 0
    data = AnalysisData(d)
    rows = data.event_rows("dispatches")
    assert rows.attrs["n_dropped"] == n_disp and not rows["episode_id"].isin(gone).any()
    assert len(rows) == len(disp) - n_disp

    (ps,) = _render(figures.fig_per_subagent, data)
    assert f"{n_disp} dispatch(es) of invalid or duplicate trials left out" in ps.notes
    assert "of a valid trial" in ps.caption
    (sc,) = _render(figures.fig_spawn_counts, data)
    assert f"{n_sp} spawn call(s) of invalid or duplicate trials left out" in sc.notes
    (ar,) = _render(figures.fig_arbiter, data)
    assert f"{n_arb} arbitration(s) of invalid or duplicate trials left out" in ar.notes
    # Nothing is dropped from an analysis dir whose episodes are all valid.
    assert AnalysisData(synth).event_rows("arbitrations").attrs["n_dropped"] == 0


def test_duplicate_trials_are_left_out(synth):
    eps = pd.read_csv(synth / "episodes.csv")
    dup = eps[eps["duplicate"].astype(bool)]
    assert len(dup) == 1 and dup["valid"].astype(bool).all()
    data = AnalysisData(synth)
    assert dup["trial_name"].iloc[0] not in set(data.episodes(valid_only=False)["trial_name"])
    cond = dup["condition"].iloc[0]
    csv = pd.read_csv(synth / "conditions.csv").set_index("condition")
    (d := synth.parent / "dups_only_episodes").mkdir(exist_ok=True)
    shutil.copy(synth / "episodes.csv", d / "episodes.csv")
    st = AnalysisData(d).resolve_stats().set_index("condition")
    assert st.loc[cond, "n_valid"] == csv.loc[cond, "n_valid"] == 9
    assert st.loc[cond, "n_resolved"] == csv.loc[cond, "n_resolved"]


# ---------------------------------------------------------------------------
# axes
# ---------------------------------------------------------------------------


def test_log_axes_label_only_major_ticks(synth):
    (p,) = _render(figures.fig_pareto_cost, AnalysisData(synth))
    logs = [a for a in p.fig.axes if a.get_xscale() == "log"]
    assert logs
    for a in logs:
        assert isinstance(a.xaxis.get_minor_formatter(), NullFormatter)


def test_calibration_shows_every_ttft_point_of_a_small_sample(synth, tmp_path):
    d = tmp_path / "cal"
    d.mkdir()
    calls = pd.read_csv(synth / "calls.csv")
    calls = calls[calls["role"] == "orchestrator"].head(40).copy()
    calls.loc[calls.index[0], "ttft_sec"] = 70.0  # one slow call among 40
    calls.to_csv(d / "calls.csv", index=False)
    shutil.copy(synth / "episodes.csv", d / "episodes.csv")
    (r,) = _render(figures.fig_calibration, AnalysisData(d))
    ax = r.fig.axes[0]
    assert ax.get_ylabel() == "TTFT (s)" and ax.get_ylim()[1] >= 70.0
    assert not any("above the y range" in n for n in r.notes)
