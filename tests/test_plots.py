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

from subagents_bench import plots
from subagents_bench.plots.data import AnalysisData, bootstrap_mean, bootstrap_ratio, wilson
from subagents_bench.plots.figures import pareto_frontier
from subagents_bench.plots.timelines import select_episodes

# The fixture module lives next to this file; make it importable under any pytest import mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_tables import default_conditions, write_synthetic_analysis  # noqa: E402

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
