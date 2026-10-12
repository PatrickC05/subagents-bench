"""Statistics helpers: Wilson CI, seeded bootstrap, exact McNemar, least squares (plain and non-negative)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from subagents_bench.analysis.stats import (
    binom_cdf,
    bootstrap_mean_ci,
    linear_fit,
    mcnemar_exact,
    median,
    nonneg_least_squares,
    quantile,
    wilson_ci,
)


def test_wilson_known_values():
    # Published Wilson 95% intervals: 8/10 -> (0.4902, 0.9433); 0/10 -> (0, 0.2775); 10/10 -> (0.7225, 1).
    # Exact: for k=0 the upper bound is 2*(z^2/2n)/(1+z^2/n) = 0.2775328.
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.490162, abs=1e-6)
    assert hi == pytest.approx(0.943318, abs=1e-6)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.277533, abs=1e-6)
    lo, hi = wilson_ci(10, 10)
    assert lo == pytest.approx(0.722467, abs=1e-6)
    assert hi == 1.0
    # Symmetry: k/n and (n-k)/n mirror each other.
    a, b = wilson_ci(3, 17)
    c, d = wilson_ci(14, 17)
    assert a == pytest.approx(1 - d) and b == pytest.approx(1 - c)


def test_wilson_failure_paths():
    lo, hi = wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)
    with pytest.raises(ValueError):
        wilson_ci(5, 3)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 1024)  # 0.021484375
    assert mcnemar_exact(9, 1) == pytest.approx(0.021484375)
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact(2, 8) == pytest.approx(2 * (1 + 10 + 45) / 1024)  # 0.109375
    assert mcnemar_exact(3, 3) == 1.0  # capped
    assert mcnemar_exact(0, 0) == 1.0  # no discordant pairs


def test_binom_cdf():
    assert binom_cdf(0, 4) == pytest.approx(1 / 16)
    assert binom_cdf(2, 4) == pytest.approx(11 / 16)
    assert binom_cdf(-1, 4) == 0.0
    assert binom_cdf(4, 4) == 1.0


def test_bootstrap_seeded_and_reproducible():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, lo, hi = bootstrap_mean_ci(x)
    assert mean == 3.0
    # Same algorithm spelled out: seed 0, 2000 resamples, percentile 2.5/97.5.
    rng = np.random.default_rng(0)
    means = np.asarray(x)[rng.integers(0, 5, size=(2000, 5))].mean(axis=1)
    exp_lo, exp_hi = np.quantile(means, [0.025, 0.975])
    assert lo == pytest.approx(exp_lo) and hi == pytest.approx(exp_hi)
    assert 1.0 < lo < 3.0 < hi < 5.0
    assert bootstrap_mean_ci(x) == (mean, lo, hi)  # deterministic
    y = list(np.linspace(0.0, 1.0, 23) ** 2)
    assert bootstrap_mean_ci(y, seed=1) != bootstrap_mean_ci(y, seed=0)


def test_bootstrap_edge_cases():
    assert bootstrap_mean_ci([2.5, 2.5, 2.5]) == (2.5, 2.5, 2.5)
    # One value has a mean but no interval (not a zero-width CI that reads as certainty).
    mean, lo, hi = bootstrap_mean_ci([7.0, None, float("nan")])
    assert mean == 7.0 and math.isnan(lo) and math.isnan(hi)
    # NaN/None are dropped, never treated as 0.
    assert bootstrap_mean_ci([1.0, float("nan"), None, 3.0])[0] == 2.0
    assert all(math.isnan(v) for v in bootstrap_mean_ci([]))
    assert all(math.isnan(v) for v in bootstrap_mean_ci([None, float("nan")]))


def test_linear_fit_exact_and_degenerate():
    xs = [0.5, 1.0, 2.0, 4.0, 8.0]
    a, b, n = linear_fit(xs, [0.3 + 0.02 * x for x in xs])
    assert (a, b, n) == (pytest.approx(0.3), pytest.approx(0.02), 5)
    a, b, n = linear_fit([1.0], [2.0])
    assert math.isnan(a) and math.isnan(b) and n == 1
    a, b, n = linear_fit([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])  # no spread in x
    assert math.isnan(a) and math.isnan(b) and n == 3
    # None/NaN pairs are skipped.
    a, b, n = linear_fit([0.0, None, 1.0, 2.0], [1.0, 5.0, 3.0, float("nan")])
    assert (a, b, n) == (pytest.approx(1.0), pytest.approx(2.0), 2)


def test_nonneg_least_squares_exact_and_constrained():
    rng = np.random.default_rng(3)
    p, c = rng.uniform(1, 20, 40), rng.uniform(10, 900, 40)
    rows = [(1.0, pi, ci) for pi, ci in zip(p, c)]
    coef, n = nonneg_least_squares(rows, 0.6 + 0.03 * p + 0.01 * c)
    assert n == 40 and coef == pytest.approx([0.6, 0.03, 0.01])
    # Unconstrained OLS would give a negative prompt slope here; the constrained fit drops that
    # regressor (coefficient 0) and refits the rest, keeping the intercept, so residuals sum to 0.
    y = 0.6 - 0.02 * p + 0.01 * c
    ols = np.linalg.lstsq(np.asarray(rows), y, rcond=None)[0]
    assert ols[1] < 0
    coef, _ = nonneg_least_squares(rows, y)
    assert coef[1] == 0.0 and (coef >= 0).all()
    exp = np.linalg.lstsq(np.asarray(rows)[:, [0, 2]], y, rcond=None)[0]
    assert (coef[0], coef[2]) == (pytest.approx(exp[0]), pytest.approx(exp[1]))
    assert (np.asarray(rows) @ coef).sum() == pytest.approx(y.sum())
    # Non-finite rows are skipped; fewer than two usable rows -> NaN.
    coef, n = nonneg_least_squares([(1.0, 1.0, 5.0), (1.0, float("nan"), 2.0)], [1.0, 2.0])
    assert n == 1 and np.isnan(coef).all() and len(coef) == 3
    # Collinear columns (every prompt the same): a full-rank subset is used, preferring the earlier column.
    coef, n = nonneg_least_squares([(1.0, 2.0, x) for x in (10.0, 20.0, 30.0)], [1.1, 1.2, 1.3])
    assert n == 3 and coef == pytest.approx([1.0, 0.0, 0.01])


def test_quantiles():
    assert median([3, 1, None, 2]) == 2.0
    assert quantile([0, 10], 0.9) == pytest.approx(9.0)
    assert math.isnan(median([]))
