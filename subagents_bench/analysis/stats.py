"""Small, dependency-light statistics used by the aggregation tables.

* :func:`wilson_ci` -- Wilson score interval for a binomial proportion (resolve
  rates);
* :func:`bootstrap_mean_ci` -- percentile bootstrap CI of a mean, seeded so the
  tables are reproducible bit for bit;
* :func:`mcnemar_exact` -- exact two-sided McNemar test (binomial on the
  discordant pairs) for paired condition-vs-baseline comparisons;
* :func:`linear_fit` -- ordinary least squares ``y = a + b x`` (TTFT vs prompt
  size in ``calibration.csv``).

Only numpy is needed (no scipy). Every function returns NaN rather than a
made-up number when the input cannot support an estimate.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

NAN = float("nan")
# Two-sided 95% normal quantile.
Z95 = 1.959963984540054


def finite(values: Iterable[float | int | None]) -> np.ndarray:
    """The finite numeric values of an iterable as a float array (None/NaN/inf dropped)."""
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return np.asarray(out, dtype=float)


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n`` trials."""
    if n <= 0:
        return (NAN, NAN)
    if k < 0 or k > n:
        raise ValueError(f"wilson_ci: k={k} outside [0, n={n}]")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # The exact bounds at k == 0 / k == n are 0 / 1; avoid float noise there.
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return (lo, hi)


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    n_resamples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap CI of the mean of the finite values.

    A fresh generator seeded with ``seed`` is used per call, so results do not
    depend on the order in which tables are computed.
    """
    x = finite(values)
    if x.size == 0:
        return (NAN, NAN, NAN)
    mean = float(x.mean())
    if x.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (mean, float(lo), float(hi))


def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k) for X ~ Binomial(n, p), computed exactly with integer combinatorics."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1)))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts ``b`` and ``c``.

    Under H0 the discordant pairs split Binomial(b + c, 0.5); the two-sided
    p-value doubles the smaller tail and is capped at 1. With no discordant
    pairs there is no evidence either way and p = 1.
    """
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(b, c), n, 0.5))


def linear_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, int]:
    """Least-squares ``y = a + b x`` over pairs where both are finite.

    Returns ``(a, b, n)``. With fewer than two points, or no spread in ``x``,
    the slope is not identifiable and ``(nan, nan, n)`` is returned.
    """
    xs, ys = [], []
    for xi, yi in zip(x, y):
        if xi is None or yi is None:
            continue
        xf, yf = float(xi), float(yi)
        if math.isfinite(xf) and math.isfinite(yf):
            xs.append(xf)
            ys.append(yf)
    n = len(xs)
    if n < 2:
        return (NAN, NAN, n)
    xa, ya = np.asarray(xs), np.asarray(ys)
    xm, ym = xa.mean(), ya.mean()
    sxx = float(((xa - xm) ** 2).sum())
    if sxx <= 0.0:
        return (NAN, NAN, n)
    b = float(((xa - xm) * (ya - ym)).sum()) / sxx
    a = float(ym - b * xm)
    return (a, b, n)


def quantile(values: Iterable[float | None], q: float) -> float:
    """Quantile (numpy linear interpolation) of the finite values; NaN if none."""
    x = finite(values)
    if x.size == 0:
        return NAN
    return float(np.quantile(x, q))


def median(values: Iterable[float | None]) -> float:
    return quantile(values, 0.5)
