"""Wilson interval for resolve rates and a seeded percentile bootstrap for means."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

NAN = float("nan")
Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval of ``k`` successes in ``n`` trials (NaN for n = 0)."""
    if n == 0:
        return NAN, NAN
    p, z2 = k / n, z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (0.0 if k == 0 else center - half), (1.0 if k == n else center + half)  # exact ends, no float noise


def bootstrap_mean(values: Iterable[float], n_resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi); NaN if any value is unknown (NaN), no interval (NaN bounds) for one value."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        return NAN, NAN, NAN
    if x.size == 1:
        return float(x[0]), NAN, NAN
    means = np.random.default_rng(seed).choice(x, (n_resamples, x.size)).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)
