"""Wilson interval for resolve rates, a seeded percentile bootstrap for means, and a seeded
cluster bootstrap (whole sessions resampled within strata) for statistics of clustered rows."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from typing import Iterable, TypeVar

import numpy as np

NAN = float("nan")
Z95 = 1.959963984540054
T = TypeVar("T")


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


def cluster_bootstrap(clusters: Sequence[T], stat: Callable[[list[T]], float], strata: Sequence[Hashable] | None = None,
                      n_resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(``stat(clusters)``, lo, hi): a 95% percentile interval from resamples that draw, within each
    stratum, as many clusters as it has, with replacement (one stratum without ``strata``). A
    cluster is whatever ``stat`` reads (e.g. one session's rows or counts); a cluster drawn twice is
    passed twice. Resamples whose statistic is NaN (e.g. a ratio with nothing in its denominator)
    are left out. NaN bounds when the statistic is NaN, when no stratum has two clusters (nothing
    to resample) or when no resample has a statistic."""
    est = float(stat(list(clusters)))
    if strata is None:
        strata = [0] * len(clusters)
    if len(strata) != len(clusters):
        raise ValueError(f"{len(strata)} strata for {len(clusters)} clusters")
    groups: dict[Hashable, list[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    if math.isnan(est) or all(len(g) < 2 for g in groups.values()):
        return est, NAN, NAN
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(g, (n_resamples, len(g))) for g in groups.values()], axis=1)
    stats = np.array([stat([clusters[i] for i in row]) for row in draws], dtype=float)
    stats = stats[~np.isnan(stats)]
    if not stats.size:
        return est, NAN, NAN
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return est, float(lo), float(hi)
