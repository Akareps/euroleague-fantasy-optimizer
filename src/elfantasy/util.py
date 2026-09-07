"""Small numeric helpers shared across the model."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ewma_weights(n: int, half_life: float) -> np.ndarray:
    """Weights for ``n`` observations ordered oldest -> newest.

    The most recent observation has weight 1; an observation ``half_life``
    games older has weight 0.5.
    """

    if n <= 0:
        return np.zeros(0)
    if half_life <= 0:
        w = np.zeros(n)
        w[-1] = 1.0
        return w
    ages = np.arange(n - 1, -1, -1, dtype=float)  # newest -> age 0
    return np.power(0.5, ages / float(half_life))


def weighted_mean(values: Sequence[float], weights: Sequence[float], default: float = 0.0) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0 or w.sum() <= 0:
        return default
    return float((v * w).sum() / w.sum())


def shrink(observed: float, prior: float, n: float, prior_n: float) -> float:
    """Empirical-Bayes style shrinkage of ``observed`` toward ``prior``.

    ``n`` is the amount of evidence behind ``observed`` (games, minutes,
    possessions -- whatever unit ``prior_n`` is expressed in).
    """

    if n <= 0:
        return prior
    denom = n + max(prior_n, 0.0)
    if denom <= 0:
        return observed
    return (n * observed + prior_n * prior) / denom


def logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normalise(weights: Iterable[float]) -> list[float]:
    w = [max(0.0, float(x)) for x in weights]
    total = sum(w)
    if total <= 0:
        n = len(w)
        return [1.0 / n] * n if n else []
    return [x / total for x in w]


def precision_blend(
    estimates: Sequence[float], variances: Sequence[float], default: float = 0.0
) -> tuple[float, float]:
    """Inverse-variance weighted combination of independent estimates.

    Returns ``(mean, variance)``. Estimates with non-positive or infinite
    variance are ignored.
    """

    num = 0.0
    den = 0.0
    for est, var in zip(estimates, variances, strict=False):
        if var is None or not math.isfinite(var) or var <= 0:
            continue
        num += est / var
        den += 1.0 / var
    if den <= 0:
        return default, float("inf")
    return num / den, 1.0 / den


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default
