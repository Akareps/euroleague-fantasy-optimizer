"""How the fantasy game turns a real game into points and prices.

Everything here is vectorised over numpy arrays so the same functions serve
single calculations and the Monte Carlo simulator.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from elfantasy.rules import GameRules

ArrayLike = float | np.ndarray


def fantasy_score(pir: ArrayLike, won: ArrayLike, rules: GameRules) -> ArrayLike:
    """Player score: PIR components plus the team-win bonus."""

    return pir * np.where(won, 1.0 + rules.win_bonus, 1.0)


def expected_win_bonus_factor(win_prob: float, rules: GameRules) -> float:
    """E[fantasy] / E[PIR] if PIR and winning were independent."""

    return 1.0 + rules.win_bonus * win_prob


def coach_points(margin: ArrayLike, rules: GameRules) -> ArrayLike:
    """Head-coach score for a final margin (continuous margins are rounded).

    The table is evaluated top to bottom; the first row whose threshold the
    rounded margin reaches applies. Games cannot end level, so a simulated
    margin that rounds to zero counts as a one-point result on its own side.
    """

    raw = np.asarray(margin, dtype=float)
    m = np.round(raw)
    m = np.where(m == 0, np.where(raw >= 0, 1.0, -1.0), m)
    thresholds = [t for t, _ in rules.coach_table]
    points = [p for _, p in rules.coach_table]
    conditions = [m >= t for t in thresholds[:-1]]
    out = np.select(conditions, points[:-1], default=points[-1])
    return out if np.ndim(margin) else float(out)


def expected_coach_points(mean_margin: float, rules: GameRules, sigma: float = 11.5) -> float:
    """E[coach points] when the margin is Normal(mean_margin, sigma)."""

    thresholds = [t for t, _ in rules.coach_table]
    points = [p for _, p in rules.coach_table]
    # P(result >= t) = P(margin >= t - 0.5), except that the win/loss line
    # sits at zero: a margin in [0, 0.5) is still a one-point win.
    cut = [0.0 if t == 1 else t - 0.5 for t in thresholds[:-1]]
    upper_tail = [1.0 - norm.cdf((c - mean_margin) / sigma) for c in cut]
    probs, prev = [], 0.0
    for tail in upper_tail:
        probs.append(tail - prev)
        prev = tail
    probs.append(1.0 - prev)
    return float(sum(p * q for p, q in zip(points, probs, strict=True)))


def price_change(score: ArrayLike, price: ArrayLike, rules: GameRules) -> ArrayLike:
    """Credits a player gains or loses after one game.

    +/- ``price_step`` for every ``price_step_points`` points his score is
    above or below his current price, in whole steps, never below the floor.
    """

    steps = np.trunc((np.asarray(score, dtype=float) - price) / rules.price_step_points)
    change = rules.price_step * steps
    return np.maximum(change, rules.price_floor - np.asarray(price, dtype=float))


def win_prob_to_margin(win_prob: float, sigma: float = 11.5) -> float:
    p = min(max(win_prob, 1e-6), 1 - 1e-6)
    return float(norm.ppf(p) * sigma)
