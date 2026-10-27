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
    """Player score: PIR plus, for a win, ``win_bonus`` of its absolute value
    (-3 PIR in a win scores -2.7)."""

    return pir + np.where(won, rules.win_bonus * np.abs(pir), 0.0)


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


def price_change(
    score: ArrayLike, price: ArrayLike, rules: GameRules, played: ArrayLike | None = None
) -> ArrayLike:
    """Credits a player gains or loses after one game.

    ``score`` is whatever ``rules.price_basis`` names (raw PIR by default).
    +/- ``price_step`` per ``price_step_points`` above or below
    ``price_hurdle`` x his current price, rounded to whole steps;
    ``price_dnp_change`` if he did not play (``played`` False); never below
    the floor.
    """

    price = np.asarray(price, dtype=float)
    x = (np.asarray(score, dtype=float) - rules.price_hurdle * price) / rules.price_step_points
    if rules.price_rounding == "half_up":
        steps = np.floor(x + 0.5 + 1e-9)  # halves go up, -0.5 -> 0
    elif rules.price_rounding == "round":
        steps = np.round(x)
    else:
        steps = np.trunc(x)
    change = rules.price_step * steps
    if played is not None:
        change = np.where(played, change, rules.price_dnp_change)
    # The floor only limits losses: a player listed below it (possible in
    # custom data) must not be lifted up to it.
    return np.maximum(change, np.minimum(0.0, rules.price_floor - price))


def win_prob_to_margin(win_prob: float, sigma: float = 11.5) -> float:
    p = min(max(win_prob, 1e-6), 1 - 1e-6)
    return float(norm.ppf(p) * sigma)
