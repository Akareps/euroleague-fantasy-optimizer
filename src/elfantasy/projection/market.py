"""Bookmaker maths: de-vigging, implied team totals, prop inversion.

A bookmaker's posted prices are the single best public forecast of a game, but
they are not probabilities -- they sum to more than one because of the vig
(overround). This module implements the standard de-vig estimators and then
converts the fair probabilities into the quantities the projection engine
needs: win probability, expected team scores, expected pace, and player means
implied by prop lines.

Estimators
----------
* Multiplicative / "proportional": divide by the booksum. Fast, but assumes the
  vig is spread proportionally, which over-taxes longshots.
* Additive: subtract the excess equally. Over-taxes favourites instead.
* Power: solve ``sum(p_i ** k) = 1``. A decent one-parameter compromise.
* Shin (1992, 1993): models the overround as arising from a fraction ``z`` of
  insider bettors. Generally the best simple closed-form estimator for two- and
  three-way markets, and the default here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.optimize import brentq
from scipy.stats import norm, poisson

Method = str


# --------------------------------------------------------------------------
# Price conversions
# --------------------------------------------------------------------------
def american_to_decimal(price: float) -> float:
    if price is None:
        raise ValueError("price is None")
    if price >= 100:
        return 1.0 + price / 100.0
    if price <= -100:
        return 1.0 + 100.0 / abs(price)
    raise ValueError(f"not a valid American price: {price}")


def decimal_to_prob(price: float) -> float:
    if price is None or price <= 1.0:
        raise ValueError(f"decimal price must be > 1.0, got {price}")
    return 1.0 / price


def to_decimal(price: float) -> float:
    """Accept either a decimal price (>1) or an American price; return decimal."""

    if price is None:
        raise ValueError("price is None")
    if 1.0 < price < 100.0:
        return float(price)
    return american_to_decimal(float(price))


def booksum(prices: Sequence[float]) -> float:
    return sum(decimal_to_prob(to_decimal(p)) for p in prices)


def overround(prices: Sequence[float]) -> float:
    """Vig as a fraction, e.g. 0.045 for a 4.5 percent book."""

    return booksum(prices) - 1.0


# --------------------------------------------------------------------------
# De-vigging
# --------------------------------------------------------------------------
def devig_multiplicative(raw: Sequence[float]) -> list[float]:
    s = sum(raw)
    if s <= 0:
        raise ValueError("implied probabilities sum to zero")
    return [p / s for p in raw]


def devig_additive(raw: Sequence[float]) -> list[float]:
    n = len(raw)
    excess = (sum(raw) - 1.0) / n
    out = [p - excess for p in raw]
    # An additive de-vig can push a heavy longshot negative; fall back.
    if any(p <= 0 for p in out):
        return devig_multiplicative(raw)
    return out


def devig_power(raw: Sequence[float]) -> list[float]:
    """Find k such that ``sum(p_i ** k) == 1``."""

    def f(k: float) -> float:
        return sum(p**k for p in raw) - 1.0

    try:
        k = brentq(f, 1.0, 12.0, xtol=1e-10)
    except ValueError:
        return devig_multiplicative(raw)
    return [p**k for p in raw]


def devig_shin(raw: Sequence[float]) -> list[float]:
    """Shin's estimator.

    Solves for the insider fraction ``z`` such that the fair probabilities

        p_i = (sqrt(z^2 + 4(1-z) * q_i^2 / S) - z) / (2 * (1 - z))

    sum to one, where ``q_i`` are the raw implied probabilities and ``S`` their
    sum. Reduces to the multiplicative method as ``z -> 0``.
    """

    s = sum(raw)
    if s <= 0:
        raise ValueError("implied probabilities sum to zero")
    if abs(s - 1.0) < 1e-12:
        return list(raw)

    def fair(z: float) -> list[float]:
        out = []
        for q in raw:
            root = math.sqrt(max(z * z + 4.0 * (1.0 - z) * q * q / s, 0.0))
            out.append((root - z) / (2.0 * (1.0 - z)))
        return out

    def f(z: float) -> float:
        return sum(fair(z)) - 1.0

    lo, hi = 1e-9, 0.35
    try:
        while f(hi) > 0 and hi < 0.95:
            hi = min(hi * 1.6, 0.95)
        z = brentq(f, lo, hi, xtol=1e-12)
    except ValueError:
        return devig_multiplicative(raw)
    return devig_multiplicative(fair(z))  # clean up tiny numerical residual


_DEVIG = {
    "multiplicative": devig_multiplicative,
    "proportional": devig_multiplicative,
    "additive": devig_additive,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(prices: Sequence[float], method: Method = "shin") -> list[float]:
    """De-vig a complete market given decimal or American prices."""

    fn = _DEVIG.get(method.lower())
    if fn is None:
        raise ValueError(f"unknown de-vig method {method!r}; try {sorted(_DEVIG)}")
    raw = [decimal_to_prob(to_decimal(p)) for p in prices]
    return fn(raw)


def two_way_prob(price_a: float, price_b: float | None, method: Method = "shin") -> float:
    """Fair probability of side A. If B is missing, fall back to the raw price."""

    if price_b is None:
        return decimal_to_prob(to_decimal(price_a))
    return devig([price_a, price_b], method)[0]


# --------------------------------------------------------------------------
# Game-level derived quantities
# --------------------------------------------------------------------------
def implied_team_totals(total: float, spread: float) -> tuple[float, float]:
    """Split a game total into (home, away) expected points.

    ``spread`` is the home handicap in the usual convention: negative means the
    home side is favoured by that many points.
    """

    margin = -spread  # expected home margin
    home = total / 2.0 + margin / 2.0
    away = total / 2.0 - margin / 2.0
    return home, away


def spread_to_win_prob(spread: float, sigma: float = 11.0) -> float:
    """Home win probability implied by a point spread.

    ``sigma`` is the standard deviation of the game margin around the spread.
    EuroLeague margins are a touch tighter than the NBA's; ~11 points is a
    reasonable default and can be re-fit with ``elfantasy backtest``.
    """

    return float(norm.cdf(-spread / sigma))


def win_prob_to_spread(p_home: float, sigma: float = 11.0) -> float:
    p = min(max(p_home, 1e-6), 1 - 1e-6)
    return float(-norm.ppf(p) * sigma)


def pace_factor(total: float, league_average_total: float, strength: float = 1.0) -> float:
    """How much faster or slower this game projects than an average one."""

    if league_average_total <= 0:
        return 1.0
    raw = total / league_average_total
    return 1.0 + strength * (raw - 1.0)


# --------------------------------------------------------------------------
# Prop inversion: line + price  ->  expected value of the statistic
# --------------------------------------------------------------------------
def prop_to_mean_normal(line: float, over_prob: float, sd: float) -> float:
    """Invert a two-way prop under a normal assumption.

    If ``X ~ N(mu, sd)`` and ``P(X > line) = p`` then ``mu = line + sd * z(p)``.
    """

    p = min(max(over_prob, 1e-4), 1 - 1e-4)
    return float(line + sd * norm.ppf(p))


def prop_to_mean_poisson(
    line: float, over_prob: float, lo: float = 0.05, hi: float = 60.0
) -> float:
    """Invert a two-way prop for a count statistic under a Poisson assumption.

    Prop lines are half-integers, so ``P(X > line) = P(X >= ceil(line))`` and
    the discreteness is handled exactly, with no continuity correction.
    """

    p = min(max(over_prob, 1e-4), 1 - 1e-4)
    k = math.ceil(line)  # smallest integer strictly above a half-integer line

    def f(lam: float) -> float:
        return float(poisson.sf(k - 1, lam)) - p

    try:
        if f(lo) > 0:
            return lo
        if f(hi) < 0:
            return hi
        return float(brentq(f, lo, hi, xtol=1e-8))
    except ValueError:  # pragma: no cover - guarded above
        return line


def prop_to_mean(
    line: float,
    over_price: float | None,
    under_price: float | None,
    *,
    dist: str = "poisson",
    sd: float | None = None,
    method: Method = "shin",
) -> float:
    """De-vig a prop and return the implied mean of the statistic.

    ``dist='poisson'`` suits counting stats (rebounds, assists, steals);
    ``dist='normal'`` suits points and PIR, where ``sd`` must be supplied.
    """

    if over_price is None and under_price is None:
        raise ValueError("need at least one price")
    if over_price is None:
        p_over = 1.0 - decimal_to_prob(to_decimal(under_price))
    elif under_price is None:
        p_over = decimal_to_prob(to_decimal(over_price))
    else:
        p_over = devig([over_price, under_price], method)[0]

    if dist == "normal":
        if sd is None or sd <= 0:
            raise ValueError("normal inversion needs a positive sd")
        return prop_to_mean_normal(line, p_over, sd)
    return prop_to_mean_poisson(line, p_over)


def pir_sd(minutes: float, sd_per_sqrt_minute: float = 1.15, floor: float = 2.5) -> float:
    """Heteroscedastic PIR spread: variance scales roughly with minutes played."""

    return max(floor, sd_per_sqrt_minute * math.sqrt(max(minutes, 0.0)))
