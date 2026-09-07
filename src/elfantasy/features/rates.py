"""Per-minute production rates.

Once minutes are projected, the remaining question is what a player does with
them. Rates are estimated component-wise (points, rebounds, assists, ... ) on a
per-minute basis, recency-weighted, and shrunk toward a positional prior so that
a player with two good games does not out-project a proven starter.

The module also handles *usage transfer*: when a team-mate is absent, the
survivors do not merely play more minutes, they also shoot and create more per
minute. Ignoring this systematically under-projects the beneficiary of an
injury -- the exact situation this project is meant to catch.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from elfantasy.config import Section
from elfantasy.models import BoxScore, Position
from elfantasy.projection.pir import StatLine, statline_from_boxscore
from elfantasy.util import ewma_weights, shrink, weighted_mean

COMPONENTS = tuple(StatLine().as_dict().keys())

# Per-minute league priors by position, in the same units as StatLine.
#
# Calibrated so that a full team-game (200 minutes, roughly 40/35/25 G/F/C
# minutes split) reproduces typical EuroLeague team totals: ~82 points, ~31
# rebounds, ~17 assists, ~91 total PIR. They are a shrinkage target for small
# samples, not a projection in their own right -- re-fit them on your own
# history before treating the absolute numbers as calibrated.
# fmt: off  (kept tabular: these are three comparable rows, not code)
POSITION_PRIORS: dict[str, dict[str, float]] = {
    "G": {
        "points": 0.400,
        "rebounds": 0.110,
        "assists": 0.130,
        "steals": 0.038,
        "blocks": 0.005,
        "fouls_drawn": 0.092,
        "missed_fg": 0.152,
        "missed_ft": 0.020,
        "turnovers": 0.065,
        "blocks_against": 0.008,
        "fouls_committed": 0.078,
    },
    "F": {
        "points": 0.400,
        "rebounds": 0.155,
        "assists": 0.065,
        "steals": 0.033,
        "blocks": 0.014,
        "fouls_drawn": 0.098,
        "missed_fg": 0.150,
        "missed_ft": 0.021,
        "turnovers": 0.055,
        "blocks_against": 0.012,
        "fouls_committed": 0.098,
    },
    "C": {
        "points": 0.420,
        "rebounds": 0.225,
        "assists": 0.050,
        "steals": 0.028,
        "blocks": 0.032,
        "fouls_drawn": 0.118,
        "missed_fg": 0.145,
        "missed_ft": 0.028,
        "turnovers": 0.060,
        "blocks_against": 0.014,
        "fouls_committed": 0.115,
    },
}
# fmt: on


@dataclass
class RateProfile:
    """A player's per-minute rates plus the evidence behind them."""

    player_id: str
    position: str
    per_minute: dict[str, float]
    minutes_sample: float
    games: int
    home_factor: float = 1.0
    away_factor: float = 1.0

    def statline(self, minutes: float) -> StatLine:
        return StatLine(**{k: v * minutes for k, v in self.per_minute.items()})

    @property
    def pir_per_minute(self) -> float:
        return self.statline(1.0).pir


def _position_prior(position: str) -> dict[str, float]:
    return POSITION_PRIORS.get(position, POSITION_PRIORS["F"])


def fit_rate_profile(
    player_id: str,
    position: Position | str,
    boxscores: list[BoxScore],
    model: Section,
) -> RateProfile:
    """Recency-weighted, shrunk per-minute rates for one player."""

    pos = position.value if isinstance(position, Position) else str(position)
    half_life = float(model.get("rates.half_life_games"))
    prior_minutes = float(model.get("rates.prior_minutes"))

    lines = sorted([b for b in boxscores if b.player_id == player_id], key=lambda b: b.round)
    lines = [b for b in lines if b.minutes > 0]
    prior = _position_prior(pos)

    if not lines:
        return RateProfile(player_id, pos, dict(prior), 0.0, 0)

    weights = ewma_weights(len(lines), half_life)
    minutes = np.asarray([b.minutes for b in lines], dtype=float)
    effective_minutes = float((minutes * weights).sum())

    per_minute: dict[str, float] = {}
    stat_lines = [statline_from_boxscore(b).as_dict() for b in lines]
    for comp in COMPONENTS:
        totals = np.asarray([sl[comp] for sl in stat_lines], dtype=float)
        # Weighted per-minute rate = weighted totals / weighted minutes.
        num = float((totals * weights).sum())
        raw = num / effective_minutes if effective_minutes > 0 else prior[comp]
        per_minute[comp] = shrink(raw, prior[comp], effective_minutes, prior_minutes)

    profile = RateProfile(
        player_id=player_id,
        position=pos,
        per_minute=per_minute,
        minutes_sample=float(minutes.sum()),
        games=len(lines),
    )
    profile.home_factor, profile.away_factor = _home_away_factors(lines, model)
    return profile


def _home_away_factors(lines: list[BoxScore], model: Section) -> tuple[float, float]:
    """Player-specific home/away production factors, shrunk to the league value.

    Home/away splits are real but noisy. Shrinking hard is essential: over ten
    games a player's split is mostly noise, and an unshrunk split will happily
    tell you a good player is unplayable on the road.
    """

    lg_home = float(model.get("context.home_factor"))
    lg_away = float(model.get("context.away_factor"))
    prior_games = float(model.get("context.home_split_prior_games"))

    home_vals = [
        b.pir / b.minutes for b in lines if b.pir is not None and b.minutes > 0 and _is_home(b)
    ]
    away_vals = [
        b.pir / b.minutes for b in lines if b.pir is not None and b.minutes > 0 and not _is_home(b)
    ]
    if not home_vals or not away_vals:
        return lg_home, lg_away

    overall = float(np.mean(home_vals + away_vals))
    if overall <= 0:
        return lg_home, lg_away

    raw_home = float(np.mean(home_vals)) / overall
    raw_away = float(np.mean(away_vals)) / overall
    home = shrink(raw_home, lg_home, len(home_vals), prior_games)
    away = shrink(raw_away, lg_away, len(away_vals), prior_games)
    return float(np.clip(home, 0.85, 1.20)), float(np.clip(away, 0.80, 1.15))


def _is_home(bs: BoxScore) -> bool:
    """Home/away flag carried on the box score by the loaders.

    Feeds vary; :mod:`elfantasy.data` sets ``bs.started`` and a private
    ``_home`` attribute where the fixture is known. Absent that, assume home so
    the split simply collapses to the league factor.
    """

    return bool(getattr(bs, "_home", True))


def usage_transfer(
    profile: RateProfile,
    absent_profiles: list[tuple[RateProfile, float]],
    beneficiary_share: float,
    model: Section,
) -> RateProfile:
    """Lift a player's per-minute rates to absorb absent team-mates' usage.

    ``absent_profiles`` pairs each absent player's profile with the probability
    they miss the game. ``beneficiary_share`` is this player's share of the
    vacated role, as computed by
    :func:`elfantasy.features.minutes.redistribution_weights`.

    Only usage-driven components move -- shots, assists, turnovers, free throws.
    Rebounds move partially. Steals, blocks and fouls are essentially per-minute
    constants and are left alone.
    """

    if not absent_profiles or beneficiary_share <= 0:
        return profile

    transfer = float(model.get("absence_redistribution.usage_transfer"))
    usage_components = {
        "points": 1.0,
        "missed_fg": 1.0,
        "missed_ft": 1.0,
        "turnovers": 0.8,
        "assists": 0.7,
        "rebounds": 0.45,
        "fouls_drawn": 0.6,
    }

    vacated: dict[str, float] = defaultdict(float)
    for absent, miss_prob in absent_profiles:
        for comp, weight in usage_components.items():
            vacated[comp] += absent.per_minute.get(comp, 0.0) * miss_prob * weight

    boosted = dict(profile.per_minute)
    for comp, amount in vacated.items():
        boosted[comp] = boosted.get(comp, 0.0) + amount * beneficiary_share * transfer

    return RateProfile(
        player_id=profile.player_id,
        position=profile.position,
        per_minute=boosted,
        minutes_sample=profile.minutes_sample,
        games=profile.games,
        home_factor=profile.home_factor,
        away_factor=profile.away_factor,
    )


def fit_all(
    players: dict[str, str],
    boxscores: list[BoxScore],
    model: Section,
) -> dict[str, RateProfile]:
    """Fit every player's rate profile. ``players`` maps id -> position code."""

    by_player: dict[str, list[BoxScore]] = defaultdict(list)
    for bs in boxscores:
        by_player[bs.player_id].append(bs)
    return {
        pid: fit_rate_profile(pid, pos, by_player.get(pid, []), model)
        for pid, pos in players.items()
    }


def consistency(boxscores: list[BoxScore], player_id: str) -> float:
    """Coefficient of variation of PIR, floored -- a per-player risk signal."""

    vals = [
        b.pir for b in boxscores if b.player_id == player_id and b.pir is not None and b.minutes > 0
    ]
    if len(vals) < 3:
        return 0.6
    arr = np.asarray(vals, dtype=float)
    mean = arr.mean()
    if mean <= 0.5:
        return 1.2
    return float(np.clip(arr.std(ddof=1) / mean, 0.15, 1.5))


def weighted_recent_pir(boxscores: list[BoxScore], player_id: str, half_life: float = 4.0) -> float:
    lines = sorted(
        [b for b in boxscores if b.player_id == player_id and b.pir is not None],
        key=lambda b: b.round,
    )
    if not lines:
        return 0.0
    w = ewma_weights(len(lines), half_life)
    return weighted_mean([b.pir for b in lines], w)
