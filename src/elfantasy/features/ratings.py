"""Team strength and pace.

Bookmaker prices are the better forecast when they exist, but they only exist
for the next round or two. Fantasy decisions need a view of rounds 3, 4 and 5 as
well -- that is exactly the case the user cares about when a slightly worse
player is preferable because he has three easy games coming and will not need to
be transferred out again.

So this module fits a margin-aware Elo model plus simple pace and defensive
ratings from completed games, and those ratings drive projections for any future
round. Where a market price *is* available for a round, the engine blends the
two (see :mod:`elfantasy.projection.engine`) rather than discarding either.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from elfantasy.models import BoxScore, Game
from elfantasy.projection.market import spread_to_win_prob, win_prob_to_spread

DEFAULT_RATING = 1500.0
# Points of margin per Elo point, i.e. a 100-Elo edge is worth ~3.3 points.
ELO_TO_POINTS = 1.0 / 30.0


@dataclass
class TeamRatings:
    """Fitted per-team quantities used to project any future fixture."""

    elo: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 2.8  # points
    pace: dict[str, float] = field(default_factory=dict)  # possessions/40min
    off_rating: dict[str, float] = field(default_factory=dict)  # pts/100 poss
    def_rating: dict[str, float] = field(default_factory=dict)
    # PIR conceded to each position group, relative to league average (1.0).
    def_vs_position: dict[str, dict[str, float]] = field(default_factory=dict)
    league_pace: float = 72.0
    league_total: float = 160.0
    margin_sigma: float = 11.0

    def rating(self, team: str) -> float:
        return self.elo.get(team, DEFAULT_RATING)

    def expected_margin(self, home: str, away: str) -> float:
        """Expected home margin in points (positive = home wins)."""

        return (self.rating(home) - self.rating(away)) * ELO_TO_POINTS + self.home_advantage

    def win_prob(self, home: str, away: str) -> float:
        return spread_to_win_prob(-self.expected_margin(home, away), self.margin_sigma)

    def expected_spread(self, home: str, away: str) -> float:
        """Home handicap in the usual convention (negative = home favoured)."""

        return -self.expected_margin(home, away)

    def expected_total(self, home: str, away: str) -> float:
        pace = 0.5 * (self.pace.get(home, self.league_pace) + self.pace.get(away, self.league_pace))
        league_ppp = self.league_total / (2.0 * self.league_pace) if self.league_pace else 1.1
        off_h = self.off_rating.get(home, league_ppp * 100.0)
        off_a = self.off_rating.get(away, league_ppp * 100.0)
        def_h = self.def_rating.get(home, league_ppp * 100.0)
        def_a = self.def_rating.get(away, league_ppp * 100.0)
        lg = league_ppp * 100.0
        exp_h = (off_h + def_a) / 2.0
        exp_a = (off_a + def_h) / 2.0
        if lg > 0:
            exp_h *= 1.0
            exp_a *= 1.0
        return (exp_h + exp_a) * pace / 100.0

    def defense_vs_position(self, team: str, position: str) -> float:
        return self.def_vs_position.get(team, {}).get(position, 1.0)


def fit_elo(
    games: list[Game],
    *,
    k: float = 20.0,
    home_advantage: float = 2.8,
    mov_scale: float = 0.35,
    initial: dict[str, float] | None = None,
) -> dict[str, float]:
    """Margin-of-victory Elo over completed games, in chronological order.

    The MOV multiplier follows the standard 538-style construction: a blowout
    moves ratings more than a one-possession game, but the effect is damped for
    heavy favourites so that beating a weak side by 25 is not over-rewarded.
    """

    elo: dict[str, float] = dict(initial or {})
    played = [
        g for g in games if g.played and g.home_score is not None and g.away_score is not None
    ]
    played.sort(key=lambda g: (g.round, g.tipoff or 0, g.game_id))

    for g in played:
        rh = elo.setdefault(g.home_code, DEFAULT_RATING)
        ra = elo.setdefault(g.away_code, DEFAULT_RATING)
        margin = float(g.home_score - g.away_score)
        exp_margin = (rh - ra) * ELO_TO_POINTS + home_advantage
        exp_home_win = 1.0 / (1.0 + 10.0 ** (-(rh + home_advantage / ELO_TO_POINTS - ra) / 400.0))
        actual = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

        surprise = abs(margin - exp_margin)
        mov_mult = np.log1p(max(abs(margin), 1.0)) * (2.2 / (0.001 * abs(rh - ra) + 2.2))
        mov_mult *= 1.0 + mov_scale * np.tanh(surprise / 12.0)

        delta = k * float(mov_mult) * (actual - exp_home_win)
        elo[g.home_code] = rh + delta
        elo[g.away_code] = ra - delta

    return elo


def estimate_possessions(team_stats: dict[str, float]) -> float:
    """Standard possessions estimate.

    ``POSS = FGA - OREB + TOV + 0.44 * FTA``
    """

    return (
        team_stats.get("fg_attempted", 0.0)
        - team_stats.get("offensive_rebounds", 0.0)
        + team_stats.get("turnovers", 0.0)
        + 0.44 * team_stats.get("ft_attempted", 0.0)
    )


def fit_ratings(
    games: list[Game],
    boxscores: list[BoxScore],
    positions: dict[str, str] | None = None,
    *,
    k: float = 20.0,
    home_advantage: float = 2.8,
) -> TeamRatings:
    """Fit Elo, pace, efficiency and positional defence from played games."""

    ratings = TeamRatings(home_advantage=home_advantage)
    ratings.elo = fit_elo(games, k=k, home_advantage=home_advantage)

    played = {g.game_id: g for g in games if g.played}

    # --- pace and efficiency ------------------------------------------------
    team_poss: dict[str, list[float]] = defaultdict(list)
    team_pts: dict[str, list[float]] = defaultdict(list)
    opp_pts: dict[str, list[float]] = defaultdict(list)

    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for bs in boxscores:
        if bs.game_id not in played:
            continue
        cell = agg[(bs.game_id, bs.team_code)]
        cell["fg_attempted"] += bs.fg_attempted
        cell["ft_attempted"] += bs.ft_attempted
        cell["turnovers"] += bs.turnovers
        cell["points"] += bs.points
        # Offensive rebounds are not always split out; approximate with 30%.
        cell["offensive_rebounds"] += 0.30 * bs.rebounds

    for (game_id, team), stats in agg.items():
        g = played[game_id]
        poss = estimate_possessions(stats)
        if poss <= 10:
            continue
        team_poss[team].append(poss)
        pts_for = g.home_score if g.is_home(team) else g.away_score
        pts_against = g.away_score if g.is_home(team) else g.home_score
        if pts_for is None or pts_against is None:
            pts_for, pts_against = stats["points"], stats["points"]
        team_pts[team].append(100.0 * float(pts_for) / poss)
        opp_pts[team].append(100.0 * float(pts_against) / poss)

    ratings.pace = {t: float(np.mean(v)) for t, v in team_poss.items() if v}
    ratings.off_rating = {t: float(np.mean(v)) for t, v in team_pts.items() if v}
    ratings.def_rating = {t: float(np.mean(v)) for t, v in opp_pts.items() if v}
    if ratings.pace:
        ratings.league_pace = float(np.mean(list(ratings.pace.values())))
    finals = [g for g in games if g.played and g.home_score is not None]
    if finals:
        ratings.league_total = float(np.mean([g.home_score + g.away_score for g in finals]))
        margins = np.array([g.home_score - g.away_score for g in finals], dtype=float)
        exp = np.array([ratings.expected_margin(g.home_code, g.away_code) for g in finals])
        resid = margins - exp
        if len(resid) >= 8:
            ratings.margin_sigma = float(np.clip(resid.std(ddof=1), 8.0, 16.0))

    # --- positional defence -------------------------------------------------
    if positions:
        conceded: dict[tuple[str, str], list[float]] = defaultdict(list)
        league: dict[str, list[float]] = defaultdict(list)
        for bs in boxscores:
            g = played.get(bs.game_id)
            if g is None or bs.minutes <= 0:
                continue
            pos = positions.get(bs.player_id)
            if not pos:
                continue
            opp = g.opponent_of(bs.team_code)
            value = (bs.pir if bs.pir is not None else 0.0) / max(bs.minutes, 1.0)
            conceded[(opp, pos)].append(value)
            league[pos].append(value)

        league_mean = {pos: float(np.mean(v)) for pos, v in league.items() if v}
        table: dict[str, dict[str, float]] = defaultdict(dict)
        for (team, pos), vals in conceded.items():
            base = league_mean.get(pos, 0.0)
            if base <= 0 or not vals:
                continue
            raw = float(np.mean(vals)) / base
            n = len(vals)
            # Shrink toward neutral; a handful of games proves very little.
            weight = n / (n + 45.0)
            table[team][pos] = 1.0 + weight * (raw - 1.0)
        ratings.def_vs_position = dict(table)

    return ratings


def market_or_model_spread(
    home: str,
    away: str,
    ratings: TeamRatings,
    market_spread: float | None,
    market_total: float | None,
) -> tuple[float, float, float]:
    """Return ``(spread, total, home_win_prob)`` preferring the market."""

    spread = market_spread if market_spread is not None else ratings.expected_spread(home, away)
    total = market_total if market_total is not None else ratings.expected_total(home, away)
    p_home = spread_to_win_prob(spread, ratings.margin_sigma)
    return float(spread), float(total), float(p_home)


def blend_spreads(market: float | None, model: float, market_weight: float = 0.8) -> float:
    if market is None:
        return model
    w = min(max(market_weight, 0.0), 1.0)
    return w * market + (1 - w) * model


__all__ = [
    "TeamRatings",
    "fit_elo",
    "fit_ratings",
    "estimate_possessions",
    "market_or_model_spread",
    "blend_spreads",
    "win_prob_to_spread",
]
