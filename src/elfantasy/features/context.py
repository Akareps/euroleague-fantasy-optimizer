"""Game context multipliers: venue, rest, opponent, pace, blowout.

Each function returns a multiplicative factor around 1.0 so the engine can
compose them transparently and report the contribution of every one of them in
``elfantasy explain``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from elfantasy.config import Section
from elfantasy.features.ratings import TeamRatings
from elfantasy.models import Game
from elfantasy.projection.market import pace_factor
from elfantasy.util import clamp


@dataclass
class GameContext:
    """Everything about one fixture that bears on a player's production."""

    game_id: str
    round: int
    team_code: str
    opponent_code: str
    home: bool
    spread: float  # home handicap; negative = home favoured
    total: float
    win_prob: float  # for `team_code`
    team_total: float
    rest_days: int | None = None
    source: str = "model"  # "market" when prices were available
    notes: list[str] = field(default_factory=list)

    @property
    def own_spread(self) -> float:
        """Handicap from this team's perspective (negative = favoured)."""

        return self.spread if self.home else -self.spread

    @property
    def blowout_margin(self) -> float:
        return abs(self.spread)


def venue_factor(profile_home: float, profile_away: float, home: bool) -> float:
    return profile_home if home else profile_away


def rest_factor(rest_days: int | None, model: Section) -> float:
    if rest_days is None:
        return 1.0
    table = model.get("context.rest_factor").as_dict()
    keys = sorted(int(k) for k in table)
    if not keys:
        return 1.0
    d = max(min(int(rest_days), keys[-1]), keys[0])
    return float(table.get(d, table.get(keys[-1], 1.0)))


def opponent_factor(
    ratings: TeamRatings,
    opponent: str,
    position: str,
    model: Section,
) -> float:
    """How generous the opponent is to this position, shrunk toward neutral."""

    strength = float(model.get("context.opponent_strength"))
    raw = ratings.defense_vs_position(opponent, position)
    return clamp(1.0 + strength * (raw - 1.0), 0.80, 1.25)


def game_pace_factor(total: float, ratings: TeamRatings, model: Section) -> float:
    strength = float(model.get("context.pace_strength"))
    return clamp(pace_factor(total, ratings.league_total, strength), 0.88, 1.15)


def team_total_factor(team_total: float, ratings: TeamRatings) -> float:
    """Scoring-environment factor for this specific team.

    A team projected for 92 points offers more fantasy production to share out
    than the same team projected for 74, independently of pace.
    """

    baseline = ratings.league_total / 2.0
    if baseline <= 0:
        return 1.0
    return clamp(0.55 + 0.45 * (team_total / baseline), 0.85, 1.18)


def build_context(
    game: Game,
    team_code: str,
    ratings: TeamRatings,
    *,
    market_spread: float | None = None,
    market_total: float | None = None,
    market_win_prob: float | None = None,
    previous_game_date: datetime | None = None,
) -> GameContext:
    """Assemble a :class:`GameContext`, preferring market inputs to the model."""

    home = game.is_home(team_code)
    spread = (
        market_spread
        if market_spread is not None
        else ratings.expected_spread(game.home_code, game.away_code)
    )
    total = (
        market_total
        if market_total is not None
        else ratings.expected_total(game.home_code, game.away_code)
    )
    p_home = (
        market_win_prob
        if market_win_prob is not None
        else ratings.win_prob(game.home_code, game.away_code)
    )
    win_prob = p_home if home else 1.0 - p_home

    margin = -spread  # home margin
    home_total = total / 2.0 + margin / 2.0
    away_total = total / 2.0 - margin / 2.0
    team_total = home_total if home else away_total

    rest = None
    if previous_game_date is not None and game.tipoff is not None:
        rest = max((game.tipoff.date() - previous_game_date.date()).days, 0)

    source = "market" if (market_spread is not None or market_total is not None) else "model"

    return GameContext(
        game_id=game.game_id,
        round=game.round,
        team_code=team_code,
        opponent_code=game.opponent_of(team_code),
        home=home,
        spread=float(spread),
        total=float(total),
        win_prob=float(clamp(win_prob, 0.02, 0.98)),
        team_total=float(team_total),
        rest_days=rest,
        source=source,
    )


def schedule_quality(contexts: list[GameContext], gamma: float = 0.55) -> float:
    """Discounted quality of a run of fixtures, in win-probability units.

    This is the quantity behind the "coach B has three easy games so I will not
    have to transfer him out next round" argument. It is not used directly as a
    projection -- the engine projects each round properly -- but it is a useful
    single number for reports and for sorting candidates.
    """

    if not contexts:
        return 0.0
    num = 0.0
    den = 0.0
    for i, ctx in enumerate(sorted(contexts, key=lambda c: c.round)):
        w = gamma**i
        num += w * ctx.win_prob
        den += w
    return num / den if den else 0.0
