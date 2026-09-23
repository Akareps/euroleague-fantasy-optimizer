"""Core domain objects.

Everything downstream (data adapters, feature builders, the projection engine
and the optimiser) speaks in these types, so a new data source only has to
produce them -- it never has to know how the model works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Position(str, Enum):
    """Fantasy position slot. The official game uses three buckets."""

    GUARD = "G"
    FORWARD = "F"
    CENTER = "C"

    @classmethod
    def parse(cls, raw: str | None) -> Position:
        if not raw:
            return cls.FORWARD
        s = raw.strip().upper()
        if s.startswith("G") or "GUARD" in s:
            return cls.GUARD
        if s.startswith("C") or "CENT" in s:
            return cls.CENTER
        return cls.FORWARD


class Availability(str, Enum):
    """Reported injury / availability status, normalised across sources."""

    ACTIVE = "active"
    PROBABLE = "probable"
    QUESTIONABLE = "questionable"
    DOUBTFUL = "doubtful"
    OUT = "out"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, raw: str | None) -> Availability:
        if not raw:
            return cls.UNKNOWN
        s = raw.strip().lower()
        table = {
            "out": cls.OUT,
            "injured": cls.OUT,
            "injury": cls.OUT,
            "unavailable": cls.OUT,
            "suspended": cls.OUT,
            "ruled out": cls.OUT,
            "doubtful": cls.DOUBTFUL,
            "unlikely": cls.DOUBTFUL,
            "questionable": cls.QUESTIONABLE,
            "game-time decision": cls.QUESTIONABLE,
            "gtd": cls.QUESTIONABLE,
            "day-to-day": cls.QUESTIONABLE,
            "probable": cls.PROBABLE,
            "likely": cls.PROBABLE,
            "available": cls.ACTIVE,
            "active": cls.ACTIVE,
            "healthy": cls.ACTIVE,
            "fit": cls.ACTIVE,
        }
        for key, value in table.items():
            if key in s:
                return value
        return cls.UNKNOWN


@dataclass(frozen=True)
class Team:
    code: str
    name: str
    country: str | None = None


@dataclass
class Player:
    """A player as the fantasy game and the stats feed jointly describe them."""

    player_id: str
    name: str
    team_code: str
    position: Position
    price: float = 0.0
    # Fantasy-game metadata, when available.
    ownership: float | None = None  # fraction of managers holding them
    purchase_price: float | None = None  # what *you* paid, for sell-value maths
    # Availability, filled in by the injury adapters.
    status: Availability = Availability.UNKNOWN
    status_note: str | None = None
    games_since_return: int | None = None
    # A stated minutes estimate (e.g. from preseason reporting). When set, the
    # minutes model uses it instead of its own baseline and never trims it.
    minutes_override: float | None = None

    @property
    def key(self) -> str:
        return self.player_id


@dataclass
class Coach:
    """A head coach as the fantasy game prices him."""

    coach_id: str
    name: str
    team_code: str
    price: float = 0.0


@dataclass
class Game:
    """One scheduled or completed fixture."""

    game_id: str
    round: int
    home_code: str
    away_code: str
    tipoff: datetime | None = None
    home_score: int | None = None
    away_score: int | None = None
    played: bool = False

    def opponent_of(self, team_code: str) -> str:
        return self.away_code if team_code == self.home_code else self.home_code

    def is_home(self, team_code: str) -> bool:
        return team_code == self.home_code


@dataclass
class BoxScore:
    """A single player's line in a single game.

    Field names follow the official PIR components so that
    :func:`elfantasy.projection.pir.pir_from_components` can be applied
    directly to either observed or projected values.
    """

    game_id: str
    round: int
    player_id: str
    team_code: str
    minutes: float = 0.0
    points: float = 0.0
    rebounds: float = 0.0
    assists: float = 0.0
    steals: float = 0.0
    blocks: float = 0.0
    fouls_drawn: float = 0.0
    fg_made: float = 0.0
    fg_attempted: float = 0.0
    ft_made: float = 0.0
    ft_attempted: float = 0.0
    turnovers: float = 0.0
    blocks_against: float = 0.0
    fouls_committed: float = 0.0
    pir: float | None = None
    started: bool = False


@dataclass
class GameOdds:
    """Closing (or latest) market prices for a fixture.

    Prices are decimal. Use :mod:`elfantasy.projection.market` to de-vig them.
    """

    game_id: str
    home_price: float | None = None
    away_price: float | None = None
    spread: float | None = None  # points, negative = home favoured
    spread_price_home: float | None = None
    spread_price_away: float | None = None
    total: float | None = None
    total_over_price: float | None = None
    total_under_price: float | None = None
    bookmaker: str | None = None
    captured_at: datetime | None = None


@dataclass
class PlayerProp:
    """A player prop line, e.g. 'Points over 13.5 at 1.87'."""

    game_id: str
    player_id: str
    market: str  # points | rebounds | assists | pir | pra ...
    line: float
    over_price: float | None = None
    under_price: float | None = None
    bookmaker: str | None = None


@dataclass
class Projection:
    """The engine's output for one player in one round."""

    player_id: str
    round: int
    mean_pir: float
    sd_pir: float
    minutes: float
    play_prob: float
    game_id: str | None = None
    opponent: str | None = None
    home: bool | None = None
    win_prob: float | None = None
    # Every multiplicative / additive step that produced `mean_pir`, kept for
    # `elfantasy explain`. Values are the factor applied at that step.
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def variance(self) -> float:
        return self.sd_pir**2


@dataclass
class Squad:
    """A fantasy roster plus the money situation around it."""

    player_ids: list[str]
    bank: float = 0.0
    purchase_prices: dict[str, float] = field(default_factory=dict)
    coach_id: str | None = None

    def __contains__(self, player_id: str) -> bool:
        return player_id in self.player_ids

    @property
    def size(self) -> int:
        return len(self.player_ids)
