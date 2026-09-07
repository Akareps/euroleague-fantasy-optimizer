"""EuroLeague official feeds: schedule, results and box scores.

The league publishes several public endpoints. They are not a documented,
versioned API and their shapes drift between seasons, so every parser here is
defensive: it looks for a field under any of the spellings it has been seen
under, and skips records it cannot understand rather than crashing the run.

Endpoints used (all overridable via environment variables):

* ``{base}/v2/competitions/{comp}/seasons/{season}/games``          -- schedule
* ``{base}/v2/competitions/{comp}/seasons/{season}/games/{n}/stats`` -- box scores
* ``{base}/v1/results?seasonCode=...``                              -- legacy results
* ``{feeds}/v2/competitions/{comp}/seasons/{season}/people``        -- rosters

If a shape changes, fix it here. Nothing above this layer needs to know.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from elfantasy.data.http import HttpClient
from elfantasy.models import BoxScore, Game, Player, Position, Team
from elfantasy.projection.pir import pir_from_boxscore

log = logging.getLogger(__name__)

COMPETITION = "E"  # E = EuroLeague, U = EuroCup


def _first(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-null key. Feeds rename fields constantly."""

    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
        # tolerate camelCase / snake_case / PascalCase drift
        for variant in (k.lower(), k.upper(), k.replace("_", "")):
            if variant in d and d[variant] is not None:
                return d[variant]
    return default


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s in {"-", "DNP", "N/A"}:
        return default
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return default


def parse_minutes(value: Any) -> float:
    """Feeds give minutes as ``MM:SS``, as a float, or as an empty string."""

    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s in {"-", "DNP"}:
        return 0.0
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) + int(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 0.0
    return _num(s)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d"):
        try:
            return datetime.fromisoformat(s) if fmt is None else datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


class EuroleagueClient:
    """Read-only client for the league's public feeds."""

    def __init__(
        self,
        http: HttpClient,
        season: str = "E2025",
        api_base: str = "https://api-live.euroleague.net",
        feeds_base: str = "https://feeds.incrowdsports.com/provider/euroleague-feeds",
        competition: str = COMPETITION,
    ) -> None:
        self.http = http
        self.season = season
        self.api_base = api_base.rstrip("/")
        self.feeds_base = feeds_base.rstrip("/")
        self.competition = competition

    # --- schedule ----------------------------------------------------------
    def fetch_games(self) -> list[Game]:
        url = f"{self.api_base}/v2/competitions/{self.competition}/seasons/{self.season}/games"
        payload = self.http.get_json(url)
        rows = _rows(payload)
        if not rows:
            log.warning("no games returned from %s", url)
            return []

        games: list[Game] = []
        for row in rows:
            try:
                games.append(self._parse_game(row))
            except (KeyError, TypeError, ValueError) as exc:
                log.debug("skipping unparseable game row: %s", exc)
        log.info("parsed %d games for %s", len(games), self.season)
        return games

    def _parse_game(self, row: dict[str, Any]) -> Game:
        home = _first(row, "local", "home", "homeTeam", default={}) or {}
        away = _first(row, "road", "away", "awayTeam", default={}) or {}
        if isinstance(home, str):
            home = {"club": {"code": home}}
        if isinstance(away, str):
            away = {"club": {"code": away}}

        home_code = _team_code(home) or str(_first(row, "homecode", "codeteama", default=""))
        away_code = _team_code(away) or str(_first(row, "awaycode", "codeteamb", default=""))
        home_score = _first(home, "score", "points")
        away_score = _first(away, "score", "points")

        played = bool(_first(row, "played", default=False)) or (
            home_score is not None and away_score is not None and _num(home_score) > 0
        )

        return Game(
            game_id=str(_first(row, "identifier", "gameCode", "id", "code", default="")),
            round=int(_num(_first(row, "round", "gameday", "roundNumber", default=0))),
            home_code=home_code,
            away_code=away_code,
            tipoff=_parse_datetime(_first(row, "date", "utcDate", "startDate")),
            home_score=int(_num(home_score)) if home_score is not None else None,
            away_score=int(_num(away_score)) if away_score is not None else None,
            played=played,
        )

    # --- rosters -----------------------------------------------------------
    def fetch_players(self) -> tuple[dict[str, Team], dict[str, Player]]:
        url = f"{self.feeds_base}/v2/competitions/{self.competition}/seasons/{self.season}/people"
        payload = self.http.get_json(url, {"personType": "J", "limit": 600})
        rows = _rows(payload)
        if not rows:
            log.warning("no roster data from %s", url)
            return {}, {}

        teams: dict[str, Team] = {}
        players: dict[str, Player] = {}
        for row in rows:
            person = _first(row, "person", default=row) or row
            club = _first(row, "club", "team", default={}) or {}
            code = _team_code(club)
            if not code:
                continue
            if code not in teams:
                teams[code] = Team(
                    code=code,
                    name=str(_first(club, "name", "tvCode", default=code)),
                    country=_first(club, "country", "countryName"),
                )
            pid = str(_first(person, "code", "id", "personCode", default="")).strip()
            if not pid:
                continue
            players[pid] = Player(
                player_id=pid,
                name=str(_first(person, "name", "fullName", "passportName", default=pid)),
                team_code=code,
                position=Position.parse(str(_first(row, "positionName", "position", default="F"))),
            )
        log.info("parsed %d players across %d clubs", len(players), len(teams))
        return teams, players

    # --- box scores --------------------------------------------------------
    def fetch_boxscores(self, games: Iterable[Game]) -> list[BoxScore]:
        out: list[BoxScore] = []
        for game in games:
            if not game.played:
                continue
            out.extend(self.fetch_game_boxscore(game))
        log.info("parsed %d player box score lines", len(out))
        return out

    def fetch_game_boxscore(self, game: Game) -> list[BoxScore]:
        url = (
            f"{self.api_base}/v2/competitions/{self.competition}"
            f"/seasons/{self.season}/games/{game.game_id}/stats"
        )
        payload = self.http.get_json(url)
        if payload is None:
            return []

        lines: list[BoxScore] = []
        for team_block in _stat_blocks(payload):
            code = _team_code(team_block) or ""
            for row in _rows(_first(team_block, "playersStats", "players", "stats", default=[])):
                bs = self._parse_boxscore(row, game, code)
                if bs is not None:
                    lines.append(bs)
        return lines

    def _parse_boxscore(self, row: dict[str, Any], game: Game, team_code: str) -> BoxScore | None:
        pid = str(_first(row, "playerCode", "player", "code", "personCode", default="")).strip()
        if not pid:
            return None
        code = team_code or str(_first(row, "team", "teamCode", default=""))
        two_made = _num(_first(row, "fieldGoalsMade2", "twoPointersMade"))
        two_att = _num(_first(row, "fieldGoalsAttempted2", "twoPointersAttempted"))
        three_made = _num(_first(row, "fieldGoalsMade3", "threePointersMade"))
        three_att = _num(_first(row, "fieldGoalsAttempted3", "threePointersAttempted"))

        bs = BoxScore(
            game_id=game.game_id,
            round=game.round,
            player_id=pid,
            team_code=code,
            minutes=parse_minutes(_first(row, "timePlayed", "minutes", "min")),
            points=_num(_first(row, "points", "pts")),
            rebounds=_num(_first(row, "totalRebounds", "rebounds", "reb")),
            assists=_num(_first(row, "assistances", "assists", "ast")),
            steals=_num(_first(row, "steals", "stl")),
            blocks=_num(_first(row, "blocksFavour", "blocks", "blk")),
            fouls_drawn=_num(_first(row, "foulsReceived", "foulsDrawn")),
            fg_made=two_made + three_made,
            fg_attempted=two_att + three_att,
            ft_made=_num(_first(row, "freeThrowsMade", "ftm")),
            ft_attempted=_num(_first(row, "freeThrowsAttempted", "fta")),
            turnovers=_num(_first(row, "turnovers", "tov")),
            blocks_against=_num(_first(row, "blocksAgainst", "blocksReceived")),
            fouls_committed=_num(_first(row, "foulsCommited", "foulsCommitted", "pf")),
            started=bool(_first(row, "startFive", "starter", default=False)),
        )
        reported = _first(row, "valuation", "pir", "performanceIndexRating")
        bs.pir = _num(reported) if reported is not None else pir_from_boxscore(bs)
        return bs


def _rows(payload: Any) -> list[dict[str, Any]]:
    """Pull a list of records out of whatever envelope the feed used."""

    if payload is None:
        return []
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "games", "players", "content"):
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
            if isinstance(value, dict):
                return _rows(value)
    return []


def _stat_blocks(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("stats", "teams", "boxScore", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [b for b in value if isinstance(b, dict)]
        local = payload.get("local")
        road = payload.get("road")
        blocks = [b for b in (local, road) if isinstance(b, dict)]
        if blocks:
            return blocks
    if isinstance(payload, list):
        return [b for b in payload if isinstance(b, dict)]
    return []


def _team_code(block: Any) -> str:
    if not isinstance(block, dict):
        return ""
    club = block.get("club")
    if isinstance(club, dict):
        code = _first(club, "code", "tvCode", "abbreviation")
        if code:
            return str(code).strip()
    code = _first(block, "code", "teamCode", "tvCode", "abbreviation")
    return str(code).strip() if code else ""
