"""Bookmaker odds: game lines and player props.

Two providers ship:

* :class:`TheOddsApiProvider` -- https://the-odds-api.com, which covers
  EuroLeague h2h/spreads/totals on a free tier and player props on paid tiers.
* :class:`CsvOddsProvider`    -- a local CSV, so you can paste in lines from any
  book (or from a screen-scrape you are permitted to take) without writing code.

Two practical notes about props for this sport:

* Books rarely price PIR / "valuation" directly outside a handful of European
  operators. Points, rebounds and assists are far more widely available, which
  is why :func:`elfantasy.projection.pir.compose_pir_from_props` assembles a
  PIR from parts instead of requiring a PIR line.
* Always take the *two-sided* price when you can. A one-sided price cannot be
  de-vigged and will bias the implied mean upward.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from elfantasy.data.http import HttpClient
from elfantasy.data.injuries import normalise_name
from elfantasy.models import Game, GameOdds, PlayerProp

log = logging.getLogger(__name__)

# The Odds API market keys -> our internal component names.
PROP_MARKET_KEYS = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_steals": "steals",
    "player_blocks": "blocks",
    "player_turnovers": "turnovers",
}


class OddsProvider(Protocol):
    name: str

    def fetch_game_odds(self, games: list[Game]) -> dict[str, GameOdds]:  # pragma: no cover
        ...

    def fetch_props(self, games: list[Game]) -> list[PlayerProp]:  # pragma: no cover
        ...


@dataclass
class TheOddsApiProvider:
    http: HttpClient
    api_key: str
    base: str = "https://api.the-odds-api.com/v4"
    sport: str = "basketball_euroleague"
    regions: str = "eu"
    bookmakers: str | None = None
    name: str = "the-odds-api"

    def _params(self, **extra: Any) -> dict[str, Any]:
        params: dict[str, Any] = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "oddsFormat": "decimal",
            **extra,
        }
        if self.bookmakers:
            params["bookmakers"] = self.bookmakers
            params.pop("regions", None)
        return params

    # --- game lines --------------------------------------------------------
    def fetch_game_odds(self, games: list[Game]) -> dict[str, GameOdds]:
        url = f"{self.base}/sports/{self.sport}/odds"
        payload = self.http.get_json(url, self._params(markets="h2h,spreads,totals"))
        if not payload:
            log.warning("no odds returned; check ODDS_API_KEY and remaining quota")
            return {}

        index = _match_index(games)
        out: dict[str, GameOdds] = {}
        for event in payload:
            game_id = _match_event(event, index)
            if game_id is None:
                continue
            odds = _consensus_from_event(event, game_id)
            if odds is not None:
                out[game_id] = odds
        log.info("matched odds for %d/%d fixtures", len(out), len(games))
        return out

    # --- player props ------------------------------------------------------
    def fetch_props(self, games: list[Game]) -> list[PlayerProp]:
        url_base = f"{self.base}/sports/{self.sport}/events"
        events = self.http.get_json(url_base, {"apiKey": self.api_key}) or []
        index = _match_index(games)
        markets = ",".join(PROP_MARKET_KEYS)

        props: list[PlayerProp] = []
        for event in events:
            game_id = _match_event(event, index)
            if game_id is None:
                continue
            event_id = event.get("id")
            payload = self.http.get_json(
                f"{url_base}/{event_id}/odds", self._params(markets=markets)
            )
            if not payload:
                continue
            props.extend(_parse_props(payload, game_id))
        log.info("parsed %d player props", len(props))
        return props


@dataclass
class CsvOddsProvider:
    """Local CSV odds, for books without an API.

    ``odds.csv``  : game_id,home_price,away_price,spread,total,bookmaker
    ``props.csv`` : game_id,player,market,line,over_price,under_price,bookmaker
    """

    odds_path: Path | None = None
    props_path: Path | None = None
    player_index: dict[str, str] | None = None  # normalised name -> player_id
    name: str = "csv"

    def fetch_game_odds(self, games: list[Game]) -> dict[str, GameOdds]:
        if not self.odds_path or not Path(self.odds_path).exists():
            return {}
        out: dict[str, GameOdds] = {}
        with Path(self.odds_path).open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                gid = (row.get("game_id") or "").strip()
                if not gid:
                    continue
                out[gid] = GameOdds(
                    game_id=gid,
                    home_price=_f(row.get("home_price")),
                    away_price=_f(row.get("away_price")),
                    spread=_f(row.get("spread")),
                    total=_f(row.get("total")),
                    bookmaker=row.get("bookmaker") or self.name,
                    captured_at=datetime.now(),
                )
        return out

    def fetch_props(self, games: list[Game]) -> list[PlayerProp]:
        if not self.props_path or not Path(self.props_path).exists():
            return []
        index = self.player_index or {}
        out: list[PlayerProp] = []
        with Path(self.props_path).open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("player") or "").strip()
                pid = row.get("player_id") or index.get(normalise_name(name))
                line = _f(row.get("line"))
                if not pid or line is None:
                    continue
                out.append(
                    PlayerProp(
                        game_id=(row.get("game_id") or "").strip(),
                        player_id=pid,
                        market=(row.get("market") or "points").strip().lower(),
                        line=line,
                        over_price=_f(row.get("over_price")),
                        under_price=_f(row.get("under_price")),
                        bookmaker=row.get("bookmaker") or self.name,
                    )
                )
        return out


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------
def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _match_index(games: list[Game]) -> dict[str, str]:
    """Map normalised team-pair keys to our game ids."""

    index: dict[str, str] = {}
    for g in games:
        index[f"{g.home_code}|{g.away_code}".lower()] = g.game_id
    return index


def _match_event(event: dict[str, Any], index: dict[str, str]) -> str | None:
    """Match a bookmaker event to a fixture.

    Books use full club names ("Panathinaikos Athens"), the feed uses codes.
    Match on a normalised token overlap, and refuse ambiguous matches rather
    than guessing -- a mis-matched fixture silently corrupts every projection
    for both teams.
    """

    home = normalise_name(str(event.get("home_team", "")))
    away = normalise_name(str(event.get("away_team", "")))
    if not home or not away:
        return None

    best: tuple[float, str] | None = None
    for key, game_id in index.items():
        hc, ac = key.split("|")
        score = _token_overlap(home, hc) + _token_overlap(away, ac)
        if best is None or score > best[0]:
            best = (score, game_id)
    if best and best[0] >= 1.0:
        return best[1]
    log.debug("could not match event %s vs %s", event.get("home_team"), event.get("away_team"))
    return None


def _token_overlap(name: str, code: str) -> float:
    code = code.lower()
    if not code:
        return 0.0
    tokens = name.split()
    for tok in tokens:
        if tok.startswith(code) or code.startswith(tok[:3]):
            return 1.0
    return 0.0


def _consensus_from_event(event: dict[str, Any], game_id: str) -> GameOdds | None:
    """Average the available books, in probability space where relevant."""

    home_team = event.get("home_team")
    h2h_home, h2h_away, spreads, totals = [], [], [], []
    books = event.get("bookmakers", []) or []

    for book in books:
        for market in book.get("markets", []) or []:
            key = market.get("key")
            outcomes = market.get("outcomes", []) or []
            if key == "h2h" and len(outcomes) >= 2:
                for o in outcomes:
                    price = _f(o.get("price"))
                    if price is None:
                        continue
                    (h2h_home if o.get("name") == home_team else h2h_away).append(price)
            elif key == "spreads":
                for o in outcomes:
                    point = _f(o.get("point"))
                    if o.get("name") == home_team and point is not None:
                        spreads.append(point)
            elif key == "totals":
                for o in outcomes:
                    point = _f(o.get("point"))
                    if point is not None and str(o.get("name", "")).lower() == "over":
                        totals.append(point)

    if not (h2h_home or spreads or totals):
        return None

    def avg(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return GameOdds(
        game_id=game_id,
        home_price=avg(h2h_home),
        away_price=avg(h2h_away),
        spread=avg(spreads),
        total=avg(totals),
        bookmaker=f"consensus({len(books)})",
        captured_at=datetime.now(),
    )


def _parse_props(payload: dict[str, Any], game_id: str) -> list[PlayerProp]:
    out: list[PlayerProp] = []
    grouped: dict[tuple[str, str, float], dict[str, float]] = {}

    for book in payload.get("bookmakers", []) or []:
        book_name = book.get("key", "")
        for market in book.get("markets", []) or []:
            component = PROP_MARKET_KEYS.get(market.get("key", ""))
            if component is None:
                continue
            for o in market.get("outcomes", []) or []:
                player = o.get("description") or o.get("participant") or o.get("name")
                point = _f(o.get("point"))
                price = _f(o.get("price"))
                side = str(o.get("name", "")).lower()
                if not player or point is None or price is None or side not in ("over", "under"):
                    continue
                key = (normalise_name(str(player)), component, point)
                grouped.setdefault(key, {})[side] = price
                grouped[key]["_book"] = book_name  # type: ignore[assignment]

    for (player_key, component, line), sides in grouped.items():
        out.append(
            PlayerProp(
                game_id=game_id,
                player_id=player_key,  # resolved to a real id by resolve_prop_players
                market=component,
                line=line,
                over_price=sides.get("over"),
                under_price=sides.get("under"),
                bookmaker=str(sides.get("_book", "")),
            )
        )
    return out


def resolve_prop_players(props: list[PlayerProp], players: dict[str, Any]) -> list[PlayerProp]:
    """Turn the normalised names used by books into our player ids."""

    index = {normalise_name(p.name): pid for pid, p in players.items()}
    resolved, dropped = [], 0
    for prop in props:
        pid = prop.player_id if prop.player_id in players else index.get(prop.player_id)
        if pid is None:
            dropped += 1
            continue
        prop.player_id = pid
        resolved.append(prop)
    if dropped:
        log.warning("dropped %d props whose player could not be matched", dropped)
    return resolved
