"""OddsPapi v4 coverage probe and conservative, timestamped prop parsing.

Keep the market catalogue: each threshold has its own market ID. An entry in
the catalogue, or hasOdds=True, is not evidence of a usable player forecast.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from elfantasy.data.injuries import normalise_name

EXPERIMENT = "oddspapi-v1"
BASE_URL = "https://api.oddspapi.io/v4"
FULL_GAME_PERIODS = {"result", "fulltime", "full time", "match", "full game"}
COMPONENTS = ("points", "rebounds", "assists")


class OddsPapiError(RuntimeError):
    pass


def utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamps must include a timezone, e.g. 2026-10-12T12:00:00Z")
    return dt.astimezone(timezone.utc)


def game_time(game: dict) -> datetime:
    if game.get("utcDate"):
        return utc(game["utcDate"])
    # The official schedule gives localTimeZone as an offset in hours.
    dt = datetime.fromisoformat(game["localDate"])
    if dt.tzinfo is None:
        if "localTimeZone" not in game:
            raise ValueError(f"game {game['identifier']} has no UTC time or timezone offset")
        dt = dt.replace(tzinfo=timezone(timedelta(hours=float(game["localTimeZone"]))))
    return dt.astimezone(timezone.utc)


class OddsPapiClient:
    """No URL logging, response-body logging or credential-bearing HTTP cache."""

    def __init__(self, api_key: str, session=None, min_interval: float = 1.05):
        if not api_key.strip():
            raise OddsPapiError("OddsPapi API key is empty")
        self._key = api_key.strip()
        self._session = session or requests.Session()
        self._interval = min_interval
        self._last = 0.0

    def get(self, endpoint: str, **params):
        pause = self._interval - (time.monotonic() - self._last)
        if pause > 0:
            time.sleep(pause)
        self._last = time.monotonic()
        try:
            response = self._session.get(
                f"{BASE_URL}/{endpoint}",
                params={**params, "apiKey": self._key, "language": "en"},
                timeout=30,
            )
        except requests.RequestException:
            # requests exceptions can contain the URL including apiKey.
            raise OddsPapiError(f"OddsPapi {endpoint}: connection failed") from None
        if response.status_code != 200:
            raise OddsPapiError(f"OddsPapi {endpoint}: HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError:
            raise OddsPapiError(f"OddsPapi {endpoint}: invalid JSON") from None


def read_api_key(path: Path | None = None) -> str:
    value = os.getenv("ODDSPAPI_API_KEY", "")
    if path is not None and path.is_file():
        value = path.read_text(encoding="utf-8-sig").strip()
    if not value:
        raise OddsPapiError(
            "Set ODDSPAPI_API_KEY or save a key in the season's private/oddspapi.key"
        )
    return value


@dataclass(frozen=True)
class PropQuote:
    fixture_id: str
    player_id: str  # provider ID, not a EuroLeague player code
    player_name: str
    bookmaker: str
    market_id: str
    components: tuple[str, ...]
    line: float
    over: float
    under: float
    changed_at: str


def _components(market: dict) -> tuple[str, ...]:
    if market.get("sportId") != 11 or market.get("playerProp") is not True:
        return ()
    period = str(market.get("period", "")).lower().strip()
    name = str(market.get("marketName", "")).lower()
    if period not in FULL_GAME_PERIODS or re.search(
        r"quarter|half|regular time|regulation|double|triple|first|most|three.pointer|field goal",
        name,
    ):
        return ()
    found = tuple(c for c in COMPONENTS if re.search(rf"\b{c}\b", name))
    # Do not accidentally interpret points+steals as points alone.
    if re.search(r"steal|block|turnover|fantasy", name):
        return ()
    return found


def parse_board(board: dict, catalogue: list[dict], captured_at: str):
    """Only complete, active pairs from the same book/player/market.

    Unpaired milestone ladders are counted but not used: their margin cannot
    be removed from one price. Integer over/under lines have a push and are
    rejected unless a Yes/No milestone specifies an unambiguous threshold.
    """
    definitions = {str(m["marketId"]): m for m in catalogue}
    quotes, counts = [], Counter()
    for book, feed in (board.get("bookmakerOdds") or {}).items():
        if feed.get("bookmakerIsActive") is False or feed.get("suspended") is True:
            counts["suspended_books"] += 1
            continue
        for mid, market in feed.get("markets", {}).items():
            definition = definitions.get(str(mid), {})
            components = _components(definition)
            if not components or market.get("marketActive") is False:
                continue
            outcomes = {str(o["outcomeId"]): o["outcomeName"] for o in definition["outcomes"]}
            pairs = defaultdict(dict)
            names = {}
            inconsistent_names = set()
            for oid, outcome in market.get("outcomes", {}).items():
                label = str(outcomes.get(str(oid), "")).lower().strip()
                side = (
                    "over"
                    if label.startswith("over")
                    else ("under" if label.startswith("under") else label)
                )
                for pid, selection in outcome.get("players", {}).items():
                    if not isinstance(selection, dict) or not selection.get("playerName"):
                        continue
                    counts["player_selections"] += 1
                    if selection.get("active") is not True:
                        counts["inactive_selections"] += 1
                        continue
                    try:
                        price = float(selection["price"])
                    except (KeyError, TypeError, ValueError):
                        counts["invalid_prices"] += 1
                        continue
                    if not math.isfinite(price) or price <= 1:
                        counts["invalid_prices"] += 1
                        continue
                    pairs[str(pid)][side] = (price, selection.get("changedAt") or captured_at)
                    if str(pid) in names and name_key(names[str(pid)]) != name_key(
                        selection["playerName"]
                    ):
                        inconsistent_names.add(str(pid))
                    names[str(pid)] = selection["playerName"]
            for pid, sides in pairs.items():
                if pid in inconsistent_names:
                    counts["inconsistent_player_names"] += 1
                    continue
                try:
                    line = float(definition["handicap"])
                except (KeyError, TypeError, ValueError):
                    counts["unknown_threshold"] += 1
                    continue
                if "yes" in sides and "no" in sides:
                    # Only a positively stated milestone such as '10+ points'.
                    label = str(definition["marketName"])
                    milestone = re.search(r"(\d+)\s*\+", label)
                    if not milestone:
                        counts["unknown_threshold"] += 1
                        continue
                    line = int(milestone.group(1)) - 0.5
                    sides = {"over": sides["yes"], "under": sides["no"]}
                if "over" not in sides or "under" not in sides:
                    counts["unpaired_markets"] += 1
                    continue
                if not math.isfinite(line) or line < 0 or not math.isclose(line % 1, 0.5):
                    counts["push_or_unknown_threshold"] += 1
                    continue
                try:
                    changed = max(utc(sides[s][1]) for s in ("over", "under"))
                except (ValueError, TypeError):
                    counts["invalid_timestamps"] += 1
                    continue
                quotes.append(
                    PropQuote(
                        str(board["fixtureId"]),
                        pid,
                        names[pid],
                        book,
                        str(mid),
                        components,
                        line,
                        sides["over"][0],
                        sides["under"][0],
                        changed.isoformat(),
                    )
                )
    counts["paired_quotes"] = len(quotes)
    return quotes, dict(counts)


def fetch_snapshot(client: OddsPapiClient, games: list[dict], *, max_fixtures: int = 10) -> dict:
    """A bounded coverage audit: two discovery calls plus at most N boards."""
    if not games:
        raise OddsPapiError("No games in this round")
    if max_fixtures < 1:
        raise OddsPapiError("max-fixtures must be positive")
    starts = [game_time(g) for g in games]
    catalogue = client.get("markets")
    fixtures = client.get(
        "fixtures",
        sportId=11,
        tournamentId=138,
        **{
            "from": (min(starts) - timedelta(days=1)).isoformat(),
            "to": (max(starts) + timedelta(days=1)).isoformat(),
        },
    )
    if not isinstance(catalogue, list) or not isinstance(fixtures, list):
        raise OddsPapiError("OddsPapi returned an unexpected catalogue or fixtures schema")
    fixtures = [
        f
        for f in fixtures
        if f.get("sportId") == 11
        and f.get("tournamentId") == 138
        and min(starts).date() <= utc(f["startTime"]).date() <= max(starts).date()
    ]
    fixtures.sort(key=lambda f: (f["startTime"], str(f["fixtureId"])))
    boards, errors = [], []
    for f in fixtures[:max_fixtures]:
        try:
            board = client.get("odds", fixtureId=f["fixtureId"], oddsFormat="decimal")
            if not isinstance(board, dict) or board.get("fixtureId") != f["fixtureId"]:
                raise OddsPapiError("OddsPapi odds: unexpected fixture schema")
            boards.append(board)
        except OddsPapiError as exc:
            errors.append({"fixture_id": f["fixtureId"], "error": str(exc)})
            if "HTTP 401" in str(exc) or "HTTP 403" in str(exc) or "HTTP 429" in str(exc):
                break  # do not burn more quota on an auth/limit error
    return {
        "experiment": EXPERIMENT,
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "round": games[0]["round"],
        "catalogue": [m for m in catalogue if m.get("sportId") == 11 and m.get("playerProp")],
        "fixtures": fixtures,
        "boards": boards,
        "errors": errors,
        "fixture_limit": max_fixtures,
    }


def name_key(name: str) -> tuple[str, ...]:
    """Allow surname-first order, accents and punctuation; no fuzzy surnames."""
    return tuple(sorted(normalise_name(name).split()))


def serialise_quotes(quotes: list[PropQuote]) -> list[dict]:
    return [asdict(q) for q in quotes]
