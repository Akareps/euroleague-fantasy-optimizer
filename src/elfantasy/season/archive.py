"""Archive: time-stamped copies of every outside source a backtest may want later.

Nothing here feeds the model. Run ``elfantasy season archive`` whenever a round
is prepared and again on game day, so that when a new idea comes along the
data to test it is already on disk.

    archive/raw/<source>/<UTC stamp>_<name>.gz   each response exactly as received
    archive/once/<source>/<name>.gz              fetched once (finished games)
    archive/tables/<table>.jsonl                 parsed rows, appended, each with
                                                 ``captured_at`` and ``source``

Tables:
    odds_lines     moneylines of upcoming (``kind: fixture``) and finished
                   (``kind: result``, the closing line) games, EuroLeague and the
                   domestic leagues of its clubs, with scores for finished games
    odds_markets   over/under and handicap consensus per EuroLeague game
    injuries       the Basketball Sphere tracker, club by club
    news           RotoWire's EuroLeague news items
    prices_public  the full fantasy price list published by Basketball Sphere
                   (it can lag the app by a round)

Plus EuroLeague play-by-play and shot data per finished game (``once/euroleague``).

All sources allow this in robots.txt (checked 17 Nov 2026). Requests carry the
project's own user agent and are paced per host.
"""

from __future__ import annotations

import gzip
import html as htmllib
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, mean
from urllib.parse import urlparse

import requests

from elfantasy.data.http import USER_AGENT

BETEXPLORER = "https://www.betexplorer.com"
LEAGUES = {  # competition -> BetExplorer path
    "euroleague": "basketball/europe/euroleague",
    "acb": "basketball/spain/acb",
    "greece": "basketball/greece/basket-league",
    "turkey": "basketball/turkey/super-lig",
    "italy": "basketball/italy/lega-a",
    "germany": "basketball/germany/bbl",
    "france": "basketball/france/lnb",
    "israel": "basketball/israel/super-league",
    "aba": "basketball/europe/admiralbet-aba-league",
    "lithuania": "basketball/lithuania/lkl",
}
TRACKER = "https://basketballsphere.com/en/injuries/euroleague/"
PRICES = "https://basketballsphere.com/en/euroleague-fantasy-player-prices/"
ROTOWIRE = "https://www.rotowire.com/euro/news.php"
LIVE_API = "https://live.euroleague.net/api"
TOTAL_SD = 16.0  # points: spread of a game's total around the market line
MARGIN_SD = 11.5  # points: spread of a game's margin around the market line

# BetExplorer's EuroLeague names -> club codes (a name not here keeps code None).
EL_CODES = {
    "anadolu efes": "IST", "baskonia": "BAS", "barcelona": "BAR", "bayern": "MUN",
    "besiktas": "BES", "crvena zvezda": "RED", "dubai": "DUB", "fenerbahce": "ULK",
    "hapoel tel aviv": "HTA", "lyon-villeurbanne": "ASV", "maccabi tel aviv": "TEL",
    "olimpia milano": "MIL", "olympiacos": "OLY", "panathinaikos": "PAN", "paris": "PRS",
    "partizan": "PAR", "real madrid": "MAD", "valencia": "PAM", "virtus bologna": "VIR",
    "zalgiris": "ZAL",
}  # fmt: skip
TRACKER_CLUBS = {
    "Anadolu Efes", "Asvel", "Barcelona", "Baskonia", "Bayern", "Besiktas", "Crvena Zvezda",
    "Dubai", "Fenerbahce", "Hapoel Tel Aviv", "Maccabi Tel Aviv", "Milano", "Olympiacos",
    "Panathinaikos", "Paris", "Partizan", "Real Madrid", "Valencia", "Virtus", "Zalgiris",
}  # fmt: skip
STATUSES = {"Out", "Doubtful", "Day to day", "Cleared"}


def el_code(name: str) -> str | None:
    low = name.lower()
    return next((c for k, c in EL_CODES.items() if low.startswith(k)), None)


# ======================================================================= storage
@dataclass
class Archive:
    root: Path
    stamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    @property
    def captured_at(self) -> str:
        return (
            datetime.strptime(self.stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
        )

    def raw(self, source: str, name: str, text: str) -> Path:
        return self._write(self.root / "raw" / source / f"{self.stamp}_{name}.gz", text)

    def once_path(self, source: str, name: str) -> Path:
        return self.root / "once" / source / f"{name}.gz"

    def once(self, source: str, name: str, text: str) -> Path:
        return self._write(self.once_path(source, name), text)

    @staticmethod
    def _write(path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def append(self, table: str, source: str, rows: list[dict]) -> int:
        path = self.root / "tables" / f"{table}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(
                    json.dumps({"captured_at": self.captured_at, "source": source, **row},
                               ensure_ascii=False) + "\n"
                )  # fmt: skip
        return len(rows)

    def read(self, table: str) -> list[dict]:
        path = self.root / "tables" / f"{table}.jsonl"
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


class Fetcher:
    """Plain GETs with the project's user agent: at most one per ``interval``
    seconds per host (``slow`` hosts get their own, longer pause), and two
    retries after a dropped connection, a timeout or a 429/5xx answer."""

    SLOW = {"www.betexplorer.com": 3.0}  # it drops connections at 1.5 s
    BACKOFF = (10.0, 30.0)

    def __init__(self, interval: float = 1.5, session: requests.Session | None = None,
                 slow: dict[str, float] | None = None):  # fmt: skip
        self.interval = interval
        self.slow = self.SLOW if slow is None else slow
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last: dict[str, float] = {}

    def _pace(self, host: str) -> None:
        pause = self.slow.get(host, self.interval) - (time.monotonic() - self._last.get(host, 0.0))
        if pause > 0:
            time.sleep(pause)
        self._last[host] = time.monotonic()

    def get(self, url: str, **headers: str) -> str:
        host = urlparse(url).netloc
        for attempt in range(len(self.BACKOFF) + 1):
            self._pace(host)
            try:
                response = self.session.get(url, headers=headers, timeout=30)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
                if attempt == len(self.BACKOFF):
                    raise
                time.sleep(self.BACKOFF[attempt])
                continue
            response.raise_for_status()
            return response.text
        raise AssertionError("unreachable")


# ======================================================================= parsers
def _text(fragment: str) -> str:
    return htmllib.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _lines(page: str) -> list[str]:
    page = re.sub(r"<script.*?</script>|<style.*?</style>", " ", page, flags=re.S)
    return [ln.strip() for ln in htmllib.unescape(re.sub(r"<[^>]+>", "\n", page)).splitlines() if ln.strip()]  # fmt: skip


def parse_betexplorer_table(page: str) -> list[dict]:
    """Rows of a BetExplorer league page (fixtures or results): round, match id,
    teams, score and winner (results), the 1/2 odds and the date text."""
    out, rnd = [], None
    for block in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        header = re.search(r"(\d+)\. Round", block)
        if header and "<th" in block:
            rnd = int(header.group(1))
            continue
        link = re.search(
            r'href="(/basketball/[^"]+?/([A-Za-z0-9]{8})/)"[^>]*class="in-match">(.*?)</a>', block, re.S
        )  # fmt: skip
        if not link:
            continue
        sides = link.group(3).split("</span> - <span>")
        if len(sides) != 2:
            continue
        home, away = _text(sides[0]), _text(sides[1])
        score = re.search(r">\s*(\d+)\s*:\s*(\d+)", block)
        odds = [float(x) for x in re.findall(r'data-odd="([\d.]+)"', block)[:2]]
        when = re.search(r'table-main__datetime">([^<]*)<', block) or re.search(
            r'h-text-right[^"]*">([^<]*)<', block
        )
        out.append({
            "round": rnd, "match_id": link.group(2), "url": link.group(1),
            "home": home, "away": away,
            "home_score": int(score.group(1)) if score else None,
            "away_score": int(score.group(2)) if score else None,
            "winner": "home" if "<strong>" in sides[0] else ("away" if "<strong>" in sides[1] else None),
            "home_odds": odds[0] if len(odds) == 2 else None,
            "away_odds": odds[1] if len(odds) == 2 else None,
            "when": when.group(1).strip() if when else None,
        })  # fmt: skip
    return out


def parse_betexplorer_market(body: str) -> list[dict]:
    """One row per bookmaker of a BetExplorer over/under or handicap table:
    the line (total or home handicap) and the two prices, active or not."""
    payload = json.loads(body)
    rows = []
    for block in re.findall(r"<tr data-bid.*?</tr>", payload.get("odds", ""), re.S):
        line = re.search(r'table-main__doubleparameter">([-\d.]+)<', block)
        odds = re.findall(r'class="([^"]*)"\s+data-odd="([\d.]+)"', block)
        book = re.search(r'data-bid="(\d+)"', block)
        if not line or len(odds) != 2:
            continue
        rows.append({
            "book": book.group(1) if book else None, "line": float(line.group(1)),
            "first": float(odds[0][1]), "second": float(odds[1][1]),
            "active": not any("inactive" in c for c, _ in odds),
        })  # fmt: skip
    return rows


def _main_line(rows: list[dict]) -> tuple[float, int, float] | None:
    """The most-quoted active line, its book count and the margin-free chance
    of its first outcome (over, or home covering)."""
    by: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r["active"] and r["first"] > 1 and r["second"] > 1:
            by[r["line"]].append((1 / r["first"]) / (1 / r["first"] + 1 / r["second"]))
    if not by:
        return None
    line = max(by, key=lambda x: (len(by[x]), -abs(mean(by[x]) - 0.5)))
    return line, len(by[line]), mean(by[line])


def total_consensus(rows: list[dict]) -> dict | None:
    """Expected game total: the most-quoted line moved by its over/under imbalance."""
    main = _main_line(rows)
    if not main:
        return None
    line, books, p_over = main
    return {"line": line, "books": books, "p_first": round(p_over, 4),
            "expected": round(line + TOTAL_SD * NormalDist().inv_cdf(p_over), 1)}  # fmt: skip


def handicap_consensus(rows: list[dict]) -> dict | None:
    """Expected home margin from the most-quoted home handicap: covering -h at
    probability p means a margin of h + SD x z(p)."""
    main = _main_line(rows)
    if not main:
        return None
    line, books, p_cover = main
    return {"line": line, "books": books, "p_first": round(p_cover, 4),
            "expected": round(-line + MARGIN_SD * NormalDist().inv_cdf(p_cover), 1)}  # fmt: skip


def parse_tracker(page: str) -> tuple[str | None, list[dict]]:
    """(update note, rows) from the Basketball Sphere injury tracker's club by club tables."""
    lines = _lines(page)
    updated = next((ln for ln in lines if ln.startswith("Updated ")), None)
    rows, club, tokens = [], None, []

    def flush():
        i = 0
        while club and i + 3 < len(tokens):
            if tokens[i + 1] in STATUSES:
                player, status, injury, back = tokens[i : i + 4]
                rows.append({"club": club, "player": player, "status": status,
                             "injury": None if injury in "–-" else injury,
                             "expected_return": None if back in "–-" else back})  # fmt: skip
                i += 4
            else:
                i += 1

    started = False
    for n, ln in enumerate(lines):
        nxt = lines[n + 1] if n + 1 < len(lines) else ""
        if ln in TRACKER_CLUBS and re.match(r"(\d+ players? out|No injured players)", nxt):
            flush()
            club, tokens, started = ln, [], True
            continue
        if not started:
            continue
        if ln == "EuroLeague":
            break
        if ln in ("Player", "Status", "Injury", "Expected return") or re.match(
            r"(\d+ players? out|No injured players)", ln
        ):
            continue
        tokens.append(ln)
    flush()
    return updated, rows


_DATE = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}$")  # fmt: skip


def parse_rotowire(page: str) -> list[dict]:
    """RotoWire EuroLeague news items: player, headline, position, team,
    injury (when given), date and text."""
    text = " | ".join(_lines(page))
    items = []
    for chunk in re.split(
        r"\| ANALYSIS \| Subscribe now \| to instantly reveal our take on this news\. \|", text
    ):
        chunk = chunk.split("Compact | Expanded |")[-1]
        parts = [p.strip() for p in chunk.split(" | ") if p.strip()]
        at = next((i for i, p in enumerate(parts) if _DATE.match(p)), None)
        if at is None or at < 4 or at > 5 or len(parts) <= at + 1:
            continue
        items.append({
            "player": parts[0], "headline": parts[1], "pos": parts[2], "team": parts[3],
            "injury": parts[4] if at == 5 else None, "date": parts[at],
            "text": " ".join(parts[at + 1 :])[:600],
        })  # fmt: skip
    return items


def parse_prices(page: str) -> list[dict]:
    """The full fantasy price list on Basketball Sphere."""
    rows = re.findall(
        r'<tr data-pos="(\w)" data-club="([^"]+)" data-price="([\d.]+)">.*?<td class="nm">(.*?)</td>',
        page, re.S,
    )  # fmt: skip
    return [{"player": _text(name), "club": htmllib.unescape(club), "pos": pos, "price": float(price)}
            for pos, club, price, name in rows]  # fmt: skip


# ==================================================================== collectors
def collect_odds(
    archive: Archive, fetch: Fetcher, competitions=None, markets_ahead: int = 2
) -> int:
    """Fixtures and results pages of each league; for the EuroLeague also the
    over/under and handicap tables of the next rounds' games and, once, of
    every finished game (the closing lines)."""
    n = 0
    for comp, path in LEAGUES.items():
        if competitions and comp not in competitions:
            continue
        for kind, page in (("fixture", "fixtures"), ("result", "results")):
            try:
                body = fetch.get(f"{BETEXPLORER}/{path}/{page}/")
            except requests.RequestException:
                continue  # the next run tries again
            archive.raw("betexplorer", f"{comp}_{page}", body)
            rows = parse_betexplorer_table(body)
            for r in rows:
                r.update(competition=comp, kind=kind)
                if comp == "euroleague":
                    r.update(home_code=el_code(r["home"]), away_code=el_code(r["away"]))
            n += archive.append("odds_lines", "betexplorer", rows)
            if comp != "euroleague":
                continue
            if kind == "fixture":
                upcoming = sorted({r["round"] for r in rows if r["round"]})[:markets_ahead]
                todo = [(r, False) for r in rows if r["round"] in upcoming]
            else:
                todo = [(r, True) for r in rows]
            n += _collect_markets(archive, fetch, path, todo)
    return n


def _collect_markets(archive: Archive, fetch: Fetcher, path: str, todo) -> int:
    referer = f"{BETEXPLORER}/{path}/"
    out = []
    for row, closed in todo:
        for market in ("ou", "ah"):
            name = f"{row['match_id']}_{market}"
            if closed and archive.once_path("betexplorer_closing", name).is_file():
                continue
            try:
                body = fetch.get(
                    f"{BETEXPLORER}/match-odds-old/{row['match_id']}/0/{market}/1/en/?sportname=basketball",
                    **{"X-Requested-With": "XMLHttpRequest", "Referer": referer},
                )  # fmt: skip
            except requests.RequestException:
                continue  # a finished game's file stays missing, so the next run retries it
            if closed:
                archive.once("betexplorer_closing", name, body)
            else:
                archive.raw("betexplorer_markets", name, body)
            summary = (total_consensus if market == "ou" else handicap_consensus)(
                parse_betexplorer_market(body)
            )
            if summary:
                out.append({"competition": "euroleague", "kind": "result" if closed else "fixture",
                            "round": row["round"], "match_id": row["match_id"],
                            "home": row["home"], "away": row["away"],
                            "home_code": el_code(row["home"]), "away_code": el_code(row["away"]),
                            "market": "total" if market == "ou" else "home_handicap", **summary})  # fmt: skip
    return archive.append("odds_markets", "betexplorer", out)


def collect_news(archive: Archive, fetch: Fetcher) -> int:
    body = fetch.get(TRACKER)
    archive.raw("basketballsphere", "injuries", body)
    updated, rows = parse_tracker(body)
    n = archive.append("injuries", "basketballsphere", [{**r, "updated": updated} for r in rows])
    for name, url in (("news", ROTOWIRE), ("injuries", f"{ROTOWIRE}?view=injuries")):
        body = fetch.get(url)
        archive.raw("rotowire", name, body)
        n += archive.append("news", f"rotowire_{name}", parse_rotowire(body))
    return n


def collect_prices(archive: Archive, fetch: Fetcher) -> int:
    body = fetch.get(PRICES)
    archive.raw("basketballsphere", "prices", body)
    return archive.append("prices_public", "basketballsphere", parse_prices(body))


def collect_play_by_play(
    archive: Archive, fetch: Fetcher, games: dict[str, dict], season: str
) -> int:
    """Play-by-play and shot data of every finished game not yet archived."""
    n = 0
    for ident, g in sorted(games.items()):
        if not g.get("played"):
            continue
        for feed in ("PlayByPlay", "Points"):
            name = f"{ident}_{feed}"
            if archive.once_path("euroleague", name).is_file():
                continue
            body = fetch.get(f"{LIVE_API}/{feed}?gamecode={g['gameCode']}&seasoncode={season}")
            json.loads(body)  # store only a valid response
            archive.once("euroleague", name, body)
            n += 1
    return n


def round_lines(archive: Archive, round_no: int) -> tuple[dict, dict]:
    """The latest archived EuroLeague moneylines and expected totals of a round,
    keyed 'HOME-AWAY' by club code, ready for a round file."""
    latest = {}
    for table in ("odds_lines", "odds_markets"):
        for r in archive.read(table):
            if r.get("competition") == "euroleague" and r.get("round") == round_no:
                key = (table, r.get("market"), r["match_id"])
                if key not in latest or r["captured_at"] >= latest[key]["captured_at"]:
                    latest[key] = r
    lines, totals = {}, {}
    for (table, market, _), r in latest.items():
        if not (r.get("home_code") and r.get("away_code")):
            continue
        fixture = f"{r['home_code']}-{r['away_code']}"
        if table == "odds_lines" and r.get("home_odds"):
            lines[fixture] = (r["home_odds"], r["away_odds"])
        elif market == "total":
            totals[fixture] = r["expected"]
    return lines, totals


COLLECTORS = ("odds", "domestic", "news", "prices", "pbp")


def collect(archive: Archive, games: dict[str, dict], season: str, only=None,
            fetch: Fetcher | None = None, log: Callable[[str], None] = print) -> dict[str, str]:  # fmt: skip
    """Run every collector (or ``only`` those named); one failing does not stop the others."""
    fetch = fetch or Fetcher()
    jobs = {
        "odds": lambda: collect_odds(archive, fetch, competitions={"euroleague"}),
        "domestic": lambda: collect_odds(
            archive, fetch, competitions=set(LEAGUES) - {"euroleague"}
        ),
        "news": lambda: collect_news(archive, fetch),
        "prices": lambda: collect_prices(archive, fetch),
        "pbp": lambda: collect_play_by_play(archive, fetch, games, season),
    }
    out = {}
    for name in COLLECTORS:
        if only and name not in only:
            continue
        try:
            out[name] = f"{jobs[name]()} rows/files"
        except (requests.RequestException, ValueError, KeyError) as exc:
            out[name] = f"failed: {type(exc).__name__}: {str(exc)[:120]}"
        log(f"{name}: {out[name]}")
    return out
