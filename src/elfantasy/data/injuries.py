"""Availability: who is injured, suspended, or otherwise not playing.

This is the highest-value and lowest-quality data in the whole pipeline. There
is no official EuroLeague injury report equivalent to the NBA's, so the truth
lives in club announcements, pre-game press conferences and local press -- which
means it arrives late, in several languages, and sometimes wrong.

The design consequence is that availability is a *pluggable list of providers*
with an explicit trust order, not a single scraper:

1. :class:`ManualInjuryProvider` -- a YAML file you maintain. Highest trust,
   because you can encode what you read in a press conference ten minutes ago.
2. :class:`FeedInjuryProvider`   -- a JSON endpoint you point at a paid or
   community feed.
3. :class:`HtmlInjuryProvider`   -- a configurable HTML table scraper, for
   sites whose terms permit it.

Merging is by trust order: an earlier provider's status wins.

Please check a site's terms of service and robots.txt before scraping it, and
prefer an official or licensed feed where one exists. No scraper targets are
shipped enabled by default for that reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml
from bs4 import BeautifulSoup

from elfantasy.data.http import HttpClient
from elfantasy.models import Availability

log = logging.getLogger(__name__)


@dataclass
class InjuryRecord:
    player_name: str
    team_code: str | None
    status: Availability
    note: str | None = None
    source: str = ""
    player_id: str | None = None


class InjuryProvider(Protocol):
    name: str

    def fetch(self) -> list[InjuryRecord]:  # pragma: no cover - protocol
        ...


@dataclass
class ManualInjuryProvider:
    """Reads a hand-maintained YAML file.

    Format (see ``examples/injuries.yaml``)::

        - player: "Surname, Name"
          team: RMB
          status: out          # out | doubtful | questionable | probable | active
          note: "ankle, out 2-3 weeks (club statement 12 Jan)"
    """

    path: Path
    name: str = "manual"

    def fetch(self) -> list[InjuryRecord]:
        if not Path(self.path).exists():
            return []
        raw = yaml.safe_load(Path(self.path).read_text(encoding="utf-8")) or []
        out = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            name = row.get("player") or row.get("name")
            if not name:
                continue
            out.append(
                InjuryRecord(
                    player_name=str(name),
                    team_code=row.get("team"),
                    status=Availability.parse(str(row.get("status", "out"))),
                    note=row.get("note"),
                    source=self.name,
                    player_id=row.get("player_id"),
                )
            )
        log.info("manual injury file: %d records", len(out))
        return out


@dataclass
class FeedInjuryProvider:
    """Reads a JSON endpoint returning a list of availability records.

    Configure ``field_map`` to match whatever the feed calls its fields.
    """

    http: HttpClient
    url: str
    field_map: dict[str, str] = field(
        default_factory=lambda: {
            "player": "player",
            "team": "team",
            "status": "status",
            "note": "note",
        }
    )
    name: str = "feed"

    def fetch(self) -> list[InjuryRecord]:
        payload = self.http.get_json(self.url)
        if payload is None:
            return []
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get(self.field_map["player"])
            if not name:
                continue
            out.append(
                InjuryRecord(
                    player_name=str(name),
                    team_code=row.get(self.field_map["team"]),
                    status=Availability.parse(str(row.get(self.field_map["status"], ""))),
                    note=row.get(self.field_map.get("note", "note")),
                    source=self.name,
                )
            )
        log.info("%s: %d records", self.name, len(out))
        return out


@dataclass
class HtmlInjuryProvider:
    """Generic HTML table scraper.

    ``row_selector`` picks the rows; ``columns`` maps a field name to the
    zero-based cell index. Nothing is hard-coded to a particular site, so you
    supply a target you are permitted to scrape.
    """

    http: HttpClient
    url: str
    row_selector: str = "table tbody tr"
    columns: dict[str, int] = field(default_factory=lambda: {"player": 0, "team": 1, "status": 2})
    name: str = "html"

    def fetch(self) -> list[InjuryRecord]:
        body = self.http.get(self.url)
        if not body:
            return []
        soup = BeautifulSoup(body, "lxml")
        out = []
        for tr in soup.select(self.row_selector):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if not cells:
                continue

            def cell(key: str, cells: list[str] = cells) -> str | None:
                idx = self.columns.get(key)
                return cells[idx] if idx is not None and idx < len(cells) else None

            name = cell("player")
            if not name:
                continue
            out.append(
                InjuryRecord(
                    player_name=name,
                    team_code=cell("team"),
                    status=Availability.parse(cell("status") or ""),
                    note=cell("note"),
                    source=self.name,
                )
            )
        log.info("%s (%s): %d records", self.name, self.url, len(out))
        return out


# --------------------------------------------------------------------------
# Merging and name matching
# --------------------------------------------------------------------------
def merge(providers: list[InjuryProvider]) -> list[InjuryRecord]:
    """Merge in trust order -- the first provider to report a player wins."""

    seen: set[str] = set()
    merged: list[InjuryRecord] = []
    for provider in providers:
        try:
            records = provider.fetch()
        except Exception as exc:  # noqa: BLE001 - a broken source must not kill the run
            log.error("injury provider %s failed: %s", getattr(provider, "name", provider), exc)
            continue
        for rec in records:
            key = normalise_name(rec.player_name)
            if key in seen:
                continue
            seen.add(key)
            merged.append(rec)
    return merged


def normalise_name(name: str) -> str:
    """Collapse the many ways a player's name is written into one key.

    Handles ``"Surname, Name"`` versus ``"Name Surname"``, accents, punctuation
    and casing. Not perfect -- transliteration differences will still slip
    through, which is why unmatched records are reported rather than silently
    dropped.
    """

    import unicodedata

    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace(".", " ").replace("-", " ").replace("'", "")
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first} {last}"
    parts = sorted(p for p in s.lower().split() if len(p) > 1)
    return " ".join(parts)


def apply_to_players(records: list[InjuryRecord], players: dict[str, Any]) -> tuple[int, list[str]]:
    """Stamp availability onto the player objects. Returns (matched, unmatched)."""

    index: dict[str, list[Any]] = {}
    for p in players.values():
        index.setdefault(normalise_name(p.name), []).append(p)

    matched = 0
    unmatched: list[str] = []
    for rec in records:
        targets = []
        if rec.player_id and rec.player_id in players:
            targets = [players[rec.player_id]]
        else:
            candidates = index.get(normalise_name(rec.player_name), [])
            if rec.team_code:
                narrowed = [p for p in candidates if p.team_code == rec.team_code]
                candidates = narrowed or candidates
            targets = candidates[:1]

        if not targets:
            unmatched.append(rec.player_name)
            continue
        for p in targets:
            p.status = rec.status
            p.status_note = rec.note
            matched += 1

    if unmatched:
        log.warning(
            "%d injury records did not match a player (check spelling/transliteration): %s",
            len(unmatched),
            ", ".join(unmatched[:8]),
        )
    return matched, unmatched


def default_providers(http: HttpClient, manual_path: Path) -> list[InjuryProvider]:
    """The shipped default: the manual file only.

    Add a feed or an HTML provider here once you have a source you are
    permitted to use. Trust order is the list order.
    """

    return [ManualInjuryProvider(path=manual_path)]
