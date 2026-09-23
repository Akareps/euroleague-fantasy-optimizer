"""Fantasy-game data: prices, ownership, and your own squad.

Prices are the constraint that makes this an optimisation problem rather than a
ranking exercise, so getting them right matters more than almost anything else
in this module. Three ways in, in order of convenience:

* :func:`load_prices_csv`   -- a two-column CSV you export or type by hand.
* :class:`FantasyApiProvider` -- the official game's own endpoint, if you have
  one that works for your season.
* :func:`load_squad_yaml`   -- your current roster, bank, and what you paid.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from elfantasy.data.http import HttpClient
from elfantasy.data.injuries import normalise_name
from elfantasy.models import Coach, Player, Position, Squad

log = logging.getLogger(__name__)


def load_prices_csv(
    path: Path | str,
    players: dict[str, Player],
    coaches: dict[str, Coach] | None = None,
) -> int:
    """Apply prices from a CSV to the player table. Returns the number applied.

    Accepted columns: ``player_id`` or ``player`` (name), ``price``, and
    optionally ``position``, ``ownership``, ``purchase_price`` and ``minutes``
    (a stated minutes estimate, see ``Player.minutes_override``).

    Rows whose position is ``HC`` are head coaches; pass ``coaches`` to collect
    them (keyed by normalised name), with the club taken from ``team``/``club``.
    """

    path = Path(path)
    if not path.exists():
        log.warning("price file not found: %s", path)
        return 0

    index = {normalise_name(p.name): pid for pid, p in players.items()}
    applied, unmatched = 0, []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if (row.get("position") or row.get("pos") or "").strip().upper() == "HC":
                if coaches is not None:
                    name = (row.get("player") or row.get("name") or "").strip()
                    cid = (row.get("coach_id") or "").strip() or normalise_name(name)
                    price = _f(row.get("price")) or 0.0
                    team = (row.get("team") or row.get("club") or "").strip()
                    coaches[cid] = Coach(coach_id=cid, name=name, team_code=team, price=price)
                    applied += 1
                continue
            pid = (row.get("player_id") or "").strip()
            if pid not in players:
                pid = index.get(normalise_name(row.get("player") or row.get("name") or ""), "")
            if not pid:
                unmatched.append(row.get("player") or row.get("player_id") or "?")
                continue
            player = players[pid]
            price = _f(row.get("price"))
            if price is not None:
                player.price = price
            own = _f(row.get("ownership"))
            if own is not None:
                player.ownership = own / 100.0 if own > 1 else own
            paid = _f(row.get("purchase_price"))
            if paid is not None:
                player.purchase_price = paid
            pos = row.get("position")
            if pos:
                player.position = Position.parse(pos)
            minutes = _f(row.get("minutes"))
            if minutes is not None:
                player.minutes_override = minutes
            applied += 1

    if unmatched:
        log.warning("%d price rows unmatched, e.g. %s", len(unmatched), ", ".join(unmatched[:6]))
    log.info("applied %d prices from %s", applied, path)
    return applied


def write_price_template(
    path: Path | str,
    players: dict[str, Player],
    coaches: dict[str, Coach] | None = None,
) -> Path:
    """Write a CSV pre-filled with every player, ready for you to add prices.

    Far less painful than typing 200 names, and it guarantees the ids match.
    Coaches, when given, are listed with position ``HC``.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(players.values(), key=lambda p: (p.team_code, p.name))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["player_id", "player", "team", "position", "price", "ownership", "minutes"]
        )
        for p in rows:
            writer.writerow(
                [p.player_id, p.name, p.team_code, p.position.value, p.price or "", "", ""]
            )
        for c in sorted((coaches or {}).values(), key=lambda c: c.team_code):
            writer.writerow(["", c.name, c.team_code, "HC", c.price or "", "", ""])
    return path


@dataclass
class FantasyApiProvider:
    """Official fantasy endpoint, if one is reachable for your season.

    The path has moved between seasons; override ``path`` rather than editing
    code. Returns ``{player_id: {"price": float, "ownership": float}}``.
    """

    http: HttpClient
    base: str
    season: str
    path: str = "/v1/fantasy/players"
    name: str = "official-fantasy"

    def fetch(self) -> dict[str, dict[str, float]]:
        url = f"{self.base.rstrip('/')}{self.path}"
        payload = self.http.get_json(url, {"seasonCode": self.season})
        if not payload:
            log.warning("no fantasy price data from %s", url)
            return {}
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        out: dict[str, dict[str, float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("code") or row.get("playerCode") or row.get("id") or "").strip()
            if not pid:
                continue
            price = _f(row.get("cr") or row.get("credits") or row.get("price"))
            own = _f(row.get("selectedBy") or row.get("ownership"))
            if price is None:
                continue
            out[pid] = {"price": price, "ownership": (own or 0.0) / 100.0 if own else 0.0}
        log.info("fetched %d fantasy prices", len(out))
        return out

    def apply(self, players: dict[str, Player]) -> int:
        data = self.fetch()
        applied = 0
        for pid, values in data.items():
            player = players.get(pid)
            if player is None:
                continue
            player.price = values["price"]
            player.ownership = values.get("ownership")
            applied += 1
        return applied


def load_squad_yaml(
    path: Path | str,
    players: dict[str, Player],
    coaches: dict[str, Coach] | None = None,
) -> Squad:
    """Load your current roster.

    Format (see ``examples/my_squad.yaml``)::

        bank: 1.5
        coach: "Saras Jasikevicius"  # optional; name or coach id
        players:
          - id: P012345          # or: name: "Surname, Name"
            paid: 12.4           # optional, only needed if you sell at cost
    """

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    index = {normalise_name(p.name): pid for pid, p in players.items()}

    ids: list[str] = []
    paid: dict[str, float] = {}
    missing: list[str] = []

    for entry in raw.get("players", []):
        if isinstance(entry, str):
            entry = {"name": entry}
        pid = str(entry.get("id") or "").strip()
        if pid not in players:
            pid = index.get(normalise_name(str(entry.get("name", ""))), "")
        if not pid:
            missing.append(str(entry.get("name") or entry.get("id") or "?"))
            continue
        ids.append(pid)
        if entry.get("paid") is not None:
            paid[pid] = float(entry["paid"])

    if missing:
        raise ValueError(
            "could not resolve these squad members to players: "
            + ", ".join(missing)
            + ". Use their player_id (see `elfantasy players`) if the name spelling differs."
        )

    coach_id = None
    coach_raw = raw.get("coach")
    if coach_raw and coaches:
        wanted = str(coach_raw).strip()
        by_name = {normalise_name(c.name): cid for cid, c in coaches.items()}
        coach_id = wanted if wanted in coaches else by_name.get(normalise_name(wanted))
        if coach_id is None:
            raise ValueError(f"could not resolve the squad's coach {wanted!r}")

    return Squad(
        player_ids=ids,
        bank=float(raw.get("bank", 0.0)),
        purchase_prices=paid,
        coach_id=coach_id,
    )


def write_squad_yaml(path: Path | str, squad: Squad, players: dict[str, Player]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for pid in squad.player_ids:
        p = players.get(pid)
        entry: dict[str, Any] = {"id": pid, "name": p.name if p else pid}
        if pid in squad.purchase_prices:
            entry["paid"] = squad.purchase_prices[pid]
        entries.append(entry)
    path.write_text(
        yaml.safe_dump(
            {"bank": squad.bank, "players": entries}, sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    return path


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
