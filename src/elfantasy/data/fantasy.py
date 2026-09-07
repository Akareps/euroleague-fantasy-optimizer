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
from elfantasy.models import Player, Position, Squad

log = logging.getLogger(__name__)


def load_prices_csv(path: Path | str, players: dict[str, Player]) -> int:
    """Apply prices from a CSV to the player table. Returns the number applied.

    Accepted columns: ``player_id`` or ``player`` (name), ``price``, and
    optionally ``position``, ``ownership`` and ``purchase_price``.
    """

    path = Path(path)
    if not path.exists():
        log.warning("price file not found: %s", path)
        return 0

    index = {normalise_name(p.name): pid for pid, p in players.items()}
    applied, unmatched = 0, []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
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
            applied += 1

    if unmatched:
        log.warning("%d price rows unmatched, e.g. %s", len(unmatched), ", ".join(unmatched[:6]))
    log.info("applied %d prices from %s", applied, path)
    return applied


def write_price_template(path: Path | str, players: dict[str, Player]) -> Path:
    """Write a CSV pre-filled with every player, ready for you to add prices.

    Far less painful than typing 200 names, and it guarantees the ids match.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(players.values(), key=lambda p: (p.team_code, p.name))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["player_id", "player", "team", "position", "price", "ownership"])
        for p in rows:
            writer.writerow([p.player_id, p.name, p.team_code, p.position.value, p.price or "", ""])
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


def load_squad_yaml(path: Path | str, players: dict[str, Player]) -> Squad:
    """Load your current roster.

    Format (see ``examples/my_squad.yaml``)::

        bank: 1.5
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

    return Squad(player_ids=ids, bank=float(raw.get("bank", 0.0)), purchase_prices=paid)


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
