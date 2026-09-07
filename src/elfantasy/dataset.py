"""The bundle of inputs a projection needs, and how to load/save it.

Keeping this as one plain container means the engine has no idea whether the
data came from a live feed, a SQLite cache, or a JSON fixture in ``examples/``.
That is what makes ``--offline`` mode and the tests possible.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from elfantasy.models import (
    Availability,
    BoxScore,
    Game,
    GameOdds,
    Player,
    PlayerProp,
    Position,
    Team,
)


@dataclass
class Dataset:
    season: str = ""
    teams: dict[str, Team] = field(default_factory=dict)
    players: dict[str, Player] = field(default_factory=dict)
    games: list[Game] = field(default_factory=list)
    boxscores: list[BoxScore] = field(default_factory=list)
    odds: dict[str, GameOdds] = field(default_factory=dict)
    props: list[PlayerProp] = field(default_factory=list)
    fetched_at: datetime | None = None
    sources: dict[str, str] = field(default_factory=dict)

    # --- queries -----------------------------------------------------------
    def team_players(self, team_code: str) -> list[Player]:
        return [p for p in self.players.values() if p.team_code == team_code]

    def games_in_round(self, round_no: int) -> list[Game]:
        return [g for g in self.games if g.round == round_no]

    def team_game(self, team_code: str, round_no: int) -> Game | None:
        for g in self.games_in_round(round_no):
            if team_code in (g.home_code, g.away_code):
                return g
        return None

    def team_games(self, team_code: str, rounds: list[int]) -> list[Game]:
        wanted = set(rounds)
        return [
            g for g in self.games if g.round in wanted and team_code in (g.home_code, g.away_code)
        ]

    def played_boxscores(self) -> list[BoxScore]:
        played = {g.game_id for g in self.games if g.played}
        return [b for b in self.boxscores if b.game_id in played]

    def boxscores_by_player(self) -> dict[str, list[BoxScore]]:
        out: dict[str, list[BoxScore]] = defaultdict(list)
        for b in self.boxscores:
            out[b.player_id].append(b)
        return out

    def props_for_game(self, game_id: str) -> dict[str, list[PlayerProp]]:
        out: dict[str, list[PlayerProp]] = defaultdict(list)
        for p in self.props:
            if p.game_id == game_id:
                out[p.player_id].append(p)
        return out

    def current_round(self) -> int:
        played = [g.round for g in self.games if g.played]
        return (max(played) + 1) if played else 1

    def last_game_before(self, team_code: str, round_no: int) -> Game | None:
        prior = [
            g for g in self.games if g.round < round_no and team_code in (g.home_code, g.away_code)
        ]
        return max(prior, key=lambda g: g.round) if prior else None

    def annotate_home_flags(self) -> None:
        """Tag each box score with whether it was a home game.

        Several rate/context features need this and the raw feeds do not always
        provide it on the player line.
        """

        index = {g.game_id: g for g in self.games}
        for bs in self.boxscores:
            g = index.get(bs.game_id)
            if g is not None:
                object.__setattr__(bs, "_home", g.is_home(bs.team_code))

    def summary(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "teams": len(self.teams),
            "players": len(self.players),
            "games": len(self.games),
            "played": sum(1 for g in self.games if g.played),
            "boxscores": len(self.boxscores),
            "odds": len(self.odds),
            "props": len(self.props),
            "sources": self.sources,
        }

    # --- persistence -------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        def dt(v: datetime | None) -> str | None:
            return v.isoformat() if v else None

        return {
            "season": self.season,
            "fetched_at": dt(self.fetched_at),
            "sources": self.sources,
            "teams": [vars(t) for t in self.teams.values()],
            "players": [
                {
                    **vars(p),
                    "position": p.position.value,
                    "status": p.status.value,
                }
                for p in self.players.values()
            ],
            "games": [{**vars(g), "tipoff": dt(g.tipoff)} for g in self.games],
            "boxscores": [
                {k: v for k, v in vars(b).items() if not k.startswith("_")} for b in self.boxscores
            ],
            "odds": [{**vars(o), "captured_at": dt(o.captured_at)} for o in self.odds.values()],
            "props": [vars(p) for p in self.props],
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> Dataset:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json(raw)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Dataset:
        def dt(v: str | None) -> datetime | None:
            return datetime.fromisoformat(v) if v else None

        ds = cls(season=raw.get("season", ""), fetched_at=dt(raw.get("fetched_at")))
        ds.sources = raw.get("sources", {})
        ds.teams = {t["code"]: Team(**t) for t in raw.get("teams", [])}
        for p in raw.get("players", []):
            p = dict(p)
            p["position"] = Position.parse(p.get("position"))
            p["status"] = Availability.parse(p.get("status"))
            ds.players[p["player_id"]] = Player(**p)
        for g in raw.get("games", []):
            g = dict(g)
            g["tipoff"] = dt(g.get("tipoff"))
            ds.games.append(Game(**g))
        ds.boxscores = [BoxScore(**b) for b in raw.get("boxscores", [])]
        for o in raw.get("odds", []):
            o = dict(o)
            o["captured_at"] = dt(o.get("captured_at"))
            ds.odds[o["game_id"]] = GameOdds(**o)
        ds.props = [PlayerProp(**p) for p in raw.get("props", [])]
        ds.annotate_home_flags()
        return ds
