"""Downloaded and third-party data for a season (kept out of git).

Layout under ``<data dir>/cache/seasons/<name>/``::

    prior/joined.json            opening prices joined to rosters and last season
    prior/people_<code>.json     rosters (EuroLeague API)
    prior/stats_<prev>_acc.json  last season's totals (EuroLeague API)
    prior/prices_round_01.csv    the opening price list (rank,player,club,pos,price)
    prior/http/                  cached last-season box scores (league-level fits)
    games.json                   this season's schedule and results
    boxes/round_NN.json          box scores, one file per round
    prices/round_NN.csv          a published price list valid for round NN
    out/                         projections and plans written by the commands
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from elfantasy.data.http import HttpClient
from elfantasy.data.injuries import normalise_name
from elfantasy.season.inputs import Key, SeasonInputs, key

API = "https://api-live.euroleague.net"


class SeasonDataError(RuntimeError):
    pass


class SeasonData:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @classmethod
    def for_season(cls, data_dir: Path, season: SeasonInputs) -> SeasonData:
        return cls(Path(data_dir) / "cache" / "seasons" / season.name)

    # ------------------------------------------------------------ paths
    @property
    def prior_dir(self) -> Path:
        return self.root / "prior"

    @property
    def games_file(self) -> Path:
        return self.root / "games.json"

    def box_file(self, round_no: int) -> Path:
        return self.root / "boxes" / f"round_{round_no:02d}.json"

    def price_list_file(self, round_no: int) -> Path:
        return self.root / "prices" / f"round_{round_no:02d}.csv"

    @property
    def out_dir(self) -> Path:
        d = self.root / "out"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def projections_file(self, round_no: int) -> Path:
        return self.out_dir / f"projections_round_{round_no:02d}.json"

    # ------------------------------------------------------------ reading
    def games(self) -> dict[str, dict]:
        """game identifier -> raw API record."""
        if not self.games_file.is_file():
            raise SeasonDataError(f"no schedule at {self.games_file}: run `elfantasy season fetch`")
        rows = json.loads(self.games_file.read_text(encoding="utf-8"))["data"]
        return {g["identifier"]: g for g in rows}

    def boxes(self, rounds) -> dict[int, dict[str, dict]]:
        """round -> player code -> box-score line."""
        out = {}
        for r in rounds:
            f = self.box_file(r)
            if not f.is_file():
                raise SeasonDataError(
                    f"no box scores for round {r} at {f}: run `elfantasy season fetch`"
                )
            out[r] = {ln["player_id"]: ln for ln in json.loads(f.read_text(encoding="utf-8"))}
        return out

    def price_lists(self, season: SeasonInputs) -> dict[int, dict[Key, float]]:
        """Published price lists by the round they are valid for (not round 1:
        opening prices come with the prior)."""
        out = {}
        for f in sorted((self.root / "prices").glob("round_[0-9][0-9].csv")):
            rnd = int(f.stem.split("_")[1])
            with f.open(encoding="utf-8") as fh:
                out[rnd] = {
                    key(season.clubs.get(r["club"], r["club"]), r["player"]): float(r["price"])
                    for r in csv.DictReader(fh)
                }
        return out

    def played_rounds(self) -> list[int]:
        """Rounds whose every game is final and whose box scores are on disk."""
        by_round: dict[int, list[bool]] = {}
        for g in self.games().values():
            by_round.setdefault(int(g["round"]), []).append(bool(g["played"]))
        return [
            r for r, done in sorted(by_round.items()) if all(done) and self.box_file(r).is_file()
        ]

    # ------------------------------------------------------------ fetching
    def _http(self) -> HttpClient:
        return HttpClient(cache_dir=self.root / "http", ttl_seconds=3600, min_interval=1.5)

    def fetch_games(self, season: SeasonInputs) -> int:
        body = self._http().get(f"{API}/v2/competitions/E/seasons/{season.code}/games", None)
        if body is None:
            raise SeasonDataError("could not download the schedule")
        self.games_file.parent.mkdir(parents=True, exist_ok=True)
        self.games_file.write_text(body, encoding="utf-8")
        return len(json.loads(body)["data"])

    def fetch_boxes(self, season: SeasonInputs, round_no: int) -> int:
        from elfantasy.data.euroleague import EuroleagueClient

        client = EuroleagueClient(self._http(), season=season.code)
        games = [g for g in client.fetch_games() if g.round == round_no]
        if not games or not all(g.played for g in games):
            raise SeasonDataError(f"round {round_no} is not finished yet")
        lines = []
        for g in games:
            lines += [asdict(b) for b in client.fetch_game_boxscore(g)]
        self.box_file(round_no).parent.mkdir(parents=True, exist_ok=True)
        self.box_file(round_no).write_text(json.dumps(lines), encoding="utf-8")
        return len(lines)

    def fetch_prior(self, season: SeasonInputs) -> list[str]:
        """Rosters and last season's totals (three throttled, cached requests)."""
        http = HttpClient(cache_dir=self.prior_dir / "http", ttl_seconds=6 * 3600, min_interval=2.0)
        targets = {
            f"people_{season.code}.json": (
                f"{API}/v2/competitions/E/seasons/{season.code}/people",
                {"personType": "J"},
            ),
            f"stats_{season.previous_code}_acc.json": (
                f"{API}/v3/competitions/E/statistics/players/traditional",
                {"seasonMode": "Single", "seasonCode": season.previous_code,
                 "statisticMode": "Accumulated", "limit": 1000},
            ),
        }  # fmt: skip
        saved = []
        self.prior_dir.mkdir(parents=True, exist_ok=True)
        for name, (url, params) in targets.items():
            body = http.get(url, params)
            if body is None:
                raise SeasonDataError(f"could not download {url}")
            (self.prior_dir / name).write_text(body, encoding="utf-8")
            saved.append(name)
        return saved

    def refresh_rosters(self, season: SeasonInputs) -> int:
        """Today's registrations from the league's roster feed (not cached), with
        a dated copy in the archive. Clubs sign and release players all season."""
        http = HttpClient(cache_dir=self.prior_dir / "http", ttl_seconds=0, min_interval=2.0)
        body = http.get(
            f"{API}/v2/competitions/E/seasons/{season.code}/people", {"personType": "J"}
        )
        if body is None:
            raise SeasonDataError("could not download the rosters")
        self.prior_dir.mkdir(parents=True, exist_ok=True)
        (self.prior_dir / f"people_{season.code}.json").write_text(body, encoding="utf-8")
        from elfantasy.season.archive import Archive

        Archive(self.root / "archive").raw("rosters", f"people_{season.code}", body)
        return len(json.loads(body)["data"])

    def _public_prices(self) -> list[dict]:
        """The latest archived public price list (it can lag the app and its clubs)."""
        path = self.root / "archive" / "tables" / "prices_public.jsonl"
        if not path.is_file():
            return []
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        latest = max((r["captured_at"] for r in rows), default=None)
        return [r for r in rows if r["captured_at"] == latest]

    def _next_round(self) -> int | None:
        if not self.games_file.is_file():
            return None
        open_rounds = [int(g["round"]) for g in self.games().values() if not g["played"]]
        return min(open_rounds) if open_rounds else None

    def join_prior(self, season: SeasonInputs) -> tuple[int, list[str], dict[str, list[str]]]:
        """Opening prices + rosters + last season's per-game stats -> joined.json.

        Matching is by normalised name within the club, then by unique token
        overlap ("TJ" vs "T.J.", middle names). Every matched player takes his
        current club and registration dates from the roster feed. Registered
        players missing from the opening list are added, priced from the
        latest public list where it names them (else ``price_unknown``).
        Returns (rows, unmatched opening-list players, report of added,
        moved, released and unpriced players).
        """
        prices_csv = self.prior_dir / "prices_round_01.csv"
        if not prices_csv.is_file():
            raise SeasonDataError(f"put the opening price list at {prices_csv}")
        with prices_csv.open(encoding="utf-8") as fh:
            prices = list(csv.DictReader(fh))
        people = json.loads(
            (self.prior_dir / f"people_{season.code}.json").read_text(encoding="utf-8")
        )["data"]
        stats = json.loads(
            (self.prior_dir / f"stats_{season.previous_code}_acc.json").read_text(encoding="utf-8")
        )["players"]
        # One row per registration: a player who moved keeps his old club's row too.
        roster = [
            {"code": p["person"]["code"], "name": p["person"]["name"], "club": p["club"]["code"],
             "last_team": p.get("lastTeam")}
            for p in people
        ]  # fmt: skip
        current = current_registrations(people)
        by_code = {s["player"]["code"]: s for s in stats}

        def tokens(name: str) -> set[str]:
            return {t for t in normalise_name(name).split() if len(t) > 1}

        def with_stats(rec: dict, code: str) -> dict:
            s = by_code.get(code)
            if s and s["gamesPlayed"]:
                gp = s["gamesPlayed"]
                rec["el25"] = {"team": s["player"]["team"]["code"], "gp": gp,
                               "min": s["minutesPlayed"] / gp, "pts": s["pointsScored"] / gp,
                               "pir": s["pir"] / gp}  # fmt: skip
            reg = current.get(code)
            if reg:
                rec.update(club=reg["club"], registered_from=reg["start"],
                           registered_until=reg["until"])  # fmt: skip
            return rec

        # A partial-name match may not take an ID another player matched exactly
        # (Vitto Brown is not on the preseason roster; "Brown" must not make him
        # Anthony Brown), nor one an earlier partial match already took.
        claimed = {
            r["code"]
            for row in prices
            for r in roster
            if r["club"] == season.clubs.get(row["club"], row["club"])
            and normalise_name(r["name"]) == normalise_name(row["player"])
        }
        out, unmatched = [], []
        report: dict[str, list[str]] = {"added": [], "moved": [], "released": [], "unpriced": []}
        for row in prices:
            club = season.clubs.get(row["club"], row["club"])
            rec = {
                "name": row["player"],
                "club": club,
                "pos": row["pos"],
                "price": float(row["price"]),
            }
            if row["pos"] == "HC":
                out.append(rec)
                continue
            want = normalise_name(row["player"])
            cands = [r for r in roster if r["club"] == club and normalise_name(r["name"]) == want]
            if not cands:
                scored = sorted(
                    (
                        (len(tokens(row["player"]) & tokens(r["name"])), r)
                        for r in roster
                        if r["club"] == club and r["code"] not in claimed
                    ),
                    key=lambda t: -t[0],
                )
                scored = [s for s in scored if s[0]]
                if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                    cands = [scored[0][1]]
                    claimed.add(scored[0][1]["code"])
            if not cands:
                unmatched.append(f"{row['player']} ({club})")
                rec["code"] = None
                out.append(rec)
                continue
            r = cands[0]
            rec.update(code=r["code"], api_name=r["name"], last_team=r["last_team"])
            with_stats(rec, r["code"])
            if rec["club"] != club:
                report["moved"].append(f"{rec['name']} ({club} -> {rec['club']})")
            if rec.get("registered_until"):
                report["released"].append(
                    f"{rec['name']} ({rec['club']}, {rec['registered_until']})"
                )
            out.append(rec)

        # Registered now but not on the opening list: signings, and players the
        # list missed. Priced from the latest public list, matched by name.
        public = self._public_prices()
        price_round = self._next_round()
        seen = {r.get("code") for r in out}
        for code, reg in sorted(current.items()):
            if code in seen or reg["until"]:
                continue
            name = feed_name(reg["name"])
            want = normalise_name(name)
            hit = [q for q in public if normalise_name(q["player"]) == want]
            if not hit:
                scored = [(len(tokens(name) & tokens(q["player"])), q) for q in public]
                best = max((s for s, _ in scored), default=0)
                top = [q for s, q in scored if s == best and s >= 2]
                hit = top if len(top) == 1 else []
            rec = {"name": name, "club": reg["club"], "code": code, "api_name": reg["name"],
                   "last_team": reg["last_team"], "added": True,
                   "pos": hit[0]["pos"] if hit else POSITIONS.get(reg["position"], "F")}  # fmt: skip
            if hit:
                rec.update(price=float(hit[0]["price"]), price_round=price_round)
            else:
                rec.update(price=4.0, price_unknown=True)
                report["unpriced"].append(f"{name} ({reg['club']})")
            out.append(with_stats(rec, code))
            report["added"].append(
                f"{name} ({reg['club']}, from {reg['start']}"
                + (f", {rec['price']:.1f}" if hit else ", price unknown")
                + ")"
            )
        (self.prior_dir / "joined.json").write_text(
            json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        return len(out), unmatched, report

    def pool_gaps(self, rounds: int = 2) -> list[str]:
        """Players with minutes in the last ``rounds`` finished rounds who are
        not in the player pool, or are there under another club."""
        pool = {
            r.get("code"): r
            for r in json.loads((self.prior_dir / "joined.json").read_text(encoding="utf-8"))
            if r.get("code")
        }
        played = self.played_rounds()[-rounds:]
        first_day = {}
        for g in self.games().values():
            day = str(g.get("localDate", ""))[:10]
            first_day[g["round"]] = min(first_day.get(g["round"], day), day)
        gaps = {}
        for r, box in self.boxes(played).items():
            for ln in box.values():
                p = pool.get(ln["player_id"])
                moved_since = p is not None and (p.get("registered_from") or "") > first_day[r]
                if ln["minutes"] > 0 and (
                    p is None or (p["club"] != ln["team_code"] and not moved_since)
                ):
                    gaps[ln["player_id"]] = f"{ln['player_id']} ({ln['team_code']}, R{r})" + (
                        "" if p is None else f": pool has {p['name']} at {p['club']}"
                    )
        return sorted(gaps.values())


POSITIONS = {"Guard": "G", "Forward": "F", "Center": "C", "Centre": "C"}


def feed_name(name: str) -> str:
    """'WALKER IV, LONNIE' -> 'Lonnie Walker IV'."""
    last, _, first = name.partition(",")
    fixed = {"Ii": "II", "Iii": "III", "Iv": "IV", "Jr": "Jr."}
    words = f"{first.strip()} {last.strip()}".title().split()
    return " ".join(fixed.get(w, w) for w in words)


def current_registrations(people: list[dict]) -> dict[str, dict]:
    """player code -> his current registration: the latest active one, else
    (released) the last one, with its start and, if released, end date."""
    out: dict[str, dict] = {}
    for p in people:
        code = p["person"]["code"]
        active = p.get("active", True)
        reg = {"name": p["person"]["name"], "club": p["club"]["code"],
               "start": (p.get("startDate") or "")[:10] or None,
               "until": None if active else (p.get("endDate") or "")[:10] or None,
               "active": active, "last_team": p.get("lastTeam"),
               "position": p.get("positionName", "")}  # fmt: skip
        old = out.get(code)
        if old is None or (active, reg["start"] or "") > (old["active"], old["start"] or ""):
            out[code] = reg
    return out
