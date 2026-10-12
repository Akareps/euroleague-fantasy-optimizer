"""Pinned bulk archive import; only raw EuroLeague box scores enter training.

The archive is third-party data, not executable code. Open only base tables,
read-only, with external access disabled. Reconcile with the official feed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

from elfantasy.data.euroleague import EuroleagueClient
from elfantasy.data.http import HttpClient

EXPERIMENT = "hierarchical-v1"
ARCHIVE_COMMIT = "2c44083749cf6c0b545a181c5e3321d4c52ca368"
ARCHIVE_SHA256 = "00ccbcb3bc1a12212005ab138a042f8296b1a65235e249839ab6d439c0935728"
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/McNews11/euroleague-open-data/"
    f"{ARCHIVE_COMMIT}/data/euroleague.duckdb"
)
SEASONS = ("E2024", "E2025")
FIELDS = {
    "fg2a": "two_attempted",
    "fg2m": "two_made",
    "fg3a": "three_attempted",
    "fg3m": "three_made",
    "fta": "ft_attempted",
    "ftm": "ft_made",
    "rebounds_total": "rebounds",
    "assists": "assists",
    "steals": "steals",
    "blocks_favour": "blocks",
    "fouls_received": "fouls_drawn",
    "turnovers": "turnovers",
    "blocks_against": "blocks_against",
    "fouls_committed": "fouls_committed",
}


def utc(value):
    d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("historical timestamps must have an explicit timezone")
    return d.astimezone(timezone.utc)


def position(value):
    return {"Guard": "G", "Forward": "F", "Center": "C", "Centre": "C"}.get(value, "F")


def import_archive(path: Path) -> dict:
    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("historical archive checksum does not match the pinned snapshot")
    try:
        import duckdb
    except ImportError as exc:
        raise ValueError(
            "Install the optional history dependency: pip install -e '.[history]'"
        ) from exc
    con = duckdb.connect(str(path), read_only=True, config={"enable_external_access": False})
    try:
        required = {"games", "players", "boxscores_player"}
        tables = dict(
            con.execute(
                "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        )
        if any(tables.get(t) != "BASE TABLE" for t in required):
            raise ValueError("archive must provide the three raw base tables")
        games = (
            con.execute(
                "SELECT * FROM games WHERE season_code IN ('E2024','E2025') AND phase_code='RS' AND played"
            )
            .fetchdf()
            .to_dict("records")
        )
        players = {
            r["person_code"]: {"name": r["name"], "pos": position(r["position_name"])}
            for r in con.execute("SELECT * FROM players").fetchdf().to_dict("records")
        }
        boxes = (
            con.execute("SELECT * FROM boxscores_player WHERE season_code IN ('E2024','E2025')")
            .fetchdf()
            .to_dict("records")
        )
    finally:
        con.close()
    by_game = {(g["season_code"], int(g["game_code"])): g for g in games}
    rows, excluded, seen = [], {}, set()
    for b in boxes:
        g = by_game.get((b["season_code"], int(b["game_code"])))
        if g is None:
            continue
        gid = f"{b['season_code']}_{int(b['game_code'])}"
        ident = (gid, b["person_code"])
        if ident in seen:
            raise ValueError(f"duplicate player-game: {ident}")
        seen.add(ident)
        own, other = ("home", "away") if b["is_home"] else ("away", "home")
        r = {
            "season": b["season_code"],
            "game_id": gid,
            "round": int(g["round"]),
            "date": utc(g["utc_date"]).isoformat(),
            "player_id": b["person_code"],
            "club": b["team_code"],
            "opp": g[f"{other}_team_code"],
            "home": bool(b["is_home"]),
            "minutes": float(b["seconds_played"]) / 60,
            "points": float(b["points"]),
            "pir": float(b["valuation"]),
            "margin": int(g[f"{own}_score"] - g[f"{other}_score"]),
            "pos": players.get(b["person_code"], {}).get("pos", "F"),
        }
        r.update({dst: float(b[src]) for src, dst in FIELDS.items()})
        recomputed = (
            r["points"]
            + sum(r[k] for k in ("rebounds", "assists", "steals", "blocks", "fouls_drawn"))
            - (r["two_attempted"] - r["two_made"])
            - (r["three_attempted"] - r["three_made"])
            - (r["ft_attempted"] - r["ft_made"])
            - sum(r[k] for k in ("turnovers", "blocks_against", "fouls_committed"))
        )
        valid = all(r[k] >= 0 for k in FIELDS.values()) and 0 <= r["minutes"] <= 60
        valid &= all(r[f"{s}_made"] <= r[f"{s}_attempted"] for s in ("two", "three", "ft"))
        valid &= abs(r["points"] - 2 * r["two_made"] - 3 * r["three_made"] - r["ft_made"]) < 0.01
        valid &= abs(recomputed - r["pir"]) < 0.01
        if not valid:
            excluded[gid] = "inconsistent shooting or PIR components"
        rows.append(r)
    totals = {}
    for r in rows:
        totals[(r["game_id"], r["club"])] = totals.get((r["game_id"], r["club"]), 0) + r["minutes"]
    for (gid, _), minutes in totals.items():
        # Entire-game clock totals must reconcile to regulation plus overtime.
        nearest = 200 + 25 * max(round((minutes - 200) / 25), 0)
        if abs(minutes - nearest) > 0.2:
            excluded[gid] = f"team minutes do not reconcile: {minutes:.3f}"
    rows = [r for r in rows if r["game_id"] not in excluded]
    for r in rows:
        r["regulation_minutes"] = r["minutes"] * 200 / totals[(r["game_id"], r["club"])]
    counts = {s: len({r["game_id"] for r in rows if r["season"] == s}) for s in SEASONS}
    if any(v < 250 for v in counts.values()):
        raise ValueError(f"insufficient historical coverage: {counts}")
    return {
        "schema_version": 1,
        "rows": sorted(rows, key=lambda r: (r["date"], r["game_id"], r["player_id"])),
        "players": players,
        "coverage": counts,
        "excluded_games": excluded,
        "source": {
            "url": ARCHIVE_URL,
            "commit": ARCHIVE_COMMIT,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "scope": "EuroLeague regular season only; EuroCup and derived fantasy tables excluded",
    }


def verify_official(history: dict, root: Path) -> dict:
    """Schedules exhaustively; deterministic first/middle/last boxes per season."""
    http = HttpClient(root / "official_http", ttl_seconds=10**10, min_interval=2.0, retries=1)
    verified = []
    for season in SEASONS:
        client = EuroleagueClient(http, season=season)
        games = client.fetch_games()
        if not games:
            raise ValueError(f"no official schedule for {season}")
        own = {r["game_id"] for r in history["rows"] if r["season"] == season}
        official = {
            g.game_id for g in games if g.played and g.round <= (34 if season == "E2024" else 38)
        }
        if own - official or len(official - own) > len(history["excluded_games"]):
            raise ValueError(f"schedule coverage mismatch for {season}")
        ordered = sorted((g for g in games if g.game_id in own), key=lambda g: g.tipoff)
        for g in (ordered[0], ordered[len(ordered) // 2], ordered[-1]):
            lines = client.fetch_game_boxscore(g)
            if not lines:
                raise ValueError(f"no official verification box for {g.game_id}")
            saved = {r["player_id"]: r for r in history["rows"] if r["game_id"] == g.game_id}
            if set(saved) != {b.player_id for b in lines}:
                raise ValueError(f"player identities differ for {g.game_id}")
            for b in lines:
                r = saved[b.player_id]
                fields = {
                    "minutes": b.minutes,
                    "pir": b.pir,
                    "points": b.points,
                    "rebounds": b.rebounds,
                    "assists": b.assists,
                    "steals": b.steals,
                    "blocks": b.blocks,
                    "turnovers": b.turnovers,
                    "fouls_drawn": b.fouls_drawn,
                    "fouls_committed": b.fouls_committed,
                    "blocks_against": b.blocks_against,
                    "ft_made": b.ft_made,
                    "ft_attempted": b.ft_attempted,
                }
                fields["fg_made"] = b.fg_made
                fields["fg_attempted"] = b.fg_attempted
                r = {
                    **r,
                    "fg_made": r["two_made"] + r["three_made"],
                    "fg_attempted": r["two_attempted"] + r["three_attempted"],
                }
                if any(abs(r[k] - v) > 0.02 for k, v in fields.items()):
                    raise ValueError(f"official statistics differ for {g.game_id}/{b.player_id}")
            verified.append({"game": g.game_id, "players": len(lines)})
    return {
        "status": "verified",
        "schedule_seasons": list(SEASONS),
        "sampled_boxes": verified,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_history(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "archive.duckdb"
    if not path.is_file():
        response = requests.get(ARCHIVE_URL, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    doc = import_archive(path)
    doc["verification"] = verify_official(doc, root)
    (root / "history.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return doc


def load_history(root: Path) -> dict:
    path = root / "history.json"
    if not path.is_file():
        raise ValueError("Historical data missing: run elfantasy experiment hierarchical fetch")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1 or doc.get("verification", {}).get("status") != "verified":
        raise ValueError("historical data has not passed official verification")
    return doc


def current_records(model, target: int, root: Path, *, fetch=False) -> list[dict]:
    """Only completed earlier rounds. Recover exact shooting splits from raw boxes.

    Missing raw data is an error, never an invented two/three-point split.
    The network is enabled only by the explicit fetch command.
    """
    production = HttpClient(model.data.root / "http", offline=True)
    independent = HttpClient(
        root / "current_http", ttl_seconds=10**10, min_interval=2.0, offline=not fetch, retries=1
    )
    people = json.loads(
        (model.data.prior_dir / f"people_{model.season.code}.json").read_text(encoding="utf-8")
    )["data"]
    positions = {p["person"]["code"]: position(p.get("positionName")) for p in people}
    joined = json.loads((model.data.prior_dir / "joined.json").read_text(encoding="utf-8"))
    duplicates = Counter(p.get("code") for p in joined if p.get("code"))
    boxes = model.data.boxes(range(1, target)) if target > 1 else {}
    rows = []
    for g in sorted(model.games.values(), key=lambda g: g["utcDate"]):
        rnd = int(g["round"])
        if rnd >= target or not g["played"]:
            continue
        url = f"https://api-live.euroleague.net/v2/competitions/E/seasons/{model.season.code}/games/{g['gameCode']}/stats"
        raw = production.get_json(url) if production._cache_path(url, None).is_file() else None
        raw = raw or independent.get_json(url)
        if not raw:
            raise ValueError(
                f"missing exact shooting data for {g['identifier']}; run hierarchical fetch --round {target}"
            )
        splits = {}
        for side in ("local", "road"):
            for p in raw[side]["players"]:
                splits[p["player"]["person"]["code"]] = p["stats"]
        team_minutes = {}
        for b in boxes[rnd].values():
            if b["game_id"] == g["identifier"]:
                team_minutes[b["team_code"]] = team_minutes.get(b["team_code"], 0) + b["minutes"]
        for b in boxes[rnd].values():
            if b["game_id"] != g["identifier"]:
                continue
            stats = splits[b["player_id"]]
            home = b["team_code"] == g["local"]["club"]["code"]
            own, other = ("local", "road") if home else ("road", "local")
            r = {
                k: b[k]
                for k in (
                    "minutes",
                    "points",
                    "pir",
                    "rebounds",
                    "assists",
                    "steals",
                    "blocks",
                    "fouls_drawn",
                    "turnovers",
                    "blocks_against",
                    "fouls_committed",
                    "ft_made",
                    "ft_attempted",
                    "player_id",
                    "game_id",
                    "round",
                )
            }
            r.update(
                season=model.season.code,
                club=b["team_code"],
                opp=g[other]["club"]["code"],
                home=home,
                date=utc(g["utcDate"]).isoformat(),
                margin=g[own]["score"] - g[other]["score"],
                pos=positions.get(b["player_id"], "F"),
                regulation_minutes=b["minutes"] * 200 / team_minutes[b["team_code"]],
            )
            r.update(
                two_made=float(stats["fieldGoalsMade2"]),
                two_attempted=float(stats["fieldGoalsAttempted2"]),
                three_made=float(stats["fieldGoalsMade3"]),
                three_attempted=float(stats["fieldGoalsAttempted3"]),
            )
            rows.append(r)
        present = {r["player_id"] for r in rows if r["game_id"] == g["identifier"]}
        injured = model.round_inputs(rnd).missed_injured()
        from elfantasy.season.inputs import key

        for p in joined:
            code = p.get("code")
            if not code or duplicates[code] > 1 or code in present or p["club"] not in team_minutes:
                continue
            if key(p["club"], p["name"]) in injured:
                continue
            home = p["club"] == g["local"]["club"]["code"]
            own, other = ("local", "road") if home else ("road", "local")
            # Shared season model convention: an unlisted player with no injury
            # annotation supplies zero-minute role evidence. Stats are zero
            # because he didn't play; no shooting split is imputed.
            row = {
                "season": model.season.code,
                "game_id": g["identifier"],
                "round": rnd,
                "date": utc(g["utcDate"]).isoformat(),
                "player_id": code,
                "club": p["club"],
                "opp": g[other]["club"]["code"],
                "home": home,
                "pos": p["pos"],
                "minutes": 0.0,
                "regulation_minutes": 0.0,
                "pir": 0.0,
                "points": 0.0,
                "margin": g[own]["score"] - g[other]["score"],
                "inferred_dnp": True,
            }
            row.update({field: 0.0 for field in FIELDS.values()})
            rows.append(row)
    return rows
