"""Round 1 data join: prices + 2026-27 rosters + 2025-26 per-game stats."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA  # noqa: E402

from elfantasy.data.injuries import normalise_name  # noqa: E402

CLUB_CODE = {
    "ASVEL": "ASV",
    "Anadolu Efes": "IST",
    "Barcelona": "BAR",
    "Baskonia": "BAS",
    "Bayern Munich": "MUN",
    "Besiktas": "BES",
    "Crvena Zvezda": "RED",
    "Dubai": "DUB",
    "Fenerbahce": "ULK",
    "Hapoel Tel Aviv": "HTA",
    "Maccabi Tel Aviv": "TEL",
    "Milano": "MIL",
    "Olympiacos": "OLY",
    "Panathinaikos": "PAN",
    "Paris": "PRS",
    "Partizan": "PAR",
    "Real Madrid": "MAD",
    "Valencia": "PAM",
    "Virtus Bologna": "VIR",
    "Zalgiris": "ZAL",
}


def tokens(name: str) -> set[str]:
    return {t for t in normalise_name(name).split() if len(t) > 1}


def load():
    with (DATA / "prices_players.csv").open(encoding="utf-8") as fh:
        prices = list(csv.DictReader(fh))
    people = json.loads(Path(DATA / "people_E2026.json").read_text(encoding="utf-8"))["data"]
    # Accumulated mode: every player, no games-played qualifier.
    stats = json.loads(Path(DATA / "stats_E2025_acc.json").read_text(encoding="utf-8"))["players"]

    roster = []
    for p in people:
        roster.append(
            {
                "code": p["person"]["code"],
                "name": p["person"]["name"],
                "club": p["club"]["code"],
                "pos_api": p.get("positionName"),
                "last_team": p.get("lastTeam"),
            }
        )

    by_code_stats = {s["player"]["code"]: s for s in stats}

    out, unmatched = [], []
    for row in prices:
        club = CLUB_CODE.get(row["club"], row["club"])
        rec = {
            "name": row["player"],
            "club": club,
            "pos": row["pos"],
            "price": float(row["price"]),
        }
        if row["pos"] == "HC":
            out.append(rec)
            continue

        # 1. exact normalised-name match within the club
        key = normalise_name(row["player"])
        cands = [r for r in roster if r["club"] == club and normalise_name(r["name"]) == key]
        # 2. token overlap within the club (handles "TJ" vs "T.J.", middle names)
        if not cands:
            want = tokens(row["player"])
            scored = []
            for r in roster:
                if r["club"] != club:
                    continue
                overlap = len(want & tokens(r["name"]))
                if overlap:
                    scored.append((overlap, r))
            scored.sort(key=lambda t: -t[0])
            if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                cands = [scored[0][1]]
        if not cands:
            unmatched.append(f"{row['player']} ({club})")
            rec["code"] = None
            out.append(rec)
            continue

        r = cands[0]
        rec["code"] = r["code"]
        rec["api_name"] = r["name"]
        rec["last_team"] = r["last_team"]
        s = by_code_stats.get(r["code"])
        if s and s["gamesPlayed"]:
            gp = s["gamesPlayed"]
            rec["el25"] = {  # per-game averages from season totals
                "team": s["player"]["team"]["code"],
                "gp": gp,
                "min": s["minutesPlayed"] / gp,
                "pts": s["pointsScored"] / gp,
                "pir": s["pir"] / gp,
            }
        out.append(rec)

    (DATA / "joined.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    return out, unmatched


if __name__ == "__main__":
    out, unmatched = load()
    players = [r for r in out if r["pos"] != "HC"]
    with_hist = [r for r in players if "el25" in r]
    moved = [r for r in with_hist if r["club"] not in r["el25"]["team"].split(";")]
    print(
        f"priced players: {len(players)}  matched to roster: {sum(1 for r in players if r.get('code'))}"
    )
    print(f"with 2025-26 EuroLeague stats: {len(with_hist)}  of which changed club: {len(moved)}")
    print(f"unmatched ({len(unmatched)}): {unmatched}")
    print("\n--- priced >= 9.0 with NO 2025-26 EuroLeague stats (new to league / role unknown) ---")
    for r in sorted(players, key=lambda r: -r["price"]):
        if r["price"] >= 9.0 and "el25" not in r:
            print(
                f"  {r['price']:5.1f} {r['pos']} {r['club']} {r['name']}  (last team: {r.get('last_team')})"
            )
    print("\n--- changed club, priced >= 9.0 ---")
    for r in sorted(moved, key=lambda r: -r["price"]):
        if r["price"] >= 9.0:
            e = r["el25"]
            print(
                f"  {r['price']:5.1f} {r['pos']} {e['team']:>7}->{r['club']} {r['name']:24} "
                f"25-26: {e['min']:4.1f}min {e['pir']:4.1f}pir"
            )
