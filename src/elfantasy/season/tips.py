"""Tipster picks: Basketball Sphere's 'best picks' per round, from the archive.

The archive keeps every capture of a round's article (``tips`` table). Here the
picks are matched to our players, flagged on the round card where our model
ranks them low, and offered to the backtest as a candidate signal. A pick
is timed against the round's first tip-off (``timing``): ``observed`` if we
archived it before then; ``dated`` if archived later but its article was last
edited before then by the publisher's own timestamp; ``late`` otherwise. A
late pick may carry hindsight: the article was edited after the games.
"""

from __future__ import annotations

import difflib
from datetime import datetime

from elfantasy.data.injuries import normalise_name

LOW_RANK = 15  # a pick we rank below this at his position is flagged


def _utc(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def first_tipoffs(games: dict[str, dict]) -> dict[int, str]:
    """round -> UTC time of its first game."""
    out: dict[int, str] = {}
    for g in games.values():
        t = g.get("utcDate")
        if t and (g["round"] not in out or _utc(t) < _utc(out[g["round"]])):
            out[g["round"]] = t
    return out


TIMINGS = ("observed", "dated", "late")


def round_tips(rows: list[dict], round_no: int, tipoff: str | None = None) -> list[dict]:
    """The picks for ``round_no`` as last captured before ``tipoff`` (else as
    last captured), each with its ``timing`` (None without a ``tipoff``)."""
    mine = [r for r in rows if r.get("round") == round_no]
    caps = sorted({r["captured_at"] for r in mine}, key=_utc)
    if not caps:
        return []
    chosen = caps[-1]
    if tipoff:
        early = [c for c in caps if _utc(c) < _utc(tipoff)]
        chosen = early[-1] if early else chosen
    out = []
    for r in mine:
        if r["captured_at"] != chosen:
            continue
        timing = None
        if tipoff is not None:
            timing = "late"
            if _utc(chosen) < _utc(tipoff):
                timing = "observed"
            elif r.get("modified") and _utc(r["modified"]) < _utc(tipoff):
                timing = "dated"
        out.append({**r, "timing": timing})
    return out


def _tokens(name: str) -> set[str]:
    return set(normalise_name(name.replace(".", "")).split())


def club_code(name: str, clubs: dict[str, str]) -> str | None:
    """A club as the tipster writes it ('Asvel', 'Virtus') -> our code."""
    low = name.lower().strip()
    hits = {c for k, c in clubs.items() if k.lower() == low}
    if not hits:
        hits = {
            c for k, c in clubs.items() if k.lower().startswith(low) or low.startswith(k.lower())
        }
    if not hits and name.upper() in clubs.values():
        hits = {name.upper()}
    return hits.pop() if len(hits) == 1 else None


def match_pick(pick: dict, people: list[dict], clubs: dict[str, str]) -> dict | None:
    """Our player (or coach) a pick names: same name tokens, else two shared
    tokens, else a close spelling at the same club; the club breaks ties."""
    want, club = _tokens(pick["player"]), club_code(pick["club"], clubs)

    def one(cands: list[dict]) -> dict | None:
        if len(cands) > 1 and club:
            cands = [p for p in cands if p["club"] == club]
        return cands[0] if len(cands) == 1 else None

    hit = one([p for p in people if _tokens(p["name"]) == want])
    if hit is None:
        hit = one([p for p in people if len(_tokens(p["name"]) & want) >= 2])
    if hit is None and club:
        flat = " ".join(sorted(want))
        close = [
            p for p in people
            if p["club"] == club
            and difflib.SequenceMatcher(None, " ".join(sorted(_tokens(p["name"]))), flat).ratio() >= 0.85
        ]  # fmt: skip
        hit = close[0] if len(close) == 1 else None
    return hit


def card_rows(result: dict, picks: list[dict], clubs: dict[str, str]) -> list[dict]:
    """One row per pick for the round card: our forecast and rank at his
    position, and a note where we disagree or cannot place him."""
    r0 = str(result["round"])
    people = {"HC": result["coaches"], "G": [], "F": [], "C": []}
    for p in result["players"]:
        people[p["pos"]].append(p)
    rank = {}
    for group in people.values():
        ranked = sorted((p for p in group if r0 in p["proj"]), key=lambda p: -p["proj"][r0]["ev"])
        rank.update({id(p): i + 1 for i, p in enumerate(ranked)})
    rows = []
    for pick in picks:
        pos = pick["pos"] if pick["pos"] in people else "F"
        hit = match_pick(pick, people[pos], clubs)
        if hit is None and pos != "HC":  # listed at another position than ours
            hit = match_pick(pick, result["players"], clubs)
        row = {"pos": pick["pos"], "player": pick["player"], "club": pick["club"],
               "price": pick["price"], "tier": pick.get("tier"), "ev": None, "rank": None,
               "name": hit["name"] if hit else None}  # fmt: skip
        if hit is None:
            row["note"] = "not in our pool"
        elif r0 not in hit["proj"]:
            row["note"] = "no forecast this round (out or not registered)"
        else:
            row.update(ev=hit["proj"][r0]["ev"], rank=rank.get(id(hit)))
            row["note"] = (
                "we rank him low: check the news" if (row["rank"] or 99) > LOW_RANK else ""
            )
        rows.append(row)
    return rows
