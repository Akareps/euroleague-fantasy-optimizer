"""Season projection model: preseason prior, evidence from games played,
game context from betting lines, calibration, and price estimates.

1. **Prior.** Last season's minutes and PIR per minute where the player has
   them (trusted more if he stayed at his club), blended with what his opening
   price implies; price alone for players new to the league; named preseason
   role evidence on top.
2. **Evidence.** Every game played since: minutes (blowouts corrected: the top
   five lose their garbage-time fourth quarter, the bench borrow it) and PIR
   per minute, as a Bayesian update on the prior. Players who missed a round
   injured are skipped; a fit player left out counts as a 0-minute game.
3. **Team context.** Each team plays 200 minutes. Absences free minutes for
   backups (rotation order first, then by position); de-vigged moneylines give
   the expected margin, which moves team PIR, garbage time, the win bonus and
   the coach's score. Fixtures without a line use team ratings fitted to every
   known line plus outright odds.
4. **Calibration.** Every past round is re-projected with today's model and
   only the information available before it; the errors, by prior type and
   projection tier, give a shrunk additive correction.
5. **Prices.** The last published list plus each round's fitted change
   (players: PIR vs price; coaches: result bucket), unless the app's real
   prices are given.

Everything that was a module-level table in the round scripts is an explicit
input here, so any past round can be re-run exactly.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

from elfantasy.data.injuries import normalise_name
from elfantasy.projection import market as mk
from elfantasy.rules import GameRules
from elfantasy.scoring import price_change
from elfantasy.season.data import SeasonData
from elfantasy.season.inputs import (
    CalibrationSettings,
    EvidenceWeights,
    Key,
    PrivateInputs,
    RoundInputs,
    SeasonInputs,
    key,
)

SIGMA = 11.5  # SD of EuroLeague game margin around the spread
HCA = 3.5  # home-court advantage, points
TEAM_MINUTES = 200.0
WIN_BONUS = 0.10  # +10% of the fantasy score when the team wins
HEALTHY = 0.97  # P(plays) for a fit player: the odd late scratch nobody reported
RANK_SHARE = [1.0] * 9 + [0.8, 0.6, 0.4]  # rotation discount beyond the ninth man


def _pkey(p: dict) -> Key:
    return key(p["club"], p["name"])


# ======================================================================= prior
def load_league_fits(prior_dir: Path):
    """Team PIR vs margin and game-to-game volatility, from last season's
    cached box scores (66 games before the API rate-limited us)."""

    team_pir: dict = defaultdict(float)
    team_pts: dict = defaultdict(float)
    player_games: dict = defaultdict(list)
    for f in (prior_dir / "http").rglob("*.json"):
        body = json.loads(json.loads(Path(f).read_text(encoding="utf-8"))["body"])
        if not isinstance(body, dict) or "local" not in body:
            continue
        gid = f.stem
        for side in ("local", "road"):
            for row in body[side]["players"]:
                st = row["stats"]
                team_pir[(gid, side)] += st["valuation"]
                team_pts[(gid, side)] += st["points"]
                if st["timePlayed"] > 600:  # 10+ minutes
                    player_games[row["player"]["person"]["code"]].append(st["valuation"])
    return team_pir, team_pts, player_games


def fit_margin_to_team_pir(team_pir, team_pts):
    """Team PIR as a linear function of the team's final margin."""
    xs, ys = [], []
    gids = {k[0] for k in team_pts}
    for gid in gids:
        for side, other in (("local", "road"), ("road", "local")):
            xs.append(team_pts[(gid, side)] - team_pts[(gid, other)])
            ys.append(team_pir[(gid, side)])
    b, a = np.polyfit(xs, ys, 1)
    return float(a), float(b), len(gids)


def player_history(agg, club_now):
    if not agg or agg["gamesPlayed"] <= 0:
        return None
    gp = agg["gamesPlayed"]
    total_min = agg["minutesPlayed"]
    teams = agg["player"]["team"]["code"].split(";")
    return {
        "gp": gp,
        "min": total_min / gp,
        "pir": agg["pir"] / gp,
        "rate": agg["pir"] / total_min if total_min > 0 else 0.0,
        "total_min": total_min,
        "sd": None,
        "last_team": teams[-1],
        "same_club": club_now in teams,
    }


def build_players(data: SeasonData, season: SeasonInputs):
    """Preseason baselines for every priced player: (players, coaches, fit_info)."""

    prior = data.prior_dir
    joined = json.loads((prior / "joined.json").read_text(encoding="utf-8"))
    team_pir, team_pts, player_games = load_league_fits(prior)
    if team_pts:
        a_pir, b_pir, n_games = fit_margin_to_team_pir(team_pir, team_pts)
    else:  # fitted on 66 games of 2025-26 before the feed rate-limited us
        a_pir, b_pir, n_games = 93.5, 1.09, 0
    stats_file = prior / f"stats_{season.previous_code}_acc.json"
    rows = json.loads(stats_file.read_text(encoding="utf-8"))["players"]
    aggs = {r["player"]["code"]: r for r in rows}

    # League-wide game-to-game volatility: SD of PIR relative to its mean,
    # from players with 4+ ten-minute games in the cached sample.
    cvs = [
        np.std(v, ddof=1) / np.mean(v)
        for v in player_games.values()
        if len(v) >= 4 and np.mean(v) >= 5
    ]
    league_cv = float(np.median(cvs)) if cvs else 0.634

    players, coaches = [], []
    for r in joined:
        if r["pos"] == "HC":
            coaches.append(r)
            continue
        r["hist"] = player_history(aggs.get(r.get("code")), r["club"]) if r.get("code") else None
        players.append(r)

    # Position PIR/min priors from rotation players.
    pos_rate = {}
    for pos in "GFC":
        rs = [
            p["hist"]["rate"]
            for p in players
            if p["hist"] and p["pos"] == pos and p["hist"]["total_min"] > 300
        ]
        pos_rate[pos] = float(np.median(rs))

    # Price-implied PIR and minutes, fitted on same-club returning players.
    fit_rows = [
        p for p in players if p["hist"] and p["hist"]["same_club"] and p["hist"]["gp"] >= 10
    ]
    prices = np.array([p["price"] for p in fit_rows])
    b1, a1 = np.polyfit(prices, [p["hist"]["pir"] for p in fit_rows], 1)
    b2, a2 = np.polyfit(prices, [p["hist"]["min"] for p in fit_rows], 1)
    fit_info = {
        "pir": (a1, b1),
        "min": (a2, b2),
        "n": len(fit_rows),
        "pos_rate": pos_rate,
        "team_pir": (a_pir, b_pir),
        "n_games": n_games,
        "league_cv": league_cv,
    }

    for p in players:
        price_pir = max(a1 + b1 * p["price"], 0.5)
        price_min = max(a2 + b2 * p["price"], 3.0)
        h = p["hist"]
        if h and h["total_min"] >= 60:
            n = h["total_min"]
            rate = (n * h["rate"] + 250 * pos_rate[p["pos"]]) / (n + 250)
            if h["same_club"]:
                w = 0.8 * h["gp"] / (h["gp"] + 6)
                minutes = w * h["min"] + (1 - w) * price_min
                pir = 0.8 * rate * minutes + 0.2 * price_pir
            else:
                minutes = 0.35 * h["min"] + 0.65 * price_min
                pir = 0.6 * rate * minutes + 0.4 * price_pir
        else:
            # New to the league: the price is the only calibrated signal.
            # The bottom of the price range is mostly deep bench who never play.
            if p["price"] <= 4.5:
                price_pir, price_min = 2.0, 5.0
            minutes, pir = price_min, price_pir
            rate = pir / minutes if minutes > 0 else pos_rate[p["pos"]]
        role = season.role.get(_pkey(p))
        if role:
            pir *= role[0]
            minutes = min(minutes * (1 + 0.8 * (role[0] - 1)), 34.0)
            p["role_note"] = role[1]
        expert_min = season.minutes_expert.get(_pkey(p))
        if expert_min is not None:
            if not (h and h["total_min"] >= 60):
                rate = pos_rate[p["pos"]] * 0.9
                pir = rate * expert_min
            else:
                pir = pir / max(minutes, 1.0) * expert_min
            minutes = expert_min
            p["minutes_locked"] = True  # a stated estimate: exempt from rotation discounts
            p["role_note"] = (p.get("role_note", "") + f" [expert: ~{expert_min:.0f} min]").strip()
        p["base_min"] = minutes
        p["base_pir"] = pir
        p["rate"] = pir / minutes if minutes > 0 else rate
        games_seen = player_games.get(p.get("code"), [])
        if len(games_seen) >= 5 and np.mean(games_seen) >= 5:
            own_cv = float(np.std(games_seen, ddof=1) / np.mean(games_seen))
            p["sd_per_pir"] = 0.5 * own_cv + 0.5 * league_cv  # shrink: few games
        else:
            p["sd_per_pir"] = league_cv
    return players, coaches, fit_info


def prior_group(p: dict) -> str:
    """How the prior was built: the calibration groups."""
    h = p.get("hist")
    if not h or h["total_min"] < 60:
        return "new to EL"
    return "same club" if h["same_club"] else "changed club"


# ==================================================================== evidence
def garbage_minutes(margin: float) -> float:
    """Fourth-quarter minutes the top five lose in a blowout (0.55 per point beyond 9)."""
    return min(max(0.55 * (abs(margin) - 9.0), 0.0), 9.0)


def update_with_evidence(
    players: list[dict],
    games: dict[str, dict],
    boxes: dict[int, dict[str, dict]],
    injured: dict[int, set[Key]],
    weights: EvidenceWeights,
    minutes_expert: dict[int, dict[Key, float]] | None = None,
) -> None:
    """Batch Bayesian update of minutes and PIR/min on the given rounds."""

    garbage = {}
    for g in games.values():
        if g["round"] in boxes and g["played"]:
            garbage[g["identifier"]] = garbage_minutes(g["local"]["score"] - g["road"]["score"])
    by_code = {p.get("code"): p for p in players if p.get("code")}
    # (game, player) pairs where the player was among his team's five biggest
    # preseason roles *among those who played that game*: injured players must
    # not hold a top-5 spot, and the starting-five flag is unreliable.
    top5 = set()
    for box in boxes.values():
        in_game: dict = {}
        for ln in box.values():
            if ln["minutes"] > 0 and ln["player_id"] in by_code:
                in_game.setdefault((ln["game_id"], ln["team_code"]), []).append(
                    by_code[ln["player_id"]]
                )
        for (game_id, _), grp in in_game.items():
            for q in sorted(grp, key=lambda q: -q["base_min"])[:5]:
                top5.add((game_id, id(q)))
    for p in players:
        h = p.get("hist")
        strong = bool(h and h.get("same_club") and h.get("gp", 0) >= 10)
        k_min = weights.k_min_strong if strong else weights.k_min_weak
        n_rate = (
            weights.n_rate_strong if strong else (weights.n_rate_hist if h else weights.n_rate_new)
        )
        k = _pkey(p)
        mins, pirs, seen = [], [], {}
        for rnd in sorted(boxes):
            if k in injured.get(rnd, set()):
                continue
            wt = weights.recency ** (rnd - 1)
            ln = boxes[rnd].get(p.get("code"))
            if ln is None:
                mins.append((wt, 0.0))  # not in the 12: a coach's decision
                continue
            m, pir = ln["minutes"], ln["pir"] or 0.0
            gt = garbage.get(ln["game_id"], 0.0)
            if m > 0 and gt > 0:
                is_top = (ln["game_id"], id(p)) in top5
                m = m + 0.75 * gt if is_top else max(m - 0.75 * gt * 5 / 6, 1.0)
            mins.append((wt, m))
            if m > 0:
                pirs.append((wt, m, pir))
            seen[rnd] = (round(m, 1), pir)
        if mins:
            p["base_min"] = (k_min * p["base_min"] + sum(a * b for a, b in mins)) / (
                k_min + sum(a for a, _ in mins)
            )
            if pirs:
                tm = sum(a * m for a, m, _ in pirs)
                p["rate"] = (n_rate * p["rate"] + sum(a * x for a, _, x in pirs)) / (n_rate + tm)
        p["base_pir"] = p["rate"] * p["base_min"]
        p["seen"] = seen
    # Stated roles, kept until a game since contradicts them.
    for since, table in sorted((minutes_expert or {}).items()):
        for p in players:
            mins = table.get(_pkey(p))
            if mins is None:
                continue
            last = max(p["seen"]) if p["seen"] else None
            if last is not None and last >= since and p["seen"][last][0] < 0.6 * mins:
                continue
            p["base_min"] = mins
            p["base_pir"] = p["rate"] * mins
            p["minutes_locked"] = True


# =================================================================== team round
def team_round(players, club, k, spread_for_team, availability, minutes_factor=None):
    """Minutes and PIR for one club in one round, conditional on each player
    playing: id(player) -> (minutes, PIR if he plays, P(plays)).

    ``k`` is the 1-based position of the round in the availability tuples.
    """

    roster = [p for p in players if p["club"] == club]
    avail = {}
    for p in roster:
        row = availability.get(_pkey(p))
        avail[id(p)] = HEALTHY if row is None else row[k - 1]

    def share(rank):
        return RANK_SHARE[rank] if rank < len(RANK_SHARE) else 0.15

    # Healthy world: the team's normal minute distribution. Excess over 200
    # comes mostly out of the back of the rotation; players with a stated
    # minutes estimate keep it.
    healthy = {}
    ranked = sorted(roster, key=lambda q: -q["base_min"])
    for rank, q in enumerate(ranked):
        healthy[id(q)] = q["base_min"] * (1.0 if q.get("minutes_locked") else share(rank))
    excess = sum(healthy.values()) - TEAM_MINUTES
    if excess > 0:
        w = {
            id(q): 0.0 if q.get("minutes_locked") else healthy[id(q)] * (0.25 + rank / 6.0)
            for rank, q in enumerate(ranked)
        }
        tot_w = sum(w.values())
        for kk in healthy:
            healthy[kk] = max(healthy[kk] - excess * w[kk] / tot_w, 0.0)
    scale = 1.0
    mins = dict(healthy)

    # Absences: backups move up the rotation first, then the rest of the
    # vacated time spreads by position and headroom.
    available = [q for q in roster if avail[id(q)] >= 0.35]
    vacated_total = sum(healthy[id(q)] * (1 - avail[id(q)]) * 0.92 for q in roster)
    restored = {}
    for rank, q in enumerate(sorted(available, key=lambda q: -q["base_min"])):
        if q.get("minutes_locked"):
            restored[id(q)] = 0.0
            continue
        restored[id(q)] = max(q["base_min"] * share(rank) * scale - healthy[id(q)], 0.0)
    gain: dict = defaultdict(float)
    pool_restore = sum(restored.values())
    used = min(pool_restore, vacated_total)
    if pool_restore > 0:
        for kk, v in restored.items():
            gain[kk] += used * v / pool_restore
    remaining = vacated_total - used

    for p in roster:
        miss = 1 - avail[id(p)]
        if miss <= 0.02 or remaining <= 0:
            continue
        vacated = remaining * (healthy[id(p)] * miss * 0.92) / vacated_total
        weights = {}
        for q in available:
            if q is p:
                continue
            pos_w = 1.0 if q["pos"] == p["pos"] else 0.35
            headroom = max(33.0 - mins[id(q)] - gain[id(q)], 0.0)
            floor = 0.2 + 0.8 * min((mins[id(q)] + gain[id(q)]) / 14.0, 1.0)
            weights[id(q)] = pos_w * headroom * floor
        tot = sum(weights.values())
        for kk, w_ in weights.items():
            gain[kk] += vacated * w_ / tot

    # Garbage time from the spread.
    garbage = garbage_minutes(spread_for_team)
    order = sorted(roster, key=lambda q: -mins[id(q)])
    starters = {id(q) for q in order[:5]}
    bench = list(order[5:11])

    out = {}
    for p in roster:
        m = mins[id(p)] + gain[id(p)]
        if k == 1 and minutes_factor:
            m *= minutes_factor.get(_pkey(p), 1.0)
        if garbage > 0:
            if id(p) in starters:
                m -= garbage * 5 * 0.75 / 5
            elif p in bench:
                m += garbage * 5 * 0.75 / max(len(bench), 1)
        m = min(max(m, 0.0), 36.0)
        # Minutes gained from absences come with a small usage bump.
        usage = 1.0 + 0.10 * min(gain[id(p)] / max(mins[id(p)], 1.0), 1.0)
        out[id(p)] = (m, p["rate"] * m * usage, avail[id(p)])
    return out


# ======================================================================= market
def spread_from_prices(home_price: float, away_price: float) -> tuple[float, float]:
    """(P(home wins), expected home margin) from a Shin-de-vigged moneyline."""
    p_home = mk.two_way_prob(home_price, away_price, "shin")
    return p_home, float(norm.ppf(p_home) * SIGMA)


def fit_ratings(lines: dict, outrights: dict[str, float]) -> dict[str, float]:
    """Team ratings (points) from known lines plus an outright-odds prior."""
    teams = sorted(outrights)
    idx = {t: i for i, t in enumerate(teams)}
    fair = mk.devig(list(outrights.values()), "multiplicative")
    logp = {t: math.log(p) for t, p in zip(outrights, fair, strict=True)}
    n = len(teams)
    rows, rhs = [], []
    for (h, a), (ph, pa) in lines.items():
        _, margin = spread_from_prices(ph, pa)
        row = np.zeros(n + 2)
        row[idx[h]], row[idx[a]] = 1.0, -1.0
        rows.append(row)
        rhs.append(margin - HCA)
    lam = 0.6
    for t in teams:  # r_t ~ a + b*log(p_title)
        row = np.zeros(n + 2)
        row[idx[t]] = lam
        row[n] = -lam
        row[n + 1] = -lam * logp[t]
        rows.append(row)
        rhs.append(0.0)
    row = np.zeros(n + 2)
    row[:n] = 1.0
    rows.append(row)
    rhs.append(0.0)
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return {t: float(sol[idx[t]]) for t in teams}


def coach_ev(margin_mean: float) -> float:
    """Expected coach score given the team's expected margin."""
    cdf = lambda x: norm.cdf((x - margin_mean) / SIGMA)  # noqa: E731
    p = {
        "w1": cdf(10.5) - cdf(0.0),
        "w2": cdf(20.5) - cdf(10.5),
        "w3": 1 - cdf(20.5),
        "l1": cdf(0.0) - cdf(-10.5),
        "l2": cdf(-10.5) - cdf(-20.5),
        "l3": cdf(-20.5),
    }
    return 10 * p["w1"] + 20 * p["w2"] + 25 * p["w3"] - 5 * p["l1"] - 10 * p["l2"] - 20 * p["l3"]


def project_round(
    players, fit_info, games, round_no, availability, lines, rating_lines, outrights, *, k=1,
    minutes_factor=None,
):  # fmt: skip
    """Raw (uncalibrated) projection of one round: (id(player) -> dict, ratings)."""

    ratings = fit_ratings(rating_lines, outrights)
    a_pir, b_pir = fit_info["team_pir"]
    out = {}
    for g in [g for g in games.values() if g["round"] == round_no]:
        home, away = g["local"]["club"]["code"], g["road"]["club"]["code"]
        if (home, away) in lines:
            p_home, margin = spread_from_prices(*lines[(home, away)])
        else:
            margin = ratings[home] - ratings[away] + HCA
            p_home = float(norm.cdf(margin / SIGMA))
        for club, m, pw, opp, is_home in (
            (home, margin, p_home, away, True),
            (away, -margin, 1 - p_home, home, False),
        ):
            # Only the part of the expected margin due to this opponent and
            # venue is new information; predicted margins carry about half
            # the signal of actual ones.
            env = 1.0 + 0.5 * b_pir * (m - ratings[club]) / a_pir
            team = team_round(players, club, k, m, availability, minutes_factor)
            for p in players:
                if p["club"] != club:
                    continue
                minutes, pir_if_plays, play = team[id(p)]
                fantasy = pir_if_plays * env * (1 + WIN_BONUS * pw)
                out[id(p)] = {"ev": play * fantasy, "if_plays": fantasy, "play": play, "min": minutes,
                              "opp": opp, "home": is_home, "win": pw, "margin": m,
                              "date": g["localDate"]}  # fmt: skip
    return out, ratings


# ================================================================== calibration
def fit_calibration(rows, settings: CalibrationSettings) -> dict[str, list[tuple]]:
    """group -> [(tier midpoint, shrunk correction, raw bias, n)].

    Observed bias ~ N(true bias, se^2) with se = CV * projected / sqrt(n);
    true bias ~ N(0, prior_sd^2).
    """
    out = {}
    for grp in ("same club", "changed club", "new to EL"):
        pts = []
        for lo, hi in settings.tiers:
            sel = [(x, y) for g, x, y in rows if g == grp and lo <= x < hi]
            if not sel:
                continue
            x = np.array([s[0] for s in sel])
            y = np.array([s[1] for s in sel])
            raw = float((y - x).mean())
            se = settings.cv * max(float(x.mean()), 3.0) / np.sqrt(len(sel))
            w = settings.prior_sd**2 / (settings.prior_sd**2 + se**2)
            pts.append((float(x.mean()), w * raw, raw, len(sel)))
        out[grp] = pts
    return out


def calibrate(cal, grp: str, projected: float) -> float:
    pts = cal.get(grp, [])
    if not pts:
        return max(projected, 0.0)
    corr = float(np.interp(projected, [p[0] for p in pts], [p[1] for p in pts]))
    return max(projected + corr, 0.0)


# ======================================================================= prices
def coach_points(margin: float) -> int:
    if margin > 0:
        return 10 if margin <= 10 else (20 if margin <= 20 else 25)
    return -5 if margin >= -10 else (-10 if margin >= -20 else -20)


def coach_price_change(points: float, price: float) -> float:
    """Fitted on all 20 Round 1 coach changes (residual SD 0.03): -10 pts ->
    -0.4, -5 -> -0.3, +10 -> +0.1, +20 -> +0.3; +25 -> +0.4 matched Obradovic
    over Rounds 2-3."""
    return round(-0.08 + 0.0246 * points - 0.012 * price, 1)


def estimate_prices(players, coaches, target, data, season, rules, games, private=None):
    """Prices for ``target``: the latest published list at or before it plus
    each later round's fitted change; the app's prices where given."""

    lists = data.price_lists(season)
    base_round = max([1] + [r for r in lists if r <= target])
    rounds = range(base_round, target)
    boxes = data.boxes(rounds) if rounds else {}
    margin = {}
    for g in games.values():
        if g["round"] in rounds and g["played"]:
            d = g["local"]["score"] - g["road"]["score"]
            margin[(g["round"], g["local"]["club"]["code"])] = d
            margin[(g["round"], g["road"]["club"]["code"])] = -d
    for x in players + coaches:
        k = _pkey(x)
        price = lists.get(base_round, {}).get(k, x["price"])  # round 1: the opening price
        for rnd in rounds:
            if x in coaches:
                price = round(
                    price + coach_price_change(coach_points(margin[(rnd, x["club"])]), price), 1
                )
            else:
                ln = boxes[rnd].get(x.get("code"))
                played = bool(ln and ln["minutes"] > 0)
                change = float(
                    price_change(ln["pir"] if played else 0.0, price, rules, played=played)
                )
                price = round(price + change, 1)
        x["price_estimated"] = price
        x["price"] = private.prices.get(k, price) if private else price


# ===================================================================== assembly
class SeasonModel:
    """Projects any round of the season from its inputs, as known before it."""

    def __init__(self, season: SeasonInputs, data: SeasonData, rules: GameRules) -> None:
        self.season = season
        self.data = data
        self.rules = rules
        self.games = data.games()
        self._rounds: dict[int, RoundInputs] = {}

    def round_inputs(self, n: int) -> RoundInputs:
        if n not in self._rounds:
            from elfantasy.season.inputs import load_round

            self._rounds[n] = load_round(self.season.root, n)
        return self._rounds[n]

    def players_before(self, target: int):
        """Prior updated on every round before ``target``."""
        players, coaches, fit_info = build_players(self.data, self.season)
        evidence = list(range(1, target))
        if evidence:
            update_with_evidence(
                players,
                self.games,
                self.data.boxes(evidence),
                {r: self.round_inputs(r).missed_injured() for r in evidence},
                self.season.evidence,
                {r: self.round_inputs(r).minutes_expert for r in range(1, target + 1)},
            )
        return players, coaches, fit_info

    def raw_round(self, players, fit_info, target, round_no, k):
        ri = self.round_inputs(target)
        return project_round(
            players, fit_info, self.games, round_no, ri.availability, ri.moneylines.get(round_no, {}),
            ri.rating_lines(), self.season.outrights, k=k,
            minutes_factor=self.season.minutes_factor_round1 if target == 1 else None,
        )  # fmt: skip

    def actuals(self, round_no: int) -> dict[str, float]:
        """player code -> fantasy points (PIR, +10% for a win) for those who played."""
        won = {}
        for g in self.games.values():
            if g["round"] == round_no and g["played"]:
                h, a = g["local"]["score"], g["road"]["score"]
                won[g["local"]["club"]["code"]], won[g["road"]["club"]["code"]] = h > a, a > h
        out = {}
        for ln in self.data.boxes([round_no])[round_no].values():
            if ln["minutes"] > 0:
                out[ln["player_id"]] = (ln["pir"] or 0.0) * (1.1 if won[ln["team_code"]] else 1.0)
        return out

    def reproject(self, round_no: int) -> list[dict]:
        """Round ``round_no`` as today's model would have projected it then:
        one row per player expected to play (P >= 0.5), with what happened."""
        players, coaches, fit_info = self.players_before(round_no)
        estimate_prices(players, coaches, round_no, self.data, self.season, self.rules, self.games)
        proj, _ = self.raw_round(players, fit_info, round_no, round_no, k=1)
        act = self.actuals(round_no)
        rows = []
        for p in players:
            r = proj.get(id(p))
            if not r or not p.get("code") or r["play"] < 0.5:
                continue
            rows.append({"name": p["name"], "club": p["club"], "pos": p["pos"], "group": prior_group(p),
                         "price": p["price"], "ev": r["ev"], "min_p": r["min"], "play": r["play"],
                         "actual": act.get(p["code"], 0.0), "round": round_no})  # fmt: skip
        return rows

    def calibration(self, target: int, rows_by_round: dict[int, list[dict]] | None = None):
        """Calibration from re-projections of every round before ``target``."""
        rows = []
        for r in range(1, target):
            rr = rows_by_round[r] if rows_by_round and r in rows_by_round else self.reproject(r)
            rows += [(x["group"], x["ev"], x["actual"]) for x in rr]
        return fit_calibration(rows, self.season.calibration)

    def project(self, target: int, private: PrivateInputs | None = None, cal=None) -> dict:
        """Projections for ``target`` and the following rounds of the horizon."""

        players, coaches, fit_info = self.players_before(target)
        estimate_prices(
            players, coaches, target, self.data, self.season, self.rules, self.games, private
        )
        cal = self.calibration(target) if cal is None else cal
        rounds = [target + i for i in range(self.season.horizon)]
        for x in players + coaches:
            x["proj"] = {}
        ratings = {}
        for k, rnd in enumerate(rounds, start=1):
            if not any(g["round"] == rnd for g in self.games.values()):
                continue
            proj, ratings = self.raw_round(players, fit_info, target, rnd, k)
            days = sorted({g["localDate"][:10] for g in self.games.values() if g["round"] == rnd})
            for p in players:
                r = proj.get(id(p))
                if not r:
                    continue
                raw = r["ev"]
                ev = calibrate(cal, prior_group(p), raw) * self.season.eye_multiplier(
                    _pkey(p), target
                )
                scale = ev / raw if raw > 0 else 1.0
                p["proj"][str(rnd)] = {**r, "ev": ev, "raw_ev": raw, "if_plays": r["if_plays"] * scale,
                                       "turn": days.index(r["date"][:10])}  # fmt: skip
            for c in coaches:
                m = next(
                    (
                        v["margin"]
                        for p in players
                        if p["club"] == c["club"] and (v := proj.get(id(p)))
                    ),
                    None,
                )
                if m is not None:
                    c["proj"][str(rnd)] = {"ev": coach_ev(m), "margin": m}
        return {"round": target, "rounds": rounds, "players": players, "coaches": coaches,
                "ratings": ratings, "calibration": cal}  # fmt: skip


def save_projection(result: dict, path: Path) -> None:
    path.write_text(
        json.dumps(result, indent=1, ensure_ascii=False, default=float), encoding="utf-8"
    )


__all__ = [
    "SeasonModel",
    "build_players",
    "calibrate",
    "coach_ev",
    "coach_points",
    "coach_price_change",
    "estimate_prices",
    "fit_calibration",
    "fit_ratings",
    "normalise_name",
    "prior_group",
    "project_round",
    "save_projection",
    "spread_from_prices",
    "team_round",
    "update_with_evidence",
]
