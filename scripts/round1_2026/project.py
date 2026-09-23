"""Round 1 projections for EuroLeague Fantasy Challenge 2026-27.

Pipeline
  1. Per-player history from every 2025-26 box score (minutes, PIR/min, SD).
  2. Price-implied priors: the game's opening prices are the operator's own
     forecast, and are the only information for players new to the league.
  3. Baseline = history for returning roles, leaning on price for new roles.
  4. Team minute budget (200), then injury redistribution by position.
  5. Game context from de-vigged moneylines: expected margin -> team PIR,
     garbage time, win bonus, coach score.
  6. Rounds 2-3 from team ratings fitted to R1 lines + outright odds.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from inputs import (  # noqa: E402
    AVAILABILITY,
    MINUTES_EXPERT,
    MINUTES_FACTOR_R1,
    OUTRIGHTS,
    R1_MONEYLINES,
    ROLE,
)
from paths import DATA  # noqa: E402

from elfantasy.data.injuries import normalise_name  # noqa: E402
from elfantasy.projection import market as mk  # noqa: E402

SIGMA = 11.5  # SD of EuroLeague game margin around the spread
HCA = 3.5  # home-court advantage, points
TEAM_MINUTES = 200.0
WIN_BONUS = 0.10  # +10% of the fantasy score when the team wins


def key(club: str, name: str) -> tuple[str, str]:
    return club, normalise_name(name)


def lookup(table: dict, club: str, name: str, default):
    k = key(club, name)
    for (c, n), v in table.items():
        if (c, normalise_name(n)) == k:
            return v
    return default


# --------------------------------------------------------------------------
# 1. history
# --------------------------------------------------------------------------
def load_cached_games():
    """The 66 full box scores fetched before the API rate-limited us.

    Too few for per-player histories, plenty for two league-level fits:
    team PIR vs margin, and game-to-game PIR volatility.
    """
    team_pir = defaultdict(float)
    team_pts = defaultdict(float)
    player_games = defaultdict(list)
    for f in (DATA / "http").rglob("*.json"):
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


def load_aggregates():
    """Every 2025-26 player's season totals (no games-played qualifier)."""
    rows = json.loads(Path(DATA / "stats_E2025_acc.json").read_text(encoding="utf-8"))["players"]
    return {r["player"]["code"]: r for r in rows}


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


# --------------------------------------------------------------------------
# 2-4. baselines, team budget, injuries
# --------------------------------------------------------------------------
def build_players():
    joined = json.loads(Path(DATA / "joined.json").read_text(encoding="utf-8"))
    team_pir, team_pts, player_games = load_cached_games()
    if team_pts:
        a_pir, b_pir, n_games = fit_margin_to_team_pir(team_pir, team_pts)
    else:  # fitted on 66 games of 2025-26 before the feed rate-limited us
        a_pir, b_pir, n_games = 93.5, 1.09, 0
    aggs = load_aggregates()

    # League-wide game-to-game volatility: SD of PIR relative to its mean,
    # from players with 4+ ten-minute games in the cached sample.
    cvs = [
        np.std(v, ddof=1) / np.mean(v)
        for v in player_games.values()
        if len(v) >= 4 and np.mean(v) >= 5
    ]
    league_cv = float(np.median(cvs)) if cvs else 0.634  # same 66 games

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
        role = lookup(ROLE, p["club"], p["name"], None)
        if role:
            pir *= role[0]
            minutes = min(minutes * (1 + 0.8 * (role[0] - 1)), 34.0)
            p["role_note"] = role[1]
        expert_min = lookup(MINUTES_EXPERT, p["club"], p["name"], None)
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


def team_round(players, club, rnd, spread_for_team):
    """Minutes and PIR for one team in one round, conditional on each player playing."""
    roster = [p for p in players if p["club"] == club]
    avail = {id(p): lookup(AVAILABILITY, club, p["name"], (1.0, 1.0, 1.0))[rnd - 1] for p in roster}
    healthy_default = 0.97  # the odd late scratch nobody reported
    for p in roster:
        if lookup(AVAILABILITY, club, p["name"], None) is None:
            avail[id(p)] = healthy_default

    # Rotation structure. Rosters are 16-21 deep but coaches play about ten,
    # so claims beyond the ninth man are discounted before the 200-minute cap.
    # Without this, fifteen bench players each claiming a few minutes force
    # every starter down proportionally.
    RANK_SHARE = [1.0] * 9 + [0.8, 0.6, 0.4]

    def share(rank):
        return RANK_SHARE[rank] if rank < len(RANK_SHARE) else 0.15

    # Healthy world: the team's normal minute distribution. If the claims
    # exceed 200, the excess comes mostly out of the back of the rotation --
    # coaches cut the ninth man's minutes, not the star's. A uniform scale
    # would shave 15% off every star on a deep roster.
    # Players with a stated minutes estimate keep it; the rest of the roster
    # absorbs the adjustment.
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
        for k in healthy:
            healthy[k] = max(healthy[k] - excess * w[k] / tot_w, 0.0)
    scale = 1.0
    mins = dict(healthy)

    # Absences. First, backups move up the rotation: a player who was the 11th
    # man is the 7th man once four teammates are out, and reclaims the minutes
    # the rank discount had taken from him. Only then does the rest of the
    # vacated time spread by position and headroom.
    available = [q for q in roster if avail[id(q)] >= 0.35]
    vacated_total = sum(healthy[id(q)] * (1 - avail[id(q)]) * 0.92 for q in roster)
    restored = {}
    for rank, q in enumerate(sorted(available, key=lambda q: -q["base_min"])):
        if q.get("minutes_locked"):
            restored[id(q)] = 0.0
            continue
        restored[id(q)] = max(q["base_min"] * share(rank) * scale - healthy[id(q)], 0.0)
    gain = defaultdict(float)
    pool_restore = sum(restored.values())
    used = min(pool_restore, vacated_total)
    if pool_restore > 0:
        for k, v in restored.items():
            gain[k] += used * v / pool_restore
    remaining = vacated_total - used

    for p in roster:
        miss = 1 - avail[id(p)]
        if miss <= 0.02 or remaining <= 0:
            continue
        # Each absentee's share of what is left, spread by position overlap.
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
        for k, w in weights.items():
            gain[k] += vacated * w / tot

    # Garbage time from the spread.
    garbage = min(max(0.55 * (abs(spread_for_team) - 9.0), 0.0), 9.0)
    order = sorted(roster, key=lambda q: -mins[id(q)])
    starters = {id(q) for q in order[:5]}
    bench = [q for q in order[5:11]]

    out = {}
    for p in roster:
        m = mins[id(p)] + gain[id(p)]
        if rnd == 1:
            m *= lookup(MINUTES_FACTOR_R1, club, p["name"], 1.0)
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


# --------------------------------------------------------------------------
# 5. market
# --------------------------------------------------------------------------
def spread_from_prices(home_price, away_price):
    p_home = mk.two_way_prob(home_price, away_price, "shin")
    return p_home, float(norm.ppf(p_home) * SIGMA)  # expected home margin


def fit_ratings():
    """Team ratings (points) from R1 margins plus an outright-odds prior."""
    teams = sorted(OUTRIGHTS)
    idx = {t: i for i, t in enumerate(teams)}
    fair = mk.devig(list(OUTRIGHTS.values()), "multiplicative")
    logp = {t: math.log(p) for t, p in zip(OUTRIGHTS, fair, strict=True)}
    n = len(teams)
    rows, rhs = [], []
    for (h, a), (ph, pa) in R1_MONEYLINES.items():
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


def coach_ev(margin_mean):
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


def fixtures():
    rows = json.loads(Path(DATA / "games_E2026.json").read_text(encoding="utf-8"))["data"]
    out = defaultdict(list)
    for r in rows:
        if r["round"] <= 3:
            out[r["round"]].append(
                (r["local"]["club"]["code"], r["road"]["club"]["code"], r["localDate"])
            )
    return out


# --------------------------------------------------------------------------
# 6. assemble
# --------------------------------------------------------------------------
def project():
    players, coaches, fit_info = build_players()
    ratings = fit_ratings()
    a_pir, b_pir = fit_info["team_pir"]
    fx = fixtures()

    for p in players:
        p["proj"] = {}
    for c in coaches:
        c["proj"] = {}

    for rnd in (1, 2, 3):
        for home, away, when in fx[rnd]:
            if rnd == 1 and (home, away) in R1_MONEYLINES:
                p_home, margin = spread_from_prices(*R1_MONEYLINES[(home, away)])
                src = "market"
            else:
                margin = ratings[home] - ratings[away] + HCA
                p_home = float(norm.cdf(margin / SIGMA))
                src = "ratings"
            for club, m, pw, opp, is_home in (
                (home, margin, p_home, away, True),
                (away, -margin, 1 - p_home, home, False),
            ):
                # Matchup environment. Team PIR rises ~1.09 per point of margin
                # (2025-26 box scores), but a player's baseline already reflects
                # his own team's quality -- only the part of the expected margin
                # due to *this* opponent and venue is new information. A
                # predicted margin also carries less signal than an actual one
                # (much of the actual-margin slope is shared shooting luck), so
                # the slope is halved.
                matchup = m - ratings[club]
                env = 1.0 + 0.5 * b_pir * matchup / a_pir
                team = team_round(players, club, rnd, m)
                for p in players:
                    if p["club"] != club:
                        continue
                    minutes, pir_if_plays, play = team[id(p)]
                    exp_pir = pir_if_plays * env
                    fantasy = exp_pir * (1 + WIN_BONUS * pw)
                    p["proj"][rnd] = {
                        "ev": play * fantasy,
                        "if_plays": fantasy,
                        "play": play,
                        "min": minutes,
                        "sd": max(2.5, p["sd_per_pir"] * max(fantasy, 2.0)),
                        "opp": opp,
                        "home": is_home,
                        "win": pw,
                        "turn": "Thu"
                        if when.startswith("2026-09-24")
                        else ("Fri" if when.startswith("2026-09-25") else when[:10]),
                        "src": src,
                    }
                for c in coaches:
                    if c["club"] == club:
                        c["proj"][rnd] = {
                            "ev": coach_ev(m),
                            "win": pw,
                            "opp": opp,
                            "home": is_home,
                            "margin": m,
                        }

    (DATA / "projections.json").write_text(
        json.dumps(
            {
                "players": players,
                "coaches": coaches,
                "ratings": ratings,
                "fit": {k: v for k, v in fit_info.items()},
            },
            indent=1,
            ensure_ascii=False,
            default=float,
        ),
        encoding="utf-8",
    )
    return players, coaches, ratings, fit_info


if __name__ == "__main__":
    players, coaches, ratings, fit = project()
    print(
        f"price fit on {fit['n']} returning players: "
        f"PIR = {fit['pir'][0]:.2f} + {fit['pir'][1]:.3f}*price, "
        f"MIN = {fit['min'][0]:.2f} + {fit['min'][1]:.3f}*price"
    )
    print(f"position PIR/min: { ({k: round(v, 3) for k, v in fit['pos_rate'].items()}) }")
    print(f"team PIR = {fit['team_pir'][0]:.1f} + {fit['team_pir'][1]:.2f}*margin")
    print("\nratings:", {t: round(r, 1) for t, r in sorted(ratings.items(), key=lambda kv: -kv[1])})
    print("\n=== top 30 Round 1 expected fantasy points ===")
    top = sorted(players, key=lambda p: -p["proj"][1]["ev"])[:30]
    for p in top:
        r1 = p["proj"][1]
        print(
            f"{p['price']:5.1f} {p['pos']} {p['club']} {p['name'][:24]:24} ev={r1['ev']:5.1f} "
            f"min={r1['min']:4.1f} play={r1['play']:.2f} vs {r1['opp']} {'H' if r1['home'] else 'A'} "
            f"win={r1['win']:.2f} {r1['turn']}"
        )
    print("\n=== coaches ===")
    for c in sorted(coaches, key=lambda c: -c["proj"][1]["ev"]):
        r = c["proj"]
        print(
            f"{c['price']:5.1f} {c['club']} {c['name']:22} R1={r[1]['ev']:5.1f} "
            f"R2={r[2]['ev']:5.1f} R3={r[3]['ev']:5.1f}  (R1 win {r[1]['win']:.2f})"
        )
