"""A synthetic league, so the project runs and is testable with no network.

The club codes are real EuroLeague codes; **everything else is invented** --
players, prices, statistics and odds are generated from a seeded random process
and bear no relation to any real person's performance. That is deliberate: a
fixture full of plausible-looking fake statistics attributed to real players
would be worse than useless.

The generator is not a toy, though. It reproduces the structure the model cares
about -- a talent hierarchy, a rotation with a real bench, home advantage,
pace differences, correlated team strength, injuries, and bookmaker prices
derived from the same latent strengths with a vig added on top -- so tests
exercise the real code paths.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from elfantasy.dataset import Dataset
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
from elfantasy.projection.pir import pir_from_boxscore

# fmt: off  (kept tabular: a lookup table, easier to scan in rows)
CLUB_CODES = [
    ("MAD", "Club MAD"),
    ("BAR", "Club BAR"),
    ("OLY", "Club OLY"),
    ("PAN", "Club PAN"),
    ("IST", "Club IST"),
    ("ULK", "Club ULK"),
    ("ASV", "Club ASV"),
    ("BER", "Club BER"),
    ("MIL", "Club MIL"),
    ("VIR", "Club VIR"),
    ("ZAL", "Club ZAL"),
    ("RED", "Club RED"),
    ("PRS", "Club PRS"),
    ("MCO", "Club MCO"),
    ("BAS", "Club BAS"),
    ("TEL", "Club TEL"),
]
# fmt: on

ROSTER_SHAPE = [
    (Position.GUARD, 4),
    (Position.FORWARD, 4),
    (Position.CENTER, 3),
]


def build_sample_dataset(
    *,
    seed: int = 7,
    n_teams: int = 12,
    rounds_played: int = 10,
    rounds_future: int = 5,
    injury_rate: float = 0.09,
    with_odds: bool = True,
    with_props: bool = True,
) -> Dataset:
    rng = random.Random(seed)
    ds = Dataset(season="SAMPLE", fetched_at=datetime(2026, 1, 15, 12, 0))
    ds.sources["generator"] = f"synthetic seed={seed}"

    codes = [c for c, _ in CLUB_CODES[:n_teams]]
    ds.teams = {c: Team(code=c, name=n) for c, n in CLUB_CODES[:n_teams]}

    # --- latent team strength and pace --------------------------------------
    strength = {c: rng.gauss(0, 4.2) for c in codes}
    pace = {c: rng.gauss(72.0, 3.0) for c in codes}

    # --- rosters -------------------------------------------------------------
    talent: dict[str, float] = {}
    minutes_role: dict[str, float] = {}
    counter = 0
    for code in codes:
        rank = 0
        for position, count in ROSTER_SHAPE:
            for _ in range(count):
                counter += 1
                pid = f"P{counter:04d}"
                # Talent is a team effect plus a within-roster hierarchy.
                tier = rng.gauss(0, 1.0) + max(0.0, 2.2 - 0.42 * rank)
                # `talent` is points per minute; the minutes-weighted league
                # average lands near 0.42, i.e. ~82 points per team-game.
                talent[pid] = 0.34 + 0.055 * tier + 0.010 * strength[code]
                minutes_role[pid] = max(4.0, 31.0 - 2.3 * rank + rng.gauss(0, 2.0))
                price = round(min(max(2.5 + 11.0 * (talent[pid] - 0.32) / 0.25, 2.0), 14.0), 1)
                ds.players[pid] = Player(
                    player_id=pid,
                    name=f"Player {counter:03d}",
                    team_code=code,
                    position=position,
                    price=price,
                    status=Availability.ACTIVE,
                    ownership=round(rng.random() * 0.4, 3),
                )
                rank += 1

    # --- schedule: a simple rotating pairing --------------------------------
    total_rounds = rounds_played + rounds_future
    start = datetime(2025, 10, 2, 20, 0)
    game_no = 0
    for rnd in range(1, total_rounds + 1):
        shuffled = codes[:]
        offset = rnd % max(len(codes) - 1, 1)
        rotated = [shuffled[0]] + shuffled[1 + offset :] + shuffled[1 : 1 + offset]
        half = len(rotated) // 2
        for i in range(half):
            home, away = rotated[i], rotated[len(rotated) - 1 - i]
            if rnd % 2 == 0:
                home, away = away, home
            game_no += 1
            ds.games.append(
                Game(
                    game_id=f"G{game_no:04d}",
                    round=rnd,
                    home_code=home,
                    away_code=away,
                    tipoff=start + timedelta(days=7 * (rnd - 1), hours=i),
                    played=rnd <= rounds_played,
                )
            )

    # --- results and box scores ---------------------------------------------
    for game in ds.games:
        if not game.played:
            continue
        home_pts, away_pts = 0, 0
        for code in (game.home_code, game.away_code):
            home = code == game.home_code
            edge = strength[code] - strength[game.opponent_of(code)] + (2.8 if home else -2.8)
            team_pace = 0.5 * (pace[game.home_code] + pace[game.away_code])
            roster = ds.team_players(code)
            night = rng.gauss(0, 0.10)  # whole-team hot/cold night
            for p in roster:
                mins = max(0.0, rng.gauss(minutes_role[p.player_id], 4.0))
                mins = min(mins, 36.0)
                if rng.random() < 0.07:  # rest / rotation choice
                    mins = 0.0
                if mins <= 0:
                    continue
                rate = talent[p.player_id] * (1 + night + 0.004 * edge) * (team_pace / 72.0)
                line = _simulate_line(rng, p.position, mins, rate)
                bs = BoxScore(
                    game_id=game.game_id,
                    round=game.round,
                    player_id=p.player_id,
                    team_code=code,
                    minutes=round(mins, 1),
                    started=minutes_role[p.player_id] > 24,
                    **line,
                )
                bs.pir = pir_from_boxscore(bs)
                ds.boxscores.append(bs)
                if home:
                    home_pts += int(bs.points)
                else:
                    away_pts += int(bs.points)
        game.home_score, game.away_score = home_pts, away_pts

    # --- injuries for the upcoming round ------------------------------------
    upcoming = rounds_played + 1
    for p in ds.players.values():
        r = rng.random()
        if r < injury_rate * 0.5:
            p.status = Availability.OUT
            p.status_note = "synthetic injury"
        elif r < injury_rate:
            p.status = Availability.QUESTIONABLE
            p.status_note = "synthetic knock"

    # --- market prices -------------------------------------------------------
    if with_odds:
        for game in ds.games:
            if game.played or game.round > upcoming + 1:
                continue
            margin = strength[game.home_code] - strength[game.away_code] + 2.8
            total = 0.5 * (pace[game.home_code] + pace[game.away_code]) * 2.24
            p_home = _norm_cdf(margin / 11.0)
            vig = 1.045
            ds.odds[game.game_id] = GameOdds(
                game_id=game.game_id,
                home_price=round(1.0 / (p_home / vig), 3),
                away_price=round(1.0 / ((1 - p_home) / vig), 3),
                spread=round(-margin * 2) / 2,
                total=round(total * 2) / 2,
                bookmaker="synthetic",
                captured_at=ds.fetched_at,
            )

    if with_props:
        for game in ds.games_in_round(upcoming):
            for code in (game.home_code, game.away_code):
                for p in ds.team_players(code):
                    if ds.players[p.player_id].status is Availability.OUT:
                        continue
                    mins = minutes_role[p.player_id]
                    if mins < 14:
                        continue  # books only price rotation players
                    exp_pts = talent[p.player_id] * mins
                    exp_reb = mins * (0.225 if p.position is Position.CENTER else 0.13)
                    exp_ast = mins * (0.130 if p.position is Position.GUARD else 0.058)
                    for market, mean in (
                        ("points", exp_pts),
                        ("rebounds", exp_reb),
                        ("assists", exp_ast),
                    ):
                        if mean < 1.5:
                            continue
                        line = round(mean * 2) / 2
                        if line == int(line):
                            line += 0.5
                        ds.props.append(
                            PlayerProp(
                                game_id=game.game_id,
                                player_id=p.player_id,
                                market=market,
                                line=line,
                                over_price=round(rng.uniform(1.80, 1.95), 2),
                                under_price=round(rng.uniform(1.80, 1.95), 2),
                                bookmaker="synthetic",
                            )
                        )

    ds.annotate_home_flags()
    return ds


def _simulate_line(rng: random.Random, position: Position, minutes: float, rate: float) -> dict:
    """Generate a plausible, internally consistent box score line."""

    pos = position.value
    reb_rate = {"G": 0.110, "F": 0.155, "C": 0.225}[pos]
    ast_rate = {"G": 0.130, "F": 0.065, "C": 0.050}[pos]
    blk_rate = {"G": 0.005, "F": 0.014, "C": 0.032}[pos]
    pf_rate = {"G": 0.078, "F": 0.098, "C": 0.115}[pos]

    def draw(mean: float) -> float:
        return float(max(0, round(rng.gauss(mean, max(mean, 0.6) ** 0.5))))

    points = draw(rate * minutes)
    # A EuroLeague player scoring N points takes roughly 0.72N + 0.8 shots and
    # makes about 0.40N of them (the rest of the points come from threes and
    # free throws), which reproduces league-average shooting splits.
    fg_attempted = draw(points * 0.72 + 0.8)
    fg_made = min(fg_attempted, draw(points * 0.40))
    ft_attempted = draw(points * 0.24)
    ft_made = min(ft_attempted, draw(ft_attempted * 0.78))

    return {
        "points": points,
        "rebounds": draw(reb_rate * minutes),
        "assists": draw(ast_rate * minutes),
        "steals": draw(0.034 * minutes),
        "blocks": draw(blk_rate * minutes),
        "fouls_drawn": draw(0.10 * minutes),
        "fg_made": fg_made,
        "fg_attempted": max(fg_attempted, fg_made),
        "ft_made": ft_made,
        "ft_attempted": max(ft_attempted, ft_made),
        "turnovers": draw(0.058 * minutes),
        "blocks_against": draw(0.011 * minutes),
        "fouls_committed": draw(pf_rate * minutes),
    }


def _norm_cdf(x: float) -> float:
    import math

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
