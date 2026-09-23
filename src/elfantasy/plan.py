"""From projections to a lineup you can actually play.

Glue between the projection engine and the two optimisers:

1. turn the pipeline's per-round projections into fantasy points (PIR plus the
   expected win bonus), per-round coach expectations, and Turn membership;
2. solve the lineup MILP for a starting roster (or the best transfers from a
   current squad);
3. price a credit from the MILP's shadow price, then hill-climb on the Turn
   simulator, which values promotions, captain switches and price changes;
4. return the roster, the pre-Turn lineup and the numbers behind the
   after-first-Turn instructions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from elfantasy import scoring
from elfantasy.models import Projection, Squad
from elfantasy.optimize.lineup import (
    CurrentSquad,
    LineupCoach,
    LineupPlayer,
    LineupSolution,
    future_slot_weight,
    optimise_lineup,
)
from elfantasy.optimize.squad import InfeasibleError
from elfantasy.optimize.turns import (
    SimCoach,
    SimPlayer,
    TurnEvaluation,
    TurnSimulator,
    candidate_pool,
    local_search,
)
from elfantasy.rules import GameRules


@dataclass
class LineupPlan:
    sim: TurnSimulator
    ids: list[int]
    coach: int | None
    evaluation: TurnEvaluation
    static: LineupSolution  # the MILP's starting point
    credit_value: float
    static_first_round: float = 0.0  # the final roster, best fixed lineup, no Turn moves
    sold: list[str] = field(default_factory=list)  # player ids
    bought: list[str] = field(default_factory=list)
    coach_changed: bool = False
    log: list[str] = field(default_factory=list)

    @property
    def players(self) -> list[SimPlayer]:
        return [self.sim.players[i] for i in self.ids]

    @property
    def captain(self) -> SimPlayer:
        return self.sim.players[self.evaluation.lineup.captain]

    def late_bench(self) -> list[int]:
        lu = self.evaluation.lineup
        return [i for i in self.ids if self.sim.late[i] and i not in lu.starters and i != lu.sixth]


def turn_of_games(dataset, round_no: int) -> dict[str, bool]:
    """game_id -> True if the game is played after the round's first day."""

    games = dataset.games_in_round(round_no)
    days = sorted({g.tipoff.date() for g in games if g.tipoff})
    first = days[0] if days else None
    return {g.game_id: bool(first and g.tipoff and g.tipoff.date() > first) for g in games}


def team_margins(
    projections: dict[str, Projection], dataset
) -> dict[str, tuple[float, float, str]]:
    """club -> (expected margin, win probability, opponent) for one round."""

    out = {}
    for pid, proj in projections.items():
        player = dataset.players.get(pid)
        if player is None or player.team_code in out or proj.opponent is None:
            continue
        spread = proj.components.get("spread")
        if spread is None or proj.win_prob is None:
            continue
        margin = -spread if proj.home else spread
        out[player.team_code] = (float(margin), float(proj.win_prob), proj.opponent)
    return out


def build_inputs(pipe, rules: GameRules, gamma: float = 0.4):
    """LineupPlayers/Coaches (for the MILP) and SimPlayers/Coaches (for Turns)."""

    ds = pipe.dataset
    rounds = sorted(pipe.horizon)
    first = rounds[0]
    late = turn_of_games(ds, first)
    w_future = future_slot_weight(rules)
    margins = {r: team_margins(pipe.horizon[r], ds) for r in rounds}

    ev: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rounds:
        for pid, proj in pipe.horizon[r].items():
            win = proj.win_prob or 0.5
            ev[pid][r] = proj.mean_pir * scoring.expected_win_bonus_factor(win, rules)

    lineup_players, sim_players = [], []
    for pid, player in ds.players.items():
        if player.price <= 0 or pid not in ev:
            continue
        proj = pipe.horizon[first].get(pid)
        is_late = bool(proj and late.get(proj.game_id, False))
        lineup_players.append(
            LineupPlayer(
                player_id=pid,
                name=player.name,
                pos=player.position.value,
                club=player.team_code,
                price=player.price,
                ev=dict(ev[pid]),
                late=is_late,
            )
        )
        if proj is None:
            continue
        future = w_future * sum(
            gamma ** (k + 1) * ev[pid].get(r, 0.0) for k, r in enumerate(rounds[1:])
        )
        fantasy = ev[pid][first]
        play = proj.play_prob
        sim_players.append(
            SimPlayer(
                player_id=pid,
                name=player.name,
                pos=player.position.value,
                club=player.team_code,
                opponent=proj.opponent or "",
                price=player.price,
                late=is_late,
                points_if_available=fantasy / play if play > 0 else 0.0,
                play_prob=play,
                minutes=proj.minutes,
                win_prob=proj.win_prob or 0.5,
                sd_ratio=float(np.clip(proj.sd_pir / max(proj.mean_pir, 1.0), 0.3, 1.2)),
                future_value=future,
            )
        )

    lineup_coaches, sim_coaches = [], []
    for cid, coach in ds.coaches.items():
        cev = {
            r: scoring.expected_coach_points(margins[r][coach.team_code][0], rules)
            for r in rounds
            if coach.team_code in margins[r]
        }
        lineup_coaches.append(LineupCoach(cid, coach.name, coach.team_code, coach.price, cev))
        future = sum(gamma ** (k + 1) * cev.get(r, 0.0) for k, r in enumerate(rounds[1:]))
        sim_coaches.append(SimCoach(cid, coach.name, coach.team_code, coach.price, future))
    return rounds, lineup_players, lineup_coaches, sim_players, sim_coaches


def plan_lineup(
    pipe,
    rules: GameRules,
    *,
    squad: Squad | None = None,
    max_transfers: int | None = None,
    gamma: float = 0.4,
    n: int = 6000,
    seed: int = 11,
    search: bool = True,
    thorough: bool = False,
    credit_multiplier: float = 1.0,
    log: Callable[[str], None] | None = None,
) -> LineupPlan:
    """Best roster and lineup for the next round, Turn mechanics included."""

    rounds, lp, lc, sp, sc = build_inputs(pipe, rules, gamma)
    messages: list[str] = []

    def say(msg: str) -> None:
        messages.append(msg)
        if log:
            log(msg)

    current = None
    budget = rules.budget
    if squad is not None:
        prices = {p.player_id: p.price for p in lp}
        coach_prices = {c.coach_id: c.price for c in lc}
        current = CurrentSquad(list(squad.player_ids), squad.coach_id, squad.bank)
        budget = squad.bank + sum(prices.get(pid, 0.0) for pid in squad.player_ids)
        budget += coach_prices.get(squad.coach_id, 0.0) if squad.coach_id else 0.0
        if max_transfers is None and not rules.unlimited_transfers_before(rounds[0]):
            max_transfers = rules.transfers_per_round

    kwargs = {"rounds": rounds, "gamma": gamma, "current": current, "max_transfers": max_transfers}
    if current is None:
        kwargs["budget"] = budget
    static = optimise_lineup(lp, lc, rules, **kwargs)

    # What one more credit is worth, in objective points: the exchange rate for
    # price changes and for credits left in the bank.
    try:
        if current is None:
            richer = optimise_lineup(lp, lc, rules, **{**kwargs, "budget": budget + 1.0})
        else:
            richer_squad = CurrentSquad(current.player_ids, current.coach_id, current.bank + 1.0)
            richer = optimise_lineup(lp, lc, rules, **{**kwargs, "current": richer_squad})
        credit_value = max(richer.objective - static.objective, 0.0) * credit_multiplier
    except InfeasibleError:
        credit_value = 1.0 * credit_multiplier
    if credit_value > 0:
        say(f"one credit is worth {credit_value:.2f} points over the horizon")
    else:
        say("the budget is not binding: an extra credit buys nothing, so price changes are ignored")

    # Owned players the engine could not project (injured, no fixture) must
    # still be representable so they can be sold or kept.
    projected = {p.player_id for p in sp}
    for pid in current.player_ids if current else []:
        if pid not in projected and pid in pipe.dataset.players:
            pl = pipe.dataset.players[pid]
            sp.append(
                SimPlayer(
                    pid,
                    pl.name,
                    pl.position.value,
                    pl.team_code,
                    "",
                    pl.price,
                    False,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                )
            )

    sim = TurnSimulator(sp, sc, rules, n=n, seed=seed, budget=budget, credit_value=credit_value)
    ids = [sim.index[p.player_id] for p in static.roster if p.player_id in sim.index]
    coach = sim.coach_index.get(static.coach.coach_id) if static.coach else None

    if search and len(ids) == rules.squad_size:
        owned = (
            (
                {sim.index[pid] for pid in current.player_ids if pid in sim.index},
                sim.coach_index.get(current.coach_id) if current.coach_id else None,
            )
            if current
            else None
        )
        starts = [(ids, coach)]
        if thorough and current is None:
            n_late = sum(1 for p in lp if p.late)
            for k in (2, 3):
                if n_late >= k:
                    try:
                        alt = optimise_lineup(lp, lc, rules, **kwargs, min_late_bench=k)
                    except InfeasibleError:
                        continue
                    starts.append(
                        (
                            [
                                sim.index[p.player_id]
                                for p in alt.roster
                                if p.player_id in sim.index
                            ],
                            sim.coach_index.get(alt.coach.coach_id) if alt.coach else None,
                        )
                    )
        best = None
        cands = candidate_pool(sim)
        pairs = candidate_pool(sim, per_pos=8, by_value=4) if thorough else None
        for start_ids, start_coach in starts:
            if len(start_ids) != rules.squad_size:
                continue
            found = local_search(
                sim,
                start_ids,
                start_coach,
                cands,
                pair_candidates=pairs,
                current=owned,
                max_transfers=max_transfers,
                log=say,
            )
            if best is None or found[2] > best[2]:
                best = found
        if best is not None:
            ids, coach = best[0], best[1]

    evaluation = sim.evaluate(ids, coach, detail=True)
    plan = LineupPlan(
        sim=sim,
        ids=ids,
        coach=coach,
        evaluation=evaluation,
        static=static,
        credit_value=credit_value,
        log=messages,
    )
    # Same roster, best *fixed* lineup: what the Turn moves are adding.
    final_ids = {sim.players[i].player_id for i in ids}
    final_coach = sim.coaches[coach].coach_id if coach is not None else None
    try:
        fixed = optimise_lineup(
            [p for p in lp if p.player_id in final_ids],
            [c for c in lc if c.coach_id == final_coach],
            rules,
            rounds=[rounds[0]],
            budget=1e6,  # roster is fixed; the budget is irrelevant here
        )
        plan.static_first_round = fixed.first_round
    except InfeasibleError:
        plan.static_first_round = float("nan")
    if current is not None:
        roster = {sim.players[i].player_id for i in ids}
        plan.sold = [pid for pid in current.player_ids if pid not in roster]
        plan.bought = [pid for pid in roster if pid not in set(current.player_ids)]
        plan.coach_changed = bool(
            coach is not None
            and current.coach_id
            and sim.coaches[coach].coach_id != current.coach_id
        )
    return plan
