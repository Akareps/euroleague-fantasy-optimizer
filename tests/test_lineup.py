"""Lineup-aware MILP and the Turn simulator, on small pools you can reason about."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from elfantasy.optimize.lineup import (
    CurrentSquad,
    LineupCoach,
    LineupPlayer,
    future_slot_weight,
    optimise_lineup,
)
from elfantasy.optimize.squad import InfeasibleError
from elfantasy.optimize.turns import (
    SimCoach,
    SimPlayer,
    TurnSimulator,
    candidate_pool,
    local_search,
)
from elfantasy.rules import GameRules

# Two first-Turn fixtures (A-B, C-D) and one later fixture (E-F).
FIXTURES = {"A": ("B", False), "B": ("A", False), "C": ("D", False), "D": ("C", False),
            "E": ("F", True), "F": ("E", True)}  # fmt: skip


@pytest.fixture(scope="module")
def rules(settings):
    return GameRules.from_settings(settings)


def make_players(n_per_pos: int = 9, seed: int = 3) -> list[LineupPlayer]:
    rng = np.random.default_rng(seed)
    clubs = list(FIXTURES)
    out = []
    for pos in "GFC":
        for k in range(n_per_pos):
            price = round(4.0 + 1.3 * k + rng.uniform(0, 0.8), 1)
            club = clubs[(k + "GFC".index(pos)) % len(clubs)]
            base = 1.1 * price + rng.normal(0, 1.5)
            out.append(
                LineupPlayer(
                    player_id=f"{pos}{k}",
                    name=f"{pos} player {k}",
                    pos=pos,
                    club=club,
                    price=price,
                    ev={1: base, 2: base * rng.uniform(0.8, 1.2), 3: base},
                    late=FIXTURES[club][1],
                )
            )
    return out


def make_coaches() -> list[LineupCoach]:
    return [
        LineupCoach(f"HC{c}", f"Coach {c}", c, 5.0 + i, {1: 2.0 * i - 3, 2: 1.0, 3: 1.0})
        for i, c in enumerate(FIXTURES)
    ]


# --------------------------------------------------------------------- MILP
class TestLineupMilp:
    def test_structure_follows_the_rules(self, rules):
        sol = optimise_lineup(make_players(), make_coaches(), rules, rounds=[1, 2, 3])
        roster = sol.roster
        assert len(roster) == 10
        assert {p: sum(1 for x in roster if x.pos == p) for p in "GFC"} == {"G": 4, "F": 4, "C": 2}
        assert len(sol.starters) == 5 and sol.sixth is not None and len(sol.bench) == 4
        counts = tuple(sum(1 for x in sol.starters if x.pos == p) for p in "GFC")
        assert rules.is_legal_formation(counts) and counts == sol.formation
        assert sol.captain in sol.starters
        assert sol.coach is not None
        assert sol.cost <= rules.budget + 1e-6

    def test_captain_is_the_best_starter_for_a_single_round(self, rules):
        sol = optimise_lineup(make_players(), make_coaches(), rules, rounds=[1])
        assert sol.captain.ev[1] == max(p.ev[1] for p in sol.starters)

    def test_bench_holds_the_weakest_for_a_single_round(self, rules):
        sol = optimise_lineup(make_players(), make_coaches(), rules, rounds=[1])
        field = sol.starters + [sol.sixth]
        # Compare within position: swapping a bench player for a field player
        # of the same position never changes the formation, so a bench player
        # better than every field player at his position would be a missed
        # improvement.
        for b in sol.bench:
            same_pos = [p for p in field if p.pos == b.pos]
            if same_pos:
                assert b.ev[1] <= max(p.ev[1] for p in same_pos) + 1e-9

    def test_objective_counts_the_lineup_weights(self, rules):
        sol = optimise_lineup(make_players(), make_coaches(), rules, rounds=[1])
        expected = (
            sum(p.ev[1] for p in sol.starters)
            + sol.sixth.ev[1]
            + sol.captain.ev[1] * (rules.captain_multiplier - 1)
            + rules.bench_weight * sum(p.ev[1] for p in sol.bench)
            + sol.coach.ev[1]
        )
        assert sol.first_round == pytest.approx(expected)

    def test_first_round_excludes_later_rounds(self, rules):
        """Regression: later-round terms leaked into the first-round score."""

        sol = optimise_lineup(make_players(), make_coaches(), rules, rounds=[1, 2, 3])
        expected = (
            sum(p.ev[1] for p in sol.starters)
            + sol.sixth.ev[1]
            + sol.captain.ev[1] * (rules.captain_multiplier - 1)
            + rules.bench_weight * sum(p.ev[1] for p in sol.bench)
            + sol.coach.ev[1]
        )
        assert sol.first_round == pytest.approx(expected)
        assert sol.objective > sol.first_round

    def test_coach_price_comes_out_of_the_same_budget(self, rules):
        players, coaches = make_players(), make_coaches()
        cheap = optimise_lineup(players, coaches, rules, rounds=[1], budget=70.0)
        assert cheap.cost <= 70.0 + 1e-6

    def test_min_late_bench(self, rules):
        sol = optimise_lineup(make_players(), make_coaches(), rules, rounds=[1], min_late_bench=2)
        assert sum(1 for p in sol.bench if p.late) >= 2

    def test_future_rounds_break_first_round_ties(self, rules):
        players = make_players()
        a = next(p for p in players if p.player_id == "G5")
        twin = dataclasses.replace(
            a, player_id="G5b", name="twin", ev={1: a.ev[1], 2: a.ev[2] + 20, 3: a.ev[3]}
        )
        sol = optimise_lineup([*players, twin], make_coaches(), rules, rounds=[1, 2])
        ids = {p.player_id for p in sol.roster}
        assert "G5b" in ids or "G5" not in ids

    def test_future_slot_weight(self, rules):
        assert future_slot_weight(rules) == pytest.approx(0.9)

    def test_impossible_budget_raises(self, rules):
        with pytest.raises(InfeasibleError):
            optimise_lineup(make_players(), make_coaches(), rules, rounds=[1], budget=20.0)


class TestTransfers:
    def _current(self, rules):
        sol = optimise_lineup(make_players(), make_coaches(), rules, rounds=[1], budget=75.0)
        return CurrentSquad(
            player_ids=[p.player_id for p in sol.roster], coach_id=sol.coach.coach_id, bank=25.0
        )

    def test_transfer_cap(self, rules):
        cur = self._current(rules)
        sol = optimise_lineup(
            make_players(), make_coaches(), rules, rounds=[1], current=cur, max_transfers=2
        )
        assert sol.n_transfers <= 2
        assert len(sol.sold) == len(sol.bought)

    def test_zero_transfers_keeps_the_roster(self, rules):
        cur = self._current(rules)
        sol = optimise_lineup(
            make_players(), make_coaches(), rules, rounds=[1], current=cur, max_transfers=0
        )
        assert {p.player_id for p in sol.roster} == set(cur.player_ids)
        assert sol.coach.coach_id == cur.coach_id

    def test_a_coach_change_uses_a_transfer(self, rules):
        cur = self._current(rules)
        coaches = make_coaches()
        best_coach = max(coaches, key=lambda c: c.ev[1])
        assert cur.coach_id != best_coach.coach_id
        sol = optimise_lineup(
            make_players(),
            coaches,
            rules,
            rounds=[1],
            current=cur,
            max_transfers=1,
            force_coach=best_coach.coach_id,
        )
        assert sol.coach_changed and sol.bought == []

    def test_sales_fund_purchases(self, rules):
        cur = self._current(rules)
        players = make_players()
        sol = optimise_lineup(
            players, make_coaches(), rules, rounds=[1], current=cur, max_transfers=4
        )
        prices = {p.player_id: p.price for p in players}
        spent = sum(p.price for p in sol.bought)
        raised = sum(prices[pid] for pid in sol.sold)
        assert spent <= cur.bank + raised + 1e-6


# ---------------------------------------------------------------- simulator
def sim_players() -> list[SimPlayer]:
    out = []
    for p in make_players():
        opp, late = FIXTURES[p.club]
        out.append(
            SimPlayer(
                player_id=p.player_id,
                name=p.name,
                pos=p.pos,
                club=p.club,
                opponent=opp,
                price=p.price,
                late=late,
                points_if_available=max(p.ev[1], 1.0) / 0.97,
                play_prob=0.97,
                minutes=8.0 + 1.6 * p.price,
                win_prob=0.6 if p.club in "ACE" else 0.4,
                future_value=0.5 * p.ev[2],
            )
        )
    return out


def sim_coaches() -> list[SimCoach]:
    return [SimCoach(c.coach_id, c.name, c.club, c.price) for c in make_coaches()]


@pytest.fixture(scope="module")
def sim(rules):
    return TurnSimulator(sim_players(), sim_coaches(), rules, n=3000, seed=1)


def milp_roster(sim, rules):
    lp = [
        LineupPlayer(p.player_id, p.name, p.pos, p.club, p.price, {1: sim.E[i]}, p.late)
        for i, p in enumerate(sim.players)
    ]
    sol = optimise_lineup(lp, make_coaches(), rules, rounds=[1])
    return [sim.index[p.player_id] for p in sol.roster], sim.coach_index[sol.coach.coach_id]


class TestSimulator:
    def test_same_roster_same_score(self, sim, rules):
        ids, coach = milp_roster(sim, rules)
        assert sim.evaluate(ids, coach).objective == sim.evaluate(ids, coach).objective

    def test_turn_moves_never_hurt(self, sim, rules):
        """Swaps and captain changes are options: they can only add value."""

        ids, coach = milp_roster(sim, rules)
        frozen_rules = dataclasses.replace(rules, field_bench_swaps=False, captain_switch=False)
        frozen = TurnSimulator(sim_players(), sim_coaches(), frozen_rules, n=3000, seed=1)
        assert (
            sim.evaluate(ids, coach).first_round.mean()
            >= frozen.evaluate(ids, coach).first_round.mean() - 1e-9
        )

    def test_captain_switch_adds_value_when_a_late_starter_exists(self, sim, rules):
        ids, coach = milp_roster(sim, rules)
        if not any(sim.late[i] for i in ids):
            pytest.skip("roster has no later-Turn player")
        no_switch = TurnSimulator(
            sim_players(),
            sim_coaches(),
            dataclasses.replace(rules, captain_switch=False),
            n=3000,
            seed=1,
        )
        assert (
            sim.evaluate(ids, coach).first_round.mean()
            >= no_switch.evaluate(ids, coach).first_round.mean()
        )

    def test_later_turn_players_start_on_the_bench(self, sim, rules):
        ids, coach = milp_roster(sim, rules)
        ev = sim.evaluate(ids, coach, detail=True)
        first_turn = [i for i in ids if not sim.late[i]]
        if len(first_turn) >= rules.field_slots:
            field = set(ev.lineup.starters) | {ev.lineup.sixth}
            assert not any(sim.late[i] for i in field)

    def test_only_later_turn_players_are_promoted(self, sim, rules):
        ids, coach = milp_roster(sim, rules)
        ev = sim.evaluate(ids, coach, detail=True)
        assert all(sim.late[i] for i in ev.promoted)
        assert all(0.0 <= p <= 1.0 for p in ev.promoted.values())

    def test_price_floor_protects_four_credit_players(self, sim):
        at_floor = [i for i, p in enumerate(sim.players) if p.price == 4.0]
        for i in at_floor:
            assert sim.dprice[i] >= 0.0

    def test_fringe_players_miss_more_games(self, sim):
        low = min(range(len(sim.players)), key=lambda i: sim.players[i].minutes)
        high = max(range(len(sim.players)), key=lambda i: sim.players[i].minutes)
        assert sim.p_zero[low] > sim.p_zero[high]

    def test_expected_points_are_preserved_by_the_dnp_model(self, sim):
        for i, p in enumerate(sim.players):
            assert sim.E[i] == pytest.approx(p.points_if_available * p.play_prob, rel=0.08)

    def test_coach_without_a_game_scores_zero(self, rules):
        coaches = [*sim_coaches(), SimCoach("HCZ", "Idle", "Z", 5.0)]
        s = TurnSimulator(sim_players(), coaches, rules, n=500, seed=1)
        assert float(np.abs(s.coach_points[-1]).sum()) == 0.0


class TestSearch:
    def test_search_keeps_every_constraint(self, sim, rules):
        ids, coach = milp_roster(sim, rules)
        start = sim.evaluate(ids, coach).objective
        new_ids, new_coach, value = local_search(
            sim, ids, coach, candidate_pool(sim, per_pos=6), max_passes=3
        )
        assert value >= start - 1e-9
        assert sim.cost(new_ids, new_coach) <= rules.budget + 1e-9
        assert sorted(sim.pos[i] for i in new_ids) == sorted("GGGGFFFFCC")
        clubs = [sim.players[i].club for i in new_ids]
        assert max(clubs.count(c) for c in set(clubs)) <= rules.max_per_club

    def test_search_respects_a_transfer_cap(self, sim, rules):
        ids, coach = milp_roster(sim, rules)
        new_ids, new_coach, _ = local_search(
            sim,
            ids,
            coach,
            candidate_pool(sim, per_pos=6),
            current=(set(ids), coach),
            max_transfers=1,
            max_passes=4,
        )
        moves = len(set(new_ids) - set(ids)) + (1 if new_coach != coach else 0)
        assert moves <= 1
