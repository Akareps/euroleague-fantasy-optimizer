"""Constraint satisfaction and optimality, on a pool small enough to reason about."""

from __future__ import annotations

import pytest

from elfantasy.models import Squad
from elfantasy.optimize.squad import Candidate, InfeasibleError, optimise_squad
from elfantasy.optimize.transfers import optimise_transfers, transfer_ladder


def _pool(n_per_pos: int = 8) -> list[Candidate]:
    """A pool where value rises with price, so the answer is not trivial."""

    out = []
    for pos in ("G", "F", "C"):
        for i in range(n_per_pos):
            price = 2.0 + i * 1.5
            out.append(
                Candidate(
                    player_id=f"{pos}{i}",
                    name=f"{pos} player {i}",
                    team_code=f"T{i % 6}",
                    position=pos,
                    price=price,
                    value=4.0 + 2.2 * i - 0.06 * i * i,
                    variance=3.0 + i,
                    next_round_pir=2.0 + 1.1 * i,
                )
            )
    return out


class TestSquad:
    def test_respects_every_constraint(self, settings):
        sol = optimise_squad(_pool(), settings)
        assert len(sol.selected) == settings.squad_size
        assert sol.total_price <= settings.budget + 1e-6

        counts = {}
        clubs = {}
        for c in sol.selected:
            counts[c.position] = counts.get(c.position, 0) + 1
            clubs[c.team_code] = clubs.get(c.team_code, 0) + 1
        for pos, (lo, hi) in settings.position_bounds().items():
            assert lo <= counts.get(pos, 0) <= hi
        assert max(clubs.values()) <= settings.max_per_club

    def test_a_smaller_budget_never_buys_more_value(self, settings):
        pool = _pool()
        rich = optimise_squad(pool, settings, budget=100.0)
        poor = optimise_squad(pool, settings, budget=60.0)
        assert poor.total_value <= rich.total_value + 1e-6

    def test_beats_greedy_value_per_credit(self, settings):
        """The reason this is a MILP and not a sort.

        Greedy picks the best ratio until the budget runs out, which strands
        credits it cannot spend well. The MILP is allowed to be worse per credit
        on one pick in order to be better overall.
        """

        pool = _pool()
        optimal = optimise_squad(pool, settings, budget=70.0)

        bounds = settings.position_bounds()
        counts: dict[str, int] = {}
        clubs: dict[str, int] = {}
        spent, greedy_value, picked = 0.0, 0.0, 0
        for c in sorted(pool, key=lambda c: -(c.value / c.price)):
            lo, hi = bounds[c.position]
            if (
                counts.get(c.position, 0) >= hi
                or clubs.get(c.team_code, 0) >= settings.max_per_club
            ):
                continue
            if spent + c.price > 70.0 or picked >= settings.squad_size:
                continue
            spent += c.price
            greedy_value += c.value
            counts[c.position] = counts.get(c.position, 0) + 1
            clubs[c.team_code] = clubs.get(c.team_code, 0) + 1
            picked += 1

        assert optimal.total_value >= greedy_value

    def test_risk_aversion_shifts_toward_lower_variance(self, settings):
        pool = _pool()
        neutral = optimise_squad(pool, settings, risk_aversion=0.0)
        averse = optimise_squad(pool, settings, risk_aversion=0.5)
        assert sum(c.variance for c in averse.selected) <= sum(c.variance for c in neutral.selected)

    def test_locked_in_player_is_always_selected(self, settings):
        pool = _pool()
        pool[0].locked_in = True  # the cheapest, least valuable guard
        sol = optimise_squad(pool, settings)
        assert pool[0].player_id in sol.player_ids

    def test_locked_out_player_is_never_selected(self, settings):
        pool = _pool()
        target = max(pool, key=lambda c: c.value)
        target.locked_out = True
        sol = optimise_squad(pool, settings)
        assert target.player_id not in sol.player_ids

    def test_impossible_budget_raises_a_clear_error(self, settings):
        with pytest.raises(InfeasibleError, match="budget too low|Infeasible"):
            optimise_squad(_pool(), settings, budget=5.0)

    def test_empty_pool_raises(self, settings):
        with pytest.raises(InfeasibleError, match="no candidates"):
            optimise_squad([], settings)


class TestTransfers:
    def _starting_squad(self, pool, settings) -> Squad:
        """A legal but deliberately cheap squad, so there is room to improve."""

        bounds = settings.position_bounds()
        counts: dict[str, int] = {}
        clubs: dict[str, int] = {}
        ids: list[str] = []
        for phase in ("min", "max"):
            for c in sorted(pool, key=lambda c: c.price):
                if len(ids) >= settings.squad_size or c.player_id in ids:
                    continue
                lo, hi = bounds[c.position]
                limit = lo if phase == "min" else hi
                if counts.get(c.position, 0) >= limit:
                    continue
                if clubs.get(c.team_code, 0) >= settings.max_per_club:
                    continue
                ids.append(c.player_id)
                counts[c.position] = counts.get(c.position, 0) + 1
                clubs[c.team_code] = clubs.get(c.team_code, 0) + 1
        return Squad(player_ids=ids, bank=25.0)

    def test_never_exceeds_the_transfer_cap(self, settings):
        pool = _pool()
        squad = self._starting_squad(pool, settings)
        plan = optimise_transfers(pool, squad, settings, max_transfers=4)
        assert plan.n_transfers <= 4

    def test_squad_size_is_preserved(self, settings):
        pool = _pool()
        squad = self._starting_squad(pool, settings)
        plan = optimise_transfers(pool, squad, settings, max_transfers=3)
        assert len(plan.squad_ids) == squad.size
        assert len(set(plan.squad_ids)) == squad.size

    def test_sales_fund_purchases(self, settings):
        """You may spend more than your bank, because selling raises cash."""

        pool = _pool()
        squad = self._starting_squad(pool, settings)
        plan = optimise_transfers(pool, squad, settings, max_transfers=4)
        spent = sum(m.in_player.price for m in plan.moves)
        raised = sum(m.out_player.price for m in plan.moves)
        assert spent <= squad.bank + raised + 1e-6
        assert plan.bank_after >= -1e-6

    def test_transfers_never_make_the_squad_worse(self, settings):
        pool = _pool()
        squad = self._starting_squad(pool, settings)
        plan = optimise_transfers(pool, squad, settings, max_transfers=4)
        assert plan.value_after >= plan.value_before - 1e-6

    def test_protected_players_are_not_sold(self, settings):
        pool = _pool()
        squad = self._starting_squad(pool, settings)
        protect = set(squad.player_ids[:3])
        plan = optimise_transfers(pool, squad, settings, max_transfers=4, protect=protect)
        sold = {m.out_player.player_id for m in plan.moves}
        assert not (sold & protect)

    def test_zero_transfers_is_a_no_op(self, settings):
        pool = _pool()
        squad = self._starting_squad(pool, settings)
        plan = optimise_transfers(pool, squad, settings, max_transfers=0)
        assert plan.n_transfers == 0
        assert set(plan.squad_ids) == set(squad.player_ids)

    def test_friction_discourages_marginal_swaps(self, settings):
        pool = _pool()
        squad = self._starting_squad(pool, settings)
        free = optimise_transfers(pool, squad, settings, max_transfers=4, friction=0.0)
        costly = optimise_transfers(pool, squad, settings, max_transfers=4, friction=25.0)
        assert costly.n_transfers <= free.n_transfers

    def test_ladder_is_monotonically_non_decreasing(self, settings):
        """More transfers can never buy less: n+1 can always mimic n."""

        pool = _pool()
        squad = self._starting_squad(pool, settings)
        plans = transfer_ladder(pool, squad, settings, max_transfers=3, friction=0.0)
        values = [p.value_after for p in plans]
        assert values == sorted(values), values

    def test_squad_member_missing_from_the_pool_is_reported(self, settings):
        pool = _pool()
        squad = self._starting_squad(pool, settings)
        squad.player_ids[0] = "ghost"
        with pytest.raises(InfeasibleError, match="not in the candidate pool"):
            optimise_transfers(pool, squad, settings)
