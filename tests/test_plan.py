"""End to end: projections -> lineup plan, on the synthetic league."""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from elfantasy.cli import app
from elfantasy.models import Squad
from elfantasy.plan import build_inputs, plan_lineup, turn_of_games
from elfantasy.rules import GameRules


@pytest.fixture(scope="module")
def rules(settings):
    return GameRules.from_settings(settings)


@pytest.fixture(scope="module")
def scratch_plan(pipeline, rules):
    return plan_lineup(pipeline, rules, n=1500, search=False)


class TestInputs:
    def test_rounds_split_into_turns(self, pipeline):
        late = turn_of_games(pipeline.dataset, pipeline.start_round)
        assert any(late.values()) and not all(late.values())

    def test_fantasy_points_include_the_win_bonus(self, pipeline, rules):
        _, lp, _, _, _ = build_inputs(pipeline, rules)
        first = pipeline.start_round
        for p in lp[:20]:
            proj = pipeline.horizon[first].get(p.player_id)
            if proj is None:
                continue
            assert p.ev[first] == pytest.approx(
                proj.mean_pir * (1 + rules.win_bonus * proj.win_prob)
            )

    def test_every_club_has_a_coach_projection(self, pipeline, rules):
        _, _, lc, _, sc = build_inputs(pipeline, rules)
        assert len(lc) == len(pipeline.dataset.coaches) == len(sc)
        assert all(pipeline.start_round in c.ev for c in lc)


class TestScratchPlan:
    def test_roster_follows_the_rules(self, scratch_plan, rules):
        sim, ids = scratch_plan.sim, scratch_plan.ids
        assert sorted(sim.pos[i] for i in ids) == sorted("GGGGFFFFCC")
        assert scratch_plan.evaluation.cost <= rules.budget + 1e-6
        assert scratch_plan.coach is not None

    def test_later_day_players_wait_on_the_bench(self, scratch_plan, rules):
        sim, lu = scratch_plan.sim, scratch_plan.evaluation.lineup
        first_day = [i for i in scratch_plan.ids if not sim.late[i]]
        if len(first_day) >= rules.field_slots:
            assert not any(sim.late[i] for i in (*lu.starters, lu.sixth))

    def test_turn_moves_add_value_over_a_fixed_lineup(self, scratch_plan):
        assert scratch_plan.evaluation.first_round.mean() >= scratch_plan.static_first_round - 1.0

    def test_first_round_is_a_distribution(self, scratch_plan):
        first = scratch_plan.evaluation.first_round
        assert np.percentile(first, 90) > np.percentile(first, 10)


class TestTransferPlan:
    def test_respects_the_transfer_cap_and_budget(self, pipeline, rules, scratch_plan):
        sim = scratch_plan.sim
        # Start from a deliberately cheap legal squad so there is room to improve.
        cheap = sorted(range(len(sim.players)), key=lambda i: sim.price[i])
        ids, counts = [], {"G": 0, "F": 0, "C": 0}
        for i in cheap:
            pos = sim.pos[i]
            if counts[pos] < rules.roster[pos]:
                ids.append(i)
                counts[pos] += 1
        squad = Squad(
            player_ids=[sim.players[i].player_id for i in ids],
            bank=5.0,
            coach_id=next(iter(pipeline.dataset.coaches)),
        )
        plan = plan_lineup(pipeline, rules, squad=squad, n=800, search=True)
        moves = len(plan.bought) + (1 if plan.coach_changed else 0)
        assert moves <= rules.transfers_per_round
        assert len(plan.sold) == len(plan.bought)
        assert plan.evaluation.cost <= plan.sim.budget + 1e-6


def test_cli_lineup_runs_on_the_sample():
    result = CliRunner().invoke(app, ["lineup", "--sample", "--no-search", "--sims", "500"])
    assert result.exit_code == 0, result.output
    assert "Turn plan" in result.output
