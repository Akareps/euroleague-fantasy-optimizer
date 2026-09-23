"""Game rules and the scoring / pricing functions built on them."""

from __future__ import annotations

import numpy as np
import pytest

from elfantasy import scoring
from elfantasy.rules import GameRules


@pytest.fixture(scope="module")
def rules(settings):
    return GameRules.from_settings(settings)


class TestRulesFile:
    def test_matches_the_2026_27_game(self, rules):
        assert rules.budget == 100.0
        assert rules.roster == {"G": 4, "F": 4, "C": 2}
        assert rules.squad_size == 10
        assert rules.head_coach
        assert rules.max_per_club == 6
        assert rules.field_slots == 6  # five starters and a sixth man
        assert rules.bench_weight == 0.5
        assert rules.captain_multiplier == 2.0
        assert rules.win_bonus == pytest.approx(0.10)
        assert rules.transfers_per_round == 4

    @pytest.mark.parametrize("formation", [(2, 2, 1), (1, 2, 2), (2, 1, 2), (1, 3, 1), (3, 1, 1)])
    def test_legal_formations(self, rules, formation):
        assert rules.is_legal_formation(formation)

    @pytest.mark.parametrize("formation", [(3, 2, 0), (1, 1, 3), (0, 3, 2), (2, 3, 0), (5, 0, 0)])
    def test_illegal_formations(self, rules, formation):
        assert not rules.is_legal_formation(formation)

    def test_unlimited_transfer_windows(self, rules):
        assert rules.unlimited_transfers_before(7)
        assert not rules.unlimited_transfers_before(2)


class TestCoachPoints:
    @pytest.mark.parametrize(
        "margin,points",
        [
            (25, 25),
            (21, 25),
            (20, 20),
            (11, 20),
            (10, 10),
            (1, 10),
            (-1, -5),
            (-10, -5),
            (-11, -10),
            (-20, -10),
            (-21, -20),
        ],
    )
    def test_official_table(self, rules, margin, points):
        assert scoring.coach_points(margin, rules) == points

    def test_a_near_zero_simulated_margin_keeps_its_sign(self, rules):
        assert scoring.coach_points(0.2, rules) == 10
        assert scoring.coach_points(-0.2, rules) == -5

    def test_analytic_expectation_matches_simulation(self, rules):
        rng = np.random.default_rng(0)
        for mean in (-12.0, 0.0, 8.0):
            sims = rng.normal(mean, 11.5, 300_000)
            assert scoring.expected_coach_points(mean, rules) == pytest.approx(
                scoring.coach_points(sims, rules).mean(), abs=0.1
            )

    def test_heavier_favourite_scores_more(self, rules):
        assert scoring.expected_coach_points(12, rules) > scoring.expected_coach_points(3, rules)


class TestPriceChange:
    def test_the_observed_rule(self, rules):
        """A 12.0 player scoring 21 moves to 12.4."""

        assert scoring.price_change(21, 12.0, rules) == pytest.approx(0.4)

    def test_whole_steps_only(self, rules):
        assert scoring.price_change(13.9, 12.0, rules) == pytest.approx(0.0)
        assert scoring.price_change(14.0, 12.0, rules) == pytest.approx(0.1)

    def test_losses_mirror_gains(self, rules):
        assert scoring.price_change(3, 12.0, rules) == pytest.approx(-0.4)

    def test_floor(self, rules):
        """A 4.0 player cannot lose value; a 4.5 player who sits out can."""

        assert scoring.price_change(0, 4.0, rules) == pytest.approx(0.0)
        assert scoring.price_change(0, 4.5, rules) == pytest.approx(-0.2)
        assert scoring.price_change(-20, 4.3, rules) == pytest.approx(-0.3)

    def test_the_floor_never_lifts_a_price(self, rules):
        """Regression: a player listed below the floor jumped up to it."""

        assert scoring.price_change(2.0, 2.0, rules) == pytest.approx(0.0)
        assert scoring.price_change(0.0, 2.0, rules) == pytest.approx(0.0)

    def test_vectorised(self, rules):
        out = scoring.price_change(np.array([0.0, 21.0]), np.array([4.0, 12.0]), rules)
        assert out == pytest.approx([0.0, 0.4])


class TestFantasyScore:
    def test_win_bonus(self, rules):
        assert scoring.fantasy_score(20.0, True, rules) == pytest.approx(22.0)
        assert scoring.fantasy_score(20.0, False, rules) == pytest.approx(20.0)

    def test_margin_from_win_probability_is_symmetric(self):
        assert scoring.win_prob_to_margin(0.5) == pytest.approx(0.0)
        assert scoring.win_prob_to_margin(0.8) == pytest.approx(-scoring.win_prob_to_margin(0.2))
