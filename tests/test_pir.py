from __future__ import annotations

import pytest

from elfantasy.models import BoxScore
from elfantasy.projection.pir import (
    StatLine,
    compose_pir_from_props,
    pir_from_boxscore,
    statline_variance,
)


class TestPirFormula:
    def test_matches_the_official_definition(self):
        # fmt: off
        bs = BoxScore(
            game_id="G1", round=1, player_id="p", team_code="T",
            minutes=30, points=20, rebounds=8, assists=4, steals=2, blocks=1,
            fouls_drawn=5, fg_made=7, fg_attempted=14, ft_made=4, ft_attempted=6,
            turnovers=3, blocks_against=1, fouls_committed=2,
        )
        # fmt: on
        # (20+8+4+2+1+5) - (7 missed FG + 2 missed FT + 3 TO + 1 BLKA + 2 PF)
        assert pir_from_boxscore(bs) == pytest.approx(40 - 15)

    def test_a_bad_game_can_be_negative(self):
        # fmt: off
        bs = BoxScore(
            game_id="G1", round=1, player_id="p", team_code="T",
            minutes=12, points=0, fg_attempted=6, fg_made=0,
            turnovers=3, fouls_committed=4,
        )
        # fmt: on
        assert pir_from_boxscore(bs) < 0

    def test_an_empty_line_is_zero(self):
        bs = BoxScore(game_id="G1", round=1, player_id="p", team_code="T")
        assert pir_from_boxscore(bs) == 0.0


class TestStatLine:
    def test_scaling_is_linear_in_pir(self):
        line = StatLine(points=10, rebounds=4, assists=2, missed_fg=5, turnovers=2)
        assert line.scaled(2.0).pir == pytest.approx(line.pir * 2.0)

    def test_blending_interpolates(self):
        a = StatLine(points=10)
        b = StatLine(points=20)
        assert a.blended(b, 0.5).points == pytest.approx(15.0)
        assert a.blended(b, 0.0).points == pytest.approx(10.0)
        assert a.blended(b, 1.0).points == pytest.approx(20.0)

    def test_variance_grows_with_production(self):
        small = StatLine(points=4, rebounds=2)
        big = StatLine(points=20, rebounds=9)
        assert statline_variance(big) > statline_variance(small)

    def test_variance_is_non_negative(self):
        assert statline_variance(StatLine()) >= 0.0


class TestComposeFromProps:
    def test_priced_components_are_replaced(self):
        modelled = StatLine(points=12, rebounds=5, assists=3, missed_fg=6)
        composed = compose_pir_from_props({"points": 16.0}, modelled)
        assert composed.points == pytest.approx(16.0)

    def test_unpriced_components_scale_with_the_market_move(self):
        """A bigger night than modelled should lift the whole line, not one term."""

        modelled = StatLine(points=12, rebounds=5, assists=3, missed_fg=6)
        composed = compose_pir_from_props({"points": 18.0}, modelled)
        assert composed.rebounds > modelled.rebounds
        assert composed.missed_fg > modelled.missed_fg

    def test_a_matching_market_leaves_the_line_alone(self):
        modelled = StatLine(points=12, rebounds=5, assists=3, missed_fg=6)
        composed = compose_pir_from_props({"points": 12.0}, modelled)
        assert composed.pir == pytest.approx(modelled.pir, abs=1e-6)

    def test_scaling_is_clamped(self):
        """One wild prop must not be allowed to triple the projection."""

        modelled = StatLine(points=10, rebounds=5, assists=3, missed_fg=5)
        composed = compose_pir_from_props({"points": 60.0}, modelled)
        assert composed.rebounds <= modelled.rebounds * 1.6 + 1e-9

    def test_partial_trust_blends_rather_than_replaces(self):
        modelled = StatLine(points=10, rebounds=5)
        composed = compose_pir_from_props({"points": 20.0}, modelled, trust={"points": 0.5})
        assert composed.points == pytest.approx(15.0)

    def test_unknown_component_is_ignored(self):
        modelled = StatLine(points=10, rebounds=5)
        composed = compose_pir_from_props({"dunks": 3.0}, modelled)
        assert composed.pir == pytest.approx(modelled.pir)
