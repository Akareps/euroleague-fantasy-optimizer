"""The bookmaking maths is the part with objectively checkable answers."""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm, poisson

from elfantasy.projection import market as mk


class TestPriceConversion:
    def test_american_to_decimal(self):
        assert mk.american_to_decimal(100) == pytest.approx(2.0)
        assert mk.american_to_decimal(-200) == pytest.approx(1.5)
        assert mk.american_to_decimal(150) == pytest.approx(2.5)

    def test_to_decimal_accepts_either_format(self):
        assert mk.to_decimal(1.91) == pytest.approx(1.91)
        assert mk.to_decimal(-110) == pytest.approx(1.9091, abs=1e-4)

    def test_rejects_impossible_prices(self):
        with pytest.raises(ValueError):
            mk.decimal_to_prob(0.9)


class TestDevig:
    PRICES = [1.45, 3.10]

    @pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
    def test_all_methods_produce_a_probability_distribution(self, method):
        probs = mk.devig(self.PRICES, method)
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)
        assert all(0 < p < 1 for p in probs)

    @pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
    def test_fair_book_is_left_alone(self, method):
        probs = mk.devig([2.0, 2.0], method)
        assert probs[0] == pytest.approx(0.5, abs=1e-6)

    def test_overround_is_positive_for_a_real_book(self):
        assert mk.overround(self.PRICES) > 0

    def test_shin_taxes_the_longshot_less_than_multiplicative(self):
        """The whole reason to prefer Shin.

        The favourite-longshot bias means proportional de-vigging overstates
        the longshot's fair probability. Shin removes more of the vig from the
        longshot side, so its fair probability comes out lower.
        """

        mult = mk.devig(self.PRICES, "multiplicative")
        shin = mk.devig(self.PRICES, "shin")
        assert shin[1] < mult[1]
        assert shin[0] > mult[0]

    def test_three_way_market(self):
        probs = mk.devig([2.4, 3.3, 3.1], "shin")
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown de-vig"):
            mk.devig(self.PRICES, "nonsense")


class TestGameDerivations:
    def test_team_totals_split_by_the_spread(self):
        home, away = mk.implied_team_totals(total=160.0, spread=-6.0)
        assert home + away == pytest.approx(160.0)
        assert home - away == pytest.approx(6.0)

    def test_pick_em_splits_evenly(self):
        home, away = mk.implied_team_totals(total=155.0, spread=0.0)
        assert home == away == pytest.approx(77.5)

    def test_spread_and_win_prob_are_inverses(self):
        for spread in (-12.5, -3.0, 0.0, 7.5):
            p = mk.spread_to_win_prob(spread, sigma=11.0)
            assert mk.win_prob_to_spread(p, sigma=11.0) == pytest.approx(spread, abs=1e-6)

    def test_favourite_has_win_prob_above_half(self):
        assert mk.spread_to_win_prob(-8.0) > 0.5
        assert mk.spread_to_win_prob(8.0) < 0.5

    def test_pace_factor_scales_with_total(self):
        assert mk.pace_factor(170.0, 160.0, 1.0) > 1.0
        assert mk.pace_factor(150.0, 160.0, 1.0) < 1.0
        assert mk.pace_factor(170.0, 160.0, 0.0) == pytest.approx(1.0)


class TestPropInversion:
    def test_normal_inversion_round_trips(self):
        mu, sd, line = 14.0, 5.0, 12.5
        p_over = 1 - norm.cdf(line, mu, sd)
        assert mk.prop_to_mean_normal(line, p_over, sd) == pytest.approx(mu, abs=1e-6)

    def test_poisson_inversion_round_trips(self):
        lam, line = 6.4, 5.5
        p_over = poisson.sf(math.ceil(line) - 1, lam)
        assert mk.prop_to_mean_poisson(line, p_over) == pytest.approx(lam, abs=1e-4)

    def test_even_money_prop_implies_the_line_is_the_median(self):
        """A 50/50 prop means the line sits at the median, above the Poisson mean."""

        mean = mk.prop_to_mean(6.5, 2.0, 2.0, dist="poisson")
        assert 6.0 < mean < 7.0

    def test_juiced_over_implies_a_higher_mean(self):
        cheap_over = mk.prop_to_mean(5.5, 1.60, 2.35, dist="poisson")
        cheap_under = mk.prop_to_mean(5.5, 2.35, 1.60, dist="poisson")
        assert cheap_over > cheap_under

    def test_one_sided_price_still_works(self):
        assert mk.prop_to_mean(5.5, 1.90, None, dist="poisson") > 0

    def test_needs_at_least_one_price(self):
        with pytest.raises(ValueError):
            mk.prop_to_mean(5.5, None, None)

    def test_normal_inversion_requires_sd(self):
        with pytest.raises(ValueError, match="positive sd"):
            mk.prop_to_mean(12.5, 1.9, 1.9, dist="normal")

    def test_devig_matters_for_the_implied_mean(self):
        """Ignoring the vig biases the implied mean upward.

        This is the mistake the module exists to prevent: taking 1/price as a
        probability makes both sides look likelier than they are, and for the
        over that translates into an inflated projection.
        """

        line = 5.5
        with_devig = mk.prop_to_mean(line, 1.87, 1.87, dist="poisson")
        naive_p = 1 / 1.87
        naive = mk.prop_to_mean_poisson(line, naive_p)
        assert naive > with_devig


class TestPirSd:
    def test_scales_with_sqrt_minutes(self):
        assert mk.pir_sd(36.0) > mk.pir_sd(18.0)
        assert mk.pir_sd(36.0) / mk.pir_sd(18.0) == pytest.approx(math.sqrt(2.0), abs=0.01)

    def test_has_a_floor(self):
        assert mk.pir_sd(0.0) >= 2.5
