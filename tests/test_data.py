"""Adapters, name matching and the numeric helpers."""

from __future__ import annotations

import pytest

from elfantasy.data.euroleague import _first, _num, _rows, parse_minutes
from elfantasy.data.injuries import (
    InjuryRecord,
    ManualInjuryProvider,
    apply_to_players,
    merge,
    normalise_name,
)
from elfantasy.models import Availability, Player, Position
from elfantasy.util import ewma_weights, precision_blend, shrink


class TestFeedParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("28:35", 28 + 35 / 60),
            ("00:00", 0.0),
            (24.5, 24.5),
            ("", 0.0),
            ("DNP", 0.0),
            (None, 0.0),
        ],
    )
    def test_minutes_formats(self, raw, expected):
        assert parse_minutes(raw) == pytest.approx(expected, abs=1e-6)

    def test_num_tolerates_junk(self):
        assert _num("-") == 0.0
        assert _num("12,5") == pytest.approx(12.5)
        assert _num(None, default=3.0) == 3.0

    def test_first_finds_any_spelling(self):
        assert _first({"gameCode": 5}, "identifier", "gameCode") == 5
        assert _first({"round": 2}, "gameday", "round") == 2
        assert _first({}, "missing", default="fallback") == "fallback"

    def test_rows_unwraps_common_envelopes(self):
        assert _rows([{"a": 1}]) == [{"a": 1}]
        assert _rows({"data": [{"a": 1}]}) == [{"a": 1}]
        assert _rows({"data": {"items": [{"a": 1}]}}) == [{"a": 1}]
        assert _rows(None) == []


class TestNameMatching:
    def test_handles_surname_first(self):
        assert normalise_name("Doe, John") == normalise_name("John Doe")

    def test_strips_accents_and_punctuation(self):
        assert normalise_name("Šarić, Dário") == normalise_name("Dario Saric")
        assert normalise_name("O'Brien, T.J.") == normalise_name("T.J. O'Brien")

    def test_is_order_insensitive(self):
        assert normalise_name("Ante Zizic") == normalise_name("Zizic Ante")

    def test_different_players_do_not_collide(self):
        assert normalise_name("John Doe") != normalise_name("Jane Doe")


class TestInjuryApplication:
    @pytest.fixture
    def players(self):
        return {
            "p1": Player("p1", "Doe, John", "AAA", Position.GUARD, 8.0),
            "p2": Player("p2", "Roe, Richard", "BBB", Position.CENTER, 5.0),
        }

    def test_applies_status_and_note(self, players):
        records = [InjuryRecord("John Doe", "AAA", Availability.OUT, "ankle", "test")]
        matched, unmatched = apply_to_players(records, players)
        assert matched == 1
        assert not unmatched
        assert players["p1"].status is Availability.OUT
        assert players["p1"].status_note == "ankle"

    def test_reports_unmatched_rather_than_failing(self, players):
        records = [InjuryRecord("Nobody At All", None, Availability.OUT, None, "test")]
        matched, unmatched = apply_to_players(records, players)
        assert matched == 0
        assert unmatched == ["Nobody At All"]

    def test_player_id_takes_precedence_over_the_name(self, players):
        records = [InjuryRecord("Totally Wrong Name", None, Availability.DOUBTFUL, None, "t", "p2")]
        matched, _ = apply_to_players(records, players)
        assert matched == 1
        assert players["p2"].status is Availability.DOUBTFUL


class TestAvailabilityParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Out", Availability.OUT),
            ("ruled out for the season", Availability.OUT),
            ("Doubtful", Availability.DOUBTFUL),
            ("game-time decision", Availability.QUESTIONABLE),
            ("Probable", Availability.PROBABLE),
            ("available", Availability.ACTIVE),
            ("", Availability.UNKNOWN),
            (None, Availability.UNKNOWN),
            ("who knows", Availability.UNKNOWN),
        ],
    )
    def test_parses_the_wording_sources_actually_use(self, raw, expected):
        assert Availability.parse(raw) is expected


class TestProviderMerge:
    def test_first_provider_wins(self, tmp_path):
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("- player: John Doe\n  status: out\n", encoding="utf-8")
        b.write_text("- player: John Doe\n  status: probable\n", encoding="utf-8")
        records = merge([ManualInjuryProvider(a, "a"), ManualInjuryProvider(b, "b")])
        assert len(records) == 1
        assert records[0].status is Availability.OUT

    def test_a_broken_provider_does_not_kill_the_run(self, tmp_path):
        class Exploding:
            name = "boom"

            def fetch(self):
                raise RuntimeError("upstream is down")

        good = tmp_path / "g.yaml"
        good.write_text("- player: Jane Roe\n  status: out\n", encoding="utf-8")
        records = merge([Exploding(), ManualInjuryProvider(good, "g")])
        assert len(records) == 1

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert ManualInjuryProvider(tmp_path / "nope.yaml").fetch() == []


class TestPositionParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Guard", Position.GUARD),
            ("G", Position.GUARD),
            ("Center", Position.CENTER),
            ("C", Position.CENTER),
            ("Forward", Position.FORWARD),
            (None, Position.FORWARD),
        ],
    )
    def test_parse(self, raw, expected):
        assert Position.parse(raw) is expected


class TestUtils:
    def test_ewma_weights_halve_at_the_half_life(self):
        w = ewma_weights(9, half_life=4.0)
        assert w[-1] == pytest.approx(1.0)
        assert w[-5] == pytest.approx(0.5)

    def test_shrink_moves_toward_the_prior_with_little_evidence(self):
        assert shrink(10.0, 5.0, n=0.0, prior_n=10.0) == pytest.approx(5.0)
        assert shrink(10.0, 5.0, n=10.0, prior_n=10.0) == pytest.approx(7.5)
        assert shrink(10.0, 5.0, n=1e6, prior_n=10.0) == pytest.approx(10.0, abs=1e-3)

    def test_precision_blend_favours_the_tighter_estimate(self):
        mean, var = precision_blend([10.0, 20.0], [1.0, 100.0])
        assert mean < 11.0
        assert var < 1.0

    def test_precision_blend_ignores_useless_estimates(self):
        mean, _ = precision_blend([10.0, 20.0], [1.0, 0.0])
        assert mean == pytest.approx(10.0)
