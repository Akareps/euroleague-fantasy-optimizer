"""Minutes are the highest-leverage input, so they get the most tests."""

from __future__ import annotations

import pytest

from elfantasy.features.minutes import (
    build_depth_chart,
    garbage_time_minutes,
    project_team_minutes,
    redistribution_weights,
)
from elfantasy.models import Availability, BoxScore, Player, Position


def _player(pid, pos, team="AAA", status=Availability.ACTIVE, name=None):
    return Player(
        player_id=pid,
        name=name or f"Player {pid}",
        team_code=team,
        position=pos,
        price=5.0,
        status=status,
    )


def _history(pid, team, minutes_per_game, rounds=8):
    return [
        BoxScore(
            game_id=f"G{r}",
            round=r,
            player_id=pid,
            team_code=team,
            minutes=minutes_per_game,
            points=minutes_per_game * 0.4,
            pir=minutes_per_game * 0.45,
        )
        for r in range(1, rounds + 1)
    ]


@pytest.fixture
def roster():
    """A ten-man rotation: two clear starters per position plus reserves."""

    spec = [
        ("g1", Position.GUARD, 32.0),
        ("g2", Position.GUARD, 26.0),
        ("g3", Position.GUARD, 12.0),
        ("f1", Position.FORWARD, 30.0),
        ("f2", Position.FORWARD, 22.0),
        ("f3", Position.FORWARD, 10.0),
        ("c1", Position.CENTER, 28.0),
        ("c2", Position.CENTER, 9.0),
        ("c3", Position.CENTER, 4.0),
    ]
    players = [_player(pid, pos) for pid, pos, _ in spec]
    box = []
    for pid, _, mins in spec:
        box.extend(_history(pid, "AAA", mins))
    return players, box, {pid: mins for pid, _, mins in spec}


class TestDepthChart:
    def test_ranks_follow_minutes(self, roster, model):
        players, box, _ = roster
        depth = build_depth_chart("AAA", players, box, model)
        assert depth.rank["g1"] < depth.rank["g3"]
        assert depth.rank["c1"] < depth.rank["c2"]

    def test_baselines_are_shrunk_toward_the_rank_prior(self, roster, model):
        players, box, actual = roster
        depth = build_depth_chart("AAA", players, box, model)
        # The top man's 32 minutes are pulled down toward the rank-1 prior of 27.
        assert actual["g1"] > depth.baseline["g1"] > 27.0

    def test_a_player_with_no_history_still_gets_a_baseline(self, roster, model):
        players, box, _ = roster
        players.append(_player("new", Position.GUARD))
        depth = build_depth_chart("AAA", players, box, model)
        assert depth.baseline["new"] >= 0.0
        assert "new" in depth.rank


class TestRedistribution:
    def test_weights_sum_to_one(self, roster, model):
        players, box, _ = roster
        depth = build_depth_chart("AAA", players, box, model)
        ids = [p.player_id for p in players]
        weights = redistribution_weights("c1", ids, depth, model)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_absent_player_gets_no_share_of_his_own_minutes(self, roster, model):
        players, box, _ = roster
        depth = build_depth_chart("AAA", players, box, model)
        ids = [p.player_id for p in players]
        assert "c1" not in redistribution_weights("c1", ids, depth, model)

    def test_same_position_absorbs_more_than_other_positions(self, roster, model):
        """The headline case: the starting centre sits, the backup centre eats."""

        players, box, _ = roster
        depth = build_depth_chart("AAA", players, box, model)
        ids = [p.player_id for p in players]
        weights = redistribution_weights("c1", ids, depth, model)
        assert weights["c2"] > weights["g2"]
        assert weights["c2"] > weights["f2"]


class TestProjection:
    def test_healthy_roster_is_close_to_baseline(self, roster, model):
        players, box, _ = roster
        depth = build_depth_chart("AAA", players, box, model)
        proj = project_team_minutes("AAA", players, depth, model)
        for pid, mp in proj.items():
            assert mp.minutes == pytest.approx(mp.baseline, abs=3.0), pid

    def test_starting_centre_out_lifts_the_backup(self, roster, model):
        """The value-play scenario the whole project is built around."""

        players, box, _ = roster
        depth = build_depth_chart("AAA", players, box, model)
        before = project_team_minutes("AAA", players, depth, model)

        for p in players:
            if p.player_id == "c1":
                p.status = Availability.OUT
        after = project_team_minutes("AAA", players, depth, model)

        gain = after["c2"].minutes - before["c2"].minutes
        assert gain > 3.0, f"backup centre only gained {gain:.1f} minutes"
        assert after["c2"].from_absences > 0
        assert any("absent" in note for note in after["c2"].notes)

    def test_out_player_has_zero_play_probability(self, roster, model):
        players, box, _ = roster
        for p in players:
            if p.player_id == "c1":
                p.status = Availability.OUT
        depth = build_depth_chart("AAA", players, box, model)
        proj = project_team_minutes("AAA", players, depth, model)
        assert proj["c1"].play_prob == 0.0

    def test_questionable_status_partially_redistributes(self, roster, model):
        players, box, _ = roster
        depth = build_depth_chart("AAA", players, box, model)
        base = project_team_minutes("AAA", players, depth, model)["c2"].minutes

        for p in players:
            if p.player_id == "c1":
                p.status = Availability.QUESTIONABLE
        partial = project_team_minutes("AAA", players, depth, model)["c2"].minutes

        for p in players:
            if p.player_id == "c1":
                p.status = Availability.OUT
        full = project_team_minutes("AAA", players, depth, model)["c2"].minutes

        assert base < partial < full

    def test_nobody_exceeds_the_game_length(self, roster, model):
        players, box, _ = roster
        for p in players[1:]:
            p.status = Availability.OUT
        depth = build_depth_chart("AAA", players, box, model)
        proj = project_team_minutes("AAA", players, depth, model)
        cap = float(model.get("minutes.max_minutes"))
        assert all(mp.minutes <= cap + 1e-6 for mp in proj.values())


class TestBlowout:
    def test_no_garbage_time_in_a_close_game(self, model):
        assert garbage_time_minutes(2.0, model) == 0.0

    def test_garbage_time_grows_with_the_spread(self, model):
        assert garbage_time_minutes(20.0, model) > garbage_time_minutes(12.0, model)

    def test_garbage_time_is_capped(self, model):
        cap = float(model.get("blowout.cap_minutes"))
        assert garbage_time_minutes(60.0, model) == pytest.approx(cap)

    def test_symmetric_in_favourite_and_underdog(self, model):
        assert garbage_time_minutes(18.0, model) == garbage_time_minutes(-18.0, model)

    def test_blowout_takes_minutes_from_starters_and_gives_them_to_the_bench(self, roster, model):
        players, box, _ = roster
        depth = build_depth_chart("AAA", players, box, model)
        close = project_team_minutes("AAA", players, depth, model, spread=-1.0)
        rout = project_team_minutes("AAA", players, depth, model, spread=-22.0)

        assert rout["g1"].minutes < close["g1"].minutes
        assert rout["c3"].minutes > close["c3"].minutes


class TestMinuteBudget:
    """When a roster claims more than 200 minutes, the bench gives them back."""

    def test_stars_keep_their_minutes_on_a_deep_roster(self, roster, model):
        players, box, _ = roster
        # Five extra 15-minute players push the team well past 200 minutes.
        for k in range(5):
            pid = f"x{k}"
            players.append(_player(pid, Position.FORWARD))
            box.extend(_history(pid, "AAA", 15.0))
        depth = build_depth_chart("AAA", players, box, model)
        proj = project_team_minutes("AAA", players, depth, model)

        star_cut = 1 - proj["g1"].minutes / proj["g1"].baseline
        deep_cut = 1 - proj["x4"].minutes / proj["x4"].baseline
        assert deep_cut > 2 * star_cut
        assert sum(mp.minutes * mp.play_prob for mp in proj.values()) == pytest.approx(
            float(model.get("minutes.team_minutes")), rel=0.03
        )


class TestStatedMinutes:
    def test_an_override_replaces_the_baseline(self, roster, model):
        players, box, _ = roster
        for p in players:
            if p.player_id == "c3":
                p.minutes_override = 13.0
        depth = build_depth_chart("AAA", players, box, model)
        proj = project_team_minutes("AAA", players, depth, model)
        assert proj["c3"].minutes == pytest.approx(13.0)

    def test_an_override_is_not_moved_by_absences_or_blowouts(self, roster, model):
        players, box, _ = roster
        for p in players:
            if p.player_id == "c3":
                p.minutes_override = 13.0
            if p.player_id == "c1":
                p.status = Availability.OUT
        depth = build_depth_chart("AAA", players, box, model)
        proj = project_team_minutes("AAA", players, depth, model, spread=-22.0)
        assert proj["c3"].minutes == pytest.approx(13.0)
        # ...while unlocked backups still absorb the starter's minutes.
        assert proj["c2"].from_absences > 0
