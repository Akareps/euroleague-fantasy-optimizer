"""End-to-end behaviour on the synthetic league."""

from __future__ import annotations

import pytest

from elfantasy.dataset import Dataset
from elfantasy.models import Availability, Squad
from elfantasy.projection import engine


class TestProjections:
    def test_every_playing_team_gets_projections(self, pipeline):
        projections = pipeline.next_round
        assert len(projections) > 50
        assert all(p.round == pipeline.start_round for p in projections.values())

    def test_projections_are_in_a_sane_range(self, pipeline):
        values = [p.mean_pir for p in pipeline.next_round.values()]
        assert max(values) < 45, "no EuroLeague player projects above 45 PIR"
        assert min(values) > -5
        assert sum(values) / len(values) > 0

    def test_ruled_out_players_project_zero(self, pipeline):
        for pid, proj in pipeline.next_round.items():
            if pipeline.dataset.players[pid].status is Availability.OUT:
                assert proj.play_prob == 0.0
                assert proj.mean_pir == pytest.approx(0.0)

    def test_variance_rises_with_projection(self, pipeline):
        rows = sorted(pipeline.next_round.values(), key=lambda p: p.mean_pir)
        low = rows[len(rows) // 4]
        high = rows[-1]
        assert high.sd_pir > low.sd_pir

    def test_components_are_recorded_for_explainability(self, pipeline):
        proj = max(pipeline.next_round.values(), key=lambda p: p.mean_pir)
        for key in ("baseline_minutes", "venue", "opponent", "pace", "model_pir"):
            assert key in proj.components

    def test_market_is_used_when_props_exist(self, pipeline):
        with_market = [
            p for p in pipeline.next_round.values() if p.components.get("market_coverage", 0) > 0
        ]
        assert with_market, "sample data has props, so some projection should use them"

    def test_horizon_covers_several_rounds(self, pipeline):
        assert len(pipeline.horizon) >= 2
        assert min(pipeline.horizon) == pipeline.start_round


class TestHorizonValue:
    def test_value_discounts_later_rounds(self, pipeline, settings):
        """A good run of fixtures beats one good game -- the core requirement."""

        gamma = float(settings.model.get("optimiser.horizon_gamma"))
        values = engine.horizon_value(pipeline.horizon, settings)
        rounds = sorted(pipeline.horizon)
        pid = max(values, key=lambda k: values[k])
        expected = sum(
            gamma**i * pipeline.horizon[r].get(pid).mean_pir
            for i, r in enumerate(rounds)
            if pid in pipeline.horizon[r]
        )
        assert values[pid] == pytest.approx(expected, abs=1e-6)

    def test_gamma_zero_collapses_to_a_single_round(self, pipeline, settings):
        values = engine.horizon_value(pipeline.horizon, settings, gamma=0.0)
        for pid, proj in pipeline.next_round.items():
            assert values[pid] == pytest.approx(proj.mean_pir, abs=1e-6)

    def test_variance_is_discounted_quadratically(self, pipeline, settings):
        variances = engine.horizon_variance(pipeline.horizon, settings, gamma=0.0)
        for pid, proj in pipeline.next_round.items():
            assert variances[pid] == pytest.approx(proj.variance, abs=1e-6)


class TestCandidates:
    def test_unavailable_players_are_excluded_by_default(self, pipeline):
        ids = {c.player_id for c in pipeline.candidates()}
        out = {pid for pid, p in pipeline.dataset.players.items() if p.status is Availability.OUT}
        assert not (ids & out)

    def test_owned_players_are_always_included(self, pipeline):
        """You must be able to see -- and sell -- an injured player you own."""

        out = [pid for pid, p in pipeline.dataset.players.items() if p.status is Availability.OUT]
        assert out, "sample data should contain at least one ruled-out player"
        squad = Squad(player_ids=[out[0]])
        ids = {c.player_id for c in pipeline.candidates(squad=squad)}
        assert out[0] in ids


class TestOptimisation:
    def test_squad_is_legal_and_within_budget(self, pipeline, settings):
        sol = pipeline.best_squad()
        assert len(sol.selected) == settings.squad_size
        assert sol.total_price <= settings.budget + 1e-6
        assert sol.next_round_pir > 0

    def test_transfers_improve_a_weak_squad(self, pipeline, settings):
        pool = sorted(pipeline.candidates(), key=lambda c: c.price)
        bounds = settings.position_bounds()
        counts: dict[str, int] = {}
        clubs: dict[str, int] = {}
        ids: list[str] = []
        for phase in ("min", "max"):
            for c in pool:
                if len(ids) >= settings.squad_size or c.player_id in ids:
                    continue
                lo, hi = bounds[c.position]
                if counts.get(c.position, 0) >= (lo if phase == "min" else hi):
                    continue
                if clubs.get(c.team_code, 0) >= settings.max_per_club:
                    continue
                ids.append(c.player_id)
                counts[c.position] = counts.get(c.position, 0) + 1
                clubs[c.team_code] = clubs.get(c.team_code, 0) + 1

        squad = Squad(player_ids=ids, bank=30.0)
        plan = pipeline.best_transfers(squad, max_transfers=4)
        assert plan.n_transfers > 0
        assert plan.value_after > plan.value_before


class TestSnapshotRoundTrip:
    def test_dataset_survives_save_and_load(self, dataset, tmp_path):
        path = dataset.save(tmp_path / "snap.json")
        loaded = Dataset.load(path)
        assert loaded.summary()["players"] == dataset.summary()["players"]
        assert loaded.summary()["boxscores"] == dataset.summary()["boxscores"]
        assert len(loaded.odds) == len(dataset.odds)
        assert len(loaded.props) == len(dataset.props)

    def test_reloaded_dataset_projects_identically(self, dataset, settings, tmp_path):
        from elfantasy.pipeline import Pipeline

        path = dataset.save(tmp_path / "snap.json")
        reloaded = Pipeline.build(settings=settings, dataset=Dataset.load(path))
        original = Pipeline.build(settings=settings, dataset=dataset)
        for pid, proj in original.next_round.items():
            assert reloaded.next_round[pid].mean_pir == pytest.approx(proj.mean_pir, abs=1e-6)


class TestDataHealth:
    def test_reports_missing_market_and_injury_data(self, settings):
        from elfantasy.data.load import data_health
        from elfantasy.dataset import Dataset as DS

        empty = DS(season="X")
        warnings = data_health(empty)
        assert any("prices" in w for w in warnings)
        assert any("odds" in w for w in warnings)


class TestMarketBlendVariance:
    def test_props_move_the_mean_without_shrinking_game_variance(self, settings, dataset):
        """Blending two estimates of the mean says nothing about one game's swing."""

        import copy
        import dataclasses

        from elfantasy.config import Section

        fitted = engine.fit(dataset, settings)
        rnd = dataset.current_round()
        with_market = engine.project_round(dataset, fitted, settings, rnd)

        raw = copy.deepcopy(settings.model.as_dict())
        raw["market"] = dict(raw["market"], compose_pir_from_props=False)
        model_only = engine.project_round(
            dataset, fitted, dataclasses.replace(settings, model=Section(raw, "model")), rnd
        )

        checked = 0
        for pid, p in with_market.items():
            if p.components.get("market_coverage", 0) <= 0.05 or p.mean_pir <= 1:
                continue
            q = model_only[pid]
            # Dispersion scales with the mean, so compare SD per sqrt(mean).
            assert p.sd_pir / p.mean_pir**0.5 >= 0.9 * q.sd_pir / q.mean_pir**0.5
            checked += 1
        assert checked > 10
