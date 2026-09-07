"""The glue between data, projections and the optimiser.

Kept separate from the CLI so the whole pipeline is usable as a library:

    from elfantasy.pipeline import Pipeline
    pipe = Pipeline.build(offline=True)
    plan = pipe.best_transfers(my_squad)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from elfantasy.config import Settings, load_settings
from elfantasy.data.load import build_dataset, data_health
from elfantasy.data.sample import build_sample_dataset
from elfantasy.dataset import Dataset
from elfantasy.models import Availability, Projection, Squad
from elfantasy.optimize.squad import Candidate, SquadSolution, optimise_squad
from elfantasy.optimize.transfers import TransferPlan, optimise_transfers, transfer_ladder
from elfantasy.projection import engine

log = logging.getLogger(__name__)


@dataclass
class Pipeline:
    settings: Settings
    dataset: Dataset
    fitted: engine.FittedModel
    horizon: dict[int, dict[str, Projection]]
    start_round: int

    # --- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        settings: Settings | None = None,
        dataset: Dataset | None = None,
        sample: bool = False,
        round_no: int | None = None,
        horizon_rounds: int | None = None,
        **load_kwargs,
    ) -> Pipeline:
        settings = settings or load_settings()
        if dataset is None:
            dataset = build_sample_dataset() if sample else build_dataset(settings, **load_kwargs)
        fitted = engine.fit(dataset, settings)
        start = round_no if round_no is not None else dataset.current_round()
        horizon = engine.project_horizon(dataset, fitted, settings, start, horizon_rounds)
        return cls(
            settings=settings,
            dataset=dataset,
            fitted=fitted,
            horizon=horizon,
            start_round=start,
        )

    # --- derived views -----------------------------------------------------
    @property
    def next_round(self) -> dict[str, Projection]:
        return self.horizon.get(self.start_round, {})

    def health(self) -> list[str]:
        return data_health(self.dataset)

    def candidates(
        self,
        *,
        include_unavailable: bool = False,
        min_price: float = 0.0,
        squad: Squad | None = None,
    ) -> list[Candidate]:
        """Every selectable player, reduced to what the optimiser needs.

        Players who are ruled out are excluded by default -- but never if you
        already own them, since the optimiser has to be able to see (and sell)
        an injured player currently in your squad.
        """

        values = engine.horizon_value(self.horizon, self.settings)
        variances = engine.horizon_variance(self.horizon, self.settings)
        next_round = self.next_round
        owned = set(squad.player_ids) if squad else set()

        out: list[Candidate] = []
        for pid, player in self.dataset.players.items():
            if player.price <= min_price and pid not in owned:
                continue
            if not include_unavailable and player.status is Availability.OUT and pid not in owned:
                continue
            proj = next_round.get(pid)
            out.append(
                Candidate(
                    player_id=pid,
                    name=player.name,
                    team_code=player.team_code,
                    position=player.position.value,
                    price=player.price,
                    value=values.get(pid, 0.0),
                    variance=variances.get(pid, 0.0),
                    next_round_pir=proj.mean_pir if proj else 0.0,
                    ownership=player.ownership,
                )
            )
        return out

    def fixture_win_probs(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for code in self.dataset.teams:
            probs = []
            for r in sorted(self.horizon):
                for proj in self.horizon[r].values():
                    player = self.dataset.players.get(proj.player_id)
                    if player and player.team_code == code and proj.win_prob is not None:
                        probs.append(proj.win_prob)
                        break
            if probs:
                out[code] = probs
        return out

    # --- decisions ---------------------------------------------------------
    def best_squad(self, **kwargs) -> SquadSolution:
        return optimise_squad(self.candidates(), self.settings, **kwargs)

    def best_transfers(self, squad: Squad, **kwargs) -> TransferPlan:
        return optimise_transfers(self.candidates(squad=squad), squad, self.settings, **kwargs)

    def transfer_ladder(self, squad: Squad, **kwargs) -> list[TransferPlan]:
        return transfer_ladder(self.candidates(squad=squad), squad, self.settings, **kwargs)

    def explain(self, player_id: str, round_no: int | None = None) -> Projection | None:
        r = round_no if round_no is not None else self.start_round
        return self.horizon.get(r, {}).get(player_id)

    def find_player(self, query: str) -> list[str]:
        """Fuzzy-ish lookup by name or id, for the CLI."""

        from elfantasy.data.injuries import normalise_name

        q = query.strip().lower()
        nq = normalise_name(query)
        hits = []
        for pid, p in self.dataset.players.items():
            if pid.lower() == q:
                return [pid]
            if q in p.name.lower() or nq == normalise_name(p.name):
                hits.append(pid)
        return hits

    def save_snapshot(self, path: Path | str) -> Path:
        return self.dataset.save(path)
