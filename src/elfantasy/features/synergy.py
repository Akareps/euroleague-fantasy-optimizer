"""Team-mate synergy (with-or-without-you effects).

Some players depend on a specific team-mate. A roll-man's production collapses
when the pick-and-roll guard is out; a spot-up shooter's efficiency falls when
the creator who feeds him is missing. That is a *different* effect from the
usage bump in :mod:`elfantasy.features.rates` -- there, an absence helps the
survivors; here, an absence can hurt them.

Two estimators are provided:

* :func:`pair_effects` -- game-level WOWY. Compares a player's per-minute PIR in
  games their team-mate played against games the team-mate missed. Simple,
  works with nothing but box scores, but confounded by everything.
* :func:`ridge_pair_effects` -- a ridge regression of per-minute PIR on
  team-mate availability indicators, which controls for the other absences in
  the same game. Preferred when there is enough data.

Both shrink hard. Pair samples in a 34-round league are tiny, and an unshrunk
WOWY number will confidently tell you nonsense.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from elfantasy.config import Section
from elfantasy.models import BoxScore
from elfantasy.util import clamp


@dataclass
class SynergyModel:
    """Multiplicative effect on player A's production of team-mate B playing."""

    effects: dict[tuple[str, str], float]
    max_abs_effect: float = 0.18

    def factor(self, player_id: str, absent_teammates: dict[str, float]) -> float:
        """Combined factor given team-mates and their probability of missing.

        ``absent_teammates`` maps team-mate id -> probability they do not play.
        """

        total = 0.0
        for mate, miss_prob in absent_teammates.items():
            eff = self.effects.get((player_id, mate))
            if eff is None or miss_prob <= 0:
                continue
            # `eff` is the effect of the mate *playing*; losing them removes it.
            total -= eff * miss_prob
        return clamp(1.0 + total, 1.0 - self.max_abs_effect, 1.0 + self.max_abs_effect)

    def explain(self, player_id: str, absent_teammates: dict[str, float]) -> list[str]:
        out = []
        for mate, miss_prob in absent_teammates.items():
            eff = self.effects.get((player_id, mate))
            if eff is None or abs(eff) < 0.02 or miss_prob <= 0.2:
                continue
            direction = "loses" if eff > 0 else "gains"
            out.append(f"{direction} {abs(eff) * 100:.0f}% without team-mate {mate}")
        return out


def _played_map(boxscores: list[BoxScore]) -> dict[tuple[str, str], float]:
    """(game_id, player_id) -> minutes."""

    return {(b.game_id, b.player_id): b.minutes for b in boxscores}


def pair_effects(boxscores: list[BoxScore], model: Section) -> SynergyModel:
    """Game-level WOWY with empirical-Bayes shrinkage toward zero effect."""

    min_pair_minutes = float(model.get("synergy.min_pair_minutes"))
    prior_minutes = float(model.get("synergy.prior_minutes"))
    max_abs = float(model.get("synergy.max_abs_effect"))

    by_game: dict[str, list[BoxScore]] = defaultdict(list)
    for bs in boxscores:
        by_game[bs.game_id].append(bs)

    # For every (player, team-mate) pair collect per-minute PIR split by
    # whether the team-mate was on the floor at all that game.
    with_mate: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    without_mate: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)

    for lines in by_game.values():
        by_team: dict[str, list[BoxScore]] = defaultdict(list)
        for bs in lines:
            by_team[bs.team_code].append(bs)
        for roster in by_team.values():
            active = {bs.player_id: bs.minutes for bs in roster}
            for bs in roster:
                if bs.minutes <= 4 or bs.pir is None:
                    continue
                rate = bs.pir / bs.minutes
                for mate_id in active:
                    if mate_id == bs.player_id:
                        continue
                    bucket = with_mate if active[mate_id] > 0 else without_mate
                    bucket[(bs.player_id, mate_id)].append((rate, bs.minutes))

    effects: dict[tuple[str, str], float] = {}
    for key, with_obs in with_mate.items():
        without_obs = without_mate.get(key, [])
        w_minutes = sum(m for _, m in with_obs)
        wo_minutes = sum(m for _, m in without_obs)
        if w_minutes < min_pair_minutes or wo_minutes < 40.0:
            continue
        w_rate = float(np.average([r for r, _ in with_obs], weights=[m for _, m in with_obs]))
        wo_rate = float(
            np.average([r for r, _ in without_obs], weights=[m for _, m in without_obs])
        )
        if wo_rate <= 0.01:
            continue
        raw = w_rate / wo_rate - 1.0
        # Shrink by the *smaller* of the two samples: the without-sample is
        # almost always the binding constraint.
        n = min(w_minutes, wo_minutes)
        shrunk = raw * (n / (n + prior_minutes))
        if abs(shrunk) < 0.015:
            continue
        effects[key] = float(clamp(shrunk, -max_abs, max_abs))

    return SynergyModel(effects=effects, max_abs_effect=max_abs)


def ridge_pair_effects(
    boxscores: list[BoxScore],
    model: Section,
    *,
    alpha: float = 25.0,
    min_games: int = 12,
) -> SynergyModel:
    """Ridge regression of per-minute PIR on team-mate availability.

    For each player, the design matrix has one column per team-mate (1 if the
    team-mate played that game, 0 otherwise) plus an intercept. Ridge shrinkage
    plays the role of the prior; ``alpha`` is deliberately large because the
    columns are highly collinear -- team-mates tend to be available together.
    """

    max_abs = float(model.get("synergy.max_abs_effect"))

    by_player_team: dict[str, str] = {b.player_id: b.team_code for b in boxscores}
    by_game_team: dict[tuple[str, str], list[BoxScore]] = defaultdict(list)
    for bs in boxscores:
        by_game_team[(bs.game_id, bs.team_code)].append(bs)

    team_games: dict[str, list[str]] = defaultdict(list)
    for game_id, team in by_game_team:
        team_games[team].append(game_id)

    effects: dict[tuple[str, str], float] = {}

    for team, games in team_games.items():
        roster = sorted({b.player_id for g in games for b in by_game_team[(g, team)]})
        if len(roster) < 6:
            continue
        index = {pid: i for i, pid in enumerate(roster)}

        # Availability matrix: games x roster.
        avail = np.zeros((len(games), len(roster)))
        rates = np.full((len(games), len(roster)), np.nan)
        weights = np.zeros((len(games), len(roster)))
        for gi, game_id in enumerate(games):
            for bs in by_game_team[(game_id, team)]:
                j = index[bs.player_id]
                if bs.minutes > 0:
                    avail[gi, j] = 1.0
                if bs.minutes > 4 and bs.pir is not None:
                    rates[gi, j] = bs.pir / bs.minutes
                    weights[gi, j] = bs.minutes

        for pid, j in index.items():
            mask = ~np.isnan(rates[:, j])
            if mask.sum() < min_games:
                continue
            y = rates[mask, j]
            w = weights[mask, j]
            mates = [k for k in range(len(roster)) if k != j]
            X = avail[np.ix_(mask.nonzero()[0], mates)]
            # Drop team-mates who were available (or absent) in every game --
            # they carry no information and only destabilise the fit.
            keep = [c for c in range(X.shape[1]) if 0 < X[:, c].sum() < X.shape[0]]
            if not keep:
                continue
            X = X[:, keep]
            mate_ids = [roster[mates[c]] for c in keep]

            Xd = np.hstack([np.ones((X.shape[0], 1)), X])
            W = np.diag(w / w.mean())
            reg = alpha * np.eye(Xd.shape[1])
            reg[0, 0] = 0.0  # do not penalise the intercept
            try:
                beta = np.linalg.solve(Xd.T @ W @ Xd + reg, Xd.T @ W @ y)
            except np.linalg.LinAlgError:  # pragma: no cover - singular design
                continue
            base = beta[0] + float(np.mean(X @ beta[1:]))
            if base <= 0.01:
                continue
            for c, mate_id in enumerate(mate_ids):
                eff = float(beta[1 + c] / base)
                if abs(eff) < 0.015:
                    continue
                effects[(pid, mate_id)] = float(clamp(eff, -max_abs, max_abs))

        _ = by_player_team  # kept for symmetry with pair_effects

    return SynergyModel(effects=effects, max_abs_effect=max_abs)


def combine(models: list[SynergyModel], weights: list[float] | None = None) -> SynergyModel:
    """Average several synergy estimates (e.g. WOWY and ridge)."""

    if not models:
        return SynergyModel(effects={})
    weights = weights or [1.0] * len(models)
    total_w = sum(weights)
    keys = {k for m in models for k in m.effects}
    merged: dict[tuple[str, str], float] = {}
    for key in keys:
        acc = sum(m.effects.get(key, 0.0) * w for m, w in zip(models, weights, strict=False))
        merged[key] = acc / total_w
    return SynergyModel(effects=merged, max_abs_effect=models[0].max_abs_effect)
