"""The projection engine.

The chain, for one player in one round:

    E[PIR] = P(play)
             x  minutes(baseline, absences, blowout, ramp)
             x  per-minute rates (recency-weighted, shrunk, usage-adjusted)
             x  venue x rest x opponent-vs-position x pace x team-total
             x  team-mate synergy

The result is then blended with any market-implied projection using
inverse-variance weighting, so the market dominates when it is available and
tightly priced, and the model carries rounds the market has not opened yet.

Everything that multiplies into the number is recorded in
``Projection.components`` so ``elfantasy explain`` can show the full derivation
rather than a black-box score.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass

from elfantasy.config import Settings
from elfantasy.dataset import Dataset
from elfantasy.features import context as ctxmod
from elfantasy.features.minutes import (
    DepthChart,
    build_depth_chart,
    project_team_minutes,
    redistribution_weights,
)
from elfantasy.features.rates import RateProfile, consistency, fit_all, usage_transfer
from elfantasy.features.ratings import TeamRatings, fit_ratings
from elfantasy.features.synergy import SynergyModel, pair_effects
from elfantasy.models import Availability, Projection
from elfantasy.projection import market as mk
from elfantasy.projection.pir import StatLine, compose_pir_from_props, statline_variance
from elfantasy.util import clamp, precision_blend

# Which prop markets map onto which PIR component.
PROP_MARKETS = {
    "points": "points",
    "player_points": "points",
    "rebounds": "rebounds",
    "player_rebounds": "rebounds",
    "assists": "assists",
    "player_assists": "assists",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
}
COUNT_MARKETS = {"rebounds", "assists", "steals", "blocks", "turnovers"}


@dataclass
class FittedModel:
    """Everything fitted once from history and reused for every round."""

    ratings: TeamRatings
    profiles: dict[str, RateProfile]
    depth_charts: dict[str, DepthChart]
    synergy: SynergyModel
    consistency: dict[str, float]


def fit(dataset: Dataset, settings: Settings) -> FittedModel:
    """Fit team ratings, rate profiles, depth charts and synergy from history."""

    dataset.annotate_home_flags()
    model = settings.model
    played = dataset.played_boxscores()
    positions = {p.player_id: p.position.value for p in dataset.players.values()}

    ratings = fit_ratings(dataset.games, played, positions)
    profiles = fit_all(positions, played, model)
    depth = {
        code: build_depth_chart(code, dataset.team_players(code), played, model)
        for code in dataset.teams
    }
    synergy = pair_effects(played, model)
    cons = {pid: consistency(played, pid) for pid in positions}

    return FittedModel(
        ratings=ratings,
        profiles=profiles,
        depth_charts=depth,
        synergy=synergy,
        consistency=cons,
    )


def _play_prob(status: Availability, settings: Settings) -> float:
    table = settings.model.get("availability.play_prob").as_dict()
    return float(table.get(status.value, table.get("unknown", 0.9)))


def _market_context(dataset: Dataset, game_id: str, settings: Settings):
    """Return ``(spread, total, home_win_prob)`` from odds, or ``(None,)*3``."""

    o = dataset.odds.get(game_id)
    if o is None:
        return None, None, None
    method = str(settings.model.get("market.devig"))
    win_prob = None
    if o.home_price and o.away_price:
        try:
            win_prob = mk.two_way_prob(o.home_price, o.away_price, method)
        except (ValueError, ZeroDivisionError):
            win_prob = None
    spread = o.spread
    if spread is None and win_prob is not None:
        spread = mk.win_prob_to_spread(win_prob)
    if win_prob is None and spread is not None:
        win_prob = mk.spread_to_win_prob(spread)
    return spread, o.total, win_prob


def _market_statline(
    dataset: Dataset,
    player_id: str,
    game_id: str,
    modelled: StatLine,
    minutes: float,
    settings: Settings,
) -> tuple[StatLine | None, float]:
    """Assemble a market-implied stat line from whatever props exist.

    Returns ``(statline, weight)`` where weight in [0, 1] reflects how much of
    the PIR the market actually priced.
    """

    if not bool(settings.model.get("market.compose_pir_from_props")):
        return None, 0.0

    props = dataset.props_for_game(game_id).get(player_id, [])
    if not props:
        return None, 0.0

    method = str(settings.model.get("market.devig"))
    sd_per_sqrt = float(settings.model.get("market.pir_sd_per_sqrt_minute"))

    implied: dict[str, float] = {}
    direct_pir: float | None = None

    for prop in props:
        market_name = prop.market.strip().lower()
        if market_name in ("pir", "player_pir", "performance_index_rating"):
            with contextlib.suppress(ValueError, ZeroDivisionError):
                direct_pir = mk.prop_to_mean(
                    prop.line,
                    prop.over_price,
                    prop.under_price,
                    dist="normal",
                    sd=mk.pir_sd(minutes, sd_per_sqrt),
                    method=method,
                )
            continue

        component = PROP_MARKETS.get(market_name)
        if component is None:
            continue
        try:
            implied[component] = mk.prop_to_mean(
                prop.line,
                prop.over_price,
                prop.under_price,
                dist="poisson" if component in COUNT_MARKETS else "normal",
                sd=max(0.9 * (prop.line**0.5), 2.0),
                method=method,
            )
        except (ValueError, ZeroDivisionError):
            continue

    if direct_pir is not None:
        # A direct PIR line beats anything assembled from parts.
        scale = direct_pir / modelled.pir if modelled.pir > 0.5 else 1.0
        return modelled.scaled(clamp(scale, 0.5, 2.0)), 1.0

    if not implied:
        return None, 0.0

    composed = compose_pir_from_props(implied, modelled)
    # How much of the projected PIR the priced components account for.
    priced_mass = sum(abs(modelled.as_dict().get(c, 0.0)) for c in implied)
    total_mass = sum(abs(v) for v in modelled.as_dict().values())
    coverage = clamp(priced_mass / total_mass, 0.0, 1.0) if total_mass > 0 else 0.0
    return composed, coverage


def project_round(
    dataset: Dataset,
    fitted: FittedModel,
    settings: Settings,
    round_no: int,
) -> dict[str, Projection]:
    """Project every player who has a fixture in ``round_no``."""

    model = settings.model
    games = dataset.games_in_round(round_no)
    out: dict[str, Projection] = {}

    for game in games:
        spread, total, p_home = _market_context(dataset, game.game_id, settings)

        for team_code in (game.home_code, game.away_code):
            roster = dataset.team_players(team_code)
            if not roster:
                continue
            depth = fitted.depth_charts.get(team_code)
            if depth is None:
                depth = build_depth_chart(team_code, roster, dataset.played_boxscores(), model)

            gc = ctxmod.build_context(
                game,
                team_code,
                fitted.ratings,
                market_spread=spread,
                market_total=total,
                market_win_prob=p_home,
                previous_game_date=_previous_tipoff(dataset, team_code, round_no),
            )

            play_probs = {p.player_id: _play_prob(p.status, settings) for p in roster}
            minutes = project_team_minutes(
                team_code,
                roster,
                depth,
                model,
                spread=gc.spread,
                play_probs=play_probs,
            )

            absent = {
                p.player_id: 1.0 - play_probs[p.player_id]
                for p in roster
                if play_probs[p.player_id] < 0.8
            }
            available_ids = [p.player_id for p in roster if play_probs[p.player_id] >= 0.35]

            for player in roster:
                pid = player.player_id
                mp = minutes[pid]
                profile = fitted.profiles.get(pid)
                if profile is None:
                    continue

                # --- usage transfer from absent team-mates -----------------
                share = 0.0
                absent_profiles = []
                for mate_id, miss_prob in absent.items():
                    if mate_id == pid:
                        continue
                    mate_profile = fitted.profiles.get(mate_id)
                    if mate_profile is None:
                        continue
                    weights = redistribution_weights(mate_id, available_ids, depth, model)
                    w = weights.get(pid, 0.0)
                    if w <= 0:
                        continue
                    share = max(share, w)
                    absent_profiles.append((mate_profile, miss_prob))
                adjusted = usage_transfer(profile, absent_profiles, share, model)

                base_line = adjusted.statline(mp.minutes)

                # --- context multipliers -----------------------------------
                venue = ctxmod.venue_factor(profile.home_factor, profile.away_factor, gc.home)
                rest = ctxmod.rest_factor(gc.rest_days, model)
                opp = ctxmod.opponent_factor(
                    fitted.ratings, gc.opponent_code, profile.position, model
                )
                pace = ctxmod.game_pace_factor(gc.total, fitted.ratings, model)
                team_env = ctxmod.team_total_factor(gc.team_total, fitted.ratings)
                syn = fitted.synergy.factor(pid, {k: v for k, v in absent.items() if k != pid})

                multiplier = venue * rest * opp * pace * team_env * syn
                model_line = base_line.scaled(multiplier)
                model_pir = model_line.pir
                model_var = statline_variance(model_line)

                # --- market blend ------------------------------------------
                market_line, coverage = _market_statline(
                    dataset, pid, game.game_id, model_line, mp.minutes, settings
                )
                if market_line is not None and coverage > 0.05:
                    market_pir = market_line.pir
                    max_w = float(model.get("market.max_market_weight"))
                    # Market variance shrinks with coverage: a fully priced
                    # line is trusted much more than one assists prop.
                    market_var = model_var * (1.0 / max(coverage, 0.05)) * (1.0 - max_w) / max_w
                    mean_pir, var = precision_blend(
                        [model_pir, market_pir], [model_var, max(market_var, 1e-6)]
                    )
                else:
                    mean_pir, var = model_pir, model_var
                    market_pir = None

                # --- availability and dispersion ---------------------------
                play = mp.play_prob
                expected = play * mean_pir
                # Var(PIR) = E[Var | play] + Var(E[PIR | play])
                total_var = play * var + play * (1 - play) * mean_pir**2
                # Widen for players whose game-to-game output is erratic.
                total_var *= 1.0 + 0.35 * (fitted.consistency.get(pid, 0.6) - 0.6)
                sd = max(total_var, 0.25) ** 0.5

                proj = Projection(
                    player_id=pid,
                    round=round_no,
                    mean_pir=float(expected),
                    sd_pir=float(sd),
                    minutes=float(mp.minutes),
                    play_prob=float(play),
                    game_id=game.game_id,
                    opponent=gc.opponent_code,
                    home=gc.home,
                    win_prob=gc.win_prob,
                    components={
                        "baseline_minutes": mp.baseline,
                        "minutes_from_absences": mp.from_absences,
                        "minutes_from_blowout": mp.from_blowout,
                        "pir_per_minute": adjusted.statline(1.0).pir,
                        "venue": venue,
                        "rest": rest,
                        "opponent": opp,
                        "pace": pace,
                        "team_total": team_env,
                        "synergy": syn,
                        "model_pir": model_pir,
                        "market_pir": market_pir if market_pir is not None else float("nan"),
                        "market_coverage": coverage,
                        "usage_share": share,
                        "spread": gc.spread,
                        "total": gc.total,
                    },
                    notes=list(mp.notes)
                    + fitted.synergy.explain(pid, absent)
                    + (["market lines"] if gc.source == "market" else []),
                )
                if player.status not in (Availability.ACTIVE, Availability.UNKNOWN):
                    proj.notes.insert(
                        0,
                        f"status: {player.status.value}"
                        + (f" ({player.status_note})" if player.status_note else ""),
                    )
                out[pid] = proj

    return out


def _previous_tipoff(dataset: Dataset, team_code: str, round_no: int):
    prev = dataset.last_game_before(team_code, round_no)
    return prev.tipoff if prev else None


def project_horizon(
    dataset: Dataset,
    fitted: FittedModel,
    settings: Settings,
    start_round: int,
    rounds: int | None = None,
) -> dict[int, dict[str, Projection]]:
    """Project ``rounds`` consecutive rounds starting at ``start_round``.

    Later rounds necessarily lean on the model rather than the market -- books
    do not price round 12 in round 9 -- which is exactly why the Elo/pace
    ratings exist.
    """

    n = rounds if rounds is not None else int(settings.model.get("optimiser.horizon_rounds"))
    available = sorted({g.round for g in dataset.games if g.round >= start_round})
    wanted = available[:n]
    return {r: project_round(dataset, fitted, settings, r) for r in wanted}


def horizon_value(
    horizon: dict[int, dict[str, Projection]],
    settings: Settings,
    gamma: float | None = None,
) -> dict[str, float]:
    """Collapse a multi-round projection into one value per player.

    ``value_i = sum_r gamma^r * E[PIR]_{i,r}``

    The discount is what encodes "I would rather own the player I do not have to
    transfer out next round". A player with one great fixture and then a brutal
    run scores worse than a slightly weaker player with three good ones, without
    any special-casing.
    """

    g = gamma if gamma is not None else float(settings.model.get("optimiser.horizon_gamma"))
    rounds = sorted(horizon)
    values: dict[str, float] = defaultdict(float)
    for i, r in enumerate(rounds):
        w = g**i
        for pid, proj in horizon[r].items():
            values[pid] += w * proj.mean_pir
    return dict(values)


def horizon_variance(
    horizon: dict[int, dict[str, Projection]],
    settings: Settings,
    gamma: float | None = None,
) -> dict[str, float]:
    g = gamma if gamma is not None else float(settings.model.get("optimiser.horizon_gamma"))
    rounds = sorted(horizon)
    out: dict[str, float] = defaultdict(float)
    for i, r in enumerate(rounds):
        w = (g**i) ** 2
        for pid, proj in horizon[r].items():
            out[pid] += w * proj.variance
    return dict(out)
