"""Chronological backtest of the season model.

Every played round is re-projected with today's model and only the
information available before it. Anything fitted -- the calibration, the price
baseline -- uses earlier rounds only, so each round is a genuine forecast of
the next. One exception the numbers cannot remove: the evidence weights in
season.yaml were chosen by looking at some rounds (``model.tuned_on``); scores
for those rounds flatter the model, later rounds do not.

Universe: every player the model expected to play (P >= 0.5). A player who
then did not play scores 0, as in the game.

Two candidate adjustments are scored alongside, without changing any
projection, until enough rounds say whether to adopt them: the opponent's
record against each position (what players of that position scored against
him beyond their forecasts in earlier rounds), and short rest (forecast errors
of projected starters on three days' rest or less, e.g. the second game of a
double-round week).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.stats import spearmanr

from elfantasy.season.model import (
    HEALTHY,
    SeasonModel,
    calibrate,
    fit_calibration,
    points_ratio,
    team_points,
)


@dataclass
class Score:
    rmse: float
    rho: float
    top: float  # mean actual points of the top-k by this predictor
    bias: float  # mean (prediction - actual)


@dataclass
class RoundScore:
    round: int
    n: int
    model: Score
    raw: Score
    price: Score | None  # actual ~ price, fitted on earlier rounds
    hindsight_top: float
    tuned_on: bool
    blend: Score | None = None  # model and price mixed by season.yaml's price_blend
    opp_adjusted: Score | None = None  # the blend plus the opponent-vs-position adjustment
    # actual - forecast for projected starters, by rest since the club's previous game
    short_rest: list[float] = field(default_factory=list)
    normal_rest: list[float] = field(default_factory=list)


OPP_PRIOR_GAMES = 60.0  # pseudo-games of "no effect" behind each opponent-position record
SHORT_REST_DAYS = 3
STARTER_MINUTES = 22.0


def opponent_adjustment(earlier: list[dict], prior_games: float = OPP_PRIOR_GAMES) -> dict:
    """(opponent, position) -> what players of that position scored against
    that opponent beyond their forecasts (``forecast``) in ``earlier`` rows,
    shrunk towards 0 with ``prior_games`` pseudo-games."""

    total: dict[tuple[str, str], float] = defaultdict(float)
    n: dict[tuple[str, str], int] = defaultdict(int)
    for x in earlier:
        total[(x["opp"], x["pos"])] += x["actual"] - x["forecast"]
        n[(x["opp"], x["pos"])] += 1
    return {k: total[k] / (n[k] + prior_games) for k in total}


def rest_days(rows_by_round: dict[int, list[dict]], round_no: int) -> dict[str, int]:
    """club -> days since its game in the previous round."""

    def dates(r):
        rows = rows_by_round.get(r, [])
        return {x["club"]: date.fromisoformat(x["date"][:10]) for x in rows if x.get("date")}

    now, before = dates(round_no), dates(round_no - 1)
    return {c: (d - before[c]).days for c, d in now.items() if c in before}


def _score(pred, act, k) -> Score:
    pred, act = np.asarray(pred, float), np.asarray(act, float)
    e = pred - act
    return Score(
        rmse=float(np.sqrt((e**2).mean())),
        rho=float(spearmanr(pred, act)[0]),
        top=float(act[np.argsort(-pred)[:k]].mean()),
        bias=float(e.mean()),
    )


def run_backtest(
    model: SeasonModel, through: int | None = None, top_k: int = 40
) -> list[RoundScore]:
    played = [r for r in model.data.played_rounds() if through is None or r <= through]
    rows = {r: model.reproject(r) for r in played}
    out = []
    for r in played:
        earlier = [x for k in played if k < r for x in rows[k]]
        cal = fit_calibration(
            [(x["group"], x["ev"], x["actual"]) for x in earlier], model.season.calibration
        )
        test = rows[r]
        act = [x["actual"] for x in test]
        raw = [x["ev"] for x in test]
        # The expected-points adjustment uses team averages (no saved over/under lines).
        teams, league = team_points(model.games, r, model.season.team_points_prior_games)
        k = model.season.team_points_weight
        calibrated = [
            calibrate(cal, x["group"], x["ev"])
            * (
                1
                + k
                * (points_ratio(x["club"], x["opp"], x["home"], x["margin"], {}, teams, league) - 1)
            )
            for x in test
        ]
        price = blend = None
        if earlier:
            b, a = np.polyfit([x["price"] for x in earlier], [x["actual"] for x in earlier], 1)
            by_price = [a + b * x["price"] for x in test]
            price = _score(by_price, act, top_k)
            w = model.season.price_blend
            mixed = [
                w * m + (1 - w) * max(q, 0.0) * x["play"] / HEALTHY
                for m, q, x in zip(calibrated, by_price, test, strict=True)
            ]
            blend = _score(mixed, act, top_k)
        forecast = mixed if earlier else calibrated
        for x, f in zip(test, forecast, strict=True):
            x["forecast"] = f
        opp_adjusted = None
        if earlier:
            adj = opponent_adjustment(earlier)
            opp_adjusted = _score(
                [x["forecast"] + adj.get((x["opp"], x["pos"]), 0.0) for x in test], act, top_k
            )
        rest = rest_days(rows, r)
        short, normal = [], []
        for x in test:
            if earlier and x["min_p"] >= STARTER_MINUTES and x["club"] in rest:
                (short if rest[x["club"]] <= SHORT_REST_DAYS else normal).append(
                    x["actual"] - x["forecast"]
                )
        out.append(
            RoundScore(
                round=r,
                n=len(test),
                model=_score(calibrated, act, top_k),
                raw=_score(raw, act, top_k),
                price=price,
                blend=blend,
                hindsight_top=float(np.sort(act)[::-1][:top_k].mean()),
                tuned_on=r in model.season.tuned_on,
                opp_adjusted=opp_adjusted,
                short_rest=short,
                normal_rest=normal,
            )
        )
    return out
