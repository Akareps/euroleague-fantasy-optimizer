"""Chronological backtest of the season model.

Every played round is re-projected with today's model and only the
information available before it. Anything fitted -- the calibration, the price
baseline -- uses earlier rounds only, so each round is a genuine forecast of
the next. One exception the numbers cannot remove: the evidence weights in
season.yaml were chosen by looking at some rounds (``model.tuned_on``); scores
for those rounds flatter the model, later rounds do not.

Universe: every player the model expected to play (P >= 0.5). A player who
then did not play scores 0, as in the game.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from elfantasy.season.model import SeasonModel, calibrate, fit_calibration


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
        calibrated = [calibrate(cal, x["group"], x["ev"]) for x in test]
        price = None
        if earlier:
            b, a = np.polyfit([x["price"] for x in earlier], [x["actual"] for x in earlier], 1)
            price = _score([a + b * x["price"] for x in test], act, top_k)
        out.append(
            RoundScore(
                round=r,
                n=len(test),
                model=_score(calibrated, act, top_k),
                raw=_score(raw, act, top_k),
                price=price,
                hindsight_top=float(np.sort(act)[::-1][:top_k].mean()),
                tuned_on=r in model.season.tuned_on,
            )
        )
    return out
