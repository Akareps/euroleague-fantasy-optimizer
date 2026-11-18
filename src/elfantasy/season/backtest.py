"""Chronological backtest of the season model.

Every played round is re-projected with today's model and only the
information available before it. Anything fitted -- the calibration, the price
baseline -- uses earlier rounds only, so each round is a genuine forecast of
the next. One exception the numbers cannot remove: the evidence weights in
season.yaml were chosen by looking at some rounds (``model.tuned_on``); scores
for those rounds flatter the model, later rounds do not.

Universe: every player the model expected to play (P >= 0.5). A player who
then did not play scores 0, as in the game.

Candidate adjustments are scored alongside, without changing any projection,
until enough rounds say whether to adopt them: the opponent's record against
each position (what players of that position scored against him beyond their
forecasts in earlier rounds); short rest (forecast errors of projected starters
on three days' rest or less, e.g. the second game of a double-round week); a
tipster bump (what Basketball Sphere's round picks scored beyond their
forecasts, over unpicked players of the same position and price band, in
earlier rounds, added to this round's picks); recent form (each player's
actual - forecast over his last rounds); and form and tipster together, to
see whether form explains what the tipster adds.

The tipster bump is scored twice. "All articles" uses every archived pick and
is exploratory only: most were archived after the games, from articles edited
after them. "Dated" fits and applies the bump on picks known before tip-off
only: archived before it, or archived later from an article its publisher
dates before it. Picks we archived ourselves before tip-off are the only fully
verified kind; the archive holds them from Round 13 on.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.stats import spearmanr

from elfantasy.season.model import SeasonModel, position_tier


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
    blend: Score | None = None  # the forecast used: model and price mixed, corrected
    plain_blend: Score | None = None  # the same without the position-and-price correction
    opp_adjusted: Score | None = None  # the blend plus the opponent-vs-position adjustment
    # actual - forecast for projected starters, by rest since the club's previous game
    short_rest: list[float] = field(default_factory=list)
    normal_rest: list[float] = field(default_factory=list)
    tipster: Score | None = None  # exploratory: the bump from every pick, whatever its timing
    tipster_dated: Score | None = None  # the bump fitted and applied on dated picks only
    form: Score | None = None  # the blend plus recent form
    form_tipster: Score | None = None  # both, the bump fitted on the form-adjusted forecast
    # picks of this round: actual - forecast beyond unpicked peers, by when the pick was
    # known (every pick; dated before tip-off by the publisher; archived before tip-off),
    # and every pick against the form-adjusted forecast
    tip_excess: list[float] = field(default_factory=list)
    tip_excess_dated: list[float] = field(default_factory=list)
    tip_excess_observed: list[float] = field(default_factory=list)
    tip_excess_after_form: list[float] = field(default_factory=list)


OPP_PRIOR_GAMES = 60.0  # pseudo-games of "no effect" behind each opponent-position record
TIP_PRIOR_PICKS = 40.0  # pseudo-picks of "no effect" behind the tipster bump
FORM_ROUNDS = 3  # recent form: his last rounds
FORM_PRIOR_GAMES = 3.0  # pseudo-games of "no effect" behind each player's form
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


def peer_excess(
    rows: list[dict], tipped: set[tuple[int, str]], key: str = "forecast",
    others: set[tuple[int, str]] | None = None,
) -> list[tuple[dict, float]]:  # fmt: skip
    """(row, excess) for each tipped row: his actual - ``key`` beyond the mean
    of the rows of the same round, position and price band that no pick names
    (``others``: every pick, when ``tipped`` is a subset)."""
    picked = tipped if others is None else others
    peer: dict[tuple, list[float]] = defaultdict(list)
    for x in rows:
        if (x["round"], x["name"]) not in picked:
            peer[(x["round"], *position_tier(x["pos"], x["price"]))].append(x["actual"] - x[key])
    out = []
    for x in rows:
        base = peer.get((x["round"], *position_tier(x["pos"], x["price"])))
        if (x["round"], x["name"]) in tipped and base:
            out.append((x, x["actual"] - x[key] - float(np.mean(base))))
    return out


def tipster_bump(
    earlier: list[dict], tipped: set[tuple[int, str]], key: str = "forecast",
    prior_picks: float = TIP_PRIOR_PICKS, others: set[tuple[int, str]] | None = None,
) -> float:  # fmt: skip
    """Points to add to a pick's forecast: the picks' mean excess over their
    peers in ``earlier`` rows, shrunk towards 0 with ``prior_picks``."""
    excess = [e for _, e in peer_excess(earlier, tipped, key, others)]
    return sum(excess) / (len(excess) + prior_picks) if excess else 0.0


def recent_form(
    rows_by_round: dict[int, list[dict]], round_no: int, rounds: int = FORM_ROUNDS,
    prior_games: float = FORM_PRIOR_GAMES,
) -> dict[str, float]:  # fmt: skip
    """name -> his actual - forecast over the ``rounds`` rounds before
    ``round_no``, shrunk towards 0 with ``prior_games`` pseudo-games."""
    total: dict[str, float] = defaultdict(float)
    n: dict[str, int] = defaultdict(int)
    for r in range(round_no - rounds, round_no):
        for x in rows_by_round.get(r, []):
            total[x["name"]] += x["actual"] - x["forecast"]
            n[x["name"]] += 1
    return {k: total[k] / (n[k] + prior_games) for k in total}


def tipped_rows(
    model: SeasonModel, rows_by_round: dict[int, list[dict]]
) -> dict[tuple[int, str], str]:
    """(round, name) of every row a round's archived picks name -> when the pick
    was known: ``observed``, ``dated`` or ``late`` (see :mod:`elfantasy.season.tips`)."""
    from elfantasy.season.archive import Archive
    from elfantasy.season.tips import first_tipoffs, match_pick, round_tips

    archived = Archive(model.data.root / "archive").read("tips")
    tipoff = first_tipoffs(model.games)
    timing: dict[tuple[int, str], str] = {}
    for r, xs in rows_by_round.items():
        for pick in round_tips(archived, r, tipoff.get(r)):
            hit = match_pick(pick, xs, model.season.clubs) if pick["pos"] != "HC" else None
            if hit:
                timing[(r, hit["name"])] = pick["timing"] or "late"
    return timing


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
    model.chronological_forecasts(rows)
    timing = tipped_rows(model, rows)
    tipped = set(timing)
    dated = {k for k, t in timing.items() if t in ("observed", "dated")}
    out = []
    for r in played:
        earlier = [x for k in played if k < r for x in rows[k]]
        test = rows[r]
        form_adj = recent_form(rows, r)
        for x in test:
            x["form_forecast"] = max(x["forecast"] + form_adj.get(x["name"], 0.0), 0.0)
        act = [x["actual"] for x in test]
        raw = [x["ev"] for x in test]
        calibrated = [x["calibrated"] for x in test]
        price = blend = plain_blend = opp_adjusted = tipster = tipster_dated = None
        form = form_tipster = None
        if earlier:
            price = _score([x["price_implied"] for x in test], act, top_k)
            blend = _score([x["forecast"] for x in test], act, top_k)
            plain_blend = _score([x["forecast_plain"] for x in test], act, top_k)
            adj = opponent_adjustment(earlier)
            opp_adjusted = _score(
                [x["forecast"] + adj.get((x["opp"], x["pos"]), 0.0) for x in test], act, top_k
            )
            bump = tipster_bump(earlier, tipped)
            bump_form = tipster_bump(earlier, tipped, key="form_forecast")
            picked = [(r, x["name"]) in tipped for x in test]
            tipster = _score(
                [x["forecast"] + bump * t for x, t in zip(test, picked, strict=True)], act, top_k
            )
            bump_dated = tipster_bump(earlier, dated, others=tipped)
            tipster_dated = _score(
                [x["forecast"] + bump_dated * ((r, x["name"]) in dated) for x in test], act, top_k
            )
            form = _score([x["form_forecast"] for x in test], act, top_k)
            form_tipster = _score(
                [x["form_forecast"] + bump_form * t for x, t in zip(test, picked, strict=True)],
                act,
                top_k,
            )
        excess = peer_excess(test, tipped)
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
                plain_blend=plain_blend,
                hindsight_top=float(np.sort(act)[::-1][:top_k].mean()),
                tuned_on=r in model.season.tuned_on,
                opp_adjusted=opp_adjusted,
                short_rest=short,
                normal_rest=normal,
                tipster=tipster,
                tipster_dated=tipster_dated,
                form=form,
                form_tipster=form_tipster,
                tip_excess=[e for _, e in excess],
                tip_excess_dated=[e for x, e in excess if timing[(r, x["name"])] == "dated"],
                tip_excess_observed=[e for x, e in excess if timing[(r, x["name"])] == "observed"],
                tip_excess_after_form=[e for _, e in peer_excess(test, tipped, "form_forecast")],
            )
        )
    return out
