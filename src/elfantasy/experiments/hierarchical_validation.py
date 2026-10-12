"""Chronological evaluation with fixed coefficients and online past-only state."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from itertools import groupby

import numpy as np
from scipy.stats import norm

from elfantasy.experiments.hierarchical import Model, apply_model
from elfantasy.experiments.history import utc
from elfantasy.season.backtest import _score


def simple_forecast(fitted, row):
    """Lagged minutes times shrunk PIR/min; independent of the production engine."""
    recent, _, _, _, _ = fitted.state.role(row)
    old = fitted.state.history(row)[-40:]
    pool = fitted.pools[row["pos"]]
    s = pool["efficiency"]
    signs = np.array([3 * s[0] - 1, 4 * s[1] - 1, 2 * s[2] - 1, 1, 1, 1, 1, 1, -1, -1, -1])
    prior = float(signs @ pool["rate"])
    weights = 0.5 ** (np.arange(len(old) - 1, -1, -1) / 8)
    rate = (120 * prior + sum(w * x["pir"] for w, x in zip(weights, old, strict=True))) / (
        120 + sum(w * x["minutes"] for w, x in zip(weights, old, strict=True))
    )
    _, _, margin = fitted.state.context(row)
    win = float(norm.cdf(margin / 11.5))
    return max(recent * rate, 0) * (1 + 0.1 * win), recent


def score_rows(rows, prefix):
    actual = [r["actual"] for r in rows]
    score = asdict(_score([r[prefix + "_ev"] for r in rows], actual, 40))
    score["minutes_rmse"] = float(
        np.sqrt(np.mean([(r[prefix + "_minutes"] - r["actual_minutes"]) ** 2 for r in rows]))
    )
    if prefix == "independent":
        score["p10_p90_coverage"] = float(
            np.mean([r["p10"] <= r["actual"] <= r["p90"] for r in rows])
        )
    return score


def historical_validation(history, *, through=None, settings=None):
    test = [
        r
        for r in history["rows"]
        if r["season"] == "E2025" and (through is None or r["round"] <= through)
    ]
    cutoff = min(r["date"] for r in test)
    fitted = Model(history["rows"], cutoff, settings)
    rounds, all_rows = [], []
    observed = set()
    prior_test = sorted(
        (r for r in history["rows"] if r["season"] == "E2025"), key=lambda r: r["date"]
    )
    for rnd in sorted(
        {r["round"] for r in test},
        key=lambda n: min(utc(r["date"]) for r in test if r["round"] == n),
    ):
        target = [r for r in test if r["round"] == rnd]
        boundary = min(utc(r["date"]) for r in target)
        # Postponed fixtures can overlap round numbers. Only completed prior
        # calendar days enter state; same-day games are conservatively excluded.
        eligible = [
            r
            for r in prior_test
            if utc(r["date"]).date() < boundary.date()
            and (r["game_id"], r["player_id"]) not in observed
        ]
        for _, grp in groupby(eligible, key=lambda r: r["date"][:10]):
            grp = list(grp)
            fitted.state.observe(grp)
            observed.update((r["game_id"], r["player_id"]) for r in grp)
        rows = []
        for r in target:
            pred = fitted.predict(r)
            own, minutes = simple_forecast(fitted, r)
            _, _, margin = fitted.state.context(r)
            win = float(norm.cdf(margin / 11.5))
            bonus = 1 + 0.1 * win
            actual = r["pir"] * (1.1 if r["margin"] > 0 else 1)
            mean, sd = pred["pir"] * bonus, pred["pir_sd"] * bonus
            rows.append(
                {
                    "player_id": r["player_id"],
                    "game_id": r["game_id"],
                    "round": rnd,
                    "actual": actual,
                    "actual_minutes": r["regulation_minutes"],
                    "simple_ev": own,
                    "simple_minutes": minutes,
                    "independent_ev": mean,
                    "independent_minutes": pred["minutes"],
                    "p10": mean + norm.ppf(0.1) * sd,
                    "p90": mean + norm.ppf(0.9) * sd,
                }
            )
        rounds.append(
            {
                "round": rnd,
                "n": len(rows),
                "simple": score_rows(rows, "simple"),
                "independent": score_rows(rows, "independent"),
                "evidence_before": fitted.state.latest_date.isoformat(),
                "first_tip": boundary.isoformat(),
            }
        )
        all_rows += rows
    return {
        "training": fitted.fit_info,
        "test_season": "E2025",
        "rounds": rounds,
        "n": len(all_rows),
        "simple": score_rows(all_rows, "simple"),
        "independent": score_rows(all_rows, "independent"),
        "mean_top40_per_round": {
            name: float(np.mean([r[name]["top"] for r in rounds]))
            for name in ("simple", "independent")
        },
        "universe": "Players on actual game-day box sheets, including rostered zero-minute players. This is conditional on box-sheet membership; it does not test pre-deadline injury/selection prediction.",
        "note": "Coefficients/pools fitted only on E2024. Before each round, E2025 state includes only games from prior calendar days, handling postponed fixtures without hindsight. No tuning on E2025. Minute errors use regulation-normalised minutes.",
        "players": all_rows,
    }


def current_validation(season_model, history, current, through=4, settings=None):
    first = min(g["utcDate"] for g in season_model.games.values() if g["round"] == 1)
    fitted = Model(history["rows"], first, settings)
    rounds, all_rows = [], []
    for rnd in season_model.data.played_rounds():
        if rnd > through:
            break
        baseline = season_model.project(rnd)
        experiment, _ = apply_model(baseline, fitted, season_model)
        before = {p["name"]: p for p in baseline["players"]}
        code_counts = Counter(p.get("code") for p in baseline["players"] if p.get("code"))
        actual = season_model.actuals(rnd)
        boxes = season_model.data.boxes([rnd])[rnd]
        seen_codes, rows = set(), []
        for p in experiment["players"]:
            code = p.get("code")
            q = p["proj"].get(str(rnd))
            if not code or code_counts[code] > 1 or code in seen_codes or not q or q["play"] < 0.5:
                continue
            seen_codes.add(code)
            own = before[p["name"]]["proj"][str(rnd)]
            detail = q.get("independent", {})
            sd = detail.get("fantasy_sd_if_available", p["sd_per_pir"] * q["if_plays"])
            # Moment match the availability mixture for interval auditing.
            variance = q["play"] * sd**2 + q["play"] * (1 - q["play"]) * q["if_plays"] ** 2
            rows.append(
                {
                    "player": p["name"],
                    "player_id": code,
                    "round": rnd,
                    "actual": actual.get(code, 0),
                    "actual_minutes": boxes.get(code, {}).get("minutes", 0),
                    "baseline_ev": own["ev"],
                    "baseline_minutes": own["min"] * own["play"],
                    "independent_ev": q["ev"],
                    "independent_minutes": q["min"] * q["play"],
                    "p10": q["ev"] + norm.ppf(0.1) * np.sqrt(variance),
                    "p90": q["ev"] + norm.ppf(0.9) * np.sqrt(variance),
                }
            )
        rounds.append(
            {
                "round": rnd,
                "n": len(rows),
                "baseline": score_rows(rows, "baseline"),
                "independent": score_rows(rows, "independent"),
                "baseline_tuned_on": rnd in season_model.season.tuned_on,
            }
        )
        all_rows += rows
        fitted.state.observe([r for r in current if r["round"] == rnd])
    return {
        "training": fitted.fit_info,
        "rounds": rounds,
        "n": len(all_rows),
        "baseline": score_rows(all_rows, "baseline"),
        "independent": score_rows(all_rows, "independent"),
        "mean_top40_per_round": {
            name: float(np.mean([r[name]["top"] for r in rounds]))
            for name in ("baseline", "independent")
        },
        "universe": "Same priced players with unique official IDs and baseline P(plays)>=0.5; actual nonparticipants score zero.",
        "note": "Retrospective replay of saved round inputs. Global experiment coefficients use E2024/E2025 only; current-season outcomes update state after each round. Baseline rounds 2/3 were tuning data. This new design has seen the results during development, so these are retrospective checks, not untouched prospective tests. OddsPapi cannot be scored without eligible historical snapshots.",
        "players": all_rows,
    }
