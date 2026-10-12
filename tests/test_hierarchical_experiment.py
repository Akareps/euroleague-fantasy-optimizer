"""Chronology, coherent shooting and independent output are material guarantees."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from elfantasy.data.http import HttpClient
from elfantasy.experiments.hierarchical import COUNTS, Model, Settings, apply_model
from elfantasy.experiments.hierarchical_cli import confirmed_pool
from elfantasy.experiments.hierarchical_validation import historical_validation
from elfantasy.experiments.history import current_records, utc
from elfantasy.rules import GameRules
from elfantasy.season.inputs import PrivateInputs, key


def training_rows():
    rng = np.random.default_rng(72)
    rows = []
    start = datetime(2024, 10, 1, tzinfo=timezone.utc)
    for game in range(15):
        for team, opp, home in (("AAA", "BBB", True), ("BBB", "AAA", False)):
            for p in range(9):
                minutes = float(np.clip(27 - p * 1.9 + rng.normal(0, 3), 1, 38))
                a2 = int(rng.poisson(minutes * 0.19))
                a3 = int(rng.poisson(minutes * 0.13))
                aft = int(rng.poisson(minutes * 0.09))
                m2 = int(rng.binomial(a2, 0.52))
                m3 = int(rng.binomial(a3, 0.37))
                mft = int(rng.binomial(aft, 0.79))
                r = {
                    "season": "E2024",
                    "game_id": f"E2024_{game}",
                    "round": game + 1,
                    "date": (start + timedelta(days=game * 7)).isoformat(),
                    "player_id": f"{team}{p}",
                    "club": team,
                    "opp": opp,
                    "home": home,
                    "pos": "GFC"[p % 3],
                    "minutes": minutes,
                    "regulation_minutes": minutes,
                    "margin": 5 if home else -5,
                    "two_attempted": a2,
                    "two_made": m2,
                    "three_attempted": a3,
                    "three_made": m3,
                    "ft_attempted": aft,
                    "ft_made": mft,
                    "points": 2 * m2 + 3 * m3 + mft,
                }
                for field, rate in zip(
                    COUNTS[3:], (0.19, 0.12, 0.035, 0.025, 0.1, 0.07, 0.02, 0.1), strict=True
                ):
                    r[field] = int(rng.poisson(minutes * rate))
                r["pir"] = (
                    r["points"]
                    + sum(r[f] for f in COUNTS[3:8])
                    - (a2 - m2)
                    - (a3 - m3)
                    - (aft - mft)
                    - sum(r[f] for f in COUNTS[8:])
                )
                rows.append(r)
    return rows


@pytest.fixture(scope="module")
def fitted():
    return Model(training_rows(), "2025-09-01T00:00:00Z", Settings(min_training_rows=50))


def future_row():
    return {
        "season": "E2026",
        "game_id": "E2026_41",
        "round": 5,
        "player_id": "AAA0",
        "date": "2026-10-13T17:45:00Z",
        "club": "AAA",
        "opp": "BBB",
        "home": True,
        "pos": "G",
    }


def test_future_and_exact_cutoff_outcomes_cannot_change_fit():
    rows = training_rows()
    cutoff = rows[-1]["date"]
    before = Model(rows, cutoff, Settings(min_training_rows=50))
    changed = copy.deepcopy(rows)
    for r in changed:
        if utc(r["date"]) >= utc(cutoff):
            r["points"] = r["two_attempted"] = r["pir"] = 100000
            r["minutes"] = r["regulation_minutes"] = 9999
    after = Model(changed, cutoff, Settings(min_training_rows=50))
    np.testing.assert_array_equal(before.minute_coef, after.minute_coef)
    np.testing.assert_array_equal(before.context_coef, after.context_coef)
    np.testing.assert_array_equal(before.selection_coef, after.selection_coef)
    np.testing.assert_array_equal(before.dispersion, after.dispersion)
    assert before.fit_info == after.fit_info


def test_same_day_tip_off_does_not_imply_outcome_is_available():
    rows = training_rows()
    cutoff = utc(rows[-1]["date"]) + timedelta(hours=1)
    fitted = Model(rows, cutoff.isoformat(), Settings(min_training_rows=50))
    assert utc(fitted.fit_info["latest_training_tip"]).date() < cutoff.date()


def test_target_time_prediction_rejects_late_evidence(fitted):
    r = future_row()
    r["date"] = fitted.rows[-1]["date"]
    with pytest.raises(ValueError, match="target game or later"):
        fitted.predict(r)


def test_shooting_and_pir_components_reconcile(fitted):
    p = fitted.predict(future_row())
    c = p["components"]
    a = p["shot_attempts"]
    e = p["shooting_efficiency"]
    assert c["points"] == pytest.approx(
        2 * a["two"] * e["two"] + 3 * a["three"] * e["three"] + a["ft"] * e["ft"]
    )
    assert c["missed_fg"] == pytest.approx(
        a["two"] * (1 - e["two"]) + a["three"] * (1 - e["three"])
    )
    recomposed = (
        c["points"]
        + sum(c[f] for f in COUNTS[3:8])
        - c["missed_fg"]
        - c["missed_ft"]
        - sum(c[f] for f in COUNTS[8:])
    )
    assert p["pir"] == pytest.approx(recomposed)
    assert all(0 < x < 1 for x in e.values())


def test_minutes_uncertainty_enters_score_variance(fitted):
    a = fitted.predict(future_row(), 20, 0)
    b = fitted.predict(future_row(), 20, 100)
    assert a["pir"] == b["pir"]
    assert b["pir_sd"] > a["pir_sd"]
    assert np.linalg.eigvalsh(fitted.correlation).min() > 0
    mean, var = fitted.minute_prediction(future_row())
    assert 0 < mean < 40 and var > 0


def test_opportunity_change_also_changes_misses(fitted):
    r = future_row()
    short = fitted.predict(r, 10, 0)
    long = fitted.predict(r, 20, 0)
    assert long["components"]["points"] == pytest.approx(2 * short["components"]["points"])
    assert long["components"]["missed_fg"] == pytest.approx(2 * short["components"]["missed_fg"])


def projection_fixture():
    players = []
    for i in range(9):
        players.append(
            {
                "name": f"P{i}",
                "code": f"AAA{i}",
                "club": "AAA",
                "pos": "GFC"[i % 3],
                "price": 10.0,
                "base_min": 25.0,
                "sd_per_pir": 0.6,
                "proj": {
                    "5": {
                        "ev": 10.0,
                        "if_plays": 10.0 / 0.97,
                        "min": 25.0,
                        "play": 0.97,
                        "home": True,
                        "opp": "BBB",
                        "win": 0.6,
                        "margin": 4.0,
                        "date": "2026-10-13T19:45:00",
                        "turn": 0,
                    }
                },
            }
        )
    players[-1]["proj"]["5"]["play"] = 0
    players[-1]["proj"]["5"]["ev"] = 0
    return {
        "round": 5,
        "rounds": [5],
        "players": players,
        "coaches": [{"name": "Coach", "price": 8, "proj": {"5": {"ev": 10}}}],
    }


def season_fixture():
    return SimpleNamespace(
        season=SimpleNamespace(code="E2026"),
        rules=GameRules(),
        games={
            "g": {
                "round": 5,
                "utcDate": "2026-10-13T17:45:00Z",
                "local": {"club": {"code": "AAA"}},
                "road": {"club": {"code": "BBB"}},
            }
        },
    )


def test_application_keeps_baseline_prices_availability_coaches_and_out_player(fitted):
    base = projection_fixture()
    saved = copy.deepcopy(base)
    result, audit = apply_model(base, fitted, season_fixture())
    assert base == saved
    assert result["coaches"] == base["coaches"]
    assert result["players"][-1] == base["players"][-1]
    assert audit["updated_players"] == 8
    assert (
        sum(p["proj"]["5"]["min"] * p["proj"]["5"]["play"] for p in result["players"]) <= 200.000001
    )
    for own, new in zip(base["players"], result["players"], strict=True):
        assert own["price"] == new["price"]
        assert own["proj"]["5"]["play"] == new["proj"]["5"]["play"]
        assert own["proj"]["5"]["win"] == new["proj"]["5"]["win"]
    assert result["experiment"]["player_props_used"] is False


def test_ambiguous_identity_keeps_own_model_exactly(fitted):
    base = projection_fixture()
    base["players"][1]["code"] = base["players"][0]["code"]
    result, audit = apply_model(base, fitted, season_fixture())
    assert result["players"][:2] == base["players"][:2]
    assert len(audit["skipped"]) == 2


def test_confirmed_pool_does_not_expose_estimated_purchases():
    base = projection_fixture()
    base["coaches"][0]["club"] = "AAA"
    private = PrivateInputs(["P0"], "Coach", 0.5, {key("AAA", "P1"): 10.0}, {})
    pool = confirmed_pool(base, private)
    assert [p["name"] for p in pool["players"]] == ["P0", "P1"]
    assert pool["coaches"] == base["coaches"]
    assert len(base["players"]) == 9


def test_small_training_set_and_invalid_priors_fail():
    with pytest.raises(ValueError, match="earlier historical"):
        Model(training_rows()[:5], "2026-01-01T00:00:00Z")
    with pytest.raises(ValueError):
        Settings(context_prior_sd=float("nan"))
    with pytest.raises(ValueError, match="timezone"):
        utc("2026-10-13T17:45:00")


def test_historical_round_numbers_do_not_override_time_order():
    train = training_rows()
    test = copy.deepcopy(train[:36])
    for i, r in enumerate(test):
        r["season"] = "E2025"
        r["game_id"] = f"E2025_{i // 18}"
        r["round"] = 7 if i < 18 else 2
        r["date"] = "2025-10-01T17:00:00Z" if i < 18 else "2025-10-08T17:00:00Z"
    result = historical_validation({"rows": train + test}, settings=Settings(min_training_rows=50))
    assert [r["round"] for r in result["rounds"]] == [7, 2]
    assert all(utc(r["evidence_before"]) < utc(r["first_tip"]) for r in result["rounds"])


def test_missing_healthy_player_is_role_evidence_but_injury_is_not(tmp_path):
    data_root = tmp_path / "season"
    prior = data_root / "prior"
    prior.mkdir(parents=True)
    joined = []
    people = []
    boxes = {}
    raw = {"local": {"players": []}, "road": {"players": []}}
    for club, side in (("AAA", "local"), ("BBB", "road")):
        for i in range(10):
            code = club + str(i)
            joined.append({"name": code, "code": code, "club": club, "pos": "G"})
            people.append({"person": {"code": code}, "positionName": "Guard"})
            boxes[code] = {
                "player_id": code,
                "game_id": "E2026_1",
                "round": 1,
                "team_code": club,
                "minutes": 20.0,
                "points": 7.0,
                "pir": 7.0,
                "rebounds": 2.0,
                "assists": 0.0,
                "steals": 0.0,
                "blocks": 0.0,
                "fouls_drawn": 0.0,
                "turnovers": 0.0,
                "blocks_against": 0.0,
                "fouls_committed": 0.0,
                "ft_made": 0.0,
                "ft_attempted": 0.0,
            }
            raw[side]["players"].append(
                {
                    "player": {"person": {"code": code}},
                    "stats": {
                        "fieldGoalsMade2": 2,
                        "fieldGoalsAttempted2": 3,
                        "fieldGoalsMade3": 1,
                        "fieldGoalsAttempted3": 2,
                    },
                }
            )
    joined += [
        {"name": "Healthy bench", "code": "HB", "club": "AAA", "pos": "G"},
        {"name": "Injured missing", "code": "IM", "club": "AAA", "pos": "G"},
    ]
    (prior / "joined.json").write_text(json.dumps(joined))
    (prior / "people_E2026.json").write_text(json.dumps({"data": people}))
    http = HttpClient(data_root / "http", offline=True)
    url = "https://api-live.euroleague.net/v2/competitions/E/seasons/E2026/games/1/stats"
    http._write_cache(http._cache_path(url, None), json.dumps(raw))
    g = {
        "round": 1,
        "played": True,
        "gameCode": 1,
        "identifier": "E2026_1",
        "utcDate": "2026-09-24T17:00:00Z",
        "local": {"club": {"code": "AAA"}, "score": 80},
        "road": {"club": {"code": "BBB"}, "score": 75},
    }
    model = SimpleNamespace(
        data=SimpleNamespace(root=data_root, prior_dir=prior, boxes=lambda rounds: {1: boxes}),
        games={"g": g},
        season=SimpleNamespace(code="E2026"),
        round_inputs=lambda n: SimpleNamespace(
            missed_injured=lambda: {key("AAA", "Injured missing")}
        ),
    )
    records = current_records(model, 2, tmp_path / "history")
    by_id = {r["player_id"]: r for r in records}
    assert by_id["HB"]["inferred_dnp"] and by_id["HB"]["minutes"] == 0
    assert "IM" not in by_id
    assert by_id["AAA0"]["two_attempted"] == 3
