"""Market identity, time cutoffs and exact own-model fallback are essential here."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import requests
from test_season import make_season
from typer.testing import CliRunner

from elfantasy.cli import app
from elfantasy.config import load_settings
from elfantasy.experiments.oddspapi import (
    EXPERIMENT,
    OddsPapiClient,
    OddsPapiError,
    fetch_snapshot,
    parse_board,
    read_api_key,
    utc,
)
from elfantasy.experiments.prop_blend import (
    ExperimentSettings,
    apply_snapshot,
    component_rates,
    constrained_components,
    match_fixtures,
    projection_cutoff,
)
from elfantasy.rules import GameRules
from elfantasy.season.data import SeasonData
from elfantasy.season.inputs import load_season
from elfantasy.season.model import SeasonModel

STAMP = "2026-10-15T10:00:00Z"


def definition(mid="1", components="Points", line=15.5, period="result"):
    return {
        "marketId": mid,
        "marketName": f"Over Under Player {components}",
        "sportId": 11,
        "playerProp": True,
        "period": period,
        "handicap": line,
        "outcomes": [
            {"outcomeId": "O", "outcomeName": "Over"},
            {"outcomeId": "U", "outcomeName": "Under"},
        ],
    }


def selection(name="AAA F dx", price=1.9, active=True, stamp=STAMP):
    return {"playerName": name, "price": price, "active": active, "changedAt": stamp}


def board(name="AAA F dx", book="one", mid="1"):
    return {
        "fixtureId": "F1",
        "bookmakerOdds": {
            book: {
                "bookmakerIsActive": True,
                "markets": {
                    mid: {
                        "marketActive": True,
                        "outcomes": {
                            "O": {"players": {"provider-id": selection(name)}},
                            "U": {"players": {"provider-id": selection(name)}},
                        },
                    }
                },
            }
        },
    }


def test_bookmakers_are_never_cross_paired():
    b = board()
    second = copy.deepcopy(b["bookmakerOdds"]["one"])
    b["bookmakerOdds"]["one"]["markets"]["1"]["outcomes"].pop("U")
    second["markets"]["1"]["outcomes"].pop("O")
    b["bookmakerOdds"]["two"] = second
    quotes, counts = parse_board(b, [definition()], STAMP)
    assert quotes == [] and counts["unpaired_markets"] == 2


@pytest.mark.parametrize("change", ["quarter", "inactive", "suspended", "integer", "invalid"])
def test_ineligible_markets_cannot_change_a_forecast(change):
    b, d = board(), definition()
    if change == "quarter":
        d["period"] = "p1"
    elif change == "inactive":
        b["bookmakerOdds"]["one"]["markets"]["1"]["outcomes"]["U"]["players"]["provider-id"][
            "active"
        ] = False
    elif change == "suspended":
        b["bookmakerOdds"]["one"]["suspended"] = True
    elif change == "integer":
        d["handicap"] = 15
    else:
        b["bookmakerOdds"]["one"]["markets"]["1"]["outcomes"]["O"]["players"]["provider-id"][
            "price"
        ] = float("nan")
    assert parse_board(b, [d], STAMP)[0] == []


def test_combo_and_milestone_catalogue_semantics():
    q, _ = parse_board(board(), [definition(components="Points + Assists + Rebounds")], STAMP)
    assert q[0].components == ("points", "rebounds", "assists")
    d = definition(components="Points 10+", line=0)
    d["outcomes"][0]["outcomeName"] = "Yes"
    d["outcomes"][1]["outcomeName"] = "No"
    q, _ = parse_board(board(), [d], STAMP)
    assert q[0].line == 9.5
    assert not parse_board(board(), [definition(components="Points + Steals")], STAMP)[0]


class FakeSession:
    def __init__(self, bodies=(), status=200, error=None):
        self.bodies = iter(bodies)
        self.status = status
        self.error = error
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        body = next(self.bodies, None)
        return type("Response", (), {"status_code": self.status, "json": lambda _: body})()


def test_client_errors_never_print_credentials():
    secret = "private-key-for-test"
    session = FakeSession(error=requests.ConnectionError(f"https://api/?apiKey={secret}"))
    with pytest.raises(OddsPapiError, match="connection failed") as error:
        OddsPapiClient(secret, session, 0).get("markets")
    assert secret not in str(error.value)
    session = FakeSession(status=401)
    with pytest.raises(OddsPapiError, match="HTTP 401"):
        OddsPapiClient(secret, session, 0).get("markets")


def test_local_key_not_required_for_offline_projection(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDSPAPI_API_KEY", raising=False)
    with pytest.raises(OddsPapiError, match="API_KEY"):
        read_api_key()
    p = tmp_path / "key"
    p.write_text("test-key\n", encoding="utf-8")
    assert read_api_key(p) == "test-key"


@pytest.fixture()
def season_model(tmp_path):
    sd, dd = make_season(tmp_path)
    data = SeasonData.for_season(dd, load_season(sd))
    games = json.loads(data.games_file.read_text())
    for g in games["data"]:
        g["utcDate"] = g["localDate"] + "Z"
    data.games_file.write_text(json.dumps(games))
    return SeasonModel(load_season(sd), data, GameRules.from_settings(load_settings()))


def snapshot(model, name="AAA F dx"):
    game = next(
        g for g in model.games.values() if g["round"] == 3 and g["local"]["club"]["code"] == "AAA"
    )
    return {
        "experiment": EXPERIMENT,
        "schema_version": 1,
        "round": 3,
        "captured_at": STAMP,
        "catalogue": [definition(line=19.5)],
        "errors": [],
        "boards": [board(name)],
        "fixtures": [
            {
                "fixtureId": "F1",
                "participant1Name": "AAA",
                "participant2Name": "DDD",
                "startTime": game["utcDate"],
            }
        ],
    }


def test_fetch_is_bounded_and_archives_catalogue(season_model):
    games = [g for g in season_model.games.values() if g["round"] == 3]
    fixture = snapshot(season_model)["fixtures"][0] | {"sportId": 11, "tournamentId": 138}
    session = FakeSession([[definition()], [fixture], board()])
    data = fetch_snapshot(OddsPapiClient("key", session, 0), games, max_fixtures=1)
    assert len(session.calls) == 3
    assert data["boards"][0]["fixtureId"] == "F1"
    assert data["catalogue"][0]["handicap"] == 15.5
    assert "apiKey" not in json.dumps(data)


def test_fixture_matching_requires_both_teams_and_time(season_model):
    s = snapshot(season_model)
    assert match_fixtures(s, season_model, ExperimentSettings())[0]["F1"]["identifier"] == "G30"
    s["fixtures"][0]["participant2Name"] = "BBB"
    assert match_fixtures(s, season_model, ExperimentSettings())[0] == {}


def test_player_aliases_are_explicit_and_club_scoped(season_model):
    baseline = season_model.project(3)
    s = snapshot(season_model, "Bookmaker nickname")
    settings = ExperimentSettings(player_aliases={"AAA": {"Bookmaker nickname": "AAA F dx"}})
    _, audit = apply_snapshot(baseline, s, season_model, settings, utc(STAMP))
    assert audit["updated_players"] == 1
    settings = ExperimentSettings(player_aliases={"BBB": {"Bookmaker nickname": "AAA F dx"}})
    result, audit = apply_snapshot(baseline, s, season_model, settings, utc(STAMP))
    assert result == baseline and audit["updated_players"] == 0


def test_no_data_and_late_data_preserve_every_projection(season_model):
    baseline = season_model.project(3)
    before = copy.deepcopy(baseline)
    cutoff = utc("2026-10-15T12:00:00Z")
    experiment, audit = apply_snapshot(baseline, None, season_model, ExperimentSettings(), cutoff)
    assert experiment == baseline and experiment is not baseline
    assert audit["status"] == "no_snapshot"
    s = snapshot(season_model)
    s["captured_at"] = "2026-10-16T10:00:00Z"
    experiment, audit = apply_snapshot(baseline, s, season_model, ExperimentSettings(), cutoff)
    assert experiment == baseline == before
    assert audit["status"] == "snapshot_after_cutoff" and audit["updated_players"] == 0


def test_props_change_only_matched_current_projection_and_keep_dnp_probability(season_model):
    baseline = season_model.project(3)
    p = next(p for p in baseline["players"] if p["name"] == "AAA F dx")
    p["proj"]["3"]["play"] = 0.5
    p["proj"]["3"]["ev"] = 0.5 * p["proj"]["3"]["if_plays"]
    old = copy.deepcopy(baseline)
    experiment, audit = apply_snapshot(
        baseline,
        snapshot(season_model),
        season_model,
        ExperimentSettings(),
        utc("2026-10-15T12:00:00Z"),
    )
    assert audit["updated_players"] == 1
    q = next(p for p in experiment["players"] if p["name"] == "AAA F dx")
    assert q["proj"]["3"]["ev"] != p["proj"]["3"]["ev"]
    assert q["proj"]["3"]["play"] == 0.5
    assert q["proj"]["3"]["ev"] == pytest.approx(0.5 * q["proj"]["3"]["if_plays"])
    assert q["proj"]["4"] == p["proj"]["4"]
    assert experiment["coaches"] == baseline["coaches"]
    assert baseline == old
    assert abs(audit["changes"][0]["pir_delta"]) <= 0.35 * 6
    for original, changed in zip(baseline["players"], experiment["players"], strict=True):
        if original["name"] != q["name"]:
            assert original == changed


def test_stale_snapshot_and_out_player_fall_back(season_model):
    baseline = season_model.project(3)
    stale = snapshot(season_model)
    stale["captured_at"] = "2026-10-10T10:00:00Z"
    stale["boards"] = [board()]
    for o in stale["boards"][0]["bookmakerOdds"]["one"]["markets"]["1"]["outcomes"].values():
        o["players"]["provider-id"]["changedAt"] = stale["captured_at"]
    result, audit = apply_snapshot(baseline, stale, season_model, ExperimentSettings(), utc(STAMP))
    assert result == baseline and audit["coverage"]["stale_quotes"] == 1
    hurt = snapshot(season_model, name="BBB C hx")
    hurt["fixtures"][0].update(
        participant1Name="BBB", participant2Name="CCC", startTime="2026-10-16T20:00:00Z"
    )
    result, audit = apply_snapshot(baseline, hurt, season_model, ExperimentSettings(), utc(STAMP))
    assert result == baseline and audit["updated_players"] == 0


def test_unchanged_active_price_is_fresh_when_observed_today(season_model):
    s = snapshot(season_model)
    for o in s["boards"][0]["bookmakerOdds"]["one"]["markets"]["1"]["outcomes"].values():
        o["players"]["provider-id"]["changedAt"] = "2026-10-01T10:00:00Z"
    _, audit = apply_snapshot(
        season_model.project(3), s, season_model, ExperimentSettings(), utc(STAMP)
    )
    assert audit["updated_players"] == 1


def test_ambiguous_baseline_player_codes_are_not_given_market_data(season_model):
    baseline = season_model.project(3)
    p = next(p for p in baseline["players"] if p["name"] == "AAA F dx")
    duplicate = copy.deepcopy(p)
    duplicate["name"] = "Different Brown"
    baseline["players"].append(duplicate)
    result, audit = apply_snapshot(
        baseline, snapshot(season_model), season_model, ExperimentSettings(), utc(STAMP)
    )
    assert result == baseline
    assert audit["coverage"]["ambiguous_player_code_quotes"] == 1


def test_duplicate_provider_prices_do_not_multiply_influence(season_model):
    baseline = season_model.project(3)
    s = snapshot(season_model)
    s["boards"][0]["bookmakerOdds"]["copied-book"] = copy.deepcopy(
        s["boards"][0]["bookmakerOdds"]["one"]
    )
    result, audit = apply_snapshot(baseline, s, season_model, ExperimentSettings(), utc(STAMP))
    assert audit["updated_players"] == 1
    assert len(audit["changes"][0]["constraints"][0]["books"]) == 1


def test_overlapping_pra_is_not_counted_twice():
    own = np.array([10.0, 5.0, 3.0])
    individual = {
        (c,): {"mean": mean, "books": ["one"]}
        for c, mean in zip(("points", "rebounds", "assists"), (12, 6, 4), strict=True)
    }
    individual[("points", "rebounds", "assists")] = {"mean": 30, "books": ["one"]}
    means, used = constrained_components(own, individual)
    assert means == pytest.approx([12, 6, 4]) and len(used) == 3
    means, used = constrained_components(
        own, {("points", "rebounds", "assists"): {"mean": 24, "books": ["one"]}}
    )
    assert means.sum() == pytest.approx(24)
    assert means / means.sum() == pytest.approx(own / own.sum())


def test_conflicting_combined_market_does_not_make_negative_components():
    means, used = constrained_components(
        np.array([10.0, 5.0, 3.0]),
        {
            ("points",): {"mean": 30, "books": ["one"]},
            ("points", "rebounds"): {"mean": 25, "books": ["one"]},
        },
    )
    assert means == pytest.approx([30, 5, 3])
    assert len(used) == 1


def test_pair_requires_consistent_player_name():
    b = board()
    b["bookmakerOdds"]["one"]["markets"]["1"]["outcomes"]["U"]["players"]["provider-id"][
        "playerName"
    ] = "Other player"
    quotes, counts = parse_board(b, [definition()], STAMP)
    assert not quotes and counts["inconsistent_player_names"] == 1


def test_existing_planner_accepts_market_adjusted_projections(season_model):
    from elfantasy.season.inputs import load_private
    from elfantasy.season.planner import RoundPlanner

    private = load_private(season_model.season.root, 3)
    baseline = season_model.project(3, private)
    experiment, audit = apply_snapshot(
        baseline, snapshot(season_model), season_model, ExperimentSettings(), utc(STAMP)
    )
    assert audit["updated_players"] == 1
    planner = RoundPlanner(experiment, season_model.rules, private, log=lambda _: None)
    rows = planner.compare(private.plans, seeds=(1,), n=400)
    assert {r["plan"] for r in rows} == {"hold", "other centre"}
    assert all(np.isfinite(r["first_round"]) for r in rows)


def test_component_forecast_cannot_see_target_box_scores(season_model):
    baseline = season_model.project(3)
    first = component_rates(season_model, 3, baseline["players"], 120)
    season_model.data.box_file(3).write_text(
        json.dumps(
            [{"player_id": "PAAA3", "minutes": 40, "points": 200, "rebounds": 100, "assists": 100}]
        )
    )
    second = component_rates(season_model, 3, baseline["players"], 120)
    assert all(np.array_equal(first[k], second[k]) for k in first)


def test_cutoff_cannot_move_past_transfer_deadline(season_model):
    with pytest.raises(ValueError, match="transfer deadline"):
        projection_cutoff(season_model, 3, "2026-10-16T23:00:00Z")
    with pytest.raises(ValueError, match="timezone"):
        utc("2026-10-15T12:00:00")
    with pytest.raises(ValueError, match="market_weight"):
        replace(ExperimentSettings(), market_weight=2)


def test_experiment_cli_preserves_baseline_files(season_model, monkeypatch):
    data_dir = season_model.data.root.parents[2]
    monkeypatch.setenv("ELFANTASY_DATA_DIR", str(data_dir))
    baseline_file = season_model.data.projections_file(3)
    baseline_file.write_text("existing-baseline")
    response = CliRunner().invoke(
        app,
        [
            "experiment",
            "oddspapi",
            "project",
            "--round",
            "3",
            "--season-dir",
            str(season_model.season.root),
        ],
    )
    assert response.exit_code == 0, response.output
    assert "0 players adjusted" in response.output
    assert baseline_file.read_text() == "existing-baseline"
    runs = list((season_model.data.root / "experiments" / EXPERIMENT / "round_03").glob("*"))
    assert len(runs) == 1
    saved = json.loads((runs[0] / "projection.json").read_text())
    assert saved["experiment"]["name"] == EXPERIMENT
    assert (runs[0] / "baseline.json").is_file()


def test_saved_coverage_can_be_reaudited_without_a_key_or_network(
    season_model, monkeypatch, tmp_path
):
    monkeypatch.setenv("ELFANTASY_DATA_DIR", str(season_model.data.root.parents[2]))
    monkeypatch.delenv("ODDSPAPI_API_KEY", raising=False)
    s = snapshot(season_model)
    s["fixture_limit"] = 10
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(s))

    def no_network(*args, **kwargs):
        raise AssertionError("An offline re-audit must not make network requests")

    monkeypatch.setattr(OddsPapiClient, "get", no_network)
    response = CliRunner().invoke(
        app,
        [
            "experiment",
            "oddspapi",
            "coverage",
            "--round",
            "3",
            "--snapshot",
            str(path),
            "--as-of",
            STAMP,
            "--season-dir",
            str(season_model.season.root),
        ],
    )
    assert response.exit_code == 0, response.output
    assert "1 matched players with usable quotes" in response.output


def test_evaluate_without_quotes_is_exactly_the_same_forecast(season_model, monkeypatch):
    monkeypatch.setenv("ELFANTASY_DATA_DIR", str(season_model.data.root.parents[2]))
    out = CliRunner().invoke(
        app,
        [
            "experiment",
            "oddspapi",
            "evaluate",
            "--round",
            "2",
            "--season-dir",
            str(season_model.season.root),
        ],
    )
    assert out.exit_code == 0, out.output
    paths = list(
        (season_model.data.root / "experiments" / EXPERIMENT / "round_02").glob("*/evaluation.json")
    )
    result = json.loads(paths[0].read_text())
    assert result["own_model"] == result["experiment_model"]
    assert result["updated_players"] == 0
