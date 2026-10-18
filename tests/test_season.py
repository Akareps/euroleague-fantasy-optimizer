"""The season workflow: inputs, evidence updates, prices, calibration, planning."""

from __future__ import annotations

import json

import pytest

from elfantasy.config import REPO_ROOT, load_settings
from elfantasy.rules import GameRules
from elfantasy.season.data import SeasonData
from elfantasy.season.inputs import (
    CalibrationSettings,
    EvidenceWeights,
    PrivateInputs,
    RoundInputs,
    key,
    load_round,
    load_season,
    rounds_recorded,
)
from elfantasy.season.model import (
    calibrate,
    coach_points,
    coach_price_change,
    estimate_prices,
    fit_calibration,
    garbage_minutes,
    team_round,
    update_with_evidence,
)
from elfantasy.season.planner import RoundPlanner

SEASON = REPO_ROOT / "seasons" / "2026-27"


@pytest.fixture(scope="module")
def rules():
    return GameRules.from_settings(load_settings())


# ---------------------------------------------------------------- inputs
class TestCommittedInputs:
    def test_the_season_file_loads(self):
        s = load_season(SEASON)
        assert s.code == "E2026" and s.previous_code == "E2025"
        assert len(s.clubs) == len(s.outrights) == 20
        assert set(s.clubs.values()) == set(s.outrights)
        assert s.evidence.k_min_strong == 0.5 and s.evidence.recency == 1.3

    def test_every_round_file_loads(self):
        rounds = rounds_recorded(SEASON)
        assert rounds[:4] == [1, 2, 3, 4]
        for n in rounds:
            r = load_round(SEASON, n)
            assert r.rating_lines(), f"round {n} has no lines to rate the teams"
            assert all(len(p) == 3 for p in r.availability.values())

    def test_names_are_normalised(self):
        r4 = load_round(SEASON, 4)
        assert r4.availability[key("IST", "JAMES, MIKE")] == (0.0, 0.0, 0.0)

    def test_eye_test_applies_from_its_round(self):
        s = load_season(SEASON)
        k = key("PRS", "Allan Dokossi")
        assert s.eye_multiplier(k, 2) == 1.0 and s.eye_multiplier(k, 3) == 0.8


def test_missed_injured_falls_back_to_low_availability():
    r = RoundInputs(
        number=7,
        availability={("AAA", "x"): (0.5, 0.9, 1.0), ("AAA", "y"): (0.95, 1.0, 1.0)},
        moneylines={7: {("AAA", "BBB"): (1.5, 2.6)}, 8: {("CCC", "AAA"): (2.0, 1.8)}},
        injured_out=None,
        minutes_expert={},
    )
    assert r.missed_injured() == {("AAA", "x")}
    assert set(r.rating_lines()) == {("AAA", "BBB"), ("CCC", "AAA")}


# ---------------------------------------------------------------- evidence
def _player(name="X", base_min=20.0, rate=0.5, hist=None, code="1"):
    return {"name": name, "club": "AAA", "pos": "G", "code": code, "base_min": base_min,
            "rate": rate, "hist": hist}  # fmt: skip


def _game(gid, rnd, margin):
    return {"identifier": gid, "round": rnd, "played": True,
            "local": {"score": 80 + margin, "club": {"code": "AAA"}},
            "road": {"score": 80, "club": {"code": "BBB"}}}  # fmt: skip


def _line(gid, minutes, pir, code="1"):
    return {"player_id": code, "game_id": gid, "team_code": "AAA", "minutes": minutes, "pir": pir}


W = EvidenceWeights(0.5, 0.5, 180.0, 180.0, 180.0, recency=1.0)


class TestEvidence:
    def test_one_game_moves_minutes_and_rate(self):
        p = _player()
        update_with_evidence(
            [p], {"g": _game("g", 1, 2)}, {1: {"1": _line("g", 30.0, 20.0)}}, {}, W
        )
        assert p["base_min"] == pytest.approx((0.5 * 20 + 30) / 1.5)
        assert p["rate"] == pytest.approx((180 * 0.5 + 20) / (180 + 30))

    def test_an_injured_absence_is_not_evidence(self):
        p = _player()
        update_with_evidence([p], {"g": _game("g", 1, 2)}, {1: {}}, {1: {key("AAA", "X")}}, W)
        assert p["base_min"] == 20.0

    def test_a_fit_player_left_out_counts_as_zero_minutes(self):
        p = _player()
        update_with_evidence([p], {"g": _game("g", 1, 2)}, {1: {}}, {}, W)
        assert p["base_min"] == pytest.approx(0.5 * 20 / 1.5)

    def test_newer_rounds_count_more(self):
        p = _player()
        games = {"g1": _game("g1", 1, 2), "g2": _game("g2", 2, 2)}
        boxes = {1: {"1": _line("g1", 10.0, 5.0)}, 2: {"1": _line("g2", 30.0, 15.0)}}
        update_with_evidence([p], games, boxes, {}, EvidenceWeights(0.5, 0.5, 180, 180, 180, 2.0))
        assert p["base_min"] == pytest.approx((0.5 * 20 + 10 + 2 * 30) / (0.5 + 1 + 2))

    def test_blowout_minutes_are_corrected(self):
        assert garbage_minutes(9) == 0.0
        assert garbage_minutes(-20) == pytest.approx(0.55 * 11)
        assert garbage_minutes(40) == 9.0
        # A starter's 25 minutes in a 25-point win were really more.
        p = _player()
        update_with_evidence(
            [p], {"g": _game("g", 1, 25)}, {1: {"1": _line("g", 25.0, 10.0)}}, {}, W
        )
        assert p["base_min"] > (0.5 * 20 + 25) / 1.5


def test_absent_players_minutes_go_mostly_to_his_position():
    roster = [
        _player(f"P{i}", base_min=m, code=str(i))
        for i, m in enumerate([30, 28, 26, 24, 22, 20, 18, 14, 10])
    ]
    for q, pos in zip(roster, "GGGFFFCCC", strict=True):
        q["pos"] = pos
    out = team_round(roster, "AAA", 1, 0.0, {key("AAA", "P6"): (0.0, 1.0, 1.0)})
    gain = {q["name"]: out[id(q)][0] - q["base_min"] for q in roster if q["name"] != "P6"}
    assert min(gain["P7"], gain["P8"]) > max(gain[n] for n in ("P0", "P1", "P2"))


# ---------------------------------------------------------------- prices
def test_coach_price_rule_matches_round_1():
    # (margin, price) -> change, as observed after Round 1 (and Obradovic's +25s).
    for margin, price, change in [(-16, 7.0, -0.4), (-1, 9.0, -0.3), (7, 8.0, 0.1), (17, 10.0, 0.3),
                                  (23, 10.3, 0.4), (23, 10.7, 0.4)]:  # fmt: skip
        assert coach_price_change(coach_points(margin), price) == pytest.approx(change)


def test_prices_chain_from_the_last_list(tmp_path, rules):
    season = load_season(SEASON)
    data = SeasonData(tmp_path)
    (tmp_path / "prices").mkdir()
    (tmp_path / "prices" / "round_02.csv").write_text(
        "rank,player,club,pos,price\n1,Xavier Lowe,Real Madrid,G,10.0\n2,Coach Y,Real Madrid,HC,8.0\n",
        encoding="utf-8",
    )
    (tmp_path / "boxes").mkdir()
    for rnd, line in (
        (2, {"player_id": "1", "game_id": "g2", "minutes": 30.0, "pir": 21.0}),
        (3, None),
    ):
        (tmp_path / "boxes" / f"round_{rnd:02d}.json").write_text(
            json.dumps([line] if line else [])
        )
    games = {
        "g2": {"round": 2, "played": True, "local": {"score": 92, "club": {"code": "MAD"}},
               "road": {"score": 80, "club": {"code": "PAR"}}},
        "g3": {"round": 3, "played": True, "local": {"score": 70, "club": {"code": "MAD"}},
               "road": {"score": 75, "club": {"code": "PAR"}}},
    }  # fmt: skip
    p = {"name": "Xavier Lowe", "club": "MAD", "pos": "G", "code": "1", "price": 9.0}
    q = {"name": "Zeke Tate", "club": "MAD", "pos": "F", "code": "2", "price": 5.0}
    c = {"name": "Coach Y", "club": "MAD", "pos": "HC", "price": 9.0}
    private = PrivateInputs([], None, None, {key("MAD", "Zeke Tate"): 6.3}, {})
    estimate_prices([p, q], [c], 4, data, season, rules, games, private)
    assert p["price"] == pytest.approx(10.4)  # 10.0 -> +0.5 (21 PIR) -> -0.1 (did not play)
    assert c["price"] == pytest.approx(8.0)  # 8.0 -> +0.3 (won by 12) -> -0.3 (lost by 5)
    assert q["price"] == 6.3  # the app's price wins


# ---------------------------------------------------------------- calibration
def test_calibration_shrinks_small_samples():
    settings = CalibrationSettings(prior_sd=1.5, cv=0.634, tiers=((0.0, 40.0),))
    few = fit_calibration([("new to EL", 10.0, 15.0)] * 2, settings)
    many = fit_calibration([("new to EL", 10.0, 15.0)] * 400, settings)
    assert 0 < few["new to EL"][0][1] < many["new to EL"][0][1] < 5.0
    assert calibrate(many, "new to EL", 10.0) == pytest.approx(10.0 + many["new to EL"][0][1])
    assert calibrate({"same club": [(5.0, -9.0, -9.0, 50)]}, "same club", 3.0) == 0.0


# ---------------------------------------------------------------- planning
def _projection():
    """Three clubs a day over two days, 7 players each, three rounds."""
    players, coaches = [], []
    clubs = {"AAA": ("BBB", 0), "BBB": ("AAA", 0), "CCC": ("DDD", 1), "DDD": ("CCC", 1)}
    for ci, (club, (opp, turn)) in enumerate(clubs.items()):
        for k, pos in enumerate("GGFFFCC"):
            price = round(4.0 + 2.0 * k + 0.3 * ci, 1)
            ev = 1.2 * price
            proj = {
                str(r): {"ev": ev, "if_plays": ev / 0.97, "play": 0.97, "min": 10 + 2 * price, "win": 0.5,
                         "opp": opp, "turn": turn, "date": f"2026-10-0{1 + turn}T20:00:00"}
                for r in (1, 2, 3)
            }  # fmt: skip
            players.append({"name": f"{club}{pos}{k}", "club": club, "pos": pos, "price": price,
                            "sd_per_pir": 0.6, "proj": proj})  # fmt: skip
        coaches.append({"name": f"Coach {club}", "club": club, "price": 5.0 + ci,
                        "proj": {str(r): {"ev": 2.0 * ci} for r in (1, 2, 3)}})  # fmt: skip
    return {"round": 1, "rounds": [1, 2, 3], "players": players, "coaches": coaches}


def test_planner_compares_named_rosters_on_the_same_simulations(rules):
    proj = _projection()
    squad = [
        "AAAG0",
        "AAAG1",
        "BBBG0",
        "BBBG1",
        "AAAF2",
        "AAAF3",
        "CCCF2",
        "CCCF3",
        "AAAC5",
        "CCCC5",
    ]
    other = [*squad[:-1], "DDDC5"]
    private = PrivateInputs(squad, "Coach AAA", 3.0, {}, {"swap a centre": (other, "Coach AAA")})
    planner = RoundPlanner(proj, rules, private, log=lambda m: None)
    assert planner.max_transfers == rules.transfers_per_round
    assert planner.title().startswith("Round 1 (Thu 1 Oct / Fri 2 Oct")
    rows = planner.compare(private.plans, seeds=(1,), n=400)
    assert {r["plan"] for r in rows} == {"swap a centre", "hold"}
    assert rows == sorted(rows, key=lambda r: -r["objective"])


def test_planner_can_buy_a_player_who_is_out_this_round(rules):
    """The MILP may pick someone unavailable now for his later rounds; the
    simulator must know him too (it used to raise KeyError)."""

    proj = _projection()
    star = next(p for p in proj["players"] if p["name"] == "AAAG0")
    star["price"] = 4.0
    for r in ("2", "3"):
        star["proj"][r]["ev"] = 60.0
    star["proj"]["1"].update(ev=0.0, play=0.0)
    planner = RoundPlanner(proj, rules, None, log=lambda m: None)
    _, milp = planner.credit_value()
    assert "AAAG0" in {p.player_id for p in milp.roster}
    assert {p.player_id for p in planner.lp} <= {p.player_id for p in planner.sp}


# ---------------------------------------------------------------- end to end
def make_season(tmp_path):
    """A four-club season: two rounds played, Round 3 to plan (Thu/Fri)."""

    import yaml

    clubs = ["AAA", "BBB", "CCC", "DDD"]
    sd, dd = tmp_path / "season", tmp_path / "data"
    root = dd / "cache" / "seasons" / "test"
    for d in (sd / "rounds", sd / "private", root / "prior", root / "boxes"):
        d.mkdir(parents=True)
    joined, stats = [], []
    for ci, club in enumerate(clubs):
        joined.append({"name": f"Coach {club}", "club": club, "pos": "HC", "price": 5.0 + ci})
        for k, pos in enumerate("GGGFFFCCC"):
            code = f"P{club}{k}"
            price = round(4.0 + 1.1 * k + 0.2 * ci, 1)
            name = f"{club} {pos} {'abcdefghi'[k]}x"
            joined.append({"name": name, "club": club, "pos": pos, "price": price, "code": code})
            if k >= 1:  # every position needs regulars with 300+ minutes
                stats.append({"player": {"code": code, "team": {"code": club}}, "gamesPlayed": 20,
                              "minutesPlayed": 20 * (14 + 2 * k), "pir": 20 * 1.2 * price,
                              "pointsScored": 20 * price})  # fmt: skip
    (root / "prior" / "joined.json").write_text(json.dumps(joined))
    (root / "prior" / "stats_E2025_acc.json").write_text(json.dumps({"players": stats}))
    fixtures = {1: [("AAA", "BBB"), ("CCC", "DDD")], 2: [("AAA", "CCC"), ("BBB", "DDD")],
                3: [("AAA", "DDD"), ("BBB", "CCC")], 4: [("DDD", "BBB"), ("CCC", "AAA")]}  # fmt: skip
    games = []
    for rnd, pairs in fixtures.items():
        for gi, (home, away) in enumerate(pairs):
            played = rnd <= 2
            games.append({"identifier": f"G{rnd}{gi}", "round": rnd, "played": played,
                          "localDate": f"2026-10-{rnd * 7 + gi - 6:02d}T20:00:00",
                          "local": {"club": {"code": home}, "score": 85 + 3 * gi if played else 0},
                          "road": {"club": {"code": away}, "score": 80 if played else 0}})  # fmt: skip
    (root / "games.json").write_text(json.dumps({"data": games}))
    for rnd in (1, 2):
        lines = []
        for g in (g for g in games if g["round"] == rnd):
            for side in ("local", "road"):
                club = g[side]["club"]["code"]
                for p in (p for p in joined if p["club"] == club and p["pos"] != "HC"):
                    k = int(p["code"][-1])
                    lines.append({"player_id": p["code"], "game_id": g["identifier"], "team_code": club,
                                  "minutes": 8.0 + 2.5 * k, "pir": round(1.1 * p["price"] + rnd)})  # fmt: skip
        (root / "boxes" / f"round_{rnd:02d}.json").write_text(json.dumps(lines))
    season = {"season": "E2026", "name": "test", "clubs": {c: c for c in clubs},
              "outrights": {c: 3.0 + 3 * i for i, c in enumerate(clubs)},
              "model": {"horizon": 3, "gamma": 0.4, "tuned_on": [2],
                        "evidence": {"k_min_strong": 0.5, "k_min_weak": 0.5, "n_rate_strong": 180,
                                     "n_rate_hist": 180, "n_rate_new": 180, "recency": 1.3}}}  # fmt: skip
    (sd / "season.yaml").write_text(yaml.safe_dump(season))
    lines = {1: {"AAA-BBB": [1.6, 2.4], "CCC-DDD": [1.9, 1.9]}, 2: {"AAA-CCC": [1.5, 2.6]},
             3: {"AAA-DDD": [1.4, 3.0], "BBB-CCC": [2.0, 1.8]}, 4: {"DDD-BBB": [1.8, 2.0]}}  # fmt: skip
    for rnd in (1, 2, 3):
        doc = {"round": rnd, "availability": [],
               "moneylines": {r: lines[r] for r in (rnd, rnd + 1) if r in lines}}  # fmt: skip
        if rnd < 3:
            doc["injured_out"] = []
        else:
            doc["availability"] = [{"club": "BBB", "player": "BBB C hx", "p": [0.0, 0.6, 1.0]}]
        (sd / "rounds" / f"r{rnd:02d}.yaml").write_text(yaml.safe_dump(doc))
    names = {
        (p["club"], p["pos"], int(p["code"][-1])): p["name"] for p in joined if p["pos"] != "HC"
    }
    squad = [names[("AAA", "G", 0)], names[("BBB", "G", 1)], names[("CCC", "G", 0)], names[("DDD", "G", 1)],
             names[("AAA", "F", 3)], names[("BBB", "F", 4)], names[("CCC", "F", 3)], names[("DDD", "F", 4)],
             names[("AAA", "C", 6)], names[("CCC", "C", 7)]]  # fmt: skip
    private = {"round": 3, "squad": squad, "coach": "Coach AAA", "bank": 2.0,
               "prices": [{"club": "AAA", "player": squad[0], "price": 4.5}],
               "plans": {"other centre": {"players": [*squad[:-1], names[("DDD", "C", 6)]],
                                          "coach": "Coach AAA"}}}  # fmt: skip
    (sd / "private" / "r03.yaml").write_text(yaml.safe_dump(private))
    return sd, dd


class TestEndToEnd:
    def test_the_model_runs_on_a_whole_season(self, tmp_path, rules):
        from elfantasy.season.backtest import run_backtest
        from elfantasy.season.model import SeasonModel

        sd, dd = make_season(tmp_path)
        season = load_season(sd)
        model = SeasonModel(season, SeasonData.for_season(dd, season), rules)
        result = model.project(3, None)
        hurt = next(p for p in result["players"] if p["name"] == "BBB C hx")
        assert hurt["proj"]["3"]["play"] == 0.0 and hurt["proj"]["4"]["play"] == 0.6
        assert {p["proj"]["3"]["turn"] for p in result["players"]} == {0, 1}  # Thu and Fri
        first = next(p for p in result["players"] if p["name"] == "AAA G ax")
        assert first["price"] != 4.0  # two rounds of games moved it
        scores = run_backtest(model)
        assert [s.round for s in scores] == [1, 2] and scores[1].tuned_on and scores[1].price

    def test_the_commands(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from elfantasy.cli import app

        sd, dd = make_season(tmp_path)
        monkeypatch.setenv("ELFANTASY_DATA_DIR", str(dd))
        runner = CliRunner()
        for args, expect in (
            (["season", "project", "--round", "3", "--top", "5"], "Round 3: top 5"),
            (["season", "backtest"], "Chronological backtest"),
            (["season", "compare", "--round", "3"], "other centre"),
        ):
            out = runner.invoke(app, [*args, "--season-dir", str(sd)], env={"COLUMNS": "200"})
            assert out.exit_code == 0, out.output
            assert expect in out.output

    def test_missing_downloads_are_explained(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from elfantasy.cli import app

        sd, dd = make_season(tmp_path)
        (dd / "cache" / "seasons" / "test" / "boxes" / "round_02.json").unlink()
        monkeypatch.setenv("ELFANTASY_DATA_DIR", str(dd))
        out = CliRunner().invoke(
            app,
            ["season", "project", "--round", "3", "--season-dir", str(sd)],
            env={"COLUMNS": "300"},
        )
        assert out.exit_code == 1
        assert "season fetch" in out.output


def test_the_planner_stops_at_a_free_transfer_window(rules):
    """After an unlimited-transfer window the squad can be rebuilt, so holding a
    player beyond it is worth nothing to today's decision."""

    import dataclasses

    free_after_1 = dataclasses.replace(rules, unlimited_after_rounds=(1,))
    planner = RoundPlanner(_projection(), free_after_1, None, log=lambda m: None)
    assert planner.rounds == [1]
    assert all(p.future_value == 0.0 for p in planner.sp)
    assert RoundPlanner(_projection(), rules, None, log=lambda m: None).rounds == [1, 2, 3]


def test_a_partial_name_match_cannot_take_another_players_id(tmp_path):
    """Vitto Brown missing from the roster must not inherit Anthony Brown's ID."""

    season = load_season(SEASON)
    data = SeasonData(tmp_path)
    data.prior_dir.mkdir(parents=True)
    (data.prior_dir / "prices_round_01.csv").write_text(
        "rank,player,club,pos,price\n1,Anthony Brown,Besiktas,F,9.7\n2,Vitto Brown,Besiktas,F,7.0\n"
        "3,Kevarrius Hayes,Partizan,C,8.0\n",
        encoding="utf-8",
    )
    people = [
        {"person": {"code": "009025", "name": "BROWN, ANTHONY"}, "club": {"code": "BES"}},
        {"person": {"code": "007777", "name": "HAYES, KEVARRIUS JAMAL"}, "club": {"code": "PAR"}},
    ]
    (data.prior_dir / "people_E2026.json").write_text(json.dumps({"data": people}))
    (data.prior_dir / "stats_E2025_acc.json").write_text(json.dumps({"players": []}))
    data.join_prior(season)
    codes = {r["name"]: r["code"] for r in json.loads((data.prior_dir / "joined.json").read_text())}
    assert codes == {"Anthony Brown": "009025", "Vitto Brown": None, "Kevarrius Hayes": "007777"}


def test_a_game_played_counts_even_if_he_was_listed_injured():
    p = _player()
    injured = {1: {key("AAA", "X")}}
    update_with_evidence(
        [p], {"g": _game("g", 1, 2)}, {1: {"1": _line("g", 30.0, 20.0)}}, injured, W
    )
    assert p["base_min"] == pytest.approx((0.5 * 20 + 30) / 1.5)


def test_prices_chain_from_the_latest_app_price(tmp_path, rules):
    """An app price recorded in an earlier round beats the older public list."""

    season = load_season(SEASON)
    data = SeasonData(tmp_path)
    (tmp_path / "prices").mkdir()
    (tmp_path / "prices" / "round_02.csv").write_text(
        "rank,player,club,pos,price\n1,Xavier Lowe,Real Madrid,G,10.0\n", encoding="utf-8"
    )
    (tmp_path / "boxes").mkdir()
    (tmp_path / "boxes" / "round_03.json").write_text(
        json.dumps([{"player_id": "1", "game_id": "g3", "minutes": 30.0, "pir": 21.0}])
    )
    p = {"name": "Xavier Lowe", "club": "MAD", "pos": "G", "code": "1", "price": 9.0}
    known = {3: {key("MAD", "Xavier Lowe"): 12.0}}
    estimate_prices([p], [], 4, data, season, rules, {}, None, known)
    assert p["price"] == pytest.approx(12.4)  # 12.0 in the app before Round 3, +0.4 for 21 PIR


def test_team_points_shrink_towards_the_league():
    from elfantasy.season.model import points_ratio, team_points

    games = {"g": _game("g", 1, 20)}  # AAA 100 - 80 BBB
    teams, league = team_points(games, 2, prior_games=3.0)
    assert league == 90.0
    assert teams["AAA"] == pytest.approx(((100 + 3 * 90) / 4, (80 + 3 * 90) / 4))
    # AAA scores well and BBB concedes a lot: AAA expects more than a league game.
    assert points_ratio("AAA", "BBB", True, 5.0, {}, teams, league) > 1.0
    # A market total overrides the averages: half the total plus half the margin.
    assert points_ratio(
        "AAA", "BBB", True, 6.0, {("AAA", "BBB"): 160.0}, teams, league
    ) == pytest.approx(83 / 90)
    assert points_ratio("AAA", "BBB", True, 0.0, {}, {}, None) == 1.0  # no games yet
