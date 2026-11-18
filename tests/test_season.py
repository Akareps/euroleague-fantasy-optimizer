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
        assert scores[1].opp_adjusted and scores[0].opp_adjusted is None

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
    # 12.0 in the app before Round 3, +0.3 for 21 PIR without a win: (21 - 13.2) / 25
    assert p["price"] == pytest.approx(12.3)


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


# ---------------------------------------------------------------- backtest candidates
def test_opponent_adjustment_is_shrunk_towards_zero():
    from elfantasy.season.backtest import opponent_adjustment

    beat = {"opp": "AAA", "pos": "C", "actual": 14.0, "forecast": 10.0}
    missed = {"opp": "AAA", "pos": "G", "actual": 8.0, "forecast": 10.0}
    adj = opponent_adjustment([beat, beat, beat, missed], prior_games=3)
    assert adj[("AAA", "C")] == pytest.approx(12.0 / 6)
    assert adj[("AAA", "G")] == pytest.approx(-2.0 / 4)


def test_rest_days_since_the_previous_round():
    from elfantasy.season.backtest import rest_days

    rows = {
        7: [{"club": "AAA", "date": "2026-10-23T20:00:00"}, {"club": "BBB", "date": "2026-10-22"}],
        8: [{"club": "AAA", "date": "2026-10-27T20:00:00"}, {"club": "CCC", "date": "2026-10-28"}],
    }
    assert rest_days(rows, 8) == {"AAA": 4}


def test_credits_count_for_the_rest_of_the_season(rules):
    """Price changes are valued past the horizon; the bank is not (spent, the
    credits stay in the squad)."""

    proj = _projection()
    planner = RoundPlanner(proj, rules, None, credit_per_round=1.2, credit_weight=0.15,
                           last_round=38, log=lambda m: None)  # fmt: skip
    assert planner.long_run_credit() == pytest.approx(1.2 * (38 - planner.rounds[-1]) * 0.15)
    sim = planner._simulator(1.0, n=200, seed=1)
    assert sim.credit_value == pytest.approx(1.0 + planner.long_run_credit())
    assert sim.bank_value == pytest.approx(0.35)
    assert RoundPlanner(proj, rules, None, log=lambda m: None).long_run_credit() == 0.0
    season = load_season(SEASON)
    assert (season.credit_per_round, season.credit_weight, season.last_round) == (1.2, 0.15, 38)


# ---------------------------------------------------------------- position-and-price correction
def test_position_tiers():
    from elfantasy.season.model import position_tier

    assert position_tier("F", 8.9) == ("F", 0)
    assert position_tier("F", 9.0) == ("F", 1)
    assert position_tier("C", 13.0) == ("C", 2)


def test_position_correction_is_shrunk_and_can_be_off():
    from elfantasy.season.model import fit_position_correction

    cheap = {"pos": "F", "price": 5.0, "actual": 8.0, "forecast_plain": 6.0}
    star = {"pos": "G", "price": 18.0, "actual": 10.0, "forecast_plain": 16.0}
    fixes = fit_position_correction([cheap, cheap, cheap, star], 3)
    assert fixes[("F", 0)] == pytest.approx(6.0 / 6)
    assert fixes[("G", 2)] == pytest.approx(-6.0 / 4)
    assert fit_position_correction([cheap], 0) == {}


def test_position_correction_runs_through_projection_and_backtest(tmp_path, rules):
    from dataclasses import replace

    from elfantasy.season.backtest import run_backtest
    from elfantasy.season.model import SeasonModel

    sd, dd = make_season(tmp_path)
    season = load_season(sd)
    data = SeasonData.for_season(dd, season)
    plain = run_backtest(SeasonModel(season, data, rules))
    assert plain[1].blend.top == pytest.approx(plain[1].plain_blend.top)

    model = SeasonModel(replace(season, position_prior_games=5.0), data, rules)
    result = model.project(3, None)
    hurt = next(p for p in result["players"] if p["name"] == "BBB C hx")
    assert hurt["proj"]["3"]["position_fix"] == 0.0  # out this round: no correction
    assert any(p["proj"]["3"]["position_fix"] != 0.0 for p in result["players"] if "3" in p["proj"])
    assert run_backtest(model)[1].plain_blend is not None


def test_this_season_uses_the_correction():
    assert load_season(SEASON).position_prior_games == 150.0


# ---------------------------------------------------------------- rosters and registrations
def _roster_data(tmp_path, people, public=None):
    season = load_season(SEASON)
    data = SeasonData(tmp_path)
    data.prior_dir.mkdir(parents=True)
    (data.prior_dir / "prices_round_01.csv").write_text(
        "rank,player,club,pos,price\n1,Kenan Kamenjas,Dubai,C,6.0\n2,Chris Silva,Fenerbahce,C,7.0\n",
        encoding="utf-8",
    )
    (data.prior_dir / "people_E2026.json").write_text(json.dumps({"data": people}))
    (data.prior_dir / "stats_E2025_acc.json").write_text(json.dumps({"players": []}))
    # Round 11 on 11-13 Nov, Round 12 on 17-18 Nov, Round 13 on 19-20 Nov.
    days = {11: ("2026-11-11", "2026-11-13"), 12: ("2026-11-17", "2026-11-18"),
            13: ("2026-11-19", "2026-11-20")}  # fmt: skip
    data.games_file.write_text(json.dumps({"data": [
        {"identifier": f"G{r}{i}", "round": r, "played": r < 12, "localDate": f"{d}T20:00:00",
         "local": {"club": {"code": "AAA"}, "score": 0}, "road": {"club": {"code": "BBB"}, "score": 0}}
        for r, ds in days.items() for i, d in enumerate(ds)
    ]}))  # fmt: skip
    if public:
        (data.root / "archive" / "tables").mkdir(parents=True)
        (data.root / "archive" / "tables" / "prices_public.jsonl").write_text(
            "\n".join(
                json.dumps({"captured_at": r.pop("captured_at", "2026-11-17T00:00:00+00:00"), **r})
                for r in public
            )
        )
    return season, data


def _person(code, name, club, start, active=True, end="2027-06-30", pos="Center"):
    return {"person": {"code": code, "name": name}, "club": {"code": club}, "active": active,
            "startDate": start, "endDate": end, "positionName": pos}  # fmt: skip


def test_a_transfer_takes_the_active_registration_not_the_stale_list_club(tmp_path):
    season, data = _roster_data(tmp_path, [
        _person("008325", "KAMENJAS, KENAN", "DUB", "2026-08-12T07:41:25", active=False,
                end="2026-09-17T14:39:15"),
        _person("008325", "KAMENJAS, KENAN", "BAS", "2026-11-13T11:03:35"),
        _person("007000", "SILVA, CHRIS", "ULK", "2026-08-01T00:00:00", active=False, end="2026-11-16"),
        _person("007000", "SILVA, CHRIS", "PAR", "2026-11-16T10:00:00"),
    ])  # fmt: skip
    _, _, report = data.join_prior(season)
    rows = {r["name"]: r for r in json.loads((data.prior_dir / "joined.json").read_text())}
    assert (rows["Kenan Kamenjas"]["club"], rows["Kenan Kamenjas"]["registered_from"]) == (
        "BAS",
        "2026-11-13",
    )
    assert rows["Kenan Kamenjas"]["registered_until"] is None
    assert rows["Chris Silva"]["club"] == "PAR"
    assert report["moved"] == ["Kenan Kamenjas (DUB -> BAS)", "Chris Silva (ULK -> PAR)"]
    # The whole history is kept: replays of earlier rounds need his club then.
    assert rows["Chris Silva"]["registrations"] == [
        {"club": "ULK", "from": "2026-08-01", "until": "2026-11-16"},
        {"club": "PAR", "from": "2026-11-16", "until": None},
    ]


def test_a_new_arrival_is_added_priced_from_the_public_list(tmp_path):
    season, data = _roster_data(
        tmp_path,
        [_person("014213", "PAYNE, CAMERON", "IST", "2026-10-07T15:09:46", pos="Guard"),
         _person("013000", "WALKER IV, LONNIE", "BES", "2026-10-20T00:00:00", pos="Guard"),
         _person("013001", "NEWCOMER, NOBODY", "BES", "2026-11-01T00:00:00", pos="Forward")],
        public=[{"player": "Cameron Payne", "club": "Anadolu Efes", "pos": "G", "price": 11.4},
                {"player": "Lonnie Walker", "club": "Besiktas", "pos": "G", "price": 9.9}],
    )  # fmt: skip
    _, _, report = data.join_prior(season)
    rows = {r["name"]: r for r in json.loads((data.prior_dir / "joined.json").read_text())}
    assert (
        rows["Cameron Payne"]["club"],
        rows["Cameron Payne"]["price"],
        rows["Cameron Payne"]["pos"],
    ) == ("IST", 11.4, "G")
    assert rows["Lonnie Walker IV"]["price"] == 9.9  # matched without the suffix
    assert rows["Nobody Newcomer"]["price_unknown"] and rows["Nobody Newcomer"]["pos"] == "F"
    assert report["unpriced"] == ["Nobody Newcomer (BES)"]
    # Each price is dated by the round it was captured for, not the round of the run.
    assert rows["Cameron Payne"]["price_obs"] == [[12, 11.4]]


def test_public_prices_are_dated_by_the_round_they_were_captured_for(tmp_path):
    season, data = _roster_data(
        tmp_path,
        [_person("014213", "PAYNE, CAMERON", "IST", "2026-10-07T15:09:46", pos="Guard")],
        public=[{"player": "Cameron Payne", "club": "Efes", "pos": "G", "price": 11.4},
                {"captured_at": "2026-11-17T11:00:00+00:00", "player": "Cameron Payne",
                 "club": "Efes", "pos": "G", "price": 11.5},
                {"captured_at": "2026-11-19T09:00:00+00:00", "player": "Cameron Payne",
                 "club": "Efes", "pos": "G", "price": 11.9}],
    )  # fmt: skip
    data.join_prior(season)
    payne = json.loads((data.prior_dir / "joined.json").read_text())[-1]
    assert payne["price_obs"] == [[12, 11.5], [13, 11.9]]  # the last capture of each round
    assert payne["price"] == 11.9


def test_a_later_price_never_stands_in_for_an_earlier_one(tmp_path, rules):
    season = load_season(SEASON)
    data = SeasonData(tmp_path)
    (tmp_path / "boxes").mkdir()
    for rnd in range(6, 13):
        (tmp_path / "boxes" / f"round_{rnd:02d}.json").write_text("[]")
    signing = {"name": "Cameron Payne", "club": "IST", "pos": "G", "code": "1", "price": 10.2,
               "price_obs": [[12, 10.2]], "added": True}  # fmt: skip
    p = dict(signing)
    estimate_prices([p], [], 6, data, season, rules, {})
    assert p["price_unknown"] and p["price"] == 4.0  # not 10.2: nothing on record by Round 6
    q = dict(signing)
    estimate_prices([q], [], 13, data, season, rules, {})
    assert not q.get("price_unknown") and q["price"] == pytest.approx(10.1)  # R12 DNP: -0.1


def test_absences_before_registration_are_not_coach_decisions():
    p = _player(base_min=20.0)
    p["registered_from"] = "2026-10-05"
    games = {"g1": _game("g1", 1, 2), "g2": _game("g2", 2, 2)}
    games["g1"]["localDate"], games["g2"]["localDate"] = (
        "2026-10-01T20:00:00",
        "2026-10-08T20:00:00",
    )
    # Round 1 (before he joined) has no line for him; round 2 he plays 24 minutes.
    update_with_evidence([p], games, {1: {}, 2: {"1": _line("g2", 24.0, 12.0)}}, {}, W)
    assert p["base_min"] == pytest.approx((0.5 * 20 + 24) / 1.5)  # round 1 not counted as 0


def test_unregistered_players_are_off_the_roster_that_round():
    from elfantasy.season.model import registered

    p = {"registered_from": "2026-11-13", "registered_until": None}
    assert not registered(p, "2026-11-12") and registered(p, "2026-11-17")
    assert not registered({"registered_until": "2026-10-24"}, "2026-11-01")
    assert registered({}, "2026-11-01")
    # Back at a club he left: both spells there count.
    back = {"club": "DUB", "registrations": [
        {"club": "DUB", "from": "2026-08-01", "until": "2026-10-01"},
        {"club": "BAS", "from": "2026-10-02", "until": "2026-10-20"},
        {"club": "DUB", "from": "2026-10-21", "until": None}]}  # fmt: skip
    assert registered(back, "2026-09-30") and not registered(back, "2026-10-10")


def test_a_price_on_record_is_found_at_any_of_his_clubs():
    from elfantasy.season.model import price_record

    signing = {"name": "Nikola Mirotic", "club": "PAM", "price": 12.2, "price_obs": [[12, 12.2]]}
    lists = {4: {key("PAM", "Nikola Mirotic"): 11.0}, 2: {key("PAM", "Someone Else"): 5.0}}
    assert price_record(signing, lists) == [(4, 11.0), (12, 12.2)]
    moved = {"name": "Kamar Baldwin", "club": "TEL", "price": 6.7,
             "registrations": [{"club": "MUN", "from": "2026-08-12", "until": "2026-11-16"},
                               {"club": "TEL", "from": "2026-11-16", "until": None}]}  # fmt: skip
    assert price_record(moved, {2: {key("MUN", "Kamar Baldwin"): 6.5}}) == [(1, 6.7), (2, 6.5)]


def test_a_transfer_is_at_his_club_of_the_day():
    from elfantasy.season.model import as_of

    stints = [{"club": "MUN", "from": "2026-08-12", "until": "2026-11-16"},
              {"club": "TEL", "from": "2026-11-16", "until": None}]  # fmt: skip
    p = {"club": "TEL", "registrations": stints}
    as_of(p, "2026-10-14")
    assert (p["club"], p["registered_until"]) == ("MUN", "2026-11-16")
    as_of(p, "2026-11-19")
    assert (p["club"], p["registered_until"]) == ("TEL", None)


def test_old_club_games_teach_the_rate_before_a_debut():
    """No game at the new club yet: his minutes there are unknown, but what he
    scored per minute elsewhere still counts."""

    p = _player(base_min=20.0)
    p["registered_from"] = "2026-11-13"
    games = {"g1": _game("g1", 1, 2)}
    games["g1"]["localDate"] = "2026-10-01T20:00:00"
    line = {**_line("g1", 24.0, 30.0), "team_code": "BBB"}
    update_with_evidence([p], games, {1: {"1": line}}, {}, W)
    assert p["rate"] == pytest.approx((180 * 0.5 + 30) / (180 + 24))
    assert p["base_min"] == 20.0


def _rosters_changed(tmp_path):
    """make_season with a transfer (BBB G bx to CCC after Round 2), a release
    (DDD F ex, owned) and two signings for AAA: one priced from Round 3, one
    with no price on record."""

    import yaml

    sd, dd = make_season(tmp_path)
    root = dd / "cache" / "seasons" / "test"
    joined = json.loads((root / "prior" / "joined.json").read_text())
    by = {p["name"]: p for p in joined}
    by["BBB G bx"].update(club="CCC", registered_from="2026-10-12", registered_until=None,
                          registrations=[{"club": "BBB", "from": "2026-08-01", "until": "2026-10-11"},
                                         {"club": "CCC", "from": "2026-10-12", "until": None}])  # fmt: skip
    by["DDD F ex"].update(registered_from="2026-08-01", registered_until="2026-10-11",
                          registrations=[{"club": "DDD", "from": "2026-08-01", "until": "2026-10-11"}])  # fmt: skip
    joined += [
        {"name": "Signed Early", "club": "AAA", "pos": "G", "code": "PNEW1", "added": True,
         "price": 7.5, "price_obs": [[3, 7.5]], "registered_from": "2026-09-01",
         "registrations": [{"club": "AAA", "from": "2026-09-01", "until": None}]},
        {"name": "Signed Unpriced", "club": "AAA", "pos": "F", "code": "PNEW2", "added": True,
         "price": 4.0, "price_unknown": True, "registered_from": "2026-09-01",
         "registrations": [{"club": "AAA", "from": "2026-09-01", "until": None}]},
    ]  # fmt: skip
    (root / "prior" / "joined.json").write_text(json.dumps(joined))
    priv = yaml.safe_load((sd / "private" / "r03.yaml").read_text())
    priv["prices"].append({"club": "AAA", "player": "Signed Unpriced", "price": 6.3})
    (sd / "private" / "r03.yaml").write_text(yaml.safe_dump(priv))
    return sd, dd


def test_replays_keep_each_player_where_he_was(tmp_path, rules):
    from elfantasy.season.model import SeasonModel

    sd, dd = _rosters_changed(tmp_path)
    season = load_season(sd)
    model = SeasonModel(season, SeasonData.for_season(dd, season), rules)
    past = {x["name"]: x for x in model.reproject(2)}
    assert past["BBB G bx"]["club"] == "BBB"  # not dropped for his later move
    assert "DDD F ex" in past  # released after Round 2: he played it
    assert "Signed Early" not in past  # first priced for Round 3: no price in Round 2
    now = {p["name"]: p for p in model.project(3, None)["players"]}
    assert now["BBB G bx"]["club"] == "CCC" and now["BBB G bx"]["proj"]["3"]["opp"] == "BBB"
    assert now["Signed Early"]["price"] == 7.5 and not now["Signed Early"].get("price_unknown")
    assert now["DDD F ex"]["proj"] == {}  # not owned here: off the board


def test_owned_released_and_newly_priced_players_reach_the_planner(tmp_path, rules):
    from elfantasy.season.inputs import load_private
    from elfantasy.season.model import SeasonModel

    sd, dd = _rosters_changed(tmp_path)
    season = load_season(sd)
    model = SeasonModel(season, SeasonData.for_season(dd, season), rules)
    private = load_private(sd, 3)
    assert "DDD F ex" in private.squad
    result = model.project(3, private)
    by = {p["name"]: p for p in result["players"]}
    gone = by["DDD F ex"]["proj"]["3"]
    assert gone["ev"] == 0.0 and gone["play"] == 0.0 and gone["unregistered"]
    assert by["Signed Unpriced"]["price"] == 6.3 and not by["Signed Unpriced"].get("price_unknown")
    planner = RoundPlanner(result, rules, private, log=lambda m: None)
    assert planner.budget() > 0  # it used to raise: not in the projections
    assert {"DDD F ex", "Signed Unpriced"} <= {p.player_id for p in planner.lp}
    plan = planner.search(n_search=300, n_final=600, seeds=(1, 2)).plan
    assert "DDD F ex" not in plan.bought


# ---------------------------------------------------------------- tipster picks
def test_tipster_picks_are_matched_to_our_players():
    from elfantasy.season.tips import club_code, match_pick

    clubs = load_season(SEASON).clubs
    assert [club_code(c, clubs) for c in ("Asvel", "Virtus", "Bayern", "Valencia")] == [
        "ASV", "VIR", "MUN", "PAM",
    ]  # fmt: skip
    people = [{"name": "TJ Shorts", "club": "PAM"}, {"name": "Azuolas Tubelis", "club": "ZAL"},
              {"name": "Tyson Carter", "club": "RED"}]  # fmt: skip
    assert (
        match_pick({"player": "T.J. Shorts", "club": "Valencia"}, people, clubs)["name"]
        == "TJ Shorts"
    )
    # a misspelling is accepted only at the same club
    assert match_pick({"player": "Azoulas Tubelis", "club": "Zalgiris"}, people, clubs)
    assert match_pick({"player": "Azoulas Tubelis", "club": "Partizan"}, people, clubs) is None
    assert match_pick({"player": "Nobody Here", "club": "Zalgiris"}, people, clubs) is None


def test_a_pick_is_timed_against_the_first_tip_off():
    from elfantasy.season.tips import round_tips

    def row(player, captured, modified):
        return {"round": 7, "player": player, "captured_at": captured, "modified": modified}

    rows = [row("Early", "2026-10-20T10:00:00+00:00", "2026-10-20T09:00:00+00:00"),
            row("Late", "2026-10-22T10:00:00+00:00", "2026-10-22T08:00:00+00:00")]  # fmt: skip
    tipoff = "2026-10-21T17:00:00Z"
    (before,) = round_tips(rows, 7, tipoff)
    assert before["player"] == "Early" and before["timing"] == "observed"  # archived then
    assert [p["player"] for p in round_tips(rows, 7)] == ["Late"]  # the card: the latest
    # Archived only after the round: the publisher's edit time is all we have.
    later = [row("A", "2026-11-18T10:00:00+00:00", "2026-10-21T09:01:00+00:00")]
    assert round_tips(later, 7, tipoff)[0]["timing"] == "dated"
    edited = [row("A", "2026-11-18T10:00:00+00:00", "2026-10-24T15:24:00+00:00")]
    assert round_tips(edited, 7, tipoff)[0]["timing"] == "late"
    undated = [row("A", "2026-11-18T10:00:00+00:00", None)]
    assert round_tips(undated, 7, tipoff)[0]["timing"] == "late"


def test_the_card_flags_picks_we_rank_low():
    from elfantasy.season.tips import LOW_RANK, card_rows

    clubs = load_season(SEASON).clubs
    names = [f"Guard {a}{b}" for a in "bcdfg" for b in "aeiou"][: LOW_RANK + 5]  # Guard ba, be...
    guards = [{"name": n, "club": "MAD", "pos": "G", "proj": {"13": {"ev": 30.0 - i}}}
              for i, n in enumerate(names)]  # fmt: skip
    guards.append({"name": "Hurt Guy", "club": "MAD", "pos": "G", "proj": {}})
    coaches = [{"name": "Coach Zed", "club": "PAN", "proj": {"13": {"ev": 10.0}}}]
    result = {"round": 13, "players": guards, "coaches": coaches}

    def pick(player, pos="G", club="Real Madrid"):
        return {"pos": pos, "player": player, "club": club, "price": 9.0}

    rows = card_rows(result, [pick(names[0]), pick(names[LOW_RANK + 2]), pick("Hurt Guy"),
                              pick("Not There"), pick("Coach Zed", "HC", "Panathinaikos")], clubs)  # fmt: skip
    assert [(r["rank"], r["note"][:12]) for r in rows] == [
        (1, ""), (LOW_RANK + 3, "we rank him "), (None, "no forecast "), (None, "not in our p"),
        (1, ""),
    ]  # fmt: skip


def test_tipster_bump_and_recent_form_are_shrunk():
    from elfantasy.season.backtest import peer_excess, recent_form, tipster_bump

    rows = [{"round": 1, "name": n, "pos": "G", "price": p, "actual": a, "forecast": 6.0}
            for n, p, a in (("A", 5.0, 10.0), ("B", 5.5, 7.0), ("C", 6.0, 5.0))]  # fmt: skip
    ((x, excess),) = peer_excess(rows, {(1, "A")})
    # A beat his forecast by 4; his unpicked peers by +1 and -1 (mean 0)
    assert x["name"] == "A" and excess == pytest.approx(4.0)
    assert tipster_bump(rows, {(1, "A")}, prior_picks=3) == pytest.approx(4.0 / 4)
    assert recent_form({1: rows}, 2, rounds=3, prior_games=1)["A"] == pytest.approx(4.0 / 2)
