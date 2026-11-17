"""The data archive: parsers on trimmed copies of the real pages, storage, collectors."""

from __future__ import annotations

import gzip
import json

import pytest
import requests

from elfantasy.season import archive as A

RESULTS = """
<table><tr><th class="h-text-left" colspan="2">11. Round</th><th>1</th><th>2</th></tr>
<tr><td class="h-text-left"><a data-test="1" href="/basketball/europe/euroleague/anadolu-efes-virtus-bologna/QgsorKdR/" class="in-match"><span><strong>Anadolu Efes</strong></span> - <span>Virtus Bologna</span></a></td><td class="h-text-center"><a href="/x/">103:91</a></td><td class="table-main__odds colored"><span><span><span data-odd="1.38"></span></span></span></td><td class="table-main__odds" data-odd="3.05"></td><td class="h-text-right h-text-no-wrap">13.11.</td></tr>
</table>"""
FIXTURES = """
<table><tr><th class="h-text-left" colspan="2">12. Round</th></tr>
<tr>
<td class="table-main__datetime">17.11. 17:00</td>
<td class="h-text-left"><a href="/basketball/europe/euroleague/dubai-anadolu-efes/UkWOzxk1/" class="in-match"><span>Dubai</span> - <span>Anadolu Efes</span></a></td>
<td class="table-main__odds"><button data-odd="1.27"></button></td><td class="table-main__odds"><button data-odd="3.57"></button></td>
</tr></table>"""


def _market(rows):
    cells = "".join(
        f'<tr data-bid="{i}"><td class="table-main__doubleparameter">{line}</td>'
        f'<td class="x {"inactive" if not active else ""}" data-odd="{a}"></td>'
        f'<td class="y" data-odd="{b}"></td></tr>'
        for i, (line, a, b, active) in enumerate(rows)
    )
    return json.dumps({"odds": f"<table><tbody>{cells}</tbody></table>"})


def test_league_page_rows():
    (r,) = A.parse_betexplorer_table(RESULTS)
    assert (r["round"], r["match_id"], r["home"], r["away"]) == (
        11,
        "QgsorKdR",
        "Anadolu Efes",
        "Virtus Bologna",
    )
    assert (r["home_score"], r["away_score"], r["winner"]) == (103, 91, "home")
    assert (r["home_odds"], r["away_odds"], r["when"]) == (1.38, 3.05, "13.11.")
    (f,) = A.parse_betexplorer_table(FIXTURES)
    assert (f["round"], f["home_score"], f["home_odds"], f["when"]) == (
        12,
        None,
        1.27,
        "17.11. 17:00",
    )
    assert A.el_code(f["home"]) == "DUB" and A.el_code("Crvena Zvezda Meridianbet") == "RED"


def test_market_consensus():
    body = _market([(170.5, 1.90, 1.90, True), (170.5, 1.85, 1.95, True), (171.5, 2.0, 1.8, True),
                    (169.5, 1.5, 2.5, False)])  # fmt: skip
    rows = A.parse_betexplorer_market(body)
    assert len(rows) == 4 and rows[3]["active"] is False
    total = A.total_consensus(rows)
    assert total["line"] == 170.5 and total["books"] == 2 and total["expected"] > 170.5
    home = A.handicap_consensus(A.parse_betexplorer_market(_market([(-4.5, 1.9, 1.9, True)])))
    assert home["expected"] == pytest.approx(4.5)
    assert A.total_consensus([]) is None


def test_tracker_club_tables():
    page = "<div>" + "".join(f"<p>{x}</p>" for x in [
        "Updated 16 November 2026, 14:19", "EuroLeague Injuries by Club", "Anadolu Efes", "Asvel",
        "Anadolu Efes", "2 players out", "Player", "Status", "Injury", "Expected return",
        "Mike James", "Out", "Achilles Tendon injury", "7-9 months",
        "P.J. Dozier", "Cleared", "Back injury", "–",
        "Bayern", "No injured players", "No injured players reported.",
        "Barcelona", "1 player out", "Juan Nunez", "Day to day", "Right ankle sprain", "–",
        "EuroLeague", "A news headline",
    ]) + "</div>"  # fmt: skip
    updated, rows = A.parse_tracker(page)
    assert updated == "Updated 16 November 2026, 14:19"
    assert [(r["club"], r["player"], r["status"]) for r in rows] == [
        ("Anadolu Efes", "Mike James", "Out"), ("Anadolu Efes", "P.J. Dozier", "Cleared"),
        ("Barcelona", "Juan Nunez", "Day to day"),
    ]  # fmt: skip
    assert rows[1]["expected_return"] is None


def test_rotowire_items():
    marker = "ANALYSIS</p><p>Subscribe now</p><p>to instantly reveal our take on this news.</p>"
    page = ("<p>Compact</p><p>Expanded</p>"
            "<p>Kevin Punter</p><p>Will miss three weeks</p><p>G</p><p>FC Barcelona</p><p>Ankle</p>"
            f"<p>November 9, 2026</p><p>Punter will be sidelined.</p><p>{marker}"
            "<p>Antonio Reeves</p><p>Makes debut</p><p>G</p><p>Partizan</p><p>October 26, 2026</p>"
            f"<p>Reeves made his debut.</p><p>{marker}")  # fmt: skip
    items = A.parse_rotowire(page)
    assert [(i["player"], i["injury"], i["date"]) for i in items] == [
        ("Kevin Punter", "Ankle", "November 9, 2026"), ("Antonio Reeves", None, "October 26, 2026"),
    ]  # fmt: skip


def test_price_list():
    page = ('<tr data-pos="F" data-club="Olympiacos" data-price="18.9"><td class="n r">1</td>'
            '<td class="nm"><a href="x">Sasha Vezenkov</a></td><td class="c">Olympiacos</td></tr>')  # fmt: skip
    assert A.parse_prices(page) == [
        {"player": "Sasha Vezenkov", "club": "Olympiacos", "pos": "F", "price": 18.9}
    ]


def test_storage(tmp_path):
    store = A.Archive(tmp_path, stamp="20261117T000000Z")
    path = store.raw("src", "page", "hello")
    assert path.name == "20261117T000000Z_page.gz"
    with gzip.open(path, "rt") as fh:
        assert fh.read() == "hello"
    assert not store.once_path("euroleague", "g1").is_file()
    store.once("euroleague", "g1", "{}")
    assert store.once_path("euroleague", "g1").is_file()
    store.append("t", "src", [{"a": 1}])
    store.append("t", "src", [{"a": 2}])
    assert [r["a"] for r in store.read("t")] == [1, 2]
    assert store.read("t")[0]["captured_at"].startswith("2026-11-17T00:00:00")


class _Fetch:
    def __init__(self, pages):
        self.pages = pages

    def get(self, url, **headers):
        for key, body in self.pages.items():
            if key in url:
                if isinstance(body, Exception):
                    raise body
                return body
        raise requests.HTTPError("404")


def test_collectors_fail_alone_and_lines_come_back(tmp_path):
    store = A.Archive(tmp_path)
    fetch = _Fetch({
        "euroleague/fixtures": FIXTURES, "euroleague/results": RESULTS,
        "match-odds-old/UkWOzxk1/0/ou": _market([(173.5, 1.9, 1.9, True)]),
        "match-odds-old/UkWOzxk1/0/ah": _market([(-8.5, 1.9, 1.9, True)]),
        "match-odds-old/QgsorKdR": _market([(170.5, 1.9, 1.9, True)]),
        "player-prices": requests.ConnectionError("down"),
        "PlayByPlay": '{"Live": false}', "Points": '{"Rows": []}',
    })  # fmt: skip
    games = {
        "E2026_1": {"gameCode": 1, "played": True},
        "E2026_2": {"gameCode": 2, "played": False},
    }
    out = A.collect(
        store, games, "E2026", only=["odds", "prices", "pbp"], fetch=fetch, log=lambda m: None
    )
    assert (
        out["prices"].startswith("failed") and "rows" in out["odds"] and out["pbp"].startswith("2 ")
    )
    assert A.round_lines(store, 12) == ({"DUB-IST": (1.27, 3.57)}, {"DUB-IST": 173.5})
    # finished games' markets are fetched once
    again = A.collect(store, games, "E2026", only=["odds", "pbp"], fetch=fetch, log=lambda m: None)
    assert again["pbp"].startswith("0 ")
    closing = [r for r in store.read("odds_markets") if r["kind"] == "result"]
    assert len(closing) == 2  # one total and one handicap, not repeated


def test_fetcher_retries_a_dropped_connection(monkeypatch):
    calls = []

    class Session:
        headers: dict = {}

        def get(self, url, headers=None, timeout=None):
            calls.append(url)
            if len(calls) == 1:
                raise requests.ConnectionError("dropped")
            response = requests.Response()
            response.status_code, response._content = 200, b"ok"
            return response

    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    fetch = A.Fetcher(interval=0, session=Session(), slow={})
    assert fetch.get("https://example.org/x") == "ok" and len(calls) == 2
