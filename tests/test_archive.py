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


def _http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status}", response=response)


class _Fetch:
    def __init__(self, pages):
        self.pages = pages

    def get(self, url, **headers):
        for key, body in self.pages.items():
            if key in url:
                if isinstance(body, Exception):
                    raise body
                return body
        raise _http_error(404)


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


TIPS = (
    '<script type="application/ld+json">{"datePublished":"2026-11-15T11:09:07+00:00",'
    '"dateModified":"2026-11-16T13:56:02+00:00"}</script><article><h2>Guards</h2>'
    "<p><strong>Best options over 10.0 credits</strong></p>"
    "<p><strong>Elijah Bryant &#8211; Hapoel Tel Aviv &#8211; 17.6 credits</strong></p>"
    "<p>Why: <strong>Elijah Bryant</strong> is on fire.</p>"
    "<p><strong>Best options from 6.0 to 10.0 credits</strong></p>"
    "<p><strong>Andres Feliz – Real Madrid – 9.7 credits</strong></p>"
    "<h2>Head coaches</h2><p><strong>Zeljko Obradovic – Panathinaikos – 10.2 credits</strong></p>"
    "<h2>FAQ</h2><p><strong>Not – a – 5 credits pick</strong></p></article>"
)


def _tips_link(r, season="26-27"):
    return f"https://basketballsphere.com/en/euroleague-fantasy-{season}-round-{r}-best-picks/"


def test_tips_article():
    published, modified, picks = A.parse_tips(TIPS)
    assert (published[:10], modified[:10]) == ("2026-11-15", "2026-11-16")
    assert [(p["pos"], p["player"], p["club"], p["price"]) for p in picks] == [
        ("G", "Elijah Bryant", "Hapoel Tel Aviv", 17.6),
        ("G", "Andres Feliz", "Real Madrid", 9.7),
        ("HC", "Zeljko Obradovic", "Panathinaikos", 10.2),
    ]
    assert picks[0]["tier"] == "Best options over 10.0 credits"
    assert picks[1]["tier"].startswith("Best options from 6.0")
    assert A.tips_slug("E2026") == "26-27"
    other = "https://basketballsphere.com/en/euroleague-fantasy-round-9-26-27-best-picks/"
    page = (f'<a href="{_tips_link(12)}">x</a><a href="{_tips_link(3, "25-26")}">last season</a>'
            f'<a href="{other}">the other format</a>')  # fmt: skip
    assert A.tip_articles(page, "E2026") == {12: _tips_link(12), 9: other}
    assert other in A.tip_addresses(9, "E2026") and _tips_link(9) in A.tip_addresses(9, "E2026")


def test_tips_without_prices():
    """Round 9's article names 'Name – Club' only: a pick when the club is a
    EuroLeague club, with an unknown price (not a guessed one)."""
    page = (
        "<article><h2>Guards</h2><p><strong>Best options over 10.0 credits</strong></p>"
        "<p><strong>TJ Shorts – Valencia</strong></p><p><strong>TJ Shorts</strong></p>"
        "<p><strong>EuroLeague Fantasy Preview – The Best Rookies</strong></p>"
        "<p><strong>Jae Crowder – Asvel</strong></p></article>"
    )
    _, _, picks = A.parse_tips(page)
    assert [(p["player"], p["club"], p["price"]) for p in picks] == [
        ("TJ Shorts", "Valencia", None), ("Jae Crowder", "Asvel", None),
    ]  # fmt: skip


def test_tips_are_backfilled_once_and_the_open_round_rechecked(tmp_path):
    calls = []

    class Fetch(_Fetch):
        def get(self, url, **headers):
            calls.append(url)
            return super().get(url, **headers)

    fetch = Fetch({
        "round-tips/": f'<a href="{_tips_link(1)}"></a><a href="{_tips_link(3)}"></a>',
        "category/": "", "round-1-best-picks": TIPS, "round-3-best-picks": TIPS,
    })  # fmt: skip
    games = {f"g{r}": {"round": r, "played": r < 3} for r in (1, 2, 3)}
    store = A.Archive(tmp_path)
    out = A.collect(store, games, "E2026", only=["tips"], fetch=fetch, log=lambda m: None)
    assert out["tips"].startswith("6 ")  # three picks in each of rounds 1 and 3
    assert {r["round"] for r in store.read("tips")} == {1, 3}
    assert store.once_path("basketballsphere_tips", "missing_02").is_file()  # finished, no article
    calls.clear()
    A.collect(store, games, "E2026", only=["tips"], fetch=fetch, log=lambda m: None)
    articles = [u for u in calls if "best-" in u]
    assert articles[0] == _tips_link(3)  # still open: read again
    assert not any("round-1-" in u or "round-2-" in u for u in articles)  # settled
    assert any("round-4-" in u for u in articles)  # the next round's article is looked for
    # A listing now links round 2's article: the 'missing' marker does not stop it.
    late = "https://basketballsphere.com/en/euroleague-fantasy-round-2-26-27-best-picks/"
    fetch.pages["round-tips/"] += f'<a href="{late}"></a>'
    fetch.pages["round-2-26-27"] = TIPS
    A.collect(store, games, "E2026", only=["tips"], fetch=fetch, log=lambda m: None)
    assert 2 in {r["round"] for r in store.read("tips")}


def test_a_failed_request_is_not_a_missing_article(tmp_path):
    fetch = _Fetch({"round-tips/": f'<a href="{_tips_link(1)}"></a>', "category/": "",
                    "round-1-best-picks": _http_error(503)})  # fmt: skip
    games = {"g1": {"round": 1, "played": True}, "g2": {"round": 2, "played": False}}
    store = A.Archive(tmp_path)
    A.collect(store, games, "E2026", only=["tips"], fetch=fetch, log=lambda m: None)
    assert not store.once_path("basketballsphere_tips", "missing_01").is_file()
    fetch.pages["round-1-best-picks"] = TIPS  # the site is back
    A.collect(store, games, "E2026", only=["tips"], fetch=fetch, log=lambda m: None)
    assert {r["round"] for r in store.read("tips")} == {1}


def test_an_article_without_picks_is_reported_and_retried(tmp_path):
    fetch = _Fetch({"round-tips/": f'<a href="{_tips_link(1)}"></a>', "category/": "",
                    "round-1-best-picks": "<article><h2>Guards</h2><p>new layout</p></article>"})  # fmt: skip
    games = {"g1": {"round": 1, "played": True}}
    store, said = A.Archive(tmp_path), []
    A.collect(store, games, "E2026", only=["tips"], fetch=fetch, log=said.append)
    assert any("no picks found" in m for m in said)
    fetch.pages["round-1-best-picks"] = TIPS
    A.collect(store, games, "E2026", only=["tips"], fetch=fetch, log=said.append)
    assert len(store.read("tips")) == 3
