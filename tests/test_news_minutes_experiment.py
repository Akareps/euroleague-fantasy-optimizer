"""Text ambiguity, deadline leakage, identity and moment consistency are material."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import yaml
from test_hierarchical_experiment import projection_fixture, season_fixture, training_rows

from elfantasy.experiments.hierarchical import Model, apply_model
from elfantasy.experiments.hierarchical import Settings as HierarchicalSettings
from elfantasy.experiments.news_cli import require_live_forecast
from elfantasy.experiments.news_collection import collect, transcript_segments
from elfantasy.experiments.news_minutes import (
    EXPERIMENT,
    Settings,
    apply_news,
    classify,
    select_evidence,
    sentiment,
)
from elfantasy.season.inputs import key

CUTOFF = datetime(2026, 10, 12, 18, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def context():
    fitted = Model(
        training_rows(), "2026-10-12T18:00:00Z", HierarchicalSettings(min_training_rows=50)
    )
    season = season_fixture()
    season.games["g"]["identifier"] = "E2026_41"
    season.round_inputs = lambda _: SimpleNamespace(availability={})
    baseline, _ = apply_model(projection_fixture(), fitted, season)
    for p in baseline["players"]:
        p["proj"]["6"] = copy.deepcopy(p["proj"]["5"])
    baseline["rounds"] = [5, 6]
    return baseline, fitted, season


def claim(**extra):
    return {
        "id": "claim1",
        "event_id": "coach-interview-20261012",
        "club": "AAA",
        "player": "P0",
        "game_id": "E2026_41",
        "source_url": "https://club.example/interview",
        "source_type": "club",
        "speaker": "Head coach",
        "published_at": "2026-10-12T12:00:00Z",
        "observed_at": "2026-10-12T13:00:00Z",
        "round": 5,
        "excerpt": "P0 will be limited to 15 minutes in tomorrow's game.",
        "kind": "minutes_cap",
        "time_scope": "upcoming_fixture",
        "confidence": 0.98,
        "reviewed": True,
        "excerpt_verified": True,
        **extra,
    }


def snapshot(*claims):
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "round": 5,
        "captured_at": "2026-10-12T14:00:00Z",
        "claims": list(claims),
    }


def identified(base):
    out = copy.deepcopy(base)
    for p in out["players"]:
        p["proj"]["5"]["game_id"] = "E2026_41"
    return out


@pytest.mark.parametrize(
    "text,kind",
    [
        ("He will be limited to 15 minutes.", "minutes_cap"),
        ("He might be limited to 15 minutes.", "context"),
        ("He was limited to 15 minutes yesterday.", "context"),
        ("He scored 15 points in 20 minutes.", "context"),
        ("He has no minutes restriction.", "context"),
        ("He is not ruled out.", "context"),
        ("He is not cleared to play.", "context"),
        ("He will not travel with the team.", "out"),
        ("He will play more minutes in the next game.", "role_up"),
        ("He could have a bigger role.", "context"),
        ("He will be limited to 15.5 minutes.", "context"),
        ("He had a great game and we are proud of him.", "context"),
    ],
)
def test_extractor_abstains_on_ambiguity_and_past_performance(text, kind):
    assert classify(text)["kind"] == kind


def test_future_capture_cannot_be_replayed_into_earlier_deadline(context):
    base, _, _ = context
    news = snapshot(claim())
    news["captured_at"] = "2026-10-13T00:00:00Z"
    with pytest.raises(ValueError, match="captured after"):
        select_evidence(news, identified(base), CUTOFF, Settings())


@pytest.mark.parametrize(
    "extra,reason",
    [
        ({"observed_at": "2026-10-12T19:00:00Z"}, "chronology"),
        ({"published_at": "2026-10-05T00:00:00Z"}, "stale"),
        ({"reviewed": False}, "unreviewed"),
        ({"excerpt_verified": False}, "unverified"),
        ({"club": "BBB"}, "identity"),
        ({"game_id": "E2026_42"}, "fixture"),
        ({"round": 6}, "wrong round"),
        ({"already_in_inputs": True}, "already incorporated"),
        ({"source_type": "forum"}, "authority"),
        ({"confidence": float("nan")}, "confidence"),
        ({"time_scope": "past_game"}, "upcoming fixture"),
    ],
)
def test_evidence_gates(context, extra, reason):
    base, _, _ = context
    chosen, audit = select_evidence(snapshot(claim(**extra)), identified(base), CUTOFF, Settings())
    assert not chosen and reason in audit[0]["reason"]


def test_reposted_interview_does_not_multiply_confidence_or_effect(context):
    base, fitted, season = context
    a = claim()
    b = claim(
        id="repost",
        source_url="https://reporter.example/same-interview",
        source_type="sourced_reporter",
    )
    one, _ = apply_news(base, fitted, season, snapshot(a), Settings(), CUTOFF)
    many, audit = apply_news(base, fitted, season, snapshot(a, b), Settings(), CUTOFF)
    assert one == many
    assert audit["selected_claims"] == 1
    assert audit["claims"][1]["reason"] == "duplicate originating event"


def test_conflicting_sources_abstain(context):
    base, fitted, season = context
    a = claim(excerpt="P0 will not play.", kind="out")
    b = claim(
        id="other", event_id="other-interview", excerpt="P0 is cleared to play.", kind="ready"
    )
    result, audit = apply_news(base, fitted, season, snapshot(a, b), Settings(), CUTOFF)
    assert result["players"] == base["players"]
    assert not audit["selected_claims"]
    assert all("conflicting" in r["reason"] for r in audit["claims"])


def test_caps_recompute_shooting_and_variance_without_mutation(context):
    base, fitted, season = context
    saved = copy.deepcopy(base)
    result, audit = apply_news(base, fitted, season, snapshot(claim()), Settings(), CUTOFF)
    q, before = result["players"][0]["proj"]["5"], base["players"][0]["proj"]["5"]
    assert 0 < q["min"] <= 15 < before["min"]
    assert q["independent"]["minute_sd"] > 0
    assert q["independent"]["minute_sd"] ** 2 <= q["min"] * (15 - q["min"])
    assert q["independent"]["components"]["points"] < before["independent"]["components"]["points"]
    assert (
        q["independent"]["components"]["missed_fg"]
        < before["independent"]["components"]["missed_fg"]
    )
    assert q["ev"] == pytest.approx(q["play"] * q["if_plays"])
    assert base == saved
    assert result["coaches"] == base["coaches"]
    assert all(
        p["price"] == b["price"] and p["proj"]["6"] == b["proj"]["6"]
        for p, b in zip(result["players"], base["players"], strict=True)
    )
    assert audit["updated_players"] > 1
    assert sum(p["proj"]["5"]["min"] * p["proj"]["5"]["play"] for p in result["players"]) <= 200


def test_restrictions_are_not_redistribution_recipients(context):
    base, fitted, season = context
    a, b = (
        claim(),
        claim(
            id="second",
            event_id="second-event",
            player="P1",
            excerpt="P1 will be limited to 10 minutes.",
        ),
    )
    result, _ = apply_news(base, fitted, season, snapshot(a, b), Settings(), CUTOFF)
    assert result["players"][0]["proj"]["5"]["min"] <= 15
    assert result["players"][1]["proj"]["5"]["min"] <= 10


def test_news_out_changes_availability_and_frees_some_minutes(context):
    base, fitted, season = context
    result, _ = apply_news(
        base,
        fitted,
        season,
        snapshot(claim(excerpt="P0 will not travel.", kind="out")),
        Settings(),
        CUTOFF,
    )
    assert result["players"][0]["proj"]["5"]["play"] == 0
    assert result["players"][0]["proj"]["5"]["ev"] == 0
    assert any(
        p["proj"]["5"]["min"] > b["proj"]["5"]["min"]
        for p, b in zip(result["players"][1:], base["players"][1:], strict=True)
    )


def test_existing_injury_input_requires_reviewed_supersession(context):
    base, fitted, season = context
    season = copy.copy(season)
    season.round_inputs = lambda _: SimpleNamespace(availability={key("AAA", "P0"): (0.97,)})
    result, audit = apply_news(
        base,
        fitted,
        season,
        snapshot(claim(excerpt="P0 will not travel.", kind="out")),
        Settings(),
        CUTOFF,
    )
    assert result["players"] == base["players"]
    assert "supersession" in audit["claims"][0]["reason"]


def test_sentiment_alone_never_changes_forecast(context):
    base, fitted, season = context
    praise = "P0 had a great game. We have confidence and trust in him."
    result, audit = apply_news(
        base, fitted, season, snapshot(claim(excerpt=praise, kind="context")), Settings(), CUTOFF
    )
    assert sentiment(praise)["score"] > 0
    assert sentiment("not good")["score"] < 0
    assert result["players"] == base["players"]
    assert not audit["updated_players"]


def test_duplicate_player_ids_and_unnamed_team_statement_are_rejected(context):
    base, _, _ = context
    changed = identified(base)
    changed["players"][1]["code"] = changed["players"][0]["code"]
    chosen, _ = select_evidence(snapshot(claim()), changed, CUTOFF, Settings())
    assert not chosen
    chosen, _ = select_evidence(snapshot(claim(player=None)), identified(base), CUTOFF, Settings())
    assert not chosen


def test_imported_caption_candidates_keep_video_offsets_and_need_review(tmp_path):
    text = "WEBVTT\n\n00:00:08.000 --> 00:00:12.000\nP0 will be limited to 15 minutes.\n\n"
    segments = transcript_segments(text)
    assert segments[0]["start"] == "00:00:08.000"
    captions = tmp_path / "interview.vtt"
    captions.write_text(text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "round": 5,
        "scope": [{"club": "AAA", "player": "P0"}],
        "sources": [
            {
                "id": "coach",
                "url": "https://club.example/video",
                "source_type": "club",
                "club": "AAA",
                "published_at": "2026-10-12T12:00:00Z",
                "format": "vtt",
                "transcript_path": "interview.vtt",
                "claims": [],
            }
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    news = collect(path, tmp_path / "out")
    assert news["candidates"][0]["transcript_start"] == "00:00:08.000"
    assert news["candidates"][0]["extracted"]["kind"] == "minutes_cap"
    assert news["candidates"][0]["reviewed"] is False
    assert not news["claims"]
    assert json.loads((tmp_path / "out" / "snapshot.json").read_text())["sources"][0][
        "content_sha256"
    ]


@pytest.mark.parametrize(
    "extra",
    [{"max_age_hours": 0}, {"half_life_hours": float("nan")}, {"redistribution_fraction": 2}],
)
def test_invalid_settings_fail(extra):
    with pytest.raises(ValueError):
        Settings(**extra)


def test_fresh_forecasts_cannot_be_built_after_deadline_even_with_old_news(context):
    _, _, season = context
    before = datetime(2026, 10, 13, 17, 44, tzinfo=timezone.utc)
    assert require_live_forecast(season, 5, before) == before
    at_lock = datetime(2026, 10, 13, 17, 45, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="already frozen run"):
        require_live_forecast(season, 5, at_lock)
