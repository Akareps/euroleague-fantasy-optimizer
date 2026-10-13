"""Reviewed, timestamped text evidence on top of hierarchical-v1.

Sentiment is an audit feature. Specific rotation statements can change forecasts;
general praise cannot. All mappings below are frozen hypotheses, not fitted effects.
"""

from __future__ import annotations

import copy
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import timezone

from scipy.stats import norm, truncnorm

from elfantasy.experiments.history import utc
from elfantasy.season.inputs import key

EXPERIMENT = "news-minutes-v1"
AUTHORITY = {
    "club": 1.0,
    "league": 1.0,
    "sourced_reporter": 0.85,
    "reporter_opinion": 0.45,
    "player": 0.5,
    "owner": 0.4,
    "forum": 0.2,
}
KINDS = {
    "out",
    "doubtful",
    "ready",
    "minutes_cap",
    "minutes_estimate",
    "role_up",
    "role_down",
    "limited",
    "context",
}


@dataclass(frozen=True)
class Settings:
    max_age_hours: float = 96.0
    min_confidence: float = 0.65
    min_effective_weight: float = 0.55
    half_life_hours: float = 72.0
    max_directional_shift: float = 2.0
    doubtful_probability: float = 0.7
    ready_probability: float = 0.97
    added_minute_sd: float = 2.0
    explicit_minutes_sd: float = 4.0
    redistribution_fraction: float = 0.5

    def __post_init__(self):
        for field, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid news setting: {field}")
        for field in ("max_age_hours", "half_life_hours", "explicit_minutes_sd"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        for field in (
            "min_confidence",
            "min_effective_weight",
            "doubtful_probability",
            "ready_probability",
            "redistribution_fraction",
        ):
            if getattr(self, field) > 1:
                raise ValueError(f"{field} must be in [0, 1]")


def normalise(text):
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[\w']+", text.replace("’", "'")))


def classify(text):
    """Small high-precision English/Spanish rule extractor; otherwise abstain.

    Reviewed translations may be supplied for other languages. No inferred minute
    cap from a player's box score, a number elsewhere on a page, or generic praise.
    """
    t = normalise(text)
    # Do not turn denied, hypothetical or uncertain restrictions into hard caps.
    uncertain = re.search(
        r"\b(may|might|could|if|possibly|rumou?r|perhaps|not limited|no minutes restriction|not ruled out|not cleared|not available|not fit|was limited|were limited|last game|yesterday|previous game)\b",
        t,
    )
    match = re.search(
        r"\b(?:limited to|capped at|at most|no more than|maximum of|limit(?:ing)? (?:him|his minutes) to|maximo de|limite de)\s+(\d+(?:\s+\d+)?)(?:\s*(?:to|a)\s*(\d+))?\s+minutes?\b",
        t,
    )
    if match and not uncertain and not re.search(r"\d+[.,]\d+", text):
        # Integers are intentional. Decimal punctuation in prose is ambiguous.
        cap = float(match.group(2) or match.group(1).split()[-1])
        if 0 < cap <= 40:
            return {"kind": "minutes_cap", "value": cap}
    match = re.search(r"\b(?:will play|expected to play|jugara)\s+(\d+)\s+(?:minutes|minutos)\b", t)
    if match and not uncertain and 0 < int(match.group(1)) <= 40:
        return {"kind": "minutes_estimate", "value": float(match.group(1))}
    if not uncertain and re.search(
        r"\b(ruled out|will not (?:play|travel)|won't (?:play|travel)|no (?:jugara|viajara))\b", t
    ):
        return {"kind": "out"}
    if re.search(
        r"\b(doubtful|game time decision|questionable|hopes? (?:he will|to) be ready|upitan)\b", t
    ):
        return {"kind": "doubtful"}
    if not uncertain and re.search(
        r"\b(cleared to play|available for (?:the|this) game|fit to play|todos sus jugadores disponibles)\b",
        t,
    ):
        return {"kind": "ready"}
    if not uncertain and re.search(
        r"\b(minutes restriction|limited minutes|not fully fit|not 100 percent|restriccion de minutos)\b",
        t,
    ):
        return {"kind": "limited"}
    if not uncertain and re.search(
        r"\b(will (?:get|play) more minutes|will have (?:a bigger role|more playing time)|jugara mas minutos)\b",
        t,
    ):
        return {"kind": "role_up"}
    if not uncertain and re.search(
        r"\b(will (?:get|play) fewer minutes|will have (?:a smaller role|less playing time)|jugara menos minutos)\b",
        t,
    ):
        return {"kind": "role_down"}
    return {"kind": "context"}


def sentiment(text):
    """Transparent English lexicon diagnostic; never a numerical model input."""
    tokens = normalise(text).split()
    positive = {
        "good",
        "great",
        "happy",
        "confidence",
        "trust",
        "improved",
        "ready",
        "proud",
        "excellent",
        "strong",
        "believed",
        "pleased",
        "delighted",
    }
    negative = {
        "injury",
        "pain",
        "injured",
        "doubtful",
        "struggling",
        "poor",
        "frustrating",
        "disappointed",
        "unfit",
    }
    hits = []
    for i, word in enumerate(tokens):
        score = int(word in positive) - int(word in negative)
        if score and any(w in {"not", "no", "never"} for w in tokens[max(0, i - 2) : i]):
            score *= -1
        if score:
            hits.append(score)
    value = sum(hits) / max(len(hits), 1)
    return {
        "score": value,
        "label": "positive" if value > 0 else "negative" if value < 0 else "neutral",
        "method": "English lexicon on reviewed translation; diagnostic only",
    }


def select_evidence(snapshot, projection, as_of, settings):
    """Reject hindsight, ambiguity, reposts, stale statements and unreviewed NLP."""
    if snapshot.get("schema_version") != 1 or snapshot.get("experiment") != EXPERIMENT:
        raise ValueError("not a news-minutes-v1 snapshot")
    cutoff = utc(as_of.isoformat()).astimezone(timezone.utc)
    captured = utc(snapshot["captured_at"])
    if captured > cutoff:
        raise ValueError("news snapshot was captured after the forecast cutoff")
    identities = defaultdict(list)
    codes = Counter(p.get("code") for p in projection["players"] if p.get("code"))
    for p in projection["players"]:
        identities[key(p["club"], p["name"])].append(p)
    audit, accepted = [], defaultdict(list)
    seen_ids = set()
    for claim in snapshot.get("claims", []):
        if claim["id"] in seen_ids:
            raise ValueError("duplicate claim IDs")
        seen_ids.add(claim["id"])
        text = claim.get("translation") or claim.get("excerpt", "")
        extracted = classify(text)
        row = {
            **claim,
            "sentiment": sentiment(text),
            "extracted": extracted,
            "decision": "rejected",
        }
        reason = None
        try:
            published = utc(claim["published_at"])
            observed = utc(claim["observed_at"])
            age = (cutoff - published).total_seconds() / 3600
            candidates = identities[key(claim.get("club", ""), claim.get("player") or "")]
            player = candidates[0] if len(candidates) == 1 else None
            confidence = float(claim["confidence"])
            weight = (
                AUTHORITY.get(claim.get("source_type"), 0)
                * confidence
                * 2 ** (-max(age, 0) / settings.half_life_hours)
            )
            row["effective_weight"] = weight
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                reason = "invalid confidence"
            elif published > observed or observed > captured or observed > cutoff:
                reason = "publication/observation chronology is invalid or after cutoff"
            elif age > settings.max_age_hours:
                reason = "stale"
            elif claim.get("round") != projection["round"]:
                reason = "wrong round (claims expire after the named fixture)"
            elif not claim.get("reviewed") or not claim.get("excerpt_verified"):
                reason = "unreviewed or unverified excerpt"
            elif (
                not claim.get("event_id") or not claim.get("source_url") or not claim.get("speaker")
            ):
                reason = "missing source, speaker or originating event"
            elif not player or not player.get("code") or codes[player["code"]] != 1:
                reason = "no unique player and club identity"
            elif str(projection["round"]) not in player["proj"]:
                reason = "no target-game forecast"
            elif claim.get("game_id") != player["proj"][str(projection["round"])].get("game_id"):
                reason = "wrong fixture"
            elif claim.get("kind") not in KINDS or claim["kind"] != extracted["kind"]:
                reason = "reviewed classification unsupported by excerpt/translation"
            elif claim["kind"] == "context":
                reason = "context/sentiment does not establish a minutes change"
            elif claim.get("time_scope") != "upcoming_fixture":
                reason = "statement does not explicitly concern the upcoming fixture"
            elif claim.get("already_in_inputs"):
                reason = "already incorporated in shared inputs"
            elif confidence < settings.min_confidence or weight < settings.min_effective_weight:
                reason = "insufficient source authority, specificity or freshness"
        except (KeyError, TypeError, ValueError) as exc:
            reason = f"invalid evidence metadata: {exc}"
        if reason:
            row["reason"] = reason
        else:
            row["decision"] = "eligible"
            accepted[(player["club"], player["name"])].append(row)
        audit.append(row)
    # Same originating interview/event is one piece of evidence regardless of
    # how many websites repeat or translate it. Precision-weighted pooling would
    # imply independence we do not have; use the strongest eligible statement.
    chosen = {}
    for identity, rows in accepted.items():
        events = defaultdict(list)
        for row in rows:
            events[row["event_id"]].append(row)
        deduped = []
        for event_rows in events.values():
            kinds = {r["kind"] for r in event_rows}
            if ("out" in kinds and kinds != {"out"}) or {"role_up", "role_down"} <= kinds:
                for row in event_rows:
                    row.update(decision="rejected", reason="conflicting claims from the same event")
                continue
            winner = max(event_rows, key=lambda r: (r["effective_weight"], r["observed_at"]))
            deduped.append(winner)
            for row in event_rows:
                if row is not winner:
                    row.update(decision="rejected", reason="duplicate originating event")
        kinds = {r["kind"] for r in deduped}
        if ("out" in kinds and kinds != {"out"}) or {"role_up", "role_down"} <= kinds:
            for row in deduped:
                row.update(
                    decision="rejected",
                    reason="unresolved conflicting sources; keep existing forecast",
                )
            continue
        if deduped:
            winner = max(deduped, key=lambda r: (r["effective_weight"], r["observed_at"]))
            winner["decision"] = "selected"
            chosen[identity] = winner
            for row in deduped:
                if row is not winner:
                    row.update(
                        decision="rejected",
                        reason="stronger independent statement selected; confidence not compounded",
                    )
    return chosen, audit


def _record(player, forecast, season_model, target):
    fixture = next(g for g in season_model.games.values() if g["identifier"] == forecast["game_id"])
    previous = [
        utc(g["utcDate"])
        for g in season_model.games.values()
        if player["club"] in (g["local"]["club"]["code"], g["road"]["club"]["code"])
        and utc(g["utcDate"]) < utc(fixture["utcDate"])
    ]
    return {
        "player_id": player["code"],
        "club": player["club"],
        "pos": player["pos"],
        "season": season_model.season.code,
        "date": fixture["utcDate"],
        "home": forecast["home"],
        "opp": forecast["opp"],
        "expected_margin": forecast["margin"],
        "seed_minutes": player["base_min"],
        "rest_days": (utc(fixture["utcDate"]) - max(previous)).total_seconds() / 86400
        if previous
        else 7,
        "round": target,
    }


def apply_news(base, fitted, season_model, snapshot, settings, as_of):
    """Keep future forecasts/prices/coaches intact; recompose current PIR moments."""
    result = copy.deepcopy(base)
    # Fixture identity is added only to this copied experiment's metadata.
    target, rnd = base["round"], str(base["round"])
    for p in result["players"]:
        q = p["proj"].get(rnd)
        if q is not None:
            fixture = next(
                g
                for g in season_model.games.values()
                if g["round"] == target
                and p["club"] in (g["local"]["club"]["code"], g["road"]["club"]["code"])
            )
            q["game_id"] = fixture["identifier"]
    chosen, evidence_audit = select_evidence(snapshot, result, as_of, settings)
    original = {p["name"]: p for p in base["players"]}
    affected, restricted, changes = set(), set(), []
    for p in result["players"]:
        claim = chosen.get((p["club"], p["name"]))
        if claim is None:
            continue
        q = p["proj"][rnd]
        if "independent" not in q:
            claim.update(
                decision="rejected", reason="hierarchical identity fallback has no fitted moments"
            )
            continue
        k, weight = claim["kind"], claim["effective_weight"]
        mu, var, play = q["min"], q["independent"]["minute_sd"] ** 2, q["play"]
        # Explicit novelty flag is required to revise researched injury inputs.
        if (
            k in {"ready", "doubtful", "out"}
            and key(p["club"], p["name"]) in season_model.round_inputs(target).availability
            and not claim.get("supersedes_existing")
        ):
            claim.update(
                decision="rejected",
                reason="existing availability input; supersession is not established",
            )
            continue
        if k == "out":
            play = 0.0
            restricted.add(p["name"])
        elif k == "ready":
            play += weight * max(settings.ready_probability - play, 0)
        elif k == "doubtful":
            play -= weight * max(play - settings.doubtful_probability, 0)
        elif k == "minutes_cap":
            cap = claim["extracted"]["value"]
            sd = math.sqrt(max(var, 1e-6))
            bounded_mu, bounded_var = truncnorm.stats(
                -mu / sd, (cap - mu) / sd, loc=mu, scale=sd, moments="mv"
            )
            mu, var = min(mu, float(bounded_mu)), float(bounded_var)
            restricted.add(p["name"])
        elif k == "minutes_estimate":
            var = max(var, 1e-6)
            precision = weight / settings.explicit_minutes_sd**2
            mu = (mu / var + claim["extracted"]["value"] * precision) / (1 / var + precision)
            var = 1 / (1 / var + precision)
        elif k in {"role_up", "role_down"}:
            mu += (1 if k == "role_up" else -1) * settings.max_directional_shift * weight
            var += settings.added_minute_sd**2 * weight
        elif k == "limited":
            # A non-numeric restriction widens uncertainty; it cannot supply a cap.
            var += settings.added_minute_sd**2 * weight
        q.update(min=min(max(mu, 0), 40), play=play)
        q["independent"]["minute_sd"] = math.sqrt(var)
        q["news_evidence_ids"] = [claim["id"]]
        affected.add(p["name"])
    # Conserve a team's expected minutes when new evidence changes its rotation.
    # Only half of a reduction is redistributed, by position and headroom. The
    # remainder is left to unpriced/unmodelled bench players, not manufactured PIR.
    direct = affected.copy()
    direct_delta = {
        p["name"]: p["proj"][rnd]["min"] * p["proj"][rnd]["play"]
        - original[p["name"]]["proj"][rnd]["min"] * original[p["name"]]["proj"][rnd]["play"]
        for p in result["players"]
        if p["name"] in direct
    }
    for club in {p["club"] for p in result["players"] if p["name"] in direct}:
        roster = [p for p in result["players"] if p["club"] == club and rnd in p["proj"]]
        for pos in {p["pos"] for p in roster if p["name"] in direct}:
            delta = sum(
                direct_delta[p["name"]] for p in roster if p["pos"] == pos and p["name"] in direct
            )
            recipients = [
                p
                for p in roster
                if p["name"] not in direct
                and p["proj"][rnd]["play"] >= 0.35
                and "independent" in p["proj"][rnd]
            ]
            weights = [
                min(p["proj"][rnd]["min"] / 12, 1)
                * (1 if p["pos"] == pos else 0.25)
                * (max(36 - p["proj"][rnd]["min"], 0) if delta < 0 else p["proj"][rnd]["min"])
                * p["proj"][rnd]["play"]
                for p in recipients
            ]
            total = sum(weights)
            if total:
                for p, w in zip(recipients, weights, strict=True):
                    if w == 0 or abs(delta) < 1e-10:
                        continue
                    q = p["proj"][rnd]
                    adjustment = (
                        -delta
                        * (settings.redistribution_fraction if delta < 0 else 1)
                        * w
                        / total
                        / q["play"]
                    )
                    q["min"] = min(max(q["min"] + adjustment, 0), 36)
                    q["news_evidence_ids"] = sorted(
                        {
                            cid
                            for other in roster
                            for cid in other["proj"][rnd].get("news_evidence_ids", [])
                        }
                    )
                    affected.add(p["name"])
        total = sum(p["proj"][rnd]["min"] * p["proj"][rnd]["play"] for p in roster)
        if total > 200:
            for p in roster:
                q = p["proj"][rnd]
                if "independent" in q:
                    scale = 200 / total
                    q["min"] *= scale
                    q["independent"]["minute_sd"] *= scale
                    affected.add(p["name"])
    for p in result["players"]:
        q = p["proj"].get(rnd)
        if p["name"] not in affected:
            if q is not None and "game_id" not in original[p["name"]]["proj"][rnd]:
                q.pop("game_id", None)
            continue
        before = original[p["name"]]["proj"][rnd]
        pred = fitted.predict(
            _record(p, q, season_model, target), q["min"], q["independent"]["minute_sd"] ** 2
        )
        bonus = 1 + season_model.rules.win_bonus * q["win"]
        fantasy, sd = max(pred["pir"], 0.25) * bonus, pred["pir_sd"] * bonus
        q.update(
            if_plays=fantasy,
            ev=q["play"] * fantasy,
            raw_ev=q["play"] * fantasy,
            independent={
                **pred,
                "fantasy_sd_if_available": sd,
                "p10_if_available": fantasy + norm.ppf(0.1) * sd,
                "p90_if_available": fantasy + norm.ppf(0.9) * sd,
            },
        )
        p["sd_per_pir"] = sd / max(fantasy, 0.25)
        changes.append(
            {
                "player": p["name"],
                "club": p["club"],
                "minutes_before": before["min"],
                "minutes_after": q["min"],
                "play_before": before["play"],
                "play_after": q["play"],
                "ev_before": before["ev"],
                "ev_after": q["ev"],
                "evidence_ids": q.get("news_evidence_ids", []),
                "direct": (p["club"], p["name"]) in chosen,
                "conditional_minute_sd_before": before["independent"]["minute_sd"],
                "conditional_minute_sd_after": pred["minute_sd"],
            }
        )
    result["experiment"] = {
        "name": EXPERIMENT,
        "parent": base.get("experiment"),
        "status": "experimental",
        "as_of": as_of.isoformat(),
        "settings": asdict(settings),
        "sentiment_used_for_minutes": False,
        "player_props_used": False,
    }
    return result, {
        "experiment": EXPERIMENT,
        "round": target,
        "as_of": as_of.isoformat(),
        "claims": evidence_audit,
        "changes": changes,
        "updated_players": len(changes),
        "selected_claims": sum(r["decision"] == "selected" for r in evidence_audit),
        "scope": snapshot.get("scope", []),
        "limitations": snapshot.get("limitations", []),
        "validation_status": "unvalidated numerical mappings; freeze for prospective evaluation",
    }
