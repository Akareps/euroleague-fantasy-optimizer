"""An additive market experiment around the unchanged SeasonModel.

Keep the baseline PIR residual (unpriced components and calibration). Change
only the priced positive components. Availability, minutes, coaches, prices,
future-round forecasts and simulation volatility still come from our model.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from elfantasy.data.injuries import normalise_name
from elfantasy.experiments.oddspapi import (
    COMPONENTS,
    EXPERIMENT,
    PropQuote,
    game_time,
    name_key,
    parse_board,
    utc,
)
from elfantasy.projection.market import prop_to_mean
from elfantasy.season.model import SeasonModel


@dataclass(frozen=True)
class ExperimentSettings:
    market_weight: float = 0.35
    max_pir_shift: float = 6.0
    max_quote_age_hours: float = 36.0
    component_prior_minutes: float = 120.0
    team_aliases: dict[str, str] = field(default_factory=dict)
    player_aliases: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.market_weight <= 1:
            raise ValueError("market_weight must be between 0 and 1")
        for value in (self.max_pir_shift, self.max_quote_age_hours, self.component_prior_minutes):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("shift, age and prior minutes must be finite and positive")

    @classmethod
    def load(cls, path: Path | None = None):
        if path is None:
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in raw.items() if k in fields})


def match_fixtures(snapshot: dict, model: SeasonModel, settings: ExperimentSettings):
    """Both teams and start time must match uniquely. Never match on a surname/date alone."""
    aliases = {normalise_name(n): c for n, c in model.season.clubs.items()}
    aliases.update({normalise_name(n): c for n, c in settings.team_aliases.items()})
    for g in model.games.values():
        for side in ("local", "road"):
            club = g[side]["club"]
            for n in (club["code"], club.get("name"), club.get("abbreviatedName")):
                if n:
                    aliases[normalise_name(n)] = club["code"]
    matched, unmatched = {}, []
    for f in snapshot.get("fixtures", []):
        codes = []
        for side in (1, 2):
            names = [
                f.get(f"participant{side}{suffix}", "") for suffix in ("Name", "ShortName", "Abbr")
            ]
            codes.append(
                next(
                    (aliases[normalise_name(n)] for n in names if normalise_name(n) in aliases),
                    None,
                )
            )
        try:
            start = utc(f["startTime"])
            candidates = [
                g
                for g in model.games.values()
                if g["local"]["club"]["code"] == codes[0]
                and g["road"]["club"]["code"] == codes[1]
                and abs((game_time(g) - start).total_seconds()) <= 7200
            ]
        except (KeyError, ValueError, TypeError):
            candidates = []
        if len(candidates) == 1:
            matched[str(f["fixtureId"])] = candidates[0]
        else:
            unmatched.append(str(f.get("fixtureId", "unknown")))
    return matched, unmatched


def component_rates(model: SeasonModel, target: int, players: list[dict], prior_minutes: float):
    """Per-minute P/R/A from last season and strictly earlier rounds.

    Missing/short individual history shrinks to the player's position pool.
    No target-round box score is read here, even when scoring a past round.
    """
    path = model.data.prior_dir / f"stats_{model.season.previous_code}_acc.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["players"]
    aggs = {r["player"]["code"]: r for r in rows}
    fields = ("pointsScored", "totalRebounds", "assists")
    pool = {}
    for pos in "GFC":
        values = [[] for _ in fields]
        for p in players:
            r = aggs.get(p.get("code"), {})
            minutes = r.get("minutesPlayed", 0)
            if p["pos"] != pos or minutes < 300:
                continue
            for i, col in enumerate(fields):
                if col in r:
                    values[i].append(r[col] / minutes)
        defaults = {"G": (0.55, 0.10, 0.16), "F": (0.50, 0.20, 0.08), "C": (0.45, 0.29, 0.06)}[pos]
        pool[pos] = np.array(
            [float(np.median(v)) if v else defaults[i] for i, v in enumerate(values)]
        )
    earlier = model.data.boxes(range(1, target)) if target > 1 else {}
    out = {}
    for p in players:
        code = p.get("code")
        if not code:
            continue
        prior = pool[p["pos"]].copy()
        agg = aggs.get(code, {})
        history_min = float(agg.get("minutesPlayed", 0))
        if history_min > 0:
            # Cap effective old-season evidence so a role change can emerge.
            strength = min(history_min, prior_minutes)
            for i, col in enumerate(fields):
                if col in agg:
                    prior[i] = (strength * agg[col] / history_min + prior_minutes * prior[i]) / (
                        strength + prior_minutes
                    )
        sums = prior_minutes * prior
        minutes = np.full(3, prior_minutes)
        for rnd, boxes in earlier.items():
            ln = boxes.get(code)
            if not ln or ln["minutes"] <= 0:
                continue
            w = model.season.evidence.recency ** (rnd - target + 1)
            for i, col in enumerate(COMPONENTS):
                if col in ln:
                    sums[i] += w * ln[col]
                    minutes[i] += w * ln["minutes"]
        out[code] = sums / minutes
    return out


def _signals(quotes: list[PropQuote], own: np.ndarray):
    """One vote per book/family, then collapse identical copied quotes.

    Alternative lines within a book do not multiply its influence. The final
    family mean is a median, so a single strange bookmaker cannot dominate.
    """
    by_book = defaultdict(list)
    for q in quotes:
        idx = [COMPONENTS.index(c) for c in q.components]
        # Fixed, deliberately conservative dispersion assumptions in v1.
        sd = max(2.0, math.sqrt(sum(max(own[i], 1) for i in idx) * 2.0))
        try:
            mean = prop_to_mean(
                q.line,
                q.over,
                q.under,
                method="multiplicative",
                dist="poisson" if q.components in (("rebounds",), ("assists",)) else "normal",
                sd=sd,
            )
        except (ValueError, ZeroDivisionError):
            continue
        if math.isfinite(mean) and mean >= 0:
            by_book[(q.components, q.bookmaker)].append((q, mean))
    by_family = defaultdict(list)
    seen = set()
    for (family, book), entries in sorted(by_book.items()):
        signature = (family, tuple(sorted((q.line, q.over, q.under) for q, _ in entries)))
        if signature in seen:
            continue
        seen.add(signature)
        by_family[family].append((book, float(np.median([m for _, m in entries]))))
    return {
        family: {"mean": float(np.median([m for _, m in values])), "books": [b for b, _ in values]}
        for family, values in by_family.items()
    }


def constrained_components(own: np.ndarray, signals: dict):
    """Use independent constraints; a PRA quote is never added to P+R+A.

    When only a combined total is priced, the own-model proportions resolve
    its underdetermined split. Full individual coverage takes precedence over
    overlapping combined markets. No artificial extra confidence is claimed.
    """
    a, y, used = [], [], []
    for family, signal in sorted(signals.items(), key=lambda item: (len(item[0]), item[0])):
        row = [float(c in family) for c in COMPONENTS]
        if a and np.linalg.matrix_rank([*a, row]) == np.linalg.matrix_rank(a):
            continue
        trial = np.array([*a, row])
        d = np.diag(np.maximum(own, 0.5))
        trial_mean = own + d @ trial.T @ np.linalg.solve(
            trial @ d @ trial.T, np.array([*y, signal["mean"]]) - trial @ own
        )
        if np.any(trial_mean < -1e-8):
            # Contradictory markets (P > PR, for example) cannot define a
            # nonnegative stat line. Keep the earlier, simpler constraints.
            continue
        a.append(row)
        y.append(signal["mean"])
        used.append({"components": list(family), **signal})
    if not a:
        return own.copy(), used
    a = np.array(a)
    # Positive constrained adjustment toward market totals. The own forecast
    # supplies proportions for components without independent market prices.
    d = np.diag(np.maximum(own, 0.5))
    adjustment = d @ a.T @ np.linalg.solve(a @ d @ a.T, np.array(y) - a @ own)
    estimate = np.maximum(own + adjustment, 0.0)
    return estimate, used


def apply_snapshot(
    baseline: dict,
    snapshot: dict | None,
    model: SeasonModel,
    settings: ExperimentSettings,
    as_of: datetime,
    *,
    audit_only: bool = False,
) -> tuple[dict, dict]:
    """Copy and adjust current-round projections, or fall back exactly.

    Capture time is mandatory, so newly fetched quotes cannot leak into a
    historical replay just because their last price change predates tip-off.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    as_of = as_of.astimezone(timezone.utc)
    result = copy.deepcopy(baseline)
    target = int(baseline["round"])
    report = {
        "experiment": EXPERIMENT,
        "round": target,
        "as_of": as_of.isoformat(),
        "status": "no_snapshot",
        "updated_players": 0,
        "coverage": {},
        "changes": [],
        "assumptions": {
            "market_weight": settings.market_weight,
            "max_pir_shift": settings.max_pir_shift,
            "max_quote_age_hours": settings.max_quote_age_hours,
        },
    }
    if snapshot is None:
        return result, report
    if snapshot.get("schema_version") != 1 or snapshot.get("experiment") != EXPERIMENT:
        raise ValueError("Unsupported OddsPapi snapshot schema")
    report["fetch_errors"] = snapshot.get("errors", [])
    matches, unmatched = match_fixtures(snapshot, model, settings)
    report["unmatched_fixtures"] = unmatched
    counts = Counter()
    per_player = defaultdict(list)
    codes = Counter(p.get("code") for p in baseline["players"] if p.get("code"))
    by_name = defaultdict(list)
    for p in baseline["players"]:
        if not p.get("code"):
            continue
        for n in {name_key(p["name"]), name_key(p.get("api_name", p["name"]))}:
            by_name[(p["club"], n)].append(p)
    aliases = {
        (club, name_key(source)): name_key(target)
        for club, mapping in settings.player_aliases.items()
        for source, target in mapping.items()
    }
    report["ambiguous_player_codes"] = {
        code: [p["name"] for p in baseline["players"] if p.get("code") == code]
        for code, n in codes.items()
        if n > 1
    }
    capture = utc(snapshot["captured_at"])
    report["captured_at"] = capture.isoformat()
    report["status"] = "snapshot_after_cutoff" if capture > as_of else "evaluated"
    # Match names only within the two clubs of the matched game.
    for board in snapshot.get("boards", []):
        quotes, stats = parse_board(board, snapshot["catalogue"], snapshot["captured_at"])
        counts.update(stats)
        game = matches.get(str(board["fixtureId"]))
        if not game or game["round"] != target:
            counts["unmatched_or_other_round_boards"] += 1
            continue
        clubs = {game[s]["club"]["code"] for s in ("local", "road")}
        for quote in quotes:
            quoted_name = name_key(quote.player_name)
            candidates = [
                p
                for club in clubs
                for p in by_name.get((club, aliases.get((club, quoted_name), quoted_name)), [])
            ]
            if len(candidates) != 1:
                counts["unmatched_players"] += 1
                continue
            p = candidates[0]
            if codes[p["code"]] > 1:
                counts["ambiguous_player_code_quotes"] += 1
                continue
            counts["matched_quotes"] += 1
            changed = utc(quote.changed_at)
            if capture > as_of or capture >= game_time(game) or changed > capture:
                counts["after_cutoff_quotes"] += 1
                continue
            # changedAt is a price movement, not the last observation. An
            # unchanged quote observed active today can still be fresh.
            if (as_of - capture).total_seconds() > settings.max_quote_age_hours * 3600:
                counts["stale_quotes"] += 1
                continue
            # Grossly implausible margins are data errors, not forecasts.
            overround = 1 / quote.over + 1 / quote.under
            if not 0.98 <= overround <= 1.25:
                counts["invalid_margin_quotes"] += 1
                continue
            per_player[(p["club"], p["code"], p["name"])].append(quote)
            counts["usable_quotes"] += 1
    counts["players_with_usable_quotes"] = len(per_player)
    report["coverage"] = dict(counts)
    report["games_requested"] = sum(g["round"] == target for g in model.games.values())
    report["boards_returned"] = len(snapshot.get("boards", []))
    report["games_matched"] = sum(g["round"] == target for g in matches.values())
    eligible = [
        p
        for p in baseline["players"]
        if str(target) in p["proj"] and p["proj"][str(target)]["play"] >= 0.5
    ]
    report["eligible_players"] = len(eligible)
    report["priced_players"] = len(baseline["players"])
    report["priced_coverage_fraction"] = len(per_player) / max(len(baseline["players"]), 1)
    if audit_only:
        report["eligible_players"] = None  # the coverage probe does not assume player availability
        return result, report
    if not per_player:
        return result, report
    rates = component_rates(model, target, baseline["players"], settings.component_prior_minutes)
    for p in result["players"]:
        quotes = per_player.get((p["club"], p.get("code"), p["name"]), [])
        r = p["proj"].get(str(target))
        if not quotes or not r or r["play"] <= 0 or r["min"] <= 0:
            continue
        own = rates[p["code"]] * r["min"]
        signals = _signals(quotes, own)
        priced, used = constrained_components(own, signals)
        if not used or settings.market_weight == 0:
            continue
        pir_delta = (
            float(np.clip((priced - own).sum(), -settings.max_pir_shift, settings.max_pir_shift))
            * settings.market_weight
        )
        bonus = 1 + 0.10 * r["win"]
        before = r["ev"]
        if_plays = max(r["if_plays"] + pir_delta * bonus, 0.0)
        r.update(ev=r["play"] * if_plays, if_plays=if_plays)
        change = {
            "player": p["name"],
            "code": p["code"],
            "club": p["club"],
            "baseline_ev": before,
            "experiment_ev": r["ev"],
            "own_components": dict(zip(COMPONENTS, own.tolist(), strict=True)),
            "market_components": dict(zip(COMPONENTS, priced.tolist(), strict=True)),
            "constraints": used,
            "quotes": len(quotes),
            "pir_delta": pir_delta,
        }
        r["market_experiment"] = change
        report["changes"].append(change)
    report["updated_players"] = len(report["changes"])
    return result, report


def projection_cutoff(model: SeasonModel, target: int, as_of: str | None = None) -> datetime:
    deadline = min(game_time(g) for g in model.games.values() if g["round"] == target)
    cutoff = utc(as_of) if as_of else min(datetime.now(timezone.utc), deadline)
    if cutoff > deadline:
        raise ValueError(
            "as-of is after the round's transfer deadline; use a separate Turn forecast"
        )
    return cutoff
