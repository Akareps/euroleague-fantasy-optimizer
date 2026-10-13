"""Opt-in text evidence experiment; network collection and forecasts are separate."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import typer
import yaml

from elfantasy import report
from elfantasy.config import REPO_ROOT
from elfantasy.experiments.hierarchical import Model, apply_model
from elfantasy.experiments.hierarchical_cli import confirmed_pool, history_root, settings_at, write
from elfantasy.experiments.history import current_records, load_history, utc
from elfantasy.experiments.news_collection import collect as collect_sources
from elfantasy.experiments.news_minutes import EXPERIMENT, Settings, apply_news
from elfantasy.experiments.prop_blend import projection_cutoff
from elfantasy.season.cli import ConfigOpt, RoundOpt, SeasonDirOpt, _context, _fail
from elfantasy.season.inputs import load_private
from elfantasy.season.model import SeasonModel, save_projection
from elfantasy.season.planner import RoundPlanner

app = typer.Typer(no_args_is_help=True, help="Reviewed news/sentiment and minutes experiment.")
SnapshotOpt = typer.Option(..., "--snapshot", exists=True, help="Frozen news snapshot JSON.")


def output(data, target):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = data.root / "experiments" / EXPERIMENT / f"round_{target:02d}" / stamp
    out.mkdir(parents=True, exist_ok=False)
    return out


def news_settings(path=None):
    path = path or REPO_ROOT / "experiments" / EXPERIMENT / "experiment.yaml"
    return Settings(**(yaml.safe_load(path.read_text(encoding="utf-8")) or {}))


def require_live_forecast(model, target, now=None):
    """Shared manual inputs have no historical archive: never rebuild after lock."""
    generated = now or datetime.now(timezone.utc)
    deadline = min(utc(g["utcDate"]) for g in model.games.values() if g["round"] == target)
    if generated >= deadline:
        raise ValueError(
            "forecast generation is after the transfer deadline; evaluate an already frozen run instead"
        )
    return generated


def prepare(round_no, snapshot_path, season_dir=None, config=None, settings=None, as_of=None):
    rules, season, data = _context(season_dir, config)
    model = SeasonModel(season, data, rules)
    require_live_forecast(model, round_no)
    cutoff = projection_cutoff(model, round_no, as_of)
    if any(
        g["round"] < round_no and utc(g["utcDate"]).date() >= cutoff.date()
        for g in model.games.values()
    ):
        raise ValueError("as-of excludes earlier-round results required by shared inputs")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    private = load_private(season.root, round_no)
    baseline = model.project(round_no, private)
    root = history_root(data)
    history = load_history(root)
    current = current_records(model, round_no, root)
    fitted = Model(history["rows"] + current, cutoff.isoformat(), settings_at())
    independent, parent_audit = apply_model(baseline, fitted, model)
    news, audit = apply_news(independent, fitted, model, snapshot, news_settings(settings), cutoff)
    generated = require_live_forecast(model, round_no)
    out = output(data, round_no)
    for name, projection in (
        ("season_baseline", baseline),
        ("hierarchical", independent),
        ("news", news),
    ):
        save_projection(projection, out / f"{name}_projection.json")
    write(out / "news_audit.json", audit)
    write(out / "parent_audit.json", parent_audit)
    write(out / "news_snapshot.json", snapshot)
    write(out / "model.json", fitted.snapshot())
    write(out / "current_evidence.json", current)
    paths = [
        snapshot_path,
        root / "history.json",
        season.root / "season.yaml",
        REPO_ROOT / "config" / "rules.yaml",
        settings or REPO_ROOT / "experiments" / EXPERIMENT / "experiment.yaml",
        REPO_ROOT / "experiments" / "hierarchical-v1" / "experiment.yaml",
        out / "current_evidence.json",
    ]
    paths += sorted((season.root / "rounds").glob("r[0-9][0-9].yaml"))
    private_path = season.root / "private" / f"r{round_no:02d}.yaml"
    if private_path.is_file():
        paths.append(private_path)
    paths += sorted((REPO_ROOT / "src" / "elfantasy" / "experiments").glob("*.py"))
    paths += [
        REPO_ROOT / "src" / "elfantasy" / "season" / name
        for name in ("inputs.py", "model.py", "planner.py")
    ]
    paths.append(REPO_ROOT / "src" / "elfantasy" / "optimize" / "turns.py")
    paths += [out / f"{label}_projection.json" for label in ("hierarchical", "news")]
    write(
        out / "provenance.json",
        {
            "experiment": EXPERIMENT,
            "as_of": cutoff.isoformat(),
            "forecast_generated_at": generated.isoformat(),
            "snapshot_captured_at": snapshot["captured_at"],
            "sha256": {
                str(p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p): sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in paths
            },
            "fit": fitted.fit_info,
            "source": history["source"],
            "scope": "New experiment output only. Numerical text-to-minutes mappings are unvalidated hypotheses.",
        },
    )
    return model, rules, private, independent, news, audit, out


@app.command()
def collect(
    round_no: int = RoundOpt,
    manifest: Path = typer.Option(None, "--manifest", help="Reviewed YAML sources/claims."),
    fetch_pages: bool = typer.Option(
        False, "--fetch-pages", help="Optional public HTML archive; obey robots.txt."
    ),
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
):
    """Freeze reviewed excerpts and discover candidates in optional HTML/captions."""
    _, _, data = _context(season_dir, config)
    manifest = (
        manifest or REPO_ROOT / "experiments" / EXPERIMENT / "research" / f"r{round_no:02d}.yaml"
    )
    out = output(data, round_no)
    try:
        snapshot = collect_sources(manifest, out, fetch_pages=fetch_pages)
        if snapshot["round"] != round_no:
            raise ValueError("manifest belongs to another round")
    except (OSError, ValueError) as exc:
        _fail(exc)
    report.console.print(
        f"Archived {len(snapshot['sources'])} sources / {len(snapshot['claims'])} reviewed or pending claims. {len(snapshot['candidates'])} unreviewed extraction candidates."
    )
    report.console.print(f"[dim]Snapshot: {out / 'snapshot.json'}[/]")


@app.command()
def project(
    round_no: int = RoundOpt,
    snapshot: Path = SnapshotOpt,
    as_of: str = typer.Option(None, "--as-of"),
    settings: Path = typer.Option(None, "--settings"),
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
):
    """Fit parent model offline, apply only eligible new evidence, save separately."""
    try:
        *_, audit, out = prepare(round_no, snapshot, season_dir, config, settings, as_of)
    except (OSError, ValueError) as exc:
        _fail(exc)
    report.console.print(
        f"{audit['selected_claims']} actionable statements; {audit['updated_players']} player forecasts changed."
    )
    report.console.print(f"[dim]News projection and complete evidence audit: {out}[/]")


@app.command()
def compare(
    round_no: int = RoundOpt,
    snapshot: Path = SnapshotOpt,
    quick: bool = typer.Option(False, "--quick"),
    as_of: str = typer.Option(None, "--as-of"),
    settings: Path = typer.Option(None, "--settings"),
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
):
    """Compare hierarchical vs text-enhanced plans on the same app-priced pool."""
    try:
        model, rules, private, parent, news, audit, out = prepare(
            round_no, snapshot, season_dir, config, settings, as_of
        )
        if not private or not private.squad:
            raise ValueError("comparison requires the private squad and app-confirmed prices")
        projections = {"hierarchical": parent, "news": news}
        planners, searches, plans = {}, {}, {}
        identical = parent["players"] == news["players"] and parent["coaches"] == news["coaches"]
        for label, projection in projections.items():
            pool = confirmed_pool(projection, private)
            planner = RoundPlanner(
                pool,
                rules,
                private,
                gamma=model.season.gamma,
                log=lambda m, label=label: report.console.print(f"[dim]{label}: {m}[/]"),
            )
            planners[label] = planner
            path = out / f"{label}_plan.json"
            if label == "news" and identical:
                write(path, searches["hierarchical"])
            else:
                result = planner.search(
                    n_search=2000 if quick else 6000, n_final=10000 if quick else 40000
                )
                planner.save(result, path)
            doc = json.loads(path.read_text(encoding="utf-8"))
            searches[label] = doc
            plans[label + "_recommendation"] = (doc["roster"], doc["coach"])
        cross = {
            label: planner.compare(plans, seeds=(101, 202, 303), n=40000)
            for label, planner in planners.items()
        }
        scope = {(r["club"], r["player"]) for r in audit["scope"]}
        rows = []
        parent_by_name = {p["name"]: p for p in parent["players"]}
        for p in news["players"]:
            if (p["club"], p["name"]) not in scope:
                continue
            before, after = (
                parent_by_name[p["name"]]["proj"][str(round_no)],
                p["proj"][str(round_no)],
            )
            claims = [
                c
                for c in audit["claims"]
                if (c.get("club"), c.get("player")) == (p["club"], p["name"])
            ]
            rows.append(
                {
                    "player": p["name"],
                    "price": p["price"],
                    "minutes_before": before["min"],
                    "minutes_after": after["min"],
                    "play_before": before["play"],
                    "play_after": after["play"],
                    "ev_before": before["ev"],
                    "ev_after": after["ev"],
                    "reviewed_claims": len(claims),
                    "actionable_claims": sum(c["decision"] == "selected" for c in claims),
                    "evidence_status": "actionable evidence"
                    if any(c["decision"] == "selected" for c in claims)
                    else "reviewed but no new actionable evidence"
                    if claims
                    else "no player-specific claim found in scoped search",
                }
            )
        comparison = {
            "experiment": EXPERIMENT,
            "round": round_no,
            "budget": planners["hierarchical"].budget(),
            "candidate_scope": "held players and app-confirmed incoming prices",
            "planner_rounds": planners["hierarchical"].rounds,
            "same_numerical_forecasts": identical,
            "recommendations": searches,
            "cross_scores": cross,
            "player_projections": rows,
            "selected_claims": audit["selected_claims"],
            "updated_players": audit["updated_players"],
            "rescore_seeds": [101, 202, 303],
            "rescore_simulations_per_seed": 40000,
            "search_simulations": 2000 if quick else 6000,
            "finalist_simulations": 10000 if quick else 40000,
            "note": "Model expectations are not accuracy evidence. Identical forecasts reuse the same search result, avoiding artificial differences caused by Monte Carlo/search noise.",
        }
        write(out / "comparison.json", comparison)
        lines = [
            f"Round {round_no}: news/minutes experiment",
            f"Evidence cutoff: {audit['as_of']}",
            f"Budget: {comparison['budget']:.1f}; candidate pool: app-confirmed prices.",
            f"Planner rounds: {comparison['planner_rounds']}",
            f"Reviewed/pending claims: {len(audit['claims'])}; actionable: {audit['selected_claims']}; changed forecasts: {audit['updated_players']}",
            f"Numerically identical to parent: {identical}",
            "",
            "Recommended transfers:",
        ]
        for label, doc in searches.items():
            score = next(r for r in cross[label] if r["plan"] == label + "_recommendation")
            lines += [
                f"{label}: sell {', '.join(doc['sold'])}; buy {', '.join(doc['bought'])}; captain {doc['captain']}; R5 mean {score['first_round']:.2f}; bank {score['bank']:.2f}."
            ]
        lines += [
            "",
            "Player | minutes parent -> news | P(available) parent -> news | expected fantasy parent -> news | evidence",
        ]
        lines += [
            f"{r['player']} | {r['minutes_before']:.2f} -> {r['minutes_after']:.2f} | {r['play_before']:.2f} -> {r['play_after']:.2f} | {r['ev_before']:.2f} -> {r['ev_after']:.2f} | {r['evidence_status']}"
            for r in rows
        ]
        lines += ["", "Evidence decisions:"]
        lines += [
            f"{r['id']} ({r.get('player') or r.get('club')}): {r['decision']} — {r.get('reason', r['kind'])}. {r['source_url']}"
            for r in audit["claims"]
        ]
        lines += [
            "",
            "Coverage limitations:",
            *audit["limitations"],
            "",
            "Numerical evidence mappings are unvalidated hypotheses. Round 5 forecasts are frozen before tip-off; no post-game outcomes have been evaluated.",
        ]
        (out / "round_05_comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    except (OSError, ValueError) as exc:
        _fail(exc)
    report.console.print(
        f"{audit['selected_claims']} actionable statements; forecasts {'unchanged' if identical else 'updated'}."
    )
    report.console.print("\n".join(lines[8:11]))
    report.console.print(f"[dim]Separate comparison: {out / 'round_05_comparison.txt'}[/]")


@app.command()
def evaluate(
    run: Path = typer.Option(
        ..., "--run", exists=True, help="Previously frozen comparison/project directory."
    ),
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
):
    """Score frozen forecasts after all games finish; never rebuild past news inputs."""
    rules, season, data = _context(season_dir, config)
    model = SeasonModel(season, data, rules)
    try:
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        projections = {
            label: json.loads((run / f"{label}_projection.json").read_text(encoding="utf-8"))
            for label in ("hierarchical", "news")
        }
        for label in projections:
            path = run / f"{label}_projection.json"
            fingerprint_key = str(
                path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            )
            if provenance["sha256"].get(fingerprint_key) != sha256(path.read_bytes()).hexdigest():
                raise ValueError(
                    "frozen forecast changed after capture; refuse retrospective evaluation"
                )
        target = projections["news"]["round"]
        projection_cutoff(model, target, provenance["as_of"])
        require_live_forecast(model, target, utc(provenance["forecast_generated_at"]))
        games = [g for g in model.games.values() if g["round"] == target]
        if not games or not all(g["played"] for g in games):
            raise ValueError(
                "round is not complete; keep the frozen forecasts for evaluation after the games"
            )
        actual = model.actual(target)
        boxes = data.boxes([target])[target]
        rows = []
        codes = Counter(p.get("code") for p in projections["news"]["players"] if p.get("code"))
        for label, projection in projections.items():
            values = []
            for p in projection["players"]:
                if not p.get("code") or codes[p["code"]] != 1:
                    continue
                q = p["proj"].get(str(target))
                if q is None:
                    continue
                observed = boxes.get(p["code"], {})
                values.append(
                    {
                        "player": p["name"],
                        "score_error": q["ev"] - actual.get(p["code"], 0),
                        "minutes_error": q["min"] * q["play"] - observed.get("minutes", 0),
                        "actual_minutes": observed.get("minutes", 0),
                    }
                )
            if not values:
                raise ValueError("no uniquely identified players to evaluate")
            rows.append(
                {
                    "model": label,
                    "n": len(values),
                    "score_rmse": math.sqrt(
                        sum(v["score_error"] ** 2 for v in values) / len(values)
                    ),
                    "expected_minutes_rmse": math.sqrt(
                        sum(v["minutes_error"] ** 2 for v in values) / len(values)
                    ),
                    "minutes_mae": sum(abs(v["minutes_error"]) for v in values) / len(values),
                    "dnp_n": sum(v["actual_minutes"] == 0 for v in values),
                    "players": values,
                }
            )
        out = output(data, target)
        write(
            out / "frozen_evaluation.json",
            {
                "round": target,
                "frozen_run": str(run),
                "forecast_as_of": provenance["as_of"],
                "models": rows,
                "note": "All uniquely identified priced players, including DNP/known absences. Minutes are unconditional expectation vs actual minutes (including overtime). Frozen projections, never retrospectively rebuilt inputs.",
            },
        )
    except (OSError, ValueError) as exc:
        _fail(exc)
    report.console.print(f"[dim]Frozen evaluation: {out / 'frozen_evaluation.json'}[/]")
