"""Separate opt-in commands; no production projection/plan files are overwritten."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.table import Table

from elfantasy import report
from elfantasy.config import REPO_ROOT
from elfantasy.experiments.hierarchical_cli import app as hierarchical_app
from elfantasy.experiments.oddspapi import (
    EXPERIMENT,
    OddsPapiClient,
    OddsPapiError,
    fetch_snapshot,
    parse_board,
    read_api_key,
)
from elfantasy.experiments.prop_blend import (
    ExperimentSettings,
    apply_snapshot,
    match_fixtures,
    projection_cutoff,
)
from elfantasy.season.cli import ConfigOpt, RoundOpt, SeasonDirOpt, _context, _fail
from elfantasy.season.data import SeasonDataError
from elfantasy.season.inputs import SeasonInputError, load_private
from elfantasy.season.model import SeasonModel, save_projection

app = typer.Typer(no_args_is_help=True, help="Opt-in experiments with separate outputs.")
odds_app = typer.Typer(no_args_is_help=True, help="OddsPapi v1: conservative player-prop blend.")
app.add_typer(odds_app, name="oddspapi")
app.add_typer(hierarchical_app, name="hierarchical")
SettingsOpt = typer.Option(None, "--settings", help="Experiment YAML (independent of season.yaml).")
SnapshotOpt = typer.Option(
    None, "--snapshot", help="Saved OddsPapi snapshot; missing = own model only."
)
AsOfOpt = typer.Option(
    None, "--as-of", help="UTC cutoff, capped at first tip-off. Default: now/deadline."
)


def _settings(path):
    default = REPO_ROOT / "experiments" / EXPERIMENT / "experiment.yaml"
    return ExperimentSettings.load(path or (default if default.is_file() else None))


def _output(data, round_no):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = data.root / "experiments" / EXPERIMENT / f"round_{round_no:02d}" / stamp
    out.mkdir(parents=True, exist_ok=False)
    return out


def _write(path, doc):
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


@odds_app.command()
def coverage(
    round_no: int = RoundOpt,
    max_fixtures: int = typer.Option(10, "--max-fixtures", min=1, max=20),
    api_key_file: Path = typer.Option(None, "--api-key-file", help="Local plain-text key file."),
    snapshot_file: Path = typer.Option(
        None, "--snapshot", help="Re-audit a saved snapshot without API calls."
    ),
    as_of: str = AsOfOpt,
    settings: Path = SettingsOpt,
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
) -> None:
    """Test live EuroLeague coverage; archive raw responses and a usable-market inventory."""
    rules, season, data = _context(season_dir, config)
    out = _output(data, round_no)
    try:
        model = SeasonModel(season, data, rules)
        cfg = _settings(settings)
        games = [g for g in model.games.values() if g["round"] == round_no]
        if snapshot_file:
            snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
        else:
            key = read_api_key(api_key_file or season.root / "private" / "oddspapi.key")
            snapshot = fetch_snapshot(OddsPapiClient(key), games, max_fixtures=max_fixtures)
        _write(out / "snapshot.json", snapshot)
        matches, unmatched = match_fixtures(snapshot, model, cfg)
        roster = json.loads((data.prior_dir / "joined.json").read_text(encoding="utf-8"))
        players = [{**p, "proj": {}} for p in roster if p["pos"] != "HC"]
        _, usable = apply_snapshot(
            {"round": round_no, "players": players},
            snapshot,
            model,
            cfg,
            projection_cutoff(model, round_no, as_of),
            audit_only=True,
        )
        rows = []
        for board in snapshot["boards"]:
            quotes, counts = parse_board(board, snapshot["catalogue"], snapshot["captured_at"])
            game = matches.get(board["fixtureId"])
            rows.append(
                {
                    "fixture_id": board["fixtureId"],
                    "game_id": game["identifier"] if game else None,
                    "counts": counts,
                    "paired_players": sorted({q.player_name for q in quotes}),
                    "paired_books": sorted({q.bookmaker for q in quotes}),
                    "paired_markets": sorted({"+".join(q.components) for q in quotes}),
                }
            )
        summary = {
            "experiment": EXPERIMENT,
            "round": round_no,
            "status": "partial" if snapshot["errors"] else "fetched",
            "captured_at": snapshot["captured_at"],
            "games_in_round": len(games),
            "fixtures_discovered": len(snapshot["fixtures"]),
            "boards_returned": len(snapshot["boards"]),
            "unmatched_fixtures": unmatched,
            "errors": snapshot["errors"],
            "fixtures": rows,
            "usable_before_deadline": usable,
            "note": "Usable counts apply time, freshness, margin and roster identity checks. "
            "The denominator is the priced roster, not an assumed healthy rotation. "
            "Unpaired ladders do not change forecasts.",
        }
        _write(out / "coverage.json", summary)
    except (OddsPapiError, SeasonDataError, ValueError, OSError) as exc:
        _write(
            out / "coverage.json",
            {
                "experiment": EXPERIMENT,
                "round": round_no,
                "status": "not_tested",
                "reason": str(exc),
            },
        )
        report.console.print(f"[dim]Coverage attempt saved in {out}[/]")
        _fail(exc)
    table = Table(title=f"OddsPapi experiment: Round {round_no} coverage")
    for col in ("Fixture", "Matched", "Players", "Paired quotes", "Unpaired"):
        table.add_column(col)
    for row in rows:
        c = row["counts"]
        table.add_row(
            row["fixture_id"],
            row["game_id"] or "no",
            str(len(row["paired_players"])),
            str(c.get("paired_quotes", 0)),
            str(c.get("unpaired_markets", 0)),
        )
    report.console.print(table)
    report.console.print(
        f"{usable['coverage'].get('players_with_usable_quotes', 0)} matched players with usable "
        f"quotes before the deadline; status: {usable['status']}"
    )
    report.console.print(f"[dim]Saved snapshot and coverage in {out}[/]")


def _prepare(round_no, snapshot_path, as_of, settings, season_dir, config):
    rules, season, data = _context(season_dir, config)
    try:
        model = SeasonModel(season, data, rules)
        cfg = _settings(settings)
        cutoff = projection_cutoff(model, round_no, as_of)
        private = load_private(season.root, round_no)
        baseline = model.project(round_no, private)
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path else None
        experiment, audit = apply_snapshot(baseline, snapshot, model, cfg, cutoff)
    except (SeasonDataError, SeasonInputError, ValueError, OSError) as exc:
        _fail(exc)
    out = _output(data, round_no)
    experiment["experiment"] = {
        "name": EXPERIMENT,
        "status": "experimental",
        "as_of": cutoff.isoformat(),
        "settings": asdict(cfg),
        "snapshot": str(snapshot_path) if snapshot_path else None,
    }
    save_projection(baseline, out / "baseline.json")
    save_projection(experiment, out / "projection.json")
    _write(out / "comparison.json", audit)
    return model, rules, private, baseline, experiment, audit, out


@odds_app.command()
def project(
    round_no: int = RoundOpt,
    snapshot: Path = SnapshotOpt,
    as_of: str = AsOfOpt,
    settings: Path = SettingsOpt,
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
) -> None:
    """Project using eligible props; use the unchanged own-model forecast for every gap."""
    _, _, _, _, _, audit, out = _prepare(round_no, snapshot, as_of, settings, season_dir, config)
    table = Table(title=f"OddsPapi experiment: Round {round_no} changes")
    for col in ("Player", "Own model", "Experiment", "Change"):
        table.add_column(col)
    for change in sorted(
        audit["changes"], key=lambda c: -abs(c["experiment_ev"] - c["baseline_ev"])
    ):
        table.add_row(
            change["player"],
            f"{change['baseline_ev']:.2f}",
            f"{change['experiment_ev']:.2f}",
            f"{change['experiment_ev'] - change['baseline_ev']:+.2f}",
        )
    report.console.print(table)
    report.console.print(f"{audit['updated_players']} players adjusted; status: {audit['status']}")
    report.console.print(f"[dim]Baseline, experiment and comparison saved in {out}[/]")


@odds_app.command()
def plan(
    round_no: int = RoundOpt,
    quick: bool = typer.Option(False, "--quick", help="Use fewer simulations for a draft."),
    snapshot: Path = SnapshotOpt,
    as_of: str = AsOfOpt,
    settings: Path = SettingsOpt,
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
) -> None:
    """Run the existing planner on experiment projections; save a separate experiment plan."""
    from elfantasy.season.planner import RoundPlanner

    model, rules, private, _, experiment, _, out = _prepare(
        round_no, snapshot, as_of, settings, season_dir, config
    )
    planner = RoundPlanner(
        experiment,
        rules,
        private,
        gamma=model.season.gamma,
        log=lambda m: report.console.print(f"[dim]{m}[/]"),
    )
    try:
        result = planner.search(n_search=2000 if quick else 6000, n_final=10000 if quick else 40000)
    except ValueError as exc:
        _fail(exc)
    planner.show(result)
    planner.save(result, out / "plan.json")
    doc = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    _write(out / "plan.json", {**doc, "experiment": experiment["experiment"]})
    report.console.print(f"[dim]Experiment plan saved in {out}[/]")


@odds_app.command()
def evaluate(
    round_no: int = RoundOpt,
    snapshot: Path = SnapshotOpt,
    as_of: str = AsOfOpt,
    settings: Path = SettingsOpt,
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
) -> None:
    """Score a finished round on the same player universe; reject quotes observed too late."""
    from elfantasy.season.backtest import _score

    model, _, _, baseline, experiment, audit, out = _prepare(
        round_no, snapshot, as_of, settings, season_dir, config
    )
    if round_no not in model.data.played_rounds():
        _fail(ValueError(f"Round {round_no} is not finished with box scores available"))
    actual = model.actuals(round_no)
    act, own, market, rows = [], [], [], []
    other = {(p["club"], p["code"], p["name"]): p for p in experiment["players"] if p.get("code")}
    boxes = model.data.boxes([round_no])[round_no]
    for p in baseline["players"]:
        r = p["proj"].get(str(round_no))
        if not p.get("code") or not r or r["play"] < 0.5:
            continue
        a = actual.get(p["code"], 0.0)
        e = other[(p["club"], p["code"], p["name"])]["proj"][str(round_no)]["ev"]
        act.append(a)
        own.append(r["ev"])
        market.append(e)
        rows.append(
            {
                "player": p["name"],
                "club": p["club"],
                "code": p["code"],
                "own_ev": r["ev"],
                "experiment_ev": e,
                "actual": a,
                "projected_minutes": r["min"],
                "actual_minutes": boxes.get(p["code"], {}).get("minutes", 0.0),
            }
        )
    evaluation = {
        "experiment": EXPERIMENT,
        "round": round_no,
        "n": len(act),
        "own_model": asdict(_score(own, act, 40)),
        "experiment_model": asdict(_score(market, act, 40)),
        "updated_players": audit["updated_players"],
        "status": audit["status"],
        "tuned_on": round_no in model.season.tuned_on,
        "players": rows,
        "note": "Replayed current season inputs; own calibration uses earlier rounds. "
        "Round scores do not reconstruct the manager's actual Turn decisions.",
    }
    _write(out / "evaluation.json", evaluation)
    for label in ("own_model", "experiment_model"):
        s = evaluation[label]
        report.console.print(
            f"{label}: RMSE {s['rmse']:.3f}, rank correlation {s['rho']:.3f}, "
            f"top-40 actual {s['top']:.2f}"
        )
    report.console.print(f"[dim]Round assessment saved in {out}[/]")
