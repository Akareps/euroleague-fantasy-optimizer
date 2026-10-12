"""Opt-in independent model; all outputs live under hierarchical-v1."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import typer
import yaml
from rich.table import Table

from elfantasy import report
from elfantasy.config import REPO_ROOT
from elfantasy.experiments.hierarchical import Model, Settings, apply_model
from elfantasy.experiments.hierarchical_validation import current_validation, historical_validation
from elfantasy.experiments.history import (
    EXPERIMENT,
    current_records,
    fetch_history,
    load_history,
    utc,
)
from elfantasy.experiments.prop_blend import projection_cutoff
from elfantasy.season.cli import ConfigOpt, RoundOpt, SeasonDirOpt, _context, _fail
from elfantasy.season.inputs import key, load_private
from elfantasy.season.model import SeasonModel, save_projection

app = typer.Typer(
    no_args_is_help=True, help="Independent hierarchical minutes/components experiment."
)
SettingsOpt = typer.Option(None, "--settings", help="Independent experiment YAML.")


def settings_at(path=None):
    path = path or REPO_ROOT / "experiments" / EXPERIMENT / "experiment.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return Settings(**(raw or {}))


def history_root(data):
    return data.root.parent.parent / "experiments" / EXPERIMENT / "history"


def output(data, target):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = data.root / "experiments" / EXPERIMENT / f"round_{target:02d}" / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def write(path, doc):
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare(target, season_dir=None, config=None, settings=None, as_of=None):
    if target < 1:
        raise ValueError("round must be positive")
    rules, season, data = _context(season_dir, config)
    model = SeasonModel(season, data, rules)
    cutoff = projection_cutoff(model, target, as_of)
    if any(
        g["round"] < target and utc(g["utcDate"]).date() >= cutoff.date()
        for g in model.games.values()
    ):
        raise ValueError(
            "as-of excludes earlier-round results required by the shared season inputs; choose a later cutoff"
        )
    private = load_private(season.root, target)
    baseline = model.project(target, private)
    root = history_root(data)
    history = load_history(root)
    current = current_records(model, target, root)
    fitted = Model(history["rows"] + current, cutoff.isoformat(), settings_at(settings))
    experiment, audit = apply_model(baseline, fitted, model)
    out = output(data, target)
    save_projection(baseline, out / "baseline.json")
    save_projection(experiment, out / "projection.json")
    write(out / "comparison.json", audit)
    write(out / "model.json", fitted.snapshot())
    write(out / "current_evidence.json", current)
    input_paths = [
        root / "history.json",
        season.root / "season.yaml",
        season.root / "rounds" / f"r{target:02d}.yaml",
    ]
    input_paths += sorted((season.root / "rounds").glob("r[0-9][0-9].yaml"))
    if (season.root / "private" / f"r{target:02d}.yaml").is_file():
        input_paths.append(season.root / "private" / f"r{target:02d}.yaml")
    input_paths += sorted(
        (REPO_ROOT / "src" / "elfantasy" / "experiments").glob("hierarchical*.py")
    )
    input_paths += [
        out / "current_evidence.json",
        REPO_ROOT / "src" / "elfantasy" / "experiments" / "history.py",
    ]
    input_paths += [
        REPO_ROOT / "src" / "elfantasy" / "season" / "model.py",
        REPO_ROOT / "src" / "elfantasy" / "season" / "planner.py",
        REPO_ROOT / "src" / "elfantasy" / "optimize" / "turns.py",
        REPO_ROOT / "config" / "rules.yaml",
        settings or REPO_ROOT / "experiments" / EXPERIMENT / "experiment.yaml",
    ]
    fingerprints = {
        str(p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p): sha256(
            p.read_bytes()
        ).hexdigest()
        for p in input_paths
    }
    write(
        out / "provenance.json",
        {
            "source": history["source"],
            "verification": history["verification"],
            "sha256": fingerprints,
            "fit": fitted.fit_info,
            "as_of": cutoff.isoformat(),
            "scope": "Separate experiment; no baseline or OddsPapi outputs overwritten.",
        },
    )
    return model, rules, private, baseline, experiment, fitted, audit, out


@app.command()
def fetch(round_no: int = RoundOpt, season_dir: Path = SeasonDirOpt, config: Path = ConfigOpt):
    """Import pinned historical data; verify official schedules and sampled boxes."""
    rules, season, data = _context(season_dir, config)
    root = history_root(data)
    try:
        history = fetch_history(root)
        rows = current_records(SeasonModel(season, data, rules), round_no, root, fetch=True)
    except (ValueError, OSError) as exc:
        _fail(exc)
    report.console.print(
        f"Verified {history['coverage']}; {len(history['rows'])} historical + {len(rows)} current rows."
    )
    report.console.print(f"[dim]History saved in {root}[/]")


@app.command()
def project(
    round_no: int = RoundOpt,
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
    settings: Path = SettingsOpt,
    as_of: str = typer.Option(None, "--as-of"),
):
    """Fit only earlier games and project separately from both other models."""
    try:
        *_, audit, out = prepare(round_no, season_dir, config, settings, as_of)
    except (ValueError, OSError) as exc:
        _fail(exc)
    table = Table(title=f"Independent experiment: Round {round_no}")
    for c in ("Player", "Baseline", "Independent", "Minutes before", "Minutes after"):
        table.add_column(c)
    for r in sorted(audit["changes"], key=lambda r: -r["experiment_ev"])[:25]:
        table.add_row(
            r["player"],
            f"{r['baseline_ev']:.1f}",
            f"{r['experiment_ev']:.1f}",
            f"{r['baseline_minutes']:.1f}",
            f"{r['experiment_minutes']:.1f}",
        )
    report.console.print(table)
    report.console.print(f"[dim]Saved independent forecasts and fitted parameters in {out}[/]")


def confirmed_pool(projection, private):
    if not private or not private.squad:
        raise ValueError("confirmed-price planning requires a private squad and app prices")
    return {
        **projection,
        "players": [
            p
            for p in projection["players"]
            if p["name"] in private.squad or key(p["club"], p["name"]) in private.prices
        ],
        "coaches": [
            c
            for c in projection["coaches"]
            if c["name"] == private.coach or key(c["club"], c["name"]) in private.prices
        ],
    }


@app.command()
def plan(
    round_no: int = RoundOpt,
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
    settings: Path = SettingsOpt,
    quick: bool = typer.Option(False, "--quick"),
    confirmed_prices_only: bool = typer.Option(False, "--confirmed-prices-only"),
):
    """Use the existing optimizer on independent forecasts; prices stay shared."""
    from elfantasy.season.planner import RoundPlanner

    try:
        model, rules, private, _, experiment, _, _, out = prepare(
            round_no, season_dir, config, settings
        )
        pool = confirmed_pool(experiment, private) if confirmed_prices_only else experiment
        planner = RoundPlanner(
            pool,
            rules,
            private,
            gamma=model.season.gamma,
            log=lambda m: report.console.print(f"[dim]{m}[/]"),
        )
        result = planner.search(n_search=2000 if quick else 6000, n_final=10000 if quick else 40000)
        planner.show(result)
        planner.save(result, out / "plan.json")
        doc = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        purchases = doc["bought"] if private and private.squad else doc["roster"]
        selected = [p for p in pool["players"] if p["name"] in purchases]
        prices = private.prices if private else {}
        unconfirmed = [
            {"player": p["name"], "price": p["price"]}
            for p in selected
            if key(p["club"], p["name"]) not in prices
        ]
        write(
            out / "plan.json",
            {
                **doc,
                "experiment": experiment["experiment"],
                "confirmed_prices_only": confirmed_prices_only,
                "unconfirmed_incoming_prices": unconfirmed,
            },
        )
    except (ValueError, OSError) as exc:
        _fail(exc)
    report.console.print(f"[dim]Separate experimental plan saved in {out}[/]")


@app.command()
def evaluate(
    through: int = typer.Option(4, "--through", min=1),
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
    settings: Path = SettingsOpt,
):
    """Historical rolling forecasts plus current-season baseline comparison."""
    rules, season, data = _context(season_dir, config)
    model = SeasonModel(season, data, rules)
    try:
        root = history_root(data)
        history = load_history(root)
        current = current_records(model, through + 1, root)
        cfg = settings_at(settings)
        past = historical_validation(history, settings=cfg)
        recent = current_validation(model, history, current, through, settings=cfg)
    except (ValueError, OSError) as exc:
        _fail(exc)
    out = output(data, through + 1)
    write(out / "historical_evaluation.json", past)
    write(out / "current_evaluation.json", recent)
    for label, doc in (("E2025 box-sheet replay", past), ("Current priced-roster replay", recent)):
        report.console.print(label)
        for name in ("simple" if doc is past else "baseline", "independent"):
            s = doc[name]
            report.console.print(
                f"  {name}: RMSE {s['rmse']:.3f}; minutes RMSE {s['minutes_rmse']:.3f}; top-40 per round {doc['mean_top40_per_round'][name]:.2f}"
            )
    report.console.print(f"[dim]Chronological evaluations saved in {out}[/]")


@app.command()
def compare(
    round_no: int = RoundOpt,
    snapshot: Path = typer.Option(..., "--snapshot", exists=True),
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
    settings: Path = SettingsOpt,
    quick: bool = typer.Option(False, "--quick"),
    confirmed_prices_only: bool = typer.Option(
        True, "--confirmed-prices-only/--allow-estimated-prices"
    ),
):
    """Three models, identical candidate prices/rules; common-seed plan rescore."""
    from elfantasy.experiments.prop_blend import ExperimentSettings, apply_snapshot
    from elfantasy.season.planner import RoundPlanner

    try:
        model, rules, private, baseline, independent, _, _, out = prepare(
            round_no, season_dir, config, settings
        )
        if not private or not private.squad:
            raise ValueError("three-way comparison requires a private squad")
        raw = json.loads(snapshot.read_text(encoding="utf-8"))
        odds_cfg = ExperimentSettings.load(
            REPO_ROOT / "experiments" / "oddspapi-v1" / "experiment.yaml"
        )
        odds, odds_audit = apply_snapshot(
            baseline, raw, model, odds_cfg, projection_cutoff(model, round_no)
        )
        save_projection(odds, out / "oddspapi_projection.json")
        write(out / "oddspapi_audit.json", odds_audit)
        projections = {"baseline": baseline, "oddspapi": odds, "hierarchical": independent}
        planners, plans, searches = {}, {}, {}
        for label, projection in projections.items():
            pool = confirmed_pool(projection, private) if confirmed_prices_only else projection
            planner = RoundPlanner(
                pool,
                rules,
                private,
                gamma=model.season.gamma,
                log=lambda m, label=label: report.console.print(f"[dim]{label}: {m}[/]"),
            )
            result = planner.search(
                n_search=2000 if quick else 6000, n_final=10000 if quick else 40000
            )
            planners[label] = planner
            path = out / f"{label}_plan.json"
            planner.save(result, path)
            doc = json.loads(path.read_text(encoding="utf-8"))
            plans[label + "_recommendation"] = (doc["roster"], doc["coach"])
            searches[label] = doc
        cross = {
            label: planner.compare(plans, seeds=(101, 202, 303), n=40000)
            for label, planner in planners.items()
        }
        rows = []
        by_model = {
            label: {p["name"]: p for p in proj["players"]} for label, proj in projections.items()
        }
        names = set(private.squad) | {n for doc in searches.values() for n in doc["bought"]}
        for name in sorted(names):
            row = {"player": name, "price": by_model["baseline"][name]["price"]}
            for label in projections:
                row[label] = by_model[label][name]["proj"][str(round_no)]["ev"]
            rows.append(row)
        write(
            out / "three_model_comparison.json",
            {
                "round": round_no,
                "budget": planners["baseline"].budget(),
                "candidate_scope": "current squad and app-confirmed prices"
                if confirmed_prices_only
                else "all priced roster; unconfirmed prices are estimates",
                "search_simulations": 2000 if quick else 6000,
                "finalist_simulations": 10000 if quick else 40000,
                "rescore_seeds": [101, 202, 303],
                "rescore_simulations_per_seed": 40000,
                "recommendations": searches,
                "cross_scores": cross,
                "player_projections": rows,
                "oddspapi_status": odds_audit["status"],
                "oddspapi_snapshot": str(snapshot),
                "note": "Scores are each model's expectations, not evidence of accuracy. Current/future prices and coaches are common. The existing optimizer includes discounted R7 value despite the reset after R6; do not treat small objective gains as decisive.",
            },
        )
    except (ValueError, OSError) as exc:
        _fail(exc)
    table = Table(title="Model recommendations: same app-priced candidate pool")
    for col in ("Model", "Sell", "Buy", "R5 mean", "Bank"):
        table.add_column(col)
    for label, doc in searches.items():
        r = next(r for r in cross[label] if r["plan"] == label + "_recommendation")
        table.add_row(
            label,
            ", ".join(doc["sold"]),
            ", ".join(doc["bought"]),
            f"{r['first_round']:.1f}",
            f"{r['bank']:.1f}",
        )
    report.console.print(table)
    report.console.print(f"[dim]Three-model comparison saved in {out}[/]")
