"""``elfantasy season ...``: the round-by-round workflow.

A round, start to finish::

    elfantasy season fetch                 # schedule + box scores of finished rounds
    # edit seasons/<name>/rounds/rNN.yaml: availability, betting lines
    # edit seasons/<name>/private/rNN.yaml: squad, bank, prices from the app
    elfantasy season project --round N     # projections for N and the next rounds
    elfantasy season plan --round N        # transfers, lineup, Turn plan
    elfantasy season compare --round N     # the named plans in the private file
    # after the round: add `injured_out` to rNN.yaml, then fetch again
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer
from rich.table import Table

from elfantasy import report
from elfantasy.config import load_settings
from elfantasy.rules import GameRules
from elfantasy.season.data import SeasonData, SeasonDataError
from elfantasy.season.inputs import SeasonInputError, load_private, load_season, private_file

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Round-by-round season workflow: inputs in seasons/<season>/, downloads in the data dir.",
)

SeasonDirOpt = typer.Option(
    None, "--season-dir", help="Season inputs folder (default: the newest under seasons/)."
)
ConfigOpt = typer.Option(None, "--config", help="Config directory (default: config/).")
RoundOpt = typer.Option(..., "--round", "-r", help="The round to prepare.")


def _context(season_dir: Path | None, config: Path | None):
    try:
        settings = load_settings(config)
        season = load_season(season_dir)
    except SeasonInputError as exc:
        report.console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    return (
        GameRules.from_settings(settings),
        season,
        SeasonData.for_season(settings.data_dir, season),
    )


def _fail(exc: Exception) -> None:
    report.console.print(f"[red]{exc}[/]")
    raise typer.Exit(1) from exc


@app.command()
def fetch(
    prior: bool = typer.Option(
        False,
        "--prior",
        help="Also download rosters and last season's totals and rebuild the prior.",
    ),
    refetch: bool = typer.Option(
        False, "--refetch", help="Download box scores already on disk again."
    ),
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
) -> None:
    """Download the schedule and the box scores of every finished round."""

    _, season, data = _context(season_dir, config)
    try:
        if prior:
            report.console.print(f"prior: saved {', '.join(data.fetch_prior(season))}")
            n, unmatched = data.join_prior(season)
            report.console.print(
                f"prior: joined {n} priced players and coaches; unmatched: {unmatched or '-'}"
            )
        report.console.print(f"schedule: {data.fetch_games(season)} games")
        by_round: dict[int, list[bool]] = {}
        for g in data.games().values():
            by_round.setdefault(int(g["round"]), []).append(bool(g["played"]))
        for rnd, done in sorted(by_round.items()):
            if not all(done):
                continue
            if data.box_file(rnd).is_file() and not refetch:
                continue
            report.console.print(f"round {rnd}: {data.fetch_boxes(season, rnd)} box-score lines")
    except SeasonDataError as exc:
        _fail(exc)


def _day(iso: str) -> str:
    return date.fromisoformat(iso[:10]).strftime("%a")


@app.command()
def project(
    round_no: int = RoundOpt,
    top: int = typer.Option(30, "--top", help="Players to list."),
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
) -> None:
    """Projections for a round and the following ones (saved to the data dir)."""

    from elfantasy.season.model import SeasonModel, save_projection

    rules, season, data = _context(season_dir, config)
    try:
        model = SeasonModel(season, data, rules)
        result = model.project(round_no, load_private(season.root, round_no))
    except (SeasonDataError, SeasonInputError) as exc:
        _fail(exc)
    out = data.projections_file(round_no)
    save_projection(result, out)

    cal = Table(title="Calibration (re-projected past rounds)", header_style="bold")
    for col in ("group", "tier mean -> correction (n)"):
        cal.add_column(col)
    for grp, pts in result["calibration"].items():
        cal.add_row(grp, "   ".join(f"{x:4.1f} -> {c:+.1f} ({n})" for x, c, _, n in pts))
    report.console.print(cal)

    r0, later = str(round_no), [str(r) for r in result["rounds"][1:]]
    table = Table(title=f"Round {round_no}: top {top}", header_style="bold")
    for col in ("Price", "P", "Club", "Player", f"R{round_no}", "Day", "Min", "Opp", "Win",
                *[f"R{r}" for r in later]):  # fmt: skip
        table.add_column(
            col, justify="right" if col not in ("P", "Club", "Player", "Day", "Opp") else "left"
        )
    ranked = sorted(result["players"], key=lambda p: -p["proj"].get(r0, {}).get("ev", 0.0))
    for p in ranked[:top]:
        r = p["proj"][r0]
        table.add_row(
            f"{p['price']:.1f}", p["pos"], p["club"], p["name"], f"{r['ev']:.1f}", _day(r["date"]),
            f"{r['min']:.1f}", r["opp"], f"{r['win']:.2f}",
            *[f"{p['proj'][x]['ev']:.1f}" if x in p["proj"] else "-" for x in later],
        )  # fmt: skip
    report.console.print(table)
    coaches = Table(title="Head coaches", header_style="bold")
    for col in ("Price", "Club", "Coach", *[f"R{r}" for r in result["rounds"]]):
        coaches.add_column(col)
    for c in sorted(result["coaches"], key=lambda c: -c["proj"].get(r0, {}).get("ev", -99))[:8]:
        coaches.add_row(f"{c['price']:.1f}", c["club"], c["name"],
                        *[f"{c['proj'][str(r)]['ev']:.1f}" if str(r) in c["proj"] else "-"
                          for r in result["rounds"]])  # fmt: skip
    report.console.print(coaches)
    report.console.print(f"[dim]saved {out}[/]")


def _planner(round_no, season_dir, config):
    from elfantasy.season.model import SeasonModel, save_projection
    from elfantasy.season.planner import RoundPlanner

    rules, season, data = _context(season_dir, config)
    private = load_private(season.root, round_no)
    if private is None or not private.squad:
        report.console.print(
            f"[yellow]no squad in {private_file(season.root, round_no)}: planning a team from scratch[/]"
        )
    try:
        result = SeasonModel(season, data, rules).project(round_no, private)
    except (SeasonDataError, SeasonInputError) as exc:
        _fail(exc)
    save_projection(result, data.projections_file(round_no))
    planner = RoundPlanner(result, rules, private, gamma=season.gamma,
                           log=lambda m: report.console.print(f"[dim]{m}[/]"))  # fmt: skip
    return planner, data, private


@app.command()
def plan(
    round_no: int = RoundOpt,
    quick: bool = typer.Option(
        False, "--quick", help="Fewer simulations (a draft in under a minute)."
    ),
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
) -> None:
    """Transfers, lineup and Turn plan, from your squad in the private file."""

    planner, data, _ = _planner(round_no, season_dir, config)
    try:
        result = planner.search(n_search=2000 if quick else 6000, n_final=10000 if quick else 40000)
    except ValueError as exc:
        _fail(exc)
    planner.show(result)
    out = data.out_dir / f"plan_round_{round_no:02d}.json"
    planner.save(result, out)
    report.console.print(f"[dim]saved {out}[/]")


@app.command()
def compare(
    round_no: int = RoundOpt,
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
) -> None:
    """The `plans:` in the private file (and holding), head to head on the same simulations."""

    planner, _, private = _planner(round_no, season_dir, config)
    if private is None or not private.plans:
        _fail(SeasonInputError("add `plans:` to the private file to compare them"))
    try:
        rows = planner.compare(private.plans)
    except ValueError as exc:
        _fail(exc)
    table = Table(
        title=f"Round {round_no}: plans head to head (3 seeds x 40k simulations)",
        header_style="bold",
    )
    for col in ("Plan", "Total", "+-seeds", f"R{round_no}", "Bad day (p10)", "Cost", "Bank"):
        table.add_column(col, justify="left" if col == "Plan" else "right")
    for r in rows:
        table.add_row(r["plan"], f"{r['objective']:.1f}", f"{r['seed_sd']:.1f}", f"{r['first_round']:.1f}",
                      f"{r['p10']:.1f}", f"{r['cost']:.1f}", f"{r['bank']:+.1f}")  # fmt: skip
    report.console.print(table)
    report.console.print(
        "[dim]Total = this round + discounted later rounds + price changes + bank, on the "
        "model's scale; differences of a point or two are inside the model's own error.[/]"
    )


@app.command()
def backtest(
    through: int = typer.Option(
        None, "--through", help="Last round to score (default: all finished)."
    ),
    top: int = typer.Option(40, "--top", help="Size of the 'top picks' group."),
    save: Path = typer.Option(None, "--save", help="Also write the scores as JSON."),
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
) -> None:
    """Every finished round, projected with only what was known before it."""

    from elfantasy.season.backtest import run_backtest
    from elfantasy.season.model import SeasonModel

    rules, season, data = _context(season_dir, config)
    try:
        scores = run_backtest(SeasonModel(season, data, rules), through, top_k=top)
    except (SeasonDataError, SeasonInputError) as exc:
        _fail(exc)
    table = Table(title="Chronological backtest", header_style="bold")
    for col in ("Round", "n", "Model RMSE", "Price RMSE", "Model rho", "Price rho",
                f"Top-{top} model", f"Top-{top} price", f"Top-{top} hindsight"):  # fmt: skip
        table.add_column(col, justify="right")
    for s in scores:
        p = s.price
        table.add_row(
            f"{s.round}{'*' if s.tuned_on else ''}", str(s.n), f"{s.model.rmse:.2f}",
            f"{p.rmse:.2f}" if p else "-", f"{s.model.rho:.3f}", f"{p.rho:.3f}" if p else "-",
            f"{s.model.top:.1f}", f"{p.top:.1f}" if p else "-", f"{s.hindsight_top:.1f}",
        )  # fmt: skip
    report.console.print(table)
    report.console.print(
        "[dim]Model = calibrated on earlier rounds only; price = actual ~ price fitted on earlier "
        "rounds (none for round 1). * = a round used to choose the evidence weights.[/]"
    )
    if save:
        save.write_text(json.dumps([s.__dict__ | {"model": s.model.__dict__, "raw": s.raw.__dict__,
                                                  "price": s.price.__dict__ if s.price else None}
                                    for s in scores], indent=1), encoding="utf-8")  # fmt: skip
