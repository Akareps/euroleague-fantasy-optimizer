"""``elfantasy season ...``: the round-by-round workflow.

A round, start to finish::

    elfantasy season fetch                 # schedule + box scores of finished rounds
    # edit seasons/<name>/rounds/rNN.yaml: availability, betting lines
    # edit seasons/<name>/private/rNN.yaml: squad, bank, prices from the app
    elfantasy season project --round N     # projections for N and the next rounds
    elfantasy season plan --round N        # transfers, lineup, Turn plan
    elfantasy season compare --round N     # the named plans in the private file
    elfantasy season archive --lines N     # snapshot odds, news, prices; print round N's lines
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


def _archive_projection(data: SeasonData, round_no: int, result: dict) -> None:
    """A dated copy for later: the files in out/ are overwritten by each run."""
    from elfantasy.season.archive import Archive

    Archive(data.root / "archive").raw(
        "projections",
        f"round_{round_no:02d}",
        json.dumps(result, ensure_ascii=False, default=float),
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
    """Download the schedule, the box scores of every finished round and
    today's rosters; rebuild the player pool from them and report changes."""

    _, season, data = _context(season_dir, config)
    try:
        if prior:
            report.console.print(f"prior: saved {', '.join(data.fetch_prior(season))}")
        report.console.print(f"schedule: {data.fetch_games(season)} games")
        report.console.print(f"rosters: {data.refresh_rosters(season)} registrations")
        if (data.prior_dir / "prices_round_01.csv").is_file():
            n, unmatched, changes = data.join_prior(season)
            report.console.print(
                f"player pool: {n} players and coaches; opening-list names not on a roster: "
                f"{', '.join(unmatched) or '-'}"
            )
            for label, names in changes.items():
                if names:
                    report.console.print(f"  {label}: {'; '.join(names)}")
        by_round: dict[int, list[bool]] = {}
        for g in data.games().values():
            by_round.setdefault(int(g["round"]), []).append(bool(g["played"]))
        for rnd, done in sorted(by_round.items()):
            if not all(done):
                continue
            if data.box_file(rnd).is_file() and not refetch:
                continue
            report.console.print(f"round {rnd}: {data.fetch_boxes(season, rnd)} box-score lines")
        gaps = data.pool_gaps()
        if gaps:
            report.console.print(
                f"[red]played recently but missing from the pool (or under another club): "
                f"{'; '.join(gaps)}[/]"
            )
    except SeasonDataError as exc:
        _fail(exc)


def _day(iso: str) -> str:
    return date.fromisoformat(iso[:10]).strftime("%a")


def _print_tips(data: SeasonData, clubs: dict[str, str], result: dict) -> None:
    """Basketball Sphere's picks for the round against our forecast; the ones
    we rank low first (a reason to check the news before locking the team)."""
    from elfantasy.season.archive import Archive
    from elfantasy.season.tips import LOW_RANK, card_rows, round_tips

    round_no = int(result["round"])
    picks = round_tips(Archive(data.root / "archive").read("tips"), round_no)
    if not picks:
        report.console.print(
            f"[dim]No tipster picks archived for round {round_no} yet: "
            "`elfantasy season archive --only tips`[/]"
        )
        return
    rows = card_rows(result, picks, clubs)
    table = Table(
        title=f"Round {round_no}: Basketball Sphere's picks against our forecast "
        f"(captured {picks[0]['captured_at'][:16].replace('T', ' ')} UTC)",
        header_style="bold",
    )
    for col in ("P", "Player", "Club", "Price", f"Our R{round_no}", "Our rank", "Note"):
        table.add_column(
            col, justify="right" if col in ("Price", f"Our R{round_no}", "Our rank") else "left"
        )
    order = {"G": 0, "F": 1, "C": 2, "HC": 3}
    for r in sorted(rows, key=lambda r: (not r["note"], order.get(r["pos"], 4), r["rank"] or 99)):
        table.add_row(
            r["pos"], r["player"], r["club"], f"{r['price']:.1f}" if r["price"] else "-",
            f"{r['ev']:.1f}" if r["ev"] is not None else "-",
            str(r["rank"]) if r["rank"] else "-", r["note"],
        )  # fmt: skip
    report.console.print(table)
    report.console.print(
        f"[dim]Our rank is at the position (coaches among coaches); a pick outside our top "
        f"{LOW_RANK} is flagged. In the backtest their picks beat our forecasts; whether to add a "
        "bump is a candidate there (`elfantasy season backtest`).[/]"
    )


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
    _archive_projection(data, round_no, result)

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
    _print_tips(data, season.clubs, result)
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
    _archive_projection(data, round_no, result)
    planner = RoundPlanner(result, rules, private, gamma=season.gamma,
                           credit_per_round=season.credit_per_round,
                           credit_weight=season.credit_weight, last_round=season.last_round,
                           log=lambda m: report.console.print(f"[dim]{m}[/]"))  # fmt: skip
    return planner, data, private, season


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

    planner, data, _, season = _planner(round_no, season_dir, config)
    try:
        result = planner.search(n_search=2000 if quick else 6000, n_final=10000 if quick else 40000)
    except ValueError as exc:
        _fail(exc)
    planner.show(result)
    _print_tips(data, season.clubs, planner.proj)
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

    planner, _, private, _ = _planner(round_no, season_dir, config)
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

    from dataclasses import asdict

    import numpy as np

    from elfantasy.season.backtest import SHORT_REST_DAYS, run_backtest
    from elfantasy.season.model import SeasonModel

    rules, season, data = _context(season_dir, config)
    try:
        scores = run_backtest(SeasonModel(season, data, rules), through, top_k=top)
    except (SeasonDataError, SeasonInputError) as exc:
        _fail(exc)
    table = Table(title="Chronological backtest", header_style="bold")
    for col in ("Round", "n", "Model RMSE", "Price RMSE", "Model rho", "Price rho",
                f"Top-{top} model", f"Top-{top} price", f"Top-{top} mix", f"Top-{top} mix+opp",
                f"Top-{top} hindsight"):  # fmt: skip
        table.add_column(col, justify="right")
    for s in scores:
        p, o = s.price, s.opp_adjusted
        table.add_row(
            f"{s.round}{'*' if s.tuned_on else ''}", str(s.n), f"{s.model.rmse:.2f}",
            f"{p.rmse:.2f}" if p else "-", f"{s.model.rho:.3f}", f"{p.rho:.3f}" if p else "-",
            f"{s.model.top:.1f}", f"{p.top:.1f}" if p else "-",
            f"{s.blend.top:.1f}" if s.blend else "-", f"{o.top:.1f}" if o else "-",
            f"{s.hindsight_top:.1f}",
        )  # fmt: skip
    report.console.print(table)
    report.console.print(
        "[dim]Model = calibrated on earlier rounds only; price = actual ~ price fitted on earlier "
        "rounds (none for round 1); mix = the forecast used: the two blended by price_blend, with "
        "the position-and-price correction; mix+opp = the mix plus each opponent's record against "
        "the position (a candidate, not used in projections). * = a round used to choose the "
        "evidence weights.[/]"
    )
    scored = [s for s in scores if s.blend and s.opp_adjusted]
    if scored:

        def mean(f):
            return float(np.mean([f(s) for s in scored]))

        report.console.print(
            f"Position-and-price correction, rounds {scored[0].round}-{scored[-1].round}: top-{top} "
            f"{mean(lambda s: s.plain_blend.top):.2f} -> {mean(lambda s: s.blend.top):.2f}, RMSE "
            f"{mean(lambda s: s.plain_blend.rmse):.3f} -> {mean(lambda s: s.blend.rmse):.3f}, rho "
            f"{mean(lambda s: s.plain_blend.rho):.3f} -> {mean(lambda s: s.blend.rho):.3f}"
        )
        report.console.print(
            f"Opponent-vs-position candidate, rounds {scored[0].round}-{scored[-1].round}: top-{top} "
            f"{mean(lambda s: s.blend.top):.2f} -> {mean(lambda s: s.opp_adjusted.top):.2f}, RMSE "
            f"{mean(lambda s: s.blend.rmse):.3f} -> {mean(lambda s: s.opp_adjusted.rmse):.3f}, rho "
            f"{mean(lambda s: s.blend.rho):.3f} -> {mean(lambda s: s.opp_adjusted.rho):.3f}"
        )

    def line(e):
        return f"{np.mean(e):+.2f} (n={len(e)}, se {np.std(e) / np.sqrt(len(e)):.2f})"

    tips = [s for s in scores if s.tipster and s.tipster_dated and s.form and s.form_tipster]
    if tips:

        def avg(f):
            return float(np.mean([f(s) for s in tips]))

        span = f"rounds {tips[0].round}-{tips[-1].round}"
        for label, get in (
            ("Tipster bump, all articles (exploratory: most edited after the games)", lambda s: s.tipster),
            ("Tipster bump, dated picks only (known before tip-off)", lambda s: s.tipster_dated),
            ("Recent form", lambda s: s.form),
            ("Form + tipster (all articles)", lambda s: s.form_tipster),
        ):  # fmt: skip
            report.console.print(
                f"{label} candidate, {span}: top-{top} {avg(lambda s: s.blend.top):.2f} -> "
                f"{avg(lambda s, g=get: g(s).top):.2f}, RMSE {avg(lambda s: s.blend.rmse):.3f} -> "
                f"{avg(lambda s, g=get: g(s).rmse):.3f}, rho {avg(lambda s: s.blend.rho):.3f} -> "
                f"{avg(lambda s, g=get: g(s).rho):.3f}"
            )

    def clustered(attr):
        """Mean over picks, with the round-clustered standard error."""
        per = [getattr(s, attr) for s in scores if getattr(s, attr)]
        if not per:
            return "none yet"
        means = [float(np.mean(e)) for e in per]
        se = float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else float("nan")
        n = sum(len(e) for e in per)
        return f"{np.mean([x for e in per for x in e]):+.2f} ({n} picks, {len(per)} rounds, se {se:.2f})"

    if any(s.tip_excess for s in scores):
        report.console.print(
            "Tipster picks, actual - forecast beyond unpicked peers: "
            f"all {clustered('tip_excess')}; dated before tip-off by the publisher "
            f"{clustered('tip_excess_dated')}; archived by us before tip-off "
            f"{clustered('tip_excess_observed')}; all, after recent form "
            f"{clustered('tip_excess_after_form')}"
        )
    short = [e for s in scores for e in s.short_rest]
    normal = [e for s in scores for e in s.normal_rest]
    if short and normal:
        report.console.print(
            f"Projected starters, actual - forecast: on {SHORT_REST_DAYS} days' rest or less "
            f"{line(short)}; with more rest {line(normal)}"
        )
    if save:
        save.write_text(json.dumps([asdict(s) for s in scores], indent=1), encoding="utf-8")


@app.command()
def archive(
    only: list[str] = typer.Option(
        None,
        "--only",
        help="Collect only these: odds, domestic, news, prices, tips, pbp (repeatable).",
    ),
    lines: int = typer.Option(
        None, "--lines", help="Afterwards print this round's lines and totals for its round file."
    ),
    season_dir: Path = SeasonDirOpt,
    config: Path = ConfigOpt,
) -> None:
    """Snapshot outside sources (odds, news, prices, play-by-play) for later backtests."""

    from elfantasy.season.archive import COLLECTORS, Archive, collect, round_lines

    _, season, data = _context(season_dir, config)
    unknown = set(only or ()) - set(COLLECTORS)
    if unknown:
        _fail(SeasonInputError(f"unknown collector(s): {', '.join(sorted(unknown))}"))
    store = Archive(data.root / "archive")
    try:
        games = data.games()
    except SeasonDataError as exc:
        _fail(exc)
    collect(store, games, season.code, only=only,
            log=lambda m: report.console.print(f"[dim]{m}[/]"))  # fmt: skip
    report.console.print(f"[dim]Archived in {store.root}[/]")
    if lines is not None:
        money, totals = round_lines(store, lines)
        print(f"moneylines:\n  {lines}:")
        for k, (h, a) in sorted(money.items()):
            print(f"    {k}: [{h:.2f}, {a:.2f}]")
        print(f"totals:\n  {lines}:")
        for k, v in sorted(totals.items()):
            print(f"    {k}: {v}")
