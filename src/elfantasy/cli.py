"""Command-line interface.

elfantasy demo                      # everything, on synthetic data, no network
elfantasy sync                      # fetch and cache a snapshot
elfantasy project --round 12        # ranked projections
elfantasy explain "Player 042"      # why that number
elfantasy lineup                    # roster + lineup + Turn plan, real game rules
elfantasy lineup -s my_squad.yaml   # the same, as transfers from your squad
elfantasy squad                     # full rebuild, squad only (no lineup rules)
elfantasy transfers -s my_squad.yaml  # transfers, squad only (no lineup rules)
elfantasy value                     # best points per credit
elfantasy fixtures                  # schedule difficulty over the horizon
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from elfantasy import report
from elfantasy.config import load_settings
from elfantasy.data.fantasy import load_squad_yaml, write_price_template
from elfantasy.data.sample import build_sample_dataset
from elfantasy.dataset import Dataset
from elfantasy.optimize.squad import InfeasibleError
from elfantasy.pipeline import Pipeline

app = typer.Typer(
    add_completion=False,
    help="Projection and optimisation for EuroLeague Fantasy.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=report.console, show_path=False, rich_tracebacks=True)],
    )


def _build(
    *,
    sample: bool,
    snapshot: Path | None,
    prices: Path | None,
    injuries: Path | None,
    odds_csv: Path | None,
    props_csv: Path | None,
    offline: bool,
    round_no: int | None,
    horizon: int | None,
    config: Path | None,
) -> Pipeline:
    settings = load_settings(config)
    if sample:
        return Pipeline.build(
            settings=settings,
            dataset=build_sample_dataset(),
            round_no=round_no,
            horizon_rounds=horizon,
        )
    return Pipeline.build(
        settings=settings,
        round_no=round_no,
        horizon_rounds=horizon,
        offline=offline,
        snapshot=snapshot,
        prices_csv=prices,
        injuries_yaml=injuries,
        odds_csv=odds_csv,
        props_csv=props_csv,
    )


# Shared options ------------------------------------------------------------
SampleOpt = typer.Option(False, "--sample", help="Use the built-in synthetic league (no network).")
SnapshotOpt = typer.Option(
    None, "--snapshot", help="Load a saved dataset JSON instead of fetching."
)
PricesOpt = typer.Option(None, "--prices", help="CSV of player prices.")
InjuriesOpt = typer.Option(None, "--injuries", help="YAML availability file.")
OddsOpt = typer.Option(None, "--odds-csv", help="CSV of game odds.")
PropsOpt = typer.Option(None, "--props-csv", help="CSV of player props.")
OfflineOpt = typer.Option(False, "--offline", help="Never hit the network; use cache only.")
RoundOpt = typer.Option(
    None, "--round", "-r", help="Round to optimise for. Defaults to the next one."
)
HorizonOpt = typer.Option(None, "--horizon", "-H", help="How many rounds to look ahead.")
ConfigOpt = typer.Option(None, "--config", help="Config directory (defaults to ./config).")
VerboseOpt = typer.Option(False, "--verbose", "-v", help="Debug logging.")


@app.command()
def demo(
    horizon: int = HorizonOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """Run the whole pipeline end to end on synthetic data.

    Fake players, fake numbers -- but every real code path, so it is the fastest
    way to see what the tool does and to check an install.
    """

    _setup_logging(verbose)
    pipe = _build(
        sample=True,
        snapshot=None,
        prices=None,
        injuries=None,
        odds_csv=None,
        props_csv=None,
        offline=True,
        round_no=None,
        horizon=horizon,
        config=None,
    )
    report.console.print(
        f"[dim]synthetic league - {len(pipe.dataset.players)} players, "
        f"round {pipe.start_round}, horizon {sorted(pipe.horizon)}[/]"
    )
    report.projections_table(pipe.next_round, pipe.dataset, top=15, title="Top projections")
    report.value_table(pipe.candidates(), pipe.dataset, top=12)

    solution = pipe.best_squad()
    report.squad_table(solution, pipe.dataset, title="Optimal squad (reset round)")

    from elfantasy.models import Squad

    # Build a deliberately mediocre starting squad, then show the transfer advice.
    pool = sorted(pipe.candidates(), key=lambda c: c.price)
    naive = Squad(player_ids=[c.player_id for c in _legal_naive(pool, pipe)], bank=8.0)
    if naive.player_ids:
        report.ladder_table(pipe.transfer_ladder(naive))


def _legal_naive(pool, pipe):
    """A cheap-but-legal squad, used only to demonstrate transfer advice.

    Fills each position's minimum first, then the remaining slots, so the
    starting point satisfies every constraint the optimiser will enforce.
    """

    bounds = pipe.settings.position_bounds()
    size = pipe.settings.squad_size
    cap = pipe.settings.max_per_club
    picked: list = []
    counts = {p: 0 for p in bounds}
    clubs: dict[str, int] = {}

    def take(candidate, limit_key: str) -> bool:
        lo, hi = bounds.get(candidate.position, (0, size))
        limit = lo if limit_key == "min" else hi
        if counts.get(candidate.position, 0) >= limit:
            return False
        if clubs.get(candidate.team_code, 0) >= cap:
            return False
        picked.append(candidate)
        counts[candidate.position] = counts.get(candidate.position, 0) + 1
        clubs[candidate.team_code] = clubs.get(candidate.team_code, 0) + 1
        return True

    for phase in ("min", "max"):
        for c in pool:
            if len(picked) >= size:
                break
            if c in picked:
                continue
            take(c, phase)
    return picked


@app.command()
def sync(
    out: Path = typer.Option(Path("data/snapshot.json"), "--out", "-o", help="Where to save."),
    prices: Path = PricesOpt,
    injuries: Path = InjuriesOpt,
    odds_csv: Path = OddsOpt,
    props_csv: Path = PropsOpt,
    offline: bool = OfflineOpt,
    config: Path = ConfigOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """Fetch everything and write a snapshot for reproducible runs."""

    _setup_logging(verbose)
    settings = load_settings(config)
    from elfantasy.data.load import build_dataset

    ds = build_dataset(
        settings,
        offline=offline,
        prices_csv=prices,
        injuries_yaml=injuries,
        odds_csv=odds_csv,
        props_csv=props_csv,
    )
    path = ds.save(out)
    report.console.print(f"saved snapshot -> [bold]{path}[/]")
    report.console.print(ds.summary())
    report.warnings_panel(_health(ds))


def _health(ds: Dataset) -> list[str]:
    from elfantasy.data.load import data_health

    return data_health(ds)


@app.command()
def project(
    top: int = typer.Option(30, "--top", "-n"),
    position: str = typer.Option(None, "--position", "-p", help="Filter to G, F or C."),
    max_price: float = typer.Option(None, "--max-price"),
    sample: bool = SampleOpt,
    snapshot: Path = SnapshotOpt,
    prices: Path = PricesOpt,
    injuries: Path = InjuriesOpt,
    odds_csv: Path = OddsOpt,
    props_csv: Path = PropsOpt,
    offline: bool = OfflineOpt,
    round_no: int = RoundOpt,
    horizon: int = HorizonOpt,
    config: Path = ConfigOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """Rank players by expected PIR for a round."""

    _setup_logging(verbose)
    pipe = _build(
        sample=sample,
        snapshot=snapshot,
        prices=prices,
        injuries=injuries,
        odds_csv=odds_csv,
        props_csv=props_csv,
        offline=offline,
        round_no=round_no,
        horizon=horizon,
        config=config,
    )
    report.warnings_panel(pipe.health())
    report.projections_table(
        pipe.next_round,
        pipe.dataset,
        top=top,
        position=position.upper() if position else None,
        max_price=max_price,
        title=f"Round {pipe.start_round} projections",
    )


@app.command()
def explain(
    player: str = typer.Argument(..., help="Player name or id."),
    sample: bool = SampleOpt,
    snapshot: Path = SnapshotOpt,
    prices: Path = PricesOpt,
    injuries: Path = InjuriesOpt,
    odds_csv: Path = OddsOpt,
    props_csv: Path = PropsOpt,
    offline: bool = OfflineOpt,
    round_no: int = RoundOpt,
    horizon: int = HorizonOpt,
    config: Path = ConfigOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """Show every term that produced a player's projection."""

    _setup_logging(verbose)
    pipe = _build(
        sample=sample,
        snapshot=snapshot,
        prices=prices,
        injuries=injuries,
        odds_csv=odds_csv,
        props_csv=props_csv,
        offline=offline,
        round_no=round_no,
        horizon=horizon,
        config=config,
    )
    hits = pipe.find_player(player)
    if not hits:
        report.console.print(f"[red]no player matched '{player}'[/]")
        raise typer.Exit(1)
    if len(hits) > 1:
        report.console.print(
            "[yellow]several matches:[/] "
            + ", ".join(f"{pipe.dataset.players[h].name} ({h})" for h in hits[:10])
        )
    proj = pipe.explain(hits[0])
    if proj is None:
        report.console.print("[yellow]that player has no fixture in this round[/]")
        raise typer.Exit(1)
    report.explain_projection(proj, pipe.dataset)


@app.command()
def lineup(
    squad_file: Path = typer.Option(
        None, "--squad", "-s", help="Your current squad (YAML); omit to build one from scratch."
    ),
    max_transfers: int = typer.Option(
        None, "--max-transfers", "-k", help="Default: the rules' per-round cap."
    ),
    thorough: bool = typer.Option(
        False, "--thorough", help="Several starting points and paired swaps (minutes, not seconds)."
    ),
    no_search: bool = typer.Option(
        False, "--no-search", help="Static MILP only; skip simulation search."
    ),
    sims: int = typer.Option(6000, "--sims", help="Simulated rounds per evaluation."),
    credit_multiplier: float = typer.Option(
        1.0,
        "--credit-value",
        help="Scale the value of a credit (e.g. 3 if gains compound all season).",
    ),
    sample: bool = SampleOpt,
    snapshot: Path = SnapshotOpt,
    prices: Path = PricesOpt,
    injuries: Path = InjuriesOpt,
    odds_csv: Path = OddsOpt,
    props_csv: Path = PropsOpt,
    offline: bool = OfflineOpt,
    round_no: int = RoundOpt,
    horizon: int = HorizonOpt,
    config: Path = ConfigOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """Best roster and lineup under the real game rules, Turn moves included.

    Plans who starts, who waits on the bench for a later game day, who
    captains, and what to change after the first day. With --squad it plans
    transfers from your current team instead.
    """

    from elfantasy.plan import plan_lineup
    from elfantasy.rules import GameRules

    _setup_logging(verbose)
    pipe = _build(
        sample=sample,
        snapshot=snapshot,
        prices=prices,
        injuries=injuries,
        odds_csv=odds_csv,
        props_csv=props_csv,
        offline=offline,
        round_no=round_no,
        horizon=horizon,
        config=config,
    )
    report.warnings_panel(pipe.health())
    rules = GameRules.from_settings(pipe.settings)

    my_squad = None
    if squad_file is not None:
        try:
            my_squad = load_squad_yaml(squad_file, pipe.dataset.players, pipe.dataset.coaches)
        except (ValueError, FileNotFoundError) as exc:
            report.console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc

    try:
        with report.console.status("simulating rounds..."):
            plan = plan_lineup(
                pipe,
                rules,
                squad=my_squad,
                max_transfers=max_transfers,
                n=sims,
                search=not no_search,
                thorough=thorough,
                credit_multiplier=credit_multiplier,
                log=lambda m: report.console.print(f"[dim]{m}[/]"),
            )
    except InfeasibleError as exc:
        report.console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    report.lineup_plan(plan, title=f"Round {pipe.start_round}")


@app.command()
def squad(
    budget: float = typer.Option(None, "--budget", "-b", help="Override the configured budget."),
    risk: float = typer.Option(None, "--risk", help="Mean-variance risk aversion (0 = neutral)."),
    sample: bool = SampleOpt,
    snapshot: Path = SnapshotOpt,
    prices: Path = PricesOpt,
    injuries: Path = InjuriesOpt,
    odds_csv: Path = OddsOpt,
    props_csv: Path = PropsOpt,
    offline: bool = OfflineOpt,
    round_no: int = RoundOpt,
    horizon: int = HorizonOpt,
    config: Path = ConfigOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """Build the best legal squad from scratch (a reset / wildcard round)."""

    _setup_logging(verbose)
    pipe = _build(
        sample=sample,
        snapshot=snapshot,
        prices=prices,
        injuries=injuries,
        odds_csv=odds_csv,
        props_csv=props_csv,
        offline=offline,
        round_no=round_no,
        horizon=horizon,
        config=config,
    )
    report.warnings_panel(pipe.health())
    try:
        solution = pipe.best_squad(budget=budget, risk_aversion=risk)
    except InfeasibleError as exc:
        report.console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    report.squad_table(solution, pipe.dataset)


@app.command()
def transfers(
    squad_file: Path = typer.Option(..., "--squad", "-s", help="YAML file with your roster."),
    max_transfers: int = typer.Option(None, "--max-transfers", "-k"),
    ladder: bool = typer.Option(True, "--ladder/--no-ladder", help="Show 0..k transfer options."),
    risk: float = typer.Option(None, "--risk"),
    friction: float = typer.Option(None, "--friction", help="Points penalty per transfer."),
    protect: list[str] = typer.Option([], "--protect", help="Player ids you refuse to sell."),
    sample: bool = SampleOpt,
    snapshot: Path = SnapshotOpt,
    prices: Path = PricesOpt,
    injuries: Path = InjuriesOpt,
    odds_csv: Path = OddsOpt,
    props_csv: Path = PropsOpt,
    offline: bool = OfflineOpt,
    round_no: int = RoundOpt,
    horizon: int = HorizonOpt,
    config: Path = ConfigOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """Recommend the best transfers for a standard round."""

    _setup_logging(verbose)
    pipe = _build(
        sample=sample,
        snapshot=snapshot,
        prices=prices,
        injuries=injuries,
        odds_csv=odds_csv,
        props_csv=props_csv,
        offline=offline,
        round_no=round_no,
        horizon=horizon,
        config=config,
    )
    report.warnings_panel(pipe.health())

    try:
        my_squad = load_squad_yaml(squad_file, pipe.dataset.players, pipe.dataset.coaches)
    except (ValueError, FileNotFoundError) as exc:
        report.console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    kwargs = {"risk_aversion": risk, "friction": friction, "protect": set(protect)}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        plan = pipe.best_transfers(my_squad, max_transfers=max_transfers, **kwargs)
        report.transfer_table(plan)
        if ladder:
            report.ladder_table(
                pipe.transfer_ladder(my_squad, max_transfers=max_transfers, **kwargs)
            )
    except InfeasibleError as exc:
        report.console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc


@app.command()
def value(
    top: int = typer.Option(25, "--top", "-n"),
    sample: bool = SampleOpt,
    snapshot: Path = SnapshotOpt,
    prices: Path = PricesOpt,
    injuries: Path = InjuriesOpt,
    odds_csv: Path = OddsOpt,
    props_csv: Path = PropsOpt,
    offline: bool = OfflineOpt,
    round_no: int = RoundOpt,
    horizon: int = HorizonOpt,
    config: Path = ConfigOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """Rank players by projected value per credit over the horizon."""

    _setup_logging(verbose)
    pipe = _build(
        sample=sample,
        snapshot=snapshot,
        prices=prices,
        injuries=injuries,
        odds_csv=odds_csv,
        props_csv=props_csv,
        offline=offline,
        round_no=round_no,
        horizon=horizon,
        config=config,
    )
    report.value_table(pipe.candidates(), pipe.dataset, top=top)


@app.command()
def fixtures(
    sample: bool = SampleOpt,
    snapshot: Path = SnapshotOpt,
    offline: bool = OfflineOpt,
    round_no: int = RoundOpt,
    horizon: int = HorizonOpt,
    config: Path = ConfigOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """Show schedule difficulty per club across the horizon."""

    _setup_logging(verbose)
    pipe = _build(
        sample=sample,
        snapshot=snapshot,
        prices=None,
        injuries=None,
        odds_csv=None,
        props_csv=None,
        offline=offline,
        round_no=round_no,
        horizon=horizon,
        config=config,
    )
    report.fixtures_table(pipe.dataset, sorted(pipe.horizon), pipe.fixture_win_probs())


@app.command("price-template")
def price_template(
    out: Path = typer.Option(Path("data/prices.csv"), "--out", "-o"),
    snapshot: Path = SnapshotOpt,
    sample: bool = SampleOpt,
    offline: bool = OfflineOpt,
    config: Path = ConfigOpt,
) -> None:
    """Write a CSV of every player, ready for you to fill in prices."""

    settings = load_settings(config)
    if sample:
        ds = build_sample_dataset()
    elif snapshot:
        ds = Dataset.load(snapshot)
    else:
        from elfantasy.data.load import build_dataset

        ds = build_dataset(settings, offline=offline)
    path = write_price_template(out, ds.players)
    report.console.print(f"wrote {len(ds.players)} rows -> [bold]{path}[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
