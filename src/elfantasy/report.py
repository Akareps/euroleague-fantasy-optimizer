"""Human-readable output.

A projection you cannot interrogate is a projection you should not act on, so
every table here is built to be argued with: the explain view shows each
multiplier that produced a number, and the transfer ladder shows what each
successive swap actually buys you.
"""

from __future__ import annotations

import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from elfantasy.dataset import Dataset
from elfantasy.models import Availability, Projection
from elfantasy.optimize.squad import Candidate, SquadSolution
from elfantasy.optimize.transfers import TransferPlan

console = Console()

STATUS_STYLE = {
    Availability.OUT: "bold red",
    Availability.DOUBTFUL: "red",
    Availability.QUESTIONABLE: "yellow",
    Availability.PROBABLE: "green",
    Availability.ACTIVE: "",
    Availability.UNKNOWN: "dim",
}


def _fmt(x: float | None, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def warnings_panel(messages: list[str]) -> None:
    if not messages:
        return
    body = "\n".join(f"- {m}" for m in messages)
    console.print(Panel(body, title="data quality", border_style="yellow", expand=False))


def projections_table(
    projections: dict[str, Projection],
    dataset: Dataset,
    *,
    top: int = 25,
    position: str | None = None,
    max_price: float | None = None,
    title: str = "Projections",
) -> None:
    rows = []
    for pid, proj in projections.items():
        player = dataset.players.get(pid)
        if player is None:
            continue
        if position and player.position.value != position:
            continue
        if max_price is not None and player.price > max_price:
            continue
        rows.append((player, proj))

    rows.sort(key=lambda r: -r[1].mean_pir)
    rows = rows[:top]

    table = Table(title=title, header_style="bold")
    table.add_column("Player", overflow="ellipsis", max_width=24)
    table.add_column("Tm", width=4)
    table.add_column("P", width=2)
    table.add_column("Price", justify="right")
    table.add_column("xPIR", justify="right")
    table.add_column("SD", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Play%", justify="right")
    table.add_column("Opp", width=4)
    table.add_column("H/A", width=3)
    table.add_column("Win%", justify="right")
    table.add_column("Val", justify="right", header_style="bold cyan")
    table.add_column("Notes", overflow="ellipsis", max_width=26, no_wrap=True)

    for player, proj in rows:
        value = proj.mean_pir / player.price if player.price > 0 else 0.0
        style = STATUS_STYLE.get(player.status, "")
        table.add_row(
            Text(player.name, style=style),
            player.team_code,
            player.position.value,
            _fmt(player.price),
            _fmt(proj.mean_pir),
            _fmt(proj.sd_pir),
            _fmt(proj.minutes),
            f"{proj.play_prob * 100:.0f}",
            proj.opponent or "-",
            "H" if proj.home else "A",
            f"{(proj.win_prob or 0) * 100:.0f}",
            _fmt(value, 2),
            "; ".join(proj.notes[:2]),
        )
    console.print(table)


def explain_projection(proj: Projection, dataset: Dataset) -> None:
    """Show the full derivation of one player's number."""

    player = dataset.players.get(proj.player_id)
    name = player.name if player else proj.player_id
    c = proj.components

    header = (
        f"[bold]{name}[/] ({player.team_code if player else '?'}, "
        f"{player.position.value if player else '?'}) - round {proj.round} "
        f"{'vs' if proj.home else '@'} {proj.opponent}"
    )
    console.print(Panel(header, border_style="cyan", expand=False))

    minutes = Table(title="minutes", show_header=True, header_style="bold")
    minutes.add_column("component")
    minutes.add_column("value", justify="right")
    minutes.add_row("baseline (recency-weighted)", _fmt(c.get("baseline_minutes")))
    minutes.add_row("from team-mate absences", f"{c.get('minutes_from_absences', 0):+.1f}")
    minutes.add_row("from blowout risk", f"{c.get('minutes_from_blowout', 0):+.1f}")
    minutes.add_row("[bold]projected minutes", f"[bold]{_fmt(proj.minutes)}")
    minutes.add_row("probability of playing", f"{proj.play_prob * 100:.0f}%")
    console.print(minutes)

    mult = Table(title="production multipliers", header_style="bold")
    mult.add_column("factor")
    mult.add_column("x", justify="right")
    mult.add_column("reading")
    labels = {
        "venue": "home / away split",
        "rest": "days since last game",
        "opponent": "opponent defence vs position",
        "pace": "game pace",
        "team_total": "team scoring environment",
        "synergy": "team-mate availability effects",
    }
    for key, label in labels.items():
        v = c.get(key)
        if v is None:
            continue
        arrow = "up" if v > 1.005 else ("down" if v < 0.995 else "neutral")
        mult.add_row(label, f"{v:.3f}", arrow)
    mult.add_row("PIR per minute (usage-adjusted)", _fmt(c.get("pir_per_minute"), 3), "")
    console.print(mult)

    blend = Table(title="model vs market", header_style="bold")
    blend.add_column("source")
    blend.add_column("xPIR", justify="right")
    blend.add_column("weight / coverage", justify="right")
    blend.add_row("statistical model", _fmt(c.get("model_pir")), "")
    market = c.get("market_pir")
    if market is not None and not math.isnan(market):
        blend.add_row("market-implied", _fmt(market), f"{c.get('market_coverage', 0) * 100:.0f}%")
    else:
        blend.add_row("market-implied", "-", "no props available")
    blend.add_row(
        "[bold]blended, x P(play)", f"[bold]{_fmt(proj.mean_pir)}", f"sd {_fmt(proj.sd_pir)}"
    )
    console.print(blend)

    game = Table(title="game context", header_style="bold")
    game.add_column("field")
    game.add_column("value", justify="right")
    game.add_row("spread (home handicap)", _fmt(c.get("spread")))
    game.add_row("total", _fmt(c.get("total")))
    game.add_row("win probability", f"{(proj.win_prob or 0) * 100:.0f}%")
    console.print(game)

    if proj.notes:
        console.print(Panel("\n".join(f"- {n}" for n in proj.notes), title="notes", expand=False))


def squad_table(solution: SquadSolution, dataset: Dataset, title: str = "Optimal squad") -> None:
    table = Table(title=title, header_style="bold")
    table.add_column("P", width=2)
    table.add_column("Player", overflow="ellipsis", max_width=24)
    table.add_column("Tm", width=4)
    table.add_column("Price", justify="right")
    table.add_column("Next xPIR", justify="right")
    table.add_column("Horizon value", justify="right")
    table.add_column("Val/credit", justify="right")

    for pos in ("G", "F", "C"):
        for c in [c for c in solution.selected if c.position == pos]:
            table.add_row(
                c.position,
                c.name,
                c.team_code,
                _fmt(c.price),
                _fmt(c.next_round_pir),
                _fmt(c.value),
                _fmt(c.value / c.price if c.price else 0, 2),
            )
    console.print(table)
    console.print(
        f"  spend [bold]{solution.total_price:.1f}[/]  "
        f"leftover [bold]{solution.leftover:.1f}[/]  "
        f"next-round xPIR [bold]{solution.next_round_pir:.1f}[/]  "
        f"horizon value [bold]{solution.total_value:.1f}[/]"
    )


def transfer_table(plan: TransferPlan, title: str = "Recommended transfers") -> None:
    if not plan.moves:
        console.print(Panel("No transfer improves the squad. Hold.", title=title, expand=False))
        return

    table = Table(title=title, header_style="bold")
    table.add_column("Out", overflow="ellipsis", max_width=22)
    table.add_column("Price", justify="right")
    table.add_column("->", width=2)
    table.add_column("In", overflow="ellipsis", max_width=22)
    table.add_column("Price", justify="right")
    table.add_column("Delta cost", justify="right")
    table.add_column("Delta next", justify="right")
    table.add_column("Delta horizon", justify="right", header_style="bold cyan")

    for m in plan.moves:
        table.add_row(
            f"{m.out_player.name} ({m.out_player.team_code})",
            _fmt(m.out_player.price),
            "->",
            f"{m.in_player.name} ({m.in_player.team_code})",
            _fmt(m.in_player.price),
            f"{m.price_delta:+.1f}",
            f"{m.in_player.next_round_pir - m.out_player.next_round_pir:+.1f}",
            f"{m.value_gain:+.1f}",
        )
    console.print(table)
    console.print(
        f"  {plan.n_transfers} transfer(s)  "
        f"next round [bold]{plan.next_round_before:.1f} -> {plan.next_round_after:.1f}[/]  "
        f"horizon [bold]{plan.value_before:.1f} -> {plan.value_after:.1f}[/] "
        f"([bold cyan]{plan.value_gain:+.1f}[/])  "
        f"bank {plan.bank_before:.1f} -> {plan.bank_after:.1f}"
    )


def ladder_table(plans: list[TransferPlan]) -> None:
    """Marginal gain of each successive transfer -- where to stop."""

    table = Table(title="Transfer ladder (marginal value of each swap)", header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Horizon value", justify="right")
    table.add_column("Gain vs hold", justify="right")
    table.add_column("Marginal gain", justify="right", header_style="bold cyan")
    table.add_column("Moves", overflow="fold")

    base = plans[0].value_after if plans else 0.0
    prev = base
    for plan in plans:
        moves = ", ".join(f"{m.out_player.name}->{m.in_player.name}" for m in plan.moves) or "hold"
        table.add_row(
            str(plan.n_transfers),
            _fmt(plan.value_after),
            f"{plan.value_after - base:+.2f}",
            f"{plan.value_after - prev:+.2f}",
            moves,
        )
        prev = plan.value_after
    console.print(table)
    console.print(
        "  [dim]Stop where the marginal gain stops paying for the flexibility you "
        "give up. Transfer friction is already priced in (optimiser.transfer_friction).[/]"
    )


def value_table(
    candidates: list[Candidate],
    dataset: Dataset,
    *,
    top: int = 20,
    title: str = "Best value per credit",
) -> None:
    ranked = sorted([c for c in candidates if c.price > 0], key=lambda c: -(c.value / c.price))[
        :top
    ]
    table = Table(title=title, header_style="bold")
    table.add_column("Player", overflow="ellipsis", max_width=24)
    table.add_column("Tm", width=4)
    table.add_column("P", width=2)
    table.add_column("Price", justify="right")
    table.add_column("Next xPIR", justify="right")
    table.add_column("Horizon", justify="right")
    table.add_column("Per credit", justify="right", header_style="bold cyan")
    table.add_column("Own%", justify="right")
    for c in ranked:
        table.add_row(
            c.name,
            c.team_code,
            c.position,
            _fmt(c.price),
            _fmt(c.next_round_pir),
            _fmt(c.value),
            _fmt(c.value / c.price, 2),
            f"{(c.ownership or 0) * 100:.0f}",
        )
    console.print(table)


def fixtures_table(dataset: Dataset, rounds: list[int], win_probs: dict[str, list[float]]) -> None:
    """Upcoming schedule difficulty by club -- the multi-round view."""

    table = Table(title="Fixture run (win probability by round)", header_style="bold")
    table.add_column("Tm", width=4)
    for r in rounds:
        table.add_column(f"R{r}", justify="right")
    table.add_column("Mean", justify="right", header_style="bold cyan")

    for code in sorted(win_probs, key=lambda c: -sum(win_probs[c]) / max(len(win_probs[c]), 1)):
        probs = win_probs[code]
        cells = []
        for p in probs:
            style = "green" if p >= 0.62 else ("red" if p <= 0.38 else "")
            cells.append(Text(f"{p * 100:.0f}", style=style))
        mean = sum(probs) / len(probs) if probs else 0.0
        table.add_row(code, *cells, f"{mean * 100:.0f}")
    console.print(table)
