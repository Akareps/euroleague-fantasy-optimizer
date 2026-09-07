"""Full-squad optimisation: the "reset round" problem.

Formulated as a mixed-integer linear program and solved with CBC (bundled with
PuLP, so there is no external solver to install).

    maximise    sum_i x_i * (V_i - lambda * Var_i)  +  bank_bonus * leftover
    subject to  sum_i x_i * price_i  <=  budget
                sum_i x_i           ==  squad_size
                pos_min_p <= sum_{i in p} x_i <= pos_max_p     for each position
                sum_{i in club c} x_i <= max_per_club          for each club
                x_i in {0, 1}

``V_i`` is the discounted multi-round value from
:func:`elfantasy.projection.engine.horizon_value`, which is what makes the
solver prefer a player with a good *run* of fixtures over one with a single good
game -- the "I will not have to transfer him again next round" effect.

Why MILP rather than a greedy points-per-credit sort: the budget constraint
makes this a multi-dimensional knapsack, and greedy value-per-credit is
provably sub-optimal on exactly the cases that matter -- it will not spend down
to a premium player even when the remaining budget cannot be used better
elsewhere. CBC solves a realistic instance (~200 players) in well under a
second.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pulp

from elfantasy.config import Settings
from elfantasy.optimize.solver import solve


@dataclass
class Candidate:
    """One selectable player, already reduced to what the solver needs."""

    player_id: str
    name: str
    team_code: str
    position: str
    price: float
    value: float  # discounted multi-round expected PIR
    variance: float = 0.0
    next_round_pir: float = 0.0
    ownership: float | None = None
    locked_in: bool = False  # must be in the squad
    locked_out: bool = False  # must not be in the squad


@dataclass
class SquadSolution:
    selected: list[Candidate] = field(default_factory=list)
    total_price: float = 0.0
    total_value: float = 0.0
    next_round_pir: float = 0.0
    leftover: float = 0.0
    status: str = "unsolved"
    objective: float = 0.0

    def by_position(self) -> dict[str, list[Candidate]]:
        out: dict[str, list[Candidate]] = {}
        for c in self.selected:
            out.setdefault(c.position, []).append(c)
        for v in out.values():
            v.sort(key=lambda c: -c.value)
        return out

    @property
    def player_ids(self) -> list[str]:
        return [c.player_id for c in self.selected]


class InfeasibleError(RuntimeError):
    """Raised when no squad satisfies the constraints."""


def _add_common_constraints(
    prob: pulp.LpProblem,
    x: dict[str, pulp.LpVariable],
    candidates: dict[str, Candidate],
    settings: Settings,
    *,
    squad_size: int,
    max_per_club: int | None = None,
) -> None:
    prob += pulp.lpSum(x.values()) == squad_size, "squad_size"

    for pos, (lo, hi) in settings.position_bounds().items():
        members = [x[pid] for pid, c in candidates.items() if c.position == pos]
        if not members:
            if lo > 0:
                raise InfeasibleError(f"no candidates available at position {pos}")
            continue
        prob += pulp.lpSum(members) >= lo, f"pos_min_{pos}"
        prob += pulp.lpSum(members) <= hi, f"pos_max_{pos}"

    cap = max_per_club if max_per_club is not None else settings.max_per_club
    if cap and cap > 0:
        clubs = {c.team_code for c in candidates.values()}
        for club in clubs:
            members = [x[pid] for pid, c in candidates.items() if c.team_code == club]
            prob += pulp.lpSum(members) <= cap, f"club_{club}"

    for pid, c in candidates.items():
        if c.locked_in:
            prob += x[pid] == 1, f"lock_in_{pid}"
        if c.locked_out:
            prob += x[pid] == 0, f"lock_out_{pid}"


def optimise_squad(
    candidates: list[Candidate],
    settings: Settings,
    *,
    budget: float | None = None,
    squad_size: int | None = None,
    risk_aversion: float | None = None,
    max_per_club: int | None = None,
    exclude_teams: set[str] | None = None,
    solver: pulp.LpSolver | None = None,
) -> SquadSolution:
    """Build the best legal squad from scratch."""

    pool = {c.player_id: c for c in candidates if c.price > 0 and not c.locked_out}
    if exclude_teams:
        pool = {k: v for k, v in pool.items() if v.team_code not in exclude_teams}
    if not pool:
        raise InfeasibleError("no candidates supplied")

    size = squad_size if squad_size is not None else settings.squad_size
    cash = budget if budget is not None else settings.budget
    lam = (
        risk_aversion
        if risk_aversion is not None
        else float(settings.model.get("optimiser.risk_aversion"))
    )
    bank_bonus = float(settings.model.get("optimiser.bank_bonus"))

    prob = pulp.LpProblem("elfantasy_squad", pulp.LpMaximize)
    x = {pid: pulp.LpVariable(f"x_{pid}", cat=pulp.LpBinary) for pid in pool}

    spend = pulp.lpSum(x[pid] * pool[pid].price for pid in pool)
    # Mean-variance objective. Because x is binary, sum(x_i * var_i) is the
    # variance of the squad total under independence -- linear, so the problem
    # stays a MILP. Same-game correlation is not modelled here; see docs/model.md.
    utility = pulp.lpSum(x[pid] * (pool[pid].value - lam * pool[pid].variance) for pid in pool)
    prob += utility + bank_bonus * (cash - spend)

    prob += spend <= cash, "budget"
    _add_common_constraints(prob, x, pool, settings, squad_size=size, max_per_club=max_per_club)

    status = solve(prob, solver)
    if status != "Optimal":
        raise InfeasibleError(
            f"solver returned {status}. Common causes: budget too low for the "
            f"position minimums, or max_per_club too tight for the candidate pool."
        )

    chosen = [pool[pid] for pid in pool if x[pid].value() and x[pid].value() > 0.5]
    total_price = sum(c.price for c in chosen)
    return SquadSolution(
        selected=sorted(chosen, key=lambda c: (c.position, -c.value)),
        total_price=total_price,
        total_value=sum(c.value for c in chosen),
        next_round_pir=sum(c.next_round_pir for c in chosen),
        leftover=cash - total_price,
        status=status,
        objective=float(pulp.value(prob.objective) or 0.0),
    )


def marginal_value_curve(
    candidates: list[Candidate],
    settings: Settings,
    budgets: list[float],
) -> list[tuple[float, float]]:
    """Optimal squad value at several budgets.

    Useful for answering "is it worth banking a credit?" -- the slope of this
    curve is the true shadow price of a credit, which is the number you should
    compare a transfer against.
    """

    out = []
    for b in budgets:
        try:
            sol = optimise_squad(candidates, settings, budget=b)
            out.append((b, sol.total_value))
        except InfeasibleError:
            out.append((b, float("nan")))
    return out


def best_at_price(
    candidates: list[Candidate],
    position: str,
    max_price: float,
    top: int = 5,
) -> list[Candidate]:
    """Highest-value players at a position within a price cap."""

    pool = [c for c in candidates if c.position == position and c.price <= max_price]
    pool.sort(key=lambda c: -c.value)
    return pool[:top]
