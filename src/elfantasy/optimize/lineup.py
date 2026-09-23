"""Lineup-aware squad and transfer optimisation under the real game's rules.

The squad optimiser in :mod:`elfantasy.optimize.squad` treats every rostered
player as equal. The actual game does not: only the starting five and a sixth
man score in full, the rest of the bench scores half, the captain scores
double, and the head coach is bought out of the same budget. Picking the ten
best points-per-credit players and then arranging them is not the same as
choosing a roster *for* a lineup -- the second can afford a premium captain
because it knows two bench slots only return half.

Formulation (one binary per role per player):

    x_i = s_i + m_i + b_i          rostered = starter + sixth man + bench
    c_i <= s_i                     the captain is a starter
    sum s_i = 5, sum m_i = 1, sum c_i = 1, exactly one coach
    starter position counts = one of the legal formations (via f_k)

    maximise  sum_i EV1_i * (s_i + m_i + (cap-1) c_i + w_bench b_i) + EV1_coach
            + sum_i x_i * w_future * sum_{r>1} gamma^(r-1) EV_r,i  (+ coach)

With a current squad it becomes a transfer problem: buys and sells are linked
to the roster variables, the transfer cap counts a coach change when the rules
say so, and sales fund purchases.

This MILP scores the first round as if the lineup were fixed. The Turn
mechanics (promoting later-Turn players, moving the captaincy) are valued by
:mod:`elfantasy.optimize.turns`, which uses this solver for starting points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pulp

from elfantasy.optimize.solver import solve
from elfantasy.optimize.squad import InfeasibleError
from elfantasy.rules import GameRules


@dataclass
class LineupPlayer:
    player_id: str
    name: str
    pos: str
    club: str
    price: float
    ev: dict[int, float]  # expected fantasy points by round number
    late: bool = False  # plays after the first Turn of the first round
    sell_price: float | None = None  # what you would receive, if you own him


@dataclass
class LineupCoach:
    coach_id: str
    name: str
    club: str
    price: float
    ev: dict[int, float]
    sell_price: float | None = None


@dataclass
class CurrentSquad:
    player_ids: list[str]
    coach_id: str | None = None
    bank: float = 0.0


@dataclass
class LineupSolution:
    starters: list[LineupPlayer]
    sixth: LineupPlayer | None
    bench: list[LineupPlayer]
    captain: LineupPlayer
    coach: LineupCoach | None
    formation: tuple[int, int, int]
    cost: float
    first_round: float
    objective: float
    bought: list[LineupPlayer] = field(default_factory=list)
    sold: list[str] = field(default_factory=list)
    coach_changed: bool = False

    @property
    def roster(self) -> list[LineupPlayer]:
        return self.starters + ([self.sixth] if self.sixth else []) + self.bench

    @property
    def n_transfers(self) -> int:
        return len(self.bought) + (1 if self.coach_changed else 0)


def future_slot_weight(rules: GameRules) -> float:
    """Average scoring weight of a roster spot in a future, re-optimised round.

    Starters and the sixth man score 1, the captain adds (multiplier - 1), the
    rest of the bench scores the bench weight: (6 + 1 + 4 * 0.5) / 10 = 0.9.
    """

    bench = rules.squad_size - rules.field_slots
    total = rules.field_slots + (rules.captain_multiplier - 1.0) + bench * rules.bench_weight
    return total / rules.squad_size


def optimise_lineup(
    players: list[LineupPlayer],
    coaches: list[LineupCoach],
    rules: GameRules,
    *,
    rounds: list[int],
    gamma: float = 0.4,
    budget: float | None = None,
    current: CurrentSquad | None = None,
    max_transfers: int | None = None,
    friction: float = 0.0,
    max_club: int | None = None,
    min_late_bench: int = 0,
    exclude: set[str] | frozenset[str] = frozenset(),
    force: set[str] | frozenset[str] = frozenset(),
    force_coach: str | None = None,
    solver: pulp.LpSolver | None = None,
) -> LineupSolution:
    """Best roster *and* lineup for ``rounds[0]``, valuing later rounds too.

    Players are identified by ``player_id``; ``exclude``/``force`` take ids.
    """

    first, later = rounds[0], rounds[1:]
    owned = set(current.player_ids) if current else set()
    pool = [
        p for p in players if p.player_id not in exclude and (first in p.ev or p.player_id in owned)
    ]
    if not pool:
        raise InfeasibleError("no candidates supplied")
    ids = range(len(pool))
    use_coach = rules.head_coach and bool(coaches)
    w_future = future_slot_weight(rules)

    prob = pulp.LpProblem("elfantasy_lineup", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in ids}
    s = {i: pulp.LpVariable(f"s{i}", cat="Binary") for i in ids}
    six = {i: pulp.LpVariable(f"m{i}", cat="Binary") for i in ids} if rules.sixth_man else {}
    cap = {i: pulp.LpVariable(f"c{i}", cat="Binary") for i in ids}
    b = {i: pulp.LpVariable(f"b{i}", cat="Binary") for i in ids}
    y = (
        {k: pulp.LpVariable(f"y{k}", cat="Binary") for k in range(len(coaches))}
        if use_coach
        else {}
    )
    f = {j: pulp.LpVariable(f"f{j}", cat="Binary") for j in range(len(rules.formations))}

    def ev(p, r):
        return p.ev.get(r, 0.0)

    def future(p):
        return sum(gamma ** (k + 1) * ev(p, r) for k, r in enumerate(later))

    first_round = pulp.lpSum(
        ev(pool[i], first)
        * (
            s[i]
            + (six[i] if six else 0)
            + (rules.captain_multiplier - 1.0) * cap[i]
            + rules.bench_weight * b[i]
        )
        for i in ids
    )
    first_round += pulp.lpSum(y[k] * ev(c, first) for k, c in enumerate(coaches) if y)
    objective = first_round
    objective += pulp.lpSum(x[i] * w_future * future(pool[i]) for i in ids)
    objective += pulp.lpSum(y[k] * future(c) for k, c in enumerate(coaches) if y)

    # --- roster and lineup structure ----------------------------------------
    for i in ids:
        prob += s[i] + (six[i] if six else 0) + b[i] == x[i]
        prob += cap[i] <= s[i]
    for pos, n in rules.roster.items():
        prob += pulp.lpSum(x[i] for i in ids if pool[i].pos == pos) == n
    prob += pulp.lpSum(s.values()) == rules.starters
    if six:
        prob += pulp.lpSum(six.values()) == 1
    prob += pulp.lpSum(cap.values()) == 1
    prob += pulp.lpSum(f.values()) == 1
    for pi, pos in enumerate("GFC"):
        prob += pulp.lpSum(s[i] for i in ids if pool[i].pos == pos) == pulp.lpSum(
            f[j] * rules.formations[j][pi] for j in f
        )
    if y:
        prob += pulp.lpSum(y.values()) == 1
    cap_club = max_club if max_club is not None else rules.max_per_club
    for club in {p.club for p in pool}:
        prob += pulp.lpSum(x[i] for i in ids if pool[i].club == club) <= cap_club
    for i in ids:
        if pool[i].player_id in force:
            prob += x[i] == 1
    if force_coach is not None and y:
        prob += pulp.lpSum(y[k] for k, c in enumerate(coaches) if c.coach_id == force_coach) == 1
    if min_late_bench:
        prob += pulp.lpSum(b[i] for i in ids if pool[i].late) >= min_late_bench

    # --- money and transfers --------------------------------------------------
    def cost(item, is_owned):
        if is_owned and item.sell_price is not None:
            return item.sell_price
        return item.price

    spend = pulp.lpSum(x[i] * cost(pool[i], pool[i].player_id in owned) for i in ids)
    if y:
        cur_coach = current.coach_id if current else None
        spend += pulp.lpSum(y[k] * cost(c, c.coach_id == cur_coach) for k, c in enumerate(coaches))
    if current is None:
        prob += spend <= (budget if budget is not None else rules.budget)
        n_moves = None
    else:
        # Keeping a player "costs" his sell value on both sides, so this is
        # the familiar "purchases <= bank + sales" written over the roster.
        owned_value = sum(cost(p, True) for p in pool if p.player_id in owned)
        if y and current.coach_id:
            owned_value += sum(cost(c, True) for c in coaches if c.coach_id == current.coach_id)
        prob += spend <= current.bank + owned_value
        n_moves = pulp.lpSum(x[i] for i in ids if pool[i].player_id not in owned)
        if y and rules.coach_counts_as_transfer and current.coach_id:
            n_moves += pulp.lpSum(
                y[k] for k, c in enumerate(coaches) if c.coach_id != current.coach_id
            )
        if max_transfers is not None:
            prob += n_moves <= max_transfers
        if friction:
            objective -= friction * n_moves
    prob += objective

    status = solve(prob, solver)
    if status != "Optimal":
        raise InfeasibleError(
            f"solver returned {status}: the budget, position counts, club cap or "
            f"transfer cap cannot all be satisfied by the candidate pool"
        )

    def picked(var):
        return [pool[i] for i in ids if var[i].value() and var[i].value() > 0.5]

    coach = next((coaches[k] for k in y if y[k].value() > 0.5), None) if y else None
    starters = picked(s)
    captain = picked(cap)[0]
    formation = next(rules.formations[j] for j in f if f[j].value() > 0.5)
    sol = LineupSolution(
        starters=sorted(starters, key=lambda p: "GFC".index(p.pos)),
        sixth=picked(six)[0] if six else None,
        bench=sorted(picked(b), key=lambda p: -ev(p, first)),
        captain=captain,
        coach=coach,
        formation=formation,
        cost=0.0,
        first_round=float(pulp.value(first_round)),
        objective=float(pulp.value(prob.objective)),
    )
    sol.cost = sum(cost(p, p.player_id in owned) for p in sol.roster) + (
        cost(coach, current is not None and coach.coach_id == current.coach_id) if coach else 0.0
    )
    if current is not None:
        sol.bought = [p for p in sol.roster if p.player_id not in owned]
        roster_ids = {p.player_id for p in sol.roster}
        sol.sold = [pid for pid in current.player_ids if pid not in roster_ids]
        sol.coach_changed = bool(coach and current.coach_id and coach.coach_id != current.coach_id)
    return sol
