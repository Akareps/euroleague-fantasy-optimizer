"""Transfer optimisation: the "standard round" problem.

Given the squad you already own, which at most ``k`` swaps most improve the
discounted multi-round value, subject to what the sales actually fund?

    maximise    sum_i x_i * (V_i - lambda * Var_i)  -  friction * sum_i b_i
    subject to  x_i = 1 - s_i          for i currently owned
                x_i = b_i              for i not owned
                sum_i b_i <= k
                sum_i b_i == sum_i s_i                     (squad size fixed)
                sum_i b_i * price_i <= bank + sum_i s_i * sell_price_i
                + the usual position and club constraints

Two details that matter in practice and are easy to get wrong:

* **Selling funds buying.** The budget constraint is not "spend less than the
  bank"; it is "spend less than the bank plus what you raise". Treating it as a
  fixed budget makes the optimiser far too conservative.
* **Transfer friction.** Without a small per-transfer penalty the solver will
  happily burn all four transfers to gain 0.3 projected points, spending
  flexibility you will want next round. ``optimiser.transfer_friction`` is the
  price of that flexibility; set it to 0 for a pure single-round answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pulp

from elfantasy.config import Settings
from elfantasy.models import Squad
from elfantasy.optimize.solver import solve
from elfantasy.optimize.squad import Candidate, InfeasibleError, _add_common_constraints


@dataclass
class TransferMove:
    out_player: Candidate
    in_player: Candidate

    @property
    def value_gain(self) -> float:
        return self.in_player.value - self.out_player.value

    @property
    def price_delta(self) -> float:
        return self.in_player.price - self.out_player.price


@dataclass
class TransferPlan:
    moves: list[TransferMove] = field(default_factory=list)
    kept: list[Candidate] = field(default_factory=list)
    value_before: float = 0.0
    value_after: float = 0.0
    next_round_before: float = 0.0
    next_round_after: float = 0.0
    bank_before: float = 0.0
    bank_after: float = 0.0
    status: str = "unsolved"

    @property
    def n_transfers(self) -> int:
        return len(self.moves)

    @property
    def value_gain(self) -> float:
        return self.value_after - self.value_before

    @property
    def squad_ids(self) -> list[str]:
        return [c.player_id for c in self.kept] + [m.in_player.player_id for m in self.moves]


def sell_price(candidate: Candidate, squad: Squad, settings: Settings) -> float:
    """What you receive for selling a player you own."""

    mode = str(settings.rules.get("pricing.sell_at", "current"))
    if mode == "current":
        return candidate.price
    paid = squad.purchase_prices.get(candidate.player_id)
    if paid is None:
        return candidate.price
    retention = float(settings.rules.get("pricing.profit_retention", 1.0))
    profit = max(candidate.price - paid, 0.0)
    loss = min(candidate.price - paid, 0.0)
    return paid + retention * profit + loss


def optimise_transfers(
    candidates: list[Candidate],
    squad: Squad,
    settings: Settings,
    *,
    max_transfers: int | None = None,
    risk_aversion: float | None = None,
    friction: float | None = None,
    force_transfers: int | None = None,
    protect: set[str] | None = None,
    solver: pulp.LpSolver | None = None,
) -> TransferPlan:
    """Find the best set of at most ``max_transfers`` swaps."""

    pool = {c.player_id: c for c in candidates if c.price > 0}
    missing = [pid for pid in squad.player_ids if pid not in pool]
    if missing:
        raise InfeasibleError(
            "these squad members are not in the candidate pool (no projection?): "
            + ", ".join(missing)
        )

    k = max_transfers if max_transfers is not None else settings.transfers_per_round
    lam = (
        risk_aversion
        if risk_aversion is not None
        else float(settings.model.get("optimiser.risk_aversion"))
    )
    fric = (
        friction
        if friction is not None
        else float(settings.model.get("optimiser.transfer_friction"))
    )
    protect = protect or set()

    owned = set(squad.player_ids)
    size = len(squad.player_ids)

    prob = pulp.LpProblem("elfantasy_transfers", pulp.LpMaximize)
    x = {pid: pulp.LpVariable(f"x_{pid}", cat=pulp.LpBinary) for pid in pool}
    buy = {pid: pulp.LpVariable(f"b_{pid}", cat=pulp.LpBinary) for pid in pool if pid not in owned}
    sell = {pid: pulp.LpVariable(f"s_{pid}", cat=pulp.LpBinary) for pid in owned}

    for pid in pool:
        if pid in owned:
            prob += x[pid] == 1 - sell[pid], f"link_owned_{pid}"
        else:
            prob += x[pid] == buy[pid], f"link_new_{pid}"

    n_in = pulp.lpSum(buy.values())
    n_out = pulp.lpSum(sell.values())
    prob += n_in == n_out, "squad_size_preserved"
    if force_transfers is not None:
        prob += n_in == force_transfers, "forced_transfers"
    else:
        prob += n_in <= k, "transfer_cap"

    raised = pulp.lpSum(sell[pid] * sell_price(pool[pid], squad, settings) for pid in owned)
    spent = pulp.lpSum(buy[pid] * pool[pid].price for pid in buy)
    prob += spent <= squad.bank + raised, "budget"

    for pid in protect:
        if pid in sell:
            prob += sell[pid] == 0, f"protect_{pid}"

    _add_common_constraints(prob, x, pool, settings, squad_size=size)

    utility = pulp.lpSum(x[pid] * (pool[pid].value - lam * pool[pid].variance) for pid in pool)
    prob += utility - fric * n_in

    status = solve(prob, solver)
    if status != "Optimal":
        raise InfeasibleError(
            f"solver returned {status}. If your current squad already breaks a "
            f"constraint (e.g. three players from one club after a rule change), "
            f"relax it in config/rules.yaml or raise --max-transfers."
        )

    sold = [pool[pid] for pid in owned if sell[pid].value() and sell[pid].value() > 0.5]
    bought = [pool[pid] for pid in buy if buy[pid].value() and buy[pid].value() > 0.5]
    kept = [pool[pid] for pid in owned if pid not in {c.player_id for c in sold}]

    moves = _pair_moves(sold, bought)

    raised_value = sum(sell_price(c, squad, settings) for c in sold)
    spent_value = sum(c.price for c in bought)

    return TransferPlan(
        moves=moves,
        kept=sorted(kept, key=lambda c: -c.value),
        value_before=sum(pool[pid].value for pid in owned),
        value_after=sum(c.value for c in kept) + sum(c.value for c in bought),
        next_round_before=sum(pool[pid].next_round_pir for pid in owned),
        next_round_after=sum(c.next_round_pir for c in kept)
        + sum(c.next_round_pir for c in bought),
        bank_before=squad.bank,
        bank_after=squad.bank + raised_value - spent_value,
        status=status,
    )


def _pair_moves(sold: list[Candidate], bought: list[Candidate]) -> list[TransferMove]:
    """Pair sales with purchases for presentation.

    The MILP does not pair them -- it just picks a set -- but a human reads
    "X out, Y in" far more easily than two lists. Pair on position first, then
    on price proximity, so the pairing reads naturally.
    """

    remaining = list(bought)
    moves: list[TransferMove] = []
    for out_player in sorted(sold, key=lambda c: -c.price):
        same_pos = [c for c in remaining if c.position == out_player.position]
        candidates = same_pos or remaining
        if not candidates:
            break
        best = min(candidates, key=lambda c: abs(c.price - out_player.price))
        remaining.remove(best)
        moves.append(TransferMove(out_player=out_player, in_player=best))
    return moves


def transfer_ladder(
    candidates: list[Candidate],
    squad: Squad,
    settings: Settings,
    *,
    max_transfers: int | None = None,
    **kwargs,
) -> list[TransferPlan]:
    """Best plan for exactly 0, 1, 2, ... transfers.

    This is the output that actually answers the manager's question. The gain
    from the fourth transfer is almost always far smaller than from the first,
    and seeing the ladder makes it obvious when to stop -- especially when
    transfers do not roll over and a marginal swap costs you optionality later.
    """

    k = max_transfers if max_transfers is not None else settings.transfers_per_round
    plans = []
    for n in range(0, k + 1):
        try:
            plans.append(
                optimise_transfers(candidates, squad, settings, force_transfers=n, **kwargs)
            )
        except InfeasibleError:
            continue
    return plans
