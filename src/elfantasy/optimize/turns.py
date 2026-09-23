"""Turn-aware evaluation and search.

A round is played over several game days ("Turns"). Between Turns the rules
allow field/bench swaps and captain changes as long as the player coming in
has not played yet. That turns every later-Turn player into an option:

* **Bench him before the first Turn.** Afterwards he can replace whichever
  first-Turn field player actually scored worst. Starting him instead fixes in
  advance who sits at bench weight, which is never better: the realised worst
  of six first-Turn players scores no more than any one of them you might
  have benched blind.
* **Captain a first-Turn starter.** If the captain flops, move the armband to a
  later-Turn starter. The captain's value becomes
  ``max(first-Turn captain's actual score, later starter's expectation)``.

The static MILP in :mod:`elfantasy.optimize.lineup` cannot see either effect,
so this module simulates the round and applies the optimal policy in every
simulation:

1. correlated game outcomes -- one margin per fixture moves every player in it
   (team PIR rises ~1.09 per point of margin) and decides win bonuses and the
   coach's score;
2. player scores, including injury risk and coach's-decision DNPs for fringe
   rotation players;
3. after the first Turn, the best legal combination of promotions and captain
   changes, chosen on *actual* first-Turn scores and *expected* later ones;
4. expected price changes from the game's price rule, and a value for credits
   left unspent.

Limitations: one decision point (after the first Turn), so a round spread over
three days is treated as "first day" vs "the rest"; and same-team correlation
beyond the shared game margin is not modelled.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from elfantasy import scoring
from elfantasy.rules import GameRules


@dataclass
class SimPlayer:
    player_id: str
    name: str
    pos: str
    club: str
    opponent: str
    price: float
    late: bool  # plays after the first Turn
    points_if_available: float  # expected fantasy points if fit (win bonus included)
    play_prob: float  # P(fit and in the squad)
    minutes: float  # expected minutes when he plays
    win_prob: float
    sd_ratio: float = 0.55  # game-to-game SD of his score / its mean
    future_value: float = 0.0  # later rounds, already discounted


@dataclass
class SimCoach:
    coach_id: str
    name: str
    club: str
    price: float
    future_value: float = 0.0


@dataclass(frozen=True)
class TurnLineup:
    starters: tuple[int, ...]
    sixth: int | None
    captain: int


@dataclass
class TurnEvaluation:
    objective: float
    first_round: np.ndarray  # realised first-round points per simulation
    lineup: TurnLineup
    future: float
    price_value: float
    bank_value: float
    cost: float
    promoted: dict[int, float] = field(default_factory=dict)  # late player -> P(promoted)
    captained: dict[int, float] = field(default_factory=dict)  # player -> P(final captain)


class TurnSimulator:
    """Simulates one round for any roster drawn from a fixed player pool.

    All rosters are scored on the same simulated games (common random
    numbers), so differences between two rosters are far less noisy than
    either roster's own score.
    """

    def __init__(
        self,
        players: list[SimPlayer],
        coaches: list[SimCoach],
        rules: GameRules,
        *,
        n: int = 6000,
        seed: int = 11,
        budget: float | None = None,
        credit_value: float = 1.0,
        bank_value: float | None = None,
        margin_sd: float = 11.5,
        pir_per_margin: float = 1.09 / 93.5,
        dnp_scale: float = 0.45,
        dnp_minutes: float = 6.0,
        dnp_cap: float = 0.4,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.rules = rules
        self.players = players
        self.coaches = coaches
        self.n = n
        self.budget = budget if budget is not None else rules.budget
        self.credit_value = credit_value
        # A credit left in the bank is spendable from next round only.
        self.bank_value = bank_value if bank_value is not None else 0.35 * credit_value

        # One margin draw per fixture, mirrored for the two clubs.
        draws: dict[tuple[str, str], np.ndarray] = {}
        self.margin: dict[str, tuple[float, np.ndarray]] = {}
        for p in players:
            key = tuple(sorted((p.club, p.opponent)))
            draws.setdefault(key, rng.standard_normal(n))
            if p.club not in self.margin:
                sign = 1.0 if p.club == key[0] else -1.0
                mu = float(norm.ppf(min(max(p.win_prob, 1e-4), 1 - 1e-4)) * margin_sd)
                self.margin[p.club] = (mu, mu + margin_sd * sign * draws[key])

        R = np.zeros((len(players), n))
        for i, p in enumerate(players):
            mu, m = self.margin[p.club]
            # Coach's-decision DNPs: likelier the fewer minutes a player
            # projects. Mean-preserving, so it adds risk without moving the
            # expectation -- risk that matters for prices, since a player who
            # sits scores 0 and loses value unless he is already at the floor.
            p_dnp = min(dnp_cap, dnp_scale * np.exp(-p.minutes / dnp_minutes))
            base = p.points_if_available / scoring.expected_win_bonus_factor(p.win_prob, rules)
            base /= 1.0 - p_dnp
            team = 1.0 + pir_per_margin * (m - mu)
            idio = np.sqrt(max(p.sd_ratio**2 - (pir_per_margin * margin_sd) ** 2, 0.05))
            pir = base * team + base * idio * rng.standard_normal(n)
            plays = rng.random(n) < p.play_prob * (1.0 - p_dnp)
            R[i] = np.where(plays, scoring.fantasy_score(pir, m > 0, rules), 0.0)
        self.R = R
        self.E = R.mean(axis=1)
        self.p_zero = (R == 0).mean(axis=1)
        self.late = np.array([p.late for p in players])
        self.pos = [p.pos for p in players]
        self.price = np.array([p.price for p in players])
        self.future = np.array([p.future_value for p in players])
        self.dprice = np.array(
            [scoring.price_change(R[i], p.price, rules).mean() for i, p in enumerate(players)]
        )
        self.coach_points = []
        for c in coaches:
            m = self.margin.get(c.club)
            self.coach_points.append(
                scoring.coach_points(m[1], rules) if m else np.zeros(n)  # bye: no game
            )
        self.index = {p.player_id: i for i, p in enumerate(players)}
        self.coach_index = {c.coach_id: k for k, c in enumerate(coaches)}

    # ------------------------------------------------------------------ lineup
    def legal(self, starters) -> bool:
        return self.rules.is_legal_formation(
            tuple(sum(1 for i in starters if self.pos[i] == x) for x in "GFC")
        )

    def initial_lineups(self, ids: list[int], limit: int = 6) -> list[TurnLineup]:
        """Plausible pre-Turn lineups: first-Turn players in the field first."""

        first = sorted([i for i in ids if not self.late[i]], key=lambda i: -self.E[i])
        later = sorted([i for i in ids if self.late[i]], key=lambda i: -self.E[i])
        pool = first + later
        k = self.rules.field_slots

        def lineups(width):
            for field_set in itertools.combinations(pool[:width], k):
                sixths = (
                    sorted(field_set, key=lambda i: self.E[i]) if self.rules.sixth_man else [None]
                )
                for sixth in sixths:
                    starters = tuple(i for i in field_set if i != sixth)
                    if not self.legal(starters):
                        continue
                    caps = sorted(starters, key=lambda i: (self.late[i], -self.E[i]))[:2]
                    rank = sum(pool.index(i) for i in field_set)
                    for c in caps:
                        yield rank, TurnLineup(starters, sixth, c)

        found = list(lineups(min(len(pool), k + 2)))
        if not found:  # e.g. both centres ranked outside the screened players
            found = list(lineups(len(pool)))
        found.sort(key=lambda t: t[0])
        seen, out = set(), []
        for _, lu in found:
            key = (frozenset(lu.starters), lu.sixth, lu.captain)
            if key not in seen:
                seen.add(key)
                out.append(lu)
            if len(out) >= limit:
                break
        return out

    def _weights(self, ids: list[int], lineup: TurnLineup) -> np.ndarray:
        """Every legal end-of-round weighting reachable from ``lineup``."""

        rules = self.rules
        slot = {i: k for k, i in enumerate(ids)}
        field_now = list(lineup.starters) + ([lineup.sixth] if lineup.sixth is not None else [])
        bench_late = (
            [i for i in ids if i not in field_now and self.late[i]]
            if rules.field_bench_swaps
            else []
        )
        weights = set()
        for plan in itertools.product([None, *field_now], repeat=len(bench_late)):
            used = [t for t in plan if t is not None]
            if len(used) != len(set(used)):
                continue
            starters, sixth = list(lineup.starters), lineup.sixth
            for j, target in zip(bench_late, plan, strict=True):
                if target is None:
                    continue
                if target == sixth:
                    sixth = j
                else:
                    starters[starters.index(target)] = j
            if not self.legal(starters):
                continue
            captains = [c for c in starters if c == lineup.captain]
            if rules.captain_switch:
                captains += [c for c in starters if self.late[c] and c != lineup.captain]
            final_field = set(starters) | ({sixth} if sixth is not None else set())
            for c in captains:
                w = np.full(len(ids), rules.bench_weight)
                for i in final_field:
                    w[slot[i]] = 1.0
                w[slot[c]] += rules.captain_multiplier - 1.0
                weights.add(tuple(w))
        return np.array(sorted(weights))

    def play_round(self, ids: list[int], lineup: TurnLineup):
        """Realised player points per simulation under the optimal policy."""

        W = self._weights(ids, lineup)
        R = self.R[ids]
        # After the first Turn: actual scores for players who have played,
        # expectations for those who have not.
        decide = np.where(self.late[ids][:, None], self.E[ids][:, None], R)
        choice = np.argmax(W @ decide, axis=0)
        realised = (W @ R)[choice, np.arange(self.n)]
        return realised, W, choice

    # -------------------------------------------------------------- evaluation
    def cost(self, ids: list[int], coach: int | None) -> float:
        total = float(self.price[ids].sum())
        if coach is not None:
            total += self.coaches[coach].price
        return total

    def evaluate(self, ids: list[int], coach: int | None, detail: bool = False) -> TurnEvaluation:
        ids = list(ids)
        best = None
        for lineup in self.initial_lineups(ids):
            realised, W, choice = self.play_round(ids, lineup)
            if best is None or realised.mean() > best[0].mean():
                best = (realised, lineup, W, choice)
        if best is None:
            raise ValueError("no legal lineup exists for this roster")
        realised, lineup, W, choice = best
        first_round = realised + (self.coach_points[coach] if coach is not None else 0.0)
        future = float(self.future[ids].sum())
        if coach is not None:
            future += self.coaches[coach].future_value
        price_value = self.credit_value * float(self.dprice[ids].sum())
        cost = self.cost(ids, coach)
        bank_value = self.bank_value * (self.budget - cost)
        ev = TurnEvaluation(
            objective=float(first_round.mean()) + future + price_value + bank_value,
            first_round=first_round,
            lineup=lineup,
            future=future,
            price_value=price_value,
            bank_value=bank_value,
            cost=cost,
        )
        if detail:
            chosen = W[choice]  # (n, roster) final weights per simulation
            cap_w = self.rules.captain_multiplier
            for k, i in enumerate(ids):
                if self.late[i] and i not in lineup.starters and i != lineup.sixth:
                    ev.promoted[i] = float((chosen[:, k] >= 1.0).mean())
                share = float((chosen[:, k] >= cap_w).mean())
                if share > 0:
                    ev.captained[i] = share
        return ev


# ---------------------------------------------------------------------- search
def candidate_pool(
    sim: TurnSimulator, per_pos: int = 22, by_value: int = 8
) -> dict[str, list[int]]:
    """Players worth trying at each position: best totals plus best per credit."""

    score = sim.E + sim.future + sim.credit_value * sim.dprice
    out = {}
    for pos in "GFC":
        idx = [i for i in range(len(sim.players)) if sim.pos[i] == pos]
        top = sorted(idx, key=lambda i: -score[i])[:per_pos]
        value = sorted(idx, key=lambda i: -score[i] / max(sim.price[i], 1e-9))[:by_value]
        out[pos] = list(dict.fromkeys(top + value))
    return out


def local_search(
    sim: TurnSimulator,
    ids: list[int],
    coach: int | None,
    candidates: dict[str, list[int]],
    *,
    pair_candidates: dict[str, list[int]] | None = None,
    current: tuple[set[int], int | None] | None = None,
    max_transfers: int | None = None,
    max_passes: int = 12,
    min_gain: float = 0.05,
    log: Callable[[str], None] | None = None,
) -> tuple[list[int], int | None, float]:
    """Hill-climb on the simulated objective with single and paired swaps.

    ``current`` = (owned player indices, owned coach index) restricts moves to
    at most ``max_transfers`` changes, counting a coach change if the rules say
    so. Pair moves run only once single swaps stop improving; they get past
    budget-locked optima where an upgrade needs a simultaneous downgrade.
    """

    ids = list(ids)
    max_club = sim.rules.max_per_club

    def feasible(new_ids, new_coach):
        if sim.cost(new_ids, new_coach) > sim.budget + 1e-9:
            return False
        clubs = [sim.players[i].club for i in new_ids]
        if max(clubs.count(c) for c in set(clubs)) > max_club:
            return False
        if current is not None and max_transfers is not None:
            owned, owned_coach = current
            moves = len(set(new_ids) - owned)
            if (
                sim.rules.coach_counts_as_transfer
                and owned_coach is not None
                and new_coach != owned_coach
            ):
                moves += 1
            if moves > max_transfers:
                return False
        return True

    def score(new_ids, new_coach):
        return sim.evaluate(new_ids, new_coach).objective

    cur = score(ids, coach)
    for step in range(max_passes):
        best: tuple[float, tuple] = (cur, ())
        for slot, i in enumerate(ids):
            for q in candidates[sim.pos[i]]:
                if q in ids:
                    continue
                new = ids.copy()
                new[slot] = q
                if feasible(new, coach) and (v := score(new, coach)) > best[0] + min_gain:
                    best = (v, ("player", slot, q))
        for k in range(len(sim.coaches)):
            if k != coach and feasible(ids, k) and (v := score(ids, k)) > best[0] + min_gain:
                best = (v, ("coach", k))
        if not best[1] and pair_candidates:
            for (s1, i1), (s2, i2) in itertools.combinations(list(enumerate(ids)), 2):
                for q1 in pair_candidates[sim.pos[i1]]:
                    if q1 in ids:
                        continue
                    for q2 in pair_candidates[sim.pos[i2]]:
                        if q2 in ids or q2 == q1:
                            continue
                        new = ids.copy()
                        new[s1], new[s2] = q1, q2
                        if feasible(new, coach) and (v := score(new, coach)) > best[0] + min_gain:
                            best = (v, ("pair", s1, q1, s2, q2))
        if not best[1]:
            break
        move = best[1]
        if move[0] == "player":
            _, slot, q = move
            msg = f"{sim.players[ids[slot]].name} -> {sim.players[q].name}"
            ids[slot] = q
        elif move[0] == "coach":
            msg = f"coach {sim.coaches[coach].name if coach is not None else '-'} -> {sim.coaches[move[1]].name}"
            coach = move[1]
        else:
            _, s1, q1, s2, q2 = move
            msg = (
                f"{sim.players[ids[s1]].name} + {sim.players[ids[s2]].name} -> "
                f"{sim.players[q1].name} + {sim.players[q2].name}"
            )
            ids[s1], ids[s2] = q1, q2
        if log:
            log(f"pass {step + 1}: {msg}  ({cur:.1f} -> {best[0]:.1f})")
        cur = best[0]
    return ids, coach, cur
