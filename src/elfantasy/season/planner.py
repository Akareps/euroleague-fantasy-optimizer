"""Transfers, lineup and Turn plan for a round, from the season projections.

1. The lineup MILP (static, fast) gives starting rosters and the value of a
   credit: how much one more credit in the bank would add to the objective.
2. A Turn simulator scores rosters on simulated games, playing the lineup the
   way you would: re-planning after each game day.
3. Local search from several starts (the MILP plans, MILP plans holding more
   last-day players on the bench, and your current squad) with single and
   paired swaps, within the transfer limit; finalists re-scored on fresh,
   larger simulations.
4. A transfer ladder: the best plan with exactly 0, 1, ... transfers, to show
   what each extra move is worth.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from elfantasy import report
from elfantasy.optimize.lineup import (
    CurrentSquad,
    LineupCoach,
    LineupPlayer,
    future_slot_weight,
    optimise_lineup,
)
from elfantasy.optimize.squad import InfeasibleError
from elfantasy.optimize.turns import (
    SimCoach,
    SimPlayer,
    TurnSimulator,
    candidate_pool,
    local_search,
)
from elfantasy.plan import LineupPlan
from elfantasy.rules import GameRules
from elfantasy.season.inputs import PrivateInputs


@dataclass
class PlanResult:
    plan: LineupPlan
    hold: float | None  # objective of keeping the squad as it is
    hold_first_round: float | None
    ladder: list[tuple[int, float, float, list[str], str | None]] = field(default_factory=list)
    finalists: list[float] = field(default_factory=list)


class RoundPlanner:
    """Plans one round from a projection produced by :class:`SeasonModel`."""

    def __init__(
        self,
        projection: dict,
        rules: GameRules,
        private: PrivateInputs | None,
        *,
        gamma: float = 0.4,
        log: Callable[[str], None] = print,
    ) -> None:
        self.proj = projection
        self.rules = rules
        self.private = private
        self.gamma = gamma
        self.log = log
        self.round = int(projection["round"])
        self.rounds = [int(r) for r in projection["rounds"]]
        self.squad = list(private.squad) if private and private.squad else []
        self.coach = private.coach if private else None
        self.bank = private.bank if private and private.bank is not None else 0.0
        self.lp, self.lc, self.sp, self.sc = self._inputs()

    # ---------------------------------------------------------------- inputs
    def _inputs(self):
        first, later = str(self.round), self.rounds[1:]
        w = future_slot_weight(self.rules)
        last_day = max(p["proj"][first]["turn"] for p in self.proj["players"] if first in p["proj"])
        lp, lc, sp, sc = [], [], [], []
        for p in self.proj["players"]:
            proj = p["proj"]
            if first not in proj:
                continue
            ev = {r: proj[str(r)]["ev"] for r in self.rounds if str(r) in proj}
            r0 = proj[first]
            # MILP "late": the last game day -- the bench the Turn options want.
            lp.append(LineupPlayer(p["name"], p["name"], p["pos"], p["club"], p["price"], ev,
                                   r0["turn"] == last_day))  # fmt: skip
            if r0["play"] <= 0.02 and p["name"] not in self.squad:
                continue
            future = w * sum(self.gamma ** (k + 1) * ev.get(r, 0.0) for k, r in enumerate(later))
            sp.append(SimPlayer(p["name"], p["name"], p["pos"], p["club"], r0["opp"], p["price"],
                                r0["turn"] > 0, r0["if_plays"], r0["play"], r0["min"], r0["win"],
                                p["sd_per_pir"], future, turn=r0["turn"]))  # fmt: skip
        for c in self.proj["coaches"]:
            ev = {r: c["proj"][str(r)]["ev"] for r in self.rounds if str(r) in c["proj"]}
            lc.append(LineupCoach(c["name"], c["name"], c["club"], c["price"], ev))
            future = sum(self.gamma ** (k + 1) * ev.get(r, 0.0) for k, r in enumerate(later))
            sc.append(SimCoach(c["name"], c["name"], c["club"], c["price"], future))
        return lp, lc, sp, sc

    @property
    def max_transfers(self) -> int | None:
        if not self.squad or self.rules.unlimited_transfers_before(self.round):
            return None
        return self.rules.transfers_per_round

    def _current(self, extra: float = 0.0) -> CurrentSquad | None:
        return CurrentSquad(self.squad, self.coach, self.bank + extra) if self.squad else None

    def budget(self) -> float:
        if not self.squad:
            return self.rules.budget
        missing = [n for n in self.squad if n not in {p.player_id for p in self.lp}]
        if missing:
            raise ValueError(f"not in the projections (check names): {', '.join(missing)}")
        coach_price = next((c.price for c in self.lc if c.coach_id == self.coach), 0.0)
        return self.bank + sum(p.price for p in self.lp if p.player_id in self.squad) + coach_price

    def credit_value(self) -> tuple[float, object]:
        """Objective points one more credit would add (MILP), and the base MILP."""
        kw = {"rounds": self.rounds, "gamma": self.gamma, "max_transfers": self.max_transfers}
        if self.squad:
            base = optimise_lineup(self.lp, self.lc, self.rules, **kw, current=self._current())
            richer = optimise_lineup(self.lp, self.lc, self.rules, **kw, current=self._current(1.0))
        else:
            base = optimise_lineup(self.lp, self.lc, self.rules, **kw, budget=self.rules.budget)
            richer = optimise_lineup(
                self.lp, self.lc, self.rules, **kw, budget=self.rules.budget + 1
            )
        return max(richer.objective - base.objective, 0.0), base

    def title(self) -> str:
        first = str(self.round)
        days = sorted(
            {p["proj"][first]["date"][:10] for p in self.proj["players"] if first in p["proj"]}
        )
        when = " / ".join(
            date.fromisoformat(d).strftime("%a %d %b").replace(" 0", " ") for d in days
        )
        return f"Round {self.round} ({when})"

    # ---------------------------------------------------------------- search
    def search(self, *, n_search: int = 6000, n_final: int = 40000, seeds=(11, 99)) -> PlanResult:
        rules, log = self.rules, self.log
        credit, base = self.credit_value()
        log(f"one credit is worth {credit:.2f} points (R{self.round} + discounted later rounds)")
        budget = self.budget()
        sim = TurnSimulator(self.sp, self.sc, rules, n=n_search, seed=seeds[0], budget=budget,
                            credit_value=credit)  # fmt: skip
        owned = (
            ({sim.index[n] for n in self.squad}, sim.coach_index.get(self.coach))
            if self.squad
            else None
        )
        cands = candidate_pool(sim)
        pairs = candidate_pool(sim, per_pos=9, by_value=3)
        kw = {"rounds": self.rounds, "gamma": self.gamma}
        kw.update(current=self._current()) if self.squad else kw.update(budget=budget)
        limit = self.max_transfers

        def ids_of(sol):
            return [sim.index[p.player_id] for p in sol.roster], sim.coach_index[sol.coach.coach_id]

        label = f"{limit} transfers" if limit is not None else "free rebuild"
        starts = {f"MILP ({label})": ids_of(base)}
        for k in (2, 3):
            with contextlib.suppress(InfeasibleError):
                starts[f"MILP, >={k} last-day bench"] = ids_of(
                    optimise_lineup(
                        self.lp, self.lc, rules, **kw, max_transfers=limit, min_late_bench=k
                    )
                )
        if self.squad:
            starts["hold"] = ([sim.index[n] for n in self.squad], sim.coach_index[self.coach])
        results = []
        for name, (ids, coach) in starts.items():
            log(f"\nsearch from {name} (objective {sim.evaluate(ids, coach).objective:.1f})")
            results.append(local_search(sim, ids, coach, cands, pair_candidates=pairs, current=owned,
                                        max_transfers=limit, log=log))  # fmt: skip

        ladder_raw = {}
        if self.squad and limit is not None:
            for k in range(0, limit + 1):
                try:
                    sol = optimise_lineup(self.lp, self.lc, rules, **kw, max_transfers=k)
                except InfeasibleError:
                    continue
                ids, coach = ids_of(sol)
                ladder_raw[k] = local_search(sim, ids, coach, cands, current=owned, max_transfers=k)

        final = TurnSimulator(self.sp, self.sc, rules, n=n_final, seed=seeds[1], budget=budget,
                              credit_value=credit)  # fmt: skip

        def remap(ids):
            return [final.index[sim.players[i].player_id] for i in ids]

        scored = sorted(
            ((final.evaluate(remap(ids), c).objective, remap(ids), c) for ids, c, _ in results),
            key=lambda t: -t[0],
        )
        out = PlanResult(plan=None, hold=None, hold_first_round=None,  # type: ignore[arg-type]
                         finalists=[s[0] for s in scored])  # fmt: skip
        log(f"\nfinalists on fresh simulations: {[round(s[0], 1) for s in scored]}")
        if ladder_raw:
            log("\ntransfer ladder (fresh simulations):")
            prev = None
            mine = {final.index[n] for n in self.squad}
            for k, (ids, coach, _) in sorted(ladder_raw.items()):
                e = final.evaluate(remap(ids), coach)
                moves = sorted(set(remap(ids)) - mine)
                new_coach = (
                    final.coaches[coach].name
                    if final.coaches[coach].coach_id != self.coach
                    else None
                )
                gain = "" if prev is None else f"  (+{e.objective - prev:.1f})"
                names = [final.players[i].name for i in moves]
                log(f"  {k} moves: objective {e.objective:6.1f}  R{self.round} {e.first_round.mean():6.1f}"
                    f"{gain}  in: {', '.join(names)}" + (f" + coach {new_coach}" if new_coach else ""))  # fmt: skip
                out.ladder.append((k, e.objective, float(e.first_round.mean()), names, new_coach))
                prev = e.objective

        _, ids, coach = scored[0]
        ev = final.evaluate(ids, coach, detail=True)
        roster = {final.players[i].player_id for i in ids}
        try:
            fixed = optimise_lineup(
                [p for p in self.lp if p.player_id in roster],
                [c for c in self.lc if c.coach_id == final.coaches[coach].coach_id],
                rules, rounds=[self.round], budget=1e6,
            )  # fmt: skip
            static_first = fixed.first_round
        except InfeasibleError:
            static_first = float("nan")
        plan = LineupPlan(sim=final, ids=ids, coach=coach, evaluation=ev, static=base,
                          credit_value=credit, static_first_round=static_first)  # fmt: skip
        if self.squad:
            plan.sold = [n for n in self.squad if n not in roster]
            plan.bought = sorted(roster - set(self.squad))
            plan.coach_changed = final.coaches[coach].coach_id != self.coach
            hold = final.evaluate(
                [final.index[n] for n in self.squad], final.coach_index[self.coach]
            )
            out.hold, out.hold_first_round = hold.objective, float(hold.first_round.mean())
        out.plan = plan
        return out

    def show(self, result: PlanResult) -> None:
        report.lineup_plan(result.plan, title=self.title())
        ev = result.plan.evaluation
        first = ev.first_round
        if result.hold is not None:
            self.log(
                f"\nholding the squad instead: objective {result.hold:.1f}, R{self.round} "
                f"{result.hold_first_round:.1f} (plan: {ev.objective:.1f}, R{self.round} {first.mean():.1f}, "
                f"p10-p90 {np.percentile(first, 10):.0f}-{np.percentile(first, 90):.0f})"
            )

    def save(self, result: PlanResult, path) -> None:
        sim, plan = result.plan.sim, result.plan
        lu = plan.evaluation.lineup
        doc = {
            "round": self.round,
            "roster": sorted(sim.players[i].name for i in plan.ids),
            "coach": sim.coaches[plan.coach].name if plan.coach is not None else None,
            "captain": sim.players[lu.captain].name,
            "starters": [sim.players[i].name for i in lu.starters],
            "sixth": sim.players[lu.sixth].name if lu.sixth is not None else None,
            "sold": plan.sold,
            "bought": plan.bought,
            "objective": plan.evaluation.objective,
            "hold_objective": result.hold,
            "ladder": result.ladder,
        }
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")

    # --------------------------------------------------------------- compare
    def compare(self, plans: dict[str, tuple[list[str], str | None]], *, seeds=(101, 202, 303),
                n: int = 40000) -> list[dict]:  # fmt: skip
        """Named rosters head to head on the same simulations, several seeds."""
        credit, _ = self.credit_value()
        budget = self.budget()
        plans = dict(plans)
        if self.squad:
            plans.setdefault("hold", (self.squad, self.coach))
        res: dict[str, list] = {k: [] for k in plans}
        for seed in seeds:
            sim = TurnSimulator(self.sp, self.sc, self.rules, n=n, seed=seed, budget=budget,
                                credit_value=credit)  # fmt: skip
            for name, (names, coach) in plans.items():
                missing = [x for x in names if x not in sim.index]
                if missing:
                    raise ValueError(f"plan {name!r}: not in the projections: {', '.join(missing)}")
                ids = [sim.index[x] for x in names]
                c = sim.coach_index[coach] if coach else None
                e = sim.evaluate(ids, c)
                res[name].append((e.objective, e.first_round.mean(), np.percentile(e.first_round, 10),
                                  sim.cost(ids, c)))  # fmt: skip
        rows = []
        for name, v in res.items():
            o = np.array(v)
            rows.append({"plan": name, "objective": float(o[:, 0].mean()), "seed_sd": float(o[:, 0].std()),
                         "first_round": float(o[:, 1].mean()), "p10": float(o[:, 2].mean()),
                         "cost": float(o[0, 3]), "bank": float(budget - o[0, 3])})  # fmt: skip
        return sorted(rows, key=lambda r: -r["objective"])
