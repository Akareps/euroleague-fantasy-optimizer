"""Plan Round 1 with the package optimisers on the preseason projections.

    python fetch.py && python build.py && python project.py && python run.py

``project.py`` writes ``projections.json`` (Rounds 1-3). This script turns it
into LineupPlayers/SimPlayers, solves the lineup MILP for starting points,
hill-climbs on the Turn simulator, and prints the lineup and the Turn plan.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA  # noqa: E402

from elfantasy import report  # noqa: E402
from elfantasy.config import load_settings  # noqa: E402
from elfantasy.optimize.lineup import (  # noqa: E402
    LineupCoach,
    LineupPlayer,
    future_slot_weight,
    optimise_lineup,
)
from elfantasy.optimize.squad import InfeasibleError  # noqa: E402
from elfantasy.optimize.turns import (  # noqa: E402
    SimCoach,
    SimPlayer,
    TurnSimulator,
    candidate_pool,
    local_search,
)
from elfantasy.plan import LineupPlan  # noqa: E402
from elfantasy.rules import GameRules  # noqa: E402

GAMMA = 0.4
ROUNDS = [1, 2, 3]


def inputs(rules: GameRules):
    d = json.loads(Path(DATA / "projections.json").read_text(encoding="utf-8"))
    w_future = future_slot_weight(rules)
    lp, sp, lc, sc = [], [], [], []
    for p in d["players"]:
        proj = p["proj"]
        if "1" not in proj:
            continue
        ev = {r: proj[str(r)]["ev"] for r in ROUNDS if str(r) in proj}
        r1 = proj["1"]
        late = r1["turn"] != "Thu"
        lp.append(LineupPlayer(p["name"], p["name"], p["pos"], p["club"], p["price"], ev, late))
        if r1["play"] <= 0.02:
            continue
        future = w_future * sum(GAMMA**k * ev.get(r, 0.0) for k, r in enumerate(ROUNDS) if k)
        sp.append(
            SimPlayer(
                player_id=p["name"],
                name=p["name"],
                pos=p["pos"],
                club=p["club"],
                opponent=r1["opp"],
                price=p["price"],
                late=late,
                points_if_available=r1["if_plays"],
                play_prob=r1["play"],
                minutes=r1["min"],
                win_prob=r1["win"],
                sd_ratio=p["sd_per_pir"],
                future_value=future,
            )
        )
    for c in d["coaches"]:
        ev = {r: c["proj"][str(r)]["ev"] for r in ROUNDS}
        lc.append(LineupCoach(c["name"], c["name"], c["club"], c["price"], ev))
        future = sum(GAMMA**k * ev[r] for k, r in enumerate(ROUNDS) if k)
        sc.append(SimCoach(c["name"], c["name"], c["club"], c["price"], future))
    return lp, lc, sp, sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=6000, help="simulations during the search")
    ap.add_argument("--final-sims", type=int, default=40000, help="simulations for the final score")
    ap.add_argument(
        "--credit-value", type=float, default=1.0, help="multiplier on the shadow price"
    )
    ap.add_argument("--thorough", action="store_true", help="paired swaps (slow)")
    args = ap.parse_args()

    rules = GameRules.from_settings(load_settings())
    lp, lc, sp, sc = inputs(rules)

    base = optimise_lineup(lp, lc, rules, rounds=ROUNDS, gamma=GAMMA)
    richer = optimise_lineup(lp, lc, rules, rounds=ROUNDS, gamma=GAMMA, budget=rules.budget + 1)
    credit = (richer.objective - base.objective) * args.credit_value
    print(f"one credit is worth {credit:.2f} points (R1 + discounted R2-R3)")

    sim = TurnSimulator(sp, sc, rules, n=args.sims, seed=11, credit_value=credit)
    cands = candidate_pool(sim)
    pairs = candidate_pool(sim, per_pos=10) if args.thorough else None

    starts = [base]
    for k in (3, 4):
        with contextlib.suppress(InfeasibleError):
            starts.append(
                optimise_lineup(lp, lc, rules, rounds=ROUNDS, gamma=GAMMA, min_late_bench=k)
            )

    results = []
    for start in starts:
        ids = [sim.index[p.player_id] for p in start.roster]
        coach = sim.coach_index[start.coach.coach_id]
        print(f"\nsearch from a MILP start (objective {sim.evaluate(ids, coach).objective:.1f})")
        results.append(local_search(sim, ids, coach, cands, pair_candidates=pairs, log=print))

    # Re-score the finalists on fresh, larger simulations.
    final = TurnSimulator(sp, sc, rules, n=args.final_sims, seed=99, credit_value=credit)
    scored = []
    for ids, coach, _ in results:
        f_ids = [final.index[sim.players[i].player_id] for i in ids]
        scored.append((final.evaluate(f_ids, coach).objective, f_ids, coach))
    _, ids, coach = max(scored, key=lambda t: t[0])

    evaluation = final.evaluate(ids, coach, detail=True)
    roster = {final.players[i].player_id for i in ids}
    fixed = optimise_lineup(
        [p for p in lp if p.player_id in roster],
        [c for c in lc if c.coach_id == final.coaches[coach].coach_id],
        rules,
        rounds=[1],
        budget=1e6,
    )
    plan = LineupPlan(
        sim=final,
        ids=ids,
        coach=coach,
        evaluation=evaluation,
        static=base,
        credit_value=credit,
        static_first_round=fixed.first_round,
    )
    report.lineup_plan(plan, title="EuroLeague Fantasy 2026-27, Round 1")
    summary = {
        "roster": sorted(roster),
        "coach": final.coaches[coach].name,
        "objective": evaluation.objective,
    }
    (DATA / "round1_plan.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
