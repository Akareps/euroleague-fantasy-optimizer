# Round 1, 2026-27: a worked example

How the Round 1 team was built, reproducible end to end. Round 1 is the
awkward case: no games have been played this season, so projections have to
come from last season's totals, the game's own prices, betting lines and
preseason reporting.

```bash
cd scripts/round1_2026
python fetch.py      # 3 requests to the official feeds (throttled, cached)
python build.py      # join prices, 2026-27 rosters and 2025-26 season totals
python project.py    # Rounds 1-3 projections -> projections.json
python run.py        # lineup MILP + Turn simulation -> the plan
```

Data goes to `data/cache/r1` (git-ignored), or `$R1_DATA`.

## Inputs

| File | Source | In git? |
|---|---|---|
| `inputs.py` | Hand-curated from the sources below, each line annotated | yes |
| `prices_players.csv` | Opening prices, [Basketball Sphere's published list](https://basketballsphere.com/en/euroleague-fantasy-player-prices/) (columns `rank,player,club,pos,price`, coaches as `HC`) | no: third-party data, supply your own |
| `games_E2026.json`, `people_E2026.json` | Official feed: 2026-27 schedule and rosters | no, `fetch.py` |
| `stats_E2025_acc.json` | Official feed: 2025-26 season totals, *Accumulated* mode | no, `fetch.py` |

`inputs.py` holds everything that changes by the hour before tip-off:

- **Bookmaker lines:** moneylines for all ten Round 1 games and outright
  winner odds (OddsPortal averages).
- **Availability:** the official Round 1 injury report, [RotoWire](https://www.rotowire.com/euro/news.php?view=injuries)
  and the Basketball Sphere injury tracker. The official report wins when they
  disagree.
- **Preseason role evidence:** the official Round 1 tips and Basketball Sphere
  previews, applied as moderate multipliers or stated minutes.

## How the projections work

`project.py` is a preseason model, separate from the in-season engine:

1. **Rates** come from 2025-26 per-minute PIR, shrunk toward a positional prior.
2. **Price-implied priors.** The game's price is the operator's own forecast,
   and the only calibrated signal for players new to the league. Returning
   players lean on their history, and players who changed club lean on the
   price.
3. **Minutes.** Each team gets a 200-minute budget with a rotation structure;
   when a roster over-claims, the back of the rotation absorbs it. Absences
   are redistributed first by moving backups up the rotation, then by
   position. Stated minutes estimates are locked.
4. **Game context** comes from Shin-de-vigged moneylines: an opponent/venue
   adjustment, garbage time, the win bonus and coach points.
5. **Rounds 2-3** use team ratings fitted to the Round 1 lines plus the
   outright odds.

`run.py` then uses the package's `optimise_lineup` and `TurnSimulator`
(`elfantasy.optimize`). There is no separate copy of the optimisation logic
here.

## The result

Two rosters came out within noise of each other. Here they are on identical
simulations (`run.py` scores its own finalists the same way):

| | Credit valued at ×1 | Credit valued at ×3 | Round 1 alone |
|---|---|---|---|
| Recommended: Francisco, Williams-Goss, **Vezenkov (C)**, Parra, Dokossi, Spagnolo · bench **Wright**, Saint-Supery, Mantzoukas, Joksimovic · coach Jasikevičius | 208.7 | **215.1** | 142.5 |
| Alternative: Robinson/Lundberg family, with Zižić and Nuñez | 208.6 | 213.8 | 143.1 |

The recommended roster was preferred for three reasons. It keeps 0.9 credits
in the bank. It carries one fewer 4.x-credit player whose projection rests on
a single minutes report (4.x players lose value when they do not play; 4.0
players cannot). And it wins once credits are valued as persisting through
the season. `python run.py --credit-value 3` reproduces the sensitivity.

Played with Turn moves, the recommended roster is worth about **+8 points**
over the same players in their best fixed lineup (134.3 → 142.5):

- **Before Thursday:** Wright and Saint-Supery (Friday games) start on the
  bench, and Vezenkov captains.
- **After Thursday:** Wright comes on for the lowest-scoring Thursday field
  player (in 100% of simulations). Saint-Supery comes on for the next-lowest
  if that player scored under ~7 (in about 65%).
- **Captaincy:** if Vezenkov scored under ~17, it moves to Wright (in about
  37%).

## Caveats

- **Not backtested.** The price priors, the role multipliers and the
  preseason minutes are judgement calls with named sources, not fitted
  parameters.
- **The price rule is a rule of thumb.** ±0.1 per 2 points against the
  price, floored at 4.0, is the rule experienced managers observe; the
  official formula is undisclosed.
- **The feed rate-limits.** Full per-game box scores for 2025-26 were
  abandoned after 66 games under 5-minute Retry-After windows. Those 66 games
  fitted the team-PIR/margin slope (93.5 + 1.09 × margin) and the PIR
  volatility (CV 0.634), which are hard-coded as fallbacks.
