# EuroLeague Fantasy Optimizer

Projection and optimisation engine for EuroLeague Fantasy. It estimates each
player's **expected PIR** for the coming rounds and then solves, under your
actual credit budget, either

- **standard round** — the best *k* transfers (4 by default), or
- **reset round** — the best complete squad from scratch.

The point is not to rank players by last week's PIR. It is to work out where the
market and the crowd are wrong: a 3.5-credit centre whose starter just went down,
against a favourable opponent, in a game his team is expected to win but not run
away with.

```bash
pip install -e .
elfantasy demo          # full pipeline on synthetic data, no network, no keys
```

---

## What it actually models

Every arrow below is a real term in the code, not a wish list.

```
                    ┌──────────────────────────────────────────────┐
   EuroLeague feeds │ schedule · results · box scores · rosters     │
   injury sources   │ manual YAML · JSON feed · HTML (opt-in)       │
   bookmakers       │ moneyline · spread · total · player props     │
   fantasy game     │ prices · ownership · your squad               │
                    └───────────────────┬──────────────────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
      ┌────────────┐            ┌──────────────┐           ┌──────────────┐
      │  MINUTES   │            │    RATES     │           │   CONTEXT    │
      │ depth chart│            │ per-minute   │           │ home/away    │
      │ absences   │            │ PIR component│           │ rest days    │
      │ blowout    │            │ EB shrinkage │           │ opp vs pos   │
      │ return ramp│            │ usage shift  │           │ pace, total  │
      └─────┬──────┘            └──────┬───────┘           └──────┬───────┘
            └──────────────────────────┼──────────────────────────┘
                                       ▼
                        E[PIR] = P(play) × minutes × rate × context × synergy
                                       │
                          blended (inverse-variance) with
                          market-implied PIR from de-vigged props
                                       ▼
                      value_i = Σ_r γ^r · E[PIR]_{i,r}     (multi-round)
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  MILP (CBC): budget · positions ·    │
                    │  club cap · ≤ k transfers · risk     │
                    └──────────────────────────────────────┘
```

### 1. Minutes come first

Minutes are the highest-leverage input, and they are where the edge is.
[`features/minutes.py`](src/elfantasy/features/minutes.py) builds a depth chart
per club from recency-weighted minutes shrunk toward a rank prior, then:

- **Absence redistribution.** An absent player's minutes are reallocated to
  team-mates, weighted by position overlap, depth-chart rank and remaining
  headroom. A missing starting centre mostly frees *centre* minutes, and the
  backup — not the star guard — absorbs them.
- **Usage transfer** ([`features/rates.py`](src/elfantasy/features/rates.py)).
  The beneficiary does not just play more minutes, he shoots more *per minute*.
  Shots, assists, turnovers and free throws shift; steals and blocks do not.
  Skipping this systematically under-projects exactly the players you want to
  find.
- **Blowout risk.** Garbage time is derived from the spread:
  `garbage = clip(slope · (|spread| − threshold), 0, cap)`. Starters lose those
  minutes, the end of the bench gains them. This is why "cheap player, team wins
  easily" is *not* automatically good — and why a value big on a team favoured
  by 4 is worth more than the same big on a team favoured by 18.
- **Return ramp** for players just back from injury.

### 2. Rates are estimated component-wise

PIR is linear in its parts:

```
PIR = (PTS + REB + AST + STL + BLK + FD) − (missed FG + missed FT + TO + BLKA + PF)
```

so [`projection/pir.py`](src/elfantasy/projection/pir.py) models each component
separately. Two payoffs: components respond differently to context (rebounds
scale with pace and vacated boards; turnovers with usage), and — crucially —
**a market-implied PIR can be assembled from points/rebounds/assists props**,
which books actually offer, instead of requiring a PIR line, which they mostly
do not.

Rates are recency-weighted (EWMA, configurable half-life) and shrunk toward a
positional prior with weight expressed in minutes played, so three good games do
not out-project a proven starter.

### 3. Context multipliers

[`features/context.py`](src/elfantasy/features/context.py) applies venue, rest
days, opponent defence *versus that position*, game pace and the team's implied
scoring environment. Personal home/away splits are shrunk hard toward the league
factor — over ten games a split is mostly noise, and an unshrunk one will
confidently tell you a good player is unplayable on the road.

### 4. Team-mate synergy

[`features/synergy.py`](src/elfantasy/features/synergy.py) estimates
with-or-without-you effects two ways: game-level WOWY, and a ridge regression of
per-minute PIR on team-mate availability that controls for simultaneous
absences. This is the *opposite* sign to usage transfer — a roll man loses when
his pick-and-roll guard sits — and both effects apply at once.

### 5. The market

[`projection/market.py`](src/elfantasy/projection/market.py) implements the
standard bookmaking toolkit:

| Function | What it does |
|---|---|
| `devig(prices, method)` | Multiplicative, additive, power, or **Shin** (default) |
| `two_way_prob` | Fair win probability from a moneyline pair |
| `implied_team_totals` | Splits a total by the spread into expected team scores |
| `spread_to_win_prob` | Normal margin model, σ re-fittable from history |
| `prop_to_mean` | Inverts a prop line into the mean of the statistic |

Prop inversion is exact rather than hand-waved. For counting stats a Poisson is
solved numerically for λ such that `P(X ≥ ⌈line⌉) = p`; half-integer lines mean
no continuity correction is needed. For points and PIR a normal is used, and
`μ = line + σ·z(p)`.

Model and market are then combined by **inverse-variance weighting**, with the
market's variance scaled by how much of the player's PIR its props actually
cover. One assists prop nudges the projection; a full points/rebounds/assists
set dominates it. Rounds the market has not opened yet fall back cleanly to the
Elo/pace ratings in [`features/ratings.py`](src/elfantasy/features/ratings.py).

### 6. The multi-round horizon

This is the "coach B has three easy games so I will not have to transfer him out
again" requirement, and it needs no special case:

```
value_i = Σ_r γ^r · E[PIR]_{i,r}
```

with `γ = 0.55` over 3 rounds by default. A player with one great fixture and
then a brutal run scores below a slightly weaker player with three good ones.
Raise `γ` to plan further ahead, drop it to 0 for a pure one-round answer.

### 7. The optimiser

A mixed-integer linear program solved with CBC (bundled with PuLP — nothing to
install).

**Reset round** — [`optimize/squad.py`](src/elfantasy/optimize/squad.py):

```
maximise   Σ x_i (V_i − λ·Var_i) + bank_bonus · leftover
s.t.       Σ x_i · price_i ≤ budget
           Σ x_i = squad_size
           pos_min_p ≤ Σ_{i∈p} x_i ≤ pos_max_p
           Σ_{i∈club c} x_i ≤ max_per_club
           x_i ∈ {0,1}
```

**Standard round** — [`optimize/transfers.py`](src/elfantasy/optimize/transfers.py)
adds buy/sell variables with two details that are easy to get wrong:

- **Selling funds buying.** The constraint is `spend ≤ bank + Σ sell_price_i`,
  not `spend ≤ bank`. Treating it as a fixed budget makes the optimiser far too
  timid.
- **Transfer friction.** Without a small per-transfer penalty the solver burns
  all four transfers to gain 0.3 points, spending flexibility you want next
  round.

Greedy points-per-credit is *provably* wrong here — the budget makes this a
multi-dimensional knapsack, and greedy will not spend down to a premium player
even when the leftover credits cannot be used better. CBC solves a realistic
200-player instance in well under a second.

Risk is handled mean-variance: because `x` is binary, `Σ x_i·Var_i` is the squad
variance under independence, so the objective stays linear. Set `--risk` above 0
to prefer floor over ceiling. (Same-game correlation is not modelled; see
[docs/model.md](docs/model.md).)

---

## Usage

```bash
elfantasy demo                          # synthetic league, no network
elfantasy sync --prices data/prices.csv --injuries examples/injuries.yaml
elfantasy project --round 12 --top 30
elfantasy explain "Player 042"          # every term behind one number
elfantasy squad --budget 100            # reset round
elfantasy transfers -s examples/my_squad.yaml -k 4
elfantasy value                          # best points per credit
elfantasy fixtures                       # schedule difficulty over the horizon
```

`elfantasy explain` is the one to reach for when the output surprises you — it
prints the minutes build-up, every context multiplier, and the model/market
blend that produced the number.

### The transfer ladder

`elfantasy transfers` shows the best plan for 0, 1, 2 … *k* swaps and the
**marginal** gain of each. The fourth transfer is almost always worth a fraction
of the first, and seeing that laid out is usually what changes the decision.

### Getting your data in

The only input the tool cannot derive is **prices**, and availability is the one
that matters most:

```bash
elfantasy price-template -o data/prices.csv   # pre-filled with ids and names
```

Then keep [`examples/injuries.yaml`](examples/injuries.yaml) current. There is no
official EuroLeague injury report, so this file — what you read in a press
conference an hour ago — is genuinely the highest-value data in the pipeline.
`elfantasy` will warn you loudly when it is empty rather than quietly assuming
everyone is fit.

Odds are optional. Set `ODDS_API_KEY` for [The Odds API](https://the-odds-api.com),
or paste lines into a CSV (`--odds-csv`, `--props-csv`) from any book.

---

## Configuration

Two YAML files, both read verbatim — no rule is hard-coded.

- [`config/rules.yaml`](config/rules.yaml) — budget, squad size, position quotas,
  club cap, transfers per round, sell-price rules.
- [`config/model.yaml`](config/model.yaml) — every model hyper-parameter, each
  commented with what it does.

> **Check `rules.yaml` against the current official rules before trusting the
> output.** The defaults (100 credits, 10 players, max 2 per club, 4 transfers)
> reflect a common configuration, but the official game changes between seasons
> and this project is not affiliated with EuroLeague Basketball.

---

## Honest limitations

- **Availability data is the weak link.** No official injury feed exists. The
  manual YAML provider is first in the trust order for that reason.
- **Public feeds are undocumented** and change shape between seasons. The
  adapters in [`data/euroleague.py`](src/elfantasy/data/euroleague.py) are
  written defensively and skip records they cannot parse; if a fetch goes empty,
  that file is what needs fixing, not the model.
- **Priors are seeded, not fitted from a full history.** `POSITION_PRIORS` and
  the dispersion table are reasonable starting values. Re-fit them on your own
  data before treating the absolute numbers as calibrated.
- **Same-game correlation is ignored** in the variance. Stacking two players from
  one game is riskier than the optimiser's variance term implies.
- **Synergy needs data.** In a 34-round season most pair effects will be shrunk
  nearly to zero, which is the correct behaviour but means the term rarely moves
  a projection early in the season.
- **No ownership/differential strategy.** The optimiser maximises expected
  points, not rank in a league where everyone owns the same three players.

## Scraping and terms of service

Only the manual availability provider is enabled by default. The HTML scraper is
a configurable tool, not a set of shipped targets — check a site's terms of
service and `robots.txt` before pointing it anywhere, and prefer official or
licensed feeds. The HTTP client caches on disk, throttles per host, and
identifies itself.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
mypy src
```

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The highest-value
areas are listed in [docs/roadmap.md](docs/roadmap.md); a working injury adapter
and a proper backtest harness are top of the list.

## License

MIT — see [LICENSE](LICENSE).

Not affiliated with, endorsed by, or connected to EuroLeague Basketball. Club
codes are used descriptively. Nothing here is betting advice.
