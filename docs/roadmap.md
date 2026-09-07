# Roadmap

Ordered by how much they would improve the actual decisions the tool makes.

## 1. A backtest harness  *(highest value)*

Nothing in this project is validated against history. Without a backtest, every
hyper-parameter in `config/model.yaml` is a guess — a defensible guess, but a
guess.

What is needed:

- Replay past rounds using only data available before tip-off.
- Score calibration (mean projected vs mean actual PIR, by decile) and
  discrimination (rank correlation, top-N hit rate).
- Compare against baselines: last-round PIR, season-average PIR, price rank.
- A `elfantasy backtest` command and a `fit-priors` command that writes the
  fitted values back into `model.yaml`.

The point-in-time constraint is the hard part: injury status and prices must be
as-of the round, not as-of today.

## 2. A real availability adapter

The single biggest source of error. Options, roughly in order of quality:

- A licensed injury feed, wired through `FeedInjuryProvider`.
- Club official channels, parsed per club.
- Pre-game press conference reports.
- Confirmed starting fives, published shortly before tip-off — these turn
  `questionable` into a fact.

Anything scraped must respect the source's terms of service. A community-
maintained YAML in a separate repo would be a legitimate and legally clean
approach.

## 3. Same-game correlation in the variance

Currently the optimiser treats players as independent. They are not: two players
in one fixture share the pace, the blowout risk and the game script. This
matters for anyone using `--risk`, and it makes stacking look safer than it is.

A block-diagonal covariance would keep the mean-variance objective quadratic —
solvable with a MIQP solver, or approximable by penalising same-game pairs with
a linear term.

## 4. Ownership and differential strategy

The optimiser maximises expected points. In a league where everyone owns the
same three players, expected *rank* is the real objective, and that changes the
answer: a lower-EV differential can beat a higher-EV chalk pick.

Needs ownership data (partially available already via `Player.ownership`) and a
rank-based objective.

## 5. Minutes as a distribution

`project_team_minutes` returns a point estimate. A coach who plays someone 8
minutes then 26 is genuinely different from one who plays 17 every night, and
the mean hides that. `rotation_stability` already measures it but only widens
the variance crudely.

## 6. In-season price modelling

Fantasy prices move with performance. Buying a player *before* the rise is worth
real credits over a season, and the current model is blind to it.

## 7. Better feed coverage

- EuroCup support (the adapters already take a `competition` parameter).
- Play-by-play data, which would give true on/off minutes for the synergy model
  instead of the game-level approximation.
- Starting fives, shot locations, defensive matchups.

## 8. Interface

A web UI or a notebook would make the transfer ladder far easier to act on than
a terminal table. Deliberately out of scope until the model is validated —
polish on an unvalidated model is a trap.
