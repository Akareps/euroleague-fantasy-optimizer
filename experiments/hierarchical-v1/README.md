# Independent hierarchical model experiment (v1)

**Opt-in, retrospective validation only, not promoted to the baseline.**
The production season model and the OddsPapi experiment remain separate. This
third model never uses player-prop quotes. Shared inputs are fantasy prices,
schedule, researched availability, game winner lines and coach forecasts.

## Run it

The fitting code uses the project's existing NumPy/SciPy dependencies. Importing
the bulk historical archive additionally needs the optional DuckDB dependency:

```powershell
pip install -e ".[history]"
elfantasy experiment hierarchical fetch --round 5
elfantasy experiment hierarchical evaluate --through 4
elfantasy experiment hierarchical project --round 5
elfantasy experiment hierarchical plan --round 5 --quick --confirmed-prices-only
elfantasy experiment hierarchical compare --round 5 --quick --snapshot <saved-oddspapi-snapshot.json>
```

`fetch` is the only command that contacts the network. It imports a pinned public
archive, verifies its schedules and sampled boxes against the official feed,
and fetches missing **earlier** current-season shooting splits. `project`, `plan`
and `evaluate` run from saved data. `compare` uses the supplied immutable odds
snapshot and saves all three models' projections, plans and common-seed scores.

Three-model comparison defaults to the current squad and candidates with
app-confirmed prices in the private round file. This holds the purchase pool
constant across models and avoids recommendations whose budget rests on an
estimated price. `--allow-estimated-prices` expands that comparison to the full
priced roster. Standalone `plan` is unrestricted unless
`--confirmed-prices-only` is supplied. Full searches use 6,000 simulations and
40,000 fresh finalist simulations; `--quick` uses 2,000 and 10,000. Comparison
then rescores every recommended roster under every model using the same three
seeds, with 40,000 simulations per seed.

History is saved under `data/cache/experiments/hierarchical-v1/history/`.
Every run gets a new directory under
`data/cache/seasons/<season>/experiments/hierarchical-v1/round_NN/<UTC timestamp>/`.
Existing output files in both other workflows are never overwritten.

## Statistical model

This is an **analytic empirical-Bayes approximation**, not a PyMC/MCMC model.

1. **Selection and minutes.** A regularised Bernoulli model predicts rotation
   participation. A Gaussian population regression with partially pooled player
   effects predicts positive minutes, bounded to a 40-minute game. Features use
   earlier recent/last/career minutes, role changes, evidence size, venue, rest,
   expected margin, past team opportunity pace and club changes. Participation
   and positive-minute uncertainty are combined. Current healthy unlisted
   players contribute zero-minute role evidence; recorded injury absences do not.
2. **Opportunities and counts.** Position-level Gamma priors are fitted from
   historical between-player variation. Each player's two/three-point attempts,
   free-throw attempts and eight other PIR counts update those priors. Older
   role evidence is capped; current-club games receive more influence through
   exponential recency weighting.
3. **Shooting efficiency.** Separate Beta-Binomial estimates for two-pointers,
   threes and free throws produce makes, points and misses from the *same*
   attempt forecasts. Increased scoring opportunities therefore also increase
   expected missed shots; fantasy scoring is recomposed from all PIR components.
4. **Context.** Regularised Poisson regressions fit venue, rest, prior team
   opportunity pace, opponent component rates and individual role changes.
   Historical expected margin comes from past-only team ratings, rather than
   the actual target-game margin. Live projections retain the shared game lines.
5. **Uncertainty.** Predictive moments combine minute variation, rate/efficiency
   posterior uncertainty, fitted NB2 overdispersion and shrunk component
   covariance. Normal moment approximations give score intervals. The existing
   Turn optimizer receives the learned mean and SD; it does not directly sample
   the complete joint shooting/count posterior.

Availability remains researched input, not a learned medical diagnosis. Minutes
redistribution following known absences and the 200-minute team ceiling are
structural adjustments. This version does not estimate causal teammate absence
effects, coach-specific rotations, domestic-game workloads or return restrictions.
New players without historical records retain researched role/minute priors and
use position-pool component rates. Missing or ambiguous official identities keep
their baseline forecast. The fitted parameters and uncertainty assumptions are
saved in `model.json`, alongside input/source fingerprints and the data cutoff.

## Data and leakage controls

The pinned [euroleague-open-data archive](https://github.com/McNews11/euroleague-open-data)
contains raw official-feed data. This experiment uses only EuroLeague regular
seasons **2024–25 and 2025–26**, not EuroCup, NBA, derived fantasy tables or roster
overrides. Commit `2c44083749cf6c0b545a181c5e3321d4c52ca368` and the archive's SHA-256
are fixed. DuckDB opens read-only with external access disabled, and only raw
base tables are queried.

The import reconciles unique player-game IDs, shot totals, reported PIR, team
minute totals, official schedules and first/middle/last sampled boxes in each
season. It imported **686 games / 16,316 box-score rows** with no excluded games.
Game-sheet zero-minute players are retained. Overtime minutes are normalised to
regulation for minute-model training; count-rate exposure uses actual minutes.

The forecast cutoff filters training data before population pools, regression,
player effects, covariance or dispersion are fitted. Target-game statistics are
never used as forecast features. Historical round groups are processed in
first-tip time order, including games played out of numerical round order.
Before each forecast, state updates use only prior calendar days, which also
avoids accidentally treating an earlier same-day tip as a completed game.

## Validation and adoption

- **Historical evaluation:** initialise coefficients/pools on 2024–25, then
  forecast every 2025–26 round and update player/team evidence chronologically.
  Compare with a simple lagged-minute × shrunk-PIR-rate forecast. The evaluation
  universe is the actual game-day box sheets, including zero-minute players.
  It therefore cannot establish accuracy for pre-deadline squad selection or
  injuries. Its comparator is not the production season model, whose historical
  fantasy price and injury snapshots are unavailable.
- **Current-season evaluation:** fit global coefficients on the two completed
  seasons, then replay rounds 1–4 with only earlier current-season evidence.
  Compare against the real season model on the same uniquely identified priced
  players with P(plays) ≥ 0.5, including zeros for actual nonparticipants.
  Baseline rounds 2/3 were tuning data. Saved manual inputs and the new design
  have been inspected during development, so these checks are retrospective,
  not an untouched prospective test.
- Report score/minute RMSE, rank correlation, per-round top-40 actual score,
  bias and 10th–90th interval coverage. Pooled top-40 and mean per-round top-40
  are stored separately. No eligible old OddsPapi snapshots exist, so that
  experiment cannot be fairly scored on those earlier rounds.

The first evaluation is mixed: improved current-season minutes and top picks,
slightly worse overall score error, and no win over the simple historical
benchmark. Keep this model experimental. Freeze settings and record upcoming
pre-deadline projections before considering adoption.

The optimizer still values discounted future rounds and estimated price changes.
Its current horizon includes Round 7 despite the reset after Round 6. Small
objective differences should not outweigh price uncertainty or model error.

Method references: [hierarchical partial pooling](https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html),
[negative-binomial count modelling](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-negative-binomial-regression.html),
[chronological validation](https://otexts.com/fpp3/tscv.html).
