# OddsPapi player-prop experiment (v1)

**Experimental; opt-in; unvalidated blend weights.** The existing `elfantasy season`
commands, inputs, projection engine and saved plans remain the baseline. Nothing
is deleted or promoted automatically.

The provider documents EuroLeague player props, but catalogue entries and
`hasOdds` flags do not establish actual coverage. Run the probe to measure
returned, active, complete quotes and the subset we can safely match and use.

## Live coverage

Create a free OddsPapi key yourself and save it as plain text in
`seasons/2026-27/private/oddspapi.key` (Git ignored), or set `ODDSPAPI_API_KEY`.
Keys are passed only to `https://api.oddspapi.io/v4`; error messages and the
experiment's saved files omit credentials. Do not put the key in this YAML.

```powershell
elfantasy experiment oddspapi coverage --round 5
```

The probe spends two discovery requests plus at most ten fixture requests by
default. `--max-fixtures 1` is a smaller smoke test. It stops on an authentication
or quota error; it does not subscribe, buy credits, or retry a quota failure.
`--api-key-file <path>` accepts an existing private key file.

Every run saves a new folder under:

```text
<data-dir>/cache/seasons/<season>/experiments/oddspapi-v1/round_NN/<UTC timestamp>/
```

Re-check an archived response after improving aliases, without spending more requests:

```powershell
elfantasy experiment oddspapi coverage --round 5 --snapshot <snapshot.json>
```

`snapshot.json` preserves the raw basketball prop catalogue, fixtures and boards.
`coverage.json` distinguishes complete pairs, unpaired ladders, suspended prices,
unmatched fixtures/players, ambiguous IDs, expired snapshots and post-deadline
quotes. The priced-roster denominator includes cheap backups and unavailable
players; it is not a claim about coverage of the healthy rotation. Provider
aliases can be extended in `experiment.yaml`; uncertain matches fall back.

## Project and plan

Use the exact `snapshot.json` path printed by the coverage run:

```powershell
elfantasy experiment oddspapi project --round 5 --snapshot <snapshot.json>
elfantasy experiment oddspapi plan --round 5 --snapshot <snapshot.json> --quick
```

Regular round inputs and prior completed-round box scores are still required.
The probe can run before the next round's manual inputs have been prepared.
Omitting `--snapshot` uses the own model for every player. These commands save
`baseline.json`, `projection.json`, `comparison.json`, and (for planning)
`plan.json` in the experiment folder. They never overwrite the normal season
outputs. Plans and projections identify the experiment and its settings.

### v1 method and limits

- Full-game points, rebounds, assists, PR, PA, RA and PRA are supported when
  the catalogue supplies their meanings. Market IDs are discovered, not hardcoded.
- Over/under must be paired within the same bookmaker, player ID, market and
  threshold. Integer over/under lines with pushes are excluded in v1. An explicit
  Yes/No milestone such as `10+` uses the correct `9.5` threshold. **Unpaired
  milestone ladders are inventoried, but not used to invent a fair probability.**
- Remove margin proportionally. Invert points/combos using a fixed normal
  dispersion assumption; rebounds/assists use Poisson inversion. Average alternative
  lines within each book, collapse identical copied boards, then take a median.
  These distribution choices are initial assumptions, not trained uncertainty estimates.
- Our component forecast uses per-minute P/R/A from last season, position priors,
  and only earlier rounds, multiplied by the season model's expected minutes.
  Combined totals constrain that vector. Independent individual markets take
  precedence over overlapping combinations; PRA is never added to P+R+A.
- Hold the own model's remaining PIR components and calibration residual fixed.
  Blend the priced-component change at 35%, capped at 6 PIR before weighting
  (at most 2.1 conditional PIR points). Our play probability and expected win
  bonus are retained. Quotes do not establish that an injured player will play.
- Coaches, price estimates, future rounds and simulation volatility retain the
  baseline forecasts. v1 does not yet fit component correlations, infer shooting
  misses from a changed points line, or recalibrate its blend.
- Only snapshots observed by the round's first tip-off can affect transfer plans.
  `--as-of` requires a timezone and cannot move past that deadline. Last price
  movement is not last observation: an old unchanged price observed active now
  can be fresh. A snapshot expires after 36 hours without a new observation.

## Evaluate without hindsight

```powershell
elfantasy experiment oddspapi evaluate --round 4 --snapshot <pre-round-4-snapshot.json>
elfantasy experiment oddspapi evaluate --round 4
```

Both forecasts are scored on the same set of players, including zeros for
nonparticipants, with RMSE, rank correlation and actual points of the top 40.
The output preserves player-level forecast and minutes errors. No snapshot
means both scores must be identical. A snapshot fetched after Round 4 is rejected
for that replay even if its prices last changed before tip-off. Historical
OddsPapi payloads have a different schema and are not yet an accepted source;
use contemporaneously archived snapshots until a verified historical adapter exists.

The assessment replays today's season inputs using earlier rounds for calibration;
it is not an immutable record of the manager's actual team or Turn decisions.
Compare saved pre-round forecasts separately, and respect the season's `tuned_on`
flags. Freeze weights before collecting forward results and review performance
over multiple rounds before considering adoption.

## Live audit: 12 October 2026, Round 5

Verified against the live API, not just the provider's catalogue:

| Check | Result |
|---|---|
| Scheduled fixtures matched by both teams and start time | 10 / 10 |
| Games with usable player props at this observation | 7 / 10 |
| Matched, priced players with usable quotes | 84 / 326 (25.8%) |
| Complete over/under quote pairs before player matching | 7,560 |
| Quote pairs passing roster and time checks | 7,409 |
| Bookmaker feeds returning complete pairs | 48 (not 48 independent forecasts) |
| Markets observed | Points, rebounds, assists, PR, PA, RA, PRA |

The three Wednesday games had no usable props at the observation time:
Hapoel–Besiktas, Efes–Bayern, and Paris–Virtus. These keep own-model projections.
The counts can change as books publish more markets. Explicit club and player
aliases now handle observed spellings such as Baskonia Vitoria-Gasteiz and Edy
Tavares; unknown players are still excluded.

A live-data integration check adjusted all 84 matched player forecasts against
the saved Round 4 model's forward forecasts for Round 5. It verified that the
baseline, other players, coaches and later-round forecasts remained unchanged.
This is a technical integration preview, **not an up-to-date Round 5 plan**:
the fresh Round 5 manual inputs were not present, and the archived own PIR
forecasts were prepared before Round 4. The preview and its provenance are saved
separately in the data cache. There is no claim of improved predictive accuracy yet.

Provider contracts: [v4 odds](https://oddspapi.io/en/docs/get-odds),
[market catalogue](https://oddspapi.io/en/docs/get-markets),
[EuroLeague example](https://oddspapi.io/blog/euroleague-basketball-odds-api/).
