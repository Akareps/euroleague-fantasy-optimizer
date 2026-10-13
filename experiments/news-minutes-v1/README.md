# News and minutes experiment (v1)

An opt-in layer on **hierarchical-v1**, with a separately archived source audit,
projection and transfer plan. Season baseline, OddsPapi and hierarchical settings
and outputs are preserved. No player props enter this experiment.

## Run

```powershell
elfantasy experiment news collect --round 5
elfantasy experiment news compare --round 5 --snapshot <saved-news-snapshot.json> --quick
elfantasy experiment news project --round 5 --snapshot <saved-news-snapshot.json>
# After the round has finished and the normal season data is refreshed:
elfantasy experiment news evaluate --run <frozen-comparison-directory>
```

Collection defaults to the reviewed excerpts in `research/r05.yaml`. These are
real public-web findings with manually reviewed player attribution, translations,
publication dates and observation timestamps. This is **assisted research, not a
comprehensive autonomous news or social-media feed**. Discovery is bounded to the
held players and five candidates with app-confirmed prices. Missing coverage does
not establish that a player is healthy or guarantee a normal rotation.

`collect --manifest <path> --fetch-pages` optionally archives the nominated public
HTML pages, after checking robots.txt, with throttling and no authentication or
access-control bypass. Failed/disallowed fetches retain separately labelled manual
excerpts and do not invent raw-page provenance. Page snippets are matched literally
and named sentences become **unreviewed candidates**, never automatic overrides.
All retrieved HTML stays in the ignored cache. Local caption files can be imported
using a source's `transcript_path` plus `format: srt` or `format: vtt`; candidate
quotes preserve video offsets. This version does not download/transcribe video
audio. Speaker attribution and translations require review.

X URLs can be recorded as sources, but this version does not implement authenticated
X API searches. The first collection identified an official interview through a
published embed; direct X content was unavailable and was not treated as verified.
An owner/teammate/fan comment has lower authority than a specific club/coach report.
No paid API/LLM service is required; the extraction and sentiment diagnostics are
transparent rules, not a trained or generative classifier.

## Evidence and forecast effects

Each claim records its unique ID, originating event, player **and club**, fixture,
speaker, original excerpt, reviewed translation, publication/observation timestamps,
confidence, time scope, verification and whether shared inputs already used it.
Day-only publication clocks are represented at midnight for conservative freshness;
they never replace the actual observation time for historical eligibility.

The English/Spanish rule extractor recognises explicit exclusions, readiness,
uncertain availability, minute caps/estimates, limited conditioning and explicit
future rotation changes. Other languages need reviewed English translations.
Uncertain, negated, past-game or generic statements abstain. Every numerical effect
requires a reviewed, verified excerpt concerning the named upcoming fixture.

- Source authority × reviewed confidence × age decay controls eligibility/weight.
  Default maximum age is 96 hours and half-life 72 hours. These are frozen design
  choices, **not empirically calibrated source reliabilities**.
- A reliable explicit restriction supplies a hard conditional minutes ceiling.
  An explicit numeric estimate updates the prior through a Gaussian precision
  approximation. Weak directional role statements are capped at a two-minute
  change before weighting; no invented numeric ceiling from vague praise.
- Availability and conditional minutes are updated separately. Existing researched
  availability requires an explicit reviewed supersession; already incorporated
  evidence is skipped. Generic nonnumeric restrictions widen minute uncertainty.
- Reposts/translations of one interview form one event. The strongest eligible
  statement is used; independent stories do not automatically multiply confidence.
  Unresolved opposite statements cause abstention rather than silent cherry-picking.
- Half of freed expected minutes is redistributed conservatively by position and
  headroom, respecting restrictions and a 200-minute expected team ceiling. Unpriced
  bench players can absorb the remainder. Added minutes come out of other roles.
- Changed minutes are passed back through the fitted PIR component model; shooting,
  missed attempts, rebounds and variance are recomputed consistently. Prices,
  coaches and future-round forecasts remain shared. Claims expire after the fixture.

Lexical English sentiment (including simple negation) is saved as a diagnostic.
**It does not change minutes or PIR.** Sarcasm, coach gamesmanship, translation
errors, tactical context and selection decisions remain material limitations.

## First Round 5 collection and validation

The first collection covers 15 held/app-priced players, nine source documents and
ten reviewed/pending claims. Tubelis and Williams-Goss injury/travel statements and
Saint-Supery's availability context were already included in shared inputs. The
coach's broader-rotation comments lack named player attribution; local performance
recaps, old player confidence quotes and fan opinions do not establish a new
upcoming minutes commitment. **No new actionable minutes statement was found.**
The correct first result is identical numerical forecasts, not an artificial
sentiment boost. Identical forecasts reuse the same transfer search result and
are checked with identical Monte Carlo seeds.

All runs have unique timestamped directories under
`data/cache/seasons/<season>/experiments/news-minutes-v1/round_NN/`.
Capture **and** individual observation times must precede the projection cutoff,
which cannot exceed the first tip. Source/settings/code/model inputs and frozen
projections have SHA-256 fingerprints. A newly scraped old article cannot enter
a past forecast merely because its publication date is old.
Fresh forecast generation after transfer lock is refused even with `--as-of`:
shared manual season inputs do not have a complete historical timestamp archive.
Historical evaluation therefore uses the already frozen projections only.

There is no historical text archive collected before earlier deadlines, so no
honest earlier-round sentiment accuracy claim is available. `evaluate` scores the
original frozen parent/text forecasts after all target games complete, including
zero-minute players, and refuses changed projections. It reports expected-minutes
RMSE/MAE and fantasy-score RMSE on the same unique priced IDs. A single round is
insufficient to calibrate effects or decide adoption. Keep recording future
deadline snapshots and compare the same parent with and without evidence; do not
tune v1 settings on its eventual evaluation results.

Source formats and all ten first-round annotations are reviewable in
[research/r05.yaml](research/r05.yaml).
