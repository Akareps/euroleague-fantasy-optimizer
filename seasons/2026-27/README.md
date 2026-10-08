# EuroLeague Fantasy 2026-27: the round-by-round workflow

Everything used to prepare each round of the season, re-runnable from these
files plus downloads.

| Path | What | In git? |
|---|---|---|
| `season.yaml` | Club names, outright odds, preseason roles, eye-test notes, model settings | yes |
| `rounds/rNN.yaml` | Round N: availability for N and the next two rounds, betting lines, and (after the round) `injured_out` | yes |
| `private/rNN.yaml` | Your squad, coach, bank, the app's prices, named plans to compare | **no** (see `private.example.yaml`) |
| `<data dir>/cache/seasons/2026-27/` | Schedule, box scores, rosters, price lists, outputs | no |

## Each round

```bash
elfantasy season fetch                 # schedule + box scores of finished rounds
```

1. **Copy last round's file** to `rounds/rNN.yaml` and update it: injury news
   (the club's own report first, then Basketball Sphere's tracker, RotoWire) as
   P(plays) for this round and the next two; moneylines for every game you can
   find (OddsPortal averages), this round and the next.
2. **Write `private/rNN.yaml`** from the app: bank, squad and coach, and the
   prices it shows. The model estimates every other price from the last
   published list and its fitted price rules, but the app is the truth.
3. Run:

```bash
elfantasy season project --round N     # projections for N, N+1, N+2
elfantasy season plan --round N        # transfers, lineup, Turn plan (a few minutes)
elfantasy season compare --round N     # `plans:` from the private file, head to head
```

4. **After the round**, add `injured_out` to `rounds/rNN.yaml`: who missed it
   injured or for personal reasons. Their absence is then not read as a coach's
   decision. (Without it, anyone given under 90% beforehand is assumed injured.)

Transfers are provisional until the round's first tip-off, so re-running close
to the deadline is free.

## Checking the model

```bash
elfantasy season backtest              # every finished round, chronologically
```

Each round is re-projected with today's model and only what was known before
it; calibration and the price baseline are fitted on earlier rounds only.
Rounds listed under `model.tuned_on` in `season.yaml` were used to choose the
evidence weights, so their scores flatter the model; later rounds are fully
out of sample.

## Starting from scratch

The prior needs the opening price list, which is third-party data and not in
git: save it as `<data dir>/cache/seasons/2026-27/prior/prices_round_01.csv`
(`rank,player,club,pos,price`, coaches with pos `HC`), then
`elfantasy season fetch --prior`. Later published lists go in `prices/round_NN.csv`.
