"""The round-by-round workflow used through the 2026-27 season.

    elfantasy season fetch    --round 5   # games and box scores so far
    elfantasy season project  --round 5   # projections for Rounds 5-7
    elfantasy season plan     --round 5   # transfers, lineup and Turn plan
    elfantasy season compare  --round 5   # named plans head to head
    elfantasy season backtest             # every past round, chronologically

Hand-researched inputs (injuries, betting lines, eye-test notes) live as YAML
under ``seasons/<season>/`` and are committed; your squad, bank and the prices
the app shows go in ``seasons/<season>/private/`` (git-ignored). Downloaded data
lives in the data directory, also git-ignored.

Modules: :mod:`.inputs` (the YAML), :mod:`.data` (downloads), :mod:`.model`
(projections), :mod:`.planner` (transfers and lineups), :mod:`.backtest`.
"""
