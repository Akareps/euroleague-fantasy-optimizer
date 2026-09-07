# Contributing

Contributions are welcome. This is a hobby project about a game, so the bar is
"does it make the recommendations better", not ceremony.

## Getting started

```bash
git clone https://github.com/Akareps/euroleague-fantasy-optimizer
cd euroleague-fantasy-optimizer
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
elfantasy demo
```

> **Windows note.** CBC (the solver) is launched as a subprocess and its path
> counts against the 260-character `MAX_PATH` limit. If you see
> `WinError 206: The filename or extension is too long`, install somewhere
> shallow such as `C:\dev\`, or enable long paths. `elfantasy` detects this and
> prints the fix.

## What is most useful

See [docs/roadmap.md](docs/roadmap.md). In short: a **backtest harness** and a
**working availability adapter** are worth more than everything else combined.

## Ground rules

**Every modelling change needs a test.** The synthetic league in
`elfantasy/data/sample.py` exists so that model behaviour is testable without a
network or a data licence. If you add a term to the projection, add a test that
demonstrates the behaviour it is supposed to produce — see
`tests/test_minutes.py::test_starting_centre_out_lifts_the_backup` for the
pattern.

**No hard-coded game rules.** Budget, squad size, quotas and transfer counts all
live in `config/rules.yaml` because the official game changes them between
seasons. If you need a new rule, add it to the config.

**Adapters absorb feed weirdness.** Public endpoints change shape without
notice. Parsers in `elfantasy/data/` should skip records they cannot understand
and log, never crash the run. The model layer must never see a feed's field
names.

**Do not commit scraped data**, and do not add a scraper target whose terms of
service forbid it. `HtmlInjuryProvider` is a configurable tool for that reason —
it ships with no targets enabled.

**Explain the why in comments, not the what.** The code is full of small
decisions that look arbitrary and are not (why shrinkage is by minutes rather
than games, why the minutes cap is one-directional). Those comments are the
point; keep them accurate when you change the code around them.

## Style

```bash
ruff check src tests --fix
ruff format src tests
mypy src
pytest
```

Type hints on public functions. Docstrings that say *why*.

## Pull requests

Describe what changed and, for a modelling change, what evidence supports it. A
projection that moves in the right direction on a case you can name is good
evidence; "it felt better" is not.
