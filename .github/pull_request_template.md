## What changed

## Why

<!-- For a modelling change: what evidence supports it? A named case where the
projection moves in the right direction beats a general argument. -->

## Checklist

- [ ] `pytest` passes
- [ ] `ruff check src tests` passes
- [ ] New/changed model behaviour has a test that demonstrates it
- [ ] No game rules hard-coded (they belong in `config/rules.yaml`)
- [ ] No scraped data committed; no scraper target added against a site's ToS
- [ ] Comments explain *why*, and existing "why" comments are still accurate
