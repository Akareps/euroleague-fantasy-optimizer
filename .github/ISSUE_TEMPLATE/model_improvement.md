---
name: Model improvement
about: A signal the projection is missing, or one it weights wrongly
labels: model
---

**The effect you think is missing or mis-weighted**

**Why you believe it is real**
A named case where the current projection is wrong is far more persuasive than a
general argument. `elfantasy explain "<player>"` shows every term, which usually
identifies exactly which one is at fault.

**Where it would live**
- [ ] minutes (`features/minutes.py`)
- [ ] rates (`features/rates.py`)
- [ ] context (`features/context.py`)
- [ ] synergy (`features/synergy.py`)
- [ ] market (`projection/market.py`)
- [ ] optimiser (`optimize/`)

**Data it would need, and whether that data is actually obtainable**
