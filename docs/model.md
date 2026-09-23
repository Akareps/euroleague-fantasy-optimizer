# The model, in detail

This document is the reference for what the engine computes and why. The
README is the tour; this is the specification.

---

## 1. Notation

For player *i*, round *r*:

| Symbol | Meaning |
|---|---|
| $\pi_i$ | probability player *i* takes the floor |
| $m_i$ | expected minutes **conditional on playing** |
| $\rho_{i,c}$ | per-minute rate of PIR component *c* |
| $M$ | product of context multipliers |
| $V_i$ | discounted multi-round value used by the optimiser |

The headline quantity is

$$\mathbb{E}[\text{PIR}_{i,r}] = \pi_i \cdot M \cdot \sum_c s_c \, \rho_{i,c} \, m_i$$

where $s_c \in \{+1, -1\}$ is the sign of component *c* in the PIR formula.

Keeping $\pi_i$ **outside** the product rather than folding it into the minutes
matters for the variance: "70% chance of 25 minutes" and "certain to play 17"
have the same mean and very different risk.

---

## 2. Minutes

### 2.1 Baseline

For each player, minutes are an exponentially weighted average of games played,
shrunk toward a depth-chart rank prior:

$$m_i^{(0)} = \frac{n_i \bar{m}_i + \kappa\, p_{\text{rank}(i)}}{n_i + \kappa}$$

with weights $w_t = 2^{-\text{age}_t / h}$, $h$ = `minutes.half_life_games`,
$\kappa$ = `minutes.prior_games`, and $p$ = `minutes.rank_prior`.

The depth chart is then **re-ranked on the shrunk values**, so it is internally
consistent — a player whose two-game cameo would otherwise rank him above a
regular starter does not.

### 2.2 Absence redistribution

An absent player *j* vacates $m_j^{(0)} \cdot (1 - \pi_j) \cdot \alpha$ minutes,
where $\alpha$ = `absence_redistribution.reabsorption` ≈ 0.92. The rest evaporates
into a shorter rotation — coaches tighten up rather than distributing every
minute evenly.

Those minutes are shared among available team-mates with weight

$$w_k \propto \underbrace{\phi(\text{pos}_k, \text{pos}_j)}_{\text{position overlap}} \cdot \underbrace{\text{rank}_k^{\beta}}_{\text{depth}} \cdot \underbrace{(m_{\max} - m_k^{(0)})}_{\text{headroom}} \cdot \underbrace{f(m_k^{(0)})}_{\text{floor damping}}$$

- **Position overlap** $\phi$ is 1.0 for the same position, 0.35 across
  positions. A missing centre mostly frees centre minutes.
- **Depth** $\text{rank}^{\beta}$ with $\beta = 0.6$ favours reserves. Starters
  are already near their ceiling; the backup is where the minutes go.
- **Headroom** prevents a 34-minute player absorbing another eight.
- **Floor damping** stops the twelfth man, who has never played, from being
  treated as the natural beneficiary.

Weights are normalised, so the redistribution conserves the vacated minutes
exactly.

### 2.3 Blowout adjustment

Garbage time from the pre-game spread:

$$g = \text{clip}\big(\text{slope} \cdot (|\text{spread}| - \tau),\; 0,\; g_{\max}\big)$$

Five players' worth of decided minutes ($5g$) is reallocated: a share
`blowout.starter_loss_share` comes off rotation ranks 1–5, and lands on the deep
bench weighted by $\text{rank}^{0.5}$.

This is why "cheap player on a team that will win easily" is **not**
automatically good. A value big on a team favoured by 4 is worth materially more
than the same big on a team favoured by 18, because in the second case he is
also competing with the coach's decision to empty the bench.

### 2.4 Cap

Finally, if the expected total exceeds the 200 minutes a team can actually
distribute, the excess comes back **mostly from the back of the rotation**.
Player $k$ at rotation rank $r_k$ gives up a share proportional to
$m_k \pi_k (0.25 + r_k/6)$. Coaches shorten the bench; they do not trim their
star. A uniform scale-down shaved ~15% off every starter on a deep roster,
putting Round 1's Vezenkov at 20 minutes against 27.8 last season.

The correction is **one-directional**. Under-allocation is never scaled up: it
means either an incomplete roster in the data or a genuinely short bench, and
scaling up would both invent minutes and double-count §2.2.

### 2.5 Stated minutes

A stated estimate (`Player.minutes_override`, e.g. "~13 minutes" from
preseason reporting) replaces the baseline and is then left alone: no
redistribution gains, no blowout shift, no trimming. Without the lock, a backup
reported at 13 minutes but ranked eleventh on a deep roster was cut to near
zero by the steps above.

---

## 3. Rates

Per-minute rates for each PIR component, recency-weighted and shrunk toward a
positional prior with weight expressed in *minutes played* (not games — a player
with six 30-minute games has far more evidence behind him than one with six
6-minute cameos):

$$\rho_{i,c} = \frac{m^{\text{eff}}_i \hat\rho_{i,c} + \kappa_\rho\, \bar\rho_{\text{pos}(i),c}}{m^{\text{eff}}_i + \kappa_\rho}$$

`POSITION_PRIORS` are calibrated so a full 200-minute team-game reproduces
typical EuroLeague totals: ~82 points, ~31 rebounds, ~17 assists, ~91 PIR.

### 3.1 Usage transfer

When a team-mate is absent, survivors do not merely play more minutes — they
shoot and create more *per minute*. The beneficiary's rates gain

$$\Delta\rho_{i,c} = \sum_j \rho_{j,c} \cdot (1-\pi_j) \cdot u_c \cdot w_i \cdot \theta$$

where $u_c$ weights how transferable component *c* is (points, missed FG and
missed FT fully; turnovers 0.8; assists 0.7; rebounds 0.45; fouls drawn 0.6),
$w_i$ is the redistribution share from §2.2, and $\theta$ =
`absence_redistribution.usage_transfer` ≈ 0.45.

Steals, blocks and fouls committed are left alone — they are close to per-minute
constants and do not scale with role.

**Omitting this step systematically under-projects the beneficiary of an
injury**, which is precisely the player this project exists to find.

---

## 4. Context

All multiplicative, all reported individually by `elfantasy explain`.

| Factor | Source |
|---|---|
| Venue | Player's own home/away split, shrunk hard toward the league factor |
| Rest | Days since previous fixture, table-driven |
| Opponent | Opponent's PIR conceded to that position, shrunk toward neutral |
| Pace | Game total relative to league average |
| Team total | Team's implied score, from spread and total |

Home/away splits deserve a note: over ten games a personal split is mostly
noise. `context.home_split_prior_games` = 25 means a player needs roughly 25
games at home before his own split outweighs the league default, and the result
is clipped to [0.85, 1.20]. An unshrunk split will confidently tell you a good
player is unplayable on the road.

---

## 5. Team-mate synergy

Two estimators, both heavily shrunk:

- **WOWY** (`pair_effects`) compares a player's per-minute PIR in games a
  team-mate played versus games they missed, shrunk by the *smaller* of the two
  samples — the without-sample is almost always binding.
- **Ridge** (`ridge_pair_effects`) regresses per-minute PIR on team-mate
  availability indicators. Controls for simultaneous absences; ridge penalty
  plays the role of the prior because the columns are highly collinear.

The effect enters as $1 - \sum_j e_{ij}(1-\pi_j)$, clipped to
±`synergy.max_abs_effect`.

Note the sign is **opposite** to usage transfer, and both apply at once: losing
a creator frees usage for the roll man but also removes the passes that made his
finishes easy. Which dominates is an empirical question the data answers per
pair.

In a 34-round season most pair effects shrink nearly to zero. That is correct
behaviour, not a bug — but it does mean the term rarely moves a projection early
in the season.

---

## 6. Market integration

### 6.1 De-vigging

Posted prices imply probabilities that sum to more than one. Four estimators are
implemented; **Shin** is the default because it models the overround as arising
from insider trading and therefore removes proportionally more vig from
longshots, which matches the observed favourite-longshot bias.

For a two-way market with raw implied probabilities $q_i$ summing to $S$, Shin
solves for the insider fraction $z$ in

$$p_i = \frac{\sqrt{z^2 + 4(1-z)q_i^2/S} - z}{2(1-z)}, \qquad \sum_i p_i = 1$$

### 6.2 Prop inversion

Given a de-vigged $P(X > \text{line}) = p$:

- **Counting stats** (rebounds, assists, steals, blocks, turnovers): solve
  numerically for $\lambda$ with $P(X \ge \lceil \text{line} \rceil) = p$ under
  a Poisson. Half-integer lines mean the discreteness is handled exactly — no
  continuity correction.
- **Points and PIR**: $\mu = \text{line} + \sigma z(p)$ under a normal, with
  $\sigma$ scaled as $\sqrt{\text{minutes}}$.

### 6.3 Composing PIR from parts

Books rarely price PIR directly. Because PIR is **linear** in its components, a
market-implied PIR is assembled from whatever props exist: priced components are
replaced, and unpriced components are scaled by the ratio of market to model on
the priced ones (clipped to [0.6, 1.6] so one wild prop cannot dominate).

### 6.4 Blending

Model and market are combined by inverse-variance weighting:

$$\mu = \frac{\mu_{\text{mod}}/\sigma^2_{\text{mod}} + \mu_{\text{mkt}}/\sigma^2_{\text{mkt}}}{1/\sigma^2_{\text{mod}} + 1/\sigma^2_{\text{mkt}}}$$

with the market's variance inflated in inverse proportion to **coverage** — what
fraction of the player's projected PIR its props actually price. One assists
prop nudges the projection; a full points/rebounds/assists set dominates it.

The blend's variance, $1/(1/\sigma^2_{\text{mod}} + 1/\sigma^2_{\text{mkt}})$,
describes how precisely we know the player's **mean**. It is *not* how much a
single game swings, and must not be used as the outcome variance: doing so
made every player with props look steadier than players without. The outcome
variance is the stat-line dispersion of §7, rescaled to the blended mean.

Rounds the market has not opened yet fall back to the Elo/pace ratings, which is
the whole reason those exist.

---

## 7. Variance

Per-component over-dispersed Poisson, $\text{Var} = \phi_c \mu_c$, summed (signs
square away), times a correlation inflation factor of 1.22 — a high-usage night
lifts points, missed shots and turnovers together.

Then the availability decomposition:

$$\text{Var}(\text{PIR}) = \underbrace{\pi \sigma^2}_{\text{playing}} + \underbrace{\pi(1-\pi)\mu^2}_{\text{might not play}}$$

Finally widened for players whose game-to-game output is erratic, measured by
the coefficient of variation of their PIR history.

**Same-game correlation** is modelled in the Turn simulator (§11), which draws
one margin per fixture and moves every player in it together. The squad-only
optimisers of §9 still assume independence.

---

## 8. The horizon

$$V_i = \sum_{r} \gamma^{r} \,\mathbb{E}[\text{PIR}_{i,r}], \qquad \gamma = 0.55,\; 3 \text{ rounds}$$

This single expression encodes the multi-round argument: a player with one great
fixture followed by a brutal run scores below a slightly weaker player with three
good ones. No special case is needed for "he will not have to be transferred out
next round" — it falls out of the discounting.

Variance discounts quadratically, $\sum_r \gamma^{2r}\text{Var}_{i,r}$, since the
rounds are treated as independent.

Set $\gamma = 0$ for a pure single-round answer; raise it to plan further ahead.

---

## 9. Optimisation

### Reset round

$$\max \sum_i x_i (V_i - \lambda \text{Var}_i) + \beta\Big(B - \sum_i x_i c_i\Big)$$

subject to budget, squad size, position quotas, club cap, $x_i \in \{0,1\}$.

Because $x$ is binary, $\sum_i x_i \text{Var}_i$ *is* the squad variance under
independence — so the mean-variance objective stays linear and the problem
remains a MILP.

### Standard round

Adds $s_i$ (sell) and $b_i$ (buy) with $x_i = 1 - s_i$ for owned players and
$x_i = b_i$ otherwise, plus:

$$\sum_i b_i \le k, \qquad \sum_i b_i = \sum_i s_i, \qquad \sum_i b_i c_i \le \text{bank} + \sum_i s_i \hat{c}_i$$

The last constraint is the one people get wrong: **selling funds buying**.
Treating the budget as fixed makes the optimiser far too timid.

The objective carries a per-transfer friction term. Without it the solver burns
all four transfers to gain 0.3 points, spending flexibility you will want next
round. `transfer_ladder` exposes the marginal gain of each successive swap so
you can see where to stop.

### Why not greedy?

Value-per-credit sorting is provably sub-optimal here. The budget makes this a
multi-dimensional knapsack; greedy strands credits it cannot spend well and will
not "spend down" to a premium player even when the alternative use of those
credits is worse. `tests/test_optimize.py::test_beats_greedy_value_per_credit`
checks this on a concrete instance.

---

## 10. The game's scoring, and lineups

### 10.1 Scoring

A player's fantasy score is his PIR, plus 10% when his team wins:

$$S_i = \text{PIR}_i \cdot (1 + 0.1 \cdot \mathbb{1}[\text{win}])$$

So $\mathbb{E}[S_i] \approx \mathbb{E}[\text{PIR}_i](1 + 0.1\,p_{\text{win}})$.
A head coach scores from the final margin $M$ (his team's view): $+10/+20/+25$
for $M \in [1,10] / [11,20] / [21,\infty)$, and $-5/-10/-20$ for the mirrored
losses. With $M \sim N(\mu, 11.5)$ the expectation is a sum over six normal
bands; a margin in $[0, 0.5)$ still counts as a one-point win.

### 10.2 The lineup MILP

Only the starting five (legal formation) and the sixth man score 100%; the
other four score $w_b = 0.5$; the captain, a starter, scores
$\kappa = 2\times$. For each player, binaries $s_i$ (starter), $m_i$ (sixth),
$b_i$ (bench), $c_i$ (captain) with $x_i = s_i + m_i + b_i$ and
$c_i \le s_i$. A formation binary $f_k$ ties the starters' position counts to
one of the legal formations. The coach choice $y_k$ comes out of the same budget.

$$\max \sum_i \text{EV}_{i,1}\,(s_i + m_i + (\kappa - 1)c_i + w_b b_i) + \text{EV}^{\text{coach}}_1
  + \bar w \sum_i x_i \sum_{r>1} \gamma^{r-1} \text{EV}_{i,r}$$

Later rounds enter with the average slot weight
$\bar w = (6 + (\kappa-1) + 4 w_b)/10 = 0.9$, since the lineup will be
re-optimised then.

With a current squad the same model solves for transfers:

- **Transfer cap:** buys are the roster variables of players not owned, and the
  cap counts a coach change.
- **Budget:** each owned player enters at his sell price on both sides, which
  is the familiar $\text{purchases} \le \text{bank} + \text{sales}$ written
  over the roster.

---

## 11. Turns

A round is played over several game days. Between them a player who has not
played yet may replace one who has, and may take the captaincy. The simulator
plays the round $N$ times with common random numbers across rosters:

1. **One margin per fixture.** $M_g \sim N(\mu_g, 11.5)$, with $\mu_g$ from the
   de-vigged moneyline, shared by both clubs.
2. **Player scores.** With injury availability $\pi_i$ and a coach's-decision
   DNP probability $q_i = \min(0.4,\, 0.45\,e^{-m_i/6})$, player $i$ plays with
   probability $\pi_i (1 - q_i)$. When he plays,
   $\text{PIR}_i = \beta_i\,[1 + \tfrac{1.09}{93.5}(M_g - \mu_g)] + \beta_i\,\sigma^{\text{idio}}_i\,\varepsilon$.
   This is mean-preserving: $\beta_i$ is scaled by $1/(1-q_i)$.
3. **The first-Turn decision.** For every reachable plan (each later-Turn bench
   player either stays or replaces one field slot, keeping the formation
   legal; captain either unchanged or moved to a later-Turn starter), score
   first-Turn players at their **realised** points and later-Turn players at
   their **expectations**, and take the argmax. The realised value then uses
   everyone's actual draws.

Two consequences, both used by the search:

- **Bench later-day players before day one.** Starting later-Turn player $j$
  fixes in advance which first-Turn player $k$ sits at weight $w_b$. Benching
  $j$ lets the policy demote $\arg\min_k S_k$ after seeing the scores, which is
  never worse.
- **The captaincy is an option.** The captain's value becomes
  $\max(S_c, \mathbb{E}[S_m])$ for the best later-Turn starter $m$.

On the 2026-27 Round 1 roster this is worth +8 points over the same players in
their best fixed lineup (134.3 → 142.5).

The search hill-climbs on the simulated objective with single swaps, then
paired swaps, which get past budget-locked optima. Finalists are re-scored on
fresh, larger simulations.

**Limitation:** one decision point. A round over three days is treated as
"first day" vs "the rest".

---

## 12. Prices and the value of a credit

Prices move after every game on that game alone:

$$\Delta p = 0.1 \cdot \operatorname{trunc}\!\left(\frac{S - p}{2}\right), \qquad p + \Delta p \ge 4.0$$

A 12.0 player scoring 21 moves to 12.4. The floor only limits losses. The
official formula is undisclosed; this is the rule experienced managers
observe.

The simulator values $\mathbb{E}[\Delta p]$ at the lineup MILP's **shadow
price** of a credit (objective at budget $B+1$ minus at $B$). Credits left in
the bank are worth 35% of that, since they can only be spent from the next
round.

This is why, at equal projections, a 4.0-credit fringe player beats a
4.5-credit one: a DNP costs the latter 0.2 credits and the former nothing.
When the budget does not bind, a credit is worth nothing and price changes
drop out.

---

## 13. Calibration

Nothing here is fitted on a full EuroLeague history. Priors and dispersion
values are principled starting points calibrated to league-average team totals,
not the output of a backtest. Before treating absolute projections as calibrated:

1. Collect several seasons of box scores.
2. Re-fit `POSITION_PRIORS`, `DISPERSION`, `margin_sigma` and the `context`
   tables.
3. Check calibration by decile: sort projections into ten buckets and compare
   mean projected against mean actual PIR. A systematic bias in any bucket
   points at a specific term above.

See [roadmap.md](roadmap.md) — a backtest harness is the highest-value
contribution anyone could make to this project.
