"""Minutes projection.

Minutes are the single highest-leverage input to a fantasy projection: a bench
big who normally plays ten minutes and suddenly plays twenty-eight is worth more
than a star whose price went up. This module produces, for one team and one
fixture:

* a baseline minutes expectation per player from recent usage,
* a redistribution of the minutes vacated by absent players, weighted by
  position overlap and depth-chart rank -- this is the "the starting centre is
  out, so the 3.5-credit backup is now a value play" case,
* a blowout adjustment derived from the game's spread, which takes minutes off
  the rotation and hands them to the end of the bench.

Everything is expressed in *expected* minutes conditional on the player being
available; the probability of playing at all is applied separately by the
engine, so that "70% chance to play 25 minutes" and "certain to play 17" stay
distinguishable in the variance estimate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from elfantasy.config import Section
from elfantasy.models import Availability, BoxScore, Player
from elfantasy.util import clamp, ewma_weights, normalise, shrink, weighted_mean


@dataclass
class DepthChart:
    """Per-team ordering of players by established role."""

    team_code: str
    baseline: dict[str, float] = field(default_factory=dict)  # player -> minutes
    rank: dict[str, int] = field(default_factory=dict)  # player -> 1-based
    position: dict[str, str] = field(default_factory=dict)

    def players(self) -> list[str]:
        return sorted(self.baseline, key=lambda p: self.rank.get(p, 99))


@dataclass
class MinutesProjection:
    player_id: str
    minutes: float
    baseline: float
    from_absences: float = 0.0
    from_blowout: float = 0.0
    play_prob: float = 1.0
    notes: list[str] = field(default_factory=list)


def build_depth_chart(
    team_code: str,
    players: list[Player],
    boxscores: list[BoxScore],
    model: Section,
) -> DepthChart:
    """Establish each player's baseline minutes and rotation rank."""

    half_life = float(model.get("minutes.half_life_games"))
    prior_games = float(model.get("minutes.prior_games"))
    rank_prior = list(model.get("minutes.rank_prior"))
    max_minutes = float(model.get("minutes.max_minutes"))

    by_player: dict[str, list[BoxScore]] = defaultdict(list)
    for bs in boxscores:
        if bs.team_code == team_code:
            by_player[bs.player_id].append(bs)

    raw: dict[str, float] = {}
    appearances: dict[str, int] = {}
    for player in players:
        lines = sorted(by_player.get(player.player_id, []), key=lambda b: b.round)
        mins = [b.minutes for b in lines]
        appearances[player.player_id] = sum(1 for m in mins if m > 0)
        if mins:
            w = ewma_weights(len(mins), half_life)
            raw[player.player_id] = weighted_mean(mins, w)
        else:
            raw[player.player_id] = 0.0

    # Provisional ranking from the raw averages, used to pick the prior.
    order = sorted(raw, key=lambda p: -raw[p])
    baseline: dict[str, float] = {}
    rank: dict[str, int] = {}
    for i, pid in enumerate(order):
        rank[pid] = i + 1
        prior = rank_prior[min(i, len(rank_prior) - 1)] if rank_prior else 12.0
        n = float(appearances.get(pid, 0))
        baseline[pid] = clamp(shrink(raw[pid], float(prior), n, prior_games), 0.0, max_minutes)

    # Re-rank on the shrunk values so the depth chart is internally consistent.
    order = sorted(baseline, key=lambda p: -baseline[p])
    rank = {pid: i + 1 for i, pid in enumerate(order)}

    return DepthChart(
        team_code=team_code,
        baseline=baseline,
        rank=rank,
        position={p.player_id: p.position.value for p in players},
    )


def redistribution_weights(
    absent: str,
    candidates: list[str],
    depth: DepthChart,
    model: Section,
) -> dict[str, float]:
    """How an absent player's minutes are shared out among available team-mates.

    Weighting has three parts:

    * position overlap -- a missing centre mostly frees centre minutes;
    * depth-chart rank -- deeper reserves absorb proportionally more, since the
      starters are already near their ceiling;
    * headroom -- a player already at 34 minutes cannot absorb another eight.
    """

    same_bonus = float(model.get("absence_redistribution.same_position_bonus"))
    cross = float(model.get("absence_redistribution.cross_position_weight"))
    rank_exp = float(model.get("absence_redistribution.backup_rank_exponent"))
    max_minutes = float(model.get("minutes.max_minutes"))

    absent_pos = depth.position.get(absent, "F")
    weights: dict[str, float] = {}
    for pid in candidates:
        if pid == absent:
            continue
        pos_w = same_bonus if depth.position.get(pid) == absent_pos else cross
        rank_w = float(depth.rank.get(pid, 10)) ** rank_exp
        headroom = max(max_minutes - depth.baseline.get(pid, 0.0), 0.0)
        if headroom <= 0.5:
            continue
        # A player who never plays at all is rarely the answer either; damp the
        # very bottom of the roster by their baseline share.
        floor = 0.25 + 0.75 * min(depth.baseline.get(pid, 0.0) / 12.0, 1.0)
        weights[pid] = pos_w * rank_w * headroom * floor

    values = normalise(weights.values())
    return dict(zip(weights.keys(), values, strict=True))


def garbage_time_minutes(spread: float, model: Section) -> float:
    """Expected minutes of decided basketball, from the pre-game spread."""

    threshold = float(model.get("blowout.threshold_points"))
    slope = float(model.get("blowout.slope"))
    cap = float(model.get("blowout.cap_minutes"))
    return clamp(slope * (abs(spread) - threshold), 0.0, cap)


def project_team_minutes(
    team_code: str,
    players: list[Player],
    depth: DepthChart,
    model: Section,
    *,
    spread: float | None = None,
    play_probs: dict[str, float] | None = None,
) -> dict[str, MinutesProjection]:
    """Project minutes for every player on one team for one fixture."""

    reabsorption = float(model.get("absence_redistribution.reabsorption"))
    max_minutes = float(model.get("minutes.max_minutes"))
    team_minutes = float(model.get("minutes.team_minutes"))
    play_prob_table = model.get("availability.play_prob").as_dict()
    ramp_games = int(model.get("availability.return_ramp_games"))
    ramp_factor = float(model.get("availability.return_ramp_factor"))

    play_probs = play_probs or {}
    roster = [p for p in players if p.team_code == team_code]
    ids = [p.player_id for p in roster]

    out: dict[str, MinutesProjection] = {}
    for p in roster:
        prob = play_probs.get(
            p.player_id,
            float(play_prob_table.get(p.status.value, play_prob_table.get("unknown", 0.9))),
        )
        # A stated minutes estimate replaces the statistical baseline and is
        # then left alone: no redistribution gains, no blowout shifts, no
        # trimming. Without this lock, a backup reported at ~13 minutes who
        # ranks 11th on a deep roster gets cut to near zero by the steps below.
        baseline = (
            float(p.minutes_override)
            if p.minutes_override is not None
            else depth.baseline.get(p.player_id, 0.0)
        )
        out[p.player_id] = MinutesProjection(
            player_id=p.player_id,
            minutes=baseline,
            baseline=baseline,
            play_prob=float(prob),
        )
    locked = {p.player_id for p in roster if p.minutes_override is not None}

    # --- 1. redistribute the minutes of players who will not play -----------
    available = [pid for pid in ids if out[pid].play_prob > 0.35]
    beneficiaries = [pid for pid in available if pid not in locked]
    for p in roster:
        pid = p.player_id
        missing_share = 1.0 - out[pid].play_prob
        if missing_share <= 1e-6:
            continue
        vacated = out[pid].baseline * missing_share * reabsorption
        if vacated <= 0.05:
            continue
        weights = redistribution_weights(pid, beneficiaries, depth, model)
        for beneficiary, w in weights.items():
            gain = vacated * w
            out[beneficiary].minutes += gain
            out[beneficiary].from_absences += gain
        if p.status in (Availability.OUT, Availability.DOUBTFUL):
            note = f"absent: {p.name} ({p.status.value})"
            for beneficiary, w in weights.items():
                if vacated * w >= 1.5:
                    out[beneficiary].notes.append(f"+{vacated * w:.1f} min from {note}")

    # A player who is himself unlikely to play should not keep his own baseline
    # in the *conditional* projection, but he must not absorb his own minutes.
    for pid in ids:
        if out[pid].play_prob <= 0.35:
            out[pid].minutes = min(out[pid].minutes, out[pid].baseline)

    # --- 2. minutes restriction for players just back from injury -----------
    for p in roster:
        if p.games_since_return is not None and 0 <= p.games_since_return < ramp_games:
            mp = out[p.player_id]
            mp.minutes *= ramp_factor
            mp.notes.append(f"return ramp ({p.games_since_return + 1}/{ramp_games})")

    # --- 3. blowout / garbage time -----------------------------------------
    if spread is not None:
        garbage = garbage_time_minutes(spread, model)
        if garbage > 0.05:
            starter_share = float(model.get("blowout.starter_loss_share"))
            rotation = sorted(beneficiaries, key=lambda pid: depth.rank.get(pid, 99))
            top, deep = rotation[:5], rotation[5:]
            if top and deep:
                # 5 players * garbage minutes of decided time to reallocate.
                pool = garbage * 5.0
                take = pool * starter_share
                per_starter = take / len(top)
                for pid in top:
                    loss = min(per_starter, max(out[pid].minutes - 8.0, 0.0))
                    out[pid].minutes -= loss
                    out[pid].from_blowout -= loss
                    if loss >= 1.0:
                        out[pid].notes.append(f"-{loss:.1f} min blowout risk")
                gain_weights = normalise([float(depth.rank.get(pid, 10)) ** 0.5 for pid in deep])
                for pid, w in zip(deep, gain_weights, strict=True):
                    gain = take * w
                    out[pid].minutes += gain
                    out[pid].from_blowout += gain
                    if gain >= 1.0:
                        out[pid].notes.append(f"+{gain:.1f} min blowout upside")

    # --- 4. clamp and renormalise to the team's minute budget ---------------
    for mp in out.values():
        mp.minutes = clamp(mp.minutes, 0.0, max_minutes)

    # A team can only hand out `team_minutes` in regulation, so if the steps
    # above have over-allocated, take the excess back -- mostly from the back
    # of the rotation. Coaches shorten the bench, they do not trim their star;
    # a uniform scale-down shaved ~15% off every starter on a deep roster.
    #
    # Deliberately one-directional. Under-allocation is *not* corrected: it
    # means either that we hold an incomplete roster (common -- feeds miss
    # two-way and youth players) or that the coach genuinely runs a short
    # bench. Scaling up in either case would invent minutes, and would also
    # double-count the absence redistribution above, which already reallocates
    # vacated minutes explicitly and intentionally reabsorbs less than 100%.
    excess = sum(out[pid].minutes * out[pid].play_prob for pid in ids) - team_minutes
    if excess > team_minutes * 0.02:
        trimmable = sorted(
            (pid for pid in ids if pid not in locked and out[pid].minutes > 0),
            key=lambda pid: -out[pid].minutes,
        )
        # Weight grows with rotation rank; the cut to a player's minutes is
        # sized so the *expected* minutes removed add up to the excess.
        weight = {
            pid: out[pid].minutes * out[pid].play_prob * (0.25 + rank / 6.0)
            for rank, pid in enumerate(trimmable)
        }
        total_w = sum(weight.values())
        for pid in trimmable:
            if total_w <= 0 or out[pid].play_prob <= 0:
                continue
            cut = excess * weight[pid] / (total_w * out[pid].play_prob)
            out[pid].minutes = clamp(out[pid].minutes - cut, 0.0, max_minutes)

    return out


def summarise_gains(
    projections: dict[str, MinutesProjection], top: int = 5
) -> list[tuple[str, float]]:
    """Biggest minute swings versus baseline -- the actual opportunity list."""

    deltas = [(pid, mp.minutes - mp.baseline) for pid, mp in projections.items()]
    deltas.sort(key=lambda kv: -kv[1])
    return deltas[:top]


def rotation_stability(depth: DepthChart, boxscores: list[BoxScore]) -> float:
    """0-1 score for how predictable a coach's rotation is.

    Used to widen the variance on teams whose minutes swing wildly, which is a
    real and often ignored risk in EuroLeague fantasy.
    """

    by_player: dict[str, list[float]] = defaultdict(list)
    for bs in boxscores:
        if bs.team_code == depth.team_code:
            by_player[bs.player_id].append(bs.minutes)
    cvs = []
    for pid, mins in by_player.items():
        if len(mins) < 3 or depth.baseline.get(pid, 0) < 8:
            continue
        arr = np.asarray(mins, dtype=float)
        if arr.mean() <= 0:
            continue
        cvs.append(arr.std(ddof=1) / arr.mean())
    if not cvs:
        return 0.5
    return float(np.clip(1.0 - np.mean(cvs), 0.0, 1.0))
