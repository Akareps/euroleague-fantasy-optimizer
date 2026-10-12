"""Independent minutes-first, component-wise empirical-Bayes experiment.

Gaussian population regression + shrunk player effects for minutes; hierarchical
Gamma-Poisson opportunities/count rates; Beta-Binomial shooting efficiency;
regularised Poisson context regression. Predictive moments include minutes,
rate/efficiency uncertainty, learned overdispersion and component covariance.
No player props. Hyperparameters and coefficients are fitted before the cutoff.
This is an analytic empirical-Bayes approximation, not an MCMC fit.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import groupby

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, truncnorm

from elfantasy.experiments.history import EXPERIMENT, utc

COUNTS = (
    "two_attempted",
    "three_attempted",
    "ft_attempted",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "fouls_drawn",
    "turnovers",
    "blocks_against",
    "fouls_committed",
)
SHOTS = ("two", "three", "ft")
MIN_FEATURES = (
    "intercept",
    "recent_minutes",
    "last_minutes",
    "career_minutes",
    "role_change",
    "evidence",
    "home",
    "rest",
    "blowout",
    "pace",
    "new_club",
)
CTX_FEATURES = ("intercept", "home", "rest", "pace", "opponent", "role_change")


@dataclass(frozen=True)
class Settings:
    recent_half_life_games: float = 8.0
    old_evidence_minutes: float = 120.0
    efficiency_prior_attempts: float = 40.0
    coefficient_prior_sd: float = 8.0
    context_prior_sd: float = 0.25
    selection_prior_sd: float = 2.0
    covariance_shrinkage: float = 0.5
    min_training_rows: int = 500

    def __post_init__(self):
        for value in (
            self.recent_half_life_games,
            self.old_evidence_minutes,
            self.efficiency_prior_attempts,
            self.coefficient_prior_sd,
            self.context_prior_sd,
            self.selection_prior_sd,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("experiment prior scales must be finite and positive")
        if not 0 <= self.covariance_shrinkage <= 1 or self.min_training_rows < 10:
            raise ValueError("invalid covariance shrinkage or training minimum")


def stats(row):
    return np.array([float(row[k]) for k in COUNTS])


class State:
    """Only already observed games. Same-day records are committed together."""

    def __init__(self, pools, settings):
        self.pools, self.settings = pools, settings
        self.players = defaultdict(list)
        self.teams = defaultdict(list)
        self.allowed = defaultdict(list)
        self.rating = defaultdict(float)
        self.latest_date = None

    def observe(self, rows):
        if rows:
            self.latest_date = max(utc(r["date"]) for r in rows)
        grouped = defaultdict(list)
        for r in rows:
            self.players[r["player_id"]].append(r)
            grouped[(r["game_id"], r["club"])].append(r)
        for (_, club), grp in grouped.items():
            total_min = sum(r["minutes"] for r in grp)
            attempts = sum(
                r["two_attempted"]
                + r["three_attempted"]
                + 0.44 * r["ft_attempted"]
                + r["turnovers"]
                for r in grp
            )
            # An opportunity proxy; offensive rebounds aren't supplied in the
            # canonical current-season boxes. Use the same definition everywhere.
            pace = attempts * 200 / max(total_min, 1)
            self.teams[club].append({"date": grp[0]["date"], "pace": pace})
            for pos in "GFC":
                sel = [r for r in grp if r["pos"] == pos]
                if sel:
                    self.allowed[(grp[0]["opp"], pos)].append(
                        (
                            sum((stats(r) for r in sel), start=np.zeros(len(COUNTS))),
                            sum(r["minutes"] for r in sel),
                        )
                    )
        for _, grp in groupby(sorted(rows, key=lambda r: r["game_id"]), key=lambda r: r["game_id"]):
            r = next(iter(grp))
            if not r["home"]:
                home, away, margin = r["opp"], r["club"], -r["margin"]
            else:
                home, away, margin = r["club"], r["opp"], r["margin"]
            error = np.clip(margin - (self.rating[home] - self.rating[away] + 3.5), -30, 30)
            self.rating[home] += 0.08 * error
            self.rating[away] -= 0.08 * error

    def history(self, r):
        return self.players[r["player_id"]]

    def role(self, r):
        rows = self.history(r)
        same = [x for x in rows if x["season"] == r["season"] and x["club"] == r["club"]]
        prior = [x["regulation_minutes"] for x in rows[-40:]]
        seed = float(
            r.get("seed_minutes", np.mean(prior) if prior else self.pools[r["pos"]]["minutes"])
        )
        recent = [x["regulation_minutes"] for x in same[-8:]]
        if recent:
            w = 0.5 ** (np.arange(len(recent) - 1, -1, -1) / 3)
            mean = float((np.dot(w, recent) + seed) / (w.sum() + 1))
            last = recent[-1]
        else:
            mean = last = seed
        career = float(np.mean(prior)) if prior else seed
        last_team = rows[-1]["club"] if rows else r["club"]
        return mean, last, career, len(same), float(last_team != r["club"])

    def context(self, r):
        date = utc(r["date"])
        team = self.teams[r["club"]]
        last_date = utc(team[-1]["date"]) if team else None
        rest = min(max((date - last_date).total_seconds() / 86400 if last_date else 7, 1), 14)
        rest = float(r.get("rest_days", rest))
        pace = np.mean([x["pace"] for x in team[-8:]]) if team else 80.0
        opp = self.teams[r["opp"]]
        opp_pace = np.mean([x["pace"] for x in opp[-8:]]) if opp else 80.0
        pace = float((pace + opp_pace) / 2)
        margin = r.get(
            "expected_margin",
            self.rating[r["club"]] - self.rating[r["opp"]] + (3.5 if r["home"] else -3.5),
        )
        return rest, pace, float(margin)

    def features(self, r):
        recent, last, career, n, changed = self.role(r)
        rest, pace, margin = self.context(r)
        x = np.array(
            [
                1,
                recent / 20,
                last / 20,
                career / 20,
                (recent - career) / 10,
                min(math.log1p(n), 4) / 3,
                float(r["home"]),
                (rest - 7) / 7,
                abs(margin) / 15,
                (pace - 80) / 10,
                changed,
            ]
        )
        ctx = np.array(
            [
                1,
                float(r["home"]),
                (rest - 7) / 7,
                math.log(max(pace, 40) / 80),
                0,
                np.clip((recent - career) / max(career, 5), -1, 1),
            ]
        )
        pool = self.pools[r["pos"]]
        previous = self.allowed[(r["opp"], r["pos"])][-20:]
        if previous:
            allowed = (
                pool["rate"] * 200 + sum((x[0] for x in previous), start=np.zeros(len(COUNTS)))
            ) / (200 + sum(x[1] for x in previous))
            opp_effect = np.log(np.clip(allowed / pool["rate"], 0.5, 2))
        else:
            opp_effect = np.zeros(len(COUNTS))
        return x, ctx, opp_effect

    def posterior(self, r):
        pool = self.pools[r["pos"]]
        old = [x for x in self.history(r) if x["season"] != r["season"] or x["club"] != r["club"]][
            -60:
        ]
        recent = [
            x for x in self.history(r) if x["season"] == r["season"] and x["club"] == r["club"]
        ][-40:]
        rows = old + recent
        weights = []
        old_min = sum(x["minutes"] for x in old)
        scale = min(1.0, self.settings.old_evidence_minutes / max(old_min, 1))
        weights += [scale] * len(old)
        weights += list(
            0.5 ** (np.arange(len(recent) - 1, -1, -1) / self.settings.recent_half_life_games)
        )
        counts = sum(
            (w * stats(x) for w, x in zip(weights, rows, strict=True)), start=np.zeros(len(COUNTS))
        )
        exposure = sum(w * x["minutes"] for w, x in zip(weights, rows, strict=True))
        alpha = pool["rate"] * pool["strength"] + counts
        beta = pool["strength"] + exposure
        rate, rate_var = alpha / beta, alpha / beta**2
        probabilities, prob_var = [], []
        for j, shot in enumerate(SHOTS):
            a = pool["efficiency"][j] * self.settings.efficiency_prior_attempts
            b = (1 - pool["efficiency"][j]) * self.settings.efficiency_prior_attempts
            for w, x in zip(weights, rows, strict=True):
                a += w * x[f"{shot}_made"]
                b += w * (x[f"{shot}_attempted"] - x[f"{shot}_made"])
            probabilities.append(a / (a + b))
            prob_var.append(a * b / ((a + b) ** 2 * (a + b + 1)))
        return rate, rate_var, np.array(probabilities), np.array(prob_var)


def fit_pools(rows):
    pools = {}
    for pos in "GFC":
        selected = [r for r in rows if r["pos"] == pos]
        if not selected:
            selected = rows
        exposure = sum(r["minutes"] for r in selected)
        counts = sum((stats(r) for r in selected), start=np.zeros(len(COUNTS)))
        rate = np.maximum(counts / max(exposure, 1), 0.001)
        by_player = defaultdict(lambda: [np.zeros(len(COUNTS)), 0.0])
        for r in selected:
            by_player[r["player_id"]][0] += stats(r)
            by_player[r["player_id"]][1] += r["minutes"]
        rates = np.array([c / m for c, m in by_player.values() if m >= 100])
        minutes = np.array([m for _, m in by_player.values() if m >= 100])
        between = (
            np.var(rates, axis=0) - rate * np.mean(1 / np.maximum(minutes, 1))
            if len(rates)
            else rate / 120
        )
        strength = np.clip(rate / np.maximum(between, 0.00001), 30, 300)
        efficiency = []
        for shot in SHOTS:
            made = sum(r[f"{shot}_made"] for r in selected)
            attempts = sum(r[f"{shot}_attempted"] for r in selected)
            efficiency.append((made + 1) / (attempts + 2))
        pools[pos] = {
            "rate": rate,
            "strength": strength,
            "efficiency": np.array(efficiency),
            "minutes": float(np.mean([r["regulation_minutes"] for r in selected])),
        }
    return pools


class Model:
    def __init__(self, rows, cutoff, settings=None):
        self.settings = settings or Settings()
        self.cutoff = utc(cutoff)
        self.rows = sorted(
            (r for r in rows if utc(r["date"]).date() < self.cutoff.date()),
            key=lambda r: (r["date"], r["game_id"]),
        )
        if len(self.rows) < self.settings.min_training_rows:
            raise ValueError("not enough strictly earlier historical training rows")
        self.pools = fit_pools(self.rows)
        state = State(self.pools, self.settings)
        X, C, offsets, y, counts, ids, dates = [], [], [], [], [], [], []
        for _, group in groupby(self.rows, key=lambda r: r["date"][:10]):
            group = list(group)
            for r in group:
                x, c, opp = state.features(r)
                rate, _, _, _ = state.posterior(r)
                ctx = np.tile(c, (len(COUNTS), 1))
                ctx[:, 4] = opp
                X.append(x)
                C.append(ctx)
                y.append(r["regulation_minutes"])
                offsets.append(np.log(np.maximum(rate * r["minutes"], 0.0001)))
                counts.append(stats(r))
                ids.append(r["player_id"])
                dates.append(utc(r["date"]))
            state.observe(group)
        self.state = state
        X, C, y, offsets, counts = map(np.asarray, (X, C, y, offsets, counts))
        age = np.array([(self.cutoff - d).total_seconds() / 86400 for d in dates])
        weights = 0.5 ** (age / 730)
        selected = (y > 0).astype(float)
        selection_prior = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0.0])
        selection_penalty = 1 / self.settings.selection_prior_sd**2

        def selection_loss(b):
            eta = X @ b
            diff = b - selection_prior
            return float(
                np.dot(weights, np.logaddexp(0, eta) - selected * eta)
                + 0.5 * selection_penalty * np.dot(diff, diff)
            ), X.T @ (weights * (expit(eta) - selected)) + selection_penalty * diff

        selection_fit = minimize(selection_loss, selection_prior, jac=True, method="L-BFGS-B")
        if not selection_fit.success:
            raise ValueError("participation fit did not converge")
        self.selection_coef = selection_fit.x
        probabilities = expit(X @ self.selection_coef)
        hessian = X.T @ (
            (weights * probabilities * (1 - probabilities))[:, None] * X
        ) + selection_penalty * np.eye(X.shape[1])
        self.selection_cov = cho_solve(cho_factor(hessian), np.eye(X.shape[1]))
        minute_weights = weights * selected
        prior = np.array([0, 12, 2, 6, 0, 0, 0, 0, -1, 0, -1.0])
        coef = prior.copy()
        sigma2 = 36.0
        tau2 = 9.0
        player_effect = {}
        groups = {pid: np.flatnonzero(np.array(ids) == pid) for pid in set(ids)}
        for _ in range(5):
            effect = np.array([player_effect.get(pid, (0, tau2))[0] for pid in ids])
            precision = (
                X.T @ (minute_weights[:, None] * X) / sigma2
                + np.eye(X.shape[1]) / self.settings.coefficient_prior_sd**2
            )
            self.minute_cov = cho_solve(cho_factor(precision), np.eye(X.shape[1]))
            coef = self.minute_cov @ (
                X.T @ (minute_weights * (y - effect)) / sigma2
                + prior / self.settings.coefficient_prior_sd**2
            )
            residual = y - X @ coef
            for pid, idx in groups.items():
                v = 1 / (1 / tau2 + minute_weights[idx].sum() / sigma2)
                player_effect[pid] = (
                    float(v * np.dot(minute_weights[idx], residual[idx]) / sigma2),
                    float(v),
                )
            error = residual - np.array([player_effect[pid][0] for pid in ids])
            sigma2 = max(float(np.average(error**2, weights=minute_weights)), 9.0)
            tau2 = float(np.clip(np.mean([m * m + v for m, v in player_effect.values()]), 1, 36))
        self.minute_coef, self.minute_sigma2 = coef, sigma2
        self.player_effect, self.tau2 = player_effect, tau2
        self.context_coef = np.zeros((len(COUNTS), len(CTX_FEATURES)))
        good = np.array([r["minutes"] >= 3 for r in self.rows])
        expected = np.zeros_like(counts)
        fit_status = []
        for j in range(len(COUNTS)):
            x, outcome, offset, w = C[good, j], counts[good, j], offsets[good, j], weights[good]
            penalty = 1 / self.settings.context_prior_sd**2

            def objective(b, x=x, outcome=outcome, offset=offset, w=w, penalty=penalty):
                eta = np.clip(offset + x @ b, -12, 8)
                mu = np.exp(eta)
                return float(np.dot(w, mu - outcome * eta) + 0.5 * penalty * np.dot(b, b)), x.T @ (
                    w * (mu - outcome)
                ) + penalty * b

            result = minimize(
                objective,
                np.zeros(x.shape[1]),
                jac=True,
                method="L-BFGS-B",
                bounds=[(-0.7, 0.7)] * x.shape[1],
            )
            if not result.success:
                raise ValueError(f"context fit did not converge for {COUNTS[j]}: {result.message}")
            self.context_coef[j] = result.x
            expected[:, j] = np.exp(np.clip(offsets[:, j] + C[:, j] @ result.x, -12, 8))
            fit_status.append(bool(result.success))
        # NB2 process variance = mean + dispersion * mean^2, fitted on training only.
        numerator = ((counts[good] - expected[good]) ** 2 - expected[good]).sum(axis=0)
        self.dispersion = np.clip(
            numerator / np.maximum((expected[good] ** 2).sum(axis=0), 1), 0, 2
        )
        standard = (counts[good] - expected[good]) / np.sqrt(
            np.maximum(expected[good] + self.dispersion * expected[good] ** 2, 0.1)
        )
        corr = np.nan_to_num(np.corrcoef(standard, rowvar=False))
        np.fill_diagonal(corr, 1)
        self.correlation = (
            1 - self.settings.covariance_shrinkage
        ) * corr + self.settings.covariance_shrinkage * np.eye(len(COUNTS))
        self.fit_info = {
            "training_rows": len(self.rows),
            "games": len({r["game_id"] for r in self.rows}),
            "seasons": sorted({r["season"] for r in self.rows}),
            "latest_training_tip": max(r["date"] for r in self.rows),
            "cutoff": self.cutoff.isoformat(),
            "minute_residual_sd": math.sqrt(sigma2),
            "context_converged": fit_status,
            "selection_converged": bool(selection_fit.success),
            "approximation": "analytic empirical Bayes",
        }

    def minute_prediction(self, r):
        x, _, _ = self.state.features(r)
        effect, effect_var = self.player_effect.get(r["player_id"], (0.0, self.tau2))
        mean = float(x @ self.minute_coef + effect)
        variance = self.minute_sigma2 + effect_var + float(x @ self.minute_cov @ x)
        sd = math.sqrt(variance)
        a, b = -mean / sd, (40 - mean) / sd
        m, v = truncnorm.stats(a, b, loc=mean, scale=sd, moments="mv")
        probability = float(
            expit(
                (x @ self.selection_coef)
                / math.sqrt(1 + math.pi * float(x @ self.selection_cov @ x) / 8)
            )
        )
        return float(probability * m), float(
            probability * v + probability * (1 - probability) * m * m
        )

    def predict(self, r, minutes=None, minute_variance=None):
        if self.state.latest_date is not None and utc(r["date"]) <= self.state.latest_date:
            raise ValueError(
                f"prediction would use evidence from the target game or later: {r.get('game_id')} at {r['date']}, evidence {self.state.latest_date.isoformat()}"
            )
        if minutes is None:
            minutes, minute_variance = self.minute_prediction(r)
        rate, rv, efficiency, ev = self.state.posterior(r)
        _, c, opponent = self.state.features(r)
        ctx = np.tile(c, (len(COUNTS), 1))
        ctx[:, 4] = opponent
        mult = np.exp(np.clip((ctx * self.context_coef).sum(axis=1), -0.8, 0.8))
        rate, rv = rate * mult, rv * mult**2
        means = minutes * rate
        variance = means + minutes**2 * (self.dispersion * rate**2 + rv)
        signs = np.array(
            [
                3 * efficiency[0] - 1,
                4 * efficiency[1] - 1,
                2 * efficiency[2] - 1,
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
            ]
        )
        shot_var = 0.0
        for j, score_multiplier in enumerate((3, 4, 2)):
            shot_var += score_multiplier**2 * (
                means[j] * (efficiency[j] * (1 - efficiency[j]) - ev[j])
                + (variance[j] + means[j] ** 2) * ev[j]
            )
        covariance = self.correlation * np.outer(np.sqrt(variance), np.sqrt(variance))
        pir_rate = float(signs @ rate)
        pir_mean = float(signs @ means)
        pir_variance = max(
            float(signs @ covariance @ signs) + shot_var + pir_rate**2 * minute_variance, 1.0
        )
        made = means[:3] * efficiency
        components = {
            "points": float(np.dot(made, [2, 3, 1])),
            "missed_fg": float(means[:2].sum() - made[:2].sum()),
            "missed_ft": float(means[2] - made[2]),
            **{k: float(means[j]) for j, k in enumerate(COUNTS) if j >= 3},
        }
        return {
            "minutes": float(minutes),
            "minute_sd": math.sqrt(minute_variance),
            "pir": pir_mean,
            "pir_sd": math.sqrt(pir_variance),
            "components": components,
            "shot_attempts": dict(zip(SHOTS, (float(x) for x in means[:3]), strict=True)),
            "shooting_efficiency": dict(zip(SHOTS, (float(x) for x in efficiency), strict=True)),
            "uncertainty": "moment approximation: minutes, gamma/beta posterior, NB2 process and shrunk covariance",
        }

    def snapshot(self):
        def plain(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            return value

        return {
            "settings": asdict(self.settings),
            "fit": self.fit_info,
            "minute_features": MIN_FEATURES,
            "selection_coefficients": self.selection_coef.tolist(),
            "selection_coefficient_covariance": self.selection_cov.tolist(),
            "minute_coefficients": self.minute_coef.tolist(),
            "minute_coefficient_covariance": self.minute_cov.tolist(),
            "player_minute_effects": self.player_effect,
            "context_features": CTX_FEATURES,
            "count_components": COUNTS,
            "context_coefficients": self.context_coef.tolist(),
            "dispersion": self.dispersion.tolist(),
            "correlation": self.correlation.tolist(),
            "position_pools": plain(self.pools),
        }


def apply_model(baseline, fitted: Model, season_model):
    """Deep-copy forecasts; common prices, availability, game context and coaches."""
    out = copy.deepcopy(baseline)
    duplicate = Counter(p.get("code") for p in out["players"] if p.get("code"))
    before_by_name = {p["name"]: p for p in baseline["players"]}
    audit, skipped = [], []
    for round_no in baseline["rounds"]:
        name = str(round_no)
        records, forecasts = {}, {}
        for p in out["players"]:
            q = p["proj"].get(name)
            if q is None or q["play"] <= 0.02:
                continue
            code = p.get("code")
            if not code or duplicate[code] > 1:
                if round_no == baseline["round"]:
                    skipped.append(
                        {"player": p["name"], "reason": "missing or ambiguous official identity"}
                    )
                continue
            fixture = next(
                g
                for g in season_model.games.values()
                if g["round"] == round_no
                and p["club"] in (g["local"]["club"]["code"], g["road"]["club"]["code"])
            )
            previous_dates = [
                utc(g["utcDate"])
                for g in season_model.games.values()
                if p["club"] in (g["local"]["club"]["code"], g["road"]["club"]["code"])
                and utc(g["utcDate"]) < utc(fixture["utcDate"])
            ]
            rest = (
                (utc(fixture["utcDate"]) - max(previous_dates)).total_seconds() / 86400
                if previous_dates
                else 7
            )
            r = {
                "player_id": code,
                "club": p["club"],
                "pos": p["pos"],
                "season": season_model.season.code,
                "date": fixture["utcDate"],
                "home": q["home"],
                "opp": q["opp"],
                "expected_margin": q["margin"],
                "seed_minutes": p["base_min"],
                "rest_days": rest,
            }
            # No history: retain researched basketball-role prior, rather than
            # assigning all cheap newcomers an average rotation role.
            if not fitted.state.history(r):
                mu, var = q["min"], fitted.minute_sigma2
            else:
                mu, var = fitted.minute_prediction(r)
            records[p["name"]] = r
            forecasts[p["name"]] = (mu, var)
        # Availability frees rotation time; distribute by position and headroom.
        for club in {p["club"] for p in out["players"]}:
            roster = [p for p in out["players"] if p["club"] == club and p["name"] in forecasts]
            lost = defaultdict(float)
            for p in out["players"]:
                if (
                    p["club"] == club
                    and name in p["proj"]
                    and p.get("code")
                    and duplicate[p["code"]] == 1
                ):
                    lost[p["pos"]] += min(p["base_min"], 32) * (1 - p["proj"][name]["play"])
            for pos, missing in lost.items():
                eligible = [p for p in roster if p["proj"][name]["play"] >= 0.35]
                weights = [
                    max(34 - forecasts[p["name"]][0], 0)
                    * (1 if p["pos"] == pos else 0.25)
                    * min(forecasts[p["name"]][0] / 12, 1)
                    for p in eligible
                ]
                total = sum(weights)
                for p, w in zip(eligible, weights, strict=True):
                    mu, var = forecasts[p["name"]]
                    forecasts[p["name"]] = (
                        (min(mu + 0.7 * missing * w / total, 36), var) if total else (mu, var)
                    )
            total = sum(forecasts[p["name"]][0] * p["proj"][name]["play"] for p in roster)
            if total > 200:
                for p in roster:
                    mu, var = forecasts[p["name"]]
                    scale = 200 / total
                    forecasts[p["name"]] = (mu * scale, var * scale**2)
        for p in out["players"]:
            if p["name"] not in forecasts:
                continue
            q = p["proj"][name]
            mu, var = forecasts[p["name"]]
            pred = fitted.predict(records[p["name"]], mu, var)
            fantasy = max(pred["pir"], 0.25) * (1 + season_model.rules.win_bonus * q["win"])
            sd = pred["pir_sd"] * (1 + season_model.rules.win_bonus * q["win"])
            before = q["ev"]
            q.update(
                min=mu,
                if_plays=fantasy,
                ev=q["play"] * fantasy,
                raw_ev=q["play"] * fantasy,
                independent={
                    **pred,
                    "fantasy_sd_if_available": sd,
                    "p10_if_available": fantasy + norm.ppf(0.1) * sd,
                    "p90_if_available": fantasy + norm.ppf(0.9) * sd,
                },
            )
            if round_no == baseline["round"]:
                p["sd_per_pir"] = sd / max(fantasy, 0.25)
                audit.append(
                    {
                        "player": p["name"],
                        "baseline_ev": before,
                        "experiment_ev": q["ev"],
                        "baseline_minutes": before_by_name[p["name"]]["proj"][name]["min"],
                        "experiment_minutes": mu,
                    }
                )
    out["experiment"] = {
        "name": EXPERIMENT,
        "status": "experimental",
        "fit": fitted.fit_info,
        "player_props_used": False,
        "settings": asdict(fitted.settings),
        "planner_distribution": "existing TurnSimulator with learned score mean and SD",
    }
    return out, {
        "updated_players": len(audit),
        "changes": audit,
        "skipped": skipped,
        "fit": fitted.fit_info,
        "note": "Availability, prices and coaches are shared. New players use researched role priors; no target-round outcomes or player props enter the fit.",
    }
