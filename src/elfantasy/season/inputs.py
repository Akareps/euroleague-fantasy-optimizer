"""Hand-researched inputs for a season, round by round.

    seasons/<name>/season.yaml        clubs, outright odds, preseason roles,
                                      eye-test notes, model settings
    seasons/<name>/rounds/rNN.yaml    availability, betting lines, and (after
                                      the round) who missed it injured
    seasons/<name>/private/rNN.yaml   your squad, bank and app prices (git-ignored)

Players are keyed by (club code, normalised name), so "Nadir Hifi" and "HIFI,
NADIR" are the same key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from elfantasy.config import REPO_ROOT
from elfantasy.data.injuries import normalise_name

Key = tuple[str, str]  # (club code, normalised player name)


class SeasonInputError(RuntimeError):
    pass


def key(club: str, name: str) -> Key:
    return club, normalise_name(name)


@dataclass(frozen=True)
class EvidenceWeights:
    """Prior weights when updating on games played. Minutes: in games.
    PIR per minute: in minutes. Each newer round counts ``recency`` times the
    one before."""

    k_min_strong: float = 0.5  # same club, 10+ games last season
    k_min_weak: float = 0.5
    n_rate_strong: float = 180.0
    n_rate_hist: float = 180.0  # some EuroLeague history
    n_rate_new: float = 90.0
    recency: float = 1.0


@dataclass(frozen=True)
class CalibrationSettings:
    prior_sd: float = 1.5
    cv: float = 0.634
    tiers: tuple[tuple[float, float], ...] = ((0.0, 10.0), (10.0, 15.0), (15.0, 40.0))


@dataclass(frozen=True)
class EyeTest:
    mult: float
    since: int
    reason: str


@dataclass
class SeasonInputs:
    root: Path
    code: str  # "E2026"
    name: str  # "2026-27"
    clubs: dict[str, str]  # full club name (as on price lists) -> code
    outrights: dict[str, float]
    role: dict[Key, tuple[float, str]]
    minutes_expert: dict[Key, float]
    minutes_factor_round1: dict[Key, float]
    eye_test: dict[Key, EyeTest]
    horizon: int = 3
    gamma: float = 0.4
    evidence: EvidenceWeights = field(default_factory=EvidenceWeights)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    tuned_on: tuple[int, ...] = ()  # rounds looked at when choosing the evidence weights

    @property
    def previous_code(self) -> str:
        return f"E{int(self.code[1:]) - 1}"

    def eye_multiplier(self, k: Key, round_no: int) -> float:
        e = self.eye_test.get(k)
        return e.mult if e and round_no >= e.since else 1.0


@dataclass
class RoundInputs:
    number: int
    availability: dict[Key, tuple[float, ...]]  # P(plays) in this round and the next ones
    moneylines: dict[int, dict[tuple[str, str], tuple[float, float]]]  # round -> (home, away)
    injured_out: set[Key] | None  # filled after the round; None = not recorded yet
    minutes_expert: dict[Key, float]  # stated roles from this round on

    def missed_injured(self) -> set[Key]:
        """Who missed this round through injury (or similar), so that their
        absence is not read as a coach's decision. Falls back to everyone
        given under a 90% chance beforehand."""
        if self.injured_out is not None:
            return self.injured_out
        return {k for k, p in self.availability.items() if p[0] < 0.9}

    def rating_lines(self) -> dict[tuple[str, str], tuple[float, float]]:
        """Every line known when this round was prepared, for team ratings."""
        out: dict[tuple[str, str], tuple[float, float]] = {}
        for rnd in sorted(self.moneylines):
            out.update(self.moneylines[rnd])
        return out


@dataclass
class PrivateInputs:
    squad: list[str]
    coach: str | None
    bank: float | None
    prices: dict[Key, float]
    plans: dict[str, tuple[list[str], str | None]]


# ---------------------------------------------------------------- locating
def default_season_dir() -> Path:
    """``$ELFANTASY_SEASON_DIR``, else the newest folder under ``seasons/``."""

    env = os.getenv("ELFANTASY_SEASON_DIR")
    if env:
        return Path(env)
    base = REPO_ROOT / "seasons"
    found = sorted(p for p in base.glob("*") if (p / "season.yaml").is_file())
    if not found:
        raise SeasonInputError(
            f"no season inputs under {base}; pass --season-dir or set ELFANTASY_SEASON_DIR"
        )
    return found[-1]


def _read(path: Path) -> dict:
    if not path.is_file():
        raise SeasonInputError(f"missing input file: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SeasonInputError(f"{path} must hold a mapping")
    return data


def _players(rows, value=None) -> dict:
    out = {}
    for row in rows or []:
        out[key(row["club"], row["player"])] = row[value] if value else row
    return out


# ---------------------------------------------------------------- loading
def load_season(root: Path | None = None) -> SeasonInputs:
    root = Path(root) if root else default_season_dir()
    d = _read(root / "season.yaml")
    model = d.get("model", {})
    ev = model.get("evidence", {})
    cal = model.get("calibration", {})
    return SeasonInputs(
        root=root,
        code=str(d["season"]),
        name=str(d.get("name", root.name)),
        clubs={str(k): str(v) for k, v in (d.get("clubs") or {}).items()},
        outrights={str(k): float(v) for k, v in d["outrights"].items()},
        role={
            key(r["club"], r["player"]): (float(r["mult"]), str(r.get("note", "")))
            for r in d.get("role") or []
        },
        minutes_expert={
            k: float(v) for k, v in _players(d.get("minutes_expert"), "minutes").items()
        },
        minutes_factor_round1={
            k: float(v) for k, v in _players(d.get("minutes_factor_round1"), "factor").items()
        },
        eye_test={
            key(r["club"], r["player"]): EyeTest(
                float(r["mult"]), int(r.get("since", 1)), str(r.get("reason", ""))
            )
            for r in d.get("eye_test") or []
        },
        horizon=int(model.get("horizon", 3)),
        gamma=float(model.get("gamma", 0.4)),
        evidence=EvidenceWeights(**{k: float(v) for k, v in ev.items()}),
        calibration=CalibrationSettings(
            prior_sd=float(cal.get("prior_sd", 1.5)),
            cv=float(cal.get("cv", 0.634)),
            tiers=tuple(tuple(float(x) for x in t) for t in cal.get("tiers", ()))
            or CalibrationSettings().tiers,
        ),
        tuned_on=tuple(int(x) for x in model.get("tuned_on") or ()),
    )


def round_file(root: Path, round_no: int) -> Path:
    return root / "rounds" / f"r{round_no:02d}.yaml"


def private_file(root: Path, round_no: int) -> Path:
    return root / "private" / f"r{round_no:02d}.yaml"


def rounds_recorded(root: Path) -> list[int]:
    return sorted(int(p.stem[1:]) for p in (root / "rounds").glob("r[0-9][0-9].yaml"))


def load_round(root: Path, round_no: int) -> RoundInputs:
    d = _read(round_file(root, round_no))
    if int(d.get("round", round_no)) != round_no:
        raise SeasonInputError(f"{round_file(root, round_no)} says round {d.get('round')}")
    lines: dict[int, dict[tuple[str, str], tuple[float, float]]] = {}
    for rnd, games in (d.get("moneylines") or {}).items():
        lines[int(rnd)] = {}
        for fixture, (home_price, away_price) in games.items():
            home, away = (s.strip() for s in str(fixture).split("-"))
            lines[int(rnd)][(home, away)] = (float(home_price), float(away_price))
    out = d.get("injured_out")
    return RoundInputs(
        number=round_no,
        availability={
            k: tuple(float(x) for x in v) for k, v in _players(d.get("availability"), "p").items()
        },
        moneylines=lines,
        injured_out=None if out is None else set(_players(out)),
        minutes_expert={
            k: float(v) for k, v in _players(d.get("minutes_expert"), "minutes").items()
        },
    )


def load_private(root: Path, round_no: int) -> PrivateInputs | None:
    path = private_file(root, round_no)
    if not path.is_file():
        return None
    d = _read(path)
    return PrivateInputs(
        squad=[str(n) for n in d.get("squad") or []],
        coach=d.get("coach"),
        bank=None if d.get("bank") is None else float(d["bank"]),
        prices={k: float(v) for k, v in _players(d.get("prices"), "price").items()},
        plans={
            str(name): ([str(n) for n in p.get("players") or []], p.get("coach"))
            for name, p in (d.get("plans") or {}).items()
        },
    )
