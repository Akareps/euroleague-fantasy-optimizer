"""The fantasy game's rules as one typed object.

``config/rules.yaml`` is the source of truth; this module only parses it, with
defaults for older rule files that predate the lineup and pricing sections so
they keep working with the squad-only optimisers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from elfantasy.config import Section, Settings

DEFAULT_FORMATIONS = ((2, 2, 1), (1, 2, 2), (2, 1, 2), (1, 3, 1), (3, 1, 1))
DEFAULT_COACH_TABLE = ((21, 25.0), (11, 20.0), (1, 10.0), (-10, -5.0), (-20, -10.0), (-999, -20.0))


@dataclass(frozen=True)
class GameRules:
    budget: float = 100.0
    roster: dict[str, int] = field(default_factory=lambda: {"G": 4, "F": 4, "C": 2})
    max_per_club: int = 6
    head_coach: bool = True

    starters: int = 5
    formations: tuple[tuple[int, int, int], ...] = DEFAULT_FORMATIONS
    sixth_man: bool = True
    bench_weight: float = 0.5
    captain_multiplier: float = 2.0

    field_bench_swaps: bool = True
    captain_switch: bool = True

    win_bonus: float = 0.10
    coach_table: tuple[tuple[float, float], ...] = DEFAULT_COACH_TABLE

    transfers_per_round: int = 4
    coach_counts_as_transfer: bool = True
    unlimited_after_rounds: tuple[int, ...] = ()

    price_step_points: float = 2.0
    price_step: float = 0.1
    price_floor: float = 4.0

    @property
    def squad_size(self) -> int:
        return sum(self.roster.values())

    @property
    def field_slots(self) -> int:
        return self.starters + (1 if self.sixth_man else 0)

    def is_legal_formation(self, counts: tuple[int, int, int]) -> bool:
        return tuple(counts) in set(self.formations)

    def unlimited_transfers_before(self, round_no: int) -> bool:
        return (round_no - 1) in self.unlimited_after_rounds

    @classmethod
    def from_settings(cls, settings: Settings) -> GameRules:
        r = settings.rules

        def get(path, default):
            value = r.get(path, default)
            return value.as_dict() if isinstance(value, Section) else value

        positions = get("squad.positions", {})
        roster = {pos: int(v["max"]) for pos, v in positions.items()} or {"G": 4, "F": 4, "C": 2}

        coach_rows = get("scoring.coach", None)
        coach_table = (
            tuple((float(row["min_margin"]), float(row["points"])) for row in coach_rows)
            if coach_rows
            else DEFAULT_COACH_TABLE
        )
        formations = tuple(
            tuple(int(x) for x in f) for f in get("lineup.formations", DEFAULT_FORMATIONS)
        )

        return cls(
            budget=float(get("budget.total", 100.0)),
            roster=roster,
            max_per_club=int(get("squad.max_per_club", 6)),
            head_coach=bool(get("squad.head_coach", False)),
            starters=int(get("lineup.starters", 5)),
            formations=formations,
            sixth_man=bool(get("lineup.sixth_man", False)),
            bench_weight=float(get("lineup.bench_weight", 1.0)),
            captain_multiplier=float(get("lineup.captain_multiplier", 1.0)),
            field_bench_swaps=bool(get("turns.field_bench_swaps", False)),
            captain_switch=bool(get("turns.captain_switch", False)),
            win_bonus=float(get("scoring.win_bonus", 0.0)),
            coach_table=coach_table,
            transfers_per_round=int(get("transfers.per_round", 4)),
            coach_counts_as_transfer=bool(get("transfers.coach_counts", True)),
            unlimited_after_rounds=tuple(
                int(x) for x in get("transfers.unlimited_after_rounds", [])
            ),
            price_step_points=float(get("pricing.step_points", 2.0)),
            price_step=float(get("pricing.step", 0.1)),
            price_floor=float(get("pricing.floor", 4.0)),
        )
