"""The PIR statistic: definition, decomposition and re-composition.

EuroLeague's Performance Index Rating is

    PIR = (points + rebounds + assists + steals + blocks + fouls drawn)
        - (missed FG + missed FT + turnovers + shots rejected + fouls committed)

That linear structure is the reason this project models PIR *component-wise*
rather than as one opaque number:

1. Bookmakers price points, rebounds and assists as separate props. Because PIR
   is linear in its parts, a market-implied PIR mean can be assembled from those
   props plus modelled residual components -- so the market's information is
   usable even though almost nobody prices PIR directly.
2. Different components respond differently to context. Rebounds scale with
   pace and with an absent team-mate's vacated boards; assists scale with
   team-mate shot-making; turnovers scale with usage. Collapsing to a single
   per-minute rate throws that away.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from elfantasy.models import BoxScore

POSITIVE_COMPONENTS = ("points", "rebounds", "assists", "steals", "blocks", "fouls_drawn")
NEGATIVE_COMPONENTS = (
    "missed_fg",
    "missed_ft",
    "turnovers",
    "blocks_against",
    "fouls_committed",
)


@dataclass
class StatLine:
    """A projected or observed set of PIR components (per game, not per minute)."""

    points: float = 0.0
    rebounds: float = 0.0
    assists: float = 0.0
    steals: float = 0.0
    blocks: float = 0.0
    fouls_drawn: float = 0.0
    missed_fg: float = 0.0
    missed_ft: float = 0.0
    turnovers: float = 0.0
    blocks_against: float = 0.0
    fouls_committed: float = 0.0

    def scaled(self, factor: float) -> StatLine:
        return StatLine(**{k: v * factor for k, v in asdict(self).items()})

    def blended(self, other: StatLine, weight: float) -> StatLine:
        """Convex blend: ``weight`` toward ``other``."""

        w = min(max(weight, 0.0), 1.0)
        a, b = asdict(self), asdict(other)
        return StatLine(**{k: (1 - w) * a[k] + w * b[k] for k in a})

    @property
    def pir(self) -> float:
        return pir_from_components(asdict(self))

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def pir_from_components(c: dict[str, float]) -> float:
    positive = sum(float(c.get(k, 0.0)) for k in POSITIVE_COMPONENTS)
    negative = sum(float(c.get(k, 0.0)) for k in NEGATIVE_COMPONENTS)
    return positive - negative


def pir_from_boxscore(bs: BoxScore) -> float:
    """Recompute PIR from a raw box score.

    Used to backfill feeds that omit the field, and to validate the ones that
    provide it.
    """

    return pir_from_components(
        {
            "points": bs.points,
            "rebounds": bs.rebounds,
            "assists": bs.assists,
            "steals": bs.steals,
            "blocks": bs.blocks,
            "fouls_drawn": bs.fouls_drawn,
            "missed_fg": max(bs.fg_attempted - bs.fg_made, 0.0),
            "missed_ft": max(bs.ft_attempted - bs.ft_made, 0.0),
            "turnovers": bs.turnovers,
            "blocks_against": bs.blocks_against,
            "fouls_committed": bs.fouls_committed,
        }
    )


def statline_from_boxscore(bs: BoxScore) -> StatLine:
    return StatLine(
        points=bs.points,
        rebounds=bs.rebounds,
        assists=bs.assists,
        steals=bs.steals,
        blocks=bs.blocks,
        fouls_drawn=bs.fouls_drawn,
        missed_fg=max(bs.fg_attempted - bs.fg_made, 0.0),
        missed_ft=max(bs.ft_attempted - bs.ft_made, 0.0),
        turnovers=bs.turnovers,
        blocks_against=bs.blocks_against,
        fouls_committed=bs.fouls_committed,
    )


# --------------------------------------------------------------------------
# Variance
# --------------------------------------------------------------------------
# Per-component dispersion, expressed as the ratio of variance to mean for a
# typical rotation player. Counting stats are mildly over-dispersed relative to
# Poisson; scoring much more so because of three-point variance. These defaults
# come from fitting EuroLeague box scores and are overridable in model.yaml.
DISPERSION = {
    "points": 1.55,
    "rebounds": 1.20,
    "assists": 1.25,
    "steals": 1.05,
    "blocks": 1.10,
    "fouls_drawn": 1.20,
    "missed_fg": 1.15,
    "missed_ft": 1.10,
    "turnovers": 1.10,
    "blocks_against": 1.05,
    "fouls_committed": 1.05,
}

# PIR components are not independent -- a high-usage night lifts points,
# missed shots and turnovers together. This inflation factor accounts for the
# net positive correlation among the terms.
CORRELATION_INFLATION = 1.22


def statline_variance(line: StatLine, dispersion: dict[str, float] | None = None) -> float:
    """Approximate Var(PIR) for a projected stat line.

    Treats each component as over-dispersed Poisson with ``Var = phi * mean``,
    sums the variances (signs square away), and applies a single inflation
    factor for the positive correlation between components.
    """

    disp = dispersion or DISPERSION
    total = 0.0
    for name, mean in line.as_dict().items():
        phi = disp.get(name, 1.15)
        total += phi * max(float(mean), 0.0)
    return total * CORRELATION_INFLATION


def compose_pir_from_props(
    props: dict[str, float],
    modelled: StatLine,
    *,
    trust: dict[str, float] | None = None,
) -> StatLine:
    """Overwrite modelled components with market-implied ones where available.

    ``props`` maps component names (``points``, ``rebounds``, ``assists``, ...)
    to market-implied means from :func:`elfantasy.projection.market.prop_to_mean`.
    Components the market does not price keep their modelled values, but are
    rescaled in proportion to how much the market moved the components it *does*
    price -- if the book expects a bigger night than the model does, the
    unpriced parts of the line should move too.
    """

    trust = trust or {}
    out = modelled.as_dict()
    anchors_model = 0.0
    anchors_market = 0.0

    for name, market_mean in props.items():
        if name not in out:
            continue
        w = min(max(trust.get(name, 1.0), 0.0), 1.0)
        anchors_model += out[name]
        anchors_market += market_mean
        out[name] = (1 - w) * out[name] + w * market_mean

    if anchors_model > 1e-9:
        scale = anchors_market / anchors_model
        scale = min(max(scale, 0.6), 1.6)  # never let one prop dominate the line
        for name in out:
            if name not in props:
                out[name] *= scale

    return StatLine(**out)
