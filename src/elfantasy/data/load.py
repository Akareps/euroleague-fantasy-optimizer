"""Assembling a :class:`~elfantasy.dataset.Dataset` from the available sources.

One function, :func:`build_dataset`, with every source optional. A missing
source degrades the projection rather than breaking it: no odds means the Elo
model carries the fixture; no injury file means everyone is assumed available
(and the report says so, loudly).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from elfantasy.config import Settings
from elfantasy.data import fantasy, injuries
from elfantasy.data import odds as odds_mod
from elfantasy.data.euroleague import EuroleagueClient
from elfantasy.data.http import HttpClient
from elfantasy.dataset import Dataset

log = logging.getLogger(__name__)


def build_dataset(
    settings: Settings,
    *,
    offline: bool = False,
    snapshot: Path | None = None,
    prices_csv: Path | None = None,
    injuries_yaml: Path | None = None,
    odds_csv: Path | None = None,
    props_csv: Path | None = None,
    use_odds_api: bool = True,
    fetch_props: bool = True,
    cache_ttl: int = 6 * 3600,
) -> Dataset:
    """Fetch (or load) everything the projection engine needs."""

    if snapshot is not None:
        ds = Dataset.load(snapshot)
        log.info("loaded snapshot %s: %s", snapshot, ds.summary())
        _apply_local_sources(ds, prices_csv, injuries_yaml, odds_csv, props_csv)
        return ds

    http = HttpClient(
        cache_dir=settings.data_dir / "cache",
        ttl_seconds=cache_ttl,
        offline=offline,
    )
    client = EuroleagueClient(
        http,
        season=settings.season,
        api_base=settings.euroleague_api_base,
        feeds_base=settings.euroleague_feeds_base,
    )

    ds = Dataset(season=settings.season, fetched_at=datetime.now())
    ds.teams, ds.players = client.fetch_players()
    ds.games = client.fetch_games()
    ds.boxscores = client.fetch_boxscores(ds.games)
    ds.sources["euroleague"] = settings.euroleague_api_base

    # Teams sometimes only appear in the schedule (e.g. before rosters publish).
    for g in ds.games:
        for code in (g.home_code, g.away_code):
            if code and code not in ds.teams:
                from elfantasy.models import Team

                ds.teams[code] = Team(code=code, name=code)

    # --- odds ---------------------------------------------------------------
    if use_odds_api and settings.odds_api_key and not offline:
        provider = odds_mod.TheOddsApiProvider(
            http=http, api_key=settings.odds_api_key, base=settings.odds_api_base
        )
        try:
            ds.odds = provider.fetch_game_odds(ds.games)
            ds.sources["odds"] = provider.name
            if fetch_props:
                props = provider.fetch_props(ds.games)
                ds.props = odds_mod.resolve_prop_players(props, ds.players)
        except Exception as exc:  # noqa: BLE001 - odds are optional
            log.error("odds fetch failed, continuing without market data: %s", exc)
    elif use_odds_api and not settings.odds_api_key:
        log.info("ODDS_API_KEY not set - projections will use model ratings only")

    _apply_local_sources(ds, prices_csv, injuries_yaml, odds_csv, props_csv)
    ds.annotate_home_flags()
    log.info("dataset ready: %s", ds.summary())
    return ds


def _apply_local_sources(
    ds: Dataset,
    prices_csv: Path | None,
    injuries_yaml: Path | None,
    odds_csv: Path | None,
    props_csv: Path | None,
) -> None:
    if prices_csv:
        fantasy.load_prices_csv(prices_csv, ds.players)
        ds.sources["prices"] = str(prices_csv)

    if injuries_yaml:
        http = HttpClient(cache_dir=Path(".cache"), offline=True)
        providers = injuries.default_providers(http, Path(injuries_yaml))
        records = injuries.merge(providers)
        matched, unmatched = injuries.apply_to_players(records, ds.players)
        ds.sources["injuries"] = f"{injuries_yaml} ({matched} matched, {len(unmatched)} unmatched)"

    if odds_csv or props_csv:
        index = {injuries.normalise_name(p.name): pid for pid, p in ds.players.items()}
        csv_provider = odds_mod.CsvOddsProvider(
            odds_path=odds_csv, props_path=props_csv, player_index=index
        )
        local_odds = csv_provider.fetch_game_odds(ds.games)
        # Local CSV overrides API odds: you typed it, so you meant it.
        ds.odds.update(local_odds)
        local_props = csv_provider.fetch_props(ds.games)
        if local_props:
            ds.props.extend(odds_mod.resolve_prop_players(local_props, ds.players))
        ds.sources["odds_csv"] = str(odds_csv or props_csv)


def data_health(ds: Dataset) -> list[str]:
    """Warnings a manager should see before trusting the output.

    Silent degradation is the main failure mode of a pipeline like this: it
    will happily produce confident numbers from no injury data and no odds.
    """

    warnings: list[str] = []
    if not ds.players:
        warnings.append("no players loaded - check the roster feed")
    priced = sum(1 for p in ds.players.values() if p.price > 0)
    if priced == 0:
        warnings.append("no prices loaded - the optimiser cannot run (use --prices)")
    elif priced < 0.7 * len(ds.players):
        warnings.append(f"only {priced}/{len(ds.players)} players have prices")

    played = sum(1 for g in ds.games if g.played)
    if played < 3:
        warnings.append(f"only {played} completed games - rates are mostly priors so far")
    if not ds.boxscores:
        warnings.append("no box scores - projections will fall back to positional priors")

    from elfantasy.models import Availability

    known = sum(1 for p in ds.players.values() if p.status is not Availability.UNKNOWN)
    if known == 0:
        warnings.append(
            "no availability data - everyone is treated as fit. This is the single "
            "biggest source of error; maintain examples/injuries.yaml"
        )

    if not ds.odds:
        warnings.append("no bookmaker odds - using Elo/pace ratings for game context")
    if not ds.props:
        warnings.append("no player props - player projections are model-only")

    return warnings
