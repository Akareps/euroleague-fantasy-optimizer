"""Configuration loading.

Two YAML files drive everything:

* ``config/rules.yaml``  -- the fantasy game's rules (budget, squad shape,
  transfer counts). These are *game* facts, and they change between seasons.
* ``config/model.yaml``  -- the projection model's hyper-parameters.

Both are plain nested dicts wrapped in a tiny accessor so that a missing key
fails loudly with a useful path rather than a bare ``KeyError``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"


class ConfigError(RuntimeError):
    pass


class Section:
    """Dict wrapper with dotted-path lookup and helpful errors."""

    def __init__(self, data: dict[str, Any], path: str = "") -> None:
        self._data = data
        self._path = path

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Section({self._path or 'root'}, keys={sorted(self._data)})"

    def get(self, dotted: str, default: Any = ...) -> Any:
        node: Any = self._data
        walked: list[str] = []
        for part in dotted.split("."):
            walked.append(part)
            if not isinstance(node, dict) or part not in node:
                if default is ...:
                    full = ".".join(filter(None, [self._path, ".".join(walked)]))
                    raise ConfigError(f"missing config key: {full}")
                return default
            node = node[part]
        if isinstance(node, dict):
            full = ".".join(filter(None, [self._path, dotted]))
            return Section(node, full)
        return node

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return self.get(key, None) is not None


@dataclass
class Settings:
    rules: Section
    model: Section
    data_dir: Path
    season: str
    odds_api_key: str | None = None
    euroleague_api_base: str = "https://api-live.euroleague.net"
    euroleague_feeds_base: str = "https://feeds.incrowdsports.com/provider/euroleague-feeds"
    odds_api_base: str = "https://api.the-odds-api.com/v4"

    # --- convenience accessors used all over the codebase -------------------
    @property
    def squad_size(self) -> int:
        return int(self.rules.get("squad.size"))

    @property
    def budget(self) -> float:
        return float(self.rules.get("budget.total"))

    @property
    def max_per_club(self) -> int:
        return int(self.rules.get("squad.max_per_club"))

    @property
    def transfers_per_round(self) -> int:
        return int(self.rules.get("transfers.per_round"))

    def position_bounds(self) -> dict[str, tuple[int, int]]:
        raw = self.rules.get("squad.positions")
        data = raw.as_dict() if isinstance(raw, Section) else dict(raw)
        return {k: (int(v["min"]), int(v["max"])) for k, v in data.items()}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"config file is not a mapping: {path}")
    return loaded


def load_settings(
    config_dir: Path | str | None = None,
    *,
    rules_file: str = "rules.yaml",
    model_file: str = "model.yaml",
) -> Settings:
    """Load rules + model config, layering environment overrides on top."""

    cdir = (
        Path(config_dir)
        if config_dir
        else Path(os.getenv("ELFANTASY_CONFIG_DIR", DEFAULT_CONFIG_DIR))
    )
    rules = Section(_read_yaml(cdir / rules_file), "rules")
    model = Section(_read_yaml(cdir / model_file), "model")

    data_dir = Path(os.getenv("ELFANTASY_DATA_DIR", REPO_ROOT / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    season = os.getenv("EL_SEASON") or str(rules.get("season", "E2025"))

    return Settings(
        rules=rules,
        model=model,
        data_dir=data_dir,
        season=season,
        odds_api_key=os.getenv("ODDS_API_KEY") or None,
        euroleague_api_base=os.getenv("EUROLEAGUE_API_BASE", "https://api-live.euroleague.net"),
        euroleague_feeds_base=os.getenv(
            "EUROLEAGUE_FEEDS_BASE", "https://feeds.incrowdsports.com/provider/euroleague-feeds"
        ),
        odds_api_base=os.getenv("ODDS_API_BASE", "https://api.the-odds-api.com/v4"),
    )
