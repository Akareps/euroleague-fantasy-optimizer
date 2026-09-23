"""Download the official EuroLeague feeds the Round 1 scripts need.

Three requests, throttled and cached. The feed rate-limits bursts with
multi-minute Retry-After windows, so this deliberately avoids per-game
endpoints: season totals come from the aggregated statistics feed.

Prices are not fetched. Put the game's price list in the data directory as
``prices_players.csv`` with columns ``rank,player,club,pos,price`` (head
coaches with pos ``HC``). The 2026-27 opening prices were taken from Basketball
Sphere's published list; see README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA  # noqa: E402

from elfantasy.data.http import HttpClient  # noqa: E402

BASE = "https://api-live.euroleague.net"
TARGETS = {
    "games_E2026.json": (f"{BASE}/v2/competitions/E/seasons/E2026/games", None),
    "people_E2026.json": (
        f"{BASE}/v2/competitions/E/seasons/E2026/people",
        {"personType": "J"},
    ),
    "stats_E2025_acc.json": (
        f"{BASE}/v3/competitions/E/statistics/players/traditional",
        {
            "seasonMode": "Single",
            "seasonCode": "E2025",
            "statisticMode": "Accumulated",
            "limit": 1000,
        },
    ),
}


def main() -> int:
    http = HttpClient(cache_dir=DATA / "http", ttl_seconds=6 * 3600, min_interval=2.0)
    for name, (url, params) in TARGETS.items():
        body = http.get(url, params)
        if body is None:
            print(f"failed: {url}")
            return 1
        (DATA / name).write_text(body, encoding="utf-8")
        print(f"saved {name} ({len(body) // 1024} KB)")
    if not (DATA / "prices_players.csv").exists():
        print(f"\nmissing {DATA / 'prices_players.csv'} -- see the module docstring")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
