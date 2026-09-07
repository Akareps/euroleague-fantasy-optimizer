"""A small caching HTTP client.

Deliberately conservative: an on-disk cache with a TTL, a minimum interval
between requests to the same host, retries with backoff, and a descriptive
User-Agent. Fantasy data is refreshed a few times a week, not a few times a
second -- there is no reason to hammer anyone's server, and cached responses
make the whole pipeline reproducible and testable.

Before pointing this at a site, check that site's terms of service and
robots.txt. Official APIs and licensed odds feeds are preferable to scraping,
and the adapters here are written so a paid feed can be dropped in without
touching the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

USER_AGENT = (
    "elfantasy/0.1 (+https://github.com/Akareps/euroleague-fantasy-optimizer) python-requests"
)


@dataclass
class HttpClient:
    cache_dir: Path
    ttl_seconds: int = 6 * 3600
    min_interval: float = 0.8
    timeout: float = 20.0
    retries: int = 3
    offline: bool = False
    session: requests.Session = field(default_factory=requests.Session)
    _last_request: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session.headers.setdefault("User-Agent", USER_AGENT)

    # --- cache -------------------------------------------------------------
    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        host = urlparse(url).netloc.replace(":", "_") or "local"
        return self.cache_dir / host / f"{digest}.json"

    def _read_cache(self, path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not self.offline:
            age = time.time() - float(payload.get("stored_at", 0))
            if age > self.ttl_seconds:
                return None
        return payload.get("body")

    def _write_cache(self, path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stored_at": time.time(), "body": body}), encoding="utf-8")

    # --- fetching ----------------------------------------------------------
    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        last = self._last_request.get(host)
        if last is not None:
            wait = self.min_interval - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
        self._last_request[host] = time.time()

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> str | None:
        """GET a URL, returning the body as text, or ``None`` on failure."""

        path = self._cache_path(url, params)
        if use_cache:
            cached = self._read_cache(path)
            if cached is not None:
                log.debug("cache hit %s", url)
                return cached

        if self.offline:
            log.warning("offline: no cached response for %s", url)
            return None

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                self._throttle(url)
                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 2**attempt))
                    log.warning("rate limited by %s, waiting %.1fs", url, wait)
                    time.sleep(min(wait, 30.0))
                    continue
                resp.raise_for_status()
                if use_cache:
                    self._write_cache(path, resp.text)
                return resp.text
            except requests.RequestException as exc:
                last_error = exc
                log.warning("request failed (%s/%s) %s: %s", attempt + 1, self.retries, url, exc)
                time.sleep(min(2**attempt, 8))

        log.error("giving up on %s: %s", url, last_error)
        # A stale cache entry beats nothing at all.
        return self._read_cache(path) if use_cache else None

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> Any | None:
        body = self.get(url, params, headers=headers)
        if body is None:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            log.error("response from %s was not JSON (first 200 chars): %s", url, body[:200])
            return None

    def clear_cache(self) -> int:
        removed = 0
        for path in self.cache_dir.rglob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:  # pragma: no cover
                pass
        return removed
