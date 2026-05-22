"""Brave Search API client.

Thin wrapper around the Brave Web Search REST endpoint. Returns a
normalized list of :class:`BraveResult` objects; callers don't see
Brave-specific JSON shape.

Rate-limited via a per-client monotonic timestamp -- Brave's free tier
caps concurrent requests, so we space calls out by a configurable
minimum interval.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from ultron.config import get_config
from ultron.errors import BraveAPIError
from ultron.resilience import CircuitBreaker, CircuitOpenError, get_error_log
from ultron.utils.logging import get_logger

logger = get_logger("web_search.brave")


# Single shared breaker — multiple BraveSearchClient instances all
# coordinate through it. Threshold/window/cooldown reflect Brave's
# free-tier behavior: typical failures are rate-limit (429) or 5xx
# bursts; 3 in 5 minutes is enough to declare "unavailable for now".
_BRAVE_BREAKER = CircuitBreaker(
    name="brave",
    failure_threshold=3,
    window_seconds=300.0,
    cooldown_seconds=300.0,
    expected_exceptions=(BraveAPIError,),
)


@dataclass(frozen=True)
class SearchResult:
    """One result row from any of the web-search providers
    (SearxNG, Brave, DuckDuckGo) -- a provider-neutral type
    used throughout the pipeline.

    Was named ``BraveResult`` historically when Brave was the
    only provider; renamed 2026-05-21 to reflect the multi-
    provider chain. ``BraveResult`` is kept as a deprecated
    alias for backward compatibility with any external code or
    pickled cache rows; both names refer to the same class
    object so ``isinstance(x, BraveResult)`` and
    ``isinstance(x, SearchResult)`` are equivalent."""

    url: str
    title: str
    snippet: str  # provider's content/description field
    rank: int  # 0-based position in the result list


# Backward-compat alias.
BraveResult = SearchResult


class BraveSearchClient:
    """Client for Brave Web Search API.

    Args:
        api_key: ``X-Subscription-Token``. Pulled from settings if not given.
        rate_limit_s: minimum seconds between requests across all callers.
        timeout_s: per-request timeout.
        endpoint: override the default URL (rarely needed).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_s: Optional[float] = None,
        timeout_s: Optional[float] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        cfg = get_config().web_search
        self.api_key = api_key or os.getenv(cfg.brave_api_key_env, "")
        if not self.api_key:
            raise ValueError(
                "Brave API key missing. Set ULTRON_BRAVE_API_KEY in your env "
                "or pass api_key=... to BraveSearchClient."
            )
        self.endpoint = endpoint if endpoint is not None else cfg.brave.endpoint
        self.rate_limit_s = (
            rate_limit_s if rate_limit_s is not None else cfg.brave.rate_limit_seconds
        )
        self.timeout_s = (
            timeout_s if timeout_s is not None else cfg.brave.timeout_seconds
        )
        self._last_call = 0.0
        self._lock = threading.Lock()

    def search(
        self,
        query: str,
        count: Optional[int] = None,
    ) -> List[BraveResult]:
        """Run a single Brave search.

        Returns up to ``count`` :class:`BraveResult` rows. On API failure
        (timeout, HTTP error, malformed JSON, rate limit, circuit open)
        returns ``[]`` and records the failure to ``logs/errors.jsonl``.
        Caller falls back to base knowledge with an uncertainty caveat.
        """
        query = query.strip()
        if not query:
            return []
        if count is None:
            count = get_config().web_search.brave.count

        try:
            return _BRAVE_BREAKER.call(self._do_search, query, count)
        except CircuitOpenError as e:
            logger.warning(
                "Brave circuit OPEN for %r — short-circuiting; %s",
                query[:80], e,
            )
            get_error_log().record(
                BraveAPIError(
                    "circuit open",
                    context={"query": query[:200], "circuit": "brave"},
                    recovery="short-circuited; fell back to base knowledge",
                ),
                dependency="brave_api",
                include_traceback=False,
            )
            return []
        except BraveAPIError as e:
            get_error_log().record(
                e.with_recovery("returned empty results; caller falls back to base knowledge"),
                dependency="brave_api",
            )
            return []

    def _do_search(self, query: str, count: int) -> List[BraveResult]:
        """Inner implementation. Raises :class:`BraveAPIError` on any
        failure; the breaker counts those toward the threshold."""
        self._respect_rate_limit()

        import requests

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": min(20, max(1, count)),
            "safesearch": "moderate",
            "result_filter": "web",
        }
        t0 = time.monotonic()
        try:
            resp = requests.get(
                self.endpoint,
                headers=headers,
                params=params,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout as e:
            raise BraveAPIError(
                f"Brave timed out after {self.timeout_s:.1f}s",
                context={"query": query[:200], "timeout_s": self.timeout_s},
            ) from e
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            raise BraveAPIError(
                f"Brave HTTP {status}",
                context={"query": query[:200], "status_code": status},
            ) from e
        except requests.exceptions.RequestException as e:
            raise BraveAPIError(
                f"Brave request failed: {e}",
                context={"query": query[:200]},
            ) from e
        except ValueError as e:
            # Malformed JSON.
            raise BraveAPIError(
                "Brave returned malformed JSON",
                context={"query": query[:200]},
            ) from e

        web_results = (data.get("web") or {}).get("results") or []
        results: List[BraveResult] = []
        for i, row in enumerate(web_results[:count]):
            url = (row.get("url") or "").strip()
            if not url:
                continue
            results.append(
                BraveResult(
                    url=url,
                    title=(row.get("title") or "").strip(),
                    snippet=(row.get("description") or "").strip(),
                    rank=i,
                )
            )
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Brave: %r -> %d results in %.0f ms",
            query[:80], len(results), elapsed_ms,
        )
        return results

    def _respect_rate_limit(self) -> None:
        """Block in-process until enough time has elapsed since the last call.

        Single-process only; the embedded prototype doesn't share a rate
        limit across machines.
        """
        with self._lock:
            now = time.monotonic()
            wait = (self._last_call + self.rate_limit_s) - now
            if wait > 0:
                logger.debug("Brave rate-limit sleep: %.2fs", wait)
                time.sleep(wait)
            self._last_call = time.monotonic()
