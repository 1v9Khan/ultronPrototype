"""SearxNG self-hosted meta-search client.

SearxNG is an open-source meta-search engine that aggregates results
from Google / Bing / DuckDuckGo / Brave / Wikipedia / etc. in
parallel, then returns a unified JSON response. Running it locally
(Docker container OR Python pip install) gives the prototype
*unlimited* web search with no API keys and typically faster
end-to-end latency than any single public API.

Setup options (operator-side; we don't auto-install):

1. **Docker (recommended):**
   ::
       docker run -d --name searxng \\
           -p 8888:8080 \\
           -v $PWD/searxng:/etc/searxng \\
           searxng/searxng

2. **Python pip:**
   ::
       pip install searxng
       searxng-run

The default endpoint is ``http://localhost:8888`` and the JSON API
lives at ``/search?q=...&format=json``. Both are configurable via
``web_search.searxng.*`` in ``config.yaml``.

Result shape matches :class:`ultron.web_search.brave.SearchResult` so
the rest of the search pipeline doesn't need to know which provider
served the query.

Failure modes (all return ``[]`` and log the failure):

- ``localhost:8888`` not listening (SearxNG service down) -> connection
  refused, returns empty, caller falls back to the next provider.
- HTTP timeout -> empty + log.
- Non-200 status -> empty + log.
- Malformed JSON -> empty + log.

Circuit-breaker protected so a flapping SearxNG doesn't keep adding
latency to every query — three consecutive failures opens the
breaker for 5 minutes, then re-tries.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from ultron.config import get_config
from ultron.errors import BraveAPIError
from ultron.resilience import CircuitBreaker, CircuitOpenError, get_error_log
from ultron.utils.logging import get_logger
from ultron.web_search.brave import SearchResult

logger = get_logger("web_search.searxng")


# Reuse the same ``BraveAPIError`` exception class for consistency in
# error logging -- callers downstream just care that a search provider
# call failed. We could introduce a separate class but it would buy
# nothing structural. Subclass for clarity:
class SearxNGError(BraveAPIError):
    """SearxNG-specific search failure. Subclasses ``BraveAPIError`` so
    error-log writers / circuit breakers can handle either uniformly."""


_SEARXNG_BREAKER = CircuitBreaker(
    name="searxng",
    failure_threshold=3,
    window_seconds=300.0,
    cooldown_seconds=300.0,
    expected_exceptions=(SearxNGError,),
)


class SearxNGSearchClient:
    """Client for a self-hosted SearxNG instance.

    Args:
        base_url: ``http://localhost:8888`` style URL. Pulled from
            ``web_search.searxng.base_url`` if not given.
        timeout_s: per-request HTTP timeout.
        categories: comma-separated list (``"general"``, ``"news"``).
            Defaults to the config value.
        engines: optional comma-separated upstream-engine list to
            constrain (e.g., ``"google,duckduckgo,wikipedia"``).
            Empty means use SearxNG's default engine set.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
        categories: Optional[str] = None,
        engines: Optional[str] = None,
    ) -> None:
        cfg = get_config().web_search.searxng
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.timeout_s = (
            timeout_s if timeout_s is not None else float(cfg.timeout_seconds)
        )
        self.categories = (
            categories if categories is not None else cfg.categories
        )
        self.engines = engines if engines is not None else cfg.engines

    def is_reachable(self) -> bool:
        """Cheap healthcheck: GET ``/`` with a tight timeout.

        Used by the provider chain to skip SearxNG quickly when it's
        not running (avoids paying the per-query connection-refused
        latency).
        """
        try:
            import requests
            r = requests.get(self.base_url + "/", timeout=1.5)
            return r.ok
        except Exception:                                              # noqa: BLE001
            return False

    def search(
        self,
        query: str,
        count: Optional[int] = None,
    ) -> List[SearchResult]:
        """Run a search via the local SearxNG instance.

        Returns up to ``count`` :class:`SearchResult` rows. On any
        failure (service down, timeout, HTTP error, malformed JSON,
        circuit open) returns ``[]`` and records the failure to
        ``logs/errors.jsonl``. Caller (the provider chain) then
        falls back to the next provider.
        """
        query = query.strip()
        if not query:
            return []
        if count is None:
            count = get_config().web_search.searxng.count

        try:
            return _SEARXNG_BREAKER.call(self._do_search, query, count)
        except CircuitOpenError as e:
            logger.warning(
                "SearxNG circuit OPEN for %r — short-circuiting; %s",
                query[:80], e,
            )
            get_error_log().record(
                SearxNGError(
                    "circuit open",
                    context={"query": query[:200], "circuit": "searxng"},
                    recovery="short-circuited; provider chain falls to next provider",
                ),
                dependency="searxng",
                include_traceback=False,
            )
            return []
        except SearxNGError as e:
            get_error_log().record(
                e.with_recovery(
                    "returned empty results; provider chain falls to next provider"
                ),
                dependency="searxng",
            )
            return []

    def _do_search(self, query: str, count: int) -> List[SearchResult]:
        """Inner implementation. Raises :class:`SearxNGError` on any
        failure; the breaker counts toward the threshold."""
        import requests

        params = {
            "q": query,
            "format": "json",
            "safesearch": "1",
        }
        if self.categories:
            params["categories"] = self.categories
        if self.engines:
            params["engines"] = self.engines

        t0 = time.monotonic()
        try:
            resp = requests.get(
                self.base_url + "/search",
                params=params,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError as e:
            # Most common case: SearxNG isn't running. Make the
            # failure cheap and silent at WARN level so we don't
            # spam errors.jsonl on every voice query when the service
            # is intentionally off.
            raise SearxNGError(
                f"SearxNG unreachable at {self.base_url}",
                context={"query": query[:200], "base_url": self.base_url,
                         "underlying": str(e)[:200]},
            ) from e
        except requests.exceptions.Timeout as e:
            raise SearxNGError(
                f"SearxNG timed out after {self.timeout_s:.1f}s",
                context={"query": query[:200], "timeout_s": self.timeout_s},
            ) from e
        except requests.exceptions.HTTPError as e:
            raise SearxNGError(
                f"SearxNG HTTP {resp.status_code}",
                context={"query": query[:200], "status": resp.status_code,
                         "body_preview": resp.text[:300]},
            ) from e
        except (ValueError, requests.exceptions.RequestException) as e:
            raise SearxNGError(
                f"SearxNG request failed: {e}",
                context={"query": query[:200]},
            ) from e

        # SearxNG JSON shape:
        #   {"query": "...", "results": [{"url": ..., "title": ...,
        #                                 "content": ..., "engine": ...,
        #                                 "category": ...}, ...]}
        raw = data.get("results", []) if isinstance(data, dict) else []
        out: list[SearchResult] = []
        seen_urls: set[str] = set()
        for i, row in enumerate(raw):
            if len(out) >= count:
                break
            url = str(row.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            out.append(SearchResult(
                url=url,
                title=str(row.get("title", "")).strip(),
                snippet=str(row.get("content", "")).strip(),
                rank=len(out),
            ))
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "SearxNG: %r -> %d results in %.0f ms",
            query[:80], len(out), elapsed_ms,
        )
        return out


__all__ = ["SearxNGSearchClient", "SearxNGError"]
