"""DuckDuckGo public-search fallback client.

Uses the community ``duckduckgo-search`` Python library, which scrapes
DuckDuckGo's HTML / Lite endpoints — no API key, no rate limit ceiling
(though DDG will slow-walk responses if you hammer it). Intended as a
fallback when the local SearxNG isn't running AND the Brave API has
been rate-limited / circuit-broken.

Result shape matches :class:`ultron.web_search.brave.SearchResult` so
the rest of the search pipeline stays provider-agnostic.

Failure modes (all return ``[]`` and log the failure):

- ``DDGS`` library raises (network down, blocked, etc.) -> empty + log.
- Timeout -> empty + log.

Slightly slower than Brave API (~500-1500 ms typical vs Brave's
~300-800 ms) because DDG returns HTML and the library parses it.
That's fine for a fallback — it's only reached after the faster
providers have failed.
"""

from __future__ import annotations

import time
from typing import List, Optional

from ultron.config import get_config
from ultron.errors import BraveAPIError
from ultron.resilience import CircuitBreaker, CircuitOpenError, get_error_log
from ultron.utils.logging import get_logger
from ultron.web_search.brave import SearchResult

logger = get_logger("web_search.duckduckgo")


class DuckDuckGoError(BraveAPIError):
    """DuckDuckGo-specific search failure. Subclasses ``BraveAPIError``
    so existing error-log writers handle it uniformly."""


_DDG_BREAKER = CircuitBreaker(
    name="duckduckgo",
    failure_threshold=3,
    window_seconds=300.0,
    cooldown_seconds=300.0,
    expected_exceptions=(DuckDuckGoError,),
)


class DuckDuckGoSearchClient:
    """Client for the DuckDuckGo public search endpoint via the
    ``duckduckgo-search`` library.

    Args:
        timeout_s: per-request timeout passed to the underlying lib.
        region: DDG region code (``"wt-wt"`` worldwide; ``"us-en"`` US
            English). Defaults to config.
        safesearch: ``"moderate"``, ``"strict"``, or ``"off"``.
    """

    def __init__(
        self,
        timeout_s: Optional[float] = None,
        region: Optional[str] = None,
        safesearch: Optional[str] = None,
    ) -> None:
        cfg = get_config().web_search.duckduckgo
        self.timeout_s = (
            timeout_s if timeout_s is not None else float(cfg.timeout_seconds)
        )
        self.region = region if region is not None else cfg.region
        self.safesearch = (
            safesearch if safesearch is not None else cfg.safesearch
        )

    def is_reachable(self) -> bool:
        """The DDG endpoint is reachable when the lib loads + we have
        network. We avoid actually pinging here (would burn an API
        call) and just return True. The provider chain will fall to
        Brave if DDG actually errors out."""
        try:
            from duckduckgo_search import DDGS  # noqa: F401
            return True
        except Exception:                                              # noqa: BLE001
            return False

    def search(
        self,
        query: str,
        count: Optional[int] = None,
    ) -> List[SearchResult]:
        """Run a DuckDuckGo text search.

        Returns up to ``count`` :class:`SearchResult` rows. On failure
        returns ``[]`` and records to ``logs/errors.jsonl``.
        """
        query = query.strip()
        if not query:
            return []
        if count is None:
            count = get_config().web_search.duckduckgo.count

        try:
            return _DDG_BREAKER.call(self._do_search, query, count)
        except CircuitOpenError as e:
            logger.warning(
                "DDG circuit OPEN for %r — short-circuiting; %s",
                query[:80], e,
            )
            get_error_log().record(
                DuckDuckGoError(
                    "circuit open",
                    context={"query": query[:200], "circuit": "duckduckgo"},
                    recovery="short-circuited; caller falls back to base knowledge",
                ),
                dependency="duckduckgo",
                include_traceback=False,
            )
            return []
        except DuckDuckGoError as e:
            get_error_log().record(
                e.with_recovery(
                    "returned empty results; caller falls back to base knowledge"
                ),
                dependency="duckduckgo",
            )
            return []

    def _do_search(self, query: str, count: int) -> List[SearchResult]:
        """Inner implementation. Raises :class:`DuckDuckGoError` on
        failure; the breaker counts toward the threshold."""
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise DuckDuckGoError(
                "duckduckgo-search not installed; "
                "pip install duckduckgo-search",
                context={"query": query[:200]},
            ) from e

        t0 = time.monotonic()
        try:
            with DDGS(timeout=self.timeout_s) as ddgs:
                raw = list(ddgs.text(
                    query,
                    region=self.region,
                    safesearch=self.safesearch,
                    max_results=min(20, max(1, count)),
                ))
        except Exception as e:                                         # noqa: BLE001
            raise DuckDuckGoError(
                f"DuckDuckGo search failed: {e}",
                context={"query": query[:200], "error": str(e)[:200]},
            ) from e

        # DDGS lib row shape:
        #   {"title": "...", "href": "...", "body": "..."}
        out: list[SearchResult] = []
        seen_urls: set[str] = set()
        for row in raw:
            if len(out) >= count:
                break
            url = str(row.get("href", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            out.append(SearchResult(
                url=url,
                title=str(row.get("title", "")).strip(),
                snippet=str(row.get("body", "")).strip(),
                rank=len(out),
            ))
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "DuckDuckGo: %r -> %d results in %.0f ms",
            query[:80], len(out), elapsed_ms,
        )
        return out


__all__ = ["DuckDuckGoSearchClient", "DuckDuckGoError"]
