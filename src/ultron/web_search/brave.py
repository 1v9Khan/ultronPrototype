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
from typing import Callable, List, Mapping, Optional

from ultron.config import get_config
from ultron.errors import BraveAPIError
from ultron.resilience import CircuitBreaker, CircuitOpenError, get_error_log
from ultron.utils.logging import get_logger


#: Callback shape passed by the chain so each request's response
#: headers can flow into the rate-limit tracker (T14). Signature
#: matches :meth:`SearchProviderChain.record_provider_outcome`
#: (provider id is bound in the closure). Headers may be the
#: :class:`requests.Response.headers` mapping, an httpx case-
#: insensitive headers object, or any plain dict; the
#: :func:`parse_rate_limit_headers` consumer normalises case.
RateLimitRecorder = Callable[[Optional[Mapping[str, object]]], None]

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
        *,
        on_response: Optional[Callable[[Optional[Mapping[str, object]], bool], None]] = None,
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
        # T14 (openclaw-clawhub catalog port). Optional callback the
        # chain installs so rate-limit envelope headers from every
        # request reach the per-provider tracker. Signature is
        # ``(headers, was_429)``. ``None`` (the default) keeps the
        # client legacy-compatible for unit tests + ad-hoc callers.
        self._on_response = on_response

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
            # T14: record headers before raise_for_status so 429s
            # still mark the tracker even when we're about to raise.
            self._record_outcome(
                getattr(resp, "headers", None),
                was_429=(resp.status_code == 429),
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

    def _record_outcome(
        self,
        headers: Optional[Mapping[str, object]],
        *,
        was_429: bool,
    ) -> None:
        """Fire the T14 rate-limit recorder if one was injected.

        Fail-open: a broken recorder must never propagate up into the
        search path. The tracker is best-effort observability.
        """
        if self._on_response is None:
            return
        try:
            self._on_response(headers, was_429)
        except Exception as e:  # noqa: BLE001
            logger.debug("brave rate-limit recorder raised (swallowed): %s", e)

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


# --- T6 auth-profile rotation (multi-key Brave) -----------------------------


def resolve_brave_api_keys(cfg=None) -> List[str]:
    """Resolve the ordered, de-duplicated list of non-empty Brave API keys.

    Reads the primary ``web_search.brave_api_key_env`` first, then each name
    in ``web_search.brave_additional_api_key_envs``. The search chain uses the
    count to decide between the single-client path (0-1 keys) and multi-key
    rotation (2+ keys) (T6).

    Args:
        cfg: optional ``web_search`` config section; pulled from the global
            config when omitted.

    Returns:
        Non-empty API keys in priority order, duplicates removed.
    """
    if cfg is None:
        cfg = get_config().web_search
    env_names: List[str] = [getattr(cfg, "brave_api_key_env", "ULTRON_BRAVE_API_KEY")]
    env_names.extend(getattr(cfg, "brave_additional_api_key_envs", []) or [])
    keys: List[str] = []
    seen = set()
    for name in env_names:
        if not name:
            continue
        val = os.getenv(name, "").strip()
        if val and val not in seen:
            seen.add(val)
            keys.append(val)
    return keys


class RotatingBraveClient:
    """Multi-key Brave client that rotates across API keys (T6).

    Built by the search chain only when two or more Brave keys are
    configured. Each key becomes an :class:`~ultron.providers.AuthProfile`
    under provider ``"brave_search"``; :meth:`search` runs the request through
    :func:`~ultron.providers.execute_with_rotation`, so a rate-limited (429)
    key is cooled down and the next key is tried before the request gives up.
    A key returning auth errors (401/403) is disabled for the session.

    Unlike :class:`BraveSearchClient`, the per-key requests bypass the shared
    ``_BRAVE_BREAKER`` -- the auth-profile store's per-key cooldown + auto-
    disable provide the equivalent protection at key granularity (the shared
    breaker would otherwise open for ALL keys on the first key's failures,
    defeating rotation). Fail-open: any rotation-layer error returns ``[]`` so
    the chain falls through to the next provider. Exposes the same
    ``search(query, count)`` surface as :class:`BraveSearchClient` so the
    chain treats them interchangeably.
    """

    def __init__(
        self,
        keys: List[str],
        *,
        on_response: Optional[Callable[[Optional[Mapping[str, object]], bool], None]] = None,
        store=None,
        provider: str = "brave_search",
    ) -> None:
        if not keys:
            raise ValueError("RotatingBraveClient requires at least one API key")
        from ultron.providers.auth_profiles import AuthProfile, get_profile_store

        self._provider = provider
        self._store = store or get_profile_store()
        self._clients: dict = {}
        for i, key in enumerate(keys):
            profile_id = f"{provider}:{i}"
            self._store.register(
                AuthProfile(
                    profile_id=profile_id,
                    provider=provider,
                    priority=i,
                    metadata={"api_key_index": i},
                )
            )
            self._clients[profile_id] = BraveSearchClient(
                api_key=key, on_response=on_response,
            )

    def search(
        self,
        query: str,
        count: Optional[int] = None,
    ) -> List[BraveResult]:
        """Run a Brave search across the configured keys in rotation.

        Returns the first key's non-empty results, or ``[]`` when every key
        is rate-limited / disabled / returns nothing (chain falls through).
        """
        query = query.strip()
        if not query:
            return []
        if count is None:
            count = get_config().web_search.brave.count

        try:
            from ultron.providers.rotation import (
                RotationOutcome,
                execute_with_rotation,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("brave rotation unavailable (%s); single-key attempt", e)
            first = next(iter(self._clients.values()), None)
            return first.search(query, count) if first is not None else []

        def _operation(profile):
            client = self._clients.get(profile.profile_id)
            if client is None:
                raise BraveAPIError(
                    "no client for profile",
                    context={"profile": profile.profile_id},
                )
            # Bypass the shared breaker; rotation handles per-key failure.
            return client._do_search(query, count)  # noqa: SLF001

        try:
            result = execute_with_rotation(
                provider=self._provider,
                operation=_operation,
                store=self._store,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("brave rotation raised (%s); returning empty", e)
            return []
        if result.outcome == RotationOutcome.SUCCESS:
            return result.value or []
        logger.info(
            "Brave rotation exhausted (outcome=%s); chain falls through",
            getattr(result.outcome, "value", result.outcome),
        )
        return []
