"""Registry discovery via /.well-known/ultron.json (T8).

T8 (openclaw-clawhub catalog port; see ``THIRD_PARTY_NOTICES.md``).
A stable, namespaced JSON file at ``<site>/.well-known/ultron.json``
returning ``{api_base, auth_base?, min_ultron_version?, extras?}``.
Clients fetch this once at startup to discover the actual API base
+ auth base + minimum-supported runtime version. The fallback
chain is:

    well-known-current
      -> well-known-legacy (``ultron.json.legacy.json`` -- present
         only when the registry has gone through a rename and wants
         old clients to keep resolving)
      -> environment-variable override (``ULTRON_REGISTRY``)
      -> hardcoded default

The ``min_ultron_version`` field lets the registry refuse to talk
to outdated clients without an explicit error code: clients see
the published minimum and self-warn / refuse to connect when
local version is below.

Net-new ultron utility: lets the user (or operator) flip endpoints
(e.g. local-network mirror) without code edits. Future skill /
MCP / voicepack registries can publish their well-known file so
ultron auto-discovers them.

Network IO is INJECTED via a fetcher callable so tests stay
hermetic and the same primitive composes with rate-limit tracker
(T14), trust envelope (T1), and trusted-hosts gate (already in
:mod:`ultron.skills.marketplace`).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)

#: Canonical path under any registry origin.
WELL_KNOWN_PATH: str = "/.well-known/ultron.json"

#: Legacy path read as fallback (registry-name rotation tolerance).
WELL_KNOWN_LEGACY_PATH: str = "/.well-known/ultron.legacy.json"

#: Default in-memory cache TTL (seconds). Sub-15-minute by design --
#: short enough that the orchestrator picks up registry-side config
#: changes on the next startup without persisting stale endpoints.
DEFAULT_DISCOVERY_TTL_SECONDS: int = 15 * 60

#: Environment-variable override read before falling back to the
#: hardcoded default.
DISCOVERY_ENV_OVERRIDE: str = "ULTRON_REGISTRY"

#: Hardcoded default when no override + no well-known file.
#: Single-user ultron mostly runs without any registry so this
#: stays empty by default; operators populate per-deployment.
DEFAULT_REGISTRY_BASE: str = ""


class DiscoveryError(RuntimeError):
    """Base class for discovery-related failures."""


class UntrustedHostError(DiscoveryError):
    """Raised when discovery is attempted against a non-allowlisted host."""

    def __init__(self, host: str) -> None:
        super().__init__(
            f"Refusing to fetch well-known JSON from non-trusted host {host!r}"
        )
        self.host = host


@dataclass(frozen=True)
class DiscoveredRegistry:
    """Result of one well-known discovery call.

    Fields:
        api_base: the resolved API base URL.
        auth_base: optional separate auth-endpoint base (None when
            auth runs on the same origin as the API).
        min_runtime_version: minimum ultron runtime version the
            registry advertises support for. Clients with a lower
            version should self-warn.
        extras: free-form metadata the registry chose to include
            (e.g. ``{"region": "eu-west", "features": [...]}``).
        source_url: the URL the values were resolved from.
        discovered_at: when the discovery succeeded (Unix epoch
            seconds).
        from_legacy: True iff resolution fell through to the
            ``ultron.legacy.json`` path.
    """

    api_base: str
    auth_base: Optional[str] = None
    min_runtime_version: Optional[str] = None
    extras: Mapping[str, object] = field(default_factory=dict)
    source_url: str = ""
    discovered_at: float = 0.0
    from_legacy: bool = False


@dataclass(frozen=True)
class FetchResponse:
    """One HTTP response from an injected fetcher.

    Kept minimal so callers can synthesise responses without taking
    a dependency on ``requests`` / ``httpx``. ``status`` of 404
    triggers fallback to the legacy path; any other non-200 is
    treated as a hard error.

    ``body`` is the raw JSON text.
    """

    status: int
    body: str = ""


FetcherCallable = Callable[[str], FetchResponse]


def _normalise_base(value: str) -> str:
    """Strip trailing slashes from a base URL for canonical comparison."""
    return value.rstrip("/") if value else value


def _parse_payload(payload: str) -> dict:
    """Parse the well-known body as a top-level JSON object.

    Raises :class:`DiscoveryError` on malformed JSON / non-object
    top-level / decode failure.
    """
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DiscoveryError(f"well-known JSON parse failure: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DiscoveryError(
            f"well-known top-level is {type(parsed).__name__}, expected object"
        )
    return parsed


def _build_discovered(payload: dict, *, source_url: str, from_legacy: bool, now: float) -> DiscoveredRegistry:
    """Construct a :class:`DiscoveredRegistry` from the parsed payload."""
    api_base = payload.get("apiBase") or payload.get("api_base")
    if not isinstance(api_base, str) or not api_base.strip():
        raise DiscoveryError("well-known payload missing 'apiBase'")
    auth_base_raw = payload.get("authBase") or payload.get("auth_base")
    auth_base: Optional[str] = None
    if isinstance(auth_base_raw, str) and auth_base_raw.strip():
        auth_base = _normalise_base(auth_base_raw.strip())
    min_version_raw = payload.get("minUltronVersion") or payload.get("min_ultron_version")
    min_version: Optional[str] = None
    if isinstance(min_version_raw, str) and min_version_raw.strip():
        min_version = min_version_raw.strip()
    extras_raw = payload.get("extras")
    extras: Mapping[str, object] = (
        dict(extras_raw) if isinstance(extras_raw, Mapping) else {}
    )
    return DiscoveredRegistry(
        api_base=_normalise_base(api_base.strip()),
        auth_base=auth_base,
        min_runtime_version=min_version,
        extras=extras,
        source_url=source_url,
        discovered_at=now,
        from_legacy=from_legacy,
    )


def discover(
    site_base_url: str,
    *,
    fetcher: FetcherCallable,
    trusted_hosts: Optional[frozenset[str]] = None,
    now: Optional[Callable[[], float]] = None,
) -> Optional[DiscoveredRegistry]:
    """Fetch ``<site>/.well-known/ultron.json`` (legacy-path fallback).

    Returns None when both the current + legacy paths return 404 (a
    registry that simply doesn't publish a well-known file). Raises
    :class:`DiscoveryError` on parse failure / payload-shape failure
    / non-404 HTTP error. Raises :class:`UntrustedHostError` when
    ``trusted_hosts`` is supplied and ``site_base_url``'s host
    isn't a member -- callers gate against the same allowlist the
    marketplace uses for git hosts.

    The :func:`fetcher` is INJECTED so tests don't take a real
    network dependency.
    """
    site = _normalise_base(site_base_url)
    if not site:
        return None
    parsed = urlparse(site)
    host = (parsed.hostname or "").casefold()
    if trusted_hosts is not None and host not in {
        h.casefold() for h in trusted_hosts
    }:
        raise UntrustedHostError(host or site)

    now_fn = now or time.time

    # Try current path.
    current_url = f"{site}{WELL_KNOWN_PATH}"
    response = fetcher(current_url)
    if response.status == 200:
        payload = _parse_payload(response.body)
        return _build_discovered(
            payload, source_url=current_url, from_legacy=False, now=now_fn()
        )
    if response.status != 404:
        raise DiscoveryError(
            f"well-known fetch failed with status {response.status} at {current_url}"
        )

    # Fall through to legacy path.
    legacy_url = f"{site}{WELL_KNOWN_LEGACY_PATH}"
    legacy_response = fetcher(legacy_url)
    if legacy_response.status == 200:
        payload = _parse_payload(legacy_response.body)
        return _build_discovered(
            payload, source_url=legacy_url, from_legacy=True, now=now_fn()
        )
    if legacy_response.status != 404:
        raise DiscoveryError(
            f"legacy well-known fetch failed with status {legacy_response.status} at {legacy_url}"
        )

    return None


def resolve_registry_base(
    *,
    site_base_url: str = "",
    env_override: str = "",
    default: str = DEFAULT_REGISTRY_BASE,
    fetcher: Optional[FetcherCallable] = None,
    trusted_hosts: Optional[frozenset[str]] = None,
) -> tuple[str, Optional[DiscoveredRegistry]]:
    """Resolve the effective registry base + (optional) full discovery.

    Resolution chain:

    1. ``site_base_url`` + ``fetcher`` -> :func:`discover` -> if it
       returns a :class:`DiscoveredRegistry`, that's the answer.
    2. ``env_override`` (stripped of trailing slash) -> use as the
       api_base directly with no full envelope.
    3. ``default`` -> use as the api_base directly with no full
       envelope.

    Returns ``(api_base, discovered_or_none)``. ``api_base`` is
    always a non-empty string when at least one of the three
    sources resolves; empty string when none do. ``discovered_or_none``
    carries the full envelope when path 1 succeeded.
    """
    if site_base_url and fetcher is not None:
        try:
            discovered = discover(
                site_base_url,
                fetcher=fetcher,
                trusted_hosts=trusted_hosts,
            )
        except (DiscoveryError, UntrustedHostError) as exc:
            LOGGER.warning(
                "Well-known discovery against %s failed: %s",
                site_base_url, exc,
            )
            discovered = None
        if discovered is not None:
            return (discovered.api_base, discovered)

    if env_override:
        return (_normalise_base(env_override.strip()), None)

    return (_normalise_base(default), None)


class DiscoveryCache:
    """Thread-safe TTL cache around :func:`discover`.

    The orchestrator constructs one of these at startup and points
    every discovery-consuming subsystem at it; subsequent calls
    within the TTL skip network IO.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_DISCOVERY_TTL_SECONDS,
        now: Optional[Callable[[], float]] = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, Optional[DiscoveredRegistry]]] = {}
        self._ttl = ttl_seconds
        self._now = now or time.time

    def get(
        self,
        site_base_url: str,
        *,
        fetcher: FetcherCallable,
        trusted_hosts: Optional[frozenset[str]] = None,
        force_refresh: bool = False,
    ) -> Optional[DiscoveredRegistry]:
        """Return the cached :class:`DiscoveredRegistry` (or fetch + cache)."""
        key = _normalise_base(site_base_url).casefold()
        now = self._now()
        with self._lock:
            cached = self._cache.get(key)
            if not force_refresh and cached is not None:
                cached_at, value = cached
                if now - cached_at <= self._ttl:
                    return value
            discovered = discover(
                site_base_url,
                fetcher=fetcher,
                trusted_hosts=trusted_hosts,
                now=self._now,
            )
            self._cache[key] = (now, discovered)
            return discovered

    def invalidate(self, site_base_url: Optional[str] = None) -> None:
        """Drop one entry (or everything when ``site_base_url is None``)."""
        with self._lock:
            if site_base_url is None:
                self._cache.clear()
                return
            key = _normalise_base(site_base_url).casefold()
            self._cache.pop(key, None)


__all__ = [
    "WELL_KNOWN_PATH",
    "WELL_KNOWN_LEGACY_PATH",
    "DEFAULT_DISCOVERY_TTL_SECONDS",
    "DISCOVERY_ENV_OVERRIDE",
    "DEFAULT_REGISTRY_BASE",
    "DiscoveryError",
    "UntrustedHostError",
    "FetchResponse",
    "DiscoveredRegistry",
    "discover",
    "resolve_registry_base",
    "DiscoveryCache",
]
