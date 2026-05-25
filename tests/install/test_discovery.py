"""Tests for the T8 well-known registry discovery."""

from __future__ import annotations

import json
from typing import Callable

import pytest

from ultron.install.discovery import (
    DEFAULT_DISCOVERY_TTL_SECONDS,
    DEFAULT_REGISTRY_BASE,
    DISCOVERY_ENV_OVERRIDE,
    DiscoveredRegistry,
    DiscoveryCache,
    DiscoveryError,
    FetchResponse,
    UntrustedHostError,
    WELL_KNOWN_LEGACY_PATH,
    WELL_KNOWN_PATH,
    discover,
    resolve_registry_base,
)


def _fixed_now(value: float = 1_000_000.0) -> Callable[[], float]:
    return lambda: value


def _ok_payload(**overrides: object) -> str:
    base = {
        "apiBase": "https://example.test/api",
        "authBase": "https://example.test/auth",
        "minUltronVersion": "u1.0.0",
        "extras": {"region": "us-west"},
    }
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# discover


def test_discover_current_path_success() -> None:
    calls: list[str] = []

    def fetcher(url: str) -> FetchResponse:
        calls.append(url)
        return FetchResponse(status=200, body=_ok_payload())

    result = discover(
        "https://example.test", fetcher=fetcher, now=_fixed_now()
    )
    assert result is not None
    assert result.api_base == "https://example.test/api"
    assert result.auth_base == "https://example.test/auth"
    assert result.min_runtime_version == "u1.0.0"
    assert result.extras == {"region": "us-west"}
    assert result.source_url == f"https://example.test{WELL_KNOWN_PATH}"
    assert result.from_legacy is False
    assert calls == [f"https://example.test{WELL_KNOWN_PATH}"]


def test_discover_falls_back_to_legacy() -> None:
    def fetcher(url: str) -> FetchResponse:
        if url.endswith(WELL_KNOWN_PATH):
            return FetchResponse(status=404)
        if url.endswith(WELL_KNOWN_LEGACY_PATH):
            return FetchResponse(status=200, body=_ok_payload())
        raise AssertionError(f"unexpected url {url}")

    result = discover(
        "https://example.test", fetcher=fetcher, now=_fixed_now()
    )
    assert result is not None
    assert result.from_legacy is True
    assert result.source_url.endswith(WELL_KNOWN_LEGACY_PATH)


def test_discover_both_404_returns_none() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=404)

    assert discover("https://example.test", fetcher=fetcher) is None


def test_discover_strips_trailing_slash() -> None:
    captured: list[str] = []

    def fetcher(url: str) -> FetchResponse:
        captured.append(url)
        return FetchResponse(status=200, body=_ok_payload())

    discover("https://example.test/", fetcher=fetcher)
    assert captured[0] == f"https://example.test{WELL_KNOWN_PATH}"


def test_discover_empty_base_returns_none() -> None:
    assert discover("", fetcher=lambda url: FetchResponse(status=404)) is None


def test_discover_non_200_non_404_raises() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=500)

    with pytest.raises(DiscoveryError):
        discover("https://example.test", fetcher=fetcher)


def test_discover_malformed_json_raises() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=200, body="{ not json")

    with pytest.raises(DiscoveryError):
        discover("https://example.test", fetcher=fetcher)


def test_discover_top_level_not_object_raises() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=200, body="[\"array\"]")

    with pytest.raises(DiscoveryError):
        discover("https://example.test", fetcher=fetcher)


def test_discover_missing_api_base_raises() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=200, body=json.dumps({"authBase": "x"}))

    with pytest.raises(DiscoveryError):
        discover("https://example.test", fetcher=fetcher)


def test_discover_accepts_snake_case_keys() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(
            status=200,
            body=json.dumps({
                "api_base": "https://example.test/api",
                "min_ultron_version": "u2.0.0",
            }),
        )

    result = discover("https://example.test", fetcher=fetcher)
    assert result is not None
    assert result.api_base == "https://example.test/api"
    assert result.min_runtime_version == "u2.0.0"


def test_discover_trusted_hosts_allowlist_passes() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=200, body=_ok_payload())

    result = discover(
        "https://example.test",
        fetcher=fetcher,
        trusted_hosts=frozenset({"example.test"}),
    )
    assert result is not None


def test_discover_trusted_hosts_blocks_untrusted() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=200, body=_ok_payload())

    with pytest.raises(UntrustedHostError) as exc_info:
        discover(
            "https://malicious.test",
            fetcher=fetcher,
            trusted_hosts=frozenset({"example.test"}),
        )
    assert "malicious.test" in str(exc_info.value)


def test_discover_trusted_hosts_case_insensitive() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=200, body=_ok_payload())

    result = discover(
        "https://EXAMPLE.TEST",
        fetcher=fetcher,
        trusted_hosts=frozenset({"example.test"}),
    )
    assert result is not None


def test_discover_normalises_trailing_slash_on_api_base() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(
            status=200,
            body=json.dumps({"apiBase": "https://example.test/api/"}),
        )

    result = discover("https://example.test", fetcher=fetcher)
    assert result is not None
    assert result.api_base == "https://example.test/api"


def test_discover_records_discovered_at_timestamp() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=200, body=_ok_payload())

    result = discover(
        "https://example.test", fetcher=fetcher, now=_fixed_now(42.0)
    )
    assert result is not None
    assert result.discovered_at == 42.0


# ---------------------------------------------------------------------------
# resolve_registry_base


def test_resolve_uses_discovery_when_present() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=200, body=_ok_payload())

    api_base, discovered = resolve_registry_base(
        site_base_url="https://example.test", fetcher=fetcher,
    )
    assert api_base == "https://example.test/api"
    assert discovered is not None


def test_resolve_falls_back_to_env_override() -> None:
    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=404)

    api_base, discovered = resolve_registry_base(
        site_base_url="https://example.test",
        env_override="https://override.example/api/",
        fetcher=fetcher,
    )
    assert api_base == "https://override.example/api"
    assert discovered is None


def test_resolve_falls_back_to_default() -> None:
    api_base, discovered = resolve_registry_base(
        site_base_url="",
        env_override="",
        default="https://default.test/api",
    )
    assert api_base == "https://default.test/api"
    assert discovered is None


def test_resolve_returns_empty_when_nothing_set() -> None:
    api_base, discovered = resolve_registry_base()
    assert api_base == ""
    assert discovered is None


def test_resolve_swallows_discovery_errors() -> None:
    """A DiscoveryError on the well-known path falls through to env / default."""

    def bad_fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=500)

    api_base, discovered = resolve_registry_base(
        site_base_url="https://example.test",
        env_override="https://fallback/api",
        fetcher=bad_fetcher,
    )
    assert api_base == "https://fallback/api"
    assert discovered is None


def test_resolve_handles_untrusted_host_silently() -> None:
    """Untrusted-host error in discovery falls through to env override."""

    def fetcher(url: str) -> FetchResponse:
        return FetchResponse(status=200, body=_ok_payload())

    api_base, discovered = resolve_registry_base(
        site_base_url="https://untrusted.test",
        env_override="https://override.test/api",
        fetcher=fetcher,
        trusted_hosts=frozenset({"trusted.test"}),
    )
    assert api_base == "https://override.test/api"
    assert discovered is None


# ---------------------------------------------------------------------------
# DiscoveryCache


def test_cache_hit_skips_fetcher() -> None:
    calls: list[str] = []

    def fetcher(url: str) -> FetchResponse:
        calls.append(url)
        return FetchResponse(status=200, body=_ok_payload())

    cache = DiscoveryCache(ttl_seconds=60, now=_fixed_now())
    a = cache.get("https://example.test", fetcher=fetcher)
    b = cache.get("https://example.test", fetcher=fetcher)
    assert a == b
    assert len(calls) == 1  # second get hit the cache


def test_cache_force_refresh_calls_fetcher() -> None:
    calls: list[str] = []

    def fetcher(url: str) -> FetchResponse:
        calls.append(url)
        return FetchResponse(status=200, body=_ok_payload())

    cache = DiscoveryCache(ttl_seconds=60, now=_fixed_now())
    cache.get("https://example.test", fetcher=fetcher)
    cache.get("https://example.test", fetcher=fetcher, force_refresh=True)
    assert len(calls) == 2


def test_cache_ttl_expiry_refetches() -> None:
    calls: list[str] = []
    times = [1_000_000.0]

    def fetcher(url: str) -> FetchResponse:
        calls.append(url)
        return FetchResponse(status=200, body=_ok_payload())

    def now() -> float:
        return times[0]

    cache = DiscoveryCache(ttl_seconds=60, now=now)
    cache.get("https://example.test", fetcher=fetcher)
    # Advance past TTL.
    times[0] += 61
    cache.get("https://example.test", fetcher=fetcher)
    assert len(calls) == 2


def test_cache_invalidate_single_key() -> None:
    calls: list[str] = []

    def fetcher(url: str) -> FetchResponse:
        calls.append(url)
        return FetchResponse(status=200, body=_ok_payload())

    cache = DiscoveryCache(ttl_seconds=60, now=_fixed_now())
    cache.get("https://example.test", fetcher=fetcher)
    cache.invalidate("https://example.test")
    cache.get("https://example.test", fetcher=fetcher)
    assert len(calls) == 2


def test_cache_invalidate_all() -> None:
    calls: list[str] = []

    def fetcher(url: str) -> FetchResponse:
        calls.append(url)
        return FetchResponse(status=200, body=_ok_payload())

    cache = DiscoveryCache(ttl_seconds=60, now=_fixed_now())
    cache.get("https://a.test", fetcher=fetcher)
    cache.get("https://b.test", fetcher=fetcher)
    cache.invalidate()
    cache.get("https://a.test", fetcher=fetcher)
    cache.get("https://b.test", fetcher=fetcher)
    assert len(calls) == 4


def test_cache_rejects_non_positive_ttl() -> None:
    with pytest.raises(ValueError):
        DiscoveryCache(ttl_seconds=0)
    with pytest.raises(ValueError):
        DiscoveryCache(ttl_seconds=-1)


def test_cache_caches_none_result() -> None:
    """A 404+404 result (None) should still cache to avoid re-querying."""
    calls: list[str] = []

    def fetcher(url: str) -> FetchResponse:
        calls.append(url)
        return FetchResponse(status=404)

    cache = DiscoveryCache(ttl_seconds=60, now=_fixed_now())
    assert cache.get("https://example.test", fetcher=fetcher) is None
    assert cache.get("https://example.test", fetcher=fetcher) is None
    # Two 404 calls (current + legacy) for the first fetch; cache
    # hit on second get -> still 2 total.
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Constants


def test_well_known_constant_value() -> None:
    assert WELL_KNOWN_PATH == "/.well-known/ultron.json"


def test_well_known_legacy_constant_value() -> None:
    assert WELL_KNOWN_LEGACY_PATH == "/.well-known/ultron.legacy.json"


def test_default_ttl_constant() -> None:
    assert DEFAULT_DISCOVERY_TTL_SECONDS == 15 * 60
