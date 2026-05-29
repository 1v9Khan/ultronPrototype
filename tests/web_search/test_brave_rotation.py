"""Tests for T6 multi-key Brave rotation.

resolve_brave_api_keys collects non-empty keys (primary + additional env
vars) in priority order; RotatingBraveClient rotates across them via the
auth-profile store (rate-limited key -> next key); the chain factory only
builds the rotating client when 2+ keys are configured (single-key path
stays the legacy single client). Local/no-auth providers need no rotation.
"""

from __future__ import annotations

import pytest

from ultron.errors import BraveAPIError
from ultron.providers.auth_profiles import reset_profile_store_for_testing
from ultron.web_search.brave import (
    BraveSearchClient,
    RotatingBraveClient,
    SearchResult,
    resolve_brave_api_keys,
)


@pytest.fixture(autouse=True)
def _fresh_store():
    reset_profile_store_for_testing()
    yield
    reset_profile_store_for_testing()


def _cfg():
    from ultron.config import get_config
    return get_config().web_search


# --- resolve_brave_api_keys -------------------------------------------------


def test_resolve_primary_key_only(monkeypatch):
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY", "primary-key")
    monkeypatch.setattr(_cfg(), "brave_additional_api_key_envs", [])
    assert resolve_brave_api_keys() == ["primary-key"]


def test_resolve_primary_plus_additional(monkeypatch):
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY", "k1")
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY_2", "k2")
    monkeypatch.setattr(_cfg(), "brave_additional_api_key_envs", ["ULTRON_BRAVE_API_KEY_2"])
    assert resolve_brave_api_keys() == ["k1", "k2"]


def test_resolve_dedups_identical_keys(monkeypatch):
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY", "same")
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY_2", "same")
    monkeypatch.setattr(_cfg(), "brave_additional_api_key_envs", ["ULTRON_BRAVE_API_KEY_2"])
    assert resolve_brave_api_keys() == ["same"]


def test_resolve_skips_empty_envs(monkeypatch):
    monkeypatch.delenv("ULTRON_BRAVE_API_KEY", raising=False)
    monkeypatch.setattr(_cfg(), "brave_additional_api_key_envs", [])
    assert resolve_brave_api_keys() == []


# --- RotatingBraveClient ----------------------------------------------------


def test_rotation_falls_through_rate_limited_key_to_next(monkeypatch):
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY", "k1")
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY_2", "k2")
    rc = RotatingBraveClient(["k1", "k2"])

    hit = SearchResult(url="https://x", title="X", snippet="s", rank=0)

    def _rate_limited(query, count):
        raise BraveAPIError("Brave HTTP 429")

    def _ok(query, count):
        return [hit]

    # First key always 429s; second key serves the result.
    rc._clients["brave_search:0"]._do_search = _rate_limited  # type: ignore[attr-defined]
    rc._clients["brave_search:1"]._do_search = _ok  # type: ignore[attr-defined]

    results = rc.search("anything")
    assert results == [hit]


def test_rotation_all_keys_rate_limited_returns_empty(monkeypatch):
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY", "k1")
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY_2", "k2")
    rc = RotatingBraveClient(["k1", "k2"])

    def _rate_limited(query, count):
        raise BraveAPIError("Brave HTTP 429")

    rc._clients["brave_search:0"]._do_search = _rate_limited  # type: ignore[attr-defined]
    rc._clients["brave_search:1"]._do_search = _rate_limited  # type: ignore[attr-defined]

    assert rc.search("anything") == []


def test_rotation_empty_query_short_circuits():
    rc = RotatingBraveClient(["k1", "k2"])
    assert rc.search("   ") == []


def test_rotating_client_requires_a_key():
    with pytest.raises(ValueError):
        RotatingBraveClient([])


# --- chain factory selection ------------------------------------------------


def test_chain_uses_plain_client_for_single_key(monkeypatch):
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY", "only-key")
    monkeypatch.setattr(_cfg(), "brave_additional_api_key_envs", [])
    from ultron.web_search.provider_chain import _make_brave

    client = _make_brave(recorder=None)
    assert isinstance(client, BraveSearchClient)
    assert not isinstance(client, RotatingBraveClient)


def test_chain_uses_rotating_client_for_two_keys(monkeypatch):
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY", "k1")
    monkeypatch.setenv("ULTRON_BRAVE_API_KEY_2", "k2")
    monkeypatch.setattr(_cfg(), "brave_additional_api_key_envs", ["ULTRON_BRAVE_API_KEY_2"])
    from ultron.web_search.provider_chain import _make_brave

    client = _make_brave(recorder=None)
    assert isinstance(client, RotatingBraveClient)
