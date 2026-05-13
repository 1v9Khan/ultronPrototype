"""Tests for the explicit-intent matcher + taint tracker."""

from __future__ import annotations

import time

import pytest

from ultron.safety.intent import IntentMatch, matches_explicit_intent
from ultron.safety.taint import (
    DEFAULT_TAINT_TTL_SECONDS,
    TaintTracker,
    get_taint_tracker,
    set_taint_tracker,
)


# ---------------------------------------------------------------------------
# Intent matcher
# ---------------------------------------------------------------------------


def test_empty_user_text_no_match():
    r = matches_explicit_intent("", tool_name="file_delete")
    assert not r.matched


def test_whitespace_only_no_match():
    r = matches_explicit_intent("   ", tool_name="file_delete")
    assert not r.matched


def test_verb_present_object_present_within_window_matches():
    r = matches_explicit_intent(
        "Please delete the temp file in the sandbox",
        tool_name="openclaw.file.delete",
    )
    assert r.matched
    assert r.verb in ("delete", "remove")


def test_verb_without_object_no_match():
    r = matches_explicit_intent(
        "Please delete that immediately",
        tool_name="openclaw.shutdown",  # uses "shutdown" verbs
    )
    # "delete" isn't in the shutdown synonym list, so no match.
    assert not r.matched


def test_object_without_verb_no_match():
    r = matches_explicit_intent(
        "The config file is sitting there",
        tool_name="openclaw.file.delete",
    )
    assert not r.matched


def test_verb_object_too_far_apart_no_match():
    text = (
        "Please delete me " + "lorem ipsum dolor sit amet " * 5
        + " the config file"
    )
    r = matches_explicit_intent(text, tool_name="openclaw.file.delete")
    # Verb at start, object far away -> no match.
    assert not r.matched


def test_shutdown_verb_synonyms():
    r = matches_explicit_intent(
        "Shut down the PC please",
        tool_name="openclaw.shutdown",
    )
    assert r.matched


def test_send_verb_for_email_tool():
    r = matches_explicit_intent(
        "Send the email to bob",
        tool_name="openclaw.message.send",
    )
    assert r.matched


def test_buy_verb_for_purchase_tool():
    r = matches_explicit_intent(
        "Buy a copy of the book from amazon",
        tool_name="openclaw.shop.buy",
    )
    assert r.matched


def test_intent_match_dataclass_shape():
    r = matches_explicit_intent("delete the file", tool_name="x.delete")
    assert isinstance(r, IntentMatch)
    assert hasattr(r, "matched")
    assert hasattr(r, "verb")
    assert hasattr(r, "object_token")
    assert hasattr(r, "reason")


def test_unknown_tool_falls_back_to_broad_verbs():
    r = matches_explicit_intent(
        "Please delete that file",
        tool_name="unknown.weird.tool",
    )
    # Falls back to the broad verb list; "delete" is in it.
    assert r.matched


# ---------------------------------------------------------------------------
# Taint tracker
# ---------------------------------------------------------------------------


def test_record_and_has_taint_exact_match():
    t = TaintTracker(ttl_seconds=10.0)
    digest = t.record(data=b"sensitive bytes", capability="screen_context")
    assert digest
    hit = t.has_taint(data=b"sensitive bytes")
    assert hit is not None
    assert hit.digest == digest
    assert hit.capability == "screen_context"


def test_has_taint_returns_none_for_unseen():
    t = TaintTracker(ttl_seconds=10.0)
    t.record(data=b"foo", capability="screen_context")
    assert t.has_taint(data=b"bar") is None


def test_taint_expires_after_ttl():
    t = TaintTracker(ttl_seconds=0.01)
    t.record(data=b"will expire", capability="screen_context")
    assert t.has_taint(data=b"will expire") is not None
    time.sleep(0.05)
    assert t.has_taint(data=b"will expire") is None


def test_empty_data_returns_none():
    t = TaintTracker()
    assert t.record(data=b"", capability="x") == ""
    assert t.has_taint(data=b"") is None


def test_text_helper():
    t = TaintTracker()
    t.record(data="hello".encode("utf-8"), capability="x")
    assert t.has_taint_str(text="hello") is not None
    assert t.has_taint_str(text="goodbye") is None


def test_max_entries_drops_oldest():
    t = TaintTracker(ttl_seconds=60.0, max_entries=3)
    for i in range(10):
        t.record(data=f"item-{i}".encode(), capability="x")
    # Only the last 3 should be present.
    assert t.size == 3
    assert t.has_taint(data=b"item-9") is not None
    assert t.has_taint(data=b"item-0") is None


def test_clear_drops_all():
    t = TaintTracker()
    t.record(data=b"a", capability="x")
    t.record(data=b"b", capability="x")
    assert t.size == 2
    t.clear()
    assert t.size == 0


def test_singleton_is_stable():
    set_taint_tracker(None)
    t1 = get_taint_tracker()
    t2 = get_taint_tracker()
    assert t1 is t2


def test_set_taint_tracker_replaces_singleton():
    custom = TaintTracker(ttl_seconds=5.0)
    set_taint_tracker(custom)
    assert get_taint_tracker() is custom
    set_taint_tracker(None)
    # After None, the singleton resets and a fresh one is created.
    fresh = get_taint_tracker()
    assert fresh is not custom


def test_default_ttl_is_60_seconds():
    """Sanity check the documented default."""
    assert DEFAULT_TAINT_TTL_SECONDS == 60.0
