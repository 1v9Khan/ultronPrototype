"""Tests for S10b — the batch selection engine (kenning.twitch.selection).

Offline + pure: builds synthetic ChatEvents, asserts the dedupe / fairness-cap /
priority / recently-answered / global-cap pipeline and the never-raises contract.
No network, no creds, no models.
"""
from __future__ import annotations

import pytest

from kenning.twitch.clients.eventsub import ChatEvent
from kenning.twitch.selection import (
    Selection,
    normalized_key,
    select_messages,
    simhash,
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _ev(
    text: str,
    *,
    uid: str = "",
    login: str = "",
    badges: list[dict] | None = None,
) -> ChatEvent:
    """Construct a minimal ChatEvent for selection tests."""
    return ChatEvent(
        broadcaster_user_id="b1",
        chatter_user_id=uid,
        chatter_login=login or (uid or "anon"),
        chatter_name=login or (uid or "anon"),
        text=text,
        badges=badges or [],
    )


def _badge(set_id: str) -> dict:
    return {"set_id": set_id, "id": "1", "info": ""}


def _texts(sel: Selection) -> list[str]:
    return [e.text for e in sel.chosen]


def _uids(sel: Selection) -> list[str]:
    return [e.chatter_user_id for e in sel.chosen]


# --------------------------------------------------------------------------- #
# 1. dedupe near-dups
# --------------------------------------------------------------------------- #
def test_dedupe_exact_and_case_and_punct():
    """Exact, case-only, and punctuation-only variants collapse to the first."""
    events = [
        _ev("POG", uid="u1"),
        _ev("pog", uid="u2"),
        _ev("Pog!!!", uid="u3"),
        _ev("pog...", uid="u4"),
    ]
    sel = select_messages(events, max_messages=10, per_user_cap=10)
    assert len(sel.chosen) == 1
    assert sel.chosen[0].text == "POG"  # first occurrence kept
    assert sel.dropped == 3
    assert "duplicate" in sel.reason


def test_dedupe_elongation_near_dup():
    """SimHash/normalized-key folds character-elongated near-duplicates."""
    events = [
        _ev("that was so clean", uid="u1"),
        _ev("that was sooo clean!!!", uid="u2"),
        _ev("that was soooooo clean", uid="u3"),
    ]
    sel = select_messages(events, max_messages=10, per_user_cap=10)
    assert len(sel.chosen) == 1
    assert sel.chosen[0].text == "that was so clean"


def test_dedupe_keeps_distinct_messages():
    """Genuinely different messages are NOT folded together."""
    events = [
        _ev("rush B now", uid="u1"),
        _ev("hold A long", uid="u2"),
        _ev("save this round", uid="u3"),
    ]
    sel = select_messages(events, max_messages=10, per_user_cap=10)
    assert len(sel.chosen) == 3
    assert sel.dropped == 0


# --------------------------------------------------------------------------- #
# 2. per-user fairness cap
# --------------------------------------------------------------------------- #
def test_per_user_cap_one():
    """At most per_user_cap=1 distinct message per chatter survives."""
    events = [
        _ev("first thought", uid="spammer"),
        _ev("second thought", uid="spammer"),
        _ev("third thought", uid="spammer"),
        _ev("a different person", uid="other"),
    ]
    sel = select_messages(events, max_messages=10, per_user_cap=1)
    by_uid = _uids(sel)
    assert by_uid.count("spammer") == 1
    assert by_uid.count("other") == 1
    assert len(sel.chosen) == 2
    assert "per_user_cap" in sel.reason


def test_per_user_cap_two_allows_two():
    """A higher cap lets more through per user."""
    events = [
        _ev("alpha one", uid="u1"),
        _ev("bravo two", uid="u1"),
        _ev("charlie three", uid="u1"),
    ]
    sel = select_messages(events, max_messages=10, per_user_cap=2)
    assert _uids(sel).count("u1") == 2
    assert sel.dropped == 1


# --------------------------------------------------------------------------- #
# 3. global max cap
# --------------------------------------------------------------------------- #
def test_max_messages_cap():
    """Never returns more than max_messages."""
    events = [_ev(f"unique message number {i}", uid=f"u{i}") for i in range(20)]
    sel = select_messages(events, max_messages=6, per_user_cap=1)
    assert len(sel.chosen) == 6
    assert sel.dropped == 14
    assert "max_cap" in sel.reason


def test_max_messages_zero():
    """max_messages=0 yields an empty chosen list, all dropped."""
    events = [_ev("hello there", uid="u1"), _ev("general kenobi", uid="u2")]
    sel = select_messages(events, max_messages=0, per_user_cap=1)
    assert sel.chosen == []
    assert sel.dropped == 2


# --------------------------------------------------------------------------- #
# 4. priority ordering (mods/vips/subs first, then recency, then quality)
# --------------------------------------------------------------------------- #
def test_priority_mod_before_plain():
    """A moderator's message outranks a plain chatter's, regardless of order."""
    events = [
        _ev("plain chatter question here", uid="plain"),
        _ev("moderator question here", uid="mod", badges=[_badge("moderator")]),
    ]
    sel = select_messages(events, max_messages=10, per_user_cap=1)
    assert sel.chosen[0].chatter_user_id == "mod"


def test_priority_full_staff_order():
    """Order is moderator > vip > subscriber > plain."""
    events = [
        _ev("plain words here please", uid="plain"),
        _ev("subscriber words here please", uid="sub", badges=[_badge("subscriber")]),
        _ev("vip words here please", uid="vip", badges=[_badge("vip")]),
        _ev("mod words here please", uid="mod", badges=[_badge("moderator")]),
    ]
    sel = select_messages(events, max_messages=10, per_user_cap=1)
    assert _uids(sel) == ["mod", "vip", "sub", "plain"]


def test_priority_recency_within_same_rank():
    """Among equal-rank plain chatters, the more recent (later) message wins the top slot."""
    events = [
        _ev("older message from earlier", uid="old"),
        _ev("newer message from later", uid="new"),
    ]
    sel = select_messages(events, max_messages=10, per_user_cap=1)
    # Same badge rank -> later index is more recent -> ranked first.
    assert sel.chosen[0].chatter_user_id == "new"


def test_priority_mod_survives_global_cap():
    """When the cap forces a drop, the mod is retained over plain chatters."""
    events = [_ev(f"plain message number {i}", uid=f"p{i}") for i in range(6)]
    events.append(_ev("a moderator asking something", uid="themod", badges=[_badge("moderator")]))
    sel = select_messages(events, max_messages=1, per_user_cap=1)
    assert len(sel.chosen) == 1
    assert sel.chosen[0].chatter_user_id == "themod"


# --------------------------------------------------------------------------- #
# 5. recently_answered skip
# --------------------------------------------------------------------------- #
def test_recently_answered_skip():
    """Chatters in recently_answered are dropped entirely."""
    events = [
        _ev("hello from alice", uid="alice"),
        _ev("hello from bob", uid="bob"),
        _ev("hello from carol", uid="carol"),
    ]
    sel = select_messages(
        events, max_messages=10, per_user_cap=1, recently_answered={"alice", "carol"}
    )
    assert _uids(sel) == ["bob"]
    assert "recently_answered" in sel.reason


def test_recently_answered_default_empty():
    """Default recently_answered does not drop anyone."""
    events = [_ev("a question", uid="u1"), _ev("another question", uid="u2")]
    sel = select_messages(events, max_messages=10, per_user_cap=1)
    assert len(sel.chosen) == 2


def test_recently_answered_none_treated_as_empty():
    """Passing None for recently_answered is treated as the empty set (no skip)."""
    events = [_ev("hi there friend", uid="u1")]
    sel = select_messages(events, max_messages=10, per_user_cap=1, recently_answered=None)
    assert len(sel.chosen) == 1


# --------------------------------------------------------------------------- #
# 6. empty in -> empty out, and empty/blank message dropping
# --------------------------------------------------------------------------- #
def test_empty_batch_empty_out():
    sel = select_messages([], max_messages=6, per_user_cap=1)
    assert sel.chosen == []
    assert sel.dropped == 0
    assert isinstance(sel, Selection)


def test_blank_messages_dropped():
    """Whitespace-only / empty bodies are dropped before they consume a slot."""
    events = [
        _ev("", uid="u1"),
        _ev("   ", uid="u2"),
        _ev("\t\n", uid="u3"),
        _ev("real content here", uid="u4"),
    ]
    sel = select_messages(events, max_messages=10, per_user_cap=10)
    assert _texts(sel) == ["real content here"]
    assert sel.dropped == 3
    assert "empty" in sel.reason


# --------------------------------------------------------------------------- #
# 7. never raises on hostile / malformed input
# --------------------------------------------------------------------------- #
def test_never_raises_on_non_event_items():
    """Hostile non-ChatEvent items in the batch are dropped, never raise."""
    hostile = [
        None,
        "just a string",
        12345,
        {"text": "a dict not an event"},
        _ev("a legit message survives", uid="ok"),
    ]
    sel = select_messages(hostile, max_messages=10, per_user_cap=10)  # type: ignore[arg-type]
    assert isinstance(sel, Selection)
    assert _texts(sel) == ["a legit message survives"]


def test_never_raises_on_garbage_fields():
    """Events with wrong-typed text/badges/uid never raise; degrade gracefully."""
    bad1 = ChatEvent(
        broadcaster_user_id="b", chatter_user_id=None, chatter_login="x",  # type: ignore[arg-type]
        chatter_name="x", text=None,  # type: ignore[arg-type]
    )
    bad2 = ChatEvent(
        broadcaster_user_id="b", chatter_user_id=12,  # type: ignore[arg-type]
        chatter_login="y", chatter_name="y", text="valid text body",
        badges=["not-a-dict", {"no_set_id": 1}, {"set_id": 99}],  # type: ignore[list-item]
    )
    sel = select_messages([bad1, bad2], max_messages=10, per_user_cap=10)
    assert isinstance(sel, Selection)
    # bad1 has no usable text -> dropped; bad2 has text -> kept (badge garbage tolerated).
    assert any(e.text == "valid text body" for e in sel.chosen)


def test_never_raises_on_bad_kwargs():
    """Non-numeric caps / weird recently_answered fall back to sane defaults."""
    events = [_ev("a stable message", uid="u1")]
    sel = select_messages(
        events,
        max_messages="lots",  # type: ignore[arg-type]
        per_user_cap=None,    # type: ignore[arg-type]
        recently_answered=object(),  # type: ignore[arg-type]
    )
    assert isinstance(sel, Selection)
    # Defaults: max_messages->6, per_user_cap->1, recently_answered->empty.
    assert len(sel.chosen) == 1


def test_never_raises_on_non_iterable_events():
    """A non-iterable events argument degrades to an empty selection, no raise."""
    sel = select_messages(object(), max_messages=6, per_user_cap=1)  # type: ignore[arg-type]
    assert isinstance(sel, Selection)
    assert sel.chosen == []


# --------------------------------------------------------------------------- #
# 8. dedupe helper unit coverage
# --------------------------------------------------------------------------- #
def test_normalized_key_folds_variants():
    assert normalized_key("POG!!!") == normalized_key("pog")
    assert normalized_key("so   clean") == normalized_key("so clean")
    assert normalized_key("clap clap clap") != ""
    assert normalized_key("") == ""
    assert normalized_key(None) == ""
    assert normalized_key(12345) == ""


def test_simhash_near_and_far():
    """Near-identical text -> small Hamming distance; different -> large."""
    a = simhash("rush B with the team right now")
    b = simhash("rush B with the team right nowww")
    c = simhash("completely unrelated economy question about saving")
    assert (a ^ b).bit_count() <= 4
    assert (a ^ c).bit_count() > 4
    assert simhash("") == 0
    assert simhash(None) == 0


# --------------------------------------------------------------------------- #
# 9. id-less spam still capped (anon bucket)
# --------------------------------------------------------------------------- #
def test_idless_spam_capped_together():
    """Distinct messages from chatters with no id share one anon fairness bucket."""
    events = [
        _ev("anon line one is here", uid=""),
        _ev("anon line two is here", uid=""),
        _ev("anon line three is here", uid=""),
    ]
    sel = select_messages(events, max_messages=10, per_user_cap=1)
    # All distinct (no dedupe), but the anon bucket caps them to 1.
    assert len(sel.chosen) == 1


# --------------------------------------------------------------------------- #
# 10. dropped accounting integrity
# --------------------------------------------------------------------------- #
def test_dropped_accounting_consistent():
    """dropped == total_in - len(chosen) across a mixed batch."""
    events = [
        _ev("keep me one", uid="a"),
        _ev("keep me one", uid="b"),       # exact dup -> dropped
        _ev("", uid="c"),                  # empty -> dropped
        _ev("keep me two", uid="d"),
        _ev("keep me three", uid="e"),
    ]
    sel = select_messages(events, max_messages=2, per_user_cap=1)
    assert sel.dropped == len(events) - len(sel.chosen)
    assert len(sel.chosen) == 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
