"""Tests for ``ultron.openclaw_bridge.events.OpenClawEventReceiver``.

Phase 3.4 ships only the prefix-matching scaffold; tests lock down
the contract so the future transport layer can ride on top without
rewriting the prefix logic.
"""

from __future__ import annotations

from typing import List

import pytest

from ultron.openclaw_bridge.events import (
    IncomingMessage,
    OpenClawEventReceiver,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_disabled_and_default_prefix() -> None:
    r = OpenClawEventReceiver()
    assert r.enabled is False
    assert r.prefix == "[voice]"


def test_rejects_empty_prefix() -> None:
    with pytest.raises(ValueError):
        OpenClawEventReceiver(prefix="")


# ---------------------------------------------------------------------------
# should_handle / extract_payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,expected",
    [
        ("[voice] hello", True),
        ("  [voice] hello", True),
        ("[voice]hello", True),                     # no space; still matches
        ("hello", False),
        ("[VOICE] hello", False),                    # case-sensitive
        ("voice: hello", False),
        ("", False),
    ],
)
def test_should_handle_default_prefix(body: str, expected: bool) -> None:
    assert OpenClawEventReceiver().should_handle(body) is expected


def test_should_handle_custom_prefix() -> None:
    r = OpenClawEventReceiver(prefix="!speak")
    assert r.should_handle("!speak hi") is True
    assert r.should_handle("[voice] hi") is False


def test_extract_payload_strips_prefix_and_whitespace() -> None:
    r = OpenClawEventReceiver()
    assert r.extract_payload("[voice]   hello there") == "hello there"
    assert r.extract_payload("[voice]hello") == "hello"
    # Non-matching body returns unchanged.
    assert r.extract_payload("just hello") == "just hello"


def test_should_handle_non_string_returns_false() -> None:
    assert OpenClawEventReceiver().should_handle(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_start_is_noop_when_disabled() -> None:
    r = OpenClawEventReceiver(enabled=False)
    await r.start()
    assert r.started is False


async def test_start_flips_started_when_enabled() -> None:
    r = OpenClawEventReceiver(enabled=True)
    await r.start()
    assert r.started is True
    await r.stop()
    assert r.started is False


async def test_start_idempotent() -> None:
    r = OpenClawEventReceiver(enabled=True)
    await r.start()
    await r.start()                                  # second call is a no-op
    assert r.started is True
    await r.stop()


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


async def test_dispatch_drops_when_disabled() -> None:
    seen: List[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        seen.append(msg)

    r = OpenClawEventReceiver(
        prefix="[voice]", on_voice_handoff=handler, enabled=False,
    )
    await r.start()
    msg = IncomingMessage(channel="telegram", sender="@me", body="[voice] hi")
    handled = await r.dispatch(msg)
    assert handled is False
    assert seen == []


async def test_dispatch_drops_when_prefix_misses() -> None:
    seen: List[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        seen.append(msg)

    r = OpenClawEventReceiver(
        prefix="[voice]", on_voice_handoff=handler, enabled=True,
    )
    await r.start()
    msg = IncomingMessage(channel="telegram", sender="@me", body="hi")
    handled = await r.dispatch(msg)
    assert handled is False
    assert seen == []
    await r.stop()


async def test_dispatch_invokes_handler_on_match() -> None:
    seen: List[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        seen.append(msg)

    r = OpenClawEventReceiver(
        prefix="[voice]", on_voice_handoff=handler, enabled=True,
    )
    await r.start()
    msg = IncomingMessage(channel="telegram", sender="@me", body="[voice] hello")
    handled = await r.dispatch(msg)
    assert handled is True
    assert len(seen) == 1
    delivered = seen[0]
    assert delivered.body == "hello"                  # prefix stripped
    assert delivered.prefix_match is True
    assert delivered.channel == "telegram"
    await r.stop()


async def test_dispatch_swallows_handler_exception() -> None:
    async def handler(_msg: IncomingMessage) -> None:
        raise RuntimeError("oops")

    r = OpenClawEventReceiver(
        prefix="[voice]", on_voice_handoff=handler, enabled=True,
    )
    await r.start()
    msg = IncomingMessage(channel="telegram", sender="@me", body="[voice] hi")
    handled = await r.dispatch(msg)
    # Handler raised; dispatch should NOT propagate.
    assert handled is False
    await r.stop()


async def test_dispatch_returns_false_when_no_handler() -> None:
    r = OpenClawEventReceiver(
        prefix="[voice]", on_voice_handoff=None, enabled=True,
    )
    await r.start()
    msg = IncomingMessage(channel="telegram", sender="@me", body="[voice] hi")
    handled = await r.dispatch(msg)
    assert handled is False
    await r.stop()
