"""S7 — structural team-mic isolation: provenance guard + static import-graph wall.

Team isolation must be a TESTED code-capability boundary, not prose. Two layers:
  1. the provenance guard refuses any non-LOCAL_VOICE source at the relay boundary;
  2. a static scan proves NO kenning.twitch module references the relay/PTT/HID/
     team-mic surface — the chat path cannot even NAME the team path.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from kenning.audio.provenance import (
    Provenance,
    TeamIsolationViolation,
    assert_team_eligible,
    is_team_eligible,
    relay_allowed,
)


def test_only_local_voice_is_team_eligible() -> None:
    assert is_team_eligible(Provenance.LOCAL_VOICE)
    for p in (Provenance.TWITCH_CHAT, Provenance.REDEEM, Provenance.SYSTEM):
        assert not is_team_eligible(p)
    # unknown string -> fail-closed (not eligible)
    assert not is_team_eligible("totally_new_source")


def test_assert_team_eligible_raises_for_non_local() -> None:
    assert_team_eligible(Provenance.LOCAL_VOICE)  # no raise
    for p in (Provenance.TWITCH_CHAT, Provenance.REDEEM, Provenance.SYSTEM):
        with pytest.raises(TeamIsolationViolation):
            assert_team_eligible(p, where="relay")


def test_relay_allowed_full_precondition() -> None:
    # local voice + relay armed + chat-mode OFF -> allowed
    assert relay_allowed(Provenance.LOCAL_VOICE, relay_runtime_enabled=True, chat_mode_active=False)
    # any unmet condition -> refused
    assert not relay_allowed(Provenance.LOCAL_VOICE, relay_runtime_enabled=False, chat_mode_active=False)
    assert not relay_allowed(Provenance.LOCAL_VOICE, relay_runtime_enabled=True, chat_mode_active=True)
    assert not relay_allowed(Provenance.TWITCH_CHAT, relay_runtime_enabled=True, chat_mode_active=False)


# --- the static import-graph wall --------------------------------------------
# No kenning.twitch module may reference the relay/PTT/HID/team-mic surface. The
# ONE future exception (the speak-to-team redeem, S13) lives in the trusted relay
# process, NOT under kenning.twitch — so this scan stays clean.
_FORBIDDEN = re.compile(
    r"_maybe_handle_relay_speech"
    r"|build_relay_line"
    r"|\bpress_key\b|\bpress_hotkey\b|SendInput"
    r"|from\s+kenning\.ptt|import\s+kenning\.ptt|kenning\.ptt\b"
    r"|kenning\.desktop"
    r"|_ptt_hold|_ptt_runtime_enabled"
    r"|relay_runtime_enabled"
)

_TWITCH_SRC = Path(__file__).resolve().parents[2] / "src" / "kenning" / "twitch"


def test_no_twitch_module_references_the_team_path() -> None:
    offenders: list[str] = []
    for py in _TWITCH_SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        m = _FORBIDDEN.search(text)
        if m:
            offenders.append(f"{py.relative_to(_TWITCH_SRC.parents[2])}: {m.group(0)!r}")
    assert not offenders, (
        "kenning.twitch must NEVER reference the relay/PTT/team-mic surface "
        "(team-isolation breakout risk): " + "; ".join(offenders)
    )


def test_orchestrator_relay_refuses_non_local_provenance() -> None:
    """The relay boundary (_maybe_handle_relay_speech) must REFUSE a non-LOCAL_VOICE
    provenance before doing any relay work — even if a future bug routed chat there."""
    import inspect

    from kenning.pipeline.orchestrator import Orchestrator

    # the provenance kwarg is wired into the signature
    sig = inspect.signature(Orchestrator._maybe_handle_relay_speech)
    assert "provenance" in sig.parameters

    class _FakeSelf:
        _relay_runtime_enabled = True

    # A chat-sourced relay attempt is refused at the guard (returns False early).
    refused = Orchestrator._maybe_handle_relay_speech(
        _FakeSelf(), "tell my team to rush B", provenance=Provenance.TWITCH_CHAT,
    )
    assert refused is False
    refused2 = Orchestrator._maybe_handle_relay_speech(
        _FakeSelf(), "push now", provenance=Provenance.REDEEM,
    )
    assert refused2 is False
