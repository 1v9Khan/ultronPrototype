"""Tests for the STOP-window RELAY toggle (2026-07-08) -- the team-relay MASTER mode.

RELAY OFF = full disengage: a matched relay command is answered with the offline
notice on the LOCAL speakers (never transmitted, all four entry points funnel
through the one choke point in _maybe_handle_relay_speech); _listening_now()
goes False so the wake word is required again (beats always-listening AND
turbo); the speculative relay never builds a line; the bare-lead re-capture
stands down; conversation switches to ULTRON_COMPANION_PERSONA with a HARD
two-sentence cap enforced on the token stream (cap_stream_sentences).
RELAY ON (default) = today's behaviour, byte-identical.
Hermetic: no Tk, no sidecar, no LLM. Mirrors tests/audio/test_twitch_chat_toggle.py.
"""
import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from kenning.audio.relay_speech import (
    cap_stream_sentences,
    set_team_relay_enabled,
    team_relay_enabled,
)
from kenning.pipeline.orchestrator import Orchestrator


@pytest.fixture(autouse=True)
def _restore_team_relay_flag():
    """The flag is a process-global; never leak an OFF state into other tests."""
    before = team_relay_enabled()
    yield
    set_team_relay_enabled(before)


# ---------------------------------------------------------------------------
# relay_speech -- the module flag
# ---------------------------------------------------------------------------

def test_team_relay_defaults_on():
    assert team_relay_enabled() is True


def test_team_relay_set_and_get():
    set_team_relay_enabled(False)
    assert team_relay_enabled() is False
    set_team_relay_enabled(True)
    assert team_relay_enabled() is True


# ---------------------------------------------------------------------------
# StopButtonOverlay -- ctor wiring
# ---------------------------------------------------------------------------

def test_stop_button_accepts_relay_kwargs():
    from kenning.audio.stop_button import StopButtonOverlay
    ov = StopButtonOverlay(
        on_stop=lambda: None,
        on_toggle_relay=lambda v: None,
        relay_enabled=False,
        relay_height=30,
        relay_label="RELAY",
    )
    assert ov._on_toggle_relay is not None
    assert ov._relay_enabled is False
    assert ov._relay_h == 30
    assert ov._relay_label == "RELAY"


def test_stop_button_relay_default_on_row_hidden():
    from kenning.audio.stop_button import StopButtonOverlay
    ov = StopButtonOverlay(on_stop=lambda: None)
    # Display state defaults ON (today's behaviour); no callback -> row hidden.
    assert ov._relay_enabled is True
    assert ov._on_toggle_relay is None


def test_stop_button_relay_callback_stored():
    from kenning.audio.stop_button import StopButtonOverlay
    sentinel = []
    ov = StopButtonOverlay(
        on_stop=lambda: None,
        on_toggle_relay=lambda v: sentinel.append(v),
    )
    assert ov._on_toggle_relay is not None
    ov._on_toggle_relay(False)
    assert sentinel == [False]


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_relay_toggle_defaults():
    from kenning.config import StopButtonConfig
    sb = StopButtonConfig()
    assert sb.relay_height == 26
    assert sb.relay_label == "RELAY"


# ---------------------------------------------------------------------------
# Orchestrator -- setter flips the shared module flag
# ---------------------------------------------------------------------------

def test_orchestrator_relay_setter_flips_module_flag():
    o = Orchestrator.__new__(Orchestrator)
    o._set_team_relay_enabled(False)
    assert team_relay_enabled() is False
    o._set_team_relay_enabled(True)
    assert team_relay_enabled() is True


def test_stop_button_wired_with_relay_callback():
    src = inspect.getsource(Orchestrator.__init__)
    assert "on_toggle_relay=" in src
    assert "_set_team_relay_enabled" in src


# ---------------------------------------------------------------------------
# Relay suppression -- the single choke point in _maybe_handle_relay_speech
# ---------------------------------------------------------------------------

def _relay_cfg(**overrides: Any) -> SimpleNamespace:
    cfg = dict(
        enabled=True,
        output_device="Voicemeeter Aux Input",
        rephrase=False,
        max_line_chars=280,
        echo_to_user=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _bare_orchestrator() -> Any:
    o = Orchestrator.__new__(Orchestrator)
    o.llm = None
    o.tts = None
    o._spoken = []
    o._speak = lambda text: o._spoken.append(text)  # type: ignore[attr-defined]
    return o


def _patch_config(monkeypatch: pytest.MonkeyPatch, cfg: SimpleNamespace) -> None:
    import kenning.config as config_mod
    monkeypatch.setattr(
        config_mod, "get_config",
        lambda: SimpleNamespace(relay_speech=cfg),
    )


def test_relay_off_command_consumed_silently_not_transmitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matched relay command while the RELAY toggle is OFF is consumed
    SILENTLY (2026-07-08 live-test direction: no spoken notice, no response at
    all) and NEVER reaches the team bus."""
    import kenning.audio.relay_speech as relay_mod

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(
        _synthesize=lambda text: (np.ones(10, dtype=np.int16), 24000),
    )
    set_team_relay_enabled(False)
    _patch_config(monkeypatch, _relay_cfg())

    def must_not_play(pcm, sr, device, **kw):  # pragma: no cover
        raise AssertionError("relay-off must not transmit")

    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: 25)
    monkeypatch.setattr(relay_mod, "play_to_device", must_not_play)

    assert o._maybe_handle_relay_speech("tell my team to rotate B") is True
    assert o._spoken == []                       # fully silent


def test_relay_off_covers_force_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """force=True (the turbo backstop / semantic-router path) funnels through
    the same choke point -- a forced callout is suppressed silently too."""
    import kenning.audio.relay_speech as relay_mod

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(
        _synthesize=lambda text: (np.ones(10, dtype=np.int16), 24000),
    )
    set_team_relay_enabled(False)
    _patch_config(monkeypatch, _relay_cfg())

    def must_not_play(pcm, sr, device, **kw):  # pragma: no cover
        raise AssertionError("relay-off must not transmit (force=True)")

    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: 25)
    monkeypatch.setattr(relay_mod, "play_to_device", must_not_play)

    assert o._maybe_handle_relay_speech("sova hit 84", force=True) is True
    assert o._spoken == []                       # fully silent


def test_followup_relay_override_stands_down_when_relay_off():
    """2026-07-08 live test: 'Tell my team Silva hit 84' engaged WITHOUT a wake
    word through the follow-up relay override while relay was OFF. The override
    must consult the live flag so relay-off means wake-word-required, period."""
    src = inspect.getsource(Orchestrator.run)
    seg = src.split("wake_or_relay_override", 1)[0]
    gate = seg.rsplit("came_from_follow_up", 1)[1]
    assert "team_relay_enabled" in gate
    assert "_relay_override" in gate


def test_relay_off_ordinary_utterance_still_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate sits AFTER the matcher: ordinary speech is untouched and falls
    to the conversational path even while relay is off (no spurious notice)."""
    o = _bare_orchestrator()
    set_team_relay_enabled(False)
    _patch_config(monkeypatch, _relay_cfg())
    assert o._maybe_handle_relay_speech("what time is it") is False
    assert o._spoken == []


def test_relay_on_default_behaviour_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the toggle ON (default) the offline gate never fires -- the muted
    notice path (the narrower voice session mute) still behaves as before."""
    o = _bare_orchestrator()
    o.tts = SimpleNamespace(
        _synthesize=lambda text: (np.ones(10, dtype=np.int16), 24000),
    )
    o._relay_runtime_enabled = False  # the pre-existing voice session mute
    _patch_config(monkeypatch, _relay_cfg())
    assert o._maybe_handle_relay_speech("tell my team to rotate B") is True
    assert "muted" in o._spoken[0]


# ---------------------------------------------------------------------------
# Wake word required + speculative / bare-lead stand-down
# ---------------------------------------------------------------------------

def test_listening_now_checks_team_relay_flag():
    """_listening_now() (a run()-local closure) consults the live flag FIRST so
    relay-off forces wake-required within one iteration, beating both the
    boot-captured always_listening and the turbo override."""
    src = inspect.getsource(Orchestrator.run)
    gate = src.split("def _listening_now", 1)[1]
    head = gate.split("if _always_listening", 1)[0]
    assert "team_relay_enabled" in head
    assert "return False" in head


def test_speculative_relay_stands_down_when_relay_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    o = Orchestrator.__new__(Orchestrator)
    o.llm = object()
    o._speculative_relay_lock = threading.Lock()
    set_team_relay_enabled(False)
    _patch_config(monkeypatch, _relay_cfg())
    from kenning.audio.relay_speech import set_u1_llm_route_enabled, u1_llm_route_enabled
    before = u1_llm_route_enabled()
    set_u1_llm_route_enabled(True)
    try:
        assert o._run_speculative_relay("tell my team to rotate B") is False
    finally:
        set_u1_llm_route_enabled(before)


def test_bare_relay_lead_stands_down_when_relay_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With relay off there is no callout worth waiting for -- the bare-lead
    re-capture never triggers, so the loop does not open an extra capture."""
    o = _bare_orchestrator()
    _patch_config(monkeypatch, _relay_cfg())
    set_team_relay_enabled(False)
    assert o._is_bare_relay_lead("tell my team") is False


# ---------------------------------------------------------------------------
# Companion persona + the HARD two-sentence stream cap
# ---------------------------------------------------------------------------

def test_companion_persona_selected_when_relay_off():
    from kenning.audio.llm_prompts import ULTRON_COMPANION_PERSONA
    o = Orchestrator.__new__(Orchestrator)
    set_team_relay_enabled(False)
    # 2026-07-24: selector appends a per-turn variety suffix -> prefix match.
    got = o._gaming_conversational_prompt()
    assert got is not None and got.startswith(ULTRON_COMPANION_PERSONA)


def test_companion_persona_not_selected_when_relay_on():
    from kenning.audio.llm_prompts import ULTRON_COMPANION_PERSONA
    o = Orchestrator.__new__(Orchestrator)
    set_team_relay_enabled(True)
    got = o._gaming_conversational_prompt()
    assert got is None or not got.startswith(ULTRON_COMPANION_PERSONA)


def test_companion_persona_is_additive_on_gaming_persona():
    """User direction (2026-07-08): the companion persona is the CURRENT
    personality PLUS an enrichment block -- enhanced, never replaced. The
    full gaming persona must be present verbatim as the base."""
    from kenning.audio.llm_prompts import (
        _COMPANION_ENRICHMENT,
        ULTRON_COMPANION_PERSONA,
        ULTRON_GAMING_PERSONA,
    )
    assert ULTRON_COMPANION_PERSONA.startswith(ULTRON_GAMING_PERSONA)
    assert ULTRON_COMPANION_PERSONA == (
        ULTRON_GAMING_PERSONA + _COMPANION_ENRICHMENT
    )


def test_companion_enrichment_is_private_and_two_sentence():
    """The enrichment re-frames the moment (relay disengaged, private with the
    operator) and keeps the two-sentence limit; BR-P2 register holds (it never
    introduces 'assistant'/vendor naming beyond the base's negative rules)."""
    from kenning.audio.llm_prompts import _COMPANION_ENRICHMENT
    e = _COMPANION_ENRICHMENT
    assert "relay is disengaged" in e
    assert "PRIVATELY" in e
    assert "TWO short" in e
    assert "Kenning" not in e
    assert "assistant" not in e.lower()


def test_respond_wires_the_stream_cap():
    src = inspect.getsource(Orchestrator._respond)
    assert "cap_stream_sentences" in src
    assert "team_relay_enabled" in src


def test_cap_stream_sentences_caps_at_two():
    toks = ["Your aim ", "is adequate. ", "For flesh. ", "Do not ", "grow proud."]
    out = "".join(cap_stream_sentences(iter(toks), max_sentences=2))
    assert out.rstrip() == "Your aim is adequate. For flesh."


def test_cap_stream_sentences_passthrough_under_cap():
    out = "".join(cap_stream_sentences(iter(["One sentence only."]), 2))
    assert out == "One sentence only."


def test_cap_stream_sentences_cuts_mid_token_on_whole_sentence():
    out = "".join(cap_stream_sentences(iter(["Two here. And two. Third dies."]), 2))
    assert out == "Two here. And two."


def test_cap_stream_sentences_ignores_decimals():
    toks = ["The universe is 13.8 billion years old. Impressive? Hardly. No."]
    out = "".join(cap_stream_sentences(iter(toks), 2))
    assert out == "The universe is 13.8 billion years old. Impressive?"


def test_cap_stream_sentences_closes_inner_stream():
    closed = {"v": False}

    def tracked():
        try:
            yield "A. "
            yield "B. "
            yield "C. "
            yield "D."
        finally:
            closed["v"] = True

    _ = "".join(cap_stream_sentences(tracked(), 2))
    assert closed["v"] is True
