"""Tests for the STOP-window CHAT toggle (2026-06-23).

Covers: StopButtonOverlay ctor stores the chat wiring; config defaults;
orchestrator has _set_twitch_chat_reply_enabled; the stop-button construction
wires the CHAT callback; the chat-mode loop reads the runtime attr.
Hermetic: no Tk, no sidecar, no LLM.
"""
import inspect


# ---------------------------------------------------------------------------
# StopButtonOverlay -- ctor wiring
# ---------------------------------------------------------------------------

def test_stop_button_accepts_chat_kwargs():
    from kenning.audio.stop_button import StopButtonOverlay
    calls = []
    ov = StopButtonOverlay(
        on_stop=lambda: None,
        on_toggle_chat=lambda v: calls.append(v),
        chat_enabled=True,
        chat_height=26,
        chat_label="CHAT",
    )
    assert ov._on_toggle_chat is not None
    assert ov._chat_enabled is True
    assert ov._chat_h == 26
    assert ov._chat_label == "CHAT"


def test_stop_button_chat_default_off():
    from kenning.audio.stop_button import StopButtonOverlay
    ov = StopButtonOverlay(on_stop=lambda: None)
    assert ov._chat_enabled is False
    assert ov._on_toggle_chat is None


def test_stop_button_chat_callback_stored():
    from kenning.audio.stop_button import StopButtonOverlay
    sentinel = []
    ov = StopButtonOverlay(
        on_stop=lambda: None,
        on_toggle_chat=lambda v: sentinel.append(v),
        chat_enabled=False,
    )
    # Manually invoke the stored callback (the Tk button would call it).
    assert ov._on_toggle_chat is not None
    ov._on_toggle_chat(True)
    assert sentinel == [True]


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_chat_toggle_defaults():
    from kenning.config import StopButtonConfig
    sb = StopButtonConfig()
    assert sb.chat_height == 26
    assert sb.chat_label == "CHAT"


# ---------------------------------------------------------------------------
# Orchestrator -- setter method
# ---------------------------------------------------------------------------

def test_orchestrator_has_chat_reply_setter():
    from kenning.pipeline import orchestrator as orch
    assert hasattr(orch.Orchestrator, "_set_twitch_chat_reply_enabled")
    src = inspect.getsource(orch.Orchestrator._set_twitch_chat_reply_enabled)
    assert "_twitch_chat_reply_enabled" in src


def test_chat_reply_setter_updates_attr():
    from kenning.pipeline.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o._twitch_chat_reply_enabled = False
    o._set_twitch_chat_reply_enabled(True)
    assert o._twitch_chat_reply_enabled is True
    o._set_twitch_chat_reply_enabled(False)
    assert o._twitch_chat_reply_enabled is False


# ---------------------------------------------------------------------------
# Orchestrator -- stop-button wiring source check
# ---------------------------------------------------------------------------

def test_stop_button_wired_with_chat_callback():
    from kenning.pipeline import orchestrator as orch
    src = inspect.getsource(orch.Orchestrator.__init__)
    assert "on_toggle_chat=" in src
    assert "_set_twitch_chat_reply_enabled" in src


# ---------------------------------------------------------------------------
# Chat-mode loop -- runtime attr takes precedence
# ---------------------------------------------------------------------------

def test_chat_loop_reads_runtime_attr():
    from kenning.pipeline import orchestrator as orch
    src = inspect.getsource(orch.Orchestrator._start_twitch_chat_mode)
    # The loop calls getattr(self, "_twitch_chat_reply_enabled", ...) so the
    # GUI toggle takes effect within one tick without a config reload.
    assert '"_twitch_chat_reply_enabled"' in src
    assert "getattr(" in src
