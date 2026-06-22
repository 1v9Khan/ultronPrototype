"""The orchestrator chat-mode hook is a flag-gated, fail-open no-op when OFF."""
from __future__ import annotations


def test_start_twitch_chat_mode_is_noop_when_disabled() -> None:
    """With twitch.enabled=False (the default), the hook must NOT construct a
    service, NOT start a thread, and NOT raise -> the voice/relay runtime is
    byte-identical. (Structurally, the kenning.twitch.service import lives AFTER
    the disabled-return, so OFF imports nothing from it.)"""
    from kenning.config import get_config
    from kenning.pipeline.orchestrator import Orchestrator

    assert get_config().twitch.enabled is False  # default-OFF
    orch = Orchestrator.__new__(Orchestrator)    # bare instance, no heavy init
    orch._start_twitch_chat_mode()               # must be a clean no-op
    assert getattr(orch, "_twitch_chat_service", None) is None


def test_start_twitch_chat_mode_method_exists_and_is_callable() -> None:
    import inspect

    from kenning.pipeline.orchestrator import Orchestrator
    assert callable(getattr(Orchestrator, "_start_twitch_chat_mode", None))
    # called from run() exactly once (right after audio.start()).
    src = inspect.getsource(Orchestrator.run)
    assert "_start_twitch_chat_mode()" in src
