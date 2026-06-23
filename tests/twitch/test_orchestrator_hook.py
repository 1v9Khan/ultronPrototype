"""The orchestrator chat-mode hook is a flag-gated, fail-open no-op when OFF."""
from __future__ import annotations


def test_start_twitch_chat_mode_is_noop_when_disabled(tmp_path) -> None:
    """With twitch.enabled=False the hook must NOT construct a service, NOT
    start a thread, and NOT raise -> the voice/relay runtime is byte-identical.
    Uses a minimal tmp config so the test is independent of the operator's live
    config.yaml (which may have twitch.enabled=True for live testing)."""
    from kenning.config import TwitchConfig, load_config
    from kenning.pipeline.orchestrator import Orchestrator

    # Schema default is OFF.
    assert TwitchConfig().enabled is False

    # Patch get_config inside the method so it sees a disabled twitch block.
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text('version: "1.0"\n', encoding="utf-8")
    disabled_cfg = load_config(cfg_path)
    assert disabled_cfg.twitch.enabled is False

    from kenning.config import get_config, set_config
    orig_cfg = get_config()
    try:
        set_config(disabled_cfg)
        orch = Orchestrator.__new__(Orchestrator)
        orch._start_twitch_chat_mode()               # must be a clean no-op
        assert getattr(orch, "_twitch_chat_service", None) is None
    finally:
        set_config(orig_cfg)


def test_start_twitch_chat_mode_method_exists_and_is_callable() -> None:
    import inspect

    from kenning.pipeline.orchestrator import Orchestrator
    assert callable(getattr(Orchestrator, "_start_twitch_chat_mode", None))
    # called from run() exactly once (right after audio.start()).
    src = inspect.getsource(Orchestrator.run)
    assert "_start_twitch_chat_mode()" in src
