"""OpenClawDispatcher stub-response tests."""

from __future__ import annotations

import asyncio

import pytest

from ultron.openclaw_routing import (
    OpenClawDispatcher,
    classify_routing,
)
from ultron.openclaw_routing.intents import (
    BrowserIntent,
    DispatchResult,
    FileOpIntent,
    MediaGenIntent,
    MessagingIntent,
    ShellOpIntent,
)


@pytest.fixture
def dispatcher():
    return OpenClawDispatcher()


def _run(coro):
    return asyncio.run(coro)


def test_handle_browser_returns_stub(dispatcher):
    intent = BrowserIntent(action="navigate", url="https://example.com")
    result = _run(dispatcher.handle_browser(intent))
    assert isinstance(result, DispatchResult)
    assert result.success is False
    assert "page" in result.voice_message.lower()
    assert "gateway" in result.voice_message.lower()
    assert result.metadata["stub"] is True
    assert result.metadata["capability"] == "browser_automation"
    assert "not yet integrated" in result.error


def test_handle_media_generation_returns_stub(dispatcher):
    intent = MediaGenIntent(medium="image", description="cat in a top hat")
    result = _run(dispatcher.handle_media_generation(intent))
    assert result.success is False
    assert "generate" in result.voice_message.lower()
    assert result.metadata["stub"] is True
    assert result.metadata["capability"] == "media_generation"


def test_handle_messaging_returns_stub(dispatcher):
    intent = MessagingIntent(channel="telegram", body="build done")
    result = _run(dispatcher.handle_messaging(intent))
    assert result.success is False
    assert "send" in result.voice_message.lower()
    assert result.metadata["stub"] is True
    assert result.metadata["capability"] == "messaging"


def test_handle_file_operation_returns_stub(dispatcher):
    intent = FileOpIntent(operation="read", path="/etc/hosts")
    result = _run(dispatcher.handle_file_operation(intent))
    assert result.success is False
    assert "files" in result.voice_message.lower()
    assert result.metadata["stub"] is True
    assert result.metadata["capability"] == "file_operations"


def test_handle_shell_operation_returns_stub(dispatcher):
    intent = ShellOpIntent(command="dir")
    result = _run(dispatcher.handle_shell_operation(intent))
    assert result.success is False
    assert "shell" in result.voice_message.lower()
    assert result.metadata["stub"] is True
    assert result.metadata["capability"] == "shell_operations"


def test_voice_messages_in_ultron_voice(dispatcher):
    """Spot-check that the stub messages avoid filler and apologetic
    phrasing per Ultron's system prompt rules."""
    bad_phrases = [
        "certainly", "of course", "happy to", "i'm so sorry",
        "i would be happy", "i'd love to",
    ]
    intents = [
        BrowserIntent(action="navigate"),
        MediaGenIntent(medium="image", description=""),
        MessagingIntent(channel="telegram", body=""),
        FileOpIntent(operation="read", path="/x"),
        ShellOpIntent(command="ls"),
    ]
    methods = [
        dispatcher.handle_browser, dispatcher.handle_media_generation,
        dispatcher.handle_messaging, dispatcher.handle_file_operation,
        dispatcher.handle_shell_operation,
    ]
    for fn, intent in zip(methods, intents):
        result = _run(fn(intent))
        msg_lower = result.voice_message.lower()
        for bad in bad_phrases:
            assert bad not in msg_lower, (
                f"voice message has banned phrase {bad!r}: {result.voice_message!r}"
            )


def test_dispatcher_reads_config_at_construction():
    """Dispatcher caches openclaw.enabled and stub_responses_enabled at
    construction time. Tests that change config afterward don't affect
    an already-built dispatcher (which is fine — operator changes config
    and restarts)."""
    d1 = OpenClawDispatcher()
    assert d1.enabled is False  # config default
    assert isinstance(d1.stub_responses_enabled, bool)


# ---------------------------------------------------------------------------
# End-to-end through classify_routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utt,expected_capability", [
    ("open hacker news", "browser_automation"),
    ("make me an image of a cat", "media_generation"),
    ("send a message to my phone", "messaging"),
    ("read the file at C:/x.txt", "file_operations"),
    ("run dir on the desktop", "shell_operations"),
])
def test_classify_then_dispatch_round_trip(dispatcher, utt, expected_capability):
    """Classify an utterance, dispatch the resulting automation_intent,
    expect a stub for the right capability."""
    intent = classify_routing(utt)
    auto = intent.automation_intent
    assert auto is not None, f"automation_intent missing for {utt!r}"

    if isinstance(auto, BrowserIntent):
        result = _run(dispatcher.handle_browser(auto))
    elif isinstance(auto, MediaGenIntent):
        result = _run(dispatcher.handle_media_generation(auto))
    elif isinstance(auto, MessagingIntent):
        result = _run(dispatcher.handle_messaging(auto))
    elif isinstance(auto, FileOpIntent):
        result = _run(dispatcher.handle_file_operation(auto))
    elif isinstance(auto, ShellOpIntent):
        result = _run(dispatcher.handle_shell_operation(auto))
    else:
        pytest.fail(f"unknown automation_intent type: {type(auto).__name__}")

    assert result.success is False
    assert result.metadata["capability"] == expected_capability


# ---------------------------------------------------------------------------
# Phase 4 — handle_messaging through a wired OpenClaw bridge
# ---------------------------------------------------------------------------


from types import SimpleNamespace
from unittest.mock import AsyncMock

from ultron.openclaw_bridge.client import SendMessageResult


def _bridge_with_send(send_result: SendMessageResult) -> SimpleNamespace:
    """Build a minimal bridge stand-in: the dispatcher only touches
    ``bridge.client.send_message``."""
    client = SimpleNamespace(
        send_message=AsyncMock(return_value=send_result),
    )
    return SimpleNamespace(client=client)


def test_messaging_dispatches_via_bridge_when_wired(monkeypatch):
    """When a bridge is wired, handle_messaging calls send_message
    instead of returning the stub."""
    monkeypatch.setenv("TELEGRAM_USER_ID", "12345")
    bridge = _bridge_with_send(SendMessageResult(
        delivered=True, channel="telegram", target="12345",
        message_id="msg-1",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = MessagingIntent(channel="telegram", body="hello", raw_text="send me hello")
    result = _run(d.handle_messaging(intent))
    assert result.success is True
    assert result.metadata["stub"] is False
    assert result.metadata["channel"] == "telegram"
    assert result.metadata["message_id"] == "msg-1"
    bridge.client.send_message.assert_awaited_once_with(
        "telegram", "12345", "hello",
    )


def test_messaging_falls_back_to_stub_without_bridge():
    """No bridge → stub voice message, exactly as before Phase 4."""
    d = OpenClawDispatcher()                              # no bridge
    intent = MessagingIntent(channel="telegram", body="hi")
    result = _run(d.handle_messaging(intent))
    assert result.success is False
    assert result.metadata["stub"] is True
    assert "gateway isn't connected" in result.voice_message.lower()


def test_messaging_handles_missing_recipient(monkeypatch):
    """No env var, no fallback, no intent.recipient → clear voice
    message, no send attempt."""
    monkeypatch.delenv("TELEGRAM_USER_ID", raising=False)
    bridge = _bridge_with_send(SendMessageResult(
        delivered=True, channel="telegram", target="anything",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = MessagingIntent(channel="telegram", body="hi", raw_text="send hi")
    result = _run(d.handle_messaging(intent))
    assert result.success is False
    assert "TELEGRAM_USER_ID" in result.voice_message
    bridge.client.send_message.assert_not_called()


def test_messaging_uses_explicit_recipient_when_present(monkeypatch):
    """An intent.recipient takes precedence over env-based fallback."""
    monkeypatch.setenv("TELEGRAM_USER_ID", "fallback")
    bridge = _bridge_with_send(SendMessageResult(
        delivered=True, channel="telegram", target="explicit",
        message_id="x",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = MessagingIntent(
        channel="telegram", body="hi", recipient="explicit",
        raw_text="send hi",
    )
    result = _run(d.handle_messaging(intent))
    assert result.success is True
    bridge.client.send_message.assert_awaited_once_with(
        "telegram", "explicit", "hi",
    )


def test_messaging_handles_send_failure(monkeypatch):
    """Bridge send returned delivered=False — voice message reflects
    the underlying error."""
    monkeypatch.setenv("TELEGRAM_USER_ID", "12345")
    bridge = _bridge_with_send(SendMessageResult(
        delivered=False, channel="telegram", target="12345",
        error="rate limited",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = MessagingIntent(channel="telegram", body="hi", raw_text="send hi")
    result = _run(d.handle_messaging(intent))
    assert result.success is False
    assert "rate limited" in result.voice_message.lower()


def test_messaging_handles_transport_exception(monkeypatch):
    """Client raises mid-call — caught and translated to voice message."""
    monkeypatch.setenv("TELEGRAM_USER_ID", "12345")
    client = SimpleNamespace(
        send_message=AsyncMock(side_effect=RuntimeError("boom")),
    )
    bridge = SimpleNamespace(client=client)
    d = OpenClawDispatcher(bridge=bridge)
    intent = MessagingIntent(channel="telegram", body="hi", raw_text="send hi")
    result = _run(d.handle_messaging(intent))
    assert result.success is False
    assert "couldn't reach the gateway" in result.voice_message.lower()


def test_messaging_rejects_empty_body(monkeypatch):
    """Empty body → clear voice error, no send attempt."""
    monkeypatch.setenv("TELEGRAM_USER_ID", "12345")
    bridge = _bridge_with_send(SendMessageResult(
        delivered=True, channel="telegram", target="12345",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = MessagingIntent(channel="telegram", body="   ")
    result = _run(d.handle_messaging(intent))
    assert result.success is False
    assert "body" in result.voice_message.lower()
    bridge.client.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 6 — handle_browser through a wired OpenClaw bridge
# ---------------------------------------------------------------------------


from ultron.openclaw_bridge.client import ToolInvocationResult


def _bridge_with_invoke(tool_result: ToolInvocationResult) -> SimpleNamespace:
    client = SimpleNamespace(
        invoke_tool=AsyncMock(return_value=tool_result),
    )
    return SimpleNamespace(client=client)


def test_browser_navigate_via_bridge():
    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="browser",
        text="Title: Example\nLoaded.",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = BrowserIntent(
        action="navigate", url="https://example.com",
        raw_text="open example.com",
    )
    result = _run(d.handle_browser(intent))
    assert result.success is True
    assert result.metadata["stub"] is False
    assert "example" in result.voice_message.lower()
    bridge.client.invoke_tool.assert_awaited_once()


def test_browser_falls_back_when_no_bridge():
    d = OpenClawDispatcher()                              # no bridge
    intent = BrowserIntent(action="navigate", url="https://x.com")
    result = _run(d.handle_browser(intent))
    assert result.success is False
    assert result.metadata["stub"] is True
    assert "gateway" in result.voice_message.lower()


def test_browser_falls_back_when_disabled():
    """Master `browser.enabled: false` → stub even with bridge."""
    from ultron.config import get_config

    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="browser", text="anything",
    ))
    cfg = get_config()
    cfg.browser.enabled = False
    try:
        d = OpenClawDispatcher(config=cfg, bridge=bridge)
        intent = BrowserIntent(action="navigate", url="https://x.com")
        result = _run(d.handle_browser(intent))
        assert result.success is False
        assert result.metadata["stub"] is True
        bridge.client.invoke_tool.assert_not_called()
    finally:
        cfg.browser.enabled = True


def test_browser_navigate_rejects_missing_url():
    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="browser", text="ok",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = BrowserIntent(action="navigate", url="")
    result = _run(d.handle_browser(intent))
    assert result.success is False
    assert "url" in result.voice_message.lower()
    bridge.client.invoke_tool.assert_not_called()


def test_browser_screenshot_via_bridge():
    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="browser",
        text="Saved to /tmp/shot.png",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = BrowserIntent(action="screenshot", raw_text="capture screen")
    result = _run(d.handle_browser(intent))
    assert result.success is True
    assert "screenshot" in result.voice_message.lower()


def test_browser_click_requires_target():
    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="browser", text="clicked",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = BrowserIntent(action="click", target=None)
    result = _run(d.handle_browser(intent))
    assert result.success is False
    bridge.client.invoke_tool.assert_not_called()


def test_browser_unknown_action_clear_voice_message():
    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="browser", text="ok",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = BrowserIntent(action="explode", url="https://x.com")
    result = _run(d.handle_browser(intent))
    assert result.success is False
    assert "explode" in result.voice_message.lower() or "don't know" in result.voice_message.lower()


def test_browser_dispatch_handles_exception():
    """Wrapper raises mid-call — translated to voice message."""
    client = SimpleNamespace(
        invoke_tool=AsyncMock(side_effect=RuntimeError("boom")),
    )
    bridge = SimpleNamespace(client=client)
    d = OpenClawDispatcher(bridge=bridge)
    intent = BrowserIntent(action="navigate", url="https://x.com")
    result = _run(d.handle_browser(intent))
    assert result.success is False
    assert "wrong" in result.voice_message.lower() or "moment" in result.voice_message.lower()


# ---------------------------------------------------------------------------
# Phase 12 — handle_media_generation through a wired OpenClaw bridge
# ---------------------------------------------------------------------------


def test_media_image_dispatches_via_bridge():
    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="image_generate",
        text="Saved to /tmp/cat.png",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = MediaGenIntent(
        medium="image", description="a cat in a top hat",
        raw_text="make me an image of a cat",
    )
    result = _run(d.handle_media_generation(intent))
    assert result.success is True
    assert result.metadata["stub"] is False
    assert result.metadata["medium"] == "image"
    assert "image" in result.voice_message.lower()
    bridge.client.invoke_tool.assert_awaited_once()
    args, _ = bridge.client.invoke_tool.call_args
    assert args[0] == "image_generate"
    assert "cat" in args[1]["prompt"]


def test_media_falls_back_to_stub_without_bridge():
    d = OpenClawDispatcher()                                 # no bridge
    intent = MediaGenIntent(medium="image", description="x")
    result = _run(d.handle_media_generation(intent))
    assert result.success is False
    assert result.metadata["stub"] is True


def test_media_falls_back_when_disabled():
    from ultron.config import get_config

    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="image_generate", text="ok",
    ))
    cfg = get_config()
    cfg.media_generation.enabled = False
    try:
        d = OpenClawDispatcher(config=cfg, bridge=bridge)
        intent = MediaGenIntent(medium="image", description="x")
        result = _run(d.handle_media_generation(intent))
        assert result.success is False
        assert result.metadata["stub"] is True
        bridge.client.invoke_tool.assert_not_called()
    finally:
        cfg.media_generation.enabled = True


def test_media_rejects_missing_description():
    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="image_generate", text="ok",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = MediaGenIntent(medium="image", description="")
    result = _run(d.handle_media_generation(intent))
    assert result.success is False
    assert "description" in result.voice_message.lower()
    bridge.client.invoke_tool.assert_not_called()


def test_media_unknown_medium():
    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="x", text="ok",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = MediaGenIntent(medium="hologram", description="x")
    result = _run(d.handle_media_generation(intent))
    assert result.success is False
    assert "hologram" in result.voice_message.lower() or "don't know" in result.voice_message.lower()


def test_media_video_uses_video_tool():
    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="video_generate",
        text="Saved.",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = MediaGenIntent(medium="video", description="a short clip")
    result = _run(d.handle_media_generation(intent))
    assert result.success is True
    args, _ = bridge.client.invoke_tool.call_args
    assert args[0] == "video_generate"


def test_media_music_alias():
    """'music' and 'audio' both route to music_generate."""
    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="music_generate", text="ok",
    ))
    d = OpenClawDispatcher(bridge=bridge)
    intent = MediaGenIntent(medium="audio", description="ambient track")
    result = _run(d.handle_media_generation(intent))
    assert result.success is True
    assert result.metadata["medium"] == "music"
    args, _ = bridge.client.invoke_tool.call_args
    assert args[0] == "music_generate"


def test_media_provider_override_in_config():
    from ultron.config import get_config

    bridge = _bridge_with_invoke(ToolInvocationResult(
        success=True, tool_name="image_generate", text="ok",
    ))
    cfg = get_config()
    # ComfyUI is the canonical local-only provider for media generation
    # in Ultron's stack — Claude Code is the only paid service.
    cfg.media_generation.default_image_provider = "comfyui"
    try:
        d = OpenClawDispatcher(config=cfg, bridge=bridge)
        intent = MediaGenIntent(medium="image", description="hi")
        result = _run(d.handle_media_generation(intent))
        assert result.success is True
        args, _ = bridge.client.invoke_tool.call_args
        assert args[1]["provider"] == "comfyui"
    finally:
        cfg.media_generation.default_image_provider = None


def test_media_dispatch_handles_exception():
    client = SimpleNamespace(
        invoke_tool=AsyncMock(side_effect=RuntimeError("boom")),
    )
    bridge = SimpleNamespace(client=client)
    d = OpenClawDispatcher(bridge=bridge)
    intent = MediaGenIntent(medium="image", description="x")
    result = _run(d.handle_media_generation(intent))
    assert result.success is False
    assert "wrong" in result.voice_message.lower() or "moment" in result.voice_message.lower()
