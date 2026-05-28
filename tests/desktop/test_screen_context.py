"""Tests for ultron.desktop.screen_context."""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock

import pytest

from ultron.desktop.capture import Screenshot
from ultron.desktop.monitors import Monitor
from ultron.desktop.screen_context import (
    ScreenContextCache,
    ScreenContextSnapshot,
    build_screen_context,
    capture_and_cache,
    get_screen_context_cache,
    get_vlm_describe,
    set_screen_context_cache,
    set_vlm_describe,
)
from ultron.desktop.windows import WindowInfo


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _mon(idx=0, primary=True) -> Monitor:
    return Monitor(
        index=idx, name=f"D{idx}",
        x=0, y=0, width=1920, height=1080,
        work_x=0, work_y=0, work_width=1920, work_height=1040,
        is_primary=primary,
    )


def _win(title="Some App", proc="some.exe", mon=0, fg=False, hwnd=1) -> WindowInfo:
    return WindowInfo(
        hwnd=hwnd, title=title, class_name="C",
        process_name=proc, pid=0,
        rect=(0, 0, 800, 600),
        monitor_index=mon, is_minimized=False, is_foreground=fg,
    )


def _shot() -> Screenshot:
    return Screenshot(
        image_bytes=b"\x89PNG_DUMMY", monitor_index=0,
        width=1920, height=1080, timestamp=0.0,
        origin_x=0, origin_y=0,
    )


# ---------------------------------------------------------------------------
# ScreenContextSnapshot.render_for_llm
# ---------------------------------------------------------------------------


def test_render_for_llm_no_foreground():
    snap = ScreenContextSnapshot(
        timestamp=0.0, monitors=(), foreground=None,
        windows=(), ui_text=(), screenshot=None,
        vlm_description=None, elapsed_ms=0.0,
    )
    out = snap.render_for_llm()
    assert "Visual context" in out
    assert "No window is currently focused" in out
    assert "End visual context" in out


def test_render_for_llm_with_foreground_and_ui_text():
    snap = ScreenContextSnapshot(
        timestamp=0.0,
        monitors=(_mon(), _mon(idx=1, primary=False)),
        foreground=_win(title="main.py - Cursor", proc="Cursor.exe", mon=0, fg=True),
        windows=(_win(title="Chrome", proc="chrome.exe", mon=1),),
        ui_text=("File", "Edit", "View", "def hello():\n    pass"),
        screenshot=_shot(),
        vlm_description=None,
        elapsed_ms=42.0,
    )
    out = snap.render_for_llm()
    assert "Cursor.exe" in out
    assert "main.py - Cursor" in out
    assert "def hello" in out
    assert "Other visible apps: chrome.exe" in out


def test_render_for_llm_with_vlm_description():
    snap = ScreenContextSnapshot(
        timestamp=0.0, monitors=(), foreground=None, windows=(),
        ui_text=(),
        screenshot=_shot(),
        vlm_description="A code editor showing a Python function definition.",
        elapsed_ms=0.0,
    )
    out = snap.render_for_llm()
    assert "Visual description" in out
    assert "Python function definition" in out


def test_render_for_llm_truncates_long_ui_text():
    long = "x" * 500
    snap = ScreenContextSnapshot(
        timestamp=0.0, monitors=(), foreground=None, windows=(),
        ui_text=(long,),
        screenshot=None,
        vlm_description=None,
        elapsed_ms=0.0,
    )
    out = snap.render_for_llm()
    # Truncated to 197 + "..."
    assert "..." in out
    assert long not in out


def test_render_for_llm_caps_ui_text_count():
    items = tuple(f"item_{i}" for i in range(100))
    snap = ScreenContextSnapshot(
        timestamp=0.0, monitors=(), foreground=None, windows=(),
        ui_text=items, screenshot=None, vlm_description=None, elapsed_ms=0.0,
    )
    out = snap.render_for_llm(max_ui_text=5)
    # First 5 are present, 6th isn't.
    for i in range(5):
        assert f"item_{i}" in out
    assert "item_5" not in out


# ---------------------------------------------------------------------------
# build_screen_context with all components mocked out
# ---------------------------------------------------------------------------


def test_build_screen_context_no_capture_no_uia(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows",
        lambda: [_win(title="Chrome", proc="chrome.exe", hwnd=2)],
    )
    snap = build_screen_context(capture=False, include_uia=False)
    assert snap.monitors == (_mon(),)
    assert snap.foreground is not None
    assert snap.foreground.is_foreground
    assert snap.screenshot is None
    assert snap.ui_text == ()
    assert snap.vlm_description is None


def test_build_screen_context_with_capture(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(fg=True, mon=0),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.collect_window_text",
        lambda *a, **kw: ["UI text 1", "UI text 2"],
    )
    fake_cap = MagicMock()
    fake_cap.capture_monitor.return_value = _shot()
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_screen_capture", lambda: fake_cap,
    )
    snap = build_screen_context(capture=True, include_uia=True)
    assert snap.screenshot is not None
    assert snap.ui_text == ("UI text 1", "UI text 2")
    fake_cap.capture_monitor.assert_called_once_with(0)


def test_build_screen_context_handles_uia_failure(monkeypatch):
    """collect_window_text raising mustn't abort the whole snapshot."""
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )

    def boom(*a, **kw):
        raise RuntimeError("simulated UIA failure")

    monkeypatch.setattr(
        "ultron.desktop.screen_context.collect_window_text", boom,
    )
    snap = build_screen_context(capture=False, include_uia=True)
    # ui_text is empty but snapshot still assembled.
    assert snap.ui_text == ()
    assert snap.foreground is not None


def test_build_screen_context_handles_capture_failure(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(fg=True, mon=0),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    fake_cap = MagicMock()
    fake_cap.capture_monitor.side_effect = RuntimeError("simulated capture failure")
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_screen_capture", lambda: fake_cap,
    )
    snap = build_screen_context(capture=True, include_uia=False)
    assert snap.screenshot is None


def test_build_screen_context_handles_window_enum_failure(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window", lambda: None,
    )

    def boom():
        raise RuntimeError("simulated enum failure")

    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", boom,
    )
    snap = build_screen_context(capture=False, include_uia=False)
    assert snap.windows == ()
    assert snap.foreground is None


def test_build_screen_context_caps_window_list(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window", lambda: None,
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows",
        lambda: [_win(hwnd=i, title=f"app_{i}", proc=f"app_{i}.exe") for i in range(50)],
    )
    snap = build_screen_context(capture=False, include_uia=False, window_list_cap=5)
    assert len(snap.windows) == 5


# ---------------------------------------------------------------------------
# VLM hook
# ---------------------------------------------------------------------------


def test_vlm_hook_starts_unset():
    set_vlm_describe(None)
    assert get_vlm_describe() is None


def test_vlm_hook_can_be_set_and_cleared():
    set_vlm_describe(None)
    try:
        def fake(img_bytes: bytes) -> str:
            return "test description"
        set_vlm_describe(fake)
        assert get_vlm_describe() is fake
        set_vlm_describe(None)
        assert get_vlm_describe() is None
    finally:
        set_vlm_describe(None)


def test_build_screen_context_includes_vlm_when_enabled(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    fake_cap = MagicMock()
    fake_cap.capture_monitor.return_value = _shot()
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_screen_capture", lambda: fake_cap,
    )

    set_vlm_describe(lambda img: "Test VLM description")
    try:
        snap = build_screen_context(
            capture=True, include_uia=False, include_vlm=True,
        )
        assert snap.vlm_description == "Test VLM description"
    finally:
        set_vlm_describe(None)


def test_build_screen_context_vlm_disabled_by_default(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    fake_cap = MagicMock()
    fake_cap.capture_monitor.return_value = _shot()
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_screen_capture", lambda: fake_cap,
    )
    vlm_calls = []

    def vlm(img):
        vlm_calls.append(img)
        return "should not be called"

    set_vlm_describe(vlm)
    try:
        snap = build_screen_context(capture=True, include_vlm=False)
        assert snap.vlm_description is None
        assert vlm_calls == []
    finally:
        set_vlm_describe(None)


# ---------------------------------------------------------------------------
# Analyze-and-discard (Phase 12)
# ---------------------------------------------------------------------------


def test_screenshot_without_bytes_clears_bytes_and_flags(monkeypatch):
    shot = Screenshot(
        image_bytes=b"\x89PNG_BIG_PAYLOAD",
        monitor_index=0, width=1920, height=1080,
        timestamp=42.0, origin_x=0, origin_y=0,
    )
    stripped = shot.without_bytes()
    assert stripped.image_bytes is None
    assert stripped.bytes_discarded is True
    # Metadata preserved.
    assert stripped.width == 1920
    assert stripped.monitor_index == 0
    assert stripped.timestamp == 42.0
    # Original unchanged (frozen dataclass).
    assert shot.image_bytes == b"\x89PNG_BIG_PAYLOAD"


def test_screenshot_without_bytes_is_idempotent():
    shot = Screenshot(
        image_bytes=None, monitor_index=0,
        width=10, height=10, timestamp=0.0,
        origin_x=0, origin_y=0, bytes_discarded=True,
    )
    again = shot.without_bytes()
    assert again is shot  # short-circuit


def test_build_screen_context_discards_bytes_after_vlm_by_default(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(fg=True, mon=0),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    fake_cap = MagicMock()
    fake_cap.capture_monitor.return_value = _shot()
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_screen_capture", lambda: fake_cap,
    )

    set_vlm_describe(lambda img: "A code editor.")
    try:
        snap = build_screen_context(
            capture=True, include_uia=False, include_vlm=True,
        )
        # VLM ran -> bytes discarded.
        assert snap.vlm_description == "A code editor."
        assert snap.screenshot is not None
        assert snap.screenshot.image_bytes is None
        assert snap.screenshot.bytes_discarded is True
        # Metadata still there.
        assert snap.screenshot.width > 0
    finally:
        set_vlm_describe(None)


def test_build_screen_context_keeps_bytes_when_discard_disabled(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    fake_cap = MagicMock()
    fake_cap.capture_monitor.return_value = _shot()
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_screen_capture", lambda: fake_cap,
    )

    set_vlm_describe(lambda img: "A code editor.")
    try:
        snap = build_screen_context(
            capture=True, include_uia=False, include_vlm=True,
            discard_image_after_analysis=False,
        )
        # Bytes preserved.
        assert snap.screenshot is not None
        assert snap.screenshot.image_bytes is not None
        assert snap.screenshot.bytes_discarded is False
    finally:
        set_vlm_describe(None)


def test_build_screen_context_no_vlm_keeps_bytes_even_with_discard_on(monkeypatch):
    """When the VLM didn't run, there's no description to fall back on
    -- bytes should be retained so the caller can run their own analysis.
    """
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    fake_cap = MagicMock()
    fake_cap.capture_monitor.return_value = _shot()
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_screen_capture", lambda: fake_cap,
    )

    set_vlm_describe(None)
    snap = build_screen_context(
        capture=True, include_vlm=False,
        discard_image_after_analysis=True,  # on, but VLM didn't run
    )
    assert snap.screenshot is not None
    assert snap.screenshot.image_bytes is not None
    assert snap.screenshot.bytes_discarded is False


def test_build_screen_context_vlm_failure_keeps_bytes(monkeypatch):
    """VLM call that returns None means no analysis was captured --
    don't discard bytes (the caller may retry or fall back).
    """
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    fake_cap = MagicMock()
    fake_cap.capture_monitor.return_value = _shot()
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_screen_capture", lambda: fake_cap,
    )

    set_vlm_describe(lambda img: None)  # VLM returns no text
    try:
        snap = build_screen_context(
            capture=True, include_vlm=True,
        )
        # No description -> bytes preserved.
        assert snap.vlm_description is None
        assert snap.screenshot is not None
        assert snap.screenshot.image_bytes is not None
    finally:
        set_vlm_describe(None)


def test_cache_strips_bytes_by_default():
    """The cache should never retain raw pixels by default."""
    cache = ScreenContextCache(ring_size=3)
    snap = ScreenContextSnapshot(
        timestamp=time.time(), monitors=(),
        foreground=None, windows=(), ui_text=(),
        screenshot=Screenshot(
            image_bytes=b"\x89PNG_LIVE",
            monitor_index=0, width=10, height=10,
            timestamp=0.0, origin_x=0, origin_y=0,
        ),
        vlm_description="something",
        elapsed_ms=10.0,
    )
    cache.store(snap)
    cached = cache.latest()
    assert cached is not None
    assert cached.screenshot is not None
    assert cached.screenshot.image_bytes is None
    assert cached.screenshot.bytes_discarded is True


def test_cache_keeps_bytes_when_discard_disabled():
    cache = ScreenContextCache(ring_size=3, discard_image_bytes=False)
    snap = ScreenContextSnapshot(
        timestamp=time.time(), monitors=(),
        foreground=None, windows=(), ui_text=(),
        screenshot=Screenshot(
            image_bytes=b"\x89PNG_PRESERVED",
            monitor_index=0, width=10, height=10,
            timestamp=0.0, origin_x=0, origin_y=0,
        ),
        vlm_description="x", elapsed_ms=0.0,
    )
    cache.store(snap)
    cached = cache.latest()
    assert cached is not None
    assert cached.screenshot is not None
    assert cached.screenshot.image_bytes == b"\x89PNG_PRESERVED"


def test_cache_handles_already_discarded_snapshot():
    """Storing a snapshot whose bytes are already discarded should not crash
    and should not re-allocate.
    """
    cache = ScreenContextCache(ring_size=3)
    snap = ScreenContextSnapshot(
        timestamp=time.time(), monitors=(),
        foreground=None, windows=(), ui_text=(),
        screenshot=Screenshot(
            image_bytes=None, monitor_index=0,
            width=10, height=10, timestamp=0.0,
            origin_x=0, origin_y=0, bytes_discarded=True,
        ),
        vlm_description="x", elapsed_ms=0.0,
    )
    cache.store(snap)
    cached = cache.latest()
    assert cached is not None
    assert cached.screenshot.bytes_discarded is True


def test_cache_handles_none_screenshot():
    """Snapshot without a screenshot at all stores cleanly."""
    cache = ScreenContextCache(ring_size=3)
    snap = ScreenContextSnapshot(
        timestamp=time.time(), monitors=(),
        foreground=None, windows=(), ui_text=(),
        screenshot=None, vlm_description=None, elapsed_ms=0.0,
    )
    cache.store(snap)
    assert cache.latest() is snap  # exact same object (no rebuild needed)


def test_build_screen_context_vlm_exception_handled(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    fake_cap = MagicMock()
    fake_cap.capture_monitor.return_value = _shot()
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_screen_capture", lambda: fake_cap,
    )

    def boom(img):
        raise RuntimeError("simulated VLM failure")

    set_vlm_describe(boom)
    try:
        snap = build_screen_context(capture=True, include_vlm=True)
        assert snap.vlm_description is None  # fail-open
    finally:
        set_vlm_describe(None)


# ---------------------------------------------------------------------------
# ScreenContextCache
# ---------------------------------------------------------------------------


def test_cache_stores_and_retrieves():
    cache = ScreenContextCache(ring_size=3)
    snap = ScreenContextSnapshot(
        timestamp=time.time(), monitors=(), foreground=None,
        windows=(), ui_text=(), screenshot=None,
        vlm_description=None, elapsed_ms=0.0,
    )
    cache.store(snap)
    assert cache.size == 1
    assert cache.latest() is snap


def test_cache_evicts_old_entries():
    cache = ScreenContextCache(ring_size=2)
    for i in range(5):
        cache.store(ScreenContextSnapshot(
            timestamp=float(i), monitors=(), foreground=None,
            windows=(), ui_text=(), screenshot=None,
            vlm_description=None, elapsed_ms=0.0,
        ))
    assert cache.size == 2
    # Newest first
    snaps = cache.all()
    assert snaps[-1].timestamp == 4.0


def test_cache_latest_fresh_respects_max_age():
    cache = ScreenContextCache(ring_size=3, max_age_seconds=0.05)
    old = ScreenContextSnapshot(
        timestamp=time.time() - 1.0,  # 1 s ago
        monitors=(), foreground=None, windows=(), ui_text=(),
        screenshot=None, vlm_description=None, elapsed_ms=0.0,
    )
    cache.store(old)
    assert cache.latest() is old  # latest ignores age
    assert cache.latest_fresh() is None  # but latest_fresh respects max_age


def test_cache_clear():
    cache = ScreenContextCache()
    cache.store(ScreenContextSnapshot(
        timestamp=time.time(), monitors=(), foreground=None,
        windows=(), ui_text=(), screenshot=None,
        vlm_description=None, elapsed_ms=0.0,
    ))
    cache.clear()
    assert cache.size == 0
    assert cache.latest() is None


def test_singleton_cache_swap():
    set_screen_context_cache(None)
    try:
        a = get_screen_context_cache()
        b = get_screen_context_cache()
        assert a is b
        custom = ScreenContextCache()
        set_screen_context_cache(custom)
        assert get_screen_context_cache() is custom
    finally:
        set_screen_context_cache(None)


# ---------------------------------------------------------------------------
# capture_and_cache convenience
# ---------------------------------------------------------------------------


def test_capture_and_cache_stores_snapshot(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window", lambda: None,
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    set_screen_context_cache(ScreenContextCache())
    try:
        snap = capture_and_cache(capture=False, include_uia=False)
        cache = get_screen_context_cache()
        assert cache.size == 1
        assert cache.latest() is snap
    finally:
        set_screen_context_cache(None)


# ---------------------------------------------------------------------------
# Live integration (Windows only)
# ---------------------------------------------------------------------------


pytestmark_windows = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows-only (full screen-context assembly)",
)


@pytestmark_windows
def test_build_screen_context_live_succeeds():
    snap = build_screen_context(capture=True, include_uia=True, include_vlm=False)
    assert isinstance(snap, ScreenContextSnapshot)
    assert snap.elapsed_ms > 0
    # On a live desktop session there should be at least one monitor.
    assert len(snap.monitors) >= 1


@pytestmark_windows
def test_render_for_llm_live_produces_readable_output():
    snap = build_screen_context(capture=False, include_uia=True, include_vlm=False)
    out = snap.render_for_llm()
    assert "Visual context" in out
    assert "End visual context" in out


# ---------------------------------------------------------------------------
# Catalog 09 wiring: extract_browser_content into screen_context
# ---------------------------------------------------------------------------


def _browser_content(*, page_title="GitHub", headings=(), text=(), buttons=(),
                     links=(), inputs=(), images=(), truncated=False):
    """Build a minimal BrowserContent for test injection."""
    from ultron.desktop.uia import BrowserContent
    return BrowserContent(
        page_title=page_title,
        browser_name="chrome",
        headings=tuple(headings),
        text=tuple(text),
        buttons=tuple(buttons),
        links=tuple(links),
        inputs=tuple(inputs),
        images=tuple(images),
        truncated=truncated,
        elapsed_ms=12,
    )


def test_browser_foreground_uses_extract_browser_content(monkeypatch):
    """When the foreground is a browser, extract_browser_content
    feeds ui_text instead of collect_window_text."""
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(title="GitHub - Chrome", proc="chrome.exe", mon=0, fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    # Force collect_window_text to a sentinel so we'd notice if it ran.
    collect_called = []

    def _sentinel(*a, **kw):
        collect_called.append(True)
        return ["should not appear"]

    monkeypatch.setattr(
        "ultron.desktop.screen_context.collect_window_text", _sentinel,
    )
    # Browser detection returns True.
    monkeypatch.setattr(
        "ultron.desktop.uia.is_browser_window", lambda title: True,
    )
    # extract_browser_content returns content.
    fake_content = _browser_content(
        page_title="Repo Home",
        headings=("Overview", "Issues"),
        text=("Welcome to the repo",),
        buttons=("Code", "Star"),
        links=(),
        inputs=(),
    )
    monkeypatch.setattr(
        "ultron.desktop.uia.extract_browser_content",
        lambda win, **kw: fake_content,
    )

    snap = build_screen_context(capture=False, include_uia=True)
    # collect_window_text must NOT have been called.
    assert collect_called == []
    # ui_text composition: title, headings, text, buttons.
    assert snap.ui_text[0] == "Repo Home"
    assert "Overview" in snap.ui_text
    assert "Welcome to the repo" in snap.ui_text
    assert any(s.startswith("button: ") for s in snap.ui_text)


def test_browser_foreground_falls_back_when_extract_returns_none(monkeypatch):
    """If extract_browser_content returns None, fall back to
    collect_window_text so we still get *some* ui_text."""
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(title="Mozilla Firefox", proc="firefox.exe", mon=0, fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.collect_window_text",
        lambda *a, **kw: ["fallback ui text"],
    )
    monkeypatch.setattr(
        "ultron.desktop.uia.is_browser_window", lambda title: True,
    )
    monkeypatch.setattr(
        "ultron.desktop.uia.extract_browser_content",
        lambda win, **kw: None,
    )
    snap = build_screen_context(capture=False, include_uia=True)
    assert snap.ui_text == ("fallback ui text",)


def test_browser_foreground_falls_back_when_extract_raises(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(title="Brave", proc="brave.exe", mon=0, fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.collect_window_text",
        lambda *a, **kw: ["safe fallback"],
    )
    monkeypatch.setattr(
        "ultron.desktop.uia.is_browser_window", lambda title: True,
    )

    def _boom(*a, **kw):
        raise RuntimeError("uia tree broken")

    monkeypatch.setattr(
        "ultron.desktop.uia.extract_browser_content", _boom,
    )
    snap = build_screen_context(capture=False, include_uia=True)
    assert snap.ui_text == ("safe fallback",)


def test_non_browser_foreground_uses_collect_window_text(monkeypatch):
    """Non-browser foreground still uses the legacy collect_window_text
    -- existing behaviour preserved."""
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(title="Visual Studio Code", proc="code.exe", mon=0, fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.collect_window_text",
        lambda *a, **kw: ["legacy UIA path", "still works"],
    )
    # extract_browser_content must not be called.
    def _should_not_be_called(*a, **kw):
        raise AssertionError("extract_browser_content called for non-browser")

    monkeypatch.setattr(
        "ultron.desktop.uia.extract_browser_content", _should_not_be_called,
    )
    snap = build_screen_context(capture=False, include_uia=True)
    assert snap.ui_text == ("legacy UIA path", "still works")


def test_browser_links_with_urls_are_rendered(monkeypatch):
    from ultron.desktop.uia import BrowserLink

    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(title="Chrome", proc="chrome.exe", mon=0, fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    monkeypatch.setattr(
        "ultron.desktop.uia.is_browser_window", lambda title: True,
    )
    content = _browser_content(
        page_title="links",
        links=(
            BrowserLink(name="GitHub", url="https://github.com",
                        center=(0, 0), enabled=True),
            BrowserLink(name="No URL", url=None,
                        center=(0, 0), enabled=True),
        ),
    )
    monkeypatch.setattr(
        "ultron.desktop.uia.extract_browser_content",
        lambda win, **kw: content,
    )
    snap = build_screen_context(capture=False, include_uia=True)
    assert any("link: GitHub -> https://github.com" in s for s in snap.ui_text)
    assert any(s == "link: No URL" for s in snap.ui_text)


def test_browser_inputs_with_values_are_rendered(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_monitors", lambda: [_mon()],
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.get_foreground_window",
        lambda: _win(title="Edge", proc="msedge.exe", mon=0, fg=True),
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.enumerate_windows", lambda: [],
    )
    monkeypatch.setattr(
        "ultron.desktop.uia.is_browser_window", lambda title: True,
    )
    content = _browser_content(
        page_title="login",
        inputs=(("Email", "user@x.com"), ("Password", "")),
    )
    monkeypatch.setattr(
        "ultron.desktop.uia.extract_browser_content",
        lambda win, **kw: content,
    )
    snap = build_screen_context(capture=False, include_uia=True)
    assert "input: Email: user@x.com" in snap.ui_text
    assert "input: Password" in snap.ui_text


# ---------------------------------------------------------------------------
# Catalog 10 batch 9 -- browser-use CDP fallback tier
# ---------------------------------------------------------------------------


class _FakeBuState:
    def __init__(self, *, success=True, url="https://x.com", title="X", elements=()):
        self.success = success
        self.url = url
        self.title = title
        self.elements = elements


class _FakeBuElement:
    def __init__(self, index, label, type_="button"):
        self.index = index
        self.label = label
        self.type = type_


class _FakeBuTool:
    def __init__(self, *, available=True, state=None):
        self._available = available
        self._state = state if state is not None else _FakeBuState()

    def is_available(self):
        return self._available

    def state(self):
        return self._state


def _set_bu_fallback_enabled(monkeypatch, enabled: bool) -> None:
    import ultron.desktop.screen_context as sc

    class _Cfg:
        class browser_use:
            screen_context_fallback_enabled = enabled

    monkeypatch.setattr(sc, "get_config", lambda: _Cfg, raising=False)
    # screen_context imports get_config lazily inside the helper, so
    # patch the source module too.
    import ultron.config as cfgmod

    monkeypatch.setattr(cfgmod, "get_config", lambda: _Cfg)


class TestBrowserUseFallbackHelper:
    def test_disabled_returns_empty(self, monkeypatch):
        from ultron.desktop import screen_context as sc
        import ultron.desktop.browser_use as bu

        _set_bu_fallback_enabled(monkeypatch, False)
        monkeypatch.setattr(bu, "get_browser_use_tool", lambda: _FakeBuTool())
        assert sc._maybe_browser_use_state_text(40) == ()

    def test_no_tool_returns_empty(self, monkeypatch):
        from ultron.desktop import screen_context as sc
        import ultron.desktop.browser_use as bu

        _set_bu_fallback_enabled(monkeypatch, True)
        monkeypatch.setattr(bu, "get_browser_use_tool", lambda: None)
        assert sc._maybe_browser_use_state_text(40) == ()

    def test_unavailable_tool_returns_empty(self, monkeypatch):
        from ultron.desktop import screen_context as sc
        import ultron.desktop.browser_use as bu

        _set_bu_fallback_enabled(monkeypatch, True)
        monkeypatch.setattr(
            bu, "get_browser_use_tool",
            lambda: _FakeBuTool(available=False),
        )
        assert sc._maybe_browser_use_state_text(40) == ()

    def test_empty_url_returns_empty(self, monkeypatch):
        from ultron.desktop import screen_context as sc
        import ultron.desktop.browser_use as bu

        _set_bu_fallback_enabled(monkeypatch, True)
        tool = _FakeBuTool(state=_FakeBuState(url=""))
        monkeypatch.setattr(bu, "get_browser_use_tool", lambda: tool)
        assert sc._maybe_browser_use_state_text(40) == ()

    def test_failed_state_returns_empty(self, monkeypatch):
        from ultron.desktop import screen_context as sc
        import ultron.desktop.browser_use as bu

        _set_bu_fallback_enabled(monkeypatch, True)
        tool = _FakeBuTool(state=_FakeBuState(success=False))
        monkeypatch.setattr(bu, "get_browser_use_tool", lambda: tool)
        assert sc._maybe_browser_use_state_text(40) == ()

    def test_success_folds_labelled_content(self, monkeypatch):
        from ultron.desktop import screen_context as sc
        import ultron.desktop.browser_use as bu

        _set_bu_fallback_enabled(monkeypatch, True)
        state = _FakeBuState(
            url="https://example.com",
            title="Example",
            elements=(
                _FakeBuElement(0, "Sign in", "button"),
                _FakeBuElement(1, "", "input"),  # empty label skipped
            ),
        )
        monkeypatch.setattr(
            bu, "get_browser_use_tool", lambda: _FakeBuTool(state=state)
        )
        result = sc._maybe_browser_use_state_text(40)
        assert any("browser-use page title: Example" in s for s in result)
        assert any("browser-use url: https://example.com" in s for s in result)
        assert any("Sign in" in s for s in result)
        # Every line carries the browser-use prefix so it's distinguishable.
        assert all(s.startswith("browser-use") for s in result)

    def test_caps_at_max_elements(self, monkeypatch):
        from ultron.desktop import screen_context as sc
        import ultron.desktop.browser_use as bu

        _set_bu_fallback_enabled(monkeypatch, True)
        many = tuple(_FakeBuElement(i, f"el{i}") for i in range(100))
        monkeypatch.setattr(
            bu, "get_browser_use_tool",
            lambda: _FakeBuTool(state=_FakeBuState(elements=many)),
        )
        result = sc._maybe_browser_use_state_text(10)
        assert len(result) <= 10

    def test_tool_exception_fails_open(self, monkeypatch):
        from ultron.desktop import screen_context as sc
        import ultron.desktop.browser_use as bu

        _set_bu_fallback_enabled(monkeypatch, True)

        class _Boom:
            def is_available(self):
                return True

            def state(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(bu, "get_browser_use_tool", lambda: _Boom())
        assert sc._maybe_browser_use_state_text(40) == ()
