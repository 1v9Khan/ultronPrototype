"""Tests for the desktop-automation tools in mcp_tools.py (Phase 7).

These cover the impl functions (not the FastMCP registration shim).
Each test mocks the underlying ultron.desktop primitives to avoid
spawning processes, capturing real screens, or loading the VLM.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ultron.openclaw_bridge.mcp_tools import (
    click_uia_impl,
    clipboard_read_impl,
    clipboard_write_impl,
    describe_screen_impl,
    enumerate_monitors_impl,
    find_image_on_screen_impl,
    focus_window_impl,
    get_screen_context_impl,
    get_window_text_impl,
    launch_app_impl,
    launch_chrome_url_impl,
    list_windows_impl,
    mouse_click_impl,
    mouse_move_impl,
    move_window_to_monitor_impl,
    open_image_search_impl,
    press_hotkey_impl,
    scroll_impl,
    take_screenshot_impl,
    type_into_uia_impl,
    type_text_impl,
    window_action_impl,
)


# ---------------------------------------------------------------------------
# enumerate_monitors_impl
# ---------------------------------------------------------------------------


def test_enumerate_monitors_returns_count_and_list(monkeypatch):
    from ultron.desktop.monitors import Monitor

    fakes = [
        Monitor(
            index=0, name="DISPLAY1",
            x=0, y=0, width=1920, height=1080,
            work_x=0, work_y=0, work_width=1920, work_height=1040,
            is_primary=True,
        ),
        Monitor(
            index=1, name="DISPLAY2",
            x=1920, y=0, width=1920, height=1080,
            work_x=1920, work_y=0, work_width=1920, work_height=1040,
            is_primary=False,
        ),
    ]
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors", lambda: fakes,
    )
    out = enumerate_monitors_impl()
    assert out["count"] == 2
    assert out["monitors"][0]["is_primary"] is True
    assert out["monitors"][1]["index"] == 1
    assert out["monitors"][0]["width"] == 1920


def test_enumerate_monitors_returns_empty_on_import_failure(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fail_on_monitors(name, *args, **kwargs):
        if name == "ultron.desktop.monitors":
            raise ImportError("simulated import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_on_monitors)
    out = enumerate_monitors_impl()
    assert out["count"] == 0
    assert "error" in out


# ---------------------------------------------------------------------------
# list_windows_impl
# ---------------------------------------------------------------------------


def test_list_windows_caps_and_serialises(monkeypatch):
    from ultron.desktop.windows import WindowInfo

    wins = [
        WindowInfo(
            hwnd=i, title=f"window {i}", class_name="C",
            process_name=f"proc{i}.exe", pid=i,
            rect=(0, 0, 100, 100),
            monitor_index=0, is_minimized=False, is_foreground=(i == 0),
        )
        for i in range(50)
    ]
    monkeypatch.setattr(
        "ultron.desktop.windows.enumerate_windows",
        lambda **kw: wins,
    )
    out = list_windows_impl(limit=5)
    assert out["count"] == 5
    assert out["windows"][0]["title"] == "window 0"
    assert out["windows"][0]["is_foreground"] is True
    assert "rect" in out["windows"][0]


def test_list_windows_no_limit_returns_all(monkeypatch):
    from ultron.desktop.windows import WindowInfo

    wins = [
        WindowInfo(
            hwnd=i, title=f"w{i}", class_name="", process_name="",
            pid=0, rect=(0, 0, 1, 1), monitor_index=0,
            is_minimized=False, is_foreground=False,
        )
        for i in range(3)
    ]
    monkeypatch.setattr(
        "ultron.desktop.windows.enumerate_windows",
        lambda **kw: wins,
    )
    out = list_windows_impl(limit=0)
    assert out["count"] == 3


# ---------------------------------------------------------------------------
# take_screenshot_impl
# ---------------------------------------------------------------------------


def _patch_capture_pipeline(monkeypatch, *, has_fg=True, capture_ok=True):
    """Common monkey-patching for the capture pipeline."""
    from ultron.desktop.capture import Screenshot
    from ultron.desktop.monitors import Monitor
    from ultron.desktop.windows import WindowInfo

    fake_mon = Monitor(
        index=0, name="D0", x=0, y=0, width=1920, height=1080,
        work_x=0, work_y=0, work_width=1920, work_height=1040,
        is_primary=True,
    )
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors", lambda: [fake_mon],
    )
    fake_fg = (
        WindowInfo(
            hwnd=1, title="t", class_name="c", process_name="p",
            pid=0, rect=(0, 0, 100, 100), monitor_index=0,
            is_minimized=False, is_foreground=True,
        )
        if has_fg else None
    )
    monkeypatch.setattr(
        "ultron.desktop.windows.get_foreground_window", lambda: fake_fg,
    )
    fake_cap = MagicMock()
    if capture_ok:
        fake_cap.capture_monitor.return_value = Screenshot(
            image_bytes=b"\x89PNG_FAKE",
            monitor_index=0, width=1920, height=1080,
            timestamp=0.0, origin_x=0, origin_y=0,
        )
    else:
        fake_cap.capture_monitor.return_value = None
    monkeypatch.setattr(
        "ultron.desktop.capture.get_screen_capture", lambda: fake_cap,
    )


def test_take_screenshot_returns_base64(monkeypatch):
    _patch_capture_pipeline(monkeypatch)
    out = take_screenshot_impl(monitor_index=0, include_image=True)
    assert out["success"] is True
    assert out["monitor_index"] == 0
    assert out["width"] == 1920
    assert "image_base64" in out
    # Base64 of "\x89PNG_FAKE" -> "iYlOR19GQUtF"
    import base64
    assert base64.b64decode(out["image_base64"]) == b"\x89PNG_FAKE"


def test_take_screenshot_omits_image_when_disabled(monkeypatch):
    _patch_capture_pipeline(monkeypatch)
    out = take_screenshot_impl(monitor_index=0, include_image=False)
    assert out["success"] is True
    assert "image_base64" not in out


def test_take_screenshot_defaults_to_foreground_monitor(monkeypatch):
    _patch_capture_pipeline(monkeypatch, has_fg=True)
    out = take_screenshot_impl(monitor_index=None, include_image=False)
    assert out["success"] is True
    assert out["monitor_index"] == 0


def test_take_screenshot_no_monitors_returns_failure(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors", lambda: [],
    )
    monkeypatch.setattr(
        "ultron.desktop.windows.get_foreground_window", lambda: None,
    )
    out = take_screenshot_impl(monitor_index=0)
    assert out["success"] is False
    assert "no monitors" in (out.get("error") or "")


def test_take_screenshot_out_of_range_returns_failure(monkeypatch):
    _patch_capture_pipeline(monkeypatch)
    out = take_screenshot_impl(monitor_index=99)
    assert out["success"] is False
    assert "out of range" in (out.get("error") or "")


def test_take_screenshot_capture_fails_returns_failure(monkeypatch):
    _patch_capture_pipeline(monkeypatch, capture_ok=False)
    out = take_screenshot_impl(monitor_index=0)
    assert out["success"] is False
    assert "capture failed" in (out.get("error") or "")


def test_take_screenshot_with_description_vlm_unset(monkeypatch):
    _patch_capture_pipeline(monkeypatch)
    monkeypatch.setattr(
        "ultron.desktop.vlm.get_vlm", lambda: None,
    )
    out = take_screenshot_impl(
        monitor_index=0, include_image=False, include_description=True,
    )
    assert out["success"] is True
    assert "VLM not configured" in (out.get("description_error") or "")


def test_take_screenshot_with_description_vlm_succeeds(monkeypatch):
    _patch_capture_pipeline(monkeypatch)
    from ultron.desktop.vlm import VLMResult

    fake_vlm = MagicMock()
    fake_vlm.describe.return_value = VLMResult(
        success=True, description="A code editor.", elapsed_ms=120.0,
    )
    monkeypatch.setattr("ultron.desktop.vlm.get_vlm", lambda: fake_vlm)
    out = take_screenshot_impl(
        monitor_index=0, include_image=False, include_description=True,
    )
    assert out["success"] is True
    assert out["description"] == "A code editor."
    assert out["description_elapsed_ms"] == 120.0


# ---------------------------------------------------------------------------
# describe_screen_impl
# ---------------------------------------------------------------------------


def test_describe_screen_returns_text_only(monkeypatch):
    _patch_capture_pipeline(monkeypatch)
    from ultron.desktop.vlm import VLMResult

    fake_vlm = MagicMock()
    fake_vlm.describe.return_value = VLMResult(
        success=True, description="cursor editor", elapsed_ms=200.0,
    )
    monkeypatch.setattr("ultron.desktop.vlm.get_vlm", lambda: fake_vlm)
    out = describe_screen_impl(monitor_index=0)
    assert out["success"] is True
    assert "image_base64" not in out
    assert out["description"] == "cursor editor"


def test_describe_screen_custom_prompt_recaptures(monkeypatch):
    _patch_capture_pipeline(monkeypatch)
    from ultron.desktop.vlm import VLMResult

    fake_vlm = MagicMock()
    fake_vlm.describe.return_value = VLMResult(
        success=True, description="answer to custom", elapsed_ms=100.0,
    )
    monkeypatch.setattr("ultron.desktop.vlm.get_vlm", lambda: fake_vlm)
    out = describe_screen_impl(
        monitor_index=0, prompt="What is the error message?",
    )
    assert out["success"] is True
    assert out["description"] == "answer to custom"
    # Both the implicit-default-prompt call AND the explicit-prompt call.
    assert fake_vlm.describe.call_count >= 1


# ---------------------------------------------------------------------------
# get_screen_context_impl
# ---------------------------------------------------------------------------


def test_get_screen_context_assembles_payload(monkeypatch):
    from ultron.desktop.monitors import Monitor
    from ultron.desktop.screen_context import ScreenContextSnapshot
    from ultron.desktop.windows import WindowInfo

    fake_mon = Monitor(
        index=0, name="D0", x=0, y=0, width=1920, height=1080,
        work_x=0, work_y=0, work_width=1920, work_height=1040,
        is_primary=True,
    )
    fake_fg = WindowInfo(
        hwnd=42, title="Cursor", class_name="", process_name="Cursor.exe",
        pid=99, rect=(0, 0, 800, 600), monitor_index=0,
        is_minimized=False, is_foreground=True,
    )
    snap = ScreenContextSnapshot(
        timestamp=0.0, monitors=(fake_mon,),
        foreground=fake_fg,
        windows=(fake_fg,),
        ui_text=("File", "Edit"),
        screenshot=None, vlm_description=None,
        elapsed_ms=10.0,
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.build_screen_context",
        lambda **kw: snap,
    )
    out = get_screen_context_impl(include_vlm=False)
    assert out["success"] is True
    assert out["foreground"]["title"] == "Cursor"
    assert out["foreground"]["process_name"] == "Cursor.exe"
    assert out["monitors"][0]["is_primary"] is True
    assert out["ui_text"] == ["File", "Edit"]
    assert "Visual context" in out["render_for_llm"]


def test_get_screen_context_no_foreground(monkeypatch):
    from ultron.desktop.screen_context import ScreenContextSnapshot

    snap = ScreenContextSnapshot(
        timestamp=0.0, monitors=(), foreground=None,
        windows=(), ui_text=(),
        screenshot=None, vlm_description=None, elapsed_ms=0.0,
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.build_screen_context",
        lambda **kw: snap,
    )
    out = get_screen_context_impl()
    assert out["success"] is True
    assert out["foreground"] is None


# ---------------------------------------------------------------------------
# launch_app_impl + launch_chrome_url_impl + open_image_search_impl
# ---------------------------------------------------------------------------


def _stub_launcher(monkeypatch, *, result):
    fake = MagicMock()
    fake.launch_app.return_value = result
    fake.launch_chrome.return_value = result
    fake.open_image_search.return_value = result
    monkeypatch.setattr(
        "ultron.desktop.launcher.get_app_launcher", lambda: fake,
    )
    return fake


def _mk_launch_result(success=True, app_name="chrome", error=None):
    from pathlib import Path
    from ultron.desktop.launcher import LaunchResult
    return LaunchResult(
        success=success, app_name=app_name,
        exe_path=Path("C:/ghost/chrome.exe"),
        pid=12345 if success else None,
        hwnd=678 if success else None,
        monitor_index=1 if success else None,
        error=error,
        window_appeared=True if success else None,
    )


def test_launch_app_empty_name_returns_failure():
    out = launch_app_impl(app_name="")
    assert out["success"] is False
    assert "required" in (out.get("error") or "")


def test_launch_app_invalid_monitor_index(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors", lambda: [],
    )
    out = launch_app_impl(app_name="chrome", monitor_index=99)
    assert out["success"] is False
    assert "out of range" in (out.get("error") or "")


def test_launch_app_happy_path(monkeypatch):
    from ultron.desktop.monitors import Monitor
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors",
        lambda: [
            Monitor(
                index=0, name="D0", x=0, y=0, width=1920, height=1080,
                work_x=0, work_y=0, work_width=1920, work_height=1040,
                is_primary=True,
            )
        ],
    )
    fake = _stub_launcher(monkeypatch, result=_mk_launch_result())
    out = launch_app_impl(
        app_name="chrome", monitor_index=0,
        fullscreen=False, maximize=True,
    )
    assert out["success"] is True
    assert out["app_name"] == "chrome"
    assert out["pid"] == 12345
    assert out["window_appeared"] is True
    fake.launch_app.assert_called_once()
    kwargs = fake.launch_app.call_args.kwargs
    assert kwargs["app_name"] == "chrome"
    assert kwargs["maximize"] is True


def test_launch_chrome_url_empty_url_returns_failure():
    out = launch_chrome_url_impl(url="")
    assert out["success"] is False


def test_launch_chrome_url_happy_path(monkeypatch):
    _stub_launcher(monkeypatch, result=_mk_launch_result())
    out = launch_chrome_url_impl(
        url="https://youtube.com", monitor_index=None, maximize=True,
    )
    assert out["success"] is True
    assert out["url"] == "https://youtube.com"


def test_open_image_search_empty_query_returns_failure():
    out = open_image_search_impl(query="")
    assert out["success"] is False


def test_open_image_search_happy_path(monkeypatch):
    fake = _stub_launcher(monkeypatch, result=_mk_launch_result())
    out = open_image_search_impl(
        query="golden retriever", monitor_index=None,
    )
    assert out["success"] is True
    assert out["query"] == "golden retriever"
    fake.open_image_search.assert_called_once()


# ---------------------------------------------------------------------------
# move_window_to_monitor_impl
# ---------------------------------------------------------------------------


def test_move_window_empty_query_returns_failure():
    out = move_window_to_monitor_impl(window_query="", monitor_index=0)
    assert out["success"] is False


def test_move_window_no_match_returns_failure(monkeypatch):
    from ultron.desktop.monitors import Monitor
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors",
        lambda: [
            Monitor(
                index=0, name="D0", x=0, y=0, width=1920, height=1080,
                work_x=0, work_y=0, work_width=1920, work_height=1040,
                is_primary=True,
            )
        ],
    )
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window", lambda *a, **kw: None,
    )
    out = move_window_to_monitor_impl(
        window_query="nonexistent", monitor_index=0,
    )
    assert out["success"] is False
    assert "no window matching" in (out.get("error") or "")


def test_move_window_happy_path(monkeypatch):
    from ultron.desktop.monitors import Monitor
    from ultron.desktop.placement import PlacementResult
    from ultron.desktop.windows import WindowInfo

    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors",
        lambda: [
            Monitor(
                index=0, name="D0", x=0, y=0, width=1920, height=1080,
                work_x=0, work_y=0, work_width=1920, work_height=1040,
                is_primary=True,
            ),
            Monitor(
                index=1, name="D1", x=1920, y=0, width=1920, height=1080,
                work_x=1920, work_y=0,
                work_width=1920, work_height=1040,
                is_primary=False,
            ),
        ],
    )
    fake_win = WindowInfo(
        hwnd=42, title="Chrome", class_name="C", process_name="chrome.exe",
        pid=1, rect=(0, 0, 800, 600), monitor_index=0,
        is_minimized=False, is_foreground=False,
    )
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window", lambda *a, **kw: fake_win,
    )
    calls = []

    def fake_move(*, hwnd, monitor, **kw):
        calls.append((hwnd, monitor, kw))
        return PlacementResult(
            success=True, hwnd=hwnd, monitor_index=monitor.index,
        )

    monkeypatch.setattr(
        "ultron.desktop.placement.move_window_to_monitor", fake_move,
    )
    out = move_window_to_monitor_impl(
        window_query="chrome", monitor_index=1, maximize=True,
    )
    assert out["success"] is True
    assert out["window_title"] == "Chrome"
    assert out["monitor_index"] == 1
    assert calls[0][0] == 42
    assert calls[0][2]["maximize"] is True


# ---------------------------------------------------------------------------
# focus_window_impl / window_action_impl
# ---------------------------------------------------------------------------


def _fake_win(hwnd=42, title="Chrome", proc="chrome.exe", mon=0):
    from ultron.desktop.windows import WindowInfo
    return WindowInfo(
        hwnd=hwnd, title=title, class_name="C", process_name=proc,
        pid=1, rect=(0, 0, 800, 600), monitor_index=mon,
        is_minimized=False, is_foreground=False,
    )


def test_focus_window_no_match(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window", lambda *a, **kw: None,
    )
    out = focus_window_impl(window_query="nonexistent")
    assert out["success"] is False
    assert "no window matching" in (out.get("error") or "")


def test_focus_window_happy_path(monkeypatch):
    from ultron.desktop.placement import PlacementResult
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window",
        lambda *a, **kw: _fake_win(),
    )
    monkeypatch.setattr(
        "ultron.desktop.placement.focus_window",
        lambda hwnd: PlacementResult(success=True, hwnd=hwnd),
    )
    out = focus_window_impl(window_query="chrome")
    assert out["success"] is True
    assert out["window_title"] == "Chrome"


def test_window_action_unknown_action():
    out = window_action_impl(window_query="x", action="explode")
    assert out["success"] is False
    assert "unknown action" in (out.get("error") or "")


def test_window_action_empty_query():
    out = window_action_impl(window_query="", action="maximize")
    assert out["success"] is False


def test_window_action_maximize_happy_path(monkeypatch):
    from ultron.desktop.placement import PlacementResult
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window",
        lambda *a, **kw: _fake_win(),
    )
    called = []
    monkeypatch.setattr(
        "ultron.desktop.placement.maximize_window",
        lambda h: called.append(("maximize", h)) or PlacementResult(success=True, hwnd=h),
    )
    monkeypatch.setattr(
        "ultron.desktop.placement.minimize_window",
        lambda h: called.append(("minimize", h)) or PlacementResult(success=True, hwnd=h),
    )
    monkeypatch.setattr(
        "ultron.desktop.placement.restore_window",
        lambda h: called.append(("restore", h)) or PlacementResult(success=True, hwnd=h),
    )
    out = window_action_impl(window_query="chrome", action="maximize")
    assert out["success"] is True
    assert out["action"] == "maximize"
    assert called == [("maximize", 42)]


def test_window_action_minimize_dispatches(monkeypatch):
    from ultron.desktop.placement import PlacementResult
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window",
        lambda *a, **kw: _fake_win(),
    )
    monkeypatch.setattr(
        "ultron.desktop.placement.minimize_window",
        lambda h: PlacementResult(success=True, hwnd=h),
    )
    out = window_action_impl(window_query="chrome", action="minimize")
    assert out["success"] is True
    assert out["action"] == "minimize"


# ---------------------------------------------------------------------------
# click_uia_impl / type_into_uia_impl / get_window_text_impl
# ---------------------------------------------------------------------------


def test_click_uia_empty_args():
    assert click_uia_impl(window_query="", element_query="x")["success"] is False
    assert click_uia_impl(window_query="x", element_query="")["success"] is False


def test_click_uia_window_not_found(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window", lambda *a, **kw: None,
    )
    out = click_uia_impl(window_query="ghost", element_query="Submit")
    assert out["success"] is False
    assert "no window" in (out.get("error") or "")


def test_click_uia_happy_path(monkeypatch):
    from ultron.desktop.uia import UIAActionResult
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window",
        lambda *a, **kw: _fake_win(),
    )
    captured = []

    def fake_click(window, query, **kw):
        captured.append((query, kw))
        return UIAActionResult(success=True, element_name=query)

    monkeypatch.setattr(
        "ultron.desktop.uia.click_element", fake_click,
    )
    out = click_uia_impl(
        window_query="chrome",
        element_query="OK",
        control_type="Button",
        user_text="press ok button",
    )
    assert out["success"] is True
    assert out["element_name"] == "OK"
    assert captured[0][0] == "OK"


def test_type_into_uia_empty_args():
    assert type_into_uia_impl(
        window_query="", element_query="x", text="y",
    )["success"] is False
    assert type_into_uia_impl(
        window_query="x", element_query="", text="y",
    )["success"] is False
    assert type_into_uia_impl(
        window_query="x", element_query="y", text=12345,  # type: ignore[arg-type]
    )["success"] is False


def test_type_into_uia_happy_path(monkeypatch):
    from ultron.desktop.uia import UIAActionResult
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window",
        lambda *a, **kw: _fake_win(),
    )
    captured = []

    def fake_type(window, query, text, **kw):
        captured.append((query, text, kw))
        return UIAActionResult(success=True, element_name=query)

    monkeypatch.setattr(
        "ultron.desktop.uia.type_text_into_element", fake_type,
    )
    out = type_into_uia_impl(
        window_query="chrome",
        element_query="search bar",
        text="hello world",
        clear_first=False,
    )
    assert out["success"] is True
    assert captured[0][1] == "hello world"


def test_get_window_text_happy_path(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window",
        lambda *a, **kw: _fake_win(),
    )
    monkeypatch.setattr(
        "ultron.desktop.uia.collect_window_text",
        lambda *a, **kw: ["File", "Edit", "View"],
    )
    out = get_window_text_impl(window_query="chrome")
    assert out["success"] is True
    assert out["count"] == 3
    assert "Edit" in out["text_lines"]


def test_get_window_text_no_match(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window", lambda *a, **kw: None,
    )
    out = get_window_text_impl(window_query="ghost")
    assert out["success"] is False


# ---------------------------------------------------------------------------
# Input primitive impls (mouse / keyboard / scroll)
# ---------------------------------------------------------------------------


def _patch_input_controller(monkeypatch, *, result_kwargs):
    """Inject a fake input controller whose methods all return the given
    InputControlResult-shaped dict.
    """
    from ultron.desktop.input_control import InputControlResult

    fake = MagicMock()
    for method in ("click", "move_mouse", "type_text", "press_hotkey", "scroll"):
        getattr(fake, method).return_value = InputControlResult(**result_kwargs)
    monkeypatch.setattr(
        "ultron.desktop.input_control.get_input_controller", lambda: fake,
    )
    return fake


def test_mouse_click_happy_path(monkeypatch):
    fake = _patch_input_controller(
        monkeypatch, result_kwargs={"success": True, "action": "click"},
    )
    out = mouse_click_impl(x=100, y=200, button="left", clicks=1)
    assert out["success"] is True
    fake.click.assert_called_once()


def test_mouse_move_happy_path(monkeypatch):
    fake = _patch_input_controller(
        monkeypatch, result_kwargs={"success": True, "action": "move_mouse"},
    )
    out = mouse_move_impl(x=500, y=300)
    assert out["success"] is True
    fake.move_mouse.assert_called_once()


def test_type_text_happy_path(monkeypatch):
    fake = _patch_input_controller(
        monkeypatch, result_kwargs={"success": True, "action": "type_text"},
    )
    out = type_text_impl(text="hello", interval_s=0.0)
    assert out["success"] is True
    fake.type_text.assert_called_once()


def test_press_hotkey_empty_keys_returns_failure():
    out = press_hotkey_impl(keys=[])
    assert out["success"] is False


def test_press_hotkey_happy_path(monkeypatch):
    fake = _patch_input_controller(
        monkeypatch, result_kwargs={"success": True, "action": "press_hotkey"},
    )
    out = press_hotkey_impl(keys=["ctrl", "s"])
    assert out["success"] is True
    fake.press_hotkey.assert_called_once_with("ctrl", "s", user_text="")


def test_scroll_happy_path(monkeypatch):
    fake = _patch_input_controller(
        monkeypatch, result_kwargs={"success": True, "action": "scroll"},
    )
    out = scroll_impl(amount=120, x=500, y=400)
    assert out["success"] is True
    fake.scroll.assert_called_once()


def test_input_impls_validator_block_propagates(monkeypatch):
    fake = _patch_input_controller(
        monkeypatch,
        result_kwargs={
            "success": False,
            "action": "click",
            "error": "safety: blocked by Cap-4",
        },
    )
    out = mouse_click_impl(x=100, y=200)
    assert out["success"] is False
    assert "safety" in (out.get("error") or "")


# ---------------------------------------------------------------------------
# Catalog 09 T1 / T3 / T7 -- new kwargs on existing impls
# ---------------------------------------------------------------------------


def test_scroll_impl_forwards_direction_horizontal(monkeypatch):
    fake = _patch_input_controller(
        monkeypatch, result_kwargs={"success": True, "action": "scroll"},
    )
    out = scroll_impl(amount=80, direction="horizontal", x=200, y=300)
    assert out["success"] is True
    _, kwargs = fake.scroll.call_args
    assert kwargs["direction"] == "horizontal"
    assert kwargs["amount"] == 80


def test_scroll_impl_default_direction_is_vertical(monkeypatch):
    """Existing scroll callers that omit direction must still get
    vertical scroll (back-compat)."""
    fake = _patch_input_controller(
        monkeypatch, result_kwargs={"success": True, "action": "scroll"},
    )
    out = scroll_impl(amount=120)
    assert out["success"] is True
    _, kwargs = fake.scroll.call_args
    assert kwargs["direction"] == "vertical"


def test_type_text_impl_forwards_wpm(monkeypatch):
    fake = _patch_input_controller(
        monkeypatch, result_kwargs={"success": True, "action": "type_text"},
    )
    out = type_text_impl(text="hello world", wpm=80)
    assert out["success"] is True
    _, kwargs = fake.type_text.call_args
    assert kwargs["wpm"] == 80
    assert kwargs["text"] == "hello world"


def test_type_text_impl_default_wpm_is_none(monkeypatch):
    """Back-compat: omitting wpm preserves the legacy interval_s
    contract -- the controller sees wpm=None and falls through to
    the caller-supplied interval."""
    fake = _patch_input_controller(
        monkeypatch, result_kwargs={"success": True, "action": "type_text"},
    )
    out = type_text_impl(text="hi", interval_s=0.02)
    assert out["success"] is True
    _, kwargs = fake.type_text.call_args
    assert kwargs.get("wpm") is None
    assert kwargs["interval_s"] == pytest.approx(0.02)


def test_mouse_move_impl_forwards_smooth(monkeypatch):
    fake = _patch_input_controller(
        monkeypatch, result_kwargs={"success": True, "action": "move_mouse"},
    )
    out = mouse_move_impl(x=100, y=200, duration_s=0.4, smooth=True)
    assert out["success"] is True
    _, kwargs = fake.move_mouse.call_args
    assert kwargs["smooth"] is True
    assert kwargs["duration_s"] == pytest.approx(0.4)


def test_mouse_move_impl_default_smooth_is_false(monkeypatch):
    """Back-compat: existing mouse_move callers without smooth keep
    using the default linear move."""
    fake = _patch_input_controller(
        monkeypatch, result_kwargs={"success": True, "action": "move_mouse"},
    )
    out = mouse_move_impl(x=10, y=20)
    assert out["success"] is True
    _, kwargs = fake.move_mouse.call_args
    assert kwargs["smooth"] is False


# ---------------------------------------------------------------------------
# Catalog 09 T4 -- clipboard read / write MCP-surface impls
# ---------------------------------------------------------------------------


def _patch_clipboard_manager(monkeypatch, *, result_kwargs):
    from ultron.desktop.clipboard import ClipboardResult

    fake = MagicMock()
    fake.read_text.return_value = ClipboardResult(**result_kwargs)
    fake.write_text.return_value = ClipboardResult(**result_kwargs)
    monkeypatch.setattr(
        "ultron.desktop.clipboard.get_clipboard_manager", lambda: fake,
    )
    return fake


def test_clipboard_read_impl_success(monkeypatch):
    fake = _patch_clipboard_manager(
        monkeypatch,
        result_kwargs={
            "success": True,
            "action": "read",
            "text": "from clipboard",
            "tainted": True,
        },
    )
    out = clipboard_read_impl(user_text="read it")
    assert out["success"] is True
    assert out["action"] == "clipboard_read"
    assert out["text"] == "from clipboard"
    assert out["tainted"] is True
    fake.read_text.assert_called_once()


def test_clipboard_read_impl_failure_propagates(monkeypatch):
    _patch_clipboard_manager(
        monkeypatch,
        result_kwargs={
            "success": False,
            "action": "read",
            "error": "pyperclip unavailable",
        },
    )
    out = clipboard_read_impl()
    assert out["success"] is False
    assert "pyperclip" in (out.get("error") or "")
    assert "text" not in out  # absent on failure


def test_clipboard_write_impl_success(monkeypatch):
    fake = _patch_clipboard_manager(
        monkeypatch,
        result_kwargs={"success": True, "action": "write", "tainted": True},
    )
    out = clipboard_write_impl(text="hello", user_text="copy this")
    assert out["success"] is True
    assert out["action"] == "clipboard_write"
    assert out["tainted"] is True
    _, kwargs = fake.write_text.call_args
    assert kwargs["user_text"] == "copy this"


def test_clipboard_write_impl_rejects_non_string():
    out = clipboard_write_impl(text=12345)  # type: ignore[arg-type]
    assert out["success"] is False
    assert "string" in (out.get("error") or "")


def test_clipboard_write_impl_failure_propagates(monkeypatch):
    _patch_clipboard_manager(
        monkeypatch,
        result_kwargs={
            "success": False,
            "action": "write",
            "error": "safety: payload contains credential",
        },
    )
    out = clipboard_write_impl(text="my password is hunter2")
    assert out["success"] is False
    assert "credential" in (out.get("error") or "")


# ---------------------------------------------------------------------------
# Catalog 09 T6 -- find_image_on_screen MCP-surface impl
# ---------------------------------------------------------------------------


def test_find_image_on_screen_impl_returns_match_payload(monkeypatch):
    from ultron.desktop.capture import TemplateMatch

    def _fake(template_path, **kw):
        return TemplateMatch(
            left=10, top=20, width=100, height=80,
            center_x=60, center_y=60, confidence=0.85,
        )

    monkeypatch.setattr(
        "ultron.desktop.capture.find_image_on_screen", _fake,
    )
    out = find_image_on_screen_impl(template_path="ok.png")
    assert out["success"] is True
    m = out["match"]
    assert m["left"] == 10 and m["top"] == 20
    assert m["width"] == 100 and m["height"] == 80
    assert m["center_x"] == 60 and m["center_y"] == 60
    assert m["confidence"] == 0.85


def test_find_image_on_screen_impl_none_match_returns_failure(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.capture.find_image_on_screen",
        lambda template_path, **kw: None,
    )
    out = find_image_on_screen_impl(template_path="missing.png")
    assert out["success"] is False
    assert "no match" in (out.get("error") or "")


def test_find_image_on_screen_impl_region_forwarded(monkeypatch):
    seen: dict = {}

    def _fake(template_path, **kw):
        seen.update(kw)
        return None

    monkeypatch.setattr(
        "ultron.desktop.capture.find_image_on_screen", _fake,
    )
    find_image_on_screen_impl(
        template_path="ok.png",
        region_left=10, region_top=20,
        region_width=300, region_height=400,
    )
    assert seen["region"] == (10, 20, 300, 400)


def test_find_image_on_screen_impl_partial_region_rejected(monkeypatch):
    # locate must not be called when region is partially specified.
    def _should_not_be_called(*a, **kw):
        raise AssertionError(
            "find_image_on_screen called despite incomplete region kwargs",
        )

    monkeypatch.setattr(
        "ultron.desktop.capture.find_image_on_screen",
        _should_not_be_called,
    )
    out = find_image_on_screen_impl(
        template_path="ok.png", region_left=10, region_top=20,
    )
    assert out["success"] is False
    assert "region" in (out.get("error") or "")
