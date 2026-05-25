"""Native desktop automation primitives.

This package replaces the (dormant) OpenClaw ``desktop-control`` and
``windows-control`` plugin paths with native Python implementations.
The user-led decision was to skip ClawHub plugins entirely; we get
the same UI Automation power via ``pywinauto`` and the same screen
capture power via ``mss`` -- with one Python stack to maintain and
one safety enforcement surface (the runtime tool-call validator).

Phase 1 (this file): monitors, capture, windows enumeration.
Phase 2+: launcher, placement, input_control, screen_context,
moondream2 VLM, MCP tool exposure for OpenClaw agents.

Module shape:

    src/ultron/desktop/
    +-- monitors.py        Win32 monitor enumeration
    +-- capture.py         mss-based multi-monitor capture
    +-- windows.py         pywin32 window enumeration + foreground detection
"""

from __future__ import annotations

from ultron.desktop.monitors import (
    Monitor,
    enumerate_monitors,
    find_monitor,
    point_to_monitor,
)
from ultron.desktop.capture import (
    Screenshot,
    ScreenCapture,
    ScreenCaptureError,
    get_screen_capture,
    set_screen_capture,
)
from ultron.desktop.windows import (
    WindowInfo,
    enumerate_windows,
    get_foreground_window,
    find_window,
    wait_for_window,
)
from ultron.desktop.placement import (
    PlacementResult,
    move_window_to_monitor,
    maximize_window,
    minimize_window,
    restore_window,
    focus_window,
)
from ultron.desktop.launcher import (
    AppEntry,
    AppLauncher,
    LaunchResult,
    get_app_launcher,
    set_app_launcher,
)
from ultron.desktop.uia import (
    UIAElement,
    UIAActionResult,
    UIElementInfo,
    collect_window_text,
    find_element,
    click_element,
    type_text_into_element,
    get_ui_element_inventory,
    wait_for_text_in_window,
)
from ultron.desktop.input_control import (
    InputControlResult,
    InputController,
    get_input_controller,
    set_input_controller,
)
from ultron.desktop.screen_context import (
    ScreenContextSnapshot,
    ScreenContextCache,
    build_screen_context,
    capture_and_cache,
    get_screen_context_cache,
    set_screen_context_cache,
    set_vlm_describe,
    get_vlm_describe,
)
from ultron.desktop.vlm import (
    Moondream2VLM,
    VLMResult,
    VLMLoadError,
    build_vlm_from_config,
    get_vlm,
    set_vlm,
)
from ultron.desktop.voice import (
    AppLaunchVoiceResult,
    ScreenContextVoiceResult,
    handle_app_launch,
    handle_screen_context_query,
)

__all__ = [
    # monitors
    "Monitor",
    "enumerate_monitors",
    "find_monitor",
    "point_to_monitor",
    # capture
    "Screenshot",
    "ScreenCapture",
    "ScreenCaptureError",
    "get_screen_capture",
    "set_screen_capture",
    # windows
    "WindowInfo",
    "enumerate_windows",
    "get_foreground_window",
    "find_window",
    "wait_for_window",
    # placement
    "PlacementResult",
    "move_window_to_monitor",
    "maximize_window",
    "minimize_window",
    "restore_window",
    "focus_window",
    # launcher
    "AppEntry",
    "AppLauncher",
    "LaunchResult",
    "get_app_launcher",
    "set_app_launcher",
    # uia
    "UIAElement",
    "UIAActionResult",
    "UIElementInfo",
    "collect_window_text",
    "find_element",
    "click_element",
    "type_text_into_element",
    "get_ui_element_inventory",
    "wait_for_text_in_window",
    # input_control
    "InputControlResult",
    "InputController",
    "get_input_controller",
    "set_input_controller",
    # screen_context
    "ScreenContextSnapshot",
    "ScreenContextCache",
    "build_screen_context",
    "capture_and_cache",
    "get_screen_context_cache",
    "set_screen_context_cache",
    "set_vlm_describe",
    "get_vlm_describe",
    # vlm
    "Moondream2VLM",
    "VLMResult",
    "VLMLoadError",
    "build_vlm_from_config",
    "get_vlm",
    "set_vlm",
    # voice (Phase 8 intent handlers)
    "AppLaunchVoiceResult",
    "ScreenContextVoiceResult",
    "handle_app_launch",
    "handle_screen_context_query",
]
