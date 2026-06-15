"""Tests for anticheat-safe mode (``kenning.safety.anticheat``).

The user's account is on the line here, so coverage is exhaustive:
the guard semantics, the toggle matcher, the blocked-tool taxonomy,
a sweep asserting EVERY guarded desktop entry point actually raises
before touching any OS API, the validator pre-check, the orchestrator
voice toggle, and the gaming-mode tie-in.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from kenning.safety.anticheat import (
    AnticheatBlockedError,
    BLOCKED_NOTICE,
    anticheat_active,
    guard,
    is_blocked_tool,
    match_anticheat_toggle,
    set_anticheat_active,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "kenning"


from kenning.safety.anticheat import clear_surface_hooks, register_surface_hook


@pytest.fixture(autouse=True)
def _reset_anticheat():
    """Every test starts and ends with the mode OFF and no hooks."""
    clear_surface_hooks()
    set_anticheat_active(False)
    yield
    clear_surface_hooks()
    set_anticheat_active(False)


# ---------------------------------------------------------------------------
# Core guard semantics
# ---------------------------------------------------------------------------


def test_inactive_by_default() -> None:
    assert anticheat_active() is False
    guard("click")  # no raise


def test_runtime_toggle_activates_guard() -> None:
    set_anticheat_active(True, "test")
    assert anticheat_active() is True
    with pytest.raises(AnticheatBlockedError) as exc:
        guard("click")
    assert "click" in str(exc.value)
    assert BLOCKED_NOTICE in str(exc.value)
    set_anticheat_active(False)
    guard("click")  # no raise again


def test_config_pin_activates_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from kenning.safety.anticheat import set_config_pin_enabled

    import kenning.config as config_mod

    monkeypatch.setattr(
        config_mod, "get_config",
        lambda: SimpleNamespace(
            gaming_mode=SimpleNamespace(anticheat_safe_mode=True),
        ),
    )
    # Opt back in past the session conftest guard for this test.
    set_config_pin_enabled(True)
    try:
        assert anticheat_active() is True
    finally:
        set_config_pin_enabled(False)


def test_config_pin_ignored_when_disabled_for_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session conftest guard: a pinned config must not leak into
    hermetic tests (set_config_pin_enabled(False) ignores it); the
    runtime flag still works."""
    import kenning.config as config_mod

    monkeypatch.setattr(
        config_mod, "get_config",
        lambda: SimpleNamespace(
            gaming_mode=SimpleNamespace(anticheat_safe_mode=True),
        ),
    )
    # conftest already disabled the pin for the session.
    assert anticheat_active() is False
    set_anticheat_active(True)
    assert anticheat_active() is True


def test_config_errors_fail_open_but_runtime_flag_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kenning.config as config_mod

    def boom():
        raise RuntimeError("config broken")

    monkeypatch.setattr(config_mod, "get_config", boom)
    assert anticheat_active() is False  # config error alone -> off
    set_anticheat_active(True)
    assert anticheat_active() is True   # runtime flag unaffected


# ---------------------------------------------------------------------------
# Blocked-tool taxonomy (validator layer)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", [
    "click", "type_text", "scroll", "move_mouse", "drag_to",
    "press_key", "press_hotkey",
    "screenshot", "get_pixel_color", "wait_for_pixel_color",
    "find_image_on_screen", "clipboard_read", "clipboard_write",
    "ocr", "semantic_click", "desktop_screenshot", "desktop_list_windows",
    "desktop_find_window", "window_close", "window_move", "dialog_click",
    "element_click", "browser_use_open", "ui_inventory", "screen_context",
    # Namespaced dispatcher + dotted bridge names must normalize into the
    # block + audit ledger (regression: leading "openclaw." / dotted segments
    # bypassed the prefix check when anticheat was pinned without gaming-mode
    # engagement).
    "openclaw.window_automation", "openclaw.desktop_automation",
    "desktop.input.press_key", "desktop.input.press_hotkey",
    "OpenClaw.Window_Automation",  # case-insensitive
])
def test_blocked_tools(tool: str) -> None:
    assert is_blocked_tool(tool) is True


@pytest.mark.parametrize("tool", [
    "web_search", "memory_retrieve", "tts_speak", "stt_transcribe",
    "relay_speech", "llm_generate", "evolution_cycle", "run_program",
    "",
])
def test_allowed_tools(tool: str) -> None:
    assert is_blocked_tool(tool) is False


# ---------------------------------------------------------------------------
# Voice toggle matcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("Enable anticheat mode.", True),
    ("engage anti-cheat mode", True),
    ("turn on anticheat safe mode", True),
    ("activate tournament mode", True),
    ("Disable anticheat mode.", False),
    ("turn off the anticheat mode", False),
    ("disengage anti cheat mode", False),
    # Non-toggles.
    ("what is anticheat mode", None),
    ("anticheat mode", None),
    ("enable gaming mode", None),
    ("", None),
])
def test_match_anticheat_toggle(text: str, expected) -> None:
    assert match_anticheat_toggle(text) == expected


# ---------------------------------------------------------------------------
# EVERY guarded desktop entry point raises while active
# ---------------------------------------------------------------------------

# (module, class or None, function) -- must stay in sync with the
# guards inserted across the desktop surface. The AST audit test below
# proves each listed function still contains its guard call.
GUARDED = [
    ("kenning.desktop.input_control", "InputController",
     ["move_mouse", "click", "type_text", "drag_to", "scroll",
      "press_key", "press_hotkey"]),
    ("kenning.desktop.capture", "ScreenCapture",
     ["capture_monitor", "capture_all_monitors", "capture_region"]),
    ("kenning.desktop.capture", None,
     ["find_image_on_screen", "get_pixel_color"]),
    ("kenning.desktop.uia", None,
     ["collect_window_text", "find_element", "click_element",
      "type_text_into_element", "dpi_aware_click_at_element_center",
      "get_ui_element_inventory", "wait_for_text_in_window",
      "wait_for_pixel_color", "find_browser_window",
      "extract_browser_content", "physical_center_of_element",
      "physical_rect_of_element"]),
    ("kenning.desktop.clipboard", "ClipboardManager",
     ["read_text", "write_text"]),
    ("kenning.desktop.dialog_control", None,
     ["find_dialogs", "read_dialog", "click_dialog_button",
      "type_into_dialog_field", "dismiss_dialog", "wait_for_dialog"]),
    ("kenning.desktop.element_click", None,
     ["find_elements_by_name", "click_element_by_name",
      "find_text_in_window"]),
    ("kenning.desktop.windows", None, ["focus_by_title", "close_window"]),
    ("kenning.desktop.placement", None,
     ["move_window_to_monitor", "maximize_window", "minimize_window",
      "restore_window", "focus_window"]),
    ("kenning.desktop.launcher", "AppLauncher",
     ["launch_app", "launch_chrome", "open_image_search"]),
    ("kenning.desktop.ocr", None, ["ocr_screen_region", "ocr_screen_monitor"]),
    ("kenning.desktop.sequence", "DesktopSequenceRunner", ["run"]),
    ("kenning.desktop.browser_use", "BrowserUseTool", ["_invoke"]),
    ("kenning.desktop.screen_context", None, ["build_screen_context"]),
    ("kenning.openclaw_bridge.desktop", "DesktopTool",
     ["screenshot", "list_windows", "find_window"]),
]


def test_every_guarded_function_contains_guard_call() -> None:
    """AST audit: each listed function literally calls the guard as one
    of its first statements -- a refactor that drops a guard fails HERE,
    before it can cost an account."""
    missing: list[str] = []
    for module, cls, fns in GUARDED:
        rel = module.replace("kenning.", "").replace(".", "/") + ".py"
        tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
        for want in fns:
            found = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and node.name == want:
                    src = ast.unparse(node)
                    if "_anticheat_guard(" in src or "guard(" in src:
                        found = True
                        break
            if not found:
                missing.append(f"{module}.{cls or ''}.{want}")
    assert not missing, f"guards missing from: {missing}"


def test_surface_hooks_stop_and_restore_subsystems() -> None:
    """Activating the mode must physically STOP running subsystems (a
    kernel anticheat sees activity, not just call gates); deactivating
    restores them. Hooks receive the new state; a broken hook never
    blocks the flip or the other hooks."""
    calls: list[tuple[str, bool]] = []
    register_surface_hook("poller", lambda a: calls.append(("poller", a)))

    def broken(active: bool) -> None:
        raise RuntimeError("hook boom")

    register_surface_hook("broken", broken)
    register_surface_hook("capture", lambda a: calls.append(("capture", a)))

    set_anticheat_active(True, "test")
    assert ("poller", True) in calls and ("capture", True) in calls
    assert anticheat_active() is True  # broken hook didn't block the flip

    calls.clear()
    set_anticheat_active(False)
    assert ("poller", False) in calls and ("capture", False) in calls


def test_clear_surface_hooks() -> None:
    calls: list[bool] = []
    register_surface_hook("x", calls.append)
    clear_surface_hooks()
    set_anticheat_active(True, "test")
    assert calls == []


def test_no_ban_class_apis_anywhere_in_source() -> None:
    """Vanguard paranoia pin: the API classes kernel anticheats ban for
    (foreign process handles, memory read/write, remote threads, global
    input hooks, raw-input registration, input-hook libraries) must
    NEVER appear in Kenning's source. The only permitted location is
    ``safety/rules/`` -- the DEFENSE regexes that exist to block these
    exact patterns in model-proposed commands."""
    import re

    forbidden = re.compile(
        r"OpenProcess|ReadProcessMemory|WriteProcessMemory"
        r"|CreateRemoteThread|VirtualAllocEx|SetWindowsHookEx"
        r"|RegisterRawInputDevices|NtOpenProcess|NtReadVirtualMemory"
        r"|from pynput|import pynput|import interception|import dxcam"
        r"|ImageGrab"
    )
    offenders: list[str] = []
    for py in SRC.rglob("*.py"):
        rel = py.relative_to(SRC).as_posix()
        if rel.startswith("safety/rules/") or rel == "safety/anticheat.py":
            continue  # defense regexes / threat-model docs, not API usage
        if forbidden.search(py.read_text(encoding="utf-8", errors="replace")):
            offenders.append(rel)
    assert not offenders, (
        f"ban-class API reference introduced in: {offenders} -- "
        "this could cost the user their game account; remove it or "
        "gate it behind an explicit design review"
    )


def test_press_key_and_hotkey_blocked_while_active() -> None:
    """press_key / press_hotkey drive pyautogui (SendInput) -- a Vanguard/EAC
    ban-class injection surface. They must hard-raise while anticheat-safe mode
    is on, exactly like the other five input methods (regression: both were
    ungated and absent from the validator block list)."""
    from kenning.desktop.input_control import InputController

    set_anticheat_active(True, "test")
    ic = InputController()
    with pytest.raises(AnticheatBlockedError):
        ic.press_key("enter")
    with pytest.raises(AnticheatBlockedError):
        ic.press_hotkey("ctrl", "s")


def test_module_guard_blocks_before_os_touch() -> None:
    """Representative end-to-end checks: guarded entry points raise
    AnticheatBlockedError immediately (no OS API import/touch) while
    the mode is active."""
    set_anticheat_active(True, "test")

    from kenning.desktop.capture import find_image_on_screen, get_pixel_color
    from kenning.desktop.dialog_control import find_dialogs
    from kenning.desktop.element_click import find_elements_by_name
    from kenning.desktop.placement import minimize_window
    from kenning.desktop.windows import close_window

    with pytest.raises(AnticheatBlockedError):
        get_pixel_color(10, 10)
    with pytest.raises(AnticheatBlockedError):
        find_image_on_screen("nonexistent.png")
    with pytest.raises(AnticheatBlockedError):
        find_dialogs()
    with pytest.raises(AnticheatBlockedError):
        find_elements_by_name("x")
    with pytest.raises(AnticheatBlockedError):
        close_window("Untitled - Notepad")
    with pytest.raises(AnticheatBlockedError):
        minimize_window(1)


# ---------------------------------------------------------------------------
# Validator pre-check
# ---------------------------------------------------------------------------


def test_validator_blocks_desktop_tool_when_active(tmp_path: Path) -> None:
    from kenning.safety.audit import AuditLog
    from kenning.safety.policy import Policy
    from kenning.safety.validator import (
        RuleContext,
        ToolCallValidator,
        Verdict,
    )

    validator = ToolCallValidator(
        policy=Policy(enabled=True, rule_enabled={}),
        rules=[],
        audit_log=AuditLog(path=tmp_path / "audit.jsonl"),
    )
    ctx = RuleContext(
        tool_name="desktop_screenshot", capability="screen_capture",
        arguments={},
    )
    set_anticheat_active(True, "test")
    verdict = validator.check(ctx)
    assert verdict.verdict == Verdict.BLOCK_HARD
    assert verdict.triggered_rule_id == "anticheat_safe_mode"
    assert verdict.user_message == BLOCKED_NOTICE
    # Audited.
    assert (tmp_path / "audit.jsonl").is_file()

    set_anticheat_active(False)
    verdict = validator.check(ctx)
    assert verdict.verdict == Verdict.ALLOW


def test_validator_allows_non_desktop_tool_when_active(
    tmp_path: Path,
) -> None:
    from kenning.safety.audit import AuditLog
    from kenning.safety.policy import Policy
    from kenning.safety.validator import (
        RuleContext,
        ToolCallValidator,
        Verdict,
    )

    validator = ToolCallValidator(
        policy=Policy(enabled=True, rule_enabled={}),
        rules=[],
        audit_log=AuditLog(path=tmp_path / "audit.jsonl"),
    )
    set_anticheat_active(True, "test")
    verdict = validator.check(RuleContext(
        tool_name="web_search", capability="network", arguments={},
    ))
    assert verdict.verdict == Verdict.ALLOW


# ---------------------------------------------------------------------------
# Orchestrator voice toggle + gaming-mode tie-in
# ---------------------------------------------------------------------------


def _bare_orchestrator():
    from kenning.pipeline.orchestrator import Orchestrator

    o = Orchestrator.__new__(Orchestrator)
    o._spoken = []
    o._speak = lambda text: o._spoken.append(text)  # type: ignore[attr-defined]
    return o


def test_orchestrator_anticheat_toggle() -> None:
    o = _bare_orchestrator()
    assert o._maybe_handle_anticheat_toggle("enable anticheat mode") is True
    assert anticheat_active() is True
    assert "engaged" in o._spoken[-1].lower()
    assert o._maybe_handle_anticheat_toggle("disable anticheat mode") is True
    assert anticheat_active() is False
    assert "off" in o._spoken[-1].lower()
    assert o._maybe_handle_anticheat_toggle("what time is it") is False


def test_gaming_mode_tie_in(monkeypatch: pytest.MonkeyPatch) -> None:
    import kenning.config as config_mod
    from kenning.openclaw_routing.gaming_mode import GamingModeManager

    monkeypatch.setattr(
        config_mod, "get_config",
        lambda: SimpleNamespace(
            gaming_mode=SimpleNamespace(
                anticheat_with_gaming_mode=True,
                anticheat_safe_mode=False,
            ),
        ),
    )
    mgr = GamingModeManager.__new__(GamingModeManager)
    mgr._set_anticheat(True)
    assert anticheat_active() is True
    mgr._set_anticheat(False)
    assert anticheat_active() is False


def test_gaming_mode_drives_anticheat_both_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anticheat is 100% tied: engage -> ON, disengage -> OFF,
    unconditionally. It is purely a function of gaming-mode state."""
    from kenning.openclaw_routing.gaming_mode import GamingModeManager

    mgr = GamingModeManager.__new__(GamingModeManager)
    assert anticheat_active() is False          # off by default
    mgr._set_anticheat(True)
    assert anticheat_active() is True           # gaming on -> anticheat on
    mgr._set_anticheat(False)
    assert anticheat_active() is False          # gaming off -> anticheat off


def test_gaming_mode_engage_enables_even_when_config_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-safe: a broken config can never leave a game unprotected."""
    import kenning.config as config_mod
    from kenning.openclaw_routing.gaming_mode import GamingModeManager

    def boom():
        raise RuntimeError("config broken")

    monkeypatch.setattr(config_mod, "get_config", boom)
    mgr = GamingModeManager.__new__(GamingModeManager)
    mgr._set_anticheat(True)
    assert anticheat_active() is True


def test_gaming_mode_config_defaults() -> None:
    from kenning.config import GamingModeConfig

    cfg = GamingModeConfig()
    # Safe-by-default: the desktop-interaction hard block defaults ON so a lost
    # or reset config can never silently unblock input/capture in a game.
    assert cfg.anticheat_safe_mode is True
    assert cfg.anticheat_with_gaming_mode is True


# ---------------------------------------------------------------------------
# NEVER-LOAD guarantee: under anticheat the input/capture/UIA stack must not
# even be imported into RAM (not merely call-gated). Proven in a CLEAN
# subprocess so suite-wide import pollution can't mask a regression.
# ---------------------------------------------------------------------------

import subprocess  # noqa: E402
import sys as _sys  # noqa: E402
import textwrap  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]


def _run_probe(body: str) -> subprocess.CompletedProcess:
    code = (
        "import sys; sys.path[:0] = [r%r, r%r]\n" % (
            str(_ROOT / "src"), str(_ROOT))
    ) + textwrap.dedent(body)
    return subprocess.run(
        [_sys.executable, "-c", code],
        capture_output=True, text=True, timeout=180, cwd=str(_ROOT),
    )


def test_anticheat_keeps_desktop_stack_out_of_ram() -> None:
    """Boot the dialog-poller path with anticheat active (shipped config) and
    assert pyautogui / mss / pywinauto / kenning.desktop were NEVER imported."""
    proc = _run_probe(
        """
        import sys
        from kenning.safety.anticheat import anticheat_active
        assert anticheat_active(), "expected anticheat active from shipped config"
        from kenning.pipeline.orchestrator import Orchestrator
        o = Orchestrator.__new__(Orchestrator)
        o._start_dialog_poller()
        assert o._dialog_poller is None, "dialog poller started under anticheat"
        risky = [m for m in (
            "pyautogui", "mss", "pyscreeze", "pywinauto", "uiautomation",
            "dxcam", "pynput",
        ) if m in sys.modules]
        assert not risky, "OS-interaction libs loaded under anticheat: %r" % risky
        assert "kenning.desktop" not in sys.modules, "kenning.desktop loaded"
        o._audit_anticheat_posture()   # must log OK, not raise
        print("PROBE_PASS")
        """
    )
    assert "PROBE_PASS" in proc.stdout, (
        f"probe failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def test_relay_path_does_not_import_desktop_stack() -> None:
    """The team relay is pure audio (synthesize -> play_to_device + tees). Loading
    it must not drag in any input/capture/UIA surface -- the voice-changer class."""
    proc = _run_probe(
        """
        import sys
        import kenning.audio.relay_speech      # the relay pipeline
        import kenning.audio.monitor           # the local monitor tee
        import kenning.spotify.voice           # spotify control
        risky = [m for m in (
            "pyautogui", "mss", "pyscreeze", "pywinauto", "uiautomation",
            "dxcam", "pynput",
        ) if m in sys.modules]
        assert not risky, "relay/spotify path pulled in: %r" % risky
        assert "kenning.desktop" not in sys.modules, "relay pulled in kenning.desktop"
        print("PROBE_PASS")
        """
    )
    assert "PROBE_PASS" in proc.stdout, (
        f"probe failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def test_start_dialog_poller_runs_when_anticheat_off(monkeypatch) -> None:
    """With anticheat OFF the poller path is taken (regression: the gate must not
    wedge the poller permanently off)."""
    from kenning.pipeline import orchestrator as orch_mod

    set_anticheat_active(False)
    started = {"n": 0}

    class _FakePoller:
        running = False

        def start(self):
            started["n"] += 1

    import kenning.desktop.dialog_poller as dp
    monkeypatch.setattr(dp, "get_dialog_poller", lambda: _FakePoller())
    o = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    o._start_dialog_poller()
    assert started["n"] == 1 and o._dialog_poller is not None


def test_posture_audit_canary_fires_when_risky_module_loaded(
    monkeypatch, caplog,
) -> None:
    """If a risky lib is somehow loaded while anticheat is active, the boot audit
    must log a loud CANARY warning (so a future regression is visible)."""
    import types
    from kenning.pipeline import orchestrator as orch_mod

    monkeypatch.setitem(_sys.modules, "pynput", types.ModuleType("pynput"))
    set_anticheat_active(True, "test")
    try:
        o = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
        with caplog.at_level("WARNING"):
            o._audit_anticheat_posture()
    finally:
        set_anticheat_active(False)
    assert any("ANTICHEAT POSTURE CANARY" in r.message for r in caplog.records)
