"""Tests for the settings panel's logic layers (spec + launch).

The tkinter app layer is deliberately untested (no GUI windows in the
sweep); everything it depends on -- the knob catalogue, the comment-
preserving YAML patcher, value rendering, the reload signal, the voice
matcher, and the spawn/close lifecycle -- is exercised here hermetically.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from kenning.settings_gui.launch import (
    close_gui,
    launch_gui,
    match_settings_command,
)
from kenning.settings_gui.spec import (
    SECTIONS,
    apply_updates,
    patch_config_text,
    read_value,
    render_value,
    write_reload_signal,
)

from kenning.pipeline.orchestrator import Orchestrator

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"

SAMPLE = """\
# Top comment stays.
alpha:
  # alpha block comment
  enabled: true            # inline comment survives
  speed: 1.3
  name: "kokoro"
  nested:
    depth: 2
    flag: false
beta:
  enabled: false
  items: []
  # trailing comment
"""


# ---------------------------------------------------------------------------
# patch_config_text
# ---------------------------------------------------------------------------


def test_patch_bool_preserves_everything_else() -> None:
    out = patch_config_text(SAMPLE, ("alpha", "enabled"), "false")
    assert "enabled: false            # inline comment survives" in out
    # Every other line is byte-identical.
    diff = [
        (a, b) for a, b in zip(SAMPLE.splitlines(), out.splitlines())
        if a != b
    ]
    assert len(diff) == 1
    assert yaml.safe_load(out)["alpha"]["enabled"] is False


def test_patch_nested_key() -> None:
    out = patch_config_text(SAMPLE, ("alpha", "nested", "depth"), "5")
    data = yaml.safe_load(out)
    assert data["alpha"]["nested"]["depth"] == 5
    assert data["alpha"]["nested"]["flag"] is False


def test_patch_same_key_name_in_other_section_untouched() -> None:
    out = patch_config_text(SAMPLE, ("beta", "enabled"), "true")
    data = yaml.safe_load(out)
    assert data["beta"]["enabled"] is True
    assert data["alpha"]["enabled"] is True  # untouched


def test_patch_list_value() -> None:
    out = patch_config_text(SAMPLE, ("beta", "items"), '["sova", "sage"]')
    assert yaml.safe_load(out)["beta"]["items"] == ["sova", "sage"]


def test_patch_missing_path_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        patch_config_text(SAMPLE, ("alpha", "nope"), "1")
    with pytest.raises(KeyError):
        patch_config_text(SAMPLE, ("gamma", "enabled"), "1")


def test_patch_comments_and_blank_lines_do_not_terminate_blocks() -> None:
    text = "sec:\n  # comment\n\n  key: 1\nother:\n  key: 2\n"
    out = patch_config_text(text, ("sec", "key"), "9")
    data = yaml.safe_load(out)
    assert data["sec"]["key"] == 9 and data["other"]["key"] == 2


# ---------------------------------------------------------------------------
# render_value / read_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,kind,expected",
    [
        (True, "bool", "true"),
        (False, "bool", "false"),
        (120.0, "float", "120.0"),
        (1.25, "float", "1.25"),
        (280, "int", "280"),
        ("Voicemeeter Aux Input", "str", '"Voicemeeter Aux Input"'),
        ("kokoro", "choice", '"kokoro"'),
        ("sova, sage", "csv", '["sova", "sage"]'),
        ([], "csv", "[]"),
        (["a"], "csv", '["a"]'),
    ],
)
def test_render_value(value, kind, expected) -> None:
    assert render_value(value, kind) == expected


def test_read_value_paths() -> None:
    data = {"a": {"b": {"c": 3}}}
    assert read_value(data, ("a", "b", "c")) == 3
    assert read_value(data, ("a", "x")) is None


# ---------------------------------------------------------------------------
# The knob catalogue stays in sync with the REAL config.yaml
# ---------------------------------------------------------------------------


def test_every_knob_path_exists_in_real_config() -> None:
    """Drift guard: a renamed/removed config key must fail loudly here,
    not silently render an empty field in the panel."""
    data = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
    missing = [
        ".".join(knob.path)
        for section in SECTIONS for knob in section.knobs
        if read_value(data, knob.path) is None
        and knob.kind not in ("csv", "str")  # genuinely-empty allowed
    ]
    assert not missing, f"knobs missing from config.yaml: {missing}"


def test_every_knob_patches_real_config_roundtrip(tmp_path: Path) -> None:
    """Every catalogued knob can be patched in a COPY of the real
    config.yaml with its current value re-rendered, and the file still
    parses to the same data."""
    original = REPO_CONFIG.read_text(encoding="utf-8")
    data = yaml.safe_load(original)
    text = original
    for section in SECTIONS:
        for knob in section.knobs:
            current = read_value(data, knob.path)
            if current is None and knob.kind not in ("csv",):
                continue
            text = patch_config_text(
                text, knob.path, render_value(current, knob.kind),
            )
    after = yaml.safe_load(text)
    assert after == data


# ---------------------------------------------------------------------------
# apply_updates
# ---------------------------------------------------------------------------


def test_apply_updates_atomic_write_and_validation(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(SAMPLE, encoding="utf-8")
    apply_updates(cfg, {
        ("alpha", "speed"): "1.1",
        ("alpha", "nested", "flag"): "true",
    })
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["alpha"]["speed"] == 1.1
    assert data["alpha"]["nested"]["flag"] is True
    assert "# Top comment stays." in cfg.read_text(encoding="utf-8")


def test_apply_updates_noop_on_empty(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(SAMPLE, encoding="utf-8")
    apply_updates(cfg, {})
    assert cfg.read_text(encoding="utf-8") == SAMPLE


def test_write_reload_signal(tmp_path: Path) -> None:
    signal = write_reload_signal(tmp_path)
    assert signal.is_file()
    assert signal.name == "config_reload.signal"


# ---------------------------------------------------------------------------
# Voice matcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Pull up your settings.", "open"),
        ("open the settings panel", "open"),
        ("Show me the control panel.", "open"),
        ("bring up your settings menu", "open"),
        ("open your knobs", "open"),
        ("launch the configuration panel", "open"),
        ("Close the settings.", "close"),
        ("close the control panel", "close"),
        ("hide your settings panel", "close"),
        # Negatives: mentions without the strict command frame.
        ("what are your settings?", None),
        ("the control panel looks nice", None),
        ("open the door", None),
        ("close the window", None),
        ("change your settings to be louder", None),
        ("", None),
    ],
)
def test_match_settings_command(text: str, expected) -> None:
    assert match_settings_command(text) == expected


# ---------------------------------------------------------------------------
# launch / close lifecycle
# ---------------------------------------------------------------------------


def test_launch_gui_spawns_without_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    calls: list[dict] = []

    def fake_popen(argv, **kwargs):
        calls.append({"argv": argv, **kwargs})
        return SimpleNamespace(pid=4321)

    pid = launch_gui(spawn_fn=fake_popen)
    assert pid == 4321
    (call,) = calls
    assert call["argv"][-2:] == ["-m", "kenning.settings_gui"]
    # No console window may EVER appear: either the GUI-subsystem
    # pythonw.exe interpreter, or CREATE_NO_WINDOW on python.exe.
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert (
        call["argv"][0].lower().endswith("pythonw.exe")
        or (call["creationflags"] & no_window)
    )
    # Never DETACHED_PROCESS (it allocates a fresh visible console
    # for console-subsystem interpreters).
    detached = getattr(subprocess, "DETACHED_PROCESS", 0)
    assert not (call["creationflags"] & detached)


def test_launch_gui_fail_open() -> None:
    def boom(argv, **kwargs):
        raise OSError("no python")

    assert launch_gui(spawn_fn=boom) is None


def test_close_gui() -> None:
    killed: list[int] = []
    assert close_gui(123, kill_fn=killed.append) is True
    assert killed == [123]
    assert close_gui(None, kill_fn=killed.append) is False
    assert close_gui(0, kill_fn=killed.append) is False


def test_close_gui_fail_open() -> None:
    def boom(pid):
        raise OSError("gone")

    assert close_gui(99, kill_fn=boom) is False


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------


def _bare_orchestrator():
    o = Orchestrator.__new__(Orchestrator)
    o._spoken = []
    o._speak = lambda text: o._spoken.append(text)  # type: ignore[attr-defined]
    return o


def test_orchestrator_settings_open(monkeypatch: pytest.MonkeyPatch) -> None:
    import kenning.settings_gui.launch as launch_mod

    o = _bare_orchestrator()
    monkeypatch.setattr(launch_mod, "launch_gui", lambda **kw: 777)
    assert o._maybe_handle_settings_gui("pull up your settings") is True
    assert o._settings_gui_pid == 777
    assert o._spoken == ["Control panel is up."]


def test_orchestrator_settings_open_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kenning.settings_gui.launch as launch_mod

    o = _bare_orchestrator()
    monkeypatch.setattr(launch_mod, "launch_gui", lambda **kw: None)
    assert o._maybe_handle_settings_gui("open the settings") is True
    assert "couldn't open" in o._spoken[0]


def test_orchestrator_settings_close(monkeypatch: pytest.MonkeyPatch) -> None:
    import kenning.settings_gui.launch as launch_mod

    closed: list = []
    o = _bare_orchestrator()
    o._settings_gui_pid = 555
    monkeypatch.setattr(
        launch_mod, "close_gui",
        lambda pid, **kw: closed.append(pid) or True,
    )
    assert o._maybe_handle_settings_gui("close the settings") is True
    assert closed == [555]
    assert o._settings_gui_pid is None
    assert o._spoken == ["Closed."]


def test_orchestrator_settings_no_match_falls_through() -> None:
    o = _bare_orchestrator()
    assert o._maybe_handle_settings_gui("what time is it") is False
    assert o._spoken == []


# ---------------------------------------------------------------------------
# Config hot reload
# ---------------------------------------------------------------------------


def test_maybe_reload_config_triggers_on_new_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import kenning.config as config_mod

    reloads: list[int] = []
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        config_mod, "reload_config", lambda: reloads.append(1),
    )
    o = _bare_orchestrator()

    # No signal file -> no-op.
    o._maybe_reload_config()
    assert reloads == []

    # Pre-existing signal on first sight -> recorded, NOT triggered
    # (a stale file from a prior session must not fire).
    signal = write_reload_signal(tmp_path / "data")
    o._maybe_reload_config()
    assert reloads == []

    # A NEWER signal -> reload + spoken ack.
    import os

    os.utime(signal, (os.path.getmtime(signal) + 5,) * 2)
    o._maybe_reload_config()
    assert reloads == [1]
    assert o._spoken == ["Settings updated."]

    # Same mtime again -> no double fire.
    o._maybe_reload_config()
    assert reloads == [1]


# ---------------------------------------------------------------------------
# Runtime-action channel (hot toggles: gaming mode / preset / device)
# ---------------------------------------------------------------------------


def test_write_action_appends_jsonl(tmp_path: Path) -> None:
    import json as _json

    from kenning.settings_gui.spec import ACTION_RELPATH, write_action

    write_action(tmp_path, "gaming_mode", True)
    write_action(tmp_path, "llm_preset", "qwen3.5-4b")
    path = tmp_path / ACTION_RELPATH
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    r0, r1 = _json.loads(lines[0]), _json.loads(lines[1])
    assert r0["action"] == "gaming_mode" and r0["value"] is True
    assert r1["action"] == "llm_preset" and r1["value"] == "qwen3.5-4b"


def test_every_action_knob_has_known_action() -> None:
    """Action knobs must use an action the orchestrator dispatches."""
    known = {"gaming_mode", "llm_preset", "kokoro_device", "wake_word"}
    for section in SECTIONS:
        for knob in section.knobs:
            if knob.action is not None:
                assert knob.action in known, knob.path


def test_no_knob_requires_restart() -> None:
    """The user contract: every exposed knob is hot (call-time or action)."""
    for section in SECTIONS:
        for knob in section.knobs:
            assert knob.restart is False, f"{knob.path} still marked restart"


# ---------------------------------------------------------------------------
# Orchestrator drain of the action channel
# ---------------------------------------------------------------------------


def _orch():
    o = Orchestrator.__new__(Orchestrator)
    o._spoken = []
    o._speak = lambda t: o._spoken.append(t)  # type: ignore[attr-defined]
    return o


def test_drain_gui_actions_dispatches(monkeypatch, tmp_path) -> None:
    import kenning.config as config_mod
    from kenning.settings_gui.spec import write_action

    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    data = tmp_path / "data"
    o = _orch()

    applied: list = []
    o._apply_gui_action = lambda a, v: applied.append((a, v))  # type: ignore

    # First sight of an existing file: skip history, fire nothing.
    write_action(data, "gaming_mode", True)
    o._drain_gui_actions()
    assert applied == []

    # New lines after the offset fire once each.
    write_action(data, "llm_preset", "qwen3.5-4b")
    write_action(data, "kokoro_device", "cpu")
    o._drain_gui_actions()
    assert applied == [("llm_preset", "qwen3.5-4b"), ("kokoro_device", "cpu")]

    # Re-drain with no new lines: nothing fires again.
    o._drain_gui_actions()
    assert len(applied) == 2


def test_apply_gui_action_gaming_mode(monkeypatch) -> None:
    o = _orch()
    calls: list = []

    class _Mgr:
        async def engage(self):
            calls.append("engage")

        async def disengage(self):
            calls.append("disengage")

    o._resolve_gaming_mode_manager = lambda: _Mgr()  # type: ignore
    o._apply_gui_action("gaming_mode", True)
    o._apply_gui_action("gaming_mode", False)
    assert calls == ["engage", "disengage"]
    assert any("engaged" in s.lower() for s in o._spoken)
    assert any("off" in s.lower() for s in o._spoken)


def test_apply_gui_action_llm_preset() -> None:
    o = _orch()
    swaps: list = []
    o.llm = SimpleNamespace(
        reload_for_preset=lambda p: (swaps.append(p) or (True, "ok")))
    o._apply_gui_action("llm_preset", "qwen3.5-9b")
    assert swaps == ["qwen3.5-9b"]


def test_apply_gui_action_kokoro_device() -> None:
    o = _orch()
    moves: list = []
    o.tts = SimpleNamespace(move_to_device=moves.append)
    o._apply_gui_action("kokoro_device", "cpu")
    assert moves == ["cpu"]


# ---------------------------------------------------------------------------
# Dynamic choices provider (relay output-device dropdown, 2026-06-12)
# ---------------------------------------------------------------------------


def test_relay_output_device_knob_is_provider_dropdown() -> None:
    knob = next(
        k for s in SECTIONS for k in s.knobs
        if k.path == ("relay_speech", "output_device")
    )
    assert knob.kind == "choice"
    assert knob.choices == ()  # dynamic, not static
    assert knob.choices_provider == "output_devices"
    # Hot via call-time config read -- no runtime action, no restart.
    assert knob.action is None
    assert knob.restart is False


def test_output_device_names_enumerates_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kenning.settings_gui.spec as spec_mod

    fake_devices = [
        {"name": "Speakers (Realtek)", "max_output_channels": 2},
        {"name": "Microphone (USB)", "max_output_channels": 0},  # input only
        {"name": "Voicemeeter Input", "max_output_channels": 8},
        {"name": "Voicemeeter Aux Input", "max_output_channels": 8},
        {"name": "Speakers (Realtek)", "max_output_channels": 2},  # dup
        {"name": "", "max_output_channels": 2},  # nameless -> skipped
    ]
    import sys

    monkeypatch.setitem(
        sys.modules, "sounddevice",
        SimpleNamespace(query_devices=lambda: fake_devices),
    )
    names = spec_mod.output_device_names()
    assert names == (
        "Speakers (Realtek)", "Voicemeeter Input", "Voicemeeter Aux Input",
    )


def test_output_device_names_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    import kenning.settings_gui.spec as spec_mod

    def boom() -> list:
        raise RuntimeError("PortAudio exploded")

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(query_devices=boom),
    )
    assert spec_mod.output_device_names() == ()


def test_resolve_choices_static_wins() -> None:
    from kenning.settings_gui.spec import Knob, resolve_choices

    knob = Knob(("tts", "kokoro", "device"), "Device", "choice",
                choices=("cuda", "cpu"))
    assert resolve_choices(knob, "cuda") == ("cuda", "cpu")


def test_resolve_choices_provider_includes_current_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kenning.settings_gui.spec as spec_mod
    from kenning.settings_gui.spec import Knob

    monkeypatch.setattr(
        spec_mod, "output_device_names",
        lambda: ("Speakers (Realtek)", "Voicemeeter Input"),
    )
    knob = Knob(("relay_speech", "output_device"), "Output device", "choice",
                choices_provider="output_devices")
    # Configured device currently unplugged -> still selectable.
    got = spec_mod.resolve_choices(knob, "Voicemeeter Aux Input")
    assert got[0] == "Voicemeeter Aux Input"
    assert "Voicemeeter Input" in got


def test_resolve_choices_provider_failure_keeps_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kenning.settings_gui.spec as spec_mod
    from kenning.settings_gui.spec import Knob

    monkeypatch.setattr(spec_mod, "output_device_names", lambda: ())
    knob = Knob(("relay_speech", "output_device"), "Output device", "choice",
                choices_provider="output_devices")
    assert spec_mod.resolve_choices(knob, "Voicemeeter Aux Input") == (
        "Voicemeeter Aux Input",
    )
