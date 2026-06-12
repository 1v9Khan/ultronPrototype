"""Knob catalogue + comment-preserving config.yaml patcher.

The GUI renders :data:`SECTIONS` -- a curated, at-a-glance subset of
``config.yaml`` (the full file is ~1600 heavily-commented lines; the
panel shows the knobs a user actually turns). Each knob carries its
YAML path, a widget kind, optional bounds/choices, and whether changing
it requires a process restart (engine/model/device construction-time
settings) vs hot-applying via the reload signal.

``patch_config_text`` edits ONE value in the raw YAML text while
preserving every comment and untouched line byte-for-byte -- PyYAML
round-tripping would destroy the file's documentation, so the patcher
is a small indent-aware block scanner instead. ``apply_updates``
patches + validates (the result must still parse, and must differ from
the original ONLY at the requested paths) before atomically writing.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

__all__ = [
    "Knob",
    "Section",
    "SECTIONS",
    "RELOAD_SIGNAL_RELPATH",
    "ACTION_RELPATH",
    "read_value",
    "render_value",
    "patch_config_text",
    "apply_updates",
    "write_reload_signal",
    "write_action",
]

# The orchestrator watches this file (relative to the repo data dir);
# the GUI touches it after a successful apply to request a hot reload.
RELOAD_SIGNAL_RELPATH = "config_reload.signal"
# Runtime-action channel: the GUI appends one JSON line per action
# (gaming-mode toggle, LLM preset swap, Kokoro device move); the
# orchestrator drains it at its idle poll point and applies each live.
ACTION_RELPATH = "gui_action.jsonl"


@dataclass(frozen=True)
class Knob:
    """One editable setting.

    Attributes:
        path: YAML path, e.g. ``("tts", "kokoro", "speed")``.
        label: short human label shown in the panel.
        kind: ``bool`` / ``int`` / ``float`` / ``str`` / ``choice`` /
            ``csv`` (a YAML list edited as comma-separated text).
        choices: allowed values for ``choice`` knobs.
        minimum / maximum: numeric bounds (display + validation).
        restart: True when the value is read at construction time and
            only applies on the next Ultron start. (No knob in the
            shipped catalogue sets this -- every exposed knob is hot,
            either call-time or via ``action``.)
        action: runtime action the orchestrator fires on apply for a
            value that isn't read call-time (``"llm_preset"`` reloads
            the model; ``"kokoro_device"`` moves the TTS engine). The
            config is still patched so the change survives a restart;
            the action makes it take effect live.
        help: one-line tooltip text.
    """

    path: tuple[str, ...]
    label: str
    kind: str
    choices: tuple[str, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    restart: bool = False
    action: Optional[str] = None
    help: str = ""


@dataclass(frozen=True)
class Section:
    """A card in the panel: a titled group of related knobs."""

    title: str
    knobs: tuple[Knob, ...] = field(default_factory=tuple)


SECTIONS: tuple[Section, ...] = (
    Section("Game Chat Relay", (
        Knob(("relay_speech", "enabled"), "Enabled", "bool",
             help="Voice relay into the game voice chat"),
        Knob(("relay_speech", "output_device"), "Output device", "str",
             help="Audio device the relay speaks on (VoiceMeeter strip)"),
        Knob(("relay_speech", "rephrase"), "LLM rephrase", "bool",
             help="Convert reported speech into a natural direct line"),
        Knob(("relay_speech", "echo_to_user"), "Echo to me", "bool",
             help="Also play relay lines on the normal output"),
        Knob(("relay_speech", "follow_up_seconds"), "Conversation window (s)",
             "float", minimum=0, maximum=600,
             help="How long relays keep listening without the wake word"),
        Knob(("relay_speech", "max_line_chars"), "Max line length", "int",
             minimum=40, maximum=600),
        Knob(("relay_speech", "addressee_names"), "Extra names", "csv",
             help="Comma-separated callout names beyond the agent roster"),
    )),
    Section("Voice", (
        # Kokoro device move is hot (gaming mode already swaps it live).
        Knob(("tts", "kokoro", "device"), "Kokoro device", "choice",
             choices=("cuda", "cpu"), action="kokoro_device",
             help="Move the TTS engine between GPU and CPU live"),
        Knob(("tts", "kokoro", "speed"), "Speech speed", "float",
             minimum=0.8, maximum=1.3),
        Knob(("tts", "pause_ms"), "Sentence pause (ms)", "int",
             minimum=0, maximum=1000),
        Knob(("tts", "output_watch", "enabled"), "Blip watcher", "bool",
             help="Analyze every clip for audio artifacts"),
        Knob(("tts", "output_watch", "waveform_enabled"), "Waveform pane",
             "bool"),
    )),
    Section("Hearing", (
        Knob(("wake_word", "threshold"), "Wake-word threshold", "float",
             minimum=0.0, maximum=1.0),
        Knob(("vad", "threshold"), "VAD threshold", "float",
             minimum=0.0, maximum=1.0),
        Knob(("vad", "min_silence_duration_ms"), "End-of-turn silence (ms)",
             "int", minimum=100, maximum=5000),
        Knob(("audio", "barge_in_enabled"), "Barge-in", "bool",
             help="Interrupt Ultron by speaking over him"),
    )),
    Section("Brain", (
        # Preset swap is hot via reload_for_preset (the gaming-mode
        # LLM swap proves it's safe at idle).
        Knob(("llm", "preset"), "Model preset", "str", action="llm_preset",
             help="Hot-swaps the local model"),
        Knob(("llm", "default_temperature"), "Temperature", "float",
             minimum=0.0, maximum=2.0),
        Knob(("llm", "default_max_tokens"), "Max tokens", "int",
             minimum=64, maximum=4096),
        Knob(("llm", "history_turns"), "History turns", "int",
             minimum=0, maximum=24),
    )),
    Section("Addressing", (
        Knob(("addressing", "follow_up_enabled"), "Follow-up window", "bool"),
        Knob(("addressing", "warm_mode_duration_seconds"),
             "Window length (s)", "float", minimum=5, maximum=300),
        Knob(("addressing", "zero_shot_addressed_min_confidence"),
             "Addressed threshold", "float", minimum=0.5, maximum=1.0,
             help="Lower = more permissive follow-up acceptance"),
    )),
    Section("Web Search", (
        Knob(("web_search", "enabled"), "Enabled", "bool"),
        Knob(("web_search", "query_reformulation", "enabled"),
             "Query reformulation", "bool"),
    )),
    Section("Evolution", (
        Knob(("evolution", "enabled"), "Self-improvement", "bool"),
        Knob(("evolution", "cycle_check_interval_turns"),
             "Cycle interval (turns)", "int", minimum=5, maximum=500),
        Knob(("evolution", "guardrail_monitoring_enabled"),
             "Guardrail brake", "bool"),
        Knob(("evolution", "pre_turn_nudge_enabled"), "Pre-turn nudge",
             "bool"),
    )),
    Section("Coding", (
        Knob(("coding", "enabled"), "Voice coding", "bool"),
        Knob(("coding", "default_model"), "Default model", "str"),
        Knob(("coding", "pre_task_confirmation_enabled"),
             "Confirm before tasks", "bool"),
    )),
    Section("Desktop & Research", (
        Knob(("desktop", "enabled"), "Desktop control", "bool"),
        Knob(("desktop", "deep_ui_discovery_enabled"),
             "Deep UI discovery", "bool"),
        Knob(("desktop", "click_preview", "enabled"), "Click preview",
             "bool"),
        Knob(("deep_research", "enabled"), "Deep research", "bool"),
    )),
    Section("Gaming / Anticheat", (
        Knob(("gaming_mode", "enabled"), "Gaming mode voice trigger",
             "bool"),
        Knob(("gaming_mode", "llm_preset"), "Gaming LLM preset", "str"),
        Knob(("gaming_mode", "toggle_docker"), "Stop Docker in game",
             "bool", help="Free Docker/WSL RAM while gaming"),
    )),
)


def read_value(config_data: dict, path: Sequence[str]) -> Any:
    """Read a knob value from parsed config data (None when absent)."""
    node: Any = config_data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def render_value(value: Any, kind: str) -> str:
    """Render a python value as the YAML scalar text the patcher writes."""
    if kind == "bool":
        return "true" if value else "false"
    if kind == "csv":
        items = [v.strip() for v in (value or [])] if isinstance(value, list) \
            else [v.strip() for v in str(value or "").split(",") if v.strip()]
        return "[" + ", ".join(json.dumps(v) for v in items) + "]"
    if kind in ("str", "choice"):
        text = str(value)
        # Quote strings: safe against colons/specials, matches the
        # file's prevailing style for device names / models.
        return json.dumps(text)
    if kind == "int":
        return str(int(value))
    if kind == "float":
        out = f"{float(value):g}"
        return out if ("." in out or "e" in out) else out + ".0"
    raise ValueError(f"unknown knob kind: {kind}")


def _block_range(
    lines: list[str], start: int, end: int, key: str, indent: int,
) -> Optional[tuple[int, int]]:
    """Find ``key:`` at exactly ``indent`` within lines[start:end].

    Returns (key_line_index, block_end_index) where block_end is the
    first subsequent line at indent <= the key's indent that starts a
    new mapping entry (comments/blank lines never terminate a block).
    """
    pat = re.compile(rf"^{' ' * indent}{re.escape(key)}\s*:")
    for i in range(start, end):
        if not pat.match(lines[i]):
            continue
        block_end = end
        for j in range(i + 1, end):
            line = lines[j]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            cur_indent = len(line) - len(line.lstrip(" "))
            if cur_indent <= indent:
                block_end = j
                break
        return i, block_end
    return None


def patch_config_text(
    text: str, path: Sequence[str], rendered_value: str,
) -> str:
    """Replace one scalar value in raw YAML text, preserving everything
    else byte-for-byte (comments, ordering, untouched lines).

    Args:
        text: full config.yaml content.
        path: YAML path of the key to change.
        rendered_value: the replacement scalar text (see
            :func:`render_value`).

    Returns:
        The patched text.

    Raises:
        KeyError: when the path cannot be located in the file.
    """
    lines = text.splitlines(keepends=True)
    start, end, indent = 0, len(lines), 0
    for depth, key in enumerate(path):
        found = _block_range(lines, start, end, key, indent)
        if found is None:
            raise KeyError(
                f"config.yaml: cannot locate {'.'.join(path)!r} "
                f"(missing {key!r} at indent {indent})"
            )
        key_line, block_end = found
        if depth == len(path) - 1:
            line = lines[key_line]
            m = re.match(
                rf"^(?P<head>\s*{re.escape(key)}\s*:\s*)"
                rf"(?P<value>[^#\r\n]*?)(?P<tail>\s*(?:#[^\r\n]*)?\r?\n?)$",
                line,
            )
            if m is None:  # pragma: no cover - defensive
                raise KeyError(f"unparseable line for {'.'.join(path)!r}")
            lines[key_line] = m.group("head") + rendered_value + m.group("tail")
            return "".join(lines)
        start, end, indent = key_line + 1, block_end, indent + 2
    raise KeyError(f"empty path")  # pragma: no cover - guarded by caller


def apply_updates(
    config_path: Path, updates: dict[tuple[str, ...], str],
) -> None:
    """Patch + validate + atomically write ``config.yaml``.

    Args:
        config_path: the YAML file to edit.
        updates: ``{path: rendered_value}`` for every changed knob.

    Raises:
        KeyError: a path could not be located.
        ValueError: the patched file no longer parses, or differs from
            the original anywhere OTHER than the requested paths.
    """
    if not updates:
        return
    original = config_path.read_text(encoding="utf-8")
    patched = original
    for path, rendered in updates.items():
        patched = patch_config_text(patched, path, rendered)

    before = yaml.safe_load(original)
    after = yaml.safe_load(patched)  # raises on broken YAML

    def scrub(data: dict, paths: Sequence[Sequence[str]]) -> dict:
        clone = json.loads(json.dumps(data, default=str))
        for p in paths:
            node = clone
            for key in p[:-1]:
                node = node.get(key, {})
            node.pop(p[-1], None)
        return clone

    if scrub(before, list(updates)) != scrub(after, list(updates)):
        raise ValueError(
            "config patch altered keys outside the requested set; aborting"
        )

    tmp = config_path.with_suffix(".yaml.tmp")
    tmp.write_text(patched, encoding="utf-8", newline="")
    tmp.replace(config_path)


def write_reload_signal(data_dir: Path) -> Path:
    """Touch the hot-reload signal file the orchestrator watches."""
    signal = data_dir / RELOAD_SIGNAL_RELPATH
    signal.parent.mkdir(parents=True, exist_ok=True)
    signal.write_text(str(time.time()), encoding="utf-8")
    return signal


def write_action(data_dir: Path, action: str, value: Any) -> Path:
    """Append one runtime-action request to the action channel.

    Args:
        data_dir: the repo data dir (holds the channel file).
        action: action name (``"gaming_mode"`` / ``"llm_preset"`` /
            ``"kokoro_device"``).
        value: the action payload (bool / str).

    Returns:
        The action-channel path.
    """
    path = data_dir / ACTION_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), "action": action, "value": value}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return path
