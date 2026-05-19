"""Tests for :mod:`ultron.coding.voice_lock`."""

from __future__ import annotations

import pytest

from ultron.coding.voice_lock import (
    DEFAULT_VOICE_LOCKED_GLOBS,
    DEFAULT_VOICE_LOCKED_PATHS,
    FileChangeScanResult,
    VoiceLockHit,
    is_voice_locked_path,
    render_warning_for_voice,
    scan_file_change,
    scan_prompt,
)


# ---------------------------------------------------------------------------
# is_voice_locked_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "src/ultron/tts/speech.py",
        "src\\ultron\\tts\\speech.py",
        "C:/STC/ultronPrototype/src/ultron/tts/speech.py",
        "C:\\STC\\ultronPrototype\\src\\ultron\\tts\\rvc.py",
        "models/piper/en_US-ryan-medium.onnx",
        "models/rvc/hubert_base.pt",
        "ultron_james_spader_mcu_6941/Ultron.pth",
        "ultron_james_spader_mcu_6941/added_IVF301_Flat_nprobe_1_Ultron_v2.index",
        "ultronVoiceAudio/Ultron_vocals_mono_v1.wav",
    ],
)
def test_default_paths_are_locked(path: str) -> None:
    assert is_voice_locked_path(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "src/ultron/tts/xtts_v3.py",  # NOT locked -- explicitly iterable zone
        "src/ultron/tts/ultron_filter.py",  # iterable zone
        "src/ultron/pipeline/orchestrator.py",
        "src/ultron/coding/runner.py",
        "tests/test_audio.py",
        "models/Qwen3.5-4B-Q4_K_M.gguf",
        "data/projects.json",
        "config.yaml",
        "",  # empty
    ],
)
def test_unrelated_paths_are_not_locked(path: str) -> None:
    assert is_voice_locked_path(path) is None


def test_extra_paths_extend_the_locked_set() -> None:
    assert is_voice_locked_path("src/ultron/tts/xtts_v3.py") is None
    pattern = is_voice_locked_path(
        "src/ultron/tts/xtts_v3.py",
        extra_paths=["src/ultron/tts/xtts_v3.py"],
    )
    assert pattern == "src/ultron/tts/xtts_v3.py"


def test_extra_globs_extend_the_locked_set() -> None:
    assert is_voice_locked_path("custom/voice/foo.bin") is None
    pattern = is_voice_locked_path(
        "custom/voice/foo.bin",
        extra_globs=["custom/voice/*"],
    )
    assert pattern == "custom/voice/*"


def test_default_constants_are_non_empty() -> None:
    assert DEFAULT_VOICE_LOCKED_PATHS, "default locked paths should not be empty"
    assert DEFAULT_VOICE_LOCKED_GLOBS, "default locked globs should not be empty"


# ---------------------------------------------------------------------------
# scan_prompt
# ---------------------------------------------------------------------------


def test_scan_prompt_empty_returns_empty() -> None:
    assert scan_prompt("") == []
    assert scan_prompt(None) == []  # type: ignore[arg-type]


def test_scan_prompt_finds_locked_path_inline() -> None:
    prompt = "Please refactor src/ultron/tts/speech.py to be cleaner."
    hits = scan_prompt(prompt)
    assert len(hits) == 1
    assert hits[0].path.endswith("speech.py")
    assert "voice-locked pattern" in hits[0].reason


def test_scan_prompt_dedups_repeated_mentions() -> None:
    prompt = (
        "Open src/ultron/tts/rvc.py, also re-open ./src/ultron/tts/rvc.py "
        "and check src/ultron/tts/rvc.py once more."
    )
    hits = scan_prompt(prompt)
    assert len(hits) == 1


def test_scan_prompt_finds_glob_matches() -> None:
    prompt = (
        "Update models/piper/en_US-ryan-medium.onnx to the new voice."
    )
    hits = scan_prompt(prompt)
    assert len(hits) == 1
    assert hits[0].matched_pattern in {"models/piper/**", "models/piper/*"}


def test_scan_prompt_ignores_unrelated_paths() -> None:
    prompt = (
        "Refactor src/ultron/pipeline/orchestrator.py "
        "and update tests/test_audio.py and data/projects.json."
    )
    hits = scan_prompt(prompt)
    assert hits == []


def test_scan_prompt_handles_windows_paths() -> None:
    prompt = (
        r"Update C:\STC\ultronPrototype\src\ultron\tts\rvc.py to use the "
        "new pitch settings."
    )
    hits = scan_prompt(prompt)
    assert len(hits) >= 1
    assert any("rvc.py" in h.path for h in hits)


def test_scan_prompt_finds_multiple_distinct_hits() -> None:
    prompt = (
        "Modify src/ultron/tts/speech.py and also touch "
        "ultronVoiceAudio/Ultron_vocals_mono_v1.wav."
    )
    hits = scan_prompt(prompt)
    paths = {h.path for h in hits}
    # The scanner returns each distinct match once.
    assert any("speech.py" in p for p in paths)
    assert any("Ultron_vocals_mono_v1.wav" in p for p in paths)


# ---------------------------------------------------------------------------
# scan_file_change
# ---------------------------------------------------------------------------


def test_scan_file_change_locked_path_blocks() -> None:
    result = scan_file_change("src/ultron/tts/speech.py")
    assert result.blocked is True
    assert result.hit is not None
    assert "FILE_CHANGE targeted voice-locked path" in result.hit.reason


def test_scan_file_change_safe_path_does_not_block() -> None:
    result = scan_file_change("src/ultron/pipeline/orchestrator.py")
    assert result.blocked is False
    assert result.hit is None


def test_scan_file_change_with_extras() -> None:
    result = scan_file_change(
        "custom/voice/foo.bin",
        extra_globs=["custom/voice/*"],
    )
    assert result.blocked is True
    assert result.hit is not None


# ---------------------------------------------------------------------------
# render_warning_for_voice
# ---------------------------------------------------------------------------


def test_render_warning_empty_is_empty_string() -> None:
    assert render_warning_for_voice([]) == ""


def test_render_warning_single_hit_names_leaf_only() -> None:
    msg = render_warning_for_voice(
        [VoiceLockHit("src/ultron/tts/speech.py", "src/ultron/tts/speech.py", "x")]
    )
    assert "speech.py" in msg
    # The whole path must not appear -- voice character lock fix
    # (2026-05-11) requires we never speak Windows paths or long slugs.
    assert "/" not in msg
    assert "\\" not in msg


def test_render_warning_multiple_hits_joins_names() -> None:
    hits = [
        VoiceLockHit("src/ultron/tts/speech.py", "src/ultron/tts/speech.py", "x"),
        VoiceLockHit("models/rvc/hubert_base.pt", "models/rvc/**", "y"),
    ]
    msg = render_warning_for_voice(hits)
    assert "speech.py" in msg
    assert "hubert_base.pt" in msg
    assert "/" not in msg
    assert "\\" not in msg


def test_render_warning_avoids_drive_letters() -> None:
    msg = render_warning_for_voice(
        [VoiceLockHit("C:\\STC\\ultronPrototype\\src\\ultron\\tts\\rvc.py", "src/ultron/tts/rvc.py", "x")]
    )
    assert "C:" not in msg
    assert "STC" not in msg
    assert "rvc.py" in msg


# ---------------------------------------------------------------------------
# VoiceLockHit shape
# ---------------------------------------------------------------------------


def test_hit_as_dict_carries_fields() -> None:
    hit = VoiceLockHit("p", "pattern", "reason")
    payload = hit.as_dict()
    assert payload == {"path": "p", "matched_pattern": "pattern", "reason": "reason"}


def test_hit_is_frozen() -> None:
    hit = VoiceLockHit("p", "pattern", "reason")
    with pytest.raises(Exception):  # FrozenInstanceError
        hit.path = "other"  # type: ignore[misc]
