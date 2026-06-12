"""Tests for the TTS output-quality watcher (``ultron.audio.output_quality``).

Hermetic: all clips are synthetic numpy waveforms; the watcher's JSONL
sink writes to ``tmp_path``; threads are joined via ``close()`` (binding
rule: thread cleanup); no audio device or voice stack is touched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from ultron.audio.output_quality import (
    OutputQualityWatcher,
    analyze_clip,
    get_output_watcher,
    reset_output_watcher,
    set_output_watcher_enabled,
)

SR = 24000


def _fade(x: np.ndarray, ms: float = 10.0) -> np.ndarray:
    """Apply linear fade-in/out so a clip is edge-clean."""
    n = int(SR * ms / 1000.0)
    out = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def _speech_like(seconds: float = 1.0, amp: float = 0.4) -> np.ndarray:
    """A faded sine burst standing in for a clean speech clip."""
    t = np.arange(int(SR * seconds), dtype=np.float32) / SR
    return _fade((amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32))


def _pad(x: np.ndarray, lead_ms: float = 100.0, tail_ms: float = 100.0) -> np.ndarray:
    lead = np.zeros(int(SR * lead_ms / 1000.0), dtype=np.float32)
    tail = np.zeros(int(SR * tail_ms / 1000.0), dtype=np.float32)
    return np.concatenate([lead, x, tail])


# ---------------------------------------------------------------------------
# analyze_clip detectors
# ---------------------------------------------------------------------------


def test_clean_clip_has_no_findings() -> None:
    report = analyze_clip(_pad(_speech_like()), SR, label="clean")
    assert report.clean, [f.detail for f in report.findings]
    assert report.duration_s > 1.0
    assert 0.3 < report.peak <= 1.0


def test_clean_int16_clip_has_no_findings() -> None:
    pcm = (_pad(_speech_like()) * 32767.0).astype(np.int16)
    report = analyze_clip(pcm, SR)
    assert report.clean, [f.detail for f in report.findings]


def test_empty_and_tiny_clips_are_skipped() -> None:
    assert analyze_clip(np.zeros(0, dtype=np.int16), SR).clean
    assert analyze_clip(np.ones(100, dtype=np.float32) * 0.5, SR).clean  # <50ms


def test_hard_onset_detected() -> None:
    x = _pad(_speech_like(), lead_ms=0.0)  # speech starts at sample 0...
    x = x.copy()
    x[:200] = 0.5  # ...at high amplitude, no fade
    report = analyze_clip(x, SR)
    kinds = {f.kind for f in report.findings}
    assert "hard_onset" in kinds


def test_hard_tail_detected() -> None:
    x = _pad(_speech_like(), tail_ms=0.0).copy()
    x[-200:] = 0.5
    report = analyze_clip(x, SR)
    kinds = {f.kind for f in report.findings}
    assert "hard_tail" in kinds


def test_leading_isolated_burst_detected() -> None:
    """The classic fine-tune blip: a short noise burst, then silence,
    then the actual speech."""
    burst = _fade(np.random.default_rng(7).normal(0, 0.3, int(SR * 0.03))
                  .astype(np.float32), ms=2.0)
    clip = np.concatenate([
        np.zeros(int(SR * 0.02), dtype=np.float32),
        burst,
        np.zeros(int(SR * 0.15), dtype=np.float32),  # >=3 silent frames
        _speech_like(),
        np.zeros(int(SR * 0.1), dtype=np.float32),
    ])
    report = analyze_clip(clip, SR)
    kinds = {f.kind: f for f in report.findings}
    assert "leading_burst" in kinds
    assert kinds["leading_burst"].position_ms < 250


def test_trailing_isolated_burst_detected() -> None:
    burst = _fade(np.random.default_rng(7).normal(0, 0.3, int(SR * 0.03))
                  .astype(np.float32), ms=2.0)
    clip = np.concatenate([
        np.zeros(int(SR * 0.1), dtype=np.float32),
        _speech_like(),
        np.zeros(int(SR * 0.15), dtype=np.float32),
        burst,
        np.zeros(int(SR * 0.02), dtype=np.float32),
    ])
    report = analyze_clip(clip, SR)
    kinds = {f.kind for f in report.findings}
    assert "trailing_burst" in kinds


def test_discontinuity_click_detected() -> None:
    """A bad concatenation join: instant jump between samples."""
    a = _speech_like(0.5)
    b = _speech_like(0.5)
    joined = np.concatenate([a[: len(a) // 2], (b + 0.8).clip(-1, 1)])
    report = analyze_clip(_pad(_fade(joined)), SR)
    kinds = {f.kind for f in report.findings}
    assert "discontinuity" in kinds


def test_internal_dropout_detected() -> None:
    gap = np.zeros(int(SR * 0.08), dtype=np.float32)  # 80ms hard gap
    clip = _pad(np.concatenate([_speech_like(0.5), gap, _speech_like(0.5)]))
    report = analyze_clip(clip, SR)
    kinds = {f.kind: f for f in report.findings}
    assert "internal_dropout" in kinds
    assert kinds["internal_dropout"].magnitude >= 40.0


def test_clipping_detected() -> None:
    t = np.arange(int(SR * 0.5), dtype=np.float32) / SR
    x = _fade(np.sign(np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    report = analyze_clip(_pad(x), SR)
    kinds = {f.kind for f in report.findings}
    assert "clipping" in kinds


def test_dc_offset_detected() -> None:
    x = _pad(_speech_like()) + 0.05
    report = analyze_clip(x.astype(np.float32), SR)
    kinds = {f.kind for f in report.findings}
    assert "dc_offset" in kinds


def test_pause_between_sentences_is_not_a_dropout() -> None:
    """The engine writes deliberate pause_ms silence between sentence
    clips -- but each CLIP is analyzed separately, and leading/trailing
    silence inside one clip is normal. Only a gap strictly INSIDE the
    speech body counts."""
    clip = _pad(_speech_like(), lead_ms=400.0, tail_ms=400.0)
    report = analyze_clip(clip, SR)
    assert report.clean, [f.detail for f in report.findings]


# ---------------------------------------------------------------------------
# OutputQualityWatcher
# ---------------------------------------------------------------------------


def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_watcher_flags_blip_and_writes_jsonl(tmp_path: Path) -> None:
    jsonl = tmp_path / "audio_quality.jsonl"
    w = OutputQualityWatcher(jsonl_path=jsonl)
    try:
        bad = _pad(_speech_like(), lead_ms=0.0).copy()
        bad[:200] = 0.5  # hard onset
        w.submit(bad, SR, label="bad clip")
        w.submit(_pad(_speech_like()), SR, label="clean clip")

        assert _wait_until(lambda: w.stats()["clips_seen"] >= 2)
        stats = w.stats()
        assert stats["clips_seen"] == 2
        assert stats["clips_flagged"] == 1
        assert stats["findings_by_kind"].get("hard_onset") == 1

        assert jsonl.is_file()
        records = [json.loads(line) for line in jsonl.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["label"] == "bad clip"
        assert records[0]["findings"][0]["kind"] == "hard_onset"
    finally:
        w.close()


def test_watcher_clean_clips_write_nothing(tmp_path: Path) -> None:
    jsonl = tmp_path / "audio_quality.jsonl"
    w = OutputQualityWatcher(jsonl_path=jsonl)
    try:
        w.submit(_pad(_speech_like()), SR, label="clean")
        assert _wait_until(lambda: w.stats()["clips_seen"] == 1)
        assert not jsonl.exists()
    finally:
        w.close()


def test_watcher_submit_never_blocks_on_full_queue(tmp_path: Path) -> None:
    w = OutputQualityWatcher(jsonl_path=tmp_path / "q.jsonl", max_queue=1)
    try:
        big = _pad(_speech_like(2.0))
        for _ in range(20):
            w.submit(big, SR)
        # All submits returned immediately; some clips were dropped.
        assert _wait_until(
            lambda: w.stats()["clips_seen"] + w.stats()["clips_dropped"] >= 20
        )
    finally:
        w.close()


def test_watcher_survives_analysis_error(tmp_path: Path) -> None:
    w = OutputQualityWatcher(jsonl_path=tmp_path / "q.jsonl")
    try:
        w.submit("not audio", SR)  # type: ignore[arg-type]
        w.submit(_pad(_speech_like()), SR, label="after error")
        assert _wait_until(lambda: w.stats()["clips_seen"] >= 1)
    finally:
        w.close()


def test_watcher_close_is_idempotent(tmp_path: Path) -> None:
    w = OutputQualityWatcher(jsonl_path=tmp_path / "q.jsonl")
    w.close()
    w.close()


def test_waveform_stream_records_every_clip(tmp_path: Path) -> None:
    wave = tmp_path / "audio_waveform.jsonl"
    w = OutputQualityWatcher(jsonl_path=tmp_path / "q.jsonl",
                             waveform_path=wave)
    try:
        w.submit(_pad(_speech_like()), SR, label="clean clip")
        bad = _pad(_speech_like(), lead_ms=0.0).copy()
        bad[:200] = 0.5
        w.submit(bad, SR, label="bad clip")
        assert _wait_until(lambda: w.stats()["clips_seen"] >= 2)
        assert _wait_until(
            lambda: wave.is_file()
            and len(wave.read_text().splitlines()) >= 2
        )
        records = [json.loads(line) for line in wave.read_text().splitlines()]
        assert len(records) == 2
        clean = next(r for r in records if r["label"] == "clean clip")
        flagged = next(r for r in records if r["label"] == "bad clip")
        assert clean["findings"] == []
        assert 0 < len(clean["env"]) <= 120
        assert max(clean["env"]) > 0.1
        assert any(f["kind"] == "hard_onset" for f in flagged["findings"])
    finally:
        w.close()


def test_waveform_stream_size_bounded(tmp_path: Path) -> None:
    wave = tmp_path / "audio_waveform.jsonl"
    w = OutputQualityWatcher(jsonl_path=tmp_path / "q.jsonl",
                             waveform_path=wave)
    try:
        # Pre-fill past the cap; the next append must rewrite keeping
        # only the newest lines.
        filler = json.dumps({"label": "old", "env": [0.0] * 120}) + "\n"
        wave.write_text(filler * 1400, encoding="utf-8")
        assert wave.stat().st_size > w._WAVEFORM_MAX_BYTES
        w.submit(_pad(_speech_like()), SR, label="newest")
        assert _wait_until(
            lambda: wave.stat().st_size <= w._WAVEFORM_MAX_BYTES
        )
        lines = wave.read_text().splitlines()
        assert len(lines) <= w._WAVEFORM_KEEP_LINES
        assert json.loads(lines[-1])["label"] == "newest"
    finally:
        w.close()


def test_waveform_stream_disabled_writes_nothing(tmp_path: Path) -> None:
    w = OutputQualityWatcher(jsonl_path=tmp_path / "q.jsonl",
                             waveform_path=None)
    try:
        w.submit(_pad(_speech_like()), SR, label="clip")
        assert _wait_until(lambda: w.stats()["clips_seen"] == 1)
        assert not (tmp_path / "audio_waveform.jsonl").exists()
    finally:
        w.close()


def test_output_watch_waveform_config_defaults() -> None:
    from ultron.config import OutputWatchConfig

    cfg = OutputWatchConfig()
    assert cfg.waveform_enabled is True
    assert cfg.waveform_jsonl_filename == "audio_waveform.jsonl"


# ---------------------------------------------------------------------------
# Singleton + config gate
# ---------------------------------------------------------------------------


def test_get_output_watcher_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import ultron.config as config_mod

    # Opt back in past the session-wide conftest guard, then exercise
    # the CONFIG gate.
    set_output_watcher_enabled(True)
    monkeypatch.setattr(
        config_mod, "get_config",
        lambda: SimpleNamespace(
            tts=SimpleNamespace(output_watch=SimpleNamespace(enabled=False)),
        ),
    )
    try:
        assert get_output_watcher() is None
    finally:
        set_output_watcher_enabled(False)


def test_get_output_watcher_override_kill_switch() -> None:
    """The conftest session guard path: override False -> always None."""
    set_output_watcher_enabled(False)
    assert get_output_watcher() is None


def test_get_output_watcher_enabled_and_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import ultron.audio.output_quality as oq
    import ultron.config as config_mod

    set_output_watcher_enabled(True)
    monkeypatch.setattr(config_mod, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(
        config_mod, "get_config",
        lambda: SimpleNamespace(
            tts=SimpleNamespace(output_watch=SimpleNamespace(
                enabled=True, jsonl_filename="aq.jsonl", max_queue=4,
            )),
        ),
    )
    try:
        w1 = get_output_watcher()
        assert w1 is not None
        assert w1 is get_output_watcher()  # cached singleton
        assert oq._watcher is w1
    finally:
        set_output_watcher_enabled(False)  # also closes + resets


def test_output_watch_config_defaults() -> None:
    from ultron.config import OutputWatchConfig

    cfg = OutputWatchConfig()
    assert cfg.enabled is True
    assert cfg.jsonl_filename == "audio_quality.jsonl"
    assert cfg.max_queue == 16
