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


def test_loud_high_frequency_speech_not_discontinuity() -> None:
    # 2026-06-12 adjudication pin: the live FP class. All 112 live
    # discontinuity findings were loud high-frequency speech content
    # (jumps 0.50-0.67 at 0.82-1.33x the local envelope). A 4 kHz tone
    # at 0.65 amplitude produces an adjacent-sample jump of ~0.65 --
    # above the old absolute 0.5 threshold -- but its max diff is only
    # ~1.41x its own median diff, far below the outlier ratio.
    t = np.arange(int(SR * 1.0), dtype=np.float32) / SR
    x = _fade((0.65 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32))
    # Precondition: this clip WOULD have flagged under the old rule.
    assert float(np.abs(np.diff(x)).max()) > 0.5
    report = analyze_clip(_pad(x), SR)
    kinds = {f.kind for f in report.findings}
    assert "discontinuity" not in kinds, [
        f.detail for f in report.findings
    ]


def test_loud_fricative_noise_not_discontinuity() -> None:
    # Broadband noise (sibilant/fricative stand-in): max diff over a
    # window of Gaussian diffs sits ~4-5x the median -- below the
    # outlier ratio even when the absolute jump clears the floor.
    rng = np.random.default_rng(11)
    x = _fade((0.22 * rng.normal(0.0, 1.0, int(SR * 0.8)))
              .clip(-0.95, 0.95).astype(np.float32))
    assert float(np.abs(np.diff(x)).max()) > 0.5  # precondition
    report = analyze_clip(_pad(x), SR)
    kinds = {f.kind for f in report.findings}
    assert "discontinuity" not in kinds, [
        f.detail for f in report.findings
    ]


def test_offset_join_inside_loud_vowel_still_flagged() -> None:
    # The hardest production-plausible true positive: a 0.6 DC-offset
    # join inside a loud 1 kHz vowel measures only ~9x the local diff
    # median (the vowel's own diffs are substantial) -- it must stay
    # above the outlier ratio.
    t = np.arange(int(SR * 0.5), dtype=np.float32) / SR
    a = (0.4 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    b = (0.4 * np.sin(2 * np.pi * 1000 * t) + 0.6).clip(-1, 1).astype(
        np.float32
    )
    report = analyze_clip(_pad(_fade(np.concatenate([a, b]))), SR)
    kinds = {f.kind for f in report.findings}
    assert "discontinuity" in kinds


def test_click_in_quiet_audio_still_flagged() -> None:
    # A click against quiet content is the clearest true positive:
    # the jump is a massive outlier vs the local diff median.
    t = np.arange(int(SR * 1.0), dtype=np.float32) / SR
    x = _fade((0.03 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    spike_at = len(x) // 2
    x = x.copy()
    x[spike_at] = 0.62  # single-sample click
    report = analyze_clip(_pad(x), SR)
    kinds = {f.kind: f for f in report.findings}
    assert "discontinuity" in kinds
    assert "x the local diff median" in kinds["discontinuity"].detail


def test_internal_dropout_detected() -> None:
    # 2026-06-12 two-tier rule: a genuine digital HARD CUT -- the gap
    # edges sit at full speech level (no decay). 150 ms is above the
    # 100 ms floor and below the 600 ms dead-air tier, so the
    # adjacent-energy rule is what fires here.
    s = _speech_like(1.0)
    cut = len(s) // 2
    clip = _pad(np.concatenate([
        s[:cut],
        np.zeros(int(SR * 0.15), dtype=np.float32),
        s[cut:],
    ]))
    report = analyze_clip(clip, SR)
    kinds = {f.kind: f for f in report.findings}
    assert "internal_dropout" in kinds
    assert kinds["internal_dropout"].magnitude >= 100.0


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
# 2026-06-12 detector adjudication pins (geometry from live
# logs/audio_quality.jsonl records)
# ---------------------------------------------------------------------------


def test_loud_runs_helper() -> None:
    from ultron.audio.output_quality import _loud_runs

    assert _loud_runs(np.array([], dtype=bool)) == []
    assert _loud_runs(np.array([True, True, True])) == [(0, 2)]
    assert _loud_runs(np.array([False, True, False, True])) == [
        (1, 1), (3, 3),
    ]
    assert _loud_runs(np.array([True, False, False, True, True])) == [
        (0, 0), (3, 4),
    ]


def test_natural_sentence_pause_not_flagged() -> None:
    # 'This is response number N. The system is operating normally.'
    # geometry: two sentences with gradual decay into a 350 ms pause
    # (live records: 290-380 ms gaps at the period boundary, x41).
    s1 = _fade(_speech_like(1.5), ms=80.0)
    s2 = _fade(_speech_like(1.5), ms=80.0)
    clip = _pad(np.concatenate([
        s1, np.zeros(int(SR * 0.35), dtype=np.float32), s2,
    ]))
    report = analyze_clip(clip, SR)
    kinds = {f.kind for f in report.findings}
    assert "internal_dropout" not in kinds, [
        f.detail for f in report.findings
    ]


def test_multiple_sentence_pauses_not_flagged() -> None:
    # 'Let me walk you through this in full detail.' repeated geometry:
    # 11 gaps of 330-400 ms every ~2.16 s in the live record.
    sentences = [_fade(_speech_like(1.0), ms=80.0) for _ in range(3)]
    gap1 = np.zeros(int(SR * 0.33), dtype=np.float32)
    gap2 = np.zeros(int(SR * 0.40), dtype=np.float32)
    clip = _pad(np.concatenate(
        [sentences[0], gap1, sentences[1], gap2, sentences[2]]
    ))
    report = analyze_clip(clip, SR)
    kinds = {f.kind for f in report.findings}
    assert "internal_dropout" not in kinds


def test_stop_consonant_closure_not_flagged() -> None:
    # 'forty-eight degrees' geometry: 60-70 ms closures with ABRUPT
    # edges (live records x7). Below the 100 ms floor, so even
    # loud-edged gaps stay unflagged.
    s = _speech_like(1.0)
    cut = len(s) // 2
    clip = _pad(np.concatenate([
        s[:cut],
        np.zeros(int(SR * 0.06), dtype=np.float32),
        s[cut:],
    ]))
    report = analyze_clip(clip, SR)
    kinds = {f.kind for f in report.findings}
    assert "internal_dropout" not in kinds


def test_long_dead_air_flagged_despite_soft_edges() -> None:
    # Tier 1: >= 600 ms of dead air inside the body flags regardless
    # of edge decay -- no natural pause is that long.
    s1 = _fade(_speech_like(1.0), ms=80.0)
    s2 = _fade(_speech_like(1.0), ms=80.0)
    clip = _pad(np.concatenate([
        s1, np.zeros(int(SR * 0.70), dtype=np.float32), s2,
    ]))
    report = analyze_clip(clip, SR)
    kinds = {f.kind: f for f in report.findings}
    assert "internal_dropout" in kinds
    assert kinds["internal_dropout"].magnitude >= 600.0


def test_quiet_trailing_burst_reclassified_not_dropout() -> None:
    # THE regression pin for the pre-trimmer-fix class ('Online.'
    # geometry): a quiet (~7% of peak) 60 ms terminal burst isolated
    # by 320 ms of dead air. The legacy 25%-of-peak burst gate
    # rejected it, so the dead air before it was misreported as an
    # internal dropout. It must now report as trailing_burst.
    lead = np.zeros(int(SR * 0.1), dtype=np.float32)
    speech = _fade(_speech_like(0.7), ms=40.0)
    gap = np.zeros(int(SR * 0.32), dtype=np.float32)
    t = np.arange(int(SR * 0.06), dtype=np.float32) / SR
    burst = (0.05 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    tail = np.zeros(int(SR * 0.03), dtype=np.float32)
    clip = np.concatenate([lead, speech, gap, burst, tail])
    report = analyze_clip(clip, SR)
    kinds = {f.kind: f for f in report.findings}
    assert "trailing_burst" in kinds, [f.detail for f in report.findings]
    assert "internal_dropout" not in kinds
    burst_start_ms = (lead.size + speech.size + gap.size) / SR * 1000.0
    assert abs(kinds["trailing_burst"].position_ms - burst_start_ms) <= 20.0


def test_quiet_leading_burst_reclassified_not_dropout() -> None:
    lead = np.zeros(int(SR * 0.03), dtype=np.float32)
    t = np.arange(int(SR * 0.06), dtype=np.float32) / SR
    burst = (0.05 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    gap = np.zeros(int(SR * 0.32), dtype=np.float32)
    speech = _fade(_speech_like(0.7), ms=40.0)
    tail = np.zeros(int(SR * 0.1), dtype=np.float32)
    clip = np.concatenate([lead, burst, gap, speech, tail])
    report = analyze_clip(clip, SR)
    kinds = {f.kind: f for f in report.findings}
    assert "leading_burst" in kinds, [f.detail for f in report.findings]
    assert "internal_dropout" not in kinds


def test_hard_cut_with_one_decayed_side_not_flagged() -> None:
    # Documents the both-sides rule for the 100-600 ms tier: a gap
    # whose pre side decays naturally does not flag even when the
    # post side is abrupt.
    s1 = _fade(_speech_like(1.0), ms=80.0)  # decays into the gap
    s2 = _speech_like(1.0).copy()
    s2[: int(SR * 0.01)] = 0.3  # abrupt re-entry
    clip = _pad(np.concatenate([
        s1, np.zeros(int(SR * 0.20), dtype=np.float32), s2,
    ]))
    report = analyze_clip(clip, SR)
    kinds = {f.kind for f in report.findings}
    assert "internal_dropout" not in kinds


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
