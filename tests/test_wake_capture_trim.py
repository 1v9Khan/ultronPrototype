"""Pins the wake-capture trim (`_trim_wake_from_capture`).

The captured buffer is ``[ wake word (pre-roll) | gap? | command ]``. The trim
must drop the wake word (no phantom "Tron"/"Franz" leads) without clipping the
command's first word, and must NEVER reach back into the wake region. Pure /
hermetic: no Orchestrator, no audio device, no models.
"""

from __future__ import annotations

import numpy as np
import pytest

from kenning.pipeline.orchestrator import _trim_wake_from_capture as trim

SR = 16000


def _buf(n: int) -> np.ndarray:
    # Monotonic ramp so we can assert WHERE the kept slice begins.
    return np.arange(n, dtype=np.float32)


def test_pause_case_drops_wake_and_gap_keeps_command():
    # pre-roll 0.5s of wake, command onset 0.5s into the live region (a pause).
    out = trim(_buf(32000), pre_roll_len=8000, speech_start_samples=8000,
               sample_rate=SR)
    # guard = 200 ms = 3200 samples -> trim at 16000-3200 = 12800.
    assert out.shape[0] == 32000 - 12800
    assert out[0] == 12800.0                      # starts a guard before onset


def test_never_trims_into_the_wake_region():
    # Even with a tiny gap (< guard), the trim clamps to pre_roll_len so the
    # wake word is never re-included.
    out = trim(_buf(32000), pre_roll_len=8000, speech_start_samples=500,
               sample_rate=SR)
    assert out[0] == 8000.0                        # clamped to the wake boundary


def test_no_pause_drops_pre_roll_only():
    out = trim(_buf(24000), pre_roll_len=8000, speech_start_samples=0,
               sample_rate=SR)
    assert out.shape[0] == 16000
    assert out[0] == 8000.0


def test_flag_off_is_passthrough(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KENNING_WAKE_TRIM_TO_SPEECH", "0")
    b = _buf(32000)
    out = trim(b, 8000, 8000, SR)
    assert out.shape[0] == 32000 and out[0] == 0.0


def test_guard_is_env_tunable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KENNING_WAKE_TRIM_GUARD_MS", "0")
    out = trim(_buf(32000), pre_roll_len=8000, speech_start_samples=8000,
               sample_rate=SR)
    assert out[0] == 16000.0                       # no guard -> exact onset


def test_degenerate_and_bad_input_fail_open():
    # pre_roll longer than the buffer -> unchanged.
    assert trim(_buf(1000), 40000, 0, SR).shape[0] == 1000
    # empty buffer -> unchanged (no raise).
    assert trim(np.zeros(0, dtype=np.float32), 8000, 8000, SR).shape[0] == 0
    # negative pre-roll -> unchanged (no raise).
    assert trim(_buf(1000), -5, 0, SR).shape[0] == 1000


# ---------------------------------------------------------------------------
# _wake_command_cut: audio-domain wake removal via VAD segmentation
# ---------------------------------------------------------------------------

from kenning.pipeline.orchestrator import _wake_command_cut as cut  # noqa: E402

FIRE = 8000  # 0.5 s pre-roll (the wake/command boundary)
GUARD = int(0.120 * SR)  # default guard 120 ms


def test_cut_paused_command_drops_wake_and_gap():
    # wake [1000,7000] ends before the fire; command [12000,28000] after a gap.
    segs = [{"start": 1000, "end": 7000}, {"start": 12000, "end": 28000}]
    assert cut(segs, 32000, FIRE, SR) == 12000 - GUARD     # clean cut at command


BACKOFF = int(0.320 * SR)  # default fire-latency backoff 320 ms


def test_cut_continuous_backs_off_fire_latency():
    # 2026-07-23: one segment spanning the fire (no pause) -> cut a fire-latency
    # BACKOFF before the fire. The detector confirms ~320 ms after the word
    # ends, so cutting AT the fire beheaded the command's first word (live:
    # "Ultron, should I push mid?" -> "I push mid").
    assert cut([{"start": 1000, "end": 28000}], 32000, FIRE, SR) == FIRE - BACKOFF


def test_cut_continuous_backoff_env_tunable(monkeypatch: pytest.MonkeyPatch):
    # 0 restores the old cut-at-fire behaviour; a custom value is honored.
    monkeypatch.setenv("KENNING_WAKE_FIRE_BACKOFF_MS", "0")
    assert cut([{"start": 1000, "end": 28000}], 32000, FIRE, SR) == FIRE
    monkeypatch.setenv("KENNING_WAKE_FIRE_BACKOFF_MS", "100")
    assert cut([{"start": 1000, "end": 28000}], 32000, FIRE, SR) == FIRE - int(
        0.100 * SR)


def test_cut_continuous_backoff_clamps_at_zero():
    # backoff larger than the fire offset -> 0 (contract: "do not trim"),
    # fail-open to the text-level wake strip.
    segs = [{"start": 0, "end": 28000}]
    assert cut(segs, 32000, 2000, SR, fire_backoff_ms=500) == 0


def test_cut_bare_wake_returns_fire():
    # only pre-fire speech (bare "Ultron") -> cut at the fire (live region empty).
    assert cut([{"start": 1000, "end": 7000}], 32000, FIRE, SR) == FIRE


def test_cut_command_only_after_fire():
    # wake not captured as a segment; command starts just after the fire.
    assert cut([{"start": 9000, "end": 26000}], 32000, FIRE, SR) == 9000 - GUARD


def test_cut_no_segments_cuts_at_fire():
    # No speech detected -> still drop the pre-roll (wake) and keep the live
    # region (a quiet sub-threshold command there survives; pure silence -> STT
    # empty -> stand down).
    assert cut([], 32000, FIRE, SR) == FIRE
    assert cut(None, 32000, FIRE, SR) == FIRE


def test_cut_clamped_in_range():
    # a degenerate cut beyond the buffer clamps to 0 (no trim).
    assert cut([{"start": 99000, "end": 99999}], 1000, FIRE, SR) == 0


# ---------------------------------------------------------------------------
# _stt_repetition_loop: Whisper repetition-hallucination gate (2026-07-23)
# ---------------------------------------------------------------------------

from kenning.pipeline.orchestrator import _stt_repetition_loop as loop  # noqa: E402


def test_live_valorant_team_comms_loop_is_dropped():
    # THE battery turn=36 line: 1.52 s of audio -> 503 chars of one phrase.
    text = ("Valorant team comms. " * 24).strip()
    assert loop(text, 1.52) is True


def test_real_speech_density_passes():
    assert loop("tell my team they're pushing B main right now", 2.4) is False
    assert loop("Ultron, what is the meaning of life?", 1.7) is False


def test_long_but_plausible_dictation_passes():
    text = ("okay so the plan is we take mid control then split A through "
            "heaven and tree while sova recons the site")
    assert loop(text, 6.0) is False       # ~17 chars/s -- human


def test_dense_but_varied_text_passes():
    # Density alone is not enough -- only phrase-dominated text drops.
    text = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
            "kilo lima mike november oscar papa quebec romeo sierra tango")
    assert loop(text, 1.0) is False       # unique words -> not a loop


def test_degenerate_inputs_pass_through():
    assert loop("", 1.0) is False
    assert loop("short loop", 0.5) is False
    assert loop("anything", 0.0) is False
