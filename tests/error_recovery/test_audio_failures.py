"""Audio-pipeline failure modes:
  - Whisper transcribe raises  -> empty string, error logged
  - Piper synth raises          -> silent clip, error logged
  - RVC convert raises          -> raw Piper passthrough, error logged

Each path validates: pipeline doesn't crash; subsequent calls work;
errors.jsonl records the failure with a recovery line.
"""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------


def _whisper_with_mock_model():
    """Build a WhisperEngine without loading a real model."""
    from kenning.transcription.whisper_engine import WhisperEngine
    eng = WhisperEngine.__new__(WhisperEngine)
    eng.model_name = "test"
    eng.device = "cpu"
    eng.compute_type = "int8"
    eng.beam_size = 1
    eng._model = MagicMock()
    return eng


def test_whisper_transcribe_failure_returns_empty(errors_log, read_errors):
    """Whisper.transcribe raises -> returns '' and logs WhisperTranscriptionError."""
    eng = _whisper_with_mock_model()
    eng._model.transcribe.side_effect = RuntimeError("CUDA OOM")

    audio = np.zeros(16000, dtype=np.float32)  # 1s of silence
    out = eng.transcribe(audio)

    assert out == ""
    records = read_errors()
    assert len(records) == 1
    rec = records[0]
    assert rec["dependency"] == "whisper"
    assert rec["error_type"] == "WhisperTranscriptionError"
    assert "CUDA OOM" in rec["message"]
    assert "skips this turn" in rec["recovery"]


def test_whisper_subsequent_transcribe_works_after_failure(errors_log, read_errors):
    """One Whisper failure doesn't poison subsequent calls."""
    eng = _whisper_with_mock_model()

    # First call fails.
    eng._model.transcribe.side_effect = RuntimeError("transient")
    assert eng.transcribe(np.zeros(16000, dtype=np.float32)) == ""

    # Second call succeeds.
    eng._model.transcribe.side_effect = None
    seg = MagicMock()
    seg.text = "hello world"
    info = MagicMock()
    info.language = "en"
    eng._model.transcribe.return_value = ([seg], info)

    out = eng.transcribe(np.zeros(16000, dtype=np.float32))
    assert out == "hello world"


# ---------------------------------------------------------------------------
# Piper
# ---------------------------------------------------------------------------


def test_piper_synth_failure_returns_silent_clip_and_logs(
    tmp_path, errors_log, read_errors,
):
    """Piper synth raising returns a zero-length int16 array; errors logged."""
    from kenning.tts.speech import TextToSpeech
    tts = TextToSpeech.__new__(TextToSpeech)
    tts._voice = MagicMock()
    tts._voice.synthesize_wav.side_effect = RuntimeError("piper crashed")
    # synthesize() also fails on the older code path
    tts._voice.synthesize.side_effect = RuntimeError("piper crashed")
    tts.piper_sample_rate = 22050
    tts.length_scale = 1.15

    pcm, sr = tts._piper_synth("hello world")
    assert pcm.size == 0
    assert sr == 22050
    records = read_errors()
    assert len(records) == 1
    rec = records[0]
    assert rec["dependency"] == "piper_tts"
    assert rec["error_type"] == "PiperSynthesisError"
    assert "terminal print" in rec["recovery"]


# ---------------------------------------------------------------------------
# RVC
# ---------------------------------------------------------------------------


def test_rvc_convert_failure_passes_piper_through(errors_log, read_errors):
    """RVC raising -> _synthesize returns the raw Piper PCM; errors logged."""
    from kenning.tts.speech import TextToSpeech
    tts = TextToSpeech.__new__(TextToSpeech)
    tts._voice = MagicMock()
    # Stub _piper_synth to skip the wav round-trip.
    tts._piper_synth = lambda text: (np.array([1, 2, 3], dtype=np.int16), 22050)
    tts.piper_sample_rate = 22050
    tts.length_scale = 1.15
    tts.rvc = MagicMock()
    tts.rvc.convert.side_effect = RuntimeError("rvc crashed")

    pcm, sr = tts._synthesize("anything")
    # raw Piper passes through unchanged
    assert list(pcm) == [1, 2, 3]
    assert sr == 22050
    records = read_errors()
    assert any(r["dependency"] == "rvc" and r["error_type"] == "RVCConversionError"
               for r in records)
    rec = next(r for r in records if r["dependency"] == "rvc")
    assert "raw Piper" in rec["recovery"]
