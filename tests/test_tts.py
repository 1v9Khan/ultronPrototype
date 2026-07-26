"""TTS wrapper tests that avoid loading real voice models."""

from __future__ import annotations

import numpy as np

from kenning.tts.speech import TextToSpeech


class FakePiperVoice:
    def __init__(self) -> None:
        self.syn_config = None

    def synthesize_wav(self, text, wav_file, syn_config=None):
        self.syn_config = syn_config
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        samples = (np.ones(512, dtype=np.int16) * 1000).tobytes()
        wav_file.writeframes(samples)


def test_piper_synth_uses_current_synthesize_wav_api():
    tts = object.__new__(TextToSpeech)
    tts._voice = FakePiperVoice()
    tts.piper_sample_rate = 22050
    tts.length_scale = 1.25

    pcm, sr = tts._piper_synth("hello")

    assert sr == 22050
    assert pcm.dtype == np.int16
    assert pcm.shape == (512,)
    assert np.all(pcm == 1000)
    assert tts._voice.syn_config.length_scale == 1.25


def test_stereo_pcm_duplicates_mono_channel():
    mono = np.array([-1, 0, 1], dtype=np.int16)

    stereo = TextToSpeech._stereo_pcm(mono)

    assert stereo.dtype == np.int16
    assert np.array_equal(
        stereo,
        np.array([[-1, -1], [0, 0], [1, 1]], dtype=np.int16),
    )
