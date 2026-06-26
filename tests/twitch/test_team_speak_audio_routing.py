"""Team-speak redeem AUDIO ROUTING (2026-06-26).

The "Make Ultron Speak To My Team" redeem (``_twitch_team_speak``) must play the
synthesized line to THREE sinks:

  * the VoiceMeeter TEAM-mic bus (``relay_speech.output_device``) -- teammates,
  * the OBS / broadcast mirror -- stream viewers,
  * the streamer's OWN default speakers (``audio.output_device``, empty '' =
    system default) -- so the streamer hears what is said to their team.

The speaker copy is ALWAYS on for the team redeem (it is NOT gated by the
"HEAR CHAT" toggle, which only governs the SAY redeem / chat-reply path). It runs
on a daemon thread so it OVERLAPS the foreground PTT + team play (not sequential,
which would play it twice in a row), and it is FULL-BAND (``shape_for_team=False``
-- the Valorant comms DSP must not touch the streamer's speakers).

Fully offline: ``Orchestrator._twitch_team_speak`` is exercised as an UNBOUND
method on a tiny fake ``self`` (no boot, no model). ``play_to_device`` is
monkeypatched to RECORD the device indices + the shape_for_team flag instead of
opening any real audio stream; the TTS synth seam is a stub returning a fake clip.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from kenning.pipeline.orchestrator import Orchestrator


class _Rs:
    """Minimal redeem_speak config object."""

    def __init__(self, *, disabled_during_ranked: bool = True) -> None:
        self.disabled_during_ranked = disabled_during_ranked


class _FakeTTS:
    def _synthesize(self, _text: str):
        # int16 mono, arbitrary sample rate.
        return (np.zeros(2400, dtype=np.int16), 24000)


class _FakeOrch:
    """Stand-in for the orchestrator: only what _twitch_team_speak touches."""

    def __init__(self) -> None:
        self.tts = _FakeTTS()
        self._ranked_active = False
        self.ptt_calls: list[str] = []

    def _ptt_hold(self) -> None:
        self.ptt_calls.append("hold")

    def _ptt_release(self) -> None:
        self.ptt_calls.append("release")


@pytest.fixture
def _patched(monkeypatch):
    """Patch the relay_speech device + play primitives + broadcast tee so the
    method routes through them without any real audio. Returns the capture list:
    each entry is (device_index, shape_for_team)."""
    plays: list[tuple] = []

    def _fake_play_to_device(pcm, sr, device_index, *, shape_for_team=True,
                             **_kw):
        plays.append((device_index, shape_for_team))
        return float(len(pcm)) / float(sr)

    broadcast_calls: list[int] = []

    def _fake_submit(pcm, sr):
        broadcast_calls.append(int(sr))

    # The team device + speaker device resolvers return distinct indices.
    monkeypatch.setattr(
        "kenning.audio.relay_speech.play_to_device", _fake_play_to_device,
        raising=True,
    )
    monkeypatch.setattr(
        "kenning.audio.relay_speech.resolve_relay_device",
        lambda _cfg: 7,  # TEAM bus index
        raising=True,
    )
    monkeypatch.setattr(
        "kenning.audio.relay_speech.resolve_speaker_device",
        lambda: 3,  # SPEAKER index
        raising=True,
    )
    monkeypatch.setattr(
        "kenning.audio.relay_speech.relay_tts_text", lambda s: s, raising=True,
    )
    monkeypatch.setattr(
        "kenning.audio.broadcast.submit", _fake_submit, raising=True,
    )
    return plays, broadcast_calls


def _run_and_join(orch, line, rs):
    """Call _twitch_team_speak then drain the daemon speaker thread."""
    Orchestrator._twitch_team_speak(orch, line, rs)
    # The speaker copy runs on a daemon thread named "redeem-team-speaker";
    # join it so the assertions see its play_to_device call deterministically.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        live = [t for t in threading.enumerate()
                if t.name == "redeem-team-speaker"]
        if not live:
            break
        for t in live:
            t.join(timeout=1.0)


# --------------------------------------------------------------------------- #
# the core requirement: BOTH the team bus AND the speakers get the clip
# --------------------------------------------------------------------------- #
def test_team_speak_plays_to_team_bus_and_speakers(_patched):
    plays, broadcast_calls = _patched
    orch = _FakeOrch()
    _run_and_join(orch, "nice round team", _Rs())

    devices = {idx for idx, _shape in plays}
    assert 7 in devices, "the TEAM-mic bus must be played"
    assert 3 in devices, "the streamer's SPEAKERS must ALSO be played"
    assert len(plays) == 2, "exactly the team play + the speaker play"
    # The broadcast/OBS tee fired too.
    assert broadcast_calls == [24000]
    # PTT wraps the (foreground) team play.
    assert orch.ptt_calls == ["hold", "release"]


def test_team_speak_drives_waveform_overlay(monkeypatch, _patched):
    # 2026-06-26 fix: the team-speak redeem must submit its synthesized PCM to the
    # on-stream waveform sink so the GUI "speaking" indicator ANIMATES (it used to
    # sit still because this path bypassed waveform.submit).
    plays, _ = _patched
    waveform_calls: list[int] = []
    monkeypatch.setattr(
        "kenning.audio.waveform.submit",
        lambda pcm, sr: waveform_calls.append(int(sr)),
        raising=True,
    )
    orch = _FakeOrch()
    _run_and_join(orch, "nice round team", _Rs())
    assert waveform_calls == [24000], "the waveform overlay must be driven once"


def test_speaker_copy_is_full_band_not_team_dsp(_patched):
    plays, _ = _patched
    orch = _FakeOrch()
    _run_and_join(orch, "you got this", _Rs())

    by_device = {idx: shape for idx, shape in plays}
    # Team bus keeps the comms DSP; the speaker copy is full-band.
    assert by_device[7] is True
    assert by_device[3] is False


def test_speaker_play_not_gated_by_hear_chat(_patched):
    # The fake self has NO _chat_audio_to_speakers attribute at all -- proving the
    # speaker copy does not consult the HEAR-CHAT toggle (that gates the SAY path).
    plays, _ = _patched
    orch = _FakeOrch()
    assert not hasattr(orch, "_chat_audio_to_speakers")
    _run_and_join(orch, "gg wp", _Rs())
    assert 3 in {idx for idx, _ in plays}


# --------------------------------------------------------------------------- #
# fail-open: a bad speaker device never breaks the team play / redeem
# --------------------------------------------------------------------------- #
def test_unresolved_speaker_device_still_plays_team(monkeypatch, _patched):
    plays, _ = _patched
    monkeypatch.setattr(
        "kenning.audio.relay_speech.resolve_speaker_device",
        lambda: None, raising=True,  # cannot resolve speakers
    )
    orch = _FakeOrch()
    _run_and_join(orch, "team line", _Rs())
    devices = [idx for idx, _ in plays]
    assert devices == [7], "team bus still played; no speaker play"
    assert orch.ptt_calls == ["hold", "release"]


def test_speaker_resolver_raising_does_not_break_team(monkeypatch, _patched):
    plays, _ = _patched

    def _boom():
        raise RuntimeError("no audio host")

    monkeypatch.setattr(
        "kenning.audio.relay_speech.resolve_speaker_device", _boom,
        raising=True,
    )
    orch = _FakeOrch()
    _run_and_join(orch, "team line", _Rs())
    assert [idx for idx, _ in plays] == [7]


def test_speaker_play_error_does_not_break_team(monkeypatch, _patched):
    plays, _ = _patched

    def _selective_play(pcm, sr, device_index, *, shape_for_team=True, **_kw):
        if device_index == 3:                 # speaker copy explodes
            raise RuntimeError("device busy")
        plays.append((device_index, shape_for_team))
        return 0.1

    monkeypatch.setattr(
        "kenning.audio.relay_speech.play_to_device", _selective_play,
        raising=True,
    )
    orch = _FakeOrch()
    _run_and_join(orch, "team line", _Rs())
    # The team play still recorded; the speaker error was swallowed.
    assert [idx for idx, _ in plays] == [7]
    assert orch.ptt_calls == ["hold", "release"]


# --------------------------------------------------------------------------- #
# pre-existing guards still hold
# --------------------------------------------------------------------------- #
def test_ranked_active_refuses_entirely(_patched):
    plays, broadcast_calls = _patched
    orch = _FakeOrch()
    orch._ranked_active = True
    _run_and_join(orch, "rush B", _Rs(disabled_during_ranked=True))
    assert plays == []
    assert broadcast_calls == []


def test_empty_line_is_a_noop(_patched):
    plays, broadcast_calls = _patched
    orch = _FakeOrch()
    _run_and_join(orch, "   ", _Rs())
    assert plays == []
    assert broadcast_calls == []
