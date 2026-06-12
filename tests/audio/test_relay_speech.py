"""Tests for the teammate voice relay (``ultron.audio.relay_speech``).

Hermetic per the binding rules: no real audio device is ever opened
(playback uses an injected stream factory; device resolution is
monkeypatched), no voice stack loads, and the orchestrator wiring tests
use the established ``Orchestrator.__new__`` pattern.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ultron.audio.relay_speech import (
    RelayCommand,
    build_relay_line,
    match_relay_command,
    play_to_device,
    resolve_relay_device,
)

# Import at module load (before any monkeypatch of get_config) so the
# transitive ``config.settings`` module loads against the REAL config.
from ultron.pipeline.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# match_relay_command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_payload",
    [
        (
            "Tell my teammates they should be smoking mid window every round.",
            "they should be smoking mid window every round.",
        ),
        (
            "tell my teammate to drop me a vandal",
            "drop me a vandal",
        ),
        ("Tell the team that we rotate B now.", "we rotate B now."),
        ("Say good luck everyone to my teammates", "good luck everyone"),
        ("ask my teammates for a save round please", "for a save round please"),
        ("ask the team if anyone has an ult", "if anyone has an ult"),
        ("tell them to watch flank", "watch flank"),
        ("Please tell my squad we are pushing A.", "we are pushing A."),
        # STT artifact: leading "One," before the relay verb is stripped.
        (
            "One, tell my teammate to drop me a vandal.",
            "drop me a vandal.",
        ),
    ],
)
def test_match_relay_positive(text: str, expected_payload: str) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.payload == expected_payload
    assert cmd.raw_text == text


@pytest.mark.parametrize(
    "text",
    [
        # Addressed to Ultron, not the team.
        "Tell me how shit my teammates are in Valorant",
        "tell me a story about my team",
        # Ordinary utterances.
        "What time is it in Paris?",
        "Show me a picture of a chicken.",
        "My teammates are bad",
        # Relay verbs without a group addressee.
        "tell her I said hi",
        "say hello",
        # Matched group but no real payload (clipped transcript).
        "tell my teammates the",
        "tell my teammates",
        "",
    ],
)
def test_match_relay_negative(text: str) -> None:
    assert match_relay_command(text) is None


# ---------------------------------------------------------------------------
# build_relay_line
# ---------------------------------------------------------------------------


def _cmd(payload: str = "they should rotate B") -> RelayCommand:
    return RelayCommand(payload=payload, raw_text=f"tell my teammates {payload}")


def test_build_relay_line_uses_generate_fn() -> None:
    captured: list[str] = []

    def fake_generate(prompt: str):
        captured.append(prompt)
        return iter(["Rotate ", "B ", "now, team."])

    line = build_relay_line(_cmd(), generate_fn=fake_generate)
    assert line == "Rotate B now, team."
    assert len(captured) == 1
    assert "they should rotate B" in captured[0]


def test_build_relay_line_fallback_on_llm_error() -> None:
    def boom(prompt: str):
        raise RuntimeError("llm down")

    line = build_relay_line(_cmd("push A together"), generate_fn=boom)
    assert line == "Team: push A together"


def test_build_relay_line_fallback_on_empty_output() -> None:
    line = build_relay_line(_cmd("push A together"), generate_fn=lambda p: iter([]))
    assert line == "Team: push A together"


def test_build_relay_line_rephrase_disabled_skips_llm() -> None:
    def fail(prompt: str):  # pragma: no cover - must not be called
        raise AssertionError("generate_fn must not be called")

    line = build_relay_line(_cmd("save this round"), rephrase=False, generate_fn=fail)
    assert line == "Team: save this round"


def test_build_relay_line_strips_quotes_newlines_and_caps_length() -> None:
    long_text = "word " * 100

    line = build_relay_line(
        _cmd(), generate_fn=lambda p: iter([f'"{long_text}\nmore"']), max_chars=50,
    )
    assert "\n" not in line and '"' not in line
    assert len(line) <= 51  # cap + closing period
    assert line.endswith(".")


def test_build_relay_line_no_llm_no_generate_fn_falls_back() -> None:
    line = build_relay_line(_cmd("watch the flank"), llm=None)
    assert line == "Team: watch the flank"


# ---------------------------------------------------------------------------
# play_to_device
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.written: list[np.ndarray] = []
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def write(self, data: np.ndarray) -> None:
        self.written.append(data)

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def test_play_to_device_writes_int16_mono() -> None:
    streams: list[_FakeStream] = []

    def factory(**kwargs: Any) -> _FakeStream:
        s = _FakeStream(**kwargs)
        streams.append(s)
        return s

    pcm = (np.ones(24000) * 1000).astype(np.int16)
    seconds = play_to_device(pcm, 24000, 7, stream_factory=factory)

    assert seconds == pytest.approx(1.0)
    (stream,) = streams
    assert stream.kwargs["device"] == 7
    assert stream.kwargs["samplerate"] == 24000
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "int16"
    assert stream.started and stream.stopped and stream.closed
    (written,) = stream.written
    assert written.dtype == np.int16
    assert written.shape == (24000, 1)


def test_play_to_device_converts_float32() -> None:
    streams: list[_FakeStream] = []

    def factory(**kwargs: Any) -> _FakeStream:
        s = _FakeStream(**kwargs)
        streams.append(s)
        return s

    pcm = np.ones(100, dtype=np.float32) * 2.0  # out of range -> clipped
    play_to_device(pcm, 16000, 3, stream_factory=factory)

    (written,) = streams[0].written
    assert written.dtype == np.int16
    assert int(written.max()) == 32767


def test_play_to_device_empty_pcm_is_noop() -> None:
    def factory(**kwargs: Any):  # pragma: no cover - must not be called
        raise AssertionError("no stream for empty pcm")

    assert play_to_device(np.zeros(0, dtype=np.int16), 24000, 1,
                          stream_factory=factory) == 0.0


def test_play_to_device_closes_stream_on_write_error() -> None:
    class _Exploding(_FakeStream):
        def write(self, data: np.ndarray) -> None:
            raise RuntimeError("device unplugged")

    streams: list[_Exploding] = []

    def factory(**kwargs: Any) -> _Exploding:
        s = _Exploding(**kwargs)
        streams.append(s)
        return s

    with pytest.raises(RuntimeError):
        play_to_device(np.ones(10, dtype=np.int16), 24000, 1,
                       stream_factory=factory)
    assert streams[0].stopped and streams[0].closed


# ---------------------------------------------------------------------------
# resolve_relay_device
# ---------------------------------------------------------------------------


def test_resolve_relay_device_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    import ultron.audio.devices as devices

    monkeypatch.setattr(devices, "resolve_device", lambda c, k: 42)
    assert resolve_relay_device("Voicemeeter Aux Input") == 42


def test_resolve_relay_device_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    import ultron.audio.devices as devices

    def boom(configured: Any, kind: str) -> int:
        raise RuntimeError("no such device")

    monkeypatch.setattr(devices, "resolve_device", boom)
    assert resolve_relay_device("Nonexistent Device") is None


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------


def _relay_cfg(**overrides: Any) -> SimpleNamespace:
    cfg = dict(
        enabled=True,
        output_device="Voicemeeter Aux Input",
        rephrase=False,
        max_line_chars=280,
        echo_to_user=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _bare_orchestrator() -> Any:
    o = Orchestrator.__new__(Orchestrator)
    o.llm = None
    o.tts = None
    o._spoken = []
    o._speak = lambda text: o._spoken.append(text)  # type: ignore[attr-defined]
    return o


def _patch_config(monkeypatch: pytest.MonkeyPatch, cfg: SimpleNamespace) -> None:
    import ultron.config as config_mod

    monkeypatch.setattr(
        config_mod, "get_config",
        lambda: SimpleNamespace(relay_speech=cfg),
    )


def test_orchestrator_relay_disabled_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    o = _bare_orchestrator()
    _patch_config(monkeypatch, _relay_cfg(enabled=False))
    assert o._maybe_handle_relay_speech("tell my teammates to rotate B") is False
    assert o._spoken == []


def test_orchestrator_relay_no_match_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    o = _bare_orchestrator()
    _patch_config(monkeypatch, _relay_cfg())
    assert o._maybe_handle_relay_speech("what time is it") is False
    assert o._spoken == []


def test_orchestrator_relay_no_tts_consumes_turn_with_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    o = _bare_orchestrator()  # tts is None
    _patch_config(monkeypatch, _relay_cfg())
    assert o._maybe_handle_relay_speech("tell my teammates to rotate B") is True
    assert o._spoken and "voice channel" in o._spoken[0]


def test_orchestrator_relay_missing_device_consumes_turn_with_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultron.audio.relay_speech as relay_mod

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(
        _synthesize=lambda text: (np.ones(10, dtype=np.int16), 24000),
    )
    _patch_config(monkeypatch, _relay_cfg())
    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: None)

    assert o._maybe_handle_relay_speech("tell my teammates to rotate B") is True
    assert o._spoken and "relay audio device" in o._spoken[0]


def test_orchestrator_relay_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import ultron.audio.relay_speech as relay_mod

    synthesized: list[str] = []
    played: list[tuple] = []

    o = _bare_orchestrator()

    def fake_synth(text: str):
        synthesized.append(text)
        return np.ones(2400, dtype=np.int16), 24000

    o.tts = SimpleNamespace(_synthesize=fake_synth)
    _patch_config(monkeypatch, _relay_cfg())
    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: 25)
    monkeypatch.setattr(
        relay_mod, "play_to_device",
        lambda pcm, sr, device, **kw: played.append((len(pcm), sr, device)) or 0.1,
    )

    handled = o._maybe_handle_relay_speech(
        "tell my teammates they should be smoking mid window every round"
    )
    assert handled is True
    # rephrase=False -> deterministic line, synthesized then played.
    assert synthesized == ["Team: they should be smoking mid window every round"]
    assert played == [(2400, 24000, 25)]
    # echo_to_user=False -> nothing on the normal output.
    assert o._spoken == []


def test_orchestrator_relay_echo_to_user(monkeypatch: pytest.MonkeyPatch) -> None:
    import ultron.audio.relay_speech as relay_mod

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(
        _synthesize=lambda text: (np.ones(10, dtype=np.int16), 24000),
    )
    _patch_config(monkeypatch, _relay_cfg(echo_to_user=True))
    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: 25)
    monkeypatch.setattr(
        relay_mod, "play_to_device", lambda pcm, sr, device, **kw: 0.1,
    )

    assert o._maybe_handle_relay_speech("tell them to watch flank") is True
    assert o._spoken == ["Team: watch flank"]


def test_orchestrator_relay_playback_failure_speaks_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultron.audio.relay_speech as relay_mod

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(
        _synthesize=lambda text: (np.ones(10, dtype=np.int16), 24000),
    )
    _patch_config(monkeypatch, _relay_cfg())
    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: 25)

    def boom(pcm, sr, device, **kw):
        raise RuntimeError("portaudio error")

    monkeypatch.setattr(relay_mod, "play_to_device", boom)

    assert o._maybe_handle_relay_speech("tell them to watch flank") is True
    assert o._spoken and "relay to your team failed" in o._spoken[0]


def test_relay_config_defaults() -> None:
    """The shipped config section exists with the documented defaults."""
    from ultron.config import RelaySpeechConfig

    cfg = RelaySpeechConfig()
    assert cfg.enabled is True
    assert "voicemeeter" in cfg.output_device.lower()
    assert cfg.rephrase is True
    assert cfg.max_line_chars == 280
    assert cfg.echo_to_user is False
