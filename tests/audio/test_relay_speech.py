"""Tests for the teammate voice relay (``kenning.audio.relay_speech``).

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

from kenning.audio.relay_speech import (
    RelayCommand,
    build_relay_line,
    match_relay_command,
    match_relay_toggle,
    play_to_device,
    resolve_relay_device,
)

# Import at module load (before any monkeypatch of get_config) so the
# transitive ``config.settings`` module loads against the REAL config.
from kenning.pipeline.orchestrator import Orchestrator


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
        # Addressed to Kenning, not the team.
        "Tell me how shit my teammates are in Valorant",
        "tell me a story about my team",
        # Ordinary utterances.
        "What time is it in Paris?",
        "Show me a picture of a chicken.",
        "My teammates are bad",
        # Relay verbs without a group addressee.
        "tell her I said hi",
        "say hello",
        # Names OUTSIDE the closed vocabulary never relay.
        "tell sarah I'll be late",
        "ask bob to bring snacks",
        # Matched group but no real payload (clipped transcript).
        "tell my teammates the",
        "tell my teammates",
        "tell sage the",
        "",
    ],
)
def test_match_relay_negative(text: str) -> None:
    assert match_relay_command(text) is None


# ---------------------------------------------------------------------------
# Named addressees + compose mode (the user's callout matrix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_name,expected_payload",
    [
        ("Ask Clove to smoke window every round.", "Clove",
         "smoke window every round."),
        ("Tell Sova to drone sewers.", "Sova", "drone sewers."),
        ("Ask Sage if I can get a heal.", "Sage", "if I can get a heal."),
        ("tell omen he should smoke A main", "Omen", "he should smoke A main"),
        ("say nice flash to breach", "Breach", "nice flash"),
        ("ask viper for a wall on mid", "Viper", "for a wall on mid"),
    ],
)
def test_match_named_addressee(
    text: str, expected_name: str, expected_payload: str,
) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.addressee == expected_name
    assert cmd.payload == expected_payload
    assert cmd.compose is False


@pytest.mark.parametrize(
    "text,expected_payload",
    [
        ("Tell my team to full buy this round.", "full buy this round."),
        ("Tell my team to save this round.", "save this round."),
        ("Tell my team nice try.", "nice try."),
        ("Tell my team the enemy is coming B.", "the enemy is coming B."),
        ("Tell my team to go A this round.", "go A this round."),
        ("Tell my team to play retake.", "play retake."),
        ("Tell my team I am lurking.", "I am lurking."),
        (
            "Tell my team that I am kenning the next step in human evolution.",
            "I am kenning the next step in human evolution.",
        ),
    ],
)
def test_match_team_callout_matrix(text: str, expected_payload: str) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.addressee == "team"
    assert cmd.payload == expected_payload


def test_match_team_rotate_short_payload_needs_two_words() -> None:
    # "tell my team to rotate" -> payload "rotate" is ONE word; the
    # two-word floor exists for clipped transcripts, so the canonical
    # phrasing keeps a qualifier ("rotate now" / "rotate B").
    assert match_relay_command("tell my team to rotate now") is not None
    assert match_relay_command("tell my team to rotate B") is not None


@pytest.mark.parametrize(
    "text",
    [
        "Give my team encouragement.",
        "give my team some encouragement",
        "Give the team a pep talk.",
        "encourage my team",
        "hype up my squad",
    ],
)
def test_match_compose_encouragement(text: str) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.compose is True
    assert cmd.addressee == "team"
    assert cmd.payload == "encouragement"


def test_custom_addressee_vocabulary() -> None:
    cmd = match_relay_command(
        "tell maverick to watch the door", names=["maverick"],
    )
    assert cmd is not None and cmd.addressee == "Maverick"
    # The custom vocabulary REPLACES the default roster.
    assert match_relay_command(
        "tell sova to drone sewers", names=["maverick"],
    ) is None


def test_named_rephrase_prompt_mentions_name() -> None:
    captured: list[str] = []

    def fake_generate(prompt: str):
        captured.append(prompt)
        return iter(["Clove, smoke window every round."])

    cmd = match_relay_command("ask clove to smoke window every round")
    assert cmd is not None
    line = build_relay_line(cmd, generate_fn=fake_generate)
    assert line == "Clove, smoke window every round."
    assert "Clove" in captured[0]
    assert "smoke window every round" in captured[0]


def test_compose_prompt_has_no_reported_speech() -> None:
    captured: list[str] = []

    def fake_generate(prompt: str):
        captured.append(prompt)
        return iter(["Heads up team, we've got this."])

    cmd = match_relay_command("give my team some encouragement")
    assert cmd is not None
    line = build_relay_line(cmd, generate_fn=fake_generate)
    assert line == "Heads up team, we've got this."
    assert "encouragement" in captured[0]
    assert "reported speech" not in captured[0]


def test_first_person_instruction_present_in_prompt() -> None:
    captured: list[str] = []
    cmd = match_relay_command("tell my team I am lurking this round")
    assert cmd is not None
    build_relay_line(
        cmd, generate_fn=lambda p: captured.append(p) or iter(["x y"]),
    )
    assert "first person" in captured[0].lower()


def test_named_fallback_line() -> None:
    cmd = match_relay_command("ask sage if I can get a heal")
    assert cmd is not None
    line = build_relay_line(cmd, rephrase=False)
    assert line == "Sage: if I can get a heal"


def test_compose_fallback_line_is_stock_encouragement() -> None:
    cmd = match_relay_command("encourage my team")
    assert cmd is not None
    line = build_relay_line(cmd, rephrase=False)
    assert line  # stock line, non-empty
    assert "Team:" not in line


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
    import kenning.audio.devices as devices

    monkeypatch.setattr(devices, "resolve_device", lambda c, k: 42)
    assert resolve_relay_device("Voicemeeter Aux Input") == 42


def test_resolve_relay_device_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    import kenning.audio.devices as devices

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
    import kenning.config as config_mod

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
    import kenning.audio.relay_speech as relay_mod

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(
        _synthesize=lambda text: (np.ones(10, dtype=np.int16), 24000),
    )
    _patch_config(monkeypatch, _relay_cfg())
    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: None)

    assert o._maybe_handle_relay_speech("tell my teammates to rotate B") is True
    assert o._spoken and "relay audio device" in o._spoken[0]


def test_orchestrator_relay_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import kenning.audio.relay_speech as relay_mod

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
    import kenning.audio.relay_speech as relay_mod

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
    import kenning.audio.relay_speech as relay_mod

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
    from kenning.config import RelaySpeechConfig

    cfg = RelaySpeechConfig()
    assert cfg.enabled is True
    assert "voicemeeter" in cfg.output_device.lower()
    assert cfg.rephrase is True
    assert cfg.max_line_chars == 280
    assert cfg.echo_to_user is False
    assert cfg.addressee_names == []
    assert cfg.follow_up_seconds == 120.0


# ---------------------------------------------------------------------------
# Conversational layer: wording variety + no-wake-word follow-ups
# ---------------------------------------------------------------------------


def test_prompt_includes_recent_lines_and_variety_instruction() -> None:
    captured: list[str] = []
    cmd = match_relay_command("tell my team to push A together")
    assert cmd is not None
    build_relay_line(
        cmd,
        recent_lines=["Rotate B now, team.", "Sage, can I get a heal?"],
        generate_fn=lambda p: captured.append(p) or iter(["Push A together."]),
    )
    prompt = captured[0]
    assert "Rotate B now, team." in prompt
    assert "Sage, can I get a heal?" in prompt
    assert "do NOT reuse their wording" in prompt
    assert "vary your phrasing" in prompt


def test_prompt_recent_lines_capped_at_six() -> None:
    captured: list[str] = []
    cmd = match_relay_command("tell my team nice try everyone")
    assert cmd is not None
    build_relay_line(
        cmd,
        recent_lines=[f"line {i}" for i in range(10)],
        generate_fn=lambda p: captured.append(p) or iter(["Nice try."]),
    )
    prompt = captured[0]
    assert "line 9" in prompt and "line 4" in prompt
    assert "line 3" not in prompt


def test_orchestrator_relay_records_recent_lines_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kenning.audio.relay_speech as relay_mod

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(
        _synthesize=lambda text: (np.ones(10, dtype=np.int16), 24000),
    )
    _patch_config(monkeypatch, _relay_cfg(follow_up_seconds=120.0))
    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: 25)
    monkeypatch.setattr(
        relay_mod, "play_to_device", lambda pcm, sr, device, **kw: 0.1,
    )

    assert o._maybe_handle_relay_speech("tell them to watch flank") is True
    assert o._maybe_handle_relay_speech("tell them to push B now") is True
    assert list(o._relay_recent_lines) == [
        "Team: watch flank", "Team: push B now",
    ]
    assert o._relay_recent_lines.maxlen == 6
    # The follow-up extension is armed for the run-loop branch.
    assert o._relay_follow_up_seconds == 120.0


def test_is_relay_command_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    o = _bare_orchestrator()
    _patch_config(monkeypatch, _relay_cfg())
    assert o._is_relay_command("tell my team to save this round") is True
    assert o._is_relay_command("ask sage if I can get a heal") is True
    assert o._is_relay_command("what time is it") is False
    assert o._is_relay_command("tell me a joke") is False


def test_is_relay_command_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    o = _bare_orchestrator()
    _patch_config(monkeypatch, _relay_cfg(enabled=False))
    assert o._is_relay_command("tell my team to save this round") is False


# ---------------------------------------------------------------------------
# Streaming safety: explicit commands ONLY + the session mute toggle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # The user's exact stream-narration examples: descriptive
        # speech about the team must NEVER relay.
        "I want my team to smoke window",
        "why is my team not smoking window",
        "I wish my team would rotate faster",
        "my team should really full buy here",
        "I told my team to save and they didn't",
        "why won't clove smoke window",
        "clove is not smoking window again",
    ],
)
def test_stream_narration_never_relays(text: str) -> None:
    assert match_relay_command(text) is None


@pytest.mark.parametrize(
    "text,addressee,payload",
    [
        # The user's exact explicit-command examples DO relay.
        ("Ask my team to smoke window every round", "team",
         "smoke window every round"),
        ("ask my clove why she is not smoking window", "Clove",
         "why she is not smoking window"),
        ("ask my team why no one is buying armor", "team",
         "why no one is buying armor"),
        ("tell my sova to drone first", "Sova", "drone first"),
    ],
)
def test_explicit_commands_relay(text: str, addressee: str,
                                 payload: str) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.addressee == addressee
    assert cmd.payload == payload


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Mute the team chat.", False),
        ("mute the relay", False),
        ("turn off the team relay", False),
        ("stop talking to my team", False),
        ("don't talk to my teammates", False),
        ("Unmute the relay.", True),
        ("turn on the team chat", True),
        ("enable the relay", True),
        ("you can talk to my team again", True),
        ("start talking to my team again", True),
        # Non-toggles.
        ("the relay is cool", None),
        ("stop", None),
        ("mute the tv", None),
        ("tell my team to rotate B", None),
    ],
)
def test_match_relay_toggle(text: str, expected) -> None:
    assert match_relay_toggle(text) == expected


def test_orchestrator_toggle_mutes_and_unmutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    o = _bare_orchestrator()
    _patch_config(monkeypatch, _relay_cfg())
    assert o._maybe_handle_relay_toggle("mute the team chat") is True
    assert o._relay_runtime_enabled is False
    assert "muted" in o._spoken[-1]
    assert o._maybe_handle_relay_toggle("unmute the relay") is True
    assert o._relay_runtime_enabled is True
    assert "back on" in o._spoken[-1]
    assert o._maybe_handle_relay_toggle("what time is it") is False


def test_orchestrator_toggle_requires_feature_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    o = _bare_orchestrator()
    _patch_config(monkeypatch, _relay_cfg(enabled=False))
    assert o._maybe_handle_relay_toggle("mute the team chat") is False


def test_muted_relay_command_acknowledged_not_transmitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kenning.audio.relay_speech as relay_mod

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(
        _synthesize=lambda text: (np.ones(10, dtype=np.int16), 24000),
    )
    o._relay_runtime_enabled = False
    _patch_config(monkeypatch, _relay_cfg())

    def must_not_play(pcm, sr, device, **kw):  # pragma: no cover
        raise AssertionError("muted relay must not transmit")

    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: 25)
    monkeypatch.setattr(relay_mod, "play_to_device", must_not_play)

    assert o._maybe_handle_relay_speech("tell my team to rotate B") is True
    assert "muted" in o._spoken[0]


def test_relay_generation_is_fully_isolated() -> None:
    """2026-06-11 live game-chat incident: without
    suppress_memory_context the engine prepends conversation history
    and the model answers the CONVERSATION ('Clove, the program is
    still in development...') instead of rephrasing the callout."""
    captured: dict = {}

    class _FakeLLM:
        def generate_stream(self, prompt, **kwargs):
            captured.update(kwargs, prompt=prompt)
            return iter(["Clove, smoke window."])

    cmd = match_relay_command("tell clove to smoke window every round")
    assert cmd is not None
    line = build_relay_line(cmd, _FakeLLM())
    assert line == "Clove, smoke window."
    assert captured["suppress_memory_context"] is True
    assert captured["record_history"] is False
    assert captured["enable_thinking"] is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        # The live leak, verbatim shape.
        ("Clove, smoke window. / no_think", "Clove, smoke window."),
        ("Rotate B now. /no_think", "Rotate B now."),
        ("Push A together./think", "Push A together."),
        ("Team: save round <|im_end|>", "Team: save round"),
    ],
)
def test_control_tokens_never_reach_the_spoken_line(
    raw: str, expected: str,
) -> None:
    cmd = match_relay_command("tell my team to rotate B now")
    assert cmd is not None
    line = build_relay_line(cmd, generate_fn=lambda p: iter([raw]))
    assert line == expected


def test_plural_teams_stt_artifact_matches() -> None:
    """Observed live: STT rendered 'teammates' as 'teams'."""
    cmd = match_relay_command("tell my teams to rotate B now")
    assert cmd is not None
    assert cmd.payload == "rotate B now"
    assert cmd.addressee == "team"


def test_is_relay_command_true_for_toggle_and_while_muted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Toggle phrases bypass the addressing gate too, and a muted
    relay-shaped utterance still routes to the handler (for the muted
    notice) instead of falling into the LLM path."""
    o = _bare_orchestrator()
    o._relay_runtime_enabled = False
    _patch_config(monkeypatch, _relay_cfg())
    assert o._is_relay_command("unmute the relay") is True
    assert o._is_relay_command("tell my team to rotate B") is True
