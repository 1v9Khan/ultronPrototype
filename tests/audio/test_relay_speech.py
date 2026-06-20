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
        # NB: bare "say hello" now DOES relay -- it defaults to the team hello
        # snap ("Hello.") rather than falling to the LLM (2026-06-19). See
        # TestSayHelloDefaultAndStop in test_corpus_audit_fixes.py.
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
        return iter(["Clove, what is the meaning of life?"])

    # A named QUESTION reaches the LLM (a short ability directive like 'smoke
    # window' is now handled deterministically by the snap path).
    cmd = match_relay_command("ask clove what the meaning of life is")
    assert cmd is not None
    line = build_relay_line(cmd, generate_fn=fake_generate)
    assert line == "Clove, what is the meaning of life?"
    assert "Clove" in captured[0]
    assert "meaning of life" in captured[0]


def test_compose_prompt_has_no_reported_speech() -> None:
    # A directive-compose ("respond") authors the line, so its prompt must NOT
    # carry a literal "reported speech" payload block. (Uses a non-calm
    # directive: 'calm' now routes to a curated pool, bypassing the LLM.)
    captured: list[str] = []

    def fake_generate(prompt: str):
        captured.append(prompt)
        return iter(["Jett, your insolence amuses me."])

    # NB: social insults ("making fun of you") now route to the curated reaction
    # pools (bypassing the LLM), so this uses a NON-social reported clause that
    # still reaches the LLM compose path.
    cmd = match_relay_command("jett told me the plan, respond")
    assert cmd is not None and cmd.compose
    line = build_relay_line(cmd, generate_fn=fake_generate)
    assert line == "Jett, your insolence amuses me."
    assert "reported speech" not in captured[0]


def test_morale_compose_uses_curated_pool() -> None:
    """Pure encouragement composes pick a curated Ultron morale line and do
    NOT call the LLM (the 4B rephrase is unreliable for abstract morale)."""
    from kenning.audio.relay_speech import DEFAULT_ENCOURAGEMENT_LINES

    called: list[str] = []
    cmd = match_relay_command("give my team some encouragement")
    assert cmd is not None
    line = build_relay_line(
        cmd, generate_fn=lambda p: called.append(p) or iter(["x"]))
    assert line in DEFAULT_ENCOURAGEMENT_LINES
    assert called == []  # curated pool short-circuits before the LLM


def test_first_person_instruction_present_in_prompt() -> None:
    captured: list[str] = []
    # An insult is off-snap -> reaches the LLM (literal callouts like 'I am
    # lurking' are now deterministic and never sent to the model).
    cmd = match_relay_command("tell my team they are bots")
    assert cmd is not None
    build_relay_line(
        cmd, generate_fn=lambda p: captured.append(p) or iter(["x y"]),
    )
    assert "first person" in captured[0].lower()


def test_named_fallback_line() -> None:
    cmd = match_relay_command("ask sage if I can get a heal")
    assert cmd is not None
    line = build_relay_line(cmd, rephrase=False)
    # Clean literal fallback (no chat-style 'Name:' label); opens with the name.
    assert line.startswith("Sage,") and "heal" in line.lower()
    assert ":" not in line.split(",")[0]


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
        return iter(["They ", "are ", "outmatched."])

    # 2026-06-17: a tactical callout ("they should rotate B") now bypasses the
    # model via the faithful-literal pre-route (any concrete count/loc/ability
    # token -> literal, never the 3B). The generate_fn seam is exercised by an
    # OFF-SNAP banter/opinion line (no tactical token), which is what genuinely
    # reaches the model.
    line = build_relay_line(_cmd("they think they can outplay us"),
                            generate_fn=fake_generate)
    assert captured, "generate_fn should be invoked for an off-snap banter line"
    assert "they think they can outplay us" in captured[0]
    assert "outmatched" in line.lower()


def test_build_relay_line_fallback_on_llm_error() -> None:
    def boom(prompt: str):
        raise RuntimeError("llm down")

    # An off-snap insult routes to the LLM, so its error path -> the clean literal
    # fallback (no 'Team:' label) is exercised, with the content preserved.
    line = build_relay_line(_cmd("they are bots"), generate_fn=boom)
    assert "bots" in line.lower() and not line.lower().startswith("team:")


def test_build_relay_line_fallback_on_empty_output() -> None:
    line = build_relay_line(_cmd("they are bots"), generate_fn=lambda p: iter([]))
    assert "bots" in line.lower() and not line.lower().startswith("team:")


def test_build_relay_line_rephrase_disabled_skips_llm() -> None:
    def fail(prompt: str):  # pragma: no cover - must not be called
        raise AssertionError("generate_fn must not be called")

    # An off-snap line (an insult) with rephrase disabled -> the deterministic
    # fallback, never the LLM. (Economy 'save' is now handled deterministically,
    # so it is no longer a clean rephrase-skip probe -- see the economy tests.)
    line = build_relay_line(_cmd("they are clueless"), rephrase=False, generate_fn=fail)
    assert "clueless" in line.lower() and not line.lower().startswith("team:")


def test_build_relay_line_strips_quotes_newlines_and_caps_length() -> None:
    long_text = "word " * 100

    line = build_relay_line(
        _cmd(), generate_fn=lambda p: iter([f'"{long_text}\nmore"']), max_chars=50,
    )
    assert "\n" not in line and '"' not in line
    assert len(line) <= 51  # cap + closing period
    assert line.endswith(".")


def test_build_relay_line_no_llm_no_generate_fn_falls_back() -> None:
    # An off-snap line (an insult) with no LLM available -> deterministic
    # fallback. ('watch the flank' is now a handled team directive -> 'Watch
    # the flank.', so it is no longer an unhandled-fallback probe.)
    line = build_relay_line(_cmd("they are absolute clowns"), llm=None)
    assert "clowns" in line.lower() and not line.lower().startswith("team:")


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


def test_play_to_device_writes_int16_stereo() -> None:
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
    # STEREO: mono PCM is widened to 2 channels so WASAPI auto-convert only has
    # to resample (not also up-mix 1->2 channels, which statics on B1 VAIO).
    assert stream.kwargs["channels"] == 2
    assert stream.kwargs["dtype"] == "int16"
    assert stream.started and stream.stopped and stream.closed
    (written,) = stream.written
    assert written.dtype == np.int16
    assert written.shape == (24000, 2)
    # Centered: both channels carry the same mono signal.
    assert np.array_equal(written[:, 0], written[:, 1])


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
# team-relay (Valorant) conditioning: _shape_for_team
# ---------------------------------------------------------------------------


def _dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))
    return 20.0 * np.log10(max(rms, 1e-12))


class TestTeamShaping:
    """The Valorant team-path conditioning chain. These exercise the helper
    DIRECTLY (play_to_device only runs it on the live, non-stream_factory path)."""

    def _isolate(self, monkeypatch: pytest.MonkeyPatch, **on: str) -> None:
        # Master ON; every stage OFF unless explicitly turned on by the caller.
        monkeypatch.setenv("KENNING_RELAY_TEAM_DSP", "1")
        for stage in ("KENNING_RELAY_COMMS_FILTER", "KENNING_RELAY_NORMALIZE",
                      "KENNING_RELAY_COMFORT_NOISE", "KENNING_RELAY_SOFTCLIP"):
            monkeypatch.setenv(stage, on.get(stage, "0"))

    def test_master_gate_off_is_passthrough(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kenning.audio.relay_speech as relay_mod

        monkeypatch.setenv("KENNING_RELAY_TEAM_DSP", "0")
        x = (0.1 * np.sin(np.linspace(0, 200, 24000))).astype(np.float32)
        out = relay_mod._shape_for_team(x, 48000)
        assert np.array_equal(out, x)

    def test_comfort_noise_fills_digital_silence_below_ceiling(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kenning.audio.relay_speech as relay_mod

        self._isolate(monkeypatch, KENNING_RELAY_COMFORT_NOISE="1")
        out = relay_mod._shape_for_team(np.zeros(24000, dtype=np.float32), 48000)
        # No longer digital zero, but never audible hiss (<= -52 dBFS hard cap;
        # ~ -58 default).
        assert -72.0 < _dbfs(out) <= -52.0

    def test_comfort_noise_hard_ceiling_honored(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kenning.audio.relay_speech as relay_mod

        self._isolate(monkeypatch, KENNING_RELAY_COMFORT_NOISE="1")
        monkeypatch.setenv("KENNING_RELAY_NOISE_DBFS", "-6")  # absurd, must clamp
        out = relay_mod._shape_for_team(np.zeros(24000, dtype=np.float32), 48000)
        assert _dbfs(out) <= -52.0 + 0.5

    def test_normalize_pulls_quiet_clip_up(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kenning.audio.relay_speech as relay_mod

        self._isolate(monkeypatch, KENNING_RELAY_NORMALIZE="1")
        monkeypatch.setenv("KENNING_RELAY_TARGET_DBFS", "-20")
        t = np.arange(24000) / 48000.0
        quiet = (0.01 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        before, after = _dbfs(quiet), _dbfs(relay_mod._shape_for_team(quiet, 48000))
        assert after > before + 6.0      # boosted (clamped at +12 dB)

    def test_normalize_gain_is_clamped(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kenning.audio.relay_speech as relay_mod

        self._isolate(monkeypatch, KENNING_RELAY_NORMALIZE="1")
        monkeypatch.setenv("KENNING_RELAY_TARGET_DBFS", "-20")
        t = np.arange(8000) / 48000.0
        tiny = (1e-4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        out = relay_mod._shape_for_team(tiny, 48000)
        # +12 dB clamp => at most ~4x, NOT normalized all the way to -20 dBFS.
        assert float(np.max(np.abs(out))) <= float(np.max(np.abs(tiny))) * 4.0 + 1e-6

    def test_softclip_caps_peaks_at_ceiling(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kenning.audio.relay_speech as relay_mod

        self._isolate(monkeypatch, KENNING_RELAY_SOFTCLIP="1")
        monkeypatch.setenv("KENNING_RELAY_CEILING_DBFS", "-1")
        hot = (np.ones(2000, dtype=np.float32) * 2.0)   # 2x over full scale
        out = relay_mod._shape_for_team(hot, 48000)
        ceil = 10.0 ** (-1.0 / 20.0)
        assert float(np.max(np.abs(out))) <= ceil + 1e-3

    def test_bandshape_highpass_strips_dc(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kenning.audio.relay_speech as relay_mod

        self._isolate(monkeypatch, KENNING_RELAY_COMMS_FILTER="1")
        out = relay_mod._shape_for_team(
            np.full(24000, 0.2, dtype=np.float32), 48000)
        assert abs(float(np.mean(out))) < 0.02     # DC removed by the high-pass

    def test_lowpass_off_by_default_keeps_highs(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kenning.audio.relay_speech as relay_mod

        self._isolate(monkeypatch, KENNING_RELAY_COMMS_FILTER="1")
        monkeypatch.delenv("KENNING_RELAY_LOWPASS_HZ", raising=False)
        t = np.arange(24000) / 48000.0
        hi = (0.3 * np.sin(2 * np.pi * 9000 * t)).astype(np.float32)
        out = relay_mod._shape_for_team(hi, 48000)
        assert _dbfs(out) > _dbfs(hi) - 3.0        # 9 kHz survives (LP is off)

    def test_lowpass_when_enabled_attenuates_highs(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kenning.audio.relay_speech as relay_mod

        self._isolate(monkeypatch, KENNING_RELAY_COMMS_FILTER="1")
        monkeypatch.setenv("KENNING_RELAY_LOWPASS_HZ", "8500")
        t = np.arange(24000) / 48000.0
        hi = (0.3 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)
        out = relay_mod._shape_for_team(hi, 48000)
        assert _dbfs(out) < _dbfs(hi) - 3.0        # 12 kHz cut by the 8.5 kHz LP

    def test_default_chain_raises_floor_and_keeps_length(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kenning.audio.relay_speech as relay_mod

        # Full DEFAULT chain (no env set beyond clearing the LP). Speech with
        # digital-silence gaps, like Kokoro.
        for v in ("KENNING_RELAY_TEAM_DSP", "KENNING_RELAY_COMMS_FILTER",
                  "KENNING_RELAY_NORMALIZE", "KENNING_RELAY_COMFORT_NOISE",
                  "KENNING_RELAY_SOFTCLIP", "KENNING_RELAY_LOWPASS_HZ"):
            monkeypatch.delenv(v, raising=False)
        n = 24000
        t = np.arange(n) / 48000.0
        x = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        x[: n // 4] = 0.0
        x[-n // 4:] = 0.0
        out = relay_mod._shape_for_team(x, 48000)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))
        # the leading silent quarter now carries the comfort-noise floor.
        assert _dbfs(out[: n // 8]) > _dbfs(x[: n // 8]) + 10.0

    def test_fail_open_never_raises(self) -> None:
        import kenning.audio.relay_speech as relay_mod

        for bad in (np.array([], dtype=np.float32),
                    np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)):
            out = relay_mod._shape_for_team(bad, 48000)
            assert isinstance(out, np.ndarray)


# ---------------------------------------------------------------------------
# consolation: crisp "nice try"
# ---------------------------------------------------------------------------


def test_nice_try_relays_crisp_recognizable_line() -> None:
    import kenning.audio.relay_speech as relay_mod

    # "nice try" / "good effort" -> crisp head + a short Ultron tail, NOT the
    # abstract DEFAULT_CONSOLATION_LINES koans ("A brief silence before the...").
    out = relay_mod._as_consolation_or_praise("nice try", None)
    assert out is not None and out.startswith("Nice try. ")
    assert any(out.endswith(t) for t in relay_mod._NICE_TRY_TAILS)
    assert relay_mod._as_consolation_or_praise(
        "good effort", None).startswith("Good effort. ")
    assert relay_mod._as_consolation_or_praise(
        "good try!", None).startswith("Good try. ")
    # OTHER consolations still use the generic pool (behavior unchanged).
    assert relay_mod._as_consolation_or_praise(
        "unlucky", None) in relay_mod.DEFAULT_CONSOLATION_LINES
    # Non-morale payload -> None (falls through to snap / LLM).
    assert relay_mod._as_consolation_or_praise("rush B", None) is None


def test_clutch_confidence_routes_deterministically() -> None:
    import kenning.audio.relay_speech as relay_mod

    # "tell my team I got this" -> a curated clutch line (no LLM).
    for p in ("I got this", "I've got this", "I have this", "I got it",
              "I'll clutch", "I can clutch this", "I'll carry this",
              "I'll win this round", "leave it to me", "this round is mine",
              "I'm gonna clutch", "watch this"):
        out = relay_mod._as_clutch(p, None)
        assert out in relay_mod.DEFAULT_CLUTCH_LINES, (p, out)
    # tactical / unrelated payloads must NEVER trip it.
    for p in ("I'll take A", "I have ult", "I got two", "I got walled",
              "I'll take main", "watch the flank", "I have no smokes",
              "rush B", "nice try"):
        assert relay_mod._as_clutch(p, None) is None, p
    # full route.
    c = relay_mod.match_relay_command("tell my team I got this")
    assert relay_mod.build_relay_line(c, None, rephrase=False) in \
        relay_mod.DEFAULT_CLUTCH_LINES
    assert len(relay_mod.DEFAULT_CLUTCH_LINES) == 20


def test_snap_registry_routes_and_is_data_extensible(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Part C: the data-driven SNAP_REGISTRY routes the existing snaps and
    auto-ingests a brand-new SnapRule with NO pipeline code change."""
    import re as _re
    import kenning.audio.relay_speech as relay_mod
    import kenning.audio.voice_lines as vl

    # existing snaps route via the registry, identical pools / format.
    assert relay_mod._apply_snap_registry("I got this", None) in \
        vl.DEFAULT_CLUTCH_LINES
    assert relay_mod._apply_snap_registry("nice try", None).startswith("Nice try. ")
    assert relay_mod._apply_snap_registry("unlucky", None) in \
        vl.DEFAULT_CONSOLATION_LINES
    assert relay_mod._apply_snap_registry("gg", None) in vl.DEFAULT_PRAISE_LINES
    # non-snap -> None (falls through to LLM / other handlers).
    assert relay_mod._apply_snap_registry("rush B", None) is None
    # off-switch -> None (the hardcoded snap functions remain as the fallback).
    monkeypatch.setenv("KENNING_SNAP_REGISTRY", "0")
    assert relay_mod._apply_snap_registry("I got this", None) is None
    monkeypatch.delenv("KENNING_SNAP_REGISTRY", raising=False)
    # DATA-DRIVEN: append ONE SnapRule -> it routes immediately, no code change.
    extra = vl.SnapRule(
        "demo_test",
        _re.compile(r"^\s*all according to plan\b", _re.IGNORECASE),
        "pool", lines=("As designed.",))
    monkeypatch.setattr(vl, "SNAP_REGISTRY", vl.SNAP_REGISTRY + (extra,))
    assert relay_mod._apply_snap_registry(
        "all according to plan", None) == "As designed."


# ---------------------------------------------------------------------------
# flavor-tail voice toggle + short hello snap
# ---------------------------------------------------------------------------


def test_flavor_toggle_matcher() -> None:
    import kenning.audio.relay_speech as relay_mod

    assert relay_mod.match_flavor_toggle("disable the flavor") is False
    assert relay_mod.match_flavor_toggle("flavor off") is False
    assert relay_mod.match_flavor_toggle("turn off the flavor tails") is False
    assert relay_mod.match_flavor_toggle("no flavor") is False
    assert relay_mod.match_flavor_toggle("turn the flavor back on") is True
    assert relay_mod.match_flavor_toggle("flavor on") is True
    assert relay_mod.match_flavor_toggle("enable flavor") is True
    # ordinary callouts / speech never trip it.
    assert relay_mod.match_flavor_toggle("rotate B") is None
    assert relay_mod.match_flavor_toggle("tell my team nice try") is None


def test_flavor_toggle_gates_tails() -> None:
    import kenning.audio.relay_speech as relay_mod

    saved = relay_mod.flavor_tails_enabled()
    try:
        relay_mod.set_flavor_tails_enabled(True)
        assert relay_mod._join_tail("Rotate B", "On my read.") == \
            "Rotate B. On my read."
        assert relay_mod._flavored("Rotate B", ["On my read."], None) == \
            "Rotate B. On my read."
        relay_mod.set_flavor_tails_enabled(False)               # flavor OFF
        assert relay_mod._join_tail("Rotate B", "On my read.") == "Rotate B"
        assert relay_mod._flavored("Rotate B", ["On my read."], None) == \
            "Rotate B"
    finally:
        relay_mod.set_flavor_tails_enabled(saved)


def test_short_hello_team_and_agent() -> None:
    import kenning.audio.relay_speech as relay_mod

    c = relay_mod.match_relay_command("say hello to my team")
    assert c is not None and getattr(c, "directive", None) == "hello"
    assert relay_mod.build_relay_line(c, None, rephrase=False) == "Hello team."
    c2 = relay_mod.match_relay_command("say hi to Jett")
    assert relay_mod.build_relay_line(c2, None, rephrase=False) == "Hello, Jett."
    c3 = relay_mod.match_relay_command("say hello to everyone")
    assert relay_mod.build_relay_line(c3, None, rephrase=False) == "Hello team."


def test_ask_day_snap_team_and_agent() -> None:
    import kenning.audio.relay_speech as relay_mod

    # team-wide -> a curated team courtesy question.
    for t in ("ask everyone how their day is going",
              "ask my team how their day is going",
              "ask the team how their day was",
              "ask everyone how the day is going"):
        c = relay_mod.match_relay_command(t)
        assert c is not None and getattr(c, "directive", None) == "ask_day", t
        assert relay_mod.build_relay_line(c, None, rephrase=False) in \
            relay_mod._ASK_DAY_TEAM_LINES, t
    # named agent -> a template with the agent's name.
    c = relay_mod.match_relay_command("ask Jett how their day is going")
    out = relay_mod.build_relay_line(c, None, rephrase=False)
    assert "Jett" in out and out.endswith("?"), out
    # the real relay "ask my team for X" must NOT be hijacked.
    c2 = relay_mod.match_relay_command("ask my team for smokes")
    assert c2 is None or getattr(c2, "directive", None) != "ask_day"


def test_target_snap_registry_is_data_extensible(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Part C target registry: hello/ask-day route through it, and a NEW
    TargetSnapRule (team + per-agent) auto-routes with NO pipeline code."""
    import re as _re
    import kenning.audio.relay_speech as relay_mod
    import kenning.audio.voice_lines as vl

    # existing target snaps still route + render correctly via the registry.
    c = relay_mod.match_relay_command("say hello to my team")
    assert relay_mod.build_relay_line(c, None, rephrase=False) == "Hello team."
    # append a brand-new target command -> routes for team AND a named agent.
    rule = vl.TargetSnapRule(
        "wish_luck",
        _re.compile(r"^(?:please\s+)?wish\s+(?P<target>.+?)\s+(?:good\s+)?luck",
                    _re.IGNORECASE),
        team_lines=("Luck is for the unprepared.",),
        agent_templates=("{name}. Win anyway.",))
    monkeypatch.setattr(vl, "TARGET_SNAP_REGISTRY", vl.TARGET_SNAP_REGISTRY + (rule,))
    ct = relay_mod.match_relay_command("wish my team good luck")
    assert getattr(ct, "directive", None) == "wish_luck"
    assert relay_mod.build_relay_line(ct, None, rephrase=False) == \
        "Luck is for the unprepared."
    ca = relay_mod.match_relay_command("wish Sova luck")
    assert relay_mod.build_relay_line(ca, None, rephrase=False) == "Sova. Win anyway."


def test_short_hello_does_not_hijack_long_intro() -> None:
    import kenning.audio.relay_speech as relay_mod

    # "introduce yourself" stays the LONG team intro (directive='greet').
    assert getattr(relay_mod.match_relay_command(
        "introduce yourself to my team"), "directive", None) == "greet"
    assert getattr(relay_mod.match_relay_command(
        "say hello to my team and introduce yourself"), "directive", None) == \
        "greet"
    # a non-team, non-agent target is NOT a hello snap.
    c = relay_mod.match_relay_command("say hello to the enemy")
    assert c is None or getattr(c, "directive", None) != "hello"


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
    # rephrase=False -> clean literal fallback (no 'Team:' label), synth + played.
    assert len(synthesized) == 1
    assert "smoking mid window" in synthesized[0].lower()
    assert not synthesized[0].lower().startswith("team:")
    assert played == [(2400, 24000, 25)]
    # echo_to_user=False -> nothing on the normal output.
    assert o._spoken == []


def test_orchestrator_relay_echo_to_user(monkeypatch: pytest.MonkeyPatch) -> None:
    import kenning.audio.relay_speech as relay_mod
    import kenning.audio.monitor as monitor_mod

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(
        _synthesize=lambda text: (np.ones(10, dtype=np.int16), 24000),
    )
    _patch_config(monkeypatch, _relay_cfg(echo_to_user=True))
    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: 25)
    monkeypatch.setattr(
        relay_mod, "play_to_device", lambda pcm, sr, device, **kw: 0.1,
    )
    teed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        monitor_mod, "maybe_submit",
        lambda pcm, sr: teed.append((len(pcm), sr)),
    )

    assert o._maybe_handle_relay_speech("tell them to watch flank") is True
    # echo_to_user now tees the SAME synthesized clip to the local monitor
    # (the user's own speakers) -- no re-synthesis, in sync with the mic write.
    assert teed == [(10, 24000)]
    # ...and it must NOT re-speak the line on the normal conversational path.
    assert o._spoken == []


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
    assert cfg.max_line_chars == 360
    assert cfg.echo_to_user is True   # default ON: user hears their own callouts
    assert cfg.addressee_names == []
    assert cfg.follow_up_seconds == 120.0


# ---------------------------------------------------------------------------
# Conversational layer: wording variety + no-wake-word follow-ups
# ---------------------------------------------------------------------------


def test_prompt_includes_recent_lines_and_variety_instruction() -> None:
    captured: list[str] = []
    # An enemy playstyle read reaches the LLM (and carries the recent-lines
    # block), so we can assert the prompt contents. ('push A together' is now a
    # deterministic directive and never reaches the model.)
    cmd = match_relay_command("tell my team the enemy is really passive")
    assert cmd is not None
    build_relay_line(
        cmd,
        recent_lines=["Rotate B now, team.", "Sage, can I get a heal?"],
        generate_fn=lambda p: captured.append(p) or iter(["They cower."]),
    )
    prompt = captured[0]
    assert "Rotate B now, team." in prompt
    assert "Sage, can I get a heal?" in prompt
    assert "do NOT reuse their wording" in prompt
    assert "ary" in prompt and "phrasing" in prompt  # variety instruction


def test_prompt_recent_lines_capped_at_six() -> None:
    captured: list[str] = []
    # An enemy playstyle read reaches the LLM (and is not a general question),
    # so the recent-lines block is included and we can assert its cap.
    cmd = match_relay_command("tell my team the enemy is camping")
    assert cmd is not None
    build_relay_line(
        cmd,
        recent_lines=[f"line {i}" for i in range(10)],
        generate_fn=lambda p: captured.append(p) or iter(["They cower."]),
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
    # Directives are resolved deterministically as clean imperatives; iter5 adds a
    # short owner-aware command tail after the preserved imperative core.
    ring = list(o._relay_recent_lines)
    assert len(ring) == 2
    assert ring[0].startswith("Watch flank.")
    assert ring[1].startswith("Push B now.")
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
            return iter(["Clove, what is your plan?"])

    # A named QUESTION reaches the LLM (snap handles short ability directives).
    cmd = match_relay_command("ask clove what the plan is")
    assert cmd is not None
    line = build_relay_line(cmd, _FakeLLM())
    assert line == "Clove, what is your plan?"
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
        ("Team: save round <|im_end|>", "save round"),
    ],
)
def test_control_tokens_never_reach_the_spoken_line(
    raw: str, expected: str,
) -> None:
    # An off-snap insult routes to the LLM, so the control-token strip on the
    # MODEL output is exercised. (Directive payloads like 'rotate B now' are now
    # handled deterministically and never reach the model.)
    cmd = match_relay_command("tell my team they are bots")
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
