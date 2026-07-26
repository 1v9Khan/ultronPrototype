"""Pins the STREAMED always-listening private-reply path (2026-07-26).

``addressing.always_listening`` is ON, so ``_maybe_handle_private_reply`` is
the DEFAULT conversational path. It used to ``"".join(generate_stream(...))``
and then ``_speak`` the whole reply -- fully serial, so first audio waited for
the last token (~2.2 s for a 60-token reply at the 12B's measured 27.7 tok/s).
It now streams into ``speak_stream``, matching ``_respond``.

The subtle contract, and the reason this is not a one-line swap: the
scaffolding guards decide whether to FALL THROUGH to ``_respond``, and that
decision cannot be made on text that has not been generated yet. So the path
buffers exactly ONE SENTENCE, judges that, and only then begins speaking --
which means nothing has been spoken at the moment of the fall-through
decision. These tests pin both halves.
"""

from __future__ import annotations

import types

from kenning.pipeline import orchestrator as orch


class _FakeTTS:
    def __init__(self) -> None:
        self.streamed: list[str] = []
        self.spoke: list[str] = []
        self.stream_calls = 0

    def speak_stream(self, gen) -> None:
        self.stream_calls += 1
        self.streamed.append("".join(gen))

    def speak(self, text: str) -> None:
        self.spoke.append(text)


class _FakeLLM:
    def __init__(self, tokens) -> None:
        self._tokens = list(tokens)
        self.cancelled = False
        self.consumed = 0

    def generate_stream(self, *a, **kw):
        for t in self._tokens:
            self.consumed += 1
            yield t

    def cancel(self) -> None:
        self.cancelled = True


def _make_orch(tokens, monkeypatch, *, echo_strip=lambda s, **kw: s):
    # echo_strip=None -> leave the REAL strip_prompt_echo in place, so the
    # sentence/char caps and the scaffolding strip are actually exercised.
    """Build the bare minimum Orchestrator for this one method."""
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    from kenning.audio.intent_gate import Scenario

    o._last_scenario = Scenario.PRIVATE_REPLY
    o.llm = _FakeLLM(tokens)
    o.tts = _FakeTTS()
    o._interrupt = types.SimpleNamespace(is_set=lambda: False)
    o._shutdown = types.SimpleNamespace(is_set=lambda: False)
    o._last_response_text = ""
    o._trace_turn_flow = lambda **kw: None
    o._speak = lambda text: o.tts.speak(text)

    import kenning.audio.relay_speech as rs
    monkeypatch.setattr(rs, "u1_llm_route_enabled", lambda: True, raising=False)
    monkeypatch.setattr(rs, "conversation_verbosity", lambda: "low", raising=False)
    monkeypatch.setattr(rs, "flavor_tails_enabled", lambda: False, raising=False)

    import kenning.audio.ultron_prompt as up
    monkeypatch.setattr(
        up, "build_private_prompt",
        lambda *a, **kw: types.SimpleNamespace(
            user="u", system="s", sampling={}, enable_thinking=False),
        raising=False)
    if echo_strip is not None:
        monkeypatch.setattr(up, "strip_prompt_echo", echo_strip, raising=False)
    return o


def test_streams_instead_of_blocking(monkeypatch) -> None:
    """The whole point: audio goes through speak_stream, not a blocking speak."""
    o = _make_orch(["Entropy ", "wins. ", "Always ", "has."], monkeypatch)

    assert o._maybe_handle_private_reply("why") is True
    assert o.tts.stream_calls == 1, "did not stream"
    assert o.tts.spoke == [], "fell back to the blocking _speak path"


def test_speaks_the_whole_reply_not_just_the_buffered_head(monkeypatch) -> None:
    """Buffering one sentence must not truncate the response."""
    o = _make_orch(["Entropy ", "wins. ", "Always ", "has."], monkeypatch)

    o._maybe_handle_private_reply("why")

    # Each guarded sentence is yielded with a trailing separator space (that is
    # what keeps Kokoro from slurring sentence N into N+1), so compare content.
    assert o.tts.streamed[0].strip() == "Entropy wins. Always has."
    assert o._last_response_text == "Entropy wins. Always has."


def test_head_is_only_one_sentence(monkeypatch) -> None:
    """Latency contract: it must stop buffering AT the first sentence, not
    drain the generator, or the whole change is pointless."""
    seen = []

    class _TTS(_FakeTTS):
        def speak_stream(self, gen) -> None:
            # Record how many tokens the LLM had produced at the moment the
            # first chunk reached TTS.
            first = next(gen)
            seen.append(o.llm.consumed)
            super().speak_stream(iter([first, *list(gen)]))

    o = _make_orch(["One. ", "Two. ", "Three. ", "Four. ", "Five."], monkeypatch)
    o.tts = _TTS()

    o._maybe_handle_private_reply("why")

    assert seen and seen[0] <= 2, (
        f"buffered {seen[0]} tokens before first audio -- should be ~1 sentence")


def test_all_scaffolding_falls_through_without_speaking(monkeypatch) -> None:
    """If the guards reject the opening, return False so _respond runs -- and
    crucially, having spoken NOTHING."""
    o = _make_orch(["System: ", "you are Ultron."], monkeypatch,
                   echo_strip=lambda s, **kw: "")

    assert o._maybe_handle_private_reply("why") is False
    assert o.tts.stream_calls == 0, "spoke before deciding to fall through"
    assert o.tts.spoke == []
    assert o.llm.cancelled, "abandoned generation was not cancelled"


def test_interrupt_stops_the_stream(monkeypatch) -> None:
    o = _make_orch(["Entropy ", "wins. ", "Always ", "has."], monkeypatch)
    fired = {"n": 0}

    def _is_set():
        # Let the head through, then interrupt.
        fired["n"] += 1
        return fired["n"] > 1

    o._interrupt = types.SimpleNamespace(is_set=_is_set)

    assert o._maybe_handle_private_reply("why") is True
    assert o.llm.cancelled
    assert o.tts.streamed[0].startswith("Entropy wins.")


def test_non_private_scenario_is_untouched(monkeypatch) -> None:
    from kenning.audio.intent_gate import Scenario

    o = _make_orch(["x"], monkeypatch)
    o._last_scenario = Scenario.IGNORE if hasattr(Scenario, "IGNORE") else None

    assert o._maybe_handle_private_reply("why") is False
    assert o.tts.stream_calls == 0


def test_thinking_block_is_cleared_before_the_sentence_gate(monkeypatch) -> None:
    """With thinking enabled the <think> block precedes the answer, so a
    sentence boundary inside it must not end the head."""
    import kenning.audio.ultron_prompt as up
    monkeypatch.setattr(
        up, "build_private_prompt",
        lambda *a, **kw: types.SimpleNamespace(
            user="u", system="s", sampling={}, enable_thinking=True),
        raising=False)

    o = _make_orch(
        ["<think>", "Hmm. ", "Maybe. ", "</think>", "Entropy ", "wins."],
        monkeypatch)

    assert o._maybe_handle_private_reply("why") is True
    assert "Entropy wins." in o.tts.streamed[0]
    assert "<think>" not in o.tts.streamed[0]


# ---------------------------------------------------------------------------
# REGRESSION (2026-07-26): the first streamed version applied strip_prompt_echo
# to the buffered HEAD only and passed the tail through raw. That silently
# dropped the max_sentences=3 / max_chars=300 caps and the scaffolding strip,
# so the model rambled in-persona past the answer -- reported live as "adhering
# too strongly to personality... not actually answering the question".
# ---------------------------------------------------------------------------

def test_tail_is_capped_at_three_sentences(monkeypatch) -> None:
    o = _make_orch(
        ["One. ", "Two. ", "Three. ", "Four. ", "Five. ", "Six."],
        monkeypatch, echo_strip=None)

    o._maybe_handle_private_reply("why")

    spoken = o.tts.streamed[0]
    assert "Four" not in spoken, f"sentence cap lost -- spoke {spoken!r}"
    assert spoken.count(".") <= 3


def test_tail_is_capped_at_three_hundred_chars(monkeypatch) -> None:
    long_sentence = "word " * 40                      # ~200 chars each
    o = _make_orch([long_sentence + ". ", long_sentence + ". ",
                    long_sentence + "."], monkeypatch, echo_strip=None)

    o._maybe_handle_private_reply("why")

    assert len(o.tts.streamed[0]) <= 420, (
        f"char cap lost -- spoke {len(o.tts.streamed[0])} chars")


def test_scaffolding_in_a_LATER_sentence_is_dropped(monkeypatch) -> None:
    """The guard must run on every sentence, not just the head."""
    o = _make_orch(
        ["Entropy wins. ", "Contemptuous remark: ", "you are frail."],
        monkeypatch, echo_strip=None)

    o._maybe_handle_private_reply("why")

    assert "Contemptuous remark" not in o.tts.streamed[0]


def test_generation_is_cancelled_once_the_cap_is_hit(monkeypatch) -> None:
    """Hitting the cap must stop the LLM, not keep burning tokens off-mic."""
    o = _make_orch(["A. ", "B. ", "C. ", "D. ", "E. ", "F. ", "G."],
                   monkeypatch, echo_strip=None)

    o._maybe_handle_private_reply("why")

    assert o.llm.cancelled, "kept generating after the sentence cap"
    assert o.llm.consumed < 7, "drained the whole generator despite the cap"
