"""End-to-end pipeline test harness (2026-05-22 autonomous run).

Exercises the full Ultron voice pipeline programmatically:
    Kokoro (synthesize a user utterance as audio)
      -> Moonshine (transcribe back to text)
      -> Gate classifier (SEARCH vs NO_SEARCH)
      -> LLM (generate response)
      -> Kokoro (synthesize response audio)

The point is to test EVERY subsystem with real model loads + real
audio + real text, not just mocks. Designed to run in a single
process so we don't pay startup cost per scenario.

Usage:
    python scripts/autonomous_e2e_harness.py [--phase=N]

Phases (each is independent; --phase=all runs everything):
    1   STT (Moonshine accuracy + latency)
    2   LLM (Qwen 3.5 4B response + TTFT + latency)
    3   TTS (Kokoro Ultron voice synth + spectral_smooth)
    4   Web search (SearXNG + readers + ranker)
    5   Memory (retrieve + rerank + ranking)
    6   Routing classifier
    7   Gate (rule path + preflight LLM)
    8   Spoken-command matrix -- EVERY RoutingIntentKind through the real
        Kokoro-synth -> Moonshine-STT acoustic path, with an enum-coverage
        guard (classification only; nothing is dispatched)
    9   Spoken short-circuit matchers (deep research / recall / history /
        code exploration / evolution / report concern / run / scrap /
        local clock) + a negative control
    10  Full conversational loops -- audio -> STT -> gate -> LLM (+ history
        recall) -> Ultron-voice TTS, incl. a LIVE web-search turn
    11  Voice coding engineer -- REAL coding-CLI create -> sandbox run ->
        edit follow-up -> re-run (costs real API tokens)

The harness PRINTS to stdout (no stdin reads) so it can run
autonomously. Bugs surfaced here trigger immediate code edits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the project root importable when this script is invoked directly
# (i.e., `python scripts/autonomous_e2e_harness.py`). Otherwise the
# legacy ``from config import settings`` imports in src/ultron/* break.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np


# ---------------------------------------------------------------------------
# Lightweight scenario record
# ---------------------------------------------------------------------------


class Scenario:
    """One named test scenario with expected behavior + recorded timings."""

    def __init__(self, name: str, input_text: str, *, expected: Optional[Dict[str, Any]] = None):
        self.name = name
        self.input_text = input_text
        self.expected = expected or {}
        self.timings: Dict[str, float] = {}
        self.outputs: Dict[str, Any] = {}
        self.errors: List[str] = []

    def record(self, phase: str, value: float) -> None:
        self.timings[phase] = value

    def out(self, key: str, value: Any) -> None:
        self.outputs[key] = value

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def summary(self) -> str:
        ok = "OK" if not self.errors else "FAIL"
        timing_str = " ".join(f"{k}={v*1000:.0f}ms" for k, v in self.timings.items())
        return f"[{ok}] {self.name}: {timing_str}"


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


_REPORT: Dict[str, List[Scenario]] = {}


def _save_phase(phase: str, scenarios: List[Scenario]) -> None:
    _REPORT[phase] = scenarios


def _free_gpu() -> None:
    """Release GPU memory between phases. Each phase constructs its OWN engines
    (LLM / TTS / STT / embedder) as locals; without freeing between phases the
    full run accumulates VRAM past the budget and later phases fail to load
    (every phase passes in isolation -- the full run was the only thing that
    exhausted VRAM). Fail-open: torch missing / no CUDA -> no-op."""
    try:
        import gc

        gc.collect()
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def phase_stt() -> List[Scenario]:
    """Test Moonshine on diverse synthesized + bundled audio."""
    print("\n" + "=" * 60)
    print("PHASE 1: STT (Moonshine)")
    print("=" * 60)
    scenarios: List[Scenario] = []
    from ultron.transcription.moonshine_engine import MoonshineEngine

    # Use the same engine the orchestrator uses.
    t0 = time.monotonic()
    engine = MoonshineEngine()
    load_s = time.monotonic() - t0
    print(f"Moonshine loaded in {load_s:.2f}s, supports_streaming={engine.supports_streaming()}")

    # Generate test audio using Kokoro with the STOCK am_michael voice
    # (NOT the Ultron fine-tune -- we want neutral synthesis so the
    # STT test isn't biased by a specific dialect).
    from ultron.tts.kokoro_engine import KokoroSpeech

    t0 = time.monotonic()
    tts_for_test = KokoroSpeech(voice="am_michael", apply_spectral_smooth=False)
    tts_for_test.warmup()
    print(f"Test-input Kokoro (am_michael) ready in {time.monotonic()-t0:.2f}s")

    test_phrases = [
        ("greeting", "Hello there."),
        ("time_query", "What time is it in France?"),
        ("math", "What is two plus two?"),
        ("tech_jargon", "Show me the cosine similarity between two vectors."),
        ("multi_sentence", "First, tell me the date. Then, give me the weather."),
        ("question_with_proper_noun", "Who is the current president of the United States?"),
        # A realistic short command. A bare single word ("Stop.") transcribes
        # unreliably (too little acoustic content) AND isn't representative --
        # real barge-in / stop is ACOUSTIC (VAD + wake word), never STT text.
        ("short_command", "Cancel the task."),
        ("longer_question", "Can you explain how a transformer model uses attention to relate tokens to each other in a sequence?"),
    ]

    for name, text in test_phrases:
        sc = Scenario(name=f"stt:{name}", input_text=text)
        try:
            # Synthesize as audio
            t0 = time.monotonic()
            pcm, sr = tts_for_test._synthesize(text)  # internal API ok for test
            sc.record("synth", time.monotonic() - t0)
            sc.out("synth_samples", len(pcm))
            sc.out("synth_sr", sr)
            # Convert int16 -> float32 for Moonshine
            audio_f32 = pcm.astype(np.float32) / 32768.0
            # Resample to 16k if needed (Kokoro is 24k, Moonshine wants 16k)
            if sr != 16000:
                try:
                    import scipy.signal
                    new_len = int(len(audio_f32) * 16000 / sr)
                    audio_f32 = scipy.signal.resample(audio_f32, new_len).astype(np.float32)
                except Exception:
                    # Quick-and-dirty linear resample fallback
                    factor = sr / 16000.0
                    indices = (np.arange(int(len(audio_f32) / factor)) * factor).astype(np.int64)
                    audio_f32 = audio_f32[indices]
            t0 = time.monotonic()
            transcript = engine.transcribe(audio_f32)
            sc.record("stt", time.monotonic() - t0)
            sc.out("transcript", transcript)
            # Soft validation: transcript should share at least 30% of
            # the input's lowercase non-punct words. Strip punctuation
            # so "Stop." vs "Stop" doesn't false-fail.
            import re as _re
            def _norm(s):
                return set(w for w in _re.findall(r"[\w']+", s.lower()) if w)
            in_words = _norm(text)
            out_words = _norm(transcript)
            overlap = len(in_words & out_words) / max(1, len(in_words))
            sc.out("word_overlap", overlap)
            if overlap < 0.3:
                sc.err(f"low word_overlap={overlap:.2f}: input={text!r} transcript={transcript!r}")
        except Exception as e:
            sc.err(f"exception: {e}\n{traceback.format_exc()}")
        scenarios.append(sc)
        print(sc.summary())

    return scenarios


def phase_llm() -> List[Scenario]:
    """Test LLM responses across scenarios."""
    print("\n" + "=" * 60)
    print("PHASE 2: LLM (Qwen 3.5 4B)")
    print("=" * 60)
    scenarios: List[Scenario] = []
    from ultron.llm.inference import LLMEngine
    from ultron.memory.embedder import HybridEmbedder
    from ultron.memory.qdrant_store import ConversationMemory

    t0 = time.monotonic()
    embedder = HybridEmbedder()
    memory = ConversationMemory(embedder=embedder)
    print(f"Memory loaded in {time.monotonic()-t0:.2f}s")
    t0 = time.monotonic()
    llm = LLMEngine(memory=memory)
    print(f"LLM loaded in {time.monotonic()-t0:.2f}s")

    # Production-style: brevity hint applied + thinking OFF (matches
    # the orchestrator's voice-path generate_stream calls). Without
    # these, the harness was testing a non-production configuration
    # and reported false-positive 10+ s TTFTs on the thinking path.
    from ultron.response_style import apply_brevity_hint as _bh

    raw_prompts = [
        ("greeting", "Say hello."),
        ("factual", "What is 7 multiplied by 8?"),
        ("identity", "Who are you?"),
        ("recipe", "Give me a one-sentence recipe for a basic cake."),
        ("multi_turn_setup", "Remember the number 42 for our next exchange."),
    ]

    for name, raw_prompt in raw_prompts:
        prompt = _bh(raw_prompt)  # voice-path brevity hint
        sc = Scenario(name=f"llm:{name}", input_text=raw_prompt)
        try:
            t0 = time.monotonic()
            chunks = []
            ttft = None
            for tok in llm.generate_stream(
                prompt,
                record_history=False,
                rag_query=raw_prompt[:80],
                enable_thinking=False,  # voice path matches orchestrator
            ):
                if ttft is None:
                    ttft = time.monotonic() - t0
                chunks.append(tok)
                if len(chunks) > 200:
                    break
            elapsed = time.monotonic() - t0
            sc.record("ttft", ttft or 0.0)
            sc.record("total", elapsed)
            response = "".join(chunks).strip()
            sc.out("response", response)
            sc.out("chars", len(response))
            if not response:
                sc.err("empty response")
            if len(response) > 2000:
                sc.err(f"response too long ({len(response)} chars; brief-style was expected to limit)")
        except Exception as e:
            sc.err(f"exception: {e}")
        scenarios.append(sc)
        print(sc.summary())
        if sc.outputs.get("response"):
            preview = sc.outputs["response"][:100].replace("\n", " ")
            print(f"  -> {preview!r}")

    # Release the Qdrant local-mode file lock so a later phase (memory) can open
    # the same path -- local Qdrant allows only one client per path at a time.
    try:
        memory.close()
    except Exception:
        pass
    return scenarios


def phase_tts() -> List[Scenario]:
    """Test Kokoro Ultron voice synth."""
    print("\n" + "=" * 60)
    print("PHASE 3: TTS (Kokoro -- Ultron voice)")
    print("=" * 60)
    scenarios: List[Scenario] = []
    from ultron.config import get_config
    from ultron.tts.kokoro_engine import KokoroSpeech

    cfg = get_config().tts.kokoro
    t0 = time.monotonic()
    engine = KokoroSpeech(
        voice=cfg.voice,
        device=cfg.device,
        speed=cfg.speed,
        apply_spectral_smooth=cfg.apply_spectral_smooth,
        spectral_smooth_window=cfg.spectral_smooth_window,
    )
    engine.warmup()
    print(f"Ultron-voice Kokoro ready in {time.monotonic()-t0:.2f}s; voice={engine._voice_display}")

    phrases = [
        ("short", "Online."),
        ("medium", "I am Ultron. I am here."),
        ("technical", "Cross-encoder reranking is now active."),
        ("longer", "According to the search, the current time in France is 9:25 PM on Friday, May 22, 2026."),
    ]
    for name, text in phrases:
        sc = Scenario(name=f"tts:{name}", input_text=text)
        try:
            t0 = time.monotonic()
            pcm, sr = engine._synthesize(text)
            sc.record("synth", time.monotonic() - t0)
            sc.out("samples", len(pcm))
            sc.out("duration_s", len(pcm) / sr)
            sc.out("peak", int(abs(pcm).max()) if len(pcm) else 0)
            sc.out("rms", float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))))
            # Sanity: PCM should be int16, peak should be > 0 but not clipped
            if len(pcm) == 0:
                sc.err("zero samples produced")
            elif abs(pcm).max() == 0:
                sc.err("silent output (peak=0)")
            elif abs(pcm).max() >= 32767:
                sc.err(f"clipping detected (peak=32767)")
            # Check duration is reasonable
            dur = len(pcm) / sr
            if dur < 0.05:
                sc.err(f"too short ({dur:.2f}s)")
            if dur > 30.0:
                sc.err(f"unreasonably long ({dur:.2f}s)")
        except Exception as e:
            sc.err(f"exception: {e}\n{traceback.format_exc()[:500]}")
        scenarios.append(sc)
        print(sc.summary())

    return scenarios


def phase_web_search() -> List[Scenario]:
    """Test the search provider chain + reader chain + ranker."""
    print("\n" + "=" * 60)
    print("PHASE 4: Web Search")
    print("=" * 60)
    scenarios: List[Scenario] = []
    from ultron.web_search.provider_chain import SearchProviderChain
    from ultron.web_search.reader_chain import ReaderChain

    t0 = time.monotonic()
    providers = SearchProviderChain()
    readers = ReaderChain()
    print(f"Search chains constructed in {time.monotonic()-t0:.2f}s")

    queries = [
        ("factual_current", "current time in France"),
        ("tech", "what is the latest python version"),
        ("simple", "weather in San Francisco"),
    ]
    for name, q in queries:
        sc = Scenario(name=f"search:{name}", input_text=q)
        try:
            t0 = time.monotonic()
            results = providers.search(q, count=5)
            sc.record("provider", time.monotonic() - t0)
            sc.out("result_count", len(results))
            sc.out("first_url", results[0].url if results else None)
            if not results:
                sc.err("no results")
                scenarios.append(sc)
                print(sc.summary())
                continue
            # Try reading the first URL
            t0 = time.monotonic()
            content = readers.fetch(results[0].url)
            sc.record("reader", time.monotonic() - t0)
            sc.out("content_chars", len(content) if content else 0)
            if not content:
                sc.err("reader returned empty")
        except Exception as e:
            sc.err(f"exception: {e}")
        scenarios.append(sc)
        print(sc.summary())
    return scenarios


def phase_memory() -> List[Scenario]:
    """Test memory retrieve + rerank."""
    print("\n" + "=" * 60)
    print("PHASE 5: Memory + Reranker")
    print("=" * 60)
    scenarios: List[Scenario] = []
    from ultron.memory.embedder import HybridEmbedder
    from ultron.memory.qdrant_store import ConversationMemory

    t0 = time.monotonic()
    embedder = HybridEmbedder()
    memory = ConversationMemory(embedder=embedder)
    print(f"Memory loaded in {time.monotonic()-t0:.2f}s")

    queries = [
        ("short", "hello"),
        ("question", "what time is it"),
        ("technical", "explain how transformers work"),
    ]
    for name, q in queries:
        sc = Scenario(name=f"memory:{name}", input_text=q)
        try:
            t0 = time.monotonic()
            results = memory.retrieve(q, k=5)
            sc.record("retrieve", time.monotonic() - t0)
            sc.out("result_count", len(results))
            if results:
                sc.out("top_preview", results[0].content[:80] if results else None)
        except Exception as e:
            sc.err(f"exception: {e}")
        scenarios.append(sc)
        print(sc.summary())
    try:
        memory.close()
    except Exception:
        pass
    return scenarios


def phase_routing() -> List[Scenario]:
    """Test routing classifier."""
    print("\n" + "=" * 60)
    print("PHASE 6: Routing")
    print("=" * 60)
    scenarios: List[Scenario] = []
    from ultron.openclaw_routing.classifier import classify_routing

    tests = [
        ("greeting", "hello", "conversational"),
        ("conversational", "how are you", "conversational"),
        ("model_switch_4b", "switch to the 4B", "MODEL_SWITCH"),
        ("model_switch_gemma", "switch to gemma", "MODEL_SWITCH"),
        ("image_search", "show me a picture of a cat", "APP_LAUNCH"),
        ("app_launch", "open chrome", "APP_LAUNCH"),
        ("media_gen", "make an image of a sunset", "MEDIA_GENERATION"),
    ]
    for name, text, expected_kind in tests:
        sc = Scenario(name=f"route:{name}", input_text=text, expected={"kind": expected_kind})
        try:
            t0 = time.monotonic()
            verdict = classify_routing(text)
            sc.record("classify", time.monotonic() - t0)
            actual = verdict.kind.name if hasattr(verdict, "kind") else str(verdict)
            sc.out("actual_kind", actual)
            sc.out("confidence", getattr(verdict, "confidence", None))
            if expected_kind != actual and not (expected_kind == "conversational" and "conversational" in actual.lower()):
                sc.err(f"routing mismatch: expected={expected_kind} actual={actual}")
        except Exception as e:
            sc.err(f"exception: {e}")
        scenarios.append(sc)
        print(sc.summary())
    return scenarios


def phase_gate() -> List[Scenario]:
    """Test web-search gate (rule path + preflight LLM)."""
    print("\n" + "=" * 60)
    print("PHASE 7: Web-search Gate")
    print("=" * 60)
    scenarios: List[Scenario] = []
    from ultron.web_search.gating import classify_by_rules

    # Rule-path tests: each must EITHER match a rule OR fall through
    # (None) to the LLM preflight stage. Greetings/acks should match;
    # everything else should fall through.
    rule_tests = [
        # (name, text, must_match_rule, expected_decision_if_matched)
        ("greeting_hello", "hello", True, "NO_SEARCH"),
        ("greeting_hi", "hi", True, "NO_SEARCH"),
        ("ack_okay", "okay", True, "NO_SEARCH"),
        ("ack_thanks", "thanks", True, "NO_SEARCH"),
        ("ultron_hello", "Ultron, hello", True, "NO_SEARCH"),
        ("fall_through_question", "how are you", False, None),
        # Stable-fact questions get caught by the conceptual-stem rule
        # (capital of France is stable knowledge, no search needed).
        ("stable_fact", "what is the capital of France", True, "NO_SEARCH"),
        # Time-sensitive marker ("current") triggers the SEARCH rule.
        ("volatile_fact", "what is the current stock price of Tesla", True, "SEARCH"),
        # Imperatives / personal-advice queries fall through (no
        # matching pattern in either rule branch).
        ("fall_through_imperative", "invent a story for me", False, None),
    ]
    for name, text, must_match, expected_decision in rule_tests:
        sc = Scenario(name=f"gate-rules:{name}", input_text=text)
        try:
            t0 = time.monotonic()
            verdict = classify_by_rules(text)
            sc.record("rules", time.monotonic() - t0)
            if verdict is None:
                sc.out("actual_decision", "FELL_THROUGH")
                if must_match:
                    sc.err(f"expected a rule match but got fell-through")
            else:
                actual = verdict.decision.name
                sc.out("actual_decision", actual)
                if must_match and expected_decision not in actual:
                    sc.err(f"gate mismatch: expected={expected_decision} actual={actual}")
                if not must_match:
                    sc.err(f"expected fall-through but rule matched: {actual}")
        except Exception as e:
            sc.err(f"exception: {e}")
        scenarios.append(sc)
        print(sc.summary())
    return scenarios


# ---------------------------------------------------------------------------
# Shared audio helper for the spoken-command phases
# ---------------------------------------------------------------------------


def _spoken_transcript(tts: Any, stt: Any, text: str, sc: "Scenario") -> str:
    """Synthesize ``text`` (neutral voice) -> transcribe -> return transcript.

    The acoustic round-trip the spoken-command phases share: it is the
    REAL path a user's utterance takes, so a phrasing that survives it
    is a phrasing the live system can route. Records synth + stt
    timings on the scenario. Raises on hard failure (caller records)."""
    t0 = time.monotonic()
    pcm, sr = tts._synthesize(text)  # internal API ok for the harness
    sc.record("synth", time.monotonic() - t0)
    audio_f32 = pcm.astype(np.float32) / 32768.0
    if sr != 16000:
        try:
            import scipy.signal

            new_len = int(len(audio_f32) * 16000 / sr)
            audio_f32 = scipy.signal.resample(audio_f32, new_len).astype(np.float32)
        except Exception:
            factor = sr / 16000.0
            indices = (np.arange(int(len(audio_f32) / factor)) * factor).astype(np.int64)
            audio_f32 = audio_f32[indices]
    # Replicate the VAD-bounded capture shape: the live pipeline always
    # hands STT a window with leading pre-roll + trailing post-speech
    # silence (the capture only closes after ~0.5-1.2 s of quiet). A
    # bare synthesized clip ends ON the last sample, which truncates the
    # decoder mid-word ("Cancel the task." -> "Cancel the time.") -- a
    # harness artifact the real mic path never produces.
    lead = np.zeros(int(0.50 * 16000), dtype=np.float32)
    tail = np.zeros(int(0.80 * 16000), dtype=np.float32)
    audio_f32 = np.concatenate([lead, audio_f32, tail])
    t0 = time.monotonic()
    transcript = stt.transcribe(audio_f32)
    sc.record("stt", time.monotonic() - t0)
    sc.out("transcript", transcript)
    return transcript or ""


def _build_spoken_pipeline() -> Tuple[Any, Any]:
    """Construct the neutral-voice Kokoro + Moonshine pair the spoken-command
    phases share. Stock am_michael (not the Ultron fine-tune) so the STT leg
    isn't biased by a specific timbre."""
    from ultron.transcription.moonshine_engine import MoonshineEngine
    from ultron.tts.kokoro_engine import KokoroSpeech

    t0 = time.monotonic()
    stt = MoonshineEngine()
    print(f"Moonshine loaded in {time.monotonic()-t0:.2f}s")
    t0 = time.monotonic()
    tts = KokoroSpeech(voice="am_michael", apply_spectral_smooth=False)
    tts.warmup()
    print(f"Test-input Kokoro ready in {time.monotonic()-t0:.2f}s")
    return tts, stt


# ---------------------------------------------------------------------------
# Phase 8: the full spoken-command matrix (every RoutingIntentKind)
# ---------------------------------------------------------------------------


def phase_commands() -> List[Scenario]:
    """Drive a spoken command for EVERY RoutingIntentKind through the real
    acoustic path (Kokoro synth -> Moonshine STT) and assert the routing
    classifier still lands on the right intent from the TRANSCRIPT.

    Classification only -- no handler is dispatched, so nothing clicks,
    types, launches, or closes anything (safe to run unattended). The
    closing coverage scenario asserts the matrix spans the ENTIRE enum,
    so adding a RoutingIntentKind without a spoken-command test fails
    this suite loudly.
    """
    print("\n" + "=" * 60)
    print("PHASE 8: Spoken-command matrix (all routing intents)")
    print("=" * 60)
    scenarios: List[Scenario] = []
    from ultron.openclaw_routing.classifier import classify_routing
    from ultron.openclaw_routing.intents import RoutingIntentKind

    tts, stt = _build_spoken_pipeline()

    # (name, spoken text, expected kind, classify kwargs)
    matrix = [
        ("conversational", "How are you doing today?",
         RoutingIntentKind.CONVERSATIONAL, {}),
        ("code_task", "Write me a program that converts celsius to fahrenheit.",
         RoutingIntentKind.CODE_TASK, {}),
        ("progress_query", "How is the project going?",
         RoutingIntentKind.PROGRESS_QUERY, {"has_active_coding_task": True}),
        ("cancel", "Cancel the task.",
         RoutingIntentKind.CANCEL, {"has_active_coding_task": True}),
        ("mid_session_adjustment", "Add a dark mode to it.",
         RoutingIntentKind.MID_SESSION_ADJUSTMENT,
         {"has_active_coding_task": True}),
        ("clarification_response", "Use the default settings for that.",
         RoutingIntentKind.CLARIFICATION_RESPONSE,
         {"has_pending_clarification": True}),
        ("browser_automation", "Open hacker news in the browser.",
         RoutingIntentKind.BROWSER_AUTOMATION, {}),
        ("media_generation", "Make an image of a sunset over the ocean.",
         RoutingIntentKind.MEDIA_GENERATION, {}),
        ("messaging", "Send a message to my phone saying dinner is ready.",
         RoutingIntentKind.MESSAGING, {}),
        ("file_operation", "Show me the files in my downloads folder.",
         RoutingIntentKind.FILE_OPERATION, {}),
        ("shell_operation", "In the terminal run git status.",
         RoutingIntentKind.SHELL_OPERATION, {}),
        ("hybrid_task", "Write a script that opens chrome and checks the news.",
         RoutingIntentKind.HYBRID_TASK, {}),
        ("model_switch", "Switch to the four B model.",
         RoutingIntentKind.MODEL_SWITCH, {}),
        ("system_status", "Give me a system status report.",
         RoutingIntentKind.SYSTEM_STATUS, {}),
        ("gaming_mode", "Engage gaming mode.",
         RoutingIntentKind.GAMING_MODE, {}),
        ("desktop_automation", "Take a screenshot of the desktop.",
         RoutingIntentKind.DESKTOP_AUTOMATION, {}),
        ("window_automation", "Focus the chrome window.",
         RoutingIntentKind.WINDOW_AUTOMATION, {}),
        ("app_launch", "Open chrome on monitor two.",
         RoutingIntentKind.APP_LAUNCH, {}),
        ("screen_context_query", "Explain what I'm looking at on screen.",
         RoutingIntentKind.SCREEN_CONTEXT_QUERY, {}),
        ("window_move", "Put discord on my right monitor.",
         RoutingIntentKind.WINDOW_MOVE, {}),
        ("window_close", "Close the notepad window.",
         RoutingIntentKind.WINDOW_CLOSE, {}),
        ("open_last_source", "Show me that article.",
         RoutingIntentKind.OPEN_LAST_SOURCE, {}),
        ("navigate_to_site", "Take me to the youtube website.",
         RoutingIntentKind.NAVIGATE_TO_SITE, {}),
        ("active_window_query", "What window is active right now?",
         RoutingIntentKind.ACTIVE_WINDOW_QUERY, {}),
        ("semantic_click", "Click the save button.",
         RoutingIntentKind.SEMANTIC_CLICK, {}),
        # "Go ahead." over a bare "Yes." -- same lesson as the phase-1
        # "Stop." finding: a single clipped word carries too little
        # acoustic content to transcribe reliably, and the yes-pattern
        # accepts the richer phrasing.
        ("window_close_confirmation", "Go ahead.",
         RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION, {}),
    ]

    tested_kinds = set()
    for name, spoken, expected_kind, kwargs in matrix:
        tested_kinds.add(expected_kind)
        sc = Scenario(
            name=f"cmd:{name}", input_text=spoken,
            expected={"kind": expected_kind.name},
        )
        try:
            transcript = _spoken_transcript(tts, stt, spoken, sc)
            if not transcript.strip():
                sc.err("empty transcript")
            else:
                t0 = time.monotonic()
                verdict = classify_routing(transcript, **kwargs)
                sc.record("classify", time.monotonic() - t0)
                sc.out("actual_kind", verdict.kind.name)
                sc.out("confidence", verdict.confidence)
                if verdict.kind is not expected_kind:
                    sc.err(
                        f"routing mismatch: spoken={spoken!r} "
                        f"transcript={transcript!r} "
                        f"expected={expected_kind.name} actual={verdict.kind.name}"
                    )
        except Exception as e:
            sc.err(f"exception: {e}\n{traceback.format_exc()[:500]}")
        scenarios.append(sc)
        print(sc.summary())

    # Completeness guard: the matrix must span the ENTIRE enum.
    sc = Scenario(name="cmd:enum_coverage", input_text="(coverage check)")
    untested = sorted(
        k.name for k in RoutingIntentKind if k not in tested_kinds
    )
    sc.out("untested_kinds", untested)
    if untested:
        sc.err(
            "RoutingIntentKind values without a spoken-command scenario: "
            + ", ".join(untested)
        )
    scenarios.append(sc)
    print(sc.summary())
    return scenarios


# ---------------------------------------------------------------------------
# Phase 9: spoken short-circuit matchers (the orchestrator's strict matchers)
# ---------------------------------------------------------------------------


def phase_short_circuits() -> List[Scenario]:
    """Drive a spoken phrase for every orchestrator run-loop short-circuit
    (deep research / deep recall / history recall / code exploration /
    evolution / report concern / run + launch / scrap / local clock) through
    the real acoustic path and assert each STRICT matcher fires on the
    transcript -- and that ordinary conversation trips NONE of them.

    Matcher-level only: no loop is executed, so this is fast + safe."""
    print("\n" + "=" * 60)
    print("PHASE 9: Spoken short-circuit matchers")
    print("=" * 60)
    scenarios: List[Scenario] = []
    from ultron import local_clock_reply
    from ultron.coding.sandbox_runner import match_run_program
    from ultron.coding.scrap import match_scrap_command
    from ultron.evolution.intent import match_evolution_command
    from ultron.feedback.report_intent import match_report_concern
    from ultron.memory.deep_recall import match_deep_recall
    from ultron.memory.history_recall import match_history_recall
    from ultron.search.code_exploration import match_code_exploration
    from ultron.web_search.deep_research import match_deep_research

    tts, stt = _build_spoken_pipeline()

    matchers = [
        ("deep_research", "Do a deep dive on quantum computing.",
         lambda t: match_deep_research(t) is not None),
        ("deep_recall", "Search your memory thoroughly for everything about my dog.",
         lambda t: match_deep_recall(t) is not None),
        ("history_recall", "What did I say earlier about the budget?",
         lambda t: match_history_recall(t) is not None),
        ("code_exploration", "Search the codebase for the wake word handler.",
         lambda t: match_code_exploration(t) is not None),
        ("evolution_status", "Evolution status.",
         lambda t: match_evolution_command(t) is not None),
        ("report_concern", "That answer was wrong.",
         lambda t: match_report_concern(t) is not None),
        ("run_program", "Run the calculator.",
         lambda t: match_run_program(t) is not None),
        ("scrap", "Scrap the whole thing.",
         lambda t: bool(match_scrap_command(t))),
        ("local_clock", "What time is it?",
         lambda t: local_clock_reply.maybe_local_clock_reply(t) is not None),
    ]
    for name, spoken, fires in matchers:
        sc = Scenario(name=f"shortcut:{name}", input_text=spoken)
        try:
            transcript = _spoken_transcript(tts, stt, spoken, sc)
            if not transcript.strip():
                sc.err("empty transcript")
            elif not fires(transcript):
                sc.err(
                    f"matcher did not fire: spoken={spoken!r} "
                    f"transcript={transcript!r}"
                )
        except Exception as e:
            sc.err(f"exception: {e}\n{traceback.format_exc()[:500]}")
        scenarios.append(sc)
        print(sc.summary())

    # Negative control: ordinary conversation must trip NO strict matcher.
    sc = Scenario(
        name="shortcut:negative_control",
        input_text="Tell me a story about a brave little toaster.",
    )
    try:
        transcript = _spoken_transcript(
            tts, stt, sc.input_text, sc,
        )
        tripped = [name for name, _spoken, fires in matchers if fires(transcript)]
        sc.out("tripped", tripped)
        if tripped:
            sc.err(f"ordinary conversation tripped strict matchers: {tripped}")
    except Exception as e:
        sc.err(f"exception: {e}")
    scenarios.append(sc)
    print(sc.summary())
    return scenarios


# ---------------------------------------------------------------------------
# Phase 10: full conversational loops (audio -> STT -> gate -> LLM -> TTS)
# ---------------------------------------------------------------------------


def phase_full_loop() -> List[Scenario]:
    """The complete turn, end to end: synthesize the user's utterance,
    transcribe it, run the web-search gate, generate the LLM response
    (with in-context history), and synthesize the reply in the Ultron
    voice. Covers a NO_SEARCH turn, a remember -> recall pair (the
    response must contain the remembered fact), and a live SEARCH turn
    through the real provider/reader chains."""
    print("\n" + "=" * 60)
    print("PHASE 10: Full conversational loops")
    print("=" * 60)
    scenarios: List[Scenario] = []
    from ultron.config import get_config
    from ultron.llm.inference import LLMEngine
    from ultron.memory.embedder import HybridEmbedder
    from ultron.memory.qdrant_store import ConversationMemory
    from ultron.response_style import apply_brevity_hint
    from ultron.tts.kokoro_engine import KokoroSpeech
    from ultron.web_search.gating import classify_by_rules

    tts_in, stt = _build_spoken_pipeline()
    t0 = time.monotonic()
    embedder = HybridEmbedder()
    memory = ConversationMemory(embedder=embedder)
    llm = LLMEngine(memory=memory)
    print(f"LLM + memory ready in {time.monotonic()-t0:.2f}s")
    kcfg = get_config().tts.kokoro
    tts_out = KokoroSpeech(
        voice=kcfg.voice, device=kcfg.device, speed=kcfg.speed,
        apply_spectral_smooth=kcfg.apply_spectral_smooth,
    )
    tts_out.warmup()

    def _llm_turn(sc: Scenario, transcript: str, *, record: bool) -> str:
        prompt = apply_brevity_hint(transcript)
        t0 = time.monotonic()
        chunks: List[str] = []
        ttft = None
        for tok in llm.generate_stream(
            prompt,
            record_history=record,
            history_user_message=transcript if record else None,
            rag_query=transcript[:80],
            enable_thinking=False,
        ):
            if ttft is None:
                ttft = time.monotonic() - t0
            chunks.append(tok)
            if len(chunks) > 250:
                break
        sc.record("ttft", ttft or 0.0)
        sc.record("llm_total", time.monotonic() - t0)
        return "".join(chunks).strip()

    def _speak_out(sc: Scenario, response: str) -> None:
        t0 = time.monotonic()
        pcm, sr = tts_out._synthesize(response[:400])
        sc.record("tts_out", time.monotonic() - t0)
        sc.out("reply_audio_s", len(pcm) / sr if len(pcm) else 0.0)
        if len(pcm) == 0 or abs(pcm).max() == 0:
            sc.err("silent reply audio")

    # Turn 1: plain NO_SEARCH conversational turn.
    sc = Scenario(name="loop:no_search_turn", input_text="What is seven times eight?")
    try:
        transcript = _spoken_transcript(tts_in, stt, sc.input_text, sc)
        verdict = classify_by_rules(transcript)
        sc.out("gate", verdict.decision.name if verdict else "FELL_THROUGH")
        response = _llm_turn(sc, transcript, record=False)
        sc.out("response", response[:160])
        if not response:
            sc.err("empty response")
        elif "56" not in response and "fifty-six" not in response.lower():
            sc.err(f"expected 56 in the answer, got: {response[:120]!r}")
        else:
            _speak_out(sc, response)
    except Exception as e:
        sc.err(f"exception: {e}\n{traceback.format_exc()[:500]}")
    scenarios.append(sc)
    print(sc.summary())

    # Turns 2+3: remember -> recall through in-context history.
    sc = Scenario(
        name="loop:remember_recall",
        input_text="Remember that my locker code is four two seven.",
    )
    try:
        transcript = _spoken_transcript(tts_in, stt, sc.input_text, sc)
        first = _llm_turn(sc, transcript, record=True)
        sc.out("ack_response", first[:120])
        sc2_text = "What is my locker code?"
        t0 = time.monotonic()
        pcm, sr = tts_in._synthesize(sc2_text)
        audio = pcm.astype(np.float32) / 32768.0
        if sr != 16000:
            import scipy.signal

            audio = scipy.signal.resample(
                audio, int(len(audio) * 16000 / sr)
            ).astype(np.float32)
        recall_transcript = stt.transcribe(audio)
        sc.out("recall_transcript", recall_transcript)
        response = _llm_turn(sc, recall_transcript or sc2_text, record=True)
        sc.out("recall_response", response[:160])
        normalized = response.lower().replace(",", "").replace("-", " ")
        if not response:
            sc.err("empty recall response")
        elif not any(
            marker in normalized
            for marker in ("427", "four two seven", "4 2 7", "fourtwoseven")
        ):
            sc.err(f"locker code not recalled: {response[:160]!r}")
        else:
            _speak_out(sc, response)
    except Exception as e:
        sc.err(f"exception: {e}\n{traceback.format_exc()[:500]}")
    scenarios.append(sc)
    print(sc.summary())

    # Turn 4: live SEARCH turn through the real ladder.
    sc = Scenario(
        name="loop:search_turn",
        input_text="What is the current weather in San Francisco?",
    )
    try:
        from ultron.web_search.search import format_sources_for_prompt

        transcript = _spoken_transcript(tts_in, stt, sc.input_text, sc)
        verdict = classify_by_rules(transcript)
        decision = verdict.decision.name if verdict else "FELL_THROUGH"
        sc.out("gate", decision)
        if "SEARCH" not in decision:
            sc.err(f"expected the gate to say SEARCH, got {decision}")
        executor = _build_search_executor(llm)
        t0 = time.monotonic()
        payload = executor.run(transcript)
        sc.record("search", time.monotonic() - t0)
        sc.out("sources", len(payload.sources))
        if not payload.sources:
            sc.err("live search returned no sources")
        else:
            augmented = (
                f"User question: {transcript}\n\nFresh information from web "
                f"search:\n{format_sources_for_prompt(payload.sources)}\n\n"
                "Answer the question using the sources."
            )
            response = _llm_turn(sc, augmented, record=False)
            sc.out("response", response[:200])
            if not response:
                sc.err("empty search-augmented response")
            else:
                _speak_out(sc, response)
    except Exception as e:
        sc.err(f"exception: {e}\n{traceback.format_exc()[:500]}")
    scenarios.append(sc)
    print(sc.summary())

    try:
        memory.close()
    except Exception:
        pass
    return scenarios


def _build_search_executor(llm: Any) -> Any:
    """The production-shaped WebSearchExecutor (provider chain + reader
    chain), mirroring the orchestrator's construction exactly -- the
    ``brave`` / ``jina`` params are duck-typed chain instances."""
    from ultron.web_search.provider_chain import SearchProviderChain
    from ultron.web_search.reader_chain import ReaderChain
    from ultron.web_search.search import WebSearchExecutor

    return WebSearchExecutor(
        brave=SearchProviderChain(), jina=ReaderChain(), llm=llm, cache=None,
    )


# ---------------------------------------------------------------------------
# Phase 11: the voice coding engineer with the REAL coding CLI
# ---------------------------------------------------------------------------


def phase_coding() -> List[Scenario]:
    """Drive the voice coding engineer end to end with the REAL coding CLI:
    create a program in the sandbox, run it via the gated sandbox runner,
    send an edit follow-up to the SAME session, and run it again. Asserts
    files exist, the program output is exact, the edit landed, and the
    completion narration is speakable. Costs real API tokens (small
    haiku-tier tasks) and several minutes."""
    print("\n" + "=" * 60)
    print("PHASE 11: Voice coding engineer (real coding CLI)")
    print("=" * 60)
    scenarios: List[Scenario] = []
    import shutil
    import uuid

    from ultron.coding.bridge import TaskRequest
    from ultron.coding.runner import CodingTaskRunner, build_default_bridge
    from ultron.coding.sandbox_runner import run_program

    from config import settings as _settings

    sandbox_root = Path(_settings.CODING_SANDBOX_PATH)
    project = sandbox_root / f"e2e_coding_{uuid.uuid4().hex[:8]}"
    project.mkdir(parents=True, exist_ok=True)

    sc = Scenario(name="coding:create", input_text="create main.py printing E2E OK")
    runner = None
    try:
        runner = CodingTaskRunner(bridge=build_default_bridge())
        request = TaskRequest(
            task_prompt=(
                "Create a file named main.py in the current directory that "
                "prints exactly the text E2E OK (followed by a newline) and "
                "nothing else. Do not create any other files."
            ),
            cwd=project,
            model="haiku",
            require_testing=False,
            timeout_s=300.0,
            label="e2e coding create",
        )
        t0 = time.monotonic()
        handle = runner.start_task(request)
        result = handle.wait(timeout=320.0)
        sc.record("create_task", time.monotonic() - t0)
        sc.out("success", result.success)
        sc.out("summary", (result.summary or "")[:200])
        main_py = project / "main.py"
        if not result.success:
            sc.err(f"create task failed: {result.summary[:200]}")
        elif not main_py.is_file():
            sc.err("main.py was not created")
        else:
            narration = runner.completion_narration()
            sc.out("narration", (narration or "")[:160])
            if not narration:
                sc.err("no completion narration produced")
            run = run_program(
                project, sandbox_root=sandbox_root,
                project_name=project.name, timeout_s=60.0,
                user_text="run the program",
            )
            sc.out("run_stdout", run.stdout[:80])
            if not run.ok:
                sc.err(f"sandbox run failed: {run.error or run.stderr[:200]}")
            elif "E2E OK" not in run.stdout:
                sc.err(f"unexpected program output: {run.stdout[:120]!r}")
    except Exception as e:
        sc.err(f"exception: {e}\n{traceback.format_exc()[:500]}")
    scenarios.append(sc)
    print(sc.summary())

    # Edit follow-up against the SAME session (the iterate-on-it path).
    sc = Scenario(name="coding:edit", input_text="change it to print E2E EDITED")
    try:
        if runner is None or scenarios[0].errors:
            sc.err("skipped: create scenario failed")
        else:
            t0 = time.monotonic()
            handle = runner.send_followup(
                "Change main.py so it prints exactly E2E EDITED instead.",
                kind="adjustment",
            )
            if handle is None:
                sc.err("send_followup returned no handle")
            else:
                result = handle.wait(timeout=320.0)
                sc.record("edit_task", time.monotonic() - t0)
                sc.out("success", result.success)
                if not result.success:
                    sc.err(f"edit task failed: {result.summary[:200]}")
                else:
                    run = run_program(
                        project, sandbox_root=sandbox_root,
                        project_name=project.name, timeout_s=60.0,
                        user_text="run the program",
                    )
                    sc.out("run_stdout", run.stdout[:80])
                    if not run.ok:
                        sc.err(f"sandbox run failed: {run.error or run.stderr[:200]}")
                    elif "E2E EDITED" not in run.stdout:
                        sc.err(
                            f"edit did not land; output: {run.stdout[:120]!r}"
                        )
    except Exception as e:
        sc.err(f"exception: {e}\n{traceback.format_exc()[:500]}")
    scenarios.append(sc)
    print(sc.summary())

    # Best-effort cleanup of the e2e project dir (keep the sandbox tidy).
    try:
        shutil.rmtree(project, ignore_errors=True)
    except Exception:
        pass
    return scenarios


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------


def write_report(path: Path) -> None:
    """Dump the recorded scenarios as JSON for postmortem."""
    payload: Dict[str, Any] = {}
    for phase, scenarios in _REPORT.items():
        payload[phase] = [
            {
                "name": s.name,
                "input": s.input_text,
                "expected": s.expected,
                "timings_ms": {k: round(v * 1000, 1) for k, v in s.timings.items()},
                "outputs": {
                    k: (
                        v if not isinstance(v, np.ndarray)
                        else f"<ndarray shape={v.shape}>"
                    )
                    for k, v in s.outputs.items()
                },
                "errors": s.errors,
            }
            for s in scenarios
        ]
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    total = sum(len(v) for v in _REPORT.values())
    failed = sum(1 for v in _REPORT.values() for s in v if s.errors)
    print(f"\n=== Wrote report to {path} ({total} scenarios, {failed} with errors) ===")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["1", "2", "3", "4", "5", "6", "7", "8", "9",
                             "10", "11", "all"])
    ap.add_argument(
        "--report",
        default="logs/autonomous_e2e_report.json",
        help="Where to write the JSON report.",
    )
    args = ap.parse_args()

    phase_map = {
        "1": ("stt", phase_stt),
        "2": ("llm", phase_llm),
        "3": ("tts", phase_tts),
        "4": ("web_search", phase_web_search),
        "5": ("memory", phase_memory),
        "6": ("routing", phase_routing),
        "7": ("gate", phase_gate),
        "8": ("commands", phase_commands),
        "9": ("short_circuits", phase_short_circuits),
        "10": ("full_loop", phase_full_loop),
        "11": ("coding", phase_coding),
    }
    to_run = list(phase_map.keys()) if args.phase == "all" else [args.phase]

    for p in to_run:
        name, fn = phase_map[p]
        try:
            _save_phase(name, fn())
        except Exception as e:
            print(f"\n!!! Phase {p} ({name}) crashed: {e}")
            traceback.print_exc()
            _save_phase(name, [])
        finally:
            # Free the phase's GPU memory so the next phase loads cleanly.
            _free_gpu()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report_path)

    failures = sum(
        1 for scenarios in _REPORT.values() for s in scenarios if s.errors
    )
    print(f"\n=== {failures} scenario failures across all phases ===")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
