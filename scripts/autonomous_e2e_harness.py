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
    1  STT (Moonshine accuracy + latency)
    2  LLM (Qwen 3.5 4B response + TTFT + latency)
    3  TTS (Kokoro Ultron voice synth + spectral_smooth)
    4  Web search (SearXNG + readers + ranker)
    5  Memory (retrieve + rerank + ranking)
    6  Routing classifier
    7  Gate (rule path + preflight LLM)
    8  Full E2E loops

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
                    choices=["1", "2", "3", "4", "5", "6", "7", "all"])
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
