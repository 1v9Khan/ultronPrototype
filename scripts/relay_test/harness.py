r"""Full-pipeline relay test harness for the Valorant teammate-relay feature.

Staged so the cheap, deterministic checks run without loading any model and
the heavy full-pipeline checks load the real voice stack:

  --stage matcher   match_relay_command on every corpus phrase; score
                    expect_match / addressee / flags. NO models. Fast.
  --stage rephrase  + build_relay_line through the REAL LLM; check the line
                    is non-empty, preserves numbers/agent/callout tokens,
                    sane length, no control tokens / stage directions.
  --stage audio     + Kokoro-synthesize each line and analyze_clip for blips/
                    bursts/dropouts (the production output watcher's detector).
  --stage asr       + transcribe Kenning's OUTPUT audio (Moonshine) and verify
                    the intended words are intelligible -- flags audio whose
                    reconstruction drops content words (a non-word-burst proxy
                    that does NOT trip on the reverb/voice-filter character).
  --stage full      + also synthesize the INPUT command and run it back through
                    STT first (exercises the spoken->STT->relay path end to end).

Results are written to logs/relay_test/<stage>_<run>.jsonl and a summary is
printed. Run from C:\STC\ultronPrototype with the runtime venv.

    .venv\\Scripts\\python.exe scripts\\relay_test\\harness.py --stage matcher
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "relay_test"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # top-level `config` package lives at repo root

from corpus import Case, build_corpus  # noqa: E402


# --- scoring helpers --------------------------------------------------------

_NUM_RE = re.compile(r"\b(?:one|two|three|four|five|\d+)\b", re.I)
_CONTROL_RE = re.compile(r"/\s*no_?think\b|<\|?[a-z_]+\|?>|<think>", re.I)
_STAGE_DIR_RE = re.compile(r"\*[^*]+\*")  # *repositions window*


def score_matcher(case: Case, cmd) -> list[str]:
    """Return a list of failure strings (empty = pass)."""
    fails = []
    matched = cmd is not None
    if matched != case.expect_match:
        fails.append(f"expect_match={case.expect_match} got={matched}")
        return fails  # nothing else meaningful to check
    if not matched:
        return fails
    # addressee
    if case.addressee != "team":
        got = getattr(cmd, "addressee", "team")
        # The matcher canonicalizes STT homophones to the real agent display
        # name (cipher->Cypher, mix->Miks, kill joy->Killjoy). Canonicalize
        # the EXPECTED addressee the same way before comparing.
        from kenning.audio.relay_speech import _NAME_CANON
        key = " ".join(case.addressee.split()).lower()
        want = _NAME_CANON.get(key, case.addressee).lower().replace(" ", "")
        gotn = str(got).lower().replace(" ", "")
        if want not in gotn and gotn not in want:
            fails.append(f"addressee want~{case.addressee!r} got={got!r}")
    # flags
    for flag in case.flags:
        if not getattr(cmd, flag, False):
            fails.append(f"flag {flag} not set")
    return fails


def score_rephrase(case: Case, line: str) -> list[str]:
    fails = []
    if not line or not line.strip():
        fails.append("empty line")
        return fails
    if _CONTROL_RE.search(line):
        fails.append(f"control token leaked: {line!r}")
    if _STAGE_DIR_RE.search(line):
        fails.append(f"stage direction leaked: {line!r}")
    if line.count('"') > 0:
        fails.append("contains quotation marks")
    if len(line) > 300:
        fails.append(f"too long ({len(line)} chars)")
    # numbers in the source phrase must survive into the line
    src_nums = set(m.group(0).lower() for m in _NUM_RE.finditer(case.text))
    # normalize digit/word equivalence loosely
    if src_nums and case.category in {"location", "ult", "team_status"}:
        line_nums = set(m.group(0).lower() for m in _NUM_RE.finditer(line))
        # allow word<->digit swap (e.g. "one" vs "1")
        if not (src_nums & line_nums) and not _word_digit_overlap(src_nums, line_nums):
            fails.append(f"number dropped: src={src_nums} line={line_nums}")
    return fails


_W2D = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}


def _word_digit_overlap(a: set, b: set) -> bool:
    na = {_W2D.get(x, x) for x in a}
    nb = {_W2D.get(x, x) for x in b}
    return bool(na & nb)


def score_audio(report) -> list[str]:
    """Failures from the production blip/burst/dropout detector. The detector
    is already calibrated to ignore the reverb/voice-filter character (only
    real onset/tail pops, isolated bursts, hard cuts, clicks, clipping, dc)."""
    fails = []
    for f in getattr(report, "findings", ()) or ():
        fails.append(f"{f.kind}@{f.position_ms:.0f}ms ({f.detail})")
    return fails


# --- content-word ASR reconstruction ---------------------------------------

_STOP = {
    "a", "an", "the", "to", "of", "and", "or", "is", "are", "am", "i", "you",
    "we", "they", "he", "she", "it", "my", "our", "your", "their", "for",
    "on", "in", "at", "so", "that", "this", "be", "do", "got", "get", "up",
    "out", "off", "with", "me", "us", "them", "him", "her", "go", "going",
}


def content_words(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in toks if t not in _STOP and len(t) > 1}


def score_asr(intended_line: str, heard: str) -> list[str]:
    """Verify the OUTPUT audio reconstructs to actual WORDS (not blips/noise),
    referenced against what Kenning was supposed to say.

    Purpose is to catch NON-WORD audio (silence, bursts, garble) -- NOT to
    grade TTS+ASR fidelity. Valorant callouts are 1-3 words and Kenning's
    voice carries a heavy reverb/filter character, so ASR routinely mis-hears
    a clean short clip ('Smoke A' -> 'Smoky'); that is still intelligible
    SPEECH and must NOT be flagged. So:

      * if ASR returns intelligible speech (any alphabetic word), the clip is
        word-audio -> pass (signal-level bursts are caught by analyze_clip).
      * flag only when ASR returns NOTHING word-like for a line that clearly
        had words -> the audio produced no recoverable speech.
      * for LONGER lines (>=5 content words, reliable for ASR) also flag a
        GROSS reconstruction miss (<35% of content words survive), which
        would indicate large dropped/garbled spans.
    """
    want = content_words(intended_line)
    if not want:
        return []
    heard_words = re.findall(r"[a-z']+", heard.lower())
    if not heard_words:
        # No word-like output at all for a line that had words: the audio is
        # silent or non-speech noise.
        return [f"no intelligible speech reconstructed (heard={heard!r})"]
    if len(want) < 5:
        return []  # short callout + speech present -> rely on analyze_clip
    got = content_words(heard)
    norm_got = {_W2D.get(w, w) for w in got}
    missing = {w for w in want if w not in got and _W2D.get(w, w) not in norm_got}
    coverage = 1.0 - (len(missing) / len(want))
    if coverage < 0.35:
        return [f"gross reconstruction miss: coverage {coverage:.0%} "
                f"missing={sorted(missing)} heard={heard!r}"]
    return []


# --- stages -----------------------------------------------------------------

def run(stage: str, limit: int | None, run_tag: str,
        categories: set | None = None) -> int:
    from corpus_packs import build_corpus_10k
    import os as _os
    _seed = int(_os.environ.get("RELAY_CORPUS_SEED", "0") or "0")
    cases = build_corpus_10k(_seed)   # expanded ~10k corpus (packs + compounds)
    if categories:
        cases = [c for c in cases if c.category in categories]
    # Deterministic shuffle so identical templates (29 "calm down" lines, the
    # location grid) aren't fired back-to-back -- that clusters the recent-line
    # ring and provokes the LLM to copy the previous output, which never
    # happens in real spread-out play. Seeded for reproducible runs.
    import random as _random
    _random.Random(7).shuffle(cases)
    if limit:
        cases = cases[:limit]
    out_dir = ROOT / "logs" / "relay_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stage}_{run_tag}.jsonl"

    need_llm = stage in {"rephrase", "audio", "asr", "full"}
    need_tts = stage in {"audio", "asr", "full"}
    need_stt = stage in {"asr", "full"}

    from kenning.audio.relay_speech import match_relay_command, build_relay_line
    # 2026-06-15: normalize FIRST, mirroring the live orchestrator
    # (normalize_command -> match_relay_command), so the harness exercises the
    # REAL routing the user gets (verbatim/greeting/possessive/disfluency/
    # reported-question/STT-correction) instead of the bare matcher on raw text.
    from kenning.audio.command_normalizer import normalize_command

    llm = tts = stt = None
    if need_llm:
        llm = _load_llm()
    if need_tts:
        from kenning.tts.kokoro_engine import KokoroSpeech
        from kenning.config import get_config
        # Build the engine the SAME way production does (make_tts_engine) so the
        # audio-blip metric reflects what teammates actually hear -- crucially the
        # max_pause_cap_ms that holds dramatic pauses under the dead-air flag.
        # A bare KokoroSpeech(voice="kenning") leaves the cap OFF and manufactures
        # phantom internal-dropout blips on long verbose lines that never occur in
        # the real relay.
        _kc = get_config().tts.kokoro
        tts = KokoroSpeech(
            voice="kenning",
            apply_runtime_filter=_kc.apply_runtime_filter,
            filter_preset=_kc.filter_preset,
            apply_spectral_smooth=_kc.apply_spectral_smooth,
            spectral_smooth_window=_kc.spectral_smooth_window,
            apply_trim_fade=_kc.apply_trim_fade,
            trim_fade_threshold_db=_kc.trim_fade_threshold_db,
            f0_contour_factor=_kc.f0_contour_factor,
            f0_shift_semitones=_kc.f0_shift_semitones,
            f0_max_excursion=_kc.f0_max_excursion,
            f0_energy_factor=_kc.f0_energy_factor,
            dur_final_factor=_kc.dur_final_factor,
            dur_internal_factor=_kc.dur_internal_factor,
            dur_stress_factor=_kc.dur_stress_factor,
            max_pause_cap_ms=_kc.max_pause_cap_ms,
        )
        tts.warmup()
    if need_stt:
        from kenning.transcription.moonshine_engine import MoonshineEngine
        stt = MoonshineEngine()

    from kenning.audio.output_quality import analyze_clip

    n = 0
    n_fail = 0
    cat_fail: dict[str, int] = {}
    recent: list[str] = []
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            rec = {"text": case.text, "category": case.category,
                   "expect_match": case.expect_match, "fails": []}

            # optional spoken->STT input round trip
            heard_in = case.text
            if stage == "full" and tts is not None and stt is not None:
                heard_in = _spoken_then_stt(tts, stt, case.text) or case.text
                rec["stt_in"] = heard_in

            cmd = match_relay_command(normalize_command(heard_in))
            rec["matched"] = cmd is not None
            rec["fails"] += [f"matcher: {x}" for x in score_matcher(case, cmd)]

            if cmd is not None and need_llm:
                try:
                    # Mirror the orchestrator: roast / fun-fact relays speak
                    # VERBATIM from the curated pools (the orchestrator
                    # intercepts these BEFORE build_relay_line), so reproduce
                    # that here for test fidelity instead of LLM-composing them.
                    if getattr(cmd, "roast", False):
                        from kenning.audio.relay_speech import (
                            load_roast_lines, pick_line,
                        )
                        line = pick_line(load_roast_lines("data/relay_roasts.txt"),
                                         recent_lines=recent[-6:])
                    elif getattr(cmd, "fun_fact", False):
                        from kenning.audio.relay_speech import (
                            load_fun_facts, pick_line,
                        )
                        line = pick_line(load_fun_facts("data/relay_fun_facts.txt"),
                                         recent_lines=recent[-6:])
                    else:
                        line = build_relay_line(cmd, llm=llm, rephrase=True,
                                                recent_lines=recent[-6:])
                except Exception as e:                               # noqa: BLE001
                    line = ""
                    rec["fails"].append(f"rephrase-exc: {e}")
                rec["line"] = line
                rec["fails"] += [f"rephrase: {x}" for x in score_rephrase(case, line)]
                if line:
                    recent.append(line)

                if need_tts and line:
                    pcm, sr = tts._synthesize(line)
                    rep = analyze_clip(pcm, sr, label=line[:60])
                    rec["fails"] += [f"audio: {x}" for x in score_audio(rep)]
                    rec["dur_s"] = round(getattr(rep, "duration_s", 0.0), 2)
                    if need_stt:
                        heard = _stt_pcm(stt, pcm, sr)
                        rec["heard_out"] = heard
                        rec["fails"] += [f"asr: {x}" for x in score_asr(line, heard)]

            if rec["fails"]:
                n_fail += 1
                cat_fail[case.category] = cat_fail.get(case.category, 0) + 1
            n += 1
            fh.write(json.dumps(rec) + "\n")
            if n % 50 == 0:
                print(f"  ... {n}/{len(cases)}  fails={n_fail}", flush=True)

    dt = time.time() - t0
    print(f"\n[{stage}] {n} cases, {n_fail} with failures "
          f"({(n - n_fail) / n:.1%} clean) in {dt:.0f}s -> {out_path}")
    if cat_fail:
        print("failures by category:")
        for cat, c in sorted(cat_fail.items(), key=lambda kv: -kv[1]):
            print(f"  {c:>4}  {cat}")
    return 0 if n_fail == 0 else 1


# --- model helpers ----------------------------------------------------------

GAMING_PRESET = "llama-3.2-3b-abliterated"


def _load_llm(preset: str = GAMING_PRESET):
    """Construct the LLM engine the relay ACTUALLY runs under.

    The relay only fires in gaming mode, which hot-swaps the LLM to the
    gaming preset (``llama-3.2-3b-abliterated`` -- a 3B *abliterated* model
    with its safety refusals removed). Testing on the default qwen3.5-4b was
    wrong: that model refuses the Ultron persona / Marvel content, the 3B
    one does not. So load the gaming preset here.

    Uses an ISOLATED Qdrant path (temp dir) so the harness can NEVER contend
    with a live Kenning instance's data/qdrant lock.
    """
    import atexit
    import os
    import pathlib
    import shutil
    import tempfile

    from kenning.llm.inference import LLMEngine
    from kenning.memory.embedder import HybridEmbedder
    from kenning.memory.qdrant_store import ConversationMemory

    # PID-unique temp path: never touches the production data/qdrant AND never
    # collides with a second concurrent harness/probe run.
    qpath = (pathlib.Path(tempfile.gettempdir())
             / f"kenning_relay_test_qdrant_{os.getpid()}")
    # TESTING MODE: gate RAG / web-search OFF (gaming parity) WITHOUT the gaming
    # device swaps, so the relay rephrase runs the same context-free path it does
    # in-game but we may keep the LLM on GPU for fast iteration.
    try:
        from kenning.safety.testing_mode import set_testing_mode_active
        set_testing_mode_active(True)
    except Exception:                                                # noqa: BLE001
        pass

    embedder = HybridEmbedder()
    memory = ConversationMemory(embedder=embedder, path=qpath)

    # Release the lock + delete the isolated storage on process exit so back-
    # to-back harness runs don't strand temp qdrant dirs (the original lingering
    # 5 GB test holder that started this whole guardrail effort).
    def _cleanup_relay_qdrant():
        try:
            if hasattr(memory, "close"):
                memory.close()
        except Exception:                                            # noqa: BLE001
            pass
        shutil.rmtree(qpath, ignore_errors=True)

    atexit.register(_cleanup_relay_qdrant)
    eng = LLMEngine(memory=memory)
    if preset and hasattr(eng, "reload_for_preset"):
        # In-game the gaming preset is CPU-only (gpu_layers=0). For TESTING we
        # default to full GPU (-1) for speed -- the model + sampling are
        # identical, so the OUTPUT we grade is the same; gpu_layers only moves
        # where compute runs. Override via RELAY_TEST_GPU_LAYERS=0 to test CPU.
        _gl_env = os.environ.get("RELAY_TEST_GPU_LAYERS")
        gpu_layers = int(_gl_env) if _gl_env not in (None, "") else -1
        try:
            try:
                ok, msg = eng.reload_for_preset(preset, gpu_layers=gpu_layers)
            except TypeError:
                ok, msg = eng.reload_for_preset(preset)  # older signature
            print(f"[llm] preset -> {preset} (gpu_layers={gpu_layers}): "
                  f"{ok} ({msg})", flush=True)
        except Exception as e:                                       # noqa: BLE001
            print(f"[llm] preset swap failed ({e}); using default", flush=True)
    if hasattr(eng, "warmup"):
        try:
            eng.warmup()
        except Exception:                                            # noqa: BLE001
            pass
    return eng


def _spoken_then_stt(tts, stt, text: str) -> str:
    """Speak the command in a neutral test voice and run it back through STT
    to exercise the spoken->STT path. Uses a stock voice so we test STT, not
    Kenning's character."""
    try:
        from kenning.tts.kokoro_engine import KokoroSpeech
        if not hasattr(_spoken_then_stt, "_neutral"):
            nv = KokoroSpeech(voice="am_michael", apply_spectral_smooth=False)
            nv.warmup()
            _spoken_then_stt._neutral = nv
        pcm, sr = _spoken_then_stt._neutral._synthesize(text)
        return _stt_pcm(stt, pcm, sr)
    except Exception:                                                # noqa: BLE001
        return ""


def _stt_pcm(stt, pcm, sr) -> str:
    import numpy as np
    audio = pcm.astype(np.float32) / 32768.0
    if sr != 16000:
        try:
            import scipy.signal
            audio = scipy.signal.resample(
                audio, int(len(audio) * 16000 / sr)).astype(np.float32)
        except Exception:                                            # noqa: BLE001
            factor = sr / 16000.0
            idx = (np.arange(int(len(audio) / factor)) * factor).astype(np.int64)
            audio = audio[idx]
    try:
        return stt.transcribe(audio) or ""
    except Exception:                                                # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="matcher",
                    choices=["matcher", "rephrase", "audio", "asr", "full"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--category", default=None,
                    help="comma-separated categories to restrict to")
    args = ap.parse_args()
    cats = set(args.category.split(",")) if args.category else None
    return run(args.stage, args.limit, args.tag, cats)


if __name__ == "__main__":
    raise SystemExit(main())
