"""Live audio-injection corpus runner.

Boots the FULL Ultron program in-process, swaps the microphone for an
InjectableCapture, and feeds each composite command WAV so it traverses the REAL
pipeline -- wake word -> pre-roll -> audio-domain wake-drop -> whisper STT ->
norm1/norm2 -> semantic routing -> tail selection -> real 3B (for LLM turns) ->
Kokoro TTS. Per command it captures the full per-stage trace, saves the spoken
response audio, and re-transcribes that response with whisper to verify it is
understandable speech. Everything is written to a session-stamped JSONL log.

Nothing in runtime src/ is modified: the injection is a drop-in swap of
orchestrator.audio, and the response capture is a class-level synth hook.

Usage:
  python scripts/relay_test/audio_corpus/run_corpus.py [--limit N] [--gpu] [--manifest path]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf            # imported BEFORE the firewall (Orchestrator())
from pydub import AudioSegment
from scipy.signal import resample_poly

# Pre-warm the heavy openwakeword -> sklearn -> pandas -> pyarrow import chain
# WHILE the repo root is NOT yet on sys.path. Otherwise pyarrow's import-cache
# _fill_cache scans the large repo root and hits a transactional-NTFS glitch
# (WinError 6714). Warming it here means the wake-model load inside Orchestrator()
# finds it already cached and never re-scans the repo root.
try:
    import openwakeword  # noqa: F401
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
# small dirs first (kenning, inject); repo root LAST so site-packages imports
# resolve before the big repo-root scan (only `config` needs the repo root).
for p in (str(_HERE), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

os.environ.setdefault("KENNING_ALLOW_MULTIPLE_INSTANCES", "1")
KOKORO_SR = 24000


def _to_16k_f32(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)
    if sr != 16000:
        from math import gcd
        g = gcd(16000, sr)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(_HERE / "out" / "manifest.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", action="store_true", help="move the 3B to GPU for speed")
    ap.add_argument("--turn-timeout", type=float, default=90.0)
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.limit:
        man = man[: args.limit]

    session = time.strftime("%Y%m%d_%H%M%S")
    outdir = _HERE / f"session_{session}"
    resp_dir = outdir / "responses"
    resp_dir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / f"corpus_{session}.log.jsonl"
    print(f"[corpus] session {session} -> {log_path}")

    # --- response-audio capture: class-level Kokoro synth hook --------------
    from kenning.tts.kokoro_engine import KokoroSpeech
    _orig_synth = KokoroSpeech._synthesize
    cap_state = {"on": False, "buf": [], "sr": None, "last": 0.0}

    def _hooked(self, text):
        clip = _orig_synth(self, text)
        try:
            pcm, sr = clip
            if cap_state["on"] and pcm is not None and len(pcm):
                cap_state["buf"].append(np.asarray(pcm).reshape(-1).copy())
                cap_state["sr"] = sr
                cap_state["last"] = time.monotonic()
        except Exception:
            pass
        return clip
    KokoroSpeech._synthesize = _hooked

    # --- boot the full orchestrator + swap the mic --------------------------
    from kenning.pipeline import Orchestrator
    from inject import InjectableCapture
    from kenning.safety.testing_mode import set_testing_mode_active

    print("[corpus] building Orchestrator (full boot)...")
    orch = Orchestrator()
    inj = InjectableCapture(realtime=True)
    orch.audio = inj
    set_testing_mode_active(True)
    if args.gpu:
        try:
            ok, msg = orch.llm.reload_for_device("gpu")
            print(f"[corpus] 3B->gpu: {ok} ({msg})")
        except Exception as e:
            print(f"[corpus] gpu move skipped: {e}")

    # deterministic stage re-derivation (same functions the live turn ran)
    from kenning.audio._stt_correct import correct_callout_stt
    from kenning.audio.command_normalizer import normalize_command
    from kenning.audio import relay_speech as RS

    usage_trace = _ROOT / "logs" / "usage_trace.jsonl"

    def _trace_count() -> int:
        try:
            return sum(1 for _ in usage_trace.open(encoding="utf-8"))
        except OSError:
            return 0

    def _last_trace_row():
        try:
            lines = usage_trace.read_text(encoding="utf-8").splitlines()
            return json.loads(lines[-1]) if lines else None
        except Exception:
            return None

    # --- start the live run loop --------------------------------------------
    t = threading.Thread(target=orch.run, daemon=True)
    t.start()
    # wait for boot to reach the wait-for-wake loop
    print("[corpus] waiting for boot to settle...")
    boot_deadline = time.monotonic() + 180
    klog = _ROOT / "logs" / "kenning.log"
    while time.monotonic() < boot_deadline:
        time.sleep(1.0)
        try:
            tailtxt = klog.read_text(encoding="utf-8", errors="ignore")[-4000:]
            if "waiting_for_wake_word" in tailtxt or "loop:iteration_start" in tailtxt:
                break
        except OSError:
            pass
    time.sleep(3.0)
    print("[corpus] boot ready; driving commands")

    results = []
    for m in man:
        cmd = m["command"]
        cap_state["buf"].clear(); cap_state["sr"] = None; cap_state["last"] = 0.0
        base_n = _trace_count()
        cap_state["on"] = True
        pcm, sr = sf.read(m["wav"], dtype="float32")
        t0 = time.monotonic()
        inj.feed_pcm(pcm)

        # wait for the turn: a new usage_trace row AND the response synth quiescent
        deadline = time.monotonic() + args.turn_timeout
        got_row = False
        while time.monotonic() < deadline:
            time.sleep(0.25)
            if _trace_count() > base_n:
                got_row = True
            quiescent = cap_state["last"] and (time.monotonic() - cap_state["last"] > 1.8)
            if got_row and quiescent:
                break
            if quiescent and inj.pending() == 0 and (time.monotonic() - cap_state["last"] > 3.0):
                break
        cap_state["on"] = False
        elapsed = round(time.monotonic() - t0, 2)

        row = _last_trace_row() if got_row else None
        raw = (row or {}).get("raw", "")

        # save + re-transcribe the response audio
        resp_txt, resp_wav = "", ""
        if cap_state["buf"]:
            resp = np.concatenate(cap_state["buf"])
            resp16 = _to_16k_f32(resp, cap_state["sr"] or KOKORO_SR)
            resp_wav = str(resp_dir / f"{m['slug']}__response.wav")
            sf.write(resp_wav, resp16, 16000, subtype="PCM_16")
            i16 = np.clip(resp16 * 32767, -32768, 32767).astype(np.int16)
            AudioSegment(i16.tobytes(), frame_rate=16000, sample_width=2, channels=1).export(
                str(resp_dir / f"{m['slug']}__response.mp3"), format="mp3", bitrate="96k")
            try:
                resp_txt = orch.stt.transcribe(resp16)
            except Exception as e:
                resp_txt = f"<RETRANSCRIBE_ERR {e}>"

        # re-derive the deterministic stages from the LIVE transcript
        stt1 = norm2 = match = route = snap = tail = None
        try:
            if raw:
                stt1 = correct_callout_stt(raw)
                norm2 = normalize_command(raw)
                cmd_obj = RS.match_relay_command(norm2)
                if cmd_obj is not None:
                    match = {"addressee": getattr(cmd_obj, "addressee", None),
                             "payload": getattr(cmd_obj, "payload", None),
                             "compose": getattr(cmd_obj, "compose", None),
                             "directive": getattr(cmd_obj, "directive", None),
                             "context": getattr(cmd_obj, "context", None)}
        except Exception as e:
            route = f"<DERIVE_ERR {e}>"

        rec = {
            "i": m["i"], "command": cmd, "expected_body": m.get("body"),
            "wav": m["wav"], "gap_s": m.get("gap_s"),
            "transcription": raw,
            "norm1_stt_correct": stt1,
            "norm2_normalized": (row or {}).get("normalized") or norm2,
            "route": (row or {}).get("route"), "reason": (row or {}).get("reason"),
            "subtype": (row or {}).get("subtype"),
            "payload": (row or {}).get("payload"), "addressee": (row or {}).get("addressee"),
            "directive": (row or {}).get("directive"), "channel": (row or {}).get("channel"),
            "match_rederived": match,
            "final_spoken": (row or {}).get("final"),
            "response_audio": resp_wav,
            "response_retranscribed": resp_txt,
            "turn_seconds": elapsed, "got_trace_row": got_row,
        }
        results.append(rec)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tag = "OK " if got_row else "??"
        print(f"  [{tag}] #{m['i']} {elapsed}s | T={raw!r} | route={rec['route']} | "
              f"final={str(rec['final_spoken'])[:60]!r}")

    print(f"[corpus] done: {len(results)} commands -> {log_path}")
    try:
        orch.shutdown()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
