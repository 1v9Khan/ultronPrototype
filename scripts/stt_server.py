"""Speech-to-text sidecar -- faster-whisper in an ISOLATED process.

WHY THIS EXISTS (2026-07-26). Running CTranslate2 in-process on a NON-DEFAULT
CUDA device (``device_index=1``, the RTX 3060 Ti added that day) corrupted
memory and killed Ultron with a silent ``__fastfail`` -- exit 0xc0000409, no
traceback, no stderr, no WER dump. The flight recorder caught CTranslate2
mid-decode (``generate_with_fallback -> model.generate``) in every snapshot
taken at the moment of death, and moving Whisper back to device 0 stopped the
crashes outright -- at the cost of thrashing, because the 12B target already
owns ~8.6 GB of that card.

The sidecar resolves both. The parent masks every GPU but one via
``CUDA_VISIBLE_DEVICES``, so INSIDE this process the target card is plain
``cuda:0`` -- CTranslate2's default, heavily-travelled path, never the
device-index path that corrupts. And because the decode lives in a child,
a repeat of that corruption kills a restartable process instead of the
assistant.

ANTICHEAT POSTURE: pure compute only (load an ASR model, return text for
submitted PCM). NO input injection, NO screen/window capture, NO foreign
process memory, NO hooks, never touches the game -- the same class as the
embedder/guard sidecars. Binds 127.0.0.1 ONLY.

Protocol (JSON over HTTP; audio is base64 float32 little-endian mono @16 kHz):
  GET  /healthz                    -> {"ok":true,"model":...,"device":...,
                                       "device_index":N,"compute_type":...}
  POST /transcribe {"audio_b64":..,"language":"en"|null}
                                   -> {"text":"...","ms":12.3}

Run:  python scripts/stt_server.py [PORT]
Env:  KENNING_STT_PORT, KENNING_STT_MODEL, KENNING_STT_DEVICE,
      KENNING_STT_COMPUTE_TYPE, KENNING_STT_BEAM_SIZE,
      KENNING_STT_PARENT_PID
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

PORT = int(os.getenv("KENNING_STT_PORT", "8779"))
MODEL = os.getenv("KENNING_STT_MODEL",
                  "deepdml/faster-whisper-large-v3-turbo-ct2")
DEVICE = os.getenv("KENNING_STT_DEVICE", "cuda")
COMPUTE = os.getenv("KENNING_STT_COMPUTE_TYPE", "float16")
BEAM = int(os.getenv("KENNING_STT_BEAM_SIZE", "1"))
PARENT_PID = os.getenv("KENNING_STT_PARENT_PID", "").strip()


def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _envb(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return default if not raw else raw in ("1", "true", "yes", "on")


# Decode knobs mirrored from the parent's stt.* config so the sidecar and the
# in-process engine produce the SAME transcript for the same audio.
TEMPERATURE = _envf("KENNING_STT_TEMPERATURE", 0.0)
CONDITION_ON_PREV = _envb("KENNING_STT_CONDITION_ON_PREVIOUS_TEXT", False)
VAD_FILTER = _envb("KENNING_STT_VAD_FILTER", False)
DOMAIN_BIAS = _envb("KENNING_STT_DOMAIN_BIAS", True)
USER_INITIAL_PROMPT = os.getenv("KENNING_STT_INITIAL_PROMPT", "")

# DECODE-QUALITY PARITY (2026-07-26). The domain vocabulary + the whole-
# transcript non-speech blocklist are shared with the in-process engine via
# kenning.transcription._whisper_text (stdlib-only, so importing it cannot die
# on a config error). Without them the sidecar would silently lose agent-name
# biasing ("Sova" -> "Silva") and re-admit "Thank you."-type phantom callouts.
# Loaded BY FILE PATH, not as ``kenning.transcription._whisper_text``: a normal
# import would execute the package __init__ chain (config.settings, resilience,
# logging) and the whole point of the sidecar is that a config problem cannot
# take STT down with it. The module is a stdlib-only leaf, so this is safe.
# Fail LOUD rather than degrade quietly.
try:
    import importlib.util as _ilu

    _wt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "kenning", "transcription", "_whisper_text.py")
    _spec = _ilu.spec_from_file_location("_stt_whisper_text", _wt_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load {_wt_path}")
    _wt = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_wt)
    build_initial_prompt = _wt.build_initial_prompt
    is_whisper_hallucination = _wt.is_whisper_hallucination
    _TEXT_RULES = True
except Exception as _tr_e:                                   # noqa: BLE001
    print(f"[stt-sidecar] WARNING: decode-quality helpers unavailable "
          f"({_tr_e}) -- domain biasing and the non-speech blocklist are OFF; "
          f"expect agent-name errors and phantom 'Thank you' callouts",
          flush=True)
    _TEXT_RULES = False

    def build_initial_prompt(user: str = "") -> str:         # noqa: D103
        return (user or "").strip()

    def is_whisper_hallucination(text: str) -> bool:         # noqa: D103
        return False

_model = None
_model_lock = threading.Lock()
# CTranslate2 is NOT safe for concurrent decodes on one model. The parent can
# have a speculative and a foreground transcribe in flight at once, so
# serialise here too -- the sidecar is the last line of defence.
_decode_lock = threading.Lock()


def _log(msg: str) -> None:
    print(f"[stt-sidecar] {msg}", flush=True)


def _load():
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        from faster_whisper import WhisperModel
        t0 = time.monotonic()
        # device_index is deliberately NOT passed: the parent masks GPUs with
        # CUDA_VISIBLE_DEVICES so the intended card is simply cuda:0 here.
        _model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
        _log(f"model ready in {time.monotonic() - t0:.1f}s "
             f"({MODEL}, {DEVICE}, {COMPUTE}, "
             f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', 'all')})")
        return _model


def _transcribe(audio: np.ndarray, language) -> str:
    model = _load()
    kw = {
        "beam_size": BEAM,
        "language": language,
        "temperature": TEMPERATURE,
        "condition_on_previous_text": CONDITION_ON_PREV,
        "vad_filter": VAD_FILTER,
    }
    if DOMAIN_BIAS:
        kw["initial_prompt"] = build_initial_prompt(USER_INITIAL_PROMPT)
    with _decode_lock:
        segments, _info = model.transcribe(audio, **kw)
        kept = []
        for seg in segments:            # decode happens during iteration
            if getattr(seg, "no_speech_prob", 0.0) > 0.85:
                continue
            piece = (seg.text or "").strip()
            if piece:
                kept.append(piece)
    text = " ".join(kept).strip()
    # Final guard: a whole-transcript stock phrase is a hallucination, not a
    # command. Same rule the in-process engine applies.
    if text and is_whisper_hallucination(text):
        _log(f"dropped non-speech hallucination {text!r}")
        return ""
    return text


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):      # noqa: A003 - silence per-request spam
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                        # noqa: N802
        if self.path.split("?")[0] != "/healthz":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {
            "ok": _model is not None, "model": MODEL, "device": DEVICE,
            "compute_type": COMPUTE,
            "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "all"),
        })

    def do_POST(self):                       # noqa: N802
        if self.path.split("?")[0] != "/transcribe":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            raw = base64.b64decode(req.get("audio_b64", ""))
            audio = np.frombuffer(raw, dtype=np.float32)
            language = req.get("language", "en")
            t0 = time.monotonic()
            text = _transcribe(audio, language)
            self._send(200, {"text": text,
                             "ms": round((time.monotonic() - t0) * 1000, 1)})
        except Exception as e:               # noqa: BLE001 - never kill the server
            _log(f"transcribe failed: {e!r}")
            self._send(500, {"error": repr(e)})


def _parent_watchdog() -> None:
    """Self-terminate if the parent dies -- no runaway orphan holding VRAM."""
    if not PARENT_PID:
        return
    try:
        pid = int(PARENT_PID)
    except ValueError:
        return
    import ctypes
    k32 = ctypes.windll.kernel32 if sys.platform == "win32" else None
    while True:
        time.sleep(3.0)
        alive = True
        try:
            if k32 is not None:
                h = k32.OpenProcess(0x1000, False, pid)
                if not h:
                    alive = False
                else:
                    k32.CloseHandle(h)
            else:
                os.kill(pid, 0)
        except Exception:                    # noqa: BLE001
            alive = False
        if not alive:
            _log(f"parent pid {pid} gone -> self-terminating")
            os._exit(0)


def main() -> int:
    global PORT
    if len(sys.argv) > 1:
        PORT = int(sys.argv[1])
    if PARENT_PID:
        threading.Thread(target=_parent_watchdog, daemon=True,
                         name="stt-parent-watchdog").start()
    try:
        _load()                              # fail fast, before serving
    except Exception as e:                   # noqa: BLE001
        _log(f"model load FAILED: {e!r}")
        return 2
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    _log(f"serving on http://127.0.0.1:{PORT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
