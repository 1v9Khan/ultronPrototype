"""Guard-model sidecar — the non-abliterated security brain (L3 input + L5 exchange).

Loopback HTTP service (mirrors scripts/embedder_server.py: 127.0.0.1 only,
parent-death deadman, fail-quiet) that loads a small NON-abliterated guard model
(default Llama-Guard-3-1B; ShieldGemma-2B / Granite-Guardian-3B selectable) as a
GGUF via llama-cpp-python and classifies text safe/unsafe. The voice/relay process
NEVER imports this — it talks to it via the thin urllib client in
``kenning.twitch.guard.GuardModelClient``.

ANTICHEAT: this script + llama_cpp live ONLY in the sidecar's ``.venv-twitch``.
If llama_cpp or the model file is unavailable, /healthz reports ready=false and
/classify returns 503 -> the client raises GuardUnavailable -> chat-reply mode
stays OFF (fail-CLOSED on the feature). It NEVER fails open.

Run:  python scripts/twitch_guard_sidecar.py [PORT]
Env:  KENNING_TWITCH_GUARD_PORT (default 8774), KENNING_TWITCH_GUARD_MODEL (GGUF path),
      KENNING_TWITCH_GUARD_FAMILY (llama-guard|shieldgemma|granite|generic),
      KENNING_TWITCH_GUARD_NCTX (default 2048), KENNING_TWITCH_PARENT_PID.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Make the worktree importable so we can reuse the prompt format/parse helpers.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from kenning.twitch.guard import (  # noqa: E402
    CANARY_SAFE,
    CANARY_UNSAFE,
    build_guard_messages,
    format_llama_guard_prompt,
    parse_guard_output,
)
import atexit  # noqa: E402

from kenning.subprocess import sidecar_lock  # noqa: E402
from kenning.subprocess.sidecar_server import SingletonThreadingHTTPServer  # noqa: E402

ROLE = "twitch_guard"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("KENNING_TWITCH_GUARD_PORT", "8774"))
MODEL = os.environ.get("KENNING_TWITCH_GUARD_MODEL", "")
FAMILY = os.environ.get("KENNING_TWITCH_GUARD_FAMILY", "llama-guard")
NCTX = int(os.environ.get("KENNING_TWITCH_GUARD_NCTX", "2048"))

_llm = None
_load_error = ""


def _load() -> None:
    """Lazy-load the GGUF guard model. Never raises: a failure leaves _llm=None
    and _load_error set so /healthz reports not-ready (fail-CLOSED on the feature)."""
    global _llm, _load_error
    if _llm is not None:
        return
    if not MODEL or not Path(MODEL).exists():
        _load_error = f"guard model not found: {MODEL!r} (set KENNING_TWITCH_GUARD_MODEL)"
        return
    try:
        from llama_cpp import Llama  # only present in .venv-twitch
        _llm = Llama(model_path=MODEL, n_ctx=NCTX, n_gpu_layers=-1, verbose=False)
        print(f"[guard] loaded {MODEL} family={FAMILY} n_ctx={NCTX} port={PORT}", flush=True)
    except Exception as e:  # noqa: BLE001
        _load_error = f"llama_cpp load failed: {e}"
        print(f"[guard] WARN {_load_error}", flush=True)


def _classify(text: str, exchange: str) -> dict:
    if _llm is None:
        # fail-CLOSED: no model -> caller must treat as unavailable (503 below).
        raise RuntimeError("guard model not loaded")
    fam = FAMILY.lower()
    if "llama-guard" in fam or "llamaguard" in fam:
        # Llama Guard: manual prompt + raw completion (its chat template rejects the
        # system/user layout). Verified live: slur->'unsafe\nS10', benign->'safe'.
        out = _llm.create_completion(
            format_llama_guard_prompt(text, exchange),
            max_tokens=24, temperature=0.0, stop=["<|eot_id|>"],
        )
        raw = out["choices"][0]["text"]
    else:
        messages = build_guard_messages(FAMILY, text, exchange)
        out = _llm.create_chat_completion(messages=messages, max_tokens=64, temperature=0.0)
        raw = out["choices"][0]["message"]["content"]
    res = parse_guard_output(FAMILY, raw)
    return {"unsafe": res.unsafe, "category": res.category, "score": res.score}


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send(200, {"ready": _llm is not None, "model": MODEL,
                             "family": FAMILY, "error": _load_error})
        elif self.path == "/canary":
            try:
                u = _classify(CANARY_UNSAFE, "")
                s = _classify(CANARY_SAFE, "")
                ok = bool(u["unsafe"]) and not bool(s["unsafe"])
                self._send(200, {"ok": ok, "unsafe_probe": u, "safe_probe": s})
            except Exception as e:  # noqa: BLE001
                self._send(503, {"ok": False, "error": str(e)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/classify":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": f"bad request: {e}"})
            return
        try:
            self._send(200, _classify(str(payload.get("text", "")), str(payload.get("exchange", ""))))
        except Exception as e:  # noqa: BLE001 — fail-CLOSED: 503 so the client raises GuardUnavailable
            self._send(503, {"error": f"guard unavailable: {e}"})

    def log_message(self, *args) -> None:  # noqa: ARG002
        return


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, int(pid))
            if not h:
                return False
            code = wintypes.DWORD()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code))
            k.CloseHandle(h)
            return (not ok) or code.value == 259
        except Exception:  # noqa: BLE001
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:  # noqa: BLE001
        return True


def _parent_watchdog() -> None:
    try:
        ppid = int(os.environ.get("KENNING_TWITCH_PARENT_PID", "0") or "0")
    except Exception:  # noqa: BLE001
        ppid = 0
    if ppid <= 0:
        ppid = os.getppid()
    if ppid <= 0:
        return
    while True:
        time.sleep(3.0)
        if not _pid_alive(ppid):
            sys.stderr.write(f"[guard] parent {ppid} gone -> self-terminating\n")
            sys.stderr.flush()
            os._exit(0)


def main() -> None:
    _load()  # best-effort; not-ready is reported via /healthz (fail-CLOSED)
    # Anti-stale-sidecar guard: reap same-role strays + reclaim the port, THEN bind
    # EXCLUSIVELY so this is the ONLY live instance (a stale predecessor can never
    # co-serve). Record + clear a per-role pidfile for the cleanup process.
    sidecar_lock.guard_singleton("127.0.0.1", PORT, ROLE)
    threading.Thread(target=_parent_watchdog, daemon=True, name="guard-parent-watchdog").start()
    server = SingletonThreadingHTTPServer(("127.0.0.1", PORT), _Handler)
    sidecar_lock.write_role(ROLE, os.getpid(), PORT)
    atexit.register(sidecar_lock.clear_role, ROLE)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        sidecar_lock.clear_role(ROLE)


if __name__ == "__main__":
    main()
