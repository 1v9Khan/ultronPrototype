"""Sidecar embedder for the Ultron command router.

Runs as a SEPARATE process so the embedding model NEVER loads into Ultron's
anticheat-pinned main process -- the main process keeps only a ~10-line urllib
client (kenning.audio._router_backends.EmbeddingBackend), so the boot anticheat
canary stays "libs loaded=none" and there's no CPU contention with the gaming
LLM/STT inside the main process.

Two interchangeable backends (chosen by env), so a heavyweight model can live in
an ISOLATED venv without disturbing the main venv's pinned deps:

  * BACKEND=fastembed (default)        -> fastembed/ONNX, run from the MAIN venv.
                                          Default model BAAI/bge-small-en-v1.5
                                          (the model the RAG path already uses).
  * BACKEND=sentence_transformers      -> sentence-transformers + torch, run from
                                          the ISOLATED .venv-embedder. Default
                                          model google/embeddinggemma-300m on GPU.
                                          Honors per-kind prompt names so
                                          asymmetric query/document prompting
                                          works (EmbeddingGemma's recommended
                                          usage).

ANTICHEAT POSTURE: pure compute only (load a text-embedding model, serve cosine-
able vectors over a loopback HTTP socket). NO input injection, NO screen/window
capture, NO foreign-process memory, NO hooks, never touches the game -- the same
class as OBS/Discord. Binds 127.0.0.1 ONLY.

Protocol (JSON over HTTP):
  GET  /healthz                          -> {"ok":true,"model":...,"dim":N,"backend":...}
  POST /embed {"texts":[...],"kind":"query"|"document"} -> {"vectors":[[...]]}

Run:  python scripts/embedder_server.py [PORT] [MODEL]
Env:  KENNING_EMBEDDER_PORT, KENNING_EMBEDDER_MODEL, KENNING_EMBEDDER_BACKEND,
      KENNING_EMBEDDER_THREADS, KENNING_EMBEDDER_QUERY_PROMPT,
      KENNING_EMBEDDER_DOC_PROMPT, KENNING_EMBEDDER_DEVICE
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Default matches the ISOLATED venv (.venv-embedder has sentence-transformers,
# not fastembed). The orchestrator always sets this explicitly when spawning;
# the default only affects a manual run, which is normally from that venv.
BACKEND = os.environ.get("KENNING_EMBEDDER_BACKEND", "sentence_transformers").lower()
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(
    os.environ.get("KENNING_EMBEDDER_PORT", "8772"))
_default_model = ("google/embeddinggemma-300m"
                  if BACKEND == "sentence_transformers"
                  else "BAAI/bge-small-en-v1.5")
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
    "KENNING_EMBEDDER_MODEL", _default_model)
THREADS = int(os.environ.get("KENNING_EMBEDDER_THREADS", "2"))
DEVICE = os.environ.get("KENNING_EMBEDDER_DEVICE", "")  # "" -> auto
# Prompt names (sentence_transformers backend only). Empty -> no prompt.
QUERY_PROMPT = os.environ.get("KENNING_EMBEDDER_QUERY_PROMPT", "") or None
DOC_PROMPT = os.environ.get("KENNING_EMBEDDER_DOC_PROMPT", "") or None

_st = None       # SentenceTransformer
_fe = None       # fastembed TextEmbedding
_dim = 0


def _load():
    global _st, _fe, _dim
    if BACKEND == "sentence_transformers":
        import torch
        from sentence_transformers import SentenceTransformer
        dev = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        _st = SentenceTransformer(MODEL, device=dev)
        _dim = int(_st.get_sentence_embedding_dimension())
        prompts = list(getattr(_st, "prompts", {}) or {})
        print("[embedder_server] ready BACKEND=sentence_transformers model=%s "
              "device=%s dim=%d port=%d prompts=%s q=%s d=%s"
              % (MODEL, dev, _dim, PORT, prompts, QUERY_PROMPT, DOC_PROMPT),
              flush=True)
    else:
        from fastembed import TextEmbedding
        _fe = TextEmbedding(MODEL, threads=THREADS)
        _dim = int(len(next(iter(_fe.embed(["probe"])))))
        print("[embedder_server] ready BACKEND=fastembed model=%s dim=%d "
              "port=%d threads=%d" % (MODEL, _dim, PORT, THREADS), flush=True)


def _embed(texts, kind):
    if not texts:
        return []
    if _st is not None:
        pn = QUERY_PROMPT if kind == "query" else DOC_PROMPT
        kw = {"normalize_embeddings": True}
        if pn:
            kw["prompt_name"] = pn
        return [list(map(float, v)) for v in _st.encode(list(texts), **kw)]
    return [list(map(float, v)) for v in _fe.embed(list(texts))]


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200, {"ok": True, "model": MODEL, "dim": _dim,
                             "backend": BACKEND})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/embed":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
            payload = json.loads(self.rfile.read(n) or b"{}")
            kind = payload.get("kind", "document")
            self._send(200, {"vectors": _embed(payload.get("texts", []), kind)})
        except Exception as e:                                    # noqa: BLE001
            self._send(500, {"error": str(e)})

    def log_message(self, *args):
        return


def _pid_alive(pid: int) -> bool:
    """True iff process ``pid`` is still running. psutil if present, else a
    ctypes OpenProcess+GetExitCodeProcess check on Windows / os.kill(0) on POSIX.
    Fail-SAFE: an indeterminate result returns True (never self-kill on doubt)."""
    if pid <= 0:
        return True
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:                                            # noqa: BLE001
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            k = ctypes.windll.kernel32
            h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False            # cannot open -> gone
            code = wintypes.DWORD()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code))
            k.CloseHandle(h)
            return (not ok) or code.value == STILL_ACTIVE
        except Exception:                                        # noqa: BLE001
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:                                            # noqa: BLE001
        return True


def _parent_watchdog() -> None:
    """Self-exit when the parent (Ultron orchestrator) dies, so a force-killed
    or crashed parent NEVER leaves this embedder as a runaway orphan. The parent
    PID is passed via KENNING_EMBEDDER_PARENT_PID (fallback: the spawn-time
    parent). Polls every few seconds; ``os._exit`` skips atexit/locks so the
    model+VRAM are freed immediately by the OS."""
    try:
        ppid = int(os.environ.get("KENNING_EMBEDDER_PARENT_PID", "0") or "0")
    except Exception:                                            # noqa: BLE001
        ppid = 0
    if ppid <= 0:
        ppid = os.getppid()
    if ppid <= 0:
        return
    sys.stderr.write(f"[embedder] parent-watchdog armed on pid {ppid}\n")
    sys.stderr.flush()
    while True:
        time.sleep(3.0)
        if not _pid_alive(ppid):
            sys.stderr.write(
                f"[embedder] parent pid {ppid} gone -> self-terminating "
                "(orphan guard)\n")
            sys.stderr.flush()
            os._exit(0)


class _ExclusiveHTTPServer(ThreadingHTTPServer):
    """Single-instance bind (anti-stale-sidecar): never SO_REUSEADDR, and
    SO_EXCLUSIVEADDRUSE on Windows so no other process can co-bind the port -- a
    duplicate fails LOUDLY at bind instead of silently co-serving stale code.
    Pure stdlib so it works in the ISOLATED embedder venv (no kenning import)."""

    allow_reuse_address = False

    def server_bind(self):
        import socket
        if sys.platform == "win32":
            excl = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if excl is not None:
                try:
                    self.socket.setsockopt(socket.SOL_SOCKET, excl, 1)
                except OSError:
                    pass
        super().server_bind()


def main():
    _load()
    # Parent-death deadman: the strongest orphan guard -- the child cleans itself
    # up on ANY parent death (crash, taskkill /F, TerminateProcess), which no
    # in-parent cleanup can cover.
    threading.Thread(target=_parent_watchdog, daemon=True,
                     name="parent-watchdog").start()
    server = _ExclusiveHTTPServer(("127.0.0.1", PORT), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
