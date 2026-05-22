"""Parakeet TDT STT HTTP server.

Long-lived process living in the isolated ``.venv-parakeet`` venv.
Loads the NVIDIA Parakeet TDT model once, then serves transcription
requests over HTTP.

The Ultron orchestrator (in the main venv) talks to this server via
:mod:`src.ultron.transcription.parakeet_engine` (the client). HTTP keeps
the two venvs decoupled -- the main venv stays on numpy<2.0,
transformers 4.41.2, librosa 0.9.1 (pinned for the rest of the voice
stack), while NeMo in this venv can use its own newer versions.

Endpoints:

* ``GET /healthz`` -- ``{ok, model_loaded, model_name}``. Used by
  the client to wait for readiness on startup.

* ``POST /transcribe`` -- multipart form with field ``audio`` (raw
  WAV bytes, mono float32 or int16). Returns ``{text, audio_seconds,
  inference_ms}``.

* ``GET /info`` -- ``{model_name, device, sample_rate}``.

* ``POST /shutdown`` -- best-effort graceful exit.

Run:

    .venv-parakeet/Scripts/python.exe parakeet_server.py \\
        [--host 127.0.0.1] [--port 8771] \\
        [--model nvidia/parakeet-tdt-0.6b-v3] [--device cuda]
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("parakeet_server")


# ---------------------------------------------------------------------------
# Model holder
# ---------------------------------------------------------------------------


class ParakeetHolder:
    """Lazy NeMo model wrapper. Loads on construction; immutable
    afterwards."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            logger.info("Loading Parakeet model %s on %s ...",
                        self.model_name, self.device)
            t0 = time.monotonic()
            import nemo.collections.asr as nemo_asr
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name,
            )
            if hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
            if hasattr(self._model, "freeze"):
                self._model.freeze()
            logger.info("Parakeet loaded in %.2fs",
                        time.monotonic() - t0)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        result = self._model.transcribe(audio=[audio], batch_size=1)
        if not result:
            return ""
        hyp = result[0]
        text = hyp.text if hasattr(hyp, "text") else str(hyp)
        return text.strip()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


_holder: Optional[ParakeetHolder] = None
_shutdown_event = threading.Event()


def make_app(holder: ParakeetHolder) -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {
            "ok": True,
            "model_loaded": holder._model is not None,
            "model_name": holder.model_name,
        }

    @app.get("/info")
    def info():
        sr = 16000
        return {
            "model_name": holder.model_name,
            "device": holder.device,
            "sample_rate": sr,
        }

    @app.post("/transcribe")
    async def transcribe(audio: UploadFile = File(...)):
        try:
            raw = await audio.read()
            audio_array, sr = sf.read(io.BytesIO(raw), dtype="float32",
                                       always_2d=False)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1).astype("float32")
            if sr != 16000:
                # Parakeet expects 16 kHz. The Ultron pipeline already
                # standardises on 16 kHz so this should rarely fire.
                logger.warning("Audio at %d Hz; expected 16000 Hz", sr)
            t0 = time.monotonic()
            text = holder.transcribe(audio_array)
            inference_ms = (time.monotonic() - t0) * 1000
            return JSONResponse({
                "text": text,
                "audio_seconds": len(audio_array) / max(sr, 1),
                "inference_ms": inference_ms,
            })
        except Exception as e:
            logger.error("transcribe failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/shutdown")
    def shutdown():
        _shutdown_event.set()
        return {"ok": True}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    global _holder
    _holder = ParakeetHolder(args.model, args.device)
    app = make_app(_holder)

    config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def watch_shutdown():
        _shutdown_event.wait()
        logger.info("Shutdown event received; stopping server.")
        server.should_exit = True

    t = threading.Thread(target=watch_shutdown, daemon=True)
    t.start()

    logger.info("Parakeet server listening on http://%s:%d",
                args.host, args.port)
    server.run()


if __name__ == "__main__":
    main()
