"""XTTS v2 HTTP server.

Long-lived process living in the isolated XTTS venv. Loads the model
once, pre-computes the Ultron speaker embedding once, then serves
streaming synthesis requests over HTTP.

The Ultron orchestrator (in the main venv) talks to this server via
:mod:`src.ultron.tts.xtts_v3` (the client). HTTP keeps the two venvs
decoupled -- the main venv doesn't have to deal with Coqui's
transformers 4.x pin or the omegaconf / hydra constraints that
fairseq/RVC also needs.

Endpoints:

* ``GET /healthz`` -- ``{ok, model_loaded, speaker_cached}``. Used
  by the client to wait for readiness on startup.

* ``POST /synthesize`` -- body ``{text, language="en", chunk_size=20,
  temperature=0.75, speed=1.0}``. Returns ``Transfer-Encoding: chunked`` stream
  of raw int16 little-endian mono PCM at the model's native sample
  rate (24000 Hz). Each network chunk corresponds to one XTTS
  inference chunk.

* ``GET /info`` -- ``{sample_rate, reference_audio, ...}``.

* ``POST /shutdown`` -- best-effort graceful exit (the supervising
  client may also just SIGKILL the subprocess; both are supported).

Run:

    python xtts_server.py [--host 127.0.0.1] [--port 8770] \\
        [--reference C:/path/to/reference.wav]

If the reference is omitted, the cleaned Ultron mono reference is
used by default.
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

# IMPORTANT: F5TTS-era investigation showed that on Windows, importing
# torch BEFORE the TTS native bits triggers a silent crash. The Coqui
# TTS path uses torch internally too, so we let TTS import first.
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_REFERENCE = PROJECT / "Ultron_vocals_mono_v1.wav"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("xtts_server")


# ---------------------------------------------------------------------------
# Model holder. Constructed at startup; immutable afterwards (single
# voice per server instance).
# ---------------------------------------------------------------------------


class XttsHolder:
    def __init__(
        self,
        reference_wav: Path,
        *,
        gpt_cond_len: int = 30,
        gpt_cond_chunk_len: int = 6,
        max_ref_length: int = 60,
    ):
        self.reference_wav = reference_wav
        # 2026-05-20 round 9: extended reference-window conditioning.
        # Coqui XTTS-v2 library defaults are ``gpt_cond_len=6`` and
        # ``max_ref_length=30`` -- so the prior code (which omitted
        # these args) only let ~6 s of prosody + ~30 s of speaker-
        # encoder audio reach the model despite us handing it a full
        # 3-minute reference. Surfaced as constructor args so the
        # client can override via CLI from config.yaml.
        self.gpt_cond_len = int(gpt_cond_len)
        self.gpt_cond_chunk_len = int(gpt_cond_chunk_len)
        self.max_ref_length = int(max_ref_length)
        self.model = None
        self.config = None
        self.gpt_latent = None
        self.speaker_embedding = None
        self.sample_rate: int = 0
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()  # XTTS isn't thread-safe

    def load(self) -> None:
        with self._load_lock:
            if self.model is not None:
                return
            t0 = time.monotonic()
            from TTS.utils.manage import ModelManager
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts

            manager = ModelManager()
            model_path, _, _ = manager.download_model(
                "tts_models/multilingual/multi-dataset/xtts_v2"
            )

            config = XttsConfig()
            config.load_json(str(Path(model_path) / "config.json"))
            model = Xtts.init_from_config(config)
            model.load_checkpoint(
                config,
                checkpoint_dir=str(model_path),
                eval=True,
            )
            import torch
            if torch.cuda.is_available():
                model.cuda()
            self.model = model
            self.config = config
            self.sample_rate = config.audio.output_sample_rate
            logger.info(
                "XTTS loaded in %.1fs, sample_rate=%d, vram=%dMB",
                time.monotonic() - t0,
                self.sample_rate,
                int(torch.cuda.memory_allocated() / 1e6) if torch.cuda.is_available() else 0,
            )

            t0 = time.monotonic()
            self.gpt_latent, self.speaker_embedding = (
                model.get_conditioning_latents(
                    audio_path=str(self.reference_wav),
                    gpt_cond_len=self.gpt_cond_len,
                    gpt_cond_chunk_len=self.gpt_cond_chunk_len,
                    max_ref_length=self.max_ref_length,
                )
            )
            logger.info(
                "speaker embedding computed in %.2fs from %s "
                "(gpt_cond_len=%d gpt_cond_chunk_len=%d max_ref_length=%d)",
                time.monotonic() - t0,
                self.reference_wav.name,
                self.gpt_cond_len,
                self.gpt_cond_chunk_len,
                self.max_ref_length,
            )

            # Warmup pass: compile any kernels so the first real request
            # doesn't pay the JIT cost.
            t0 = time.monotonic()
            for _ in model.inference_stream(
                text="Hello.",
                language="en",
                gpt_cond_latent=self.gpt_latent,
                speaker_embedding=self.speaker_embedding,
                stream_chunk_size=20,
            ):
                pass
            logger.info("warmup pass in %.2fs", time.monotonic() - t0)

    def stream(
        self,
        text: str,
        language: str = "en",
        chunk_size: int = 20,
        temperature: float = 0.75,
        speed: float = 1.0,
    ):
        """Generator yielding raw int16 little-endian PCM bytes, one
        XTTS inference chunk at a time. Acquires the inference lock
        for the duration.

        ``speed`` is XTTS v2's native duration multiplier (>1.0 =
        faster). Adjusts the GPT duration tokens before waveform
        decoding, so pitch and timbre are unchanged."""
        if self.model is None:
            raise RuntimeError("model not loaded")
        with self._inference_lock:
            for chunk in self.model.inference_stream(
                text=text,
                language=language,
                gpt_cond_latent=self.gpt_latent,
                speaker_embedding=self.speaker_embedding,
                stream_chunk_size=chunk_size,
                temperature=temperature,
                speed=speed,
            ):
                # chunk is a torch tensor (float32) in [-1, 1] on GPU
                pcm_f32 = chunk.detach().cpu().numpy().astype(np.float32)
                # Clip to [-1, 1] then convert to int16 little-endian.
                np.clip(pcm_f32, -1.0, 1.0, out=pcm_f32)
                pcm_i16 = (pcm_f32 * 32767.0).astype(np.int16)
                yield pcm_i16.tobytes()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


class SynthRequest(BaseModel):
    text: str
    language: str = "en"
    chunk_size: int = 20
    temperature: float = 0.75
    # XTTS native duration multiplier. 1.0 = native rate; >1.0 = faster.
    # Set by the client from ``tts.xtts_v3.speed`` in config.
    speed: float = 1.0


def build_app(holder: XttsHolder) -> FastAPI:
    app = FastAPI(title="Ultron XTTS server", version="1.0")

    @app.get("/healthz")
    def healthz():
        return {
            "ok": True,
            "model_loaded": holder.model is not None,
            "speaker_cached": holder.speaker_embedding is not None,
        }

    @app.get("/info")
    def info():
        return {
            "sample_rate": holder.sample_rate,
            "reference_audio": str(holder.reference_wav),
            "model_loaded": holder.model is not None,
            # 2026-05-20 round 9: surface the conditioning-window
            # params so the sample-gen driver can A/B verify the
            # server is actually using the configured values.
            "gpt_cond_len": holder.gpt_cond_len,
            "gpt_cond_chunk_len": holder.gpt_cond_chunk_len,
            "max_ref_length": holder.max_ref_length,
        }

    @app.post("/synthesize")
    async def synthesize(req: SynthRequest):
        if holder.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="empty text")

        # XTTS inference is synchronous + slow; run it in a thread and
        # push results onto an asyncio.Queue that the response generator
        # awaits. Async generator path avoids FastAPI's sync-generator
        # threadpool overhead which added ~280 ms TTFB in initial
        # measurement (the sync gen was scheduled into Starlette's
        # default 40-thread pool with batch flush behaviour).
        import asyncio
        loop = asyncio.get_event_loop()
        # Bounded queue: backpressure if client reads slowly. 8 chunks
        # is plenty for a single sentence.
        out_q: asyncio.Queue = asyncio.Queue(maxsize=8)
        SENTINEL = object()
        t_start = time.monotonic()
        first_chunk_logged = [False]

        def producer():
            try:
                for pcm_bytes in holder.stream(
                    text=req.text,
                    language=req.language,
                    chunk_size=req.chunk_size,
                    temperature=req.temperature,
                    speed=req.speed,
                ):
                    if not first_chunk_logged[0]:
                        first_chunk_logged[0] = True
                        logger.info(
                            "first chunk produced at %.0f ms (text=%.40r)",
                            (time.monotonic() - t_start) * 1000,
                            req.text,
                        )
                    asyncio.run_coroutine_threadsafe(
                        out_q.put(pcm_bytes), loop
                    ).result()
            except Exception as e:
                logger.exception("producer failed: %s", e)
            finally:
                asyncio.run_coroutine_threadsafe(
                    out_q.put(SENTINEL), loop
                ).result()

        # Kick off XTTS in a worker thread; the generator below awaits
        # the queue.
        threading.Thread(target=producer, daemon=True, name="xtts-producer").start()

        async def agen():
            sent_first = False
            while True:
                item = await out_q.get()
                if item is SENTINEL:
                    return
                if not sent_first:
                    sent_first = True
                    logger.info(
                        "first chunk sent at %.0f ms",
                        (time.monotonic() - t_start) * 1000,
                    )
                yield item

        headers = {
            "X-Sample-Rate": str(holder.sample_rate),
            "X-Audio-Format": "s16le-mono",
        }
        return StreamingResponse(
            agen(),
            media_type="application/octet-stream",
            headers=headers,
        )

    @app.post("/shutdown")
    def shutdown():
        logger.info("shutdown requested")
        # Schedule the process exit on a background thread so the
        # response can flush first.
        def _exit():
            time.sleep(0.2)
            os._exit(0)
        threading.Thread(target=_exit, daemon=True).start()
        return JSONResponse({"shutting_down": True})

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ultron XTTS HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Path to the speaker reference WAV (default: cleaned Ultron mono).",
    )
    parser.add_argument(
        "--gpt-cond-len",
        type=int,
        default=30,
        help=(
            "Seconds of reference audio fed to the XTTS GPT prosody encoder. "
            "Coqui library default is 6; bumped to 30 (2026-05-20 round 9) "
            "so the 3-min Ultron reference actually contributes more than "
            "the first ~6 s."
        ),
    )
    parser.add_argument(
        "--gpt-cond-chunk-len",
        type=int,
        default=6,
        help=(
            "Per-chunk size (s) for GPT prosody conditioning. XTTS averages "
            "over gpt_cond_len/gpt_cond_chunk_len chunks. Keep at 6 unless "
            "you know why you're changing it."
        ),
    )
    parser.add_argument(
        "--max-ref-length",
        type=int,
        default=60,
        help=(
            "Seconds of reference audio fed to the HiFi-GAN speaker encoder. "
            "Coqui library default is 30; bumped to 60 (2026-05-20 round 9)."
        ),
    )
    args = parser.parse_args(argv)

    if not args.reference.is_file():
        print(f"ERROR: reference audio missing: {args.reference}")
        return 1

    holder = XttsHolder(
        args.reference,
        gpt_cond_len=args.gpt_cond_len,
        gpt_cond_chunk_len=args.gpt_cond_chunk_len,
        max_ref_length=args.max_ref_length,
    )
    print(
        f"loading XTTS + reference {args.reference} "
        f"(gpt_cond_len={args.gpt_cond_len} "
        f"gpt_cond_chunk_len={args.gpt_cond_chunk_len} "
        f"max_ref_length={args.max_ref_length}) ..."
    )
    holder.load()

    app = build_app(holder)
    print(f"\nXTTS server ready: http://{args.host}:{args.port}")
    print(f"  GET  /healthz")
    print(f"  GET  /info")
    print(f"  POST /synthesize {{ text }}")
    print(f"  POST /shutdown")

    # Single-worker uvicorn -- XTTS isn't thread-safe so we don't want
    # multiple worker processes contending for the GPU.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
