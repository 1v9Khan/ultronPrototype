"""Client shim for the out-of-process STT sidecar (2026-07-26).

Speaks the same surface as :class:`WhisperEngine` -- ``transcribe(audio,
language) -> str`` -- but the decode happens in ``scripts/stt_server.py``,
a child process the orchestrator masks onto ONE GPU with
``CUDA_VISIBLE_DEVICES``.

WHY: CTranslate2 on a NON-DEFAULT CUDA device (``device_index=1``) corrupted
memory in-process and killed Ultron with a silent 0xc0000409 fastfail -- no
traceback, no stderr, no WER dump. Every flight-recorder snapshot at the
moment of death had CTranslate2 mid-decode. Inside the child the target card
is plain ``cuda:0``, CTranslate2's default path, and a repeat of the
corruption kills a restartable child rather than the assistant.

Anticheat posture: stdlib only (urllib/base64/json) -- no new import surface
on the voice path.

Fail-open by construction: any transport error returns ``""``, exactly as
``WhisperEngine.transcribe`` does on a decode failure, so a dead sidecar
degrades to "heard nothing" instead of raising into the run loop.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

import numpy as np

from kenning.utils.logging import get_logger

logger = get_logger("transcription.sidecar")


class SidecarWhisperEngine:
    """Drop-in STT engine that delegates decoding to the sidecar process."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8779,
        timeout_s: float = 30.0,
        model_name: str = "sidecar",
    ) -> None:
        self._base = f"http://{host}:{int(port)}"
        self._timeout = float(timeout_s)
        self.model_name = model_name
        self.device = "sidecar"
        # Mirrors WhisperEngine's attribute surface so callers that read these
        # for logging/telemetry keep working.
        self.compute_type = "sidecar"
        self.beam_size = 1
        self._warned = False
        self._lock = threading.Lock()

    # -- health -----------------------------------------------------------
    def healthy(self, timeout_s: float = 2.0) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self._base}/healthz", timeout=timeout_s
            ) as r:
                return bool((json.loads(r.read() or b"{}") or {}).get("ok"))
        except Exception:                                        # noqa: BLE001
            return False

    def wait_until_ready(self, timeout_s: float = 180.0) -> bool:
        """Poll /healthz until the child has its weights loaded."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.healthy():
                return True
            time.sleep(0.5)
        return False

    # -- the WhisperEngine surface ----------------------------------------
    def transcribe(self, audio: np.ndarray,
                   language: Optional[str] = "en") -> str:
        if audio is None or audio.size == 0:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # Same silence gate as the in-process engine: never pay a round trip
        # (or a hallucinated "Thank you.") on room tone.
        if float(np.max(np.abs(audio))) < 0.008:
            return ""

        payload = json.dumps({
            "audio_b64": base64.b64encode(
                np.ascontiguousarray(audio).tobytes()).decode("ascii"),
            "language": language,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/transcribe", data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            # One decode at a time from this side too: the speculative and
            # foreground transcribes can overlap, and the sidecar serialises
            # anyway -- queueing here keeps the round trips orderly.
            with self._lock:
                with urllib.request.urlopen(req, timeout=self._timeout) as r:
                    body = json.loads(r.read() or b"{}") or {}
            text = str(body.get("text", "") or "").strip()
            self._warned = False
            logger.debug(
                "sidecar STT: %.2fs audio -> %d chars in %.0fms",
                audio.size / 16000.0, len(text),
                (time.monotonic() - t0) * 1000.0,
            )
            return text
        except Exception as e:                                   # noqa: BLE001
            if not self._warned:
                logger.error(
                    "STT sidecar unreachable/failed (%s) -- transcription "
                    "returns empty until it recovers", e)
                self._warned = True
            return ""
