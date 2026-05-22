"""NVIDIA Parakeet TDT STT engine (frontier item 5, 2026-05-21).

Drop-in replacement for :class:`WhisperEngine` -- same
``transcribe(audio: np.ndarray, language: Optional[str]) -> str``
interface, very different model underneath.

**Isolation architecture (default):** NeMo runs in an isolated venv
at ``ultronVoiceAudio/.venv-parakeet/`` via a FastAPI HTTP server
(``ultronVoiceAudio/scripts/parakeet_server.py``). The main venv
stays clean of NeMo's dependency cascade (transformers 4.57+,
numpy>=2, librosa 0.11+, hydra 1.3+) which is incompatible with
the rest of the voice stack (fairseq/RVC need hydra<1.1; torchcrepe
needs librosa==0.9.1; the LLM stack uses transformers 4.41.2).
Mirrors the XTTS pattern at ``.venv-xtts``.

**Easy reversibility (the "variable switch"):**
- ``stt.engine: whisper`` in config.yaml -> Whisper, instant revert.
- ``stt.engine: parakeet`` -> Parakeet via the isolated venv.
- ``stt.engine: auto`` -> Parakeet if .venv-parakeet exists, else Whisper.
- ``stt.parakeet_use_isolated_venv: false`` (advanced) -> attempt to
  import NeMo from the main venv. NOT recommended -- the main venv
  pin set is incompatible with NeMo.

Why Parakeet?
- RNN-Transducer architecture: streaming-native.
- ~RTFx 2000+ on consumer GPUs (Whisper base ~100). On 5 s of audio,
  expect ~5-20 ms inference vs Whisper's ~80 ms.

*** IF VOICE TRANSCRIPTION QUALITY REGRESSES AFTER 2026-05-21,
SUSPECT THIS ENGINE FIRST. *** Roll back with
``stt.engine: whisper`` to confirm whether the regression is
Parakeet-specific before chasing other causes.
"""

from __future__ import annotations

import atexit
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import settings
from ultron.errors import WhisperTranscriptionError
from ultron.resilience import get_error_log
from ultron.utils.logging import get_logger

logger = get_logger("transcription.parakeet")


PARAKEET_INSTALL_HINT = (
    "Parakeet requires either:\n"
    "  (a) An isolated venv at ultronVoiceAudio/.venv-parakeet/ with\n"
    "      nemo_toolkit[asr] + fastapi + uvicorn + soundfile installed.\n"
    "      Set up with:\n"
    "        python -m venv ultronVoiceAudio/.venv-parakeet\n"
    "        ultronVoiceAudio/.venv-parakeet/Scripts/python.exe \\\n"
    "            -m pip install nemo_toolkit[asr] fastapi uvicorn soundfile\n"
    "  (b) Or NeMo in the main venv (set stt.parakeet_use_isolated_venv:\n"
    "      false). NOT RECOMMENDED -- breaks the main venv's pinned\n"
    "      dependencies (numpy<2.0, transformers 4.41.2, librosa 0.9.1).\n"
    "\n"
    "Or revert to Whisper with stt.engine: whisper in config.yaml."
)


def is_nemo_available() -> bool:
    """Check if NeMo can be imported.

    Two modes:
    1. Isolated venv (default): check that
       ``ultronVoiceAudio/.venv-parakeet/`` exists with a python
       executable AND the parakeet_server.py script is present.
       We don't actually try to import NeMo from the main venv --
       that's expected to fail.
    2. Main venv (opt-in): use ``importlib.util.find_spec`` to
       check NeMo presence in the current interpreter.

    Returns False on any error.
    """
    try:
        from ultron.config import get_config
        stt_cfg = get_config().stt
        use_isolated = getattr(stt_cfg, "parakeet_use_isolated_venv", True)
    except Exception:                                                  # noqa: BLE001
        use_isolated = True

    if use_isolated:
        try:
            from ultron.config import resolve_path
            python_exe = resolve_path(
                getattr(stt_cfg, "parakeet_server_python",
                        "ultronVoiceAudio/.venv-parakeet/Scripts/python.exe"),
            )
            script = resolve_path(
                getattr(stt_cfg, "parakeet_server_script",
                        "ultronVoiceAudio/scripts/parakeet_server.py"),
            )
            return Path(python_exe).is_file() and Path(script).is_file()
        except Exception:                                              # noqa: BLE001
            return False
    else:
        try:
            import importlib.util
            return importlib.util.find_spec("nemo.collections.asr") is not None
        except Exception:                                              # noqa: BLE001
            return False


# Module-level singleton for the spawned server process so we don't
# accidentally launch multiple servers on repeated ParakeetEngine
# constructions (e.g., in tests).
_SERVER_PROCESS: Optional[subprocess.Popen] = None
_SERVER_URL_CACHED: Optional[str] = None


def _spawn_server_if_needed(stt_cfg) -> str:
    """Ensure the Parakeet FastAPI server is running. Returns its URL.

    Idempotent: if the server is already running (either spawned by
    this process or by a previous one on the same port), the existing
    instance is reused.
    """
    global _SERVER_PROCESS, _SERVER_URL_CACHED

    import requests

    url = getattr(stt_cfg, "parakeet_server_url",
                  "http://127.0.0.1:8771")
    startup_timeout = float(
        getattr(stt_cfg, "parakeet_server_startup_timeout_seconds", 60.0)
    )

    # Already healthy?
    try:
        r = requests.get(f"{url}/healthz", timeout=1.0)
        if r.ok and r.json().get("model_loaded"):
            logger.info("Parakeet server already running at %s", url)
            _SERVER_URL_CACHED = url
            return url
    except Exception:
        pass  # not running; spawn below

    from ultron.config import resolve_path
    python_exe = str(resolve_path(
        getattr(stt_cfg, "parakeet_server_python",
                "ultronVoiceAudio/.venv-parakeet/Scripts/python.exe"),
    ))
    script = str(resolve_path(
        getattr(stt_cfg, "parakeet_server_script",
                "ultronVoiceAudio/scripts/parakeet_server.py"),
    ))
    if not Path(python_exe).is_file():
        raise ImportError(
            f"Parakeet venv not found at {python_exe}. {PARAKEET_INSTALL_HINT}"
        )
    if not Path(script).is_file():
        raise ImportError(
            f"Parakeet server script not found at {script}."
        )

    model = getattr(stt_cfg, "parakeet_model", "nvidia/parakeet-tdt-0.6b-v3")
    device = getattr(stt_cfg, "parakeet_device", "cuda")
    # Parse port from URL.
    from urllib.parse import urlparse
    parsed = urlparse(url)
    port = parsed.port or 8771
    host = parsed.hostname or "127.0.0.1"

    logger.info("Spawning Parakeet server: %s %s (model=%s device=%s port=%d)",
                python_exe, script, model, device, port)
    cmd = [
        python_exe, script,
        "--host", host,
        "--port", str(port),
        "--model", model,
        "--device", device,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32" else 0
        ),
    )
    _SERVER_PROCESS = proc

    # Register cleanup so the server dies with us. Best-effort -- if
    # the user kills the orchestrator with -9, the server stays
    # running (operator can check `tasklist` and kill manually).
    def _cleanup():
        try:
            if _SERVER_PROCESS is not None and _SERVER_PROCESS.poll() is None:
                # Try graceful first.
                try:
                    requests.post(f"{url}/shutdown", timeout=2.0)
                except Exception:
                    pass
                _SERVER_PROCESS.terminate()
        except Exception:
            pass

    atexit.register(_cleanup)

    # Health-check poll loop.
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{url}/healthz", timeout=2.0)
            if r.ok and r.json().get("model_loaded"):
                logger.info("Parakeet server ready at %s", url)
                _SERVER_URL_CACHED = url
                return url
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(
                f"Parakeet server exited unexpectedly during startup "
                f"(returncode={proc.returncode}). Run the server "
                f"manually for full logs: {' '.join(cmd)}"
            )
        time.sleep(1.0)
    raise TimeoutError(
        f"Parakeet server failed to become healthy within "
        f"{startup_timeout:.0f}s. Check the .venv-parakeet venv has "
        f"NeMo installed."
    )


class ParakeetEngine:
    """NVIDIA Parakeet TDT speech-to-text via an isolated venv +
    FastAPI server.

    Args:
        model_name: HuggingFace / NGC model id. Defaults to
            ``nvidia/parakeet-tdt-0.6b-v3``.
        device: ``"cuda"`` (default) or ``"cpu"``.

    Raises:
        ImportError: when the isolated venv is not set up AND
            fallback to main-venv NeMo isn't available.
        TimeoutError: when the server fails to become healthy within
            ``parakeet_server_startup_timeout_seconds``.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        from ultron.config import get_config
        stt_cfg = get_config().stt

        self.model_name = model_name or getattr(
            stt_cfg, "parakeet_model", "nvidia/parakeet-tdt-0.6b-v3",
        )
        self.device = device or getattr(stt_cfg, "parakeet_device", "cuda")
        self.use_isolated_venv = bool(
            getattr(stt_cfg, "parakeet_use_isolated_venv", True)
        )
        self.request_timeout = float(
            getattr(stt_cfg, "parakeet_request_timeout_seconds", 30.0)
        )

        if self.use_isolated_venv:
            # Surface a clear ImportError BEFORE attempting subprocess
            # spawn when the venv isn't set up. The spawn function
            # would catch the same case but gives a more confusing
            # error mid-init.
            if not is_nemo_available():
                raise ImportError(PARAKEET_INSTALL_HINT)
            self._server_url = _spawn_server_if_needed(stt_cfg)
        else:
            # Main-venv path. NOT recommended; the main venv pin set
            # is incompatible with NeMo (transformers 4.41 vs NeMo's
            # 4.57 requirement, numpy<2.0 vs NeMo's 2.x, etc.).
            if not is_nemo_available():
                raise ImportError(PARAKEET_INSTALL_HINT)
            logger.warning(
                "Parakeet running in main venv (parakeet_use_isolated_venv=False). "
                "This may break the rest of the voice stack -- proceed only "
                "if you know what you're doing."
            )
            import nemo.collections.asr as nemo_asr
            t0 = time.monotonic()
            self._main_venv_model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name,
            )
            if hasattr(self._main_venv_model, "to"):
                self._main_venv_model = self._main_venv_model.to(self.device)
            if hasattr(self._main_venv_model, "freeze"):
                self._main_venv_model.freeze()
            logger.info("Parakeet (main-venv path) ready in %.2fs",
                        time.monotonic() - t0)
            self._server_url = None

    def __enter__(self) -> "ParakeetEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Server lifecycle is managed by atexit; nothing to do here.
        pass

    def transcribe(self, audio: np.ndarray, language: Optional[str] = "en") -> str:
        """Transcribe a mono float32 16 kHz audio segment to text."""
        if audio.size == 0:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        t0 = time.monotonic()
        try:
            if self.use_isolated_venv:
                text = self._transcribe_http(audio)
            else:
                result = self._main_venv_model.transcribe(
                    audio=[audio], batch_size=1,
                )
                hyp = result[0] if result else ""
                text = (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.error(
                "Parakeet transcribe failed in %.0fms: %s "
                "(if this is recurring, swap to stt.engine: whisper)",
                elapsed_ms, e,
            )
            get_error_log().record(
                WhisperTranscriptionError(
                    f"Parakeet transcribe failed: {e}",
                    context={
                        "audio_seconds": len(audio) / settings.SAMPLE_RATE,
                        "model": self.model_name,
                        "device": self.device,
                        "engine": "parakeet",
                        "isolated_venv": self.use_isolated_venv,
                    },
                    recovery=(
                        "returned empty transcription; orchestrator "
                        "skips this turn. Operator: consider "
                        "``stt.engine: whisper`` to revert."
                    ),
                ),
                dependency="parakeet",
            )
            return ""

        elapsed_ms = (time.monotonic() - t0) * 1000
        audio_seconds = len(audio) / settings.SAMPLE_RATE
        logger.info(
            "Parakeet: %.2fs audio -> %d chars in %.0fms (RTF=%.3f)",
            audio_seconds, len(text), elapsed_ms,
            elapsed_ms / 1000 / max(audio_seconds, 1e-6),
        )
        return text

    def _transcribe_http(self, audio: np.ndarray) -> str:
        """Send audio to the isolated-venv server and return the text."""
        import requests
        import soundfile as sf

        # Encode as WAV in-memory (16 kHz mono float32). The server
        # accepts any soundfile-readable format; WAV is the simplest.
        buf = io.BytesIO()
        sf.write(buf, audio, settings.SAMPLE_RATE, format="WAV", subtype="FLOAT")
        buf.seek(0)

        url = self._server_url or _SERVER_URL_CACHED
        if not url:
            raise RuntimeError(
                "Parakeet server URL not set; engine may have been "
                "constructed without going through the spawn helper."
            )

        r = requests.post(
            f"{url}/transcribe",
            files={"audio": ("audio.wav", buf, "audio/wav")},
            timeout=self.request_timeout,
        )
        r.raise_for_status()
        return str(r.json().get("text", "")).strip()


__all__ = ["ParakeetEngine", "is_nemo_available", "PARAKEET_INSTALL_HINT"]
