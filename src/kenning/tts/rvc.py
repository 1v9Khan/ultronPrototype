"""RVC v2 voice conversion via ``infer-rvc-python``.

Takes Piper's neutral-voice audio and re-paints it as the trained target
voice (Kenning / James Spader). Adds ~300 ms per sentence on a 3060 Ti.

Inference runs in-memory on numpy arrays — no temp WAV files in the hot
path — via ``BaseLoader.generate_from_cache((audio, sr), tag=...)``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import settings
from kenning.utils.fairseq_compat import (
    patch_fairseq_dataclasses,
    patch_torch_load_for_fairseq,
)
from kenning.utils.logging import get_logger

logger = get_logger("tts.rvc")

_TAG = "kenning"  # internal handle for the loaded voice


class RvcConverter:
    """Wraps a trained RVC v2 ``.pth`` + ``.index`` pair.

    Args:
        model_path: Path to the ``.pth`` weights file.
        index_path: Path to the FAISS retrieval ``.index``.
        device: ``cuda:0`` recommended; ``cpu`` works but is much slower.
    """

    def __init__(
        self,
        model_path: Path = settings.RVC_MODEL_PATH,
        index_path: Path = settings.RVC_INDEX_PATH,
        hubert_path: Path = settings.RVC_HUBERT_PATH,
        rmvpe_path: Path = settings.RVC_RMVPE_PATH,
        device: str = settings.RVC_DEVICE,
    ) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"RVC model not found: {model_path}")
        if not Path(index_path).is_file():
            raise FileNotFoundError(f"RVC index not found: {index_path}")

        self.model_path = Path(model_path)
        self.index_path = Path(index_path)
        self.hubert_path = Path(hubert_path)
        self.rmvpe_path = Path(rmvpe_path)
        self.device = device
        self._converter = None
        self._load()

    # --- model loading -------------------------------------------------------

    def _load(self) -> None:
        patch_fairseq_dataclasses()
        patch_torch_load_for_fairseq()
        from infer_rvc_python import BaseLoader

        logger.info("Loading RVC model (%s) on %s…", self.model_path.name, self.device)
        t0 = time.monotonic()
        try:
            self._converter = BaseLoader(
                only_cpu=self.device.startswith("cpu"),
                hubert_path=str(self.hubert_path) if self.hubert_path.is_file() else None,
                rmvpe_path=str(self.rmvpe_path) if self.rmvpe_path.is_file() else None,
            )
            self._converter.apply_conf(
                tag=_TAG,
                file_model=str(self.model_path),
                pitch_algo=settings.RVC_F0_METHOD,
                pitch_lvl=settings.RVC_PITCH_SHIFT,
                file_index=str(self.index_path),
                index_influence=settings.RVC_INDEX_RATE,
                respiration_median_filtering=settings.RVC_FILTER_RADIUS,
                envelope_ratio=settings.RVC_RMS_MIX_RATE,
                consonant_breath_protection=settings.RVC_PROTECT,
            )
        except Exception as e:
            logger.error("RVC load failed: %s", e)
            raise
        logger.info("RVC ready in %.2fs", time.monotonic() - t0)

    # --- context manager -----------------------------------------------------

    def __enter__(self) -> "RvcConverter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the model so VRAM can be reclaimed at GC."""
        self._converter = None

    # --- inference -----------------------------------------------------------

    def convert(self, pcm_int16: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Convert Piper's neutral PCM into the target voice.

        Args:
            pcm_int16: mono int16 samples from Piper.
            sample_rate: source sample rate (Piper medium = 22050).

        Returns:
            ``(converted_pcm_int16, output_sample_rate)``. RVC chooses its
            own output rate (typically 40000 Hz for v2 models) — pass it
            through to whoever plays the audio.
        """
        if self._converter is None:
            raise RuntimeError("RvcConverter is not loaded")
        if pcm_int16.size == 0:
            return pcm_int16, sample_rate

        # infer-rvc-python wants float32 in [-1, 1].
        audio_f32 = pcm_int16.astype(np.float32) / 32768.0

        t0 = time.monotonic()
        try:
            out_audio, out_sr = self._converter.generate_from_cache(
                audio_data=(audio_f32, sample_rate),
                tag=_TAG,
            )
        except Exception as e:
            logger.exception("RVC inference failed: %s", e)
            return pcm_int16, sample_rate  # fail soft

        # Normalize output to int16 PCM regardless of upstream dtype.
        out_audio = np.asarray(out_audio).reshape(-1)
        if out_audio.dtype.kind == "f":
            out_pcm = np.clip(out_audio * 32767.0, -32768, 32767).astype(np.int16)
        else:
            out_pcm = out_audio.astype(np.int16, copy=False)

        logger.debug(
            "RVC: %.2fs in @ %d Hz → %.2fs out @ %d Hz in %.0fms",
            len(pcm_int16) / sample_rate,
            sample_rate,
            len(out_pcm) / max(out_sr, 1),
            int(out_sr),
            (time.monotonic() - t0) * 1000,
        )
        return out_pcm, int(out_sr)
