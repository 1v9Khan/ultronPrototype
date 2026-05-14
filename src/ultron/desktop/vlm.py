"""Moondream2 vision-language model wrapper -- CPU, on-demand.

The default voice-path LLM (``josiefied-qwen3-8b``) is text-only.
This module gives Ultron a small vision-language model so screen
captures can be turned into structured descriptions for "explain
what I'm looking at" flows.

Design:

- **CPU-only** -- no VRAM impact. Voice path already peaks at ~10 GB
  on the 8B abliterated LLM; the 11.5 GB hard cap leaves no room for
  a GPU VLM. Moondream2 on CPU is slow (~5-8 s per query) but the
  user only pays that cost when they explicitly ask a contextual
  question.

- **Lazy load.** First :meth:`describe` call loads weights into RAM
  (~3.5 GB on disk, ~4-5 GB RAM after load). Subsequent calls reuse.
  An orchestrator that never asks a contextual question never pays
  the load.

- **transformers backend.** ``vikhyatk/moondream2`` is the canonical
  moondream2 distribution on HuggingFace. It ships with custom
  inference code via ``trust_remote_code=True``. The publisher
  (vikhyatk) is the moondream2 author; this matches the standard
  installation pattern documented at huggingface.co/vikhyatk/moondream2.

- **Fail-open at every layer.** Missing transformers / failed model
  load / inference exception / out-of-memory all return ``None``.
  The screen-context module treats ``None`` as "no VLM description
  available" and the LLM falls back to text-only context (window
  title, UIA text, etc.).

- **No tokenization of secrets.** The PNG bytes flowing in are
  already tainted by :func:`ultron.safety.taint.get_taint_tracker`
  via the capture pipeline. The VLM output (text description) is
  NOT itself stamped as tainted -- it's a paraphrase, not the raw
  bytes -- so the model's own response about the image can flow
  freely into the LLM context. The bytes-exact taint protects
  against the model uploading the screenshot wholesale; the
  paraphrase is the product.
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ultron.utils.logging import get_logger

logger = get_logger("desktop.vlm")

# HuggingFace repo for the moondream2 model. The publisher is the model
# author; trust_remote_code is required (their custom inference code).
DEFAULT_MOONDREAM_REPO = "vikhyatk/moondream2"

# Pin to a stable revision. The model author updates ``main`` regularly
# and recent ``tokenizer.json`` revisions are incompatible with the
# tokenizers builds we have pinned -- surfaces as the runtime error
# "data did not match any variant of untagged enum ModelWrapper at
# line 255192 column 3" during from_pretrained(). ``2025-06-21`` is
# the documented stable release per huggingface.co/vikhyatk/moondream2
# README. Must match the value in scripts/download_models.py.
DEFAULT_MOONDREAM_REVISION = "2025-06-21"

# Default prompt when the caller doesn't supply one. Intentionally
# open-ended -- Ultron will frame the answer in its own voice via the
# LLM, not parrot back the description.
DEFAULT_DESCRIBE_PROMPT = (
    "Describe what is visible in this screenshot in 2-4 sentences. "
    "Identify the application, the main content area, any visible text "
    "or controls, and what the user appears to be doing."
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VLMResult:
    """One VLM query outcome.

    Attributes:
        success: True iff ``description`` is populated.
        description: model output, or None on failure.
        elapsed_ms: wall-clock from query start to result.
        error: failure reason when ``success=False``.
    """

    success: bool
    description: Optional[str] = None
    elapsed_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VLMLoadError(RuntimeError):
    """Raised at construction time when the VLM can't be initialised.

    Inference-time failures degrade to :class:`VLMResult` with
    ``success=False`` rather than raising.
    """


# ---------------------------------------------------------------------------
# Lazy transformers import
# ---------------------------------------------------------------------------


def _import_backend():
    """Lazy import of transformers + PIL + torch. Returns dict or None."""
    try:
        from PIL import Image  # type: ignore
        from transformers import (  # type: ignore[import]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        import torch  # type: ignore[import]
        return {
            "Image": Image,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "torch": torch,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("transformers / torch unavailable for VLM: %s", e)
        return None


# ---------------------------------------------------------------------------
# Moondream2 wrapper
# ---------------------------------------------------------------------------


class Moondream2VLM:
    """Vision-language model wrapper using moondream2 on CPU.

    Construction validates that the transformers + PIL + torch stack is
    importable but does NOT load model weights. Weights are loaded on
    the first :meth:`describe` call. Subsequent calls reuse the loaded
    model.
    """

    def __init__(
        self,
        *,
        repo: str = DEFAULT_MOONDREAM_REPO,
        revision: Optional[str] = DEFAULT_MOONDREAM_REVISION,
        device: str = "cpu",
        max_tokens: int = 200,
    ) -> None:
        backend = _import_backend()
        if backend is None:
            raise VLMLoadError(
                "transformers / torch not available; VLM disabled"
            )
        if device not in ("cpu", "cuda"):
            raise VLMLoadError(f"unsupported VLM device: {device}")
        if max_tokens < 8 or max_tokens > 1024:
            raise VLMLoadError(f"max_tokens out of range: {max_tokens}")
        self._backend = backend
        self._repo = repo
        self._revision = revision
        self._device = device
        self._max_tokens = int(max_tokens)
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._load_failed = False
        self._load_error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        """True once :meth:`_ensure_loaded` has succeeded."""
        return self._model is not None and self._tokenizer is not None

    @property
    def device(self) -> str:
        return self._device

    def warmup(self) -> bool:
        """Force the lazy-load now. Returns True on success."""
        try:
            self._ensure_loaded()
            return self.loaded
        except Exception as e:  # noqa: BLE001
            logger.warning("VLM warmup failed: %s", e)
            return False

    def _ensure_loaded(self) -> None:
        """Lazy-load the model + tokenizer. Idempotent. Thread-safe."""
        if self._model is not None:
            return
        if self._load_failed:
            # Don't retry a known-bad load every call -- the failure was
            # logged; the caller continues with VLM disabled.
            raise VLMLoadError(self._load_error or "previous load failed")

        with self._load_lock:
            if self._model is not None:
                return
            if self._load_failed:
                raise VLMLoadError(self._load_error or "previous load failed")
            try:
                t0 = time.time()
                logger.info(
                    "loading moondream2 (%s) on %s ...",
                    self._repo, self._device,
                )
                kwargs = {"trust_remote_code": True}
                if self._revision:
                    kwargs["revision"] = self._revision
                tok = self._backend["AutoTokenizer"].from_pretrained(
                    self._repo, **kwargs,
                )
                model = self._backend["AutoModelForCausalLM"].from_pretrained(
                    self._repo, **kwargs,
                )
                if self._device == "cuda":
                    model = model.to("cuda")
                model.eval()
                self._tokenizer = tok
                self._model = model
                logger.info(
                    "moondream2 loaded in %.1f s", time.time() - t0,
                )
            except Exception as e:  # noqa: BLE001
                self._load_failed = True
                self._load_error = str(e)[:300]
                logger.warning("moondream2 load failed: %s", e)
                raise VLMLoadError(f"moondream2 load failed: {e}") from e

    def describe(
        self,
        image_bytes: bytes,
        *,
        prompt: Optional[str] = None,
    ) -> VLMResult:
        """Return a description of an image.

        Args:
            image_bytes: PNG / JPEG / etc. -- anything Pillow can open.
            prompt: optional question for the VLM. When None, asks for
                a generic scene description.

        Returns:
            :class:`VLMResult`. ``success=False`` on any failure with
            a brief error string; the caller should treat as "no
            description available" and proceed.
        """
        t0 = time.time()
        if not image_bytes:
            return VLMResult(
                success=False, error="empty image bytes",
                elapsed_ms=(time.time() - t0) * 1000.0,
            )
        question = (prompt or DEFAULT_DESCRIBE_PROMPT).strip()
        if not question:
            question = DEFAULT_DESCRIBE_PROMPT

        try:
            self._ensure_loaded()
        except VLMLoadError as e:
            return VLMResult(
                success=False, error=str(e)[:200],
                elapsed_ms=(time.time() - t0) * 1000.0,
            )

        try:
            Image = self._backend["Image"]
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            return VLMResult(
                success=False, error=f"image decode failed: {e}",
                elapsed_ms=(time.time() - t0) * 1000.0,
            )

        try:
            torch = self._backend["torch"]
            with torch.no_grad():
                enc = self._model.encode_image(img)
                answer = self._model.answer_question(
                    enc, question, self._tokenizer,
                )
        except Exception as e:  # noqa: BLE001
            return VLMResult(
                success=False, error=f"inference failed: {e}",
                elapsed_ms=(time.time() - t0) * 1000.0,
            )

        text = (answer or "").strip()
        if not text:
            return VLMResult(
                success=False, error="empty model output",
                elapsed_ms=(time.time() - t0) * 1000.0,
            )
        return VLMResult(
            success=True, description=text,
            elapsed_ms=(time.time() - t0) * 1000.0,
        )

    def close(self) -> None:
        """Release model weights from memory. Idempotent."""
        self._model = None
        self._tokenizer = None


# ---------------------------------------------------------------------------
# Singleton + screen_context hook
# ---------------------------------------------------------------------------


_vlm_singleton: Optional[Moondream2VLM] = None
_vlm_lock = threading.Lock()


def get_vlm() -> Optional[Moondream2VLM]:
    """Module-level singleton accessor. May return None if no VLM was set."""
    return _vlm_singleton


def set_vlm(vlm: Optional[Moondream2VLM]) -> None:
    """Set the module-level VLM singleton.

    The orchestrator constructs a :class:`Moondream2VLM` at init time
    (when ``vlm.enabled=true`` in config) and pushes it via this hook.
    Also wires the screen_context hook so :func:`build_screen_context`
    can include a VLM description on demand.
    """
    global _vlm_singleton
    with _vlm_lock:
        _vlm_singleton = vlm
        # Wire (or unwire) the screen_context describe hook.
        from ultron.desktop.screen_context import set_vlm_describe
        if vlm is None:
            set_vlm_describe(None)
        else:
            set_vlm_describe(_describe_via_singleton)


def _describe_via_singleton(image_bytes: bytes) -> Optional[str]:
    """Bridge function passed to :func:`set_vlm_describe`.

    Routes image bytes through the singleton VLM and returns either
    the description text or None on failure.
    """
    vlm = get_vlm()
    if vlm is None:
        return None
    result = vlm.describe(image_bytes)
    return result.description if result.success else None


def build_vlm_from_config(
    *,
    enabled: bool,
    repo: str = DEFAULT_MOONDREAM_REPO,
    revision: Optional[str] = DEFAULT_MOONDREAM_REVISION,
    device: str = "cpu",
    max_tokens: int = 200,
) -> Optional[Moondream2VLM]:
    """Construct a :class:`Moondream2VLM` from configuration.

    Returns ``None`` when ``enabled=False`` OR construction fails
    (missing transformers, etc.). The orchestrator treats ``None`` as
    "VLM unavailable; fall back to text-only screen context".
    """
    if not enabled:
        return None
    try:
        return Moondream2VLM(
            repo=repo,
            revision=revision,
            device=device,
            max_tokens=max_tokens,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("VLM construction failed: %s", e)
        return None


__all__ = [
    "DEFAULT_MOONDREAM_REPO",
    "DEFAULT_MOONDREAM_REVISION",
    "DEFAULT_DESCRIBE_PROMPT",
    "Moondream2VLM",
    "VLMResult",
    "VLMLoadError",
    "get_vlm",
    "set_vlm",
    "build_vlm_from_config",
]
