"""OCR text extraction via Tesseract.

Catalog 08 T7 (GREEN). Fills the MIDDLE TIER between UIA text
extraction (semantic, instant, but unavailable on canvas-rendered
content) and the Moondream2 VLM (semantic, 300-800 ms, requires
GPU and 330 MB VRAM):

    UIA text     -> 5-30 ms,    semantic, UIA-only surfaces
    OCR (this)   -> 100-500 ms, raw text,  ALL rendered pixels
    Moondream2   -> 300-800 ms, semantic, ALL rendered pixels + reasoning

Use OCR when you need machine-readable text from rendered content
that UIA can't see (game HUDs, image viewers, PDF previews, video
overlays, Electron apps with shallow UIA trees, screenshots in
chat clients). Use the VLM tier for "explain what's on screen" /
"is this an error dialog?" reasoning.

Implementation
==============

Tesseract is invoked via the ``pytesseract`` Python wrapper, which
shells out to the system ``tesseract`` binary. Both the wrapper and
the binary are OPTIONAL dependencies -- the module is fail-open at
both layers:

* If ``pytesseract`` cannot be imported, every public function
  returns a structured "unavailable" :class:`OCRResult`.
* If the ``tesseract`` binary is missing (or pytesseract raises a
  :class:`TesseractNotFoundError`), the same fail-open contract
  applies.
* If a region capture fails (mss / display unavailable), the call
  returns an "unavailable" result without crashing.

The Tesseract binary location can be configured via the
``ULTRON_TESSERACT_CMD`` environment variable; otherwise pytesseract
uses its default PATH lookup. Setting the path explicitly is the
right move on Windows where the installer drops the binary at
``C:\\Program Files\\Tesseract-OCR\\tesseract.exe`` and PATH is not
always updated.

Region capture is delegated to ultron's existing
:class:`ultron.desktop.capture.ScreenCapture` so the same mss
thread-locality + taint-tracker contract applies.

Safety
======

Cap-2 (read-only screen observation). Captures pay the standard
taint-tracker record so any subsequent outbound tool call carrying
those bytes trips the validator's exfil layer. The returned text
is treated as ephemeral by default (NOT taint-tracked) -- callers
that want OCR'd text to flow into the taint chain should record it
explicitly via the safety taint API.

The OCR call passes Tesseract's ``--psm 6`` (assume a single
uniform block of text) by default, which is the right choice for
most screen-region extraction. Callers can override via the ``psm``
argument when extracting structured content (e.g. ``--psm 11`` for
sparse text on a busy background).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from ultron.utils.logging import get_logger

logger = get_logger("desktop.ocr")


#: Default page-segmentation mode. PSM 6 = "Assume a single uniform
#: block of text", which works well for most app windows, dialogs,
#: and game HUDs where the OCR target is a coherent region.
DEFAULT_PSM: int = 6


#: Default language for OCR. English-only by default; callers can
#: pass other languages (e.g. "eng+spa") when needed.
DEFAULT_LANG: str = "eng"


#: Environment variable that, when set, points pytesseract at a
#: specific tesseract binary. Useful on Windows where the installer
#: doesn't always update PATH.
TESSERACT_CMD_ENV: str = "ULTRON_TESSERACT_CMD"


@dataclass(frozen=True)
class OCRResult:
    """Outcome of one OCR call.

    Attributes:
        success: True iff text was extracted (even an empty result
            from a blank region is success=True). False indicates
            an error (binary missing, pytesseract import failure,
            capture failure, etc.).
        text: extracted text on success, empty string on failure.
        elapsed_ms: wall-clock duration including capture + OCR.
        engine: ``"tesseract"`` on success; ``"unavailable"``
            otherwise. Useful for audit-log differentiation.
        region: ``(left, top, width, height)`` 4-tuple of the region
            that was OCR'd, or ``None`` for a full-screen capture.
        psm: Tesseract page-segmentation mode used.
        lang: Tesseract language used.
        error: structured failure reason when ``success=False``.
    """

    success: bool
    text: str = ""
    elapsed_ms: int = 0
    engine: str = "unavailable"
    region: Optional[tuple[int, int, int, int]] = None
    psm: int = DEFAULT_PSM
    lang: str = DEFAULT_LANG
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# pytesseract lazy import
# ---------------------------------------------------------------------------


_pytesseract_cache: dict = {}


def _import_pytesseract():
    """Lazy import + Windows PATH override.

    Returns the ``pytesseract`` module, or None when unavailable.
    Cached so repeated calls don't pay the import cost.
    """
    if "module" in _pytesseract_cache:
        return _pytesseract_cache["module"]
    try:
        import pytesseract  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.debug("pytesseract unavailable: %s", exc)
        _pytesseract_cache["module"] = None
        return None
    # Windows path override: setting tesseract_cmd lets the wrapper
    # find the binary even when PATH wasn't updated by the installer.
    cmd_env = os.environ.get(TESSERACT_CMD_ENV, "").strip()
    if cmd_env:
        try:
            pytesseract.pytesseract.tesseract_cmd = cmd_env  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.debug("pytesseract tesseract_cmd override failed: %s", exc)
    _pytesseract_cache["module"] = pytesseract
    return pytesseract


def reset_pytesseract_cache_for_testing() -> None:
    """Test hook: clear the lazy-import cache."""
    _pytesseract_cache.clear()


# ---------------------------------------------------------------------------
# PIL conversion helper
# ---------------------------------------------------------------------------


def _png_bytes_to_pil_image(png_bytes: bytes):
    """Decode PNG bytes to a PIL :class:`Image.Image`.

    Returns None on any decode failure. Tesseract accepts a PIL Image
    directly; we go through it rather than writing a temp file so the
    pixels never touch disk.
    """
    if not png_bytes:
        return None
    try:
        from io import BytesIO

        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        logger.debug("PIL import failed: %s", exc)
        return None
    try:
        return Image.open(BytesIO(png_bytes))
    except Exception as exc:  # noqa: BLE001
        logger.debug("PIL decode failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_ocr_available() -> bool:
    """True iff pytesseract is importable AND the tesseract binary is
    reachable.

    Checks both layers so callers can decide upfront whether to use
    OCR or fall straight through to the VLM tier.
    """
    pyt = _import_pytesseract()
    if pyt is None:
        return False
    try:
        # get_tesseract_version raises TesseractNotFoundError when the
        # binary is missing; any other exception is treated as
        # unavailable too.
        pyt.get_tesseract_version()
    except Exception as exc:  # noqa: BLE001
        logger.debug("tesseract binary not reachable: %s", exc)
        return False
    return True


def ocr_image_bytes(
    png_bytes: bytes,
    *,
    psm: int = DEFAULT_PSM,
    lang: str = DEFAULT_LANG,
    region: Optional[tuple[int, int, int, int]] = None,
) -> OCRResult:
    """OCR a PNG-encoded byte buffer directly.

    Use when the caller already has captured bytes in hand (e.g.
    after a screenshot for the VLM tier failed to produce semantic
    output, hand the same bytes to OCR rather than re-capturing).

    Args:
        png_bytes: PNG-encoded image data.
        psm: Tesseract page-segmentation mode (default 6 = uniform
            block of text).
        lang: Tesseract language code (default ``"eng"``).
        region: optional region tuple stored on the result for audit
            traceability; does NOT crop the image. Pre-crop the
            bytes before calling if you need the OCR scope limited.

    Returns:
        :class:`OCRResult`. ``success=True`` with the extracted text
        on success; ``success=False`` with structured error on any
        failure path.
    """
    started = time.perf_counter()
    if not isinstance(png_bytes, (bytes, bytearray)) or not png_bytes:
        return OCRResult(
            success=False, error="png_bytes must be non-empty bytes",
            engine="unavailable", region=region, psm=psm, lang=lang,
        )
    pyt = _import_pytesseract()
    if pyt is None:
        return OCRResult(
            success=False, error="pytesseract unavailable",
            engine="unavailable", region=region, psm=psm, lang=lang,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    image = _png_bytes_to_pil_image(bytes(png_bytes))
    if image is None:
        return OCRResult(
            success=False, error="PIL decode failed",
            engine="unavailable", region=region, psm=psm, lang=lang,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    try:
        text = pyt.image_to_string(
            image, lang=str(lang), config=f"--psm {int(psm)}",
        )
    except Exception as exc:  # noqa: BLE001
        return OCRResult(
            success=False, error=f"tesseract error: {str(exc)[:200]}",
            engine="unavailable", region=region, psm=psm, lang=lang,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    if not isinstance(text, str):
        text = str(text)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return OCRResult(
        success=True,
        text=text.strip(),
        engine="tesseract",
        region=region,
        psm=psm,
        lang=lang,
        elapsed_ms=elapsed_ms,
    )


def ocr_screen_region(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    psm: int = DEFAULT_PSM,
    lang: str = DEFAULT_LANG,
    capture: Optional[object] = None,
) -> OCRResult:
    """OCR an arbitrary screen rectangle.

    Captures via ultron's existing :class:`ScreenCapture` (so the
    taint tracker stamps the bytes as ``screen_context``) then
    delegates to :func:`ocr_image_bytes`.

    Args:
        x, y: top-left coordinates of the region in physical pixels.
        width, height: region dimensions in pixels. Must be > 0.
        psm: Tesseract page-segmentation mode.
        lang: Tesseract language.
        capture: optional :class:`ScreenCapture` instance for tests;
            production calls resolve the module singleton via
            :func:`ultron.desktop.capture.get_screen_capture`.

    Returns:
        :class:`OCRResult`.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('ocr')
    region = (int(x), int(y), int(width), int(height))
    if width <= 0 or height <= 0:
        return OCRResult(
            success=False, error="width and height must be > 0",
            engine="unavailable", region=region, psm=psm, lang=lang,
        )
    cap = capture
    if cap is None:
        try:
            from ultron.desktop.capture import get_screen_capture
            cap = get_screen_capture()
        except Exception as exc:  # noqa: BLE001
            return OCRResult(
                success=False, error=f"capture unavailable: {exc}",
                engine="unavailable", region=region, psm=psm, lang=lang,
            )
    try:
        shot = cap.capture_region(
            x=int(x), y=int(y), width=int(width), height=int(height),
        )
    except Exception as exc:  # noqa: BLE001
        return OCRResult(
            success=False, error=f"capture raised: {str(exc)[:200]}",
            engine="unavailable", region=region, psm=psm, lang=lang,
        )
    if shot is None or not getattr(shot, "image_bytes", None):
        return OCRResult(
            success=False, error="capture returned no image",
            engine="unavailable", region=region, psm=psm, lang=lang,
        )
    result = ocr_image_bytes(
        shot.image_bytes, psm=psm, lang=lang, region=region,
    )
    return result


def ocr_screen_monitor(
    monitor_index: int = 0,
    *,
    psm: int = DEFAULT_PSM,
    lang: str = DEFAULT_LANG,
    capture: Optional[object] = None,
) -> OCRResult:
    """OCR an entire monitor.

    Useful for the "what's on screen?" voice-context flow when UIA
    text is sparse and the VLM is unavailable / too slow.

    Args:
        monitor_index: 0-based monitor index.
        psm: Tesseract page-segmentation mode.
        lang: Tesseract language.
        capture: optional :class:`ScreenCapture` instance for tests.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('ocr')
    cap = capture
    if cap is None:
        try:
            from ultron.desktop.capture import get_screen_capture
            cap = get_screen_capture()
        except Exception as exc:  # noqa: BLE001
            return OCRResult(
                success=False, error=f"capture unavailable: {exc}",
                engine="unavailable", psm=psm, lang=lang,
            )
    try:
        shot = cap.capture_monitor(int(monitor_index))
    except Exception as exc:  # noqa: BLE001
        return OCRResult(
            success=False, error=f"capture raised: {str(exc)[:200]}",
            engine="unavailable", psm=psm, lang=lang,
        )
    if shot is None or not getattr(shot, "image_bytes", None):
        return OCRResult(
            success=False, error="capture returned no image",
            engine="unavailable", psm=psm, lang=lang,
        )
    region = None
    try:
        region = (
            int(getattr(shot, "origin_x", 0)),
            int(getattr(shot, "origin_y", 0)),
            int(getattr(shot, "width", 0)),
            int(getattr(shot, "height", 0)),
        )
    except Exception:  # noqa: BLE001
        region = None
    return ocr_image_bytes(
        shot.image_bytes, psm=psm, lang=lang, region=region,
    )


__all__ = [
    "DEFAULT_PSM",
    "DEFAULT_LANG",
    "TESSERACT_CMD_ENV",
    "OCRResult",
    "is_ocr_available",
    "ocr_image_bytes",
    "ocr_screen_region",
    "ocr_screen_monitor",
    "reset_pytesseract_cache_for_testing",
]
