"""Multi-monitor screen capture via ``mss``.

mss is ~5 ms per monitor on this hardware and ships as a single
pure-Python wheel. Its native handle is NOT thread-safe in a single
instance, so :class:`ScreenCapture` keeps one ``mss.mss()`` per
thread via thread-local storage.

Every successful capture is recorded in the safety taint tracker
(capability=``screen_context``) by default so the validator's
exfil-detection layer can match if those exact bytes show up as
an outbound tool argument later. Recording is sub-millisecond
(SHA-256 over the PNG bytes); set ``record_taint=False`` to disable
for tests.

The :class:`Screenshot` dataclass holds PNG-encoded bytes. PNG is
the right format here: lossless (so OCR / VLM see exactly what the
user sees), well-compressed for typical desktop content, and the
universal interchange format the moondream2 VLM accepts.
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import mss
from PIL import Image

from ultron.desktop.monitors import Monitor, enumerate_monitors
from ultron.utils.logging import get_logger

logger = get_logger("desktop.capture")


class ScreenCaptureError(RuntimeError):
    """Raised when a capture call cannot be satisfied (caught + fail-open by callers)."""


@dataclass(frozen=True)
class Screenshot:
    """One captured frame.

    Attributes:
        image_bytes: PNG-encoded image data. ``None`` once the bytes
            have been discarded (post-VLM analysis under the
            analyze-and-discard pattern -- see :meth:`without_bytes` and
            :func:`ultron.desktop.screen_context.build_screen_context`'s
            ``discard_image_after_analysis`` flag).
        monitor_index: source monitor index, or None for arbitrary regions.
        width: pixel width.
        height: pixel height.
        timestamp: ``time.time()`` at capture moment.
        origin_x: leftmost pixel coordinate of the capture in virtual-screen space.
        origin_y: topmost pixel coordinate of the capture in virtual-screen space.
        bytes_discarded: True iff the original bytes were intentionally
            dropped after analysis. Lets callers distinguish "no
            capture was made" (image_bytes=None, bytes_discarded=False)
            from "capture was made + analysed, then discarded for
            storage efficiency / privacy" (image_bytes=None,
            bytes_discarded=True). Defaults False.
    """

    image_bytes: Optional[bytes]
    monitor_index: Optional[int]
    width: int
    height: int
    timestamp: float
    origin_x: int
    origin_y: int
    bytes_discarded: bool = False

    def without_bytes(self) -> "Screenshot":
        """Return a copy with ``image_bytes`` cleared and
        ``bytes_discarded=True``.

        Used by the screen-context layer post-VLM analysis so the
        cache only ever retains the textual description, not the raw
        pixels. Idempotent on already-discarded screenshots.
        """
        if self.image_bytes is None and self.bytes_discarded:
            return self
        return Screenshot(
            image_bytes=None,
            monitor_index=self.monitor_index,
            width=self.width,
            height=self.height,
            timestamp=self.timestamp,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            bytes_discarded=True,
        )


def _bgra_to_png_bytes(bgra: bytes, width: int, height: int) -> bytes:
    """Convert mss's BGRA raw buffer to PNG-encoded bytes."""
    img = Image.frombytes("RGB", (width, height), bgra, "raw", "BGRX")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _record_taint_safe(image_bytes: bytes) -> None:
    """Record capture bytes in the safety taint tracker. Fail-open."""
    if not image_bytes:
        return
    try:
        from ultron.safety.taint import get_taint_tracker

        get_taint_tracker().record(data=image_bytes, capability="screen_context")
    except Exception as e:  # noqa: BLE001 -- safety side must never break capture
        logger.debug("taint record skipped: %s", e)


class ScreenCapture:
    """Per-process screen capture facade.

    One instance per orchestrator. Thread-safe via a thread-local mss
    handle (mss's underlying GDI / DXGI objects are not safe to share
    across threads).
    """

    def __init__(self, *, record_taint: bool = True) -> None:
        self._tls = threading.local()
        self._record_taint = bool(record_taint)
        self._closed = False

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has been called."""
        return self._closed

    def _sct(self) -> "mss.MSS":
        if self._closed:
            raise ScreenCaptureError("ScreenCapture is closed")
        sct = getattr(self._tls, "sct", None)
        if sct is None:
            sct = mss.MSS()
            self._tls.sct = sct
        return sct

    def capture_monitor(self, monitor: Union[Monitor, int]) -> Optional[Screenshot]:
        """Capture one monitor by :class:`Monitor` instance or index.

        Returns None on any failure (missing monitor, mss error). Caller
        treats None as "couldn't see the screen right now".
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('screenshot')
        if isinstance(monitor, int):
            mons = enumerate_monitors()
            if not (0 <= monitor < len(mons)):
                logger.warning("capture_monitor: index %d out of range", monitor)
                return None
            mon = mons[monitor]
        else:
            mon = monitor

        return self._capture_region(
            x=mon.x,
            y=mon.y,
            width=mon.width,
            height=mon.height,
            monitor_index=mon.index,
        )

    def capture_all_monitors(self) -> list[Screenshot]:
        """Capture every connected monitor in index order."""
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('screenshot')
        results: list[Screenshot] = []
        for mon in enumerate_monitors():
            shot = self.capture_monitor(mon)
            if shot is not None:
                results.append(shot)
        return results

    def capture_region(
        self,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Optional[Screenshot]:
        """Capture an arbitrary rectangle in virtual-screen coordinates."""
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('screenshot')
        if width <= 0 or height <= 0:
            return None
        return self._capture_region(
            x=x, y=y, width=width, height=height, monitor_index=None,
        )

    def _capture_region(
        self,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        monitor_index: Optional[int],
    ) -> Optional[Screenshot]:
        try:
            sct = self._sct()
            grab = sct.grab({
                "left": x,
                "top": y,
                "width": width,
                "height": height,
            })
        except Exception as e:  # noqa: BLE001 -- mss raises ScreenShotError and others
            logger.warning(
                "screen capture failed at (%d,%d,%d,%d): %s",
                x, y, width, height, e,
            )
            return None

        try:
            png_bytes = _bgra_to_png_bytes(
                bytes(grab.raw), grab.size[0], grab.size[1],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("PNG encode failed: %s", e)
            return None

        if self._record_taint:
            _record_taint_safe(png_bytes)

        return Screenshot(
            image_bytes=png_bytes,
            monitor_index=monitor_index,
            width=grab.size[0],
            height=grab.size[1],
            timestamp=time.time(),
            origin_x=x,
            origin_y=y,
        )

    def close(self) -> None:
        """Release the mss handle on every thread that touched this instance.

        Idempotent. Subsequent capture calls raise :class:`ScreenCaptureError`.
        """
        self._closed = True
        sct = getattr(self._tls, "sct", None)
        if sct is not None:
            try:
                sct.close()
            except Exception:  # noqa: BLE001
                pass
            self._tls.sct = None


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_capture_singleton: Optional[ScreenCapture] = None
_capture_lock = threading.Lock()


def get_screen_capture() -> ScreenCapture:
    """Module-level singleton accessor.

    The orchestrator constructs the production :class:`ScreenCapture`
    on init and pushes it via :func:`set_screen_capture`. Callers that
    arrive before the orchestrator (tests, scripts) get a default
    instance with taint recording enabled.
    """
    global _capture_singleton
    if _capture_singleton is None:
        with _capture_lock:
            if _capture_singleton is None:
                _capture_singleton = ScreenCapture()
    return _capture_singleton


def set_screen_capture(capture: Optional[ScreenCapture]) -> None:
    """Test / orchestrator hook -- swap the singleton."""
    global _capture_singleton
    with _capture_lock:
        _capture_singleton = capture


# ---------------------------------------------------------------------------
# Catalog 09 T6: image template matching (find a UI element by saved image)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateMatch:
    """One template-match hit on the screen.

    Attributes:
        left: left edge of the matched rectangle in physical pixels.
        top: top edge.
        width: matched-rect width in pixels (== template width).
        height: matched-rect height in pixels (== template height).
        center_x: x coordinate of the rect centre (precomputed for
            direct routing to :meth:`InputController.click`).
        center_y: y coordinate of the rect centre.
        confidence: confidence threshold that was applied (callers
            can store this alongside the match for audit).
    """

    left: int
    top: int
    width: int
    height: int
    center_x: int
    center_y: int
    confidence: float


_DEFAULT_TEMPLATE_CONFIDENCE: float = 0.8


def find_image_on_screen(
    template_path: str,
    *,
    confidence: float = _DEFAULT_TEMPLATE_CONFIDENCE,
    region: Optional[tuple[int, int, int, int]] = None,
) -> Optional[TemplateMatch]:
    """Find a saved template image on screen via OpenCV template matching.

    Catalog 09 T6 (YELLOW): fills the middle tier between fast UIA
    (semantic, but unavailable for canvas-rendered apps and game HUDs)
    and slow VLM (semantic, but 300-800 ms + GPU). Template matching
    is ~10-100 ms per search on typical hardware with a ``region``
    constraint, and works on every visible pixel-rendered surface --
    including older Win32 apps without UIA roles, game inventory
    icons, browser content not exposed via accessibility, and any
    UI element whose appearance is stable across sessions.

    Routing through the gated :class:`InputController.click` for the
    returned centre coordinate keeps the whole click safety stack
    (foreground security, validator, click_preview, rate limit)
    intact. The template-matching step itself is read-only and
    therefore Cap-2.

    Args:
        template_path: filesystem path to the template image (PNG /
            JPEG). The path is canonicalised via
            :class:`ultron.safety.path_resolver.PathResolver` -- raw
            paths with bidi-override / percent-escape evasion patterns
            are rejected. This protects against attacker-controlled
            template paths that could match a spoofed UI element.
        confidence: 0.0-1.0 OpenCV match confidence. Default 0.8
            (matches the upstream plugin). Higher values reduce false
            positives at the cost of false negatives on anti-aliased
            edges and gamma-shifted backgrounds.
        region: optional ``(left, top, width, height)`` 4-tuple
            restricting the search to a sub-rectangle of the screen.
            Cuts search time by 5-20x for small known regions.
            ``None`` (default) scans the full virtual screen.

    Returns:
        :class:`TemplateMatch` on a successful match within the
        confidence threshold. ``None`` when:

        * The template path canonicalisation fails (evasion pattern,
          path traversal, broken symlink chain).
        * pyautogui is unavailable.
        * opencv-python is not installed (pyautogui.locateOnScreen
          requires opencv for confidence-based matching).
        * No match meeting the threshold was found.
        * pyautogui raised any other exception (display unavailable,
          region out of bounds, etc.).

    Fail-open at every layer: the return shape is always
    ``Optional[TemplateMatch]`` and the orchestrator can simply branch
    on ``None`` instead of catching exceptions.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('find_image_on_screen')
    if not isinstance(template_path, str) or not template_path:
        return None
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return None
    if not (0.0 < conf <= 1.0):
        return None
    if region is not None:
        try:
            region_tuple: Optional[tuple[int, int, int, int]] = (
                int(region[0]), int(region[1]),
                int(region[2]), int(region[3]),
            )
        except (TypeError, IndexError, ValueError):
            return None
        if region_tuple[2] <= 0 or region_tuple[3] <= 0:
            return None
    else:
        region_tuple = None

    try:
        from ultron.safety.path_resolver import get_path_resolver

        resolved = get_path_resolver().safe_realpath(template_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "find_image_on_screen path resolver failed: %s", exc,
        )
        return None
    if resolved is None:
        logger.debug(
            "find_image_on_screen rejected template path: %s",
            template_path,
        )
        return None

    template_resolved = str(resolved)

    try:
        import pyautogui  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.debug("find_image_on_screen pyautogui unavailable: %s", exc)
        return None

    try:
        box = pyautogui.locateOnScreen(
            template_resolved,
            confidence=conf,
            region=region_tuple,
        )
    except Exception as exc:  # noqa: BLE001
        # pyautogui.locateOnScreen raises ImageNotFoundException when
        # no match is found AND PyAutoGUIException / ImportError when
        # opencv-python is missing. All are mapped to a single None
        # contract so callers can branch on the return value rather
        # than catching exceptions.
        logger.debug(
            "find_image_on_screen locateOnScreen exception: %s", exc,
        )
        return None

    if box is None:
        return None

    try:
        left = int(box[0])
        top = int(box[1])
        width = int(box[2])
        height = int(box[3])
    except (TypeError, IndexError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None

    center_x = left + width // 2
    center_y = top + height // 2

    return TemplateMatch(
        left=left,
        top=top,
        width=width,
        height=height,
        center_x=center_x,
        center_y=center_y,
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# Catalog 09 T2: pixel-color probe (single coordinate, no screenshot)
# ---------------------------------------------------------------------------


def get_pixel_color(x: int, y: int) -> Optional[tuple[int, int, int]]:
    """Return the on-screen RGB colour at ``(x, y)`` without a full capture.

    Catalog 09 T2 (GREEN, read-only observation). Wraps
    :func:`pyautogui.pixel` which internally calls Win32 ``GetPixel``
    on Windows -- a single GDI call, no PNG encoding, no taint tracker
    record (the RGB tuple is ephemeral data, not durable bytes).

    Use cases:

    * Cheap game-state polling (HUD pixel, health bar tip) where a VLM
      capture + analyse round-trip is overkill.
    * Loading-spinner disappearance detection (poll the spinner pixel
      until it matches the background colour).
    * Pixel-based "is this dialog gone yet?" after a dismiss attempt
      when UIA tree presence is not a reliable signal.
    * Status-LED indicators in legacy Win32 dashboards that expose no
      UIA accessibility role.

    Fail-open: any exception (out-of-bounds coordinate, mss / pyautogui
    error, missing display) returns ``None`` rather than raising so the
    polling-loop caller can simply continue. The upstream plugin lets
    the underlying pyautogui exception propagate; ultron's contract is
    "observation primitives never crash the orchestrator".

    Returns:
        ``(r, g, b)`` 3-tuple of 0-255 integers when successful, or
        ``None`` on failure.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('get_pixel_color')
    try:
        # Local import to avoid pulling pyautogui at module load (mss
        # is the standard capture path; pyautogui is only needed here).
        import pyautogui  # type: ignore[import]

        rgb = pyautogui.pixel(int(x), int(y))
    except Exception as exc:  # noqa: BLE001
        logger.debug("get_pixel_color(%d, %d) failed: %s", x, y, exc)
        return None
    if rgb is None:
        return None
    try:
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    except (TypeError, IndexError, ValueError):
        return None
