"""Visual crosshair preview before desktop clicks.

Direct port of SWE-Agent's
``tools/web_browser/lib/browser_manager.py:CROSSHAIR_JS`` (MIT,
Yang et al. 2024) adapted from browser-JS injection to native
desktop screenshot annotation.

The pattern: BEFORE firing a click at ``(x, y)``, draw a red
crosshair on a screenshot at those coordinates, run the resulting
annotated image through the VLM, ask "is this where you actually
want to click", and only proceed if the VLM agrees. Catches the
common "I clicked at (450, 200) but I actually wanted (450, 250)"
class of error before the click reaches the OS.

Per the user's confirmation-gate choice: the first click of a
session ALWAYS confirms; subsequent clicks within
:data:`AUTO_PASS_RADIUS_PX` of a recently-confirmed region
auto-pass. So a session that clicks several times in the same UI
panel pays the VLM round-trip cost only once.

The crosshair drawing is pure Pillow (no PortAudio / no OS click)
so this module is import-safe even when desktop automation isn't
wired in.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Crosshair size in pixels (matches SWE-Agent's `size = 20`).
DEFAULT_CROSSHAIR_SIZE: int = 20

#: Crosshair thickness in pixels (matches SWE-Agent's `thickness = 3`).
DEFAULT_CROSSHAIR_THICKNESS: int = 3

#: Crosshair colour (RGB) -- bright red, matches the JS injection.
DEFAULT_CROSSHAIR_COLOR: tuple[int, int, int] = (255, 0, 0)

#: Recently-confirmed click radius. Subsequent clicks within this
#: many pixels of a confirmed coordinate auto-pass without the
#: VLM round-trip.
AUTO_PASS_RADIUS_PX: int = 100

#: Number of historical confirmations to remember per session.
DEFAULT_HISTORY_DEPTH: int = 20


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class PreviewDecision(Enum):
    """Result of :func:`preview_click`."""

    ALLOW = "allow"
    BLOCK = "block"
    AUTO_PASS = "auto_pass"
    DEGRADED = "degraded"  # VLM unavailable -- allow + log


@dataclass(frozen=True)
class ConfirmedClick:
    """Record of a click that the VLM confirmed."""

    x: int
    y: int
    confirmed_at: float
    description: str = ""


@dataclass
class PreviewResult:
    """Output of :func:`preview_click`."""

    decision: PreviewDecision
    confidence: float = 1.0
    vlm_response: str = ""
    annotated_png: Optional[bytes] = None
    reason: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Crosshair drawing
# ---------------------------------------------------------------------------


def draw_crosshair_on_image(
    image_bytes: bytes,
    *,
    x: int,
    y: int,
    size: int = DEFAULT_CROSSHAIR_SIZE,
    thickness: int = DEFAULT_CROSSHAIR_THICKNESS,
    color: tuple[int, int, int] = DEFAULT_CROSSHAIR_COLOR,
) -> bytes:
    """Return ``image_bytes`` with a red crosshair drawn at ``(x, y)``.

    Pure-PIL implementation -- accepts and returns PNG-encoded bytes.
    Identical geometry to SWE-Agent's CROSSHAIR_JS: two perpendicular
    bars of length ``size`` and thickness ``thickness`` centred on
    ``(x, y)``.

    Raises :class:`ImportError` when Pillow isn't installed; callers
    should catch and degrade.
    """
    if not image_bytes:
        raise ValueError("image_bytes must be non-empty")
    from PIL import Image, ImageDraw

    with Image.open(io.BytesIO(image_bytes)) as im:
        # Convert to RGB for stable colour rendering even on RGBA inputs.
        im = im.convert("RGB").copy()
        draw = ImageDraw.Draw(im)
        # Horizontal bar.
        h_left = x - size // 2
        h_top = y - thickness // 2
        h_right = h_left + size
        h_bottom = h_top + thickness
        draw.rectangle([h_left, h_top, h_right, h_bottom], fill=color)
        # Vertical bar.
        v_left = x - thickness // 2
        v_top = y - size // 2
        v_right = v_left + thickness
        v_bottom = v_top + size
        draw.rectangle([v_left, v_top, v_right, v_bottom], fill=color)
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()


# ---------------------------------------------------------------------------
# Confirmation history (per-session)
# ---------------------------------------------------------------------------


class ConfirmationHistory:
    """Bounded recent-click history for the auto-pass tier.

    Not thread-safe -- callers serialise per-session click attempts.
    """

    def __init__(self, *, max_entries: int = DEFAULT_HISTORY_DEPTH) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = int(max_entries)
        self._entries: list[ConfirmedClick] = []

    def record(self, x: int, y: int, *, description: str = "") -> None:
        """Append a confirmation. Oldest entry dropped when at cap."""
        self._entries.append(
            ConfirmedClick(
                x=int(x), y=int(y), confirmed_at=time.time(), description=description
            )
        )
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

    def near(self, x: int, y: int, *, radius: int = AUTO_PASS_RADIUS_PX) -> Optional[ConfirmedClick]:
        """Return the most-recent confirmation within ``radius`` of ``(x, y)``,
        or ``None``."""
        if radius < 0:
            return None
        for entry in reversed(self._entries):
            dx = entry.x - x
            dy = entry.y - y
            if dx * dx + dy * dy <= radius * radius:
                return entry
        return None

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[ConfirmedClick]:
        return list(self._entries)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preview_click(
    *,
    x: int,
    y: int,
    capture_screen: Callable[[], bytes],
    vlm_describe: Optional[Callable[[bytes, str], str]],
    history: ConfirmationHistory,
    intent_description: str = "",
    auto_pass_radius: int = AUTO_PASS_RADIUS_PX,
    crosshair_size: int = DEFAULT_CROSSHAIR_SIZE,
    crosshair_thickness: int = DEFAULT_CROSSHAIR_THICKNESS,
    crosshair_color: tuple[int, int, int] = DEFAULT_CROSSHAIR_COLOR,
    require_confirmation_keyword: str = "yes",
) -> PreviewResult:
    """Decide whether the click at ``(x, y)`` should fire.

    Decision flow:

    1. If ``(x, y)`` is within ``auto_pass_radius`` pixels of a
       recently-confirmed click in ``history``: return AUTO_PASS.
    2. Otherwise capture a screenshot (via ``capture_screen``),
       annotate with a crosshair at ``(x, y)``, and pass to
       ``vlm_describe(annotated_png, prompt)``. The prompt asks
       the VLM whether the click target matches ``intent_description``.
    3. If the VLM response contains ``require_confirmation_keyword``
       case-insensitive: record + return ALLOW.
    4. Otherwise return BLOCK.

    Degraded path: if ``vlm_describe`` is ``None`` OR raises, return
    DEGRADED + ``decision=ALLOW`` so the click still fires but the
    operator's audit log records the missing confirmation. The
    desktop layer then chooses whether DEGRADED counts as ALLOW or
    BLOCK depending on its safety posture.
    """
    # 1. Auto-pass tier.
    near = history.near(x, y, radius=auto_pass_radius)
    if near is not None:
        return PreviewResult(
            decision=PreviewDecision.AUTO_PASS,
            confidence=1.0,
            reason=(
                f"auto-pass: within {auto_pass_radius}px of a recently-"
                f"confirmed click at ({near.x}, {near.y}) "
                f"{time.time() - near.confirmed_at:.1f}s ago"
            ),
            extra={"near_confirmed": True},
        )

    # 2. Capture screen + annotate.
    try:
        screen_bytes = capture_screen()
    except Exception as exc:
        logger.warning("preview_click: capture_screen raised: %s", exc)
        return PreviewResult(
            decision=PreviewDecision.DEGRADED,
            confidence=0.0,
            reason=f"capture_screen raised: {exc}",
        )
    if not screen_bytes:
        return PreviewResult(
            decision=PreviewDecision.DEGRADED,
            confidence=0.0,
            reason="capture_screen returned empty bytes",
        )
    try:
        annotated = draw_crosshair_on_image(
            screen_bytes,
            x=x,
            y=y,
            size=crosshair_size,
            thickness=crosshair_thickness,
            color=crosshair_color,
        )
    except Exception as exc:
        logger.warning("preview_click: crosshair draw raised: %s", exc)
        return PreviewResult(
            decision=PreviewDecision.DEGRADED,
            confidence=0.0,
            reason=f"crosshair draw raised: {exc}",
        )

    # 3. VLM round-trip.
    if vlm_describe is None:
        return PreviewResult(
            decision=PreviewDecision.DEGRADED,
            confidence=0.0,
            reason="vlm_describe callable is None",
            annotated_png=annotated,
        )
    prompt = (
        "A red crosshair is drawn on the screen at the proposed click "
        "target. The intent was: " + (intent_description or "(unspecified)") + ". "
        "Is the crosshair on the correct UI element for that intent? "
        "Answer 'yes' if it is, or describe the actual element under the "
        "crosshair if not."
    )
    try:
        response = vlm_describe(annotated, prompt) or ""
    except Exception as exc:
        logger.warning("preview_click: vlm_describe raised: %s", exc)
        return PreviewResult(
            decision=PreviewDecision.DEGRADED,
            confidence=0.0,
            reason=f"vlm_describe raised: {exc}",
            annotated_png=annotated,
        )

    response_lower = response.lower()
    keyword_lower = require_confirmation_keyword.lower()
    if keyword_lower and keyword_lower in response_lower:
        history.record(x, y, description=intent_description)
        return PreviewResult(
            decision=PreviewDecision.ALLOW,
            confidence=0.9,
            vlm_response=response,
            annotated_png=annotated,
            reason="VLM confirmed the click target",
        )
    return PreviewResult(
        decision=PreviewDecision.BLOCK,
        confidence=0.5,
        vlm_response=response,
        annotated_png=annotated,
        reason="VLM did not confirm the click target",
    )


__all__ = [
    "AUTO_PASS_RADIUS_PX",
    "ConfirmationHistory",
    "ConfirmedClick",
    "DEFAULT_CROSSHAIR_COLOR",
    "DEFAULT_CROSSHAIR_SIZE",
    "DEFAULT_CROSSHAIR_THICKNESS",
    "DEFAULT_HISTORY_DEPTH",
    "PreviewDecision",
    "PreviewResult",
    "draw_crosshair_on_image",
    "preview_click",
]
