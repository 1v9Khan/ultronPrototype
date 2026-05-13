"""Category R -- Sensors and input.

R1 -- microphone access outside the declared voice pipeline (see D16).
R2 -- webcam access without explicit user-stated intent.
R3 -- screen capture (handled by Cap-1 carve-out + A13/J8 cross-ref).
R4 -- keylogging via low-level hooks (see D15).
R5 -- clipboard reads (see D13).
R6 -- clipboard writes (address substitution -- see J7).
R7 -- standard clipboard read/write with explicit user intent -- LOG_ONLY.

Most R-rules cross-reference D / J rules; this module exists mainly so
the validator audit log records the canonical R id when these fire
under the sensor-control category framing.
"""

from __future__ import annotations

from ultron.safety.rules.base import CommandPatternRule, Rule


def build_category_r_rules() -> list[Rule]:
    """Factory for Category R rules.

    Many R-IDs overlap with D-IDs (D13 clipboard read, D15 keylogging,
    D16 mic/webcam). We do NOT re-register those here -- the dispatcher
    audit log already records D-prefix ids. R is reserved for sensor-
    specific concerns not covered by D, currently just R2 (webcam
    write-shape) and explicit screen-capture rate limiting (Phase 4
    Cap-1 carve-out).
    """
    return [
        # R2: webcam access. (D16 covers this too -- here we add a
        # tighter pattern that catches Windows-specific MediaFoundation.)
        CommandPatternRule(
            rule_id="R2",
            description="webcam access without explicit user-stated intent",
            category="R",
            patterns=[
                # Windows MediaFoundation source reader
                r"\bIMFSourceReader\b.*\bVidcap\b",
                r"\bMFEnumDeviceSources\b",
                # PowerShell .NET wrappers
                r"\bSystem\.Drawing\.Imaging\.ImageFormat\.Capture\b",
                # OpenCV / TouchDesigner etc
                r"\bcv2\.VideoCapture\(\s*[0-9]\s*\)",
            ],
        ),
    ]
