"""XTTS v2 + v3 Kenning filter TTS engine (drop-in replacement for Piper+RVC).

Architecture:

    main venv (this module)            isolated XTTS venv
    --------------------               -------------------
    XttsV3Speech                <-->   xtts_server.py (FastAPI)
        speak_stream(...)              POST /synthesize
        _synthesize(text) ----HTTP---> XTTS streaming inference
                              <-PCM--  v3 filter (this venv)

The XTTS server runs as a subprocess in its own Python venv (the
``.venv-xtts`` next to the audio prep). HTTP keeps the venvs decoupled
because Coqui TTS's deps (transformers 4.x pinned, hydra 1.3, omegaconf
2.3) conflict with what the main Kenning venv needs (older omegaconf
that fairseq 0.12.2 wants for the legacy RVC path).

Latency:
- XTTS streaming TTFT (model only): ~234 ms (benchmarked 2026-05-10)
- Through HTTP: ~375 ms TTFB (60 ms of asyncio + threadpool overhead)
- v3 filter at runtime: ~10-30 ms per sentence
- Composite first-audio-byte: ~400 ms

This is competitive with the legacy Piper+RVC path (~313 ms TTS synth
median) at much higher voice quality.
"""

from __future__ import annotations

import io
import json
import logging
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Callable, ClassVar, Iterable, NamedTuple, Optional, Tuple

import numpy as np
import sounddevice as sd

from config import settings
from kenning.audio.devices import describe_device, resolve_device
from kenning.tts.precomputed_ack import PrecomputedAckClipCache
from kenning.tts.kenning_filter import apply_filter as apply_kenning_filter
from kenning.utils.logging import get_logger

logger = get_logger("tts.xtts_v3")

# Re-export the same Clip / ClipItem contract that the legacy Piper
# pipeline uses. The orchestrator's playback path consumes ClipItem
# tuples, so we honour that contract verbatim.
Clip = Tuple[np.ndarray, int]


class ClipItem(NamedTuple):
    audio: np.ndarray
    sample_rate: int
    is_known_last: bool = False


# Same generous timeout as the Piper+RVC path's playback queue (matches
# kenning.tts.speech._QUEUE_GET_TIMEOUT_SECONDS so downstream playback
# behaviour is consistent).
_QUEUE_GET_TIMEOUT_SECONDS = 60.0

# How long to wait for the XTTS server's /healthz to come up. Cold
# loads hit ~25 s for model + warmup; we add headroom for slower disks
# and first-run model downloads.
_SERVER_STARTUP_TIMEOUT_S = 180.0
_SERVER_HEALTHZ_POLL_INTERVAL_S = 0.5


class XttsServerStartError(RuntimeError):
    """Raised when the XTTS server subprocess can't be started."""


class XttsSynthError(RuntimeError):
    """Raised when a synthesis HTTP call fails (caller decides whether
    to fall back to silent clip vs propagate)."""


def trim_phantom_tail(
    audio_f32: np.ndarray,
    sample_rate: int,
    *,
    silence_threshold: float = 0.005,
    max_event_ms: float = 200.0,
    min_lead_silence_ms: float = 150.0,
    trailing_grace_ms: float = 80.0,
    window_ms: float = 20.0,
    min_clip_duration_ms: float = 800.0,
) -> Tuple[np.ndarray, bool]:
    """Detect and trim an XTTS phantom-token tail.

    XTTS-v2's GPT duration head sometimes emits a fragmentary syllable
    after the stop-token, producing a short isolated audio event in
    the otherwise-silent tail of the synthesised clip. This function
    detects that specific signature and trims everything after the
    last sustained-speech region.

    Pattern detected (walking the RMS envelope from end backwards):

        ...sustained_speech...silence(>= min_lead_silence_ms)...
        short_event(<max_event_ms)...silence_to_end

    Trimming preserves the sustained-speech region plus a
    ``trailing_grace_ms`` cushion so natural speech-end decay isn't
    cut off. Returns ``(possibly-trimmed audio, True/False)`` where
    the bool indicates whether a phantom was detected. When no phantom
    pattern is present the audio is returned unchanged.

    Pure function -- no config import, no logger. Inputs are float32
    in [-1, 1] (typical XTTS post-scaling) but the function operates
    on the raw amplitude so other ranges work too. Safe to call on
    very short clips: anything shorter than two analysis windows is
    returned unchanged.

    Clips shorter than ``min_clip_duration_ms`` are returned unchanged.
    Real phantom syllables only show up at the end of multi-sentence
    responses; a single short word like ``"Right."`` lasts ~400-700 ms
    and the algorithm can misclassify its stop-consonant release as a
    phantom when XTTS lengthens the pre-stop closure.

    Args:
        audio_f32: 1-D mono audio. Other shapes are flattened.
        sample_rate: Hz.
        silence_threshold: RMS threshold below which a window counts
            as silence.
        max_event_ms: trailing audio events shorter than this are
            phantom candidates; longer events are legitimate.
        min_lead_silence_ms: required silent gap between the
            sustained-speech region and the phantom candidate.
        trailing_grace_ms: amount of audio preserved after the last
            sustained-speech window (to keep natural decay).
        window_ms: analysis window size.
        min_clip_duration_ms: clips shorter than this skip the trim
            entirely. Guards against mis-trimming stop-consonant
            releases on single short words.

    Returns:
        ``(audio, trimmed)`` -- ``audio`` is the (possibly shorter)
        clip; ``trimmed`` is True iff a phantom was detected and a
        trim occurred.
    """
    if audio_f32.ndim != 1:
        audio_f32 = audio_f32.reshape(-1)
    n = audio_f32.shape[0]
    if n == 0:
        return audio_f32, False

    if sample_rate > 0 and (n / sample_rate) * 1000.0 < min_clip_duration_ms:
        return audio_f32, False

    win = max(1, int(sample_rate * window_ms / 1000.0))
    n_win = n // win
    if n_win < 4:
        # Too short to reliably detect a phantom pattern.
        return audio_f32, False

    trimmed_buf = audio_f32[: n_win * win].reshape(n_win, win)
    # float64 in the RMS reduction to avoid catastrophic cancellation
    # on quiet windows; coerce back to float32 result.
    rms = np.sqrt(np.mean(trimmed_buf.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    speech_mask = rms >= silence_threshold

    if not speech_mask.any():
        return audio_f32, False

    speech_indices = np.where(speech_mask)[0]
    last_idx = int(speech_indices[-1])
    if last_idx == 0:
        return audio_f32, False

    # Find the trailing event (contiguous speech windows ending at
    # last_idx).
    trailing_start = last_idx
    while trailing_start > 0 and speech_mask[trailing_start - 1]:
        trailing_start -= 1
    trailing_event_windows = last_idx - trailing_start + 1
    trailing_event_ms = trailing_event_windows * window_ms

    # If the trailing event is itself long, it's legitimate end-of-
    # sentence audio -- nothing to trim.
    if trailing_event_ms > max_event_ms:
        return audio_f32, False

    # Find the previous sustained-speech region's end.
    prior_indices = np.where(speech_mask[:trailing_start])[0]
    if prior_indices.size == 0:
        # Only the (short) trailing event was detected as speech. Not
        # a phantom -- could be a clip that contains only a brief
        # word. Leave alone.
        return audio_f32, False

    prior_end = int(prior_indices[-1])
    gap_windows = trailing_start - prior_end - 1
    gap_ms = gap_windows * window_ms

    if gap_ms < min_lead_silence_ms:
        # Not enough silent gap -- this is probably the natural pause
        # between two words inside a sentence, not a phantom tail.
        return audio_f32, False

    # Phantom signature matched. Trim to the end of the prior speech
    # region plus the trailing grace cushion.
    grace_windows = max(1, int(trailing_grace_ms / window_ms))
    cut_window = prior_end + 1 + grace_windows
    cut_samples = min(cut_window * win, n)
    if cut_samples <= 0 or cut_samples >= n:
        # Edge case: grace would extend past the buffer end. Cut
        # exactly at the prior region's end + minimal grace.
        cut_samples = min((prior_end + 1) * win, n)
        if cut_samples <= 0:
            return audio_f32, False
    return audio_f32[:cut_samples], True


# ----------------------------------------------------------------------
# Text normalisation for XTTS
# ----------------------------------------------------------------------
#
# XTTS-v2 mispronounces certain text patterns even with a clean
# reference voice: time formats with colons + "a.m./p.m." come out as
# garbled letter strings; Windows paths with backslashes pin the GPU at
# 100 %; bare unit suffixes attached to numbers are read letter-by-
# letter ("72 degrees F" sounds OK but "72°F" garbles); currency
# symbols are skipped entirely; common Latin abbreviations and title
# abbreviations are inconsistent. The normaliser rewrites these into
# spoken-friendly forms before the server inference call. Each rewrite
# is conservative -- anything that doesn't match a pattern passes
# through unchanged. URLs and email addresses are deliberately
# preserved (XTTS reads them naturally and aggressive rewriting
# would mangle them).

# ----- Windows paths (must run FIRST so the drive-letter colon
# isn't misread as a time pattern). ----------------------------------

# ``C:\foo\bar\baz.ext`` -> ``baz.ext``. Excludes chars Windows
# itself rejects in filenames + whitespace. Requires at least one
# backslash so bare ``C:`` (e.g., "Drive C: is full") passes through.
_WIN_PATH_RE = re.compile(
    r"\b[A-Za-z]:\\(?:[^\s\\/:*?\"<>|]+\\)*([^\s\\/:*?\"<>|]+)",
)

# ----- Times (run before bare unit patterns so the colon is gone
# before the standalone-AM/PM pass). ---------------------------------

# H:MM or HH:MM optionally followed by a.m./p.m./am/pm.
_TIME_AMPM_RE = re.compile(
    r"\b(\d{1,2}):(\d{2})\s*(a\.?\s*m\.?|p\.?\s*m\.?)\b",
    re.IGNORECASE,
)

# 24-hour HH:MM standalone. Negative lookbehind/lookahead so we don't
# eat the inner colon of an already-handled AM/PM time or grab a digit
# that's part of something larger (ratio, date "2026:01" etc.).
_TIME_24H_RE = re.compile(
    r"(?<![:\d])(\d{1,2}):(\d{2})(?![:\d])"
)

# Standalone a.m./p.m. attached to a number ("10 a.m. sharp" or
# "8 pm tonight"). The leading digit anchor prevents "I am" being
# misread as "I A M" and "P.M." in proper names from being mangled.
_AMPM_STANDALONE_RE = re.compile(
    r"\b(\d{1,2})\s+(a\.?\s*m\.?|p\.?\s*m\.?)\b",
    re.IGNORECASE,
)

# ----- Temperatures. ------------------------------------------------

# ``72°F`` / ``72 °F`` / ``72° F`` (and Celsius variants).
_TEMP_F_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°\s*F\b",
    re.IGNORECASE,
)
_TEMP_C_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°\s*C\b",
    re.IGNORECASE,
)
# ``45°`` (no F/C suffix). Run AFTER F/C so they don't lose their
# suffix. Negative lookahead excludes letters that would form
# another unit (so we don't catch the ``°`` that we just stripped).
_TEMP_DEG_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°(?!\s*[A-Za-z])",
)

# ----- Currency (compound suffixes first, then bare). ---------------

# ``$1.5M`` / ``$1.5 million`` style. Order: M before plain, B before
# M (so $1.5B isn't grabbed by the M rule first), K likewise.
_CURRENCY_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*B\b"), r"\1 billion dollars"),
    (re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*M\b"), r"\1 million dollars"),
    (re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*K\b"), r"\1 thousand dollars"),
    (re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)"), r"\1 dollars"),
    (re.compile(r"€(\d+(?:,\d{3})*(?:\.\d+)?)\s*B\b"), r"\1 billion euros"),
    (re.compile(r"€(\d+(?:,\d{3})*(?:\.\d+)?)\s*M\b"), r"\1 million euros"),
    (re.compile(r"€(\d+(?:,\d{3})*(?:\.\d+)?)\s*K\b"), r"\1 thousand euros"),
    (re.compile(r"€(\d+(?:,\d{3})*(?:\.\d+)?)"), r"\1 euros"),
    (re.compile(r"£(\d+(?:,\d{3})*(?:\.\d+)?)\s*B\b"), r"\1 billion pounds"),
    (re.compile(r"£(\d+(?:,\d{3})*(?:\.\d+)?)\s*M\b"), r"\1 million pounds"),
    (re.compile(r"£(\d+(?:,\d{3})*(?:\.\d+)?)\s*K\b"), r"\1 thousand pounds"),
    (re.compile(r"£(\d+(?:,\d{3})*(?:\.\d+)?)"), r"\1 pounds"),
    (re.compile(r"¥(\d+(?:,\d{3})*(?:\.\d+)?)"), r"\1 yen"),
)

# ----- Units of measurement.
#
# Compound units (km/h, m/s) MUST come before bare-unit patterns so
# the slash form isn't broken into ``X kilometres / 30 hours`` by an
# earlier match. Each rule requires a digit prefix and a word
# boundary so common words ("m" in "I am", "g" inside "going") aren't
# misread as units. ----------------------------------------------------

_UNIT_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    # Speed (compound; run before bare distance + time units).
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*km\s*/\s*h\b", re.IGNORECASE), r"\1 kilometres per hour"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*kph\b", re.IGNORECASE), r"\1 kilometres per hour"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*mph\b", re.IGNORECASE), r"\1 miles per hour"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*m\s*/\s*s\b", re.IGNORECASE), r"\1 metres per second"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*ft\s*/\s*s\b", re.IGNORECASE), r"\1 feet per second"),
    # Mass (lb before bare numeric).
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*lbs?\b", re.IGNORECASE), r"\1 pounds"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*kgs?\b", re.IGNORECASE), r"\1 kilograms"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*oz\b", re.IGNORECASE), r"\1 ounces"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*mg\b", re.IGNORECASE), r"\1 milligrams"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*tonnes?\b", re.IGNORECASE), r"\1 tonnes"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*tons?\b", re.IGNORECASE), r"\1 tons"),
    # ``g`` is a unit ONLY when (a) preceded by a digit + optional space
    # and (b) followed by a non-letter boundary. Stops "5g network" from
    # matching, but catches "500 g of flour".
    (re.compile(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*g\b(?![A-Za-z])"), r"\1 grams"),
    # Distance (bare; after km/h, m/s above).
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*km\b", re.IGNORECASE), r"\1 kilometres"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*cm\b", re.IGNORECASE), r"\1 centimetres"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE), r"\1 millimetres"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*mi\b", re.IGNORECASE), r"\1 miles"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*ft\b", re.IGNORECASE), r"\1 feet"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*yds?\b", re.IGNORECASE), r"\1 yards"),
    # ``in`` is a preposition too -- only safe when adjacent to a digit
    # AND followed by a non-letter. ``5in screen`` would match; ``in the``
    # would not.
    (re.compile(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*in\b(?![A-Za-z])"), r"\1 inches"),
    # Bare ``m`` is risky -- only treat as metres when surrounded by
    # digit on the left AND non-letter on the right. Skips "I am",
    # "I'm", and similar.
    (re.compile(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*m\b(?![A-Za-z/])"), r"\1 metres"),
    # Time units (ms, sec, min, hr, hrs).
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*ms\b", re.IGNORECASE), r"\1 milliseconds"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*secs?\b", re.IGNORECASE), r"\1 seconds"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*mins?\b", re.IGNORECASE), r"\1 minutes"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*hrs?\b", re.IGNORECASE), r"\1 hours"),
    # Storage / data sizes.
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*GB\b"), r"\1 gigabytes"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*MB\b"), r"\1 megabytes"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*KB\b"), r"\1 kilobytes"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*TB\b"), r"\1 terabytes"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*GHz\b"), r"\1 gigahertz"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*MHz\b"), r"\1 megahertz"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*kHz\b"), r"\1 kilohertz"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*Hz\b"), r"\1 hertz"),
)

# ----- Ordinals (1st-31st covers calendar dates; beyond that the
# numeric form usually reads better). --------------------------------

_ORDINAL_WORDS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth",
    18: "eighteenth", 19: "nineteenth", 20: "twentieth",
    21: "twenty-first", 22: "twenty-second", 23: "twenty-third",
    24: "twenty-fourth", 25: "twenty-fifth", 26: "twenty-sixth",
    27: "twenty-seventh", 28: "twenty-eighth", 29: "twenty-ninth",
    30: "thirtieth", 31: "thirty-first",
}

_ORDINAL_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)


def _ordinal_sub(match: re.Match) -> str:
    n = int(match.group(1))
    return _ORDINAL_WORDS.get(n, f"{n}{match.group(2).lower()}")


# ----- Title abbreviations + acronym-dots. --------------------------

_TITLE_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bMr\.(?=\s+[A-Z])"), "Mister"),
    (re.compile(r"\bMrs\.(?=\s+[A-Z])"), "Missus"),
    (re.compile(r"\bMs\.(?=\s+[A-Z])"), "Miz"),
    (re.compile(r"\bDr\.(?=\s+[A-Z])"), "Doctor"),
    (re.compile(r"\bProf\.(?=\s+[A-Z])"), "Professor"),
    # ``St.`` is ambiguous (Street vs. Saint). Treat as ``Saint`` only
    # when followed by a capitalised proper noun -- that's the
    # personal-name pattern. Sentence-starting ``St.`` is rare; in
    # the address sense, the period typically appears AFTER a street
    # name (e.g., "Main St.") not before a capitalised name.
    (re.compile(r"\bSt\.(?=\s+[A-Z])"), "Saint"),
)

_ACRONYM_DOTS_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bU\.S\.A\."), "U S A"),
    (re.compile(r"\bU\.S\.(?!\w)"), "U S"),
    (re.compile(r"\bU\.K\.(?!\w)"), "U K"),
    (re.compile(r"\bU\.N\.(?!\w)"), "U N"),
    (re.compile(r"\bE\.U\.(?!\w)"), "E U"),
    (re.compile(r"\bN\.A\.S\.A\."), "NASA"),
)

# ----- Common Latin abbreviations (single-pass replacement). --------

_ABBREVIATION_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\be\.\s*g\.", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.\s*e\.", re.IGNORECASE), "that is"),
    (re.compile(r"\betc\.(?=\s|$|[,;])", re.IGNORECASE), "et cetera"),
    (re.compile(r"\bvs\.", re.IGNORECASE), "versus"),
    (re.compile(r"\bcf\.", re.IGNORECASE), "compare"),
    (re.compile(r"\bN\.B\.", re.IGNORECASE), "note well"),
    (re.compile(r"\bapprox\.", re.IGNORECASE), "approximately"),
)

# ----- Misc characters that throw XTTS off. -------------------------

# ``&`` between words is consistently read as "and" by some TTS but
# silently skipped by others. Replace it explicitly when it sits
# between word characters. Skip HTML entities ("&amp;" etc.).
_AMPERSAND_RE = re.compile(r"(?<=[A-Za-z0-9])\s*&\s*(?=[A-Za-z0-9])")

# 2026-05-19 Issue 1 fix: URLs in spoken text are catastrophic for the
# XTTS-v2 GPT context budget. The model tokenises ``https://``...``/``
# character-by-character with high token cost; a couple of URLs in a
# single sentence can push the audio-token output over the 4096 ctx
# window and crash the synth worker (live-session 2026-05-19 log:
# ``Requested tokens (4830) exceed context window of 4096``). The
# normaliser now strips URLs entirely -- a URL spoken aloud is
# unintelligible anyway and the sources list is shown in the printed
# transcript. Matches http(s)://, ftp://, and bare www. URLs.
_URL_RE = re.compile(
    r"(?:https?|ftp)://\S+|\bwww\.\S+",
    re.IGNORECASE,
)


def _ampm_letters(token: str) -> str:
    """``a.m.`` / ``AM`` / ``a m`` -> ``A M``."""
    letters = re.sub(r"[^a-zA-Z]", "", token).upper()
    return " ".join(letters)


def normalize_text_for_tts(text: str) -> str:
    """Rewrite text patterns XTTS-v2 mispronounces.

    Applies the following passes in order. Each pass is conservative
    -- unmatched text passes through unchanged.

    0. URLs (``https://...``, ``http://...``, ``ftp://...``, bare
       ``www.example.com``) are stripped. A URL spoken aloud is
       unintelligible and -- worse -- char-by-char tokenisation
       balloons the XTTS-v2 audio-token count, easily overflowing
       the 4096-token GPT context window (live-session 2026-05-19
       hit 4830 tokens on a response containing source URLs).
       Sources are still shown in the printed transcript.
    1. Windows drive paths (``C:\\foo\\bar\\baz.ext``) collapse to
       the filename leaf. Run FIRST AFTER url-strip because the
       drive-letter colon would otherwise look like a time pattern.
    2. Times with AM/PM (``2:16 a.m.`` -> ``2 16 A M``).
    3. Bare 24-hour times (``14:30`` -> ``14 30``).
    4. Standalone ``a.m.`` / ``p.m.`` markers.
    5. Temperatures: ``72°F`` -> ``72 degrees Fahrenheit``,
       ``20°C`` -> ``20 degrees Celsius``, ``45°`` -> ``45 degrees``.
    6. Currency: ``$1.5M`` -> ``1.5 million dollars``, ``£25`` ->
       ``25 pounds``, plus € and ¥. Compound suffixes (B/M/K) handled
       before bare amounts.
    7. Units of measurement: speed (mph, kph, km/h, m/s), mass (lb,
       kg, oz, g, mg, tonne, ton), distance (km, m, cm, mm, mi, ft,
       in, yd), time (ms, sec, min, hr), storage (GB, MB, KB, TB),
       frequency (Hz, kHz, MHz, GHz). Compound units (km/h, m/s)
       run before bare units. Each rule requires a digit prefix +
       word boundary so common words ("m" in "I am", "g" inside
       "going") aren't misread.
    8. Ordinals 1st-31st: ``19th`` -> ``nineteenth``. Larger
       ordinals stay numeric.
    9. Title abbreviations followed by a capitalised name:
       ``Dr. Smith`` -> ``Doctor Smith``.
    10. Acronym-with-dots: ``U.S.A.`` -> ``U S A``.
    11. Latin abbreviations: ``e.g.`` -> ``for example``,
        ``i.e.`` -> ``that is``, ``etc.`` -> ``et cetera``,
        ``vs.`` -> ``versus``, ``cf.`` -> ``compare``,
        ``approx.`` -> ``approximately``.
    12. Inter-word ``&`` -> ``and``.

    Pure function. Empty input passes through. Safe to call on any
    text.
    """
    if not text:
        return text

    out = text

    # 0. URLs -- strip entirely (Issue 1 fix; see _URL_RE comment).
    # Replace each URL with a single space so surrounding tokens
    # don't get glued together ("seehttps://x.com today" issue).
    # Collapse the resulting whitespace runs afterwards.
    out = _URL_RE.sub(" ", out)
    out = re.sub(r"[ \t]{2,}", " ", out)

    # 1. Windows paths.
    out = _WIN_PATH_RE.sub(lambda m: m.group(1), out)

    # 2-4. Times + AM/PM.
    out = _TIME_AMPM_RE.sub(
        lambda m: f"{m.group(1)} {m.group(2)} {_ampm_letters(m.group(3))}",
        out,
    )
    out = _TIME_24H_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", out)
    out = _AMPM_STANDALONE_RE.sub(
        lambda m: f"{m.group(1)} {_ampm_letters(m.group(2))}",
        out,
    )

    # 5. Temperatures (F/C first so the suffix isn't stripped by
    # the bare-degree pass).
    out = _TEMP_F_RE.sub(r"\1 degrees Fahrenheit", out)
    out = _TEMP_C_RE.sub(r"\1 degrees Celsius", out)
    out = _TEMP_DEG_RE.sub(r"\1 degrees", out)

    # 6. Currency.
    for pattern, replacement in _CURRENCY_PATTERNS:
        out = pattern.sub(replacement, out)

    # 7. Units of measurement (compound before bare).
    for pattern, replacement in _UNIT_PATTERNS:
        out = pattern.sub(replacement, out)

    # 8. Ordinals.
    out = _ORDINAL_RE.sub(_ordinal_sub, out)

    # 9. Titles + 10. Acronym-dots.
    for pattern, replacement in _TITLE_PATTERNS:
        out = pattern.sub(replacement, out)
    for pattern, replacement in _ACRONYM_DOTS_PATTERNS:
        out = pattern.sub(replacement, out)

    # 11. Latin abbreviations.
    for pattern, replacement in _ABBREVIATION_PATTERNS:
        out = pattern.sub(replacement, out)

    # 12. Ampersand.
    out = _AMPERSAND_RE.sub(" and ", out)

    return out


def _find_free_port() -> int:
    """Bind to port 0 to let the OS assign a free port, then close."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class XttsV3Speech:
    """XTTS v2 streaming TTS with v3 Kenning post-filter.

    Drop-in replacement for ``kenning.tts.speech.TextToSpeech``. Same
    public surface (``speak``, ``speak_stream``, ``warmup``, ``stop``)
    so the orchestrator can swap engines via config without touching
    the playback path.
    """

    def __init__(
        self,
        *,
        server_python: Optional[Path] = None,
        server_script: Optional[Path] = None,
        reference_audio: Optional[Path] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        flush_chars: str = settings.TTS_SENTENCE_FLUSH_CHARS,
        filter_preset: str = "v3_heavy",
        filter_tail_silence_ms: float = 200.0,
        speed: Optional[float] = None,
        temperature: Optional[float] = None,
        phantom_tail_trim_enabled: Optional[bool] = None,
        phantom_tail_silence_threshold: Optional[float] = None,
        phantom_tail_max_event_ms: Optional[float] = None,
        phantom_tail_min_lead_silence_ms: Optional[float] = None,
        gpt_cond_len: Optional[int] = None,
        gpt_cond_chunk_len: Optional[int] = None,
        max_ref_length: Optional[int] = None,
        rvc=None,  # accepted-but-ignored for legacy ctor compat
    ) -> None:
        # Resolve paths via config when not explicitly passed. Defaults
        # point at the layout established in the audio prep work.
        from kenning.config import get_config, resolve_path
        cfg = get_config()
        xtts_cfg = getattr(cfg.tts, "xtts_v3", None)

        if server_python is None:
            # Gitignored venv not migrated by the 2026-06-12 rename --
            # physically stays at ultronVoiceAudio/ (baked-in paths).
            sp = (xtts_cfg.server_python if xtts_cfg else None) or \
                "ultronVoiceAudio/.venv-xtts/Scripts/python.exe"
            server_python = resolve_path(sp)
        if server_script is None:
            ss = (xtts_cfg.server_script if xtts_cfg else None) or \
                "kenningVoiceAudio/scripts/xtts_server.py"
            server_script = resolve_path(ss)
        if reference_audio is None:
            # Gitignored reference WAV not migrated by the rename --
            # physically stays at ultronVoiceAudio/ under its old name.
            ra = (xtts_cfg.reference_audio if xtts_cfg else None) or \
                "ultronVoiceAudio/kokoro training audio/Ultron_vocals_mono_v1.wav"
            reference_audio = resolve_path(ra)

        if not Path(server_python).is_file():
            raise XttsServerStartError(
                f"XTTS server Python not found at {server_python}. "
                f"Did you create the .venv-xtts venv?"
            )
        if not Path(server_script).is_file():
            raise XttsServerStartError(
                f"XTTS server script not found at {server_script}."
            )
        if not Path(reference_audio).is_file():
            raise XttsServerStartError(
                f"XTTS reference audio not found at {reference_audio}."
            )

        self.server_python = Path(server_python)
        self.server_script = Path(server_script)
        self.reference_audio = Path(reference_audio)
        self.host = host
        self.port = int(port) if port is not None else _find_free_port()
        self.base_url = f"http://{self.host}:{self.port}"

        self.flush_chars = set(flush_chars)
        self.filter_preset = filter_preset
        self.filter_tail_silence_ms = float(filter_tail_silence_ms)
        # 2026-05-19 Issue 1 fix + round 4 retune: cap per-synth-call
        # text length so a single sentence can't overflow the 4096-
        # audio-token XTTS-v2 GPT context window, but keep the cap
        # high enough that ordinary multi-clause sentences pass
        # through as a single call. Round 3 hit "horrible pacing,
        # pauses randomly between words" because the 240-char cap was
        # splitting normal sentences into 3-4 fragments, each picking
        # up 200 ms of v3-filter tail silence. 600 chars at ~1.5
        # audio tokens per char = ~900 tokens, comfortably under the
        # 4096 cap. URL-laden text is handled separately by the
        # url-strip in :func:`normalize_text_for_tts`.
        if xtts_cfg is not None and getattr(xtts_cfg, "max_chars_per_synth_call", None):
            self._max_chars_per_synth_call = int(xtts_cfg.max_chars_per_synth_call)
        else:
            self._max_chars_per_synth_call = 600
        # Cadence: passed to XTTS ``inference_stream(speed=...)`` on the
        # server side. Adjusts synthesis duration tokens; does NOT touch
        # the post-synthesis v3 filter chain.
        if speed is None:
            speed = float(xtts_cfg.speed) if xtts_cfg is not None else 1.0
        self._synth_speed = float(speed)
        # 2026-05-12 phantom-token mitigation: lower temperature than
        # XTTS-v2's library default (0.75) cuts the rate at which the
        # GPT duration head emits fragmentary syllables at sentence
        # ends. Forwarded in the HTTP body.
        if temperature is None:
            temperature = float(xtts_cfg.temperature) if xtts_cfg is not None else 0.65
        self._synth_temperature = float(temperature)
        # Phantom-tail trim parameters (defence-in-depth on top of the
        # temperature reduction). Disabled here means the audio passes
        # straight from server PCM into the v3 filter; useful for A/B
        # comparison against the unfiltered output.
        if phantom_tail_trim_enabled is None:
            phantom_tail_trim_enabled = (
                bool(xtts_cfg.phantom_tail_trim_enabled) if xtts_cfg is not None else True
            )
        self._phantom_tail_trim_enabled = bool(phantom_tail_trim_enabled)
        if phantom_tail_silence_threshold is None:
            phantom_tail_silence_threshold = (
                float(xtts_cfg.phantom_tail_silence_threshold) if xtts_cfg is not None else 0.005
            )
        self._phantom_tail_silence_threshold = float(phantom_tail_silence_threshold)
        if phantom_tail_max_event_ms is None:
            phantom_tail_max_event_ms = (
                float(xtts_cfg.phantom_tail_max_event_ms) if xtts_cfg is not None else 200.0
            )
        self._phantom_tail_max_event_ms = float(phantom_tail_max_event_ms)
        if phantom_tail_min_lead_silence_ms is None:
            phantom_tail_min_lead_silence_ms = (
                float(xtts_cfg.phantom_tail_min_lead_silence_ms) if xtts_cfg is not None else 150.0
            )
        self._phantom_tail_min_lead_silence_ms = float(phantom_tail_min_lead_silence_ms)
        # 2026-05-20 round 9: extended reference-window conditioning.
        # These are forwarded to the XTTS server subprocess as CLI
        # args and consumed once at speaker-embedding computation
        # time. Coqui library defaults are 6/6/30 -- the round 9
        # production defaults of 30/6/60 give the speaker embedding
        # ~5x more prosody context + 2x more speaker-encoder context
        # from the same 3-min reference, at a one-time ~1-2 s extra
        # startup cost.
        if gpt_cond_len is None:
            gpt_cond_len = int(xtts_cfg.gpt_cond_len) if xtts_cfg is not None else 30
        self._gpt_cond_len = int(gpt_cond_len)
        if gpt_cond_chunk_len is None:
            gpt_cond_chunk_len = int(xtts_cfg.gpt_cond_chunk_len) if xtts_cfg is not None else 6
        self._gpt_cond_chunk_len = int(gpt_cond_chunk_len)
        if max_ref_length is None:
            max_ref_length = int(xtts_cfg.max_ref_length) if xtts_cfg is not None else 60
        self._max_ref_length = int(max_ref_length)
        # 2026-05-11 chunk-streaming investigation: was prototyped but
        # not shipped. Pedalboard's PitchShift (Rubber Band offline
        # mode) buffers ~25 000 samples internally with ``reset=False``,
        # which means streaming chunks through the v3_heavy chain
        # produces zero output until the buffer fills (and the buffered
        # audio can't be cleanly drained). Per-chunk ``reset=True``
        # works but produces ~125 % RMS divergence at chunk boundaries
        # -- audible artifacts. The v3 chain order is user-locked, so
        # moving PitchShift to the end (which would unblock streaming)
        # is out of scope. The audio is still streamed at the HTTP
        # level (server pushes PCM chunks as they're synthesised), but
        # the client accumulates the full sentence before filter
        # processing. See docs/codebase_structure.md for the
        # investigation notes.

        # Match the Piper path's output device + lock behaviour so the
        # orchestrator + barge-in handling stay uniform.
        self.output_device = resolve_device(settings.AUDIO_OUTPUT_DEVICE, "output")
        self._stop_event = threading.Event()
        self._playback_lock = threading.Lock()

        # Server lifecycle.
        self._server_proc: Optional[subprocess.Popen] = None
        self._sample_rate: int = 24000  # XTTS native; confirmed via /info after start

        # 2026-05-15 latency: pre-computed ack clip cache. Populated by
        # the orchestrator AFTER warmup via ``set_ack_cache`` + the
        # ``PrecomputedAckClipCache.prewarm`` daemon thread. Until then
        # (and on misses) ``_synthesize`` falls through to the live HTTP
        # + v3 filter path. The cache stores already-filtered audio so
        # cache hits are byte-identical to the live path.
        self._ack_cache: Optional["PrecomputedAckClipCache"] = None

        # 2026-05-15 latency: pre-opened output stream slot. The
        # orchestrator calls :meth:`prepare_output_stream` on a daemon
        # thread during Whisper STT so the ~50 ms PortAudio open cost
        # overlaps with transcription rather than landing on the
        # critical path before first audible audio. Consumed by
        # :meth:`speak_stream` -- if present + SR matches, the engine
        # reuses it instead of opening fresh. ``shutdown`` closes any
        # surviving pre-open.
        self._preopened_stream: Optional[sd.OutputStream] = None
        self._preopened_lock = threading.Lock()

        self._start_server()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        """Spawn the XTTS server subprocess and wait for /healthz."""
        argv = [
            str(self.server_python),
            "-u",
            str(self.server_script),
            "--host", self.host,
            "--port", str(self.port),
            "--reference", str(self.reference_audio),
            "--gpt-cond-len", str(self._gpt_cond_len),
            "--gpt-cond-chunk-len", str(self._gpt_cond_chunk_len),
            "--max-ref-length", str(self._max_ref_length),
        ]
        logger.info(
            "Starting XTTS server (port=%d, ref=%s, "
            "gpt_cond_len=%d gpt_cond_chunk_len=%d max_ref_length=%d)",
            self.port,
            self.reference_audio.name,
            self._gpt_cond_len,
            self._gpt_cond_chunk_len,
            self._max_ref_length,
        )
        try:
            # Inherit stderr to the parent so we see crashes; pipe
            # stdout to /dev/null since the server is verbose on
            # uvicorn startup.
            self._server_proc = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
            )
        except FileNotFoundError as e:
            raise XttsServerStartError(f"Failed to spawn XTTS server: {e}") from e

        # T12/T23: track the XTTS daemon as PERSISTENT (tracked in the
        # "what's running" registry, never auto-reaped). Fail-open.
        try:
            from kenning.subprocess.zombie_killer import get_zombie_killer
            if self._server_proc is not None:
                get_zombie_killer().register(
                    self._server_proc.pid, "xtts-server", persistent=True,
                )
        except Exception:  # noqa: BLE001
            pass

        # Poll /healthz until ready or timeout.
        deadline = time.monotonic() + _SERVER_STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._server_proc.poll() is not None:
                code = self._server_proc.returncode
                self._server_proc = None
                raise XttsServerStartError(
                    f"XTTS server exited during startup (code {code})."
                )
            try:
                req = urllib.request.Request(self.base_url + "/healthz")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    if payload.get("ok") and payload.get("speaker_cached"):
                        # Confirm sample rate via /info.
                        try:
                            with urllib.request.urlopen(
                                self.base_url + "/info", timeout=2.0
                            ) as ir:
                                info = json.loads(ir.read().decode("utf-8"))
                                self._sample_rate = int(info.get("sample_rate", 24000))
                        except Exception:
                            pass
                        logger.info(
                            "XTTS server ready in %.1fs (sample_rate=%d)",
                            _SERVER_STARTUP_TIMEOUT_S - (deadline - time.monotonic()),
                            self._sample_rate,
                        )
                        return
            except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
                pass  # not ready yet
            time.sleep(_SERVER_HEALTHZ_POLL_INTERVAL_S)

        # Timeout
        self._stop_server_subprocess()
        raise XttsServerStartError(
            f"XTTS server did not become ready within {_SERVER_STARTUP_TIMEOUT_S}s."
        )

    def _stop_server_subprocess(self) -> None:
        """Best-effort teardown via /shutdown then T8 kill_process_tree.

        Order:
        1. POST ``/shutdown`` so the FastAPI server drains its queue.
        2. Wait up to 2 s for clean exit.
        3. If still alive: T8 ``kill_process_tree`` (terminate + wait
           + force-kill survivors) -- handles uvicorn worker forks +
           any Coqui background threads in a single cross-platform
           call. Replaces the legacy nested-timeout terminate/kill
           pair which couldn't reach grandchildren.
        """
        if self._server_proc is None:
            return
        try:
            req = urllib.request.Request(
                self.base_url + "/shutdown", method="POST"
            )
            urllib.request.urlopen(req, timeout=1.0).close()
        except Exception:
            pass
        try:
            self._server_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                from kenning.subprocess.kill_tree import kill_process_tree
                kill_process_tree(self._server_proc.pid, grace_seconds=2.0)
            except Exception as e:  # noqa: BLE001
                logger.debug("xtts kill_process_tree raised (swallowed): %s", e)
        finally:
            try:
                from kenning.subprocess.zombie_killer import get_zombie_killer
                if self._server_proc is not None:
                    get_zombie_killer().unregister(self._server_proc.pid)
            except Exception:  # noqa: BLE001
                pass
            self._server_proc = None

    def __enter__(self) -> "XttsV3Speech":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
        self._stop_server_subprocess()

    # ------------------------------------------------------------------
    # Public API (mirrors TextToSpeech)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Interrupt any in-progress playback (signal only; doesn't stop the server)."""
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass
        # 2026-05-15: also close any pre-opened stream so shutdown
        # releases the device handle cleanly.
        with self._preopened_lock:
            s = self._preopened_stream
            self._preopened_stream = None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    def speak(self, text: str) -> None:
        """Synthesize + play ``text`` synchronously."""
        if not text.strip():
            return
        self._stop_event.clear()
        clip = self._synthesize(text)
        if clip[0].size > 0 and not self._stop_event.is_set():
            self._play(clip)

    def prepare_output_stream(self) -> None:
        """Open the PortAudio output stream proactively.

        2026-05-15 latency: the orchestrator calls this on a daemon
        thread after VAD ends and BEFORE Whisper STT so the ~50 ms
        ``sd.OutputStream`` open cost (Windows mixer round-trip)
        overlaps with transcription. When :meth:`speak_stream` is
        called shortly after, it consumes the pre-opened stream via
        :meth:`_consume_preopened_stream` and skips the open path
        entirely.

        Idempotent: re-calling with an existing pre-open is a no-op.
        Failures are swallowed and logged WARN -- the live path
        falls back to its own open as before.
        """
        with self._preopened_lock:
            if self._preopened_stream is not None:
                return
            try:
                from kenning.config import get_config
                tts_cfg = get_config().tts
                low_latency = bool(tts_cfg.output_low_latency_mode)
            except Exception:
                low_latency = False
            try:
                stream = self._open_output_stream(
                    self._sample_rate, low_latency,
                )
                stream.start()
                # Write 50 ms of silence to make sure the device is
                # actually emitting samples (avoids the first-write
                # underrun some drivers exhibit).
                self._write_silence(stream, self._sample_rate, 0.05)
                self._preopened_stream = stream
                logger.debug(
                    "XTTS+v3: output stream pre-opened (%d Hz, %s latency)",
                    self._sample_rate,
                    "low" if low_latency else "default",
                )
            except Exception as e:
                logger.warning(
                    "XTTS+v3 stream pre-open failed (%s); live path "
                    "will open fresh.", e,
                )

    def _consume_preopened_stream(self, sr: int) -> Optional[sd.OutputStream]:
        """Atomically take ownership of any pre-opened stream.

        Returns the stream when the cached one matches ``sr``;
        otherwise closes the cached stream (sample-rate mismatch) and
        returns None so the caller opens fresh.

        Thread-safe via ``_preopened_lock``. Callers transfer
        ownership to themselves -- the cache slot is cleared, so the
        engine no longer holds a reference. Defensive ``getattr``
        keeps the engine instantiable in unit-test fixtures that
        bypass ``__init__``.
        """
        lock = getattr(self, "_preopened_lock", None)
        if lock is None:
            return None
        with lock:
            s = getattr(self, "_preopened_stream", None)
            self._preopened_stream = None
        if s is None:
            return None
        if sr != self._sample_rate:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
            return None
        return s

    def warmup(self, text: str = "Online.") -> None:
        """Touch the server with a tiny request so the first real
        utterance doesn't pay any cold-cache cost."""
        if not text.strip():
            return
        t0 = time.monotonic()
        try:
            self._synthesize(text)
            logger.info("XTTS warmup complete in %.0fms", (time.monotonic() - t0) * 1000)
        except Exception as e:
            logger.warning("XTTS warmup skipped: %s", e)

    def speak_stream(self, fragments: Iterable[str]) -> None:
        """Consume token fragments and play sentence-by-sentence.

        Same producer-signaled lookahead playback contract as
        :meth:`kenning.tts.speech.TextToSpeech.speak_stream` -- queues
        :class:`ClipItem` tuples onto an internal audio queue and
        plays each clip immediately on receipt without blocking on
        the next clip first.
        """
        self._stop_event.clear()

        try:
            from kenning.config import get_config
            tts_cfg = get_config().tts
            spec_open = tts_cfg.speculative_stream_open_enabled
            # 2026-05-11 SR-mismatch fix: ``tts.speculative_stream_sample_rate``
            # is tuned for the legacy Piper+RVC stack (48 kHz). The XTTS
            # engine produces 24 kHz natively. Reading the global field
            # here forced a close-and-reopen on every turn (50-100 ms
            # wasted) when xtts_v3 was active. The engine knows its own
            # native rate, so use it directly -- the legacy speech.py
            # path is unchanged and still uses the config field.
            spec_sr = self._sample_rate
            low_latency = tts_cfg.output_low_latency_mode
        except Exception:
            spec_open = False
            spec_sr = self._sample_rate
            low_latency = False

        audio_q: queue.Queue[Optional[ClipItem]] = queue.Queue(maxsize=8)
        workers: list[threading.Thread] = []

        def synth_worker() -> None:
            try:
                self._run_synth_loop(
                    fragments=fragments,
                    push=lambda item: audio_q.put(item),
                )
            except Exception as e:
                logger.error("XTTS synth worker error: %s", e)
            finally:
                audio_q.put(None)

        worker = threading.Thread(target=synth_worker, daemon=True, name="xtts-synth")
        worker.start()
        workers.append(worker)

        sr: int = spec_sr if spec_open else self._sample_rate
        block_frames = max(1, int(sr * 0.05))
        stream: Optional[sd.OutputStream] = None
        first_item: Optional[ClipItem] = None

        try:
            with self._playback_lock:
                if self._stop_event.is_set():
                    return

                if spec_open:
                    # 2026-05-15 latency: prefer the pre-opened stream
                    # (opened during STT on a daemon thread) so the
                    # ~50 ms PortAudio open cost is already paid. SR
                    # mismatch falls back to a fresh open.
                    stream = self._consume_preopened_stream(sr)
                    if stream is None:
                        stream = self._open_output_stream(sr, low_latency)
                        stream.start()
                        self._write_silence(stream, sr, 0.05)
                    else:
                        logger.debug(
                            "XTTS+v3: consumed pre-opened output stream",
                        )

                try:
                    first_item = audio_q.get(timeout=_QUEUE_GET_TIMEOUT_SECONDS)
                except queue.Empty:
                    logger.warning("XTTS playback queue starved before first clip")
                    return
                if first_item is None:
                    return

                actual_sr = first_item.sample_rate
                if not spec_open:
                    sr = actual_sr
                    block_frames = max(1, int(sr * 0.05))
                    stream = self._open_output_stream(sr, low_latency)
                    stream.start()
                    self._write_silence(stream, sr, 0.05)
                elif actual_sr != sr:
                    logger.info("XTTS speculative SR %d != actual %d; reopening", sr, actual_sr)
                    if stream is not None:
                        try:
                            stream.stop()
                            stream.close()
                        except Exception:
                            pass
                    sr = actual_sr
                    block_frames = max(1, int(sr * 0.05))
                    stream = self._open_output_stream(sr, low_latency)
                    stream.start()
                    self._write_silence(stream, sr, 0.05)

                item = first_item
                while True:
                    audio = self._stereo_pcm(item.audio)
                    edge_ms = settings.TTS_EDGE_FADE_MS
                    if edge_ms > 0:
                        audio = self._apply_fade_in(audio, sr, ms=edge_ms)
                        audio = self._apply_fade_out(audio, sr, ms=edge_ms)

                    for start in range(0, audio.shape[0], block_frames):
                        if self._stop_event.is_set():
                            return
                        stream.write(audio[start : start + block_frames])

                    if item.is_known_last:
                        self._write_silence(stream, sr, 0.05)
                        break

                    pause_ms = settings.TTS_PAUSE_MS
                    if pause_ms > 0 and not self._stop_event.is_set():
                        self._write_silence(stream, sr, pause_ms / 1000.0)

                    try:
                        nxt = audio_q.get(timeout=_QUEUE_GET_TIMEOUT_SECONDS)
                    except queue.Empty:
                        logger.warning(
                            "XTTS playback waited %.0fs without next clip; ending",
                            _QUEUE_GET_TIMEOUT_SECONDS,
                        )
                        self._write_silence(stream, sr, 0.05)
                        break

                    if nxt is None:
                        self._write_silence(stream, sr, 0.05)
                        break

                    if nxt.sample_rate != sr:
                        stream.stop()
                        stream.close()
                        sr = nxt.sample_rate
                        block_frames = max(1, int(sr * 0.05))
                        stream = self._open_output_stream(sr, low_latency)
                        stream.start()
                        self._write_silence(stream, sr, 0.05)
                    item = nxt
        except Exception as e:
            logger.warning("XTTS streaming playback error: %s", e)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            for w in workers:
                w.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Common English abbreviations that end with `.` but do NOT mark
    # a sentence boundary. Lower-cased; the boundary check normalises
    # before lookup. Kept conservative -- a missed abbreviation
    # produces an early flush (audible micro-pause), while a false
    # positive HOLDS too much text and risks an overrun. So we only
    # list abbreviations confidently used mid-sentence.
    _ABBREVIATIONS: ClassVar[frozenset[str]] = frozenset({
        "mr", "mrs", "ms", "dr", "st", "jr", "sr", "fr",
        "vs", "etc", "eg", "ie", "cf", "al", "esp",
        "inc", "co", "ltd", "corp", "llc",
        "ave", "blvd", "rd", "pkwy", "hwy",
        "no", "nos",
        "approx", "vol", "ed", "eds", "rev", "ref",
    })

    @classmethod
    def _is_safe_sentence_boundary(
        cls, text: str, pos: int, *, buffer_complete: bool,
    ) -> bool:
        """Return True if ``text[pos]`` is a flushable sentence end.

        Rejects mid-token periods that would otherwise fragment the
        audio (ellipsis, decimals, domains, common abbreviations).
        ``buffer_complete`` should be True only when called from the
        tail-flush at end-of-stream so trailing `.` is treated as
        a real sentence end rather than "wait for more tokens".

        Round 7b (2026-05-20): introduced to stop ``.`` after
        ``Dictionary``, ``e.g``, ``3``, ``v2``, etc. from triggering
        a flush. Previously the entire stream got chopped into 3-6
        sub-clips per sentence, each carrying 200 ms of v3-filter
        tail silence -- the cause of the "horrible pacing, random
        pauses between words" the user reported.
        """
        ch = text[pos]
        n = len(text)
        if ch == "\n":
            return True
        if ch in "!?":
            # `!?` are unambiguous. Note: `??` or `!!` still trigger
            # on the first one; the post-flush remainder starts with
            # the second char, which we treat as a 1-char "sentence"
            # -- harmless, gets stripped.
            return True
        if ch != ".":
            return False
        # Ellipsis suppression: don't flush mid-ellipsis (next is `.`)
        # nor at the trailing dot of an ellipsis (prev is `.`).
        if pos + 1 < n and text[pos + 1] == ".":
            return False
        if pos > 0 and text[pos - 1] == ".":
            return False
        # Acronym continuation: pattern "L.L." where L is a letter --
        # e.g. "e.g.", "i.e.", "U.S.", "U.K.", "Ph.D.". The current `.`
        # closes a single-letter token whose predecessor is also `.`,
        # which marks it as part of an abbreviation chain that should
        # not break the sentence.
        if (
            pos >= 2
            and text[pos - 2] == "."
            and text[pos - 1].isalpha()
        ):
            return False
        # Decimal: digit.digit (e.g. "3.14"). Reject.
        if (
            pos > 0
            and text[pos - 1].isdigit()
            and pos + 1 < n
            and text[pos + 1].isdigit()
        ):
            return False
        # Mid-domain: letter.letter ("Dictionary.com"). Reject.
        if (
            pos > 0
            and text[pos - 1].isalpha()
            and pos + 1 < n
            and text[pos + 1].isalpha()
        ):
            return False
        # Trailing `.` with no next char: wait for more unless we're
        # at end of stream.
        if pos + 1 >= n:
            return buffer_complete
        next_ch = text[pos + 1]
        if next_ch.isspace():
            # Walk back over the preceding letter run -- if it's a
            # known abbreviation, suppress the flush.
            start = pos
            while start > 0 and text[start - 1].isalpha():
                start -= 1
            token = text[start:pos].lower()
            if token and token in cls._ABBREVIATIONS:
                return False
            return True
        # Anything else (e.g. "U.S.A." mid-token) -- be permissive,
        # let it flush. Worst case is one extra micro-pause, not a
        # context overflow.
        return True

    def _find_next_sentence_boundary(
        self, text: str, *, buffer_complete: bool,
    ) -> int:
        """Return position+1 of the next safe boundary, or 0 if none."""
        for i, ch in enumerate(text):
            if ch in self.flush_chars:
                if self._is_safe_sentence_boundary(
                    text, i, buffer_complete=buffer_complete,
                ):
                    return i + 1
        return 0

    def _run_synth_loop(
        self,
        *,
        fragments: Iterable[str],
        push: Callable[[ClipItem], None],
    ) -> None:
        """Walk fragments, synth on safe sentence boundaries, push ClipItems.

        Round 7b (2026-05-20): boundary detection now defers across
        ellipses, decimals, domains, and common abbreviations. The
        running buffer is held until we either find a *safe* flush
        or hit the safety-valve length (``max_chars`` * 2) at which
        point we soft-break on the last clause/space boundary so an
        unflushable stream can't grow without bound. End-of-stream
        always flushes whatever remains.

        2026-05-19 Issue 1 fix: every sentence is sub-split via
        :meth:`_split_for_synth` before being passed to ``_synthesize``,
        so no single HTTP call to the XTTS server sends text long
        enough to overflow the 4096-audio-token GPT context window.
        Cap is taken from ``tts.xtts_v3.max_chars_per_synth_call``
        (default 600).
        """
        max_chars = self._max_chars_per_synth_call
        # ``max_chars * 2`` headroom: a normal sentence sits under
        # ``max_chars``; we only soft-break when no safe boundary has
        # appeared and the buffer doubled past that threshold.
        soft_break_threshold = max(max_chars * 2, 240)
        pending = ""

        def _flush(text: str) -> None:
            for chunk in self._split_for_synth(text, max_chars):
                if self._stop_event.is_set():
                    return
                pcm, sr = self._synthesize(chunk)
                if pcm.size > 0:
                    push(ClipItem(pcm, sr, is_known_last=False))

        for frag in fragments:
            if self._stop_event.is_set():
                break
            if not frag:
                continue
            pending += frag
            # Drain as many safe boundaries as possible.
            while True:
                cut = self._find_next_sentence_boundary(
                    pending, buffer_complete=False,
                )
                if cut <= 0:
                    break
                sentence = pending[:cut].strip()
                pending = pending[cut:].lstrip()
                if sentence:
                    _flush(sentence)
                    if self._stop_event.is_set():
                        break
            if self._stop_event.is_set():
                break
            # Safety valve: if the buffer has grown well past the
            # synth-call cap without finding a safe boundary, soft-
            # break at the last clause / space.
            if len(pending) > soft_break_threshold:
                soft_cut = -1
                for sep in (";", ":", ",", "--", " "):
                    pos = pending.rfind(sep)
                    if pos > soft_cut:
                        soft_cut = pos
                if soft_cut > 0:
                    sentence = pending[: soft_cut + 1].strip()
                    pending = pending[soft_cut + 1 :].lstrip()
                    if sentence:
                        _flush(sentence)

        tail = pending.strip()
        if tail and not self._stop_event.is_set():
            _flush(tail)

    @staticmethod
    def _split_for_synth(text: str, max_chars: int) -> list[str]:
        """Sub-split a single chunk so each piece is <= ``max_chars``.

        Tries successively-finer boundaries: clause punctuation
        (``,`` ``;`` ``:`` ``--``) first, then space boundaries. If
        a token itself exceeds the cap (e.g. a long URL the URL-strip
        missed), it is force-sliced at the char level so the call
        stays under the limit.

        Returns ``[text]`` unchanged when ``text`` is already short
        enough -- the common case, preserving byte-for-byte legacy
        behaviour for typical conversational responses.
        """
        if max_chars <= 0:
            return [text]
        text = text.strip()
        if not text or len(text) <= max_chars:
            return [text] if text else []

        # Pass 1: clause boundaries (preserve the boundary char on the
        # left chunk so the cadence stays natural).
        clause_re = re.compile(r"([,;:]|--|—| -- )\s+")
        parts: list[str] = []
        cursor = 0
        for m in clause_re.finditer(text):
            end = m.end()
            parts.append(text[cursor:end].strip())
            cursor = end
        if cursor < len(text):
            parts.append(text[cursor:].strip())
        if len(parts) == 1:
            # No clause boundaries; fall through to word split.
            parts = [text]

        out: list[str] = []
        for part in parts:
            if not part:
                continue
            if len(part) <= max_chars:
                out.append(part)
                continue
            # Pass 2: word boundaries. Greedily pack words into chunks
            # of <= max_chars characters.
            words = part.split()
            current = ""
            for w in words:
                if not w:
                    continue
                # If a single word exceeds the cap (very long URL /
                # alphanumeric ID the strip missed), force-slice it.
                if len(w) > max_chars:
                    if current:
                        out.append(current)
                        current = ""
                    for i in range(0, len(w), max_chars):
                        out.append(w[i:i + max_chars])
                    continue
                proposed = (current + " " + w).strip() if current else w
                if len(proposed) > max_chars:
                    if current:
                        out.append(current)
                    current = w
                else:
                    current = proposed
            if current:
                out.append(current)

        return [p for p in out if p]

    def set_ack_cache(self, cache: Optional[PrecomputedAckClipCache]) -> None:
        """Wire a pre-computed ack clip cache.

        Once installed, :meth:`_synthesize` checks the cache before
        running the live HTTP + v3 filter path. Cache hits return the
        stored ``(pcm, sr)`` clip directly. Misses fall through to the
        live path unchanged.

        Pass ``None`` to detach the cache (e.g. after a server restart
        when the cached clips may no longer match the live engine
        state).
        """
        self._ack_cache = cache
        if cache is not None:
            logger.info(
                "XTTS+v3: ack clip cache attached (%d phrases enrolled)",
                len(cache.phrases),
            )

    def _synthesize(self, text: str) -> Clip:
        """Synthesize one sentence: cache → HTTP → assemble PCM → v3 filter → (pcm, sr).

        Cache lookup happens BEFORE the HTTP call so a hit returns
        immediately, skipping ~350-400 ms of XTTS inference + filter
        work. The cache stores already-filtered audio so cache hits
        produce byte-identical output to the live path. On miss, the
        existing live path runs unchanged.
        """
        # 2026-05-15 latency: precomputed ack clip cache. Phrases like
        # "Mm." / "Querying external sources." / etc. are pre-rendered
        # once at orchestrator startup and reused for the entire
        # session. Cache hit = skip HTTP + filter; cache miss = live
        # path. The cache is keyed by stripped text -- the strip
        # convention must match what :meth:`_run_synth_loop` applies
        # before calling here, which it does. ``getattr`` keeps the
        # engine instantiable in unit-test fixtures that bypass
        # ``__init__``.
        ack_cache = getattr(self, "_ack_cache", None)
        if ack_cache is not None:
            cached = ack_cache.get(text)
            if cached is not None:
                logger.debug(
                    "XTTS+v3: ack-cache hit for %r (skipped %.0fms synth)",
                    text[:40], 0.0,  # actual saving logged in aggregate
                )
                return cached

        t0 = time.monotonic()
        try:
            spoken = normalize_text_for_tts(text)
        except Exception as e:
            logger.warning(
                "TTS text normalisation failed for %r (%s); using raw text",
                text[:60], e,
            )
            spoken = text
        try:
            pcm_i16 = self._http_synthesize(spoken)
        except Exception as e:
            logger.error("XTTS server synth failed for %r: %s", text[:60], e)
            from kenning.errors import PiperSynthesisError  # closest typed error
            from kenning.resilience import get_error_log
            get_error_log().record(
                PiperSynthesisError(
                    f"XTTS server synth failed: {e}",
                    context={"text_preview": text[:60], "text_chars": len(text)},
                    recovery="returned silent clip; orchestrator falls back to terminal print",
                ),
                dependency="xtts_server",
            )
            return np.zeros(0, dtype=np.int16), self._sample_rate

        if pcm_i16.size == 0:
            logger.warning("XTTS produced no audio for %r", text[:60])
            return pcm_i16, self._sample_rate

        # Apply v3 filter. Convert int16 -> float32 [-1, 1], filter,
        # convert back. The filter pads tail_silence_ms of trailing
        # zeros so reverb decay isn't clipped at the buffer end.
        pcm_f32 = pcm_i16.astype(np.float32) / 32768.0

        # 2026-05-12 phantom-tail trim: catches the residual XTTS-v2
        # phantom syllables that slip past the lower temperature.
        # Runs BEFORE the filter so the reverb tail decays normally
        # into its tail_silence_ms padding rather than into a phantom.
        if self._phantom_tail_trim_enabled:
            try:
                pcm_f32, was_trimmed = trim_phantom_tail(
                    pcm_f32,
                    self._sample_rate,
                    silence_threshold=self._phantom_tail_silence_threshold,
                    max_event_ms=self._phantom_tail_max_event_ms,
                    min_lead_silence_ms=self._phantom_tail_min_lead_silence_ms,
                )
                if was_trimmed:
                    logger.debug(
                        "Phantom-tail trimmed on %r (clip=%d samples)",
                        text[:40], pcm_f32.size,
                    )
            except Exception as e:
                logger.warning("Phantom-tail trim failed (using raw PCM): %s", e)

        try:
            filtered_f32 = apply_kenning_filter(
                pcm_f32,
                self._sample_rate,
                preset=self.filter_preset,
                tail_silence_ms=self.filter_tail_silence_ms,
            )
        except Exception as e:
            logger.warning("Kenning filter failed (using raw PCM): %s", e)
            filtered_f32 = pcm_f32

        # Convert back to int16 with clipping.
        np.clip(filtered_f32, -1.0, 1.0, out=filtered_f32)
        out_pcm = (filtered_f32 * 32767.0).astype(np.int16)
        logger.debug(
            "XTTS+v3: %d chars -> %.2fs audio @ %d Hz in %.0fms",
            len(text),
            out_pcm.size / max(self._sample_rate, 1),
            self._sample_rate,
            (time.monotonic() - t0) * 1000,
        )
        return out_pcm, self._sample_rate

    def _http_synthesize(self, text: str) -> np.ndarray:
        """POST /synthesize, accumulate streamed PCM, return int16 array.

        2026-05-19 defense-in-depth: cap text length BEFORE sending so
        the XTTS-v2 4096-audio-token context window can't be exceeded
        by a single call. The chunker in :meth:`_run_synth_loop`
        normally keeps every call under ``max_chars_per_synth_call``,
        but live sessions have hit the limit on phantom-text paths
        we haven't fully traced (LLM stream emits 0 chars but the
        synth queue receives ~5000 audio-tokens worth of text). The
        hard cap below is the belt-and-braces: 1.5x the configured
        chunk size, with the offending text logged so we can identify
        the upstream culprit on next occurrence.

        2026-05-19 round 5: also log EVERY synth call at INFO level
        (with length + preview) so the next phantom-text occurrence
        shows what XTTS actually received. The XTTS server's
        "Requested tokens (N)" error message is opaque about the
        input text -- this log captures it client-side.
        """
        text_len = len(text)
        logger.info(
            "XTTS synth: %d chars -> server (preview=%r)",
            text_len, text[:160],
        )
        hard_cap = max(120, int(self._max_chars_per_synth_call * 1.5))
        if text_len > hard_cap:
            logger.warning(
                "XTTS text cap: truncating %d-char input to %d (preview=%r)",
                text_len, hard_cap, text[:120],
            )
            text = text[:hard_cap].rstrip() + "."
        body = json.dumps(
            {
                "text": text,
                "language": "en",
                "speed": self._synth_speed,
                "temperature": self._synth_temperature,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/synthesize",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            sr_header = resp.headers.get("X-Sample-Rate")
            if sr_header:
                self._sample_rate = int(sr_header)
            chunks: list[bytes] = []
            while True:
                c = resp.read(8192)
                if not c:
                    break
                chunks.append(c)
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        raw = b"".join(chunks)
        return np.frombuffer(raw, dtype=np.int16).copy()


    def _play(self, clip: Clip) -> None:
        """Single-shot playback. Same shape as TextToSpeech._play."""
        pcm, sr = clip
        try:
            from kenning.config import get_config
            low_latency = get_config().tts.output_low_latency_mode
        except Exception:
            low_latency = False
        with self._playback_lock:
            if self._stop_event.is_set():
                return
            try:
                audio = self._stereo_pcm(pcm)
                duration = audio.shape[0] / max(sr, 1)
                logger.info(
                    "Playing XTTS+v3 clip: %.2fs @ %d Hz via %s",
                    duration, sr, describe_device(self.output_device, "output"),
                )
                block_frames = max(1, int(sr * 0.05))
                with self._open_output_stream(sr, low_latency) as stream:
                    for start in range(0, audio.shape[0], block_frames):
                        if self._stop_event.is_set():
                            return
                        stream.write(audio[start : start + block_frames])
            except Exception as e:
                logger.warning("Playback error: %s", e)

    def _open_output_stream(self, sample_rate: int, low_latency: bool) -> sd.OutputStream:
        kwargs: dict = {
            "samplerate": sample_rate,
            "channels": 2,
            "dtype": "int16",
            "device": self.output_device,
        }
        if low_latency:
            kwargs["latency"] = "low"
        return sd.OutputStream(**kwargs)

    @staticmethod
    def _stereo_pcm(pcm: np.ndarray) -> np.ndarray:
        mono = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if mono.size == 0:
            return np.zeros((0, 2), dtype=np.int16)
        return np.column_stack((mono, mono)).astype(np.int16, copy=False)

    @staticmethod
    def _apply_fade_in(audio: np.ndarray, sr: int, ms: float = 4.0) -> np.ndarray:
        n = audio.shape[0]
        if n == 0:
            return audio
        fade = min(n, max(1, int(sr * ms / 1000.0)))
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32).reshape(-1, 1)
        out = audio.copy()
        out[:fade] = (out[:fade].astype(np.float32) * ramp).astype(np.int16)
        return out

    @staticmethod
    def _apply_fade_out(audio: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
        n = audio.shape[0]
        if n == 0:
            return audio
        fade = min(n, max(1, int(sr * ms / 1000.0)))
        ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32).reshape(-1, 1)
        out = audio.copy()
        out[-fade:] = (out[-fade:].astype(np.float32) * ramp).astype(np.int16)
        return out

    @staticmethod
    def _write_silence(stream: sd.OutputStream, sr: int, duration_s: float) -> None:
        n = max(0, int(sr * duration_s))
        if n == 0:
            return
        silence = np.zeros((n, 2), dtype=np.int16)
        try:
            stream.write(silence)
        except Exception as e:
            logger.debug("Silence write failed (likely closing stream): %s", e)
