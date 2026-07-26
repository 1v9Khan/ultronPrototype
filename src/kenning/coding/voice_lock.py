"""Voice-character-lock pre-dispatch guardrails (E5).

The project carries a long-standing **voice-quality lock**: certain
files (Piper voice, RVC weights, the legacy `tts/speech.py` engine,
`tts/rvc.py`, the cleaned vocal reference WAV, SOUL.md persona file)
must not be modified without explicit user direction. Today the
convention is enforced by:

* The runtime tool-call validator's Category K (K2 covers the vocal
  WAV; K8 covers AI-pipeline-ingested files like SOUL.md).
* Reviewer discipline.

This module adds a **regex-grade pre-dispatch scanner** for the
coding bridge: when AI coding agent is about to be dispatched against a
task whose prompt explicitly targets a voice-locked file, the
orchestrator gets a structured warning it can surface to the user
(and optionally block on) BEFORE the subprocess spawns.

It also provides a FILE_CHANGE listener helper that catches actual
write attempts at runtime. The runtime safety validator already
intercepts these in production for K-listed paths; the voice-lock
check covers the larger set of voice-quality-relevant paths that K
doesn't enumerate.

Pure-Python, no IO, no config dependence (callers pass in the
protected set or rely on the documented defaults).
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Sequence


# Default set of voice-locked paths and globs. Repository-relative.
# Callers can extend or override at construction time.
DEFAULT_VOICE_LOCKED_PATHS: tuple[str, ...] = (
    # Persona file (workspace-side, also K8-protected when ingested).
    "~/.openclaw/workspace/SOUL.md",
    # Legacy TTS engine + RVC wrapper.
    "src/kenning/tts/speech.py",
    "src/kenning/tts/rvc.py",
    # Reference vocal sample for XTTS. Both the legacy root-level
    # location and the post-disk-cleaning home are locked (additive --
    # the file moved 2026-06-11; protections follow it, never shrink).
    # The 2026-06-12 Kenning rename did NOT migrate this gitignored WAV
    # (baked-in venv paths) -- it physically stays at ultronVoiceAudio/
    # under its original name, so BOTH trees are locked (never shrink).
    "ultronVoiceAudio/Ultron_vocals_mono_v1.wav",
    "ultronVoiceAudio/kokoro training audio/Ultron_vocals_mono_v1.wav",
    "kenningVoiceAudio/Kenning_vocals_mono_v1.wav",
    "kenningVoiceAudio/kokoro training audio/Kenning_vocals_mono_v1.wav",
)

DEFAULT_VOICE_LOCKED_GLOBS: tuple[str, ...] = (
    # Piper voice weights.
    "models/piper/**",
    "models/piper/*",
    # RVC support files.
    "models/rvc/**",
    "models/rvc/*",
    # RVC voice model directory (specific to Kenning's character).
    "kenning_rvc_voice/**",
    "kenning_rvc_voice/*",
    # Coqui XTTS speaker embeddings (any future reference files). Both
    # the real (ultronVoiceAudio, gitignored) and post-rename
    # (kenningVoiceAudio, tracked scripts) workshop trees are locked.
    "ultronVoiceAudio/*.wav",
    "ultronVoiceAudio/*.npy",
    "kenningVoiceAudio/*.wav",
    "kenningVoiceAudio/*.npy",
)


def _normalise_path(path: str) -> str:
    """Return ``path`` with normalised slashes + drive case for matching.

    Lowercases the drive letter on Windows, collapses ``\\`` to ``/``,
    and resolves ``~`` to the user home directory. Idempotent.
    """
    if not path:
        return ""
    expanded = str(Path(path).expanduser())
    # Normalise separators for glob matching.
    posix = expanded.replace("\\", "/")
    return posix


def _matches_any(path_norm: str, patterns: Sequence[str]) -> Optional[str]:
    """Return the first matching pattern (or None) for ``path_norm``."""
    for pat in patterns:
        pat_norm = _normalise_path(pat)
        if fnmatch.fnmatch(path_norm, pat_norm):
            return pat
        # Also try suffix match -- "src/kenning/tts/speech.py" should
        # match against the canonical relative path even if the input is
        # an absolute path under the repo.
        if path_norm.endswith("/" + pat_norm.lstrip("/")):
            return pat
    return None


def is_voice_locked_path(
    path: str,
    *,
    extra_paths: Iterable[str] = (),
    extra_globs: Iterable[str] = (),
) -> Optional[str]:
    """Return the matching pattern when ``path`` is voice-locked.

    Returns ``None`` when ``path`` is NOT voice-locked. The returned
    string is the matching pattern so callers can surface "why" in
    user-facing narration.
    """
    if not path:
        return None
    path_norm = _normalise_path(path)

    # Exact paths + appended extras
    exact = tuple(DEFAULT_VOICE_LOCKED_PATHS) + tuple(extra_paths or ())
    match = _matches_any(path_norm, exact)
    if match is not None:
        return match

    # Glob patterns
    globs = tuple(DEFAULT_VOICE_LOCKED_GLOBS) + tuple(extra_globs or ())
    return _matches_any(path_norm, globs)


# ---------------------------------------------------------------------------
# Prompt scanner
# ---------------------------------------------------------------------------


# Heuristic for spotting file paths inside a free-form coding prompt.
# Catches anything that looks like ``path/with/slashes.ext`` -- a fairly
# broad net deliberately (we'd rather over-flag than under-flag for a
# voice-character-lock guardrail).
_PATH_LIKE = re.compile(
    r"(?P<path>"
    # Absolute Windows paths with drive letter.
    r"[A-Za-z]:[\\/][^\s\"'<>|?*]+"
    # Absolute POSIX paths.
    r"|/[^\s\"'<>|?*]+"
    # Tilde-prefixed user paths.
    r"|~/[^\s\"'<>|?*]+"
    # Relative project-style paths (must include at least one slash + an
    # extension or recognisable bottom segment).
    r"|(?:[A-Za-z0-9_\-.]+[\\/])+[A-Za-z0-9_\-.]+"
    r")"
)


@dataclass(frozen=True)
class VoiceLockHit:
    """One detected voice-locked target."""

    path: str
    matched_pattern: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "matched_pattern": self.matched_pattern,
            "reason": self.reason,
        }


def scan_prompt(
    prompt: str,
    *,
    extra_paths: Iterable[str] = (),
    extra_globs: Iterable[str] = (),
) -> list[VoiceLockHit]:
    """Scan a free-form coding-task prompt for voice-locked targets.

    Returns the deduplicated list of hits (by path). An empty list
    means the prompt is safe under the current lock.
    """
    if not prompt:
        return []
    seen: dict[str, VoiceLockHit] = {}
    for match in _PATH_LIKE.finditer(prompt):
        candidate = match.group("path")
        # Strip trailing sentence punctuation -- the path regex
        # character class includes ``.`` so a sentence-terminating
        # period (or comma / colon) at the tail gets absorbed and would
        # break the strict-match check below.
        candidate = candidate.rstrip(".,;:!?)")
        pattern = is_voice_locked_path(
            candidate,
            extra_paths=extra_paths,
            extra_globs=extra_globs,
        )
        if pattern is None:
            continue
        # Dedupe by normalised path so "models/piper/foo" and
        # "./models/piper/foo" report once.
        key = _normalise_path(candidate).lower()
        if key in seen:
            continue
        seen[key] = VoiceLockHit(
            path=candidate,
            matched_pattern=pattern,
            reason=(
                f"path {candidate!r} matches voice-locked pattern "
                f"{pattern!r}; voice-character-lock convention forbids "
                "modification without explicit user direction"
            ),
        )
    return list(seen.values())


# ---------------------------------------------------------------------------
# FILE_CHANGE listener helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileChangeScanResult:
    """Verdict for one FILE_CHANGE event."""

    path: str
    hit: Optional[VoiceLockHit]

    @property
    def blocked(self) -> bool:
        return self.hit is not None


def scan_file_change(
    path: str,
    *,
    extra_paths: Iterable[str] = (),
    extra_globs: Iterable[str] = (),
) -> FileChangeScanResult:
    """Check a single FILE_CHANGE path against the voice-lock list."""
    pattern = is_voice_locked_path(
        path, extra_paths=extra_paths, extra_globs=extra_globs
    )
    if pattern is None:
        return FileChangeScanResult(path=path, hit=None)
    return FileChangeScanResult(
        path=path,
        hit=VoiceLockHit(
            path=path,
            matched_pattern=pattern,
            reason=(
                f"FILE_CHANGE targeted voice-locked path {path!r} "
                f"(pattern {pattern!r})"
            ),
        ),
    )


def render_warning_for_voice(hits: Sequence[VoiceLockHit]) -> str:
    """Compose an in-character voice line summarising ``hits``.

    Returns an empty string when ``hits`` is empty. Designed to drop
    into the existing voice-narration path (TTS-safe -- no backslashes,
    no drive letters, no long unbroken slugs that confuse XTTS).
    """
    if not hits:
        return ""
    if len(hits) == 1:
        target = Path(hits[0].path).name
        return (
            f"I noticed this task would touch the voice-locked file "
            f"{target}. That is off limits without your explicit "
            "permission. Confirm if you actually want me to proceed."
        )
    targets = ", ".join(sorted({Path(h.path).name for h in hits}))
    return (
        f"I noticed this task would touch voice-locked files: {targets}. "
        "Those are off limits without your explicit permission. "
        "Confirm if you actually want me to proceed."
    )
