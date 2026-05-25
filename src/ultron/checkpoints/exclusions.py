"""Exclusion patterns for the shadow-repo checkpoint system.

Two layers of exclusion compose into the shadow repo's ``.gitignore``:

1. :data:`DEFAULT_CHECKPOINT_EXCLUSIONS` — the cline-style "always
   skip" list (node_modules, .venv, __pycache__, dist, build, binary
   media). Kept generous because the shadow repo only needs to track
   text artifacts the user might want to rewind.

2. :data:`VOICE_BASELINE_PROTECTED_PATTERNS` — the LOAD-BEARING
   ultron-specific list: voice-quality-locked files (SOUL.md, RVC
   weights, Piper voice, Kokoro voicepack, LLM model file). These
   MUST be excluded so a checkpoint restore cannot accidentally
   roll the voicepack to a stale snapshot (catastrophic per the
   voice-quality contract).

The caller composes the two via :func:`compose_gitignore`, optionally
extending with patterns from the workspace's own ``.gitattributes``
(LFS-tracked files, etc.). The result is written to the shadow repo's
``info/exclude`` file at init.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

#: Default cross-cutting exclusion patterns. These mirror the
#: gitignore-shape entries the cline catalog calls out.
DEFAULT_CHECKPOINT_EXCLUSIONS: tuple[str, ...] = (
    # Python build / venv detritus.
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".venv/",
    ".venv-*/",
    "venv/",
    "env/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".tox/",
    "*.egg-info/",
    "build/",
    "dist/",
    # JS / TS build detritus.
    "node_modules/",
    ".next/",
    ".nuxt/",
    ".turbo/",
    "out/",
    "coverage/",
    ".cache/",
    # OS noise.
    ".DS_Store",
    "Thumbs.db",
    # Binary media (the shadow repo is for text rewind, not assets).
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.bmp",
    "*.tiff",
    "*.mp3",
    "*.mp4",
    "*.mov",
    "*.wav",
    "*.flac",
    "*.aac",
    "*.ogg",
    "*.pdf",
    "*.zip",
    "*.gz",
    "*.tar",
    "*.7z",
    "*.iso",
    # Large model artefacts (defer to the voice-baseline lock layer).
    "*.gguf",
    "*.pt",
    "*.pth",
    "*.bin",
    "*.safetensors",
    "*.ckpt",
    "*.onnx",
    "*.joblib",
    "*.pkl",
    # Runtime data Ultron mutates outside the checkpointed workspace.
    "logs/",
    "data/qdrant/",
    "data/checkpoints/",
    "data/streaming-overflow/",
    "data/maintenance.sqlite",
    "data/coding/sessions/",
    # IDE noise.
    ".idea/",
    ".vscode/",
    ".cursor/",
    ".vs/",
)


#: Voice-baseline-lock patterns. These MUST stay excluded from any
#: checkpoint or restore — the voice-quality contract treats these
#: files as immutable runtime constants. Restoring them from a stale
#: snapshot would silently regress the voice character.
VOICE_BASELINE_PROTECTED_PATTERNS: tuple[str, ...] = (
    # The persona + voice quality files.
    "SOUL.md",
    "IDENTITY.md",
    "ultronVoiceAudio/Ultron_vocals_mono_v1.wav",
    "ultronVoiceAudio/Ultron_vocals_mono_v1.*",
    # Piper baseline voice + RVC support files.
    "models/piper/**",
    "models/rvc/**",
    "ultron_james_spader_mcu_6941/**",
    # Kokoro fine-tune voicepack + fine-tune weights.
    "models/kokoro/**",
    # LLM GGUFs (swapping is via preset only, never via checkpoint).
    "models/*.gguf",
    "models/*.bin",
    "models/*.safetensors",
    # Wake-word ONNX + Smart Turn ONNX.
    "models/openwakeword/**",
    "models/smart_turn/**",
    # Moondream2 + addressing classifier caches.
    "models/moondream2/**",
    "models/flan-t5-small/**",
    "models/.hf-cache/**",
)


def compose_gitignore(
    *,
    include_defaults: bool = True,
    include_voice_baseline: bool = True,
    extra_patterns: Iterable[str] = (),
    workspace_gitattributes: str = "",
) -> str:
    """Render the composed gitignore body used by the shadow repo.

    Args:
        include_defaults: include :data:`DEFAULT_CHECKPOINT_EXCLUSIONS`.
        include_voice_baseline: include
            :data:`VOICE_BASELINE_PROTECTED_PATTERNS`. Default True;
            disabling requires explicit caller intent (the voice
            contract makes this load-bearing).
        extra_patterns: optional caller-supplied additional patterns.
        workspace_gitattributes: optional contents of the workspace's
            ``.gitattributes``; LFS-tracked entries are extracted and
            appended to the exclusion list.

    Returns:
        Newline-joined string suitable for writing to the shadow repo's
        ``.git/info/exclude`` file.
    """
    sections: list[str] = []
    if include_defaults:
        sections.append("# DEFAULT_CHECKPOINT_EXCLUSIONS")
        sections.extend(DEFAULT_CHECKPOINT_EXCLUSIONS)
    if include_voice_baseline:
        sections.append("")
        sections.append("# VOICE_BASELINE_PROTECTED_PATTERNS")
        sections.extend(VOICE_BASELINE_PROTECTED_PATTERNS)
    lfs_patterns = _extract_lfs_patterns(workspace_gitattributes)
    if lfs_patterns:
        sections.append("")
        sections.append("# LFS-tracked patterns (from .gitattributes)")
        sections.extend(lfs_patterns)
    extras = [p.strip() for p in extra_patterns if p and p.strip()]
    if extras:
        sections.append("")
        sections.append("# extra patterns")
        sections.extend(extras)
    return "\n".join(sections) + "\n"


def _extract_lfs_patterns(text: str) -> list[str]:
    """Recover ``filter=lfs`` patterns from a ``.gitattributes`` body."""
    out: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "filter=lfs" not in line:
            continue
        # Pattern is the first token before whitespace.
        pattern = line.split(None, 1)[0]
        if pattern:
            out.append(pattern)
    return out


__all__ = [
    "DEFAULT_CHECKPOINT_EXCLUSIONS",
    "VOICE_BASELINE_PROTECTED_PATTERNS",
    "compose_gitignore",
]
