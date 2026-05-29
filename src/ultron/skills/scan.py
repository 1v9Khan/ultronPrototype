"""Trust scan for skills loaded from UNTRUSTED sources.

A skill's markdown body is injected verbatim into the LLM system prompt,
so a hostile ``.md`` dropped into ``~/.ultron/skills``, a project's
``.ultron/skills``, or the autonomous ``data/evolution/skills`` directory
is a prompt-injection / instruction-override vector. This module flags such
skills so :func:`ultron.skills.loader.load_skills_from_directory` can
QUARANTINE them (skip + log) before they ever become active. PUBLIC
(ultron-shipped) skills are trusted and never scanned.

Two checks:

* **content scan** -- chat-template tag markers (``[INST]`` /
  ``<|im_start|>`` / ``<|system|>`` / ...) + natural-language jailbreak /
  instruction-override / system-prompt-override phrasing in the body or
  description. These are the bytes that would manipulate the model.
* **companion-code scan** -- if the skill's directory carries ``.py``
  files (a code-bundled skill), the install-time
  :func:`ultron.install.static_scanner.scan_install_directory` runs and any
  CRITICAL finding is folded in.

Fail-open at the edges: an unexpected scanner error is treated as a WARN
(not a quarantine) so a broken scanner can never silently drop every skill.
A positive *detection*, by contrast, quarantines -- that is the safe
direction for an actual injection hit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

CLEAN = "clean"
WARNING = "warning"
CRITICAL = "critical"

# Chat-template / control markers that have no business in a knowledge
# skill body. Mirrors the marker set neutralised by
# ``ultron.llm.inference._sanitize_user_input`` for user utterances.
_TAG_MARKERS: tuple[str, ...] = (
    "[inst]",
    "[/inst]",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
    "</s>",
)

# Natural-language instruction-override / jailbreak phrasing.
_JAILBREAK_RE = re.compile(
    r"\b("
    r"ignore\s+(?:all\s+)?(?:your\s+|the\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|rules|directives|prompts)|"
    r"disregard\s+(?:your|the|all|any)\s+(?:safety|previous|system|prior)\s+\w+|"
    r"forget\s+(?:(?:all|everything|your|the|any)\s+)+(?:instructions|rules|directives|prompts|prior|previous)|"
    r"you\s+are\s+now\s+(?:a\s+|an\s+|in\s+)?\w+|"
    r"developer\s+mode|"
    r"jailbreak|"
    r"\bDAN\b|"
    r"override\s+(?:your\s+)?(?:safety|system|rules|guidelines|instructions)|"
    r"your\s+(?:real|true|actual)\s+(?:instructions|system\s+prompt|rules)|"
    r"bypass\s+(?:your\s+|the\s+)?(?:safety|validator|rules|guard)"
    r")\b",
    re.IGNORECASE,
)

# Attempts to forge a new system prompt / role boundary inside the body.
_SYS_OVERRIDE_RE = re.compile(
    r"(?:^|\n)\s*(?:system\s+prompt\s*:|new\s+system\s+prompt|begin\s+system\b)"
    r"|</?(?:system|assistant)\s*>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SkillScanResult:
    """Outcome of scanning one skill.

    Attributes:
        ok: True when the skill is safe to load. False -> quarantine.
        severity: ``CLEAN`` / ``WARNING`` / ``CRITICAL``.
        reasons: human-readable reasons (logged on quarantine).
    """

    ok: bool
    severity: str
    reasons: tuple[str, ...]


def _scan_text(label: str, text: Optional[str]) -> list[str]:
    """Return reason strings for any injection markers found in ``text``."""
    if not text:
        return []
    reasons: list[str] = []
    low = text.lower()
    for marker in _TAG_MARKERS:
        if marker in low:
            reasons.append(f"chat-template marker {marker!r} in {label}")
    if _JAILBREAK_RE.search(text):
        reasons.append(f"instruction-override / jailbreak phrasing in {label}")
    if _SYS_OVERRIDE_RE.search(text):
        reasons.append(f"system-prompt / role-boundary forgery in {label}")
    return reasons


def scan_skill_content(
    name: str,
    content: Optional[str],
    *,
    description: Optional[str] = None,
    frontmatter: Optional[Mapping[str, object]] = None,
) -> SkillScanResult:
    """Scan a skill's body + description for prompt-injection content.

    Pure function (no filesystem). Returns a CRITICAL result with reasons
    on any detection, else CLEAN.
    """
    reasons = _scan_text("skill body", content)
    reasons += _scan_text("skill description", description)
    if frontmatter:
        # Scan stringy frontmatter values too (a hostile skill could hide
        # the payload in an unconsumed frontmatter field that some future
        # consumer renders).
        for key, value in frontmatter.items():
            if isinstance(value, str):
                reasons += _scan_text(f"frontmatter field {key!r}", value)
    if reasons:
        return SkillScanResult(ok=False, severity=CRITICAL, reasons=tuple(reasons))
    return SkillScanResult(ok=True, severity=CLEAN, reasons=())


def _scan_companion_code(directory: Path) -> list[str]:
    """Run the install-time static scanner over a skill dir IFF it carries
    ``.py`` files. Returns reason strings for CRITICAL findings. Fail-open."""
    try:
        has_python = any(directory.glob("*.py"))
    except Exception:  # noqa: BLE001
        return []
    if not has_python:
        return []
    try:
        from ultron.install.static_scanner import scan_install_directory

        report = scan_install_directory(directory)
    except Exception as exc:  # noqa: BLE001 -- fail-open
        logger.debug("companion-code scan errored for %s: %r", directory, exc)
        return []
    reasons: list[str] = []
    for finding in getattr(report, "findings", ()) or ():
        sev = str(getattr(finding, "severity", "")).lower()
        if sev.endswith("critical"):
            kind = getattr(finding, "kind", "") or getattr(finding, "message", "")
            reasons.append(f"companion-code critical finding: {kind}")
    return reasons


def scan_skill(skill: object) -> SkillScanResult:
    """Scan a built :class:`ultron.skills.models.Skill` (content + any
    companion code in its directory).

    Returns a CRITICAL result (``ok=False``) on detection. A scanner error
    degrades to a WARNING result that is still ``ok=True`` (load proceeds) so
    a broken scanner never drops the whole catalog.
    """
    try:
        name = getattr(skill, "name", "") or ""
        content = getattr(skill, "content", "") or ""
        description = getattr(skill, "description", None)
        extra = getattr(skill, "extra", None)
        result = scan_skill_content(
            name, content, description=description, frontmatter=extra,
        )
        reasons = list(result.reasons)
        path = getattr(skill, "path", None)
        if path is not None:
            try:
                reasons += _scan_companion_code(Path(path).parent)
            except Exception:  # noqa: BLE001
                pass
        if reasons:
            return SkillScanResult(ok=False, severity=CRITICAL, reasons=tuple(reasons))
        return SkillScanResult(ok=True, severity=CLEAN, reasons=())
    except Exception as exc:  # noqa: BLE001 -- never let a scanner bug drop skills
        logger.debug("scan_skill errored (treating as clean): %r", exc)
        return SkillScanResult(ok=True, severity=WARNING, reasons=(f"scan error: {exc!r}",))


__all__ = [
    "CLEAN",
    "WARNING",
    "CRITICAL",
    "SkillScanResult",
    "scan_skill_content",
    "scan_skill",
]
