"""Skill discovery + frontmatter -> :class:`Skill` conversion.

Walks a directory, parses every ``.md`` file's YAML frontmatter via
:func:`ultron.parsing.parse_frontmatter` (T11), and turns the result into
a :class:`Skill` value object. Per-file errors are logged and skipped so
one malformed skill never breaks discovery -- the fail-open posture from
the frontmatter parser carries through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ultron.parsing import FrontmatterResult, parse_frontmatter, walk_directory_with_frontmatter
from ultron.skills.models import (
    KeywordTrigger,
    Skill,
    SkillSource,
    SkillType,
    TaskTrigger,
    Trigger,
)

logger = logging.getLogger(__name__)

_TASK_TRIGGER_PREFIX = "/"


@dataclass
class SkillLoadStats:
    """Per-directory load statistics for diagnostics."""

    directory: Path
    skills_loaded: int = 0
    skipped_no_name: int = 0
    skipped_parse_error: int = 0
    skipped_quarantined: int = 0
    files_scanned: int = 0
    errors: list[str] = field(default_factory=list)


def _coerce_triggers(value) -> list[str]:
    """Normalise a frontmatter ``triggers`` value to a list of strings.

    Accepts:
        * ``None`` -> ``[]``
        * ``str`` -> ``[value]``
        * ``list[str]`` -> the list
        * anything else -> ``[]`` (logged caller-side)
    """

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if isinstance(item, (str, int, float))]
    return []


def _build_trigger(
    triggers: list[str], *, default_min_chars: int
) -> Trigger | None:
    """Choose between :class:`TaskTrigger` and :class:`KeywordTrigger`.

    OpenHands convention: any trigger starting with ``/`` makes the WHOLE
    list a task trigger; otherwise it's a keyword trigger. Mixed lists
    (some slash, some plain) fall to task-trigger semantics to match the
    OpenHands shape -- the catalog called the split out explicitly.
    """

    if not triggers:
        return None

    if any(t.startswith(_TASK_TRIGGER_PREFIX) for t in triggers):
        return TaskTrigger(commands=tuple(triggers))

    return KeywordTrigger(
        keywords=tuple(triggers),
        min_user_text_chars=default_min_chars,
    )


def _coerce_skill_type(value, has_trigger: bool) -> SkillType:
    """Map a frontmatter ``type`` string to :class:`SkillType`."""

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"knowledge", "knowledge_v1", "kb"}:
            return SkillType.KNOWLEDGE
        if normalized in {"task", "command", "slash"}:
            return SkillType.TASK
        if normalized in {"always_on", "always-on", "ambient"}:
            return SkillType.ALWAYS_ON
    if not has_trigger:
        return SkillType.ALWAYS_ON
    return SkillType.KNOWLEDGE


def _skill_from_frontmatter(
    result: FrontmatterResult,
    *,
    source: SkillSource,
    default_min_user_text_chars: int,
) -> Skill | None:
    """Convert a :class:`FrontmatterResult` to a :class:`Skill`.

    Returns ``None`` when the file is structurally unusable as a skill
    (parse error AND no fallback name). When there's no frontmatter at
    all, the file is treated as an always-on skill using the filename
    stem as the name.
    """

    if not result.ok and result.frontmatter is None:
        # An error AND no parsed data. Try the filename stem.
        name = result.path.stem if result.path else None
        if not name:
            return None
        return Skill(
            name=name,
            content=result.body,
            trigger=None,
            source=source,
            type=SkillType.ALWAYS_ON,
            description=None,
            path=result.path,
            version=None,
            extra={"_load_error": result.error or "unknown"},
        )

    frontmatter = result.frontmatter or {}
    raw_name = frontmatter.get("name")
    if isinstance(raw_name, str):
        name = raw_name.strip()
    else:
        name = result.path.stem if result.path else ""
    if not name:
        logger.warning(
            "Skill at %s has no usable name; skipping", result.path
        )
        return None

    triggers_raw = _coerce_triggers(frontmatter.get("triggers"))
    if "triggers" in frontmatter and not triggers_raw:
        logger.warning(
            "Skill %s at %s has non-list 'triggers' value (%r); treating as always-on",
            name,
            result.path,
            frontmatter.get("triggers"),
        )

    min_user_text_chars = frontmatter.get("min_user_text_chars")
    if not isinstance(min_user_text_chars, int) or min_user_text_chars < 0:
        min_user_text_chars = default_min_user_text_chars

    trigger = _build_trigger(triggers_raw, default_min_chars=min_user_text_chars)
    skill_type = _coerce_skill_type(
        frontmatter.get("type"),
        has_trigger=trigger is not None,
    )

    description = frontmatter.get("description")
    if not isinstance(description, str):
        description = None

    version = frontmatter.get("version")
    if not isinstance(version, str):
        version = None

    # Pass through any non-consumed keys for forward-compat.
    consumed = {"name", "triggers", "type", "description", "version", "min_user_text_chars"}
    extra = {
        key: value
        for key, value in frontmatter.items()
        if key not in consumed
    }

    return Skill(
        name=name,
        content=result.body,
        trigger=trigger,
        source=source,
        type=skill_type,
        description=description,
        path=result.path,
        version=version,
        extra=extra,
    )


def load_skill_from_path(
    path: Path | str,
    *,
    source: SkillSource = SkillSource.OTHER,
    default_min_user_text_chars: int = 0,
) -> Skill | None:
    """Load a single skill from a file. Returns ``None`` on unusable input."""

    result = parse_frontmatter(path)
    return _skill_from_frontmatter(
        result,
        source=source,
        default_min_user_text_chars=default_min_user_text_chars,
    )


def load_skills_from_directory(
    directory: Path | str,
    *,
    source: SkillSource = SkillSource.OTHER,
    recursive: bool = True,
    default_min_user_text_chars: int = 0,
    skip_directories: Iterable[str] | None = None,
    skip_filenames: Iterable[str] | None = None,
    scan_untrusted: bool = True,
) -> tuple[list[Skill], SkillLoadStats]:
    """Walk ``directory`` and return a list of :class:`Skill` + load stats.

    Per-file failures are logged at WARN and recorded in
    :attr:`SkillLoadStats.errors`; they never raise.

    When ``scan_untrusted`` is True (default) AND ``source`` is not
    :attr:`SkillSource.PUBLIC`, each built skill is run through
    :func:`ultron.skills.scan.scan_skill` and QUARANTINED (skipped, counted
    in :attr:`SkillLoadStats.skipped_quarantined`, logged at WARN) on any
    prompt-injection / instruction-override detection. PUBLIC (ultron-
    shipped) skills are trusted and never scanned. Fail-open: a scanner
    error degrades to "clean" so the catalog is never silently emptied.
    """

    root = Path(directory)
    stats = SkillLoadStats(directory=root)
    skills: list[Skill] = []
    if not root.exists() or not root.is_dir():
        return skills, stats

    walker_kwargs: dict = {"recursive": recursive}
    if skip_directories is not None:
        walker_kwargs["skip_directories"] = skip_directories
    if skip_filenames is not None:
        walker_kwargs["skip_filenames"] = skip_filenames

    for result in walk_directory_with_frontmatter(root, **walker_kwargs):
        stats.files_scanned += 1
        try:
            skill = _skill_from_frontmatter(
                result,
                source=source,
                default_min_user_text_chars=default_min_user_text_chars,
            )
        except Exception as exc:
            stats.skipped_parse_error += 1
            error_message = f"{result.path}: unexpected error -- {exc!r}"
            stats.errors.append(error_message)
            logger.warning("Skill parse failure: %s", error_message)
            continue

        if skill is None:
            stats.skipped_no_name += 1
            if result.error:
                stats.errors.append(f"{result.path}: {result.error}")
            continue

        # Trust gate: scan skills from untrusted sources (USER / PROJECT /
        # OTHER -- e.g. ~/.ultron/skills, a project .ultron/skills, or the
        # autonomous data/evolution/skills dir) for prompt-injection content
        # BEFORE they can be injected into the system prompt. PUBLIC
        # (ultron-shipped) skills are trusted. Fail-open.
        if scan_untrusted and source != SkillSource.PUBLIC:
            try:
                from ultron.skills.scan import scan_skill

                scan_result = scan_skill(skill)
            except Exception as exc:  # noqa: BLE001 -- never drop on scanner bug
                scan_result = None
                logger.debug(
                    "skill scan errored for %s (treating as clean): %r",
                    skill.name, exc,
                )
            if scan_result is not None and not scan_result.ok:
                stats.skipped_quarantined += 1
                reasons = "; ".join(scan_result.reasons)
                stats.errors.append(f"{result.path}: quarantined ({reasons})")
                logger.warning(
                    "Quarantined untrusted skill %r from %s: %s",
                    skill.name, result.path, reasons,
                )
                continue

        skills.append(skill)
        stats.skills_loaded += 1

    return skills, stats
