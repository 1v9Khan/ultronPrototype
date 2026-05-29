"""Skill registry: discovery, mtime-invalidated cache, match + render.

The registry holds the loaded skill catalog and answers
``matching_skills(user_text)`` queries. It also produces the
``[Skills: name1, name2]\\n<content>\\n`` block that the orchestrator
prepends to the system prompt when the LLM is invoked.

The :class:`SkillRegistry` is intentionally simple:

* It walks one OR MORE directories at construction time.
* Per-directory it tracks an mtime fingerprint so a touched file
  triggers a partial reload on the next ``matching_skills`` call.
* Skills carry a :class:`SkillSource`; on duplicate name the higher-
  precedence source wins (project > user > public). Within the same
  source, last-loaded wins (mirrors the OpenHands ``_merge_skills``).

A module-level singleton accessor mirrors ``ultron.desktop.vlm``'s
pattern. Tests use :func:`reset_skill_registry_for_testing` to drop
the singleton between cases.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ultron.skills.models import (
    KeywordTrigger,
    Skill,
    SkillMatch,
    SkillSource,
    SkillType,
    TaskTrigger,
    find_matched_commands,
    find_matched_keywords,
)
from ultron.skills.loader import (
    SkillLoadStats,
    load_skills_from_directory,
)

logger = logging.getLogger(__name__)

DEFAULT_KEYWORD_MIN_USER_TEXT_CHARS = 8
"""Default min-chars guard for keyword triggers loaded without an explicit override.

Catches the "user says 'ssh' as an interjection and the ssh skill loads
stale ops guidance" foot-gun the OpenHands catalog called out.
"""

DEFAULT_PUBLIC_SKILLS_DIRNAME = "skills"
DEFAULT_USER_SKILLS_DIR_NAME = ".ultron/skills"
DEFAULT_PROJECT_SKILLS_DIRNAME = ".ultron/skills"


@dataclass
class _SourceSpec:
    directory: Path
    source: SkillSource


def _directory_mtime_fingerprint(directory: Path) -> tuple[float, int]:
    """Return ``(latest_mtime, file_count)`` for cache invalidation.

    Walks the directory once. A change to any file (added / removed /
    modified) flips one of the two values, which is enough for the
    registry to know "reload me".
    """

    latest_mtime = 0.0
    file_count = 0
    try:
        for candidate in directory.rglob("*"):
            try:
                if not candidate.is_file():
                    continue
                if candidate.suffix.lower() != ".md":
                    continue
                stat = candidate.stat()
            except OSError:
                continue
            file_count += 1
            if stat.st_mtime > latest_mtime:
                latest_mtime = stat.st_mtime
    except OSError:
        pass
    return latest_mtime, file_count


class SkillRegistry:
    """In-process registry of loaded skills with mtime invalidation."""

    def __init__(
        self,
        sources: Sequence[_SourceSpec] | None = None,
        *,
        disabled_skills: Iterable[str] | None = None,
        default_min_user_text_chars: int = DEFAULT_KEYWORD_MIN_USER_TEXT_CHARS,
        always_on_only: bool = False,
        max_matches_per_turn: int = 6,
        scan_untrusted: bool = True,
    ) -> None:
        self._sources: list[_SourceSpec] = list(sources or [])
        self._disabled_skills: set[str] = {s.lower() for s in (disabled_skills or [])}
        self._default_min_user_text_chars = default_min_user_text_chars
        self._always_on_only = always_on_only
        self._max_matches_per_turn = max(0, max_matches_per_turn)
        self._scan_untrusted = bool(scan_untrusted)

        self._lock = threading.RLock()
        self._cache: dict[str, Skill] = {}
        self._fingerprints: dict[Path, tuple[float, int]] = {}
        self._last_load_stats: list[SkillLoadStats] = []
        self._loaded_at: float = 0.0

    # -- catalog management --

    def add_source(self, directory: Path | str, *, source: SkillSource) -> None:
        """Register an additional source directory.

        Future calls to :meth:`matching_skills` / :meth:`reload` pick it up.
        """

        spec = _SourceSpec(directory=Path(directory), source=source)
        with self._lock:
            self._sources.append(spec)
            # Drop any cached fingerprint for the new dir so the next
            # invalidation check sees fresh state.
            self._fingerprints.pop(spec.directory, None)

    def set_disabled_skills(self, names: Iterable[str]) -> None:
        with self._lock:
            self._disabled_skills = {s.lower() for s in names}

    @property
    def disabled_skills(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._disabled_skills)

    @property
    def sources(self) -> tuple[Path, ...]:
        with self._lock:
            return tuple(spec.directory for spec in self._sources)

    @property
    def loaded_at(self) -> float:
        with self._lock:
            return self._loaded_at

    @property
    def last_load_stats(self) -> list[SkillLoadStats]:
        with self._lock:
            return list(self._last_load_stats)

    # -- loading --

    def reload(self) -> list[SkillLoadStats]:
        """Force a fresh walk of every source directory.

        Returns the per-source stats. The catalog becomes whatever the
        walk discovers; previously-loaded skills not on disk are
        dropped.
        """

        with self._lock:
            new_cache: dict[str, Skill] = {}
            stats_list: list[SkillLoadStats] = []
            new_fingerprints: dict[Path, tuple[float, int]] = {}

            for spec in self._sources:
                skills, stats = load_skills_from_directory(
                    spec.directory,
                    source=spec.source,
                    default_min_user_text_chars=self._default_min_user_text_chars,
                    scan_untrusted=self._scan_untrusted,
                )
                stats_list.append(stats)
                new_fingerprints[spec.directory] = _directory_mtime_fingerprint(spec.directory)
                for skill in skills:
                    existing = new_cache.get(skill.name.lower())
                    if existing is None:
                        new_cache[skill.name.lower()] = skill
                        continue
                    if existing.source.precedence < skill.source.precedence:
                        new_cache[skill.name.lower()] = skill
                    elif existing.source.precedence == skill.source.precedence:
                        # Within the same source class, last wins (matches
                        # OpenHands ``_merge_skills`` semantics).
                        new_cache[skill.name.lower()] = skill

            self._cache = new_cache
            self._fingerprints = new_fingerprints
            self._last_load_stats = stats_list
            self._loaded_at = time.time()
            return stats_list

    def _maybe_reload_if_stale(self) -> None:
        """Reload only when any source directory's mtime fingerprint changed."""

        with self._lock:
            if self._loaded_at == 0.0:
                self.reload()
                return
            for spec in self._sources:
                fingerprint = _directory_mtime_fingerprint(spec.directory)
                if self._fingerprints.get(spec.directory) != fingerprint:
                    self.reload()
                    return

    # -- introspection --

    def all_skills(self) -> list[Skill]:
        """Return the deduped skill catalog."""

        self._maybe_reload_if_stale()
        with self._lock:
            return list(self._cache.values())

    def list_skill_names(self) -> list[str]:
        return sorted(skill.name for skill in self.all_skills())

    # -- matching --

    def matching_skills(
        self,
        user_text: str,
        *,
        mode: str = "standby",
        gaming_mode: bool | None = None,
        vlm_loaded: bool = True,
        has_internet: bool = True,
    ) -> list[SkillMatch]:
        """Return the skills the registry would inject for ``user_text``.

        Always-on skills are always included (the orchestrator can
        choose to skip them by setting ``always_on_only=False``).
        Trigger-matching skills add their match if and only if the
        trigger matches AND the skill isn't on the disabled list.

        2026-05-26 (openclaw-clawhub catalog T5 wiring):
        ``mode`` filters out skills whose frontmatter ``modes`` list
        excludes the current mode. ``"standby"`` (the default) +
        ``"gaming"`` are the two ultron modes today. A skill with no
        ``modes`` declaration matches every mode (legacy
        compatibility). Useful for gaming-mode where a coding skill
        with a heavy system prompt would burn budget the user doesn't
        want spent.

        2026 (catalog 07 T4 wiring): ``capability_tags`` in the
        frontmatter filters skills against the current runtime
        capability context.

        * ``gaming_mode`` (when ``True``, or when ``None`` and
          ``mode == "gaming"``): drops skills with any tag in
          :data:`ultron.skills.capability_tags.GAMING_MODE_INCOMPATIBLE_TAGS`.
          Default ``None`` derives the flag from ``mode``.
        * ``vlm_loaded=False``: drops skills tagged
          :attr:`CapabilityTag.REQUIRES_VLM`.
        * ``has_internet=False``: drops skills tagged
          :attr:`CapabilityTag.REQUIRES_INTERNET`.

        Skills with no ``capability_tags`` declaration pass every
        capability check (legacy / unscoped). Unknown tag strings
        are ignored (forward-compatible with future tag additions).
        Fail-open if the capability_tags module can't be imported.

        At most :attr:`_max_matches_per_turn` non-always-on skills are
        returned, in source-precedence order then by name.
        """

        self._maybe_reload_if_stale()
        normalised_mode = (mode or "standby").lower().strip()
        if gaming_mode is None:
            gaming_mode = normalised_mode == "gaming"
        results: list[SkillMatch] = []
        triggered_results: list[SkillMatch] = []
        with self._lock:
            for skill in self._cache.values():
                if skill.name.lower() in self._disabled_skills:
                    continue
                if not _skill_active_in_mode(skill, normalised_mode):
                    continue
                if not _skill_active_for_capability_tags(
                    skill,
                    gaming_mode=bool(gaming_mode),
                    vlm_loaded=vlm_loaded,
                    has_internet=has_internet,
                ):
                    continue
                if skill.is_always_on:
                    results.append(SkillMatch(skill=skill, matched_terms=()))
                    continue
                if self._always_on_only:
                    continue
                match_terms = self._terms_matching(skill, user_text)
                if not match_terms:
                    continue
                triggered_results.append(SkillMatch(skill=skill, matched_terms=match_terms))

        triggered_results.sort(
            key=lambda m: (-m.skill.source.precedence, m.skill.name.lower())
        )
        if self._max_matches_per_turn:
            triggered_results = triggered_results[: self._max_matches_per_turn]
        return results + triggered_results

    @staticmethod
    def _terms_matching(skill: Skill, user_text: str) -> tuple[str, ...]:  # noqa: D401
        return _terms_matching_inner(skill, user_text)


def _skill_active_in_mode(skill: Skill, mode: str) -> bool:
    """Return True iff ``skill`` should be active in ``mode``.

    Reads the optional ``modes`` list from ``skill.extra`` (sourced
    from frontmatter at load time). A skill with no declaration
    matches every mode (legacy / unscoped). When the declaration is
    present it must contain ``mode`` (case-insensitive) for the
    skill to be active.
    """
    extra = getattr(skill, "extra", None) or {}
    raw = extra.get("modes")
    if raw is None:
        return True
    if isinstance(raw, str):
        # Tolerate a single string in addition to the list form.
        modes = [raw]
    elif isinstance(raw, (list, tuple)):
        modes = list(raw)
    else:
        # Unknown shape -> fail-open (skill stays active).
        return True
    normalised = {str(m).lower().strip() for m in modes if m is not None}
    if not normalised:
        return True
    return mode.lower().strip() in normalised


def _coerce_capability_tag_strings(raw: object) -> list[str]:
    """Normalise a frontmatter ``capability_tags`` value to a list of strings.

    Accepts:

    * ``None`` -> ``[]``
    * a single string -> wrapped in a one-element list
    * a list / tuple of strings or stringifiable values -> stringified
      and stripped, dropping empty entries
    * any other shape -> ``[]`` (fail-open; the predicate then treats
      the skill as unrestricted)
    """

    if raw is None:
        return []
    if isinstance(raw, str):
        stripped = raw.strip()
        return [stripped] if stripped else []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _skill_active_for_capability_tags(
    skill: Skill,
    *,
    gaming_mode: bool,
    vlm_loaded: bool,
    has_internet: bool,
) -> bool:
    """Return True iff ``skill`` survives the current capability context.

    Reads the optional ``capability_tags`` list from
    ``skill.extra`` (sourced from frontmatter at load time). Skills
    with no declaration -- the common case for the six built-in
    skills -- pass every check (legacy / unscoped).

    The filter rules mirror
    :func:`ultron.skills.capability_tags.filter_capabilities`:

    * Gaming mode drops skills tagged with anything in
      :data:`GAMING_MODE_INCOMPATIBLE_TAGS`.
    * ``vlm_loaded=False`` drops skills tagged
      :attr:`REQUIRES_VLM`.
    * ``has_internet=False`` drops skills tagged
      :attr:`REQUIRES_INTERNET`.

    Unknown tag strings are silently ignored so adding tags upstream
    is forward-compatible. The capability_tags module is imported
    lazily so the skill-registry import doesn't pay its cost when
    capability filtering isn't engaged.
    """

    extra = getattr(skill, "extra", None) or {}
    raw_tags = _coerce_capability_tag_strings(extra.get("capability_tags"))
    if not raw_tags:
        return True

    try:
        from ultron.skills.capability_tags import (
            CapabilityTag,
            GAMING_MODE_INCOMPATIBLE_TAGS,
        )
    except Exception as exc:  # noqa: BLE001  -- fail-open
        logger.debug("capability_tags filter skipped: %s", exc)
        return True

    tags: set[CapabilityTag] = set()
    for value in raw_tags:
        try:
            tags.add(CapabilityTag(value))
        except ValueError:
            # Unknown tag -- ignore (forward-compatible with upstream
            # additions before the enum knows about them).
            continue

    if not tags:
        return True

    if gaming_mode and (tags & GAMING_MODE_INCOMPATIBLE_TAGS):
        return False
    if not vlm_loaded and CapabilityTag.REQUIRES_VLM in tags:
        return False
    if not has_internet and CapabilityTag.REQUIRES_INTERNET in tags:
        return False
    return True


def _terms_matching_inner(skill: Skill, user_text: str) -> tuple[str, ...]:
    trigger = skill.trigger
    if isinstance(trigger, KeywordTrigger):
        if user_text and len(user_text) < trigger.min_user_text_chars:
            return ()
        return find_matched_keywords(user_text, trigger.keywords)
    if isinstance(trigger, TaskTrigger):
        return find_matched_commands(user_text, trigger.commands)
    return ()


# -- module-level singleton accessor --


_REGISTRY: SkillRegistry | None = None
_REGISTRY_LOCK = threading.RLock()


def get_skill_registry() -> SkillRegistry | None:
    """Return the process-wide :class:`SkillRegistry`, or ``None``."""

    with _REGISTRY_LOCK:
        return _REGISTRY


def set_skill_registry(registry: SkillRegistry | None) -> None:
    """Replace the process-wide registry.

    Pass ``None`` to clear (effectively disables skill injection until
    a fresh registry is set).
    """

    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = registry


def reset_skill_registry_for_testing() -> None:
    """Test escape hatch: clear the module-level singleton."""

    set_skill_registry(None)


# -- rendering --


def format_skills_block(
    matches: Sequence[SkillMatch],
    *,
    leading_label: str = "Skills",
    include_descriptions: bool = False,
    max_chars: int | None = None,
) -> str:
    """Render matched skills as a system-prompt-injectable block.

    Layout (matches the catalog suggestion):

    ::

        [Skills: gaming, coding]

        # gaming
        <body...>

        # coding
        <body...>

    Returns an empty string when ``matches`` is empty.
    """

    if not matches:
        return ""

    header_names = ", ".join(match.skill.name for match in matches)
    parts: list[str] = [f"[{leading_label}: {header_names}]", ""]
    for match in matches:
        title = f"# {match.skill.name}"
        if include_descriptions and match.skill.description:
            title += f" -- {match.skill.description}"
        parts.append(title)
        body = match.content.rstrip()
        if body:
            parts.append(body)
        parts.append("")
    text = "\n".join(parts).rstrip() + "\n"

    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        truncated = text[: max_chars - 32].rstrip()
        text = truncated + "\n\n... <skills truncated>\n"
    return text


# -- one-call orchestrator helper --


def maybe_get_skills_block(
    user_text: str,
    *,
    leading_label: str = "Skills",
    max_chars: int | None = None,
    mode: str = "standby",
    gaming_mode: bool | None = None,
    vlm_loaded: bool = True,
    has_internet: bool = True,
) -> str:
    """Return the formatted skills block for ``user_text`` (or empty).

    Convenience for callers that just want "give me the system-prompt
    addendum if any skills match, otherwise nothing". When no module-
    level registry is set OR the registry returns no matches, returns
    ``""`` so callers can concatenate unconditionally.

    ``mode`` (default ``"standby"``) is forwarded to
    :meth:`SkillRegistry.matching_skills` so the caller can ask for
    the gaming-mode-only catalog when ``GamingModeManager`` is engaged.

    ``gaming_mode`` / ``vlm_loaded`` / ``has_internet`` are forwarded
    to :meth:`SkillRegistry.matching_skills` (catalog 07 T4) so skills
    whose frontmatter ``capability_tags`` list excludes the current
    capability context are filtered out. ``gaming_mode=None`` (the
    default) defers to the ``mode`` argument.

    Every error is swallowed (fail-open) -- a malformed skill catalog
    must never break the voice loop. The catalog's `min_user_text_chars`
    guard is respected per-skill.
    """

    try:
        registry = get_skill_registry()
        if registry is None:
            return ""
        matches = registry.matching_skills(
            user_text,
            mode=mode,
            gaming_mode=gaming_mode,
            vlm_loaded=vlm_loaded,
            has_internet=has_internet,
        )
        if not matches:
            return ""
        return format_skills_block(
            matches, leading_label=leading_label, max_chars=max_chars
        )
    except Exception as exc:
        logger.warning("skills block construction failed: %r", exc)
        return ""


# -- default registry construction --


def build_default_registry(
    *,
    project_root: Path | str,
    user_home: Path | str | None = None,
    extra_project_dirs: Iterable[Path | str] | None = None,
    disabled_skills: Iterable[str] | None = None,
    always_on_only: bool = False,
    default_min_user_text_chars: int = DEFAULT_KEYWORD_MIN_USER_TEXT_CHARS,
    max_matches_per_turn: int = 6,
    scan_untrusted: bool = True,
) -> SkillRegistry:
    """Construct a :class:`SkillRegistry` wired to the three default sources.

    Args:
        project_root: Project root (used to locate ``skills/`` and
            ``.ultron/skills/``).
        user_home: Override for the user home; defaults to ``Path.home()``.
        extra_project_dirs: Additional directories to scan with
            :attr:`SkillSource.PROJECT` precedence.
        disabled_skills: Skill names to suppress.
        always_on_only: When True, the registry only emits always-on
            skills (debug toggle).
        default_min_user_text_chars: Keyword-trigger guard floor.
        max_matches_per_turn: Cap on the number of triggered skills the
            registry emits per ``matching_skills`` call.

    Returns:
        A :class:`SkillRegistry` ready for use. Reload is deferred to
        the first :meth:`matching_skills` call (lazy).
    """

    project_root_path = Path(project_root)
    home_path = Path(user_home) if user_home else Path.home()

    public_dir = project_root_path / DEFAULT_PUBLIC_SKILLS_DIRNAME
    user_dir = home_path / DEFAULT_USER_SKILLS_DIR_NAME
    project_skill_dir = project_root_path / DEFAULT_PROJECT_SKILLS_DIRNAME

    sources: list[_SourceSpec] = []
    sources.append(_SourceSpec(directory=public_dir, source=SkillSource.PUBLIC))
    sources.append(_SourceSpec(directory=user_dir, source=SkillSource.USER))
    sources.append(_SourceSpec(directory=project_skill_dir, source=SkillSource.PROJECT))
    if extra_project_dirs:
        for extra in extra_project_dirs:
            sources.append(_SourceSpec(directory=Path(extra), source=SkillSource.PROJECT))

    return SkillRegistry(
        sources,
        disabled_skills=disabled_skills,
        default_min_user_text_chars=default_min_user_text_chars,
        always_on_only=always_on_only,
        max_matches_per_turn=max_matches_per_turn,
        scan_untrusted=scan_untrusted,
    )
