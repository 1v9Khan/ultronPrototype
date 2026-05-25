"""Conditional rule evaluation with frontmatter ``paths`` / ``intents`` / ``topics``.

Adapted from cline's ``evaluateRuleConditionals`` pattern (Apache 2.0;
see ``THIRD_PARTY_NOTICES.md``). Each rule is a markdown file with a
YAML frontmatter block; the engine parses the block, evaluates every
declared condition against a per-turn context, and yields a
:class:`RuleActivation` for each match. The activations carry the
``matched_conditions`` set so the orchestrator can render
``<rule_activated source="..." matched="paths,intents">...</rule_activated>``
blocks for the LLM.

Conditions are intentionally simple — string globs, substring tests,
small comparator strings — to keep evaluation under a millisecond per
rule. The path-extraction heuristic mirrors cline's: strip fenced code
blocks, strip URLs, then match identifier-with-slash and
identifier-with-dot-extension patterns. Token length is capped so a
mis-extracted 1024-character path cannot blow up the matcher.
"""

from __future__ import annotations

import logging
import operator
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from ultron.parsing.frontmatter import parse_frontmatter

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum length of a path-like token recovered from a transcript.
MAX_EXTRACTED_TOKEN_LENGTH: int = 256

#: Minimum length of a path-like token (filters single-letter noise).
MIN_EXTRACTED_TOKEN_LENGTH: int = 3

#: File-extension regex used for the "identifier.ext" fallback path match.
_EXT_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9_.\-]+\.[A-Za-z0-9]{1,10}\b",
)

#: Slash-segmented path regex (``foo/bar`` or ``foo/bar/baz.py``).
_SLASH_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+\b",
)

#: Fenced code-block strip pattern (triple-backtick fences only).
_CODE_FENCE_PATTERN: re.Pattern[str] = re.compile(
    r"```[\s\S]*?```|`[^`\n]*`",
)

#: URL strip pattern.
_URL_PATTERN: re.Pattern[str] = re.compile(
    r"https?://\S+|ftp://\S+",
)

#: Mapping of comparator prefix -> operator.
_COMPARATORS: dict[str, Any] = {
    ">=": operator.ge,
    "<=": operator.le,
    "!=": operator.ne,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,
}


# ---------------------------------------------------------------------------
# Path extraction heuristic
# ---------------------------------------------------------------------------

def extract_path_like_strings(text: str) -> set[str]:
    """Recover candidate path tokens from a free-form transcript.

    Strips fenced code blocks + URLs first, then collects any token
    that looks like a slash-segmented path OR an identifier with a
    1-10-char file extension. Tokens are length-capped at
    :data:`MAX_EXTRACTED_TOKEN_LENGTH` and de-duplicated.

    Args:
        text: arbitrary transcript / user message text.

    Returns:
        Deterministic-iteration set of candidate tokens.
    """
    if not text:
        return set()
    cleaned = _CODE_FENCE_PATTERN.sub(" ", text)
    cleaned = _URL_PATTERN.sub(" ", cleaned)
    tokens: set[str] = set()
    for match in _SLASH_TOKEN_PATTERN.findall(cleaned):
        token = match.strip("/.")
        if not token:
            continue
        if MIN_EXTRACTED_TOKEN_LENGTH <= len(token) <= MAX_EXTRACTED_TOKEN_LENGTH:
            tokens.add(token)
    for match in _EXT_TOKEN_PATTERN.findall(cleaned):
        if MIN_EXTRACTED_TOKEN_LENGTH <= len(match) <= MAX_EXTRACTED_TOKEN_LENGTH:
            tokens.add(match)
    return tokens


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleEvaluationContext:
    """Per-turn context the evaluator matches against.

    Attributes:
        user_text: raw user transcript (no brevity-hint preamble).
        intent_kind: stringified ``RoutingIntentKind`` value, or empty.
        intent_label: optional broader intent label (used as a topic match).
        topics: optional iterable of detected topic strings (RAG topic ids,
            ConversationMemory `Channel`-tagged topics).
        system_state: mapping of state keys ultron exposes per-turn:
            ``gaming_mode``, ``n_active_skills``, ``coding_in_progress``,
            ``hour``, ``minute``, ``rss_mb``, ``vram_used_mb``, etc.
        extra_paths: optional extra path tokens to add to the candidate
            set (callers that already know the file context can pass
            ``("src/foo/bar.py",)`` without relying on the heuristic).
    """

    user_text: str = ""
    intent_kind: str = ""
    intent_label: str = ""
    topics: Sequence[str] = field(default_factory=tuple)
    system_state: Mapping[str, Any] = field(default_factory=dict)
    extra_paths: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ConditionalRule:
    """One parsed rule file with metadata + body.

    Attributes:
        name: stable identifier (defaults to the file stem if
            frontmatter omits ``name``).
        source_path: path on disk the rule was loaded from.
        body: the markdown body AFTER the frontmatter block.
        conditions: parsed frontmatter mapping (the raw ``paths`` /
            ``intents`` / etc.).
        source_layer: ``"global"`` / ``"project"`` / ``"workspace"`` /
            ``"skill"`` — useful for telemetry + dedup precedence.
    """

    name: str
    source_path: Path
    body: str
    conditions: Mapping[str, Any]
    source_layer: str = "project"


@dataclass(frozen=True)
class RuleActivation:
    """The result of activating a rule for one turn.

    Attributes:
        rule: the underlying rule.
        matched_conditions: set of condition kinds (``"paths"``,
            ``"intents"``, ``"topics"``, ``"system_state"``, ``"all_of"``,
            ``"always"``) that fired.
        matched_values: per-condition match details for debugging.
    """

    rule: ConditionalRule
    matched_conditions: frozenset[str]
    matched_values: Mapping[str, Sequence[str]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.rule.name

    @property
    def body(self) -> str:
        return self.rule.body


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

def _matches_path_conditions(
    candidate_paths: Iterable[str],
    patterns: Iterable[str],
) -> list[str]:
    """Return the subset of ``candidate_paths`` matching any glob."""
    try:
        import fnmatch
    except ImportError:  # pragma: no cover
        return []
    matched: list[str] = []
    pat_list = [p.strip() for p in patterns if isinstance(p, str) and p.strip()]
    if not pat_list:
        return []
    for candidate in candidate_paths:
        normalised = candidate.replace("\\", "/")
        for pattern in pat_list:
            normalised_pat = pattern.replace("\\", "/")
            if fnmatch.fnmatchcase(normalised, normalised_pat):
                matched.append(candidate)
                break
    return matched


def _matches_intent_conditions(
    intent_kind: str, allowed: Iterable[str],
) -> Optional[str]:
    """Return the matched intent value when ``intent_kind`` is allowed."""
    if not intent_kind:
        return None
    upper_kind = intent_kind.upper()
    for entry in allowed:
        if not isinstance(entry, str):
            continue
        if entry.upper() == upper_kind:
            return entry
    return None


def _matches_topic_conditions(
    text: str, topics: Sequence[str], patterns: Iterable[str],
) -> list[str]:
    """Return the topic / substring / regex patterns that fire."""
    matched: list[str] = []
    lowered_text = (text or "").lower()
    lowered_topics = [t.lower() for t in topics]
    for pattern in patterns:
        if not isinstance(pattern, str):
            continue
        compiled: Optional[re.Pattern[str]] = None
        bare = pattern.strip()
        if not bare:
            continue
        # Treat patterns wrapped in slashes as regex (``/foo.*/``).
        if bare.startswith("/") and bare.endswith("/") and len(bare) > 2:
            try:
                compiled = re.compile(bare[1:-1], re.IGNORECASE)
            except re.error:
                compiled = None
        if compiled is not None:
            if compiled.search(text or ""):
                matched.append(bare)
                continue
            continue
        lowered_pat = bare.lower()
        if lowered_pat in lowered_text or lowered_pat in lowered_topics:
            matched.append(bare)
    return matched


def _coerce_number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matches_state_clause(state_value: Any, requirement: Any) -> bool:
    """Evaluate one ``system_state`` clause."""
    if isinstance(requirement, bool):
        return bool(state_value) is requirement
    if isinstance(requirement, (int, float)):
        actual = _coerce_number(state_value)
        return actual is not None and actual == float(requirement)
    if isinstance(requirement, str):
        text = requirement.strip()
        # Comparator string: ``">=2"`` / ``"<10"`` / ``"=production"``.
        for prefix, op in _COMPARATORS.items():
            if text.startswith(prefix):
                payload = text[len(prefix):].strip()
                requirement_num = _coerce_number(payload)
                actual_num = _coerce_number(state_value)
                if requirement_num is not None and actual_num is not None:
                    return bool(op(actual_num, requirement_num))
                return bool(op(str(state_value), payload))
        # Plain string: exact (case-insensitive) match.
        return str(state_value).lower() == text.lower()
    if isinstance(requirement, list):
        return any(_matches_state_clause(state_value, r) for r in requirement)
    return False


def _matches_state_conditions(
    state: Mapping[str, Any],
    clauses: Mapping[str, Any],
) -> list[str]:
    """Return the list of state keys that satisfied their requirement."""
    matched: list[str] = []
    if not clauses:
        return matched
    for key, requirement in clauses.items():
        if not _matches_state_clause(state.get(key), requirement):
            return []  # all_of semantics — bail on first mismatch
        matched.append(f"{key}={requirement}")
    return matched


def evaluate_rule(
    rule: ConditionalRule,
    context: RuleEvaluationContext,
) -> Optional[RuleActivation]:
    """Evaluate ``rule`` against ``context`` and return an activation or None.

    A rule activates when at least one declared condition fires. Rules
    that declare no conditions are always active (the "global rule" case).
    Empty condition lists (``paths: []``) explicitly DEACTIVATE the rule.

    Args:
        rule: the rule to evaluate.
        context: per-turn context.

    Returns:
        :class:`RuleActivation` when the rule should be injected, else
        ``None``.
    """
    conditions = rule.conditions or {}
    matched: dict[str, list[str]] = {}
    declared_count = 0
    deactivated = False

    # ``paths`` condition.
    if "paths" in conditions:
        declared_count += 1
        path_patterns = conditions["paths"]
        if isinstance(path_patterns, list):
            if not path_patterns:
                deactivated = True
            else:
                candidates: set[str] = set(context.extra_paths)
                candidates.update(extract_path_like_strings(context.user_text))
                hits = _matches_path_conditions(candidates, path_patterns)
                if hits:
                    matched["paths"] = hits

    # ``intents`` condition.
    if "intents" in conditions and not deactivated:
        declared_count += 1
        intent_list = conditions["intents"]
        if isinstance(intent_list, list):
            if not intent_list:
                deactivated = True
            else:
                hit = _matches_intent_conditions(context.intent_kind, intent_list)
                if hit:
                    matched["intents"] = [hit]

    # ``topics`` condition.
    if "topics" in conditions and not deactivated:
        declared_count += 1
        topic_patterns = conditions["topics"]
        if isinstance(topic_patterns, list):
            if not topic_patterns:
                deactivated = True
            else:
                hits = _matches_topic_conditions(
                    context.user_text, list(context.topics), topic_patterns,
                )
                if hits:
                    matched["topics"] = hits

    # ``system_state`` condition.
    if "system_state" in conditions and not deactivated:
        declared_count += 1
        clauses = conditions["system_state"]
        if isinstance(clauses, Mapping):
            if not clauses:
                deactivated = True
            else:
                hits = _matches_state_conditions(context.system_state, clauses)
                if hits:
                    matched["system_state"] = hits

    # ``not_in_gaming_mode`` convenience inverse.
    if "not_in_gaming_mode" in conditions and not deactivated:
        declared_count += 1
        wants_off = bool(conditions["not_in_gaming_mode"])
        gaming = bool(context.system_state.get("gaming_mode", False))
        if wants_off and not gaming:
            matched.setdefault("system_state", []).append("not_in_gaming_mode=true")

    # ``all_of`` combinator — every nested block must yield an activation.
    if "all_of" in conditions and not deactivated:
        declared_count += 1
        nested_blocks = conditions["all_of"]
        if isinstance(nested_blocks, list) and nested_blocks:
            inner_matches: list[str] = []
            for block in nested_blocks:
                if not isinstance(block, Mapping):
                    inner_matches = []
                    break
                inner_rule = ConditionalRule(
                    name=f"{rule.name}.all_of",
                    source_path=rule.source_path,
                    body="",
                    conditions=block,
                    source_layer=rule.source_layer,
                )
                inner = evaluate_rule(inner_rule, context)
                if inner is None:
                    inner_matches = []
                    break
                inner_matches.extend(inner.matched_conditions)
            if inner_matches:
                matched["all_of"] = inner_matches

    if deactivated:
        return None

    if declared_count == 0:
        # No conditions declared → always-on.
        return RuleActivation(
            rule=rule,
            matched_conditions=frozenset({"always"}),
            matched_values={"always": ("declared no conditions",)},
        )

    if not matched:
        return None

    return RuleActivation(
        rule=rule,
        matched_conditions=frozenset(matched.keys()),
        matched_values={k: tuple(v) for k, v in matched.items()},
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_rule_from_path(
    path: Path,
    *,
    source_layer: str = "project",
) -> Optional[ConditionalRule]:
    """Parse one ``.md`` file into a :class:`ConditionalRule`.

    Args:
        path: rule file path.
        source_layer: precedence layer label.

    Returns:
        :class:`ConditionalRule` or ``None`` when the file cannot be
        parsed (the engine logs WARN and skips).
    """
    if not path.is_file():
        return None
    parse = parse_frontmatter(path)
    if parse.error:
        LOGGER.warning("rule %s frontmatter error: %s", path, parse.error)
        return None
    conditions = parse.frontmatter or {}
    name = str(conditions.get("name") or path.stem)
    return ConditionalRule(
        name=name,
        source_path=path.resolve(),
        body=(parse.body or "").strip(),
        conditions=conditions,
        source_layer=source_layer,
    )


@dataclass(frozen=True)
class ConditionalRuleSet:
    """Collection of loaded rules keyed by name (later layers win).

    Attributes:
        rules: name -> rule mapping.
        layers: per-layer rule names for telemetry.
    """

    rules: Mapping[str, ConditionalRule]
    layers: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def evaluate(
        self, context: RuleEvaluationContext,
    ) -> list[RuleActivation]:
        """Evaluate every rule against ``context`` and return activations.

        Args:
            context: per-turn evaluation context.

        Returns:
            List of activated :class:`RuleActivation` objects in
            insertion order.
        """
        out: list[RuleActivation] = []
        for rule in self.rules.values():
            activation = evaluate_rule(rule, context)
            if activation is not None:
                out.append(activation)
        return out

    def names(self) -> list[str]:
        return list(self.rules.keys())


def load_rule_set(
    directories: Sequence[tuple[Path, str]],
    *,
    extensions: tuple[str, ...] = (".md", ".markdown"),
) -> ConditionalRuleSet:
    """Load every rule file across the supplied ``(directory, layer)`` pairs.

    Args:
        directories: ordered list of ``(directory, layer_label)`` tuples;
            later directories override earlier ones (so global -> project
            -> workspace -> skill matches the precedence catalog says we
            want).
        extensions: tuple of file extensions to load (lower-cased).

    Returns:
        :class:`ConditionalRuleSet` with the merged rule mapping.
    """
    rules: dict[str, ConditionalRule] = {}
    layers: dict[str, list[str]] = {}
    for directory, label in directories:
        if not directory.is_dir():
            continue
        seen_for_layer: list[str] = []
        for entry in sorted(directory.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in extensions:
                continue
            rule = load_rule_from_path(entry, source_layer=label)
            if rule is None:
                continue
            rules[rule.name] = rule
            seen_for_layer.append(rule.name)
        layers[label] = seen_for_layer
    return ConditionalRuleSet(
        rules=rules,
        layers={k: tuple(v) for k, v in layers.items()},
    )


__all__ = [
    "MAX_EXTRACTED_TOKEN_LENGTH",
    "MIN_EXTRACTED_TOKEN_LENGTH",
    "ConditionalRule",
    "ConditionalRuleSet",
    "RuleActivation",
    "RuleEvaluationContext",
    "evaluate_rule",
    "extract_path_like_strings",
    "load_rule_from_path",
    "load_rule_set",
]
