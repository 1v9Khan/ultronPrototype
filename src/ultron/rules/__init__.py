"""Conditional rule activation with frontmatter ``paths`` / ``intents`` / etc.

This package implements the activation engine for ``.ultron/rules/*.md``
markdown files (and their global equivalent at ``~/.ultron/rules/``).
The frontmatter on each rule file declares per-turn activation
conditions; the engine evaluates those conditions against the current
user-transcript / intent / system-state and returns the subset of
rules that should be injected into the system prompt this turn.

Conditions implemented:

* ``paths: [glob, glob, ...]`` — fire when any path-like token
  extracted from the user transcript matches any glob.
* ``intents: [RoutingIntentKind, ...]`` — fire when the current
  intent kind is in the list.
* ``topics: [substring|regex, ...]`` — fire when the user transcript
  matches any topic substring/regex.
* ``system_state: {gaming_mode: true, n_active_skills: ">=2", ...}``
  — fire when the orchestrator's state snapshot matches every clause.
* ``all_of: [other condition map, ...]`` — combinator; every nested
  block must match.
* ``not_in_gaming_mode: true`` — convenience inverse of a system_state
  clause.
"""

from __future__ import annotations

from .conditionals import (
    ConditionalRule,
    ConditionalRuleSet,
    RuleActivation,
    RuleEvaluationContext,
    evaluate_rule,
    extract_path_like_strings,
    load_rule_from_path,
    load_rule_set,
)

__all__ = [
    "ConditionalRule",
    "ConditionalRuleSet",
    "RuleActivation",
    "RuleEvaluationContext",
    "evaluate_rule",
    "extract_path_like_strings",
    "load_rule_from_path",
    "load_rule_set",
]
