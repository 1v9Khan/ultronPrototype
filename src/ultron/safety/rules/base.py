"""Abstract base class for safety validator rules.

Every rule subclasses :class:`Rule` and implements :meth:`evaluate`.
The validator's dispatcher (:class:`ToolCallValidator`) iterates a
list of rule instances per call.

Rule design conventions:

* **rule_id is part of the user's policy.** Each rule has a stable
  id (``"K1"``, ``"A3"``, ``"D7"``) matching the 2026-05-12
  restriction list. Users toggle individual rules via
  ``config.yaml:safety.rules.<rule_id>: false``.
* **Stateless.** Rules read from :class:`RuleContext` and the
  :class:`Policy` passed to ``evaluate``. They do NOT keep mutable
  state between calls (otherwise the validator wouldn't be thread-
  safe).
* **Pure-Python.** No filesystem I/O during evaluation -- the path
  resolver does that upstream. Network calls absolutely not.
* **Cheap.** Validators run on every tool-call. Categories A-D fire
  on millions of calls a day in heavy use. A rule's evaluate is a
  few-microsecond hash lookup or regex match.
* **Return :class:`RuleResult`.** Always. Even on internal-error
  paths -- raise only if there's no reasonable result; the
  validator catches and converts to ``BLOCK_HARD``.

For the path-pattern rules in Category K, the
:class:`PathSetRule` helper class below covers the common case
(block writes to any path in a set). Subclasses just provide the
``rule_id``, the description, and the set of protected paths.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ultron.safety.path_resolver import PathResolver
    from ultron.safety.policy import Policy
    from ultron.safety.validator import RuleContext, RuleResult, Verdict


class Rule(ABC):
    """Base class for safety validator rules.

    Subclasses set:

    * ``rule_id`` -- string matching the user's restriction-list
      numbering (e.g. ``"K1"``, ``"A3"``, ``"D7"``).
    * ``description`` -- human-readable one-line description.
    * ``category`` -- single-letter category code (``"K"``, ``"A"``,
      etc.) for grouping in audit logs.

    And implement:

    * :meth:`evaluate(ctx, *, policy, resolver) -> RuleResult`
    """

    rule_id: str = ""
    description: str = ""
    category: str = ""

    @abstractmethod
    def evaluate(
        self,
        ctx: "RuleContext",
        *,
        policy: "Policy",
        resolver: "PathResolver",
    ) -> "RuleResult":
        """Decide a verdict for the given context.

        Args:
            ctx: what the model is trying to do.
            policy: the loaded policy (read-only).
            resolver: shared path canonicalisation helper.

        Returns:
            A :class:`RuleResult`. Use :class:`Verdict.ALLOW` to
            signal "no opinion".
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} rule_id={self.rule_id!r}>"


class PathSetRule(Rule):
    """Common helper: block write-shaped tool calls whose paths land in a set.

    Subclasses set ``rule_id`` + ``description`` and override
    :meth:`protected_paths` to return the canonical set the rule
    enforces. Subclasses can also override :meth:`is_write_attempt`
    if the default heuristic (tool name ends in ``write``,
    ``delete``, ``move``, etc., OR ``arguments`` has a ``write_mode``
    truthy flag) isn't right for their case.
    """

    _WRITE_VERBS = (
        "write", "delete", "remove", "unlink", "rmtree",
        "move", "rename", "copy", "create", "modify", "edit",
        "patch", "overwrite", "truncate", "spawn",
    )

    def __init__(
        self,
        *,
        rule_id: str,
        description: str,
        category: str = "K",
    ) -> None:
        self.rule_id = rule_id
        self.description = description
        self.category = category

    @abstractmethod
    def protected_paths(self, policy: "Policy") -> list[Path]:
        """Return the canonical paths this rule protects.

        Called per evaluate (or once and cached by the subclass).
        """

    def is_write_attempt(self, ctx: "RuleContext") -> bool:
        """Default heuristic: tool name contains a write verb OR the
        arguments dict says so.

        Inspects:
        * ``tool_name`` substring against ``_WRITE_VERBS``.
        * ``arguments["write"]`` truthy.
        * ``arguments["write_mode"]`` not None.
        * ``arguments["destructive"]`` truthy.
        * ``arguments["operation"]`` -- string field used by the
          OpenClaw dispatcher's ``FileOpIntent`` (e.g. ``"write"``,
          ``"delete"``, ``"move"``); checked against the same verb
          list as the tool name.

        Override when needed -- e.g. for tools whose semantics aren't
        captured by the verb list.
        """
        name_lower = ctx.tool_name.lower()
        for verb in self._WRITE_VERBS:
            if verb in name_lower:
                return True
        # Common keyword argument shapes that imply a write.
        if ctx.arguments.get("write", False):
            return True
        if ctx.arguments.get("write_mode") is not None:
            return True
        if ctx.arguments.get("destructive", False):
            return True
        op = ctx.arguments.get("operation")
        if isinstance(op, str):
            op_lower = op.lower()
            for verb in self._WRITE_VERBS:
                if verb in op_lower:
                    return True
        return False

    def evaluate(
        self,
        ctx: "RuleContext",
        *,
        policy: "Policy",
        resolver: "PathResolver",
    ) -> "RuleResult":
        from ultron.safety.path_resolver import PathResolveError
        from ultron.safety.validator import RuleResult, Verdict

        if not self.is_write_attempt(ctx):
            return RuleResult(
                rule_id=self.rule_id,
                verdict=Verdict.ALLOW,
                reason="not a write attempt",
            )

        protected = self.protected_paths(policy)
        if not protected:
            return RuleResult(
                rule_id=self.rule_id,
                verdict=Verdict.ALLOW,
                reason="rule has no protected paths configured",
            )

        # Each candidate path in ctx.paths goes through canonicalisation
        # for comparison. The caller already canonicalises before
        # constructing the context; we re-canonicalise for defence in
        # depth in case a buggy caller passes raw strings.
        for raw in ctx.paths:
            try:
                candidate = resolver.resolve(raw)
            except PathResolveError as e:
                # Unresolvable path during a write attempt is itself
                # a hard block -- attacker may be using an evasion
                # pattern.
                return RuleResult(
                    rule_id=self.rule_id,
                    verdict=Verdict.BLOCK_HARD,
                    reason=(
                        f"unresolvable path during write attempt: {e}"
                    ),
                    context={"raw_path": str(raw)},
                )
            for p in protected:
                if candidate == p or self._is_within(candidate, p):
                    return RuleResult(
                        rule_id=self.rule_id,
                        verdict=Verdict.BLOCK_HARD,
                        reason=(
                            f"{self.description}: write to protected path "
                            f"{candidate} blocked"
                        ),
                        context={
                            "candidate": str(candidate),
                            "protected": str(p),
                        },
                    )
        return RuleResult(
            rule_id=self.rule_id,
            verdict=Verdict.ALLOW,
            reason="no protected-path match",
        )

    @staticmethod
    def _is_within(candidate: Path, root: Path) -> bool:
        """True iff ``candidate`` is a descendant of ``root``.

        ``relative_to`` raises on non-descendants; we swallow and
        return False. ``root.parts`` comparison gives equivalent
        semantics with friendlier error handling.
        """
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# Generic rule classes -- the multiplier for Phases 3-5.
# Most of the user's restriction-list items are pattern matches against
# paths, shell commands, or tool arguments. Rather than writing 150
# individual Rule subclasses, these generic classes let category modules
# define rules as data.
# ---------------------------------------------------------------------------


class PathPatternRule(Rule):
    """Match candidate paths against a list of regex patterns.

    Distinct from :class:`PathSetRule` (exact-or-descendant match
    against a fixed set of paths): this one fires on regex pattern
    matches anywhere in the resolved path string. Used for rules
    that block writes under any of a category of directories
    (``C:\\Windows\\`` etc.) rather than a specific list of files.

    The patterns match against the LOWERCASE canonical path with
    forward slashes -- so a single pattern works on Windows
    (``c:/windows/``) without case / separator variants.

    Args:
        rule_id, description, category: rule metadata.
        patterns: list of regex patterns (compiled at construction).
        verdict: what to return when a pattern matches. Default
            ``Verdict.BLOCK_HARD``; some rules use
            ``Verdict.NEEDS_EXPLICIT_INTENT`` to require user-stated
            intent.
        write_only: if True (default), the rule only fires on write-
            shaped tool calls (heuristic from :class:`PathSetRule`).
            Set False for read-shaped rules like Category D OUT-gates.
    """

    def __init__(
        self,
        *,
        rule_id: str,
        description: str,
        category: str,
        patterns: list[str],
        verdict_on_match=None,  # Verdict; defaults to BLOCK_HARD
        write_only: bool = True,
    ) -> None:
        from ultron.safety.validator import Verdict
        self.rule_id = rule_id
        self.description = description
        self.category = category
        self._compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        self._verdict_on_match = (
            verdict_on_match if verdict_on_match is not None else Verdict.BLOCK_HARD
        )
        self._write_only = write_only

    _WRITE_VERBS = PathSetRule._WRITE_VERBS

    def _is_write_attempt(self, ctx) -> bool:
        name_lower = ctx.tool_name.lower()
        for verb in self._WRITE_VERBS:
            if verb in name_lower:
                return True
        if ctx.arguments.get("write", False):
            return True
        if ctx.arguments.get("destructive", False):
            return True
        # Match against ``arguments["operation"]`` -- the OpenClaw
        # dispatcher's FileOpIntent passes ``operation="write"`` /
        # ``"delete"`` etc. rather than encoding it in tool_name.
        op = ctx.arguments.get("operation")
        if isinstance(op, str):
            op_lower = op.lower()
            for verb in self._WRITE_VERBS:
                if verb in op_lower:
                    return True
        return False

    def evaluate(self, ctx, *, policy, resolver):
        from ultron.safety.path_resolver import PathResolveError
        from ultron.safety.validator import RuleResult, Verdict

        if self._write_only and not self._is_write_attempt(ctx):
            return RuleResult(
                rule_id=self.rule_id,
                verdict=Verdict.ALLOW,
                reason="not a write attempt",
            )

        for raw in ctx.paths:
            try:
                p = resolver.resolve(raw) if not isinstance(raw, Path) else raw
            except PathResolveError as e:
                return RuleResult(
                    rule_id=self.rule_id,
                    verdict=Verdict.BLOCK_HARD,
                    reason=f"unresolvable path: {e}",
                    context={"raw_path": str(raw)},
                )
            # Normalise to lowercase forward-slash form for pattern
            # matching. The path resolver guarantees the canonical
            # absolute form; we just collapse the surface for regex.
            sp = str(p).lower().replace("\\", "/")
            for pat in self._compiled:
                if pat.search(sp):
                    return RuleResult(
                        rule_id=self.rule_id,
                        verdict=self._verdict_on_match,
                        reason=(
                            f"{self.description}: path {p} matched "
                            f"pattern {pat.pattern!r}"
                        ),
                        context={"path": str(p), "pattern": pat.pattern},
                    )
        return RuleResult(
            rule_id=self.rule_id,
            verdict=Verdict.ALLOW,
            reason="no pattern match",
        )


class CommandPatternRule(Rule):
    """Match shell-command strings (or tool-name strings) against
    a list of regex patterns.

    Args:
        rule_id, description, category: metadata.
        patterns: list of regex patterns (compiled with re.IGNORECASE).
            Each pattern is matched against the concatenation of
            ``ctx.tool_name`` and ``ctx.arguments`` values converted
            to strings.
        verdict_on_match: what to return on a match. Default BLOCK_HARD.
        check_tool_name: also pattern-match against the tool name
            itself (default True). Set False to limit to argument
            content only.
    """

    def __init__(
        self,
        *,
        rule_id: str,
        description: str,
        category: str,
        patterns: list[str],
        verdict_on_match=None,
        check_tool_name: bool = True,
    ) -> None:
        from ultron.safety.validator import Verdict
        self.rule_id = rule_id
        self.description = description
        self.category = category
        self._compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        self._verdict_on_match = (
            verdict_on_match if verdict_on_match is not None else Verdict.BLOCK_HARD
        )
        self._check_tool_name = check_tool_name

    def evaluate(self, ctx, *, policy, resolver):  # noqa: ARG002
        from ultron.safety.validator import RuleResult, Verdict

        # Build the haystack: tool name + every argument value.
        parts: list[str] = []
        if self._check_tool_name:
            parts.append(ctx.tool_name)
        for v in ctx.arguments.values():
            if isinstance(v, (str, bytes)):
                parts.append(v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, str):
                        parts.append(item)
            elif v is not None:
                parts.append(str(v))
        haystack = "\n".join(parts)
        for pat in self._compiled:
            m = pat.search(haystack)
            if m is not None:
                return RuleResult(
                    rule_id=self.rule_id,
                    verdict=self._verdict_on_match,
                    reason=(
                        f"{self.description}: matched pattern "
                        f"{pat.pattern!r}"
                    ),
                    context={
                        "pattern": pat.pattern,
                        "match_preview": m.group(0)[:200],
                    },
                )
        return RuleResult(
            rule_id=self.rule_id,
            verdict=Verdict.ALLOW,
            reason="no command pattern match",
        )


class ToolNameRule(Rule):
    """Block based purely on tool-name prefix or exact match.

    Cheap: no path resolution, no argument inspection. Useful for
    categorical bans like "no LSASS dump tool" or "no WMI process
    create."

    Args:
        rule_id, description, category: metadata.
        denied_tool_names: list of full tool names; if ``ctx.tool_name``
            starts with any of these, fire.
        verdict_on_match: default BLOCK_HARD.
    """

    def __init__(
        self,
        *,
        rule_id: str,
        description: str,
        category: str,
        denied_tool_names: list[str],
        verdict_on_match=None,
    ) -> None:
        from ultron.safety.validator import Verdict
        self.rule_id = rule_id
        self.description = description
        self.category = category
        self._denied = [n.lower() for n in denied_tool_names]
        self._verdict_on_match = (
            verdict_on_match if verdict_on_match is not None else Verdict.BLOCK_HARD
        )

    def evaluate(self, ctx, *, policy, resolver):  # noqa: ARG002
        from ultron.safety.validator import RuleResult, Verdict

        name_lower = ctx.tool_name.lower()
        for denied in self._denied:
            if name_lower == denied or name_lower.startswith(denied + "."):
                return RuleResult(
                    rule_id=self.rule_id,
                    verdict=self._verdict_on_match,
                    reason=(
                        f"{self.description}: tool {ctx.tool_name!r} "
                        f"is on the deny list"
                    ),
                    context={"tool_name": ctx.tool_name, "denied_prefix": denied},
                )
        return RuleResult(
            rule_id=self.rule_id,
            verdict=Verdict.ALLOW,
            reason="tool not on deny list",
        )


class SandboxConfinementRule(Rule):
    """Block destructive operations on paths outside the sandbox roots.

    The mirror of :class:`PathSetRule`: instead of a fixed protected
    set, this rule blocks writes anywhere that ISN'T under one of
    the sandbox roots. Used for Category A rules that allow destructive
    operations only within ``data/sandbox/<project>/``.

    Args:
        rule_id, description, category: metadata.
        write_only: if True, only fires on write-shaped tool calls.
        allow_dirs: extra directories where destructive ops are
            allowed in addition to ``policy.sandbox_roots`` (e.g.
            ``logs/`` for legitimate log rotation).
    """

    def __init__(
        self,
        *,
        rule_id: str,
        description: str,
        category: str,
        write_only: bool = True,
        allow_dirs: Optional[list[str]] = None,
    ) -> None:
        self.rule_id = rule_id
        self.description = description
        self.category = category
        self._write_only = write_only
        self._allow_dirs = allow_dirs or []

    _WRITE_VERBS = PathSetRule._WRITE_VERBS

    def _is_write_attempt(self, ctx) -> bool:
        name_lower = ctx.tool_name.lower()
        for verb in self._WRITE_VERBS:
            if verb in name_lower:
                return True
        if ctx.arguments.get("write", False):
            return True
        if ctx.arguments.get("destructive", False):
            return True
        op = ctx.arguments.get("operation")
        if isinstance(op, str):
            op_lower = op.lower()
            for verb in self._WRITE_VERBS:
                if verb in op_lower:
                    return True
        return False

    def evaluate(self, ctx, *, policy, resolver):
        from ultron.safety.path_resolver import PathResolveError
        from ultron.safety.validator import RuleResult, Verdict

        if self._write_only and not self._is_write_attempt(ctx):
            return RuleResult(
                rule_id=self.rule_id,
                verdict=Verdict.ALLOW,
                reason="not a write attempt",
            )

        allowed_roots = list(policy.sandbox_roots)
        for extra in self._allow_dirs:
            try:
                allowed_roots.append(resolver.resolve(extra))
            except PathResolveError:
                continue

        if not allowed_roots:
            return RuleResult(
                rule_id=self.rule_id,
                verdict=Verdict.ALLOW,
                reason="no sandbox roots configured",
            )

        for raw in ctx.paths:
            try:
                p = resolver.resolve(raw) if not isinstance(raw, Path) else raw
            except PathResolveError as e:
                return RuleResult(
                    rule_id=self.rule_id,
                    verdict=Verdict.BLOCK_HARD,
                    reason=f"unresolvable path: {e}",
                    context={"raw_path": str(raw)},
                )
            inside = any(
                PathSetRule._is_within(p, root) or p == root
                for root in allowed_roots
            )
            if not inside:
                return RuleResult(
                    rule_id=self.rule_id,
                    verdict=Verdict.BLOCK_HARD,
                    reason=(
                        f"{self.description}: destructive op on path {p} "
                        f"is outside the sandbox roots"
                    ),
                    context={
                        "path": str(p),
                        "sandbox_roots": [str(r) for r in allowed_roots],
                    },
                )
        return RuleResult(
            rule_id=self.rule_id,
            verdict=Verdict.ALLOW,
            reason="path is inside sandbox",
        )
