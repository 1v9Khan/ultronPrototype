"""Format-error requery without history pollution.

Direct port of SWE-Agent's
``sweagent/agent/agents.py:get_model_requery_history`` (MIT,
Yang et al. 2024). The pattern: when the model emits a malformed
action (missing tool call, multiple tool calls, blocked action,
bash syntax error, content-policy violation), the harness builds
a TEMPORARY history -- existing real history + fake assistant turn
with the broken output + fake user turn with the error template --
and re-queries the model on it. If the re-query succeeds, the
original BROKEN exchange is NOT added to the permanent history;
only the corrected version is. From the model's perspective the
broken attempt never happened.

For ultron the helper is used wherever an in-process LLM call
produces output that needs structured re-parsing:

* The web-search gate's preflight JSON output (when the LLM
  returns malformed JSON).
* The intent classifier's structured output.
* The addressing classifier when zero-shot returns an unexpected
  label.
* Future structured-output call sites (decomposer, disambiguator).

The helper itself is pure -- it builds the temp history + delegates
to a caller-supplied LLM function so the caller controls retry
budgets, model selection, etc. The :class:`RequeryLoop` orchestrator
caps retry depth (default 3, matching SWE-Agent's ``max_requeries``)
and surfaces a tagged failure when the cap is reached.

Failure modes that warrant a requery (subclasses of
:class:`RequeryReason`):

* :data:`RequeryReason.FORMAT` -- output doesn't parse as expected.
* :data:`RequeryReason.BASH_SYNTAX` -- bash command failed `bash -n`.
* :data:`RequeryReason.BLOCKED_ACTION` -- safety validator blocked
  the proposed action.
* :data:`RequeryReason.CONTENT_POLICY` -- LLM API returned a content
  policy violation; resample with the same history.
* :data:`RequeryReason.EMPTY` -- output was empty / whitespace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults + constants
# ---------------------------------------------------------------------------

#: Default max requery attempts. Matches SWE-Agent's
#: ``max_requeries=3`` cap.
DEFAULT_MAX_REQUERIES: int = 3


# ---------------------------------------------------------------------------
# Reason enum + result records
# ---------------------------------------------------------------------------


class RequeryReason(Enum):
    """Classification of why a requery is being attempted."""

    FORMAT = "format"
    BASH_SYNTAX = "bash_syntax"
    BLOCKED_ACTION = "blocked_action"
    CONTENT_POLICY = "content_policy"
    EMPTY = "empty"
    OTHER = "other"


@dataclass(frozen=True)
class RequeryAttempt:
    """One requery attempt's record (for diagnostics + audit log)."""

    attempt_index: int  # 0-indexed
    reason: RequeryReason
    error_message: str
    broken_output: str
    corrected_output: Optional[str]
    succeeded: bool


@dataclass
class RequeryResult:
    """Output of :class:`RequeryLoop.run`."""

    final_output: Optional[str]
    succeeded: bool
    attempts: list[RequeryAttempt] = field(default_factory=list)
    max_retries_reached: bool = False


# ---------------------------------------------------------------------------
# Temp-history builder (the core trick)
# ---------------------------------------------------------------------------


def build_requery_history(
    base_messages: Sequence[Mapping[str, Any]],
    *,
    broken_output: str,
    error_template: str,
    context: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Build a TEMPORARY history for the requery.

    Mirrors SWE-Agent's ``get_model_requery_history``: appends a
    fake assistant turn carrying the model's broken output + a fake
    user turn carrying the rendered error message. The result is
    safe to pass to ``llm.generate`` -- the model sees the failure
    AND the remediation prompt, but the caller never adds the
    failure to the permanent message list.

    ``error_template`` supports str.format substitution with
    ``context`` keys; missing keys render as empty.

    :param base_messages: the real (untouched) message list.
    :param broken_output: the LLM's last broken response.
    :param error_template: a template describing the failure +
        remediation. Receives ``context`` as format kwargs.
    :param context: optional substitution dict for the template.
    """
    context = dict(context or {})
    try:
        rendered = error_template.format_map(_DefaultDict(context))
    except Exception as exc:
        logger.warning(
            "requery template format error: %s; using raw template", exc
        )
        rendered = error_template
    temp: list[dict[str, Any]] = []
    for msg in base_messages:
        if isinstance(msg, Mapping):
            temp.append(dict(msg))
    temp.append({"role": "assistant", "content": broken_output})
    temp.append({"role": "user", "content": rendered})
    return temp


class _DefaultDict(dict):
    """dict subclass returning '' for missing keys (for str.format_map)."""

    def __missing__(self, key: str) -> str:  # noqa: D401
        return ""


# ---------------------------------------------------------------------------
# RequeryLoop
# ---------------------------------------------------------------------------


GenerateFn = Callable[[Sequence[Mapping[str, Any]]], str]
ValidateFn = Callable[[str], tuple[bool, RequeryReason, str]]


class RequeryLoop:
    """Orchestrator for the temp-history requery cycle.

    Usage::

        loop = RequeryLoop(
            generate_fn=lambda msgs: my_llm.generate(msgs),
            validate_fn=lambda out: parse_or_reason(out),
            error_template="Your output failed validation: {reason}.\\n"
                "Please re-emit a valid response.",
            max_retries=3,
        )
        result = loop.run(base_messages, initial_output)

    ``generate_fn`` receives the temp history (real + broken +
    error) and returns the model's new output. ``validate_fn``
    inspects an output and returns ``(ok, reason, error_message)``.
    On ok, the loop exits with ``succeeded=True`` and the final
    output. On not-ok, the loop iterates up to ``max_retries``
    times before returning ``succeeded=False`` with the last
    attempt's details.

    Per SWE-Agent's contract, the broken exchanges NEVER make it
    into the caller's permanent message list -- only the corrected
    output does (the caller appends that themselves).
    """

    def __init__(
        self,
        *,
        generate_fn: GenerateFn,
        validate_fn: ValidateFn,
        error_template: str = (
            "Your previous response failed validation: {reason} -- {error}\n"
            "Please re-emit a valid response."
        ),
        max_retries: int = DEFAULT_MAX_REQUERIES,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0 (got {max_retries})")
        self.generate_fn = generate_fn
        self.validate_fn = validate_fn
        self.error_template = error_template
        self.max_retries = int(max_retries)

    def run(
        self,
        base_messages: Sequence[Mapping[str, Any]],
        initial_output: str,
        *,
        context_per_attempt: Optional[Callable[[int, str], Mapping[str, Any]]] = None,
    ) -> RequeryResult:
        """Run the validate/requery loop.

        :param base_messages: the real message list (NOT mutated).
        :param initial_output: the first output to validate.
        :param context_per_attempt: optional callable returning the
            template context for each retry; signature
            ``(attempt_index, error_message) -> dict``.
        """
        attempts: list[RequeryAttempt] = []
        current_output = initial_output
        for idx in range(self.max_retries + 1):
            ok, reason, err_msg = self._validate(current_output)
            if ok:
                if attempts:
                    attempts[-1] = RequeryAttempt(
                        attempt_index=attempts[-1].attempt_index,
                        reason=attempts[-1].reason,
                        error_message=attempts[-1].error_message,
                        broken_output=attempts[-1].broken_output,
                        corrected_output=current_output,
                        succeeded=True,
                    )
                return RequeryResult(
                    final_output=current_output,
                    succeeded=True,
                    attempts=attempts,
                    max_retries_reached=False,
                )
            attempts.append(
                RequeryAttempt(
                    attempt_index=idx,
                    reason=reason,
                    error_message=err_msg,
                    broken_output=current_output,
                    corrected_output=None,
                    succeeded=False,
                )
            )
            if idx == self.max_retries:
                # Cap reached -- don't query again.
                return RequeryResult(
                    final_output=None,
                    succeeded=False,
                    attempts=attempts,
                    max_retries_reached=True,
                )
            # Build the temp history + re-query.
            ctx: Mapping[str, Any]
            if context_per_attempt is not None:
                try:
                    ctx = context_per_attempt(idx, err_msg) or {}
                except Exception as exc:
                    logger.warning(
                        "RequeryLoop context_per_attempt raised: %s", exc
                    )
                    ctx = {}
            else:
                ctx = {"reason": reason.value, "error": err_msg}
            temp_history = build_requery_history(
                base_messages,
                broken_output=current_output,
                error_template=self.error_template,
                context=ctx,
            )
            try:
                current_output = self.generate_fn(temp_history)
            except Exception as exc:
                logger.warning(
                    "RequeryLoop generate_fn raised on attempt %d: %s",
                    idx,
                    exc,
                )
                # Treat as fatal -- record + exit.
                attempts.append(
                    RequeryAttempt(
                        attempt_index=idx + 1,
                        reason=RequeryReason.OTHER,
                        error_message=f"generate_fn raised: {exc}",
                        broken_output="",
                        corrected_output=None,
                        succeeded=False,
                    )
                )
                return RequeryResult(
                    final_output=None,
                    succeeded=False,
                    attempts=attempts,
                    max_retries_reached=False,
                )
        # Loop body always returns; safety net.
        return RequeryResult(
            final_output=None,
            succeeded=False,
            attempts=attempts,
            max_retries_reached=True,
        )

    def _validate(self, output: str) -> tuple[bool, RequeryReason, str]:
        try:
            result = self.validate_fn(output)
        except Exception as exc:
            logger.warning(
                "RequeryLoop validate_fn raised: %s", exc
            )
            return False, RequeryReason.OTHER, f"validator raised: {exc}"
        if not isinstance(result, tuple) or len(result) != 3:
            return (
                False,
                RequeryReason.OTHER,
                "validator returned malformed result",
            )
        ok, reason, err_msg = result
        if not isinstance(reason, RequeryReason):
            return False, RequeryReason.OTHER, "validator returned non-enum reason"
        return bool(ok), reason, str(err_msg)


# ---------------------------------------------------------------------------
# Pre-built validators for the common cases
# ---------------------------------------------------------------------------


def validate_non_empty(output: str) -> tuple[bool, RequeryReason, str]:
    """Pass iff ``output`` is non-empty after stripping."""
    if not output or not str(output).strip():
        return False, RequeryReason.EMPTY, "output is empty"
    return True, RequeryReason.EMPTY, ""


def validate_json(output: str) -> tuple[bool, RequeryReason, str]:
    """Pass iff ``output`` parses as JSON (after stripping fences /
    surrounding whitespace)."""
    import json

    s = (output or "").strip()
    # Strip ```json ... ``` fences if present.
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        json.loads(s)
    except json.JSONDecodeError as exc:
        return False, RequeryReason.FORMAT, f"JSON parse error: {exc}"
    return True, RequeryReason.FORMAT, ""


__all__ = [
    "DEFAULT_MAX_REQUERIES",
    "RequeryAttempt",
    "RequeryLoop",
    "RequeryReason",
    "RequeryResult",
    "build_requery_history",
    "validate_json",
    "validate_non_empty",
]
