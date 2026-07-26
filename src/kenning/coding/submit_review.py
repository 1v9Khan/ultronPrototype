"""Multi-stage submit review with kenning-specific gates.

Direct port of SWE-Agent's
``tools/review_on_submit_m/bin/submit`` (MIT, Yang et al. 2024).
The pattern: when a coding session would "complete", the supervisor
first walks through a list of review prompts that ask the model
(or the supervisor itself) to self-check before the user-facing
"Done" narration fires.

SWE-Agent's default stage tells the model to:

1. Re-run the reproduction script
2. Remove the reproduction script
3. Revert any test files touched
4. Submit again

Kenning's stages are domain-specific:

* **VOICE_LOCK** -- did this session touch any voice-quality-locked
  file (``SOUL.md``, ``RVC``, ``Piper``, the LLM model file, the
  vocal WAV)? If so, refuse to declare complete -- those files are
  under the partial-lift contract and can't be modified without
  explicit user direction.
* **TESTS** -- did `scripts/run_tests.py` pass since the last edit?
  If files were written but no test sweep was run, prompt the
  supervisor to run one.
* **DOC_DRIFT** -- did the session touch
  ``src/kenning/`` modules / ``scripts/`` files / ``tests/``
  directories without also touching ``docs/codebase_structure.md``?
  If so, prompt to update the doc.
* **FREEFORM** -- catch-all stage where additional reviewers can
  add custom prompts via config.

Each stage carries a substitution-friendly template (``{diff}``,
``{problem_statement}``, ``{files_touched}``, ``{voice_locked_hits}``,
etc). State (``current_stage``, ``stages_resolved``) lives in
:class:`SessionRegistry` (T15) so a crash mid-review can resume from
the last completed stage.

Single-resolution invariant: ``resolve(stage)`` once advances the
counter; calling resolve on the same stage twice raises. ``force_complete()``
short-circuits the remaining stages -- the user can interject to
say "skip the check, just commit."
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from kenning.coding.session_registry import (
    SessionRegistry,
    get_session_registry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------


class StageOutcome(Enum):
    """Result of resolving one review stage."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    FORCED = "forced"


@dataclass(frozen=True)
class ReviewStage:
    """One review-prompt template.

    :param name: short identifier (``"VOICE_LOCK"``, ``"TESTS"``, ...).
        Used as the registry key suffix + audit-log tag.
    :param prompt_template: substitution template; placeholders use
        Python str.format syntax (``{diff}``, ``{problem_statement}``,
        ``{files_touched}``, ``{voice_locked_hits}``). Missing
        placeholders render as empty.
    :param description: one-line human description (for narration
        + audit log).
    :param required: when True, a FAILED resolution blocks the
        eventual completion. When False, FAILED is logged but the
        loop continues.
    """

    name: str
    prompt_template: str
    description: str = ""
    required: bool = True


# ---------------------------------------------------------------------------
# Default stages
# ---------------------------------------------------------------------------

#: Default stage 1 -- voice-quality-lock check. Refers verbatim to
#: the files under the partial-lift contract.
DEFAULT_VOICE_LOCK_STAGE = ReviewStage(
    name="VOICE_LOCK",
    description="Voice-quality-lock contract check",
    prompt_template=(
        "Before declaring complete, scan the session diff for any "
        "modifications to voice-quality-locked files (SOUL.md, RVC "
        "weights, Piper voice model, the LLM model GGUF, the vocal "
        "WAV reference). If any of these appear in the diff, REVERT "
        "the change -- the voice baseline is contract-locked and "
        "cannot be modified without explicit user direction.\n\n"
        "Files touched this session:\n{files_touched}\n\n"
        "Voice-locked file hits detected:\n{voice_locked_hits}"
    ),
)

#: Default stage 2 -- test sweep check.
DEFAULT_TESTS_STAGE = ReviewStage(
    name="TESTS",
    description="Test sweep status check",
    prompt_template=(
        "Before declaring complete, confirm `scripts/run_tests.py` "
        "passes against the changes made this session. The current "
        "baseline is 4750+ passing, 16 skipped, 0 failed. A drop in "
        "the passing count or any failure is a regression that must "
        "be fixed before the session is considered complete.\n\n"
        "Files touched this session:\n{files_touched}"
    ),
)

#: Default stage 3 -- documentation drift check.
DEFAULT_DOC_DRIFT_STAGE = ReviewStage(
    name="DOC_DRIFT",
    description="codebase_structure.md drift check",
    prompt_template=(
        "Before declaring complete, confirm `docs/codebase_structure.md` "
        "reflects the session's changes. If you added a new module, "
        "function, script, test directory, or runtime artifact, update "
        "the corresponding section. The doc's maintenance contract is "
        "binding -- skipping the update means future sessions waste "
        "time re-deriving truth.\n\n"
        "Files touched this session:\n{files_touched}\n\n"
        "Was docs/codebase_structure.md included in this session? "
        "{doc_touched}"
    ),
)

DEFAULT_STAGES: tuple[ReviewStage, ...] = (
    DEFAULT_VOICE_LOCK_STAGE,
    DEFAULT_TESTS_STAGE,
    DEFAULT_DOC_DRIFT_STAGE,
)


# ---------------------------------------------------------------------------
# Voice-lock detection
# ---------------------------------------------------------------------------

#: Files under the partial-lift voice-quality lock. Paths are relative
#: to the project root; the matcher uses a case-insensitive substring
#: check so a session may catch a file the matcher's exact-list missed.
DEFAULT_VOICE_LOCKED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"SOUL\.md",
        r"\bIDENTITY\.md\b",
        r"models[\\/]piper[\\/]",
        r"kenning_rvc_voice[\\/]",
        # Matches the legacy root-level location AND the
        # post-disk-cleaning "kokoro training audio/" home, in BOTH the
        # real ultronVoiceAudio/ tree (gitignored; the 2026-06-12 rename
        # did not migrate it) and the post-rename kenningVoiceAudio/ tree.
        r"(?:ultron|kenning)VoiceAudio[\\/](?:kokoro training audio[\\/])?(?:Ultron|Kenning)_vocals_mono_v1\.wav",
        r"models[\\/]Qwen3\.5-4B-Q4_K_M\.gguf",
        r"models[\\/]Qwen3\.5-0\.8B-Q4_K_M\.gguf",
        r"models[\\/]kokoro[\\/]voices[\\/]kenning\.pt",
        r"models[\\/]kokoro[\\/]kenning_finetune\.pth",
        r"src[\\/]kenning[\\/]tts[\\/]rvc\.py",
        r"src[\\/]kenning[\\/]tts[\\/]kenning_filter\.py",
    )
)


def detect_voice_lock_hits(
    files_touched: Iterable[str],
    *,
    patterns: Sequence[re.Pattern[str]] = DEFAULT_VOICE_LOCKED_PATTERNS,
) -> list[str]:
    """Return the subset of ``files_touched`` matching any voice-lock
    pattern."""
    hits: list[str] = []
    for f in files_touched:
        if not f:
            continue
        for pat in patterns:
            if pat.search(f):
                hits.append(f)
                break
    return hits


# ---------------------------------------------------------------------------
# Loop state + result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageResult:
    """Outcome of resolving one review stage."""

    name: str
    outcome: StageOutcome
    note: str = ""
    resolved_at: float = 0.0


@dataclass
class ReviewState:
    """Persisted state for the submit-review loop.

    Lives in :class:`SessionRegistry` under
    ``"submit_review_state"``.
    """

    current_stage: int = 0
    results: list[StageResult] = field(default_factory=list)
    forced: bool = False
    started_at: float = 0.0


# Registry keys
_REGISTRY_STATE_KEY: str = "submit_review_state"
_REGISTRY_STAGES_KEY: str = "submit_review_stages"


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class SubmitReviewLoop:
    """Multi-stage review state machine.

    Construct via :func:`build_submit_review_loop` or directly.
    Public API:

    * :meth:`current_stage()` -- name of the next stage to resolve,
      or ``None`` if all are done.
    * :meth:`current_prompt(context)` -- render the current stage's
      prompt template with ``context``.
    * :meth:`resolve(stage_name, outcome, note)` -- mark one stage
      done. Advances the counter when ``outcome`` is PASSED /
      SKIPPED; FAILED on a required stage halts but doesn't advance.
    * :meth:`force_complete(reason)` -- short-circuit remaining
      stages. Records :data:`StageOutcome.FORCED` for each.
    * :meth:`is_complete()` -- True when every stage was PASSED /
      SKIPPED / FORCED.
    * :meth:`is_blocked()` -- True when a required stage is FAILED.
    * :meth:`reset()` -- drop state (start the loop over).
    """

    def __init__(
        self,
        *,
        stages: Sequence[ReviewStage],
        registry: SessionRegistry,
    ) -> None:
        if not stages:
            raise ValueError("at least one stage is required")
        # Dedup by name; later stages with the same name lose.
        seen: set[str] = set()
        unique: list[ReviewStage] = []
        for s in stages:
            if s.name in seen:
                continue
            seen.add(s.name)
            unique.append(s)
        self.stages: tuple[ReviewStage, ...] = tuple(unique)
        self.registry = registry
        self._load_state()

    # ----- public surface ----------------------------------------------

    def current_stage(self) -> Optional[ReviewStage]:
        """Return the next un-resolved stage, or None when done."""
        state = self._state
        if state.forced or state.current_stage >= len(self.stages):
            return None
        # A FAILED required stage blocks the loop.
        for r in state.results:
            if r.outcome == StageOutcome.FAILED:
                stage = self._find_stage(r.name)
                if stage is not None and stage.required:
                    return stage
        return self.stages[state.current_stage]

    def current_prompt(
        self,
        *,
        context: Optional[dict] = None,
    ) -> str:
        """Render the current stage's prompt template."""
        stage = self.current_stage()
        if stage is None:
            return ""
        context = context or {}
        try:
            return stage.prompt_template.format_map(_DefaultDict(context))
        except Exception as exc:
            logger.warning(
                "submit_review.current_prompt format error for stage %s: %s",
                stage.name,
                exc,
            )
            return stage.prompt_template

    def resolve(
        self,
        stage_name: str,
        outcome: StageOutcome,
        *,
        note: str = "",
    ) -> StageResult:
        """Mark one stage done. Returns the recorded result.

        Raises :class:`RuntimeError` on attempts to:
        * resolve an unknown stage name
        * resolve a stage AFTER the loop was forced
        * resolve the SAME stage twice (single-resolution invariant)
        """
        if self._state.forced:
            raise RuntimeError(
                "submit_review loop was force-completed; resolve() is invalid"
            )
        stage = self._find_stage(stage_name)
        if stage is None:
            raise RuntimeError(f"unknown stage: {stage_name!r}")
        if any(r.name == stage_name for r in self._state.results):
            raise RuntimeError(
                f"stage {stage_name!r} already resolved"
            )
        result = StageResult(
            name=stage_name,
            outcome=outcome,
            note=note,
            resolved_at=time.time(),
        )
        self._state.results.append(result)
        if outcome in (StageOutcome.PASSED, StageOutcome.SKIPPED):
            # Advance counter to the next un-resolved stage.
            resolved = {r.name for r in self._state.results}
            idx = self._state.current_stage
            while idx < len(self.stages) and self.stages[idx].name in resolved:
                idx += 1
            self._state.current_stage = idx
        self._save_state()
        return result

    def force_complete(self, *, reason: str = "user_override") -> None:
        """Short-circuit remaining stages. Records FORCED for each
        un-resolved one."""
        resolved = {r.name for r in self._state.results}
        for s in self.stages:
            if s.name in resolved:
                continue
            self._state.results.append(
                StageResult(
                    name=s.name,
                    outcome=StageOutcome.FORCED,
                    note=reason,
                    resolved_at=time.time(),
                )
            )
        self._state.forced = True
        self._state.current_stage = len(self.stages)
        self._save_state()

    def is_complete(self) -> bool:
        """True when every stage is PASSED / SKIPPED / FORCED."""
        if self._state.forced:
            return True
        if self._state.current_stage < len(self.stages):
            return False
        # And no FAILED required stage outstanding.
        for r in self._state.results:
            if r.outcome == StageOutcome.FAILED:
                s = self._find_stage(r.name)
                if s is not None and s.required:
                    return False
        return True

    def is_blocked(self) -> bool:
        """True iff at least one REQUIRED stage failed."""
        if self._state.forced:
            return False
        for r in self._state.results:
            if r.outcome == StageOutcome.FAILED:
                s = self._find_stage(r.name)
                if s is not None and s.required:
                    return True
        return False

    def history(self) -> list[StageResult]:
        """Return a copy of the per-stage history (in resolution order)."""
        return list(self._state.results)

    def reset(self) -> None:
        """Drop state -- start the loop over from stage 0."""
        self._state = ReviewState(started_at=time.time())
        self._save_state()

    def status_summary(self) -> str:
        """Single-line human-readable status."""
        if self.is_complete():
            return f"review complete ({len(self.stages)}/{len(self.stages)} stages)"
        cur = self.current_stage()
        if cur is None:
            return "review complete"
        if self.is_blocked():
            return f"review BLOCKED at {cur.name}"
        return (
            f"review at stage {self._state.current_stage + 1}/{len(self.stages)}: "
            f"{cur.name}"
        )

    # ----- internals ---------------------------------------------------

    def _find_stage(self, name: str) -> Optional[ReviewStage]:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def _load_state(self) -> None:
        raw = self.registry.get(_REGISTRY_STATE_KEY)
        if not isinstance(raw, dict):
            self._state = ReviewState(started_at=time.time())
            return
        results_raw = raw.get("results", [])
        results: list[StageResult] = []
        if isinstance(results_raw, list):
            for r in results_raw:
                if not isinstance(r, dict):
                    continue
                try:
                    results.append(
                        StageResult(
                            name=str(r.get("name", "")),
                            outcome=StageOutcome(r.get("outcome", "passed")),
                            note=str(r.get("note", "")),
                            resolved_at=float(r.get("resolved_at", 0.0)),
                        )
                    )
                except (ValueError, TypeError):
                    continue
        self._state = ReviewState(
            current_stage=int(raw.get("current_stage", 0)),
            results=results,
            forced=bool(raw.get("forced", False)),
            started_at=float(raw.get("started_at", time.time())),
        )

    def _save_state(self) -> None:
        payload = {
            "current_stage": self._state.current_stage,
            "results": [
                {
                    "name": r.name,
                    "outcome": r.outcome.value,
                    "note": r.note,
                    "resolved_at": r.resolved_at,
                }
                for r in self._state.results
            ],
            "forced": self._state.forced,
            "started_at": self._state.started_at,
        }
        self.registry[_REGISTRY_STATE_KEY] = payload


class _DefaultDict(dict):
    """dict subclass that returns ``""`` for missing keys (for
    str.format_map)."""

    def __missing__(self, key: str) -> str:  # noqa: D401
        return ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_submit_review_loop(
    session_id: str,
    *,
    extra_stages: Sequence[ReviewStage] = (),
    skip_defaults: bool = False,
    registry: Optional[SessionRegistry] = None,
) -> SubmitReviewLoop:
    """Construct a :class:`SubmitReviewLoop` for ``session_id``.

    ``extra_stages`` are appended after the defaults (or replace
    them entirely when ``skip_defaults`` is True). Operators can
    inject domain-specific stages without modifying the default
    list.
    """
    if registry is None:
        registry = get_session_registry(session_id)
    stages: list[ReviewStage] = []
    if not skip_defaults:
        stages.extend(DEFAULT_STAGES)
    stages.extend(extra_stages)
    return SubmitReviewLoop(stages=tuple(stages), registry=registry)


__all__ = [
    "DEFAULT_DOC_DRIFT_STAGE",
    "DEFAULT_STAGES",
    "DEFAULT_TESTS_STAGE",
    "DEFAULT_VOICE_LOCK_STAGE",
    "DEFAULT_VOICE_LOCKED_PATTERNS",
    "ReviewStage",
    "ReviewState",
    "StageOutcome",
    "StageResult",
    "SubmitReviewLoop",
    "build_submit_review_loop",
    "detect_voice_lock_hits",
]
