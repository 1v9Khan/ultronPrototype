"""Desktop sequence runner with per-step screenshot bracketing.

Catalog 09 T5 (GREEN): closes the "no visual audit trail" gap for
multi-step desktop sequences. Existing desktop actions produce
text-only audit log entries (which window was clicked, which key was
pressed); for debugging failed sequences and for voice feedback
("did it actually work?"), the before/after screenshot bracket is
the right primitive.

Adapted from the upstream :class:`AIDesktopAgent.execute_task` step
loop -- the screenshot-bracketing pattern is the highest-leverage
piece even though the upstream's natural-language planner is
deliberately not ported (ultron's LLM intent router is more
sophisticated; see "Things deliberately NOT done" in the catalog
entry).

Architecture
============

* Each step is a :class:`SequenceStep` -- a callable + description +
  optional verification mode.
* The runner calls :meth:`step.action` between two captures via
  :class:`ultron.desktop.capture.ScreenCapture` (the same primitive
  the click-preview gate uses, so mss thread-locality holds).
* When ``verify_with_vlm=True`` AND the click-preview gate is enabled,
  the after-screenshot is routed through the VLM via the existing
  :mod:`ultron.desktop.click_preview` infrastructure. The VLM is
  asked "did the step succeed at: <description>?" and the response
  is recorded in the step result. The bytes are discarded post-VLM
  per the analyze-and-discard pattern (catalog 08 T12).
* On step failure, the runner aborts -- ``status=failed`` and the
  remaining steps are not executed. This matches the upstream loop's
  fail-fast contract.
* Auto-pass radius: identical to click_preview, sequential steps
  whose after-frame matches the VLM's prior verdict within
  :data:`SEQUENCE_AUTO_PASS_RADIUS_PX` skip the redundant VLM round-
  trip (covers the "click ten things in a row" case).

The result schema mirrors the upstream :class:`AIDesktopAgent` shape
(``task / status / success / steps / screenshots / failed_at_step /
error``) while adapting the screenshots to ultron's
:class:`Screenshot` dataclass (with ``image_bytes`` cleared post-
analysis instead of holding raw PIL Image objects in memory).

Safety
======

The runner does NOT itself simulate input; each step's action is the
caller's responsibility. Pass actions that go through the gated
:class:`ultron.desktop.input_control.InputController` and the runner
adds an observation envelope (Cap-2) around them. Failure modes:

* Capture error before the step -> step records ``capture_before_failed``;
  the step still runs.
* Capture error after the step -> step records ``capture_after_failed``;
  the step result reflects whatever the action returned.
* VLM verification raises -> degraded verdict, step continues per
  caller policy.

The :class:`SequenceResult` is JSON-serialisable (the ``Screenshot``
references are stripped to (width, height, monitor_index, timestamp,
``bytes_discarded=True``) tuples for the audit channel).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from ultron.utils.logging import get_logger

logger = get_logger("desktop.sequence")


# Sequential-step auto-pass radius: a per-step after-capture that
# centres within this many pixels of the previous confirmed step
# skips the redundant VLM call. Mirrors click_preview's auto-pass.
SEQUENCE_AUTO_PASS_RADIUS_PX: int = 150


# ---------------------------------------------------------------------------
# Step and result types
# ---------------------------------------------------------------------------


class SequenceStatus(str, Enum):
    """Overall outcome of a sequence run."""

    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


class StepOutcome(str, Enum):
    """Outcome of a single step within a sequence."""

    OK = "ok"
    FAILED = "failed"
    EXCEPTION = "exception"


class VlmVerdict(str, Enum):
    """VLM verification outcome for one step's after-screenshot."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"        # bracket_screenshots=False or no VLM wired
    AUTO_PASSED = "auto_passed"  # within auto-pass radius of prior step
    DEGRADED = "degraded"      # VLM raised; not a verdict


@dataclass(frozen=True)
class SequenceStep:
    """One step in a desktop sequence.

    Attributes:
        description: human-readable description spoken to the VLM
            ("Open the File menu", "Type the filename") AND surfaced
            in the audit log. Mirrors the upstream plugin's
            ``description`` field on plan steps.
        action: zero-arg callable invoked between captures. Its
            return value is recorded in the step result under
            ``action_result``. Convention: a truthy return means
            "step succeeded at the OS level"; a falsy / None return
            means "step's own contract failed".
        target_x / target_y: optional anchor coordinates for the
            auto-pass-radius check. Set to the click target so
            multi-click sequences in the same panel can skip VLM
            round-trips.
        timeout_s: per-step soft timeout. The runner does not enforce
            this directly (the action itself owns blocking semantics)
            but records it in the step entry for audit / debug.
    """

    description: str
    action: Callable[[], Any]
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    timeout_s: Optional[float] = None


@dataclass(frozen=True)
class ScreenshotRef:
    """Pointer to one bracketed screenshot.

    Attributes:
        step: 1-based step number (matches the upstream plugin's
            ``"step"`` field on screenshot dicts).
        when: ``"before"`` or ``"after"``.
        width: pixel width.
        height: pixel height.
        monitor_index: source monitor index when known.
        timestamp: ``time.time()`` at capture.
        bytes_discarded: True iff the original bytes were dropped
            after VLM analysis under the analyze-and-discard pattern.
            Always True on the cache record; callers needing the
            actual bytes must wire their own retainer (e.g. a
            session-scoped temp dir).
        error: capture failure reason when no screenshot was taken.
    """

    step: int
    when: str
    width: int = 0
    height: int = 0
    monitor_index: Optional[int] = None
    timestamp: float = 0.0
    bytes_discarded: bool = True
    error: Optional[str] = None


@dataclass(frozen=True)
class StepResult:
    """Outcome of one step in a sequence.

    Attributes:
        step_index: 1-based.
        description: copied from the :class:`SequenceStep`.
        outcome: :class:`StepOutcome`.
        action_result: whatever the action callable returned.
        error: action-failure or capture-failure reason.
        before: :class:`ScreenshotRef` for the before-frame
            (may carry ``error`` when capture failed).
        after: :class:`ScreenshotRef` for the after-frame.
        vlm_verdict: :class:`VlmVerdict`.
        vlm_message: VLM's textual response (when invoked).
        elapsed_s: wall-clock time from before-capture to after-capture.
    """

    step_index: int
    description: str
    outcome: StepOutcome
    action_result: Any = None
    error: Optional[str] = None
    before: Optional[ScreenshotRef] = None
    after: Optional[ScreenshotRef] = None
    vlm_verdict: VlmVerdict = VlmVerdict.SKIPPED
    vlm_message: str = ""
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class SequenceResult:
    """Outcome of a full sequence run.

    Schema mirrors the upstream :class:`AIDesktopAgent.execute_task`
    return dict:

    * ``task``: original task description.
    * ``status``: ``"completed"`` / ``"failed"`` / ``"error"``.
    * ``success``: True iff all steps succeeded.
    * ``steps``: list of :class:`StepResult`.
    * ``screenshots``: list of (before, after) :class:`ScreenshotRef`
        pairs, one entry per executed step.
    * ``failed_at_step``: 1-based step index when ``status=failed``.
    * ``error``: top-level exception message when ``status=error``.
    * ``elapsed_s``: total wall-clock duration.
    """

    task: str
    status: SequenceStatus
    success: bool
    steps: tuple[StepResult, ...]
    screenshots: tuple[tuple[ScreenshotRef, ScreenshotRef], ...]
    failed_at_step: Optional[int] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class DesktopSequenceRunner:
    """Run a list of desktop steps with before/after screenshot bracketing.

    Construct with optional :class:`ScreenCapture` injection (tests
    swap in a mock; the orchestrator uses the singleton). The VLM
    hook is also injected -- callers wire it from the existing
    click_preview holder so the same Moondream2 instance is reused
    rather than spinning up a second VLM session.
    """

    def __init__(
        self,
        *,
        capture: Optional[object] = None,
        vlm_describe: Optional[Callable[[bytes, str], str]] = None,
        bracket_screenshots: bool = True,
        verify_with_vlm: bool = False,
        auto_pass_radius_px: int = SEQUENCE_AUTO_PASS_RADIUS_PX,
        confirmation_keyword: str = "yes",
        monitor_index: int = 0,
        discard_image_bytes: bool = True,
        clock_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self._capture = capture
        self._vlm_describe = vlm_describe
        self._bracket = bool(bracket_screenshots)
        self._verify = bool(verify_with_vlm)
        self._auto_pass = max(0, int(auto_pass_radius_px))
        self._keyword = str(confirmation_keyword).lower().strip() or "yes"
        self._monitor_index = int(monitor_index)
        self._discard = bool(discard_image_bytes)
        self._clock = clock_fn if callable(clock_fn) else time.time

    def _get_capture(self):
        """Resolve the screen-capture surface.

        Returns ``None`` when capture is not wired AND
        ``bracket_screenshots=False`` -- the runner can still execute
        steps but produces empty :class:`ScreenshotRef` entries.
        """
        if self._capture is not None:
            return self._capture
        try:
            from ultron.desktop.capture import get_screen_capture
            return get_screen_capture()
        except Exception as exc:  # noqa: BLE001
            logger.debug("DesktopSequenceRunner: no capture available: %s", exc)
            return None

    def _capture_screenshot(self, step_index: int, when: str) -> tuple[
        ScreenshotRef, Optional[bytes],
    ]:
        """Capture a frame for the before/after bracket.

        Returns the :class:`ScreenshotRef` (always populated) plus the
        raw PNG bytes when the caller needs them for VLM analysis.
        The bytes are dropped from the ref under the analyze-and-
        discard contract.
        """
        if not self._bracket:
            return (
                ScreenshotRef(step=step_index, when=when,
                              bytes_discarded=False),
                None,
            )
        capture = self._get_capture()
        if capture is None:
            return (
                ScreenshotRef(
                    step=step_index, when=when,
                    error="no capture wired", bytes_discarded=False,
                ),
                None,
            )
        try:
            shot = capture.capture_monitor(self._monitor_index)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "DesktopSequenceRunner capture %s failed at step %d: %s",
                when, step_index, exc,
            )
            return (
                ScreenshotRef(
                    step=step_index, when=when,
                    error=f"capture exception: {exc}",
                    bytes_discarded=False,
                ),
                None,
            )
        if shot is None:
            return (
                ScreenshotRef(
                    step=step_index, when=when,
                    error="capture returned None",
                    bytes_discarded=False,
                ),
                None,
            )
        ref = ScreenshotRef(
            step=step_index,
            when=when,
            width=int(getattr(shot, "width", 0)),
            height=int(getattr(shot, "height", 0)),
            monitor_index=getattr(shot, "monitor_index", None),
            timestamp=float(getattr(shot, "timestamp", self._clock())),
            bytes_discarded=self._discard,
        )
        # Some test doubles return raw bytes directly; tolerate both
        # the dataclass shape and the raw-bytes shape.
        png_bytes = getattr(shot, "image_bytes", None)
        if png_bytes is None and isinstance(shot, (bytes, bytearray)):
            png_bytes = bytes(shot)
        return ref, png_bytes

    def _within_auto_pass(
        self,
        step: SequenceStep,
        last_confirmed: Optional[tuple[int, int]],
    ) -> bool:
        if last_confirmed is None or self._auto_pass <= 0:
            return False
        if step.target_x is None or step.target_y is None:
            return False
        dx = int(step.target_x) - last_confirmed[0]
        dy = int(step.target_y) - last_confirmed[1]
        return (dx * dx + dy * dy) <= (self._auto_pass * self._auto_pass)

    def _vlm_verdict_for_step(
        self,
        *,
        step: SequenceStep,
        after_png: Optional[bytes],
    ) -> tuple[VlmVerdict, str]:
        """Ask the VLM whether the after-frame shows the step succeeded.

        Returns ``(verdict, message)``. When the VLM is not wired or
        the after-frame is missing, returns ``(SKIPPED, "")``.
        """
        if not self._verify:
            return VlmVerdict.SKIPPED, ""
        if self._vlm_describe is None:
            return VlmVerdict.SKIPPED, ""
        if after_png is None:
            return VlmVerdict.DEGRADED, "no after-frame to verify"
        prompt = (
            f"Did this desktop step succeed: '{step.description}'? "
            f"Reply '{self._keyword}' if yes, otherwise describe what "
            "is wrong in one sentence."
        )
        try:
            response = self._vlm_describe(after_png, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.debug("VLM verify raised at step %s: %s", step.description, exc)
            return VlmVerdict.DEGRADED, f"vlm exception: {exc}"
        text = (response or "").strip().lower()
        if not text:
            return VlmVerdict.DEGRADED, "vlm returned empty response"
        if self._keyword in text.split():
            return VlmVerdict.SUCCEEDED, response
        # Tolerant heuristic: "yes" anywhere in the first ~10 chars is
        # also accepted to cover ", yes" / "yes," / etc.
        if text.startswith(self._keyword):
            return VlmVerdict.SUCCEEDED, response
        return VlmVerdict.FAILED, response

    def _coerce_action_outcome(self, result: Any) -> tuple[StepOutcome, Optional[str]]:
        """Map an action's return value to a :class:`StepOutcome`.

        Convention:
        * ``True`` / truthy non-result-shaped value -> OK.
        * Object with ``success`` attribute -> OK iff truthy, else FAILED.
        * ``False`` / ``None`` -> FAILED with action-returned-falsy reason.
        """
        if hasattr(result, "success"):
            if result.success:
                return StepOutcome.OK, None
            return (
                StepOutcome.FAILED,
                getattr(result, "error", None) or "action returned failure",
            )
        if result is None or result is False:
            return StepOutcome.FAILED, "action returned falsy"
        return StepOutcome.OK, None

    def run(
        self,
        task: str,
        steps: Sequence[SequenceStep],
    ) -> SequenceResult:
        """Execute ``steps`` in order with before/after bracketing.

        On the first step failure (action returned falsy / raised /
        VLM rejected), the runner stops -- remaining steps are NOT
        executed. The :class:`SequenceResult` carries the prefix of
        executed steps plus their screenshot pairs.
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('desktop_sequence')
        start = self._clock()
        executed: list[StepResult] = []
        pairs: list[tuple[ScreenshotRef, ScreenshotRef]] = []
        last_confirmed: Optional[tuple[int, int]] = None

        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, SequenceStep):
                err = f"step {idx}: not a SequenceStep ({type(step).__name__})"
                return SequenceResult(
                    task=task,
                    status=SequenceStatus.ERROR,
                    success=False,
                    steps=tuple(executed),
                    screenshots=tuple(pairs),
                    failed_at_step=idx,
                    error=err,
                    elapsed_s=self._clock() - start,
                )
            step_start = self._clock()
            before_ref, _ = self._capture_screenshot(idx, "before")
            try:
                action_result = step.action()
                outcome, action_error = self._coerce_action_outcome(action_result)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "DesktopSequenceRunner step %d (%s) raised: %s",
                    idx, step.description, exc,
                )
                after_ref, _ = self._capture_screenshot(idx, "after")
                step_result = StepResult(
                    step_index=idx,
                    description=step.description,
                    outcome=StepOutcome.EXCEPTION,
                    error=str(exc)[:300],
                    before=before_ref,
                    after=after_ref,
                    vlm_verdict=VlmVerdict.SKIPPED,
                    elapsed_s=self._clock() - step_start,
                )
                executed.append(step_result)
                pairs.append((before_ref, after_ref))
                return SequenceResult(
                    task=task,
                    status=SequenceStatus.FAILED,
                    success=False,
                    steps=tuple(executed),
                    screenshots=tuple(pairs),
                    failed_at_step=idx,
                    error=str(exc)[:300],
                    elapsed_s=self._clock() - start,
                )

            after_ref, after_png = self._capture_screenshot(idx, "after")

            # VLM verification: auto-pass within radius, else ask VLM.
            if self._within_auto_pass(step, last_confirmed):
                verdict, vlm_msg = VlmVerdict.AUTO_PASSED, ""
            else:
                verdict, vlm_msg = self._vlm_verdict_for_step(
                    step=step, after_png=after_png,
                )

            # If VLM rejects an otherwise-successful action, downgrade
            # to FAILED. SKIPPED / DEGRADED / AUTO_PASSED do NOT
            # downgrade -- the runner can't know better than the
            # action's own contract there.
            if outcome is StepOutcome.OK and verdict is VlmVerdict.FAILED:
                outcome = StepOutcome.FAILED
                action_error = action_error or f"vlm rejected: {vlm_msg[:160]}"

            step_result = StepResult(
                step_index=idx,
                description=step.description,
                outcome=outcome,
                action_result=action_result,
                error=action_error,
                before=before_ref,
                after=after_ref,
                vlm_verdict=verdict,
                vlm_message=vlm_msg,
                elapsed_s=self._clock() - step_start,
            )
            executed.append(step_result)
            pairs.append((before_ref, after_ref))

            if outcome is not StepOutcome.OK:
                return SequenceResult(
                    task=task,
                    status=SequenceStatus.FAILED,
                    success=False,
                    steps=tuple(executed),
                    screenshots=tuple(pairs),
                    failed_at_step=idx,
                    error=action_error,
                    elapsed_s=self._clock() - start,
                )

            # Track the most recently confirmed anchor for the next
            # step's auto-pass check.
            if (
                verdict in (VlmVerdict.SUCCEEDED, VlmVerdict.AUTO_PASSED)
                and step.target_x is not None
                and step.target_y is not None
            ):
                last_confirmed = (int(step.target_x), int(step.target_y))

        return SequenceResult(
            task=task,
            status=SequenceStatus.COMPLETED,
            success=True,
            steps=tuple(executed),
            screenshots=tuple(pairs),
            failed_at_step=None,
            error=None,
            elapsed_s=self._clock() - start,
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


_runner_singleton: Optional[DesktopSequenceRunner] = None


def get_sequence_runner() -> DesktopSequenceRunner:
    """Module-level singleton accessor."""
    global _runner_singleton
    if _runner_singleton is None:
        _runner_singleton = DesktopSequenceRunner()
    return _runner_singleton


def set_sequence_runner(runner: Optional[DesktopSequenceRunner]) -> None:
    """Test / orchestrator hook -- swap the singleton."""
    global _runner_singleton
    _runner_singleton = runner


__all__ = [
    "SequenceStatus",
    "StepOutcome",
    "VlmVerdict",
    "SequenceStep",
    "ScreenshotRef",
    "StepResult",
    "SequenceResult",
    "DesktopSequenceRunner",
    "SEQUENCE_AUTO_PASS_RADIUS_PX",
    "get_sequence_runner",
    "set_sequence_runner",
]
