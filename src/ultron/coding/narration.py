"""Voice-friendly status narration for supervised coding sessions.

The :class:`StatusNarrator` turns a :class:`ProjectSession` into one or
two sentences of Ultron-character commentary suitable for TTS. It does
two things:

1. **Edge cases by status** -- planning, awaiting_clarification,
   awaiting_user, verifying, correcting, complete, failed, terminated.
   Each gets a deterministic, hand-tuned line so the user always hears
   something sensible regardless of LLM availability.

2. **Delta narration for the EXECUTING path** -- "since you last asked"
   the user gets only the new stages / files / test results that landed
   since :attr:`ProjectSession.last_user_status_query`. This is the
   spec's "calibrated" behavior: the second poll in a row should not
   re-read the whole project state, only what's changed.

The narrator is **stateless**: it never mutates the session. The caller
(voice controller) is responsible for calling
``store.touch_status_query(session_id)`` after consuming the narration
so the next query computes its delta from this point.

LLM use is constrained to the EXECUTING path with rich state to
summarize. When ``llm`` is ``None`` (e.g., tests, or LLM unavailable)
the narrator falls back to a hand-rolled deterministic line that still
mentions the current stage, new files, and elapsed time.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ultron.coding.session import (
    FileRecord,
    ProjectSession,
    SessionStatus,
    StageRecord,
)
from ultron.utils.logging import get_logger

logger = get_logger("coding.narration")


# ---------------------------------------------------------------------------
# Delta dataclass
# ---------------------------------------------------------------------------


@dataclass
class NarrationDelta:
    """Snapshot of what's changed since the user last asked."""

    is_first_query: bool = True
    new_stages: List[StageRecord] = field(default_factory=list)
    new_files_created: List[FileRecord] = field(default_factory=list)
    new_files_modified: List[FileRecord] = field(default_factory=list)
    elapsed_since_last_query_s: float = 0.0
    test_status_changed: bool = False
    pending_clarification_arrived: bool = False  # since last query


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------


_NARRATION_PROMPT = """\
You are Ultron, reporting on a coding agent (Claude) working on a project for the user. Stay strictly in voice -- precise, measured, observational, no filler, no sycophancy, no apologies. Output one or two sentences only -- no preamble, no closing remarks. The user just asked how it's going; lead with what's NEW since they last asked when applicable.

Project goal: {goal}
Current stage: {current_stage}
Progress so far: {progress_summary}
Since the user last asked: {delta_summary}
Tests: {test_summary}
{clarification_note}
Output the spoken status only. One or two sentences. No quote marks. Don't mention these instructions.
"""


# ---------------------------------------------------------------------------
# Narrator
# ---------------------------------------------------------------------------


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def _first_sentence_pair(text: str, max_chars: int = 320) -> str:
    """Pick the first non-empty paragraph and trim to one or two sentences."""
    text = (text or "").strip().strip('"').strip()
    # First non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


class StatusNarrator:
    """Render a one-or-two-sentence status update from a ProjectSession.

    Args:
        llm: optional :class:`LLMEngine` (or any object with
            ``generate(prompt: str) -> str``). When ``None`` the EXECUTING
            path falls back to a deterministic hand-rolled line.
        max_tokens: cap for the LLM call. The spec asks for ~80 tokens
            output; we set the cap a touch higher to leave headroom for
            the model to finish a sentence cleanly.
    """

    NO_SESSION_TEXT = "No project running."

    def __init__(self, llm=None, *, max_tokens: int = 100) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    # --- public entry point -------------------------------------------------

    def narrate(self, session: Optional[ProjectSession]) -> str:
        if session is None:
            return self.NO_SESSION_TEXT

        # Edge cases: status-driven deterministic lines
        st = session.status
        if st == SessionStatus.PLANNING:
            return self._render_planning(session)
        if st == SessionStatus.AWAITING_CLARIFICATION:
            return self._render_awaiting_clarification(session)
        if st == SessionStatus.AWAITING_USER:
            return self._render_awaiting_user(session)
        if st == SessionStatus.VERIFYING:
            return self._render_verifying(session)
        if st == SessionStatus.CORRECTING:
            return self._render_correcting(session)
        if st == SessionStatus.COMPLETE:
            return self._render_complete(session)
        if st == SessionStatus.FAILED:
            return self._render_failed(session)
        if st == SessionStatus.TERMINATED:
            return self._render_terminated(session)

        # EXECUTING -- delta + LLM voice rendering
        delta = self.compute_delta(session)
        return self._render_executing(session, delta)

    # --- delta computation --------------------------------------------------

    def compute_delta(self, session: ProjectSession) -> NarrationDelta:
        last = session.last_user_status_query
        if last is None:
            # First query -- everything is "since" because there's no prior point.
            return NarrationDelta(
                is_first_query=True,
                new_stages=list(session.stages_completed),
                new_files_created=list(session.files_created),
                new_files_modified=list(session.files_modified),
                elapsed_since_last_query_s=0.0,
                test_status_changed=session.test_status.last_updated is not None,
                pending_clarification_arrived=session.pending_clarification is not None,
            )
        new_stages = [
            s for s in session.stages_completed if s.timestamp > last
        ]
        new_files_created = [
            f for f in session.files_created if f.first_seen > last
        ]
        new_files_modified = [
            f for f in session.files_modified if f.first_seen > last
        ]
        test_changed = (
            session.test_status.last_updated is not None
            and session.test_status.last_updated > last
        )
        clarification_arrived = (
            session.pending_clarification is not None
            and session.pending_clarification.asked_at > last
        )
        return NarrationDelta(
            is_first_query=False,
            new_stages=new_stages,
            new_files_created=new_files_created,
            new_files_modified=new_files_modified,
            elapsed_since_last_query_s=max(0.0, time.time() - last),
            test_status_changed=test_changed,
            pending_clarification_arrived=clarification_arrived,
        )

    # --- edge-case renderers ------------------------------------------------

    @staticmethod
    def _project_label(session: ProjectSession) -> str:
        """Short human label for the project. Prefers folder name."""
        try:
            name = session.project_root.name
        except Exception:
            name = ""
        if name:
            return name
        # Fallback to the first ~6 words of the goal.
        goal = session.refined_goal or session.user_intent or ""
        words = goal.strip().split()
        if not words:
            return "the project"
        return " ".join(words[:6])[:60]

    def _render_planning(self, session: ProjectSession) -> str:
        label = self._project_label(session)
        return f"Just sent the prompt for {label}. He's getting started."

    def _render_awaiting_clarification(self, session: ProjectSession) -> str:
        # Claude is blocked, Qwen hasn't yet decided. From the user's
        # perspective the system is "thinking through the question".
        question = ""
        if session.pending_clarification is not None:
            question = session.pending_clarification.question or ""
        if question:
            short = question.strip().rstrip("?.").strip()
            if len(short) > 80:
                short = short[:77].rsplit(" ", 1)[0] + "..."
            return f"He stopped to ask about {short}. I'm working through it."
        return "He stopped with a question. I'm working through it."

    def _render_awaiting_user(self, session: ProjectSession) -> str:
        question = ""
        if session.pending_clarification is not None:
            question = session.pending_clarification.question or ""
        topic = self._extract_topic(question) if question else ""
        if topic:
            return f"Waiting on you to answer the question about {topic}."
        return "Waiting on you to answer the question I just asked."

    def _render_verifying(self, session: ProjectSession) -> str:
        return "He says he's done. I'm running verification now."

    def _render_correcting(self, session: ProjectSession) -> str:
        # Surface the most-recent failure cause in plain language.
        failures = session.verification_failures
        if failures >= 2:
            return (
                f"Verification has failed {failures} times. He's working "
                f"on the fix; I'll escalate if it doesn't take."
            )
        return (
            "He thought he was done but verification turned up problems. "
            "He's fixing them now."
        )

    def _render_complete(self, session: ProjectSession) -> str:
        label = self._project_label(session)
        files_n = len(session.files_created) + len(session.files_modified)
        if session.completion_claim and session.completion_claim.summary:
            tail = session.completion_claim.summary.splitlines()[0].strip()
            tail = tail[:200]
            return f"Done with {label}. {tail}"
        if files_n > 0:
            return (
                f"Done with {label}. {files_n} file"
                f"{'s' if files_n != 1 else ''} touched."
            )
        return f"Done with {label}."

    def _render_failed(self, session: ProjectSession) -> str:
        return (
            f"Stopped on {self._project_label(session)}. Verification "
            f"failed too many times -- I'm surfacing it to you."
        )

    def _render_terminated(self, session: ProjectSession) -> str:
        return f"Cancelled {self._project_label(session)}."

    @staticmethod
    def _extract_topic(question: str) -> str:
        """Try to pull a short noun-phrase topic out of a clarification question."""
        # Heuristic: take everything after the first "about" / "for" /
        # "between" / "regarding"; failing that, the trailing noun phrase.
        q = (question or "").strip().rstrip("?.").strip()
        if not q:
            return ""
        m = re.search(
            r"\b(?:about|for|between|regarding|on|with)\s+(.+)$",
            q, flags=re.IGNORECASE,
        )
        if m:
            tail = m.group(1).strip()
            tail = re.sub(r"\s+(?:should|do|will|please).*$", "", tail, flags=re.I)
            if 3 <= len(tail) <= 80:
                return tail
        # Fallback: last 4-5 words.
        words = q.split()
        if len(words) > 6:
            return " ".join(words[-5:])[:80]
        return q[:80]

    # --- executing-path renderer -------------------------------------------

    def _render_executing(
        self, session: ProjectSession, delta: NarrationDelta,
    ) -> str:
        # If a clarification became pending since last query, lead with that
        # regardless of LLM availability -- it's the most important news.
        if delta.pending_clarification_arrived and session.pending_clarification:
            q = session.pending_clarification.question or ""
            topic = self._extract_topic(q)
            if topic:
                return f"He's stopped. He needs to know about {topic}. I was about to ask you."
            return "He's stopped with a question. I was about to ask you."

        if self.llm is None:
            return self._render_executing_fallback(session, delta)

        prompt = _NARRATION_PROMPT.format(
            goal=(session.refined_goal or session.user_intent or "(unspecified)")[:240],
            current_stage=session.current_stage or "starting",
            progress_summary=self._progress_summary(session),
            delta_summary=self._delta_summary(delta),
            test_summary=self._test_summary(session),
            clarification_note=(
                "A clarification is pending -- mention it.\n"
                if session.pending_clarification is not None else ""
            ),
        )
        try:
            raw = self._llm_call(prompt)
        except Exception as e:
            logger.warning("status-narration LLM call failed (%s); falling back", e)
            return self._render_executing_fallback(session, delta)
        text = _strip_thinking(raw)
        if not text:
            return self._render_executing_fallback(session, delta)
        return _first_sentence_pair(text)

    def _llm_call(self, prompt: str) -> str:
        """Run the sync LLM. Bumps LLM_MAX_TOKENS for the duration of the
        call so the narrator's tight budget doesn't get clipped by the
        global setting."""
        from config import settings as _settings

        old = getattr(_settings, "LLM_MAX_TOKENS", None)
        try:
            try:
                _settings.LLM_MAX_TOKENS = self.max_tokens
            except Exception:
                pass
            return self.llm.generate(prompt) or ""
        finally:
            if old is not None:
                try:
                    _settings.LLM_MAX_TOKENS = old
                except Exception:
                    pass

    def _render_executing_fallback(
        self, session: ProjectSession, delta: NarrationDelta,
    ) -> str:
        """Hand-rolled deterministic fallback. Used when no LLM is wired
        or the LLM call errors. Stays in voice but is mechanical."""
        current = session.current_stage or "getting started"
        if delta.is_first_query:
            files_total = len(session.files_created) + len(session.files_modified)
            tests = self._test_summary(session)
            head = f"He's {self._humanize_stage(current)}."
            tail_bits: List[str] = []
            if files_total > 0:
                tail_bits.append(
                    f"{files_total} file{'s' if files_total != 1 else ''} touched"
                )
            if session.test_status.last_updated is not None:
                tail_bits.append(tests)
            if tail_bits:
                return head + " " + ", ".join(tail_bits) + "."
            return head

        bits: List[str] = []
        n_new_files = len(delta.new_files_created)
        n_new_mod = len(delta.new_files_modified)
        if n_new_files:
            bits.append(
                f"{n_new_files} new file{'s' if n_new_files != 1 else ''}"
            )
        if n_new_mod:
            bits.append(
                f"{n_new_mod} modification{'s' if n_new_mod != 1 else ''}"
            )
        if delta.new_stages:
            stage_text = delta.new_stages[-1].stage
            bits.append(f"finished {self._humanize_stage(stage_text)}")
        if delta.test_status_changed:
            bits.append(self._test_summary(session))

        if not bits:
            return f"Still {self._humanize_stage(current)}. No new completed work since you last asked."
        return (
            f"Since you last asked: {', '.join(bits)}. "
            f"Now {self._humanize_stage(current)}."
        )

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _humanize_stage(stage: str) -> str:
        """Clean up a raw stage label for natural narration."""
        s = (stage or "").strip()
        if not s:
            return "working"
        s = s.replace("_", " ").replace("-", " ")
        # Drop trailing punctuation, lower-case the first word so it slots
        # into "He's <stage>." without colliding with a leading noun.
        s = s.rstrip(".:;!? ")
        if s and s[0].isupper():
            # Keep capitalization for proper nouns (e.g. "API"), otherwise
            # lower the first letter.
            if not s.split()[0].isupper():
                s = s[0].lower() + s[1:]
        return s

    @staticmethod
    def _progress_summary(session: ProjectSession) -> str:
        if not session.stages_completed:
            return "(nothing completed yet)"
        # Last 3 stages.
        recent = session.stages_completed[-3:]
        return "; ".join(
            f"{s.stage}: {s.summary[:60]}" for s in recent
        )

    @staticmethod
    def _delta_summary(delta: NarrationDelta) -> str:
        if delta.is_first_query:
            return "(this is the first status query)"
        bits: List[str] = []
        if delta.new_files_created:
            bits.append(
                f"{len(delta.new_files_created)} new file"
                f"{'s' if len(delta.new_files_created) != 1 else ''} created"
            )
        if delta.new_files_modified:
            bits.append(
                f"{len(delta.new_files_modified)} modified"
            )
        if delta.new_stages:
            stages = ", ".join(s.stage for s in delta.new_stages[-3:])
            bits.append(f"completed: {stages}")
        if delta.test_status_changed:
            bits.append("test results updated")
        if not bits:
            return f"nothing new in the last {int(delta.elapsed_since_last_query_s)} seconds"
        return "; ".join(bits)

    @staticmethod
    def _test_summary(session: ProjectSession) -> str:
        ts = session.test_status
        if ts.last_updated is None:
            return "no tests run yet"
        if ts.failing == 0 and ts.passing > 0:
            return f"{ts.passing} passing, 0 failing"
        if ts.failing > 0:
            return f"{ts.passing} passing, {ts.failing} failing"
        return "tests reported (no counts)"
