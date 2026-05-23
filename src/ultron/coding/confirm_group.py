"""Batched confirmation prompts (catalog T14, 2026-05-22 batch 14).

Lifted in spirit from aider's ``ConfirmGroup``: instead of asking the
user one yes/no per side effect (which produces a noisy, draining
voice loop), the orchestrator accumulates a small set of related
confirmation items and asks ONE batched question covering all of
them.

Example::

    cg = ConfirmGroup()
    cg.add("Modify foo.py")
    cg.add("Modify bar.py")
    cg.add("Delete baz.py")

    if cg.is_pending():
        question = cg.render_question()
        # speak `question`, wait for yes/no, then call .resolve(True)
        # or .resolve(False)

The class is intentionally minimal: it stores items, renders a
single TTS-safe question, and tracks resolution state. The decision
about HOW to ask the question (voice prompt, text prompt, GUI
modal, etc.) and HOW to collect the answer is the caller's
responsibility.

Public surface:

  * :class:`ConfirmGroup` -- the batch container.
  * :class:`ConfirmGroupResolution` -- frozen result dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfirmGroupResolution:
    """Outcome of one confirmation cycle.

    Attributes:
        approved: True when the user said yes, False on no.
        items: The items that were in the batch at resolution time.
        question: The rendered question shown to the user.
    """

    approved: bool
    items: tuple
    question: str


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


class ConfirmGroup:
    """A small bounded buffer of confirmation items.

    Items are deduplicated by string equality so repeated ``add()`` calls
    for the same action don't bloat the question. The group can be
    resolved exactly once -- a second :meth:`resolve` call raises.

    Args:
        prefix: Lead-in text. Default ``"I'll"``. Plain ASCII; the
            voice layer will narrate this verbatim.
        max_items: Cap on stored items. Adds beyond this point are
            recorded as ``"...and N more"`` so the rendered question
            stays short.
    """

    def __init__(
        self,
        *,
        prefix: str = "I'll",
        max_items: int = 6,
    ) -> None:
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self._prefix = prefix
        self._max_items = int(max_items)
        self._items: List[str] = []
        self._overflow_count = 0
        self._resolved: Optional[ConfirmGroupResolution] = None

    # --- mutation ---------------------------------------------------------

    def add(self, item: str) -> None:
        """Add an item to the batch. No-op for empty / duplicate strings."""
        if self._resolved is not None:
            raise RuntimeError(
                "ConfirmGroup already resolved; construct a new one."
            )
        cleaned = (item or "").strip()
        if not cleaned:
            return
        if cleaned in self._items:
            return
        if len(self._items) >= self._max_items:
            self._overflow_count += 1
            return
        self._items.append(cleaned)

    def add_many(self, items) -> None:
        for it in items:
            self.add(it)

    # --- query -----------------------------------------------------------

    def is_pending(self) -> bool:
        """True when there are unresolved items waiting for confirmation."""
        return self._resolved is None and bool(self._items)

    def items(self) -> List[str]:
        """Snapshot of currently-buffered items."""
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: str) -> bool:
        return (item or "").strip() in self._items

    # --- render ----------------------------------------------------------

    def render_question(self) -> str:
        """Build the batched yes/no question.

        Empty buffer renders an empty string; otherwise the form
        depends on item count:

        * 1 item: ``"I'll <item>. Okay?"``
        * 2 items: ``"I'll <a> and <b>. Okay?"``
        * 3+ items: ``"I'll <a>, <b>, and <c>. Okay?"``

        Overflow items are summarised as ``"...and N more"`` before
        the comma-separated tail.
        """
        if not self._items:
            return ""
        items = list(self._items)
        if self._overflow_count > 0:
            items = items + [f"and {self._overflow_count} more"]
        if len(items) == 1:
            return f"{self._prefix} {items[0]}. Okay?"
        if len(items) == 2:
            return f"{self._prefix} {items[0]} and {items[1]}. Okay?"
        body = ", ".join(items[:-1])
        return f"{self._prefix} {body}, and {items[-1]}. Okay?"

    # --- resolution ------------------------------------------------------

    def resolve(self, approved: bool) -> ConfirmGroupResolution:
        """Record the user's yes/no and freeze the group.

        Raises:
            RuntimeError: when called twice.
        """
        if self._resolved is not None:
            raise RuntimeError("ConfirmGroup already resolved.")
        question = self.render_question()
        self._resolved = ConfirmGroupResolution(
            approved=bool(approved),
            items=tuple(self._items),
            question=question,
        )
        return self._resolved

    @property
    def resolution(self) -> Optional[ConfirmGroupResolution]:
        """The resolution if :meth:`resolve` has been called, else None."""
        return self._resolved


__all__ = [
    "ConfirmGroup",
    "ConfirmGroupResolution",
]
