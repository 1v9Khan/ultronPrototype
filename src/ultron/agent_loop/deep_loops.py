"""Cross-system deep-gather loops -- the catalog-12 (felo-search T3) creative
extensions of the deep-research pattern, generalised beyond web search.

The :class:`~ultron.web_search.deep_research.DeepResearchLoop` proved the
pattern for web search. The catalog's insight is that the SAME bounded
"decompose the question -> gather over a source -> identify gaps -> gather
the gaps -> stop (max_steps cap)" loop applies to ultron's other retrieval
surfaces:

* **memory** -- iterative RAG over the Qdrant conversation store, for
  complex multi-faceted recall that a single retrieve pass under-serves;
* **codebase** -- iterative ripgrep exploration ("how does X work across
  this repo");
* **desktop UI** -- iterative UIA element discovery for complex / dynamic
  windows where a single semantic-name lookup misses.

Rather than copy the loop three times, this module factors the shared logic
into one generic :class:`DeepGatherLoop` (a :class:`~ultron.agent_loop.base.AgentLoop`
subclass) parameterised by three injected callables -- ``gather`` (a
sub-query string -> an iterable of domain items), ``item_key`` (item ->
hashable dedup key), and ``item_summary`` (item -> a short string for the
gap-analysis digest) -- plus the two domain LLM prompts. The three named
loops (:class:`DeepMemoryLoop` / :class:`DeepExplorationLoop` /
:class:`DeepUIDiscoveryLoop`) are thin subclasses that wire their domain's
primitive in via dependency injection, which also makes them trivially
testable with fakes (no Qdrant / ripgrep / live desktop needed).

Every layer FAILS OPEN: the decompose / gap-analysis LLM calls degrade to
"search the question verbatim" / "stop", and a raising ``gather`` for one
sub-query is logged + skipped rather than aborting the loop. The base
class's ``max_steps`` cap is the load-bearing safety invariant.

These ship as importable primitives with a clean public entry method
(``recall`` / ``explore`` / ``discover``); wiring them to a concrete
trigger (a voice command, a coding-agent tool, a click-fallback) is a
one-call integration left to the consuming surface.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Hashable, Iterable, List, Optional, Sequence

from ultron.agent_loop.base import AgentLoop, StepRecord
from ultron.utils.logging import get_logger

# Reuse the tolerant JSON-list parser + sub-query dedup from the web
# deep-research module (same package-family helpers; keeps the prompt-output
# parsing identical across every deep loop).
from ultron.web_search.deep_research import _dedupe_subqueries, _parse_json_list

logger = get_logger("agent_loop.deep_loops")

DEFAULT_MAX_STEPS = 3
DEFAULT_MAX_SUB_QUERIES_PER_STEP = 3
DEFAULT_MAX_ACCUMULATED = 8


from dataclasses import dataclass, field


@dataclass
class DeepGatherResult:
    """Outcome of a deep-gather run.

    ``items`` is the accumulated, de-duplicated domain item set (memory
    turns / ripgrep matches / UI element matches); ``sub_queries`` is every
    sub-query actually issued (the research strategy).
    """

    kind: str
    question: str
    items: List[Any] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    loop_status: str = "completed"
    steps: int = 0
    elapsed_s: float = 0.0

    @property
    def found(self) -> int:
        return len(self.items)


class DeepGatherLoop(AgentLoop):
    """Generic bounded decompose -> gather -> gap-fill loop.

    Args:
        llm: engine exposing ``._llm.create_chat_completion`` (decompose +
            gap analysis). ``None`` degrades to a single verbatim gather.
        kind: short domain label (``"memory"`` / ``"code"`` / ``"ui"``) used
            in logs + the loop name.
        gather: ``Callable[[str], Iterable[item]]`` -- issue one sub-query
            against the domain source and return its items.
        item_key: ``Callable[[item], Hashable]`` -- a stable dedup key.
        item_summary: ``Callable[[item], str]`` -- a short string used to
            build the gap-analysis findings digest.
        decompose_prompt / gap_prompt: domain prompts. ``decompose_prompt``
            is ``.format(n=, query=)``; ``gap_prompt`` is
            ``.format(n=, query=, findings=)``. Both must instruct the model
            to emit ``{"sub_questions": [...]}`` / ``{"gaps": [...]}``.
        max_steps / max_sub_queries_per_step / max_accumulated: bounds.
    """

    def __init__(
        self,
        *,
        llm,
        kind: str,
        gather: Callable[[str], Iterable[Any]],
        item_key: Callable[[Any], Hashable],
        item_summary: Callable[[Any], str],
        decompose_prompt: str,
        gap_prompt: str,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_sub_queries_per_step: int = DEFAULT_MAX_SUB_QUERIES_PER_STEP,
        max_accumulated: int = DEFAULT_MAX_ACCUMULATED,
        on_step=None,
        clock=time.monotonic,
    ) -> None:
        super().__init__(
            max_steps=max_steps, name=f"deep_{kind}", on_step=on_step, clock=clock
        )
        self._llm = llm
        self._kind = kind
        self._gather = gather
        self._item_key = item_key
        self._item_summary = item_summary
        self._decompose_prompt = decompose_prompt
        self._gap_prompt = gap_prompt
        self._max_sub = max(1, int(max_sub_queries_per_step))
        self._max_acc = max(1, int(max_accumulated))
        # per-run state
        self._question = ""
        self._items: List[Any] = []
        self._keys: set = set()
        self._searched: set = set()
        self._all_sub_queries: List[str] = []

    def gather_for(self, question: str) -> DeepGatherResult:
        """Run the loop for ``question`` and return the gathered items."""
        self._question = (question or "").strip()
        self._items = []
        self._keys = set()
        self._searched = set()
        self._all_sub_queries = []
        if not self._question:
            return DeepGatherResult(kind=self._kind, question="", loop_status="empty")
        result = self.run(goal=self._question)
        return DeepGatherResult(
            kind=self._kind,
            question=self._question,
            items=list(self._items),
            sub_queries=list(self._all_sub_queries),
            loop_status=result.status.value,
            steps=result.final_step,
            elapsed_s=result.elapsed_s,
        )

    # -- AgentLoop contract --------------------------------------------

    def plan(self, observation: Any, history: Sequence[StepRecord]) -> Any:
        if len(self._items) >= self._max_acc:
            return None
        if not history:
            subqs = self._decompose(self._question)
            return subqs or [self._question]
        gaps = self._gap_fill(self._question)
        fresh = [g for g in gaps if g.strip().lower() not in self._searched]
        return fresh or None

    def act(self, action: Any) -> Any:
        subs = action if isinstance(action, (list, tuple)) else [action]
        new = 0
        for raw in subs:
            q = str(raw).strip()
            canonical = q.lower()
            if not q or canonical in self._searched:
                continue
            if len(self._items) >= self._max_acc:
                break
            self._searched.add(canonical)
            self._all_sub_queries.append(q)
            try:
                items = self._gather(q) or []
            except Exception as e:  # noqa: BLE001 -- one bad sub-query never aborts
                logger.debug("deep_%s gather %r failed: %s", self._kind, q, e)
                continue
            for it in items:
                try:
                    key = self._item_key(it)
                except Exception:  # noqa: BLE001
                    continue
                if key in self._keys:
                    continue
                self._keys.add(key)
                self._items.append(it)
                new += 1
                if len(self._items) >= self._max_acc:
                    break
        return {"new": new, "total": len(self._items)}

    def action_succeeded(self, result: Any) -> bool:
        return result is not None

    def action_signature(self, action: Any) -> str:
        if isinstance(action, (list, tuple)):
            return " | ".join(sorted(str(a).strip().lower() for a in action))
        return str(action).strip().lower()

    def is_done(self, result: Any, history: Sequence[StepRecord]) -> bool:
        if len(self._items) >= self._max_acc:
            return True
        if isinstance(result, dict) and result.get("new", 0) == 0 and self._items:
            return True
        return False

    # -- LLM helpers (fail-open) ---------------------------------------

    def _decompose(self, question: str) -> List[str]:
        raw = self._llm_json(self._decompose_prompt.format(n=self._max_sub, query=question))
        return _dedupe_subqueries(_parse_json_list(raw, "sub_questions"), self._max_sub)

    def _gap_fill(self, question: str) -> List[str]:
        findings = self._findings_digest()
        raw = self._llm_json(
            self._gap_prompt.format(n=self._max_sub, query=question, findings=findings)
        )
        return _dedupe_subqueries(_parse_json_list(raw, "gaps"), self._max_sub)

    def _llm_json(self, prompt: str) -> str:
        if self._llm is None:
            return ""
        try:
            user_msg = prompt if "/no_think" in prompt else prompt.rstrip() + " /no_think"
            out = self._llm._llm.create_chat_completion(  # noqa: SLF001
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=192,
            )
            raw = (out["choices"][0]["message"]["content"] or "").strip()
            try:
                from ultron.llm.inference import strip_thinking_text

                raw = strip_thinking_text(raw).strip()
            except Exception:  # noqa: BLE001
                pass
            return raw
        except Exception as e:  # noqa: BLE001
            logger.debug("deep_%s LLM call failed: %s", self._kind, e)
            return ""

    def _findings_digest(self, *, max_items: int = 8) -> str:
        if not self._items:
            return "(nothing found yet)"
        lines: List[str] = []
        for it in self._items[:max_items]:
            try:
                lines.append(f"- {self._item_summary(it)}")
            except Exception:  # noqa: BLE001
                continue
        return "\n".join(lines) if lines else "(nothing found yet)"


# ---------------------------------------------------------------------------
# Domain prompts
# ---------------------------------------------------------------------------

_MEMORY_DECOMPOSE = """Break this recall question into {n} focused sub-queries to run against a personal conversation-memory store. Each should target a distinct facet (decisions made, preferences stated, facts mentioned, timeline). Return ONLY JSON: {{"sub_questions": ["...", "..."]}}.

Recall question: {query}
"""
_MEMORY_GAP = """You are doing iterative recall over conversation memory. Given the recall question and what's been retrieved, list the KEY facets still UNCOVERED that another memory query could surface. Empty list if well covered. Return ONLY JSON (at most {n}): {{"gaps": ["...", "..."]}}.

Recall question: {query}

Retrieved so far:
{findings}
"""

_CODE_DECOMPOSE = """Break this codebase question into {n} focused search patterns to run against a code repository (function names, class names, key identifiers, distinctive strings). Return ONLY JSON: {{"sub_questions": ["...", "..."]}}.

Codebase question: {query}
"""
_CODE_GAP = """You are exploring a codebase iteratively. Given the question and the matches found so far, list the KEY follow-up search patterns still needed (callers, imports, related identifiers). Empty list if the picture is complete. Return ONLY JSON (at most {n}): {{"gaps": ["...", "..."]}}.

Codebase question: {query}

Matches so far:
{findings}
"""

_UI_DECOMPOSE = """Break this UI-target description into {n} candidate element names / labels to look for on screen (the literal label, common synonyms, an accessibility-style name). Return ONLY JSON: {{"sub_questions": ["...", "..."]}}.

UI target: {query}
"""
_UI_GAP = """You are locating a UI element iteratively. Given the target and the elements found so far, list other candidate names / labels still worth trying (synonyms, adjacent controls, parent menus). Empty list if the target is clearly found. Return ONLY JSON (at most {n}): {{"gaps": ["...", "..."]}}.

UI target: {query}

Elements found so far:
{findings}
"""


# ---------------------------------------------------------------------------
# Named domain loops (thin subclasses)
# ---------------------------------------------------------------------------


def _memory_key(turn: Any) -> Hashable:
    tid = getattr(turn, "id", None)
    if tid:
        return ("id", tid)
    return ("rc", getattr(turn, "role", ""), (getattr(turn, "content", "") or "")[:200])


def _memory_summary(turn: Any) -> str:
    role = getattr(turn, "role", "") or ""
    content = (getattr(turn, "content", "") or "").strip().replace("\n", " ")
    if len(content) > 160:
        content = content[:160] + "..."
    return f"{role}: {content}"


class DeepMemoryLoop(DeepGatherLoop):
    """Iterative RAG over conversation memory (felo-search T3 extension).

    ``retrieve`` is a ``Callable[[str, int], Iterable[turn]]`` -- typically
    a thin adapter over :meth:`ConversationMemory.retrieve`. Items are
    memory turns (anything exposing ``.id`` / ``.role`` / ``.content``).
    """

    def __init__(
        self,
        *,
        retrieve: Callable[[str, int], Iterable[Any]],
        llm,
        k_per_query: int = 3,
        max_steps: int = 2,
        max_sub_queries_per_step: int = DEFAULT_MAX_SUB_QUERIES_PER_STEP,
        max_accumulated: int = DEFAULT_MAX_ACCUMULATED,
        on_step=None,
        clock=time.monotonic,
    ) -> None:
        k = max(1, int(k_per_query))
        super().__init__(
            llm=llm,
            kind="memory",
            gather=lambda q: retrieve(q, k),
            item_key=_memory_key,
            item_summary=_memory_summary,
            decompose_prompt=_MEMORY_DECOMPOSE,
            gap_prompt=_MEMORY_GAP,
            max_steps=max_steps,
            max_sub_queries_per_step=max_sub_queries_per_step,
            max_accumulated=max_accumulated,
            on_step=on_step,
            clock=clock,
        )

    def recall(self, question: str) -> DeepGatherResult:
        return self.gather_for(question)


def _code_key(match: Any) -> Hashable:
    # ripgrep-style match: prefer (file, line); fall back to str().
    fp = getattr(match, "file_path", None) or getattr(match, "path", None)
    ln = getattr(match, "line_number", None) or getattr(match, "line", None)
    if fp is not None:
        return ("fl", str(fp), ln)
    return ("s", str(match))


def _code_summary(match: Any) -> str:
    fp = getattr(match, "file_path", None) or getattr(match, "path", None) or "?"
    ln = getattr(match, "line_number", None) or getattr(match, "line", None) or "?"
    text = (getattr(match, "text", None) or getattr(match, "line_text", None) or "")
    text = str(text).strip().replace("\n", " ")
    if len(text) > 120:
        text = text[:120] + "..."
    return f"{fp}:{ln}: {text}"


class DeepExplorationLoop(DeepGatherLoop):
    """Iterative codebase exploration via ripgrep (felo-search T3 extension).

    ``search`` is a ``Callable[[str], Iterable[match]]`` -- typically a thin
    adapter over :func:`ultron.search.ripgrep.regex_search_files` returning
    its match rows (objects exposing ``.file_path`` / ``.line_number`` /
    ``.text``, or any object -- the key/summary fall back to ``str()``).
    """

    def __init__(
        self,
        *,
        search: Callable[[str], Iterable[Any]],
        llm,
        max_steps: int = 4,
        max_sub_queries_per_step: int = DEFAULT_MAX_SUB_QUERIES_PER_STEP,
        max_accumulated: int = 20,
        on_step=None,
        clock=time.monotonic,
    ) -> None:
        super().__init__(
            llm=llm,
            kind="code",
            gather=search,
            item_key=_code_key,
            item_summary=_code_summary,
            decompose_prompt=_CODE_DECOMPOSE,
            gap_prompt=_CODE_GAP,
            max_steps=max_steps,
            max_sub_queries_per_step=max_sub_queries_per_step,
            max_accumulated=max_accumulated,
            on_step=on_step,
            clock=clock,
        )

    def explore(self, question: str) -> DeepGatherResult:
        return self.gather_for(question)


def _ui_key(el: Any) -> Hashable:
    return (
        "ui",
        getattr(el, "window", None) or getattr(el, "window_title", None),
        getattr(el, "name", None),
        getattr(el, "control_type", None),
    )


def _ui_summary(el: Any) -> str:
    name = getattr(el, "name", None) or "?"
    ctype = getattr(el, "control_type", None) or "?"
    window = getattr(el, "window", None) or getattr(el, "window_title", None) or "?"
    return f"{name} ({ctype}) @ {window}"


class DeepUIDiscoveryLoop(DeepGatherLoop):
    """Iterative UIA element discovery (felo-search T3 extension).

    ``find`` is a ``Callable[[str], Iterable[match]]`` -- typically a thin
    adapter over :func:`ultron.desktop.element_click.find_elements_by_name`
    returning its match rows (objects exposing ``.name`` / ``.control_type``
    / ``.window``). Intended as a FALLBACK when a single semantic-name
    lookup misses on a complex / dynamically-loaded window.
    """

    def __init__(
        self,
        *,
        find: Callable[[str], Iterable[Any]],
        llm,
        max_steps: int = 3,
        max_sub_queries_per_step: int = DEFAULT_MAX_SUB_QUERIES_PER_STEP,
        max_accumulated: int = 12,
        on_step=None,
        clock=time.monotonic,
    ) -> None:
        super().__init__(
            llm=llm,
            kind="ui",
            gather=find,
            item_key=_ui_key,
            item_summary=_ui_summary,
            decompose_prompt=_UI_DECOMPOSE,
            gap_prompt=_UI_GAP,
            max_steps=max_steps,
            max_sub_queries_per_step=max_sub_queries_per_step,
            max_accumulated=max_accumulated,
            on_step=on_step,
            clock=clock,
        )

    def discover(self, target: str) -> DeepGatherResult:
        return self.gather_for(target)


__all__ = [
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_SUB_QUERIES_PER_STEP",
    "DEFAULT_MAX_ACCUMULATED",
    "DeepGatherResult",
    "DeepGatherLoop",
    "DeepMemoryLoop",
    "DeepExplorationLoop",
    "DeepUIDiscoveryLoop",
]
