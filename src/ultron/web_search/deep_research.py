"""Bounded agentic deep-research loop over the FREE local-first search ladder.

Catalog 12 (clawhub-felo-search) T3 [YELLOW]. Felo's product differentiator
is an iterative server-side loop -- decompose the question -> search each
sub-question -> read -> identify gaps -> search again -> synthesize -- of
which the client only ever sees the final answer + the ``query_analysis``
sub-queries. Felo's own API is PAID (RED, not ported), but the LOOP PATTERN
is entirely implementable over ultron's existing FREE providers (SearxNG ->
Brave -> DuckDuckGo + Trafilatura -> Jina), and the catalog-11
:class:`~ultron.agent_loop.base.AgentLoop` base provides the exact
safety-instrumented skeleton -- crucially its load-bearing ``max_steps``
cap, which bounds the otherwise-unbounded "keep researching" autonomy.

:class:`DeepResearchLoop` subclasses :class:`AgentLoop`:

* **plan**: step 1 decomposes the question into focused sub-questions (one
  in-process LLM call); later steps run a gap analysis ("what's still
  unanswered given what we've found?") and return new sub-questions, or
  ``None`` to finish. Both LLM calls FAIL OPEN -- decompose falls back to
  searching the question verbatim; gap analysis falls back to "no gaps"
  (finish) so an LLM hiccup can never spin the loop.
* **act**: searches each new sub-question through the SAME
  :class:`~ultron.web_search.search.WebSearchExecutor` the normal path uses
  (so T1 reformulation + the provider/reader chains + the cross-encoder
  ranker + the ``web_results`` cache + the per-provider rate-limit tracker
  ALL apply for free), accumulating de-duplicated sources up to a hard cap.
* **is_done / action_succeeded / action_signature**: a step that adds no
  new sources finishes the loop (further searching won't help); an empty
  search is NOT a failure (the base's default fail-fast is overridden);
  the canonical sub-question set drives the base's loop detector.

The loop only GATHERS sources. Final-answer SYNTHESIS stays with the
orchestrator's existing search-augmented streaming path (so the answer
streams to TTS incrementally and citation handling is identical) --
:meth:`DeepResearchResult.to_payload` hands back a standard
:class:`SearchPayload`.

Because each step issues several full searches, a deep-research turn costs
~10-18 s; it is therefore EXPLICIT opt-in (the orchestrator triggers it only
when the user actually asks to "research X in depth" / "do a deep dive on
X", matched by :func:`match_deep_research`). The normal sub-second search
path is untouched.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from ultron.agent_loop.base import AgentLoop, StepRecord
from ultron.utils.logging import get_logger
from ultron.web_search.search import SearchPayload, SearchSource

logger = get_logger("web_search.deep_research")

#: Default research rounds before forced synthesis (the AgentLoop max_steps
#: cap). Step 1 decomposes; steps 2..N fill gaps.
DEFAULT_MAX_STEPS = 3
#: Default sub-questions generated per step.
DEFAULT_MAX_SUB_QUERIES_PER_STEP = 3
#: Default ranked sources kept per sub-question search.
DEFAULT_TOP_N_PER_QUERY = 3
#: Hard cap on accumulated sources fed to synthesis -- bounds the LLM prompt
#: size (n_ctx=8192) regardless of how many sub-questions ran.
DEFAULT_MAX_ACCUMULATED_SOURCES = 8


@dataclass
class DeepResearchResult:
    """Outcome of a deep-research run.

    ``sources`` is the accumulated, de-duplicated source set;
    ``sub_queries`` is every sub-question actually searched (the "research
    strategy" surfaced via T4). :meth:`to_payload` adapts it to the
    standard :class:`SearchPayload` the orchestrator's synthesis path
    consumes.
    """

    question: str
    sources: List[SearchSource] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    loop_status: str = "completed"
    steps: int = 0
    elapsed_s: float = 0.0

    def to_payload(self) -> SearchPayload:
        return SearchPayload(
            query=self.question,
            sources=list(self.sources),
            cache_hit=False,
            elapsed_ms=self.elapsed_s * 1000.0,
            notes=[f"deep_research:{self.loop_status}:{self.steps}steps"],
            queries=list(self.sub_queries),
        )


# ---------------------------------------------------------------------------
# LLM prompts (decompose + gap analysis)
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """Break this research question into {n} focused, keyword-rich web-search sub-questions that together cover it thoroughly. Each should target a DISTINCT facet (background, current state, comparisons, caveats, etc.). Do NOT just rephrase the question.

Return ONLY a JSON object, no commentary, no markdown fences:
{{"sub_questions": ["...", "..."]}}

Research question: {question}
"""

_GAP_PROMPT = """You are doing iterative web research. Given the research question and a digest of what has been found so far, list the KEY sub-questions that are still UNANSWERED and that another web search could resolve. If the findings already cover the question well, return an EMPTY list.

Return ONLY a JSON object, no commentary, no markdown fences (at most {n} items):
{{"gaps": ["...", "..."]}}

Research question: {question}

Findings so far:
{findings}
"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_json_list(text: str, key: str) -> List[str]:
    """Extract a list of strings under ``key`` (or a bare array) from an LLM
    response, tolerant of think-blocks / fences / prose. ``[]`` on failure."""
    if not text:
        return []
    text = _THINK_RE.sub("", text).strip()
    candidates: List[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    i = text.find("{")
    if i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[i : j + 1])
                    break
    candidates.append(text)
    for c in candidates:
        try:
            v = json.loads(c)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(v, dict):
            items = v.get(key) or []
        elif isinstance(v, list):
            items = v
        else:
            continue
        return [str(x).strip() for x in items if isinstance(x, str) and str(x).strip()]
    return []


class DeepResearchLoop(AgentLoop):
    """Bounded decompose -> search -> gap-fill loop over the free search ladder.

    Args:
        executor: a :class:`~ultron.web_search.search.WebSearchExecutor`
            (its ``.run`` is used per sub-question; ``None`` makes the loop
            a no-op that returns no sources).
        llm: an LLM engine exposing ``._llm.create_chat_completion`` (used
            for decomposition + gap analysis). ``None`` degrades to a single
            verbatim search of the question.
        max_steps: research rounds (AgentLoop cap). Must be positive.
        max_sub_queries_per_step: sub-questions per round.
        top_n_per_query: ranked sources kept per sub-question search.
        max_accumulated_sources: hard cap on the gathered source set.
        on_step: optional per-step callback (forwarded to AgentLoop; e.g.
            live narration).
        clock: monotonic time source (injectable for tests).
    """

    def __init__(
        self,
        *,
        executor,
        llm,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_sub_queries_per_step: int = DEFAULT_MAX_SUB_QUERIES_PER_STEP,
        top_n_per_query: int = DEFAULT_TOP_N_PER_QUERY,
        max_accumulated_sources: int = DEFAULT_MAX_ACCUMULATED_SOURCES,
        on_step=None,
        clock=time.monotonic,
    ) -> None:
        super().__init__(
            max_steps=max_steps, name="deep_research", on_step=on_step, clock=clock
        )
        self._executor = executor
        self._llm = llm
        self._max_sub = max(1, int(max_sub_queries_per_step))
        self._top_n = max(1, int(top_n_per_query))
        self._max_sources = max(1, int(max_accumulated_sources))
        # Per-run state (reset in research()).
        self._question = ""
        self._accumulated: List[SearchSource] = []
        self._searched_canonical: set[str] = set()
        self._all_sub_queries: List[str] = []

    # -- public entry --------------------------------------------------

    def research(self, question: str) -> DeepResearchResult:
        """Run the loop for ``question`` and return the gathered sources."""
        self._question = (question or "").strip()
        self._accumulated = []
        self._searched_canonical = set()
        self._all_sub_queries = []
        if not self._question:
            return DeepResearchResult(question="", loop_status="empty")
        result = self.run(goal=self._question)
        return DeepResearchResult(
            question=self._question,
            sources=list(self._accumulated),
            sub_queries=list(self._all_sub_queries),
            loop_status=result.status.value,
            steps=result.final_step,
            elapsed_s=result.elapsed_s,
        )

    # -- AgentLoop contract --------------------------------------------

    def plan(self, observation: Any, history: Sequence[StepRecord]) -> Any:
        # Stop early once the source cap is reached -- no point planning more.
        if len(self._accumulated) >= self._max_sources:
            return None
        if not history:
            subqs = self._decompose(self._question)
            return subqs or [self._question]
        gaps = self._identify_gaps(self._question, self._accumulated)
        # Drop gaps we've already searched; if none remain, we're done.
        fresh = [
            g for g in gaps if g.strip().lower() not in self._searched_canonical
        ]
        return fresh or None

    def act(self, action: Any) -> Any:
        sub_questions = action if isinstance(action, (list, tuple)) else [action]
        new_sources = 0
        for raw in sub_questions:
            subq = str(raw).strip()
            canonical = subq.lower()
            if not subq or canonical in self._searched_canonical:
                continue
            if len(self._accumulated) >= self._max_sources:
                break
            self._searched_canonical.add(canonical)
            self._all_sub_queries.append(subq)
            if self._executor is None:
                continue
            try:
                payload = self._executor.run(subq, [subq], self._top_n)
            except Exception as e:  # noqa: BLE001 -- one bad sub-query never aborts
                logger.debug("deep-research sub-query %r failed: %s", subq, e)
                continue
            for s in getattr(payload, "sources", None) or []:
                if any(s.url == existing.url for existing in self._accumulated):
                    continue
                self._accumulated.append(s)
                new_sources += 1
                if len(self._accumulated) >= self._max_sources:
                    break
        return {"new_sources": new_sources, "total": len(self._accumulated)}

    def action_succeeded(self, result: Any) -> bool:
        # A completed search round is success even when it finds nothing new
        # -- the gap analysis / is_done decide whether to continue, NOT the
        # base's default fail-fast (which would abort on an empty search).
        return result is not None

    def action_signature(self, action: Any) -> str:
        if isinstance(action, (list, tuple)):
            return " | ".join(sorted(str(a).strip().lower() for a in action))
        return str(action).strip().lower()

    def is_done(self, result: Any, history: Sequence[StepRecord]) -> bool:
        if len(self._accumulated) >= self._max_sources:
            return True
        # A round that added zero new sources means further searching is
        # unlikely to help -- finish with what we have (if anything).
        if (
            isinstance(result, dict)
            and result.get("new_sources", 0) == 0
            and self._accumulated
        ):
            return True
        return False

    # -- LLM helpers (fail-open) ---------------------------------------

    def _decompose(self, question: str) -> List[str]:
        raw = self._llm_json(_DECOMPOSE_PROMPT.format(n=self._max_sub, question=question))
        return _dedupe_subqueries(_parse_json_list(raw, "sub_questions"), self._max_sub)

    def _identify_gaps(self, question: str, sources: List[SearchSource]) -> List[str]:
        findings = self._findings_digest(sources)
        raw = self._llm_json(
            _GAP_PROMPT.format(n=self._max_sub, question=question, findings=findings)
        )
        return _dedupe_subqueries(_parse_json_list(raw, "gaps"), self._max_sub)

    def _llm_json(self, prompt: str) -> str:
        """Single short in-process LLM call; FAIL-OPEN to ``""``."""
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
            logger.debug("deep-research LLM call failed: %s", e)
            return ""

    @staticmethod
    def _findings_digest(sources: List[SearchSource], *, max_items: int = 8) -> str:
        if not sources:
            return "(nothing found yet)"
        lines: List[str] = []
        for s in sources[:max_items]:
            title = (s.title or s.url or "").strip()
            preview = (s.snippet or "").strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:160] + "..."
            lines.append(f"- {title}: {preview}")
        return "\n".join(lines)


def _dedupe_subqueries(queries: List[str], cap: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for q in queries:
        qs = (q or "").strip()
        key = qs.lower()
        if not qs or key in seen:
            continue
        seen.add(key)
        out.append(qs)
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
# Voice-intent matcher (run-loop short-circuit; no new RoutingIntentKind)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepResearchMatch:
    """A matched deep-research request. ``topic`` is the subject to research."""

    topic: str


# Each pattern requires an EXPLICIT deep / thorough / in-depth marker so a
# normal "search X" / "look up X" / "what is X" never trips deep research.
# Ordered most-specific first; the first match wins.
_DEEP_RESEARCH_PATTERNS = [
    # "research X in depth / in detail / thoroughly / exhaustively"
    re.compile(
        r"\bresearch\s+(?P<topic>.+?)\s+(?:in\s+depth|in\s+detail|thoroughly|exhaustively|deeply)\b",
        re.IGNORECASE,
    ),
    # "deeply / thoroughly / exhaustively research X"
    re.compile(
        r"\b(?:deeply|thoroughly|exhaustively)\s+research\s+(?P<topic>.+)$",
        re.IGNORECASE,
    ),
    # "do a deep dive on/into X", "deep dive on X", "give me a deep dive into X"
    re.compile(
        r"\bdeep[\s-]?dive\s+(?:on|into|about|of|for)\s+(?P<topic>.+)$",
        re.IGNORECASE,
    ),
    # "do a deep/thorough/in-depth/exhaustive/comprehensive
    #  research/search/investigation/analysis/dive on/into/about X"
    re.compile(
        r"\b(?:do|run|perform|conduct|give\s+me|i\s+want)?\s*(?:a\s+|an\s+|some\s+)?"
        r"(?:deep|thorough|in[\s-]?depth|exhaustive|comprehensive)\s+"
        r"(?:research|search|investigation|analysis|dive|study)\s+"
        r"(?:on|into|about|of|for|regarding)\s+(?P<topic>.+)$",
        re.IGNORECASE,
    ),
    # "dig deep(er) into/on X"
    re.compile(
        r"\bdig\s+deep(?:er)?\s+(?:into|on|about)\s+(?P<topic>.+)$",
        re.IGNORECASE,
    ),
]

#: A topic shorter than this (after stripping) is treated as no real subject.
_MIN_TOPIC_CHARS = 2


def match_deep_research(text: str) -> Optional[DeepResearchMatch]:
    """Detect an explicit "research X in depth" / "deep dive on X" request.

    Returns a :class:`DeepResearchMatch` with the extracted ``topic`` when
    the utterance carries an explicit deep / thorough / in-depth research
    marker, else ``None`` (so normal search / conversational queries fall
    through untouched). Strict by design -- "search X", "look up X", "what
    is X" do NOT match.
    """
    t = (text or "").strip()
    if not t:
        return None
    for pattern in _DEEP_RESEARCH_PATTERNS:
        m = pattern.search(t)
        if not m:
            continue
        topic = (m.group("topic") or "").strip().strip("?.!,;:").strip()
        # Drop a leading filler article the patterns may have left.
        topic = re.sub(r"^(?:the|a|an)\s+", "", topic, flags=re.IGNORECASE).strip()
        if len(topic) >= _MIN_TOPIC_CHARS:
            return DeepResearchMatch(topic=topic)
    return None


__all__ = [
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_SUB_QUERIES_PER_STEP",
    "DEFAULT_TOP_N_PER_QUERY",
    "DEFAULT_MAX_ACCUMULATED_SOURCES",
    "DeepResearchResult",
    "DeepResearchLoop",
    "DeepResearchMatch",
    "match_deep_research",
]
