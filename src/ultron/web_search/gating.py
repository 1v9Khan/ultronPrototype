"""Two-stage web-search gate (hard rules + LLM pre-flight).

Stage 1 (rules, instant):
  - Strong YES: time-sensitive markers ("today", "now", "current", date
    references, weather/sports/stock/news indicators), URLs in the
    utterance.
  - Strong NO: questions about the user themselves (memory handles those),
    creative writing, opinions on already-discussed material.
  - Otherwise: UNCLEAR -> stage 2.

Stage 2 (LLM pre-flight pass):
  - Single short LLM call returning structured JSON: needs_search,
    confidence, reason, search_queries, knowledge_confidence,
    has_temporal_dependency.
  - Caller can use the same call to inform Phase 5's uncertainty
    response style (high/medium/low) without re-prompting.

Both layers run on the orchestrator's existing main LLM. The pre-flight
call is short (<=200 tokens out) so a single sequential call adds
~50-200 ms. Future work: run in parallel with retrieval per spec section
3.4 once parallel LLM access is wired.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterable, List, Optional

from ultron.utils.logging import get_logger

logger = get_logger("web_search.gating")


class GateDecision(str, Enum):
    SEARCH = "SEARCH"
    NO_SEARCH = "NO_SEARCH"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class GateVerdict:
    """Structured output of the gate. Carries enough metadata for the
    caller to (a) act, (b) write a useful audit log, and (c) feed
    uncertainty signals into Phase 5's response-style chooser."""

    decision: GateDecision
    confidence: str  # "high" | "medium" | "low"
    source: str  # "rule" | "preflight" | "default"
    reason: str
    search_queries: List[str] = field(default_factory=list)
    # Phase-5-friendly uncertainty signals (populated by the LLM pre-flight).
    knowledge_confidence: Optional[str] = None
    knowledge_source: Optional[str] = None
    has_temporal_dependency: Optional[bool] = None
    latency_ms: float = 0.0
    # V1-gap A2: context categories + memory-side search queries the
    # multi-pass retrieval path uses to fan out. Populated by the LLM
    # pre-flight pass; rule-only verdicts leave these empty so the
    # default single-pass retrieval applies.
    context_categories: List[str] = field(default_factory=list)
    memory_search_queries: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# B1: knowledge_source resolution
# ---------------------------------------------------------------------------


# Spec Part 5.2: knowledge_source ranges over five values that drive the
# Phase-5 response-style chooser:
#   - "weights":           model is answering from training data.
#   - "retrieved_memory":  prior conversation memory is in scope.
#   - "retrieved_facts":   a stored user fact is in scope.
#   - "web_search_needed": answer requires fresh web fetch.
#   - "unknown":           model can't tell where the answer would come from.
#
# This helper is the single resolution site so every GateVerdict
# construction site -- the four rule branches in classify_by_rules and
# the LLM preflight branch in classify_by_preflight -- maps inputs to the
# enumeration consistently.


def _resolve_knowledge_source(
    *,
    needs_search: bool,
    confidence: str,
    memory_snippets: Optional[Iterable] = None,
    rule_reason: Optional[str] = None,
) -> str:
    """Pick the spec's five-value ``knowledge_source`` label.

    Args:
        needs_search: did the gate decide a web search is required?
        confidence: ``"high"``/``"medium"``/``"low"`` knowledge confidence
            (LLM preflight uses this; rule branches pass ``"high"`` for
            their confident verdicts and ``"low"`` for the catch-all).
        memory_snippets: any prior-turn memory passed into the gate.
            Non-empty implies the LLM will lean on retrieved conversation
            context when answering.
        rule_reason: the ``reason`` field a rule branch is about to set.
            We inspect it for "personal/memory question" and "stored fact"
            markers so memory-flavoured rule decisions surface their
            source label.

    Returns:
        One of the five spec values. Order of precedence: web_search_needed
        wins (fresh data is always a stronger signal than weights/memory),
        then rule-derived memory/fact decisions, then memory snippets in
        scope, then high-confidence weights, then unknown.
    """
    if needs_search:
        return "web_search_needed"
    reason = (rule_reason or "").lower()
    if "personal" in reason or "memory question" in reason:
        return "retrieved_memory"
    if "stored fact" in reason or "retrieved fact" in reason:
        return "retrieved_facts"
    if memory_snippets:
        try:
            has_any = any(True for _ in memory_snippets)
        except TypeError:
            has_any = bool(memory_snippets)
        if has_any:
            return "retrieved_memory"
    if confidence == "high":
        return "weights"
    return "unknown"


# ---------------------------------------------------------------------------
# Stage 1: hard rules
# ---------------------------------------------------------------------------


# Time-sensitive markers -- if any fires the answer almost certainly
# changed since the model's training cutoff.
_TIME_SENSITIVE = re.compile(
    r"\b(?:"
    r"today|tonight|tomorrow|yesterday|"
    r"right\s+now|right\s+this\s+(?:moment|minute|second)|currently|current|"
    r"this\s+(?:week|month|year|morning|afternoon|evening|hour|minute)|"
    r"latest|recent|recently|"
    r"just\s+(?:released?|launched?|happened?|came\s+out|announced?|dropped?|shipped?|posted?|published?)|"
    r"newest|new(?:\s+release)?|"
    r"breaking|live|in\s+the\s+news|"
    r"upcoming|next\s+(?:week|month|year|game|match|episode)|"
    r"as\s+of"
    r")\b",
    re.IGNORECASE,
)

# Topical categories that change frequently enough to need fresh data.
_VOLATILE_TOPICS = re.compile(
    r"\b(?:"
    r"weather|forecast|temperature|rain(?:fall|ing)?|snow(?:fall|ing)?|"
    r"stock\s+(?:price|market)|share\s+price|ticker|"
    r"score|standings|playoff|"
    r"election\s+results?|polling|"
    r"flight\s+(?:status|delay)|train\s+schedule|"
    r"exchange\s+rate|currency|"
    r"box\s+office"
    r")\b",
    re.IGNORECASE,
)

# Year mentions that look post-cutoff. The training cutoff for Qwen3.5 is
# ~Feb 2026 per project memory; queries about 2026+ events should search.
# We don't try to be too clever -- any 4-digit year >= cutoff_year qualifies.
_YEAR_MARKER = re.compile(r"\b(20\d{2})\b")
_TRAINING_CUTOFF_YEAR = 2026

# Embedded URLs always need fetching (we can't make up page contents).
_URL_MARKER = re.compile(r"https?://\S+", re.IGNORECASE)

# Anti-search rules: things memory or the model alone should handle.
_PERSONAL_QUESTIONS = re.compile(
    r"\b(?:"
    r"what\s+(?:do|did)\s+(?:i|we)\s+|"
    r"do\s+(?:you|I)\s+(?:remember|recall)|"
    r"my\s+(?:name|address|preference|favorite|password|todo|schedule|calendar|note)|"
    r"who\s+am\s+I|"
    r"my\s+last\s+|"
    r"earlier\s+I\s+said"
    r")",
    re.IGNORECASE,
)

# Catch-all for clearly-stable factual / conceptual / philosophical
# questions. Fires only after all SEARCH rules (time-sensitive, volatile
# topics, post-cutoff years, URLs) have had their chance, so e.g.
# "What's the weather today?" still routes SEARCH (time-sensitive wins).
# This shortcut keeps the LLM preflight off the hot path for the
# overwhelming majority of stable-knowledge queries.
_STABLE_FACTUAL_REQUEST = re.compile(
    r"^\s*(?:(?:hey\s+|okay\s+|alright\s+)?ultron[\s,.\-:]+)?"
    r"(?:and\s+|but\s+|so\s+|also\s+|then\s+)?"
    r"(?:"
    r"what(?:'s|\s+is|\s+are|\s+was|\s+were|\s+does|\s+do|\s+did|\s+kind|\s+type|\s+sort|\s+about|\s+do\s+you\s+think)|"
    r"how(?:'s|\s+is|\s+does|\s+do|\s+did|\s+can|\s+could|\s+would|\s+should|\s+tall|\s+long|\s+much|\s+many|\s+big|\s+small|\s+fast|\s+hard|\s+heavy|\s+wide|\s+old|\s+far)|"
    r"why(?:'s|\s+is|\s+are|\s+was|\s+were|\s+does|\s+do|\s+did|\s+am)|"
    r"who(?:'s|\s+is|\s+was|\s+were|\s+are|\s+wrote|\s+made|\s+invented|\s+founded|\s+discovered|\s+created)|"
    r"where(?:'s|\s+is|\s+are|\s+was|\s+were|\s+does|\s+do|\s+did)|"
    r"when(?:'s|\s+is|\s+was|\s+did|\s+do|\s+does)|"
    r"which(?:\s+is|\s+are|\s+one|\s+of|\s+kind)|"
    r"are\s+you|"
    r"do\s+you\s+(?:think|believe|know|feel|understand|find|like|prefer)|"
    r"is\s+(?:it|that|this)\s+(?:true|possible|right|fair|safe|reasonable|necessary|enough|even)|"
    r"explain|describe|tell\s+me|walk\s+me\s+through|"
    r"define|"
    r"give\s+me\s+(?:an|a|some)?\s*(?:overview|background|summary|explanation|definition)"
    r")\b",
    re.IGNORECASE,
)

# Creative-writing / brainstorming -- no factual claims to verify.
_CREATIVE_TASKS = re.compile(
    r"\b(?:"
    r"write\s+(?:me\s+)?(?:an?\s+)?(?:poem|story|song|joke|haiku|essay|limerick|sonnet|email|letter|note|message)|"
    r"compose\s+(?:an?\s+)?(?:poem|song|message|letter|haiku|essay|limerick|sonnet|email|note)|"
    r"draft\s+(?:an?\s+)?(?:email|letter|message|note|reply|response|memo)|"
    r"brainstorm|"
    r"come\s+up\s+with\s+(?:names|ideas|titles|suggestions)|"
    r"imagine\s+(?:if|that)|"
    r"pretend\s+(?:to\s+be|you're|you\s+are)|"
    r"role[-\s]?play"
    r")",
    re.IGNORECASE,
)


def _preflight_call(llm, prompt: str, max_tokens: int) -> str:
    """Single greedy call OR self-consistency N-vote on the JSON output.

    Centralised so the verdict-parsing code below sees one ``raw`` string
    regardless of whether self-consistency was used. Returns ``""`` on
    error (callers treat that as "preflight failed -> default NO_SEARCH").
    """
    from ultron.config import get_config
    from ultron.llm.self_consistency import (
        json_winner_aggregator,
        run_self_consistency,
        should_apply_self_consistency,
    )

    cfg = get_config()
    use_consistency = should_apply_self_consistency("web_gating_preflight", cfg)

    def _one_call(temperature: float) -> str:
        try:
            out = llm._llm.create_chat_completion(  # noqa: SLF001
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (out["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            logger.warning("preflight LLM call failed: %s", e)
            return ""

    if not use_consistency:
        # Single greedy call (temperature=0.0 — same as before).
        return _one_call(0.0)

    sc = cfg.llm.self_consistency
    result = run_self_consistency(
        _one_call,
        n=sc.n,
        temperature=sc.temperature,
        aggregator=json_winner_aggregator,
    )
    return result.answer or ""


def classify_by_rules(utterance: str) -> Optional[GateVerdict]:
    """Stage-1 hard-rule classification.

    Returns a confident :class:`GateVerdict` if any rule fires, else
    ``None`` (caller should escalate to the LLM pre-flight pass).
    """
    text = (utterance or "").strip()
    if not text:
        reason = "empty utterance"
        return GateVerdict(
            GateDecision.NO_SEARCH, "high", "rule",
            reason,
            knowledge_source=_resolve_knowledge_source(
                needs_search=False, confidence="high", rule_reason=reason,
            ),
        )

    # Anti-search wins over time-sensitive: "what did I say earlier today"
    # is personal, not a web query.
    if _PERSONAL_QUESTIONS.search(text):
        reason = "personal/memory question"
        return GateVerdict(
            GateDecision.NO_SEARCH, "high", "rule",
            reason,
            has_temporal_dependency=False,
            knowledge_source=_resolve_knowledge_source(
                needs_search=False, confidence="high", rule_reason=reason,
            ),
        )
    if _CREATIVE_TASKS.search(text):
        reason = "creative / brainstorming task"
        return GateVerdict(
            GateDecision.NO_SEARCH, "high", "rule",
            reason,
            has_temporal_dependency=False,
            knowledge_source=_resolve_knowledge_source(
                needs_search=False, confidence="high", rule_reason=reason,
            ),
        )

    # Strong YES rules.
    url_match = _URL_MARKER.search(text)
    if url_match:
        reason = "embedded URL needs fetching"
        return GateVerdict(
            GateDecision.SEARCH, "high", "rule",
            reason,
            search_queries=[url_match.group(0)],
            has_temporal_dependency=True,
            knowledge_source=_resolve_knowledge_source(
                needs_search=True, confidence="high", rule_reason=reason,
            ),
        )

    if _TIME_SENSITIVE.search(text):
        reason = "time-sensitive marker"
        return GateVerdict(
            GateDecision.SEARCH, "high", "rule",
            reason,
            has_temporal_dependency=True,
            knowledge_source=_resolve_knowledge_source(
                needs_search=True, confidence="high", rule_reason=reason,
            ),
        )

    if _VOLATILE_TOPICS.search(text):
        reason = "topic with volatile factual content"
        return GateVerdict(
            GateDecision.SEARCH, "high", "rule",
            reason,
            has_temporal_dependency=True,
            knowledge_source=_resolve_knowledge_source(
                needs_search=True, confidence="high", rule_reason=reason,
            ),
        )

    for m in _YEAR_MARKER.finditer(text):
        year = int(m.group(1))
        if year >= _TRAINING_CUTOFF_YEAR:
            reason = (
                f"references year {year} "
                f"(>= training cutoff {_TRAINING_CUTOFF_YEAR})"
            )
            return GateVerdict(
                GateDecision.SEARCH, "high", "rule",
                reason,
                has_temporal_dependency=True,
                knowledge_source=_resolve_knowledge_source(
                    needs_search=True, confidence="high", rule_reason=reason,
                ),
            )

    # Catch-all NO_SEARCH for clear stable/conceptual question stems with
    # no time/volatile markers above. Confidence is "medium" -- we may be
    # wrong on edge cases like "Who is the current President?" which the
    # time-sensitive rule should have caught first; if a hard case slips
    # through, the LLM model will at worst answer from training.
    if _STABLE_FACTUAL_REQUEST.search(text):
        reason = "stable / conceptual question stem (no time / volatile markers)"
        return GateVerdict(
            GateDecision.NO_SEARCH, "medium", "rule",
            reason,
            has_temporal_dependency=False,
            knowledge_source=_resolve_knowledge_source(
                needs_search=False, confidence="medium", rule_reason=reason,
            ),
        )

    return None  # No rule fired; caller falls through to the LLM gate.


# ---------------------------------------------------------------------------
# Stage 2: LLM pre-flight pass
# ---------------------------------------------------------------------------


_PREFLIGHT_PROMPT = """You decide whether a query needs a fresh web search before answering, you rate your own knowledge confidence so the assistant can adjust its tone, and you suggest 2-4 short context categories the assistant could pull from prior conversation memory.

Query: {query}

Retrieved memory (from prior conversations, may be empty):
{memory_block}

Rules:
- needs_search=true if the query asks about specific facts that change (current events, prices, releases, statistics, dates after early 2026, named services that may have changed), OR if you'd otherwise have to fabricate verifiable facts.
- needs_search=false if the query is conceptual, philosophical, asks for opinion, is about the user's own context, or is creative.
- knowledge_confidence reflects how sure you are answering WITHOUT a web call: "high" if it's standard textbook material, "medium" if you'd hedge, "low" if you'd guess.
- has_temporal_dependency: would the correct answer change over time?
- search_queries: 1-3 concise search queries if needs_search; otherwise empty.
- context_categories: 0-4 SHORT category labels (each <8 words) describing background topics that would make the answer better. Examples: "user's coding preferences", "recent decisions about the project", "user's location". Empty array when none apply.
- memory_search_queries: 0-3 short reformulated queries to issue against prior conversation memory beyond the literal question (e.g., the literal query is "what should I do next on this?" so the memory query becomes "what was decided last about the project"). Empty when not useful.
- reason: one short sentence.

Return ONLY this JSON object, no commentary, no markdown fences:
{{"needs_search": <bool>, "knowledge_confidence": "high"|"medium"|"low", "has_temporal_dependency": <bool>, "search_queries": [<string>, ...], "context_categories": [<string>, ...], "memory_search_queries": [<string>, ...], "reason": "<one sentence>"}}
"""


def _build_memory_block(memory_snippets) -> str:
    if not memory_snippets:
        return "(none)"
    lines = []
    for s in memory_snippets[:5]:
        role = getattr(s, "role", "") or ""
        content = (getattr(s, "content", "") or "").strip().replace("\n", " ")
        if len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"- {role}: {content}")
    return "\n".join(lines)


def classify_by_preflight(
    llm,
    utterance: str,
    memory_snippets=None,
    max_tokens: int = 256,
) -> GateVerdict:
    """Stage-2 LLM-based classification.

    Issues a short call to the main LLM with a tight JSON schema prompt.
    On parse failure or empty answer, defaults to NO_SEARCH (the safer
    side -- a missed search adds a follow-up turn; a false-positive
    search burns API quota and adds latency).

    4B plan Item 6: when ``llm.self_consistency.enabled`` is True and
    site ``"web_gating_preflight"`` isn't disabled, the JSON output is
    sampled N times (at the configured non-zero temperature, since
    self-consistency requires diverse samples) and majority-voted via
    :func:`ultron.llm.self_consistency.json_winner_aggregator`. The
    winning JSON is parsed identically to the single-greedy path.
    Default OFF preserves byte-for-byte single-call behaviour.
    """
    t0 = time.monotonic()
    prompt = _PREFLIGHT_PROMPT.format(
        query=utterance.strip(),
        memory_block=_build_memory_block(memory_snippets),
    )

    raw = _preflight_call(llm, prompt, max_tokens)
    if not raw:
        return GateVerdict(
            GateDecision.NO_SEARCH, "low", "default",
            "preflight error",
            latency_ms=(time.monotonic() - t0) * 1000,
            knowledge_source=_resolve_knowledge_source(
                needs_search=False, confidence="low",
                memory_snippets=memory_snippets,
                rule_reason="preflight error",
            ),
        )

    parsed = _parse_preflight_json(raw)
    if parsed is None:
        logger.warning("preflight returned unparseable JSON: %r", raw[:200])
        return GateVerdict(
            GateDecision.NO_SEARCH, "low", "default",
            "preflight unparseable",
            latency_ms=(time.monotonic() - t0) * 1000,
            knowledge_source=_resolve_knowledge_source(
                needs_search=False, confidence="low",
                memory_snippets=memory_snippets,
                rule_reason="preflight unparseable",
            ),
        )

    needs_search = bool(parsed.get("needs_search"))
    confidence = str(parsed.get("knowledge_confidence", "medium")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    queries = [q for q in parsed.get("search_queries", []) if isinstance(q, str) and q.strip()]
    # V1-gap A2: optional category + memory-query enrichment from the
    # preflight. Older preflight payloads (and rule verdicts) won't have
    # these keys; back-compat is preserved by defaulting to [].
    context_categories = [
        c for c in parsed.get("context_categories", []) or []
        if isinstance(c, str) and c.strip()
    ][:4]
    memory_search_queries = [
        q for q in parsed.get("memory_search_queries", []) or []
        if isinstance(q, str) and q.strip()
    ][:3]
    reason = str(parsed.get("reason", "") or "").strip() or "preflight verdict"
    temporal = parsed.get("has_temporal_dependency")
    if not isinstance(temporal, bool):
        temporal = None

    decision = GateDecision.SEARCH if needs_search else GateDecision.NO_SEARCH
    # The gate's confidence in the SEARCH/NO_SEARCH call differs from the
    # model's knowledge confidence; record both. We bias the gate
    # confidence toward "medium" since the LLM is the one making the call.
    gate_confidence = "medium"
    if confidence == "high" and not needs_search:
        gate_confidence = "high"  # high confidence answer => high confidence no-search
    if needs_search and queries:
        gate_confidence = "high"  # explicit queries => high confidence search

    return GateVerdict(
        decision=decision,
        confidence=gate_confidence,
        source="preflight",
        reason=reason,
        search_queries=queries[:3],
        knowledge_confidence=confidence,
        knowledge_source=_resolve_knowledge_source(
            needs_search=needs_search, confidence=confidence,
            memory_snippets=memory_snippets, rule_reason=reason,
        ),
        has_temporal_dependency=temporal,
        latency_ms=(time.monotonic() - t0) * 1000,
        context_categories=context_categories,
        memory_search_queries=memory_search_queries,
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse_preflight_json(text: str) -> Optional[dict]:
    """Extract a single JSON object from the LLM's response.

    Tolerant of (a) <think>...</think> reasoning blocks, (b) markdown
    code fences, (c) prose preamble before the JSON body.
    """
    if not text:
        return None
    text = _THINK_RE.sub("", text).strip()

    candidates = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    # Greedy first balanced {...}.
    i = text.find("{")
    if i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[i: j + 1])
                    break
    candidates.append(text)

    for c in candidates:
        try:
            v = json.loads(c)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class WebSearchGate:
    """Decide whether an utterance needs a web search.

    Args:
        llm: an :class:`LLMEngine` (or anything with a ``._llm`` attribute
            exposing ``create_chat_completion``). Required for stage 2.
            ``None`` disables stage 2 -- callers get UNCERTAIN for anything
            rules can't decide.
    """

    def __init__(self, llm=None) -> None:
        self.llm = llm

    def classify(self, utterance: str, memory_snippets=None) -> GateVerdict:
        t0 = time.monotonic()
        rule_verdict = classify_by_rules(utterance)
        if rule_verdict is not None:
            rule_verdict.latency_ms = (time.monotonic() - t0) * 1000
            return rule_verdict
        if self.llm is None:
            return GateVerdict(
                GateDecision.UNCERTAIN, "low", "default",
                "no rule matched, no LLM available for preflight",
                latency_ms=(time.monotonic() - t0) * 1000,
                knowledge_source=_resolve_knowledge_source(
                    needs_search=False, confidence="low",
                    memory_snippets=memory_snippets,
                    rule_reason="no LLM available for preflight",
                ),
            )
        return classify_by_preflight(self.llm, utterance, memory_snippets)
