"""Web search subsystem (Phase 4).

Brave Search API + Jina Reader for clean markdown extraction. A two-stage
gate decides whether a query needs the web at all:

  1. Hard rules (instant, no LLM): time-sensitive markers, fresh-data
     categories, embedded URLs.
  2. LLM pre-flight pass for everything not caught by stage 1.

Results cache into the ``web_results`` Qdrant collection keyed by the
search query so repeated queries within the freshness window skip the
API call entirely.
"""

from ultron.web_search.acknowledgments import AcknowledgmentSource
from ultron.web_search.brave import (
    BraveResult,  # deprecated alias for backward-compat
    BraveSearchClient,
    SearchResult,
)
from ultron.web_search.cache import WebResultsCache
from ultron.web_search.gating import GateDecision, GateVerdict, WebSearchGate
from ultron.web_search.jina import JinaReaderClient
from ultron.web_search.search import (
    SearchPayload,
    SearchSource,
    WebSearchExecutor,
    format_sources_for_prompt,
    format_sources_for_transcript,
)

__all__ = [
    "AcknowledgmentSource",
    "BraveResult",  # deprecated alias
    "BraveSearchClient",
    "SearchResult",
    "GateDecision",
    "GateVerdict",
    "JinaReaderClient",
    "SearchPayload",
    "SearchSource",
    "WebResultsCache",
    "WebSearchExecutor",
    "WebSearchGate",
    "format_sources_for_prompt",
    "format_sources_for_transcript",
]
