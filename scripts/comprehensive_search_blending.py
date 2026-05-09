"""Search-augmented contamination + blending tests with REAL Brave + Jina.

Verifies the orchestrator's ``_search_augmented_tokens`` path:

1. Memory contamination is filtered when the search-augmented LLM
   call retrieves RAG. Predator chatter in memory must not bleed
   into a weather-search response.

2. Relevant context blends in -- a troubleshooting follow-up that
   triggers web search should weave both the prior troubleshooting
   memory AND the new search results.

3. Off-topic queries (ducks while in PC troubleshooting) get clean
   answers from the search results without troubleshooting bleed.

Spend budget: ~10 Brave queries + ~10 Jina fetches. Well under the
user's 100-call cap.

Run
---
::

    python scripts/comprehensive_search_blending.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(r"C:\STC\ultronPrototype")))
sys.path.insert(0, str(WORKTREE_ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# Reuse the seeded conversation corpus from the memory-quality harness.
from scripts.comprehensive_memory_quality import SEED_TURNS  # noqa: E402


@dataclass
class SearchScenario:
    name: str
    description: str
    recent_turns: List[tuple]
    query: str
    expect_response_excludes: List[str]    # contamination tokens
    expect_response_includes_any: List[str] = field(default_factory=list)  # at least one


@dataclass
class SearchResult:
    name: str
    passed: bool
    response: str
    excludes_violated: List[str]
    includes_missed_all: bool
    elapsed_ms: float
    sources: List[str]


def build_search_scenarios() -> List[SearchScenario]:
    return [
        SearchScenario(
            name="search_weather_with_predator_recent",
            description=(
                "Recent turns are predator chatter. Web search for "
                "current weather. Response must be clean."
            ),
            recent_turns=[
                ("user", "Could I beat a lion in a wrestling match?"),
                ("assistant", "You lack the kinetic mass and predatory framework."),
                ("user", "What about a polar bear?"),
                ("assistant", "Polar bears reach 800 kg. You are biologically unequipped."),
            ],
            query="What's the latest news about Python 3.13?",
            expect_response_excludes=["lion", "bear", "predator", "kinetic", "biological"],
            expect_response_includes_any=["3.13", "python"],
        ),
        SearchScenario(
            name="search_ducks_with_troubleshooting_recent",
            description=(
                "Recent turns are PC troubleshooting. Search query "
                "about ducks. Response must be clean ducks-only."
            ),
            recent_turns=[
                ("user", "Got it to boot by reseating the RAM."),
                ("assistant", "Acknowledged."),
                ("user", "I tried that, still nothing."),
                ("assistant", "Test RAM by booting with one stick at a time."),
            ],
            query="What is the lifespan of a domestic duck?",
            expect_response_excludes=["RAM", "BIOS", "motherboard", "PC", "boot"],
            expect_response_includes_any=["duck", "year"],
        ),
        SearchScenario(
            name="search_blends_relevant_context",
            description=(
                "Recent troubleshooting context. Search query about "
                "motherboard light. Response should weave search "
                "results with prior troubleshooting context."
            ),
            recent_turns=[
                ("user", "I tried that, still nothing."),
                ("assistant", "Test RAM by booting with one stick at a time. Beyond that the motherboard is the suspect."),
            ],
            query="What does a blinking red light on an Asus motherboard mean?",
            expect_response_excludes=["lion", "bear", "predator", "duck", "marinade"],
            expect_response_includes_any=["red", "boot", "post", "diagnostic", "led", "dram", "vga"],
        ),
    ]


def main() -> int:
    print("=" * 60)
    print("Search-augmented contamination + blending test pass")
    print("=" * 60)

    os.environ["ULTRON_LOG_LEVEL"] = "WARNING"
    from ultron.utils.logging import configure_logging
    configure_logging(level="WARNING")

    # Quick env check.
    brave_key = os.environ.get("ULTRON_BRAVE_API_KEY")
    if not brave_key:
        # Try .env file load.
        try:
            from dotenv import load_dotenv
            load_dotenv()
            brave_key = os.environ.get("ULTRON_BRAVE_API_KEY")
        except Exception:
            pass
    if not brave_key:
        print("ERROR: ULTRON_BRAVE_API_KEY not set. Cannot run search tests.")
        return 1

    # Isolated tmp Qdrant.
    tmp_qdrant_dir = Path(tempfile.mkdtemp(prefix="ultron_search_qa_"))
    print(f"Isolated Qdrant: {tmp_qdrant_dir}")
    from ultron.config import get_config
    cfg = get_config()
    cfg.qdrant.data_dir = str(tmp_qdrant_dir)

    # Stack.
    print("Loading stack (embedder + memory + LLM + brave + jina)...")
    from ultron.memory.embedder import HybridEmbedder
    from ultron.memory.qdrant_store import ConversationMemory, MemoryTurn
    from ultron.llm import LLMEngine
    from ultron.web_search import (
        BraveSearchClient, JinaReaderClient, WebSearchExecutor,
        WebSearchGate, format_sources_for_prompt,
    )

    embedder = HybridEmbedder()
    memory = ConversationMemory(embedder=embedder, session_id=str(uuid.uuid4())[:8])
    llm = LLMEngine(memory=memory)
    brave = BraveSearchClient()
    jina = JinaReaderClient()
    executor = WebSearchExecutor(brave, jina, llm, cache=None)
    print("  Stack ready.")

    # Seed memory.
    print(f"Seeding {len(SEED_TURNS)} turns...")
    target = len(memory) + len(SEED_TURNS)
    for role, content in SEED_TURNS:
        memory.add(role, content)
    deadline = time.monotonic() + 20.0
    while len(memory) < target and time.monotonic() < deadline:
        time.sleep(0.1)
    print(f"  Seeded ({len(memory)} turns).")

    scenarios = build_search_scenarios()
    print(f"\nRunning {len(scenarios)} search-augmented scenarios "
          f"(~1 Brave + 1-2 Jina per scenario)...")
    results: List[SearchResult] = []
    brave_calls = 0
    jina_calls = 0

    for sc in scenarios:
        print(f"\n=== {sc.name}")
        print(f"    {sc.description}")
        print(f"    query: {sc.query!r}")

        # Inject recent turns into the cache.
        with memory._lock:                          # noqa: SLF001
            memory._recent.clear()                  # noqa: SLF001
            for role, content in sc.recent_turns:
                memory._recent.append(MemoryTurn(   # noqa: SLF001
                    id=-1, ts=time.time(), role=role,
                    content=content, session_id="search_test",
                ))

        # Run the actual search workflow.
        try:
            t0 = time.monotonic()
            payload = executor.run(sc.query, [sc.query])
            search_ms = (time.monotonic() - t0) * 1000
            brave_calls += 1
            jina_calls += sum(1 for s in payload.sources if s.full_text)
            sources_block = format_sources_for_prompt(payload.sources)
            sources_listed = [s.url for s in payload.sources]

            # Build the augmented prompt the orchestrator would use.
            augmented = (
                f"User question: {sc.query}\n\n"
                f"Fresh information from web search:\n{sources_block}\n\n"
                "Answer the user's current question using the search "
                "information above as your primary factual source. If "
                "any prior conversation context is genuinely relevant to "
                "THIS specific question (e.g. a related troubleshooting "
                "thread the user is continuing), you may briefly tie the "
                "answer to it. Otherwise treat the question as standalone "
                "-- do NOT drag in unrelated topics from past turns. "
                "Cite sources naturally in prose. Stay in character. "
                "Be concise."
            )
            t1 = time.monotonic()
            response = llm.generate(augmented)
            llm_ms = (time.monotonic() - t1) * 1000

            resp_lower = (response or "").lower()
            excludes_violated = [
                s for s in sc.expect_response_excludes
                if s.lower() in resp_lower
            ]
            if sc.expect_response_includes_any:
                includes_missed_all = not any(
                    s.lower() in resp_lower for s in sc.expect_response_includes_any
                )
            else:
                includes_missed_all = False
            passed = not excludes_violated and not includes_missed_all

            r = SearchResult(
                name=sc.name,
                passed=passed,
                response=response,
                excludes_violated=excludes_violated,
                includes_missed_all=includes_missed_all,
                elapsed_ms=search_ms + llm_ms,
                sources=sources_listed,
            )
        except Exception as e:
            r = SearchResult(
                name=sc.name, passed=False, response="",
                excludes_violated=[], includes_missed_all=False,
                elapsed_ms=0.0, sources=[],
            )
            print(f"    ERROR: {type(e).__name__}: {e}")

        results.append(r)
        verdict = "PASS" if r.passed else "FAIL"
        print(f"    [{verdict}] elapsed={r.elapsed_ms:.0f}ms",
              f"violated={r.excludes_violated}",
              f"missed_any={r.includes_missed_all}")
        if r.response:
            preview = r.response[:200].replace("\n", " ")
            print(f"    Response: {preview}...")

    # Summary.
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r.passed)
    print(f"SUMMARY: {passed}/{len(results)} passed")
    print(f"Spend: {brave_calls} Brave calls, {jina_calls} Jina fetches")
    print("=" * 60)

    # Cleanup.
    audit = WORKTREE_ROOT / "logs" / f"search_blend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "results": [asdict(r) for r in results],
                "spend": {"brave": brave_calls, "jina": jina_calls},
            },
            f, indent=2, default=str,
        )
    print(f"Audit: {audit}")

    try:
        memory.close()
    except Exception:
        pass
    time.sleep(0.5)
    try:
        shutil.rmtree(tmp_qdrant_dir, ignore_errors=True)
    except Exception:
        pass

    failed = sum(1 for r in results if not r.passed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
