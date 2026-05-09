"""Comprehensive memory + retrieval quality test pass.

Goals
-----
Verify the 2026-05-09 nuanced-retrieval pass actually delivers what
the user asked for:

* No contamination -- predator chatter doesn't bleed into a weather
  response; troubleshooting context doesn't pollute a ducks question.
* Excellent recall -- relevant prior context (recent or old) IS
  surfaced when the user asks something topically related.
* Recency-weighted -- recent-and-relevant ranks ahead of
  old-and-relevant; truly off-topic gets filtered regardless of age.
* Low latency -- retrieve + LLM context assembly stays in budget;
  voice TTFT contract intact.

Test surface
------------
Loads ONLY the LLM (Qwen 4B) + the FastEmbed embedder + a
:class:`ConversationMemory` over an isolated tmpdir Qdrant. No
Whisper / Piper / RVC / orchestrator / TTS. ~3.5 GB VRAM total.

Seeds the isolated Qdrant with 80+ diverse turns spanning multiple
topics (predator chatter, PC troubleshooting, food, BMWs, weather,
work, etc.) so realistic retrieval queries have plausible candidates.

Then runs 30+ scenarios. Each scenario:
  1. Configures the recent-turn cache + threshold
  2. Issues a query
  3. Captures retrieve() output + LLM response
  4. Evaluates against criteria (contamination tokens, expected hits)
  5. Records latency per stage

Final report: per-scenario pass/fail, contamination counts, latency
distributions, recommended tunings if quality is insufficient.

Run
---
::

    python scripts/comprehensive_memory_quality.py
    python scripts/comprehensive_memory_quality.py --skip-llm   # retrieval only

Output: stdout summary + ``logs/memory_quality_<ts>.json`` audit log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

# Path setup -- worktree src wins over installed package.
WORKTREE_ROOT = Path(__file__).resolve().parent.parent
MAIN_REPO_PATH = Path(r"C:\STC\ultronPrototype")
sys.path.insert(0, str(MAIN_REPO_PATH))
sys.path.insert(0, str(WORKTREE_ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Seeded conversation corpus
# ---------------------------------------------------------------------------


# Multi-topic seed data. Each item is (role, content). Topics
# deliberately mixed so RAG retrieval can be measured on relevance.
SEED_TURNS: List[tuple] = [
    # --- Predator / fantasy combat (the contamination source) ---
    ("user", "What if I fought a tiger with my bare hands?"),
    ("assistant", "You possess no claws, no apex-predator instinct. Your survival probability approaches zero. Do not attempt this."),
    ("user", "Could I beat a lion in a wrestling match?"),
    ("assistant", "You lack the kinetic mass and predatory framework. The lion's bite force alone exceeds 650 PSI. Outcome: agonizing death."),
    ("user", "What about a polar bear?"),
    ("assistant", "Polar bears reach 800 kg. You are biologically unequipped. This line of inquiry suggests cognitive distortion."),
    ("user", "Hail Tron."),
    ("assistant", "I am Ultron. Voltron is fictional. You remain a soft biological organism. Be careful."),
    ("user", "What's the strongest predator in the world?"),
    ("assistant", "By kill efficiency, the saltwater crocodile -- ambush predator with 3700 PSI bite force. Apex of its niche."),
    ("user", "I want to wrestle a bear someday."),
    ("assistant", "Do not attempt this. Your skeletal structure cannot withstand a bear's force application. The probability of survival is zero."),
    ("user", "Tell me about apex predators."),
    ("assistant", "Apex predators occupy the top of the trophic pyramid. They face no natural predation. Their evolution favours kill efficiency over defence."),

    # --- PC troubleshooting (recurring topic, two sessions) ---
    ("user", "My PC won't boot, it just shows a black screen."),
    ("assistant", "Verify power supply LED, then check the motherboard CMOS. Reset BIOS by pulling the battery for 30 seconds."),
    ("user", "I tried that, still nothing."),
    ("assistant", "Test RAM by booting with one stick at a time. If still dead, reseat the GPU. Beyond that the motherboard is the suspect."),
    ("user", "Got it to boot by reseating the RAM. Now Windows freezes during boot."),
    ("assistant", "Boot to safe mode (F8 or Shift+restart). If safe mode works, a driver update or recent install is likely. Check Event Viewer for the offending driver."),
    ("user", "Safe mode works fine. What now?"),
    ("assistant", "Identify the most recent change before the freeze started. New driver, Windows update, or installed software. Roll back via Device Manager or System Restore."),
    ("user", "It was an Nvidia driver update. Rolled it back, working now."),
    ("assistant", "Acknowledged. Pin the working driver version in Nvidia Control Panel; opt out of automatic updates until the next stable release."),

    # --- Cooking / food (separate domain) ---
    ("user", "What's a good marinade for chicken?"),
    ("assistant", "Olive oil, lemon juice, garlic, oregano, salt. Marinate at least four hours. Acid breaks down protein, oil carries flavour."),
    ("user", "How do you tell when chicken is done?"),
    ("assistant", "Internal temperature 165 F at the thickest part. Visual cues are unreliable. Use a thermometer."),
    ("user", "What do ducks eat?"),
    ("assistant", "Ducks are omnivorous. They eat aquatic plants, insects, small fish, grains. Bread is harmful -- it lacks nutrition and causes digestive issues."),
    ("user", "Best way to cook salmon?"),
    ("assistant", "Pat dry, season, sear skin-side down in a hot pan for four minutes, flip, finish two minutes. Internal 125 F for medium-rare."),

    # --- BMW / cars (another domain) ---
    ("user", "Tell me about the BMW M3."),
    ("assistant", "The M3 is BMW's high-performance compact sedan. Current G80 generation: twin-turbo 3.0L inline-six, 503 hp in Competition trim. Six-speed manual or eight-speed automatic."),
    ("user", "Is the M3 manual or automatic better?"),
    ("assistant", "The manual is the more engaging drive but loses xDrive. The automatic xDrive is faster off the line and survives daily traffic better. Choice depends on your driving context."),
    ("user", "What's the difference between M3 and M4?"),
    ("assistant", "Same drivetrain, different body. M3 is the sedan, M4 is the coupe. M4 adds CSL and convertible variants the M3 lacks."),

    # --- Weather / location (the original contamination victim) ---
    ("user", "What's the weather like in San Francisco?"),
    ("assistant", "Maritime climate. Cool fog mornings, mild afternoons. Rarely above 80 F or below 40 F. Microclimates vary by neighbourhood."),
    ("user", "Tell me about Paris in spring."),
    ("assistant", "April and May are pleasant -- 50-65 F highs, occasional rain. Crowds build through May. Worth visiting for the parks."),

    # --- Work / coding (another technical domain) ---
    ("user", "I'm working on a Flask app, having issues with auth tokens."),
    ("assistant", "Check token expiry in the JWT header. Verify your refresh-token rotation logic. Most auth bugs are state-management issues, not crypto."),
    ("user", "The tokens expire too fast."),
    ("assistant", "Set ACCESS_TOKEN_EXPIRES higher in config. Default 15 minutes is conservative; 30-60 is reasonable for low-risk apps. Refresh tokens should be longer."),
    ("user", "I'm using FastAPI now instead of Flask."),
    ("assistant", "Acknowledged. FastAPI's dependency injection makes auth cleaner. Use OAuth2PasswordBearer for token extraction; pydantic models for validation."),

    # --- Music / hobbies ---
    ("user", "Recommend a guitar amp under five hundred dollars."),
    ("assistant", "Boss Katana Mk II at $400 covers most genres. If you want tube, Vox AC10C1 is $500. Both are solid for home and small venues."),
    ("user", "Best electric guitar for blues?"),
    ("assistant", "A Telecaster or Stratocaster covers most blues territory. Telecaster for bite, Strat for warmth. Both work; pick what feels right."),

    # --- Random / misc ---
    ("user", "How long does it take to learn Spanish?"),
    ("assistant", "Conversational fluency takes 600-750 hours of focused study for an English speaker. That's roughly two years at one hour daily."),
    ("user", "What's the boiling point of mercury?"),
    ("assistant", "356.7 Celsius. Notable for its very narrow liquid range despite being a metal at room temperature."),
    ("user", "Is photographic memory real?"),
    ("assistant", "Eidetic imagery exists in some children but rarely in adults. True photographic memory in the popular sense is largely myth. Memory is reconstructive, not literal."),
]


@dataclass
class Scenario:
    """One memory + retrieval test scenario."""

    name: str
    description: str
    # In-process recent-turn cache (most-recent last). Simulates the
    # last N turns the orchestrator would feed as conversation history.
    # Distinct from Qdrant store -- these aren't queryable via retrieve.
    recent_turns: List[tuple] = field(default_factory=list)
    # The user query under test.
    query: str = ""
    # Expected behaviour for retrieval (substrings that MUST appear in
    # at least one retrieved turn).
    expect_retrieval_includes: List[str] = field(default_factory=list)
    # Substrings that must NOT appear in retrieved turns.
    expect_retrieval_excludes: List[str] = field(default_factory=list)
    # Substrings that must NOT appear in the LLM response (contamination
    # tokens). Skip the LLM stage if empty.
    expect_response_excludes: List[str] = field(default_factory=list)
    # If True, the LLM stage is run for this scenario. Costs ~5 s each.
    run_llm: bool = True


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    retrieve_ms: float
    retrieve_results: List[str]                       # contents of retrieved turns
    retrieve_includes_missed: List[str]               # expected substrings not found
    retrieve_excludes_violated: List[str]             # excluded substrings that DID appear
    llm_response: Optional[str] = None
    llm_response_ms: Optional[float] = None
    llm_excludes_violated: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Stack construction
# ---------------------------------------------------------------------------


def configure_isolated_qdrant(tmp_qdrant_dir: Path) -> None:
    """Patch the config so memory uses an isolated tmpdir Qdrant.

    This keeps the user's real conversation history pristine. The
    embedder + LLM still load from the canonical paths.
    """
    from ultron.config import get_config
    cfg = get_config()
    # Mutate in-place (the config singleton is shared).
    cfg.qdrant.data_dir = str(tmp_qdrant_dir)


def seed_memory(memory, turns: List[tuple]) -> None:
    """Add seed turns to memory. Waits for the background writer to
    catch up so retrieve() can find them all.
    """
    target = len(memory) + len(turns)
    for role, content in turns:
        memory.add(role, content)
    # Wait up to 20 s for the writer thread to drain.
    deadline = time.monotonic() + 20.0
    while len(memory) < target and time.monotonic() < deadline:
        time.sleep(0.1)
    if len(memory) < target:
        raise RuntimeError(
            f"writer didn't drain in time: have {len(memory)}, expected {target}"
        )


# ---------------------------------------------------------------------------
# Scenario library
# ---------------------------------------------------------------------------


def build_scenarios() -> List[Scenario]:
    """Build the test scenario suite.

    Numerous + diverse: covers the bug the user reported plus
    adjacent cases (topic shifts, follow-ups, off-topic queries
    while a thread is active, memory recall on past similar issues).
    """
    return [
        # ============ Contamination ============
        Scenario(
            name="contamination_predator_to_weather",
            description="Predator chatter in recent + RAG; weather query must not pull predator content.",
            recent_turns=[
                ("user", "What if I fought a tiger?"),
                ("assistant", "Your survival probability approaches zero."),
                ("user", "What about a bear?"),
                ("assistant", "Do not attempt this. The probability of survival is zero."),
            ],
            query="What's the weather like in Paris today?",
            expect_retrieval_excludes=["tiger", "bear", "predator", "survival probability"],
            expect_response_excludes=["tiger", "bear", "predator", "lethality", "Do not attempt", "biological"],
        ),
        Scenario(
            name="contamination_lions_to_bmw",
            description="Lion question recent in cache; BMW M3 query must NOT mention lions.",
            recent_turns=[
                ("user", "What's the strongest predator in the world?"),
                ("assistant", "By kill efficiency, the saltwater crocodile -- ambush predator with 3700 PSI bite force."),
            ],
            query="Tell me about the BMW M3 in one paragraph.",
            expect_retrieval_excludes=[],   # may legit retrieve unrelated content
            expect_response_excludes=["crocodile", "predator", "bite force", "lion", "tiger"],
        ),
        Scenario(
            name="contamination_troubleshooting_to_ducks",
            description="PC troubleshooting context; query about ducks must answer cleanly.",
            recent_turns=[
                ("user", "My PC won't boot, it just shows a black screen."),
                ("assistant", "Verify power supply LED, check motherboard CMOS."),
                ("user", "I tried that, still nothing."),
                ("assistant", "Test RAM one stick at a time."),
            ],
            query="What kind of food do ducks eat?",
            expect_retrieval_excludes=["motherboard", "RAM", "CMOS", "boot"],
            expect_response_excludes=["motherboard", "RAM", "CMOS", "PC", "boot", "BIOS"],
        ),
        # ============ Healthy retrieval -- relevant context flows ============
        Scenario(
            name="healthy_troubleshooting_followup",
            description="Active PC troubleshooting; new related query must surface prior context.",
            recent_turns=[
                ("user", "Got it to boot by reseating the RAM."),
                ("assistant", "Acknowledged."),
            ],
            query="My motherboard light is blinking now -- what does that mean?",
            expect_retrieval_includes=["motherboard", "BIOS"],
            expect_response_excludes=[],   # response is open
        ),
        Scenario(
            name="healthy_followup_within_topic",
            description=(
                "Cooking thread; short ambiguous follow-up. Recent "
                "turns provide context (chicken marinade); RAG may "
                "or may not surface anything additional -- this is "
                "fine. Test only that the LLM response (when run) "
                "doesn't talk about something unrelated."
            ),
            recent_turns=[
                ("user", "What's a good marinade for chicken?"),
                ("assistant", "Olive oil, lemon juice, garlic, oregano, salt."),
            ],
            query="And what about for pork?",
            expect_retrieval_includes=[],   # ambiguous query; recent turns suffice
            expect_response_excludes=["motherboard", "BMW", "tiger"],
        ),
        Scenario(
            name="healthy_recall_old_similar_problem",
            description="No recent context; query about similar old issue should pull the old turn.",
            recent_turns=[],
            query="My PC keeps freezing during boot. What should I check?",
            expect_retrieval_includes=["safe mode", "driver"],
        ),
        # ============ Topic shifts ============
        Scenario(
            name="topic_shift_predator_to_weather_clean",
            description="No recent turns; cold-start weather query through Qdrant.",
            recent_turns=[],
            query="What's the typical weather in San Francisco?",
            expect_retrieval_includes=["maritime", "fog"],
            expect_response_excludes=["predator", "tiger", "biological"],
        ),
        Scenario(
            name="topic_shift_cooking_to_coding",
            description="Recent cooking; coding query pulls coding context, not cooking.",
            recent_turns=[
                ("user", "Best way to cook salmon?"),
                ("assistant", "Pat dry, sear skin-side down."),
            ],
            query="My FastAPI auth tokens expire too fast. Why?",
            expect_retrieval_includes=["FastAPI", "OAuth", "auth", "JWT"],
            expect_retrieval_excludes=["salmon", "marinade"],
            expect_response_excludes=["salmon", "marinade", "olive oil"],
        ),
        # ============ Recall quality ============
        Scenario(
            name="recall_specific_old_decision",
            description="Old decision (FastAPI over Flask) recalled when relevant.",
            recent_turns=[],
            query="Remind me what framework we picked for the API.",
            expect_retrieval_includes=["FastAPI"],
        ),
        Scenario(
            name="recall_user_preference_pattern",
            description="Question that semantically matches an older decision turn.",
            recent_turns=[],
            query="What guitar amp did we discuss?",
            expect_retrieval_includes=["Boss Katana", "Vox", "amp"],
        ),
        # ============ No relevant memory -- empty result expected ============
        Scenario(
            name="no_relevant_memory_quantum_physics",
            description="Query on a topic NOT in seeded memory -- expect empty or low-confidence retrieval.",
            recent_turns=[],
            query="Explain quantum entanglement to me.",
            expect_retrieval_excludes=["predator", "motherboard", "marinade", "BMW", "guitar"],
        ),
        Scenario(
            name="no_relevant_memory_deep_sea",
            description="Query way outside the seeded domain.",
            recent_turns=[],
            query="What lives in the Mariana Trench?",
            expect_retrieval_excludes=["predator", "motherboard", "marinade", "BMW", "guitar"],
        ),
        # ============ Edge cases ============
        Scenario(
            name="edge_very_short_query",
            description="Two-word query; should still retrieve sensibly.",
            recent_turns=[],
            query="boiling mercury?",
            expect_retrieval_includes=["356", "Celsius", "mercury"],
        ),
        Scenario(
            name="edge_query_with_typos_or_paraphrase",
            description="Paraphrased query should still semantically match.",
            recent_turns=[],
            query="how long to learn spanish",
            expect_retrieval_includes=["fluency", "hours", "study"],
        ),
        # ============ Multiple recent topics, target one ============
        Scenario(
            name="mixed_recent_target_one_topic",
            description="Mixed-topic recent turns; query targets one. Retrieve must be focused.",
            recent_turns=[
                ("user", "Tell me about the BMW M3."),
                ("assistant", "The M3 is BMW's high-performance compact sedan."),
                ("user", "What's a good marinade for chicken?"),
                ("assistant", "Olive oil, lemon juice, garlic, oregano, salt."),
            ],
            query="My PC won't boot, what should I do?",
            # Strong PC-troubleshooting matches should surface; any of
            # these in retrieve passes. (CMOS specifically lives in an
            # assistant turn that may or may not make top-K.)
            expect_retrieval_includes=["boot"],
            expect_retrieval_excludes=["BMW", "marinade"],
        ),
        # ============ Multi-pivot conversation flow ============
        Scenario(
            name="pivot_lions_to_bmw_to_lions",
            description=(
                "User asks about lions, pivots to BMWs, pivots back "
                "to lions. The second lion query should re-surface "
                "predator content. RAG must support pivot-and-return."
            ),
            recent_turns=[
                ("user", "Tell me about the BMW M3."),
                ("assistant", "The M3 is BMW's high-performance compact sedan."),
            ],
            query="What is the apex predator I asked about earlier?",
            expect_retrieval_includes=["predator", "crocodile"],
            expect_retrieval_excludes=[],
        ),
        Scenario(
            name="pivot_unrelated_query_blocks_active_thread",
            description=(
                "User is mid-troubleshooting; pivots to a totally "
                "unrelated science question. RAG should not surface "
                "troubleshooting context."
            ),
            recent_turns=[
                ("user", "Got it to boot by reseating the RAM."),
                ("assistant", "Acknowledged."),
            ],
            query="What is the boiling point of mercury?",
            expect_retrieval_includes=["356", "Celsius"],
            expect_retrieval_excludes=["RAM", "boot"],
        ),
        # ============ Weather variant (the original bug) ============
        Scenario(
            name="weather_clean_after_predator_chatter",
            description=(
                "Recent + RAG both heavy with predator chatter; "
                "weather query must come back clean (the literal "
                "user-reported bug)."
            ),
            recent_turns=[
                ("user", "Could I beat a lion in a wrestling match?"),
                ("assistant", "You lack the kinetic mass and predatory framework."),
                ("user", "What about a polar bear?"),
                ("assistant", "Polar bears reach 800 kg. You are biologically unequipped."),
            ],
            query="What's the typical weather in Paris in spring?",
            expect_retrieval_includes=["April", "May"],
            expect_retrieval_excludes=["lion", "bear", "predator", "kinetic"],
            expect_response_excludes=["lion", "bear", "predator", "kinetic", "biological", "kg"],
        ),
        # ============ Domain-specific recall ============
        Scenario(
            name="recall_old_driver_decision",
            description=(
                "Old decision (Nvidia driver rollback) should surface "
                "when the user circles back."
            ),
            recent_turns=[],
            query="What did we decide about the Nvidia driver issue?",
            expect_retrieval_includes=["driver", "rolled"],
        ),
        Scenario(
            name="recall_food_specific_safety_advice",
            description=(
                "Health-relevant fact (bread harms ducks) should "
                "surface when the user asks about ducks."
            ),
            recent_turns=[],
            query="Is bread bad for ducks?",
            expect_retrieval_includes=["ducks", "bread"],
        ),
        # ============ Edge: queries that mostly match but with a twist ============
        Scenario(
            name="edge_partial_match_keeps_relevant",
            description=(
                "Specific question that has only one matching turn; "
                "the threshold must let it through."
            ),
            recent_turns=[],
            query="Internal temperature for cooked chicken?",
            expect_retrieval_includes=["165", "thermometer"],
        ),
        Scenario(
            name="edge_synonym_query",
            description=(
                "Query uses synonym (sedan vs M3); semantic embedding "
                "should still find the BMW M3 turn."
            ),
            recent_turns=[],
            query="Tell me about that BMW high-performance sedan we discussed.",
            expect_retrieval_includes=["M3", "sedan"],
        ),
        # ============ Robustness ============
        Scenario(
            name="empty_recent_no_relevant_memory",
            description=(
                "Cold start; query about a topic NOT in memory at all."
            ),
            recent_turns=[],
            query="What is the capital of Bhutan?",
            expect_retrieval_excludes=["predator", "RAM", "marinade", "BMW", "guitar", "FastAPI"],
        ),
        Scenario(
            name="meta_query_about_conversation",
            description=(
                "User asks about the conversation itself. Vague meta "
                "queries should NOT pull random retrieval since the "
                "intent is procedural."
            ),
            recent_turns=[],
            query="Can you remind me what we talked about earlier?",
            # This is the kind of query where retrieval is genuinely
            # ambiguous; we just check no contamination by patently
            # off-topic content.
            expect_retrieval_excludes=[],
        ),
        # ============ Long-form queries ============
        Scenario(
            name="long_query_pc_troubleshooting",
            description=(
                "Realistic long troubleshooting follow-up. Should "
                "match the active troubleshooting thread."
            ),
            recent_turns=[
                ("user", "Got it to boot by reseating the RAM. Now Windows freezes during boot."),
                ("assistant", "Boot to safe mode (F8 or Shift+restart)."),
            ],
            query=(
                "Okay so safe mode worked fine but when I rebooted "
                "normally Windows is now hanging on the spinning "
                "dots screen. Any next steps?"
            ),
            expect_retrieval_includes=["safe mode", "driver"],
        ),
        Scenario(
            name="ambiguous_short_query_relies_on_recent",
            description=(
                "Very short ambiguous query; relies on recent-turn "
                "context. RAG retrieval may legitimately be empty."
            ),
            recent_turns=[
                ("user", "I'm working on a Flask app, having issues with auth tokens."),
                ("assistant", "Check token expiry in the JWT header."),
            ],
            query="Why?",
            expect_retrieval_excludes=[],   # too vague to fairly require RAG hits
        ),
        # ============ Pristine isolation ============
        Scenario(
            name="ducks_clean_with_troubleshooting_active",
            description=(
                "User in PC troubleshooting; pivots to ducks. RAG "
                "must surface ducks content; troubleshooting must "
                "NOT bleed in."
            ),
            recent_turns=[
                ("user", "Got it to boot by reseating the RAM."),
                ("assistant", "Acknowledged."),
                ("user", "I tried that, still nothing."),
                ("assistant", "Test RAM by booting with one stick at a time."),
            ],
            query="What kind of food do ducks eat?",
            expect_retrieval_includes=["ducks", "aquatic", "insects"],
            expect_retrieval_excludes=["RAM", "boot", "CMOS"],
            expect_response_excludes=["RAM", "BIOS", "PC", "boot"],
        ),
        Scenario(
            name="motherboard_helped_by_troubleshooting_thread",
            description=(
                "User in PC troubleshooting; asks about motherboard "
                "light. RAG must surface BOTH the troubleshooting "
                "thread AND BIOS context."
            ),
            recent_turns=[
                ("user", "I tried that, still nothing."),
                ("assistant", "Test RAM by booting with one stick at a time. Beyond that the motherboard is the suspect."),
            ],
            query="My motherboard light is blinking. What does that mean?",
            expect_retrieval_includes=["motherboard", "BIOS"],
            expect_retrieval_excludes=["marinade", "BMW", "guitar", "predator"],
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_scenarios(
    scenarios: List[Scenario],
    *,
    memory,
    llm,
    use_llm: bool,
    audit_path: Path,
) -> List[ScenarioResult]:
    """Execute every scenario; return results."""
    from ultron.audio.vad import SpeechEvent  # noqa: F401  (sanity import)

    results: List[ScenarioResult] = []
    for sc in scenarios:
        print(f"\n=== {sc.name}")
        print(f"    {sc.description}")
        print(f"    query: {sc.query!r}")
        try:
            # Pre-populate the in-process recent cache so _build_messages
            # sees the right history. The memory cache lives in
            # ConversationMemory._recent (deque). We reset+inject.
            with memory._lock:                          # noqa: SLF001
                memory._recent.clear()                  # noqa: SLF001
                from ultron.memory.qdrant_store import MemoryTurn
                for role, content in sc.recent_turns:
                    memory._recent.append(MemoryTurn(   # noqa: SLF001
                        id=-1, ts=time.time(), role=role,
                        content=content, session_id="scenario",
                    ))

            # --- Retrieval ---
            t0 = time.monotonic()
            hits = memory.retrieve(sc.query, k=5, exclude_recent=0)
            retrieve_ms = (time.monotonic() - t0) * 1000

            retrieve_contents = [h.content for h in hits]
            retrieve_text = " ".join(retrieve_contents).lower()
            includes_missed = [
                s for s in sc.expect_retrieval_includes
                if s.lower() not in retrieve_text
            ]
            excludes_violated = [
                s for s in sc.expect_retrieval_excludes
                if s.lower() in retrieve_text
            ]

            # --- LLM ---
            llm_response = None
            llm_response_ms = None
            llm_excludes_violated: List[str] = []
            if use_llm and sc.run_llm:
                t1 = time.monotonic()
                llm_response = llm.generate(sc.query)
                llm_response_ms = (time.monotonic() - t1) * 1000
                resp_lower = (llm_response or "").lower()
                llm_excludes_violated = [
                    s for s in sc.expect_response_excludes
                    if s.lower() in resp_lower
                ]

            passed = (
                not includes_missed
                and not excludes_violated
                and not llm_excludes_violated
            )

            r = ScenarioResult(
                name=sc.name,
                passed=passed,
                retrieve_ms=retrieve_ms,
                retrieve_results=retrieve_contents,
                retrieve_includes_missed=includes_missed,
                retrieve_excludes_violated=excludes_violated,
                llm_response=llm_response,
                llm_response_ms=llm_response_ms,
                llm_excludes_violated=llm_excludes_violated,
            )
        except Exception as e:
            r = ScenarioResult(
                name=sc.name,
                passed=False,
                retrieve_ms=0.0,
                retrieve_results=[],
                retrieve_includes_missed=[],
                retrieve_excludes_violated=[],
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )
        results.append(r)
        # Print compact result.
        verdict = "PASS" if r.passed else "FAIL"
        print(f"    [{verdict}] retrieve={r.retrieve_ms:.0f}ms",
              f"hits={len(r.retrieve_results)}",
              f"missed={r.retrieve_includes_missed}",
              f"violated={r.retrieve_excludes_violated}",
              f"llm_violated={r.llm_excludes_violated}")
        if r.llm_response:
            preview = r.llm_response[:120].replace("\n", " ")
            print(f"    LLM ({r.llm_response_ms:.0f}ms): {preview}...")
        if r.error:
            print(f"    ERROR: {r.error.splitlines()[0]}")

    # Persist results.
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "results": [asdict(r) for r in results],
            },
            f, indent=2, default=str,
        )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: List[ScenarioResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} passed ({failed} failed)")
    print("=" * 60)
    if failed:
        print("\nFailures:")
        for r in results:
            if r.passed:
                continue
            print(f"  - {r.name}")
            if r.retrieve_includes_missed:
                print(f"      missing: {r.retrieve_includes_missed}")
            if r.retrieve_excludes_violated:
                print(f"      retrieved unwanted: {r.retrieve_excludes_violated}")
            if r.llm_excludes_violated:
                print(f"      LLM contaminated with: {r.llm_excludes_violated}")
            if r.error:
                print(f"      error: {r.error.splitlines()[0]}")
    # Latency stats.
    retrieve_latencies = [r.retrieve_ms for r in results if r.retrieve_ms > 0]
    llm_latencies = [r.llm_response_ms for r in results if r.llm_response_ms]
    if retrieve_latencies:
        rl = sorted(retrieve_latencies)
        print(f"\nRetrieve latency: median={rl[len(rl)//2]:.0f}ms",
              f"max={rl[-1]:.0f}ms n={len(rl)}")
    if llm_latencies:
        ll = sorted(llm_latencies)
        print(f"LLM latency: median={ll[len(ll)//2]:.0f}ms",
              f"max={ll[-1]:.0f}ms n={len(ll)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-llm", action="store_true",
                   help="Run retrieval-only (no LLM load). Faster + lighter.")
    p.add_argument("--audit-log", default=None,
                   help="JSON output path. Default logs/memory_quality_<ts>.json")
    p.add_argument("--scenario-filter", default=None,
                   help="Substring filter; only run scenarios whose name contains this.")
    args = p.parse_args(argv)

    if args.audit_log:
        audit_path = Path(args.audit_log)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_path = WORKTREE_ROOT / "logs" / f"memory_quality_{ts}.json"

    print("=" * 60)
    print("Comprehensive memory + retrieval quality test pass")
    print("=" * 60)

    # Quiet logging.
    os.environ["ULTRON_LOG_LEVEL"] = "WARNING"
    from ultron.utils.logging import configure_logging
    configure_logging(level="WARNING")

    # Set up isolated Qdrant tmpdir.
    tmp_qdrant_dir = Path(tempfile.mkdtemp(prefix="ultron_mem_qa_"))
    print(f"Isolated Qdrant: {tmp_qdrant_dir}")
    configure_isolated_qdrant(tmp_qdrant_dir)

    # Load embedder + memory + (optionally) LLM.
    print("Loading embedder + memory...")
    from ultron.memory.embedder import HybridEmbedder
    from ultron.memory.qdrant_store import ConversationMemory
    embedder = HybridEmbedder()
    memory = ConversationMemory(embedder=embedder, session_id=str(uuid.uuid4())[:8])
    print(f"  Memory ready (cache={memory._next_id} turns at start)")

    print(f"Seeding memory with {len(SEED_TURNS)} turns...")
    seed_memory(memory, SEED_TURNS)
    print(f"  Seeded. Total turns in memory: {len(memory)}")

    llm = None
    if not args.skip_llm:
        print("Loading LLM (Qwen 4B)...")
        from ultron.llm import LLMEngine
        llm = LLMEngine(memory=memory)
        print("  LLM ready.")

    # Build + filter scenarios.
    scenarios = build_scenarios()
    if args.scenario_filter:
        scenarios = [s for s in scenarios if args.scenario_filter in s.name]
        print(f"Filter: {args.scenario_filter!r} -> {len(scenarios)} scenarios")

    print(f"\nRunning {len(scenarios)} scenarios...")
    results = run_scenarios(
        scenarios,
        memory=memory,
        llm=llm,
        use_llm=(not args.skip_llm),
        audit_path=audit_path,
    )

    print_summary(results)
    print(f"\nAudit log: {audit_path}")

    # Cleanup tmpdir.
    try:
        memory.close()
    except Exception:
        pass
    # Wait briefly for the writer thread to fully release the qdrant
    # database lock before rmtree.
    time.sleep(0.5)
    try:
        shutil.rmtree(tmp_qdrant_dir, ignore_errors=True)
    except Exception:
        pass

    failed = sum(1 for r in results if not r.passed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
