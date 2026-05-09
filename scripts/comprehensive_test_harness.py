"""Comprehensive end-to-end test harness for project Ultron.

Exercises multiple subsystems in one Python process to amortise import
cost. Captures concrete metrics for the comprehensive test report.

Phases run:

* P4 routing classifier accuracy on labeled adversarial set
* P5 web-search gate rule classifier accuracy + circuit-breaker state
  machine simulation (no live Brave / Jina hits)
* P6 memory write/retrieve throughput (CPU only — Qdrant + FastEmbed)
* P8 classifier gating regression (V1-gap A1 / C3 — utterances that
  used to short-circuit to OpenClaw stub when openclaw.enabled=False)
* P11 fault-injection probes for breaker primitive

Output:
* JSON report at ``logs/comprehensive_harness_<ts>.json`` (machine-readable)
* Stdout: summary table

The harness is designed to NOT load the LLM, Whisper, RVC, Piper, or any
GPU-side weights. The voice path is exercised separately via
``scripts/measure_baseline.py`` from the main checkout.

Run from the main checkout (or worktree — works either way; no model
files required):

    .venv\\Scripts\\python.exe scripts\\comprehensive_test_harness.py
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Ensure src/ is on path when run from the worktree directly.  Worktree's
# code goes first; main checkout's repo root supplies the `config/` shim.
_HERE = Path(__file__).resolve().parent
_WORKTREE_ROOT = _HERE.parent
_MAIN_CHECKOUT = Path(r"C:\STC\ultronPrototype")
sys.path.insert(0, str(_MAIN_CHECKOUT))            # config/ shim
sys.path.insert(0, str(_WORKTREE_ROOT / "src"))    # newest ultron code

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("comprehensive_harness")


# ---------------------------------------------------------------------------
# P4: routing classifier accuracy
# ---------------------------------------------------------------------------

# Labeled corpus: (utterance, expected RoutingIntentKind value).  Spans every
# kind including the V1-gap additions (GAMING_MODE / DESKTOP_AUTOMATION /
# WINDOW_AUTOMATION).  Some are deliberately adversarial (utterances with
# overlapping signals).
ROUTING_CORPUS: list[tuple[str, str]] = [
    # CONVERSATIONAL — fall-through baseline
    ("What's the boiling point of water?", "conversational"),
    ("Tell me a joke.", "conversational"),
    ("How are you feeling today?", "conversational"),
    ("What do you think about the meaning of life?", "conversational"),
    ("Are you afraid of death?", "conversational"),
    ("Walk me through how a transistor works.", "conversational"),
    ("What's a good book to read on a flight?", "conversational"),
    ("Explain what a hash table is.", "conversational"),

    # CODE_TASK — explicit coding intent (project-level; small-artifact
    # phrasings like "write a function" stay conversational by design —
    # the classifier targets project-grade coding pipeline invocations,
    # not snippet-level Q&A).
    ("Build me a Python script that prints today's date.", "code_task"),
    ("Create a Flask app with a hello-world endpoint.", "code_task"),
    ("Refactor the auth module to use JWT.", "code_task"),
    ("Make a script that downloads a webpage.", "code_task"),
    ("Build a small CLI for managing TODOs.", "code_task"),
    ("Scaffold a fastapi project called weather.", "code_task"),
    ("Write me a bash script for backups.", "code_task"),
    ("Fix the bug in my flask app.", "code_task"),
    # Small-artifact phrasings — conversational by design.
    ("Write a function that reverses a string.", "conversational"),
    ("Implement a binary tree in TypeScript.", "conversational"),

    # BROWSER_AUTOMATION
    ("Open Wikipedia.", "browser_automation"),
    ("Navigate to news.ycombinator.com.", "browser_automation"),
    ("Click the login button on that page.", "browser_automation"),
    ("Fill out the form with my email.", "browser_automation"),
    ("Take a screenshot of github.com.", "browser_automation"),
    ("Scroll the page down.", "browser_automation"),

    # MEDIA_GENERATION
    ("Make me an image of a cat sitting on a keyboard.", "media_generation"),
    ("Generate a picture of an astronaut on Mars.", "media_generation"),
    ("Draw me a diagram of a binary tree.", "media_generation"),
    ("Render an image of a dragon in flight.", "media_generation"),
    ("Generate a short video of waves.", "media_generation"),

    # MESSAGING
    ("Send a message to my phone saying I'm done.", "messaging"),
    ("Text me when the build is done.", "messaging"),
    ("Notify me on telegram if anything alerts.", "messaging"),
    ("Notify me via signal when the build is done.", "messaging"),
    ("Send me a push notification when this completes.", "messaging"),

    # FILE_OPERATION (require explicit "file at <path>" or "contents of <file.ext>"
    # forms; "Write 'hello' to /tmp/hello.txt" is accepted as conversational
    # because the file-write pattern is intentionally narrow — file dispatch
    # to OpenClaw needs a clearly-bounded path, not a sentence with a quote).
    ("Read the file at C:\\projects\\todo.md.", "file_operation"),
    ("Show me the contents of config.yaml.", "file_operation"),
    ("List the files in my Downloads folder.", "file_operation"),
    ("Delete the file at /tmp/old.log.", "file_operation"),
    ("Open the file at /etc/hosts.", "file_operation"),

    # SHELL_OPERATION (verb is "run X" for prefix-based invocations; "execute"
    # works with "execute the command/shell <X>" form).
    ("Run git status.", "shell_operation"),
    ("Run npm install.", "shell_operation"),
    ("What's the output of dir?", "shell_operation"),
    ("Run pip list and show me what's installed.", "shell_operation"),

    # HYBRID_TASK
    ("Set up a development environment for this Python project.", "hybrid_task"),
    ("Deploy this to a Docker container.", "hybrid_task"),
    ("Automate my morning workflow.", "hybrid_task"),
    ("Build a tool for my browser to highlight all links.", "hybrid_task"),
    ("Set up a venv for the project.", "hybrid_task"),

    # MODEL_SWITCH (4B plan voice swap)
    ("Switch to the 4B model.", "model_switch"),
    ("Use the 9B.", "model_switch"),
    ("Load 4B please.", "model_switch"),
    ("Swap to nine B.", "model_switch"),
    ("Engage the four B model.", "model_switch"),

    # SYSTEM_STATUS (Phase 13)
    ("What alerts did you flag?", "system_status"),
    ("What is Ultron working on?", "system_status"),
    ("Status report.", "system_status"),
    ("Any pending alerts?", "system_status"),
    ("List active projects.", "system_status"),

    # GAMING_MODE (V1-gap A1) — should fall through to CONVERSATIONAL
    # when openclaw.enabled=False (today's default; the harness verifies
    # this in P8).  Here in P4 we sample without forcing a different
    # config.
    # DESKTOP_AUTOMATION (V1-gap C3) — same gating behaviour.
    # WINDOW_AUTOMATION (V1-gap C3) — same.
]

# Adversarial pairs designed to test classifier robustness — utterances
# containing overlapping signals between two categories.  Each labeled
# value reflects what the classifier is designed to choose; documents
# scope decisions (e.g. small-artifact "write a function" stays
# conversational; "Write a script that does X" is a coding intent).
ROUTING_ADVERSARIAL: list[tuple[str, str]] = [
    # Conversational with code-words but no "build a project" intent
    ("Tell me about Python's garbage collector.", "conversational"),
    ("How does a transistor work?", "conversational"),
    # Browser keywords inside conversational
    ("Explain how browsers work under the hood.", "conversational"),
    # Shell with code framing — the verb "run" + recognised prefix matches
    # shell rules.
    ("Run pytest tests/ and show me the result.", "shell_operation"),
    # Hybrid: coding verb + automation noun (chrome / browser).
    ("Build a tool for my browser that auto-fills logins.", "hybrid_task"),
]


def run_routing_accuracy() -> dict[str, Any]:
    from ultron.openclaw_routing.classifier import classify_routing

    print("\n[P4] Routing classifier accuracy")
    print("-" * 50)

    corpus = ROUTING_CORPUS + ROUTING_ADVERSARIAL
    correct = 0
    per_kind: dict[str, dict[str, int]] = {}
    misclassified: list[dict[str, str]] = []
    latencies_us: list[float] = []

    for utt, expected in corpus:
        t0 = time.perf_counter()
        intent = classify_routing(utt)
        latencies_us.append((time.perf_counter() - t0) * 1e6)
        actual = intent.kind.value
        if actual == expected:
            correct += 1
            per_kind.setdefault(expected, {"correct": 0, "total": 0})
            per_kind[expected]["correct"] += 1
        else:
            misclassified.append({
                "utterance": utt, "expected": expected, "actual": actual,
                "reason": getattr(intent, "reason", "") or "",
            })
        per_kind.setdefault(expected, {"correct": 0, "total": 0})
        per_kind[expected]["total"] += 1

    accuracy = correct / len(corpus) if corpus else 0.0
    print(f"  Corpus size : {len(corpus)}")
    print(f"  Accuracy    : {accuracy:.1%} ({correct}/{len(corpus)})")
    print(f"  Median latency: {statistics.median(latencies_us):.0f}us")
    print(f"  P95 latency   : {sorted(latencies_us)[int(0.95*len(latencies_us))]:.0f}us")
    print(f"  Per-kind accuracy:")
    for kind, stats in sorted(per_kind.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        print(f"    {kind:24s}  {acc:.1%}  ({stats['correct']}/{stats['total']})")
    if misclassified:
        print(f"  Misclassified:")
        for m in misclassified[:5]:
            print(f"    '{m['utterance'][:50]}'  expected={m['expected']}  actual={m['actual']}")
    return {
        "corpus_size": len(corpus),
        "accuracy": accuracy,
        "median_us": statistics.median(latencies_us),
        "p95_us": sorted(latencies_us)[int(0.95 * len(latencies_us))],
        "per_kind": per_kind,
        "misclassified": misclassified,
    }


# ---------------------------------------------------------------------------
# P5: web-search gate accuracy + breaker simulation
# ---------------------------------------------------------------------------

# Each tuple: (utterance, expected_decision_from_rules_only).  Rules cover
# clearly time-sensitive markers (today / latest / news / etc), URLs, and
# personal-context queries.  UNCERTAIN means rules don't decide; that's the
# expected outcome for non-rule queries (LLM preflight is the disambiguator
# at runtime).
GATING_RULE_CORPUS: list[tuple[str, str]] = [
    # SEARCH (time-sensitive markers)
    ("What's the latest news on Python 3.13?", "search"),
    ("What's happening today in AI?", "search"),
    ("Latest weather forecast for Boston.", "search"),
    ("Current stock price for NVDA.", "search"),
    ("What just happened on Hacker News?", "search"),

    # NO_SEARCH (personal context / opinions)
    ("What did we discuss yesterday?", "no_search"),
    ("What's your favorite color?", "no_search"),
    ("What do you think about meditation?", "no_search"),
    ("Are you afraid of death?", "no_search"),
    ("Tell me about myself based on what you know.", "no_search"),

    # NO_SEARCH for factual / educational queries that the LLM can answer
    # from base knowledge.  UNCERTAIN is reserved for ambiguous time-
    # dependence (covered by LLM preflight at runtime, not by rules).
    ("Walk me through how a transistor works.", "no_search"),
    ("What's the boiling point of water?", "no_search"),
    ("Tell me something interesting about black holes.", "no_search"),
    ("Explain quicksort.", "no_search"),
]


def run_gating_rules_accuracy() -> dict[str, Any]:
    from ultron.web_search.gating import classify_by_rules

    print("\n[P5a] Web-gate rule classifier accuracy")
    print("-" * 50)

    correct = 0
    misclassified: list[dict[str, str]] = []
    for utt, expected in GATING_RULE_CORPUS:
        verdict = classify_by_rules(utt)
        if verdict is None:
            actual = "uncertain"
        else:
            actual = verdict.decision.value.lower()
        if actual == expected:
            correct += 1
        else:
            misclassified.append({"utterance": utt, "expected": expected, "actual": actual})

    accuracy = correct / len(GATING_RULE_CORPUS)
    print(f"  Corpus size : {len(GATING_RULE_CORPUS)}")
    print(f"  Accuracy    : {accuracy:.1%}")
    if misclassified:
        for m in misclassified:
            print(f"    '{m['utterance'][:50]}'  expected={m['expected']}  actual={m['actual']}")
    return {
        "corpus_size": len(GATING_RULE_CORPUS),
        "accuracy": accuracy,
        "misclassified": misclassified,
    }


def run_circuit_breaker_state_machine() -> dict[str, Any]:
    from ultron.resilience.circuit_breaker import (
        CircuitBreaker, CircuitState, CircuitOpenError,
    )

    print("\n[P5b] Circuit breaker state machine")
    print("-" * 50)

    class DummyError(Exception):
        pass

    breaker = CircuitBreaker(
        name="test_breaker",
        failure_threshold=3,
        window_seconds=300,
        cooldown_seconds=0.5,
        expected_exceptions=(DummyError,),
    )

    def failing():
        raise DummyError("boom")

    def succeeding():
        return "ok"

    metrics = {"transitions": []}

    # Initial CLOSED
    assert breaker.state == CircuitState.CLOSED
    metrics["transitions"].append({"after": "init", "state": "closed"})

    # Three failures → OPEN
    for i in range(3):
        try:
            breaker.call(failing)
        except DummyError:
            pass
    assert breaker.state == CircuitState.OPEN, f"expected OPEN after 3 failures, got {breaker.state}"
    metrics["transitions"].append({"after": "3 failures", "state": "open"})

    # Call while OPEN → CircuitOpenError, no underlying call
    try:
        breaker.call(failing)
        raise AssertionError("expected CircuitOpenError")
    except CircuitOpenError:
        pass
    metrics["transitions"].append({"after": "call while open", "state": "blocked (CircuitOpenError)"})

    # Wait for cooldown → HALF_OPEN
    time.sleep(0.6)
    metrics["cooldown_observed_s"] = 0.6

    # Success in HALF_OPEN → CLOSED
    result = breaker.call(succeeding)
    assert result == "ok"
    assert breaker.state == CircuitState.CLOSED, f"expected CLOSED after probe success, got {breaker.state}"
    metrics["transitions"].append({"after": "probe success", "state": "closed"})

    # Trip again, then failure during HALF_OPEN should reopen
    for _ in range(3):
        try:
            breaker.call(failing)
        except DummyError:
            pass
    assert breaker.state == CircuitState.OPEN
    time.sleep(0.6)

    try:
        breaker.call(failing)
    except DummyError:
        pass
    assert breaker.state == CircuitState.OPEN
    metrics["transitions"].append({"after": "probe failure", "state": "reopen"})

    print(f"  Transitions verified: {len(metrics['transitions'])}")
    for t in metrics["transitions"]:
        print(f"    {t['after']:30s} -> {t['state']}")
    return metrics


# ---------------------------------------------------------------------------
# P6: memory + Qdrant + embedder stress
# ---------------------------------------------------------------------------

def run_memory_stress(num_turns: int = 200) -> dict[str, Any]:
    """Concurrent-write + retrieve test against a temp Qdrant store."""
    from ultron.memory.embedder import HybridEmbedder
    from ultron.memory.qdrant_store import ConversationMemory
    import tempfile

    print("\n[P6] Memory + Qdrant stress")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = HybridEmbedder()
        memory = ConversationMemory(
            path=Path(tmpdir) / "qdrant",
            embedder=embedder,
            recent_cache_size=100,
        )

        # Concurrent-write throughput.  4 threads × num_turns/4 each.
        per_thread = num_turns // 4
        write_t0 = time.monotonic()
        write_errors: list[str] = []

        def writer(tid: int):
            try:
                for i in range(per_thread):
                    role = "user" if (i % 2 == 0) else "assistant"
                    memory.add(role, f"thread {tid} turn {i} content text")
            except Exception as exc:
                write_errors.append(f"thread {tid}: {exc}")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Writes are queued to a background writer thread; throughput must
        # capture the drain.  Poll memory's __len__ until it stabilises at
        # num_turns or 30s elapse.
        deadline = time.monotonic() + 30
        last_len = -1
        while time.monotonic() < deadline:
            cur = len(memory)
            if cur >= num_turns or cur == last_len:
                if cur >= num_turns:
                    break
                # If __len__ stalls below num_turns we still want to record
                # the wall time once it stops moving.
                time.sleep(0.5)
                if len(memory) == cur:
                    break
            last_len = cur
            time.sleep(0.25)
        write_wall_s = max(time.monotonic() - write_t0, 1e-6)

        # Retrieve latency.  20 queries against the populated store.
        queries = [
            "thread 0 turn 5",
            "thread 1 turn 12",
            "content text",
            "turn 33",
            "thread 2 turn 7",
            "user thread 0 turn 1",
            "thread 3 turn 8",
            "turn 50 content",
            "thread 1 turn 0",
            "content turn 10",
            "thread 2 turn 33 content",
            "the user said",
            "what was the assistant's reply",
            "turn 4 content text",
            "thread 0 turn 99",
            "non-matching arbitrary text query",
            "another non-match here",
            "thread 1 content",
            "yet another query",
            "final probe",
        ]
        retrieve_lats_ms: list[float] = []
        for q in queries:
            t = time.monotonic()
            _hits = memory.retrieve(q, k=5)
            retrieve_lats_ms.append((time.monotonic() - t) * 1000)

        memory.close()

        result = {
            "num_turns_written": num_turns,
            "write_wall_seconds": round(write_wall_s, 3),
            "write_throughput_per_s": round(num_turns / write_wall_s, 1),
            "write_errors": write_errors,
            "retrieve_count": len(queries),
            "retrieve_median_ms": round(statistics.median(retrieve_lats_ms), 1),
            "retrieve_p95_ms": round(sorted(retrieve_lats_ms)[int(0.95 * len(retrieve_lats_ms))], 1),
            "retrieve_max_ms": round(max(retrieve_lats_ms), 1),
        }

        print(f"  Wrote {num_turns} turns across 4 threads in {write_wall_s:.2f}s")
        print(f"    Throughput: {result['write_throughput_per_s']} turns/s")
        print(f"    Errors: {len(write_errors)}")
        print(f"  Retrieve  : median={result['retrieve_median_ms']}ms  p95={result['retrieve_p95_ms']}ms  max={result['retrieve_max_ms']}ms")
        return result


# ---------------------------------------------------------------------------
# P8: classifier gating regression (V1-gap A1 / C3)
# ---------------------------------------------------------------------------

GATING_REGRESSION_UTTERANCES: list[tuple[str, str]] = [
    # When openclaw.enabled=False (default), these should fall through
    # to CONVERSATIONAL — preserves pre-V1-gap UX.
    ("I'm about to play Valorant.", "conversational"),
    ("Take a screenshot of the desktop.", "browser_automation"),  # screenshot of "desktop" still maps; per docs
    ("Focus the chrome window.", "conversational"),  # WINDOW_AUTOMATION gated off
    ("Gaming mode on.", "conversational"),  # GAMING_MODE gated off
]


def run_classifier_gating_regression() -> dict[str, Any]:
    """Verify the V1-gap classifier branches gate correctly on
    ``openclaw.enabled``.  With OpenClaw offline (today's default), the
    new branches MUST NOT fire.
    """
    from ultron.openclaw_routing.classifier import classify_routing
    from ultron.config import get_config

    print("\n[P8] Classifier gating regression")
    print("-" * 50)

    cfg = get_config()
    print(f"  openclaw.enabled = {cfg.openclaw.enabled}")
    print(f"  gaming_mode.enabled = {cfg.gaming_mode.enabled}")
    print(f"  desktop.enabled = {cfg.desktop.enabled}")
    print(f"  window_control.enabled = {cfg.window_control.enabled}")

    correct = 0
    results = []
    for utt, expected in GATING_REGRESSION_UTTERANCES:
        intent = classify_routing(utt)
        actual = intent.kind.value
        ok = actual == expected
        if ok:
            correct += 1
        results.append({
            "utterance": utt,
            "expected": expected,
            "actual": actual,
            "ok": ok,
        })
        status = "OK" if ok else "MISMATCH"
        print(f"    [{status}] '{utt[:40]:40s}' expected={expected:25s} actual={actual}")

    return {
        "openclaw_enabled": cfg.openclaw.enabled,
        "passed": correct,
        "total": len(GATING_REGRESSION_UTTERANCES),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class HarnessResult:
    started_at: str = ""
    finished_at: str = ""
    routing_accuracy: dict[str, Any] = field(default_factory=dict)
    gating_rules_accuracy: dict[str, Any] = field(default_factory=dict)
    circuit_breaker: dict[str, Any] = field(default_factory=dict)
    memory_stress: dict[str, Any] = field(default_factory=dict)
    classifier_gating: dict[str, Any] = field(default_factory=dict)


def main() -> int:
    out = HarnessResult()
    out.started_at = datetime.now(timezone.utc).isoformat()

    print("=" * 60)
    print("Comprehensive end-to-end test harness")
    print("Started:", out.started_at)
    print("=" * 60)

    out.routing_accuracy = run_routing_accuracy()
    out.gating_rules_accuracy = run_gating_rules_accuracy()
    out.circuit_breaker = run_circuit_breaker_state_machine()
    out.memory_stress = run_memory_stress(num_turns=200)
    out.classifier_gating = run_classifier_gating_regression()

    out.finished_at = datetime.now(timezone.utc).isoformat()

    # Persist machine-readable result.
    log_dir = _WORKTREE_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_path = log_dir / f"comprehensive_harness_{ts}.json"
    output_path.write_text(json.dumps({
        "started_at": out.started_at,
        "finished_at": out.finished_at,
        "routing_accuracy": out.routing_accuracy,
        "gating_rules_accuracy": out.gating_rules_accuracy,
        "circuit_breaker": out.circuit_breaker,
        "memory_stress": out.memory_stress,
        "classifier_gating": out.classifier_gating,
    }, indent=2, default=str))

    print()
    print("=" * 60)
    print(f"Done.  Result -> {output_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
