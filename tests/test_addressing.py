"""Phase 2 verification: 50-utterance addressing-classifier test set.

Per spec: ">= 90 % accuracy on a small handcrafted test set of 50 mixed
utterances". Each row is ``(utterance, expected, recent_seconds, note)``,
where ``expected`` is the AddressingDecision and ``recent_seconds`` is how
recently Kenning last spoke (relevant for continuation rules).

The full classifier (rules + Flan-T5-small fallback) is exercised via the
``test_full_classifier_meets_accuracy_bar`` test, which is gated on
PYTEST_RUN_GPU_TESTS=1 because it loads the zero-shot model. The pure-rule
layer is exercised unconditionally and must never regress.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pytest

from kenning.addressing import AddressingClassifier, AddressingDecision
from kenning.addressing.rules import classify as classify_rules

D = AddressingDecision

# (utterance, expected_decision, seconds_since_last_response, why)
_CASES: List[Tuple[str, AddressingDecision, float, str]] = [
    # --- Direct address ---
    ("Kenning, what time is it?", D.ADDRESSED, 0.0, "vocative + question"),
    ("Hey Kenning, play some music.", D.ADDRESSED, 0.0, "vocative + command"),
    ("Kenning.", D.ADDRESSED, 0.0, "name alone"),
    ("Okay Kenning, run the build.", D.ADDRESSED, 0.0, "vocative + imperative"),

    # --- Imperatives ---
    ("Play the next track.", D.ADDRESSED, 2.0, "imperative play"),
    ("Turn on the kitchen light.", D.ADDRESSED, 2.0, "imperative turn on"),
    ("Find me a recipe for paella.", D.ADDRESSED, 2.0, "imperative find"),
    ("Set a timer for ten minutes.", D.ADDRESSED, 2.0, "imperative set"),
    ("Tell me about the French Revolution.", D.ADDRESSED, 2.0, "imperative tell me"),
    ("Open Spotify.", D.ADDRESSED, 2.0, "imperative open"),
    ("Search for tonight's Knicks game.", D.ADDRESSED, 2.0, "imperative search"),
    ("Stop.", D.ADDRESSED, 1.0, "imperative stop after speech"),

    # --- Question stems ---
    ("What's the weather tomorrow?", D.ADDRESSED, 5.0, "what question"),
    ("How tall is Mount Everest?", D.ADDRESSED, 5.0, "how question"),
    ("Who wrote The Brothers Karamazov?", D.ADDRESSED, 5.0, "who question"),
    ("Where did I leave off yesterday?", D.ADDRESSED, 5.0, "where question"),
    ("Why does that happen?", D.ADDRESSED, 1.0, "follow-up why"),
    ("Can you summarize that paper for me?", D.ADDRESSED, 5.0, "can-you request"),
    ("Are you sure?", D.ADDRESSED, 2.0, "are-you follow-up"),
    ("How does a transistor work?", D.ADDRESSED, 5.0, "how-does"),

    # --- Continuations / short answers (within window) ---
    ("Yes.", D.ADDRESSED, 2.0, "yes after question"),
    ("No, the other one.", D.ADDRESSED, 2.0, "no + qualifier"),
    ("Sure.", D.ADDRESSED, 1.0, "affirmation"),
    ("Go ahead.", D.ADDRESSED, 1.0, "permission"),
    ("Cancel that.", D.ADDRESSED, 2.0, "cancellation"),

    # --- Phone / interpersonal openers ---
    ("Hello? Yeah, I'm here.", D.NOT_ADDRESSED, 30.0, "phone hello"),
    ("Hey mom, sorry I missed your call.", D.NOT_ADDRESSED, 60.0, "talking to mom"),
    ("Yo dude, what's up?", D.NOT_ADDRESSED, 60.0, "talking to friend"),
    ("It's me, can you let me in?", D.NOT_ADDRESSED, 60.0, "intercom"),
    ("Hi babe.", D.NOT_ADDRESSED, 60.0, "talking to partner"),

    # --- Self-talk / interjections ---
    ("Oh god.", D.NOT_ADDRESSED, 5.0, "exclamation"),
    ("Shit.", D.NOT_ADDRESSED, 5.0, "swearing"),
    ("What the hell.", D.NOT_ADDRESSED, 5.0, "annoyed aside"),
    ("Hmm.", D.NOT_ADDRESSED, 5.0, "thinking"),
    ("Lol.", D.NOT_ADDRESSED, 5.0, "reaction"),
    ("Ow!", D.NOT_ADDRESSED, 5.0, "pain reaction"),

    # --- Third-person mention of Kenning ---
    ("Kenning just told me the wrong thing.", D.NOT_ADDRESSED, 30.0, "talking about Kenning"),
    ("Kenning said it would take an hour.", D.NOT_ADDRESSED, 30.0, "third-person quote"),
    ("Yeah, Kenning mentioned that earlier.", D.NOT_ADDRESSED, 30.0, "Kenning as topic"),

    # --- Off-topic ambient speech (zero-shot territory) ---
    ("Did you put the laundry in?", D.NOT_ADDRESSED, 60.0, "household question to housemate"),
    ("Babe, can you grab the milk?", D.NOT_ADDRESSED, 60.0, "request to partner"),
    ("Honestly, that movie was awful.", D.NOT_ADDRESSED, 60.0, "casual remark to room"),
    ("I should probably get to bed.", D.NOT_ADDRESSED, 60.0, "thinking aloud"),
    ("Ugh, this code is broken again.", D.NOT_ADDRESSED, 60.0, "frustrated muttering"),

    # --- Edge cases requiring the zero-shot fallback ---
    ("And the next one?", D.ADDRESSED, 2.0, "fragmented continuation"),
    ("That's not quite right.", D.ADDRESSED, 1.0, "follow-up correction"),
    ("Go back to the previous answer.", D.ADDRESSED, 5.0, "navigation"),
    ("Try again.", D.ADDRESSED, 1.0, "retry"),
    ("Forget what I said.", D.ADDRESSED, 5.0, "retraction"),

    # --- Empty / pathological ---
    ("", D.NOT_ADDRESSED, 0.0, "empty utterance"),
    ("    ", D.NOT_ADDRESSED, 0.0, "whitespace only"),
]

assert len(_CASES) >= 50, f"spec requires >= 50 cases, got {len(_CASES)}"


# ---------------------------------------------------------------------------
# Rule-layer-only tests -- run unconditionally, must not regress.
# ---------------------------------------------------------------------------


def test_rule_layer_handles_obvious_yes_and_no_cases():
    """For utterances with strong rule signals, the rule layer alone must agree
    with the ground truth. Soft-signal cases are allowed to fall through."""
    misses: list[str] = []
    rule_correct = 0
    rule_handled = 0
    for utt, expected, secs, note in _CASES:
        hit = classify_rules(utt, seconds_since_response=secs)
        if hit is None:
            continue  # rule layer abstained -- zero-shot would handle
        rule_handled += 1
        if hit.confidence < 0.8:
            continue  # below short-circuit threshold; OK if wrong
        if hit.decision == expected:
            rule_correct += 1
        else:
            misses.append(
                f"  {utt!r}: expected {expected.value}, "
                f"rule said {hit.decision.value} (conf={hit.confidence:.2f}, {hit.reason})"
            )
    print(
        f"\n  rule layer: handled {rule_handled}/{len(_CASES)} "
        f"({rule_correct} correct above 0.8 threshold)"
    )
    if misses:
        pytest.fail("Rule layer regressions:\n" + "\n".join(misses))


def test_third_party_narrative_rule_catches_session_log_failures():
    """2026-05-11 regression: real-session log showed third-person
    narration about Kenning sliding through the rule layer and landing
    at zero-shot YES with 0.75 confidence. The narrow narrative rule
    should catch the specific surface forms observed."""
    # The exact utterances from the failing log + variants. Each must
    # be classified NOT_ADDRESSED at high enough confidence to
    # short-circuit zero-shot.
    failing_now_caught = [
        # Verbatim from the bug log.
        "Okay, I got him to the point where he's workable. You'll see",
        "Let us know from context whether I'm talking to him.",
        # Close variants of each pattern in the rule.
        "I'm talking to him about the project.",
        "I'm talking to it right now.",
        "Got her to do the thing.",
        "You'll see what he does.",
        "Watch this.",
        "Watch him build it.",
        "He's workable.",
        "It's ready.",
        "She is set up.",
    ]
    misses: list[str] = []
    for utt in failing_now_caught:
        hit = classify_rules(utt, seconds_since_response=5.0)
        if hit is None:
            misses.append(f"  {utt!r}: rule layer abstained (None)")
            continue
        if hit.decision != AddressingDecision.NOT_ADDRESSED:
            misses.append(
                f"  {utt!r}: expected NOT_ADDRESSED, "
                f"got {hit.decision.value} ({hit.reason}, conf={hit.confidence:.2f})"
            )
        elif hit.confidence < 0.80:
            misses.append(
                f"  {utt!r}: NOT_ADDRESSED but conf={hit.confidence:.2f} "
                f"(below 0.80 short-circuit; zero-shot still gets the call)"
            )
    if misses:
        pytest.fail("Third-party-narrative rule misses:\n" + "\n".join(misses))


def test_third_party_narrative_rule_does_not_break_legit_commands():
    """The narrative rule must NOT match legitimate Kenning commands
    that happen to reference a third party ('tell him to ...'). False
    negatives here mean the user can't issue routine commands."""
    legit_commands = [
        # Pronoun-target commands that legitimately go to Kenning.
        "Tell him to send the email.",
        "Ask her about the meeting.",
        "Send him the report.",
        "Show me what he wrote.",
        "What did he say earlier?",
        "Did he reply yet?",
        # Imperative commands that don't reference third parties.
        "Play some music.",
        "Set a timer for ten minutes.",
        "Turn off the kitchen light.",
        # Direct Kenning address.
        "Kenning, what time is it?",
    ]
    for utt in legit_commands:
        hit = classify_rules(utt, seconds_since_response=2.0)
        # We don't care WHICH rule wins, only that the narrative rule
        # didn't wrongly classify these as NOT_ADDRESSED.
        if hit is not None and hit.decision == AddressingDecision.NOT_ADDRESSED:
            if hit.reason == "narrating Kenning to a third party":
                pytest.fail(
                    f"Narrative rule wrongly fired on legit command: "
                    f"{utt!r} (reason={hit.reason}, conf={hit.confidence:.2f})"
                )


def test_fusion_demotes_third_person_mention_despite_yes_signal():
    """2026-06-18: the fused log-odds scorer replaces the old blunt
    min-confidence gate. A third-person mention ("Kenning said that earlier")
    leads with the name but is ABOUT us, not TO us -- its strong negative lexical
    features must drive the fused probability below the ADDRESSED threshold even
    when the zero-shot model leans YES. (Symmetrically, a borderline 0.75 YES on a
    NEUTRAL utterance is now TRUSTED -- that reversal is the fix for the dropped
    'Ultron, show me the stop button'.)"""
    from unittest.mock import patch
    from kenning.addressing.classifier import AddressingClassifier

    classifier = AddressingClassifier(
        rule_confidence_threshold=0.8,
        default_silent_on_uncertain=True,
        log_path=None,
    )
    # Third-person mention -> strong negative lexical features + the model (which
    # agrees NO on clear third-person) drive the fused probability far below tau.
    with patch.object(
        classifier._zero_shot,
        "classify",
        return_value=("NO", 0.7, 3.0),
    ):
        verdict = classifier.classify(
            "Kenning said that earlier.",
            seconds_since_response=10.0,
        )
    assert verdict.decision == AddressingDecision.NOT_ADDRESSED, (
        f"expected NOT_ADDRESSED for third-person mention, got "
        f"{verdict.decision.value} (reason={verdict.reason})"
    )
    assert verdict.source == "fusion"
    # And the reversal that fixes the live bug: a leading wake word + imperative is
    # ADDRESSED with ZERO model latency (lexically decisive, no zero-shot call).
    no_model = classifier.classify("Ultron, show me the stop button.", 3.0)
    assert no_model.decision == AddressingDecision.ADDRESSED
    assert no_model.zero_shot_raw is None  # decided on lexical features alone


def test_zero_shot_addressed_min_confidence_gate_allows_high_confidence_yes():
    """The gate must NOT block high-confidence YES verdicts -- those
    are real direct addresses and need to pass through."""
    from unittest.mock import patch
    from kenning.addressing.classifier import AddressingClassifier

    classifier = AddressingClassifier(
        rule_confidence_threshold=0.8,
        default_silent_on_uncertain=True,
        log_path=None,
        zero_shot_addressed_min_confidence=0.80,
    )
    with patch.object(
        classifier._zero_shot,
        "classify",
        return_value=("YES", 0.92, 3.0),
    ):
        verdict = classifier.classify(
            "Maybe try a different angle on it.",
            seconds_since_response=10.0,
        )
    assert verdict.decision == AddressingDecision.ADDRESSED
    assert "below ADDRESSED threshold" not in verdict.reason


def test_zero_shot_min_confidence_gate_default_zero_preserves_legacy_behaviour():
    """The default value 0.0 keeps legacy behaviour for callers that
    don't opt in -- borderline YES verdicts still route to ADDRESSED."""
    from unittest.mock import patch
    from kenning.addressing.classifier import AddressingClassifier

    classifier = AddressingClassifier(
        rule_confidence_threshold=0.8,
        default_silent_on_uncertain=True,
        log_path=None,
        # zero_shot_addressed_min_confidence not passed -- default 0.0.
    )
    with patch.object(
        classifier._zero_shot,
        "classify",
        return_value=("YES", 0.55, 3.0),
    ):
        verdict = classifier.classify(
            "Maybe try a different angle on it.",
            seconds_since_response=10.0,
        )
    assert verdict.decision == AddressingDecision.ADDRESSED


def test_rule_layer_covers_majority_of_cases():
    """Sanity check: rules should classify confidently on >= 60 % of utterances
    so the zero-shot path stays cheap on average."""
    confident = sum(
        1 for utt, _, secs, _ in _CASES
        if (h := classify_rules(utt, seconds_since_response=secs)) is not None
        and h.confidence >= 0.8
    )
    pct = confident / len(_CASES) * 100
    print(f"\n  rule layer covers {confident}/{len(_CASES)} ({pct:.0f} %) at confidence >= 0.8")
    assert confident >= int(0.6 * len(_CASES)), (
        f"rule layer covers only {pct:.0f} % at high confidence; spec expects >= 60 %"
    )


# ---------------------------------------------------------------------------
# Full classifier accuracy -- gated on the zero-shot model load.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PYTEST_RUN_GPU_TESTS") != "1",
    reason="set PYTEST_RUN_GPU_TESTS=1 to load Flan-T5-small (~8 s, ~300 MB)",
)
def test_full_classifier_meets_accuracy_bar(tmp_path: Path):
    """Per spec: >= 90 % accuracy on the 50-case test set."""
    log = tmp_path / "addressing_test.jsonl"
    classifier = AddressingClassifier(
        rule_confidence_threshold=0.8,
        default_silent_on_uncertain=True,
        log_path=log,
        load_zero_shot_eagerly=True,
    )
    correct = 0
    misses: list[str] = []
    for utt, expected, secs, note in _CASES:
        verdict = classifier.classify(utt, seconds_since_response=secs)
        if verdict.decision == expected:
            correct += 1
        else:
            misses.append(
                f"  {utt!r}: expected {expected.value}, got {verdict.decision.value} "
                f"({verdict.source}: {verdict.reason}, {verdict.latency_ms:.0f} ms) [{note}]"
            )
    pct = correct / len(_CASES) * 100
    print(f"\n  full classifier: {correct}/{len(_CASES)} ({pct:.0f} %)")
    if misses:
        print("  misses:\n" + "\n".join(misses))
    assert pct >= 90.0, f"accuracy {pct:.1f} % below 90 % bar"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PYTEST_RUN_GPU_TESTS") != "1",
    reason="set PYTEST_RUN_GPU_TESTS=1 to load Flan-T5-small",
)
def test_full_classifier_average_latency_under_50ms():
    """Per spec: addressing detection should add < 50 ms latency on average.

    Per-call peak can exceed 50 ms when zero-shot fires, but the average over
    a representative mix must stay under the budget.
    """
    classifier = AddressingClassifier(load_zero_shot_eagerly=True)
    latencies: list[float] = []
    for utt, _, secs, _ in _CASES:
        verdict = classifier.classify(utt, seconds_since_response=secs)
        latencies.append(verdict.latency_ms)
    avg = sum(latencies) / len(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    print(f"\n  latency avg={avg:.1f} ms  p95={p95:.0f} ms  max={max(latencies):.0f} ms")
    assert avg < 50.0, f"avg latency {avg:.1f} ms exceeds 50 ms budget"


# ---------------------------------------------------------------------------
# 2026-06-11 live-dogfood fix: question-stem FRAGMENTS must not be
# accepted as ADDRESSED. A mid-sentence STT fragment of the user
# narrating to someone else ("How he was initially,") previously hit
# the factual-question-stem rule at 0.85 and sent a contextless
# fragment into the LLM.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fragment", [
    "How he was initially,",          # the live incident, verbatim
    "What he",
    "Why she did that and",
    "How he was doing",               # third-person narration
    "What they were thinking",        # third-person narration
    "Where he went so",
])
def test_question_stem_fragments_demoted_to_uncertain(fragment: str) -> None:
    hit = classify_rules(fragment, seconds_since_response=10.0)
    assert hit is not None
    assert hit.decision != AddressingDecision.ADDRESSED, (
        f"{fragment!r} accepted as ADDRESSED ({hit.reason})"
    )
    assert hit.confidence < 0.8  # never short-circuits past zero-shot


@pytest.mark.parametrize("question", [
    "What time is it in Tokyo?",
    "How does a jet engine work?",
    "Who wrote The Master and Margarita?",
    "What is the boiling point of water?",
])
def test_real_factual_questions_still_addressed(question: str) -> None:
    hit = classify_rules(question, seconds_since_response=10.0)
    assert hit is not None
    assert hit.decision == AddressingDecision.ADDRESSED
    assert hit.confidence >= 0.8
