"""Rule-based addressing classifier.

A small set of regex / keyword rules that fire when the signal is strong
enough to short-circuit the zero-shot fallback. Goal: handle 70-90 % of
WARM-mode utterances at near-zero latency, so the slower zero-shot model
only touches genuinely ambiguous speech.

Each rule returns a (decision, confidence, reason) triple. The dispatcher
takes the highest-confidence rule per utterance. If no rule clears the
0.8 confidence bar the dispatcher falls through to zero-shot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class AddressingDecision(str, Enum):
    ADDRESSED = "ADDRESSED"
    NOT_ADDRESSED = "NOT_ADDRESSED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class RuleHit:
    decision: AddressingDecision
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# Strong YES signals: utterances that look like direct address to Kenning.
# Word boundaries (\b) keep "what" from matching "somewhat".
# ---------------------------------------------------------------------------

# Factual / informational question stems -- almost always a query the
# assistant should handle. We bias these high because mistakenly routing a
# "what time is it?" through the zero-shot fallback adds ~90 ms for no gain.
_FACTUAL_QUESTION_STEMS = re.compile(
    r"^\s*(?:"
    r"what(?:'s|s|\s+is|\s+are|\s+was|\s+were)?"
    r"|who(?:'s|s|\s+is|\s+are|\s+was)?"
    r"|when(?:'s|s|\s+is|\s+was)?"
    r"|where(?:'s|s|\s+is|\s+are)?"
    r"|why(?:'s|s|\s+is)?"
    r"|how(?:'s|s|\s+is|\s+do(?:es)?|\s+did|\s+much|\s+many|\s+long|\s+far|\s+old)?"
    r"|which"
    r"|is\s+(?:there|that|this)"
    r")\b",
    re.IGNORECASE,
)

# Second-person-target questions ("did you...", "are you..."). These are
# genuinely ambiguous between "directed at Kenning" and "directed at a human"
# (roommate, partner, child). We deliberately keep them below the 0.8
# threshold so the zero-shot model gets the final call using context.
_SECOND_PERSON_QUESTIONS = re.compile(
    r"^\s*(?:"
    r"do\s+you|did\s+you|don't\s+you|didn't\s+you"
    r"|are\s+you|aren't\s+you"
    r"|have\s+you|haven't\s+you"
    r"|were\s+you|weren't\s+you"
    r"|will\s+you|won't\s+you"
    r"|can\s+you|can't\s+you"
    r"|could\s+you"
    r"|would\s+you"
    r"|should\s+you"
    r"|how\s+(?:can|could|would|should)\s+you"
    r")\b",
    re.IGNORECASE,
)

# Imperative second-person commands. Conservative list -- we match the verb at
# the START of the utterance to avoid catching things like "I'll play later".
_IMPERATIVE_VERBS = re.compile(
    r"^\s*(?:"
    r"tell\s+me|show\s+me|find|search|look\s+up|look\s+for|google|"
    r"play|pause|stop|skip|resume|"
    r"turn\s+(?:on|off|up|down)|set|"
    r"open|close|launch|run|start|kill|"
    r"call|text|message|send|"
    r"remind\s+me|set\s+a\s+(?:reminder|timer|alarm)|"
    r"add\s+to|put\s+on|"
    r"explain|describe|summarize|translate|"
    r"draft|write|compose|"
    r"give\s+me"
    r")\b",
    re.IGNORECASE,
)

# Direct address by name. "Kenning, ..." in the second-person sense.
# We detect "kenning" at the START followed by a comma / vocative pause.
# Mid-sentence "kenning" (third-person mention) is handled by NO rules below.
_DIRECT_ADDRESS = re.compile(
    r"^\s*(?:hey\s+|okay\s+|alright\s+)?kenning\b[\s,.\-:]+",
    re.IGNORECASE,
)

# Short continuation answers most likely directed at us when said in a
# follow-up window after Kenning asked / proposed something. Standalone
# YES/NO are easy to misclassify so we keep this pool tight.
_CONTINUATION_TOKENS = {
    "yes", "yeah", "yep", "yup", "sure", "correct", "right",
    "no", "nope", "nah",
    "okay", "ok",
    "go", "go ahead", "do it", "do that", "proceed",
    "stop", "cancel",
    "more", "continue", "keep going",
    "next", "the next one", "the first one", "the second one", "the third one",
    "thanks", "thank you",
}


# ---------------------------------------------------------------------------
# Strong NO signals: utterances that clearly aren't directed at Kenning.
# ---------------------------------------------------------------------------

# Third-person mention of Kenning ("Kenning said X", "kenning's response was..."),
# i.e. the speaker is talking ABOUT Kenning, not TO Kenning. Anchored
# mid-sentence to avoid catching the YES "kenning, what's..." case which is
# matched by _DIRECT_ADDRESS first.
_THIRD_PERSON_MENTION = re.compile(
    r"\b(?:ultron|kenning)\s+(?:just|already|previously|earlier|said|thinks|told|wrote|claimed|reported|mentioned|is|was|keeps|seems|lagged|crashed|broke)\b",
    re.IGNORECASE,
)

# 2026-05-11 false-positive guard: tight patterns that catch the
# common "I'm narrating Kenning's behaviour to a third party in the
# room" cases observed in real session logs. These were sliding
# through the rule layer at exactly 0.75 zero-shot confidence and
# getting wrongly routed to ADDRESSED.
#
# The patterns are deliberately narrow -- legitimate Kenning commands
# like "tell him to send the email" or "ask her about the meeting"
# don't match, but "I'm talking to him", "got him to the point...",
# "you'll see", "watch this/him" all do.
_THIRD_PARTY_NARRATIVE = re.compile(
    r"(?:"
    # "I'm talking to him/her/it/them" / "I am talking to ..." -- meta
    # commentary about who you're addressing, not a command.
    r"\b(?:i'?m|i\s+am)\s+talking\s+(?:to|about|at|with)\s+(?:him|her|it|them)\b"
    # Causative "got/made/let him to <something>" -- past-tense
    # narrative about Kenning's state, not a directive to Kenning.
    r"|\b(?:got|made|let|forced|coaxed)\s+(?:him|her|it)\s+to\s+(?:the\s+)?\w+"
    # Meta-narration to a third party: "you'll see", "watch this/him",
    # "look at this/him" at the start of an utterance.
    r"|^\s*you'?ll\s+see\b"
    r"|^\s*watch\s+(?:this|him|it|what\s+he|what\s+it|what\s+she)\b"
    # Subject-pronoun status updates: "he's workable / broken / done /
    # ready / stuck / thinking / good / working" -- describing Kenning's
    # state to someone else.
    r"|^\s*(?:he|it|she)(?:'s|\s+is)\s+(?:workable|broken|done|ready|stuck|"
    r"thinking|working|good|fine|set|set\s+up|all\s+set)\b"
    # 2026-05-19 Issue 6 fix: continuation of third-party reference
    # like "I'm running him through his paces" / "running him" /
    # "showing him to the team". The first-person verb + him/her
    # signals the user is reporting about Kenning to a third party.
    r"|\b(?:i'?m|i\s+am|i)\s+(?:running|showing|telling|teaching|"
    r"explaining|introducing|demoing|testing|debugging)\s+(?:him|her|it)\b"
    r")",
    re.IGNORECASE,
)

# 2026-05-19 Issue 6 fix: third-party possessive questions.
# "Where's his sandbox?" / "What's her name?" / "When is their birthday?"
# are about-Kenning-to-a-third-party, not directed AT Kenning. Without
# this guard the factual-question-stem rule below fires (because the
# utterance starts with "where's") and ADDRESSED wins with conf 0.85
# -- live session 2026-05-19: "Where's his sandbar?" (Whisper-mangled
# from "where's his sandbox") was wrongly accepted as a follow-up to
# Kenning and triggered a SEARCH for a JPEG that didn't exist.
_THIRD_PARTY_POSSESSIVE_QUESTION = re.compile(
    r"^\s*(?:what|where|when|why|how|which)(?:'s|s|\s+is|\s+are|\s+was|\s+were)?\s+"
    r"(?:his|her|their|its)\s+\w+",
    re.IGNORECASE,
)

# Phone-call / interpersonal openers. If we hear these in WARM mode the user
# is almost certainly addressing another human.
_PHONE_OPENERS = re.compile(
    r"^\s*(?:"
    r"hello\?+|hey\s+(?:dude|man|bro|babe|honey|mom|dad|sis|guys)|"
    r"hi\s+(?:there\s+)?(?:dude|man|bro|babe|honey|mom|dad|sis|guys)|"
    r"yo\b|"
    r"what\s+up\s+(?:dude|man|bro|guys)|"
    r"it'?s\s+me\b"
    r")",
    re.IGNORECASE,
)

# Self-talk / exclamations. Short interjections almost never directed at us.
_INTERJECTIONS = {
    "oh god", "oh no", "oh shit", "oh fuck", "oh damn", "oh boy",
    "jesus", "jesus christ", "christ",
    "fuck", "shit", "damn", "crap",
    "lol", "haha", "lmao",
    "huh", "wat", "what the", "what the hell", "what the fuck",
    "wow", "whoa", "oof", "yikes",
    "ow", "ouch",
    "ugh", "argh", "hmm", "uhh", "umm", "uhhh", "err",
}


# ---------------------------------------------------------------------------
# 2026-06-18 FUSION feature extractor. Instead of one short-circuiting RuleHit,
# emit a graded feature vector the classifier fuses in log-odds space. Reuses
# every regex/set above. POSITION-AWARE: a LEADING assistant name is a strong
# ADDRESSED cue; a name mentioned mid-sentence / third-person is a NEGATIVE one.
# ---------------------------------------------------------------------------

# Leading wake word (the assistant's name) at the very start = a vocative address.
# Narrow, audited names only (not the broad ASR-mishear set) so chatter that
# merely contains a homophone doesn't fire. Captures the "Ultron, ..." the
# kenning-only _DIRECT_ADDRESS regex above was blind to (the live failure).
_LEADING_WAKE = re.compile(
    r"^\s*(?:hey\s+|okay\s+|ok\s+|alright\s+)?"
    r"(?:ultron|kenning|altron|ultraun)\b[\s,.\-:!?]*",
    re.IGNORECASE,
)
_NAME_ANYWHERE = re.compile(r"\b(?:ultron|kenning)\b", re.IGNORECASE)
# Human-directed openers: a subject pronoun (NOT 'you', which can address us) or
# an answer particle at the very start -- the empirically dominant NOT-ADDRESSED
# markers (Amazon ICASSP'20).
_SUBJ_PRONOUN_OPENER = re.compile(
    r"^\s*(?:i|we|he|she|they|it)(?:['’]\w+|\s+\w+)", re.IGNORECASE)
_PARTICLE_OPENER = re.compile(
    r"^\s*(?:yeah|yep|yup|nah|nope|okay|ok|oh|uh+|um+|hmm+|er+)\b", re.IGNORECASE)
_TRAILS_OFF = (",", " and", " but", " so", " or", " because")
_THIRD_PERSON_SUBJ_Q = re.compile(
    r"^(?:how|what|why|when|where|who)\s+(?:he|she|they|him|her|them)\b", re.IGNORECASE)


def features(
    utterance: str,
    seconds_since_response: float = 0.0,
) -> dict:
    """Graded addressee features in [0,1] (+ recency_s) for log-odds fusion.

    Every signal reuses a regex/set defined above. The assistant-name check is
    POSITION-AWARE and, when a leading name is present, the imperative/question
    features are evaluated on the REMAINDER ("Ultron, show me X" -> imperative)."""
    text = (utterance or "").strip()
    lowered = text.lower().rstrip(".!?,")
    wm = _LEADING_WAKE.match(text)
    third_person = bool(_THIRD_PERSON_MENTION.search(text))
    # "Ultron said ..." / "Kenning is broken" leads with the name but is ABOUT us,
    # not a vocative TO us -> suppress the leading-wake boost; count it negative.
    leading = bool(wm) and not third_person
    rest = text[wm.end():].strip() if wm else text
    rwords = rest.split()
    trails = rest.rstrip(".!?").rstrip().endswith(_TRAILS_OFF)
    fq = bool(_FACTUAL_QUESTION_STEMS.match(rest)) and len(rwords) >= 4 \
        and not trails and not _THIRD_PERSON_SUBJ_Q.match(rest)
    imperative = bool(_IMPERATIVE_VERBS.match(rest))
    return {
        "leading_wake": 1.0 if leading else 0.0,
        "embedded_or_3p_name": 1.0 if (
            (not leading and _NAME_ANYWHERE.search(text)) or third_person
        ) else 0.0,
        "initial_imperative": 1.0 if imperative else 0.0,
        "factual_question": 1.0 if fq else 0.0,
        "second_person_q": 1.0 if _SECOND_PERSON_QUESTIONS.match(rest) else 0.0,
        "continuation": 1.0 if lowered in _CONTINUATION_TOKENS else 0.0,
        "subj_pronoun_opener": 1.0 if (not wm and _SUBJ_PRONOUN_OPENER.match(text)) else 0.0,
        "particle_opener": 1.0 if (not wm and _PARTICLE_OPENER.match(text)) else 0.0,
        "phone_opener": 1.0 if _PHONE_OPENERS.search(text) else 0.0,
        "interjection": 1.0 if lowered in _INTERJECTIONS else 0.0,
        "third_party_narrative": 1.0 if _THIRD_PARTY_NARRATIVE.search(text) else 0.0,
        "possessive_q": 1.0 if _THIRD_PARTY_POSSESSIVE_QUESTION.match(text) else 0.0,
        "trails_off": 1.0 if (trails and not wm and not imperative) else 0.0,
        "recency_s": float(max(0.0, seconds_since_response)),
    }


def classify(
    utterance: str,
    seconds_since_response: float = 0.0,
) -> Optional[RuleHit]:
    """Run rule-based classification on ``utterance``.

    Returns the best matching :class:`RuleHit`, or ``None`` if no rule fired
    with confidence >= 0.5 (caller should fall through to zero-shot).

    ``seconds_since_response`` slightly biases the continuation pool: short
    answers are more likely to be a real continuation if the user spoke
    within ~5 s of Kenning's last response.
    """
    text = utterance.strip()
    if not text:
        return RuleHit(AddressingDecision.NOT_ADDRESSED, 0.95, "empty utterance")

    lowered = text.lower().rstrip(".!?,")

    # NO rules fire first: a direct phone opener wins over a question stem
    # ("hey mom, what time is it?" is not for Kenning).
    if _PHONE_OPENERS.search(text):
        return RuleHit(
            AddressingDecision.NOT_ADDRESSED, 0.92, "phone-call / interpersonal opener"
        )
    if _THIRD_PERSON_MENTION.search(text):
        return RuleHit(
            AddressingDecision.NOT_ADDRESSED, 0.85, "third-person mention of Kenning"
        )
    if _THIRD_PARTY_NARRATIVE.search(text):
        return RuleHit(
            AddressingDecision.NOT_ADDRESSED, 0.85, "narrating Kenning to a third party"
        )
    # Issue 6 fix: "where's his X?" / "what's her Y?" -- about Kenning,
    # not to Kenning. Must run BEFORE the factual-question-stem rule
    # below, which would otherwise grab "where's" and short-circuit
    # to ADDRESSED at 0.85.
    if _THIRD_PARTY_POSSESSIVE_QUESTION.match(text):
        return RuleHit(
            AddressingDecision.NOT_ADDRESSED, 0.85,
            "third-party possessive question (about Kenning, not to Kenning)",
        )
    if lowered in _INTERJECTIONS:
        return RuleHit(
            AddressingDecision.NOT_ADDRESSED, 0.85, "standalone interjection / self-talk"
        )

    # YES rules.
    if _DIRECT_ADDRESS.match(text):
        return RuleHit(
            AddressingDecision.ADDRESSED, 0.95, "direct address by name"
        )
    if _IMPERATIVE_VERBS.match(text):
        return RuleHit(
            AddressingDecision.ADDRESSED, 0.88, "imperative command stem"
        )
    if _FACTUAL_QUESTION_STEMS.match(text):
        # Factual question stems are a strong YES signal: "what time is it",
        # "who wrote X", "how does Y work" -- almost always directed at us.
        # 2026-06-11 live-dogfood fix: a mid-sentence STT FRAGMENT of the
        # user narrating to someone else ("How he was initially,") also
        # starts with a stem and was accepted at 0.85, sending a
        # contextless fragment into the LLM (which then recited stale
        # RAG memories as current facts). Guards: a real question has at
        # least four words, doesn't trail off with a comma/conjunction,
        # and isn't third-person narration about "him/her/them". Demote
        # fragments to UNCERTAIN so the zero-shot layer (and its
        # default-to-not-addressed posture) decides instead.
        stripped = text.strip()
        words = stripped.split()
        trails_off = stripped.rstrip(".!?").rstrip().endswith((
            ",", " and", " but", " so", " or", " because",
        ))
        third_person_subject = bool(re.match(
            r"^(?:how|what|why|when|where|who)\s+(?:he|she|they|him|her|them)\b",
            stripped, re.IGNORECASE,
        ))
        if len(words) < 4 or trails_off or third_person_subject:
            return RuleHit(
                AddressingDecision.UNCERTAIN, 0.55,
                "question-stem fragment (incomplete or third-person)",
            )
        return RuleHit(
            AddressingDecision.ADDRESSED, 0.85, "factual question stem"
        )
    if _SECOND_PERSON_QUESTIONS.match(text):
        # Second-person-target questions are ambiguous between addressing
        # Kenning and addressing a human in the room. Stay below the 0.8
        # short-circuit threshold so the zero-shot fallback can decide.
        return RuleHit(
            AddressingDecision.UNCERTAIN, 0.55, "second-person question (ambiguous)"
        )

    # Continuation tokens are bumped slightly higher when we just spoke.
    if lowered in _CONTINUATION_TOKENS:
        bias = 0.05 if seconds_since_response < 5.0 else 0.0
        return RuleHit(
            AddressingDecision.ADDRESSED, 0.78 + bias, "continuation token"
        )

    return None  # No confident rule -- caller should escalate to zero-shot.


def explain_rules() -> List[Tuple[str, str]]:
    """Lightweight introspection for the review tool. Returns a list of
    ``(rule_name, summary)`` for documentation."""
    return [
        ("phone_openers", "interpersonal openers like 'hey mom', 'yo', \"it's me\""),
        ("third_person_mention", "'Kenning said ...', talking about Kenning not to him"),
        ("third_party_narrative", "'I'm talking to him', 'got him to ...', 'you'll see' -- narrating Kenning to a third party"),
        ("interjections", "'oh god', 'lol', 'shit' -- self-talk"),
        ("direct_address", "starts with 'Kenning, ...' (vocative)"),
        ("imperative_verbs", "starts with command verb: play, find, turn on, ..."),
        ("question_stems", "starts with what/who/how/why/can you/..."),
        ("continuation_tokens", "single-word answers: yes, no, ok, do that, ..."),
    ]
