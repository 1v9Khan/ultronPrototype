"""Engine-agnostic intent recognition for Ultron.

Thin wrapper over the ``moonshine_voice`` package's
:class:`IntentRecognizer` that consumes transcript text from ANY STT
engine -- Moonshine, Parakeet, Whisper, or even typed input -- and
matches it against a registered set of canonical phrases via cosine
similarity on Gemma-300M embeddings.

The recognizer is decoupled from any specific STT pipeline. The
orchestrator calls :meth:`UltronIntentRecognizer.process_utterance`
after each transcribe call; a match above the configured threshold
short-circuits the LLM gating path and fires the registered handler.

Loaded lazily on first use; ~300 MB of CPU RAM in q4 quantization;
zero VRAM cost.
"""

from ultron.intent.recognizer import (
    IntentMatch,
    IntentRegistration,
    UltronIntentRecognizer,
    get_intent_recognizer,
    set_intent_recognizer,
)

__all__ = [
    "IntentMatch",
    "IntentRegistration",
    "UltronIntentRecognizer",
    "get_intent_recognizer",
    "set_intent_recognizer",
]
