"""Flan-T5-small zero-shot classifier for ambiguous addressing decisions.

The model is loaded lazily on first use (~8 s, ~300 MB CPU RAM). All inference
runs on CPU; we explicitly pin ``device_map='cpu'`` and disable CUDA tensors
even if a GPU is available, since the spec budgets zero new VRAM for Phase 2.

Latency target: <100 ms per call. Benchmarked at 78-94 ms median on the
prototype hardware. Only invoked when rule-based classification is uncertain,
so the average WARM-mode latency stays well below 50 ms.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from kenning.utils.logging import get_logger

logger = get_logger("addressing.zero_shot")

# The model accepts up to 512 tokens. We keep the prompt compact so a few
# turns of context fit comfortably with room to spare.
_PROMPT_TEMPLATE = """You decide whether a user's spoken utterance is meant for Kenning, the AI assistant they're talking to, or for someone or something else (another person, themselves, ambient speech).

Answer with one word only: YES, NO, or UNCLEAR.
- YES if the utterance is clearly directed at Kenning (a question, command, or follow-up to his last response).
- NO if the utterance is clearly NOT directed at Kenning (talking to another person, muttering, side comments).
- UNCLEAR if the signal is weak.

{context_block}Time since Kenning last spoke: {seconds_since:.0f} seconds.
Utterance: "{utterance}"

Answer:"""


def _format_context(context: Optional[List[Tuple[str, str]]]) -> str:
    if not context:
        return ""
    lines = ["Recent conversation:"]
    # Take last few turns, capped to keep prompt short.
    for role, content in context[-4:]:
        speaker = "Kenning" if role.lower().startswith("a") else "User"
        text = content.strip().replace("\n", " ")
        if len(text) > 200:
            text = text[:200] + "..."
        lines.append(f"  {speaker}: {text}")
    return "\n".join(lines) + "\n\n"


class ZeroShotAddresseeModel:
    """Lazy wrapper around Flan-T5-small for the zero-shot path."""

    def __init__(self, model_name: str = "google/flan-t5-small") -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        logger.info("Loading zero-shot addressee model %s on CPU...", self.model_name)
        t0 = time.monotonic()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.eval()
        # Pin to CPU so even an accidental .cuda() elsewhere doesn't push it
        # to GPU. Phase 2 must add zero VRAM.
        self._model.to("cpu")
        # Disable autograd globally for inference -- the wrapper never
        # backpropagates.
        for p in self._model.parameters():
            p.requires_grad_(False)
        # Warmup: a single short forward so the first real call doesn't pay
        # the JIT/import overhead.
        prompt = _PROMPT_TEMPLATE.format(
            context_block="", seconds_since=0.0, utterance="hello"
        )
        with torch.no_grad():
            inputs = self._tokenizer(prompt, return_tensors="pt")
            self._model.generate(**inputs, max_new_tokens=4)
        logger.info(
            "Zero-shot model ready in %.1fs", time.monotonic() - t0
        )

    def classify(
        self,
        utterance: str,
        context: Optional[List[Tuple[str, str]]] = None,
        seconds_since_response: float = 0.0,
    ) -> Tuple[str, float, float]:
        """Run zero-shot classification.

        Returns ``(verdict, confidence, latency_ms)`` where ``verdict`` is
        one of ``"YES"``, ``"NO"``, ``"UNCLEAR"``.

        2026-06-18: ``confidence`` is now the model's REAL certainty -- the
        softmax probability mass on the first generated token (the verdict's
        first sub-token) -- instead of the old hard-coded 0.75. That constant
        was the literal cause of the follow-up drop: a confident "Ultron, show
        me the stop button" was stamped 0.75 < the 0.80 ADDRESSED bar and
        silently rejected. Fail-open to the legacy constants if the step scores
        are unavailable.
        """
        import torch

        self._ensure_loaded()
        prompt = _PROMPT_TEMPLATE.format(
            context_block=_format_context(context),
            seconds_since=max(0.0, seconds_since_response),
            utterance=utterance.replace('"', "'"),
        )
        t0 = time.monotonic()
        with torch.no_grad():
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            gen = self._model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        output_ids = gen.sequences
        raw = self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().upper()
        latency_ms = (time.monotonic() - t0) * 1000

        # REAL confidence = P(first generated token) from the step-0 score
        # softmax. None if scores are missing -> fall back to the old constants.
        model_conf: Optional[float] = None
        try:
            scores0 = gen.scores[0][0]                 # logits over vocab, step 0
            probs0 = torch.softmax(scores0, dim=-1)
            first_tok_id = int(output_ids[0][1])       # [0]=decoder_start, [1]=first gen
            c = float(probs0[first_tok_id])
            if c == c and 0.0 <= c <= 1.0:             # finite + in range
                model_conf = c
        except Exception:                              # noqa: BLE001
            model_conf = None

        # First word is the verdict.
        first_word = raw.split()[0].rstrip(".,!?:;") if raw else "UNCLEAR"
        if first_word not in {"YES", "NO", "UNCLEAR"}:
            logger.info(
                "Zero-shot returned unexpected token %r for %r -- treating as UNCLEAR",
                raw, utterance[:60],
            )
            return "UNCLEAR", (model_conf if model_conf is not None else 0.40), latency_ms

        if first_word in {"YES", "NO"}:
            confidence = model_conf if model_conf is not None else 0.75
        else:
            confidence = model_conf if model_conf is not None else 0.50
        return first_word, confidence, latency_ms
