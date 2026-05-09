"""HybridTaskDecomposer — split a HYBRID_TASK utterance into ordered subtasks.

When the routing classifier returns ``HYBRID_TASK``, the orchestrator
asks Qwen to decompose the utterance into a structured plan: a list of
:class:`HybridSubtask` items, each labeled "coding" or "automation".

Phase 5 implementation:
  * ``decompose()`` calls into the local LLM (the one that's already
    loaded for the voice path) with a small prompt asking for JSON.
  * Output is validated; malformed responses fall back to a one-element
    plan that preserves the original utterance under "coding" type so
    the user gets a reasonable best-effort.
  * The decomposer is awaited inline; total token budget for the
    decomposition prompt is small (<=200 tokens output).

After OpenClaw integration the structure stays — only the downstream
dispatch differs (automation subtasks reach OpenClaw instead of stubs).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List

from ultron.config import get_config
from ultron.openclaw_routing.intents import HybridSubtask
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_routing.decomposer")


_DECOMPOSE_PROMPT = """\
The user said: "{utterance}"

Break this into ordered subtasks. Each subtask is either "coding" \
(generate code, edit a project) or "automation" (browser, file, shell, \
messaging, media). Output ONLY a JSON object with a single "subtasks" \
key. Each subtask has: order (int, 1-based), type ("coding" | \
"automation"), subtype (optional, e.g. "file_op" / "browser" / "shell"), \
description (short).

Example output:
{{"subtasks": [
  {{"order": 1, "type": "automation", "subtype": "file_op", \
"description": "Read the file at C:/path/to/data.csv"}},
  {{"order": 2, "type": "coding", \
"description": "Build a Python script that processes the data"}}
]}}

Output ONLY the JSON object, no commentary, no markdown.
"""


@dataclass
class DecompositionResult:
    """Outcome of a decomposition attempt."""
    subtasks: List[HybridSubtask]
    fallback_used: bool
    raw_response: str = ""


class HybridTaskDecomposer:
    """Calls the local LLM to split a HYBRID_TASK utterance into subtasks.

    Args:
        llm: an :class:`LLMEngine`-like object with a ``generate`` method
            that takes a prompt and returns a string. The orchestrator's
            existing LLM is the right one to pass in.
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def decompose(self, utterance: str) -> DecompositionResult:
        """Return a structured plan. Always returns at least one subtask
        — falls back to "coding: original utterance" on parse failure.

        4B plan Item 6: when ``llm.self_consistency.enabled`` is True
        (and the ``decomposer`` site isn't in
        ``self_consistency.disabled_sites``), the prompt is sampled N
        times and the JSON outputs are majority-voted. The majority
        winner is returned. Otherwise this is a single greedy call.
        """
        if not get_config().routing.hybrid_task_decomposition_enabled:
            return _coding_only_fallback(utterance, fallback_used=True)
        prompt = _DECOMPOSE_PROMPT.format(utterance=utterance.replace('"', "'"))

        raw = self._call_llm_with_optional_self_consistency(prompt)
        if not raw:
            return _coding_only_fallback(utterance, fallback_used=True, raw="")

        subtasks = _parse_subtasks(raw)
        if not subtasks:
            return _coding_only_fallback(utterance, fallback_used=True, raw=raw)
        return DecompositionResult(
            subtasks=subtasks, fallback_used=False, raw_response=raw,
        )

    def _call_llm_with_optional_self_consistency(self, prompt: str) -> str:
        """Single LLM call by default; N-sample majority vote when the
        ``decomposer`` self-consistency site is enabled.

        Failure of the LLM call returns an empty string — the caller
        treats that as "fallback to coding-only".
        """
        if self._llm is None:
            return ""
        from ultron.llm.self_consistency import (
            majority_vote_json,
            run_self_consistency,
            should_apply_self_consistency,
        )

        cfg = get_config()
        if not should_apply_self_consistency("decomposer", cfg):
            try:
                return self._llm.generate(prompt)
            except Exception as e:
                logger.warning("HybridTaskDecomposer LLM call failed: %s", e)
                return ""

        sc = cfg.llm.self_consistency

        def _sampler(temperature: float) -> str:
            try:
                # Pass temperature only when the LLM accepts it; the
                # in-process LLMEngine reads default_temperature from
                # config but generate() doesn't currently take an
                # override. Use the config-default sampling and rely
                # on the model's natural variability across calls.
                return self._llm.generate(prompt) or ""
            except Exception as e:
                logger.warning("HybridTaskDecomposer sampler call failed: %s", e)
                return ""

        result = run_self_consistency(
            _sampler, n=sc.n, temperature=sc.temperature,
            aggregator=lambda samples: _serialise_json_winner(majority_vote_json(samples)),
        )
        return result.answer or ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_subtasks(text: str) -> List[HybridSubtask]:
    """Best-effort JSON extract. Returns [] if anything goes wrong; caller
    converts that to a one-subtask fallback."""
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
                    candidates.append(text[i: j + 1])
                    break
    candidates.append(text)

    for c in candidates:
        try:
            obj = json.loads(c)
            if not isinstance(obj, dict):
                continue
            raw_subs = obj.get("subtasks")
            if not isinstance(raw_subs, list):
                continue
            out: List[HybridSubtask] = []
            for entry in raw_subs:
                if not isinstance(entry, dict):
                    continue
                order = int(entry.get("order", len(out) + 1))
                stype = str(entry.get("type", "coding")).lower()
                if stype not in ("coding", "automation"):
                    continue
                subtype = entry.get("subtype")
                if subtype is not None:
                    subtype = str(subtype).lower()
                desc = str(entry.get("description", "")).strip()
                if not desc:
                    continue
                out.append(HybridSubtask(
                    order=order, type=stype, subtype=subtype, description=desc,
                ))
            if out:
                # Sort by order to be safe.
                out.sort(key=lambda s: s.order)
                return out
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return []


def _serialise_json_winner(vote_result):
    """Adapter: ``majority_vote_json`` returns ``(parsed_dict, votes)``.
    ``run_self_consistency`` expects ``(answer, votes)`` where answer is
    a string. Re-serialise the winning dict so downstream
    :func:`_parse_subtasks` can re-parse it uniformly with the
    single-call code path."""
    parsed_winner, votes = vote_result
    if parsed_winner is None:
        return "", votes
    return json.dumps(parsed_winner), votes


def _coding_only_fallback(
    utterance: str, *, fallback_used: bool, raw: str = "",
) -> DecompositionResult:
    return DecompositionResult(
        subtasks=[HybridSubtask(
            order=1, type="coding",
            description=utterance,
        )],
        fallback_used=fallback_used,
        raw_response=raw,
    )


__all__ = ["HybridTaskDecomposer", "DecompositionResult"]
