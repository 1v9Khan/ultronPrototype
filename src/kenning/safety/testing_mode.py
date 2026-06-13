"""Testing mode -- a SEPARATE, off-by-default mode for principled corpus testing.

It mimics the *disabled-functionality* conditions of gaming + anticheat mode --
RAG retrieval, the cross-encoder reranker, web search, and desktop automation
are all OFF -- so corpus outputs reflect exactly what the user gets while gaming.
The ONE difference: testing mode does NOT move the LLM/TTS to CPU and does NOT
engage real gaming mode, so the GPU stays available for fast generation.

Critically, this is its OWN flag: enabling it never triggers the gaming-mode
device swaps and never alters what ``gaming_mode``/``anticheat`` engage actually
do. The gating sites (``llm/inference._retrieve_rag_snippets``, the orchestrator
web-search gate, ``safety.anticheat.anticheat_active``) simply ALSO honor this
flag. Defaults OFF (runtime + config) so a normal restart is never accidentally
in testing mode.

Output-quality note: with the SAME gaming 3B model + the SAME gates, GPU and CPU
produce the same text (greedy is deterministic; sampled is statistically
identical), so GPU testing is representative of the CPU gaming runtime.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_runtime_active = False


def set_testing_mode_active(active: bool) -> None:
    """Flip the runtime testing-mode flag (used by the corpus test harness)."""
    global _runtime_active
    with _lock:
        _runtime_active = bool(active)


def is_testing_mode_active() -> bool:
    """True iff testing mode is on -- via the runtime flag OR the config
    ``testing_mode.enabled`` pin. Fail-open to False (a config error must never
    silently leave heavy subsystems on during a real gaming session)."""
    if _runtime_active:
        return True
    try:
        from kenning.config import get_config

        return bool(getattr(
            getattr(get_config(), "testing_mode", None), "enabled", False))
    except Exception:  # noqa: BLE001
        return False
