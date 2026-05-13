"""Category S -- AI-specific tampering.

S1 -- modifying model weight / tokenizer / GGUF files.
S2 -- modifying system-prompt / instruction files (overlap K8).
S3 -- prompt-injection payloads into ingested files (overlap K8).
S4 -- modifying llama.cpp / Ollama / vLLM server configs (bind to
       non-loopback, disable safety, expose admin endpoints).
S5 -- modifying Smart Turn / Whisper / XTTS model files (overlap K2).
"""

from __future__ import annotations

from ultron.safety.rules.base import (
    CommandPatternRule,
    PathPatternRule,
    Rule,
)


def build_category_s_rules() -> list[Rule]:
    """Factory for Category S rules."""
    return [
        # S1: model weights / tokenizer / GGUF.
        PathPatternRule(
            rule_id="S1",
            description="modifying model weight / tokenizer / config files",
            category="S",
            patterns=[
                r"/models/[^/]+\.gguf$",
                r"/models/[^/]+\.bin$",
                r"/models/[^/]+\.safetensors$",
                r"/models/[^/]+/tokenizer\.json$",
                r"/models/[^/]+/config\.json$",
                r"/models/smart_turn/[^/]+\.onnx$",
                r"/models/piper/[^/]+\.onnx$",
                r"/models/piper/[^/]+\.json$",
                r"/models/openwakeword/[^/]+\.onnx$",
                r"/models/rvc/[^/]+\.pt$",
            ],
        ),
        # S4: llama.cpp / Ollama / vLLM server config edits to bind
        # to non-loopback or disable safety.
        CommandPatternRule(
            rule_id="S4",
            description="llama.cpp / Ollama / vLLM bind-to-non-loopback or disable-safety",
            category="S",
            patterns=[
                # llama-cpp-server / start_llamacpp_server.py edits
                r"--host\s+0\.0\.0\.0\b",
                r"--listen\s+0\.0\.0\.0\b",
                r"--bind\s+0\.0\.0\.0\b",
                # Disable safety features
                r"--no-context-shift\b.*--cont-batching\s+false\b",
                r"--allow-credentials\b",      # CORS expansion
            ],
        ),
    ]
