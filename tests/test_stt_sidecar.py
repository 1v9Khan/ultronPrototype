"""Pins the out-of-process STT sidecar (2026-07-26).

The sidecar exists because faster-whisper/CTranslate2 constructed with a
non-zero ``device_index`` corrupts memory IN-PROCESS on this box and kills
Ultron with a silent ``0xc0000409``. The child is masked onto one GPU with
``CUDA_VISIBLE_DEVICES`` so the target card is plain ``cuda:0`` there.

Two contracts are load-bearing and easy to lose:

1. **Placement** — the server must NEVER pass ``device_index`` to
   ``WhisperModel``; masking is the whole mechanism.
2. **Decode-quality parity** — moving STT to another process must not
   silently cost the Valorant domain biasing (agent names) or the
   whole-transcript non-speech blocklist. Both live in
   ``kenning.transcription._whisper_text`` and are shared with the
   in-process engine.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SERVER = _ROOT / "scripts" / "stt_server.py"


def _load_server(monkeypatch, **env):
    """Import scripts/stt_server.py fresh with a given environment.

    Module-level constants are read from the environment at import time, so
    each test gets its own module object.
    """
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location("_stt_server_under_test",
                                                  _SERVER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Seg:
    def __init__(self, text: str, no_speech_prob: float = 0.0) -> None:
        self.text = text
        self.no_speech_prob = no_speech_prob


class _FakeModel:
    """Records the kwargs the server passes to faster-whisper."""

    def __init__(self, segments) -> None:
        self._segments = segments
        self.calls: list[dict] = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return iter(self._segments), object()


@pytest.mark.skipif(not _SERVER.is_file(), reason="stt_server.py missing")
def test_never_passes_device_index(monkeypatch) -> None:
    """The masking mechanism -- a device_index here would reintroduce the crash."""
    import ast

    src = _SERVER.read_text(encoding="utf-8")
    assert "device_index" in src, "the deliberate omission must stay documented"
    # Parse rather than grep: the docstring and comments MUST mention
    # device_index (that is the documentation), only a real kwarg is a bug.
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", "")) == "WhisperModel"]
    assert calls, "no WhisperModel construction found in stt_server.py"
    for call in calls:
        names = [kw.arg for kw in call.keywords]
        assert "device_index" not in names, (
            "stt_server.py passes device_index to WhisperModel -- that is the "
            "exact in-process path that corrupts memory; the parent masks GPUs "
            "with CUDA_VISIBLE_DEVICES instead")


@pytest.mark.skipif(not _SERVER.is_file(), reason="stt_server.py missing")
def test_decode_quality_helpers_load(monkeypatch) -> None:
    """The shared rules must load by FILE PATH, without the package __init__
    chain -- a config error must not be able to take STT down."""
    mod = _load_server(monkeypatch)
    assert mod._TEXT_RULES is True, (
        "the sidecar fell back to no-domain-bias / no-blocklist")
    assert mod.build_initial_prompt("").startswith("Valorant team comms")
    assert mod.is_whisper_hallucination("Thank you.") is True
    assert mod.is_whisper_hallucination("push mid") is False


@pytest.mark.skipif(not _SERVER.is_file(), reason="stt_server.py missing")
def test_applies_domain_biasing(monkeypatch) -> None:
    mod = _load_server(monkeypatch, KENNING_STT_DOMAIN_BIAS="1")
    fake = _FakeModel([_Seg(" push mid ")])
    monkeypatch.setattr(mod, "_load", lambda: fake)

    out = mod._transcribe(np.zeros(16000, dtype=np.float32), "en")

    assert out == "push mid"
    kw = fake.calls[0]
    assert "Valorant team comms" in kw["initial_prompt"], (
        "agent-name biasing lost in the sidecar")
    # The rest of the decode settings must mirror the in-process engine.
    assert kw["temperature"] == 0.0
    assert kw["condition_on_previous_text"] is False
    assert kw["language"] == "en"


@pytest.mark.skipif(not _SERVER.is_file(), reason="stt_server.py missing")
def test_domain_bias_can_be_disabled(monkeypatch) -> None:
    mod = _load_server(monkeypatch, KENNING_STT_DOMAIN_BIAS="0")
    fake = _FakeModel([_Seg("hello")])
    monkeypatch.setattr(mod, "_load", lambda: fake)

    mod._transcribe(np.zeros(16000, dtype=np.float32), "en")

    assert "initial_prompt" not in fake.calls[0]


@pytest.mark.skipif(not _SERVER.is_file(), reason="stt_server.py missing")
def test_drops_whole_transcript_hallucination(monkeypatch) -> None:
    """A stock non-speech phrase must not reach the parent as a command."""
    mod = _load_server(monkeypatch)
    fake = _FakeModel([_Seg(" Thank you. ")])
    monkeypatch.setattr(mod, "_load", lambda: fake)

    assert mod._transcribe(np.zeros(16000, dtype=np.float32), "en") == ""


@pytest.mark.skipif(not _SERVER.is_file(), reason="stt_server.py missing")
def test_keeps_real_speech_that_merely_contains_a_stock_word(monkeypatch) -> None:
    """The blocklist is WHOLE-transcript only -- never substring."""
    mod = _load_server(monkeypatch)
    fake = _FakeModel([_Seg("thank you for the heal, Sage")])
    monkeypatch.setattr(mod, "_load", lambda: fake)

    out = mod._transcribe(np.zeros(16000, dtype=np.float32), "en")
    assert out == "thank you for the heal, Sage"


@pytest.mark.skipif(not _SERVER.is_file(), reason="stt_server.py missing")
def test_drops_high_no_speech_segments(monkeypatch) -> None:
    mod = _load_server(monkeypatch)
    fake = _FakeModel([_Seg("phantom", no_speech_prob=0.99), _Seg("push mid")])
    monkeypatch.setattr(mod, "_load", lambda: fake)

    assert mod._transcribe(np.zeros(16000, dtype=np.float32), "en") == "push mid"


# ---------------------------------------------------------------------------
# The shared module itself -- both STT paths depend on these exact rules.
# ---------------------------------------------------------------------------

def test_shared_rules_are_the_same_objects_the_engine_uses() -> None:
    from kenning.transcription import _whisper_text as wt
    from kenning.transcription import whisper_engine as we

    assert we._DOMAIN_PROMPT is wt.DOMAIN_PROMPT
    assert we._WHISPER_HALLUCINATIONS is wt.WHISPER_HALLUCINATIONS
    assert we._is_whisper_hallucination is wt.is_whisper_hallucination


def test_user_initial_prompt_augments_never_replaces() -> None:
    """Regression: a short override once SHADOWED the whole domain vocabulary."""
    from kenning.transcription._whisper_text import build_initial_prompt

    out = build_initial_prompt("Kenning.")
    assert out.startswith("Valorant team comms")
    assert out.endswith("Kenning.")


def test_shared_module_is_stdlib_only() -> None:
    """It is imported by the sidecar precisely so a config error cannot kill
    STT -- any heavier import would defeat that."""
    src = (_ROOT / "src" / "kenning" / "transcription"
           / "_whisper_text.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            assert s in ("from __future__ import annotations", "import re"), (
                f"non-stdlib import in the shared decode module: {line!r}")
