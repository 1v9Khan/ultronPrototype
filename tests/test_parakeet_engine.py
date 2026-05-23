"""Tests for the non-streaming :class:`ParakeetEngine` surface.

Streaming-specific behaviour is covered by
``tests/test_parakeet_streaming_client.py``. This file pins the
constructor, transcribe / cache-hit semantics, and HTTP wire format
for the one-shot ``transcribe()`` path.

No real NeMo / GPU / server load happens -- ``is_nemo_available`` and
``_spawn_server_if_needed`` are monkeypatched, and ``requests`` is
swapped for a recording stub.
"""

from __future__ import annotations

import io
import sys
from typing import Any, Dict, List

import numpy as np
import pytest

from ultron.transcription import parakeet_engine as parakeet_module


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self._payload}")

    def json(self):
        return self._payload


class _FakeRequests:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.next_payload: Dict[str, Any] = {"text": ""}

    class exceptions:
        class RequestException(Exception):
            pass

        class ConnectionError(RequestException):
            pass

        class Timeout(RequestException):
            pass

    def post(self, url, *, data=None, files=None, headers=None, timeout=None):
        self.calls.append({
            "method": "POST",
            "url": url,
            "files": files is not None,
            "headers": headers or {},
            "timeout": timeout,
        })
        return _FakeResponse(self.next_payload)


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(monkeypatch):
    """Build a ParakeetEngine with HTTP layer + NeMo availability stubbed."""
    fake_requests = _FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setattr(parakeet_module, "is_nemo_available", lambda: True)
    monkeypatch.setattr(
        parakeet_module, "_spawn_server_if_needed",
        lambda cfg: "http://127.0.0.1:8771",
    )
    e = parakeet_module.ParakeetEngine()
    # Make the engine's request_timeout deterministic in case the
    # default ever changes.
    e.request_timeout = 7.5
    return e, fake_requests


# ---------------------------------------------------------------------------
# Constructor behaviour
# ---------------------------------------------------------------------------


def test_construct_pulls_server_url_from_spawn_helper(monkeypatch):
    monkeypatch.setattr(parakeet_module, "is_nemo_available", lambda: True)
    seen: list[Any] = []

    def _fake_spawn(cfg):
        seen.append(cfg)
        return "http://example.invalid:1234"

    monkeypatch.setattr(parakeet_module, "_spawn_server_if_needed", _fake_spawn)
    engine = parakeet_module.ParakeetEngine()
    assert engine._server_url == "http://example.invalid:1234"
    assert engine.use_isolated_venv is True
    assert len(seen) == 1


def test_construct_refuses_when_nemo_missing(monkeypatch):
    monkeypatch.setattr(parakeet_module, "is_nemo_available", lambda: False)
    with pytest.raises(ImportError) as exc_info:
        parakeet_module.ParakeetEngine()
    # The error must point operators at the install hint.
    assert "venv-parakeet" in str(exc_info.value) or "nemo" in str(
        exc_info.value
    ).lower()


# ---------------------------------------------------------------------------
# Streaming-cache → transcribe handoff
# ---------------------------------------------------------------------------


def test_transcribe_returns_cached_streaming_text(engine):
    """If a stream just stashed a non-empty result, the next
    ``transcribe`` call returns it instead of hitting the model.

    This is the orchestrator's happy-path optimisation: the
    streaming partial that completed during the user's speech is
    already the right answer, so the post-capture transcribe call
    returns instantly.
    """
    e, fake = engine
    e._last_streaming_text = "hello from the stream"
    out = e.transcribe(np.zeros(16000, dtype=np.float32))
    assert out == "hello from the stream"
    assert e._last_streaming_text is None  # consumed
    # No HTTP call should have happened.
    assert fake.calls == []


def test_transcribe_falls_through_on_empty_cache_signal(engine):
    """``_last_streaming_text = ""`` is the explicit cache-miss
    sentinel from ``stop_stream``: the cached value is empty, so the
    one-shot fallback transcribe MUST run.
    """
    e, fake = engine
    e._last_streaming_text = ""
    fake.next_payload = {"text": "fallback result"}
    out = e.transcribe(np.ones(16000, dtype=np.float32))
    assert out == "fallback result"
    # One HTTP call to /transcribe.
    assert len(fake.calls) == 1
    assert fake.calls[0]["url"].endswith("/transcribe")
    assert fake.calls[0]["files"] is True


def test_transcribe_returns_empty_for_empty_audio(engine):
    e, fake = engine
    e._last_streaming_text = None
    out = e.transcribe(np.zeros(0, dtype=np.float32))
    assert out == ""
    assert fake.calls == []


def test_transcribe_handles_http_error_returns_empty(engine, monkeypatch):
    """A 500 from the server must not propagate -- voice loop continues."""
    e, fake = engine
    e._last_streaming_text = None

    def _broken_post(*args, **kwargs):
        return _FakeResponse({"error": "boom"}, status_code=500)

    fake.post = _broken_post
    out = e.transcribe(np.ones(16000, dtype=np.float32))
    assert out == ""


def test_transcribe_coerces_int16_audio(engine):
    e, fake = engine
    e._last_streaming_text = None
    fake.next_payload = {"text": "ok"}
    pcm_int16 = (np.ones(8000) * 16000).astype(np.int16)
    out = e.transcribe(pcm_int16)
    assert out == "ok"
    # POST happened (engine ran the model path).
    assert any(call["url"].endswith("/transcribe") for call in fake.calls)


# ---------------------------------------------------------------------------
# Streaming protocol nuance: supports_streaming returns True even
# when the gaming-mode engine swap kills the server underneath.
# ---------------------------------------------------------------------------


def test_supports_streaming_is_true(engine):
    e, _ = engine
    assert e.supports_streaming() is True


# ---------------------------------------------------------------------------
# Server URL guards (the wire helpers refuse to fire without a URL)
# ---------------------------------------------------------------------------


def test_stream_url_raises_when_unset(engine, monkeypatch):
    """Without a server URL, the streaming HTTP helpers must raise so
    the caller knows the stream can't start (rather than silently
    losing partials)."""
    e, _ = engine
    e._server_url = None
    # Also clear the module-level cache so the helper truly has no URL.
    monkeypatch.setattr(parakeet_module, "_SERVER_URL_CACHED", None)
    with pytest.raises(RuntimeError) as exc_info:
        e._stream_url()
    assert "server URL not set" in str(exc_info.value)
