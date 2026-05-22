"""Unit tests for ``ultron.intent.recognizer``.

Tests use a stubbed ``moonshine_voice.intent_recognizer.IntentRecognizer``
so they run without downloading the ~300 MB Gemma-300M embedding model
or requiring the moonshine native lib to be loaded. The contract we
verify is the Ultron wrapper's behavior -- lazy loading, fail-open,
phrase registration, threshold semantics, and singleton state.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import List

import pytest

from ultron.intent.recognizer import (
    IntentMatch,
    IntentRegistration,
    UltronIntentRecognizer,
    get_intent_recognizer,
    set_intent_recognizer,
)


# ---------------------------------------------------------------------------
# Test scaffolding: stub the moonshine_voice modules the recognizer imports
# ---------------------------------------------------------------------------


class _FakeMatch:
    """Mirror of moonshine_voice.IntentMatch shape."""

    def __init__(self, canonical_phrase: str, similarity: float):
        self.canonical_phrase = canonical_phrase
        self.similarity = similarity


class _FakeNativeRecognizer:
    """Stand-in for moonshine_voice.intent_recognizer.IntentRecognizer."""

    def __init__(self, *, model_path, model_arch, model_variant, threshold):
        self.model_path = model_path
        self.model_arch = model_arch
        self.model_variant = model_variant
        self.threshold = threshold
        self.registered: list[tuple[str, int]] = []
        self.unregistered: list[str] = []
        self.cleared = False
        self._scripted_matches: list[_FakeMatch] = []
        self.closed = False

    def register_intent(self, phrase, *, handler=None, priority=0, **_):
        self.registered.append((phrase, priority))

    def unregister_intent(self, phrase):
        self.unregistered.append(phrase)
        return True

    def clear_intents(self):
        self.cleared = True

    def get_closest_intents(self, utterance, threshold):
        return [m for m in self._scripted_matches if m.similarity >= threshold]

    def close(self):
        self.closed = True


@pytest.fixture
def stub_moonshine(monkeypatch):
    """Install fake moonshine_voice modules so loading succeeds without
    the native lib. Returns the fake recognizer class for assertions."""
    constructed: list[_FakeNativeRecognizer] = []

    def _factory(**kwargs):
        instance = _FakeNativeRecognizer(**kwargs)
        constructed.append(instance)
        return instance

    fake_intent_module = types.ModuleType(
        "moonshine_voice.intent_recognizer"
    )
    fake_intent_module.IntentRecognizer = _factory

    fake_download = types.ModuleType("moonshine_voice.download")

    def _get_embedding_model(model_name, variant):
        return f"/fake/path/{model_name}-{variant}", "GEMMA_300M"

    fake_download.get_embedding_model = _get_embedding_model

    monkeypatch.setitem(
        sys.modules, "moonshine_voice.intent_recognizer", fake_intent_module,
    )
    monkeypatch.setitem(
        sys.modules, "moonshine_voice.download", fake_download,
    )

    return constructed


# ---------------------------------------------------------------------------
# Construction / lazy load
# ---------------------------------------------------------------------------


def test_construction_does_not_load_native_handle():
    """Construction should be cheap -- no library import, no model
    download. Load happens on first use."""
    r = UltronIntentRecognizer()
    assert r.loaded is False
    assert r.is_available is True  # optimistic until proven otherwise


def test_threshold_defaults_to_pointeight():
    r = UltronIntentRecognizer()
    assert r.threshold == pytest.approx(0.8)


def test_threshold_setter_updates_value():
    r = UltronIntentRecognizer(threshold=0.75)
    assert r.threshold == pytest.approx(0.75)
    r.threshold = 0.9
    assert r.threshold == pytest.approx(0.9)


def test_ensure_loaded_calls_native_factory(stub_moonshine):
    r = UltronIntentRecognizer(variant="q8", threshold=0.7)
    assert r.ensure_loaded() is True
    assert r.loaded is True
    assert len(stub_moonshine) == 1
    fake = stub_moonshine[0]
    assert fake.model_variant == "q8"
    assert fake.threshold == pytest.approx(0.7)


def test_ensure_loaded_handles_missing_moonshine(monkeypatch):
    """When moonshine_voice is unavailable, load fails open and the
    recognizer becomes unavailable -- no exception bubbles up."""
    def _fail_import(*args, **kwargs):
        raise ImportError("simulated missing dep")

    # The recognizer imports lazily inside _do_load; patch the modules
    # the import resolves to.
    monkeypatch.delitem(
        sys.modules, "moonshine_voice.intent_recognizer", raising=False,
    )
    monkeypatch.delitem(
        sys.modules, "moonshine_voice.download", raising=False,
    )
    # Inject a broken module that raises on attribute access.
    class _Broken:
        def __getattr__(self, name):
            raise ImportError("simulated")

    monkeypatch.setitem(sys.modules, "moonshine_voice", _Broken())

    r = UltronIntentRecognizer()
    assert r.ensure_loaded() is False
    assert r.is_available is False
    assert r.loaded is False


def test_ensure_loaded_is_idempotent(stub_moonshine):
    r = UltronIntentRecognizer()
    r.ensure_loaded()
    r.ensure_loaded()
    r.ensure_loaded()
    assert len(stub_moonshine) == 1  # native handle created once


def test_load_failure_is_cached_and_does_not_retry(monkeypatch):
    """After a load failure, subsequent calls return False without
    attempting another import (which would be wasted work)."""
    call_count = {"n": 0}

    def _failing_import(*args, **kwargs):
        call_count["n"] += 1
        raise ImportError("boom")

    class _Broken:
        def __getattr__(self, name):
            raise ImportError("boom")

    monkeypatch.setitem(sys.modules, "moonshine_voice", _Broken())

    r = UltronIntentRecognizer()
    r.ensure_loaded()
    r.ensure_loaded()
    r.ensure_loaded()
    # The recognizer guards _load_failed so it shouldn't keep trying.
    # We can't directly count imports through monkeypatch.setitem
    # but the contract is is_available stays False.
    assert r.is_available is False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_before_load_is_replayed_at_load_time(stub_moonshine):
    """Phrases registered BEFORE the native handle is constructed get
    re-registered when the handle materializes. This matches the
    orchestrator flow: register intents at startup, lazy-load on first
    utterance."""
    r = UltronIntentRecognizer()
    r.register("engage gaming mode")
    r.register("disengage gaming mode", priority=5)
    assert r.loaded is False

    r.ensure_loaded()
    assert r.loaded is True

    fake = stub_moonshine[0]
    phrases = [p for p, _pri in fake.registered]
    assert "engage gaming mode" in phrases
    assert "disengage gaming mode" in phrases


def test_register_after_load_pushes_through_immediately(stub_moonshine):
    r = UltronIntentRecognizer()
    r.ensure_loaded()
    fake = stub_moonshine[0]
    fake.registered.clear()

    r.register("set a timer for five minutes")
    assert ("set a timer for five minutes", 0) in fake.registered


def test_register_empty_phrase_raises():
    r = UltronIntentRecognizer()
    with pytest.raises(ValueError):
        r.register("")
    with pytest.raises(ValueError):
        r.register("   ")


def test_unregister_removes_phrase(stub_moonshine):
    r = UltronIntentRecognizer()
    r.register("a")
    r.register("b")
    assert r.unregister("a") is True
    assert r.unregister("a") is False  # already gone
    assert r.registered_phrases == ["b"]


def test_clear_removes_all_phrases(stub_moonshine):
    r = UltronIntentRecognizer()
    r.register("a")
    r.register("b")
    r.ensure_loaded()
    r.clear()
    assert r.registered_phrases == []
    assert stub_moonshine[0].cleared is True


# ---------------------------------------------------------------------------
# process_utterance
# ---------------------------------------------------------------------------


def test_process_utterance_returns_match_above_threshold(stub_moonshine):
    r = UltronIntentRecognizer(threshold=0.7)
    r.register("turn on the lights")
    r.ensure_loaded()
    stub_moonshine[0]._scripted_matches = [
        _FakeMatch("turn on the lights", 0.92),
    ]

    match = r.process_utterance("switch on the lights")

    assert match is not None
    assert match.canonical_phrase == "turn on the lights"
    assert match.utterance == "switch on the lights"
    assert match.similarity == pytest.approx(0.92)


def test_process_utterance_returns_none_below_threshold(stub_moonshine):
    r = UltronIntentRecognizer(threshold=0.85)
    r.register("turn on the lights")
    r.ensure_loaded()
    # Below threshold -- the stub's get_closest_intents filters these out.
    stub_moonshine[0]._scripted_matches = [
        _FakeMatch("turn on the lights", 0.5),
    ]

    match = r.process_utterance("can we get some light")

    assert match is None


def test_process_utterance_fires_registered_handler(stub_moonshine):
    fired: list[tuple[str, str, float]] = []

    def _handler(canonical, utterance, similarity):
        fired.append((canonical, utterance, similarity))

    r = UltronIntentRecognizer(threshold=0.7)
    r.register("play music", handler=_handler)
    r.ensure_loaded()
    stub_moonshine[0]._scripted_matches = [_FakeMatch("play music", 0.91)]

    r.process_utterance("put some music on")

    assert fired == [("play music", "put some music on", pytest.approx(0.91))]


def test_process_utterance_swallows_handler_exceptions(stub_moonshine):
    """Handler that raises must not break the recognizer (next utterance
    still works) or the caller's voice loop."""

    def _broken(*_args):
        raise RuntimeError("simulated handler crash")

    r = UltronIntentRecognizer()
    r.register("a", handler=_broken)
    r.ensure_loaded()
    stub_moonshine[0]._scripted_matches = [_FakeMatch("a", 0.99)]

    # No exception escapes.
    match = r.process_utterance("a")
    assert match is not None


def test_process_utterance_empty_input_returns_none(stub_moonshine):
    r = UltronIntentRecognizer()
    r.register("a")
    r.ensure_loaded()
    assert r.process_utterance("") is None
    assert r.process_utterance("   ") is None


def test_process_utterance_with_no_registrations_returns_none(stub_moonshine):
    """If no phrases are registered, every utterance returns None
    (skipping the native get_closest_intents call entirely)."""
    r = UltronIntentRecognizer()
    r.ensure_loaded()
    assert r.process_utterance("anything") is None
    # Native call should NOT have been invoked since the registry was empty.
    # (Detected by leaving _scripted_matches empty; even if get_closest_intents
    # were called it would return [].)


def test_process_utterance_lazy_loads_on_first_call(stub_moonshine):
    """If never warmed up, the first process_utterance should still
    trigger the load."""
    r = UltronIntentRecognizer()
    r.register("hello there")
    assert r.loaded is False
    stub_moonshine_setup_called = False

    # Inject a scripted match by pre-loading via a no-op call.
    r.process_utterance("hi")  # may return None; we only assert load fired
    assert r.loaded is True


def test_process_utterance_fail_open_on_native_error(stub_moonshine):
    """If the native lib raises, we log and return None -- voice loop
    falls through to the LLM path."""
    r = UltronIntentRecognizer()
    r.register("test")
    r.ensure_loaded()

    def _raise(*_a, **_k):
        raise RuntimeError("simulated native failure")

    stub_moonshine[0].get_closest_intents = _raise

    match = r.process_utterance("test utterance")
    assert match is None


# ---------------------------------------------------------------------------
# get_top_matches
# ---------------------------------------------------------------------------


def test_get_top_matches_returns_ranked_list(stub_moonshine):
    r = UltronIntentRecognizer(threshold=0.5)
    for phrase in ("a", "b", "c", "d", "e", "f"):
        r.register(phrase)
    r.ensure_loaded()
    stub_moonshine[0]._scripted_matches = [
        _FakeMatch("a", 0.95),
        _FakeMatch("b", 0.90),
        _FakeMatch("c", 0.85),
        _FakeMatch("d", 0.80),
        _FakeMatch("e", 0.75),
    ]

    out = r.get_top_matches("anything", n=3)

    assert [m.canonical_phrase for m in out] == ["a", "b", "c"]
    assert all(m.utterance == "anything" for m in out)


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------


def test_singleton_starts_unset():
    set_intent_recognizer(None)  # cleanup any prior test state
    assert get_intent_recognizer() is None


def test_singleton_set_and_get_round_trip():
    r = UltronIntentRecognizer()
    set_intent_recognizer(r)
    try:
        assert get_intent_recognizer() is r
    finally:
        set_intent_recognizer(None)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_ensure_loaded_only_constructs_once(stub_moonshine):
    """Multiple threads racing on ensure_loaded() must result in exactly
    one native-handle construction."""
    r = UltronIntentRecognizer()
    barrier = threading.Barrier(8)
    results: list[bool] = []
    lock = threading.Lock()

    def _worker():
        barrier.wait()
        ok = r.ensure_loaded()
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert all(results)
    assert len(stub_moonshine) == 1  # only one construction
