"""Tests for the real model-based speculative draft (2026-05-22).

The factory ``make_qwen08b_draft_model`` constructs an internal class
that closes over the fake Llama instance, so most tests patch the
``Llama`` import via ``ultron.llm.draft_model._import_base``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake Llama / LlamaDraftModel
# ---------------------------------------------------------------------------


class _FakeLlamaDraftModel:
    """Stand-in for llama_cpp.llama_speculative.LlamaDraftModel ABC.
    Provides a no-op base so our subclass can be instantiated without
    the real C library."""


def _build_fake_llama(*, vocab_size: int = 32, eos_id: int = 0):
    """Build a fake Llama mock that simulates the eval / get_logits
    surface our draft model depends on."""
    fake = MagicMock()
    fake._n_vocab = vocab_size
    fake.token_eos.return_value = eos_id

    # Track state for the prefix-cache logic.
    fake._evaluated: list[int] = []

    def _reset():
        fake._evaluated.clear()

    def _eval(tokens):
        fake._evaluated.extend(list(tokens))

    fake.reset = MagicMock(side_effect=_reset)
    fake.eval = MagicMock(side_effect=_eval)

    # The draft samples via ``_greedy_sample_last_token`` which calls
    # ``llama._ctx.get_logits()`` and runs argmax. Provide logits where
    # token (len(_evaluated) % vocab_size) is the max each time, so
    # successive eval calls yield predictable draft tokens.
    def _get_logits():
        logits = np.full(vocab_size, -1e9, dtype=np.float32)
        # Pick a token that's not EOS so the loop doesn't bail early.
        pick = (len(fake._evaluated) + 1) % vocab_size
        if pick == eos_id:
            pick = (pick + 1) % vocab_size
        logits[pick] = 1.0
        return _PtrLike(logits)

    class _Ctx:
        def get_logits(self):
            return _get_logits()

    fake._ctx = _Ctx()
    return fake


class _PtrLike:
    """Minimal stand-in for the ctypes pointer; np.ctypeslib.as_array
    can wrap a numpy array as-is when the shape matches."""

    def __init__(self, arr):
        self._arr = arr

    @property
    def __array_interface__(self):
        return self._arr.__array_interface__


def _patched_import_base(fake_llama):
    """Patch _import_base to return (FakeLlamaCls, _FakeLlamaDraftModel)
    so the subclass can be created and the test can drive it."""

    class _FakeLlamaCls:
        def __init__(self, *args, **kwargs):
            pass

        def __new__(cls, *args, **kwargs):
            return fake_llama

    return (_FakeLlamaCls, _FakeLlamaDraftModel)


@pytest.fixture
def fake_draft_model(monkeypatch):
    """Build a draft model wired against a fake Llama."""
    fake_llama = _build_fake_llama(vocab_size=32, eos_id=0)

    monkeypatch.setattr(
        "ultron.llm.draft_model._import_base",
        lambda: _patched_import_base(fake_llama),
    )

    # Replace np.ctypeslib.as_array with a passthrough that
    # accepts our _PtrLike + shape and returns the underlying array.
    real_as_array = np.ctypeslib.as_array

    def _as_array(ptr, shape=None):
        if isinstance(ptr, _PtrLike):
            return ptr._arr.reshape(shape) if shape else ptr._arr
        return real_as_array(ptr, shape=shape)

    monkeypatch.setattr(np.ctypeslib, "as_array", _as_array)

    from ultron.llm.draft_model import make_qwen08b_draft_model

    draft = make_qwen08b_draft_model(
        draft_model_path="/fake/qwen-0.8b.gguf",
        num_pred_tokens=4,
        n_ctx=512,
    )
    return draft, fake_llama


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_draft_model_constructs(fake_draft_model):
    draft, fake_llama = fake_draft_model
    assert draft is not None
    assert draft.num_pred_tokens == 4
    assert draft._n_vocab == 32


# ---------------------------------------------------------------------------
# __call__ behavior
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_draft(fake_draft_model):
    draft, fake_llama = fake_draft_model
    out = draft(np.array([], dtype=np.intc))
    assert out.dtype == np.intc
    assert len(out) == 0
    fake_llama.eval.assert_not_called()


def test_first_call_evaluates_full_input(fake_draft_model):
    draft, fake_llama = fake_draft_model
    out = draft(np.array([10, 11, 12], dtype=np.intc))
    # The initial eval should cover the input plus each subsequent
    # drafted token's solo eval.
    eval_args = [c.args[0] for c in fake_llama.eval.call_args_list]
    assert eval_args[0] == [10, 11, 12]
    assert len(out) == draft.num_pred_tokens


def test_second_call_uses_prefix_cache_no_rewind(fake_draft_model):
    """When the second call extends the first, we should only eval the
    new tail -- not reset + re-eval the whole prompt."""
    draft, fake_llama = fake_draft_model
    draft(np.array([10, 11, 12], dtype=np.intc))
    initial_evals = list(fake_llama.eval.call_args_list)
    fake_llama.reset.reset_mock()
    # Second call extends with one new accepted token. The previous
    # call also extended _evaluated with drafts; the test's fake_llama
    # tracks those via _evaluated. Extending past those is fine.
    extension = fake_llama._evaluated + [99]
    draft(np.asarray(extension, dtype=np.intc))
    # No full reset on a clean extension.
    fake_llama.reset.assert_not_called()
    # The new tail (the single 99 token plus any subsequent drafts) was eval'd.
    new_eval_args = [
        c.args[0] for c in fake_llama.eval.call_args_list[len(initial_evals):]
    ]
    assert [99] in new_eval_args


def test_diverged_input_rewinds_to_common_prefix(fake_draft_model):
    """If the main model rejects our drafts, the next input_ids will
    diverge before our cached tail; the draft must reset + re-eval
    from the common prefix."""
    draft, fake_llama = fake_draft_model
    draft(np.array([10, 11, 12], dtype=np.intc))
    fake_llama.reset.reset_mock()
    # Diverge at position 3 (after the original prompt's last token).
    draft(np.array([10, 11, 12, 999], dtype=np.intc))
    fake_llama.reset.assert_called_once()


def test_eos_stops_drafting_early(fake_draft_model):
    """When the draft would emit EOS, it should stop and return only
    the tokens generated up to that point."""
    # Re-build with a logits function that emits EOS on the 2nd draft step.
    fake_llama = _build_fake_llama(vocab_size=32, eos_id=5)

    # Override logits to return EOS after one drafted token.
    state = {"calls": 0}

    def _get_logits_eos():
        state["calls"] += 1
        logits = np.full(32, -1e9, dtype=np.float32)
        if state["calls"] >= 2:
            logits[5] = 1.0  # EOS
        else:
            logits[7] = 1.0
        return _PtrLike(logits)

    fake_llama._ctx.get_logits = _get_logits_eos

    with patch(
        "ultron.llm.draft_model._import_base",
        lambda: _patched_import_base(fake_llama),
    ), patch(
        "numpy.ctypeslib.as_array",
        side_effect=lambda ptr, shape=None: (
            ptr._arr.reshape(shape) if shape else ptr._arr
        ) if isinstance(ptr, _PtrLike) else ptr,
    ):
        from ultron.llm.draft_model import make_qwen08b_draft_model
        draft = make_qwen08b_draft_model(
            draft_model_path="/fake.gguf", num_pred_tokens=8, n_ctx=512,
        )
        out = draft(np.array([1, 2, 3], dtype=np.intc))
        # First call returned token 7; second call hit EOS and stopped.
        assert list(out) == [7]


def test_draft_failure_returns_empty_and_resets_state(fake_draft_model):
    """If the underlying eval raises, the draft must surface an empty
    array (so the main model proceeds without drafts) and reset state
    so subsequent calls aren't poisoned."""
    draft, fake_llama = fake_draft_model
    fake_llama.eval.side_effect = RuntimeError("boom")
    out = draft(np.array([10, 11, 12], dtype=np.intc))
    assert out.dtype == np.intc
    assert len(out) == 0
    # State cleared so the next call starts fresh.
    assert draft._evaluated_tokens == []


def test_draft_returns_intc_dtype(fake_draft_model):
    """The llama-cpp draft-model contract requires np.intc dtype."""
    draft, _ = fake_draft_model
    out = draft(np.array([10, 11, 12], dtype=np.intc))
    assert out.dtype == np.intc


# ---------------------------------------------------------------------------
# Config dispatch
# ---------------------------------------------------------------------------


def test_default_draft_kind_is_none():
    """Live config should default ``draft_kind`` to ``"none"`` until
    we verify the model path is stable."""
    from ultron.config import get_config
    assert get_config().llm.draft_kind == "none"
