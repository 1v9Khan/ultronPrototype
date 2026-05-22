"""Real model-based speculative draft (2026-05-22 experiment).

Until now Ultron's "speculative decoding" surface has been:
- ``LlamaPromptLookupDecoding`` (PLD) -- n-gram matching against the
  prompt, no model loaded. Hit a ``llama_decode returned -1`` bug
  in llama-cpp-python 0.3.22 that we couldn't work around. Disabled.

This module is the *other* speculative-decoding flavour: load the
0.8B Qwen draft GGUF as a real model and use its predictions to draft
candidate tokens that the 4B main model then verifies. Theoretical
benefit (when the draft and main agree on most tokens) is 30-50%
generation throughput vs no draft -- much better than PLD's
typical 5-15% on conversational prompts because the 0.8B can
actually predict diverse continuations rather than only repeating
prompt n-grams.

**Cost-benefit caveat:** the verification batch inside llama-cpp-python
is the SAME C code path that fails on PLD. So there's a real chance
this hits the same ``llama_decode returned -1`` crash. We tag this
as an experiment; the default config keeps ``draft_kind: "none"``
until live verification proves stable.

**Implementation choices:**

- Logits-all stays OFF on the draft itself. We fetch the last-token
  logits via the C-level pointer (``_ctx.get_logits()``) and pick
  greedy argmax in NumPy. Saves the ~5 GB host RAM the scores buffer
  would otherwise consume.

- Prefix caching: the main model's verification cycle feeds us
  ``input_ids`` that mostly extends our prior state -- usually by 1-10
  tokens (the previously-accepted drafts). We track what's already
  in our KV cache and only evaluate the genuinely new tokens. Without
  this, every draft call would re-evaluate the full 1000+-token
  context, costing ~100 ms per call and erasing all speed gains.

- Greedy sampling only. The main model verifies anyway; sampling
  diversity on the draft doesn't help acceptance rate.

The class implements ``LlamaDraftModel`` from
``llama_cpp.llama_speculative`` -- the abstract base llama-cpp-python
expects for the ``draft_model=`` kwarg.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def _import_base():
    """Late import of LlamaDraftModel + Llama. The wrapping ``Llama``
    class triggers a ctypes dlopen of llama.dll at import time, which
    we want gated behind the explicit ``draft_kind: "model"`` opt-in
    so plain ``import ultron.llm.draft_model`` doesn't load the C lib
    on machines that don't have it."""
    from llama_cpp import Llama
    from llama_cpp.llama_speculative import LlamaDraftModel
    return Llama, LlamaDraftModel


def _greedy_sample_last_token(llama, n_vocab: int) -> int:
    """Read the last-position logits from the C library and pick argmax.

    Avoids requiring ``logits_all=True`` on the draft (which would
    cost ~5 GB host RAM for the scores buffer at n_ctx=8192,
    n_vocab=152064). Instead we go directly through the ctx pointer:
    ``_ctx.get_logits()`` always returns the just-evaluated logits.
    """
    logits_ptr = llama._ctx.get_logits()
    # When the draft model was built with logits_all=False, the
    # pointer addresses a flat array of just the last position's
    # n_vocab logits.
    logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))
    return int(np.argmax(logits))


def make_qwen08b_draft_model(
    draft_model_path: str,
    *,
    num_pred_tokens: int = 4,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
):
    """Construct a real model-based speculative draft wrapping a
    second ``Llama`` instance pointing at the 0.8B draft GGUF.

    Args:
        draft_model_path: path to the draft GGUF (e.g.
            ``models/Qwen3.5-0.8B-Q4_K_M.gguf``).
        num_pred_tokens: how many draft tokens to predict per call.
            4 is a conservative starting point -- llama.cpp typically
            accepts ~3-5 of these per verification round in
            conversational workloads, so emitting more wastes the
            extra draft compute on tokens that will be rejected.
        n_ctx: context length. Should match the main model's n_ctx so
            the draft can see the same prompt history.
        n_gpu_layers: -1 means offload all layers to GPU (the 0.8B
            Q4_K_M is ~530 MB on disk; comfortably fits alongside the
            main 4B on a 12 GB card).

    Returns:
        An instance of a ``LlamaDraftModel`` subclass ready to pass to
        ``Llama(draft_model=...)``.

    Notes on the prefix-cache state machine:
        - ``self._evaluated_tokens`` mirrors the ``input_ids`` the
          draft's KV cache currently contains.
        - On each call we compare incoming ``input_ids`` to that list,
          find the longest common prefix, and rewind / extend
          accordingly. This is essential -- without it, a 1000-token
          prompt would cost ~100 ms per draft call, erasing the win.
        - The draft KV state is consumed for our own drafting after
          we extend it with predicted tokens. If the main model later
          rejects some of those drafts, the next call's input_ids
          will diverge before our cached tail, and the prefix match
          detects the rewind cleanly.
    """
    Llama, LlamaDraftModel = _import_base()

    class Qwen08BDraftModel(LlamaDraftModel):
        """Real model-based speculative draft using the 0.8B Qwen GGUF."""

        def __init__(self) -> None:
            self.num_pred_tokens = int(num_pred_tokens)
            self._llama = Llama(
                model_path=str(draft_model_path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=int(n_ctx),
                # The draft must NOT itself wire a draft_model -- that
                # would recurse. Keep logits_all OFF; we read logits
                # via the C ptr below.
                logits_all=False,
                verbose=False,
            )
            self._evaluated_tokens: List[int] = []
            try:
                self._n_vocab = int(self._llama._n_vocab)
            except Exception:                                          # noqa: BLE001
                # Fallback: query the model directly.
                self._n_vocab = int(self._llama.model.n_vocab())
            try:
                self._eos = int(self._llama.token_eos())
            except Exception:                                          # noqa: BLE001
                self._eos = -1
            logger.info(
                "Qwen08BDraftModel ready: path=%s num_pred=%d n_ctx=%d",
                draft_model_path, num_pred_tokens, n_ctx,
            )

        def _resync(self, target: List[int]) -> None:
            """Rewind the draft's KV cache to the longest common prefix
            with ``target`` and re-evaluate any divergent tail."""
            cached = self._evaluated_tokens
            common = 0
            for a, b in zip(cached, target):
                if a == b:
                    common += 1
                else:
                    break

            if common < len(cached):
                # We've gone past the accepted prefix (the main model
                # rejected some of our prior drafts). Rewind by reset
                # + re-eval the common prefix.
                self._llama.reset()
                if common > 0:
                    self._llama.eval(target[:common])
                self._evaluated_tokens = list(target[:common])

            # Eval the genuinely-new tokens.
            tail = target[len(self._evaluated_tokens):]
            if tail:
                self._llama.eval(tail)
                self._evaluated_tokens.extend(tail)

        def __call__(
            self,
            input_ids: npt.NDArray[np.intc],
            /,
            **kwargs: Any,
        ) -> npt.NDArray[np.intc]:
            try:
                target = input_ids.tolist()
                if not target:
                    return np.array([], dtype=np.intc)

                self._resync(target)

                drafts: List[int] = []
                for _ in range(self.num_pred_tokens):
                    next_token = _greedy_sample_last_token(
                        self._llama, self._n_vocab,
                    )
                    if next_token == self._eos:
                        break
                    drafts.append(next_token)
                    # Commit this draft into the draft's KV cache. If
                    # the main rejects it later, the next call's
                    # ``_resync`` will detect the divergence and rewind.
                    self._llama.eval([next_token])
                    self._evaluated_tokens.append(next_token)

                return np.array(drafts, dtype=np.intc)
            except Exception as e:                                     # noqa: BLE001
                logger.warning(
                    "Qwen08BDraftModel: draft step failed (%s); "
                    "returning empty draft so the main model "
                    "generates fresh. Future calls retry.", e,
                )
                # Best-effort reset so we don't carry corrupt state.
                try:
                    self._llama.reset()
                except Exception:
                    pass
                self._evaluated_tokens = []
                return np.array([], dtype=np.intc)

    return Qwen08BDraftModel()


__all__ = ["make_qwen08b_draft_model"]
