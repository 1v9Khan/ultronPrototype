"""Pluggable similarity backends for the command router.

A ``SimilarityBackend`` scores a query string against a set of exemplar strings
and returns one similarity in [0,1] per exemplar. Two implementations:

  * ``LexicalBackend`` (DEFAULT, gaming-safe): rapidfuzz token-set / WRatio
    fused with a Metaphone phonetic score. Fully in-process, CPU-light,
    deterministic, no model, no new dependency, zero anticheat/RAM footprint.
    Ideal for short, formulaic Valorant callouts.

  * ``EmbeddingBackend`` (OPTIONAL, slot-in): cosine similarity over sentence
    embeddings produced by a SIDECAR process (``scripts/embedder_server.py``),
    NOT in Ultron's process. This module imports ONLY ``urllib`` -- the embedder
    model + fastembed/onnx never load into the anticheat-pinned main process, so
    the boot canary stays ``libs loaded=none`` and there is no CPU contention
    with the gaming LLM/STT. The sidecar is pure compute (no input/capture/
    injection), so it is anticheat-irrelevant -- the same class as OBS/Discord.
    If the sidecar is unreachable, ``available()`` is False and the router
    transparently falls back to the lexical backend.

The router picks ONE backend at construction; switching is a config flag, so an
embedder can be slotted in later with no routing-logic change.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, List, Sequence

import numpy as np

from kenning.utils.logging import get_logger

logger = get_logger("audio.router_backends")

try:
    from rapidfuzz import fuzz as _fuzz
except Exception:                                                 # noqa: BLE001
    _fuzz = None
try:
    import jellyfish as _jf
except Exception:                                                 # noqa: BLE001
    _jf = None

_WORD = re.compile(r"[a-z0-9']+")

# Sentinel cached when a query embed fails, so the remaining per-family calls in
# the SAME turn short-circuit (raising _SidecarUnavailable) instead of each
# paying a full HTTP timeout -- and the HybridBackend failure counter ticks once
# per turn rather than once per family.
_EMBED_FAILED = object()


class _SidecarUnavailable(Exception):
    """This turn's query embed already failed (cached short-circuit)."""


def _norm(s: str) -> str:
    return " ".join(_WORD.findall((s or "").lower()))


def _phonetic(s: str) -> str:
    """Per-word Metaphone code, space-joined (sound-alike comparison key)."""
    words = _WORD.findall((s or "").lower())
    if _jf is None:
        return " ".join(words)
    out = []
    for w in words:
        try:
            out.append(_jf.metaphone(w) or w)
        except Exception:                                         # noqa: BLE001
            out.append(w)
    return " ".join(out)


class SimilarityBackend(ABC):
    """Scores a query against exemplars; returns one similarity in [0,1] each."""

    name = "base"

    def available(self) -> bool:
        return True

    @abstractmethod
    def prepare(self, exemplars: Sequence[str]) -> Any:
        """Precompute whatever ``score`` needs for this exemplar set (called
        once per route at startup). Returns an opaque handle passed to score."""

    @abstractmethod
    def score(self, query: str, prepared: Any) -> List[float]:
        """Return a similarity in [0,1] for ``query`` vs each prepared exemplar."""


class LexicalBackend(SimilarityBackend):
    """rapidfuzz (token-set / WRatio) fused with a Metaphone phonetic ratio.

    Robust to word-order, partial matches, and sound-alike STT errors on short
    callouts. Deterministic + microsecond-cheap; the default gaming backend."""

    name = "lexical"
    LEX_W = 0.75
    PHON_W = 0.25

    def available(self) -> bool:
        return _fuzz is not None

    def prepare(self, exemplars: Sequence[str]) -> Any:
        return [(_norm(e), _phonetic(e)) for e in exemplars]

    def score(self, query: str, prepared: Any) -> List[float]:
        if _fuzz is None:
            return [0.0] * len(prepared)
        qn, qp = _norm(query), _phonetic(query)
        out: List[float] = []
        for en, ep in prepared:
            lex = max(_fuzz.token_set_ratio(qn, en),
                      _fuzz.WRatio(qn, en)) / 100.0
            ph = (_fuzz.ratio(qp, ep) / 100.0) if (qp and ep) else 0.0
            out.append(self.LEX_W * lex + self.PHON_W * ph)
        return out


class EmbeddingBackend(SimilarityBackend):
    """Cosine similarity over sentence embeddings from the SIDECAR process.

    Imports nothing heavy -- only talks HTTP to ``scripts/embedder_server.py``
    on a local port. ``available()`` pings the sidecar; if it's down the router
    falls back to the lexical backend, so this is a pure additive enhancement."""

    name = "embedding"

    def __init__(self, host: str = "127.0.0.1", port: int = 8772,
                 timeout: float = 0.5, prepare_timeout: float = 25.0) -> None:
        self._base = f"http://{host}:{port}"
        # Per-QUERY embed (latency-critical, 1 text): a >0.5s loopback call has
        # already failed. A tight timeout caps the cost when the sidecar dies.
        self._timeout = timeout
        # PREPARE embeds a whole family of exemplars AT ONCE (dozens of texts)
        # and may hit a COLD model on first use -> a generous ONE-TIME timeout.
        # (Using the per-query timeout here silently failed the router build.)
        self._prepare_timeout = prepare_timeout
        # 1-entry cache: the router scores the SAME query against EVERY family in
        # a turn, so cache the last (texts, kind) embedding -> embed each query
        # ONCE per turn instead of once per family.
        self._cache_key = None
        self._cache_val = None

    def _post(self, path: str, payload: dict, timeout: "float | None" = None) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._base + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout or self._timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def available(self) -> bool:
        try:
            req = urllib.request.Request(self._base + "/healthz", method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                return r.status == 200
        except Exception:                                         # noqa: BLE001
            return False

    def _embed(self, texts: Sequence[str], kind: str = "document",
               timeout: "float | None" = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        cache_key = (tuple(texts), kind)
        if cache_key == self._cache_key:
            if self._cache_val is _EMBED_FAILED:
                # Same query already failed THIS turn -> short-circuit the other
                # families without paying another HTTP timeout.
                raise _SidecarUnavailable("sidecar embed failed this turn")
            return self._cache_val
        # ``kind`` lets an asymmetric model (EmbeddingGemma: query vs document
        # prompts -- best routing margins) prompt each side correctly; symmetric
        # models (bge) ignore it.
        try:
            out = self._post("/embed", {"texts": list(texts), "kind": kind},
                             timeout=timeout)
        except Exception:
            # Cache the failure for this (query, kind) so the remaining families
            # this turn skip the HTTP call; re-raise the REAL error so the
            # HybridBackend failure counter ticks exactly ONCE per turn.
            self._cache_key = cache_key
            self._cache_val = _EMBED_FAILED
            raise
        vecs = np.asarray(out.get("vectors", []), dtype=np.float32)
        if vecs.size == 0:
            return np.zeros((len(texts), 1), dtype=np.float32)
        # L2-normalize so a dot product IS cosine similarity.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result = vecs / norms
        self._cache_key = cache_key
        self._cache_val = result
        return result

    def prepare(self, exemplars: Sequence[str]) -> Any:
        # Exemplars are the "documents" in the retrieval framing. This batch
        # embed (dozens of texts, possibly a cold model) runs ONCE at build, so
        # it uses the GENEROUS prepare timeout -- not the tight per-query one.
        return self._embed(exemplars, kind="document",
                           timeout=self._prepare_timeout)

    def score(self, query: str, prepared: Any) -> List[float]:
        ex = prepared
        if ex is None or len(ex) == 0:
            return []
        q = self._embed([query], kind="query")     # utterance = the query side
        if q.size == 0 or q.shape[1] != ex.shape[1]:
            return [0.0] * len(ex)
        return (ex @ q[0]).clip(0.0, 1.0).tolist()


class HybridBackend(SimilarityBackend):
    """Fuses LEXICAL (always) + EMBEDDING (when the sidecar is up).

    The two fail in complementary ways: lexical/phonetic nails exact vocab +
    sound-alikes (agent names, formulaic phrasings) but mis-fires on paraphrases;
    embeddings nail semantic intent but can be fuzzy on exact tokens. Fusing them
    (weighted sum) is the research-recommended hybrid for short, noisy ASR
    commands. If the embedding sidecar is unreachable, this degrades to
    lexical-only transparently, so it is always safe to select."""

    name = "hybrid"

    def __init__(self, embedding: "EmbeddingBackend | None" = None,
                 lexical: "LexicalBackend | None" = None,
                 emb_weight: float = 0.6) -> None:
        self.lex = lexical or LexicalBackend()
        self.emb = embedding
        self.emb_weight = float(emb_weight)
        self._emb_ok = bool(embedding is not None and embedding.available())
        self._emb_fails = 0      # consecutive embedding failures -> latch off

    def available(self) -> bool:
        return self.lex.available()

    def using_embedding(self) -> bool:
        return self._emb_ok

    def prepare(self, exemplars: Sequence[str]) -> Any:
        lp = self.lex.prepare(exemplars)
        ep = None
        if self._emb_ok:
            import time
            last_err = None
            # Retry once: the sidecar /healthz can be up while the FIRST /embed
            # is still slow (EmbeddingGemma warm-up) -- a single retry covers that
            # window so we don't needlessly drop to lexical.
            for attempt in range(2):
                try:
                    ep = self.emb.prepare(exemplars)
                    last_err = None
                    break
                except Exception as e:                            # noqa: BLE001
                    last_err = e
                    if attempt == 0:
                        time.sleep(2.0)
            if last_err is not None:
                # Degrading to lexical here is the SAFETY NET (never crash the
                # build); it is meant to be UNREACHABLE in normal operation, so
                # log it LOUD (ERROR) -- the boot respawn-on-lexical retry acts
                # on this.
                self._emb_ok = False
                logger.error("command router: embedding prepare FAILED after "
                             "retries (%s) -> LEXICAL-ONLY this session", last_err)
        return (lp, ep)

    def score(self, query: str, prepared: Any) -> List[float]:
        lp, ep = prepared
        ls = self.lex.score(query, lp)
        if ep is None or not self._emb_ok:
            return ls
        try:
            es = self.emb.score(query, ep)
            self._emb_fails = 0                  # healthy -> reset
        except _SidecarUnavailable:
            # This turn's embed already failed + was counted on an earlier
            # family -> use lexical without double-counting.
            return ls
        except Exception:                                         # noqa: BLE001
            # Sidecar died: return lexical, and after a few consecutive FAILED
            # TURNS latch the embedding OFF so subsequent turns take the
            # short-circuit above (zero HTTP overhead) instead of eating a
            # timeout every turn forever.
            self._emb_fails += 1
            if self._emb_fails >= 3:
                self._emb_ok = False
                logger.error("command router: embedding sidecar failed %dx -> "
                             "LEXICAL-ONLY for the rest of this session (sidecar "
                             "may have died; try_recover re-enables it if it returns)",
                             self._emb_fails)
            return ls
        if len(es) != len(ls):
            return ls
        w = self.emb_weight
        return [w * e + (1.0 - w) * l for e, l in zip(es, ls)]

    def try_recover(self) -> bool:
        """Re-enable embedding if the sidecar has come back. No-op when already
        on. Called THROTTLED from the idle voice loop so a transient sidecar
        outage (the 3x failure latch) doesn't disable the hybrid for the whole
        session. Returns True iff it (re)enabled embedding on this call."""
        if self._emb_ok or self.emb is None:
            return False
        try:
            if self.emb.available():
                self._emb_fails = 0
                self._emb_ok = True
                logger.info("command router: embedding sidecar recovered -> "
                            "hybrid re-enabled (was lexical-only)")
                return True
        except Exception:                                         # noqa: BLE001
            pass
        return False


def get_backend(prefer: str = "hybrid", *, host: str = "127.0.0.1",
                port: int = 8772, emb_weight: float = 0.6,
                wait_seconds: float = 0.0) -> SimilarityBackend:
    """Build the configured backend. ``hybrid`` (default) = lexical + embedding
    sidecar; ``embedding`` = sidecar-only; ``lexical`` = no sidecar. Any
    embedding path falls back to lexical when the sidecar is unreachable, so
    gaming never blocks on it. ``wait_seconds`` > 0 polls for the sidecar (it
    loads the model async at boot) so a COLD boot still gets the embedding
    backend instead of latching lexical-only for the session."""
    if prefer == "lexical":
        return LexicalBackend()
    emb = EmbeddingBackend(host=host, port=port)
    emb_ok = emb.available()
    if not emb_ok and wait_seconds > 0:
        import time
        deadline = time.monotonic() + wait_seconds
        logger.info("command router: waiting up to %.0fs for the embedding "
                    "sidecar to finish loading...", wait_seconds)
        while time.monotonic() < deadline:
            time.sleep(1.5)
            if emb.available():
                emb_ok = True
                break
    if not emb_ok:
        logger.warning(
            "command router: embedding sidecar unreachable at %s:%d -> "
            "lexical backend (a pure-CPU, no-model fallback)", host, port)
    if prefer == "embedding":
        return emb if emb_ok else LexicalBackend()
    # hybrid (default)
    backend = HybridBackend(embedding=emb if emb_ok else None, emb_weight=emb_weight)
    logger.info("command router: HYBRID backend (embedding=%s + lexical)",
                "on" if emb_ok else "off")
    return backend
