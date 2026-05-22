"""Multi-reader chain for full-page text extraction.

Same cascade pattern as :class:`SearchProviderChain` (providers for
search) but for the READING step: try local trafilatura first, fall
through to Jina Reader (external) when local extraction fails or
returns empty (typical for JS-heavy SPAs).

Configured via ``web_search.readers`` (ordered list of reader IDs).
Default order:

    ["trafilatura", "jina"]

This gives:

- **Trafilatura (local):** ~50-150 ms, no external dependency, no
  privacy leak. Handles 80%+ of typical news / blog / docs pages.
- **Jina Reader (external):** ~1-3 s round-trip, runs a real
  headless browser server-side, handles JS-heavy + Cloudflare-
  challenged pages. Fallback for what trafilatura can't extract.

Same ``fetch(url) -> Optional[str]`` interface as either client
individually, so the search executor can use the chain
transparently in place of ``JinaReaderClient``.

Failure semantics:
- Reader returns non-empty markdown -> chain stops, returns it.
- Reader returns ``None`` (its own failure) -> chain falls through.
- All readers return ``None`` -> chain returns ``None`` (search
  executor downgrades the source to snippet-only).

Each reader has its own circuit breaker, so a flapping one gets
short-circuited without slowing the chain down.
"""

from __future__ import annotations

import time
from typing import List, Optional

from ultron.config import get_config
from ultron.utils.logging import get_logger

logger = get_logger("web_search.reader_chain")


class ReaderChain:
    """Sequenced page-extraction readers with local-first fallback.

    Args:
        reader_ids: ordered list of reader names. None -> read from
            ``web_search.readers`` config.

    Raises:
        ValueError: if an unknown reader id is configured.
    """

    _READER_FACTORIES = {
        "trafilatura": lambda: _make_trafilatura(),
        "jina": lambda: _make_jina(),
    }

    def __init__(self, reader_ids: Optional[List[str]] = None) -> None:
        if reader_ids is None:
            cfg = get_config().web_search
            reader_ids = list(getattr(cfg, "readers", ["jina"]))
        if not reader_ids:
            raise ValueError("reader_ids cannot be empty")

        self.reader_ids: List[str] = []
        self._clients: dict = {}
        for rid in reader_ids:
            rid = rid.lower().strip()
            if rid not in self._READER_FACTORIES:
                raise ValueError(
                    f"Unknown reader {rid!r}; "
                    f"valid options: {sorted(self._READER_FACTORIES)}"
                )
            self.reader_ids.append(rid)
        logger.info(
            "Reader chain: %s (in order; first non-empty wins)",
            " -> ".join(self.reader_ids),
        )

    def _get_client(self, rid: str):
        """Lazy-construct + cache the client for ``rid``. Returns
        ``None`` if construction failed (reader gets skipped)."""
        if rid in self._clients:
            return self._clients[rid]
        try:
            client = self._READER_FACTORIES[rid]()
            self._clients[rid] = client
            return client
        except Exception as e:                                         # noqa: BLE001
            logger.warning(
                "Failed to construct %r reader (%s); skipping in chain.",
                rid, e,
            )
            self._clients[rid] = None
            return None

    def fetch(self, url: str) -> Optional[str]:
        """Run the chain. Returns the first non-None / non-empty
        extraction, or ``None`` if every reader fails."""
        url = url.strip()
        if not url:
            return None

        for rid in self.reader_ids:
            client = self._get_client(rid)
            if client is None:
                continue
            t0 = time.monotonic()
            try:
                text = client.fetch(url)
            except Exception as e:                                     # noqa: BLE001
                # Defence-in-depth: clients SHOULD return None rather
                # than raise, but defend against bugs / new readers.
                logger.warning(
                    "Reader %r raised unexpectedly (%s); falling through.",
                    rid, e,
                )
                text = None
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if text and text.strip():
                logger.info(
                    "Chain: %s extracted %d chars from %r in %.0f ms",
                    url[:80], len(text), rid, elapsed_ms,
                )
                return text
            else:
                logger.debug(
                    "Chain: %s empty from %r in %.0f ms; trying next reader",
                    url[:80], rid, elapsed_ms,
                )

        logger.info(
            "Chain: %s exhausted all %d readers; returning None "
            "(source falls back to snippet-only)",
            url[:80], len(self.reader_ids),
        )
        return None


# --- Lazy factory helpers ---------------------------------------------------


def _make_trafilatura():
    from ultron.web_search.trafilatura_reader import TrafilaturaReaderClient
    return TrafilaturaReaderClient()


def _make_jina():
    from ultron.web_search.jina import JinaReaderClient
    return JinaReaderClient()


__all__ = ["ReaderChain"]
