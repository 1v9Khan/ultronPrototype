"""Local full-page extraction via the ``trafilatura`` library.

Pulls the raw HTML for a URL with ``requests`` and runs it through
:mod:`trafilatura`'s boilerplate-stripping extractor to get clean
text suitable for LLM context augmentation. Replaces the Jina Reader
external service for sites where local extraction is sufficient
(the vast majority of plain-HTML articles).

Trade-offs vs Jina Reader (the external fallback):

- **Latency:** trafilatura is ~50-150 ms per page on a typical 4070
  Ti machine (just HTTP + Python parsing), vs Jina's ~1-3 s round-
  trip to r.jina.ai. Big win on the voice hot path.
- **Quality:** trafilatura is among the best open-source boilerplate
  removers (consistently top-2 on extraction benchmarks). Sufficient
  for clean news / blog / docs content -- the kind of page the
  voice assistant actually gets asked about.
- **JS-heavy sites:** trafilatura only sees the *raw HTML response*.
  Single-page apps (React/Vue dashboards) that hydrate content with
  JavaScript will return an empty body. Jina Reader handles those
  because it runs a real headless browser. The reader chain falls
  through to Jina in those cases.
- **Cloudflare / WAF challenges:** trafilatura inherits whatever
  ``requests`` sees. Sites blocking direct ``python-requests``
  user-agent will 403 us; Jina (with its real browser fingerprint)
  often gets through. Again the chain handles fallback.
- **Privacy:** trafilatura does NOT phone home anywhere -- the
  outbound HTTP is YOUR machine fetching the target page directly.
  Jina sees every URL you fetch.

Same ``fetch(url) -> Optional[str]`` interface as
:class:`ultron.web_search.jina.JinaReaderClient` so the reader chain
can swap them transparently.
"""

from __future__ import annotations

import time
from typing import Optional

from ultron.config import get_config
from ultron.errors import JinaReaderError
from ultron.resilience import CircuitBreaker, CircuitOpenError, get_error_log
from ultron.utils.logging import get_logger

logger = get_logger("web_search.trafilatura")


class TrafilaturaReaderError(JinaReaderError):
    """Trafilatura-specific extraction failure. Subclasses
    :class:`JinaReaderError` so the existing error-log writers /
    circuit-breaker helpers handle either reader uniformly."""


_TRAFILATURA_BREAKER = CircuitBreaker(
    name="trafilatura",
    failure_threshold=5,
    window_seconds=300.0,
    cooldown_seconds=120.0,
    expected_exceptions=(TrafilaturaReaderError,),
)


# Browser-ish User-Agent — many sites refuse the default Python one.
# Not a workaround for serious WAF challenges (those need a real
# headless browser via Jina), just for sites that gate on the most
# obvious "python-requests/..." string.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
}


class TrafilaturaReaderClient:
    """Local-extraction reader. Drop-in replacement for
    :class:`JinaReaderClient` -- same ``fetch(url) -> Optional[str]``.

    Args:
        timeout_s: per-request HTTP timeout for downloading the page.
            trafilatura's parse step is purely CPU and adds ~50-100 ms
            on typical pages.
        max_bytes: truncate output at this many chars (trailing-edge)
            so giant pages don't balloon the LLM prompt. Mirrors
            the JinaReaderClient cap.
    """

    def __init__(
        self,
        timeout_s: Optional[float] = None,
        max_bytes: Optional[int] = None,
        max_html_bytes: Optional[int] = None,
    ) -> None:
        # Trafilatura-side config lives at web_search.trafilatura.*
        # but it inherits defaults from the Jina block when fields are
        # absent so operators don't have to write both.
        ws_cfg = get_config().web_search
        traf_cfg = getattr(ws_cfg, "trafilatura", None)
        jina_cfg = ws_cfg.jina
        self.timeout_s = (
            timeout_s
            if timeout_s is not None
            else float(
                getattr(traf_cfg, "timeout_seconds", None)
                or jina_cfg.timeout_seconds
            )
        )
        self.max_bytes = (
            max_bytes
            if max_bytes is not None
            else int(
                getattr(traf_cfg, "max_bytes", None) or jina_cfg.max_bytes
            )
        )
        # 2026-05-22: cap the raw HTML BEFORE running trafilatura.extract.
        # Live session hit a 200k-output page that took 5.75 s of CPU
        # extraction -- the LLM context can never use that much per
        # source, so paying the parse cost is pure waste. Truncating
        # the HTML at ~1 MB caps worst-case extraction time at ~1-2 s.
        self.max_html_bytes = (
            max_html_bytes
            if max_html_bytes is not None
            else int(getattr(traf_cfg, "max_html_bytes", 1_048_576))
        )

    def fetch(self, url: str) -> Optional[str]:
        """Return cleaned-up text for ``url`` or ``None`` on failure.

        Failures (timeout, HTTP error, empty extraction, circuit open)
        record to ``logs/errors.jsonl`` and return ``None``; the
        reader chain then falls through to Jina.
        """
        url = url.strip()
        if not url:
            return None
        try:
            return _TRAFILATURA_BREAKER.call(self._do_fetch, url)
        except CircuitOpenError as e:
            logger.warning(
                "Trafilatura circuit OPEN for %s — short-circuiting; %s",
                url[:80], e,
            )
            get_error_log().record(
                TrafilaturaReaderError(
                    "circuit open",
                    context={"url": url[:200], "circuit": "trafilatura"},
                    recovery="short-circuited; reader chain falls to Jina",
                ),
                dependency="trafilatura",
                include_traceback=False,
            )
            return None
        except TrafilaturaReaderError as e:
            get_error_log().record(
                e.with_recovery(
                    "returned None; reader chain falls to Jina"
                ),
                dependency="trafilatura",
            )
            return None

    def _do_fetch(self, url: str) -> Optional[str]:
        """Inner implementation. Raises :class:`TrafilaturaReaderError`
        on any failure; the breaker counts toward the threshold."""
        import requests
        import trafilatura

        t0 = time.monotonic()
        try:
            resp = requests.get(
                url,
                headers=_REQUEST_HEADERS,
                timeout=self.timeout_s,
                allow_redirects=True,
            )
            resp.raise_for_status()
            html = resp.text
            # 2026-05-22 perf: cap input HTML so giant pages don't pin
            # CPU. Trafilatura.extract on multi-MB HTML can take 5-10 s
            # on CPU; truncating to ~1 MB still gives us the article
            # content (which is always near the top of the document).
            if self.max_html_bytes and len(html) > self.max_html_bytes:
                logger.debug(
                    "Trafilatura: truncating %d-char HTML to %d for %s",
                    len(html), self.max_html_bytes, url[:80],
                )
                html = html[: self.max_html_bytes]
        except requests.exceptions.Timeout as e:
            raise TrafilaturaReaderError(
                f"trafilatura GET timed out after {self.timeout_s:.1f}s",
                context={"url": url[:200], "timeout_s": self.timeout_s},
            ) from e
        except requests.exceptions.HTTPError as e:
            raise TrafilaturaReaderError(
                f"HTTP {resp.status_code} on {url}",
                context={
                    "url": url[:200],
                    "status": resp.status_code,
                    "body_preview": resp.text[:300],
                },
            ) from e
        except requests.exceptions.RequestException as e:
            raise TrafilaturaReaderError(
                f"trafilatura GET failed: {e}",
                context={"url": url[:200], "error": str(e)[:200]},
            ) from e

        try:
            # ``include_comments=False`` strips reader-comments sections.
            # ``include_tables=True`` keeps table content (useful for
            # specs / pricing / docs pages).
            # ``output_format='markdown'`` returns cleaner-than-text
            # output the LLM can parse without losing structure.
            extracted = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                output_format="markdown",
                with_metadata=False,
                favor_recall=False,
            )
        except Exception as e:                                         # noqa: BLE001
            raise TrafilaturaReaderError(
                f"trafilatura.extract raised: {e}",
                context={"url": url[:200], "error": str(e)[:200]},
            ) from e

        if not extracted or not extracted.strip():
            # Common case: JS-heavy SPA where the raw HTML has no
            # body content. Let the chain fall through to Jina.
            raise TrafilaturaReaderError(
                "trafilatura returned empty extraction; "
                "likely JS-rendered page",
                context={"url": url[:200], "html_bytes": len(html)},
            )

        # Truncate at the trailing edge (mirrors Jina behaviour).
        if self.max_bytes and len(extracted) > self.max_bytes:
            extracted = (
                extracted[: self.max_bytes]
                + "\n\n[... truncated; original was "
                + f"{len(extracted)} chars]"
            )

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "Trafilatura: %s -> %d chars in %.0f ms",
            url[:80], len(extracted), elapsed_ms,
        )
        return extracted


__all__ = ["TrafilaturaReaderClient", "TrafilaturaReaderError"]
