"""Image-as-base64-markdown encoding + multimodal segmenter.

Direct port of SWE-Agent's
``tools/image_tools/bin/view_image`` + the companion
``ImageParsingHistoryProcessor`` in
``sweagent/agent/history_processors.py`` (MIT, Yang et al. 2024).

The pattern: encode image bytes as
``![<path>](data:<mime>;base64,<base64>)`` -- a vanilla markdown
image link with a data: URL. Subsequent processing walks the LLM
history, finds these markdown patterns via regex, and SPLITS the
matching content into multi-modal segments
``[{type: text}, {type: image_url}, {type: text}]`` that
multimodal-capable LLMs consume natively.

For ultron the encoder + parser are pure-Python and import-safe.
They're useful even before a multimodal LLM is wired in because:

* The supervisor / narrator can pass images through audit logs in
  human-readable form without bespoke serialisation.
* The desktop crosshair preview (T16) returns annotated PNG bytes
  that this encoder can wrap for the eventual VLM-via-LLM-channel
  path.
* The base64 encoding pipeline gets a stable interface NOW so the
  multimodal call sites are a config-flip + handler swap when the
  time comes.

Optional Pillow integration: when Pillow is present and the
caller passes ``max_dim`` to :func:`encode_image_as_markdown`, the
image is resized so the longer side fits ``max_dim`` before
encoding -- caps the data-URL payload at a manageable size.

The default whitelist matches SWE-Agent's verbatim:
``image/png``, ``image/jpeg``, ``image/webp``.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Allowed MIME types -- verbatim from SWE-Agent's view_image tool.
DEFAULT_ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/webp"}
)

#: Default thumbnail cap -- when set, the longer dim of the image is
#: resized to this many pixels before encoding. Cuts a typical
#: 1920x1080 screenshot to ~250 KB encoded.
DEFAULT_MAX_DIM: int = 1024

#: Regex that locates the base64-markdown pattern in any content
#: string. Pattern verbatim from SWE-Agent's
#: `ImageParsingHistoryProcessor._pattern`.
_PATTERN = re.compile(
    r"(!\[([^\]]*)\]\(data:)([^;]+);base64,([^)]+)(\))"
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodedImage:
    """One encoded image ready to embed in markdown."""

    alt_text: str
    mime_type: str
    base64_data: str
    markdown: str
    original_bytes: int
    encoded_bytes: int

    def as_data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.base64_data}"


@dataclass(frozen=True)
class TextSegment:
    """One text-type multimodal segment."""

    text: str

    def as_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True)
class ImageSegment:
    """One image-type multimodal segment."""

    url: str
    mime_type: str = ""
    alt_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": self.url},
        }


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_image_as_markdown(
    image_bytes: bytes,
    *,
    alt_text: str = "",
    mime_type: str = "image/png",
    max_dim: Optional[int] = DEFAULT_MAX_DIM,
    allowed_mime_types: Iterable[str] = DEFAULT_ALLOWED_MIME_TYPES,
) -> EncodedImage:
    """Return ``image_bytes`` encoded as a markdown image link.

    The output format matches SWE-Agent's
    ``![<alt>](data:<mime>;base64,<base64>)`` verbatim so any
    consumer of the data-URL convention can ingest it.

    :param image_bytes: raw image bytes (PNG / JPEG / WebP).
    :param alt_text: alt-text for the markdown image; defaults to
        empty.
    :param mime_type: MIME type of ``image_bytes``. Must appear in
        ``allowed_mime_types`` or :class:`ValueError`.
    :param max_dim: when set + Pillow available, resize the image so
        its longer side fits this many pixels before encoding. Pass
        ``None`` to disable resizing entirely.
    :param allowed_mime_types: override the whitelist (rarely needed).
    """
    if not image_bytes:
        raise ValueError("image_bytes must be non-empty")
    allowed = frozenset(allowed_mime_types)
    if mime_type not in allowed:
        raise ValueError(
            f"mime_type {mime_type!r} not in allowed set {sorted(allowed)!r}"
        )
    original_bytes = len(image_bytes)
    resized = image_bytes
    if max_dim is not None and max_dim > 0:
        try:
            resized = _maybe_resize(image_bytes, max_dim=max_dim, mime_type=mime_type)
        except Exception as exc:
            logger.warning(
                "encode_image_as_markdown: resize failed: %s; "
                "encoding original",
                exc,
            )
    b64 = base64.b64encode(resized).decode("ascii")
    markdown = f"![{alt_text}](data:{mime_type};base64,{b64})"
    return EncodedImage(
        alt_text=alt_text,
        mime_type=mime_type,
        base64_data=b64,
        markdown=markdown,
        original_bytes=original_bytes,
        encoded_bytes=len(b64),
    )


def _maybe_resize(
    image_bytes: bytes,
    *,
    max_dim: int,
    mime_type: str,
) -> bytes:
    """Resize via Pillow when available + the longer side exceeds ``max_dim``.

    Returns the original bytes when Pillow isn't installed or the
    image already fits.
    """
    try:
        from PIL import Image
    except ImportError:
        return image_bytes
    with Image.open(io.BytesIO(image_bytes)) as im:
        w, h = im.size
        longer = max(w, h)
        if longer <= max_dim:
            return image_bytes
        scale = max_dim / longer
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        # Pillow expects an explicit format on save; map mime->format.
        save_format = {
            "image/png": "PNG",
            "image/jpeg": "JPEG",
            "image/webp": "WEBP",
        }.get(mime_type, "PNG")
        resized = im.resize(new_size)
        out = io.BytesIO()
        # JPEG doesn't accept RGBA -- coerce.
        if save_format == "JPEG" and resized.mode != "RGB":
            resized = resized.convert("RGB")
        resized.save(out, format=save_format)
        return out.getvalue()


# ---------------------------------------------------------------------------
# Parsing -- markdown -> multimodal segments
# ---------------------------------------------------------------------------


def parse_image_markdown(
    content: str,
    *,
    allowed_mime_types: Iterable[str] = DEFAULT_ALLOWED_MIME_TYPES,
) -> list[dict[str, Any]]:
    """Split ``content`` into multimodal segments at markdown image
    boundaries.

    Returns a list whose entries are either
    ``{"type": "text", "text": "..."}`` or
    ``{"type": "image_url", "image_url": {"url": "data:..."}}``
    (matches the LiteLLM / OpenAI multimodal API shape).

    Allowed MIME types come from SWE-Agent's whitelist by default;
    images with disallowed MIME pass through as TEXT (preserving the
    raw markdown so the operator can see what wasn't ingestible).

    Returns ``[{"type": "text", "text": content}]`` when no images
    are present (single-text segment). This makes the function safe
    to call unconditionally on every history item.
    """
    if not content:
        return [{"type": "text", "text": ""}]
    allowed = frozenset(allowed_mime_types)
    segments: list[dict[str, Any]] = []
    last_end = 0
    has_images = False

    def push_text(t: str) -> None:
        if not t:
            return
        if segments and segments[-1].get("type") == "text":
            segments[-1]["text"] += t
        else:
            segments.append({"type": "text", "text": t})

    for m in _PATTERN.finditer(content):
        full_prefix, alt_text, mime_type, base64_data, suffix = m.groups()
        push_text(content[last_end : m.start()])
        # Normalise "image/jpg" -> "image/jpeg" to match SWE-Agent.
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"
        if mime_type in allowed:
            segments.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_data}"
                    },
                }
            )
            has_images = True
        else:
            # Disallowed MIME: keep the raw markdown as text so the
            # operator can see what was rejected.
            push_text(m.group(0))
        last_end = m.end()
    push_text(content[last_end:])

    if not has_images:
        # No images parsed -- collapse to a single text segment.
        return [{"type": "text", "text": content}]
    return segments


def history_to_multimodal(
    history: Iterable[dict[str, Any]],
    *,
    allowed_mime_types: Iterable[str] = DEFAULT_ALLOWED_MIME_TYPES,
) -> list[dict[str, Any]]:
    """Walk ``history`` + rewrite items whose content has base64-markdown
    image patterns into multimodal segment lists.

    Items without image patterns pass through unchanged. Items that
    DO have images get their ``content`` replaced with a list of
    multimodal segments. Mirrors SWE-Agent's
    :class:`ImageParsingHistoryProcessor`.

    Returns a new list (input is not mutated).
    """
    out: list[dict[str, Any]] = []
    allowed = frozenset(allowed_mime_types)
    for item in history:
        if not isinstance(item, dict):
            out.append(item)
            continue
        content = item.get("content")
        if not isinstance(content, str) or not _PATTERN.search(content):
            out.append(dict(item))
            continue
        new_item = dict(item)
        segments = parse_image_markdown(content, allowed_mime_types=allowed)
        # If the segments are JUST one text entry, keep the str shape
        # for backward compat.
        if len(segments) == 1 and segments[0].get("type") == "text":
            new_item["content"] = segments[0]["text"]
        else:
            new_item["content"] = segments
        out.append(new_item)
    return out


def has_image_markdown(content: str) -> bool:
    """True iff ``content`` contains at least one base64-markdown
    image pattern."""
    if not content:
        return False
    return _PATTERN.search(content) is not None


__all__ = [
    "DEFAULT_ALLOWED_MIME_TYPES",
    "DEFAULT_MAX_DIM",
    "EncodedImage",
    "ImageSegment",
    "TextSegment",
    "encode_image_as_markdown",
    "has_image_markdown",
    "history_to_multimodal",
    "parse_image_markdown",
]
