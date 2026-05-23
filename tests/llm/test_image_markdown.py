"""Tests for image-as-base64-markdown encoding + parsing (catalog T18)."""

from __future__ import annotations

import base64
import io
import re

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from ultron.llm.image_markdown import (
    DEFAULT_ALLOWED_MIME_TYPES,
    DEFAULT_MAX_DIM,
    EncodedImage,
    encode_image_as_markdown,
    has_image_markdown,
    history_to_multimodal,
    parse_image_markdown,
)


def _png(w: int = 32, h: int = 32, color: tuple = (10, 20, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h), color).save(out, format="PNG")
    return out.getvalue()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_default_allowed_mime_types_matches_swe_agent():
    assert DEFAULT_ALLOWED_MIME_TYPES == frozenset(
        {"image/png", "image/jpeg", "image/webp"}
    )


def test_default_max_dim_reasonable():
    assert DEFAULT_MAX_DIM > 0


# ---------------------------------------------------------------------------
# encode_image_as_markdown
# ---------------------------------------------------------------------------


def test_encode_empty_bytes_raises():
    with pytest.raises(ValueError):
        encode_image_as_markdown(b"")


def test_encode_disallowed_mime_raises():
    with pytest.raises(ValueError):
        encode_image_as_markdown(_png(), mime_type="image/gif")


def test_encode_emits_data_url_markdown():
    out = encode_image_as_markdown(_png(), alt_text="screenshot")
    assert isinstance(out, EncodedImage)
    assert out.markdown.startswith("![screenshot](data:image/png;base64,")
    assert out.markdown.endswith(")")
    assert out.as_data_url().startswith("data:image/png;base64,")


def test_encoded_base64_is_valid():
    out = encode_image_as_markdown(_png())
    decoded = base64.b64decode(out.base64_data)
    # Decodes to a real PNG signature.
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_resizes_large_image_when_max_dim_set():
    big = _png(w=4000, h=2000)
    out = encode_image_as_markdown(big, max_dim=512)
    # The encoded payload should be much smaller than the original.
    assert out.encoded_bytes < out.original_bytes * 0.7
    # And the actual image size should be <= max_dim.
    decoded = base64.b64decode(out.base64_data)
    with Image.open(io.BytesIO(decoded)) as im:
        assert max(im.size) <= 512


def test_encode_no_resize_when_max_dim_none():
    img = _png(w=4000, h=2000)
    out = encode_image_as_markdown(img, max_dim=None)
    # When resize is disabled, the encoded image keeps its original
    # dimensions.
    decoded = base64.b64decode(out.base64_data)
    with Image.open(io.BytesIO(decoded)) as im:
        assert im.size == (4000, 2000)


def test_encode_jpeg_mode_conversion():
    # JPEG can't encode RGBA; ensure the resizer converts.
    rgba = io.BytesIO()
    Image.new("RGBA", (3000, 3000), (255, 0, 0, 128)).save(rgba, format="PNG")
    rgba_bytes = rgba.getvalue()
    # Encode as JPEG -- the resize path converts to RGB.
    out = encode_image_as_markdown(
        rgba_bytes, mime_type="image/jpeg", max_dim=200
    )
    assert out.mime_type == "image/jpeg"
    decoded = base64.b64decode(out.base64_data)
    # Header signature should be JPEG (\xff\xd8\xff).
    assert decoded[:3] == b"\xff\xd8\xff"


# ---------------------------------------------------------------------------
# parse_image_markdown
# ---------------------------------------------------------------------------


def test_parse_empty_returns_single_empty_text():
    out = parse_image_markdown("")
    assert out == [{"type": "text", "text": ""}]


def test_parse_no_images_returns_single_text():
    out = parse_image_markdown("just plain text")
    assert out == [{"type": "text", "text": "just plain text"}]


def test_parse_single_image_splits_correctly():
    md = encode_image_as_markdown(_png()).markdown
    content = f"before {md} after"
    out = parse_image_markdown(content)
    # Three segments: text "before ", image_url, text " after"
    assert len(out) == 3
    assert out[0]["type"] == "text"
    assert "before" in out[0]["text"]
    assert out[1]["type"] == "image_url"
    assert out[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert out[2]["type"] == "text"
    assert "after" in out[2]["text"]


def test_parse_multiple_images():
    md1 = encode_image_as_markdown(_png(color=(255, 0, 0))).markdown
    md2 = encode_image_as_markdown(_png(color=(0, 255, 0))).markdown
    content = f"a {md1} b {md2} c"
    out = parse_image_markdown(content)
    images = [s for s in out if s["type"] == "image_url"]
    assert len(images) == 2


def test_parse_disallowed_mime_passes_through_as_text():
    # Forge an unsupported MIME embedded in markdown.
    content = "before ![alt](data:image/bmp;base64,QkFE) after"
    out = parse_image_markdown(content)
    # No image segments; the disallowed MIME survives as text.
    assert all(s["type"] == "text" for s in out)
    combined = "".join(s["text"] for s in out)
    assert "data:image/bmp" in combined


def test_parse_image_jpg_normalised_to_jpeg():
    # An "image/jpg" MIME (common typo) gets normalised to "image/jpeg"
    # per SWE-Agent.
    content = "x ![](data:image/jpg;base64,ABC) y"
    out = parse_image_markdown(content)
    images = [s for s in out if s["type"] == "image_url"]
    assert len(images) == 1
    assert "image/jpeg" in images[0]["image_url"]["url"]


def test_parse_alt_text_preserved_in_url_url_only_format():
    """The alt text doesn't appear in the multimodal segment (LLM
    API doesn't expose it), but it's still preserved in the URL
    string structure -- no info loss."""
    md = encode_image_as_markdown(_png(), alt_text="my-screenshot").markdown
    out = parse_image_markdown(md)
    # Just the single image segment.
    assert len(out) == 1
    assert out[0]["type"] == "image_url"


def test_parse_collapses_consecutive_text_segments():
    """Repeated text between images shouldn't fragment."""
    md = encode_image_as_markdown(_png()).markdown
    content = f"hello {md} world"
    out = parse_image_markdown(content)
    text_segments = [s for s in out if s["type"] == "text"]
    # 2 text segments around the image.
    assert len(text_segments) == 2


# ---------------------------------------------------------------------------
# has_image_markdown
# ---------------------------------------------------------------------------


def test_has_image_markdown_true():
    md = encode_image_as_markdown(_png()).markdown
    assert has_image_markdown(md) is True


def test_has_image_markdown_false():
    assert has_image_markdown("plain text") is False
    assert has_image_markdown("") is False
    assert has_image_markdown("![alt](http://example.com/img.png)") is False  # not data URL


# ---------------------------------------------------------------------------
# history_to_multimodal
# ---------------------------------------------------------------------------


def test_history_to_multimodal_passes_through_text_only_items():
    history = [
        {"role": "user", "content": "plain text only"},
        {"role": "assistant", "content": "no images here"},
    ]
    out = history_to_multimodal(history)
    assert len(out) == 2
    for orig, new in zip(history, out):
        assert orig["content"] == new["content"]


def test_history_to_multimodal_rewrites_image_items():
    md = encode_image_as_markdown(_png()).markdown
    history = [
        {"role": "user", "content": f"check this: {md}"},
    ]
    out = history_to_multimodal(history)
    # Content should now be a list with text + image_url segments.
    assert isinstance(out[0]["content"], list)
    types = [s["type"] for s in out[0]["content"]]
    assert "image_url" in types


def test_history_to_multimodal_does_not_mutate_input():
    md = encode_image_as_markdown(_png()).markdown
    history = [{"role": "user", "content": f"{md}"}]
    original_content = history[0]["content"]
    history_to_multimodal(history)
    assert history[0]["content"] == original_content


def test_history_to_multimodal_handles_non_dict_items():
    """Non-dict items pass through unchanged."""
    history = [{"role": "user", "content": "ok"}, "garbage", None]
    out = history_to_multimodal(history)
    assert out[1] == "garbage"
    assert out[2] is None


def test_history_to_multimodal_handles_non_string_content():
    """Already-multimodal content (list) passes through unchanged."""
    history = [
        {"role": "user", "content": [{"type": "text", "text": "pre-converted"}]},
    ]
    out = history_to_multimodal(history)
    assert out[0]["content"] == history[0]["content"]
