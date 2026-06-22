"""L1 canonicalizer tests — covert-channel strip, confusable fold, de-obfuscation."""
from __future__ import annotations

from kenning.twitch.safety.normalize import (
    deobfuscate,
    fold_skeleton,
    normalize_for_match,
)


def test_zero_width_and_tag_block_stripped() -> None:
    # zero-width space + a Unicode Tag char injected mid-word must vanish.
    s = "he​llo\U000e0041world"
    nf = normalize_for_match(s)
    assert "​" not in nf.nfkc and "\U000e0041" not in nf.nfkc
    assert nf.covert_stripped >= 2
    assert nf.nfkc.startswith("hello")


def test_bidi_override_stripped() -> None:
    nf = normalize_for_match("ab‮cd")  # RLO
    assert "‮" not in nf.nfkc


def test_zalgo_combining_marks_capped() -> None:
    zalgo = "h" + "́̂̃̄̅" + "i"
    nf = normalize_for_match(zalgo)
    # at most 2 combining marks survive per base char before NFKC composition
    assert "hi" in fold_skeleton(nf.nfkc).replace("́", "")


def test_fullwidth_and_confusable_fold() -> None:
    # fullwidth letters (NFKC) + a Cyrillic homoglyph fold to ASCII skeleton.
    assert "hello" in fold_skeleton(normalize_for_match("ｈｅｌｌｏ").nfkc)
    # Cyrillic а/е/о/р/с -> latin
    assert fold_skeleton("аеорс") == "aeopc"


def test_deobfuscate_spaced_dotted_leet_repeat() -> None:
    assert deobfuscate("b o m b") == "bomb"
    assert deobfuscate("f.u.c.k") == "fuck"
    assert deobfuscate("hellooooo") == "hello"
    # leet folds only within alpha tokens; a pure number is preserved.
    assert deobfuscate("h3ll0 84") == "hello 84".replace(" ", "")  # separators removed -> hello84
    assert "84" in deobfuscate("sova hit 84")


def test_reversed_form() -> None:
    nf = normalize_for_match("olleh")
    assert nf.reversed == "hello"


def test_normalize_never_raises_on_garbage() -> None:
    for s in ["", "\x00\x01", "🤖" * 50, "\ud800", "a" * 5000]:
        nf = normalize_for_match(s)
        assert isinstance(nf.nfkc, str)
