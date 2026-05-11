"""Addressing classifier failure mode:
  - Zero-shot classifier raises  -> default-silent verdict, error logged

Validates the addressing pipeline never crashes on zero-shot failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from ultron.addressing.classifier import (
    AddressingClassifier, AddressingDecision,
)


@pytest.fixture
def classifier(tmp_path):
    """A real AddressingClassifier with the zero-shot mocked. Skips the
    Flan-T5 load so the test runs without GPU."""
    import threading
    c = AddressingClassifier.__new__(AddressingClassifier)
    c.rule_threshold = 0.99  # force fall-through to zero-shot for any utt
    c.default_silent = True
    # 0.0 disables the 2026-05-11 zero-shot min-confidence gate, so
    # existing error-recovery tests keep validating the legacy path.
    c.zero_shot_addressed_min_confidence = 0.0
    c._zero_shot = MagicMock()
    c._recent_turns_provider = None
    c.log_path = tmp_path / "addressing.jsonl"
    c._log_lock = threading.Lock()
    return c


def test_zero_shot_failure_returns_default_silent(
    classifier, errors_log, read_errors,
):
    """Zero-shot raises -> NOT_ADDRESSED verdict + error logged."""
    classifier._zero_shot.classify.side_effect = RuntimeError("model dead")

    verdict = classifier.classify("ambiguous utterance with no rule hit", 5.0)

    assert verdict.decision == AddressingDecision.NOT_ADDRESSED
    assert verdict.source == "default_silent"
    records = read_errors()
    assert len(records) == 1
    rec = records[0]
    assert rec["dependency"] == "addressing_zero_shot"
    assert rec["error_type"] == "AddressingClassifierError"
    assert "default-silent" in rec["recovery"]


def test_zero_shot_failure_with_default_loud_returns_uncertain(
    classifier, errors_log, read_errors,
):
    """When default_silent=False, the fallback is UNCERTAIN, not NOT_ADDRESSED."""
    classifier.default_silent = False
    classifier._zero_shot.classify.side_effect = RuntimeError("model dead")

    verdict = classifier.classify("ambiguous", 5.0)
    assert verdict.decision == AddressingDecision.UNCERTAIN
    records = read_errors()
    assert "uncertain" in records[0]["recovery"]


def test_classifier_subsequent_call_works_after_failure(
    classifier, errors_log, read_errors,
):
    """Single zero-shot failure doesn't poison the next call."""
    # First call fails
    classifier._zero_shot.classify.side_effect = RuntimeError("transient")
    classifier.classify("first", 5.0)

    # Second call succeeds. Zero-shot returns the raw verdict string
    # ("YES" / "NO" / "UNCLEAR"); the classifier maps that to an
    # AddressingDecision.
    classifier._zero_shot.classify.side_effect = None
    classifier._zero_shot.classify.return_value = ("YES", 0.85, 12.0)
    verdict = classifier.classify("second", 5.0)
    assert verdict.decision == AddressingDecision.ADDRESSED
    assert verdict.source == "zero_shot"
