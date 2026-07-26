"""Compatibility shims for legacy RVC dependencies."""

from __future__ import annotations

import dataclasses
import sys
import types

from kenning.utils.fairseq_compat import patch_fairseq_dataclasses


def test_patch_allows_legacy_nested_dataclass_default():
    patch_fairseq_dataclasses()

    namespace = _module_namespace("fairseq.fake")
    exec(
        """
from dataclasses import dataclass

@dataclass
class Inner:
    value: int = 1

@dataclass
class Outer:
    inner: Inner = Inner()
""",
        namespace,
    )

    outer = namespace["Outer"]()

    assert outer.inner.value == 1
    assert namespace["Outer"].__dataclass_fields__["inner"].default is not dataclasses.MISSING


def test_patch_allows_legacy_field_wrapped_dataclass_default():
    patch_fairseq_dataclasses()

    namespace = _module_namespace("hydra.fake")
    exec(
        """
from dataclasses import dataclass, field

@dataclass
class Inner:
    value: int = 2

@dataclass
class Outer:
    inner: Inner = field(default=Inner())
""",
        namespace,
    )

    outer = namespace["Outer"]()

    assert outer.inner.value == 2
    assert namespace["Outer"].__dataclass_fields__["inner"].default is not dataclasses.MISSING


def _module_namespace(name: str) -> dict[str, object]:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module.__dict__
