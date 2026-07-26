"""Runtime compatibility for Fairseq/Hydra 1.x on Python 3.11+."""

from __future__ import annotations

import dataclasses
import traceback
from typing import Any


def patch_fairseq_dataclasses() -> None:
    """Allow legacy nested dataclass defaults on Python 3.11.

    Fairseq 0.12 declares fields such as ``common: CommonConfig =
    CommonConfig()``. Python 3.11 rejects those mutable defaults at import time
    by treating unhashable defaults as mutable. Fairseq also imports Hydra 1.0,
    which uses the same pattern. RVC only needs this stack for HuBERT checkpoint
    loading, so a narrow runtime patch is less invasive than editing installed
    site-packages.
    """
    if getattr(dataclasses, "_kenning_fairseq_patch", False):
        return

    original_get_field = dataclasses._get_field  # type: ignore[attr-defined]

    def patched_get_field(
        cls: type[Any],
        field_name: str,
        field_type: Any,
        default_kw_only: bool,
    ):
        try:
            return original_get_field(cls, field_name, field_type, default_kw_only)
        except ValueError as e:
            default = getattr(cls, field_name, dataclasses.MISSING)
            legacy_default = _legacy_dataclass_default(cls, default)
            if legacy_default is not None:
                # Preserve the actual default object. Fairseq's Hydra
                # initializer reads __dataclass_fields__[name].default
                # directly, so default_factory would break config registration.
                if legacy_default.__class__.__hash__ is None:
                    legacy_default.__class__.__hash__ = object.__hash__
                return original_get_field(
                    cls, field_name, field_type, default_kw_only
                )
            raise e

    dataclasses._get_field = patched_get_field  # type: ignore[attr-defined]
    dataclasses._kenning_fairseq_patch = True  # type: ignore[attr-defined]


def patch_torch_load_for_fairseq() -> None:
    """Let Fairseq load legacy trusted HuBERT checkpoints under PyTorch 2.6.

    PyTorch 2.6 defaults ``torch.load`` to ``weights_only=True``. Fairseq 0.12
    checkpoints include Fairseq classes, so their own loader must opt into the
    old behavior. This wrapper only changes calls made from Fairseq's
    ``checkpoint_utils.py`` and leaves every other ``torch.load`` call alone.
    """
    import torch

    if getattr(torch.load, "_kenning_fairseq_patch", False):
        return

    original_load = torch.load

    def patched_load(*args: Any, **kwargs: Any):
        if "weights_only" not in kwargs and _called_from_fairseq_checkpoint_utils():
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    patched_load._kenning_fairseq_patch = True  # type: ignore[attr-defined]
    patched_load._kenning_original_load = original_load  # type: ignore[attr-defined]
    torch.load = patched_load


def _legacy_dataclass_default(cls: type[Any], default: Any) -> Any | None:
    if isinstance(default, dataclasses.Field):
        default = default.default
    if _is_legacy_fairseq_default(cls, default):
        return default
    return None


def _is_legacy_fairseq_default(cls: type[Any], default: Any) -> bool:
    cls_module = getattr(cls, "__module__", "")
    default_module = getattr(default.__class__, "__module__", "")
    legacy_roots = ("fairseq.", "hydra.")
    return (
        cls_module.startswith(legacy_roots)
        and default_module.startswith(legacy_roots)
        and hasattr(default, "__dataclass_fields__")
    )


def _called_from_fairseq_checkpoint_utils() -> bool:
    for frame in traceback.extract_stack(limit=12):
        path = frame.filename.replace("\\", "/")
        if path.endswith("/fairseq/checkpoint_utils.py"):
            return True
    return False
