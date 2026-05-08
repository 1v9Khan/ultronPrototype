"""4B optimization plan Stage C — start_llamacpp_server launcher tests.

Tests the pure-Python pieces of ``scripts/start_llamacpp_server.py``:
- ``_build_arg_parser`` — argparse construction, help renders.
- ``_resolve_kwargs`` — args + optional config overlay -> ModelSettings kwargs.

Does NOT test ``main()`` (which loads CUDA DLLs and starts uvicorn).
The launcher is loaded via ``importlib.util`` so the test doesn't pollute
sys.path with the ``scripts/`` directory.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "start_llamacpp_server.py"
)


def _load_launcher():
    spec = importlib.util.spec_from_file_location("start_llamacpp_server", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def launcher():
    return _load_launcher()


# ---------------------------------------------------------------------------
# argparse + help
# ---------------------------------------------------------------------------


def test_help_renders(launcher) -> None:
    parser = launcher._build_arg_parser()
    text = parser.format_help()
    # Spot-check the new Stage C flags appear in --help
    assert "--model-draft" in text
    assert "--draft-num-pred-tokens" in text
    assert "--from-config" in text
    # And that the legacy flags still appear
    assert "--model" in text
    assert "--n-ctx" in text
    assert "--api-key" in text


def test_default_args_back_compat(launcher) -> None:
    """With no args, kwargs match pre-Stage-C behaviour exactly: 9B
    target model, n_ctx=8192, no draft_model field at all."""
    parser = launcher._build_arg_parser()
    args = parser.parse_args([])
    kwargs = launcher._resolve_kwargs(args)
    assert kwargs["model"].endswith("Qwen3.5-9B-Q4_K_M.gguf")
    assert kwargs["n_ctx"] == 8192
    assert kwargs["n_gpu_layers"] == -1
    assert kwargs["flash_attn"] is True
    assert kwargs["type_k"] == 8
    assert kwargs["type_v"] == 8
    assert kwargs["model_alias"] == "qwen3.5-9b-local"
    # Critical: no draft_model leaked into kwargs when --model-draft not set
    assert "draft_model" not in kwargs
    assert "draft_model_num_pred_tokens" not in kwargs


def test_model_draft_flag_attaches_speculative(launcher, tmp_path: Path) -> None:
    target = tmp_path / "target.gguf"
    target.write_bytes(b"x")
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"x")
    parser = launcher._build_arg_parser()
    args = parser.parse_args(
        ["--model", str(target), "--model-draft", str(draft)]
    )
    kwargs = launcher._resolve_kwargs(args)
    assert kwargs["model"] == str(target.resolve())
    assert kwargs["draft_model"] == str(draft.resolve())
    # Default --draft-num-pred-tokens
    assert kwargs["draft_model_num_pred_tokens"] == 8


def test_draft_num_pred_tokens_override(launcher, tmp_path: Path) -> None:
    target = tmp_path / "t.gguf"
    target.write_bytes(b"x")
    draft = tmp_path / "d.gguf"
    draft.write_bytes(b"x")
    parser = launcher._build_arg_parser()
    args = parser.parse_args(
        [
            "--model", str(target),
            "--model-draft", str(draft),
            "--draft-num-pred-tokens", "16",
        ]
    )
    kwargs = launcher._resolve_kwargs(args)
    assert kwargs["draft_model_num_pred_tokens"] == 16


def test_draft_num_pred_tokens_ignored_without_model_draft(launcher) -> None:
    """Specifying --draft-num-pred-tokens but not --model-draft must
    NOT inject a stray draft_model_num_pred_tokens key (avoid noisy
    ModelSettings construction)."""
    parser = launcher._build_arg_parser()
    args = parser.parse_args(["--draft-num-pred-tokens", "16"])
    kwargs = launcher._resolve_kwargs(args)
    assert "draft_model" not in kwargs
    assert "draft_model_num_pred_tokens" not in kwargs


# ---------------------------------------------------------------------------
# Overlay (--from-config) behaviour
# ---------------------------------------------------------------------------


def _argv(*items: str) -> list[str]:
    return list(items)


def _args_with_argv(launcher, argv: list[str]):
    parser = launcher._build_arg_parser()
    args = parser.parse_args(argv)
    args._explicit_argv = argv
    return args


def test_from_config_4b_overlay(launcher) -> None:
    """Overlay simulates a config with preset='qwen3.5-4b': model points
    at the 4B GGUF, n_ctx=16384, draft_model points at 0.8B."""
    argv = ["--from-config"]
    args = _args_with_argv(launcher, argv)
    overlay: dict[str, Any] = {
        "model": "/abs/models/Qwen3.5-4B-Q4_K_M.gguf",
        "n_ctx": 16384,
        "model_draft": "/abs/models/Qwen3.5-0.8B-Q4_K_M.gguf",
    }
    kwargs = launcher._resolve_kwargs(args, overlay=overlay)
    # model + draft come from overlay; n_ctx too
    assert kwargs["model"].endswith("Qwen3.5-4B-Q4_K_M.gguf")
    assert kwargs["n_ctx"] == 16384
    assert kwargs["draft_model"].endswith("Qwen3.5-0.8B-Q4_K_M.gguf")
    assert kwargs["draft_model_num_pred_tokens"] == 8


def test_from_config_9b_overlay_no_draft(launcher) -> None:
    """Overlay with the default 9B preset has no model_draft key — so
    the launcher must NOT enable speculative decoding."""
    argv = ["--from-config"]
    args = _args_with_argv(launcher, argv)
    overlay: dict[str, Any] = {
        "model": "/abs/models/Qwen3.5-9B-Q4_K_M.gguf",
        "n_ctx": 8192,
        # no model_draft
    }
    kwargs = launcher._resolve_kwargs(args, overlay=overlay)
    assert kwargs["model"].endswith("Qwen3.5-9B-Q4_K_M.gguf")
    assert kwargs["n_ctx"] == 8192
    assert "draft_model" not in kwargs


def test_cli_model_overrides_overlay(launcher, tmp_path: Path) -> None:
    """--model on the CLI must win even when --from-config supplies an
    overlay. Useful for ad-hoc swaps without editing YAML."""
    explicit_target = tmp_path / "explicit.gguf"
    explicit_target.write_bytes(b"x")
    argv = ["--from-config", "--model", str(explicit_target)]
    args = _args_with_argv(launcher, argv)
    overlay: dict[str, Any] = {
        "model": "/abs/models/Qwen3.5-4B-Q4_K_M.gguf",
        "n_ctx": 16384,
        "model_draft": "/abs/models/Qwen3.5-0.8B-Q4_K_M.gguf",
    }
    kwargs = launcher._resolve_kwargs(args, overlay=overlay)
    # Explicit --model wins
    assert kwargs["model"] == str(explicit_target.resolve())
    # Overlay still provides n_ctx + model_draft
    assert kwargs["n_ctx"] == 16384
    assert kwargs["draft_model"].endswith("Qwen3.5-0.8B-Q4_K_M.gguf")


def test_cli_n_ctx_overrides_overlay(launcher) -> None:
    argv = ["--from-config", "--n-ctx", "4096"]
    args = _args_with_argv(launcher, argv)
    overlay: dict[str, Any] = {
        "model": "/abs/models/Qwen3.5-4B-Q4_K_M.gguf",
        "n_ctx": 16384,
        "model_draft": "/abs/models/Qwen3.5-0.8B-Q4_K_M.gguf",
    }
    kwargs = launcher._resolve_kwargs(args, overlay=overlay)
    assert kwargs["n_ctx"] == 4096


def test_cli_model_draft_overrides_overlay(launcher, tmp_path: Path) -> None:
    explicit_draft = tmp_path / "explicit_draft.gguf"
    explicit_draft.write_bytes(b"x")
    argv = ["--from-config", "--model-draft", str(explicit_draft)]
    args = _args_with_argv(launcher, argv)
    overlay: dict[str, Any] = {
        "model": "/abs/models/Qwen3.5-4B-Q4_K_M.gguf",
        "n_ctx": 16384,
        "model_draft": "/abs/models/Qwen3.5-0.8B-Q4_K_M.gguf",
    }
    kwargs = launcher._resolve_kwargs(args, overlay=overlay)
    assert kwargs["draft_model"] == str(explicit_draft.resolve())


def test_chat_format_passed_through(launcher) -> None:
    parser = launcher._build_arg_parser()
    args = parser.parse_args(["--chat-format", "chatml"])
    kwargs = launcher._resolve_kwargs(args)
    assert kwargs["chat_format"] == "chatml"


def test_chat_format_omitted_when_default(launcher) -> None:
    parser = launcher._build_arg_parser()
    args = parser.parse_args([])
    kwargs = launcher._resolve_kwargs(args)
    # Omitted (so ModelSettings uses GGUF metadata template)
    assert "chat_format" not in kwargs


def test_no_flash_attn_disables(launcher) -> None:
    parser = launcher._build_arg_parser()
    args = parser.parse_args(["--no-flash-attn"])
    kwargs = launcher._resolve_kwargs(args)
    assert kwargs["flash_attn"] is False
