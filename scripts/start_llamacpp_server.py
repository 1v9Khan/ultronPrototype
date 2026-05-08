"""Start the shared llama-cpp-server for Ultron + OpenClaw.

This is the OpenAI-compatible HTTP server that exposes the active LLM
GGUF to both consumers, replacing the in-process llama-cpp-python load
currently used by Ultron's voice pipeline. The OpenClaw Gateway
connects via ``@openclaw/lmstudio-provider`` configured with
``baseUrl: http://127.0.0.1:8765`` (see ``~/.openclaw/openclaw.json``).

Why a Python wrapper rather than ``python -m llama_cpp.server``:
``llama_cpp`` needs the bundled torch CUDA DLL directory on PATH on
Windows. ``ultron.__init__`` adds it automatically; importing ultron
first ensures the load order is right. Running ``python -m
llama_cpp.server`` directly fails to find ``llama.dll``.

Run from the main checkout (where models/ lives):

    cd C:\\STC\\ultronPrototype
    .venv\\Scripts\\python.exe scripts/start_llamacpp_server.py

The server stays in the foreground until Ctrl+C. Send Ctrl+C cleanly to
unload the model from VRAM. The default ``--api_key local-ultron``
matches the OpenClaw config; rotate later if hardening for non-loopback.

Flags mirror Ultron voice-pipeline llama-cpp config (see
[config.yaml:llm](../config.yaml)) so character + VRAM behaviour are
preserved when we switch the voice path off in-process loading.

4B optimization plan Stage C — speculative decoding:
``--model-draft <path>`` enables speculative decoding by pairing the
target model with a small draft model (the 4B + 0.8B preset
recommended in [docs/4b_optimization_plan.md](../docs/4b_optimization_plan.md)).
``--draft-num-pred-tokens N`` controls how many tokens the draft
predicts ahead before the target verifies (default 8). Note: the
upstream ``llama.cpp`` CLI exposes both ``--draft-max`` and
``--draft-min``, but ``llama-cpp-python==0.3.22`` only surfaces a
single combined parameter (mapped here from ``--draft-num-pred-tokens``).

``--from-config`` loads the values from ``config.yaml:llm`` so a
``preset: "qwen3.5-4b"`` config switches the launched model + draft +
n_ctx without command-line flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# CLI parsing + kwargs resolution (pure Python — no llama_cpp / uvicorn import,
# so this is independently testable without GPU or DLL loading).
# ---------------------------------------------------------------------------


def _default_model_path() -> Path:
    return Path("models") / "Qwen3.5-9B-Q4_K_M.gguf"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default=str(_default_model_path()),
        help="Path to the target GGUF (default: ./models/Qwen3.5-9B-Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--model-draft",
        dest="model_draft",
        type=str,
        default=None,
        help=(
            "Optional path to a draft GGUF for speculative decoding. "
            "Default: None (no spec decoding). The 4B optimization plan "
            "pairs Qwen3.5-4B (target) with Qwen3.5-0.8B (draft). The "
            "draft must share the target's tokenizer/vocab — verified by "
            "Stage B's vocab_only check (n_vocab=248320 for both)."
        ),
    )
    parser.add_argument(
        "--draft-num-pred-tokens",
        dest="draft_num_pred_tokens",
        type=int,
        default=8,
        help=(
            "How many tokens the draft predicts before the target "
            "verifies (default 8; recipe says 8 for conversational, "
            "higher for predictable code output). Maps to "
            "llama-cpp-python's draft_model_num_pred_tokens. "
            "Ignored when --model-draft is unset."
        ),
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help=(
            "Listen port (default: 8765). 8080 is commonly in a Windows "
            "Hyper-V / HNS reserved range — bind fails with WinError 13."
        ),
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--api-key", dest="api_key", type=str, default="local-ultron")
    parser.add_argument(
        "--model-alias",
        dest="model_alias",
        type=str,
        default="qwen3.5-9b-local",
        help="OpenAI-compat model id consumers use in API calls",
    )
    parser.add_argument("--n-ctx", dest="n_ctx", type=int, default=8192)
    parser.add_argument("--n-gpu-layers", dest="n_gpu_layers", type=int, default=-1)
    parser.add_argument(
        "--no-flash-attn", dest="flash_attn", action="store_false",
        help="Disable flash-attention (default: enabled, mirrors voice config)",
    )
    parser.add_argument(
        "--type-k", dest="type_k", type=int, default=8,
        help="GGML KV cache type for K (default: 8 = Q8_0; matches voice)",
    )
    parser.add_argument(
        "--type-v", dest="type_v", type=int, default=8,
        help="GGML KV cache type for V (default: 8 = Q8_0; matches voice)",
    )
    parser.add_argument(
        "--chat-format", dest="chat_format", type=str, default=None,
        help=(
            "Override the chat format. Default: None (use the GGUF's metadata "
            "template — Qwen3.5's full template with <think> scaffolding). "
            "Pass 'chatml' to bypass <think> handling for OpenClaw "
            "compatibility (the voice path uses the GGUF metadata template "
            "directly via in-process llama-cpp-python and is unaffected). "
            "Other values: see llama_cpp.llama_chat_format.LlamaChatCompletionHandlerRegistry."
        ),
    )
    parser.add_argument(
        "--from-config",
        dest="from_config",
        action="store_true",
        help=(
            "Load model / draft / n_ctx from config.yaml:llm (resolves the "
            "active preset via LLMConfig). CLI flags still override what "
            "the config provides — useful for ad-hoc swaps without editing "
            "YAML. The 4B plan flips 'preset: qwen3.5-4b' in config.yaml; "
            "running with --from-config picks up the change automatically."
        ),
    )
    parser.set_defaults(flash_attn=True)
    return parser


def _config_overlay() -> dict[str, Any]:
    """Read ``config.yaml:llm`` and return launcher-relevant fields.

    Imports are deferred so the launcher's CLI parsing + tests don't
    require the ultron package to be importable.
    """
    # Make the worktree's src/ importable from any cwd (the launcher is
    # often run from main checkout).
    here = Path(__file__).resolve().parent.parent
    src = here / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from ultron.config import get_config, resolve_path

    cfg = get_config().llm
    overlay: dict[str, Any] = {
        "model": str(resolve_path(cfg.model_path)),
        "n_ctx": cfg.n_ctx,
    }
    if cfg.draft_model_path:
        overlay["model_draft"] = str(resolve_path(cfg.draft_model_path))
    return overlay


def _resolve_kwargs(args: argparse.Namespace, *, overlay: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Build the ModelSettings kwargs dict.

    The overlay (from ``--from-config``) provides defaults that the
    CLI flags can still override. Tests inject ``overlay`` directly to
    avoid importing the ultron package.
    """
    parser = _build_arg_parser()
    cli_set: set[str] = set()
    # Walk argparse to learn which dest names exist + which were
    # explicitly supplied on the command line. ``args`` always carries
    # all destinations (filled with defaults), so we need a separate
    # pass to know which were *explicit* — argparse doesn't track that.
    # Approach: re-parse with no defaults, see which actions hit.
    if overlay:
        no_default_parser = argparse.ArgumentParser(add_help=False)
        for action in parser._actions:
            if not action.option_strings or action.dest in ("help",):
                continue
            kw: dict[str, Any] = {}
            if isinstance(action, argparse._StoreFalseAction):
                kw["action"] = "store_false"
                kw["default"] = None
            elif isinstance(action, argparse._StoreTrueAction):
                kw["action"] = "store_true"
                kw["default"] = None
            else:
                kw["type"] = action.type
                kw["default"] = None
            no_default_parser.add_argument(*action.option_strings, dest=action.dest, **kw)
        # Reconstruct argv from sys.argv at parse time? Tests pass
        # argv directly; main() uses sys.argv. We track via the
        # presence of ``_explicit_argv`` attached to args by the
        # caller (parse_args does not).
        explicit_argv = getattr(args, "_explicit_argv", sys.argv[1:])
        no_default_args, _ = no_default_parser.parse_known_args(explicit_argv)
        for k, v in vars(no_default_args).items():
            if v is not None:
                cli_set.add(k)

    model = args.model
    n_ctx = args.n_ctx
    model_draft = args.model_draft

    if overlay:
        if "model" not in cli_set and "model" in overlay:
            model = overlay["model"]
        if "n_ctx" not in cli_set and "n_ctx" in overlay:
            n_ctx = overlay["n_ctx"]
        if "model_draft" not in cli_set and "model_draft" in overlay:
            model_draft = overlay["model_draft"]

    kwargs: dict[str, Any] = dict(
        model=str(Path(model).resolve()) if model else model,
        model_alias=args.model_alias,
        n_ctx=n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        flash_attn=args.flash_attn,
        type_k=args.type_k,
        type_v=args.type_v,
    )
    if args.chat_format:
        kwargs["chat_format"] = args.chat_format
    if model_draft:
        kwargs["draft_model"] = str(Path(model_draft).resolve())
        kwargs["draft_model_num_pred_tokens"] = args.draft_num_pred_tokens
    return kwargs


# ---------------------------------------------------------------------------
# Entry point (DLL-loading + uvicorn live here, not imported by tests)
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    args._explicit_argv = list(argv) if argv is not None else sys.argv[1:]

    overlay = _config_overlay() if args.from_config else None
    kwargs = _resolve_kwargs(args, overlay=overlay)

    model_path = Path(kwargs["model"])
    if not model_path.is_file():
        sys.stderr.write(
            f"error: model file not found at {model_path}\n"
            f"hint: run from C:\\STC\\ultronPrototype (main checkout) "
            f"so the relative ./models path resolves, or pass --model "
            f"with an absolute path.\n"
        )
        return 2
    if "draft_model" in kwargs and not Path(kwargs["draft_model"]).is_file():
        sys.stderr.write(
            f"error: draft model file not found at {kwargs['draft_model']}\n"
            f"hint: run scripts/download_models.py to fetch the draft GGUF.\n"
        )
        return 2

    # Crucial: ultron.__init__ adds <venv>/Lib/site-packages/torch/lib to
    # PATH so llama-cpp-python can find its bundled CUDA DLLs.
    import ultron  # noqa: F401

    from llama_cpp.server.app import create_app
    from llama_cpp.server.settings import (
        ConfigFileSettings,
        ModelSettings,
        ServerSettings,
    )
    import uvicorn

    server_settings = ServerSettings(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    model_settings = [ModelSettings(**kwargs)]
    config = ConfigFileSettings(  # noqa: F841 — kept for future config-file mode
        host=server_settings.host,
        port=server_settings.port,
        api_key=server_settings.api_key,
        models=model_settings,
    )

    spec_msg = ""
    if "draft_model" in kwargs:
        spec_msg = (
            f" speculative=on draft={Path(kwargs['draft_model']).name} "
            f"draft_num_pred={kwargs['draft_model_num_pred_tokens']}"
        )

    sys.stderr.write(
        f"[start_llamacpp_server] starting on http://{args.host}:{args.port} "
        f"model={Path(kwargs['model']).name} model_alias={args.model_alias} "
        f"n_ctx={kwargs['n_ctx']} kv_q8=true flash_attn={args.flash_attn} "
        f"chat_format={args.chat_format or 'gguf-default'}{spec_msg}\n"
    )

    app = create_app(server_settings=server_settings, model_settings=model_settings)
    uvicorn.run(
        app,
        host=server_settings.host,
        port=server_settings.port,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
