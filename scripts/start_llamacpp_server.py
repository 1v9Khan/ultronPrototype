"""Start the shared llama-cpp-server for Ultron + OpenClaw.

This is the OpenAI-compatible HTTP server that exposes
``models/Qwen3.5-9B-Q4_K_M.gguf`` to both consumers, replacing the
in-process llama-cpp-python load currently used by Ultron's voice
pipeline. The OpenClaw Gateway connects via ``@openclaw/lmstudio-provider``
configured with ``baseUrl: http://127.0.0.1:8080`` (see
``~/.openclaw/openclaw.json``).

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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default=str(_default_model_path()),
        help="Path to the Qwen GGUF (default: ./models/Qwen3.5-9B-Q4_K_M.gguf)",
    )
    parser.add_argument("--port", type=int, default=8080)
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
    parser.set_defaults(flash_attn=True)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        sys.stderr.write(
            f"error: model file not found at {model_path}\n"
            f"hint: run from C:\\STC\\ultronPrototype (main checkout) "
            f"so the relative ./models path resolves, or pass --model "
            f"with an absolute path.\n"
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
    model_settings = [
        ModelSettings(
            model=str(model_path),
            model_alias=args.model_alias,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            flash_attn=args.flash_attn,
            type_k=args.type_k,
            type_v=args.type_v,
        ),
    ]
    config = ConfigFileSettings(
        host=server_settings.host,
        port=server_settings.port,
        api_key=server_settings.api_key,
        models=model_settings,
    )

    sys.stderr.write(
        f"[start_llamacpp_server] starting on http://{args.host}:{args.port} "
        f"model_alias={args.model_alias} kv_q8=true flash_attn={args.flash_attn}\n"
    )

    app = create_app(server_settings=server_settings, model_settings=model_settings)
    uvicorn.run(
        app,
        host=server_settings.host,
        port=server_settings.port,
        log_level="warning",
    )
    return 0


def _default_model_path() -> Path:
    return Path("models") / "Qwen3.5-9B-Q4_K_M.gguf"


if __name__ == "__main__":
    raise SystemExit(main())
