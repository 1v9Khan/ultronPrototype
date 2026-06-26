"""Pure planning for the Twitch loopback-sidecar auto-spawn (anticheat: stdlib only).

The orchestrator calls :func:`plan_sidecars` to decide WHICH sidecars to launch and
with WHAT env, then Popens each spec itself (mirroring ``_start_embedder_sidecar``).
The planning is split out here so it is fully unit-testable WITHOUT spawning a
process: a test passes a tiny ``tcfg`` stand-in (or the real ``TwitchConfig``) plus
injectable ``path_exists``/``expanduser`` and asserts the returned specs.

Sidecars (each a SEPARATE process in :data:`kenning.subprocess.sidecar_lock`'s role
registry, with a parent-death deadman + exclusive port bind):
  * ``twitch_read``   — ALWAYS (when enabled): the EventSub chat/redeem reader (8773).
  * ``twitch_guard``  — only when a guard GGUF path is configured AND present (8774).
                        Chat-reply is fail-CLOSED on the guard, so no path => no guard
                        => the chat-reply runtime simply never enables.
  * ``twitch_helper`` — only when ``twitch.helper.enabled`` AND its GGUF is present (8776).

ANTICHEAT (BR-P1): this module imports ONLY stdlib (``os``/``dataclasses``/
``urllib.parse``). It is imported by the orchestrator ONLY inside the
``twitch.enabled`` gate, so a flag-OFF boot never loads it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

__all__ = ["SidecarSpec", "plan_sidecars"]


@dataclass(frozen=True)
class SidecarSpec:
    """One sidecar to spawn. ``env`` is merged OVER ``os.environ`` by the caller
    (plus ``KENNING_TWITCH_PARENT_PID``); ``script`` is repo-root-relative."""

    role: str          # sidecar_lock role: twitch_read | twitch_guard | twitch_helper
    script: str        # repo-relative launcher path
    port: int          # loopback port (argv[1] to the script)
    env: dict          # extra env vars for this sidecar


def _port_of(endpoint: object, default: int) -> int:
    """Parse the port out of a loopback endpoint URL (``http://127.0.0.1:8773``).
    Fail-safe: anything unparseable falls back to ``default``."""
    try:
        port = urlsplit(str(endpoint or "")).port
        return int(port) if port else int(default)
    except (ValueError, TypeError):
        return int(default)


def plan_sidecars(
    tcfg: object,
    *,
    path_exists: Callable[[str], bool] = os.path.exists,
    expanduser: Callable[[str], str] = os.path.expanduser,
) -> list[SidecarSpec]:
    """Return the ordered list of sidecar specs to spawn for ``tcfg`` (the
    ``TwitchConfig``). Caller MUST have already confirmed ``tcfg.enabled``; this
    function does not re-check the master gate. Pure + side-effect-free."""
    specs: list[SidecarSpec] = []
    auth = getattr(tcfg, "auth", None)
    safety = getattr(tcfg, "safety", None)
    economy = getattr(tcfg, "economy", None)
    helper = getattr(tcfg, "helper", None)
    moderation = getattr(tcfg, "moderation", None)
    raid = getattr(tcfg, "raid", None)
    client_id = str(getattr(auth, "client_id", "") or "")
    broadcaster_login = str(getattr(auth, "broadcaster_login", "") or "")
    bot_login = str(getattr(auth, "bot_login", "") or "")
    broadcaster_token_path = expanduser(str(getattr(auth, "token_path", "~/.kenning/twitch.json")))

    # --- READ sidecar: chat (+ optional channel-point redeems). Always. ---
    read_port = _port_of(getattr(tcfg, "read_sidecar_endpoint", ""), 8773)
    specs.append(SidecarSpec(
        role="twitch_read",
        script="scripts/twitch_read_sidecar.py",
        port=read_port,
        env={
            "KENNING_TWITCH_READ_PORT": str(read_port),
            "KENNING_TWITCH_CLIENT_ID": str(getattr(auth, "client_id", "") or ""),
            "KENNING_TWITCH_BROADCASTER_LOGIN": str(getattr(auth, "broadcaster_login", "") or ""),
            "KENNING_TWITCH_BOT_LOGIN": str(getattr(auth, "bot_login", "") or ""),
            "KENNING_TWITCH_BOT_TOKEN_PATH": expanduser(
                str(getattr(auth, "bot_token_path", "~/.kenning/twitch_bot.json"))),
            "KENNING_TWITCH_BROADCASTER_TOKEN_PATH": expanduser(
                str(getattr(auth, "token_path", "~/.kenning/twitch.json"))),
            "KENNING_TWITCH_SUBSCRIBE_REDEEMS": "1" if getattr(economy, "enabled", False) else "0",
            # channel.raid rides the SAME isolated broadcaster-token session as
            # redeems; subscribe when raid handling is enabled (default ON).
            "KENNING_TWITCH_SUBSCRIBE_RAIDS": "1" if getattr(raid, "enabled", False) else "0",
        },
    ))

    # --- GUARD sidecar: only when a GGUF path is configured AND present. ---
    guard_path = getattr(safety, "guard_model_path", None)
    if guard_path:
        gp = expanduser(str(guard_path))
        if path_exists(gp):
            guard_port = _port_of(getattr(safety, "guard_endpoint", ""), 8774)
            specs.append(SidecarSpec(
                role="twitch_guard",
                script="scripts/twitch_guard_sidecar.py",
                port=guard_port,
                env={
                    "KENNING_TWITCH_GUARD_PORT": str(guard_port),
                    "KENNING_TWITCH_GUARD_MODEL": gp,
                    "KENNING_TWITCH_GUARD_FAMILY": str(getattr(safety, "guard_family", "llama-guard")),
                    # 2026-06-24 VRAM: keep the guard OFF the GPU by default (it's a
                    # second llama.cpp process => its own CUDA context). CPU + a
                    # thread cap; latency-tolerant chat moderation. Configurable.
                    "KENNING_TWITCH_GUARD_GPU_LAYERS": str(int(getattr(safety, "guard_gpu_layers", 0))),
                    "KENNING_TWITCH_GUARD_THREADS": str(int(getattr(safety, "guard_threads", 6))),
                },
            ))

    # --- HELPER sidecar: only when enabled AND its GGUF is present. ---
    if getattr(helper, "enabled", False):
        hpath = getattr(helper, "model_path", None)
        if hpath:
            hp = expanduser(str(hpath))
            if path_exists(hp):
                hport = int(getattr(helper, "port", 8776) or 8776)
                specs.append(SidecarSpec(
                    role="twitch_helper",
                    script="scripts/twitch_helper_sidecar.py",
                    port=hport,
                    env={
                        "KENNING_TWITCH_HELPER_PORT": str(hport),
                        "KENNING_TWITCH_HELPER_MODEL": hp,
                    },
                ))

    # --- WRITE/Helix moderation sidecar: only when voice moderation is enabled
    # AND we have the creds to resolve the broadcaster id (client id + login). It
    # holds the broadcaster write token in a SEPARATE process (Twitch I/O off the
    # anticheat-pinned path) and serves the propose/confirm moderation API. ---
    if getattr(moderation, "voice_commands_enabled", False) and client_id and broadcaster_login:
        write_port = _port_of(getattr(tcfg, "write_sidecar_endpoint", ""), 8777)
        read_port = specs[0].port  # the read sidecar (always specs[0]) for the roster source
        specs.append(SidecarSpec(
            role="twitch_write",
            script="scripts/twitch_write_sidecar.py",
            port=write_port,
            env={
                "KENNING_TWITCH_WRITE_PORT": str(write_port),
                "KENNING_TWITCH_CLIENT_ID": client_id,
                "KENNING_TWITCH_BROADCASTER_LOGIN": broadcaster_login,
                "KENNING_TWITCH_BOT_LOGIN": bot_login,
                "KENNING_TWITCH_BROADCASTER_TOKEN_PATH": broadcaster_token_path,
                "KENNING_TWITCH_BOT_TOKEN_PATH": expanduser(
                    str(getattr(auth, "bot_token_path", "~/.kenning/twitch_bot.json"))),
                "KENNING_TWITCH_READ_ENDPOINT": f"http://127.0.0.1:{read_port}",
                "KENNING_TWITCH_MOD_REQUIRE_CONFIRM":
                    "1" if getattr(moderation, "require_readback_confirm", True) else "0",
                "KENNING_TWITCH_MOD_BREAKER_LIMIT":
                    str(int(getattr(moderation, "mass_action_limit_per_60s", 0) or 0)),
            },
        ))

    return specs
