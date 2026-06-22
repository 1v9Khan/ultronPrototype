"""S0 — Twitch config defaults + the anticheat byte-identical invariant.

The whole Twitch capability is flag-gated default-OFF. These tests pin:
  1. every switch defaults OFF / safe (so a fresh config never enables anything);
  2. the schema is ``extra=forbid`` (typo'd keys are rejected, not silently kept);
  3. importing ``kenning.config`` does NOT pull in the ``kenning.twitch`` package;
  4. importing the ``kenning.twitch`` package imports NONE of the forbidden
     network/ML libraries (it is voice-process-safe by construction).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pydantic
import pytest

from kenning.config import KenningConfig, TwitchConfig


def test_twitch_master_and_subswitches_default_off() -> None:
    t = KenningConfig().twitch
    assert t.enabled is False
    assert t.chat.reply_enabled is False
    assert t.chat.allow_during_ranked is False
    assert t.economy.enabled is False
    assert t.economy.lose_all_segment_enabled is False
    assert t.economy.transfers_enabled is False
    assert t.overlay.enabled is False
    assert t.overlay.obs_websocket_enabled is False
    assert t.speak_to_team.enabled is False
    assert t.helper.enabled is False


def test_twitch_safety_defaults_are_fail_closed() -> None:
    s = KenningConfig().twitch.safety
    # The guard model is REQUIRED and the stack fails CLOSED by default.
    assert s.guard_required is True
    assert s.fail_closed is True
    assert s.phoneme_gate_enabled is True
    assert s.asr_backstop_enabled is True
    m = KenningConfig().twitch.moderation
    assert m.protect_roles is True
    assert m.require_readback_confirm is True


def test_twitch_extra_forbid() -> None:
    with pytest.raises(pydantic.ValidationError):
        TwitchConfig(definitely_not_a_real_key=1)
    with pytest.raises(pydantic.ValidationError):
        TwitchConfig(chat={"reply_enabled": True, "bogus": 2})


def test_config_import_does_not_pull_in_twitch_package() -> None:
    """A fresh interpreter that imports kenning.config must not import the
    kenning.twitch package — config only DEFINES the schema classes."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    code = textwrap.dedent(
        """
        import sys
        import kenning.config  # noqa: F401
        leaked = sorted(m for m in sys.modules
                        if m == "kenning.twitch" or m.startswith("kenning.twitch."))
        print("LEAKED=" + ",".join(leaked))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "LEAKED=\n" in (out.stdout + "\n") or out.stdout.strip().endswith("LEAKED="), (
        f"kenning.config import leaked twitch modules: {out.stdout!r}"
    )


def test_twitch_package_imports_no_forbidden_libs() -> None:
    """Importing the kenning.twitch package namespace must not ADD any forbidden
    network / sidecar-only library beyond what ``import kenning`` already loads.

    We measure the DELTA: ``import kenning`` (the base package registers CUDA DLLs
    and may pull torch/numpy — allowed in-process compute) is the baseline; then
    ``import kenning.twitch`` may only add the package namespace itself, never a
    new network/ML transport. (torch/numpy/llama_cpp are NOT in the forbidden set
    here — they are the main process's allowed in-process compute; the forbidden
    set is the networking + sidecar-only transports that must stay in sidecars.)
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    code = textwrap.dedent(
        """
        import sys
        import kenning  # noqa: F401   -- baseline (CUDA DLL setup, torch/numpy)
        before = set(sys.modules)
        import kenning.twitch  # noqa: F401
        added = set(sys.modules) - before
        forbidden = ("requests", "aiohttp", "httpx", "websockets", "websocket",
                     "transformers", "sqlite_vec", "obsws_python", "twitchio")
        newly = sorted(f for f in forbidden if f in added)
        print("NEWLY_FORBIDDEN=" + ",".join(newly))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("NEWLY_FORBIDDEN=")), None)
    assert line is not None, out.stdout
    assert line == "NEWLY_FORBIDDEN=", f"kenning.twitch added forbidden libs: {line}"
