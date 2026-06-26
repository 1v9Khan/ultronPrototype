"""Unit tests for the pure Twitch sidecar-spawn planner (kenning.twitch.sidecar_launch).

The planner decides WHICH sidecars to launch + with WHAT env from a TwitchConfig,
without spawning anything -- so we assert the gating + env contract here and leave
the actual Popen to the orchestrator-method test.
"""
from __future__ import annotations

from kenning.config import TwitchConfig
from kenning.twitch.sidecar_launch import SidecarSpec, plan_sidecars


def _roles(specs: list[SidecarSpec]) -> list[str]:
    return [s.role for s in specs]


def test_default_config_plans_only_the_read_sidecar() -> None:
    """No guard_model_path + helper disabled => just the read sidecar."""
    specs = plan_sidecars(TwitchConfig(), path_exists=lambda _p: False)
    assert _roles(specs) == ["twitch_read"]
    read = specs[0]
    assert read.script == "scripts/twitch_read_sidecar.py"
    assert read.port == 8773
    assert read.env["KENNING_TWITCH_READ_PORT"] == "8773"
    # redeems default OFF (economy disabled)
    assert read.env["KENNING_TWITCH_SUBSCRIBE_REDEEMS"] == "0"


def test_read_env_carries_auth_and_token_paths() -> None:
    cfg = TwitchConfig(
        enabled=True,
        auth={
            "client_id": "abc123",
            "broadcaster_login": "1v9khan",
            "bot_login": "ultron_kenning",
        },
        economy={"enabled": True},
    )
    read = plan_sidecars(cfg, path_exists=lambda _p: False)[0]
    assert read.env["KENNING_TWITCH_CLIENT_ID"] == "abc123"
    assert read.env["KENNING_TWITCH_BROADCASTER_LOGIN"] == "1v9khan"
    assert read.env["KENNING_TWITCH_BOT_LOGIN"] == "ultron_kenning"
    # token paths are expanded (no leading ~) and point at the two stores
    assert read.env["KENNING_TWITCH_BOT_TOKEN_PATH"].endswith("twitch_bot.json")
    assert "~" not in read.env["KENNING_TWITCH_BOT_TOKEN_PATH"]
    assert read.env["KENNING_TWITCH_BROADCASTER_TOKEN_PATH"].endswith("twitch.json")
    # economy enabled => subscribe to redeems
    assert read.env["KENNING_TWITCH_SUBSCRIBE_REDEEMS"] == "1"


def test_read_env_subscribes_raids_when_raid_enabled() -> None:
    # raid.enabled defaults ON => the read sidecar subscribes to channel.raid.
    on = plan_sidecars(TwitchConfig(enabled=True), path_exists=lambda _p: False)[0]
    assert on.env["KENNING_TWITCH_SUBSCRIBE_RAIDS"] == "1"
    # raid.enabled OFF => no raid subscription (main runtime unchanged).
    off = plan_sidecars(
        TwitchConfig(enabled=True, raid={"enabled": False}),
        path_exists=lambda _p: False,
    )[0]
    assert off.env["KENNING_TWITCH_SUBSCRIBE_RAIDS"] == "0"


def test_guard_planned_only_when_path_present() -> None:
    cfg = TwitchConfig(
        enabled=True,
        safety={"guard_model_path": "E:/UltronModels/Llama-Guard-3-1B.Q5_K_M.gguf",
                "guard_family": "llama-guard"},
    )
    # path missing => guard NOT planned (chat-reply stays fail-CLOSED)
    assert _roles(plan_sidecars(cfg, path_exists=lambda _p: False)) == ["twitch_read"]
    # path present => guard planned with the resolved model env
    specs = plan_sidecars(cfg, path_exists=lambda _p: True)
    assert _roles(specs) == ["twitch_read", "twitch_guard"]
    guard = specs[1]
    assert guard.script == "scripts/twitch_guard_sidecar.py"
    assert guard.port == 8774
    assert guard.env["KENNING_TWITCH_GUARD_MODEL"].endswith("Llama-Guard-3-1B.Q5_K_M.gguf")
    assert guard.env["KENNING_TWITCH_GUARD_FAMILY"] == "llama-guard"
    assert guard.env["KENNING_TWITCH_GUARD_PORT"] == "8774"


def test_helper_planned_only_when_enabled_and_present() -> None:
    cfg = TwitchConfig(
        enabled=True,
        helper={"enabled": True, "model_path": "models/qwen2.5-1.5b.gguf", "port": 8776},
    )
    # enabled but model missing => not planned
    assert "twitch_helper" not in _roles(plan_sidecars(cfg, path_exists=lambda _p: False))
    # enabled + present => planned
    specs = plan_sidecars(cfg, path_exists=lambda _p: True)
    helper = [s for s in specs if s.role == "twitch_helper"]
    assert len(helper) == 1
    assert helper[0].port == 8776
    assert helper[0].env["KENNING_TWITCH_HELPER_MODEL"].endswith("qwen2.5-1.5b.gguf")

    # disabled => never planned even if a path exists
    cfg_off = TwitchConfig(enabled=True, helper={"enabled": False, "model_path": "x.gguf"})
    assert "twitch_helper" not in _roles(plan_sidecars(cfg_off, path_exists=lambda _p: True))


def test_read_port_derived_from_endpoint() -> None:
    cfg = TwitchConfig(enabled=True, read_sidecar_endpoint="http://127.0.0.1:9001")
    read = plan_sidecars(cfg, path_exists=lambda _p: False)[0]
    assert read.port == 9001
    assert read.env["KENNING_TWITCH_READ_PORT"] == "9001"


def test_write_sidecar_planned_only_with_moderation_and_creds() -> None:
    # moderation on by default, but no client_id => NOT planned
    assert "twitch_write" not in _roles(plan_sidecars(TwitchConfig(enabled=True),
                                                      path_exists=lambda _p: False))
    # creds present + moderation on => planned with the broadcaster env
    cfg = TwitchConfig(
        enabled=True,
        auth={"client_id": "abc", "broadcaster_login": "1v9khan", "bot_login": "ultron_kenning"},
    )
    specs = plan_sidecars(cfg, path_exists=lambda _p: False)
    write = [s for s in specs if s.role == "twitch_write"]
    assert len(write) == 1
    assert write[0].script == "scripts/twitch_write_sidecar.py"
    assert write[0].port == 8777
    assert write[0].env["KENNING_TWITCH_CLIENT_ID"] == "abc"
    assert write[0].env["KENNING_TWITCH_BROADCASTER_LOGIN"] == "1v9khan"
    assert write[0].env["KENNING_TWITCH_READ_ENDPOINT"] == "http://127.0.0.1:8773"
    # moderation explicitly disabled => NOT planned even with creds
    cfg_off = TwitchConfig(
        enabled=True,
        auth={"client_id": "abc", "broadcaster_login": "1v9khan"},
        moderation={"voice_commands_enabled": False},
    )
    assert "twitch_write" not in _roles(plan_sidecars(cfg_off, path_exists=lambda _p: False))
