"""OpenClaw Gateway lifecycle awareness.

The bridge does NOT start or stop the Gateway — the user manages that
via ``gateway.cmd`` or the supervisor. This module just knows how to:

- locate the Gateway URL (from config, falling back to OpenClaw's
  default loopback bind),
- check whether it's reachable (a fast sub-second probe),
- wait for it to come up at startup with a configurable timeout, and
- read the bearer token out of OpenClaw's local config so the bridge
  can authenticate to the HTTP API.

Critical contract: ``is_reachable()`` and ``wait_for_ready()`` must
NEVER raise. They return False / a timeout result. The voice path
must keep working when the Gateway is down (``openclaw.fail_open:
true``); the bridge logs a warning and degrades capabilities.

Auth model (per OpenClaw 2026.5.x): the Gateway issues a bearer
token stored in ``~/.openclaw/openclaw.json`` under
``gateway.auth.token``. Bridge clients send it as
``Authorization: Bearer <token>``. Reading it from disk is fine
(the file is local and the user's). Do NOT log the token at any
level.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.lifecycle")


# Gateway local-loopback default. Confirmed via `openclaw status` on a
# fresh OpenClaw 2026.5.7 install. Override via ULTRON_OPENCLAW_GATEWAY_URL.
_DEFAULT_GATEWAY_URL = "http://127.0.0.1:18789"

# Path to OpenClaw's user config. The auth token lives at
# ``gateway.auth.token``; we read it lazily so a missing file doesn't
# crash bridge construction.
_DEFAULT_CONFIG_PATH = Path.home() / ".openclaw" / "openclaw.json"


@dataclass(frozen=True)
class OpenClawStatus:
    """Snapshot of Gateway state. Returned by :meth:`get_status`.

    All fields are best-effort — a partial response from the Gateway
    still produces a value here, with the missing pieces left as None
    or empty.
    """

    reachable: bool
    runtime_version: Optional[str] = None
    default_agent_id: Optional[str] = None
    configured_channels: tuple[str, ...] = ()
    error: Optional[str] = None


def _read_token(config_path: Path = _DEFAULT_CONFIG_PATH) -> Optional[str]:
    """Read the Gateway auth token from OpenClaw's user config.

    Returns None on any error (missing file, malformed JSON, missing
    key). Never raises. Never logs the token.
    """
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.debug("openclaw config not found at %s", config_path)
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("could not read openclaw config (%s)", e)
        return None
    auth = (data.get("gateway") or {}).get("auth") or {}
    token = auth.get("token")
    if not isinstance(token, str) or not token.strip():
        return None
    return token.strip()


def _resolve_gateway_url(override: Optional[str] = None) -> str:
    """Resolve the Gateway HTTP URL.

    Order: explicit ``override`` arg → ``ULTRON_OPENCLAW_GATEWAY_URL``
    env var → ``http://127.0.0.1:18789`` default.

    The Gateway listens on HTTP for tool/agent calls and on WebSocket
    for streaming agent runs. This URL is the HTTP base — health,
    tool invocations, etc. WebSocket calls construct their own
    ``ws://...`` URL from the same host:port.
    """
    if override:
        return override.rstrip("/")
    env = os.getenv("ULTRON_OPENCLAW_GATEWAY_URL")
    if env:
        return env.rstrip("/")
    return _DEFAULT_GATEWAY_URL


class OpenClawLifecycle:
    """Health + reachability for the Gateway. Never raises.

    Args:
        gateway_url: HTTP base URL. If None, resolves via
            :func:`_resolve_gateway_url`.
        config_path: location of OpenClaw's user config. The auth
            token is read from there.
        probe_timeout_s: per-request timeout for health probes
            (kept short — health checks must not block startup).
    """

    def __init__(
        self,
        gateway_url: Optional[str] = None,
        *,
        config_path: Optional[Path] = None,
        probe_timeout_s: float = 2.0,
    ) -> None:
        self.gateway_url = _resolve_gateway_url(gateway_url)
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._probe_timeout_s = probe_timeout_s

    @property
    def auth_token(self) -> Optional[str]:
        """Current Gateway bearer token. Re-read on every access so a
        post-startup auth rotation is picked up without restart."""
        return _read_token(self._config_path)

    def is_reachable(self) -> bool:
        """Fast probe: the Gateway responds to its canvas-host static
        path on the HTTP port. Used as a liveness signal because no
        unauth'd JSON endpoint is guaranteed across versions.

        Returns False on connection refused, timeout, or any error.
        Never raises.
        """
        # The Gateway mounts the canvas host at /__openclaw__/canvas/
        # (visible in the startup log). Even a 404 from there means the
        # Gateway is running and serving HTTP. Connection refused or
        # timeout means it isn't.
        url = f"{self.gateway_url}/__openclaw__/canvas/"
        try:
            resp = requests.get(url, timeout=self._probe_timeout_s)
            return resp.status_code < 500
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            return False
        except requests.exceptions.RequestException as e:
            logger.debug("Gateway probe error: %s", e)
            return False

    def wait_for_ready(
        self,
        timeout_s: float = 30.0,
        *,
        poll_interval_s: float = 1.0,
    ) -> bool:
        """Block until the Gateway is reachable or ``timeout_s`` elapses.

        Returns True if reachable within the deadline, False otherwise.
        Useful at Ultron startup when we want to register Ultron's MCP
        with the Gateway as soon as it's up.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            if self.is_reachable():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_interval_s)

    def get_status(self) -> OpenClawStatus:
        """Snapshot of Gateway state. Always returns a value (with
        ``reachable=False`` and an ``error`` string on failure)."""
        if not self.is_reachable():
            return OpenClawStatus(reachable=False, error="not reachable")
        # OpenClaw's HTTP API doesn't expose a stable unauth /status
        # endpoint. Read what we can from the config file — runtime
        # version is in there as ``meta.lastTouchedVersion``, default
        # agent comes from ``agents.defaults.model`` heuristically, etc.
        # This is good enough for the bridge to decide whether to
        # proceed; richer status comes from the CLI.
        runtime_version = None
        default_agent_id = None
        configured_channels: tuple[str, ...] = ()
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return OpenClawStatus(reachable=True)
        meta = data.get("meta") or {}
        if isinstance(meta.get("lastTouchedVersion"), str):
            runtime_version = meta["lastTouchedVersion"]
        agents = data.get("agents") or {}
        # Prefer the first listed agent's id if present, else fall back
        # to the model alias from defaults.
        agent_list = agents.get("list") or []
        if agent_list and isinstance(agent_list, list):
            for entry in agent_list:
                if isinstance(entry, dict) and entry.get("default") is True:
                    default_agent_id = entry.get("id")
                    break
            if default_agent_id is None:
                first = agent_list[0]
                if isinstance(first, dict):
                    default_agent_id = first.get("id")
        channels = data.get("channels") or {}
        configured_channels = tuple(
            name for name, cfg in channels.items()
            if isinstance(cfg, dict) and cfg.get("enabled") is True
        )
        return OpenClawStatus(
            reachable=True,
            runtime_version=runtime_version,
            default_agent_id=default_agent_id,
            configured_channels=configured_channels,
        )


__all__ = [
    "OpenClawLifecycle",
    "OpenClawStatus",
]
