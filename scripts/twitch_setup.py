"""Interactive Twitch device-flow setup — the streamer's one-time wake-up step.

Runs the OAuth Device Code Flow (RFC 8628): prints the short user code + the
verification URL, polls until the user approves in a browser, then stores the
token set atomically to ``~/.kenning/twitch.json`` (gitignored, pre-push
secret-deny like ``spotify.json``).

ANTICHEAT (BR-P1): pure stdlib + ``urllib`` (via :mod:`kenning.twitch.auth`).
Import-safe — running ``import scripts.twitch_setup`` has NO side effects
(everything is inside functions; the device flow only runs under ``__main__``).

Usage::

    python scripts/twitch_setup.py --client-id <ID> --identity broadcaster
    python scripts/twitch_setup.py --client-id <ID> --identity bot --path ~/.kenning/twitch_bot.json

Token VALUES are never printed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from kenning.twitch.auth import (
    BOT_SCOPES,
    BROADCASTER_SCOPES,
    DeviceFlowError,
    RevokedError,
    TokenStore,
    TwitchAuth,
    TwitchAuthError,
)

logger = logging.getLogger("kenning.twitch.setup")

_IDENTITY_SCOPES = {
    "broadcaster": BROADCASTER_SCOPES,
    "bot": BOT_SCOPES,
}
_DEFAULT_PATHS = {
    "broadcaster": "~/.kenning/twitch.json",
    "bot": "~/.kenning/twitch_bot.json",
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser (no parsing side effects)."""
    p = argparse.ArgumentParser(
        prog="twitch_setup",
        description="One-time Twitch OAuth device-flow setup for Ultron.",
    )
    p.add_argument(
        "--client-id",
        required=True,
        help="Twitch application client id (public; no client secret needed).",
    )
    p.add_argument(
        "--identity",
        choices=sorted(_IDENTITY_SCOPES),
        default="broadcaster",
        help="Which identity to authorize (selects the least-privilege scope set).",
    )
    p.add_argument(
        "--path",
        default=None,
        help="Override the token store path (defaults per identity under ~/.kenning/).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Overall seconds to wait for the user to approve (default 900).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging (never logs token values).",
    )
    return p


def run_setup(
    client_id: str,
    *,
    identity: str = "broadcaster",
    path: Optional[str] = None,
    timeout: float = 900.0,
    auth: Optional[TwitchAuth] = None,
    out=None,
) -> dict:
    """Run the device flow end-to-end and persist the tokens; return them.

    ``auth`` is injectable so tests can drive the whole flow with a mock
    transport. ``out`` is the stream for user-facing prompts (defaults to
    ``sys.stdout``). Token values are never written to ``out`` or the log.
    """
    out = out or sys.stdout
    scopes = _IDENTITY_SCOPES.get(identity, BROADCASTER_SCOPES)
    store_path = path or _DEFAULT_PATHS.get(identity, "~/.kenning/twitch.json")

    if auth is None:
        store = TokenStore(store_path)
        auth = TwitchAuth(client_id, store, scopes=scopes)

    device = auth.start_device_flow()

    print("", file=out)
    print("=" * 60, file=out)
    print("  Twitch authorization required", file=out)
    print("=" * 60, file=out)
    print(f"  1. Open: {device.verification_uri}", file=out)
    print(f"  2. Enter this code: {device.user_code}", file=out)
    print(f"  (waiting up to {int(timeout)}s; code expires in {device.expires_in}s)", file=out)
    print("=" * 60, file=out)
    out.flush()

    tokens = auth.poll_device_token(
        device.device_code, interval=device.interval, timeout=timeout
    )

    print("", file=out)
    print("Authorization complete. Tokens stored securely.", file=out)
    print(f"  store: {auth.store.path}", file=out)
    print(f"  scopes: {tokens.get('scope', '')}", file=out)
    out.flush()
    return tokens


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 ok, non-zero on error)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run_setup(
            args.client_id,
            identity=args.identity,
            path=args.path,
            timeout=args.timeout,
        )
    except RevokedError as e:
        print(f"\nAuthorization was revoked or rejected: {e}", file=sys.stderr)
        return 3
    except DeviceFlowError as e:
        print(f"\nDevice flow failed: {e}", file=sys.stderr)
        return 2
    except TwitchAuthError as e:
        print(f"\nSetup failed: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
