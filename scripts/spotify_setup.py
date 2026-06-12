"""One-time Spotify authorization for Ultron.

Opens the consent page in your browser, catches the redirect on a tiny
local server, exchanges the code for a refresh token, and saves it into
the gitignored credentials file (``~/.ultron/spotify.json``). Run once::

    python scripts/spotify_setup.py

Prerequisites (in the Spotify dashboard for the app whose client
id/secret are in the credentials file):
  * Add the redirect URI EXACTLY as it appears in the file
    (default ``http://127.0.0.1:8899/callback``).
  * The Spotify account must be Premium for playback control.
"""

from __future__ import annotations

import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ultron.spotify.auth import (  # noqa: E402
    build_authorize_url,
    exchange_code,
    load_credentials,
    save_refresh_token,
)

DEFAULT_CREDS = "~/.ultron/spotify.json"


def main() -> int:
    creds_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CREDS
    creds = load_credentials(creds_path)
    parsed = urllib.parse.urlparse(creds.redirect_uri)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 8899

    captured: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            # Ignore favicon / prefetch hits -- only the real callback
            # carries ``code`` or ``error``.
            if "code" in q:
                captured["code"] = q["code"][0]
                body = b"Ultron is now connected to Spotify. You can close this tab."
            elif "error" in q:
                captured["error"] = q["error"][0]
                body = b"Authorization was denied. Check the console."
            else:
                self.send_response(204)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):  # silence the default logging
            return

    server = HTTPServer((host, port), Handler)
    # serve_forever in a daemon thread so favicon/prefetch requests don't
    # consume a one-shot handler before the real callback arrives.
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = build_authorize_url(creds)
    print("Opening the Spotify consent page in your browser...", flush=True)
    print(f"If it doesn't open, paste this URL:\n{url}\n", flush=True)
    print(f"Waiting up to 3 minutes for you to click Agree "
          f"(listening on {host}:{port})...", flush=True)
    webbrowser.open(url)

    import time

    for _ in range(1800):  # ~180 s
        if captured:
            break
        time.sleep(0.1)
    server.shutdown()
    if "code" not in captured:
        print(f"No authorization code received "
              f"({captured.get('error', 'timeout')}).", flush=True)
        return 1

    payload = exchange_code(creds, captured["code"])
    refresh = payload.get("refresh_token")
    if not refresh:
        print("No refresh token returned. Did you already authorize once?")
        return 1
    save_refresh_token(creds.path or creds_path, refresh)
    print(f"Saved refresh token to {creds.path}. Ultron can now control Spotify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
