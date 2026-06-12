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
            if "code" in q:
                captured["code"] = q["code"][0]
                body = b"Ultron is now connected to Spotify. You can close this tab."
            else:
                captured["error"] = q.get("error", ["unknown"])[0]
                body = b"Authorization failed. Check the console."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):  # silence the default logging
            return

    server = HTTPServer((host, port), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()

    url = build_authorize_url(creds)
    print("Opening the Spotify consent page in your browser...")
    print(f"If it doesn't open, paste this URL:\n{url}\n")
    webbrowser.open(url)

    # Wait for the redirect (handle_request serves exactly one).
    import time

    for _ in range(600):  # ~60 s
        if captured:
            break
        time.sleep(0.1)
    if "code" not in captured:
        print(f"No authorization code received ({captured.get('error', 'timeout')}).")
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
