"""Tiny logging HTTP proxy.

Listens on :8766, forwards everything to http://127.0.0.1:8765
(llama-cpp-server), logs request method/path/body and response status
to stdout. Used to capture what OpenClaw actually sends so we can
diagnose response-parsing issues.

Run:
    python scripts/_log_proxy.py
"""

from __future__ import annotations

import http.server
import json
import sys
import urllib.request
import urllib.error

UPSTREAM = "http://127.0.0.1:8765"


class LogProxy(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Quieter default access log; we print what we want manually.
        return

    def _forward(self, method: str) -> None:
        body = b""
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            body = self.rfile.read(n)
        url = UPSTREAM + self.path
        sys.stdout.write(f"\n=== {method} {self.path} ===\n")
        if body:
            try:
                parsed = json.loads(body)
                preview = json.dumps(parsed, indent=2, default=str)
                if len(preview) > 8000:
                    preview = preview[:8000] + "\n...(truncated)"
                sys.stdout.write(f"REQUEST BODY:\n{preview}\n")
            except Exception:
                sys.stdout.write(f"REQUEST BODY (non-json): {body[:200]!r}\n")
        sys.stdout.flush()
        req = urllib.request.Request(url, data=body or None, method=method)
        for h in ("Authorization", "Content-Type", "Accept"):
            v = self.headers.get(h)
            if v:
                req.add_header(h, v)
        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() in ("content-length", "transfer-encoding"):
                    continue
                self.send_header(k, v)
            self.end_headers()
            data = e.read()
            self.wfile.write(data)
            sys.stdout.write(f"RESPONSE: {e.code}\n")
            sys.stdout.flush()
            return
        # Stream the body through, log a preview.
        self.send_response(resp.status)
        is_sse = False
        for k, v in resp.headers.items():
            if k.lower() in ("content-length", "transfer-encoding"):
                continue
            if k.lower() == "content-type" and "event-stream" in v:
                is_sse = True
            self.send_header(k, v)
        if is_sse:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        sys.stdout.write(f"RESPONSE: {resp.status} sse={is_sse}\n")
        sys.stdout.flush()
        # Stream + tee to stdout for SSE; for non-SSE, read all + dump.
        if is_sse:
            collected = []
            while True:
                chunk = resp.readline()
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                collected.append(chunk.decode("utf-8", errors="replace"))
            preview = "".join(collected)
            if len(preview) > 8000:
                preview = preview[:8000] + "\n...(truncated)"
            sys.stdout.write(f"SSE STREAM:\n{preview}\n=== end ===\n")
            sys.stdout.flush()
        else:
            data = resp.read()
            self.wfile.write(data)
            try:
                parsed = json.loads(data)
                preview = json.dumps(parsed, indent=2, default=str)
            except Exception:
                preview = data.decode("utf-8", errors="replace")
            if len(preview) > 4000:
                preview = preview[:4000] + "\n...(truncated)"
            sys.stdout.write(f"RESPONSE BODY:\n{preview}\n=== end ===\n")
            sys.stdout.flush()

    def do_GET(self) -> None:  # noqa: N802
        self._forward("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._forward("POST")


def main() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 8766), LogProxy)
    sys.stdout.write("[log-proxy] :8766 -> :8765\n")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stdout.write("\n[log-proxy] stopped\n")


if __name__ == "__main__":
    main()
