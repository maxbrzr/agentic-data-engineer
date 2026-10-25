import json
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MARKER_PATH = Path(
    os.environ.get(
        "ADE_SANDBOX_MARKER",
        "/tmp/agentic-data-engineer-sandbox.json",
    )
)
OUTPUT_DIR = Path(os.environ["ADE_OUTPUT_DIR"])
# Each probe gets a fresh file name: Docker Desktop's file sharing can keep a
# stale entry for a name the host just deleted, so rewriting it fails (ENOENT).
PROBE_PREFIX = ".opencode-sandbox-live-probe-"


class AttestationHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        if route == "/health":
            payload = b'{"healthy":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if route != "/agentic-data-engineer-sandbox.json":
            self.send_error(404)
            return

        try:
            marker = json.loads(MARKER_PATH.read_text(encoding="utf-8"))
            probe_token = secrets.token_hex(16)
            probe_name = PROBE_PREFIX + probe_token
            (OUTPUT_DIR / probe_name).write_text(
                probe_token,
                encoding="utf-8",
            )
            marker["probe_name"] = probe_name
            marker["probe_token"] = probe_token
            payload = json.dumps(marker, sort_keys=True).encode("utf-8")
        except Exception as exc:
            self.send_error(500, explain=str(exc))
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 54322), AttestationHandler).serve_forever()
