"""Standard-library web view: shared game state plus a polling JSON API."""
from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).resolve().parent / "static"


class PlaygroundState:
    """Written by the decision loop, read by HTTP threads."""

    def __init__(self, *, step_ms=150, log_size=300):
        self._cond = threading.Condition()
        self.board = None
        self.phase = "warming"
        self.paused = False
        self.step_ms = int(step_ms)
        self.game_index = 1
        self.games = []
        self.decisions = 0
        self.forced = 0
        self.request_s_total = 0.0
        self.tokens_total = 0
        self._seq = 0
        self._log = deque(maxlen=log_size)

    def set_board(self, snapshot, phase=None):
        with self._cond:
            self.board = snapshot
            if phase:
                self.phase = phase

    def add_decision(self, entry):
        with self._cond:
            self._seq += 1
            entry = dict(entry, seq=self._seq, game=self.game_index)
            self._log.append(entry)
            if entry.get("forced"):
                self.forced += 1
            else:
                self.decisions += 1
                self.request_s_total += entry.get("request_s") or 0.0
                self.tokens_total += entry.get("input_tokens") or 0
            return entry

    def finish_game(self, record):
        with self._cond:
            self.games.append(dict(record, game=self.game_index))
            self.game_index += 1

    def control(self, *, pause=None, step_ms=None):
        with self._cond:
            if pause is not None:
                self.paused = bool(pause)
            if step_ms is not None:
                self.step_ms = max(0, min(5000, int(step_ms)))
            self._cond.notify_all()
            return {"paused": self.paused, "step_ms": self.step_ms}

    def wait_while_paused(self):
        with self._cond:
            while self.paused:
                self._cond.wait(0.5)

    def get_step_ms(self):
        with self._cond:
            return self.step_ms

    def summary(self):
        with self._cond:
            return self._summary()

    def _summary(self):
        scores = [g["score"] for g in self.games]
        return {
            "games_finished": len(self.games),
            "games": list(self.games),
            "best_score": max(scores, default=0),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "decisions": self.decisions,
            "forced_moves": self.forced,
            "mean_request_ms": 1000 * self.request_s_total / self.decisions if self.decisions else 0.0,
            "mean_input_tokens": self.tokens_total / self.decisions if self.decisions else 0.0,
        }

    def view(self, since=0):
        with self._cond:
            return {
                "board": self.board,
                "phase": self.phase,
                "paused": self.paused,
                "step_ms": self.step_ms,
                "game_index": self.game_index,
                "stats": self._summary(),
                "log": [e for e in self._log if e["seq"] > since],
                "log_reset": bool(self._log) and since < self._log[0]["seq"] - 1,
            }


def _handler(state: PlaygroundState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser closed or reloaded mid-poll

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                try:
                    since = int(parse_qs(url.query).get("since", ["0"])[0])
                except ValueError:
                    since = 0
                self._json(state.view(since))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/control":
                return self._json({"error": "not found"}, 404)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                result = state.control(pause=body.get("pause"), step_ms=body.get("step_ms"))
            except (ValueError, TypeError, AttributeError) as exc:
                return self._json({"error": str(exc)}, 400)
            self._json(result)

    return Handler


def start_server(state: PlaygroundState, host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), _handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="playground-http", daemon=True)
    thread.start()
    return server
