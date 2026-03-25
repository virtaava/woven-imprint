"""Sidecar HTTP API for SillyTavern extension integration.

Exposes REST endpoints so the SillyTavern extension can record dialogue
events and query character memory/relationships without going through the
OpenAI-compatible proxy.

Usage:
    python -m woven_imprint.server.sidecar --port 8765

Endpoints:
    GET  /health                          Health check
    POST /characters                      Create character (deduplicates by name)
    GET  /characters                      List all characters
    GET  /characters/{id}                 Get character info + emotion
    POST /characters/{id}/session         Start session
    DELETE /characters/{id}/session       End session
    POST /record                          Record a message via ingest()
    GET  /memory?character_id=X&query=Z   Query relevant memories
    GET  /relationships/{char_id}/{tid}   Get relationship dimensions
"""

from __future__ import annotations

import json
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

from ..engine import Engine
from .services import (
    create_character_service,
    list_characters_service,
    get_character_state_service,
    start_session_service,
    end_session_service,
    record_message_service,
    recall_memories_service,
    get_relationship_service,
)


_engine: Engine | None = None
_config: dict = {}


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        from ..config import get_config
        from ..providers import create_llm, create_embedding

        cfg = get_config()
        db_path = _config.get("db_path") or cfg.storage.db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        model = _config.get("model") or cfg.llm.model
        cfg.llm.model = model
        _engine = Engine(
            db_path=db_path,
            llm=create_llm(cfg),
            embedding=create_embedding(cfg),
        )
    return _engine


# Route patterns for path parameters
_CHAR_ID_RE = re.compile(r"^/characters/([^/]+)$")
_CHAR_SESSION_RE = re.compile(r"^/characters/([^/]+)/session$")
_REL_RE = re.compile(r"^/relationships/([^/]+)/([^/]+)$")


class SidecarHandler(BaseHTTPRequestHandler):
    """Handle sidecar REST API requests."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        if path == "/health":
            self._handle_health()
        elif path == "/characters":
            self._handle_list_characters()
        elif path == "/memory":
            self._handle_memory_query(qs)
        elif m := _CHAR_ID_RE.match(path):
            self._handle_get_character(m.group(1))
        elif m := _REL_RE.match(path):
            self._handle_get_relationship(m.group(1), m.group(2))
        else:
            self._send_error("not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/characters":
            self._handle_create_character()
        elif path == "/record":
            self._handle_record()
        elif m := _CHAR_SESSION_RE.match(path):
            self._handle_start_session(m.group(1))
        else:
            self._send_error("not found", 404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if m := _CHAR_SESSION_RE.match(path):
            self._handle_end_session(m.group(1))
        else:
            self._send_error("not found", 404)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    # ---- Handlers ----

    def _handle_health(self):
        from .. import __version__

        self._send_json({"status": "ok", "version": __version__})

    def _handle_create_character(self):
        body = self._read_body()
        if body is None:
            self._send_error("invalid or empty request body", 400)
            return

        name = body.get("name", "").strip()
        if not name:
            self._send_error("name is required", 400)
            return

        # Build persona dict from body (request parsing stays here)
        raw_persona = body.get("persona")
        if isinstance(raw_persona, str):
            persona = {"personality": raw_persona}
        elif isinstance(raw_persona, dict):
            persona = raw_persona
        else:
            persona = {}
        for key in ("personality", "speaking_style", "occupation", "appearance", "backstory"):
            if key in body and key not in persona:
                persona[key] = body[key]

        try:
            result = create_character_service(
                _get_engine(),
                name,
                persona,
                body.get("birthdate"),
            )
            self._send_json(result, 201 if result["created"] else 200)
        except Exception as exc:
            self._send_error(str(exc), 500)

    def _handle_list_characters(self):
        chars = list_characters_service(_get_engine())
        self._send_json({"characters": chars})

    def _handle_get_character(self, char_id: str):
        try:
            data = get_character_state_service(_get_engine(), char_id)
            self._send_json(data)
        except KeyError:
            self._send_error(f"character '{char_id}' not found", 404)

    def _handle_start_session(self, char_id: str):
        try:
            result = start_session_service(_get_engine(), char_id)
            self._send_json(result)
        except KeyError:
            self._send_error(f"character '{char_id}' not found", 404)

    def _handle_end_session(self, char_id: str):
        try:
            result = end_session_service(_get_engine(), char_id)
            self._send_json(result)
        except KeyError:
            self._send_error(f"character '{char_id}' not found", 404)

    def _handle_record(self):
        body = self._read_body()
        if body is None:
            self._send_error("invalid or empty request body", 400)
            return

        char_id = body.get("character_id")
        role = body.get("role")
        content = body.get("content")
        user_id = body.get("user_id")

        if not char_id or not role or not content:
            self._send_error("character_id, role, and content are required", 400)
            return

        try:
            record_message_service(
                _get_engine(),
                char_id,
                role,
                content,
                user_id,
                strict_roles=True,
            )
            self._send_json({"ok": True})
        except KeyError:
            self._send_error(f"character '{char_id}' not found", 404)
        except ValueError as exc:
            self._send_error(str(exc), 400)

    def _handle_memory_query(self, qs: dict):
        char_id = qs.get("character_id", [None])[0]
        query = qs.get("query", [None])[0]

        if not char_id or not query:
            self._send_error("character_id and query are required", 400)
            return

        limit = 10
        if "limit" in qs:
            try:
                limit = int(qs["limit"][0])
            except (ValueError, IndexError):
                pass

        user_id = qs.get("user_id", [None])[0]

        try:
            result = recall_memories_service(
                _get_engine(),
                char_id,
                query,
                limit=limit,
                user_id=user_id,
            )
            self._send_json(result)
        except KeyError:
            self._send_error(f"character '{char_id}' not found", 404)

    def _handle_get_relationship(self, char_id: str, target_id: str):
        try:
            rel = get_relationship_service(_get_engine(), char_id, target_id)
            self._send_json({"relationship": rel})
        except KeyError:
            self._send_error(f"character '{char_id}' not found", 404)

    # ---- Helpers ----

    def _read_body(self, max_size: int = 1_048_576) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return None
        if length > max_size:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _send_error(self, message: str, status: int = 400):
        self._send_json({"error": message}, status)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def run_server(
    port: int = 8765,
    db_path: str | None = None,
    model: str | None = None,
):
    """Start the sidecar API server."""
    global _config
    import os

    resolved_model = model or os.environ.get("WOVEN_IMPRINT_MODEL", "llama3.2")
    _config = {"db_path": db_path, "model": resolved_model}

    server = HTTPServer(("127.0.0.1", port), SidecarHandler)
    print(f"Woven Imprint sidecar running on http://127.0.0.1:{port}")
    print("Endpoints: /health, /characters, /record, /memory, /relationships\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Woven Imprint Sidecar API")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default=None)
    parser.add_argument("--model", default="llama3.2")
    args = parser.parse_args()
    run_server(port=args.port, db_path=args.db, model=args.model)
