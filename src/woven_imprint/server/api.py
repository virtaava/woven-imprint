"""OpenAI-compatible API server — drop-in persistent character proxy.

Any system that can point to an OpenAI base URL gets persistent characters.
The model name maps to a character: model="alice" → chat as Alice.

Usage:
    python -m woven_imprint.server.api --port 8650

Then point any OpenAI client to http://localhost:8650/v1:
    client = OpenAI(base_url="http://localhost:8650/v1", api_key="not-needed")
    response = client.chat.completions.create(model="alice", messages=[...])
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from ..engine import Engine
from .services import (
    find_character_by_name_or_id,
    extract_last_user_message,
    extract_user_id_from_messages,
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


class OpenAIHandler(BaseHTTPRequestHandler):
    """Handle OpenAI-compatible chat completions requests."""

    def _check_auth(self) -> bool:
        """Check bearer token if API key is configured."""
        api_key = _config.get("api_key")
        if not api_key:
            return True  # no auth configured
        auth_header = self.headers.get("Authorization", "")
        if auth_header == f"Bearer {api_key}":
            return True
        self._send_error("Invalid or missing API key", 401)
        return False

    def do_GET(self):
        if not self._check_auth():
            return
        if self.path == "/v1/models":
            self._send_models()
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_error("not found", 404)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(204)
        self.send_header(
            "Access-Control-Allow-Origin", _config.get("cors_origin", "http://localhost")
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_POST(self):
        if not self._check_auth():
            return
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        else:
            self._send_error("not found", 404)

    def _handle_chat(self):
        body = self._read_body()
        if not body:
            self._send_error("empty body", 400)
            return

        model_name = body.get("model", "").lower().strip()
        messages = body.get("messages", [])

        if not model_name or not messages:
            self._send_error("model and messages required", 400)
            return

        engine = _get_engine()

        # Find character by model name
        match = find_character_by_name_or_id(engine, model_name)
        if not match:
            self._send_error(
                f"Character '{model_name}' not found. "
                f"Use GET /v1/models to list available characters.",
                404,
            )
            return

        char = engine.load_character(match["id"])

        user_msg = extract_last_user_message(messages)
        if not user_msg:
            self._send_error("no user message found", 400)
            return

        user_id = extract_user_id_from_messages(messages)

        # Generate response
        response = char.chat(user_msg, user_id=user_id)

        # Format as OpenAI response
        self._send_json(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_msg) // 4,
                    "completion_tokens": len(response) // 4,
                    "total_tokens": len(user_msg) // 4 + len(response) // 4,
                },
                # Extra: woven imprint metadata
                "woven_imprint": {
                    "character": char.name,
                    "emotion": char.emotion.to_dict(),
                    "arc_phase": char.arc.current_phase.value,
                    "arc_tension": char.arc.tension,
                },
            }
        )

    def _send_models(self):
        engine = _get_engine()
        chars = engine.list_characters()
        models = []
        for c in chars:
            models.append(
                {
                    "id": c["name"].lower().replace(" ", "-"),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "woven-imprint",
                }
            )
        self._send_json({"object": "list", "data": models})

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
        """Send OpenAI-compatible error response."""
        self._send_json(
            {"error": {"message": message, "type": "invalid_request_error", "code": status}},
            status,
        )

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Access-Control-Allow-Origin", _config.get("cors_origin", "http://localhost")
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quiet logging
        pass


def run_server(
    port: int = 8650,
    db_path: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
):
    """Start the OpenAI-compatible API server."""
    global _config
    import os

    resolved_model = model or os.environ.get("WOVEN_IMPRINT_MODEL", "llama3.2")
    resolved_key = api_key or os.environ.get("WOVEN_IMPRINT_API_KEY")
    _config = {"db_path": db_path, "model": resolved_model, "api_key": resolved_key}

    server = HTTPServer(("127.0.0.1", port), OpenAIHandler)
    print(f"Woven Imprint API server running on http://127.0.0.1:{port}")
    print(f"OpenAI base_url: http://127.0.0.1:{port}/v1")
    print("Models = characters. Use model='character-name' in API calls.")
    print("GET /v1/models to list available characters.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Woven Imprint API Server")
    parser.add_argument("--port", type=int, default=8650)
    parser.add_argument("--db", default=None)
    parser.add_argument("--model", default="llama3.2")
    args = parser.parse_args()
    run_server(port=args.port, db_path=args.db, model=args.model)
