# SillyTavern Extension for Woven Imprint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a SillyTavern extension that gives every ST character persistent memory, relationship tracking, and cross-session recall — powered by woven-imprint running as a Python sidecar.

**Architecture:** Three components: (1) a Python sidecar HTTP API extending woven-imprint's existing OpenAI-compatible server with memory/relationship query endpoints, (2) a SillyTavern server plugin (Node.js) that proxies requests to the sidecar, and (3) a SillyTavern UI extension that hooks into the chat pipeline to inject memory context into prompts and record dialogue for extraction. The sidecar intercepts NO traffic — SillyTavern talks to its own LLM backend directly. The sidecar is called asynchronously to record events and queried synchronously to inject memory before generation.

**Tech Stack:** Python 3.11+ (woven-imprint, http.server), Node.js (SillyTavern server plugin, Express router), JavaScript (SillyTavern UI extension), SQLite (woven-imprint character DB)

---

## Key Insight: The Existing API Server

Woven-imprint already ships an OpenAI-compatible proxy (`woven_imprint.server.api`) that routes `model=character-name` to `character.chat()`. However, for SillyTavern we do NOT want to proxy the LLM — ST already handles its own LLM connection. Instead we need a **sidecar API** that:

1. **Records** user messages and AI responses (for memory formation)
2. **Queries** relevant memories + relationship state (for prompt injection)
3. **Manages** character lifecycle (create, load, session start/end)

This is a new HTTP API built alongside the existing one.

## Data Flow

```
User types in SillyTavern
  ↓
UI Extension: MESSAGE_SENT event
  → POST /api/plugins/woven-imprint/record  (async, non-blocking)
  ↓
UI Extension: generate_interceptor fires (before LLM call)
  → GET /api/plugins/woven-imprint/memory?character=<name>  (sync)
  → Injects memory context into chat array as system message
  ↓
SillyTavern sends enriched prompt to its own LLM backend
  ↓
LLM responds
  ↓
UI Extension: MESSAGE_RECEIVED event
  → POST /api/plugins/woven-imprint/record  (async, AI response)
  ↓
Sidecar: woven-imprint processes message
  → fact extraction, relationship updates, memory consolidation
```

## File Structure

```
~/sona/projects/woven-imprint/
├── src/woven_imprint/server/
│   ├── api.py                    # EXISTING: OpenAI-compatible proxy
│   └── sidecar.py                # NEW: Memory/relationship query API for extensions
├── tests/
│   └── test_sidecar.py           # NEW: Sidecar API tests

~/sona/projects/sillytavern-woven-imprint/   # NEW: Separate repo
├── README.md
├── package.json
├── plugin/                        # ST server plugin
│   ├── index.js                   # Express routes → sidecar proxy
│   └── package.json
├── extension/                     # ST UI extension
│   ├── manifest.json
│   ├── index.js                   # Chat hooks + memory injection
│   ├── settings.html              # Config panel
│   └── style.css
└── tests/
    └── test_plugin.js
```

Two deliverables:
1. **Sidecar API** — lives in woven-imprint repo (Python, new file)
2. **ST Extension** — separate repo `sillytavern-woven-imprint` (Node.js + browser JS)

---

## Task 1: Sidecar HTTP API

The sidecar exposes REST endpoints for the ST extension to record events and query memory. It wraps woven-imprint's Engine, Character, and relationship APIs.

**Files:**
- Create: `src/woven_imprint/server/sidecar.py`
- Create: `tests/test_sidecar.py`

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/characters` | Create character from ST card data |
| GET | `/characters` | List all characters |
| GET | `/characters/{id}` | Get character info + emotion + relationships |
| POST | `/characters/{id}/session` | Start session |
| DELETE | `/characters/{id}/session` | End session (consolidate) |
| POST | `/record` | Record a message (user or AI) → triggers memory/relationship |
| GET | `/memory` | Query relevant memories for prompt injection |
| GET | `/relationships/{char_id}/{target_id}` | Get relationship dimensions |
| GET | `/health` | Health check |

- [ ] **Step 1: Write failing tests for sidecar endpoints**

```python
# tests/test_sidecar.py
"""Tests for the sidecar HTTP API."""
import json
import threading
import time
import urllib.request
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def sidecar_url(tmp_path_factory):
    """Start sidecar server on a random port for testing."""
    from woven_imprint.server.sidecar import run_sidecar
    import socket

    # Find free port
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    db_path = str(tmp_path_factory.mktemp("sidecar") / "test.db")
    server_thread = threading.Thread(
        target=run_sidecar,
        kwargs={"port": port, "db_path": db_path},
        daemon=True,
    )
    server_thread.start()
    time.sleep(1)  # Wait for startup
    return f"http://127.0.0.1:{port}"


def _request(url, path, method="GET", data=None):
    req = urllib.request.Request(
        f"{url}{path}",
        method=method,
        data=json.dumps(data).encode() if data else None,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health(sidecar_url):
    status, body = _request(sidecar_url, "/health")
    assert status == 200
    assert body["status"] == "ok"


def test_create_character(sidecar_url):
    status, body = _request(sidecar_url, "/characters", "POST", {
        "name": "Alice",
        "persona": "A witty detective",
        "personality": "sharp, skeptical",
        "speaking_style": "clipped sentences",
    })
    assert status == 200
    assert body["name"] == "Alice"
    assert "id" in body


def test_list_characters(sidecar_url):
    status, body = _request(sidecar_url, "/characters")
    assert status == 200
    assert len(body["characters"]) >= 1


def test_record_and_query_memory(sidecar_url):
    # Get character ID
    _, chars = _request(sidecar_url, "/characters")
    char_id = chars["characters"][0]["id"]

    # Start session
    status, _ = _request(sidecar_url, f"/characters/{char_id}/session", "POST")
    assert status == 200

    # Record a user message
    status, _ = _request(sidecar_url, "/record", "POST", {
        "character_id": char_id,
        "role": "user",
        "content": "I found the missing artifact in the old warehouse.",
        "user_id": "player_1",
    })
    assert status == 200

    # Record an AI response
    status, _ = _request(sidecar_url, "/record", "POST", {
        "character_id": char_id,
        "role": "assistant",
        "content": "The warehouse? That's where Donovan was last seen.",
        "user_id": "player_1",
    })
    assert status == 200

    # Query memory
    status, body = _request(
        sidecar_url,
        f"/memory?character_id={char_id}&user_id=player_1&query=warehouse"
    )
    assert status == 200
    assert "context" in body  # formatted memory string for prompt injection
    assert "memories" in body  # raw memory list


def test_relationship_query(sidecar_url):
    _, chars = _request(sidecar_url, "/characters")
    char_id = chars["characters"][0]["id"]

    status, body = _request(
        sidecar_url,
        f"/relationships/{char_id}/player_1"
    )
    assert status == 200
    assert "dimensions" in body
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cd ~/sona/projects/woven-imprint && python -m pytest tests/test_sidecar.py -v`
Expected: ImportError — `sidecar` module doesn't exist yet

- [ ] **Step 3: Implement sidecar.py**

```python
# src/woven_imprint/server/sidecar.py
"""Sidecar HTTP API for SillyTavern integration.

Exposes REST endpoints for recording dialogue, querying memory,
and managing character sessions. Designed to run alongside SillyTavern
as a persistent memory backend.

Usage:
    python -m woven_imprint.server.sidecar --port 8765
    # Or:
    woven-imprint sidecar --port 8765
"""

from __future__ import annotations

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

from ..engine import Engine


_engine: Engine | None = None
_config: dict = {}
_sessions: dict[str, bool] = {}  # char_id → session active


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        from ..config import get_config
        from ..providers import create_llm, create_embedding

        cfg = get_config()
        db_path = _config.get("db_path") or cfg.storage.db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _engine = Engine(
            db_path=db_path,
            llm=create_llm(cfg),
            embedding=create_embedding(cfg),
        )
    return _engine


class SidecarHandler(BaseHTTPRequestHandler):
    """Handle sidecar REST API requests."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        if path == "/health":
            self._send_json({"status": "ok"})
        elif path == "/characters":
            self._handle_list_characters()
        elif path.startswith("/characters/") and not path.endswith("/session"):
            char_id = path.split("/")[2]
            self._handle_get_character(char_id)
        elif path == "/memory":
            self._handle_query_memory(params)
        elif path.startswith("/relationships/"):
            parts = path.split("/")
            if len(parts) >= 4:
                self._handle_get_relationship(parts[2], parts[3])
            else:
                self._send_error("invalid path", 400)
        else:
            self._send_error("not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/characters":
            body = self._read_body()
            if body:
                self._handle_create_character(body)
        elif path.endswith("/session") and path.startswith("/characters/"):
            char_id = path.split("/")[2]
            self._handle_start_session(char_id)
        elif path == "/record":
            body = self._read_body()
            if body:
                self._handle_record(body)
        else:
            self._send_error("not found", 404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path.endswith("/session") and path.startswith("/characters/"):
            char_id = path.split("/")[2]
            self._handle_end_session(char_id)
        else:
            self._send_error("not found", 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # --- Handlers ---

    def _handle_create_character(self, body: dict):
        engine = _get_engine()
        name = body.get("name", "").strip()
        if not name:
            self._send_error("name required", 400)
            return

        persona = {}
        for key in ("personality", "speaking_style", "occupation", "appearance", "backstory"):
            if key in body:
                persona[key] = body[key]
        if body.get("persona"):
            persona["personality"] = body["persona"]

        char = engine.create_character(
            name=name,
            persona=persona if persona else None,
            character_id=body.get("id"),
            birthdate=body.get("birthdate"),
        )
        self._send_json({"id": char.id, "name": char.name})

    def _handle_list_characters(self):
        engine = _get_engine()
        chars = engine.list_characters()
        self._send_json({"characters": chars})

    def _handle_get_character(self, char_id: str):
        engine = _get_engine()
        char = engine.load_character(char_id)
        if not char:
            self._send_error(f"character {char_id} not found", 404)
            return
        emo = char.emotion
        self._send_json({
            "id": char.id,
            "name": char.name,
            "emotion": emo.to_dict() if emo else None,
            "session_active": _sessions.get(char_id, False),
        })

    def _handle_start_session(self, char_id: str):
        engine = _get_engine()
        char = engine.load_character(char_id)
        if not char:
            self._send_error(f"character {char_id} not found", 404)
            return
        session_id = char.start_session()
        _sessions[char_id] = True
        self._send_json({"session_id": session_id})

    def _handle_end_session(self, char_id: str):
        engine = _get_engine()
        char = engine.load_character(char_id)
        if not char:
            self._send_error(f"character {char_id} not found", 404)
            return
        summary = char.end_session()
        _sessions[char_id] = False
        self._send_json({"summary": summary})

    def _handle_record(self, body: dict):
        """Record a message — triggers chat() for AI processing."""
        engine = _get_engine()
        char_id = body.get("character_id")
        content = body.get("content", "")
        user_id = body.get("user_id", "st_user")
        role = body.get("role", "user")

        if not char_id or not content:
            self._send_error("character_id and content required", 400)
            return

        char = engine.load_character(char_id)
        if not char:
            self._send_error(f"character {char_id} not found", 404)
            return

        # Ensure session is active
        if not _sessions.get(char_id):
            char.start_session()
            _sessions[char_id] = True

        if role == "user":
            # Process user message through woven-imprint's chat
            # This triggers memory formation, relationship updates, etc.
            # We discard the response — ST has its own LLM response
            try:
                char.chat(content, user_id=user_id)
            except Exception:
                pass  # Non-blocking — don't fail the ST pipeline

        self._send_json({"recorded": True})

    def _handle_query_memory(self, params: dict):
        """Query relevant memories for prompt injection."""
        engine = _get_engine()
        char_id = params.get("character_id", [""])[0]
        user_id = params.get("user_id", ["st_user"])[0]
        query = params.get("query", [""])[0]
        limit = int(params.get("limit", ["10"])[0])

        if not char_id:
            self._send_error("character_id required", 400)
            return

        char = engine.load_character(char_id)
        if not char:
            self._send_error(f"character {char_id} not found", 404)
            return

        # Recall memories
        memories = []
        if query:
            memories = char.recall(query, limit=limit)

        # Get relationship with user
        rel = char.relationships.get(user_id)
        rel_desc = ""
        if rel:
            rel_desc = char.relationships.describe(user_id)

        # Format for prompt injection
        context_parts = []
        if rel_desc:
            context_parts.append(f"[Relationship with {user_id}: {rel_desc}]")
        if memories:
            context_parts.append("[Relevant memories:]")
            for m in memories[:5]:
                context_parts.append(f"- {m.get('content', '')[:200]}")

        context = "\n".join(context_parts) if context_parts else ""

        self._send_json({
            "context": context,
            "memories": memories[:limit],
            "relationship": rel,
        })

    def _handle_get_relationship(self, char_id: str, target_id: str):
        engine = _get_engine()
        char = engine.load_character(char_id)
        if not char:
            self._send_error(f"character {char_id} not found", 404)
            return
        rel = char.relationships.get(target_id)
        if rel:
            self._send_json(rel)
        else:
            self._send_json({
                "dimensions": {"trust": 0, "affection": 0, "respect": 0,
                               "familiarity": 0, "tension": 0},
                "type": "stranger",
            })

    # --- Helpers ---

    def _read_body(self, max_size: int = 1_048_576) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0 or length > max_size:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return None

    def _send_error(self, message: str, status: int = 400):
        self._send_json({"error": message}, status)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        pass


def run_sidecar(port: int = 8765, db_path: str | None = None):
    """Start the sidecar API server."""
    global _config
    _config = {"db_path": db_path}

    server = HTTPServer(("127.0.0.1", port), SidecarHandler)
    print(f"Woven Imprint sidecar running on http://127.0.0.1:{port}")
    print("Endpoints: /characters, /record, /memory, /relationships, /health\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Woven Imprint Sidecar")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    run_sidecar(port=args.port, db_path=args.db)
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd ~/sona/projects/woven-imprint && python -m pytest tests/test_sidecar.py -v`
Expected: All pass

- [ ] **Step 5: Add CLI entry point for sidecar**

Modify: `src/woven_imprint/__main__.py` or the CLI to add `woven-imprint sidecar --port 8765`

- [ ] **Step 6: Commit**

```bash
git add src/woven_imprint/server/sidecar.py tests/test_sidecar.py
git commit -m "feat: add sidecar HTTP API for SillyTavern integration"
```

---

## Task 2: SillyTavern Server Plugin

The server plugin runs inside SillyTavern's Node.js process and proxies requests to the Python sidecar. It handles connection management, error recovery, and character auto-creation from ST character cards.

**Files:**
- Create: `sillytavern-woven-imprint/plugin/index.js`
- Create: `sillytavern-woven-imprint/plugin/package.json`

- [ ] **Step 1: Create plugin package.json**

```json
{
  "name": "sillytavern-woven-imprint",
  "version": "1.0.0",
  "description": "Woven Imprint persistent memory bridge for SillyTavern",
  "main": "index.js",
  "author": "virtaava",
  "license": "Apache-2.0"
}
```

- [ ] **Step 2: Implement server plugin**

```javascript
// plugin/index.js
const http = require('http');

const SIDECAR_URL = process.env.WOVEN_IMPRINT_URL || 'http://127.0.0.1:8765';

/**
 * Proxy a request to the woven-imprint sidecar.
 */
function proxySidecar(method, path, body = null) {
    return new Promise((resolve, reject) => {
        const url = new URL(path, SIDECAR_URL);
        const options = {
            hostname: url.hostname,
            port: url.port,
            path: url.pathname + url.search,
            method,
            headers: { 'Content-Type': 'application/json' },
            timeout: 10000,
        };

        const req = http.request(options, (res) => {
            let data = '';
            res.on('data', (chunk) => { data += chunk; });
            res.on('end', () => {
                try {
                    resolve({ status: res.statusCode, body: JSON.parse(data) });
                } catch {
                    resolve({ status: res.statusCode, body: data });
                }
            });
        });

        req.on('error', (err) => reject(err));
        req.on('timeout', () => { req.destroy(); reject(new Error('Sidecar timeout')); });

        if (body) {
            req.write(JSON.stringify(body));
        }
        req.end();
    });
}

/**
 * SillyTavern server plugin entry point.
 */
module.exports = {
    init: async (router) => {
        // Health check (also checks sidecar connectivity)
        router.get('/health', async (req, res) => {
            try {
                const result = await proxySidecar('GET', '/health');
                res.json({ plugin: 'ok', sidecar: result.body });
            } catch (err) {
                res.status(503).json({ plugin: 'ok', sidecar: 'unreachable', error: err.message });
            }
        });

        // Proxy: create character
        router.post('/characters', async (req, res) => {
            try {
                const result = await proxySidecar('POST', '/characters', req.body);
                res.status(result.status).json(result.body);
            } catch (err) {
                res.status(502).json({ error: 'Sidecar unreachable' });
            }
        });

        // Proxy: list characters
        router.get('/characters', async (req, res) => {
            try {
                const result = await proxySidecar('GET', '/characters');
                res.json(result.body);
            } catch (err) {
                res.status(502).json({ error: 'Sidecar unreachable' });
            }
        });

        // Proxy: record message
        router.post('/record', async (req, res) => {
            try {
                const result = await proxySidecar('POST', '/record', req.body);
                res.status(result.status).json(result.body);
            } catch (err) {
                // Non-blocking — don't fail the chat
                res.json({ recorded: false, error: err.message });
            }
        });

        // Proxy: query memory
        router.get('/memory', async (req, res) => {
            try {
                const qs = new URLSearchParams(req.query).toString();
                const result = await proxySidecar('GET', `/memory?${qs}`);
                res.status(result.status).json(result.body);
            } catch (err) {
                // Graceful degradation — return empty context
                res.json({ context: '', memories: [], relationship: null });
            }
        });

        // Proxy: get relationship
        router.get('/relationships/:charId/:targetId', async (req, res) => {
            try {
                const result = await proxySidecar('GET', `/relationships/${req.params.charId}/${req.params.targetId}`);
                res.status(result.status).json(result.body);
            } catch (err) {
                res.json({ dimensions: {}, type: 'unknown' });
            }
        });

        // Proxy: session management
        router.post('/characters/:charId/session', async (req, res) => {
            try {
                const result = await proxySidecar('POST', `/characters/${req.params.charId}/session`);
                res.status(result.status).json(result.body);
            } catch (err) {
                res.status(502).json({ error: 'Sidecar unreachable' });
            }
        });

        router.delete('/characters/:charId/session', async (req, res) => {
            try {
                const result = await proxySidecar('DELETE', `/characters/${req.params.charId}/session`);
                res.status(result.status).json(result.body);
            } catch (err) {
                res.status(502).json({ error: 'Sidecar unreachable' });
            }
        });

        console.log('[Woven Imprint] Server plugin loaded. Sidecar URL:', SIDECAR_URL);
    },

    exit: async () => {
        console.log('[Woven Imprint] Server plugin unloaded.');
    },

    info: {
        id: 'woven-imprint-bridge',
        name: 'Woven Imprint Bridge',
        description: 'Bridges SillyTavern to woven-imprint persistent memory sidecar',
    },
};
```

- [ ] **Step 3: Commit**

```bash
git add plugin/
git commit -m "feat: SillyTavern server plugin — sidecar proxy"
```

---

## Task 3: SillyTavern UI Extension

The UI extension hooks into ST's event system to intercept messages, inject memory context into prompts via `generate_interceptor`, and provide a settings panel.

**Files:**
- Create: `sillytavern-woven-imprint/extension/manifest.json`
- Create: `sillytavern-woven-imprint/extension/index.js`
- Create: `sillytavern-woven-imprint/extension/settings.html`
- Create: `sillytavern-woven-imprint/extension/style.css`

- [ ] **Step 1: Create manifest.json**

```json
{
    "display_name": "Woven Imprint",
    "loading_order": 5,
    "js": "index.js",
    "css": "style.css",
    "author": "virtaava",
    "version": "1.0.0",
    "requires": [],
    "optional": [],
    "generate_interceptor": "wovenImprintInterceptor",
    "auto_update": true
}
```

- [ ] **Step 2: Implement UI extension index.js**

```javascript
// extension/index.js
// Woven Imprint — persistent memory for SillyTavern characters

const PLUGIN_BASE = '/api/plugins/woven-imprint-bridge';
const MODULE_NAME = 'woven_imprint';

// Character ID mapping: ST character name → woven-imprint character ID
let characterMap = {};
let settings = {
    enabled: true,
    sidecarUrl: 'http://127.0.0.1:8765',
    injectionDepth: 4,
    maxMemories: 5,
    showRelationship: true,
    memoryTemplate: '[Character Memory]\n{{memories}}\n[Relationship: {{relationship}}]',
};

/**
 * POST to the server plugin (which proxies to sidecar).
 */
async function pluginFetch(path, method = 'GET', body = null) {
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) options.body = JSON.stringify(body);
    try {
        const resp = await fetch(`${PLUGIN_BASE}${path}`, options);
        return await resp.json();
    } catch (err) {
        console.warn('[Woven Imprint] Plugin fetch failed:', err);
        return null;
    }
}

/**
 * Ensure a woven-imprint character exists for the given ST character.
 * Creates one if needed, using the ST character card data.
 */
async function ensureCharacter(stChar) {
    if (!stChar || !stChar.name) return null;

    const key = stChar.name.toLowerCase().replace(/\s+/g, '_');
    if (characterMap[key]) return characterMap[key];

    // Check if already exists in sidecar
    const list = await pluginFetch('/characters');
    if (list?.characters) {
        const match = list.characters.find(c =>
            c.name.toLowerCase().replace(/\s+/g, '_') === key ||
            c.name.toLowerCase() === stChar.name.toLowerCase()
        );
        if (match) {
            characterMap[key] = match.id;
            return match.id;
        }
    }

    // Create from ST card
    const result = await pluginFetch('/characters', 'POST', {
        name: stChar.name,
        persona: stChar.description || stChar.personality || '',
        personality: stChar.personality || '',
        speaking_style: stChar.mes_example ? 'see examples' : '',
    });

    if (result?.id) {
        characterMap[key] = result.id;
        // Start session
        await pluginFetch(`/characters/${result.id}/session`, 'POST');
        return result.id;
    }
    return null;
}

/**
 * Record a message to woven-imprint (non-blocking).
 */
async function recordMessage(charId, role, content, userId = 'st_user') {
    if (!settings.enabled || !charId) return;
    pluginFetch('/record', 'POST', {
        character_id: charId,
        role,
        content,
        user_id: userId,
    });
    // Fire and forget — don't await
}

/**
 * Query memory context for prompt injection.
 */
async function queryMemory(charId, query = '', userId = 'st_user') {
    if (!settings.enabled || !charId) return '';
    const params = new URLSearchParams({
        character_id: charId,
        user_id: userId,
        query: query || 'recent events',
        limit: String(settings.maxMemories),
    });
    const result = await pluginFetch(`/memory?${params}`);
    return result?.context || '';
}

/**
 * Generate interceptor — called before every LLM request.
 * Injects memory context into the chat array.
 */
globalThis.wovenImprintInterceptor = async function (chat, contextSize, abort, type) {
    if (!settings.enabled) return;
    if (type === 'quiet') return; // Don't inject into quiet prompts

    const context = SillyTavern.getContext();
    const char = context.characters?.[context.characterId];
    if (!char) return;

    const charId = await ensureCharacter(char);
    if (!charId) return;

    // Get the last user message as query
    let query = '';
    for (let i = chat.length - 1; i >= 0; i--) {
        if (chat[i].is_user) {
            query = chat[i].mes?.substring(0, 200) || '';
            break;
        }
    }

    const memoryContext = await queryMemory(charId, query);
    if (!memoryContext) return;

    // Inject as a system message at configured depth
    const injectionMsg = {
        role: 'system',
        content: memoryContext,
        identifier: 'woven_imprint_memory',
    };

    // Insert at depth (from the end of the array)
    const insertIdx = Math.max(0, chat.length - settings.injectionDepth);
    chat.splice(insertIdx, 0, injectionMsg);
};

/**
 * Initialize the extension.
 */
(function init() {
    const context = SillyTavern.getContext();
    const { eventSource, event_types, extensionSettings, saveSettingsDebounced } = context;

    // Load settings
    if (extensionSettings[MODULE_NAME]) {
        Object.assign(settings, extensionSettings[MODULE_NAME]);
    }
    extensionSettings[MODULE_NAME] = settings;

    // On user message: record to sidecar
    eventSource.on(event_types.MESSAGE_SENT, async (messageIndex) => {
        if (!settings.enabled) return;
        const ctx = SillyTavern.getContext();
        const msg = ctx.chat?.[messageIndex];
        if (!msg || !msg.is_user) return;

        const char = ctx.characters?.[ctx.characterId];
        const charId = await ensureCharacter(char);
        recordMessage(charId, 'user', msg.mes);
    });

    // On AI response: record to sidecar
    eventSource.on(event_types.MESSAGE_RECEIVED, async (messageIndex) => {
        if (!settings.enabled) return;
        const ctx = SillyTavern.getContext();
        const msg = ctx.chat?.[messageIndex];
        if (!msg || msg.is_user) return;

        const char = ctx.characters?.[ctx.characterId];
        const charId = await ensureCharacter(char);
        recordMessage(charId, 'assistant', msg.mes);
    });

    // On chat change: reset state
    eventSource.on(event_types.CHAT_CHANGED, async () => {
        // Could pre-load memory for new character here
    });

    // Settings panel
    const settingsHtml = `
        <div class="woven-imprint-settings">
            <div class="inline-drawer">
                <div class="inline-drawer-toggle inline-drawer-header">
                    <b>Woven Imprint</b>
                    <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
                </div>
                <div class="inline-drawer-content">
                    <label class="checkbox_label">
                        <input id="woven_enabled" type="checkbox" ${settings.enabled ? 'checked' : ''} />
                        <span>Enable persistent memory</span>
                    </label>
                    <label>
                        <span>Memory injection depth:</span>
                        <input id="woven_depth" type="number" value="${settings.injectionDepth}" min="1" max="20" />
                    </label>
                    <label>
                        <span>Max memories per query:</span>
                        <input id="woven_max_memories" type="number" value="${settings.maxMemories}" min="1" max="20" />
                    </label>
                    <div id="woven_status" class="woven-status">Checking sidecar...</div>
                </div>
            </div>
        </div>
    `;
    document.getElementById('extensions_settings2')?.insertAdjacentHTML('beforeend', settingsHtml);

    // Bind settings
    document.getElementById('woven_enabled')?.addEventListener('change', (e) => {
        settings.enabled = e.target.checked;
        extensionSettings[MODULE_NAME] = settings;
        saveSettingsDebounced();
    });

    document.getElementById('woven_depth')?.addEventListener('input', (e) => {
        settings.injectionDepth = parseInt(e.target.value) || 4;
        extensionSettings[MODULE_NAME] = settings;
        saveSettingsDebounced();
    });

    document.getElementById('woven_max_memories')?.addEventListener('input', (e) => {
        settings.maxMemories = parseInt(e.target.value) || 5;
        extensionSettings[MODULE_NAME] = settings;
        saveSettingsDebounced();
    });

    // Check sidecar health
    pluginFetch('/health').then(result => {
        const statusEl = document.getElementById('woven_status');
        if (statusEl) {
            if (result?.sidecar?.status === 'ok') {
                statusEl.textContent = 'Sidecar connected';
                statusEl.style.color = '#4caf50';
            } else {
                statusEl.textContent = 'Sidecar not reachable — run: woven-imprint sidecar';
                statusEl.style.color = '#f44336';
            }
        }
    });

    console.log('[Woven Imprint] UI extension loaded.');
})();
```

- [ ] **Step 3: Create style.css**

```css
.woven-imprint-settings .inline-drawer-content {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 8px 0;
}
.woven-imprint-settings label {
    display: flex;
    align-items: center;
    gap: 8px;
}
.woven-imprint-settings input[type="number"] {
    width: 60px;
}
.woven-status {
    font-size: 0.85em;
    margin-top: 4px;
    color: #999;
}
```

- [ ] **Step 4: Commit**

```bash
git add extension/
git commit -m "feat: SillyTavern UI extension — memory injection + event recording"
```

---

## Task 4: Repo Scaffolding and README

**Files:**
- Create: `sillytavern-woven-imprint/README.md`
- Create: `sillytavern-woven-imprint/package.json`
- Create: `sillytavern-woven-imprint/LICENSE`

- [ ] **Step 1: Write README with install instructions**

Key sections:
- What it does (1 paragraph)
- Prerequisites (`pip install woven-imprint`, SillyTavern with server plugins enabled)
- Install steps:
  1. Start sidecar: `woven-imprint sidecar --port 8765`
  2. Copy `plugin/` to SillyTavern's `plugins/woven-imprint-bridge/`
  3. Paste extension URL in ST's "Install Extension" (or copy `extension/` to `third-party/`)
  4. Restart SillyTavern, enable "Woven Imprint" in extensions
- How it works (data flow diagram)
- Configuration options
- Troubleshooting

- [ ] **Step 2: Create GitHub repo**

```bash
cd ~/sona/projects/sillytavern-woven-imprint
git init && git add -A
git commit -m "initial: SillyTavern extension for woven-imprint persistent memory"
gh repo create virtaava/sillytavern-woven-imprint --public --source=. --push
```

- [ ] **Step 3: Commit**

---

## Task 5: Integration Testing

End-to-end test: start sidecar, install extension in a test SillyTavern instance, chat, verify memory persists.

**Files:**
- Create: `sillytavern-woven-imprint/tests/test_integration.sh`

- [ ] **Step 1: Write integration test script**

```bash
#!/bin/bash
# Test: start sidecar, create character, record messages, query memory

SIDECAR_PORT=18765
SIDECAR_PID=""

cleanup() { kill $SIDECAR_PID 2>/dev/null; }
trap cleanup EXIT

# Start sidecar
python -m woven_imprint.server.sidecar --port $SIDECAR_PORT --db /tmp/test_st.db &
SIDECAR_PID=$!
sleep 2

BASE="http://127.0.0.1:$SIDECAR_PORT"

# Health check
curl -sf "$BASE/health" | jq .status || { echo "FAIL: health"; exit 1; }

# Create character
CHAR_ID=$(curl -sf -X POST "$BASE/characters" \
  -H 'Content-Type: application/json' \
  -d '{"name":"Test Alice","persona":"A witty detective"}' | jq -r .id)
echo "Created: $CHAR_ID"

# Start session
curl -sf -X POST "$BASE/characters/$CHAR_ID/session" | jq .

# Record messages
curl -sf -X POST "$BASE/record" \
  -H 'Content-Type: application/json' \
  -d "{\"character_id\":\"$CHAR_ID\",\"role\":\"user\",\"content\":\"I found a clue at the docks\"}"

curl -sf -X POST "$BASE/record" \
  -H 'Content-Type: application/json' \
  -d "{\"character_id\":\"$CHAR_ID\",\"role\":\"assistant\",\"content\":\"The docks? That matches the harbor report.\"}"

# Query memory
curl -sf "$BASE/memory?character_id=$CHAR_ID&query=docks" | jq .

# Check relationship
curl -sf "$BASE/relationships/$CHAR_ID/st_user" | jq .

echo "ALL TESTS PASSED"
```

- [ ] **Step 2: Run integration test**

- [ ] **Step 3: Commit**

---

## Execution Order

1. **Task 1** (Sidecar API) — must be first, everything depends on it
2. **Task 2** (Server Plugin) — depends on Task 1
3. **Task 3** (UI Extension) — depends on Task 2
4. **Task 4** (Repo + README) — can run in parallel with Task 3
5. **Task 5** (Integration test) — depends on all above

## Risks

- **Sidecar latency**: `generate_interceptor` is synchronous — memory query must complete before LLM call starts. Mitigation: cache recent queries, timeout after 2s with empty context.
- **Character deduplication**: ST users may rename characters. Mitigation: `POST /characters` must check for existing character with same name before creating.
- **Group chats**: `characterId` is undefined in group context. Mitigation: use `GROUP_MEMBER_DRAFTED` event to track active speaker, query per-member.
- **Single-threaded sidecar**: `http.server` is blocking — a slow `/record` (LLM fact extraction) blocks `/memory` queries. Mitigation: document as known limitation; future upgrade to `ThreadingHTTPServer` or FastAPI.

---

## Review Resolutions

The following issues were identified during plan review and MUST be addressed during implementation:

### C1/C2: `Character.ingest()` method (CRITICAL — Task 0)

The `/record` endpoint cannot call `char.chat()` because:
1. `chat()` triggers a full LLM generation (wasted — ST has its own response)
2. Memory/relationships are assessed against the *generated* response, not the actual ST AI response
3. AI responses (`role=assistant`) are never processed at all

**Fix:** Add a `Character.ingest()` method to woven-imprint that:
- Stores the message in the conversation buffer (user or assistant)
- Triggers fact extraction on the message pair (user + assistant)
- Triggers relationship assessment
- Does NOT generate an LLM response

This is a new **Task 0** that must be implemented before the sidecar. The `/record` endpoint should accept both user and assistant messages and call `char.ingest()` for each.

### C3: Missing HTTP error responses

All `do_POST` handlers must send `self._send_error("invalid or empty request body", 400)` when `_read_body()` returns `None`.

### I2: Remove `generate_interceptor` from manifest.json

This is not a real SillyTavern manifest field. The interceptor is registered via `globalThis.wovenImprintInterceptor` in `index.js`, which is the correct approach. Remove the field from manifest.json.

### I3/I4: Verify ST chat array format

The `generate_interceptor` function signature and the injected message format must be verified against the target SillyTavern version. ST internal messages use `{ is_user, mes, name }` format, not OpenAI `{ role, content }`. The implementer must check `public/scripts/openai.js` in the ST source.

### I5: Add `sidecar_port` to config.py

Add `sidecar_port: int = 8765` to `ServerConfig` and `WOVEN_IMPRINT_SIDECAR_PORT` env var mapping.

### I6: Name deduplication in `POST /characters`

Check for existing character with same name before creating a new one. Return existing character's ID if found.

### I7: Robust integration test

Add `set -euo pipefail`, check for `jq` availability, clean up temp DB.

### S2: Remove `settings.html` from file structure

Settings UI is inlined in `index.js`. Remove `settings.html` from the file list.

### S3: Remove unused `memoryTemplate` setting

Not implemented — remove from settings object to avoid confusion.
