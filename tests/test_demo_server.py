"""Tests for the FastAPI demo server.

Requires the [demo] extra (fastapi, uvicorn) and httpx.
Skipped automatically if not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Skip entire module if demo dependencies aren't installed
pytest.importorskip("fastapi", reason="demo extras not installed")
pytest.importorskip("httpx", reason="httpx not installed")

# Ensure tests/ helpers are importable
sys.path.insert(0, str(Path(__file__).parent))

from helpers import make_test_engine
from woven_imprint.server.demo import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app_client():
    """Authed test client against a test app."""
    from starlette.testclient import TestClient

    engine = make_test_engine()
    app, token = create_app(engine=engine)
    client = TestClient(app, base_url="http://127.0.0.1:7860")
    client.headers["Authorization"] = f"Bearer {token}"
    yield client, token, engine
    engine.close()


@pytest.fixture()
def unauthed_client(app_client):
    """Client with no auth header (shares same app instance)."""
    from starlette.testclient import TestClient

    client, _token, _engine = app_client
    unauthed = TestClient(client.app, base_url="http://127.0.0.1:7860")
    yield unauthed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_character(client, name="TestChar"):
    """Create a character via the API and return the response dict."""
    resp = client.post("/api/characters", json={"name": name})
    assert resp.status_code in (200, 201)
    return resp.json()


# ---------------------------------------------------------------------------
# TestHealth
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_no_auth(self, unauthed_client):
        """Health endpoint works without auth."""
        resp = unauthed_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_with_auth(self, app_client):
        client, _token, _engine = app_client
        resp = client.get("/api/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TestAuth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_missing_auth_rejected(self, unauthed_client):
        """Requests without Bearer token get 401."""
        resp = unauthed_client.get("/api/characters")
        assert resp.status_code == 401

    def test_wrong_token_rejected(self, app_client):
        """Requests with wrong token get 401."""
        from starlette.testclient import TestClient

        client, _token, _engine = app_client
        bad = TestClient(client.app, base_url="http://127.0.0.1:7860")
        bad.headers["Authorization"] = "Bearer wrong-token-value"
        resp = bad.get("/api/characters")
        assert resp.status_code == 401

    def test_valid_auth_accepted(self, app_client):
        client, _token, _engine = app_client
        resp = client.get("/api/characters")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TestCORS
# ---------------------------------------------------------------------------

class TestCORS:
    def test_evil_origin_no_cors_header(self, app_client):
        """Requests from evil.com should not get CORS allow header."""
        client, _token, _engine = app_client
        resp = client.options(
            "/api/characters",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        allow = resp.headers.get("access-control-allow-origin", "")
        assert "evil.com" not in allow

    def test_localhost_origin_allowed(self, app_client):
        client, _token, _engine = app_client
        resp = client.options(
            "/api/characters",
            headers={
                "Origin": "http://127.0.0.1:7860",
                "Access-Control-Request-Method": "GET",
            },
        )
        allow = resp.headers.get("access-control-allow-origin", "")
        assert "127.0.0.1" in allow


# ---------------------------------------------------------------------------
# TestCharacterEndpoints
# ---------------------------------------------------------------------------

class TestCharacterEndpoints:
    def test_create_and_list(self, app_client):
        client, _token, _engine = app_client
        # Create
        resp = client.post("/api/characters", json={"name": "Alice"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Alice"
        assert data["created"] is True

        # Duplicate returns 200
        resp2 = client.post("/api/characters", json={"name": "Alice"})
        assert resp2.status_code == 200
        assert resp2.json()["created"] is False

        # List
        resp3 = client.get("/api/characters")
        assert resp3.status_code == 200
        names = [c["name"] for c in resp3.json()["characters"]]
        assert "Alice" in names

    def test_get_character_state(self, app_client):
        client, _token, _engine = app_client
        created = _create_test_character(client, "Bob")
        char_id = created["id"]

        resp = client.get(f"/api/characters/{char_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Bob"
        assert "emotion" in data
        assert "arc" in data
        assert "phase" in data["arc"]
        assert "tension" in data["arc"]

    def test_get_nonexistent_character(self, app_client):
        client, _token, _engine = app_client
        resp = client.get("/api/characters/nonexistent-id-999")
        assert resp.status_code == 404

    def test_session_lifecycle(self, app_client):
        client, _token, _engine = app_client
        created = _create_test_character(client, "Carol")
        char_id = created["id"]

        # Start session
        resp = client.post(f"/api/characters/{char_id}/session")
        assert resp.status_code == 200
        assert "session_id" in resp.json()

        # End session
        resp2 = client.delete(f"/api/characters/{char_id}/session")
        assert resp2.status_code == 200

    def test_record_message(self, app_client):
        client, _token, _engine = app_client
        created = _create_test_character(client, "Dave")
        char_id = created["id"]

        resp = client.post("/api/record", json={
            "character_id": char_id,
            "role": "user",
            "content": "Hello Dave!",
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


# ---------------------------------------------------------------------------
# TestProviderConfig
# ---------------------------------------------------------------------------

class TestProviderConfig:
    def test_get_provider_redacts_key(self, app_client):
        """Provider config response must not leak the actual API key."""
        client, _token, _engine = app_client
        resp = client.get("/api/config/provider")
        assert resp.status_code == 200
        data = resp.json()
        assert "api_key_configured" in data
        assert isinstance(data["api_key_configured"], bool)
        assert "api_key" not in data
        assert "provider" in data
        assert "model" in data


# ---------------------------------------------------------------------------
# TestXRaySanitization
# ---------------------------------------------------------------------------

class TestXRaySanitization:
    def test_character_state_no_path_leak(self, app_client):
        """Character state must not leak filesystem paths."""
        client, _token, engine = app_client
        created = _create_test_character(client, "Sentinel")
        char_id = created["id"]

        resp = client.get(f"/api/characters/{char_id}")
        assert resp.status_code == 200
        assert "/home/" not in resp.text

    def test_character_state_no_token_leak(self, app_client):
        """Character state must not leak the bearer token."""
        client, token, _engine = app_client
        created = _create_test_character(client, "Guard")
        char_id = created["id"]

        resp = client.get(f"/api/characters/{char_id}")
        assert token not in resp.text

    def test_provider_config_no_sk_leak(self, app_client):
        """Provider config must not leak sk- prefixed keys."""
        client, _token, _engine = app_client
        resp = client.get("/api/config/provider")
        assert "sk-" not in resp.text


# ---------------------------------------------------------------------------
# TestChatCompletions
# ---------------------------------------------------------------------------

class TestChatCompletions:
    def test_chat_completions_basic(self, app_client):
        """OpenAI-compatible chat endpoint returns valid structure."""
        client, _token, _engine = app_client
        _create_test_character(client, "Echo")

        resp = client.post("/v1/chat/completions", json={
            "model": "Echo",
            "messages": [{"role": "user", "content": "Hello Echo!"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "Echo"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert "usage" in data
        assert "woven_imprint" in data
        assert data["woven_imprint"]["character"] == "Echo"

    def test_chat_completions_no_character(self, app_client):
        client, _token, _engine = app_client
        resp = client.post("/v1/chat/completions", json={
            "model": "nonexistent-character",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 404
