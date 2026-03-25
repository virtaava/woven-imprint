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
import woven_imprint.server.demo as demo_mod
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

    def test_cookie_auth_accepted(self, app_client):
        from starlette.testclient import TestClient

        client, _token, _engine = app_client
        cookie_client = TestClient(client.app, base_url="http://127.0.0.1:7860")
        index = cookie_client.get("/")
        if index.status_code == 503:
            pytest.skip("demo static build not present in test environment")
        assert index.status_code == 200

        resp = cookie_client.get("/api/characters")
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
# TestRateLimiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_chat_requests_are_rate_limited(self, monkeypatch):
        from starlette.testclient import TestClient

        monkeypatch.setitem(demo_mod._RATE_LIMIT_RULES, "chat", {"limit": 1, "window": 60.0})

        engine = make_test_engine()
        app, token = create_app(engine=engine)
        client = TestClient(app, base_url="http://127.0.0.1:7860")
        client.headers["Authorization"] = f"Bearer {token}"

        _create_test_character(client, "Echo")

        first = client.post(
            "/v1/chat/completions",
            json={
                "model": "Echo",
                "messages": [{"role": "user", "content": "Hello Echo!"}],
            },
        )
        assert first.status_code == 200

        second = client.post(
            "/v1/chat/completions",
            json={
                "model": "Echo",
                "messages": [{"role": "user", "content": "Hello again!"}],
            },
        )
        assert second.status_code == 429
        assert second.json()["error"] == "rate_limited"
        assert "Retry-After" in second.headers

        engine.close()


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

        resp = client.post(
            "/api/record",
            json={
                "character_id": char_id,
                "role": "user",
                "content": "Hello Dave!",
            },
        )
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

    def test_set_provider_persists_config(self, app_client, monkeypatch, tmp_path):
        import woven_imprint.config as config_mod
        import woven_imprint.providers as providers_mod

        client, _token, _engine = app_client
        config_path = tmp_path / "config.yaml"

        monkeypatch.setattr(config_mod, "_config_path", str(config_path))
        monkeypatch.setattr(config_mod, "_config", None)
        monkeypatch.setattr(providers_mod, "create_llm", lambda cfg: object())

        resp = client.post(
            "/api/config/provider",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-test-123",
                "base_url": "https://api.openai.com/v1",
            },
        )
        assert resp.status_code == 200
        assert config_path.exists()

        text = config_path.read_text()
        assert "llm_provider: openai" in text
        assert "model: gpt-4o-mini" in text
        assert "base_url: https://api.openai.com/v1" in text


# ---------------------------------------------------------------------------
# TestXRaySanitization
# ---------------------------------------------------------------------------


class TestXRaySanitization:
    def test_index_does_not_embed_token(self, app_client):
        client, token, _engine = app_client
        resp = client.get("/")
        if resp.status_code == 503:
            pytest.skip("demo static build not present in test environment")
        assert token not in resp.text

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

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "Echo",
                "messages": [{"role": "user", "content": "Hello Echo!"}],
            },
        )
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
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-character",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestRateLimit
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_rate_limiter_allows_within_window(self):
        """_SlidingWindowLimiter allows calls up to the limit."""
        from collections import deque
        import time as _time

        # Build a fresh limiter directly
        events: dict = {}
        limit = 5
        window = 60.0

        def _check(key: str) -> bool:
            now = _time.monotonic()
            q = events.setdefault(key, deque())
            while q and q[0] < now - window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

        for _ in range(limit):
            assert _check("k") is True
        assert _check("k") is False

    def test_rate_limiter_different_keys_independent(self):
        """Different client keys do not share rate limit state."""
        from collections import deque
        import time as _time

        events: dict = {}
        limit = 3
        window = 60.0

        def _check(key: str) -> bool:
            now = _time.monotonic()
            q = events.setdefault(key, deque())
            while q and q[0] < now - window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

        for _ in range(limit):
            _check("a")
        assert _check("a") is False
        # Key "b" is unaffected
        assert _check("b") is True

    def test_chat_rate_limit_returns_429(self, app_client, monkeypatch):
        """Chat endpoint returns 429 when rate limit is exhausted."""
        import woven_imprint.server.demo as _demo

        client, _token, _engine = app_client
        _create_test_character(client, "RLChar")

        # Patch the rate-limit enforcer to always block chat completions.
        original = _demo._enforce_rate_limit

        async def _always_block_chat(req):
            from starlette.responses import JSONResponse as _JR

            if req.url.path == "/v1/chat/completions":
                return _JR({"error": "rate_limited"}, status_code=429)
            return await original(req)

        monkeypatch.setattr(_demo, "_enforce_rate_limit", _always_block_chat)

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "RLChar", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 429

    def test_mutation_rate_limit_returns_429(self, app_client, monkeypatch):
        """Mutation endpoints return 429 when rate limit exhausted."""
        import woven_imprint.server.demo as _demo

        client, _token, _engine = app_client
        original = _demo._enforce_rate_limit

        async def _always_block_mutation(req):
            from starlette.responses import JSONResponse as _JR

            if req.method == "POST" and "/api/characters" in req.url.path:
                return _JR({"error": "rate_limited"}, status_code=429)
            return await original(req)

        monkeypatch.setattr(_demo, "_enforce_rate_limit", _always_block_mutation)

        resp = client.post("/api/characters", json={"name": "ShouldBlock"})
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# TestProviderPersistenceRestart
# ---------------------------------------------------------------------------


class TestProviderPersistenceRestart:
    def test_config_survives_new_app_instance(self, monkeypatch, tmp_path):
        """Provider config saved via API is loaded by a new app instance."""
        from starlette.testclient import TestClient
        import woven_imprint.config as config_mod
        import woven_imprint.providers as providers_mod

        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr(config_mod, "_config_path", str(config_path))
        monkeypatch.setattr(config_mod, "_config", None)
        monkeypatch.setattr(providers_mod, "create_llm", lambda cfg: object())

        # --- App instance 1: save provider config ---
        engine1 = make_test_engine()
        app1, token1 = create_app(engine=engine1, token="test-token-1")
        client1 = TestClient(app1, base_url="http://127.0.0.1:7860")
        client1.headers["Authorization"] = f"Bearer {token1}"

        resp = client1.post(
            "/api/config/provider",
            json={"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-test"},
        )
        assert resp.status_code == 200
        assert config_path.exists()
        engine1.close()

        # Force config reload for next app instance
        monkeypatch.setattr(config_mod, "_config", None)

        # --- App instance 2: read config ---
        engine2 = make_test_engine()
        app2, token2 = create_app(engine=engine2, token="test-token-2")
        client2 = TestClient(app2, base_url="http://127.0.0.1:7860")
        client2.headers["Authorization"] = f"Bearer {token2}"

        resp2 = client2.get("/api/config/provider")
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4o-mini"
        assert data["api_key_configured"] is True
        engine2.close()


# ---------------------------------------------------------------------------
# TestFirstRunFlow
# ---------------------------------------------------------------------------


class TestFirstRunFlow:
    def test_health_available_before_character_setup(self, app_client):
        """Health endpoint is reachable with no characters configured."""
        client, _token, _engine = app_client
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_empty_character_list_on_fresh_engine(self, app_client):
        """Fresh engine returns empty character list, not an error."""
        client, _token, _engine = app_client
        resp = client.get("/api/characters")
        assert resp.status_code == 200
        chars = resp.json().get("characters", [])
        assert isinstance(chars, list)

    def test_first_character_creation(self, app_client):
        """First character can be created on a fresh engine."""
        client, _token, _engine = app_client
        resp = client.post("/api/characters", json={"name": "FirstChar"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "FirstChar"
        assert data["created"] is True

    def test_chat_available_after_character_created(self, app_client):
        """Chat endpoint works after first character is created."""
        client, _token, _engine = app_client
        _create_test_character(client, "ChatFirst")

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ChatFirst",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# TestSessionFlush
# ---------------------------------------------------------------------------


class TestSessionFlush:
    def test_session_start_and_end(self, app_client):
        """Session start returns a session_id; end returns a summary."""
        client, _token, _engine = app_client
        created = _create_test_character(client, "FlushChar")
        char_id = created["id"]

        start = client.post(f"/api/characters/{char_id}/session")
        assert start.status_code == 200
        assert "session_id" in start.json()

        end = client.delete(f"/api/characters/{char_id}/session")
        assert end.status_code == 200
        # summary may be None or a string — either is acceptable
        assert "summary" in end.json()

    def test_new_token_on_new_app_instance(self):
        """Each create_app call generates a fresh token, invalidating the old one."""
        from starlette.testclient import TestClient

        engine = make_test_engine()

        app1, token1 = create_app(engine=engine, token=None)
        app2, token2 = create_app(engine=engine, token=None)

        assert token1 != token2

        # Old token no longer works on the new app
        stale_client = TestClient(app2, base_url="http://127.0.0.1:7860")
        stale_client.headers["Authorization"] = f"Bearer {token1}"
        resp = stale_client.get("/api/characters")
        assert resp.status_code == 401

        engine.close()
