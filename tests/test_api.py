"""Tests for the OpenAI-compatible API server."""

import json
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer

import pytest

import woven_imprint.server.api as api_mod
from woven_imprint import Engine


class FakeLLM:
    def __init__(self):
        self.call_count = 0

    def generate(self, messages, **kw):
        self.call_count += 1
        return "I hear you."

    def generate_json(self, messages, **kw):
        self.call_count += 1
        return {}


class FakeEmbedder:
    def __init__(self):
        self._vocab = {}
        self._next = 0

    def embed(self, text):
        vec = [0.0] * 50
        for word in text.lower().split()[:10]:
            if word not in self._vocab:
                self._vocab[word] = self._next % 50
                self._next += 1
            vec[self._vocab[word]] += 1.0
        mag = sum(x * x for x in vec) ** 0.5
        if mag > 0:
            vec = [x / mag for x in vec]
        return vec

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def dimensions(self):
        return 50


def _make_engine():
    llm = FakeLLM()
    embedder = FakeEmbedder()
    engine = Engine(db_path=":memory:", llm=llm, embedding=embedder)
    orig = engine.create_character

    def _create_seq(*a, **kw):
        c = orig(*a, **kw)
        c.parallel = False
        return c

    engine.create_character = _create_seq
    return engine


@pytest.fixture()
def api_url():
    """Start an API server on a random port and inject an in-memory engine."""
    engine = _make_engine()
    api_mod._engine = engine
    api_mod._config = {
        "db_path": ":memory:",
        "model": "test",
        "api_key": None,
    }

    server = HTTPServer(("127.0.0.1", 0), api_mod.OpenAIHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    engine.close()
    api_mod._engine = None


def _post(url, data=None):
    body = json.dumps(data).encode() if data is not None else b""
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(url):
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestModels:
    def test_models_endpoint(self, api_url):
        """Test that /v1/models lists available models."""
        # Create a character first
        engine = api_mod._engine
        engine.create_character("Alice")

        status, data = _get(f"{api_url}/v1/models")
        assert status == 200
        assert "data" in data
        models = [m["id"] for m in data["data"]]
        assert "alice" in models


class TestChatCompletions:
    def test_chat_completions_basic(self, api_url):
        """Test basic chat completions request."""
        # Create a character
        engine = api_mod._engine
        engine.create_character("Bob")

        status, data = _post(
            f"{api_url}/v1/chat/completions",
            {
                "model": "bob",
                "messages": [{"role": "user", "content": "Hello Bob!"}],
            },
        )
        assert status == 200
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]

    def test_chat_completions_missing_model(self, api_url):
        """Test that missing model returns 400."""
        status, data = _post(
            f"{api_url}/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        assert status == 400
        assert "error" in data

    def test_chat_completions_nonexistent_model(self, api_url):
        """Test that nonexistent model returns 404."""
        status, data = _post(
            f"{api_url}/v1/chat/completions",
            {
                "model": "nonexistent",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        assert status == 404
        assert "error" in data

    def test_chat_completions_no_thread_leak(self, api_url):
        """Regression test: chat request should not leak background worker threads.

        When _handle_chat loads a character and calls char.chat(), it should run
        bookkeeping synchronously, not spawn a background worker thread that gets
        leaked. This test verifies that no new woven-bg-* threads are created
        after a chat completions request.
        """
        # Create a character
        engine = api_mod._engine
        engine.create_character("Charlie")

        # Capture baseline thread count to isolate NEW threads created by this test
        baseline_bg_threads = {
            t.ident for t in threading.enumerate() if t.name.startswith("woven-bg-")
        }

        # Make a chat request with a user_id to trigger relationship assessment
        status, data = _post(
            f"{api_url}/v1/chat/completions",
            {
                "model": "charlie",
                "messages": [
                    {"role": "user", "content": "Hello Charlie!", "user_id": "player_001"}
                ],
            },
        )
        assert status == 200
        assert "choices" in data
        assert data["choices"][0]["message"]["content"]

        # Check no NEW background worker threads leaked from this chat call
        current_bg_threads = {
            t.ident for t in threading.enumerate() if t.name.startswith("woven-bg-")
        }
        new_threads = current_bg_threads - baseline_bg_threads
        assert len(new_threads) == 0, (
            f"Leaked {len(new_threads)} new background worker thread(s) from chat request"
        )
