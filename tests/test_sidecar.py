"""Tests for the sidecar HTTP API."""

import json
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer

import pytest

from woven_imprint import Engine
from woven_imprint.server.sidecar import SidecarHandler


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


class FakeLLM:
    def __init__(self):
        self.call_count = 0

    def generate(self, messages, **kw):
        self.call_count += 1
        return "I hear you."

    def generate_json(self, messages, **kw):
        self.call_count += 1
        user = messages[-1].get("content", "") if messages else ""
        system = messages[0].get("content", "") if messages else ""

        if "extract" in system.lower() or "facts" in user.lower():
            return ["The user mentioned something notable"]

        if "relationship" in system.lower() or "trust" in user.lower():
            return {
                "trust": 0.03,
                "affection": 0.01,
                "respect": 0.02,
                "familiarity": 0.05,
                "tension": 0.0,
            }

        return {}


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
def sidecar_url():
    """Start a sidecar server on a random port and inject an in-memory engine."""
    import woven_imprint.server.sidecar as mod

    engine = _make_engine()
    mod._engine = engine

    server = HTTPServer(("127.0.0.1", 0), SidecarHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    engine.close()
    mod._engine = None


def _get(url):
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


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


def _delete(url):
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post_empty(url):
    """POST with no body at all (Content-Length 0)."""
    req = urllib.request.Request(
        url,
        data=b"",
        headers={"Content-Type": "application/json", "Content-Length": "0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestHealth:
    def test_health_returns_ok(self, sidecar_url):
        status, data = _get(f"{sidecar_url}/health")
        assert status == 200
        assert data["status"] == "ok"
        assert "version" in data


class TestCharacters:
    def test_create_and_list(self, sidecar_url):
        status, data = _post(f"{sidecar_url}/characters", {"name": "Alice"})
        assert status == 201
        assert data["created"] is True
        char_id = data["id"]
        assert data["name"] == "Alice"

        status, data = _get(f"{sidecar_url}/characters")
        assert status == 200
        names = [c["name"] for c in data["characters"]]
        assert "Alice" in names

        status, data = _get(f"{sidecar_url}/characters/{char_id}")
        assert status == 200
        assert data["name"] == "Alice"
        assert "emotion" in data

    def test_name_deduplication(self, sidecar_url):
        s1, d1 = _post(f"{sidecar_url}/characters", {"name": "Bob"})
        assert s1 == 201
        assert d1["created"] is True
        bob_id = d1["id"]

        s2, d2 = _post(f"{sidecar_url}/characters", {"name": "Bob"})
        assert s2 == 200
        assert d2["created"] is False
        assert d2["id"] == bob_id

        # Case-insensitive dedup
        s3, d3 = _post(f"{sidecar_url}/characters", {"name": "bob"})
        assert d3["id"] == bob_id
        assert d3["created"] is False

    def test_get_nonexistent(self, sidecar_url):
        status, data = _get(f"{sidecar_url}/characters/nope-123")
        assert status == 404
        assert "error" in data

    def test_create_missing_body(self, sidecar_url):
        status, data = _post_empty(f"{sidecar_url}/characters")
        assert status == 400
        assert "error" in data

    def test_create_missing_name(self, sidecar_url):
        status, data = _post(f"{sidecar_url}/characters", {"persona": {}})
        assert status == 400


class TestSession:
    def test_start_and_end_session(self, sidecar_url):
        _, d = _post(f"{sidecar_url}/characters", {"name": "Carol"})
        cid = d["id"]

        status, data = _post(f"{sidecar_url}/characters/{cid}/session")
        assert status == 200
        assert "session_id" in data

        status, data = _delete(f"{sidecar_url}/characters/{cid}/session")
        assert status == 200
        assert "summary" in data

    def test_session_nonexistent_character(self, sidecar_url):
        status, _ = _post(f"{sidecar_url}/characters/nope/session")
        assert status == 404


class TestRecord:
    def test_record_user_message(self, sidecar_url):
        _, d = _post(f"{sidecar_url}/characters", {"name": "Diana"})
        cid = d["id"]

        status, data = _post(
            f"{sidecar_url}/record",
            {"character_id": cid, "role": "user", "content": "Hello Diana!"},
        )
        assert status == 200
        assert data["ok"] is True

    def test_record_assistant_message(self, sidecar_url):
        _, d = _post(f"{sidecar_url}/characters", {"name": "Eve"})
        cid = d["id"]

        status, data = _post(
            f"{sidecar_url}/record",
            {
                "character_id": cid,
                "role": "assistant",
                "content": "Hello, nice to meet you!",
            },
        )
        assert status == 200
        assert data["ok"] is True

    def test_record_invalid_role(self, sidecar_url):
        _, d = _post(f"{sidecar_url}/characters", {"name": "Fiona"})
        cid = d["id"]

        status, data = _post(
            f"{sidecar_url}/record",
            {"character_id": cid, "role": "system", "content": "sneaky"},
        )
        assert status == 400

    def test_record_missing_body(self, sidecar_url):
        status, data = _post_empty(f"{sidecar_url}/record")
        assert status == 400
        assert "error" in data

    def test_record_missing_fields(self, sidecar_url):
        status, data = _post(f"{sidecar_url}/record", {"character_id": "x"})
        assert status == 400

    def test_record_nonexistent_character(self, sidecar_url):
        status, data = _post(
            f"{sidecar_url}/record",
            {"character_id": "nope", "role": "user", "content": "hello"},
        )
        assert status == 404


class TestMemory:
    def test_memory_query(self, sidecar_url):
        _, d = _post(f"{sidecar_url}/characters", {"name": "Grace"})
        cid = d["id"]

        # Ingest some content first
        _post(
            f"{sidecar_url}/record",
            {"character_id": cid, "role": "user", "content": "I love painting landscapes"},
        )

        status, data = _get(f"{sidecar_url}/memory?character_id={cid}&query=painting")
        assert status == 200
        assert "memories" in data

    def test_memory_query_missing_params(self, sidecar_url):
        status, data = _get(f"{sidecar_url}/memory?character_id=x")
        assert status == 400

    def test_memory_query_nonexistent_character(self, sidecar_url):
        status, data = _get(f"{sidecar_url}/memory?character_id=nope&query=hello")
        assert status == 404


class TestRelationship:
    def test_relationship_query(self, sidecar_url):
        _, d = _post(f"{sidecar_url}/characters", {"name": "Hannah"})
        cid = d["id"]

        # Ingest with user_id to create relationship
        _post(
            f"{sidecar_url}/record",
            {
                "character_id": cid,
                "role": "user",
                "content": "I trust you completely",
                "user_id": "player1",
            },
        )

        status, data = _get(f"{sidecar_url}/relationships/{cid}/player1")
        assert status == 200
        assert "relationship" in data

    def test_relationship_no_existing(self, sidecar_url):
        _, d = _post(f"{sidecar_url}/characters", {"name": "Iris"})
        cid = d["id"]

        status, data = _get(f"{sidecar_url}/relationships/{cid}/nobody")
        assert status == 200
        assert data["relationship"] is None

    def test_relationship_nonexistent_character(self, sidecar_url):
        status, data = _get(f"{sidecar_url}/relationships/nope/player1")
        assert status == 404


class TestCORS:
    def test_cors_header_on_json_response(self, sidecar_url):
        req = urllib.request.Request(f"{sidecar_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_options_preflight(self, sidecar_url):
        req = urllib.request.Request(f"{sidecar_url}/characters", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 204
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
