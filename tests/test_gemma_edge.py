"""Tests for the Gemma edge adapter-facing LLM provider."""

from __future__ import annotations

import pytest

from woven_imprint.llm.gemma_edge import GemmaEdgeLLM


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_generate_uses_content_field(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        return _FakeResponse({"content": "  Miss Vale lowers her voice.  "})

    monkeypatch.setattr("woven_imprint.llm.gemma_edge.requests.post", fake_post)

    llm = GemmaEdgeLLM(model="gemma-3n", base_url="http://127.0.0.1:8766", timeout=9)
    text = llm.generate([{"role": "user", "content": "Tell me about the drawer."}])

    assert text == "Miss Vale lowers her voice."
    assert calls == [
        (
            "http://127.0.0.1:8766/generate",
            {
                "model": "gemma-3n",
                "messages": [{"role": "user", "content": "Tell me about the drawer."}],
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            9,
        )
    ]


def test_generate_json_returns_direct_json_payload(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        assert url == "http://127.0.0.1:8766/generate_json"
        return _FakeResponse({"mood": "guarded", "intensity": 0.6})

    monkeypatch.setattr("woven_imprint.llm.gemma_edge.requests.post", fake_post)

    llm = GemmaEdgeLLM(model="gemma-3n", base_url="http://127.0.0.1:8766")
    result = llm.generate_json([{"role": "user", "content": "How does she feel?"}])

    assert result == {"mood": "guarded", "intensity": 0.6}


def test_generate_json_accepts_direct_list_payload(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        assert url == "http://127.0.0.1:8766/generate_json"
        return _FakeResponse([{"topic": "drawer"}, {"topic": "paper"}])

    monkeypatch.setattr("woven_imprint.llm.gemma_edge.requests.post", fake_post)

    llm = GemmaEdgeLLM(model="gemma-3n", base_url="http://127.0.0.1:8766")
    result = llm.generate_json([{"role": "user", "content": "List the topics."}])

    assert result == [{"topic": "drawer"}, {"topic": "paper"}]


def test_generate_json_falls_back_to_embedded_code_block(monkeypatch):
    responses = [
        RuntimeError("adapter JSON path unavailable"),
        _FakeResponse({"content": '```json\n{"stance": "guarded", "mentions": ["drawer"]}\n```'}),
    ]

    def fake_post(url, json=None, timeout=None):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("woven_imprint.llm.gemma_edge.requests.post", fake_post)

    llm = GemmaEdgeLLM(model="gemma-3n", base_url="http://127.0.0.1:8766")
    result = llm.generate_json([{"role": "user", "content": "Return structured stance."}])

    assert result == {"stance": "guarded", "mentions": ["drawer"]}


def test_requires_base_url_or_env(monkeypatch):
    monkeypatch.delenv("WOVEN_IMPRINT_GEMMA_EDGE_URL", raising=False)
    monkeypatch.delenv("WOVEN_IMPRINT_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="requires a base_url"):
        GemmaEdgeLLM(model="gemma-3n")
