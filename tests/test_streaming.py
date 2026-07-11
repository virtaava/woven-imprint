import json

from tests.helpers import FakeLLM


def test_default_stream_yields_single_chunk():
    llm = FakeLLM()
    # FakeLLM doesn't define generate_stream — default must fall back to generate()
    from woven_imprint.llm.base import LLMProvider

    chunks = list(LLMProvider.generate_stream(llm, [{"role": "user", "content": "hi"}]))
    assert chunks == ["I hear you."]


def test_ollama_stream_parses_ndjson(monkeypatch):
    from woven_imprint.llm.ollama import OllamaLLM

    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}),
        json.dumps({"message": {"content": "lo"}, "done": False}),
        json.dumps({"message": None}),  # message: null case
        json.dumps({"message": {"content": ""}, "done": True}),
        "[]",  # bare array (not dict) case
    ]

    class FakeResp:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(line.encode() for line in lines)

        def close(self):
            pass

    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResp())
    llm = OllamaLLM(model="test")
    chunks = list(llm.generate_stream([{"role": "user", "content": "hi"}]))
    assert "".join(chunks) == "Hello"


def test_ollama_stream_closes_response_on_abandonment(monkeypatch):
    """Verify response.close() is called even when caller abandons generator early."""
    from woven_imprint.llm.ollama import OllamaLLM
    import gc

    lines = [
        json.dumps({"message": {"content": "First"}, "done": False}),
        json.dumps({"message": {"content": "Second"}, "done": False}),
        json.dumps({"message": {"content": "Third"}, "done": True}),
    ]

    close_called = False

    class FakeResp:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(line.encode() for line in lines)

        def close(self):
            nonlocal close_called
            close_called = True

    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResp())
    llm = OllamaLLM(model="test")
    gen = llm.generate_stream([{"role": "user", "content": "hi"}])

    # Consume only the first chunk
    first = next(gen)
    assert first == "First"

    # Close the generator (simulating early abandonment)
    gen.close()
    gc.collect()

    assert close_called, "resp.close() should have been called"


class StreamingFakeLLM(FakeLLM):
    def generate_stream(self, messages, **kw):
        for chunk in ["I ", "hear ", "you."]:
            yield chunk


def test_chat_stream_yields_and_bookkeeps():
    from tests.helpers import FakeEmbedder
    from woven_imprint.engine import Engine

    engine = Engine(db_path=":memory:", llm=StreamingFakeLLM(), embedding=FakeEmbedder())
    char = engine.create_character("Streamy")
    char.background = False  # sync bookkeeping for deterministic assertions
    char.parallel = False

    chunks = list(char.chat_stream("hello", user_id="u1"))
    assert "".join(chunks) == "I hear you."
    # Same post-processing as chat(): buffers + memory + relationship
    assert char.get_relationship("u1") is not None
    buffer = char.memory.get_all(tier="buffer", limit=10)
    assert any("I hear you." in m["content"] for m in buffer)
    assert char._context.turn_count == 2


class FailingStreamFakeLLM(FakeLLM):
    """LLM that yields one chunk then raises RuntimeError."""

    def generate_stream(self, messages, **kw):
        yield "Partial"
        raise RuntimeError("Stream generation failed")


def test_chat_stream_records_metrics_on_failure():
    """Verify that metrics are recorded when stream generation fails."""
    import pytest

    from tests.helpers import FakeEmbedder
    from woven_imprint.engine import Engine

    engine = Engine(db_path=":memory:", llm=FailingStreamFakeLLM(), embedding=FakeEmbedder())
    char = engine.create_character("Failing")
    char.background = False
    char.enforce_consistency = False

    # Consume the generator with pytest.raises
    with pytest.raises(RuntimeError, match="Stream generation failed"):
        list(char.chat_stream("hello"))

    # Verify metrics were recorded despite the failure
    assert "total_ms" in char.last_chat_metrics
    assert char.last_chat_metrics["total_ms"] > 0
