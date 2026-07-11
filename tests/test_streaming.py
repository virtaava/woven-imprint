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
