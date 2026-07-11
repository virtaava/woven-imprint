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
        json.dumps({"message": {"content": ""}, "done": True}),
    ]

    class FakeResp:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(line.encode() for line in lines)

    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResp())
    llm = OllamaLLM(model="test")
    chunks = list(llm.generate_stream([{"role": "user", "content": "hi"}]))
    assert "".join(chunks) == "Hello"
