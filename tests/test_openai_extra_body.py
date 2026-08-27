"""Tests for OpenAILLM extra_body forwarding into chat.completions.create."""

from types import SimpleNamespace

from woven_imprint.llm.openai_llm import OpenAILLM


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content='{"ok": true}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _llm(extra):
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.model = "m"
    llm.extra_body = extra
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    return llm


def test_extra_body_forwarded_when_set():
    llm = _llm({"chat_template_kwargs": {"enable_thinking": False}})
    llm.generate([{"role": "user", "content": "hi"}])
    llm.generate_json([{"role": "user", "content": "hi"}])
    calls = llm.client.chat.completions.calls
    assert all(
        c.get("extra_body") == {"chat_template_kwargs": {"enable_thinking": False}} for c in calls
    )


def test_extra_body_absent_when_none():
    llm = _llm(None)
    llm.generate([{"role": "user", "content": "hi"}])
    assert "extra_body" not in llm.client.chat.completions.calls[0]
