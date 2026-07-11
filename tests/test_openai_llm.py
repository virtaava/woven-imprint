"""Tests for OpenAILLM system-message merging (no SDK/network required).

Strict OpenAI-compatible servers (vLLM's chat template, our primary
deployment) reject requests with more than one system-role message, but
`Character._build_context` intentionally emits two leading system messages
(stable persona prefix + volatile block). `_merge_system_messages` coalesces
those at the provider boundary.
"""

from woven_imprint.llm.openai_llm import _merge_system_messages


def test_two_leading_system_messages_merge_stable_first():
    messages = [
        {"role": "system", "content": "stable persona prefix"},
        {"role": "system", "content": "volatile emotion/arc block"},
        {"role": "user", "content": "hello"},
    ]

    merged = _merge_system_messages(messages)

    assert len(merged) == 2
    assert merged[0]["role"] == "system"
    assert merged[0]["content"] == "stable persona prefix\n\nvolatile emotion/arc block"
    assert merged[1] == {"role": "user", "content": "hello"}


def test_system_message_after_user_message_not_merged():
    messages = [
        {"role": "system", "content": "stable persona prefix"},
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "an injected mid-conversation system note"},
        {"role": "assistant", "content": "hi there"},
    ]

    merged = _merge_system_messages(messages)

    # Only the leading run is a candidate for merging; with a single leading
    # system message there's nothing to merge, and the later system message
    # (after a user message) must be left exactly where it was.
    assert merged == messages


def test_single_leading_system_message_untouched():
    messages = [
        {"role": "system", "content": "only one"},
        {"role": "user", "content": "hi"},
    ]

    assert _merge_system_messages(messages) == messages


def test_no_system_messages_untouched():
    messages = [{"role": "user", "content": "hi"}]

    assert _merge_system_messages(messages) == messages
