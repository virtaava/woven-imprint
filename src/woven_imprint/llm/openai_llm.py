"""OpenAI-compatible LLM provider — works with OpenAI, Azure, and compatible APIs."""

from __future__ import annotations

import json
import re

from .base import LLMProvider


def _merge_system_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge a leading run of system-role messages into a single message.

    Strict OpenAI-compatible servers (observed: vLLM's OpenAI shim in front
    of Qwen3.5-35B-A3B-FP8 — our primary deployment) reject any request with
    more than one system-role message ("System message must be at the
    beginning."). `Character._build_context` intentionally emits two leading
    system messages — a stable persona prefix plus a volatile block
    (emotion/arc/relationship/memories) — kept separate for provider
    prefix-caching. This merges only the *leading* run of system messages,
    joining their content with "\\n\\n" (stable prefix first, preserving
    order); a system message that appears after a non-system message (e.g.
    injected mid-conversation) is left untouched.
    """
    out: list[dict[str, str]] = []
    i = 0
    n = len(messages)
    leading_system: list[dict[str, str]] = []
    while i < n and messages[i].get("role") == "system":
        leading_system.append(messages[i])
        i += 1

    if len(leading_system) > 1:
        merged = dict(leading_system[0])
        merged["content"] = "\n\n".join(m.get("content", "") for m in leading_system)
        out.append(merged)
    else:
        out.extend(leading_system)

    out.extend(messages[i:])
    return out


class OpenAILLM(LLMProvider):
    """Generate completions via OpenAI-compatible API.

    Works with: OpenAI, Azure OpenAI, vLLM, llama.cpp server, LiteLLM, etc.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        extra_body: dict | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install woven-imprint[openai]"
            )

        kwargs: dict = {"timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self.model = model
        # Forwarded verbatim into every chat.completions.create call, e.g.
        # {"chat_template_kwargs": {"enable_thinking": False}} for vLLM's
        # Qwen3.5 reasoning toggle. None/empty means "omit the kwarg".
        self.extra_body = extra_body

    def generate(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ) -> str:
        from .resilience import resilient_call

        messages = _merge_system_messages(messages)
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = resilient_call(
            self.client.chat.completions.create,
            **kwargs,
            provider_name="openai",
        )
        return response.choices[0].message.content or ""

    def generate_stream(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ):
        from .resilience import resilient_call

        messages = _merge_system_messages(messages)
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        stream = resilient_call(
            self.client.chat.completions.create,
            **kwargs,
            provider_name="openai",
        )
        for event in stream:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield delta

    def generate_json(
        self, messages: list[dict[str, str]], temperature: float = 0.3, max_tokens: int = 2048
    ) -> dict:
        from .resilience import resilient_call

        messages = _merge_system_messages(messages)
        # Use JSON mode if model supports it
        try:
            kwargs: dict = dict(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            response = resilient_call(
                self.client.chat.completions.create,
                **kwargs,
                provider_name="openai",
            )
            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception:
            # Fallback: regular generation + parse
            raw = self.generate(messages, temperature=temperature)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
            for match in re.finditer(r"\{.*\}", raw, re.DOTALL):
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue
            raise ValueError(f"Could not parse JSON: {raw[:200]}")
