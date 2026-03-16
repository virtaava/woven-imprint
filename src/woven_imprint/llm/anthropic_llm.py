"""Anthropic Claude LLM provider."""

from __future__ import annotations

import json
import re

from .base import LLMProvider


class AnthropicLLM(LLMProvider):
    """Generate completions via Anthropic's Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        timeout: int = 120,
        max_tokens: int = 4096,
    ):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install woven-imprint[anthropic]"
            )

        kwargs: dict = {"timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key

        self.client = Anthropic(**kwargs)
        self.model = model
        self.default_max_tokens = max_tokens

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.7,
                 max_tokens: int = 2048) -> str:
        # Anthropic uses system param separately
        system = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system += msg["content"] + "\n"
            else:
                chat_messages.append(msg)

        # Ensure alternating user/assistant messages
        if not chat_messages or chat_messages[0]["role"] != "user":
            chat_messages.insert(0, {"role": "user", "content": "Hello."})

        response = self.client.messages.create(
            model=self.model,
            system=system.strip() or "You are a helpful assistant.",
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text

    def generate_json(self, messages: list[dict[str, str]], temperature: float = 0.3) -> dict:
        # Add JSON instruction
        augmented = messages.copy()
        if augmented and augmented[-1]["role"] == "user":
            augmented[-1] = {
                "role": "user",
                "content": augmented[-1]["content"] + "\n\nRespond with valid JSON only.",
            }

        raw = self.generate(augmented, temperature=temperature)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        for match in re.finditer(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL):
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

        for match in re.finditer(r"\{.*\}", raw, re.DOTALL):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not parse JSON: {raw[:200]}")
