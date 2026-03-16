"""OpenAI-compatible LLM provider — works with OpenAI, Azure, and compatible APIs."""

from __future__ import annotations

import json
import re

from .base import LLMProvider


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

    def generate(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_json(self, messages: list[dict[str, str]], temperature: float = 0.3) -> dict:
        # Use JSON mode if model supports it
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
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
