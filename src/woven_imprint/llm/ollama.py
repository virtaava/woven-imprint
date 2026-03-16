"""Ollama LLM provider — local-first inference."""

from __future__ import annotations

import json
import re

import requests

from .base import LLMProvider


class OllamaLLM(LLMProvider):
    """Generate completions via Ollama's native /api/chat endpoint."""

    def __init__(
        self,
        model: str = "qwen3-coder:30b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 120,
        num_ctx: int = 8192,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx

    def generate(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": self.num_ctx,
                },
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        # Strip thinking tags (qwen3-coder, deepseek)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content

    def generate_json(self, messages: list[dict[str, str]], temperature: float = 0.3) -> dict:
        # Add explicit JSON instruction
        if messages and messages[-1]["role"] == "user":
            messages = messages.copy()
            messages[-1] = {
                "role": "user",
                "content": messages[-1]["content"] + "\n\nRespond with valid JSON only.",
            }

        raw = self.generate(messages, temperature=temperature)

        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Extract JSON from markdown code blocks
        for match in re.finditer(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL):
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

        # Extract any JSON object
        for match in re.finditer(r"\{.*\}", raw, re.DOTALL):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}")
