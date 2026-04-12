"""Gemma Edge LLM provider.

Adapter-facing provider for Gemma runtimes exposed through a small local HTTP
bridge. This is intended for Google AI Edge / MediaPipe style deployments where
the actual inference runtime is outside the Python process.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from .base import LLMProvider


class GemmaEdgeLLM(LLMProvider):
    """Generate completions through a Gemma edge adapter service.

    Expected adapter endpoints:
    - POST /generate
    - POST /generate_json

    Minimal request shape:
    {
      "model": "...",
      "messages": [...],
      "temperature": 0.7,
      "max_tokens": 256
    }

    Minimal response shape for /generate:
    {"content": "..."} or {"text": "..."}

    Minimal response shape for /generate_json:
    any valid JSON object/array, or {"content": "{...json...}"}.
    """

    def __init__(
        self,
        model: str = "gemma",
        base_url: str | None = None,
        timeout: int = 120,
    ):
        import os

        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("WOVEN_IMPRINT_GEMMA_EDGE_URL")
            or os.environ.get("WOVEN_IMPRINT_BASE_URL")
        )
        if not self.base_url:
            raise ValueError("GemmaEdgeLLM requires a base_url or WOVEN_IMPRINT_GEMMA_EDGE_URL.")
        self.base_url = self.base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict) -> requests.Response:
        resp = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp

    def generate(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ) -> str:
        resp = self._post(
            "/generate",
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        data = resp.json()
        if isinstance(data, dict):
            if isinstance(data.get("content"), str):
                return data["content"].strip()
            if isinstance(data.get("text"), str):
                return data["text"].strip()
        raise ValueError("Gemma edge adapter returned no text content")

    def generate_json(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> dict[str, Any] | list[Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            resp = self._post("/generate_json", payload)
            data = resp.json()
            if isinstance(data, (dict, list)):
                return data
        except Exception:
            pass

        raw = self.generate(messages, temperature=temperature)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        for match in re.finditer(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL):
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

        for match in re.finditer(r"\{.*\}|\[.*\]", raw, re.DOTALL):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not parse JSON from Gemma edge response: {raw[:200]}")
