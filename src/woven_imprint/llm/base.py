"""Abstract LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Generate text completions from an LLM."""

    @abstractmethod
    def generate(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ) -> str:
        """Generate a completion from a message list.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's response text.
        """

    @abstractmethod
    def generate_json(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> dict[str, Any] | list[Any]:
        """Generate a JSON response. Must return valid parsed JSON."""

    def generate_stream(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ):
        """Yield response text chunks. Default: one chunk via generate().

        Providers with native streaming override this. Note: providers that
        post-process whole responses (e.g. think-tag stripping) may behave
        differently in stream mode — document per provider.
        """
        yield self.generate(messages, temperature=temperature, max_tokens=max_tokens)

    def generate_json_robust(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> dict[str, Any] | list[Any]:
        """generate_json with one bounded retry at temperature 0.1 on parse failure.

        Uniform policy for subsystem JSON calls on weak/small models
        (mirrors the consistency checker's existing retry behavior).
        """
        try:
            return self.generate_json(messages, temperature=temperature)
        except ValueError:
            return self.generate_json(messages, temperature=0.1)
