"""Abstract LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Generate text completions from an LLM."""

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], temperature: float = 0.7,
                 max_tokens: int = 2048) -> str:
        """Generate a completion from a message list.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's response text.
        """

    @abstractmethod
    def generate_json(self, messages: list[dict[str, str]], temperature: float = 0.3) -> dict:
        """Generate a JSON response. Must return valid parsed JSON."""
