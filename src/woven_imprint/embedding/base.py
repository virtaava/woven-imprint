"""Abstract embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Generate vector embeddings from text."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns float vector."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns list of float vectors."""

    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
