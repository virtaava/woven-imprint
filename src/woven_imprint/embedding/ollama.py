"""Ollama embedding provider — local-first, no API key needed."""

from __future__ import annotations

import requests

from .base import EmbeddingProvider


class OllamaEmbedding(EmbeddingProvider):
    """Generate embeddings via Ollama's /api/embed endpoint."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 30,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dims: int | None = None

    def embed(self, text: str) -> list[float]:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        vec = embeddings[0]
        if self._dims is None:
            self._dims = len(vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        vecs = resp.json()["embeddings"]
        if self._dims is None and vecs:
            self._dims = len(vecs[0])
        return vecs

    def dimensions(self) -> int:
        if self._dims is None:
            self.embed("test")
        return self._dims  # type: ignore
