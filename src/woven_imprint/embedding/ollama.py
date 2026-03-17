"""Ollama embedding provider — local-first, no API key needed."""

from __future__ import annotations

import requests

from .base import EmbeddingProvider


class OllamaEmbedding(EmbeddingProvider):
    """Generate embeddings via Ollama's /api/embed endpoint."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str | None = None,
        timeout: int = 30,
    ):
        import os

        self.model = model
        self.base_url = (
            base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.timeout = timeout
        self._dims: int | None = None

    def embed(self, text: str) -> list[float]:
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Install from https://ollama.com"
            ) from None
        except requests.Timeout:
            raise TimeoutError(
                f"Ollama embedding did not respond within {self.timeout}s."
            ) from None
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise ValueError(
                    f"Embedding model '{self.model}' not found. Run: ollama pull {self.model}"
                ) from None
            raise
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
