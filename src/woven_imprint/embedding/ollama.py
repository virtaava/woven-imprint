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

    def _post(self, payload: dict) -> requests.Response:
        """Resilient POST to Ollama embedding endpoint."""
        from ..llm.resilience import resilient_call

        def _do_post():
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp

        try:
            return resilient_call(_do_post, provider_name="ollama_embed")
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

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            # Return zero vector of appropriate dimensionality
            dims = self.dimensions()
            return [0.0] * dims
        resp = self._post({"model": self.model, "input": text})
        embeddings = resp.json()["embeddings"]
        vec = embeddings[0]
        if self._dims is None:
            self._dims = len(vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Handle empty strings by returning zero vectors
        if not texts:
            return []
        # Get dimensionality (will call embed("test") if unknown)
        dims = self.dimensions()
        # Prepare result list
        results = []
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if not text.strip():
                results.append([0.0] * dims)
            else:
                results.append(None)  # placeholder
                non_empty_indices.append(i)
                non_empty_texts.append(text)
        # If all texts were empty, return zero vectors
        if not non_empty_texts:
            return results
        # Get embeddings for non-empty texts
        resp = self._post({"model": self.model, "input": non_empty_texts})
        vecs = resp.json()["embeddings"]
        if self._dims is None and vecs:
            self._dims = len(vecs[0])
        # Place embeddings at correct positions
        for idx, vec in zip(non_empty_indices, vecs):
            results[idx] = vec
        return results

    def dimensions(self) -> int:
        if self._dims is None:
            self.embed("test")
        return self._dims  # type: ignore
