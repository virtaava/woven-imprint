"""OpenAI embedding provider."""

from __future__ import annotations

from .base import EmbeddingProvider


class OpenAIEmbedding(EmbeddingProvider):
    """Generate embeddings via OpenAI API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install woven-imprint[openai]"
            )

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self.model = model
        self._dims: int | None = None

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        vec = response.data[0].embedding
        if self._dims is None:
            self._dims = len(vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        vecs = [d.embedding for d in response.data]
        if self._dims is None and vecs:
            self._dims = len(vecs[0])
        return vecs

    def dimensions(self) -> int:
        if self._dims is None:
            self.embed("test")
        return self._dims  # type: ignore
