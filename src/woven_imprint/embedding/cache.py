"""Content-hash LRU cache wrapping any EmbeddingProvider.

Identical text is embedded once. Thread-safe (background workers embed too).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from .base import EmbeddingProvider


class CachedEmbedder(EmbeddingProvider):
    def __init__(self, inner: EmbeddingProvider, maxsize: int = 512):
        self.inner = inner
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> list[float]:
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
                return list(self._cache[key])
        vec = self.inner.embed(text)
        with self._lock:
            self.misses += 1
            self._cache[key] = list(vec)
            self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def dimensions(self) -> int:
        return self.inner.dimensions()
