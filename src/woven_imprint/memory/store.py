"""Memory store — manages the three-tier memory lifecycle."""

from __future__ import annotations


from ..clock import sqlite_ts
from ..embedding.base import EmbeddingProvider
from ..storage.sqlite import SQLiteStorage
from ..utils.text import generate_id


def guard_embedding_dimension(storage: SQLiteStorage, embedding: list[float] | None) -> None:
    """Guard against mixed embedding models corrupting cosine math (meta contract).

    Records the DB's embedding dimensionality (and model) the first time an
    embedding is saved, then raises ValueError on any later save whose
    embedding has a different dimension. Shared by every write path that can
    persist an embedding — `MemoryStore.add` and the consolidation engine's
    cluster-summary writes — so a swapped embedder can't silently write
    mixed-dimension vectors via either path.
    """
    if not embedding:
        return
    known = storage.meta_get("embedding_dimensions")
    if known is None:
        storage.meta_set("embedding_dimensions", str(len(embedding)))
        from ..config import get_config

        storage.meta_set("embedding_model", get_config().llm.embedding_model)
    elif int(known) != len(embedding):
        raise ValueError(
            f"Embedding dimension mismatch: DB stores {known}-d vectors, "
            f"got {len(embedding)}-d. Mixed embedding models corrupt retrieval — "
            f"re-embed the database or restore the original embedding model."
        )


class MemoryStore:
    """Manages buffer/core/bedrock memory tiers for a character."""

    def __init__(self, storage: SQLiteStorage, embedder: EmbeddingProvider, character_id: str):
        self.storage = storage
        self.embedder = embedder
        self.character_id = character_id

    def add(
        self,
        content: str,
        tier: str = "buffer",
        role: str | None = None,
        session_id: str | None = None,
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> dict:
        """Add a new memory entry."""
        embedding = self.embedder.embed(content)
        guard_embedding_dimension(self.storage, embedding)

        memory = {
            "id": generate_id("mem-"),
            "character_id": self.character_id,
            "tier": tier,
            "content": content,
            "embedding": embedding,
            "importance": importance,
            "certainty": 1.0,
            "status": "active",
            "source_refs": [],
            "session_id": session_id,
            "role": role,
            "metadata": metadata or {},
            "created_at": sqlite_ts(),
            "accessed_at": sqlite_ts(),
        }
        self.storage.save_memory(memory)
        return memory

    def add_without_embedding(
        self,
        content: str,
        tier: str = "buffer",
        role: str | None = None,
        session_id: str | None = None,
        importance: float = 0.5,
    ) -> dict:
        """Add memory without computing embedding (for batch processing)."""
        memory = {
            "id": generate_id("mem-"),
            "character_id": self.character_id,
            "tier": tier,
            "content": content,
            "embedding": None,
            "importance": importance,
            "certainty": 1.0,
            "status": "active",
            "source_refs": [],
            "session_id": session_id,
            "role": role,
            "metadata": {},
            "created_at": sqlite_ts(),
            "accessed_at": sqlite_ts(),
        }
        self.storage.save_memory(memory)
        return memory

    def get(self, memory_id: str) -> dict | None:
        return self.storage.get_memory(memory_id)

    def get_all(self, tier: str | None = None, limit: int = 1000) -> list[dict]:
        return self.storage.get_memories(self.character_id, tier=tier, limit=limit)

    def count(self, tier: str | None = None) -> int:
        return self.storage.count_memories(self.character_id, tier=tier)

    def touch(self, memory_id: str) -> None:
        """Mark memory as recently accessed.

        Updates accessed_at for analytics; recency scoring is anchored on
        created_at by default (see `memory.recency_anchor` config).
        """
        self.storage.touch_memory(memory_id)

    def archive(self, memory_id: str) -> None:
        """Move memory to archived status (excluded from retrieval)."""
        self.storage.update_memory_status(memory_id, "archived")

    def needs_consolidation(self, threshold: int = 100) -> bool:
        """Check if buffer has exceeded consolidation threshold."""
        return self.count(tier="buffer") >= threshold
