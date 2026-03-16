"""Memory store — manages the three-tier memory lifecycle."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ..embedding.base import EmbeddingProvider
from ..storage.sqlite import SQLiteStorage
from ..utils.text import generate_id


class MemoryStore:
    """Manages buffer/core/bedrock memory tiers for a character."""

    def __init__(self, storage: SQLiteStorage, embedder: EmbeddingProvider,
                 character_id: str):
        self.storage = storage
        self.embedder = embedder
        self.character_id = character_id

    def add(self, content: str, tier: str = "buffer", role: str | None = None,
            session_id: str | None = None, importance: float = 0.5,
            metadata: dict | None = None) -> dict:
        """Add a new memory entry."""
        embedding = self.embedder.embed(content)
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
        }
        self.storage.save_memory(memory)
        return memory

    def add_without_embedding(self, content: str, tier: str = "buffer",
                              role: str | None = None, session_id: str | None = None,
                              importance: float = 0.5) -> dict:
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
        """Mark memory as recently accessed (for recency scoring)."""
        self.storage.touch_memory(memory_id)

    def archive(self, memory_id: str) -> None:
        """Move memory to archived status (excluded from retrieval)."""
        self.storage.update_memory_status(memory_id, "archived")

    def needs_consolidation(self, threshold: int = 100) -> bool:
        """Check if buffer has exceeded consolidation threshold."""
        return self.count(tier="buffer") >= threshold
