"""Multi-strategy memory retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ..embedding.base import EmbeddingProvider
from ..storage.sqlite import SQLiteStorage
from ..utils.rrf import reciprocal_rank_fusion


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_decay_rates() -> dict:
    from ..config import get_config

    cfg = get_config().memory
    return {"bedrock": cfg.decay_bedrock, "core": cfg.decay_core, "buffer": cfg.decay_buffer}


def _get_tier_boosts() -> dict:
    from ..config import get_config

    cfg = get_config().memory
    return {
        "bedrock": cfg.tier_boost_bedrock,
        "core": cfg.tier_boost_core,
        "buffer": cfg.tier_boost_buffer,
    }


def _recency_score(memory: dict, tier: str = "buffer") -> float:
    """Exponential decay based on hours since the anchor timestamp.

    Anchor is `created_at` by default (config memory.recency_anchor).
    Anchoring on `accessed_at` makes frequently-retrieved memories
    self-reinforcing — kept only as an opt-in legacy mode.
    """
    from ..config import get_config

    anchor_field = (
        "accessed_at" if get_config().memory.recency_anchor == "accessed" else "created_at"
    )
    decay_rate = _get_decay_rates().get(tier, 0.995)
    raw = memory.get(anchor_field) or memory.get("created_at") or ""
    try:
        anchored = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0.5
    if anchored.tzinfo is None:
        anchored = anchored.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    hours = max(0, (now - anchored).total_seconds() / 3600)
    return decay_rate**hours


class MemoryRetriever:
    """Retrieve memories using multi-strategy RRF.

    Retrieval strategies:
    1. Semantic similarity (embedding cosine distance)
    2. Keyword match (FTS5 BM25)
    3. Recency (tier-aware exponential decay)
    4. Importance (score × certainty + tier boost)
    5. Relationship boost (if target specified)

    Strategies are fused via weighted Reciprocal Rank Fusion (configurable
    per-strategy weights and k via MemoryConfig). Tier no longer contributes
    its own ranked list — it only influences decay rate (recency) and the
    importance tier boost, preventing bedrock seed floods from drowning out
    query-relevant personal facts.

    Tier-aware scoring:
    - Bedrock memories decay extremely slowly and get importance boosts
    - Core memories decay slowly
    - Buffer memories decay quickly (ephemeral by design)
    """

    def __init__(self, storage: SQLiteStorage, embedder: EmbeddingProvider, character_id: str):
        self.storage = storage
        self.embedder = embedder
        self.character_id = character_id

    def retrieve(
        self, query: str, limit: int = 10, relationship_target: str | None = None
    ) -> list[dict]:
        """Retrieve the most relevant memories using RRF across strategies."""
        # Two-phase retrieval (C5 fix):
        # Phase 1: Load recent memories per tier (recency window)
        bedrock = self.storage.get_memories(self.character_id, tier="bedrock", limit=50)
        core = self.storage.get_memories(self.character_id, tier="core", limit=200)
        buffer = self.storage.get_memories(self.character_id, tier="buffer", limit=100)

        # Phase 2: FTS pre-filter finds relevant OLD memories beyond the recency window
        # This ensures a memory from months ago can be found if the query matches
        try:
            fts_candidates = self.storage.fts_search(self.character_id, query, limit=50)
        except Exception:
            fts_candidates = []

        # Merge — deduplicate by ID
        memory_map: dict[str, dict] = {}
        for m in bedrock + core + buffer + fts_candidates:
            if m["id"] not in memory_map:
                memory_map[m["id"]] = m

        all_memories = list(memory_map.values())
        if not all_memories:
            return []

        # Sort by rowid for stable input order (ensures tiebreakers in rankings are deterministic)
        all_memories.sort(key=lambda m: m.get("rowid", 0))

        # Strategy 1: Semantic ranking (skip if query empty)
        semantic_ranked = []
        if query.strip():
            query_embedding = self.embedder.embed(query)
            semantic_scores = []
            for m in all_memories:
                if m.get("embedding"):
                    sim = _cosine_similarity(query_embedding, m["embedding"])
                    semantic_scores.append((m["id"], sim))
            semantic_scores.sort(key=lambda x: x[1], reverse=True)
            semantic_ranked = [mid for mid, _ in semantic_scores]

        # Strategy 2: Keyword ranking (BM25 via FTS5) — uses pre-fetched candidates
        keyword_ranked = [m["id"] for m in fts_candidates]

        # Strategy 3: Tier-aware recency ranking (with rowid tiebreaker for determinism)
        recency_scores = [
            (m["id"], _recency_score(m, m.get("tier", "buffer")), m.get("rowid", 0))
            for m in all_memories
        ]
        # Sort by score descending, then by rowid ascending (newer=higher rowid comes last in tie)
        recency_scores.sort(key=lambda x: (-x[1], x[2]))
        recency_ranked = [mid for mid, _, _ in recency_scores]

        # Strategy 4: Importance with tier boost + user affinity (with rowid tiebreaker)
        importance_scores = []
        for m in all_memories:
            base = m.get("importance", 0.5) * m.get("certainty", 1.0)
            boost = _get_tier_boosts().get(m.get("tier", "buffer"), 0.0)
            # User affinity bonus
            if relationship_target:
                meta = m.get("metadata", {})
                if meta.get("user_id") == relationship_target:
                    base += 0.2
            importance_scores.append((m["id"], base + boost, m.get("rowid", 0)))
        # Sort by score descending, then by rowid ascending (newer=higher rowid comes last in tie)
        importance_scores.sort(key=lambda x: (-x[1], x[2]))
        importance_ranked = [mid for mid, _, _ in importance_scores]

        # Strategy 5: Relationship boost (if target specified)
        from ..config import get_config

        mem_cfg = get_config().memory

        ranked_lists = [
            semantic_ranked,
            keyword_ranked,
            recency_ranked,
            importance_ranked,
        ]
        weights = [
            mem_cfg.weight_semantic,
            mem_cfg.weight_keyword,
            mem_cfg.weight_recency,
            mem_cfg.weight_importance,
        ]

        if relationship_target:
            rel_scores = []
            target_lower = relationship_target.lower()
            for m in all_memories:
                content_lower = m["content"].lower()
                meta = m.get("metadata", {})
                involves_target = (
                    target_lower in content_lower or meta.get("target_id") == relationship_target
                )
                rel_scores.append((m["id"], 1.0 if involves_target else 0.0))
            rel_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_lists.append([mid for mid, _ in rel_scores])
            weights.append(mem_cfg.weight_relationship)

        # Fuse with weighted RRF
        fused = reciprocal_rank_fusion(ranked_lists, k=mem_cfg.rrf_k, weights=weights)

        # Return top-N memories
        results = []
        touch_ids = []
        for mem_id, score in fused[:limit]:
            if mem_id in memory_map:
                mem = memory_map[mem_id].copy()
                mem["_retrieval_score"] = score
                results.append(mem)
                touch_ids.append(mem_id)

        # Batch-update accessed_at (single transaction, no FTS reindex)
        self.storage.touch_memories_batch(touch_ids)

        return results
