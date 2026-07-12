"""Consolidation engine — compress buffer memories into core memories."""

from __future__ import annotations

import math

from ..llm.base import LLMProvider
from ..embedding.base import EmbeddingProvider
from ..storage.sqlite import SQLiteStorage
from ..utils.text import generate_id


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cluster_memories(memories: list[dict], similarity_threshold: float = 0.75) -> list[list[dict]]:
    """Cluster memories by semantic similarity using simple greedy clustering."""
    if not memories:
        return []

    # Filter to memories with embeddings
    with_emb = [m for m in memories if m.get("embedding")]
    without_emb = [m for m in memories if not m.get("embedding")]

    clusters: list[list[dict]] = []
    assigned: set[str] = set()

    for mem in with_emb:
        if mem["id"] in assigned:
            continue

        cluster = [mem]
        assigned.add(mem["id"])

        for other in with_emb:
            if other["id"] in assigned:
                continue
            sim = _cosine_similarity(mem["embedding"], other["embedding"])
            if sim >= similarity_threshold:
                cluster.append(other)
                assigned.add(other["id"])

        clusters.append(cluster)

    # Put memories without embeddings in their own singleton clusters
    for mem in without_emb:
        clusters.append([mem])

    return clusters


class ConsolidationEngine:
    """Compress buffer memories into consolidated core memories.

    When the buffer exceeds a threshold, semantically similar memories
    are clustered and summarized by the LLM into dense core entries.
    Original buffer entries are archived (not deleted).
    """

    def __init__(
        self,
        storage: SQLiteStorage,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        character_id: str,
        threshold: int = 100,
        similarity: float = 0.75,
    ):
        self.storage = storage
        self.llm = llm
        self.embedder = embedder
        self.character_id = character_id
        self.threshold = threshold
        self.similarity = similarity

    def needs_consolidation(self) -> bool:
        count = self.storage.count_memories(self.character_id, tier="buffer")
        return count >= self.threshold

    def consolidate(
        self, dry_run: bool = False, chunk_size: int | None = None, budget=None
    ) -> dict:
        """Run one bounded consolidation pass. Returns stats.

        Args:
            dry_run: If True, compute clusters and counts but write nothing
                and make no LLM calls.
            chunk_size: Max buffer rows to pull for this pass. Defaults to
                ``get_config().maintenance.consolidate_chunk_size``. Use
                `drain()` to fully clear a buffer larger than one chunk.
            budget: Optional duck-typed budget object with ``take(n: int = 1)
                -> bool``. Checked once per multi-member cluster, immediately
                before the LLM call that summarizes it (singleton promotions
                make no LLM call and are never budget-checked). None means
                unlimited. When the budget is exhausted, processing of
                remaining clusters stops and ``budget_exhausted`` is set on
                the returned stats.

        Returns:
            Dict with keys: clusters, summarized, created, archived,
            llm_calls, and (only when it happened) budget_exhausted=True.
        """
        from ..config import get_config

        chunk_size = chunk_size or get_config().maintenance.consolidate_chunk_size
        buffer = self.storage.get_memories(self.character_id, tier="buffer", limit=chunk_size)
        if len(buffer) < 10:
            return {"clusters": 0, "summarized": 0, "created": 0, "archived": 0, "llm_calls": 0}

        clusters = _cluster_memories(buffer, self.similarity)

        stats = {
            "clusters": len(clusters),
            "summarized": 0,
            "created": 0,
            "archived": 0,
            "llm_calls": 0,
        }

        for cluster in clusters:
            if len(cluster) < 2:
                # Singleton — promote directly to core if important enough.
                # No LLM call involved, so no budget check here.
                mem = cluster[0]
                if mem.get("importance", 0) >= 0.6:
                    if not dry_run:
                        self.storage.save_memory(
                            {
                                **mem,
                                "id": generate_id("mem-"),
                                "tier": "core",
                                "source_refs": [mem["id"]],
                            }
                        )
                        self.storage.update_memory_status(mem["id"], "archived")
                        stats["created"] += 1
                        stats["archived"] += 1
                continue

            # Multi-memory cluster — summarize
            if dry_run:
                # Count only: no LLM call, no writes, no budget consumed.
                stats["summarized"] += len(cluster)
                stats["created"] += 1
                continue

            if budget is not None and not budget.take(1):
                stats["budget_exhausted"] = True
                break

            content_texts = [m["content"][:300] for m in cluster]
            cluster_text = "\n".join(f"- {t}" for t in content_texts)

            summary = self._summarize_cluster(cluster_text)
            stats["llm_calls"] += 1
            if not summary:
                continue

            # Compute embedding for the summary
            embedding = self.embedder.embed(summary)

            # Compute importance as max of cluster
            max_importance = max(m.get("importance", 0.5) for m in cluster)

            # Create consolidated core memory
            source_ids = [m["id"] for m in cluster]
            self.storage.save_memory(
                {
                    "id": generate_id("mem-"),
                    "character_id": self.character_id,
                    "tier": "core",
                    "content": f"[Consolidated] {summary}",
                    "embedding": embedding,
                    "importance": max_importance,
                    "certainty": 1.0,
                    "status": "active",
                    "source_refs": source_ids,
                    "role": "observation",
                    "metadata": {"type": "consolidation", "source_count": len(cluster)},
                }
            )
            stats["created"] += 1

            # Archive original buffer entries
            for mem in cluster:
                self.storage.update_memory_status(mem["id"], "archived")
                stats["archived"] += 1

            stats["summarized"] += len(cluster)

        return stats

    def drain(self, max_chunks: int = 10, dry_run: bool = False, budget=None) -> dict:
        """Run consolidation passes until the buffer is below threshold.

        Replaces the old single-pass 500-row cap: a heavy day fully drains
        across multiple bounded passes. ``budget`` (see `consolidate()`) is
        threaded through every pass; once a pass reports budget_exhausted,
        draining stops even if the buffer is still above threshold.
        """
        totals = {
            "passes": 0,
            "clusters": 0,
            "summarized": 0,
            "created": 0,
            "archived": 0,
            "llm_calls": 0,
            "budget_exhausted": False,
        }
        for _ in range(max_chunks):
            if not self.needs_consolidation():
                break
            result = self.consolidate(dry_run=dry_run, budget=budget)
            totals["passes"] += 1
            for key in ("clusters", "summarized", "created", "archived", "llm_calls"):
                totals[key] += result.get(key, 0)
            if result.get("budget_exhausted"):
                totals["budget_exhausted"] = True
                break
            if dry_run:
                break  # dry_run archives nothing → would loop forever
        return totals

    def _summarize_cluster(self, cluster_text: str) -> str | None:
        """Use LLM to summarize a cluster of related memories."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory consolidation system. Summarize the following "
                    "related memories into a single dense entry that preserves all "
                    "important facts, emotions, and relationships. Be concise but "
                    "complete. Write in third person or as an observation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Consolidate these related memories into one summary:\n\n"
                    f"{cluster_text}\n\n"
                    f"Write a single paragraph capturing the key information."
                ),
            },
        ]
        try:
            return self.llm.generate(messages, temperature=0.3, max_tokens=300)
        except Exception:
            return None
