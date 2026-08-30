"""Consolidation engine — compress buffer memories into core memories."""

from __future__ import annotations

from .. import clock
from ..llm.base import LLMProvider
from ..embedding.base import EmbeddingProvider
from ..prompts import render
from ..storage.sqlite import SQLiteStorage
from ..utils.text import generate_id
from .retrieval import _cosine_similarity
from .store import guard_embedding_dimension


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
    Sources stay active and retrievable (metadata.consolidated_into) unless
    `consolidation_keep_sources` is False, in which case original buffer
    entries are archived (not deleted) as before.
    """

    def __init__(
        self,
        storage: SQLiteStorage,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        character_id: str,
        threshold: int = 100,
        similarity: float = 0.75,
        keep_sources: bool | None = None,
    ):
        self.storage = storage
        self.llm = llm
        self.embedder = embedder
        self.character_id = character_id
        self.threshold = threshold
        self.similarity = similarity
        # None (default) means "follow get_config().memory.consolidation_keep_sources
        # at call time"; an explicit True/False overrides the config (used by tests
        # that want deterministic behavior regardless of the ambient config).
        self.keep_sources = keep_sources

    def _resolve_keep_sources(self) -> bool:
        if self.keep_sources is not None:
            return self.keep_sources
        from ..config import get_config

        return get_config().memory.consolidation_keep_sources

    def needs_consolidation(self) -> bool:
        """True once the *unconsolidated* buffer count reaches threshold.

        When keep_sources is on, rows already claimed by a previous
        consolidation pass (`metadata.consolidated_into` / `consolidation_seen`
        set) don't recount toward the threshold. When keep_sources is off this
        is a plain buffer count — byte-identical to the pre-Tier-3c behavior.
        """
        keep_sources = self._resolve_keep_sources()
        count = self.storage.count_memories(
            self.character_id, tier="buffer", unconsolidated=keep_sources
        )
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
            Dict with keys: clusters, summarized, created, archived, kept,
            promoted, seen, llm_calls, and (only when it happened)
            budget_exhausted=True. ``kept``/``promoted``/``seen`` are always
            present (0 when `consolidation_keep_sources` is off, matching the
            pre-Tier-3c archive-everything behavior byte-for-byte).
        """
        from ..config import get_config

        keep_sources = self._resolve_keep_sources()
        chunk_size = chunk_size or get_config().maintenance.consolidate_chunk_size
        buffer = self.storage.get_memories(
            self.character_id, tier="buffer", limit=chunk_size, unconsolidated=keep_sources
        )
        if len(buffer) < 10:
            return {
                "clusters": 0,
                "summarized": 0,
                "created": 0,
                "archived": 0,
                "kept": 0,
                "promoted": 0,
                "seen": 0,
                "llm_calls": 0,
            }

        clusters = _cluster_memories(buffer, self.similarity)

        stats = {
            "clusters": len(clusters),
            "summarized": 0,
            "created": 0,
            "archived": 0,
            "kept": 0,
            "promoted": 0,
            "seen": 0,
            "llm_calls": 0,
        }

        for cluster in clusters:
            if len(cluster) < 2:
                # Singleton — promote directly to core if important enough.
                # No LLM call involved, so no budget check here.
                mem = cluster[0]
                if mem.get("importance", 0) >= 0.6:
                    if not dry_run:
                        if keep_sources:
                            # Promote in place — no copy, no archive, so the
                            # source text never appears twice in retrieval.
                            self.storage.update_memory_fields(mem["id"], tier="core")
                            self.storage.update_memory_metadata(
                                mem["id"], {"promoted_from_buffer": True}
                            )
                            stats["promoted"] += 1
                        else:
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
                elif keep_sources and not dry_run:
                    # Below the promotion bar: leave the row alone but stop it
                    # from recounting toward the threshold every pass.
                    self.storage.update_memory_metadata(mem["id"], {"consolidation_seen": True})
                    stats["seen"] += 1
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

            content_texts = [
                f"({m.get('created_at', '')[:10]}) {m['content'][:300]}" for m in cluster
            ]
            cluster_text = "\n".join(f"- {t}" for t in content_texts)

            summary = self._summarize_cluster(cluster_text)
            stats["llm_calls"] += 1
            if not summary:
                continue

            # Compute embedding for the summary
            embedding = self.embedder.embed(summary)
            # Apply the same dimension guard save_memory's sibling path
            # (MemoryStore.add) enforces — consolidation writes summary
            # memories directly via storage.save_memory, bypassing that
            # guard, so a swapped embedder could otherwise write a
            # mixed-dimension vector on this nightly path silently.
            guard_embedding_dimension(self.storage, embedding)

            # Compute importance as max of cluster
            max_importance = max(m.get("importance", 0.5) for m in cluster)

            # Create consolidated core memory. created_at is stamped as the
            # *latest* source memory's created_at (not "now") so a
            # consolidation pass doesn't make old memories look freshly
            # formed; date_range preserves the full span for the prompt/UI.
            source_ids = [m["id"] for m in cluster]
            dates = sorted(m["created_at"] for m in cluster if m.get("created_at"))
            memory_dict = {
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
                "metadata": {
                    "type": "consolidation",
                    "source_count": len(cluster),
                    "date_range": [dates[0], dates[-1]] if dates else None,
                },
            }
            if dates:
                memory_dict["created_at"] = dates[-1]
            self.storage.save_memory(memory_dict)
            stats["created"] += 1

            # Sources stay active and retrievable (keep_sources) or are
            # archived (legacy behavior) — either way they're claimed so a
            # later pass doesn't re-cluster them.
            if keep_sources:
                now_iso = clock.now().isoformat()
                for mem in cluster:
                    self.storage.update_memory_metadata(
                        mem["id"],
                        {"consolidated_into": memory_dict["id"], "consolidated_at": now_iso},
                    )
                    stats["kept"] += 1
            else:
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
            "kept": 0,
            "promoted": 0,
            "seen": 0,
            "llm_calls": 0,
            "budget_exhausted": False,
        }
        for _ in range(max_chunks):
            if not self.needs_consolidation():
                break
            result = self.consolidate(dry_run=dry_run, budget=budget)
            totals["passes"] += 1
            for key in (
                "clusters",
                "summarized",
                "created",
                "archived",
                "kept",
                "promoted",
                "seen",
                "llm_calls",
            ):
                totals[key] += result.get(key, 0)
            if result.get("budget_exhausted"):
                totals["budget_exhausted"] = True
                break
            if dry_run:
                break  # dry_run archives nothing → would loop forever
            if (
                result.get("archived", 0)
                + result.get("kept", 0)
                + result.get("promoted", 0)
                + result.get("seen", 0)
                == 0
            ):
                # No-progress short-circuit: a pass that clustered rows but
                # made no progress on any row — archived nothing (legacy
                # path), kept/promoted/marked-seen nothing (keep_sources
                # path) — e.g. every summary came back empty from a
                # failing/misbehaving LLM, would re-cluster the exact same
                # buffer rows next pass, burning budget up to max_chunks
                # times for zero gain. Stop draining instead.
                break
        return totals

    def _summarize_cluster(self, cluster_text: str) -> str | None:
        """Use LLM to summarize a cluster of related memories."""
        messages = render("consolidation_summary", cluster_text=cluster_text)
        try:
            return self.llm.generate(messages, temperature=0.3, max_tokens=300)
        except Exception:
            return None
