"""Memory store — manages the three-tier memory lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..clock import sqlite_ts
from ..embedding.base import EmbeddingProvider
from ..storage.sqlite import SQLiteStorage
from ..utils.text import generate_id
from .retrieval import cosine_matrix

if TYPE_CHECKING:
    from .facts import FactStore


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


def build_embed_text(
    content: str,
    role: str | None,
    created_at: str | None,
    metadata: dict | None,
    character_name: str,
) -> str:
    """Build the date+speaker-contextualized text a memory's vector is computed from.

    Ported from the offline ranking experiment's winning template
    (`eval/external/runs/diagnostics/ranking/ranking_experiments_report.md`
    section (c)): `f"search_document: [{date}] {speaker}: {content}"`, with
    two deliberate departures that experiment flagged as worth fixing before
    shipping — that template embedded the speaker name twice (once from the
    injected `speaker:` prefix, once from the pre-existing `"[User] "`/
    `"[Name] "` bracket tag already in stored `content`), and it carried a
    nomic `search_document:` prefix that the same offline sweep measured as
    harmful on raw content. Here the bracket tag is stripped before the
    speaker is injected, and no prefix is added at all.

    - user rows: ``f"[{date}] User: {user_id}: {body}"`` when
      ``metadata.user_id`` is known, else ``f"[{date}] User: {body}"``.
    - character rows: ``f"[{date}] {character_name}: {body}"``.
    - everything else (facts, summaries, reflections, consolidations,
      events, ...): ``f"[{date}] {content}"`` — content is left exactly as
      stored, since these rows carry no leading speaker bracket to strip.

    ``body`` is ``content`` with a leading ``"[User] "`` / ``"[<name>] "``
    tag stripped (that tag is how ``content`` is stored — see
    ``Character.chat``/``ingest``/``ingest_exchange``); stored ``content``
    itself is never modified, this is embedding-only. ``date`` is the
    ``YYYY-MM-DD`` prefix of ``created_at`` (the memory's own creation
    timestamp, not "now").
    """
    date = (created_at or "")[:10]
    meta = metadata or {}
    if role == "user":
        tag = "[User] "
        body = content[len(tag) :] if content.startswith(tag) else content
        user_id = meta.get("user_id")
        if user_id:
            return f"[{date}] User: {user_id}: {body}"
        return f"[{date}] User: {body}"
    if role == "character":
        tag = f"[{character_name}] "
        body = content[len(tag) :] if content.startswith(tag) else content
        return f"[{date}] {character_name}: {body}"
    return f"[{date}] {content}"


def _content_has_all_tokens(content: str, tokens: list[str]) -> bool:
    """Case-insensitive whole-word check: does `content` contain every one of `tokens`?

    Used by the entity-delta dedup guard (`MemoryStore._find_dedup_match`'s
    `require_tokens`). Word boundaries matter: the object "red" must not be
    satisfied by "Fred" or "shredded", or the guard would wave through the very
    merges it exists to block.
    """
    import re

    lowered = content.lower()
    return all(re.search(rf"(?<!\w){re.escape(tok.lower())}(?!\w)", lowered) for tok in tokens)


class MemoryStore:
    """Manages buffer/core/bedrock memory tiers for a character."""

    build_embed_text = staticmethod(build_embed_text)

    def __init__(
        self,
        storage: SQLiteStorage,
        embedder: EmbeddingProvider,
        character_id: str,
        character_name: str = "Character",
    ):
        self.storage = storage
        self.embedder = embedder
        self.character_id = character_id
        # Used to strip the `[<name>] ` tag and label the speaker on character
        # rows when building contextualized embed text (see `_embed_text_for`).
        # Set by `Character.__init__` to `persona.name`; the default here only
        # matters for direct/test construction that never adds `role="character"`
        # rows through `add()`.
        self.character_name = character_name
        # Wired up by Character right after construction (`char.memory.facts = char.facts`)
        # so `delete()` can retract facts pointing at a deleted memory.
        self.facts: FactStore | None = None

    def _embed_text_for(
        self, content: str, role: str | None, created_at: str | None, metadata: dict | None
    ) -> str:
        """The text actually embedded for a memory row — contextualized (date +
        speaker) when `memory.embedding_context` is on, else exactly `content`
        (the pre-Tier-3d behavior). Shared by `add()` and `reembed()` so both
        always agree on what a memory's vector should be."""
        from ..config import get_config

        if not get_config().memory.embedding_context:
            return content
        return build_embed_text(content, role, created_at, metadata, self.character_name)

    def embed_for(
        self,
        content: str,
        role: str | None = None,
        created_at: str | None = None,
        metadata: dict | None = None,
    ) -> list[float]:
        """Compute the vector a memory of this shape would get from `add()`:
        `_embed_text_for(content, role, created_at, metadata)` (contextualized
        by default — see `build_embed_text`) through this store's own embedder,
        with the same dimension guard `add()`/`edit()`/`reembed()` apply.

        The single seam every memory-writing subsystem that persists an
        embedding *outside* `add()`/`edit()` itself should go through —
        consolidation summaries, belief contradictions, growth events, and
        `FactStore.edit`'s linked-memory re-embed (injected there as
        `embed_fn`) — so all of them agree on what "the vector for this
        content" means instead of each embedding raw content independently
        (and, in a couple of those call sites, skipping the dimension guard
        entirely).
        """
        embed_text = self._embed_text_for(content, role, created_at, metadata)
        embedding = self.embedder.embed(embed_text)
        guard_embedding_dimension(self.storage, embedding)
        return embedding

    def add(
        self,
        content: str,
        tier: str = "buffer",
        role: str | None = None,
        session_id: str | None = None,
        importance: float = 0.5,
        metadata: dict | None = None,
        dedup_similarity: float | None = None,
        dedup_scope: str = "core",
        dedup_require_tokens: list[str] | None = None,
    ) -> dict:
        """Add a new memory entry.

        The stored `content` is exactly what was passed in; the *vector* is
        computed from `_embed_text_for(...)`, which — by default
        (`memory.embedding_context: true`) — embeds a date+speaker
        contextualized string instead of raw `content` (see `build_embed_text`).

        `dedup_similarity` (0/None = off) opts this call into semantic dedup:
        when `tier == dedup_scope` (default `"core"`) and the candidate's
        embedding matches an existing active `dedup_scope` memory at or above
        this cosine threshold, no new row is inserted — the *existing* row is
        reinforced instead (`importance` bumped by 0.05 capped at 1.0,
        `metadata.dup_count` incremented, `metadata.last_confirmed` stamped)
        and returned. Every call — deduped or not — returns a memory dict
        with a transient `"deduped": bool` key (not persisted to storage) so
        callers can tell which happened; on a dedup, the row's `id` is the
        *existing* memory's id, which matters to callers that link a
        secondary record (e.g. `FactStore.add(memory_id=...)`) to whichever
        memory ends up representing this content.

        `dedup_require_tokens` — the entity-delta guard: when given, a
        candidate is rejected as a dedup match — even at/above
        `dedup_similarity` — unless its stored content contains every one of
        these tokens (case-insensitive). `_store_structured_fact` passes the
        new fact's `object` tokens: a bag-of-words embedder (or a real one on
        boilerplate-heavy statements) can score two updates to the same
        (subject, predicate) with a different object as near-identical
        cosine similarity when most of the sentence is shared preamble — this
        guard stops that from silently merging a value change into the old
        row instead of recording it.
        """
        created_at = sqlite_ts()
        embed_text = self._embed_text_for(content, role, created_at, metadata)
        embedding = self.embedder.embed(embed_text)
        guard_embedding_dimension(self.storage, embedding)

        if dedup_similarity and tier == dedup_scope:
            match = self._find_dedup_match(
                content, embedding, dedup_scope, dedup_similarity, dedup_require_tokens
            )
            if match is not None:
                return self._reinforce_duplicate(match)

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
            "created_at": created_at,
            "accessed_at": created_at,
        }
        self.storage.save_memory(memory)
        memory["deduped"] = False
        return memory

    def _find_dedup_match(
        self,
        content: str,
        embedding: list[float],
        scope: str,
        threshold: float,
        require_tokens: list[str] | None = None,
    ) -> dict | None:
        """Find the best-matching active `scope`-tier memory for `content`/`embedding`,
        if any scores >= `threshold` cosine similarity.

        Candidate pool: FTS hits for `content` (limit 50) union the newest 500
        active `scope` rows — cheap enough (one extra O(500) numpy cosine pass)
        to run per fact insert without a second embedding call (the candidate
        vector is already in hand from `add()`).

        `require_tokens` (entity-delta guard, see `add()`): candidates missing
        any one of these tokens (case-insensitive substring match against the
        candidate's stored `content`) are dropped from the pool before the
        cosine comparison runs, regardless of embedding similarity.
        """
        try:
            fts_candidates = self.storage.fts_search(self.character_id, content, limit=50)
        except Exception:
            fts_candidates = []
        recent = self.storage.get_memories(self.character_id, tier=scope, limit=500)

        candidates: dict[str, dict] = {}
        for m in fts_candidates + recent:
            if m.get("tier") == scope:
                candidates[m["id"]] = m
        if not candidates:
            return None

        rows = list(candidates.values())
        if require_tokens:
            rows = [
                r for r in rows if _content_has_all_tokens(r.get("content") or "", require_tokens)
            ]
            if not rows:
                return None
        sims = cosine_matrix(embedding, [r.get("embedding") or [] for r in rows])
        best_idx, best_sim = -1, -1.0
        for i, sim in enumerate(sims):
            if sim > best_sim:
                best_idx, best_sim = i, sim
        if best_idx == -1 or best_sim < threshold:
            return None
        return rows[best_idx]

    def _reinforce_duplicate(self, existing: dict) -> dict:
        """Bump `importance`/`metadata.dup_count`/`metadata.last_confirmed` on an
        existing memory found to duplicate an incoming fact, and return the
        refreshed row (with the transient `deduped: True` key set) instead of
        inserting a new one."""
        new_importance = min(1.0, existing.get("importance", 0.5) + 0.05)
        self.storage.update_memory_fields(existing["id"], importance=new_importance)
        dup_count = int((existing.get("metadata") or {}).get("dup_count", 0)) + 1
        self.storage.update_memory_metadata(
            existing["id"], {"dup_count": dup_count, "last_confirmed": sqlite_ts()}
        )
        refreshed = self.storage.get_memory(existing["id"])
        assert refreshed is not None
        refreshed["deduped"] = True
        return refreshed

    def reembed(self, batch_size: int = 64) -> int:
        """Recompute vectors for every active memory using the current embed-text
        builder (`memory.embedding_context` + `build_embed_text`) — for bringing
        a database created before this feature (or before a builder change) up
        to date. Idempotent: re-running recomputes the same vectors from the
        same stored content/config, byte-for-byte.

        Streams pages of `batch_size` rows via
        `storage.iter_memory_rows_for_reembed` — each page carries only
        id/content/role/created_at/metadata, never an `embedding` column — so
        a history far larger than fits comfortably in memory never has every
        row's *vector* materialized at once the way a plain
        `get_memories(..., limit=None)` would. Each page is embedded in one
        `embed_batch` call via the store's own `self.embedder` (so a
        `CachedEmbedder` wrapper, if configured, still dedupes identical embed
        texts across the run). Only the `embedding` column changes;
        `content`/`metadata`/tier/etc. are untouched. Returns the number of
        memories re-embedded.
        """
        count = 0
        for chunk in self.storage.iter_memory_rows_for_reembed(
            self.character_id, page=max(1, batch_size)
        ):
            texts = [
                self._embed_text_for(
                    m["content"], m.get("role"), m.get("created_at"), m["metadata"]
                )
                for m in chunk
            ]
            vectors = self.embedder.embed_batch(texts)
            assert len(vectors) == len(chunk)
            for vec in vectors:
                guard_embedding_dimension(self.storage, vec)
            self.storage.update_memory_embeddings_batch(
                [(m["id"], vec) for m, vec in zip(chunk, vectors)]
            )
            count += len(chunk)
        return count

    def link_entities(self, llm, batch_size: int = 10) -> int:
        """Backfill `metadata.entities` for active memories that predate Tier 3o.

        One `generate_json_robust` call per `batch_size` memories. Idempotent:
        rows whose metadata already CONTAINS the `entities` key (even `[]`)
        are skipped, and a batch whose LLM call fails is skipped (logged via
        return count only — never raises). Rows the LLM omits from its answer
        are stamped `entities: []` so reruns don't loop on them. Returns the
        number of memories updated.
        """
        from ..persona.assessment import TurnAssessor

        _LINK_ENTITIES_PROMPT = (
            "For each numbered memory below, list up to 8 short canonical names of specific "
            "people, pets, places, organizations, or distinctive objects/events it mentions "
            "(proper nouns preferred; no generic nouns, no dates).\n"
            'Return ONLY a JSON object mapping the number to the list, e.g. {{"1": ["Rocket"], '
            '"2": []}}. Every number must appear.\n\n{items}'
        )

        rows = self.storage.get_memories(self.character_id, limit=100000)
        todo = [r for r in rows if "entities" not in (r.get("metadata") or {})]
        updated = 0
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            items = "\n".join(
                f"{i + 1}. {(r.get('content') or '')[:400]}" for i, r in enumerate(batch)
            )
            try:
                data = llm.generate_json_robust(
                    [{"role": "user", "content": _LINK_ENTITIES_PROMPT.format(items=items)}],
                    temperature=0.0,
                )
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            for i, r in enumerate(batch):
                ents = TurnAssessor._parse_entities(data.get(str(i + 1)))
                try:
                    self.set_entities(r["id"], ents)
                    updated += 1
                except KeyError:
                    continue
        return updated

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

    def get_all(self, tier: str | None = None, limit: int | None = 1000) -> list[dict]:
        """`limit=None` returns all matching rows (no LIMIT clause)."""
        return self.storage.get_memories(self.character_id, tier=tier, limit=limit)

    def count(self, tier: str | None = None, unconsolidated: bool = False) -> int:
        return self.storage.count_memories(
            self.character_id, tier=tier, unconsolidated=unconsolidated
        )

    def touch(self, memory_id: str) -> None:
        """Mark memory as recently accessed.

        Updates accessed_at for analytics; recency scoring is anchored on
        created_at by default (see `memory.recency_anchor` config).
        """
        self.storage.touch_memory(memory_id)

    def archive(self, memory_id: str) -> None:
        """Move memory to archived status (excluded from retrieval)."""
        self.storage.update_memory_status(memory_id, "archived")

    def edit(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        tier: str | None = None,
    ) -> dict:
        """Edit a memory in place. Re-embeds when `content` actually changes.

        Raises `KeyError` if the memory doesn't exist (or belongs to another
        character); `ValueError` on an invalid `tier`.
        """
        row = self.storage.get_memory(memory_id)
        if row is None or row.get("character_id") != self.character_id:
            raise KeyError(memory_id)
        embedding = None
        if content is not None and content != row["content"]:
            embedding = self.embedder.embed(
                self._embed_text_for(
                    content, row.get("role"), row.get("created_at"), row.get("metadata")
                )
            )
            guard_embedding_dimension(self.storage, embedding)
        else:
            content = None
        self.storage.update_memory_fields(
            memory_id, content=content, embedding=embedding, importance=importance, tier=tier
        )
        updated = self.storage.get_memory(memory_id)
        assert updated is not None
        return updated

    def delete(self, memory_id: str) -> None:
        """Hard delete a memory. Retracts any fact still pointing at it — deleting a
        memory means "forget this"."""
        row = self.storage.get_memory(memory_id)
        if row is None or row.get("character_id") != self.character_id:
            raise KeyError(memory_id)
        if self.facts is not None:
            for fid in self.storage.unlink_fact_memory(memory_id):
                self.facts.retract(fid)
        self.storage.delete_memory(memory_id)

    def pin(self, memory_id: str, pinned: bool = True) -> dict:
        """Set `metadata.pinned` on a memory (pinned memories are always in the prompt)."""
        row = self.storage.get_memory(memory_id)
        if row is None or row.get("character_id") != self.character_id:
            raise KeyError(memory_id)
        meta = dict(row.get("metadata") or {})
        meta["pinned"] = bool(pinned)
        self.storage.update_memory_fields(memory_id, metadata=meta)
        updated = self.storage.get_memory(memory_id)
        assert updated is not None
        return updated

    def set_entities(self, memory_id: str, entities: list[str]) -> dict:
        """Set `metadata.entities` on a memory (Tier 3o entity handles).

        Merge-style like `pin`: other metadata keys survive. The key's
        PRESENCE (even as []) marks the row as entity-processed — the
        backfill job (`link_entities`) skips rows that have it.
        """
        row = self.storage.get_memory(memory_id)
        if row is None or row.get("character_id") != self.character_id:
            raise KeyError(memory_id)
        meta = dict(row.get("metadata") or {})
        meta["entities"] = list(entities)
        self.storage.update_memory_fields(memory_id, metadata=meta)
        updated = self.storage.get_memory(memory_id)
        assert updated is not None
        return updated

    def pinned(self) -> list[dict]:
        return self.storage.list_pinned_memories(self.character_id)

    def needs_consolidation(self, threshold: int = 100) -> bool:
        """Check if buffer has exceeded consolidation threshold.

        Mirrors `ConsolidationEngine.needs_consolidation`: when
        `get_config().memory.consolidation_keep_sources` is on, counts only
        *unconsolidated* buffer rows — memories already claimed by a previous
        consolidation pass (`metadata.consolidated_into` / `consolidation_seen`)
        don't recount toward the threshold. When the flag is off, this is a
        plain buffer count (byte-identical to the pre-Tier-3c behavior).
        """
        from ..config import get_config

        keep_sources = get_config().memory.consolidation_keep_sources
        return self.count(tier="buffer", unconsolidated=keep_sources) >= threshold
