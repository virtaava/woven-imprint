"""Bi-temporal structured facts: what the character knows, since when, and what it used to believe."""

from __future__ import annotations

import re

from .. import clock
from ..storage.sqlite import SQLiteStorage
from ..utils.text import generate_id
from .store import guard_embedding_dimension

_WS = re.compile(r"[\s_]+")


def normalize_key(subject: str, predicate: str) -> tuple[str, str]:
    s = _WS.sub("_", str(subject).strip().casefold()).strip("_")
    p = _WS.sub("_", str(predicate).strip().casefold()).strip("_")
    return s, p


def normalize_object(obj: str) -> str:
    return str(obj).strip().casefold().rstrip(".!?,;:")


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _norm_time(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip()
    if len(v) == 10 and _DATE_RE.match(v):  # YYYY-MM-DD
        return f"{v} 00:00:00"
    try:
        return clock.sqlite_ts(clock.parse_ts(v))
    except ValueError:
        return None


def _derive_statement(old_statement: str, old_object: str, new_object: str) -> str:
    """Derive a new statement for an object-only fact edit.

    Word-boundary, case-insensitive replace of the old object text in the old
    statement if it appears there as a whole word/phrase (not as a substring
    of a larger word — e.g. object "tea" must not match inside "steam");
    otherwise append a "— now: <new_object>." clause so the statement still
    reflects the update.
    """
    if old_object:
        pattern = re.compile(r"(?<!\w)" + re.escape(old_object) + r"(?!\w)", re.IGNORECASE)
        if pattern.search(old_statement):
            return pattern.sub(new_object, old_statement, count=1)
    return f"{old_statement.rstrip('.')} — now: {new_object}."


class FactStore:
    def __init__(self, storage: SQLiteStorage, character_id: str, embedder=None, embed_fn=None):
        self.storage = storage
        self.character_id = character_id
        # Used by `edit()` as a fallback to re-embed a linked memory when its
        # content changes, if `embed_fn` (below) isn't wired up.
        self.embedder = embedder
        # Optional `(content, role, created_at, metadata) -> list[float]`
        # callable — typically `MemoryStore.embed_for`, injected by Character
        # (`FactStore(..., embed_fn=char.memory.embed_for)`) — so `edit()`
        # re-embeds with the same contextualized text every other write path
        # uses. FactStore has no `MemoryStore` reference of its own (facts and
        # memories are independent tables), hence this seam rather than
        # calling `MemoryStore.embed_for` directly. Falls back to a raw
        # `embedder.embed(content)` (no date/speaker context, no dimension
        # guard) when absent, matching the pre-Tier-3d-fix-wave behavior for
        # direct/test construction that never wires it up.
        self.embed_fn = embed_fn

    def add(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,
        statement: str,
        event_time: str | None = None,
        certainty: float = 1.0,
        importance: float = 0.75,
        memory_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        s, p = normalize_key(subject, predicate)
        recorded = clock.sqlite_ts()
        et = _norm_time(event_time)
        fact = {
            "id": generate_id("fact-"),
            "character_id": self.character_id,
            "subject": s,
            "predicate": p,
            "object": str(object).strip(),
            "statement": statement.strip(),
            "event_time": et,
            "valid_from": et or recorded,
            "valid_to": None,
            "recorded_at": recorded,
            "expired_at": None,
            "certainty": certainty,
            "importance": importance,
            "memory_id": memory_id,
            "superseded_by": None,
            "session_id": session_id,
            "user_id": user_id,
            "metadata": metadata or {},
        }
        self.storage.save_fact(fact)
        return fact

    def get(self, fact_id: str) -> dict | None:
        return self.storage.get_fact(fact_id)

    def current(
        self, subject: str | None = None, predicate: str | None = None, limit: int | None = 200
    ) -> list[dict]:
        s, p = self._key(subject, predicate)
        return self.storage.query_facts(
            self.character_id, subject=s, predicate=p, active_only=True, limit=limit
        )

    def as_of(
        self, when: str, subject: str | None = None, predicate: str | None = None
    ) -> list[dict]:
        s, p = self._key(subject, predicate)
        return self.storage.query_facts(
            self.character_id, subject=s, predicate=p, as_of=_norm_time(when), limit=None
        )

    def history(self, subject: str, predicate: str) -> list[dict]:
        s, p = normalize_key(subject, predicate)
        return self.storage.query_facts(
            self.character_id,
            subject=s,
            predicate=p,
            active_only=False,
            limit=None,
            order="valid_asc",
        )

    def find_active(self, subject: str, predicate: str) -> dict | None:
        rows = self.current(subject, predicate, limit=1)
        return rows[0] if rows else None

    def expire(self, fact_id: str, *, valid_to: str, superseded_by: str | None) -> None:
        self.storage.expire_fact(
            fact_id,
            valid_to=_norm_time(valid_to) or clock.sqlite_ts(),
            expired_at=clock.sqlite_ts(),
            superseded_by=superseded_by,
        )

    def edit(
        self, fact_id: str, *, object: str | None = None, statement: str | None = None
    ) -> dict:
        """Edit a fact in place; if linked to a memory and `statement` changes (or is
        derived from an `object`-only edit), updates the memory's content too so text
        and record stay consistent — re-embedding through `embed_fn` (contextualized,
        dimension-guarded — see `MemoryStore.embed_for`) when wired up, else falling
        back to a raw `embedder.embed(statement)` (also dimension-guarded).

        When `object` is given without `statement`, the new statement is derived from
        the old one: a case-insensitive replace of the old object text if it appears
        in the old statement, else an appended "— now: <object>." clause.

        Shared-row guard: semantic dedup (`MemoryStore.add`'s `dedup_similarity`
        path) can link more than one fact to the same memory row. When the linked
        memory is still referenced by another active fact, a content change does
        NOT rewrite that shared row — a fresh memory is created for the edited
        statement and this fact is relinked to it instead, so the sibling fact's
        still-current text is left untouched.
        """
        f = self.get(fact_id)
        if f is None or f.get("character_id") != self.character_id:
            raise KeyError(fact_id)
        if object is not None and statement is None:
            statement = _derive_statement(f.get("statement") or "", f.get("object") or "", object)
        self.storage.update_fact_fields(fact_id, object=object, statement=statement)

        if statement is not None and f.get("memory_id"):
            memory_id = f["memory_id"]
            old_mem = self.storage.get_memory(memory_id)
            shared = self.storage.count_active_facts_for_memory(memory_id) > 1
            if shared and old_mem is not None:
                new_memory_id = self._relink_to_fresh_memory(statement, old_mem)
                self.storage.update_fact_fields(fact_id, memory_id=new_memory_id)
            elif self.embed_fn is not None or self.embedder is not None:
                embedding = self._embed(statement, old_mem)
                self.storage.update_memory_fields(memory_id, content=statement, embedding=embedding)
        updated = self.get(fact_id)
        assert updated is not None
        return updated

    def _embed(self, content: str, old_mem: dict | None) -> list[float]:
        """Embed `content` for a memory linked to a fact — contextualized via
        `embed_fn` (using `old_mem`'s role/created_at/metadata) when wired up,
        else a raw, dimension-guarded `embedder.embed(content)`."""
        if self.embed_fn is not None:
            role = old_mem.get("role") if old_mem else None
            created_at = old_mem.get("created_at") if old_mem else None
            metadata = old_mem.get("metadata") if old_mem else None
            return self.embed_fn(content, role, created_at, metadata)
        embedding = self.embedder.embed(content)
        guard_embedding_dimension(self.storage, embedding)
        return embedding

    def _relink_to_fresh_memory(self, statement: str, old_mem: dict) -> str:
        """Create a fresh memory row carrying the edited `statement` — copying
        `old_mem`'s tier/role/session/importance/certainty — and return its id.
        Used by `edit()`'s shared-row guard: the old row stays exactly as it
        was for whichever other active fact still references it."""
        embedding = self._embed(statement, old_mem)
        new_id = generate_id("mem-")
        meta = dict(old_mem.get("metadata") or {})
        meta.pop("fact_id", None)
        meta["relinked_from"] = old_mem["id"]
        new_memory = {
            "id": new_id,
            "character_id": self.character_id,
            "tier": old_mem.get("tier", "core"),
            "content": statement,
            "embedding": embedding,
            "importance": old_mem.get("importance", 0.75),
            "certainty": old_mem.get("certainty", 1.0),
            "status": "active",
            "source_refs": [],
            "session_id": old_mem.get("session_id"),
            "role": old_mem.get("role"),
            "metadata": meta,
        }
        self.storage.save_memory(new_memory)
        return new_id

    def retract(self, fact_id: str) -> dict:
        """Expire a fact now with no successor and mark it retracted; archives the
        linked memory (if it still exists) ONLY when no other active fact still
        references it — semantic dedup (`MemoryStore.add`'s `dedup_similarity`
        path) can link more than one fact to the same memory row, and retracting
        one of them must not pull the memory out from under the other."""
        f = self.get(fact_id)
        if f is None or f.get("character_id") != self.character_id:
            raise KeyError(fact_id)
        now = clock.sqlite_ts()
        if f.get("valid_to") is None:
            self.storage.expire_fact(fact_id, valid_to=now, expired_at=now, superseded_by=None)
        meta = dict(f.get("metadata") or {})
        meta["retracted"] = True
        self.storage.update_fact_fields(fact_id, metadata=meta)
        memory_id = f.get("memory_id")
        if memory_id and self.storage.get_memory(memory_id) is not None:
            if self.storage.count_active_facts_for_memory(memory_id) == 0:
                self.storage.update_memory_status(memory_id, "archived")
        updated = self.get(fact_id)
        assert updated is not None
        return updated

    def delete(self, fact_id: str) -> None:
        """Hard delete of the fact row only (memory untouched)."""
        f = self.get(fact_id)
        if f is None or f.get("character_id") != self.character_id:
            raise KeyError(fact_id)
        self.storage.delete_fact(fact_id)

    def bump_certainty(self, fact_id: str, delta: float = 0.15) -> None:
        f = self.get(fact_id)
        if f:
            f["certainty"] = max(0.0, min(1.0, float(f.get("certainty", 1.0)) + delta))
            self.storage.save_fact(f)

    def count(self) -> int:
        return self.storage.count_facts(self.character_id)

    @staticmethod
    def _key(subject, predicate):
        s = normalize_key(subject, "x")[0] if subject is not None else None
        p = normalize_key("x", predicate)[1] if predicate is not None else None
        return s, p
