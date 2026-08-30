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
    def __init__(self, storage: SQLiteStorage, character_id: str, embedder=None):
        self.storage = storage
        self.character_id = character_id
        # Used by `edit()` to re-embed a linked memory when its content changes.
        self.embedder = embedder

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
        derived from an `object`-only edit), updates the memory's content too
        (re-embedding through `self.embedder` when present) so text and record stay
        consistent.

        When `object` is given without `statement`, the new statement is derived from
        the old one: a case-insensitive replace of the old object text if it appears
        in the old statement, else an appended "— now: <object>." clause.
        """
        f = self.get(fact_id)
        if f is None or f.get("character_id") != self.character_id:
            raise KeyError(fact_id)
        if object is not None and statement is None:
            statement = _derive_statement(f.get("statement") or "", f.get("object") or "", object)
        self.storage.update_fact_fields(fact_id, object=object, statement=statement)
        if statement is not None and f.get("memory_id") and self.embedder is not None:
            embedding = self.embedder.embed(statement)
            guard_embedding_dimension(self.storage, embedding)
            self.storage.update_memory_fields(
                f["memory_id"], content=statement, embedding=embedding
            )
        updated = self.get(fact_id)
        assert updated is not None
        return updated

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
