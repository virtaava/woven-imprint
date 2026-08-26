# Tier 2 "Facts · Relationships · Relevance" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Structured bi-temporal facts with whole-store supersession, a relationship state machine in code (slow trust / fast betrayal / tiers / key moments / tension decay), and a relevance-first retrieval gate — all measured in the long-horizon benchmark.

**Architecture:** Schema v5 adds a `facts` table and `relationships.state`. `memory/facts.py::FactStore` owns bi-temporal queries; `Character._store_facts` supersedes by `(subject, predicate)` and links each fact to its memory row. `relationship/model.py::RelationshipModel.update` becomes a state machine gated by `relationship.dynamics`. `memory/retrieval.py` gates recency/importance/relationship lists to the semantic∪keyword eligible set. New prompt section version + snapshot drivers. Four new long-horizon benchmarks.

**Tech Stack:** Python 3.11+, sqlite3, existing prompt registry / clock / eval framework. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-26-tier2-facts-relationships-relevance.md`

## Global Constraints

- Branch `feat/tier2-facts-relationships` off master `9dfd13a`. Commit per task. Before each commit: `.venv/bin/python -m pytest -q` (baseline 408 passed, 3 skipped), `.venv/bin/ruff check src/ tests/ eval/`, `.venv/bin/ruff format src/ tests/ eval/`, `.venv/bin/pyright --project pyrightconfig.json` (0 errors).
- Timestamps via `woven_imprint.clock`; DB strings `"YYYY-MM-DD HH:MM:SS"`; date-only `event_time` normalized to `"YYYY-MM-DD 00:00:00"`.
- Public API additive only. With `relationship.dynamics=False`, `memory.relevance_gate=False`, `context.facts_block=False` and `unified_assessment=False`, behavior is byte-identical to master.
- Every new/changed prompt: version bump + snapshot driver in `tests/test_prompts_registry.py` (regenerate fixture with `UPDATE_SNAPSHOTS=1` only for the changed ids; other ids must stay identical).
- Never modify `experiments/parametric_spike/` or `kotlin/`.
- Config defaults exactly as in the spec. Tier thresholds exactly as in the spec.

---

## File structure

```
src/woven_imprint/
  storage/sqlite.py             _MIGRATIONS[5]; facts CRUD; relationships.state persisted
  memory/facts.py               NEW FactStore (add/current/as_of/history/find_active/expire/normalize)
  character.py                  self.facts; _parse_facts → dicts; _store_facts structured path; facts block in _build_context; export
  engine.py                     import_character restores facts
  persona/assessment.py         facts parsing accepts objects
  prompts.py                    TURN_ASSESSMENT_FACTS_SECTION v2; turn_assessment version=2
  relationship/model.py         dynamics state machine, tiers, key moments, decay, describe()
  memory/retrieval.py           relevance gate
  config.py                     RelationshipConfig fields; MemoryConfig.relevance_gate/relevance_semantic_topk; ContextConfig.facts_block/facts_block_limit
  mcp_server.py                 get_facts tool; get_stats facts_current
  server/services.py, server/demo.py   get_facts_service + GET /api/facts/{character_id}
eval/bench_longhorizon.py       4 new benchmarks; scripted structured facts
tests/ test_facts_store.py, test_structured_facts.py, test_facts_block.py, test_relationship_dynamics.py,
       test_relevance_gate.py, test_storage.py (v5), test_relationship.py (legacy pins), test_prompts_registry.py, test_longhorizon.py
docs/ SCHEMA.md, ARCHITECTURE.md, CONFIGURATION.md, README.md, CHANGELOG.md, RESULTS.md (generated)
```

---

### Task 1: Schema v5 + `FactStore`

**Files:**
- Modify: `src/woven_imprint/storage/sqlite.py` (`_MIGRATIONS`, new facts methods, `save_relationship`/`get_relationship(s)` handle `state`), `tests/test_storage.py` (version assertion 4→5)
- Create: `src/woven_imprint/memory/facts.py`, `tests/test_facts_store.py`
- Modify: `docs/SCHEMA.md` (version 5, facts table, relationships.state, protocol bullet)

**Interfaces:**
- `SQLiteStorage.save_fact(fact: dict) -> None` (upsert by id; fields per spec), `get_fact(fact_id) -> dict|None`, `query_facts(character_id, *, subject=None, predicate=None, active_only=True, as_of: str|None=None, limit: int|None=200, order="recorded_desc"|"valid_asc") -> list[dict]`, `expire_fact(fact_id, *, valid_to: str, expired_at: str, superseded_by: str|None) -> None`, `count_facts(character_id, active_only=True) -> int`, `delete_facts(character_id)` not needed (cascade).
- `get_relationship`/`get_relationships` return `state: dict` (json-decoded, default `{}`); `save_relationship` persists `rel.get("state", {})`.
- `FactStore(storage, character_id)` with methods from the spec; `normalize_key`, `normalize_object` are module functions.

- [ ] **Step 1: Failing tests**

`tests/test_facts_store.py`:
```python
from datetime import datetime, timedelta, timezone

import pytest

from woven_imprint import clock
from woven_imprint.memory.facts import FactStore, normalize_key, normalize_object
from woven_imprint.storage.sqlite import SQLiteStorage

T0 = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store():
    s = SQLiteStorage(":memory:")
    s.save_character("c1", "Ada", {})
    yield FactStore(s, "c1"), s
    s.close()


def test_schema_version_is_5():
    s = SQLiteStorage(":memory:")
    v = s._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert v == 5
    cols = {r[1] for r in s._conn.execute("PRAGMA table_info(facts)")}
    assert {"subject", "predicate", "object", "valid_from", "valid_to", "recorded_at", "expired_at", "memory_id"} <= cols
    rcols = {r[1] for r in s._conn.execute("PRAGMA table_info(relationships)")}
    assert "state" in rcols


def test_normalizers():
    assert normalize_key(" Has_Cat_Named ", "USER") == ("user", "has_cat_named")
    assert normalize_key("has cat named", "user") == ("user", "has_cat_named")
    assert normalize_object("Pixel.") == "pixel"


def test_add_current_and_supersede(store):
    facts, _ = store
    with clock.override(T0):
        a = facts.add(subject="user", predicate="has_cat_named", object="Pixel", statement="The visitor's cat is named Pixel.")
    assert a["valid_from"] == "2026-05-03 12:00:00" and a["valid_to"] is None
    with clock.override(T0 + timedelta(days=40)):
        b = facts.add(subject="user", predicate="has_cat_named", object="Moss", statement="The visitor's cat is named Moss.")
        facts.expire(a["id"], valid_to=b["valid_from"], superseded_by=b["id"])
    cur = facts.current(subject="user", predicate="has_cat_named")
    assert [f["object"] for f in cur] == ["Moss"]
    hist = facts.history("user", "has_cat_named")
    assert [f["object"] for f in hist] == ["Pixel", "Moss"]
    assert hist[0]["valid_to"] == b["valid_from"] and hist[0]["superseded_by"] == b["id"]
    assert hist[0]["expired_at"] == "2026-06-12 12:00:00"


def test_as_of_world_time(store):
    facts, _ = store
    with clock.override(T0):
        a = facts.add(subject="user", predicate="lives_in", object="Tampere", statement="Lives in Tampere.")
    with clock.override(T0 + timedelta(days=30)):
        b = facts.add(subject="user", predicate="lives_in", object="Oulu", statement="Lives in Oulu.", event_time="2026-06-01")
        facts.expire(a["id"], valid_to=b["valid_from"], superseded_by=b["id"])
    assert b["valid_from"] == "2026-06-01 00:00:00"
    assert [f["object"] for f in facts.as_of("2026-05-20 00:00:00", subject="user")] == ["Tampere"]
    assert [f["object"] for f in facts.as_of("2026-06-15 00:00:00", subject="user")] == ["Oulu"]


def test_find_active_uses_normalized_key(store):
    facts, _ = store
    facts.add(subject="User", predicate="Works As", object="a luthier", statement="Works as a luthier.")
    assert facts.find_active("user", "works_as")["object"] == "a luthier"
    assert facts.find_active("user", "plays") is None


def test_relationship_state_roundtrip(store):
    _, s = store
    s.save_relationship({"id": "rel-1", "character_id": "c1", "target_id": "t", "dimensions": {"trust": 0.1},
                         "state": {"updates": 3, "recent": [0.1]}})
    rel = s.get_relationship("c1", "t")
    assert rel["state"] == {"updates": 3, "recent": [0.1]}
    assert s.get_relationships("c1")[0]["state"]["updates"] == 3


def test_legacy_relationship_row_has_empty_state(store):
    _, s = store
    s._conn.execute("INSERT INTO relationships (id, character_id, target_id, dimensions) VALUES ('rel-x','c1','u','{}')")
    assert s.get_relationship("c1", "u")["state"] == {}
```
Run → FAIL (module missing / version 4).

- [ ] **Step 2: Migration + storage methods**

`storage/sqlite.py` — add to `_MIGRATIONS`:
```python
    5: """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    statement TEXT NOT NULL,
    event_time TEXT,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    recorded_at TEXT NOT NULL,
    expired_at TEXT,
    certainty REAL DEFAULT 1.0,
    importance REAL DEFAULT 0.75,
    memory_id TEXT,
    superseded_by TEXT,
    session_id TEXT,
    user_id TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(character_id, subject, predicate, valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_recorded ON facts(character_id, recorded_at DESC);
ALTER TABLE relationships ADD COLUMN state TEXT DEFAULT '{}';
""",
```
Methods (after the relationship methods):
```python
    # ── Facts (bi-temporal) ─────────────────────────────────────────────
    _FACT_COLS = ("id", "character_id", "subject", "predicate", "object", "statement", "event_time",
                  "valid_from", "valid_to", "recorded_at", "expired_at", "certainty", "importance",
                  "memory_id", "superseded_by", "session_id", "user_id", "metadata")

    def save_fact(self, fact: dict) -> None:
        row = {c: fact.get(c) for c in self._FACT_COLS}
        row["metadata"] = json.dumps(fact.get("metadata") or {})
        row["certainty"] = fact.get("certainty", 1.0)
        row["importance"] = fact.get("importance", 0.75)
        cols = ", ".join(self._FACT_COLS)
        marks = ", ".join("?" for _ in self._FACT_COLS)
        updates = ", ".join(f"{c}=excluded.{c}" for c in self._FACT_COLS if c != "id")
        with self._lock:
            self._conn.execute(
                f"INSERT INTO facts ({cols}) VALUES ({marks}) ON CONFLICT(id) DO UPDATE SET {updates}",
                tuple(row[c] for c in self._FACT_COLS),
            )
            self._commit()

    def _row_to_fact(self, row) -> dict:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except (TypeError, ValueError):
            d["metadata"] = {}
        return d

    def get_fact(self, fact_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return self._row_to_fact(row) if row else None

    def query_facts(self, character_id: str, *, subject: str | None = None, predicate: str | None = None,
                    active_only: bool = True, as_of: str | None = None, limit: int | None = 200,
                    order: str = "recorded_desc") -> list[dict]:
        q = "SELECT * FROM facts WHERE character_id = ?"
        params: list[Any] = [character_id]
        if subject is not None:
            q += " AND subject = ?"; params.append(subject)
        if predicate is not None:
            q += " AND predicate = ?"; params.append(predicate)
        if as_of is not None:
            q += " AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)"; params += [as_of, as_of]
        elif active_only:
            q += " AND valid_to IS NULL AND expired_at IS NULL"
        q += " ORDER BY valid_from ASC, rowid ASC" if order == "valid_asc" else " ORDER BY recorded_at DESC, rowid DESC"
        if limit is not None:
            q += " LIMIT ?"; params.append(limit)
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def expire_fact(self, fact_id: str, *, valid_to: str, expired_at: str, superseded_by: str | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE facts SET valid_to = ?, expired_at = ?, superseded_by = ? WHERE id = ?",
                (valid_to, expired_at, superseded_by, fact_id),
            )
            self._commit()

    def count_facts(self, character_id: str, active_only: bool = True) -> int:
        q = "SELECT COUNT(*) FROM facts WHERE character_id = ?"
        if active_only:
            q += " AND valid_to IS NULL AND expired_at IS NULL"
        with self._lock:
            return self._conn.execute(q, (character_id,)).fetchone()[0]
```
Relationships: add `state` to the INSERT column list/VALUES and `state=excluded.state` in the upsert (`json.dumps(rel.get("state") or {})`); in `get_relationship`/`get_relationships` add `d["state"] = json.loads(d.get("state") or "{}")`.

- [ ] **Step 3: `memory/facts.py`**

```python
"""Bi-temporal structured facts: what the character knows, since when, and what it used to believe."""
from __future__ import annotations

import re

from .. import clock
from ..storage.sqlite import SQLiteStorage
from ..utils.text import generate_id

_WS = re.compile(r"[\s_]+")


def normalize_key(subject: str, predicate: str) -> tuple[str, str]:
    s = _WS.sub("_", str(subject).strip().casefold()).strip("_")
    p = _WS.sub("_", str(predicate).strip().casefold()).strip("_")
    return s, p


def normalize_object(obj: str) -> str:
    return str(obj).strip().casefold().rstrip(".!?,;:")


def _norm_time(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip()
    if len(v) == 10:  # YYYY-MM-DD
        return f"{v} 00:00:00"
    try:
        return clock.sqlite_ts(clock.parse_ts(v))
    except ValueError:
        return None


class FactStore:
    def __init__(self, storage: SQLiteStorage, character_id: str):
        self.storage = storage
        self.character_id = character_id

    def add(self, *, subject: str, predicate: str, object: str, statement: str, event_time: str | None = None,
            certainty: float = 1.0, importance: float = 0.75, memory_id: str | None = None,
            session_id: str | None = None, user_id: str | None = None, metadata: dict | None = None) -> dict:
        s, p = normalize_key(subject, predicate)
        recorded = clock.sqlite_ts()
        et = _norm_time(event_time)
        fact = {
            "id": generate_id("fact-"), "character_id": self.character_id,
            "subject": s, "predicate": p, "object": str(object).strip(), "statement": statement.strip(),
            "event_time": et, "valid_from": et or recorded, "valid_to": None,
            "recorded_at": recorded, "expired_at": None,
            "certainty": certainty, "importance": importance, "memory_id": memory_id,
            "superseded_by": None, "session_id": session_id, "user_id": user_id, "metadata": metadata or {},
        }
        self.storage.save_fact(fact)
        return fact

    def get(self, fact_id: str) -> dict | None:
        return self.storage.get_fact(fact_id)

    def current(self, subject: str | None = None, predicate: str | None = None, limit: int | None = 200) -> list[dict]:
        s, p = self._key(subject, predicate)
        return self.storage.query_facts(self.character_id, subject=s, predicate=p, active_only=True, limit=limit)

    def as_of(self, when: str, subject: str | None = None, predicate: str | None = None) -> list[dict]:
        s, p = self._key(subject, predicate)
        return self.storage.query_facts(self.character_id, subject=s, predicate=p, as_of=_norm_time(when), limit=None)

    def history(self, subject: str, predicate: str) -> list[dict]:
        s, p = normalize_key(subject, predicate)
        return self.storage.query_facts(self.character_id, subject=s, predicate=p, active_only=False, limit=None, order="valid_asc")

    def find_active(self, subject: str, predicate: str) -> dict | None:
        rows = self.current(subject, predicate, limit=1)
        return rows[0] if rows else None

    def expire(self, fact_id: str, *, valid_to: str, superseded_by: str | None) -> None:
        self.storage.expire_fact(fact_id, valid_to=_norm_time(valid_to) or clock.sqlite_ts(),
                                 expired_at=clock.sqlite_ts(), superseded_by=superseded_by)

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
```

- [ ] **Step 4: docs/SCHEMA.md + test_storage + run + commit**

SCHEMA.md: "Current schema version: **5**"; add `### facts (migration v5)` with the columns and the bi-temporal semantics (valid_* = world time, recorded_at/expired_at = character's knowledge time; invalidate-never-delete); add `relationships.state` (JSON, dynamics state) under the relationships section; protocol bullet for v5. `tests/test_storage.py`: `assert version == 5`.
```bash
.venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "feat(storage): schema v5 — bi-temporal facts table, relationships.state; FactStore"
```

---

### Task 2: Structured extraction + whole-store supersession

**Files:**
- Modify: `prompts.py` (`TURN_ASSESSMENT_FACTS_SECTION`, `turn_assessment` version=2), `persona/assessment.py`, `character.py` (`self.facts`, `_parse_facts`, `_store_facts`, `_run_bookkeeping` call), `tests/helpers.py` (FakeLLM unified branch returns one structured fact), `tests/test_prompts_registry.py` (driver unchanged; fixture regenerated for `turn_assessment` only), `tests/fixtures/prompt_snapshots.json`
- Create: `tests/test_structured_facts.py`

**Interfaces:**
- `Character._parse_facts(result, max_facts) -> list[dict]`: each `{"statement": str, "subject": str|None, "predicate": str|None, "object": str|None, "event_time": str|None}`; strings become `{"statement": s, others None}`; entries with `len(statement) <= 10` dropped; slice AFTER filtering (fix the old slice-before-filter quirk; note it in the commit body).
- `Character._store_facts(facts: list[dict], user_id, session_id, importance)`; legacy `_extract_memories` wraps its string list via `_parse_facts` so it also passes dicts (unstructured branch = old behavior).
- `TurnAssessment.facts: list[dict]`.

- [ ] **Step 1: Failing tests**

`tests/test_structured_facts.py`:
```python
from datetime import datetime, timedelta, timezone

from tests.helpers import FakeLLM, make_test_engine
from woven_imprint import clock
from woven_imprint.character import Character

T0 = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


class FactLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.next_facts = []

    def generate_json(self, messages, temperature=0.3):
        head = messages[0]["content"].lower()
        if "bookkeeping assistant" in head:
            return {"emotion": {"mood": "content", "intensity": 0.3, "cause": ""},
                    "relationship": {"trust": 0.01}, "beat": None, "facts": self.next_facts}
        return super().generate_json(messages, temperature)

    def generate_json_robust(self, messages, temperature=0.3):
        return self.generate_json(messages, temperature)


def _char():
    engine = make_test_engine()
    engine.llm = FactLLM()
    orig = engine.create_character

    def create(*a, **kw):
        c = orig(*a, **kw)
        c.parallel = False; c.background = False; c.enforce_consistency = False; c.unified_assessment = True
        return c

    engine.create_character = create
    return engine, engine.create_character("Ada")


def test_parse_facts_accepts_strings_and_objects():
    out = Character._parse_facts(["The visitor moved to Oulu last spring.", "short", {"statement": "The visitor's cat is named Pixel.", "subject": "user", "predicate": "has_cat_named", "object": "Pixel", "event_time": "2026-04-01"}, {"statement": "x"}], 5)
    assert out[0] == {"statement": "The visitor moved to Oulu last spring.", "subject": None, "predicate": None, "object": None, "event_time": None}
    assert out[1]["predicate"] == "has_cat_named" and out[1]["event_time"] == "2026-04-01"
    assert len(out) == 2


def test_structured_supersession_across_whole_store():
    engine, char = _char()
    llm = engine.llm
    with clock.override(T0):
        llm.next_facts = [{"statement": "The visitor's cat is named Pixel.", "subject": "user", "predicate": "has_cat_named", "object": "Pixel"}]
        for _ in range(3):
            char.chat("hello", user_id="toni")   # extraction fires on turn 3
    pixel = char.facts.find_active("user", "has_cat_named")
    assert pixel and pixel["object"] == "Pixel" and pixel["memory_id"]
    # bury it under >60 unrelated core memories — beyond the old 50-row heuristic window
    llm.next_facts = []
    for i in range(70):
        char.memory.add(f"Routine note {i} about weather.", tier="core")
    with clock.override(T0 + timedelta(days=40)):
        llm.next_facts = [{"statement": "The visitor's cat is named Moss.", "subject": "user", "predicate": "has_cat_named", "object": "Moss", "event_time": "2026-06-10"}]
        for _ in range(3):
            char.chat("news", user_id="toni")
    cur = char.facts.current("user", "has_cat_named")
    assert [f["object"] for f in cur] == ["Moss"]
    hist = char.facts.history("user", "has_cat_named")
    assert hist[0]["object"] == "Pixel" and hist[0]["valid_to"] == "2026-06-10 00:00:00" and hist[0]["superseded_by"] == cur[0]["id"]
    old_mem = engine.storage.get_memory(pixel["memory_id"])
    assert old_mem["status"] == "contradicted"
    new_mem = engine.storage.get_memory(cur[0]["memory_id"])
    assert new_mem["metadata"]["fact_id"] == cur[0]["id"] and new_mem["metadata"]["contradicts"] == pixel["memory_id"]
    assert new_mem["metadata"]["user_id"] == "toni"


def test_same_object_reinforces_instead_of_duplicating():
    engine, char = _char()
    llm = engine.llm
    llm.next_facts = [{"statement": "The visitor lives in Oulu.", "subject": "user", "predicate": "lives_in", "object": "Oulu"}]
    for _ in range(6):
        char.chat("hi", user_id="toni")   # extraction on turns 3 and 6
    assert len(char.facts.history("user", "lives_in")) == 1
    f = char.facts.find_active("user", "lives_in")
    mem = engine.storage.get_memory(f["memory_id"])
    assert mem["certainty"] >= 1.0  # reinforced (clamped)


def test_unstructured_fact_keeps_legacy_behavior():
    engine, char = _char()
    engine.llm.next_facts = ["The visitor likes tea very much."]
    for _ in range(3):
        char.chat("hi", user_id="toni")
    assert char.facts.count() == 0
    assert any("likes tea" in m["content"] for m in char.memory.get_all(tier="core"))
```
Run → FAIL.

- [ ] **Step 2: Prompt + parsing**

`prompts.py`: replace `TURN_ASSESSMENT_FACTS_SECTION` with
```python
TURN_ASSESSMENT_FACTS_SECTION = (
    '"facts": up to {max_facts} objects — specific NEW facts, opinions, preferences, biographical '
    'details or commitments worth remembering long-term. Each object: {{"statement": one sentence, '
    '"subject": "user" | "self" | a name, "predicate": snake_case verb phrase (e.g. has_cat_named, '
    'lives_in, works_as, likes, dislikes, plans_to), "object": the value as a short phrase, '
    '"event_time": "YYYY-MM-DD" or null}}; [] if nothing notable.'
)
```
and set `turn_assessment` `version=2`, `expects="JSON: emotion + optional relationship/beat/facts (facts are objects)"`. Regenerate the snapshot for `turn_assessment` only: run `UPDATE_SNAPSHOTS=1 .venv/bin/python -m pytest tests/test_prompts_registry.py -q`, then `git diff tests/fixtures/prompt_snapshots.json` must show changes ONLY under the `turn_assessment` key.

`character.py`:
```python
    @staticmethod
    def _parse_facts(result, max_facts: int) -> list[dict]:
        raw = result if isinstance(result, list) else (result.get("facts", []) if isinstance(result, dict) else [])
        out: list[dict] = []
        for item in raw:
            if isinstance(item, str):
                stmt, rec = item, {}
            elif isinstance(item, dict):
                stmt, rec = str(item.get("statement") or ""), item
            else:
                continue
            stmt = stmt.strip()
            if len(stmt) <= 10:
                continue
            def _s(key):
                v = rec.get(key)
                return str(v).strip() if isinstance(v, (str, int, float)) and str(v).strip() else None
            out.append({"statement": stmt, "subject": _s("subject"), "predicate": _s("predicate"),
                        "object": _s("object"), "event_time": _s("event_time")})
            if len(out) >= max_facts:
                break
        return out
```
`persona/assessment.py`: `TurnAssessment.facts: list[dict]`; parsing call unchanged (it delegates). `tests/helpers.py` FakeLLM unified branch: `"facts": [{"statement": "A notable fact was shared", "subject": None, ...}]` — simplest: keep the string `"A notable fact was shared"` (strings still parse) so existing expectations hold.

- [ ] **Step 3: `_store_facts` structured path**

```python
    def _store_facts(self, facts: list[dict], user_id, session_id, importance: float) -> None:
        """Store extracted facts. Structured facts supersede by (subject, predicate) over the whole
        store; unstructured facts keep the legacy antonym heuristic."""
        for fact in facts:
            if isinstance(fact, str):  # tolerate legacy callers
                fact = {"statement": fact, "subject": None, "predicate": None, "object": None, "event_time": None}
            stmt = fact["statement"]
            if fact.get("subject") and fact.get("predicate") and fact.get("object"):
                self._store_structured_fact(fact, user_id, session_id, importance)
                continue
            existing = self.memory.get_all(tier="core", limit=50)
            contradictions = self.belief.detect_contradictions(stmt, existing)
            for old_mem in contradictions:
                self.belief.contradict(old_mem["id"], stmt, source="extraction", session_id=session_id)
            if not contradictions:
                self.memory.add(content=stmt, tier="core", role="observation", session_id=session_id,
                                importance=importance, metadata={"source": "extraction", "user_id": user_id})

    def _store_structured_fact(self, fact: dict, user_id, session_id, importance: float) -> None:
        from .memory.facts import normalize_object

        old = self.facts.find_active(fact["subject"], fact["predicate"])
        if old and normalize_object(old["object"]) == normalize_object(fact["object"]):
            if old.get("memory_id"):
                self.belief.reinforce(old["memory_id"])
            self.facts.bump_certainty(old["id"])
            return
        meta = {"source": "extraction", "user_id": user_id}
        if old and old.get("memory_id"):
            meta["contradicts"] = old["memory_id"]
        mem = self.memory.add(content=fact["statement"], tier="core", role="observation", session_id=session_id,
                              importance=importance, metadata=meta)
        new = self.facts.add(subject=fact["subject"], predicate=fact["predicate"], object=fact["object"],
                             statement=fact["statement"], event_time=fact.get("event_time"), importance=importance,
                             memory_id=mem["id"], session_id=session_id, user_id=user_id)
        mem["metadata"]["fact_id"] = new["id"]
        self.storage.save_memory(mem)
        if old:
            self.facts.expire(old["id"], valid_to=new["valid_from"], superseded_by=new["id"])
            if old.get("memory_id"):
                self.storage.update_memory_status(old["memory_id"], "contradicted", certainty=0.0)
```
`__init__`: `from .memory.facts import FactStore; self.facts = FactStore(storage, char_id)`. `_extract_memories` (legacy): pass `self._parse_facts(result, max_facts)` (dicts with None keys) into `_store_facts` — behavior unchanged. Check `save_memory` re-save keeps `created_at` (it honors the dict's `created_at`, present on the returned mem dict — confirm `memory.add` returns `created_at`; if not, re-fetch via `storage.get_memory(mem["id"])` before saving).

- [ ] **Step 4: Run, commit**

```bash
.venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "feat(facts): structured fact extraction with whole-store bi-temporal supersession"
```

---

### Task 3: "What I know" block, export/import, MCP + HTTP exposure

**Files:**
- Modify: `config.py` (ContextConfig `facts_block: bool = True`, `facts_block_limit: int = 12`), `character.py` (`_format_facts_block`, `_build_context`, `export`), `engine.py` (`import_character` facts), `mcp_server.py` (`get_facts`, `get_stats`), `server/services.py` (`get_facts_service`), `server/demo.py` (`GET /api/facts/{character_id}`), `docs/CONFIGURATION.md`
- Create: `tests/test_facts_block.py`

**Interfaces:**
- `Character._format_facts_block(user_id: str | None) -> str` → `""` when no facts; else the block from the spec (user facts; "previously: X" when `superseded` history exists; separate self block).
- `_build_context` inserts `("facts", f"\n\n{facts_text}")` between relationship and memories when `context.facts_block`.
- `export()["facts"] = self.facts.current(limit=None) + expired ones` (all rows: `storage.query_facts(..., active_only=False, limit=None)`); `import_character` re-inserts via `storage.save_fact` (ids preserved; memory_id references re-mapped if the memory id changed — memories are re-added with new ids, so store a `memory_content` fallback: on import, set `memory_id=None` and `metadata["imported_memory_content"]=…`; keep it simple and document).
- MCP `get_facts(character_id: str, subject: str | None = None, as_of: str | None = None) -> str` JSON list.
- HTTP `GET /api/facts/{character_id}?subject=&as_of=` → `{"facts": [...]}`.

- [ ] **Step 1: Failing tests**

`tests/test_facts_block.py`:
```python
from datetime import datetime, timedelta, timezone

from tests.helpers import make_test_engine
from woven_imprint import clock

T0 = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


def _char():
    engine = make_test_engine()
    char = engine.create_character("Ada")
    char.background = False; char.parallel = False; char.enforce_consistency = False
    return engine, char


def test_facts_block_lists_current_facts_with_previous():
    engine, char = _char()
    with clock.override(T0):
        a = char.facts.add(subject="user", predicate="has_cat_named", object="Pixel", statement="The visitor's cat is named Pixel.", user_id="toni")
        char.facts.add(subject="self", predicate="fears", object="forgetting", statement="I fear forgetting.")
    with clock.override(T0 + timedelta(days=40)):
        b = char.facts.add(subject="user", predicate="has_cat_named", object="Moss", statement="The visitor's cat is named Moss.", user_id="toni")
        char.facts.expire(a["id"], valid_to=b["valid_from"], superseded_by=b["id"])
        text = char._format_facts_block("toni")
    assert "What you currently know about toni" in text
    assert "(since 2026-06-12, previously: Pixel) The visitor's cat is named Moss." in text
    assert "Things you have said about yourself" in text and "I fear forgetting." in text


def test_facts_block_in_volatile_context_and_gated():
    from woven_imprint.config import get_config

    engine, char = _char()
    char.facts.add(subject="user", predicate="lives_in", object="Oulu", statement="The visitor lives in Oulu.", user_id="toni")
    char.chat("hi", user_id="toni")
    msgs = char.last_chat_messages
    volatile = msgs[1]["content"]
    assert "What you currently know about toni" in volatile and "Oulu" in volatile
    get_config().context.facts_block = False
    try:
        char.chat("again", user_id="toni")
        assert "What you currently know" not in char.last_chat_messages[1]["content"]
    finally:
        get_config().context.facts_block = True


def test_export_import_roundtrip_facts(tmp_path):
    engine, char = _char()
    char.facts.add(subject="user", predicate="lives_in", object="Oulu", statement="The visitor lives in Oulu.")
    data = char.export(str(tmp_path / "c.json"))
    assert data["facts"][0]["predicate"] == "lives_in"
    engine2 = make_test_engine()
    char2 = engine2.import_character(str(tmp_path / "c.json"))
    assert char2.facts.find_active("user", "lives_in")["object"] == "Oulu"
```
(Verify `last_chat_messages` exists on Character — the map says it does at `character.py:97-98`; confirm it holds the messages sent to the LLM. If it is not populated in the sync path, capture via a RecordingLLM instead.)

- [ ] **Step 2: Implement**

`character.py`:
```python
    def _format_facts_block(self, user_id: str | None) -> str:
        from .config import get_config

        ctx = get_config().context
        if not ctx.facts_block:
            return ""
        limit = ctx.facts_block_limit
        user_facts = [f for f in self.facts.current(subject="user", limit=None)
                      if not user_id or not f.get("user_id") or f.get("user_id") == user_id]
        user_facts.sort(key=lambda f: (-float(f.get("importance", 0.75)), f.get("recorded_at", "")), )
        user_facts = user_facts[:limit]
        lines: list[str] = []
        if user_facts:
            who = user_id or "the user"
            lines.append(f"What you currently know about {who} (facts you learned; dates are when they became true):")
            for f in user_facts:
                since = (f.get("valid_from") or "")[:10]
                prev = ""
                hist = self.facts.history(f["subject"], f["predicate"])
                older = [h for h in hist if h.get("superseded_by") == f["id"]]
                if older:
                    prev = f", previously: {older[-1]['object']}"
                lines.append(f"- (since {since}{prev}) {f['statement']}")
        self_facts = self.facts.current(subject="self", limit=5)
        if self_facts:
            lines.append("Things you have said about yourself:")
            for f in self_facts:
                lines.append(f"- {f['statement']}")
        return "\n".join(lines)
```
`_build_context`: compute `facts_text = self._format_facts_block(user_id)` — `_build_context` needs `user_id`; check its signature (`grep -n "def _build_context"`); if it lacks `user_id`, add an optional keyword `user_id: str | None = None` and pass it from both `chat` and `chat_stream` (public API of `_build_context` is private; keep positional args unchanged). Insert `if facts_text: optional_parts.append(("facts", f"\n\n{facts_text}"))` after the relationship part.

`export`: `"facts": self.storage.query_facts(self.id, active_only=False, limit=None)`. `import_character`: after relationships, `for f in data.get("facts", []): f = dict(f); f["character_id"] = char_id; f["memory_id"] = None; self.storage.save_fact(f)`.

MCP:
```python
@mcp.tool()
def get_facts(character_id: str, subject: str | None = None, as_of: str | None = None) -> str:
    """List what the character currently knows (structured facts), optionally as of a past date (YYYY-MM-DD)."""
    char = _get_character(character_id)
    if not char:
        return json.dumps({"error": f"Character {character_id} not found"})
    rows = char.facts.as_of(as_of, subject=subject) if as_of else char.facts.current(subject=subject)
    keep = ("subject", "predicate", "object", "statement", "valid_from", "valid_to", "certainty")
    return json.dumps([{k: r.get(k) for k in keep} for r in rows], indent=2)
```
`get_stats`: add `"facts_current": char.facts.count()`. `services.py`: `def get_facts_service(engine, character_id, subject=None, as_of=None)` mirroring above; `demo.py` route `@app.get("/api/facts/{character_id}")` with query params → `{"facts": rows}`; test in `tests/test_demo_server.py` style (one test: create character via service, add a fact, GET returns it — follow the existing test client pattern in that file).

CONFIGURATION.md: document `context.facts_block`, `context.facts_block_limit`.

- [ ] **Step 3: Run, commit**

```bash
.venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "feat(facts): 'What I know' prompt block, export/import, MCP get_facts, GET /api/facts"
```

---

### Task 4: Relationship dynamics — trust asymmetry, betrayal damping, key moments, trajectory window, tension decay, tiers

**Files:**
- Modify: `config.py` (RelationshipConfig fields per spec), `relationship/model.py`, `character.py` (`note=` at the two `update` call sites), `tests/test_relationship.py` (legacy pins set `dynamics=False`), `eval/bench_memory.py` (`bench_relationship_bounded_change` sets dynamics False explicitly OR keep — check it still passes: with dynamics, +1.0 trust → clamped 0.15 × 0.5 = 0.075 ≤ 0.15 ✓, −1.0 affection → −0.15 ✓; leave as is)
- Create: `tests/test_relationship_dynamics.py`

**Interfaces:**
- `RelationshipModel.update(target_id, deltas, new_type=None, *, note: str | None = None) -> dict`.
- `RelationshipModel.tier_for(dims: dict) -> tuple[str, float]` (tier name, affinity) — static.
- `rel["state"]` keys: `updates:int, recent:list[[net, tension_delta]], damping_left:int, betrayals:int, last_update_at:str`.
- `describe()` adds `  tier: {type} (affinity {aff:+.2f})` after the dimension lines and `  recent: <moment>` lines (≤2) before trajectory.

- [ ] **Step 1: Failing tests**

`tests/test_relationship_dynamics.py`:
```python
from datetime import datetime, timedelta, timezone

import pytest

from woven_imprint import clock
from woven_imprint.config import get_config
from woven_imprint.relationship.model import RelationshipModel
from woven_imprint.storage.sqlite import SQLiteStorage

T0 = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def rm():
    s = SQLiteStorage(":memory:")
    s.save_character("c1", "Ada", {})
    get_config().relationship.dynamics = True
    yield RelationshipModel(s, "c1")
    s.close()


def test_trust_gains_are_halved_losses_are_not(rm):
    with clock.override(T0):
        rm.update("u", {"trust": 0.10})
        assert abs(rm.get("u")["dimensions"]["trust"] - 0.05) < 1e-9
        rm.update("u", {"trust": -0.10})
        assert abs(rm.get("u")["dimensions"]["trust"] - (-0.05)) < 1e-9


def test_betrayal_damps_recovery_and_records_key_moment(rm):
    with clock.override(T0):
        for _ in range(4):
            rm.update("u", {"trust": 0.15})          # 4 × 0.075 = 0.30
        rm.update("u", {"trust": -0.12}, note="lied about the map")   # betrayal
        rel = rm.get("u")
        assert rel["state"]["damping_left"] == 10 and rel["state"]["betrayals"] == 1
        assert any("betrayal" in m and "lied about the map" in m for m in rel["key_moments"])
        before = rel["dimensions"]["trust"]
        rm.update("u", {"trust": 0.15})
        after = rm.get("u")["dimensions"]["trust"]
        assert abs((after - before) - 0.15 * 0.5 * 0.25) < 1e-9
        assert rm.get("u")["state"]["damping_left"] == 9


def test_key_moment_threshold_and_cap(rm):
    get_config().relationship.key_moments_limit = 3
    try:
        with clock.override(T0):
            for i in range(5):
                rm.update("u", {"affection": 0.09}, note=f"moment {i}")
        moments = rm.get("u")["key_moments"]
        assert len(moments) == 3 and moments[-1].endswith("moment 4")
        rm.update("u", {"affection": 0.02}, note="tiny")
        assert not any("tiny" in m for m in rm.get("u")["key_moments"])
    finally:
        get_config().relationship.key_moments_limit = 20


def test_trajectory_uses_window(rm):
    with clock.override(T0):
        for _ in range(5):
            rm.update("u", {"trust": 0.15, "affection": 0.15})
        assert rm.get("u")["trajectory"] == "warming"
        rm.update("u", {"trust": -0.05})            # one small dip must not flip a 5-turn warm streak
        assert rm.get("u")["trajectory"] == "warming"
        for _ in range(5):
            rm.update("u", {"trust": -0.15, "affection": -0.15})
        assert rm.get("u")["trajectory"] == "cooling"


def test_tension_decays_with_elapsed_days(rm):
    with clock.override(T0):
        rm.update("u", {"tension": 0.15})
        rm.update("u", {"tension": 0.15})
        assert abs(rm.get("u")["dimensions"]["tension"] - 0.30) < 1e-9
    with clock.override(T0 + timedelta(days=4)):
        rm.update("u", {"trust": 0.01})
        assert abs(rm.get("u")["dimensions"]["tension"] - 0.10) < 1e-9   # 0.30 − 4×0.05


def test_tiers():
    from woven_imprint.relationship.model import RelationshipModel as R

    assert R.tier_for({"trust": 0, "affection": 0, "respect": 0, "familiarity": 0.0, "tension": 0})[0] == "stranger"
    assert R.tier_for({"trust": 0, "affection": 0, "respect": 0, "familiarity": 0.2, "tension": 0})[0] == "acquaintance"
    assert R.tier_for({"trust": 0.4, "affection": 0.3, "respect": 0.2, "familiarity": 0.35, "tension": 0})[0] == "friend"
    assert R.tier_for({"trust": 0.8, "affection": 0.6, "respect": 0.5, "familiarity": 0.7, "tension": 0})[0] == "close_friend"
    assert R.tier_for({"trust": -0.6, "affection": -0.4, "respect": 0.0, "familiarity": 0.5, "tension": 0.5})[0] == "adversary"


def test_type_follows_tier_and_describe_shows_it(rm):
    with clock.override(T0):
        for _ in range(8):
            rm.update("u", {"trust": 0.15, "affection": 0.15, "respect": 0.15, "familiarity": 0.15})
    rel = rm.get("u")
    assert rel["type"] in ("friend", "close_friend")
    text = rm.describe("u")
    assert "tier:" in text and "affinity" in text


def test_dynamics_off_is_legacy(rm):
    get_config().relationship.dynamics = False
    try:
        with clock.override(T0):
            rm.update("u", {"trust": 0.10})
        assert abs(rm.get("u")["dimensions"]["trust"] - 0.10) < 1e-9
        assert rm.get("u")["type"] == "stranger" and rm.get("u")["key_moments"] == []
    finally:
        get_config().relationship.dynamics = True
```
Also in `tests/test_relationship.py`: add an autouse fixture that sets `get_config().relationship.dynamics = False` for that module (comment: "pins legacy arithmetic") and restores it.

- [ ] **Step 2: Config + model**

`config.py` RelationshipConfig:
```python
    max_delta: float = 0.15
    key_moments_limit: int = 20
    dynamics: bool = True
    trust_gain_factor: float = 0.5
    betrayal_threshold: float = -0.10
    betrayal_damping_turns: int = 10
    betrayal_gain_damping: float = 0.25
    key_moment_threshold: float = 0.08
    trajectory_window: int = 5
    tension_decay_per_day: float = 0.05
```
Env: `WOVEN_IMPRINT_RELATIONSHIP_DYNAMICS` → ("relationship","dynamics"). YAML template comments.

`relationship/model.py` — replace `update` and `add_key_moment`, add `tier_for`, extend `describe`:
```python
    TIERS = ("stranger", "acquaintance", "friend", "close_friend", "adversary")

    @staticmethod
    def tier_for(dims: dict) -> tuple[str, float]:
        aff = 0.5 * dims.get("trust", 0.0) + 0.3 * dims.get("affection", 0.0) + 0.2 * dims.get("respect", 0.0)
        fam = dims.get("familiarity", 0.0)
        if aff <= -0.3:
            return "adversary", aff
        if aff >= 0.5 and fam >= 0.6:
            return "close_friend", aff
        if aff >= 0.25 and fam >= 0.3:
            return "friend", aff
        if fam >= 0.1:
            return "acquaintance", aff
        return "stranger", aff

    def update(self, target_id: str, deltas: dict[str, float], new_type: str | None = None, *, note: str | None = None) -> dict:
        from ..config import get_config
        from .. import clock

        cfg = get_config().relationship
        rel = self.get_or_create(target_id)
        dims = rel["dimensions"]
        if not cfg.dynamics:
            return self._update_legacy(rel, deltas, new_type)

        state = rel.setdefault("state", {}) or {}
        state.setdefault("updates", 0); state.setdefault("recent", []); state.setdefault("damping_left", 0)
        state.setdefault("betrayals", 0)
        now = clock.now()
        last = state.get("last_update_at")
        if last:
            try:
                elapsed_days = max(0.0, (now - clock.parse_ts(last)).total_seconds() / 86400.0)
            except ValueError:
                elapsed_days = 0.0
            if elapsed_days > 0 and dims.get("tension", 0.0) > 0:
                dims["tension"] = _clamp(dims["tension"] - cfg.tension_decay_per_day * elapsed_days, 0.0, 1.0)

        today = now.date().isoformat()
        applied: dict[str, float] = {}
        for key, delta in deltas.items():
            if key not in dims:
                continue
            clamped = max(-cfg.max_delta, min(cfg.max_delta, float(delta)))
            if key == "trust":
                if clamped <= cfg.betrayal_threshold:
                    state["damping_left"] = cfg.betrayal_damping_turns
                    state["betrayals"] += 1
                    self._push_moment(rel, f"{today}: betrayal — trust {clamped:+.2f}" + (f" — {note}" if note else ""), cfg)
                elif clamped > 0:
                    clamped *= cfg.trust_gain_factor
                    if state["damping_left"] > 0:
                        clamped *= cfg.betrayal_gain_damping
            if key == "familiarity":
                dims[key] = _clamp(dims[key] + abs(clamped), 0.0, 1.0)
            elif key == "tension":
                dims[key] = _clamp(dims[key] + clamped, 0.0, 1.0)
            else:
                dims[key] = _clamp(dims[key] + clamped)
            applied[key] = clamped
            if abs(clamped) >= cfg.key_moment_threshold and not (key == "trust" and clamped <= cfg.betrayal_threshold):
                self._push_moment(rel, f"{today}: {key} {clamped:+.2f}" + (f" — {note}" if note else ""), cfg)
        if state["damping_left"] > 0:
            state["damping_left"] -= 1

        net = applied.get("trust", 0.0) + applied.get("affection", 0.0) + applied.get("respect", 0.0)
        state["recent"] = (state["recent"] + [[net, applied.get("tension", 0.0)]])[-cfg.trajectory_window:]
        net_sum = sum(r[0] for r in state["recent"])
        tension_sum = sum(abs(r[1]) for r in state["recent"])
        if net_sum > 0.1:
            rel["trajectory"] = "warming"
        elif net_sum < -0.1:
            rel["trajectory"] = "cooling"
        elif tension_sum > 0.1:
            rel["trajectory"] = "volatile"
        else:
            rel["trajectory"] = "stable"

        rel["type"] = new_type or self.tier_for(dims)[0]
        state["updates"] += 1
        state["last_update_at"] = clock.sqlite_ts(now)
        rel["state"] = state
        self.storage.save_relationship(rel)
        return rel

    def _update_legacy(self, rel: dict, deltas: dict[str, float], new_type: str | None) -> dict:
        # (move the existing update() body here verbatim, minus the get_or_create line)
        ...

    def _push_moment(self, rel: dict, moment: str, cfg) -> None:
        moments = rel.get("key_moments", []) or []
        moments.append(moment)
        rel["key_moments"] = moments[-cfg.key_moments_limit:]

    def add_key_moment(self, target_id: str, moment: str) -> None:
        from ..config import get_config
        rel = self.get_or_create(target_id)
        self._push_moment(rel, moment, get_config().relationship)
        self.storage.save_relationship(rel)
```
`describe()`: after the dimension lines add `tier, aff = self.tier_for(dims); lines.append(f"  tier: {rel.get('type', tier)} (affinity {aff:+.2f})")`; then `for m in (rel.get("key_moments") or [])[-2:]: lines.append(f"  recent: {m}")`; keep the trajectory line last. Note `get_or_create` should initialize `"state": {}`.

`character.py`: `self.relationships.update(user_id, out.relationship, note=message[:80])` in `_run_bookkeeping`; `_update_relationship_event`: `note=event[:80]`; legacy `_update_relationship`: `note=user_msg[:80]` (harmless when dynamics off — legacy path ignores note).

- [ ] **Step 3: Run (expect test_relationship.py legacy pins green via the autouse fixture), lint, commit**

```bash
.venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ eval/ && .venv/bin/ruff format src/ tests/ eval/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "feat(relationship): dynamics state machine — asymmetric trust, betrayal damping, key moments, windowed trajectory, tension decay, tiers"
```

---

### Task 5: Retrieval relevance gate

**Files:**
- Modify: `config.py` (MemoryConfig `relevance_gate: bool = True`, `relevance_semantic_topk: int = 100`), `memory/retrieval.py`, `docs/ARCHITECTURE.md` (Retrieval section)
- Create: `tests/test_relevance_gate.py`

- [ ] **Step 1: Failing test**

```python
from tests.helpers import make_test_engine
from woven_imprint.config import get_config


def _setup():
    engine = make_test_engine()
    char = engine.create_character("Ada", persona={"personality": "patient and wise", "speaking_style": "warm"})
    for i in range(30):   # bedrock flood, off-topic
        char.memory.add(f"[Self] I value {i} things about the archive and its silence.", tier="bedrock", importance=0.9)
    fresh = char.memory.add("The visitor dislikes tea and prefers coffee.", tier="core", importance=0.75)
    return engine, char, fresh


def test_gate_puts_on_topic_fact_first():
    engine, char, fresh = _setup()
    top = char.retriever.retrieve("tea", limit=5)
    assert top[0]["id"] == fresh["id"]


def test_gate_off_restores_legacy_flood():
    engine, char, fresh = _setup()
    get_config().memory.relevance_gate = False
    try:
        top = char.retriever.retrieve("tea", limit=5)
        # legacy: bedrock floor may outrank; we only assert the fresh fact is still retrievable
        assert any(m["id"] == fresh["id"] for m in top)
    finally:
        get_config().memory.relevance_gate = True


def test_empty_query_ignores_gate():
    engine, char, fresh = _setup()
    assert len(char.retriever.retrieve("", limit=5)) == 5
```

- [ ] **Step 2: Implement**

In `retrieve()` after `keyword_ranked` is built:
```python
        gated = all_memories
        if mem_cfg.relevance_gate and query.strip():
            eligible = set(semantic_ranked[: mem_cfg.relevance_semantic_topk]) | set(keyword_ranked)
            gated = [m for m in all_memories if m["id"] in eligible]
```
and build the recency, importance and relationship lists from `gated` instead of `all_memories`. Keep the final top-N materialization from `memory_map`. ARCHITECTURE.md: replace the "known ranking limitation" text with the gate description (and keep a sentence that `relevance_gate=False` restores the old fusion).

- [ ] **Step 3: Run, commit**

```bash
.venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "feat(retrieval): relevance gate — recency/importance re-rank only semantically or lexically relevant memories"
```

---

### Task 6: Long-horizon benchmarks for facts, betrayal, relevance

**Files:**
- Modify: `eval/bench_longhorizon.py` (scripted structured facts; four new benchmarks; `contradiction_supersession` regains global assertion), `docs/RESULTS.md` (regenerated), `README.md` claims table (rows for structured supersession + betrayal consequences move from Planned → Measured with benchmark names)

- [ ] **Step 1: Script + benchmarks**

`LongHorizonLLM.generate_json` bookkeeping branch: facts become objects — regular facts `{"statement": ..., "subject": "user", "predicate": f"mentioned_{noun}", "object": noun}`? No — keep regular facts unstructured strings (they must keep exercising the legacy memory path and dedup pressure). Add structured ones: day 12 → `{"statement": "The visitor's cat is named Pixel.", "subject": "user", "predicate": "has_cat_named", "object": "Pixel"}`; day 45 → same predicate, object "Moss", `event_time` = that day's date. Tea facts stay strings (they test the legacy heuristic). Relationship deltas unchanged (betrayal window days 41–50 already sends −0.10 trust → with dynamics that is exactly `betrayal_threshold` → betrayal fires).

Benchmarks appended in `_score`:
```python
    # 8 structured supersession (bi-temporal)
    cur = char.facts.current("user", "has_cat_named")
    hist = char.facts.history("user", "has_cat_named")
    day45 = (T0 + timedelta(days=44)).date().isoformat()
    as_of_day30 = char.facts.as_of((T0 + timedelta(days=29)).strftime("%Y-%m-%d %H:%M:%S"), subject="user", predicate="has_cat_named")
    ok8 = ([f["object"] for f in cur] == ["Moss"] and [f["object"] for f in hist] == ["Pixel", "Moss"]
           and (hist[0]["valid_to"] or "").startswith(day45) and [f["object"] for f in as_of_day30] == ["Pixel"]
           and engine.storage.get_memory(hist[0]["memory_id"])["status"] == "contradicted")
    out.append(BenchmarkResult("structured_supersession", ok8, 1.0 if ok8 else 0.0,
                               {"current": [f["object"] for f in cur], "history": [(f["object"], f["valid_to"]) for f in hist]}))
    # 9 facts block rendered
    text = char._format_facts_block("toni")
    ok9 = "What you currently know about toni" in text and "Moss" in text and "previously: Pixel" in text
    out.append(BenchmarkResult("facts_block_rendered", ok9, 1.0 if ok9 else 0.0, {"block": text[:300]}))
    # 10 betrayal has consequences
    rel = char.relationships.get("toni")
    ok10 = (trust_at[50] < trust_at[40] and trust_at[60] < trust_at[40] and trust_at[60] > trust_at[50]
            and any("betrayal" in m for m in rel["key_moments"]) and type_at[50] == "adversary"
            and trajectory_at[45] == "cooling")
    out.append(BenchmarkResult("betrayal_has_consequences", ok10, 1.0 if ok10 else 0.0,
               {"t40": trust_at[40], "t50": trust_at[50], "t60": trust_at[60], "type50": type_at[50], "traj45": trajectory_at[45],
                "moments": rel["key_moments"][-3:]}))
    # 11 relevance gate: global rank
    ranked = char.retriever.retrieve("tea", limit=20)
    ok11 = bool(ranked) and "dislikes tea" in ranked[0]["content"].lower()
    out.append(BenchmarkResult("relevance_gate_global_rank", ok11, 1.0 if ok11 else 0.0, {"top": ranked[0]["content"][:80] if ranked else None}))
```
`_simulate` records `type_at[day]` and `trajectory_at[day]` alongside `trust_at` and passes them to `_score` (extend the signature). `contradiction_supersession`: change `first_is_new` to use `ranked[0]` (global) now that the gate exists, keep the on-topic filter as the details.

- [ ] **Step 2: Run, iterate honestly (library fixes allowed if small and justified — report them), regenerate, README rows, commit**

```bash
.venv/bin/python eval/bench_longhorizon.py && .venv/bin/python -m pytest tests/test_longhorizon.py -q
.venv/bin/python eval/run_eval.py && .venv/bin/python eval/render_results.py
.venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ eval/ && .venv/bin/ruff format src/ tests/ eval/
git add -A && git commit -m "eval(longhorizon): structured supersession, facts block, betrayal consequences, relevance gate benchmarks"
```
README claims table: "Slow trust / lasting consequences of betrayal" → **Measured** (`bench_longhorizon: betrayal_has_consequences`); add row "Facts are structured and bi-temporal (what she believed on a given date)" → **Measured** (`structured_supersession`); "Contradictions supersede older beliefs" evidence → both `contradiction_supersession` and `structured_supersession`.

---

### Task 7: Legacy-parity guard test + snapshot hygiene

**Files:**
- Create: `tests/test_tier2_legacy_parity.py`

- [ ] **Step 1: Test that the four switches restore master behavior**

```python
from tests.helpers import make_test_engine
from woven_imprint.config import get_config


def test_all_tier2_switches_off_matches_legacy_shapes():
    cfg = get_config()
    cfg.relationship.dynamics = False
    cfg.memory.relevance_gate = False
    cfg.context.facts_block = False
    try:
        engine = make_test_engine()
        char = engine.create_character("Ada")
        char.background = False; char.parallel = False; char.enforce_consistency = False; char.unified_assessment = False
        char.chat("hello there friend", user_id="toni")
        rel = char.relationships.get("toni")
        assert rel["type"] == "stranger" and rel["key_moments"] == [] and rel["state"] in ({}, None) or "updates" not in (rel["state"] or {})
        assert "What you currently know" not in char.last_chat_messages[1]["content"]
        assert char.facts.count() == 0
    finally:
        cfg.relationship.dynamics = True
        cfg.memory.relevance_gate = True
        cfg.context.facts_block = True
```
Run, commit: `git commit -m "test: Tier 2 legacy-parity guard"`.

---

### Task 8: Docs + CHANGELOG

**Files:** `CHANGELOG.md`, `README.md` (Quick Start: `char.facts.current()`, `as_of`), `docs/ARCHITECTURE.md` (Facts, Relationship dynamics, Relevance gate sections), `docs/CONFIGURATION.md` (all new keys), `docs/SCHEMA.md` (verify v5 section from Task 1)

- [ ] **Step 1: CHANGELOG `[Unreleased]`**
```markdown
### Added (Tier 2 — facts · relationships · relevance)
- Schema v5: bi-temporal `facts` table (subject/predicate/object, valid_from/valid_to, recorded_at/expired_at,
  superseded_by) and `relationships.state`. Existing DBs migrate on open.
- `Character.facts` (`FactStore`): `current()`, `as_of(date)`, `history(subject, predicate)`; facts extracted as
  structured objects (turn_assessment v2) and superseded by `(subject, predicate)` across the whole store —
  the old 50-row antonym heuristic remains only for unstructured facts.
- "What you currently know about <user>" block in the volatile context (`context.facts_block`), with
  "previously: X" for superseded values.
- Relationship dynamics in code (`relationship.dynamics`): trust gains halved (`trust_gain_factor`), a
  single-turn trust drop ≤ `betrayal_threshold` damps trust gain for `betrayal_damping_turns`, automatic key
  moments, windowed trajectory, tension decay per elapsed day, derived tiers (stranger/acquaintance/friend/
  close_friend/adversary) shown to the model.
- Retrieval relevance gate (`memory.relevance_gate`): recency/importance/relationship lists re-rank only
  memories that are semantically or lexically relevant to the query.
- Long-horizon benchmarks: structured_supersession, facts_block_rendered, betrayal_has_consequences,
  relevance_gate_global_rank; `contradiction_supersession` asserts global rank again.
- MCP `get_facts`, `GET /api/facts/{character_id}`; export/import include facts.

### Changed
- `RelationshipModel.update` default behavior changed (dynamics on); set `relationship.dynamics: false` for
  the previous arithmetic. `key_moments_limit` is now honored.
- Kotlin C1 branch: must add the `facts` table, `relationships.state`, the relevance gate and the facts block
  before merging (schema version 5).
```
- [ ] **Step 2: Docs sections, final verification, commit**
```bash
.venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ eval/ && .venv/bin/ruff format --check src/ tests/ eval/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "docs: Tier 2 changelog, architecture, configuration, schema"
```

---

## Self-review
- **Spec coverage:** T2.1 schema/FactStore → T1; extraction + supersession → T2; prompt block/export/MCP/HTTP → T3. T2.2 → T4 (all config fields, tiers, key moments, decay, describe, note plumbing). T2.3 → T5. Benchmarks → T6. Acceptance parity → T7. Docs/CHANGELOG/Kotlin note → T8.
- **Placeholders:** the only elided block is `_update_legacy` ("move the existing body verbatim") — explicit instruction, not a TODO.
- **Type consistency:** `FactStore.add(...)` kwargs identical in T1/T2/T3/T6; `expire(fact_id, *, valid_to, superseded_by)` identical; `query_facts` kwargs identical between storage and FactStore; `update(..., note=)` used in T4 and by T2's call sites (T2 lands before T4 → T2 must NOT pass `note=` yet; T4 adds it — plan order respected); `_parse_facts` returns dicts everywhere after T2; `trust_at/type_at/trajectory_at` threaded through `_simulate`→`_score` in T6.
