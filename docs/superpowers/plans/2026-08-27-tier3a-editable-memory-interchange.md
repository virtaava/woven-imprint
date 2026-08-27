# Tier 3a "Editable Memory · Interchange" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Memories and facts become editable artifacts (edit / pin / delete / retract) from the library, HTTP, MCP and the demo X-Ray panel; pinned memories are always in the prompt; SillyTavern cards round-trip with lorebooks.

**Architecture:** Storage gains field-level UPDATE/DELETE for memories and facts (no migration; `pinned` in `metadata` via `json_extract`). `MemoryStore`/`FactStore` expose `edit/delete/pin/retract`. `_build_context` renders a non-sheddable pinned block. Demo server adds PATCH/DELETE routes + MCP tools. The React X-Ray panel gets pinned/edit/delete/facts controls and the bundle is rebuilt. The card parser reads `character_book`; `Character.export_card()` writes a V2 card.

**Tech Stack:** Python 3.11+, sqlite3 JSON1, FastAPI (demo extra), React 19 + Vite 8 + Tailwind 4 + shadcn/base-ui (existing `demo/node_modules`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-tier3a-editable-memory-interchange.md`

## Global Constraints

- Branch `feat/tier3a-editable-memory` off master `2d9ce9a`. Commit per task. Before each commit: `.venv/bin/python -m pytest -q` (baseline 443 passed, 3 skipped), `.venv/bin/ruff check src/ tests/ eval/`, `.venv/bin/ruff format src/ tests/ eval/`, `.venv/bin/pyright --project pyrightconfig.json` (0 errors).
- Demo-server tests need `fastapi`/`httpx` — install once with `uv pip install --python .venv/bin/python -e ".[demo]"` in Task 3 and keep it installed (these tests currently skip; they must RUN from Task 3 on).
- No schema migration. No new required deps. Public API additive. Pinned = `metadata["pinned"] is True`.
- Frontend: `cd demo && npm run build` regenerates `src/woven_imprint/demo_static/` (commit the bundle); run `npx tsc -b` for type errors; no new npm dependencies.
- Never touch `experiments/parametric_spike/` or `kotlin/`.

---

## File structure

```
src/woven_imprint/
  storage/sqlite.py        update_memory_fields, delete_memory, list_pinned_memories, update_fact_fields, delete_fact, unlink_fact_memory
  memory/store.py          edit, delete, pin, pinned
  memory/facts.py          edit, retract, delete
  character.py             _format_pinned_block, _build_context pinned base block, export_card
  config.py                ContextConfig.pinned_block/pinned_limit
  server/models.py         MemoryPatchRequest, FactPatchRequest
  server/services.py       recall strips embeddings; pinned/patch/delete services; facts include id
  server/demo.py           routes: GET /api/memory/pinned, PATCH/DELETE /api/memory/{id}, PATCH/DELETE /api/facts/{id}, GET /api/characters/{id}/card
  mcp_server.py            edit_memory, delete_memory, pin_memory, list_pinned, retract_fact, edit_fact
  migrate/parsers.py       character_book + extra fields, ccv3 chunk
  migrate/importer.py      lorebook → memories; scenario/system_prompt/greetings/tags
  cli.py                   export-card
demo/src/lib/{api.ts,types.ts}, demo/src/App.tsx, demo/src/components/XRayPanel.tsx, demo/src/components/FactsSection.tsx (new)
src/woven_imprint/demo_static/**  (rebuilt)
eval/bench_longhorizon.py  pinned_always_present
tests/ test_memory_edit.py, test_facts_edit.py, test_pinned_block.py, test_demo_memory_routes.py, test_card_import.py, test_card_export.py, test_mcp_memory_tools.py
docs/ UI_GUIDE.md, MIGRATION.md, README.md, CHANGELOG.md, CONTRIBUTING.md, ARCHITECTURE.md
```

---

### Task 1: Storage + MemoryStore + FactStore mutation API

**Files:**
- Modify: `src/woven_imprint/storage/sqlite.py`, `src/woven_imprint/memory/store.py`, `src/woven_imprint/memory/facts.py`
- Create: `tests/test_memory_edit.py`, `tests/test_facts_edit.py`

**Interfaces:**
```python
# storage
def update_memory_fields(self, memory_id, *, content=None, embedding=None, importance=None, tier=None, metadata=None) -> None
def delete_memory(self, memory_id) -> bool
def list_pinned_memories(self, character_id) -> list[dict]
def update_fact_fields(self, fact_id, *, object=None, statement=None, certainty=None, importance=None, metadata=None) -> None
def delete_fact(self, fact_id) -> bool
def unlink_fact_memory(self, memory_id) -> list[str]          # returns affected fact ids
# MemoryStore
def edit(self, memory_id, *, content=None, importance=None, tier=None) -> dict
def delete(self, memory_id) -> None                             # retracts linked facts
def pin(self, memory_id, pinned: bool = True) -> dict
def pinned(self) -> list[dict]
# FactStore
def edit(self, fact_id, *, object=None, statement=None, embed=None) -> dict   # embed: callable(str)->list[float] for the linked memory; MemoryStore passes its embedder
def retract(self, fact_id) -> dict
def delete(self, fact_id) -> None
```
`MemoryStore.__init__` gets `facts: FactStore | None = None` (Character wires `char.memory.facts = char.facts` after construction, or construct FactStore inside — choose: Character sets `self.memory.facts = self.facts` right after creating both).

- [ ] **Step 1: Failing tests**

`tests/test_memory_edit.py`:
```python
import pytest

from tests.helpers import make_test_engine


def _char():
    engine = make_test_engine()
    return engine, engine.create_character("Ada")


def test_edit_content_reembeds_and_keeps_created_at():
    engine, char = _char()
    m = char.memory.add("The visitor likes tea.", tier="core", importance=0.6)
    before = engine.storage.get_memory(m["id"])
    updated = char.memory.edit(m["id"], content="The visitor likes coffee.")
    row = engine.storage.get_memory(m["id"])
    assert row["content"] == "The visitor likes coffee." and updated["content"] == row["content"]
    assert row["created_at"] == before["created_at"]
    assert row["embedding"] != before["embedding"]
    assert [x["id"] for x in engine.storage.fts_search(char.id, "coffee")] == [m["id"]]
    assert engine.storage.fts_search(char.id, "tea") == []


def test_edit_importance_and_tier():
    engine, char = _char()
    m = char.memory.add("A routine note.", tier="buffer")
    char.memory.edit(m["id"], importance=0.9, tier="core")
    row = engine.storage.get_memory(m["id"])
    assert row["importance"] == 0.9 and row["tier"] == "core"
    with pytest.raises(ValueError):
        char.memory.edit(m["id"], tier="granite")
    with pytest.raises(KeyError):
        char.memory.edit("mem-nope", importance=0.1)


def test_pin_unpin_and_list():
    engine, char = _char()
    a = char.memory.add("Brother died in March.", tier="core")
    b = char.memory.add("Prefers window seats.", tier="core")
    char.memory.pin(a["id"])
    char.memory.pin(b["id"])
    char.memory.pin(b["id"], pinned=False)
    pinned = char.memory.pinned()
    assert [p["id"] for p in pinned] == [a["id"]]
    assert engine.storage.get_memory(a["id"])["metadata"]["pinned"] is True
    assert engine.storage.get_memory(b["id"])["metadata"].get("pinned") is False


def test_delete_removes_row_fts_and_retracts_linked_fact():
    engine, char = _char()
    m = char.memory.add("The visitor's cat is named Pixel.", tier="core")
    f = char.facts.add(subject="user", predicate="has_cat_named", object="Pixel", statement=m["content"], memory_id=m["id"])
    char.memory.delete(m["id"])
    assert engine.storage.get_memory(m["id"]) is None
    assert engine.storage.fts_search(char.id, "Pixel") == []
    fact = char.facts.get(f["id"])
    assert fact["memory_id"] is None and fact["expired_at"] is not None and fact["metadata"].get("retracted") is True
    assert char.facts.find_active("user", "has_cat_named") is None
```
`tests/test_facts_edit.py`:
```python
from tests.helpers import make_test_engine


def _char():
    engine = make_test_engine()
    return engine, engine.create_character("Ada")


def test_edit_fact_updates_record_and_linked_memory():
    engine, char = _char()
    m = char.memory.add("The visitor lives in Tampere.", tier="core")
    f = char.facts.add(subject="user", predicate="lives_in", object="Tampere", statement=m["content"], memory_id=m["id"])
    out = char.facts.edit(f["id"], object="Oulu", statement="The visitor lives in Oulu.")
    assert out["object"] == "Oulu"
    assert engine.storage.get_memory(m["id"])["content"] == "The visitor lives in Oulu."
    assert engine.storage.fts_search(char.id, "Oulu")


def test_retract_expires_without_successor_and_archives_memory():
    engine, char = _char()
    m = char.memory.add("The visitor plays chess.", tier="core")
    f = char.facts.add(subject="user", predicate="plays", object="chess", statement=m["content"], memory_id=m["id"])
    out = char.facts.retract(f["id"])
    assert out["valid_to"] is not None and out["superseded_by"] is None and out["metadata"]["retracted"] is True
    assert char.facts.current("user", "plays") == []
    assert engine.storage.get_memory(m["id"])["status"] == "archived"


def test_delete_fact_only():
    engine, char = _char()
    m = char.memory.add("Likes rye bread.", tier="core")
    f = char.facts.add(subject="user", predicate="likes", object="rye bread", statement=m["content"], memory_id=m["id"])
    char.facts.delete(f["id"])
    assert char.facts.get(f["id"]) is None
    assert engine.storage.get_memory(m["id"]) is not None
```
Run → FAIL.

- [ ] **Step 2: Storage**

```python
    _MEMORY_TIERS = ("buffer", "core", "bedrock")

    def update_memory_fields(self, memory_id: str, *, content: str | None = None, embedding: list[float] | None = None,
                             importance: float | None = None, tier: str | None = None, metadata: dict | None = None) -> None:
        sets, params = [], []
        if content is not None:
            if embedding is None:
                raise ValueError("content changes require a new embedding")
            sets += ["content = ?", "embedding = ?"]; params += [content, _serialize_embedding(embedding)]
        if importance is not None:
            sets.append("importance = ?"); params.append(max(0.0, min(1.0, float(importance))))
        if tier is not None:
            if tier not in self._MEMORY_TIERS:
                raise ValueError(f"invalid tier {tier!r}")
            sets.append("tier = ?"); params.append(tier)
        if metadata is not None:
            sets.append("metadata = ?"); params.append(json.dumps(metadata))
        if not sets:
            return
        params.append(memory_id)
        with self._lock:
            self._conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params)
            self._commit()

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._commit()
            return cur.rowcount > 0

    def list_pinned_memories(self, character_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT *, rowid FROM memories WHERE character_id = ? AND status = 'active' "
                "AND json_extract(metadata, '$.pinned') = 1 ORDER BY created_at ASC, rowid ASC",
                (character_id,),
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]

    def update_fact_fields(self, fact_id: str, *, object: str | None = None, statement: str | None = None,
                           certainty: float | None = None, importance: float | None = None, metadata: dict | None = None) -> None:
        sets, params = [], []
        if object is not None: sets.append("object = ?"); params.append(object)
        if statement is not None: sets.append("statement = ?"); params.append(statement)
        if certainty is not None: sets.append("certainty = ?"); params.append(max(0.0, min(1.0, float(certainty))))
        if importance is not None: sets.append("importance = ?"); params.append(max(0.0, min(1.0, float(importance))))
        if metadata is not None: sets.append("metadata = ?"); params.append(json.dumps(metadata))
        if not sets: return
        params.append(fact_id)
        with self._lock:
            self._conn.execute(f"UPDATE facts SET {', '.join(sets)} WHERE id = ?", params)
            self._commit()

    def delete_fact(self, fact_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            self._commit()
            return cur.rowcount > 0

    def unlink_fact_memory(self, memory_id: str) -> list[str]:
        with self._lock:
            ids = [r[0] for r in self._conn.execute("SELECT id FROM facts WHERE memory_id = ?", (memory_id,)).fetchall()]
            self._conn.execute("UPDATE facts SET memory_id = NULL WHERE memory_id = ?", (memory_id,))
            self._commit()
        return ids
```
Note: `sqlite3` `json_extract` on `'true'` JSON returns 1 — `json.dumps({"pinned": True})` stores `true`, so `= 1` works.

- [ ] **Step 3: MemoryStore + FactStore**

`memory/store.py`:
```python
    def edit(self, memory_id: str, *, content: str | None = None, importance: float | None = None, tier: str | None = None) -> dict:
        row = self.storage.get_memory(memory_id)
        if row is None or row.get("character_id") != self.character_id:
            raise KeyError(memory_id)
        embedding = None
        if content is not None and content != row["content"]:
            embedding = self.embedder.embed(content)
            guard_embedding_dimension(self.storage, embedding)
        else:
            content = None
        self.storage.update_memory_fields(memory_id, content=content, embedding=embedding, importance=importance, tier=tier)
        return self.storage.get_memory(memory_id)

    def delete(self, memory_id: str) -> None:
        row = self.storage.get_memory(memory_id)
        if row is None or row.get("character_id") != self.character_id:
            raise KeyError(memory_id)
        if self.facts is not None:
            for fid in self.storage.unlink_fact_memory(memory_id):
                self.facts.retract(fid)
        self.storage.delete_memory(memory_id)

    def pin(self, memory_id: str, pinned: bool = True) -> dict:
        row = self.storage.get_memory(memory_id)
        if row is None or row.get("character_id") != self.character_id:
            raise KeyError(memory_id)
        meta = dict(row.get("metadata") or {})
        meta["pinned"] = bool(pinned)
        self.storage.update_memory_fields(memory_id, metadata=meta)
        return self.storage.get_memory(memory_id)

    def pinned(self) -> list[dict]:
        return self.storage.list_pinned_memories(self.character_id)
```
with `self.facts = None` in `__init__` (typed `FactStore | None`, imported under `TYPE_CHECKING`).

`memory/facts.py`:
```python
    def edit(self, fact_id: str, *, object: str | None = None, statement: str | None = None, embed=None) -> dict:
        f = self.get(fact_id)
        if f is None or f.get("character_id") != self.character_id:
            raise KeyError(fact_id)
        self.storage.update_fact_fields(fact_id, object=object, statement=statement)
        if statement is not None and f.get("memory_id") and embed is not None:
            self.storage.update_memory_fields(f["memory_id"], content=statement, embedding=embed(statement))
        return self.get(fact_id)

    def retract(self, fact_id: str) -> dict:
        f = self.get(fact_id)
        if f is None or f.get("character_id") != self.character_id:
            raise KeyError(fact_id)
        now = clock.sqlite_ts()
        if f.get("valid_to") is None:
            self.storage.expire_fact(fact_id, valid_to=now, expired_at=now, superseded_by=None)
        meta = dict(f.get("metadata") or {}); meta["retracted"] = True
        self.storage.update_fact_fields(fact_id, metadata=meta)
        if f.get("memory_id") and self.storage.get_memory(f["memory_id"]) is not None:
            self.storage.update_memory_status(f["memory_id"], "archived")
        return self.get(fact_id)

    def delete(self, fact_id: str) -> None:
        f = self.get(fact_id)
        if f is None or f.get("character_id") != self.character_id:
            raise KeyError(fact_id)
        self.storage.delete_fact(fact_id)
```
`FactStore.edit(..., embed=...)`: `Character` wires `char.facts.embed = char.embedder.embed`? Simpler: `FactStore.__init__(storage, character_id, embedder=None)`; Character passes `embedder`; `edit` uses `self.embedder.embed` when present (drop the `embed` kwarg). Adopt that.

`character.py` `__init__`: `self.facts = FactStore(storage, char_id, embedder=embedder)` and `self.memory.facts = self.facts`.

- [ ] **Step 4: Run, commit**

```bash
.venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "feat(memory): edit/delete/pin memories and edit/retract/delete facts (library)"
```

---

### Task 2: Pinned block in the prompt + benchmark

**Files:**
- Modify: `config.py` (ContextConfig `pinned_block: bool = True`, `pinned_limit: int = 10`), `character.py` (`_format_pinned_block`, `_build_context`), `eval/bench_longhorizon.py` (`pinned_always_present`), `docs/RESULTS.md` (regen)
- Create: `tests/test_pinned_block.py`

- [ ] **Step 1: Failing tests**

```python
from tests.helpers import make_test_engine
from woven_imprint.config import get_config


def _char():
    engine = make_test_engine()
    char = engine.create_character("Ada")
    char.background = False; char.parallel = False; char.enforce_consistency = False
    return engine, char


def test_pinned_block_in_base_and_not_duplicated():
    engine, char = _char()
    m = char.memory.add("The visitor's brother died in March; never joke about it.", tier="core", importance=0.9)
    char.memory.pin(m["id"])
    char.chat("hello", user_id="toni")
    volatile = char.last_chat_messages[1]["content"]
    assert "Things you always remember:" in volatile
    assert volatile.count("brother died in March") == 1


def test_pinned_block_survives_tiny_budget():
    engine, char = _char()
    m = char.memory.add("Always call the visitor Captain.", tier="core")
    char.memory.pin(m["id"])
    for i in range(30):
        char.memory.add(f"Filler memory number {i} about the weather and errands.", tier="core")
    get_config().context.total_tokens = 400
    try:
        char.chat("hello", user_id="toni")
    finally:
        get_config().context.total_tokens = 6000
    assert "Always call the visitor Captain." in char.last_chat_messages[1]["content"]


def test_pinned_block_gated():
    engine, char = _char()
    m = char.memory.add("Pinned thing.", tier="core"); char.memory.pin(m["id"])
    get_config().context.pinned_block = False
    try:
        char.chat("hi", user_id="toni")
        assert "Things you always remember" not in char.last_chat_messages[1]["content"]
    finally:
        get_config().context.pinned_block = True
```

- [ ] **Step 2: Implement**

```python
    def _format_pinned_block(self) -> tuple[str, set[str]]:
        from .config import get_config

        ctx = get_config().context
        if not ctx.pinned_block:
            return "", set()
        rows = self.memory.pinned()[: ctx.pinned_limit]
        if not rows:
            return "", set()
        lines = ["Things you always remember:"]
        for m in rows:
            when = (m.get("created_at") or "")[:10]
            lines.append(f"- ({when}) {m['content'][:300]}" if when else f"- {m['content'][:300]}")
        return "\n".join(lines), {m["id"] for m in rows}
```
In `_build_context`: after the date line, `pinned_text, pinned_ids = self._format_pinned_block()`; if `pinned_text`: `volatile += ("\n\n" if volatile else "") + pinned_text`; then `memories = [m for m in memories if m["id"] not in pinned_ids]` BEFORE `memory_text = self._format_memories(memories)`; `base_size` already includes `len(volatile)`.

Benchmark: in `_simulate`, on day 3 after the session, `pinned_id = char.memory.add("The visitor asked to be reminded about the lighthouse key.", tier="core"); char.memory.pin(pinned_id["id"])`; in `_score`: run `char.chat("hello again", user_id="toni")` under the clock and check `char.last_chat_messages[1]["content"]` contains "lighthouse key" exactly once → `BenchmarkResult("pinned_always_present", ...)`. Regenerate RESULTS.md (`run_eval.py` + `render_results.py`) → 26/26.

- [ ] **Step 3: Run, commit**

```bash
.venv/bin/python eval/bench_longhorizon.py | tail -3 && .venv/bin/python eval/run_eval.py >/dev/null && .venv/bin/python eval/render_results.py && .venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ eval/ && .venv/bin/ruff format src/ tests/ eval/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "feat(prompt): pinned memories always in the prompt; pinned_always_present benchmark"
```

---

### Task 3: HTTP routes + MCP tools + demo bug fixes

**Files:**
- Modify: `server/models.py`, `server/services.py`, `server/demo.py`, `mcp_server.py`
- Create: `tests/test_demo_memory_routes.py`, `tests/test_mcp_memory_tools.py`

- [ ] **Step 1: Install demo extra so server tests run**

`uv pip install --python .venv/bin/python -e ".[demo]"` then `.venv/bin/python -m pytest tests/test_demo_server.py -q` → must run (not skip) and pass.

- [ ] **Step 2: Failing tests**

`tests/test_demo_memory_routes.py` (reuse the `app_client` fixture pattern from `tests/test_demo_server.py` — copy `_create_test_character` helper if needed):
```python
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_demo_server import app_client, _create_test_character  # noqa: F401,E402


def _setup(app_client):
    client, _t, engine = app_client
    cid = _create_test_character(client, "EditChar")["id"]
    char = engine.get_character(cid)
    m = char.memory.add("The visitor likes tea.", tier="core", importance=0.6)
    return client, engine, cid, m


def test_recall_strips_embeddings_and_has_ids(app_client):
    client, engine, cid, m = _setup(app_client)
    r = client.get("/api/memory", params={"character_id": cid, "query": "tea"})
    assert r.status_code == 200
    mem = r.json()["memories"][0]
    assert "embedding" not in mem and mem["id"] == m["id"] and "created_at" in mem


def test_patch_memory_content_importance_tier_pin(app_client):
    client, engine, cid, m = _setup(app_client)
    r = client.patch(f"/api/memory/{m['id']}", json={"character_id": cid, "content": "The visitor likes coffee.", "importance": 0.9, "tier": "bedrock", "pinned": True})
    assert r.status_code == 200
    mem = r.json()["memory"]
    assert mem["content"] == "The visitor likes coffee." and mem["tier"] == "bedrock" and mem["metadata"]["pinned"] is True
    assert "embedding" not in mem
    r = client.get("/api/memory/pinned", params={"character_id": cid})
    assert [x["id"] for x in r.json()["memories"]] == [m["id"]]


def test_patch_memory_validation_and_404(app_client):
    client, engine, cid, m = _setup(app_client)
    assert client.patch(f"/api/memory/{m['id']}", json={"character_id": cid, "tier": "granite"}).status_code == 400
    assert client.patch(f"/api/memory/{m['id']}", json={"character_id": cid, "importance": 2}).status_code == 400
    assert client.patch("/api/memory/mem-nope", json={"character_id": cid, "importance": 0.5}).status_code == 404
    other = _create_test_character(client, "Other")["id"]
    assert client.patch(f"/api/memory/{m['id']}", json={"character_id": other, "importance": 0.5}).status_code == 404


def test_delete_memory(app_client):
    client, engine, cid, m = _setup(app_client)
    assert client.delete(f"/api/memory/{m['id']}", params={"character_id": cid}).json() == {"deleted": True}
    assert client.delete(f"/api/memory/{m['id']}", params={"character_id": cid}).status_code == 404


def test_facts_patch_retract_delete(app_client):
    client, engine, cid, m = _setup(app_client)
    char = engine.get_character(cid)
    f = char.facts.add(subject="user", predicate="likes", object="tea", statement=m["content"], memory_id=m["id"])
    r = client.get(f"/api/facts/{cid}")
    assert r.json()["facts"][0]["id"] == f["id"]
    r = client.patch(f"/api/facts/{f['id']}", json={"character_id": cid, "object": "coffee", "statement": "The visitor likes coffee."})
    assert r.status_code == 200 and r.json()["fact"]["object"] == "coffee"
    r = client.delete(f"/api/facts/{f['id']}", params={"character_id": cid})   # default mode=retract
    assert r.status_code == 200 and r.json()["fact"]["metadata"]["retracted"] is True
    assert client.delete(f"/api/facts/{f['id']}", params={"character_id": cid, "mode": "delete"}).json() == {"deleted": True}
    assert client.patch(f"/api/facts/{f['id']}", json={"character_id": cid, "object": "x"}).status_code == 404


def test_unauthed_mutations_rejected(app_client):
    client, engine, cid, m = _setup(app_client)
    from starlette.testclient import TestClient
    un = TestClient(client.app, base_url="http://127.0.0.1:7860")
    assert un.patch(f"/api/memory/{m['id']}", json={"character_id": cid, "importance": 0.5}).status_code == 401
    assert un.delete(f"/api/memory/{m['id']}", params={"character_id": cid}).status_code == 401
```
`tests/test_mcp_memory_tools.py`: `pytest.importorskip("mcp")`; use the existing pattern in `tests/test_mcp*.py` if present (grep) to call tool functions directly with a test engine (`mcp_server._engine` / `_char_cache` monkeypatched); assert `edit_memory`, `pin_memory` + `list_pinned`, `delete_memory`, `retract_fact`, `edit_fact` JSON outputs and the `error` shape for a bad id. If no MCP test harness exists, create the minimal one: `monkeypatch.setattr(mcp_server, "_engine", make_test_engine())`, clear `_char_cache`.

- [ ] **Step 3: Implement**

`server/models.py`:
```python
class MemoryPatchRequest(BaseModel):
    character_id: str
    content: str | None = Field(None, max_length=50_000)
    importance: float | None = Field(None, ge=0.0, le=1.0)
    tier: str | None = Field(None, pattern="^(buffer|core|bedrock)$")
    pinned: bool | None = None


class FactPatchRequest(BaseModel):
    character_id: str
    object: str | None = Field(None, max_length=2_000)
    statement: str | None = Field(None, max_length=10_000)
```
(Pydantic validation errors return 422 by default — the tests expect 400 for bad tier/importance: add an exception handler in `create_app` mapping `RequestValidationError` → 400 for `/api/memory/*` and `/api/facts/*` paths only, or validate manually in the route and raise 400. Choose the manual route validation (simpler, explicit): keep the fields untyped-loose (`tier: str | None`, `importance: float | None`) and check in the handler.)

`server/services.py`:
```python
_MEMORY_KEEP = ("id", "tier", "content", "importance", "certainty", "status", "created_at", "accessed_at", "metadata", "session_id", "role")

def _public_memory(m: dict) -> dict:
    return {k: m.get(k) for k in _MEMORY_KEEP}

def recall_memories_service(...):   # memories = [_public_memory(m) for m in memories]
def list_pinned_service(engine, character_id): char = engine.get_character(character_id); return [_public_memory(m) for m in char.memory.pinned()]
def patch_memory_service(char, memory_id, *, content=None, importance=None, tier=None, pinned=None) -> dict:
    if tier is not None and tier not in ("buffer", "core", "bedrock"): raise ValueError("invalid tier")
    if importance is not None and not (0.0 <= float(importance) <= 1.0): raise ValueError("importance must be within [0, 1]")
    if content is not None or importance is not None or tier is not None:
        char.memory.edit(memory_id, content=content, importance=importance, tier=tier)
    if pinned is not None:
        char.memory.pin(memory_id, pinned)
    row = char.storage.get_memory(memory_id)
    if row is None: raise KeyError(memory_id)
    return _public_memory(row)
def delete_memory_service(char, memory_id) -> None: char.memory.delete(memory_id)
def patch_fact_service(char, fact_id, *, object=None, statement=None) -> dict: return _public_fact(char.facts.edit(fact_id, object=object, statement=statement))
def retract_fact_service(char, fact_id) -> dict: return _public_fact(char.facts.retract(fact_id))
def delete_fact_service(char, fact_id) -> None: char.facts.delete(fact_id)
```
`_FACT_KEEP` gains `"id", "superseded_by", "memory_id", "metadata"`; `_public_fact(r)`.

`server/demo.py` routes (all `dependencies=[Depends(_check_auth)]`, mutation ones inside `async with _character_mutation(cid)` using `_get_character`):
```python
    @app.get("/api/memory/pinned", dependencies=[Depends(_check_auth)])
    async def pinned_memories(character_id: str): ... KeyError → 404 ...; return {"memories": list_pinned_service(_engine, character_id)}

    @app.patch("/api/memory/{memory_id}", dependencies=[Depends(_check_auth)])
    async def patch_memory(memory_id: str, body: MemoryPatchRequest):
        async with _character_mutation(body.character_id):
            try: char = _get_character(body.character_id)
            except KeyError: raise HTTPException(404, ...)
            try: mem = patch_memory_service(char, memory_id, content=body.content, importance=body.importance, tier=body.tier, pinned=body.pinned)
            except KeyError: raise HTTPException(404, f"Memory '{memory_id}' not found")
            except ValueError as exc: raise HTTPException(400, str(exc))
        return {"memory": mem}

    @app.delete("/api/memory/{memory_id}", ...)  async def delete_memory(memory_id: str, character_id: str): ... → {"deleted": True}; KeyError → 404
    @app.patch("/api/facts/{fact_id}", ...)      body FactPatchRequest → {"fact": ...}
    @app.delete("/api/facts/{fact_id}", ...)     (fact_id, character_id, mode: str = "retract") → mode=="delete" → {"deleted": True} else {"fact": ...}; unknown mode → 400
```
Route ordering: declare `GET /api/memory/pinned` BEFORE any `/api/memory/{memory_id}` GET if one is added (none is); PATCH/DELETE with a path param don't conflict with the existing `GET /api/memory`.

Rate-limit bucket: the existing middleware assigns buckets by method (check `_enforce_rate_limit`) — PATCH/DELETE should land in `mutation`; verify and adjust the method set if it only lists POST.

`mcp_server.py` tools (mirror existing style, JSON strings, `{"error": ...}` on failure):
`edit_memory(character_id, memory_id, content=None, importance=None, tier=None)`, `delete_memory(character_id, memory_id)`, `pin_memory(character_id, memory_id, pinned=True)`, `list_pinned(character_id)`, `retract_fact(character_id, fact_id)`, `edit_fact(character_id, fact_id, object=None, statement=None)` — each strips embeddings via the services' public shapers (import from `server.services` is fine; it has no FastAPI import at module level — verify; if it does, duplicate the two small shapers in `mcp_server.py`).

- [ ] **Step 4: Run, commit**

```bash
.venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "feat(api): memory/fact edit, pin, delete, retract over HTTP and MCP; recall no longer leaks embeddings"
```

---

### Task 4: Demo UI — pinned, edit, delete, facts; bundle rebuild

**Files:**
- Modify: `demo/src/lib/types.ts`, `demo/src/lib/api.ts`, `demo/src/App.tsx`, `demo/src/components/XRayPanel.tsx`
- Create: `demo/src/components/FactsSection.tsx`, `demo/src/components/MemoryItem.tsx`
- Rebuild: `src/woven_imprint/demo_static/**`
- Create: `tests/test_demo_bundle.py`

- [ ] **Step 1: Bundle freshness test (fails until rebuilt)**

```python
from pathlib import Path


def test_built_bundle_references_new_endpoints():
    static = Path(__file__).resolve().parent.parent / "src" / "woven_imprint" / "demo_static" / "assets"
    js = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in static.glob("*.js"))
    for needle in ("/api/memory/pinned", "/api/facts/", "Things the character always remembers"):
        assert needle in js, f"bundle stale: {needle} missing — run `cd demo && npm run build`"
```

- [ ] **Step 2: types + api client**

`types.ts`: `Memory` → `{ id: string; content: string; tier: string; importance: number; certainty?: number; created_at?: string; metadata?: { pinned?: boolean; fact_id?: string; historical?: boolean; [k: string]: unknown } }`; add
```ts
export interface Fact {
  id: string; subject: string; predicate: string; object: string; statement: string
  valid_from: string; valid_to: string | null; certainty: number; memory_id?: string | null
}
```
`api.ts`: add
```ts
async function request<T = unknown>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers: headers(), ...init })
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch { /* keep statusText */ }
    throw new Error(`${res.status} ${detail}`)
  }
  return res.json() as Promise<T>
}
export const fetchPinned = (characterId: string) =>
  request<{ memories: Memory[] }>(`/api/memory/pinned?${new URLSearchParams({ character_id: characterId })}`)
export const patchMemory = (characterId: string, memoryId: string, patch: { content?: string; importance?: number; tier?: string; pinned?: boolean }) =>
  request<{ memory: Memory }>(`/api/memory/${memoryId}`, { method: 'PATCH', body: JSON.stringify({ character_id: characterId, ...patch }) })
export const deleteMemory = (characterId: string, memoryId: string) =>
  request<{ deleted: boolean }>(`/api/memory/${memoryId}?${new URLSearchParams({ character_id: characterId })}`, { method: 'DELETE' })
export const fetchFacts = (characterId: string, subject = 'user') =>
  request<{ facts: Fact[] }>(`/api/facts/${characterId}?${new URLSearchParams({ subject })}`)
export const patchFact = (characterId: string, factId: string, patch: { object?: string; statement?: string }) =>
  request<{ fact: Fact }>(`/api/facts/${factId}`, { method: 'PATCH', body: JSON.stringify({ character_id: characterId, ...patch }) })
export const retractFact = (characterId: string, factId: string) =>
  request<{ fact: Fact }>(`/api/facts/${factId}?${new URLSearchParams({ character_id: characterId })}`, { method: 'DELETE' })
```
and migrate `fetchRelationship` to `request` returning `res.relationship` (fixes the envelope bug); import `Memory`, `Fact` types.

- [ ] **Step 3: Components**

`MemoryItem.tsx`:
```tsx
import { useState } from 'react'
import { Pin, PinOff, Pencil, Trash2, Check, X } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { Memory } from '@/lib/types'

const TIER_COLORS: Record<string, string> = {
  bedrock: 'text-amber-400 border-amber-400/30 bg-amber-400/10',
  core: 'text-blue-400 border-blue-400/30 bg-blue-400/10',
  buffer: 'text-zinc-400 border-zinc-400/30 bg-zinc-400/10',
}

interface Props {
  memory: Memory
  onPin: (id: string, pinned: boolean) => void
  onEdit: (id: string, content: string) => Promise<void>
  onDelete: (id: string) => void
}

export function MemoryItem({ memory, onPin, onEdit, onDelete }: Props) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(memory.content)
  const [busy, setBusy] = useState(false)
  const pinned = memory.metadata?.pinned === true
  const date = memory.created_at?.slice(0, 10)

  const save = async () => {
    if (!draft.trim() || draft === memory.content) { setEditing(false); return }
    setBusy(true)
    try { await onEdit(memory.id, draft.trim()); setEditing(false) } finally { setBusy(false) }
  }

  return (
    <div className={`flex flex-col gap-1 rounded-md border p-2 ${pinned ? 'border-amber-400/40 bg-amber-400/5' : 'border-border/50 bg-background/50'}`}>
      <div className="flex items-center gap-2">
        <Badge variant="outline" className={`text-[10px] px-1.5 py-0 h-4 ${TIER_COLORS[memory.tier] || ''}`}>{memory.tier}</Badge>
        <span className="text-[10px] text-muted-foreground">{Math.round(memory.importance * 100)}% imp.</span>
        {date && <span className="text-[10px] text-muted-foreground">{date}</span>}
        <span className="ml-auto flex items-center gap-0.5">
          <Button variant="ghost" size="icon-xs" title={pinned ? 'Unpin' : 'Pin (always in prompt)'} onClick={() => onPin(memory.id, !pinned)}>
            {pinned ? <PinOff className="size-3" /> : <Pin className="size-3" />}
          </Button>
          <Button variant="ghost" size="icon-xs" title="Edit" onClick={() => { setDraft(memory.content); setEditing(true) }}><Pencil className="size-3" /></Button>
          <Button variant="ghost" size="icon-xs" title="Delete" onClick={() => { if (window.confirm('Forget this memory? This cannot be undone.')) onDelete(memory.id) }}><Trash2 className="size-3" /></Button>
        </span>
      </div>
      {editing ? (
        <div className="flex flex-col gap-1">
          <textarea className="w-full rounded-md border border-border bg-background p-1 text-xs" rows={3} value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Escape') setEditing(false); if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void save() }} />
          <div className="flex gap-1">
            <Button size="xs" disabled={busy} onClick={() => void save()}><Check className="size-3" /> Save</Button>
            <Button size="xs" variant="ghost" onClick={() => setEditing(false)}><X className="size-3" /> Cancel</Button>
          </div>
        </div>
      ) : (
        <p className="text-xs leading-relaxed text-foreground/80">{memory.content}</p>
      )}
    </div>
  )
}
```
(Check `button.tsx` for the available `size` variants — use the closest existing ones, e.g. `sm`/`icon`, if `xs`/`icon-xs` don't exist; do not add variants unless trivial.)

`FactsSection.tsx`: list `facts` (current, subject user) as rows `statement` + `(since YYYY-MM-DD)` + inline object edit (input + save) + "Retract" button (confirm). Props: `facts: Fact[]`, `onEditObject(id, object)`, `onRetract(id)`.

`XRayPanel.tsx`: new props `pinned: Memory[]`, `facts: Fact[]`, `onPin`, `onEditMemory`, `onDeleteMemory`, `onEditFact`, `onRetractFact`; render a **Pinned** card (title "Things the character always remembers") above Memory Feed using `MemoryItem`; Memory Feed rows → `MemoryItem` keyed by `id` (exclude pinned ids); Facts card after Relationship Radar.

`App.tsx`: state `pinned`, `facts`; `refreshXRay` also `fetchPinned` + `fetchFacts`; handlers call `patchMemory`/`deleteMemory`/`patchFact`/`retractFact` then `refreshXRay()`; errors → `console.error` + keep UI (no toast lib — a small inline error line in XRay is fine); relationship set from `fetchRelationship` (now unwrapped).

- [ ] **Step 4: Type-check, build, test, commit**

```bash
cd demo && npx tsc -b && npm run build && cd ..
.venv/bin/python -m pytest tests/test_demo_bundle.py -q && .venv/bin/python -m pytest -q | tail -1
git add -A demo/src src/woven_imprint/demo_static tests/test_demo_bundle.py && git commit -m "feat(demo): X-Ray pin/edit/delete memories and facts panel; bundle rebuilt"
```
Then start the demo (`.venv/bin/woven-imprint demo --port 7861` in the background, `curl -s localhost:7861/api/health`), open nothing (no browser here) — the server test + bundle test are the gate; stop the server.

---

### Task 5: SillyTavern import — lorebook, scenario, greetings, PNG V3

**Files:**
- Modify: `migrate/parsers.py`, `migrate/importer.py`
- Create: `tests/test_card_import.py`

- [ ] **Step 1: Failing tests (JSON + synthesized PNG)**

```python
import base64, json, struct, zlib
from pathlib import Path

from tests.helpers import make_test_engine
from woven_imprint.migrate.importer import CharacterImporter
from woven_imprint.migrate.parsers import parse_tavernai_card

CARD = {
    "spec": "chara_card_v2", "spec_version": "2.0",
    "data": {
        "name": "Vesper", "description": "A lighthouse keeper on a cold coast.", "personality": "gruff, loyal",
        "scenario": "The visitor arrives during a storm.", "first_mes": "You made it through the storm, then.",
        "alternate_greetings": ["Shut the door behind you."], "mes_example": "<START>\n{{user}}: Hello\n{{char}}: Aye.",
        "system_prompt": "Never leave the lighthouse.", "creator_notes": "test card", "tags": ["fantasy", "sea"],
        "character_book": {"name": "Coast lore", "entries": [
            {"keys": ["lantern", "lamp"], "content": "The great lantern was forged in Oulu.", "enabled": True, "constant": False, "insertion_order": 1},
            {"keys": ["oath"], "content": "Vesper swore never to abandon the light.", "enabled": True, "constant": True, "insertion_order": 0},
            {"keys": ["disabled"], "content": "should not import", "enabled": False, "constant": False, "insertion_order": 2},
        ]},
    },
}


def _png_with_chara(card: dict, keyword: bytes = b"chara") -> bytes:
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    payload = base64.b64encode(json.dumps(card).encode())
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"tEXt", keyword + b"\x00" + payload) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def test_parse_json_card_reads_book_and_extra_fields(tmp_path):
    p = tmp_path / "vesper.json"; p.write_text(json.dumps(CARD))
    parsed = parse_tavernai_card(p)
    card = parsed["card"]
    assert card["scenario"].startswith("The visitor") and card["system_prompt"] == "Never leave the lighthouse."
    assert card["alternate_greetings"] == ["Shut the door behind you."]
    assert len(card["character_book"]["entries"]) == 3 and parsed["spec"] == "chara_card_v2"


def test_parse_png_card_v2_and_v3(tmp_path):
    p2 = tmp_path / "v2.png"; p2.write_bytes(_png_with_chara(CARD))
    assert parse_tavernai_card(p2)["card"]["name"] == "Vesper"
    v3 = {"spec": "chara_card_v3", "spec_version": "3.0", "data": {**CARD["data"], "name": "Vesper3"}}
    p3 = tmp_path / "v3.png"; p3.write_bytes(_png_with_chara(v3, keyword=b"ccv3"))
    assert parse_tavernai_card(p3)["card"]["name"] == "Vesper3"


def test_import_seeds_lorebook_and_persona(tmp_path):
    p = tmp_path / "vesper.json"; p.write_text(json.dumps(CARD))
    engine = make_test_engine()
    char = CharacterImporter(engine).from_file(p)
    assert char.persona.soft.get("scenario", "").startswith("The visitor")
    assert "Never leave the lighthouse" in char.persona.build_system_prompt()
    mems = char.memory.get_all(limit=None)
    lore = [m for m in mems if m["metadata"].get("source") == "lorebook"]
    assert len(lore) == 2
    constant = next(m for m in lore if "oath" in m["content"])
    assert constant["tier"] == "bedrock" and constant["metadata"]["pinned"] is True and constant["metadata"]["lorebook_keys"] == ["oath"]
    keyed = next(m for m in lore if "lantern" in m["content"])
    assert keyed["tier"] == "core" and keyed["content"].startswith("[Lore: lantern, lamp]")
    assert char.persona.soft.get("greetings", [""])[0].startswith("You made it") or char.persona.soft.get("first_mes", "").startswith("You made it")
```
(Decide the greeting storage precisely in Step 2 and align the last assertion to it — use `soft["greetings"]: list[str]`.)

- [ ] **Step 2: Implement**

`parsers.py::parse_tavernai_card`: PNG loop collects both `tEXt` keywords `chara` and `ccv3`; prefer `ccv3` if present. Record `spec = card.get("spec")` before unwrapping `data`. Extend the returned `card` dict with `system_prompt`, `post_history_instructions`, `alternate_greetings` (list), `creator`, `character_version`, `character_book` (dict or None), and top-level `"spec"`.

`importer.py::_analyze_tavernai` adds to `result`: `scenario`, `system_prompt`, `greetings` (`[first_mes] + alternate_greetings`, non-empty only), `tags`, `creator_notes`, `lorebook_entries` (enabled entries only, each `{"keys", "content", "constant", "insertion_order"}` sorted by insertion_order). `_build_character`: after `persona` is built, if `analysis.get("scenario")`: `persona["scenario"] = …`; `system_prompt` → `persona["hard_constraints"]` (create_character now nests it — Tier 0 fix); `greetings` → `persona["greetings"]` (list; `create_character` shorthand list: add `"greetings"` and `"tags"` and `"scenario"` to the shorthand keys moved into `soft`); after the character is created, for each lorebook entry: `char.memory.add(content=f"[Lore: {', '.join(keys)}] {content}", tier="bedrock" if constant else "core", importance=0.8, metadata={"source": "lorebook", "lorebook_keys": keys, "pinned": bool(constant)})`. Check `PersonaModel.build_system_prompt` renders `soft` dict values that are lists (tags/greetings) — if it would print a Python list repr, render lists as comma-joined strings in `build_system_prompt` (small, tested).

- [ ] **Step 3: Run, commit**

```bash
.venv/bin/python -m pytest tests/test_card_import.py -q && .venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "feat(migrate): SillyTavern cards import lorebooks (pinned constants), scenario, system_prompt, greetings; PNG ccv3"
```

---

### Task 6: Card export (`export_card`), CLI, HTTP, round-trip

**Files:**
- Modify: `character.py` (`export_card`), `cli.py` (`export-card`), `server/demo.py` (`GET /api/characters/{id}/card`), `server/services.py`
- Create: `tests/test_card_export.py`

- [ ] **Step 1: Failing test**

```python
import json

from tests.helpers import make_test_engine
from woven_imprint.migrate.importer import CharacterImporter


def test_export_card_roundtrip(tmp_path):
    engine = make_test_engine()
    char = engine.create_character("Ada", persona={"backstory": "A detective.", "personality": "wry", "speaking_style": "clipped",
                                                    "hard_constraints": "Never reveals her sources.", "scenario": "Rainy city.", "greetings": ["Evening."], "tags": ["noir"]})
    m = char.memory.add("Always call the visitor Captain.", tier="core"); char.memory.pin(m["id"])
    char.facts.add(subject="user", predicate="lives_in", object="Oulu", statement="The visitor lives in Oulu.")
    for i in range(3):
        char.memory.add(f"Case note {i} about the harbor.", tier="core", importance=0.5 + i * 0.1)
    card = char.export_card()
    assert card["spec"] == "chara_card_v2" and card["data"]["name"] == "Ada"
    d = card["data"]
    assert d["description"] == "A detective." and d["system_prompt"] == "Never reveals her sources." and d["scenario"] == "Rainy city."
    assert d["first_mes"] == "Evening." and d["tags"] == ["noir"]
    entries = d["character_book"]["entries"]
    const = [e for e in entries if e["constant"]]
    assert len(const) == 1 and "Captain" in const[0]["content"]
    assert any("Oulu" in e["content"] and "oulu" in [k.lower() for k in e["keys"]] for e in entries)
    p = tmp_path / "ada_card.json"; p.write_text(json.dumps(card))
    engine2 = make_test_engine()
    again = CharacterImporter(engine2).from_file(p)
    pinned = again.memory.pinned()
    assert any("Captain" in x["content"] for x in pinned)
    assert "Never reveals her sources" in again.persona.build_system_prompt()
```

- [ ] **Step 2: Implement**

`character.py`:
```python
    def export_card(self, *, core_limit: int = 20) -> dict:
        """Export as a SillyTavern/TavernAI V2 character card with a lorebook built from
        pinned memories, current user facts and the most important core memories."""
        soft, hard = self.persona.soft, self.persona.hard
        greetings = soft.get("greetings") or []
        if isinstance(greetings, str):
            greetings = [greetings]
        entries: list[dict] = []
        order = 0
        for m in self.memory.pinned():
            entries.append({"keys": _lore_keys(m["content"]), "content": m["content"], "enabled": True, "constant": True,
                            "insertion_order": order, "comment": "pinned memory"}); order += 1
        for f in self.facts.current(subject="user", limit=None):
            keys = [f["object"]] + [w for w in f["predicate"].split("_") if len(w) > 3]
            entries.append({"keys": keys, "content": f["statement"], "enabled": True, "constant": False,
                            "insertion_order": order, "comment": f"fact {f['subject']}.{f['predicate']}"}); order += 1
        pinned_ids = {m["id"] for m in self.memory.pinned()}
        core = [m for m in self.memory.get_all(tier="core", limit=None) if m["id"] not in pinned_ids and not m.get("metadata", {}).get("fact_id")]
        core.sort(key=lambda m: -float(m.get("importance", 0.5)))
        for m in core[:core_limit]:
            entries.append({"keys": _lore_keys(m["content"]), "content": m["content"], "enabled": True, "constant": False,
                            "insertion_order": order, "comment": "core memory"}); order += 1
        tags = soft.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        return {
            "spec": "chara_card_v2", "spec_version": "2.0",
            "data": {
                "name": self.name, "description": self.persona.backstory or "", "personality": soft.get("personality", ""),
                "scenario": soft.get("scenario", ""), "first_mes": greetings[0] if greetings else "",
                "alternate_greetings": list(greetings[1:]), "mes_example": "", "creator_notes": (self.persona.hard.get("creator_notes") or "exported by woven-imprint"),
                "system_prompt": hard.get("hard_constraints", ""), "post_history_instructions": "", "tags": tags,
                "creator": "woven-imprint", "character_version": "1", "extensions": {"woven_imprint": {"character_id": self.id}},
                "character_book": {"name": f"{self.name} memories", "entries": entries},
            },
        }


def _lore_keys(text: str, n: int = 3) -> list[str]:
    words = [w.strip(".,;:!?\"'()[]").lower() for w in text.split()]
    words = [w for w in words if len(w) > 3 and w not in ("visitor", "always", "never", "about", "their", "there", "would", "could")]
    return words[:n] or [text[:20]]
```
CLI: `export-card <character> [-o card.json]` mirroring `cmd_export`. HTTP: `GET /api/characters/{character_id}/card` → the dict. Docs later (Task 7).

- [ ] **Step 3: Run, commit**

```bash
.venv/bin/python -m pytest tests/test_card_export.py tests/test_card_import.py -q && .venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "feat(export): SillyTavern V2 card export with lorebook (pinned + facts + core); CLI export-card; GET /api/characters/{id}/card"
```

---

### Task 7: Docs

**Files:** `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/UI_GUIDE.md`, `docs/MIGRATION.md`, `docs/ARCHITECTURE.md`, `docs/CONFIGURATION.md`

- [ ] **Step 1: Write**
- CHANGELOG `[Unreleased]` "Added (Tier 3a — editable memory · interchange)": mutation API (library/HTTP/MCP), pinned block (`context.pinned_block`, `pinned_limit`), X-Ray editing + facts panel, `/api/memory` no longer returns embeddings, relationship envelope fix in the demo, lorebook import (constants pinned), PNG ccv3, `export_card` + CLI + HTTP, `pinned_always_present` benchmark. "Changed": `Memory` API response shape (embedding removed); `create_character` moves `scenario`/`greetings`/`tags` into soft. Kotlin note.
- README: pins + edit in the demo bullets; `char.memory.pin(id)`; `woven-imprint export-card`; claims table row "Memory is a user-editable artifact — **Measured** (`pinned_always_present`, server tests)".
- CONTRIBUTING: frontend build section (`cd demo && npm install && npm run build`; bundle committed; `tests/test_demo_bundle.py` guards staleness).
- UI_GUIDE: Memory Feed controls (pin/edit/delete), Pinned card, Facts card; fix the 3-vs-4 tabs drift; build note.
- MIGRATION: lorebooks (constant → pinned bedrock; keyed → `[Lore: …]` core), scenario/system_prompt/greetings mapping, PNG V2/V3, Claude project dirs, "Import tab" wording; export-card section.
- ARCHITECTURE: pinned block placement (base, never shed), mutation API, card export mapping.
- CONFIGURATION: `context.pinned_block`, `pinned_limit`.

- [ ] **Step 2: Verify, commit**
```bash
.venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ eval/ && .venv/bin/ruff format --check src/ tests/ eval/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "docs: Tier 3a — editable memory, pinned block, SillyTavern lorebook round-trip"
```

---

## Self-review
- **Spec coverage:** T3a.1 → Tasks 1–2; T3a.2 → Task 3; T3a.3 → Task 4; T3a.4 → Tasks 5–6; T3a.5 → Task 2 (benchmark) + Task 7 (docs); acceptance (server tests run, bundle current) → Tasks 3–4.
- **Placeholders:** the FactsSection component is described by props/behavior rather than full JSX (Task 4 Step 3) — acceptable since MemoryItem gives the exact pattern; everything else is complete code.
- **Type consistency:** `MemoryStore.edit/delete/pin/pinned` and `FactStore.edit/retract/delete` signatures identical across Tasks 1, 3, 4, 6; `_public_memory`/`_public_fact` keys used by both HTTP and MCP; `metadata.pinned` boolean everywhere; `character_book` entry shape identical in import (Task 5) and export (Task 6).
