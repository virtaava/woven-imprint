# Tier 3o — Entity-Linking at Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Memories carry LLM-extracted entity handles (`metadata.entities`) — attached at ingest via the existing unified-assessment call, backfillable for old stores — and the retrieval second pass can pivot on them instead of heuristic salient terms.

**Architecture:** Four additive layers: (1) the `TurnAssessor` JSON gains an `entities` section; (2) `Character` captures the turn's buffer-memory ids and attaches the parsed entities after assessment (sync + background paths); (3) `MemoryStore.link_entities` backfills stores that predate the feature, batched N memories per LLM call, idempotent via the presence of the `entities` metadata key; (4) `memory.second_pass_entities` switches the Tier 3f second-pass FTS terms from `_salient_terms` heuristics to the seeds' entity union, with fallback when the union is empty.

**Tech Stack:** Python 3.11, no new deps. Existing patterns: `MemoryStore.pin` (metadata merge), `MemoryStore.reembed` + `cli.cmd_reembed` (backfill job + CLI), Tier 3f second-pass block in `memory/retrieval.py`.

**Spec:** `docs/superpowers/specs/2026-09-04-tier3o-entity-linking.md`

## Global Constraints

- Branch `feat/tier3o-entity-linking` off master `cd9606f`; commit per task.
- Never touch `experiments/parametric_spike/` or `kotlin/`.
- No schema migration — entities live in the `metadata` JSON. Public API additive only.
- Gates: `.venv/bin/python -m pytest`, `.venv/bin/ruff check src tests eval demo`, `.venv/bin/pyright` — all clean per task.
- `second_pass_entities=False` (default) must leave retrieval byte-identical; assessor `entities` parsing must NEVER raise (absent/malformed → `[]`).
- Benchmark measurement happens AFTER the branch is reviewed (controller-run chains); not part of these tasks.

---

### Task 1: Assessor `entities` field

**Files:**
- Modify: `src/woven_imprint/prompts.py` (after `TURN_ASSESSMENT_FACTS_SECTION`, ~line 90)
- Modify: `src/woven_imprint/persona/assessment.py` (`TurnAssessment` ~line 22, `build_messages` sections ~line 52, `assess` ~line 98)
- Test: `tests/test_entity_assessment.py` (create)

**Interfaces:**
- Consumes: existing `TurnAssessor.assess(**kwargs) -> TurnAssessment`, `generate_json_robust`.
- Produces: `TurnAssessment.entities: list[str]` (default `[]`); `TurnAssessor._parse_entities(raw) -> list[str]` (staticmethod: non-list → `[]`; non-str items skipped; strip; drop empties; case-fold dedup preserving first-seen casing; cap 8). Task 2 reads `out.entities`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_entity_assessment.py`:

```python
"""Tier 3o: `entities` field in the unified turn assessment.

Spec: docs/superpowers/specs/2026-09-04-tier3o-entity-linking.md

The unified JSON gains an optional `entities` list (<=8 short canonical
names). Parsing is fault-tolerant like every other section: absent,
malformed, or non-list payloads yield [] and never raise.
"""

from __future__ import annotations

from woven_imprint.persona.assessment import TurnAssessor


def test_parse_entities_happy_path():
    out = TurnAssessor._parse_entities(["Rocket", "Caroline", "San Diego"])
    assert out == ["Rocket", "Caroline", "San Diego"]


def test_parse_entities_cleans_dedups_caps():
    raw = ["  Rocket ", "rocket", "", 42, None, "A", "B", "C", "D", "E", "F", "G", "H"]
    out = TurnAssessor._parse_entities(raw)
    assert out[0] == "Rocket"          # first-seen casing kept, stripped
    assert "rocket" not in out[1:]     # case-fold dedup
    assert 42 not in out and None not in out and "" not in out
    assert len(out) <= 8


def test_parse_entities_non_list_returns_empty():
    assert TurnAssessor._parse_entities("Rocket") == []
    assert TurnAssessor._parse_entities({"e": 1}) == []
    assert TurnAssessor._parse_entities(None) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_entity_assessment.py -v`
Expected: FAIL with `AttributeError: ... '_parse_entities'`

- [ ] **Step 3: Implement** — three edits:

(a) `src/woven_imprint/prompts.py`, directly after the `TURN_ASSESSMENT_FACTS_SECTION` constant:

```python
TURN_ASSESSMENT_ENTITIES_SECTION = (
    '"entities": up to 8 short canonical names of specific people, pets, places, '
    "organizations, or distinctive objects/events mentioned in this exchange "
    "(proper nouns preferred; no generic nouns, no dates); [] if none."
)
```

(b) `src/woven_imprint/persona/assessment.py`:
- Import `TURN_ASSESSMENT_ENTITIES_SECTION` in the existing `from ..prompts import (...)` block (alphabetical position after `TURN_ASSESSMENT_EMOTION_SECTION`... keep the block's existing ordering style).
- `TurnAssessment` gains, between `facts` and `raw`:

```python
    entities: list[str] = field(default_factory=list)
```

- In `build_messages`, after the `want_facts` section append (the sections list is unconditional for emotion; entities is likewise always requested — it costs a few tokens and rides the same call):

```python
        sections.append(TURN_ASSESSMENT_ENTITIES_SECTION)
```

Place it AFTER the `if want_facts:` block so section order is emotion, [relationship], [beat], [facts], entities.
- New staticmethod on `TurnAssessor` (before `assess`):

```python
    @staticmethod
    def _parse_entities(raw) -> list[str]:
        """Fault-tolerant `entities` parsing: never raises, [] on anything odd.

        Non-list payloads -> []. Non-string items are skipped (unlike a hard
        void: entities are advisory handles, partial credit is fine). Items
        are stripped, empties dropped, case-fold deduped keeping the first
        casing seen, capped at 8 (spec 2026-09-04-tier3o-entity-linking).
        """
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
            if len(out) >= 8:
                break
        return out
```

- In `assess`, before the `return`:

```python
        entities = self._parse_entities(data.get("entities"))
```

and add `entities=entities` to the `TurnAssessment(...)` constructor call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_entity_assessment.py tests/test_unified_assessment.py -v` (second file: whatever existing test module covers `TurnAssessor` — find it with `grep -rl TurnAssessor tests/`; run it to prove no regression)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/woven_imprint/prompts.py src/woven_imprint/persona/assessment.py tests/test_entity_assessment.py
git commit -m "feat(tier3o): entities field in unified turn assessment"
```

---

### Task 2: Attach entities to the turn's buffer memories

**Files:**
- Modify: `src/woven_imprint/memory/store.py` (new `set_entities`, after `pin` ~line 458)
- Modify: `src/woven_imprint/character.py` (`chat` user-add ~line 481 and response-add ~line 571; worker submit ~line 598; `ingest` add ~line 835; `_run_bookkeeping` signature ~line 1892 and body end ~line 1956)
- Test: `tests/test_entity_assessment.py` (append)

**Interfaces:**
- Consumes: `TurnAssessment.entities` (Task 1); `storage.get_memory` / `storage.update_memory_fields` (existing); `MemoryStore.add(...) -> dict` (existing — returns the row incl. `id`).
- Produces: `MemoryStore.set_entities(memory_id: str, entities: list[str]) -> dict` (pin-style metadata merge; KeyError on unknown/foreign id); `_run_bookkeeping(..., turn_memory_ids: tuple[str, ...] = ())` — attaches `out.entities` to each id when non-empty. Task 3's backfill relies on `metadata.entities` key semantics established here.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_entity_assessment.py`:

```python
import json


def _mk_char(payload):
    """Character with a FakeLLM whose generate_json_robust returns `payload`."""
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Entity Test")
    class _LLM:
        def generate(self, messages, **kw):
            return "ok"
        def generate_json_robust(self, messages, temperature=0.3, **kw):
            return payload
        def generate_json(self, messages, **kw):
            return payload
    char.llm = _LLM()
    char.assessor.llm = char.llm
    return char


def test_set_entities_merges_metadata():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Store Test")
    row = char.memory.add(content="[User] I adopted Rocket", metadata={"user_id": "u1"})
    updated = char.memory.set_entities(row["id"], ["Rocket"])
    assert updated["metadata"]["entities"] == ["Rocket"]
    assert updated["metadata"]["user_id"] == "u1"  # merge, not replace


def test_ingest_attaches_entities_to_turn_memory():
    char = _mk_char({"emotion": {}, "facts": [], "entities": ["Rocket", "Caroline"]})
    char.ingest("user", "Caroline adopted a beagle named Rocket", user_id="Melanie")
    mems = char.storage.get_memories(char.id, limit=5)
    turn = [m for m in mems if "Rocket" in m["content"]]
    assert turn and turn[0]["metadata"].get("entities") == ["Rocket", "Caroline"]


def test_background_chat_attaches_entities_after_flush():
    char = _mk_char({"emotion": {}, "facts": [], "entities": ["Rocket"]})
    char.background = True
    char.chat("tell me about Rocket", user_id="Melanie")
    char.flush()
    mems = char.storage.get_memories(char.id, limit=8)
    tagged = [m for m in mems if (m.get("metadata") or {}).get("entities") == ["Rocket"]]
    assert len(tagged) == 2  # user turn + response turn


def test_ingest_no_entities_leaves_metadata_untouched():
    char = _mk_char({"emotion": {}, "facts": [], "entities": []})
    char.ingest("user", "hello there", user_id="Melanie")
    mems = char.storage.get_memories(char.id, limit=5)
    turn = [m for m in mems if "hello there" in m["content"]]
    assert turn and "entities" not in (turn[0].get("metadata") or {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_entity_assessment.py -v`
Expected: new tests FAIL (`AttributeError: ... 'set_entities'`)

- [ ] **Step 3: Implement**

(a) `MemoryStore.set_entities` in `src/woven_imprint/memory/store.py`, modeled byte-for-byte on `pin` directly above it:

```python
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
```

(b) `src/woven_imprint/character.py`:
- `chat()`: capture both buffer adds — change `self.memory.add(...)` at the user-message site (~line 481) to `_user_mem = self.memory.add(...)` and the response site (~line 571) to `_resp_mem = self.memory.add(...)`; build `turn_memory_ids = (_user_mem["id"], _resp_mem["id"])`.
- The bookkeeping dispatch (both branches — worker submit AND the inline call a few lines below it) passes `turn_memory_ids` as an additional positional arg after `session_id` (the worker submit at ~line 598 already forwards positional args to `target`; append it there).
- `ingest()`: `_turn_mem = self.memory.add(...)` at ~line 835; pass `(_turn_mem["id"],)` into the `_run_bookkeeping(user_msg, response, user_id, self._session_id, ...)` call.
- `_run_bookkeeping` signature gains `turn_memory_ids: tuple[str, ...] = ()`. At the END of the method body (after the `want_facts` block), add:

```python
        if out.entities and turn_memory_ids:
            for mid in turn_memory_ids:
                try:
                    self.memory.set_entities(mid, out.entities)
                except Exception as e:
                    logger.debug("Entity attach failed for %s: %s", mid, e)
```

(non-fatal like every other subsystem; note the legacy `_run_subsystems_sequential` path is untouched — entities are a unified-assessment feature, document that in the method docstring).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_entity_assessment.py tests/test_ingest.py tests/test_engine.py -v`
Expected: PASS (ingest/engine suites prove the signature change broke nothing)

- [ ] **Step 5: Commit**

```bash
git add src/woven_imprint/memory/store.py src/woven_imprint/character.py tests/test_entity_assessment.py
git commit -m "feat(tier3o): attach assessed entities to the turn's buffer memories"
```

---

### Task 3: `link_entities` backfill job + CLI

**Files:**
- Modify: `src/woven_imprint/memory/store.py` (new `link_entities`, after `reembed` ~line 349)
- Modify: `src/woven_imprint/cli.py` (new `cmd_link_entities` after `cmd_reembed` ~line 426; parser registration after the reembed parser ~line 666)
- Test: `tests/test_entity_assessment.py` (append)

**Interfaces:**
- Consumes: `set_entities` (Task 2); `self.storage.get_memories(character_id, limit=...)`; an `llm` handle passed IN (MemoryStore has no LLM member — the caller supplies it, exactly like `reembed` uses the store's embedder).
- Produces: `MemoryStore.link_entities(llm, batch_size: int = 10) -> int` (returns memories updated; skips rows whose metadata already contains the `entities` key; batch LLM failure skips that batch, never raises). CLI subcommand `link-entities <character>`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_entity_assessment.py`:

```python
class _BatchLLM:
    """Returns {"1": [...], "2": [...]} keyed by 1-based batch position."""

    def __init__(self, mapping=None, exc=None):
        self.mapping = mapping or {}
        self.exc = exc
        self.calls = 0

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.mapping


def test_link_entities_backfills_only_unprocessed():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Backfill Test")
    a = char.memory.add(content="[User] Caroline adopted Rocket")
    b = char.memory.add(content="[User] dinner was pasta")
    char.memory.set_entities(b["id"], [])  # already processed (empty marker)
    llm = _BatchLLM({"1": ["Caroline", "Rocket"]})
    n = char.memory.link_entities(llm, batch_size=10)
    assert n == 1 and llm.calls == 1
    assert char.storage.get_memory(a["id"])["metadata"]["entities"] == ["Caroline", "Rocket"]
    # second run: nothing left to do, no LLM call
    llm2 = _BatchLLM({})
    assert char.memory.link_entities(llm2) == 0 and llm2.calls == 0


def test_link_entities_batch_failure_skips_not_raises():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Backfill Fail")
    char.memory.add(content="[User] one")
    n = char.memory.link_entities(_BatchLLM(exc=RuntimeError("down")), batch_size=10)
    assert n == 0  # skipped, no exception


def test_link_entities_marks_empty_result_as_processed():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Backfill Empty")
    a = char.memory.add(content="[User] nothing notable")
    n = char.memory.link_entities(_BatchLLM({}), batch_size=10)  # LLM returns {} -> no entities
    assert n == 1
    assert char.storage.get_memory(a["id"])["metadata"]["entities"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_entity_assessment.py -k link_entities -v`
Expected: FAIL (`AttributeError: ... 'link_entities'`)

- [ ] **Step 3: Implement**

(a) `MemoryStore.link_entities` in `src/woven_imprint/memory/store.py` (module-level prompt constant next to it):

```python
_LINK_ENTITIES_PROMPT = (
    "For each numbered memory below, list up to 8 short canonical names of specific "
    "people, pets, places, organizations, or distinctive objects/events it mentions "
    "(proper nouns preferred; no generic nouns, no dates).\n"
    'Return ONLY a JSON object mapping the number to the list, e.g. {{"1": ["Rocket"], '
    '"2": []}}. Every number must appear.\n\n{items}'
)
```

```python
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
```

NOTE for the implementer: check `storage.get_memories`'s signature for the active-only filter (Tier 3f-era code filters `status='active'` inside `get_memories` — verify with a quick read; if it returns non-active rows, filter `r.get("status", "active") == "active"` in the `todo` comprehension).

(b) CLI in `src/woven_imprint/cli.py`, mirroring `cmd_reembed` (~line 406) exactly (same character-resolution helper, same output style):

```python
def cmd_link_entities(args):
    engine, char = _load_character(args)  # use the same helper cmd_reembed uses — read it first
    count = char.memory.link_entities(char.llm, batch_size=args.batch)
    print(f"Linked entities on {count} memories for {char.name}")
```

and parser registration next to the reembed parser:

```python
    p_link = sub.add_parser(
        "link-entities", help="Backfill metadata.entities on existing memories (Tier 3o)"
    )
    p_link.add_argument("character", help="Character name or ID")
    p_link.add_argument("--batch", type=int, default=10)
    p_link.set_defaults(func=cmd_link_entities)
```

(Adjust the two helper names — character loading and defaults wiring — to whatever `cmd_reembed`'s block actually uses; read it before writing.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_entity_assessment.py tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/woven_imprint/memory/store.py src/woven_imprint/cli.py tests/test_entity_assessment.py
git commit -m "feat(tier3o): link_entities backfill job + CLI"
```

---

### Task 4: `second_pass_entities` retrieval flag

**Files:**
- Modify: `src/woven_imprint/config.py` (MemoryConfig, after `query_expansion_weight`; example-YAML block next to `# query_expansion:` lines)
- Modify: `src/woven_imprint/memory/retrieval.py` (Tier 3f block, the `terms = _salient_terms(combined_text)` line ~line 402)
- Modify: `docs/CONFIGURATION.md` (YAML block + table row, next to `retrieval_second_pass`)
- Test: `tests/test_retrieval_second_pass.py` (append)

**Interfaces:**
- Consumes: `metadata.entities` on seed rows (Tasks 2/3); existing Tier 3f block internals (`seeds`, `combined_text`, `_salient_terms`).
- Produces: `MemoryConfig.second_pass_entities: bool = False`. No other surface.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_retrieval_second_pass.py` (reuse its `setup` fixture, `WordEmbedder`, `_add` helper; extend the fixture teardown to restore `second_pass_entities` the way it restores `retrieval_second_pass`):

```python
def _add_with_entities(storage, embedder, content, entities, importance=0.5):
    mid = _add(storage, embedder, content, importance=importance)
    row = storage.get_memory(mid)
    meta = dict(row.get("metadata") or {})
    meta["entities"] = entities
    storage.update_memory_fields(mid, metadata=meta)
    return mid


def test_second_pass_uses_seed_entities_when_enabled(setup):
    storage, embedder, retriever, cfg = setup
    cfg.retrieval_second_pass = 2
    cfg.second_pass_entities = True
    # Seed: matches the query, carries an entity handle that shares no
    # vocabulary with the query.
    _add_with_entities(storage, embedder, "adopted a puppy last week", ["Rocket"])
    # Co-dependent evidence reachable ONLY via the entity term:
    target = _add(storage, embedder, "Rocket chewed the garden hose")
    for i in range(8):
        _add(storage, embedder, f"weather note number {i} sunny")
    results = retriever.retrieve("tell me about the puppy adoption", limit=6)
    assert target in {m["id"] for m in results}


def test_second_pass_entities_empty_union_falls_back_to_salient_terms(setup):
    storage, embedder, retriever, cfg = setup
    cfg.retrieval_second_pass = 2
    cfg.second_pass_entities = True
    # No seed has metadata.entities -> behavior must equal the flag-off run.
    _add(storage, embedder, "Bermuda Triangle mystery discussed at length")
    _add(storage, embedder, "Bermuda shorts purchased yesterday")
    baseline_cfg = cfg.second_pass_entities
    cfg.second_pass_entities = False
    off = [m["id"] for m in retriever.retrieve("mystery discussion", limit=5)]
    cfg.second_pass_entities = True
    on = [m["id"] for m in retriever.retrieve("mystery discussion", limit=5)]
    cfg.second_pass_entities = baseline_cfg
    assert on == off


def test_second_pass_entities_flag_off_is_byte_identical(setup):
    storage, embedder, retriever, cfg = setup
    cfg.retrieval_second_pass = 2
    cfg.second_pass_entities = False
    _add_with_entities(storage, embedder, "adopted a puppy last week", ["Rocket"])
    _add(storage, embedder, "Rocket chewed the garden hose")
    r1 = [m["id"] for m in retriever.retrieve("puppy adoption", limit=5)]
    # flag stays off: entities on rows must not influence anything
    r2 = [m["id"] for m in retriever.retrieve("puppy adoption", limit=5)]
    assert r1 == r2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_retrieval_second_pass.py -v`
Expected: new tests FAIL (`AttributeError: ... 'second_pass_entities'`)

- [ ] **Step 3: Implement**

(a) `MemoryConfig` field after `query_expansion_weight`:

```python
    # Tier 3o (docs/superpowers/specs/2026-09-04-tier3o-entity-linking.md):
    # when true AND retrieval_second_pass > 0, the second-pass FTS terms are
    # the union of the seeds' `metadata.entities` (case-fold deduped, first
    # casing kept, cap 16) instead of `_salient_terms` heuristics; an empty
    # union falls back to the heuristics so stores without entity metadata
    # behave exactly as before. Off by default pending the pre-registered
    # locomo-t3oEnt bar.
    second_pass_entities: bool = False
```

plus the example-YAML lines under the `# query_expansion_weight:` comment:

```python
  # second_pass_entities: false   # second-pass FTS pivots on seeds' metadata.entities
```

(b) `memory/retrieval.py` — replace the single line `terms = _salient_terms(combined_text)` inside the Tier 3f block with:

```python
                # Tier 3o: entity-quality terms when available. Union of the
                # seeds' metadata.entities; empty union (stores predating
                # entity linking, or flag off) falls back to the heuristics.
                terms: list[str] = []
                if mem_cfg.second_pass_entities:
                    seen_ents: set[str] = set()
                    for s_ in seeds:
                        for ent in (s_.get("metadata") or {}).get("entities") or []:
                            if not isinstance(ent, str):
                                continue
                            cleaned = ent.strip()
                            key = cleaned.casefold()
                            if not cleaned or key in seen_ents:
                                continue
                            seen_ents.add(key)
                            terms.append(cleaned)
                            if len(terms) >= 16:
                                break
                        if len(terms) >= 16:
                            break
                if not terms:
                    terms = _salient_terms(combined_text)
```

(c) `docs/CONFIGURATION.md`: add `second_pass_entities: false` to the YAML block under `retrieval_second_pass: 0` and a table row after the `retrieval_second_pass` row:

```
| `second_pass_entities` | `false` | — | Tier 3o: when `retrieval_second_pass > 0`, the second-pass FTS terms become the union of the seed memories' `metadata.entities` (LLM-extracted entity handles attached at ingest by the unified assessment, or backfilled via `woven-imprint link-entities <character>`), case-fold deduped and capped at 16; an empty union falls back to the `_salient_terms` heuristics, so stores without entity metadata behave exactly as before. Off by default pending the pre-registered `locomo-t3oEnt` bar (`docs/superpowers/specs/2026-09-04-tier3o-entity-linking.md`). |
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_retrieval_second_pass.py tests/test_query_expansion.py tests/test_retrieval.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/woven_imprint/config.py src/woven_imprint/memory/retrieval.py docs/CONFIGURATION.md tests/test_retrieval_second_pass.py
git commit -m "feat(tier3o): second_pass_entities — entity-pivot terms for the retrieval second pass"
```

---

### Task 5: Full-suite gate

**Files:** none expected — regression gate before measurement.

**Interfaces:** consumes everything above; produces a green tree for the controller's measurement chains.

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: 885 pre-existing + new tests pass (1 pre-existing skip OK).

- [ ] **Step 2: Lint + types**

Run: `.venv/bin/ruff check src tests eval demo && .venv/bin/pyright`
Expected: "All checks passed!" and 0 errors.

- [ ] **Step 3: Long-horizon bench**

Run: `.venv/bin/python -m eval.bench_longhorizon`
Expected: 12/12 (defaults unchanged — proves the off path).

- [ ] **Step 4: Commit (only if fixes were needed)**

```bash
git add -A && git commit -m "fix(tier3o): suite/bench fixes"
```
