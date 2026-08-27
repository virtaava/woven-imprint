# Tier 3a — Memory as an editable artifact + SillyTavern interchange — Spec

**Date:** 2026-08-27  
**Owner:** Toni  
**Origin:** audit roadmap Tier 3 ("memory as editable artifact", "interchange"); landscape research (C.AI pins, Kindroid journals, Nomi Mind Map, Replika dashboard all expose editable memory; SillyTavern cards + lorebooks are the de-facto interchange).  
**Status:** approved ("merge and proceed", 2026-08-27). Design choices are the controller's stated assumptions; Toni can override.

## Goal
1. **Every memory tier and every fact is viewable and editable**: edit text, change importance/tier, pin, delete; edit/retract facts — from the library, HTTP, MCP and the demo X-Ray panel.
2. **Pinned memories are always in the prompt** (the single most-praised companion property), never shed by the budget.
3. **SillyTavern cards round-trip**: import reads `character_book` (lorebook), `first_mes`, `scenario`, `system_prompt`; export writes a V2 card with a lorebook built from pinned memories and current facts.
4. Fix the two demo bugs found: `/api/memory` leaks embeddings; the frontend misreads the relationship envelope.

Out of scope: ego's Character Context Protocol (no public spec verified), a frontend test runner (documented as a follow-up), Kotlin parity (CHANGELOG note), V3 card `ccv3` chunks (accept-only if trivial; not required).

## T3a.1 Library: memory & fact mutation, pinning

**Storage (`storage/sqlite.py`, no migration — `pinned` lives in `metadata` JSON, queried via `json_extract`):**
- `update_memory_fields(memory_id, *, content=None, embedding=None, importance=None, tier=None, metadata=None) -> None` — single UPDATE building only the provided SET clauses; `content` change must come with `embedding`; tier validated against the CHECK set.
- `delete_memory(memory_id) -> bool` — hard delete (FTS trigger keeps the index in sync); returns whether a row was removed.
- `list_pinned_memories(character_id) -> list[dict]` — `WHERE status='active' AND json_extract(metadata,'$.pinned') = 1 ORDER BY created_at ASC`.
- `delete_fact(fact_id) -> bool`; `update_fact_fields(fact_id, *, object=None, statement=None, certainty=None, importance=None, metadata=None)`; `unlink_fact_memory(memory_id)` sets `memory_id=NULL` on facts referencing it.

**MemoryStore:**
- `edit(memory_id, *, content=None, importance=None, tier=None) -> dict` — re-embeds when content changes (through `guard_embedding_dimension`); returns the updated row; `KeyError` if missing.
- `delete(memory_id) -> None` — also `unlink_fact_memory` and **retracts** any fact that pointed to it (see FactStore.retract) — deleting a memory means "forget this".
- `pin(memory_id, pinned: bool = True) -> dict` — sets `metadata.pinned`.
- `pinned() -> list[dict]`.

**FactStore:**
- `edit(fact_id, *, object=None, statement=None) -> dict` — updates the fact and, if linked, the memory's content (re-embed) so text and record stay consistent.
- `retract(fact_id) -> dict` — expires now with `superseded_by=None`, `metadata.retracted=True`; linked memory → status `archived`.
- `delete(fact_id) -> None` — hard delete of the fact row only (memory untouched).

**Prompt — pinned block (`character.py::_build_context`):** `context.pinned_block: bool = True`, `context.pinned_limit: int = 10`. Rendered into the non-sheddable `volatile` base (after the "Today is" line):
```
Things you always remember:
- (2026-05-03) The visitor's brother died in March; do not bring it up casually.
```
Pinned memories are excluded from the retrieved-memories list (no duplicates). The pinned block counts toward `base_size`.

**Character:** `char.memory.edit/delete/pin/pinned`, `char.facts.edit/retract/delete` are the public surface (no new Character methods). `export()` already carries `metadata` (pins survive export/import).

## T3a.2 HTTP + MCP

Demo server (all auth-guarded, `mutation` rate bucket):
- `GET /api/memory` — response memories **strip `embedding`**, keep `id, tier, content, importance, certainty, status, created_at, metadata` (bug fix).
- `GET /api/memory/pinned?character_id=` → `{"memories": [...]}`.
- `PATCH /api/memory/{memory_id}` body `{character_id, content?, importance?, tier?, pinned?}` → `{"memory": {...}}`; 404 if unknown; 400 on invalid tier/importance.
- `DELETE /api/memory/{memory_id}?character_id=` → `{"deleted": true}`.
- `GET /api/facts/{character_id}` — items now include `id`, `superseded_by`, `memory_id`.
- `PATCH /api/facts/{fact_id}` body `{character_id, object?, statement?}` → `{"fact": {...}}`; `DELETE /api/facts/{fact_id}?character_id=&mode=retract|delete` (default retract) → `{"fact": {...}}` / `{"deleted": true}`.
- Every mutation route resolves the character via the cache (`_get_character`) under `_character_mutation(cid)`; memories/facts must belong to `character_id` (404 otherwise).

MCP tools: `edit_memory(character_id, memory_id, content=None, importance=None, tier=None)`, `delete_memory(character_id, memory_id)`, `pin_memory(character_id, memory_id, pinned=True)`, `list_pinned(character_id)`, `retract_fact(character_id, fact_id)`, `edit_fact(character_id, fact_id, object=None, statement=None)`.

## T3a.3 Demo UI (React/Vite, `demo/`)

- `types.ts`: `Memory` gains `id`, `certainty?`, `metadata?` (`{pinned?: boolean; fact_id?: string; historical?: boolean}`); new `Fact` type (`id, subject, predicate, object, statement, valid_from, valid_to, certainty, previously?`).
- `api.ts`: `patchMemory`, `deleteMemory`, `fetchPinned`, `fetchFacts`, `patchFact`, `retractFact`; a shared `request()` helper that throws on `!res.ok` (fixes the silent-error pattern) — existing helpers migrate to it.
- `App.tsx`: relationship envelope fix (`res.relationship`); `refreshXRay` also loads pinned + facts.
- `XRayPanel.tsx`: **Pinned** section (top of memory area) with unpin; **Memory Feed** rows keyed by `id` with pin toggle, inline edit (textarea + save/cancel → PATCH), delete (confirm) and a date; **Facts** section listing current user-facts with `previously` and an edit-object / retract control. Keyboard: Escape cancels edit.
- Build: `cd demo && npm run build` regenerates `src/woven_imprint/demo_static/` (committed). `CONTRIBUTING.md` + `docs/UI_GUIDE.md` document the build. A server test greps the built bundle for `/api/memory/pinned` to prove the bundle is current.

## T3a.4 SillyTavern interchange

**Import (`migrate/parsers.py`, `importer.py`):**
- `parse_tavernai_card` also returns `character_book` (entries: `keys`, `content`, `enabled`, `constant`, `insertion_order`, `comment`), `system_prompt`, `post_history_instructions`, `alternate_greetings`, `creator`, `character_version`, `spec`/`spec_version`; PNG: accept `tEXt` `chara` (V2) and, if present, `ccv3` (V3 JSON, base64) preferring V3.
- `_analyze_tavernai` maps: `description` → backstory; `personality` → personality; `scenario` → `soft.scenario`; `system_prompt` → `hard.hard_constraints` (if non-empty); `first_mes` + `alternate_greetings` → `persona.greetings` (metadata list, used by the demo as suggested opener — store; UI use optional); `creator_notes` → `metadata.creator_notes`; `tags` → `soft.tags` (comma list). Lorebook: each enabled entry → memory `tier="bedrock"` if `constant` else `"core"`, content `f"[Lore: {', '.join(keys)}] {content}"`, `importance=0.8`, `metadata={"source":"lorebook","lorebook_keys":[...],"pinned": constant}` — constant entries are pinned.
- Tests: JSON card fixture with a 3-entry book (one constant); PNG fixture synthesized in-test (valid PNG signature + IHDR + `tEXt chara` + IEND with correct CRCs) — both import without an LLM (importer's LLM calls use FakeLLM).

**Export:** `Character.export_card() -> dict` (V2: `{"spec":"chara_card_v2","spec_version":"2.0","data":{...}}`) with `name`, `description` (backstory), `personality`, `scenario`, `first_mes` (first greeting or ""), `mes_example` "", `creator_notes` (from metadata or "woven-imprint export"), `system_prompt` (hard_constraints), `tags`, `character_book`: `{"name": f"{name} memories", "entries": [...]}` where entries = pinned memories (`constant: true`, keys from the first 3 content words) + current user facts (`keys: [object] + predicate words`, content = statement) + top-N core memories by importance (N = 20, `constant: false`). CLI `woven-imprint export-card <id> --out card.json`; HTTP `GET /api/characters/{id}/card`; MCP not needed.
- Round-trip test: create → pin/facts → `export_card()` → import via `from_file` → pinned bedrock lore present.

## T3a.5 Benchmark + docs
- Long-horizon: `pinned_always_present` — pin a day-3 memory; at day 60 the volatile system message contains its text and it is NOT duplicated in the memories block.
- Docs: UI_GUIDE (editing, pinning, facts panel, build), MIGRATION (lorebooks, Claude project, tab-name drift fixed), README (pins + card export), CHANGELOG, CONTRIBUTING (frontend build), ARCHITECTURE (pinned block placement, mutation API).

## Acceptance
- pytest/ruff/pyright green; demo server tests cover every new route (incl. 404/400 paths); frontend builds (`tsc -b && vite build`) and the committed bundle is current.
- Existing behavior unchanged when nothing is pinned and no card fields are present.

## Global constraints
- No schema migration (pins in metadata). No new required Python deps. Public API additive. Branch `feat/tier3a-editable-memory` off master `2d9ce9a`. Commit per task. Never touch `experiments/parametric_spike/` or `kotlin/`.
- Frontend work runs with the existing `demo/node_modules` (npm 350 MB present); no new npm deps unless a shadcn component is generated locally.
