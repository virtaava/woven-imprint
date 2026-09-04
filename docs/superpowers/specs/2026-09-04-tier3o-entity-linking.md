# Tier 3o — Entity-linking at ingest — Spec

**Date:** 2026-09-04
**Owner:** Toni ("proceed with the entity-linking at ingest")
**Origin:** recorded next retrieval candidate since Tier 3i/3m. Remaining gaps: LoCoMo
multi-hop tail (cat-1 0.500) and open-domain (cat-3 0.271); Tier 3f's heuristic
salient-term second pass failed its bar — the hypothesis is that CLEAN entity handles
succeed where capitalized-token heuristics failed.

## Goal
Give memories a retrievable entity handle at ingest time (zero additional LLM calls —
the field rides the existing unified-assessment call), plus a backfill job for existing
stores, and make the retrieval second pass pivot on those entities. Ship library-side;
default retrieval behavior unchanged until the pre-registered bar rules.

## Mechanism
1. **Assessor** (`persona/assessment.py`): the unified JSON gains an optional
   `entities` field — up to 8 short canonical strings (proper names of people, pets,
   places, organizations, distinctive objects/events mentioned in the exchange; no
   generic nouns, no dates). Parsed fault-tolerantly like every other section (absent,
   malformed, or non-list → `[]`, never an error); `TurnAssessment` carries
   `entities: list[str]`. Prompt section added to `build_messages` mirroring the facts
   section's style.
2. **Ingest attach** (`character.py::_run_bookkeeping`): the episodic memory row(s)
   recorded for the current exchange get `metadata.entities = [...]` (case-preserved,
   case-fold deduped, capped 8). The plan pins the id plumbing (the turn memory is
   created before bookkeeping runs; use the existing
   `storage.update_memory_fields(memory_id, metadata=...)` from Tier 3a). Applies on
   both the sync and background bookkeeping paths. `fact`-derived core rows may also
   carry them if trivially reachable; not required.
3. **Backfill job** (`maintenance.py` + engine/CLI, mirroring `reembed`):
   `link_entities(character_id, batch_size=10)` — for active memories missing
   `metadata.entities`, one `generate_json_robust` call per batch of `batch_size`
   memory contents returns `{memory_index: [entities]}`; failures skip the batch
   (never raise); progress streamed; idempotent (skips rows that already have the
   key, even an empty list — `{"entities": []}` marks "processed, none found").
   CLI: `woven-imprint link-entities <character_id>`; engine method for scripting.
4. **Retrieval** (`memory/retrieval.py`): new `memory.second_pass_entities: bool =
   False`. Inside the EXISTING Tier 3f block (`retrieval_second_pass > 0`), when the
   flag is on: the second-pass FTS terms become the union of the seeds'
   `metadata.entities` (case-fold dedup, cap 16, preserve first-seen casing); seeds
   with no entities contribute nothing; if the union is EMPTY, fall back to
   `_salient_terms` exactly as today. Everything else in the block (one `fts_search`,
   widening, mean-vector gate, re-fuse) unchanged. Flag off → byte-identical.
5. **Config**: `second_pass_entities` validated bool, documented in CONFIGURATION.md
   and the config.py example block, spec-cited docstring.

## Measurement (pre-registered bars, declared BEFORE any run)
Backfill copies of the kept DBs (originals untouched):
`cp -r locomo-mem-v3b -> locomo-mem-v3b-ent`, run `link_entities` over its 10 DBs
(sharded, ~11.5k memories, ~1-2 h), then answer-only at the shipping protocol
(K=100, hardened prompt, `--agg-stage`):
- **`locomo-t3oEnt`**: `--set memory.retrieval_second_pass=3
  --set memory.second_pass_entities=true`.
- **`locomo-t3oSal`** (attribution arm): `--set memory.retrieval_second_pass=3` only —
  Tier 3f heuristics at the current protocol, so entity-vs-heuristic term quality is
  isolated.
- Baseline: `locomo-t3nAgg` (overall 0.674, cat-1 0.500, cat-3 0.271, adversarial 0.877).
- **Ship-on bar (flips `second_pass_entities` AND `retrieval_second_pass=3` defaults):**
  overall ≥ 0.669 AND cat-1 ≥ 0.530 AND adversarial ≥ 0.87 AND t3oEnt > t3oSal on cat-1
  (the entity handles must beat the heuristic, else Tier 3f's verdict stands).
- Bar met → LME confirmation on backfilled `lme-s-100-v5` copies (`lme-t3oEnt`):
  keep iff overall ≥ 0.651 AND `_abs` ≥ 6/7.
- Bar missed → ships opt-in with honest docs (Tier 3f/3i precedent). Ingest-path entity
  storage ships regardless (zero-cost, additive metadata; it is the enabler either way).

## Deliverables
Assessor field + parsing tests (FakeLLM payload variants); bookkeeping attach + tests
(sync + background paths; memory row carries entities); maintenance job + CLI + tests
(idempotency, batch failure skip, empty-marker semantics); retrieval flag + tests
(entity terms used, empty-union fallback, off byte-identical, composition with
query_expansion/agg unaffected); config validation + docs; backfill tooling for the
benchmark DB copies (driver script, not committed results until measured); BENCHMARKS
Tier 3o section + RESULTS/README/CHANGELOG per outcome; results JSONs committed.

## Acceptance
pytest (885+new)/ruff (src tests eval demo)/pyright green; `bench_longhorizon` 12/12;
deterministic suites unaffected; numbers published exactly as measured.

## Global constraints
Branch `feat/tier3o-entity-linking` off master `cd9606f`. Never touch
`experiments/parametric_spike/` or `kotlin/`. No schema migration (entities live in
metadata JSON). Public API additive. Local brain/embedder only; never stop vllm-brain;
memory < 85%. Commit per task.
