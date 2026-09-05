# Tier 3p — DF-aware entity pivoting — Spec

**Date:** 2026-09-05
**Origin:** Tier 3o's measured diagnosis — the entity second pass failed (0.638 vs 0.674)
because two-speaker stores make most entity handles low-discriminative: speaker names
match half the store, and the FTS pivot floods the candidate pool. The recorded follow-up
is to pivot ONLY on rare entities.

## Mechanism (small, additive)
1. **Storage**: `SQLiteStorage.fts_term_count(character_id, term) -> int` — count of
   active memories matching the (sanitized, quoted, single-term) FTS query. Same
   sanitization discipline as `fts_search` (strip to `\\w+` words, quote); a term that
   sanitizes to nothing returns 0. Under `self._lock`, one `SELECT count(*)`.
2. **Config**: `memory.second_pass_entity_max_df: float = 0.05` — an entity is pivot-
   eligible only if its FTS document count ≤ `max_df × active-memory count` (the
   active count comes from the candidate pool already in hand:
   `len(all_memories)` at that point in `retrieve()` — no extra query). Validated in
   `MemoryConfig.__post_init__`: `0 < value <= 1`.
3. **Retrieval** (inside the existing Tier 3o entity-terms block, only when
   `second_pass_entities` is on): after building the deduped entity union (cap 16 as
   today), filter it: keep entity `e` iff
   `storage.fts_term_count(character_id, e) <= max_df * max(len(all_memories), 1)`.
   At most 16 count queries per retrieve (µs-scale SQLite counts). Empty result after
   filtering → the existing `_salient_terms` fallback fires unchanged. Off path
   (`second_pass_entities=False`) untouched — the filter lives entirely inside the
   entity branch.
4. Docs: config docstring + example line + CONFIGURATION.md row amendment (the
   `second_pass_entities` row gains the DF sentence).

## Measurement (pre-registered bar — declared BEFORE any run)
Answer-only on the EXISTING `locomo-mem-v3b-ent` backfilled copies, shipping protocol
(K=100 hardened + agg-stage), `--set memory.retrieval_second_pass=3
--set memory.second_pass_entities=true` (max_df at its 0.05 default): run
`locomo-t3pDF`. Baselines: t3nAgg 0.674/cat-1 0.500/adv 0.877; t3oEnt 0.638/0.408
(unfiltered entities); t3oSal 0.658/0.450.
- **Adopt (flip `retrieval_second_pass=3` + `second_pass_entities=true` defaults) iff:**
  overall ≥ 0.669 AND cat-1 ≥ 0.530 AND adversarial ≥ 0.87 — the same absolute bar
  Tier 3o pre-registered; the DF filter must make the mechanism BEAT the no-second-pass
  baseline, not merely un-hurt it.
- Missed → ships opt-in; the Tier 3o section gains the DF data point; the
  entity-retrieval direction is then CLOSED (two measured failures) and the queue moves
  on (preference protocol fix next).
- One diagnostic permitted before the run (not a tuning loop): print the DF distribution
  of extracted entities on one v3b-ent DB to sanity-check 0.05 leaves a non-empty
  eligible set; if it filters everything, set the default from that histogram ONCE,
  document the choice in the spec before the measurement run.

## Acceptance
Tests: fts_term_count (counts, sanitization, empty term); config validation bounds; the
DF filter drops a corpus-flooding entity while keeping a rare one (mutation-bound: test
must fail if the filter is bypassed); empty-after-filter falls back to salient terms;
off path byte-identical. Suite/ruff/pyright green; bench 12/12. BENCHMARKS/CHANGELOG per
outcome; results committed.

## Global constraints
Branch `feat/tier3p-df-entity-pivot` off master `3744e0b`. Never touch
`experiments/parametric_spike/` or `kotlin/`. No schema migration. Additive only.
