# Tier 3c — Recall: keep consolidated sources retrievable — Spec

**Date:** 2026-08-30  **Owner:** Toni  **Status:** approved ("proceed" after Tier 3b merge; Toni approved the vllm `--max_num_seqs 8` step, done and verified 02:50).
**Origin:** Tier 3b results (LoCoMo J 0.444 memory vs 0.696 full-context) + recall diagnostic (`eval/external/diagnose_recall.py`, branch `analysis/recall-diagnostic` fdbe3b3): of 857 WRONG answers, **743 (86.7%) had their evidence only in `archived` buffer rows** (consolidation archives sources; retrieval excludes archived), 85 (9.9%) were ranking misses (active evidence outside top-20), 24 (2.8%) generation failures. 91.1% of buffer rows end up archived. Recall@100 of evidence memories is 8.6%.

## Goal
Make the verbatim source memories stay retrievable after consolidation, without breaking the buffer/consolidation lifecycle, and re-measure LoCoMo (memory mode), LoCoMo-Plus and LongMemEval-S with the same harness and judge.

## Design
1. **Config** `memory.consolidation_keep_sources: bool = True` (new default; behavior change documented in CHANGELOG + CONFIGURATION).
2. **Consolidation** (`memory/consolidation.py`): when keep_sources is on,
   - multi-member cluster: create the `[Consolidated]` core row as today; source rows stay `status='active'`, `tier='buffer'`, and get `metadata.consolidated_into = <core id>` (plus `metadata.consolidated_at = now`) via a new storage `update_memory_metadata(id, patch)` (merge, not replace). Not archived.
   - singleton with importance ≥ 0.6: **promote in place** (`UPDATE tier='core'`, metadata `promoted_from_buffer = true`) instead of creating a copy + archiving (no duplicate text in retrieval). Below 0.6: mark `consolidated_into = null`?? — no: leave untouched but mark `metadata.consolidation_seen = true` so it no longer counts toward the threshold (see 3). 
   - when keep_sources is off: byte-identical to today's behavior.
   - stats gain `kept` (sources kept active) alongside `archived`.
3. **Threshold bookkeeping**: `needs_consolidation()` and the consolidation chunk query count/pull only **unconsolidated** buffer rows: storage `count_memories(character_id, tier="buffer", unconsolidated=True)` / `get_memories(..., unconsolidated=True)` filter `json_extract(metadata,'$.consolidated_into') IS NULL AND json_extract(metadata,'$.consolidation_seen') IS NULL`. `MemoryStore.count`/`needs_consolidation` mirror. `drain()`'s "archived==0 → stop" guard becomes "archived+kept+promoted == 0 → stop".
4. **Retrieval**: unchanged — kept sources are active and therefore candidates. Prompt assembly unchanged (a summary and its sources may co-occur; acceptable, measured).
5. **Maintenance/decay**: verify no other path archives buffer rows (`grep archived`: only consolidation + `MemoryStore.archive`); `describe()`/X-Ray tier counts keep working (buffer count now includes kept sources — show "buffer (n, of which consolidated m)" only if trivial; else leave).
6. **Export/import**: metadata already round-trips; no change.

## Measurement (harness unchanged, brain now `--max_num_seqs 8`)
- `locomo-mem-v2`: full LoCoMo memory run with the new default (10 shards) → compare with v1 per category; then `plus-mem-v2` (`--reuse-run locomo-mem-v2`) and `lme-s-50-v2`. Full-context baselines unchanged (reuse v1 numbers).
- Success bar: LoCoMo J improves materially (expect single-hop to move most); report exactly as measured either way. Also re-run `diagnose_recall` on v2 to show the new miss distribution.
- Docs: BENCHMARKS.md gains a v2 column/rows + a "what changed" note; RESULTS.md re-rendered; README headline updated; CHANGELOG.

## Scope extension (2026-08-30 07:55, after the v2 diagnostic + offline ranking experiments)
The v2 diagnostic moved 85% of misses to "evidence active but outside top-20". Offline experiments (`eval/external/ranking_experiments.py`) showed a query-time-only fix: RRF `weight_recency=weight_importance=0` and a bug fix in the relationship strategy (unboosted candidates tied at 0 kept insertion order = oldest-first bias) lift evidence recall@20 from 23.9% to 50.8% on the same stored embeddings. Task 5 ships both (new defaults; relationship list ranks only boosted candidates). Measurement adds `locomo-mem-v2b` / `plus-mem-v2b`: re-answer the v2 DBs (no re-ingestion). Contextualized embedding text (+rrf_k 120, keyword weight 2.0 → 54.7% @20) needs re-embedding and is deferred to Tier 3d.

## Out of scope (follow-ups)
Ranking misses (9.9%: temporal/date-aware retrieval, query rewriting), generation failures (2.8%), `single-session-preference` instrument, run_plus aggregation guard, `_run_bookkeeping` beat no-op.

## Constraints
Branch `feat/tier3c-recall-keep-sources` off master 53157a3; merge `analysis/recall-diagnostic` into it first. pytest/ruff/pyright green; long-horizon bench (`tests/test_longhorizon.py`) must stay green — if it depends on archival counts, adapt expectations explicitly and say so. Never touch `experiments/`, `kotlin/`, `eval/results/latest.json`. Runs: never stop vllm-brain; watch memory < 85%.
