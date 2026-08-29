# Tier 3c "Recall: keep consolidated sources" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidation keeps source memories active and retrievable (config `memory.consolidation_keep_sources`, default True); threshold bookkeeping counts only unconsolidated buffer rows; re-measure LoCoMo/Plus/LME with the unchanged harness.

**Spec:** `docs/superpowers/specs/2026-08-30-tier3c-recall-keep-sources.md` (binding; read it first).

**Global constraints:** branch `feat/tier3c-recall-keep-sources` off master 53157a3 (with `analysis/recall-diagnostic` merged in). Before each commit: `.venv/bin/python -m pytest -q` (baseline 649 passed, 1 skipped — plus the diagnostic branch adds none), `.venv/bin/ruff check src/ tests/ eval/`, `.venv/bin/ruff format src/ tests/ eval/`, `.venv/bin/pyright --project pyrightconfig.json` (0 errors), `.venv/bin/pyright eval/external` (0 errors). Never `git add -A` (kotlin/, uv.lock untracked). Never call the LLM servers from tests.

---

### Task 1: Storage + config — unconsolidated filters, metadata patch, config flag

**Files:** `src/woven_imprint/storage/sqlite.py` (`count_memories`, `get_memories`: new kwarg `unconsolidated: bool = False` adding `AND json_extract(metadata,'$.consolidated_into') IS NULL AND json_extract(metadata,'$.consolidation_seen') IS NULL`; new `update_memory_metadata(memory_id, patch: dict) -> None` merging into the JSON — read the existing `update_memory_fields(metadata=…)` and reuse), `src/woven_imprint/storage/base.py` (abstract signatures if any), `src/woven_imprint/config.py` (`MemoryConfig.consolidation_keep_sources: bool = True` + the YAML template block + CONFIGURATION.md row), `src/woven_imprint/memory/store.py` (`count(tier, unconsolidated=False)`, `needs_consolidation` uses unconsolidated=True), tests `tests/test_storage_unconsolidated.py`.
- [ ] Tests: count/get with unconsolidated filter (rows with `consolidated_into`, with `consolidation_seen`, and plain rows); `update_memory_metadata` merges and preserves other keys; config default True and YAML override False.
- [ ] Implement; run checks; commit `feat(storage): unconsolidated buffer filters, metadata patch, consolidation_keep_sources config`.

### Task 2: Consolidation keep-sources behavior

**Files:** `src/woven_imprint/memory/consolidation.py`, `src/woven_imprint/character.py` (docstrings of `consolidate`/session-end note), tests `tests/test_consolidation_keep_sources.py` (+ adapt `tests/test_consolidation*.py` / any test asserting `archived` counts: run them with `get_config().memory.consolidation_keep_sources = False` in a fixture OR update expectations — state which in the report).
- [ ] Tests (FakeLLM/FakeEmbedder via tests/helpers): with keep_sources True: multi-member cluster → one `[Consolidated]` core row, sources remain `active`/`buffer` with `metadata.consolidated_into == core id` and `consolidated_at`; singleton importance ≥ 0.6 → same row promoted in place to `core` (no new id, `metadata.promoted_from_buffer`), singleton < 0.6 → untouched except `metadata.consolidation_seen = True`; `needs_consolidation()` returns False after a pass even though total buffer count ≥ threshold; second `consolidate()` pass over the same rows is a no-op (pulls only unconsolidated rows; `len(buffer) < 10` → early return); `drain()` terminates; stats include `kept`; retrieval (`MemoryRetriever`) can still return a kept source by exact FTS phrase after consolidation. With keep_sources False: existing behavior byte-identical (archived counts as before).
- [ ] `consolidate()`: pull `get_memories(tier="buffer", limit=chunk_size, unconsolidated=True)`; branch per spec; `drain()` stop guard = `archived + kept + promoted == 0`. `needs_consolidation()` → `count_memories(..., tier="buffer", unconsolidated=True)`.
- [ ] Long-horizon bench: run `.venv/bin/python -m pytest tests/test_longhorizon.py -q` and `.venv/bin/python eval/bench_longhorizon.py` (deterministic, no LLM) — if any expectation depends on archival (e.g. "crosses 233 core rows", buffer sizes), report the before/after numbers and adapt only with an explicit note in the commit message.
- [ ] Checks; commit `feat(consolidation): keep sources retrievable (consolidation_keep_sources); promote singletons in place`.

### Task 3: Measure — locomo-mem-v2 (controller runs; implementer only prepares)
- [ ] Controller: `for i in 0..9: python -m eval.external run --bench locomo --mode memory --run-id locomo-mem-v2 --shard i/10 --no-results --timeout 900 &` then unsharded aggregate; then `python -m eval.external.diagnose_recall` pointed at v2 (add `--run-id` arg to the diagnostic script in Task 2 if missing — implementer: add `--run-id` and `--results` CLI args, default v1 paths).
- [ ] Controller: `plus-mem-v2` (`--reuse-run locomo-mem-v2`), `lme-s-50-v2` (10 shards, `--interval 1 --delete-db-after-answer`).

### Task 4: Docs
- [ ] `docs/BENCHMARKS.md`: results tables gain v2 columns (memory v1 → v2, full-context unchanged); "What changed in v2" paragraph citing the diagnostic numbers (86.7% archived-evidence misses; 91.1% buffer archived) and the new default; diagnostic section (how to run `diagnose_recall`, v1 vs v2 miss distribution); runtime with `--max_num_seqs 8`. `docs/RESULTS.md` re-rendered (renderer prints whatever keys are in `external_latest.json` — v2 runs overwrite the `bench:mode` keys; keep v1 per-run files and cite them). README headline; CHANGELOG (behavior change: consolidation no longer archives sources by default; config flag); `docs/CONFIGURATION.md` row; ARCHITECTURE consolidation paragraph.
- [ ] Commit results JSON (`external_latest.json`, `external_*-v2.json`, judge sample) with `git add -f`.
