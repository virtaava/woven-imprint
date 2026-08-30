# Tier 3d "Recall II" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-08-30-tier3d-rendering-embeddings-dedup.md` (binding; read first).
**Global constraints:** branch `feat/tier3d-rendering-embeddings-dedup`; before each commit: `.venv/bin/python -m pytest -q` (baseline 691 passed, 1 skipped), `.venv/bin/ruff check src/ tests/ eval/`, `.venv/bin/ruff format src/ tests/ eval/`, `.venv/bin/pyright --project pyrightconfig.json` (0), `.venv/bin/pyright eval/external` (0), `.venv/bin/python eval/bench_longhorizon.py` (12/12). Never `git add -A`. Mock tests only (no LLM/embedding servers from tests).

### Task 1: Rendering — content cap, user identity, weekday
**Files:** `src/woven_imprint/character.py` (`_format_memories` ~1782; user-turn stores at ~264, ~442, ~703 (`chat` paths) and `ingest`/`ingest_exchange` — add `metadata={"user_id": user_id}` when user_id is given; check `MemoryStore.add` accepts `metadata`), `src/woven_imprint/config.py` (`ContextConfig.memory_content_max_chars: int = 800` + YAML template + docs/CONFIGURATION.md row), `docs/ARCHITECTURE.md` (prompt section), tests `tests/test_format_memories.py` (new) + adapt any test pinning the old line format (grep tests/ for `[:200]`, `"(20`, `[User]`), `eval/bench_longhorizon.py` `dates_rendered` only if it pins the exact format.
- [ ] Tests: line keeps 500-char content intact at default cap; cap 0 = unlimited; cap 100 truncates; user turn with `metadata.user_id="caroline"` renders `[User: caroline]`, without → `[User]`; character turn `[Ada]`; weekday present `(2023-05-08 Mon, …)`; `chat()`/`ingest()`/`ingest_exchange()` store `metadata.user_id`.
- [ ] Implement; checks; commit `feat(prompt): memory lines — configurable content cap (800), user identity, weekday`.

### Task 2: Embedding context + reembed
**Files:** `src/woven_imprint/memory/store.py` (`add`: build embed_text per spec; helper `build_embed_text(row_like) -> str` reusable by reembed), `src/woven_imprint/config.py` (`embedding_context`, `rrf_k` 120, `weight_keyword` 2.0 + YAML + CONFIGURATION), `src/woven_imprint/maintenance.py` (job `reembed`, opt-in), `src/woven_imprint/cli.py` (`reembed` command), `src/woven_imprint/storage/sqlite.py` (batch embedding update helper if missing), tests `tests/test_embedding_context.py`.
- [ ] Tests: the embedder receives `"[2023-05-08] User: caroline: I adopted a cat"` for a user buffer row (tag stripped), `"[2023-05-08] Ada: …"` for character rows, `"[date] content"` for core/fact rows; flag off → raw content; `reembed` recomputes all active rows (FakeEmbedder call count == rows) and is idempotent; retrieval still works after reembed; config defaults rrf_k 120 / weight_keyword 2.0 documented with the offline numbers (54.7% @20 on contextualized docs).
- [ ] Implement; checks; commit `feat(embedding): contextualized embedding text (date+speaker), reembed job/CLI; rrf_k 120, keyword weight 2.0`.

### Task 3: Fact dedup at extraction
**Files:** `src/woven_imprint/character.py` (fact → core memory insertion path; find it: grep `_store_facts` / where extracted facts become `tier="core"` rows), `src/woven_imprint/config.py` (`fact_dedup_similarity` 0.92), `src/woven_imprint/memory/store.py` or `retrieval.py` (reuse cosine helpers), tests `tests/test_fact_dedup.py`.
- [ ] Tests (FakeEmbedder bag-of-words): identical statement twice → one core row, `dup_count` 1, importance bumped; paraphrase below threshold → two rows; threshold 0 → off; the `facts` table still gets both rows; ingest stats expose `dedup_skipped`.
- [ ] Implement; checks; commit `feat(facts): semantic dedup of fact-derived core memories (fact_dedup_similarity 0.92)`.

### Task 4: Measurement (controller): v2e answer-only after T1; v3 fresh ingest after T3 → diagnostics → plus-mem-v3 → lme-s-50-v3.

### Task 5: Docs + results (after numbers): BENCHMARKS Tier 3d section, ARCHITECTURE, CONFIGURATION, CHANGELOG, README, RESULTS re-render, results JSON commit.
