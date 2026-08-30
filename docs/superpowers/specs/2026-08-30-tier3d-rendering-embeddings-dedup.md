# Tier 3d — Recall II: what the character sees, what it embeds, fact dedup — Spec

**Date:** 2026-08-30  **Owner:** Toni  **Status:** approved ("merge then proceed" after Tier 3c merge 4c2e6c9).
**Origin:** Tier 3c diagnostics. v2d misses: 64% evidence not retrieved, 35% evidence in prompt but wrong. The abstention analysis (`eval/external/abstain_analysis.py`, merged) found product-side rendering defects: `_format_memories` truncates every memory line at 200 chars (133/251 cases exposed, 23 with the gold cut off), user turns render as anonymous `[User]` (51 speaker-ambiguous), dates carry no weekday so "tomorrow"/"last weekend" are uncomputable (41). Offline ranking experiments: contextualized embedding text "[date] speaker: content" + rrf_k 120 + keyword weight 2.0 → evidence recall@20 54.7% (vs 50.8%). Near-duplicate extracted facts dominate the top-20 (17.7/20; 78/491 core rows in paraphrase groups in conv-26).

## Goals (measured with the unchanged harness/judge)
1. **Rendering** — memory lines carry the full content up to a configurable cap, the user's identity, and a weekday.
2. **Embedding context** — buffer memories are embedded with date + speaker context; existing DBs can be re-embedded.
3. **Fact dedup** — near-duplicate fact-derived core memories are not inserted twice.
4. Publish v2e (rendering only, answer-only on the v2d DBs) and v3 (fresh ingest with everything) for LoCoMo, LoCoMo-Plus, and LongMemEval-S (50-question sample, first re-run since v1).

## Design
**T1 Rendering (`character.py::_format_memories`, memory storage of user turns):**
- `ContextConfig.memory_content_max_chars: int = 800` (0 = no cap) replaces the hard-coded `[:200]`. The overall prompt budget mechanism is unchanged.
- User turns store `metadata.user_id` (from `chat(user_id=…)` / `ingest(user_id=…)` / `ingest_exchange`); rendering shows `[User: <user_id>]` when known, `[User]` otherwise. Character turns stay `[<name>]`. Stored content is unchanged (`"[User] …"`), so no migration.
- Date prefix gains the weekday: `(2023-05-08 Mon, 3 months ago)`.
- Long-horizon bench `dates_rendered` adapts only if it pins the old exact format (state it).
- Measurement **v2e**: answer-only re-run on the v2d DBs (`--reuse-ingest locomo-mem-v2d`), current defaults.

**T2 Embedding context (`memory/store.py::add`, config, CLI):**
- `MemoryConfig.embedding_context: bool = True`. When on, the vector is computed from `embed_text = f"[{YYYY-MM-DD}] {speaker}: {body}"` where `speaker` = `User: <user_id>`/`User` for user rows, the character name for character rows (the stored `[Name] ` / `[User] ` tag is stripped from `body` to avoid the name twice), and `f"[{date}] {content}"` for other rows (facts/summaries/observations). Stored `content` unchanged. No nomic `search_*` prefixes (measured harmful).
- `woven-imprint reembed <character_id> [--batch 64]` CLI + `MaintenanceRunner` job `reembed` (opt-in, not in the nightly default set): recompute vectors for all active memories with the current builder; idempotent; logs count.
- Retrieval defaults per the offline sweep on contextualized docs: `rrf_k 60 → 120`, `weight_keyword 1.0 → 2.0` (validated live by v3; revert if v3 contradicts).

**T3 Fact dedup (`character.py::_store_facts` or wherever fact-derived core rows are inserted):**
- `MemoryConfig.fact_dedup_similarity: float = 0.92` (0 = off). Before inserting a fact-derived core memory, compare its embedding against active core rows (FTS candidates by statement words ∪ newest 500 core rows; cosine via the existing numpy path). If max ≥ threshold: skip the insert, bump the existing row `importance = min(1.0, importance + 0.05)`, `metadata.dup_count += 1`, `metadata.last_confirmed = now`. The structured `facts` table row is still written (its own supersession logic is unchanged). Stats/logging for the benchmark (`dedup_skipped` count in ingest stats if cheap).

**T4 Measurement (controller):** v2e (answer-only, T1); v3 fresh ingest 10 shards (T1+T2+T3, new rrf_k/keyword) → diagnostics (`diagnose_recall`, `abstain_analysis`) → `plus-mem-v3` → `lme-s-50-v3` (10 shards, `--interval 1 --delete-db-after-answer`). Full-context baselines unchanged.

**T5 Docs:** BENCHMARKS.md Tier 3d section (v2d → v2e → v3 tables incl. LME v1 → v3), ARCHITECTURE (rendering, embedding context, dedup), CONFIGURATION rows, CHANGELOG, README headline, RESULTS re-render, results JSON.

## Out of scope (follow-ups)
Multi-hop second-pass retrieval (79 cases), tier-aware candidate window at product scale, shared fusion function refactor, harness abstention wording (16 cases), photo-caption rendering as a fact.

## Constraints
Branch `feat/tier3d-rendering-embeddings-dedup` off master (4c2e6c9 + analysis merge). pytest/ruff/pyright green; long-horizon 12/12 (adapt only with explicit notes). Never touch `experiments/`, `kotlin/`, `eval/results/latest.json`. Never stop vllm-brain (8 seqs); memory < 85%.
