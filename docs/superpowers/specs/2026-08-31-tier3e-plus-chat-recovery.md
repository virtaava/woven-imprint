# Tier 3e — LoCoMo-Plus chat-path recovery — Spec

**Date:** 2026-08-31  **Owner:** Toni  **Status:** approved ("merge then proceed" after Tier 3d merge 7936577).
**Origin:** Tier 3d decomposition (ledger archived at eval/external/runs/driver/sdd-ledger-tier3d.md): Plus cognitive fell 0.421 → 0.269 under Tier 3d. Exonerated individually: vectors, rrf_k/keyword, dedup, 800-char cap (cap-200 run 0.284), identity tag (≈−2.7). Remaining suspects: the WEEKDAY date prefix and the `_build_context` SECTION REORDER (facts/memories now precede emotion/arc/relationship). Confound: `chat()` samples at a hardcoded temp 0.7 → ±4-pt run noise; old-code cluster 0.374–0.421 vs new-code cluster 0.242–0.307.

## Goal
Identify and fix (or justify) the chat-path change that costs ~0.10 on the cognitive cue test, with a noise-controlled protocol, without regressing LoCoMo 0.562 / LME-S 0.542.

## Design
**T1 (product, small):**
- `chat()`/`chat_stream()` read `get_config().llm.temperature` (config existed since forever; the hardcoded 0.7 ignored it — same default, now honored). Benchmarks can then `--set llm.temperature=0.3`.
- `ContextConfig.weekday_in_dates: bool = True` — gates the `Mon`/`Tue` token in `_format_memories` date prefixes (and only that). Product-justifiable (date-format preference) and gives the weekday separator.
- Tests; no behavior change at defaults except honoring the config field.

**T2 (measurement, controller):** all on the v2 DBs, v2d-equivalent retrieval settings (`rrf_k 60, keyword 1.0, embedding_context false`), `llm.temperature=0.3`:
- A `plus-t3eA`: new code as-is (noise-reduced baseline; run TWICE — A2 replica — to size residual noise).
- B `plus-t3eB`: + `context.weekday_in_dates=false`.
- Decision table: B−A ≥ +0.05 → weekday guilty → flip `weekday_in_dates` default to False for character chat (keep for QA? it lives in `_format_memories` shared by both — measure LoCoMo v2e-style re-answer with weekday off before flipping; the weekday helped nothing measurably on QA, it was an abstain-analysis hunch). B ≈ A → suspect = section reorder → C: one-commit scratch revert of the order, measurement only, then decide.
- Success bar: Plus ≥ 0.40 at temp 0.3 on the winning config, LoCoMo re-answer (v3b DBs) within noise of 0.562.

**T3:** docs (BENCHMARKS Tier 3e subsection incl. protocol note that Plus now runs at temp 0.3), defaults per measurement, results JSON, merge decision → Toni.

## Constraints
Branch `feat/tier3e-plus-chat-recovery` off master 7936577. Green suite/bench as always. Never touch `experiments/`, `kotlin/`. Brain 8 seqs; memory < 85%.
