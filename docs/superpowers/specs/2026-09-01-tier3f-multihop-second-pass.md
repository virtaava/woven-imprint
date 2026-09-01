# Tier 3f — Multi-hop second-pass retrieval — Spec

**Date:** 2026-09-01  **Owner:** Toni  **Status:** approved ("merge and proceed" after Tier 3e merge dd860e8).
**Origin:** v3 abstain-with-evidence analysis (eval/external/runs/diagnostics/abstain/locomo-mem-v3/): `c_multi_hop` = 127/317 retrieved-but-wrong cases — the gold needs 2+ evidence turns and only part of the set reached the top-K block. LoCoMo cat-1 (multi-hop) J is 0.305 vs single-hop 0.709 (v3b). Query-time problem → measurable answer-only on existing DBs.

## Goal
When the initial retrieval surfaces partial evidence, a second pass keyed on what was found pulls in the co-dependent memories — raising multi-hop J without hurting the other categories or Plus.

## Design
**T1 (product):** `MemoryRetriever.retrieve(..)` gains an expansion stage, config-gated:
- `MemoryConfig.retrieval_second_pass: int = 0` (0 = off; N>0 = number of seed hits to expand from). When on: take the top `retrieval_second_pass` fused hits; for each, extract salient terms from its content (capitalized tokens, digits/dates, and words ≥ 5 chars not in a small stopword set — pure heuristics, no LLM call); run ONE combined FTS query over the union of those terms (limit 50) plus ONE semantic query using the mean of the seeds' stored vectors (limit relevance_semantic_topk); score the new candidates with the existing fusion against the ORIGINAL query rankings (they join the candidate pool and the gate; implementation may simply widen `gated` and re-fuse); final top-K unchanged in size. At most +1 FTS + 0 embed calls (vectors already stored; the mean needs no server) per retrieve.
- Deterministic, no LLM, bounded cost. Tests with FakeEmbedder: a two-part fact split across two memories where part B shares terms with part A but not with the query — B enters top-K only with the second pass on.
**T2 (measurement, controller):** locomo re-answer on the v3b DBs (`--reuse-ingest locomo-mem-v3b`), `--set memory.retrieval_second_pass=5` → t3fA vs v3b 0.562 (esp. cat-1 0.305); if cat-1 gains ≥ +0.05 with overall not down: Plus check at temp 0.3 (t3fB vs 0.354/0.359); LME-S 50 re-answer is NOT possible (DBs deleted) — skip, note.
**T3:** default ruling (flip `retrieval_second_pass` to 5 only if measured net-positive), docs, results, merge package.

## Constraints
Branch `feat/tier3f-multihop-second-pass` off master dd860e8. Green suite/bench as always; never touch experiments/, kotlin/.
