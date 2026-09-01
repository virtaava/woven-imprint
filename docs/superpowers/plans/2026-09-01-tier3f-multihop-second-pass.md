# Tier 3f "Multi-hop second pass" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-09-01-tier3f-multihop-second-pass.md` (binding).
**Global constraints:** per commit: full pytest (baseline 772 passed, 1 skipped), ruff, pyright (project + eval/external), bench_longhorizon 12/12. Mock tests only.

### Task 1: expansion stage in MemoryRetriever
**Files:** src/woven_imprint/memory/retrieval.py (after the first fusion: seed selection, term extraction helper `_salient_terms(text) -> list[str]`, one FTS + one mean-vector semantic pull, candidate-pool widening + re-fusion; keep the relevance gate semantics), src/woven_imprint/config.py (`retrieval_second_pass: int = 0` + YAML), docs/CONFIGURATION.md, CHANGELOG, tests/test_retrieval_second_pass.py (split-evidence scenario; off = byte-identical ranking to today; cost bound: exactly one extra fts_search and no embedder.embed calls when on — spy).
- [ ] TDD; commit `feat(retrieval): optional second-pass expansion for multi-hop recall (retrieval_second_pass)`.

### Task 2 (controller): t3fA locomo re-answer on v3b DBs (second_pass=5) vs 0.562/cat-1 0.305; if good → t3fB Plus at temp 0.3.
### Task 3: default ruling + docs + results + merge package.
