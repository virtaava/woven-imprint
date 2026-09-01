# Tier 3g "Relative time + photo captions" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-09-01-tier3g-relative-time-photo-render.md` (binding).
**Global constraints:** per commit: full pytest (baseline 789 passed, 1 skipped), ruff, pyright (project + eval/external), bench_longhorizon 12/12. Mock tests only. Both features off → byte-identical rendering (tests must prove it).

### Task 1: both rendering features
**Files:** src/woven_imprint/character.py (`_format_memories` + two pure helpers `_relative_date_hints(content, created) -> str` and `_split_photo_captions(content) -> tuple[str, list[str]]` — module-level, unit-testable), src/woven_imprint/config.py (two ContextConfig flags + YAML), docs/CONFIGURATION.md, CHANGELOG, tests/test_relative_date_hints.py + tests/test_photo_caption_line.py (closed-set cases incl. last/next weekday arithmetic across month/year boundaries, weekend rule, cap at 3, no false hits on "lastly"/"nextdoor"; caption split single/multiple/absent/nested parens; flags off byte-identical; caps interaction: hints don't count toward memory_content_max_chars? — RULING: the cap applies to the content BEFORE hint/caption processing; hints and [photo] lines are never truncated).
- [ ] TDD; commit `feat(prompt): relative-date hints and photo-caption continuation lines in memory rendering`.

### Task 2 (controller): locomo-t3gA answer-only on v3b DBs; bar per spec; defaults per outcome.
### Task 3: docs + results + merge package.
