# Tier 3e "Plus chat recovery" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-08-31-tier3e-plus-chat-recovery.md` (binding).
**Global constraints:** branch `feat/tier3e-plus-chat-recovery`; per commit: `.venv/bin/python -m pytest -q` (baseline 765 passed, 1 skipped), ruff check/format src/ tests/ eval/, pyright project + eval/external (0), `.venv/bin/python eval/bench_longhorizon.py` (12/12). Mock tests only.

### Task 1: chat temperature honors config; weekday flag
**Files:** src/woven_imprint/character.py (`chat` ~370, `chat_stream` ~524: `temperature=get_config().llm.temperature`; `_format_memories` weekday token gated on `get_config().context.weekday_in_dates`), src/woven_imprint/config.py (`ContextConfig.weekday_in_dates: bool = True` + YAML), docs/CONFIGURATION.md rows, CHANGELOG, tests (tests/test_format_memories.py weekday on/off; a chat test asserting the FakeLLM saw the configured temperature — spy on generate kwargs).
- [ ] TDD; commit `feat(chat): sampling temperature honors llm.temperature; weekday date token configurable`.

### Task 2 (controller): measurement matrix per spec (A, A2, B on v2 DBs at temp 0.3; LoCoMo weekday-off re-answer on v3b DBs; optional C scratch revert).
### Task 3: defaults per measurement + docs + results + merge package.
