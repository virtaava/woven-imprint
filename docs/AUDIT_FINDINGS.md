# Woven Imprint — Audit Findings (2026-03-17)

Four parallel audits: general, security, performance, API/UX.

## Critical (show-stoppers)

| # | Finding | Source |
|---|---------|--------|
| C1 | `generate_id()` uses `time.time_ns()` — same-nanosecond calls produce identical IDs. `_seed_bedrock` loop clobbers all but last seed. | General |
| C2 | MCP server calls `load_character()` per tool call → fresh object → emotion/arc/session/context all lost between calls | General |
| C3 | Emotional state + narrative arc not persisted — `load_character()` creates fresh objects | General |
| C4 | Feb 29 birthdate crashes `days_until_birthday` — `date.replace(year=)` raises ValueError | General |
| C5 | Retrieval blind at scale — `ORDER BY created_at DESC LIMIT 200` acts as recency filter BEFORE semantic search. At 10K+ core memories, relevant old memories never loaded | Performance |
| C6 | 5-9 sequential LLM calls per chat() = 12-16s latency. No way to disable subsystems except consistency | Performance |

## High Priority

| # | Finding | Source |
|---|---------|--------|
| H1 | `belief.contradict()` creates memory with `embedding=None` → invisible to semantic retrieval | General |
| H2 | `touch_memory` triggers FTS5 reindex 10x per retrieval (trigger has no column filter) | Performance |
| H3 | Every write is its own SQLite COMMIT — 5-10+ disk syncs per chat() | Performance |
| H4 | O(n²) clustering in consolidation — degrades past 500 buffer entries | Performance |
| H5 | Prompt injection — memory content injected unsanitized into 11 LLM prompt surfaces | Security |
| H6 | No input size limits — 100MB message would embed, store, and send to LLM | Security |
| H7 | API server: no CORS preflight (do_OPTIONS), error format doesn't match OpenAI spec, no streaming | API/UX |
| H8 | `load_character()` returns None silently — causes AttributeError downstream | API/UX |
| H9 | Lazy imports in `llm/__init__.py` are factory functions, not class re-exports — isinstance/type checking breaks | API/UX |
| H10 | `woven-imprint list` fails if Ollama is down (constructs LLM unnecessarily) | API/UX |

## Medium Priority

| # | Finding | Source |
|---|---------|--------|
| M1 | No thread safety — single SQLite connection, single-threaded HTTP server | Performance |
| M2 | No `PRAGMA busy_timeout` — concurrent access gets SQLITE_BUSY | Performance |
| M3 | FTS5 query syntax could search across character boundaries (mitigated by SQL WHERE) | Security |
| M4 | Path traversal in export/import — no path validation (only CLI-exposed currently) | Security |
| M5 | API server: no auth, CORS wildcard, user_id spoofing via system message | Security |
| M6 | No schema migration system — future changes break existing databases | General |
| M7 | No logging or observability anywhere | General |
| M8 | Architecture doc lists 4 modules that don't exist (vllm, qdrant, sentence_transformers, relationship/graph) | API/UX |
| M9 | Missing CLI commands: delete, import, --version | API/UX |
| M10 | Default model `qwen3-coder:30b` — fails on laptops, no auto-detection | API/UX |
| M11 | No `Engine.__enter__/__exit__` context manager | API/UX |
| M12 | No `Character.get_relationship()` convenience method (docs promise it) | API/UX |

## Low Priority

| # | Finding | Source |
|---|---------|--------|
| L1 | Pure Python cosine similarity — numpy would be ~100x faster | Performance |
| L2 | No async support anywhere | Performance |
| L3 | `detect_contradictions()` exists but is never called automatically | General |
| L4 | No `py.typed` marker for PEP 561 | General |
| L5 | No CHANGELOG, CONTRIBUTING.md | General |
| L6 | Test coverage ~30% — character.py, engine.py, retrieval.py, CLI, servers untested | General |
| L7 | OpenAI proxy token counts are word counts, not token estimates | API/UX |
| L8 | MCP chat tool returns no metadata (emotion, arc, relationship) | API/UX |
| L9 | `group_interaction` accumulated_context grows unboundedly | General |
