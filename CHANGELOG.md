# Changelog

All notable changes to Woven Imprint will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-03-25

### Added
- React demo UI replacing Gradio (`woven-imprint demo`)
  - Chat with any character, markdown rendering, suggested prompts
  - X-Ray sidebar: live memory feed, relationship radar chart, emotion indicator
  - Collapsible X-Ray panel (toggle + localStorage persistence)
  - Character management: create, delete, export (JSON), import (JSON/PNG/markdown), migrate from text
  - Character selector in top bar for switching between characters
  - Reflect button for character self-reflection
  - Provider configuration with live model discovery
  - Provider presets: Ollama, OpenAI, Anthropic, DeepSeek, NVIDIA NIM, Custom (any OpenAI-compatible API)
  - Connection test required before saving provider
- FastAPI demo server with security hardening
  - Bearer token auth on all API routes
  - CORS locked to localhost (relaxed with --host 0.0.0.0)
  - Provider secrets never exposed to frontend
  - Graceful shutdown with session flushing
- Service-layer extraction from sidecar/API handlers
- `--host` flag for remote access (Tailscale, network)
- `--port` flag (default 7860)
- `--no-browser` flag
- Persistence regression tests
- Meridian seed database build script

### Removed
- Gradio web UI (`woven-imprint ui` command removed)
- `[ui]` optional dependency (replaced by `[demo]`)

### Changed
- `woven-imprint demo` now launches the React demo (previously a terminal REPL)

## [0.4.0] - 2026-03-18

### Added
- **Provider agnosticism**: All entry points use factory functions instead of hardcoded Ollama. Configure `llm_provider` (ollama/openai/anthropic) and `embedding_provider` (ollama/openai) in config or via `WOVEN_IMPRINT_LLM_PROVIDER` / `WOVEN_IMPRINT_EMBEDDING_PROVIDER` env vars.
- New `providers.py` module with `create_llm()` and `create_embedding()` factory functions.
- `WOVEN_IMPRINT_API_KEY_LLM` and `WOVEN_IMPRINT_BASE_URL` env vars for provider API keys and custom endpoints.
- `WOVEN_IMPRINT_ENFORCE_CONSISTENCY` env var (was missing from env_map).
- Consistency checker now accepts `CharacterConfig` for configurable retries, temperature, and fail-open score.
- JSON parse retry: consistency checker retries once at temperature=0.1 when `generate_json()` returns non-dict.
- Conversation context passed to consistency `check()` via `enforce()` — last 3 message pairs included for growth justification.
- Dynamic fact extraction cap: scales by exchange length (>2000 chars = 2x cap up to 15; <200 chars = half cap, min 2). Configurable via `max_facts_per_extraction` and `fact_density_scaling`.
- Enriched extraction prompt: includes "preferences, biographical details" in categories; adds recent conversation context with "do not re-extract" instruction.
- `WOVEN_IMPRINT_MAX_FACTS` env var for fact extraction cap.
- `MigrationConfig` dataclass: `max_messages`, `max_message_length`, `chunk_size` — all configurable.
- Chunked conversation analysis for large exports: `_analyze_conversations_chunked()` processes in chunks, `_synthesize_analyses()` merges via LLM.
- Scaled relationship sample size in migration: `min(60, max(30, n//10))` instead of fixed 30.
- `tests/test_providers.py`: 9 tests for factory functions.
- `tests/test_migration.py`: 9 tests for parsers and chunked analysis.

### Changed
- ChatGPT export parser: default is now unlimited messages (was hardcoded 500) and unlimited message length (was hardcoded 2000). Use `MigrationConfig` to set limits.
- Claude project parser: `rglob("*.md")` for all markdown files (was only `memory/*.md`), also scans `.claude/` directory, no character limits on file content.
- CLI, UI, MCP server, API server, and Engine all use `create_llm()`/`create_embedding()` factories — no direct Ollama imports in entry points.
- `CharacterConfig` gains `consistency_max_retries`, `consistency_temperature`, `consistency_fail_open_score`.
- `MemoryConfig` gains `max_facts_per_extraction`, `fact_density_scaling`.
- Config default template includes all new settings.
- 167 tests (was 146+), 1 skipped (optional anthropic dependency).

### Fixed
- `enforce_consistency` missing from environment variable map — now configurable via `WOVEN_IMPRINT_ENFORCE_CONSISTENCY`.
- `enforce()` never passed conversation context to `check()` despite `check()` having a `context` parameter.
- Fact extraction used hardcoded `facts[:5]` regardless of exchange density.

## [0.3.1] - 2026-03-17

### Added
- LLM provider resilience: retry with exponential backoff + circuit breaker
- Configurable: max_retries, retry delays, circuit breaker threshold/cooldown

## [0.3.0] - 2026-03-17

### Added
- Centralized configuration: `~/.woven_imprint/config.yaml` with 50 settings
- `woven-imprint config --init` and `woven-imprint config` commands
- Configuration reference documentation (docs/CONFIGURATION.md)
- PyYAML as core dependency
- All modules wired to read from config (no more editing source)
- 6th RRF retrieval strategy (explicit tier priority ranking)

### Changed
- Config priority: CLI flags > env vars > config file > defaults

## [0.2.1] - 2026-03-17

### Added
- Parallel subsystem calls via ThreadPoolExecutor (opt-in: `character.parallel = True`)
- Thread-safe SQLite with `check_same_thread=False`
- CI step timeouts to catch hangs

### Fixed
- Infinite recursion in SQLite `_commit()` method

## [0.2.0] - 2026-03-17

### Added
- Auto-consolidation every 20 turns + at session end (was never called)
- Belief revision wired into fact extraction (auto-detects contradictions)
- Periodic state save every 10 turns (emotion/arc survives mid-session crash)
- Tier priority as 6th RRF retrieval strategy
- Consolidation correctness benchmark (14th benchmark)
- Live persistence benchmarks: 50-session recall, adversarial persona (8/8),
  contradiction handling (4/4), held-out character (100%)
- Evaluation methodology doc with exact prompts and honest limitations
- Docker Compose setup (Ollama + Woven Imprint self-contained)
- OLLAMA_HOST env var for remote/containerized Ollama
- API server bearer token auth (`--api-key` or `WOVEN_IMPRINT_API_KEY`)
- Actionable Ollama error messages
- Web UI with 4 tabs (Chat, Characters, Migrate, Settings)
- `woven-imprint ui --browser chrome` configurable browser
- `woven-imprint update` command with pipx support
- `woven-imprint migrate` from ChatGPT, SillyTavern, Custom GPTs, Claude
- Custom GPT knowledge file import (`--knowledge` flag)
- PDF extraction via pymupdf
- Platform-specific setup guides (Windows, macOS, Linux, Docker)
- `/slash` commands in CLI chat

### Changed
- All model defaults unified to `llama3.2`
- Cross-session persistence: 67% → 100%
- Benchmarks: 13/13 (94.8%) → 14/14 (97.9%)
- Session summary importance: 0.7 → 0.85
- Extracted fact importance: 0.6 → 0.75
- Core tier boost: 0.15 → 0.2, Bedrock: 0.3 → 0.35

### Fixed
- Gradio 6.0 compatibility
- Proper PNG chunk parsing for TavernAI cards
- save_character no longer wipes state
- Relationship trajectory uses clamped deltas
- Growth memories have embeddings
- Emotion mood case-insensitive
- FTS5 update trigger only fires on content changes
- Consistency checker handles non-dict LLM response

## [0.1.2] - 2026-03-17

### Added
- Full-featured web UI with 4 tabs: Chat, Characters, Migrate, Settings
- `woven-imprint update` command — upgrades core + all installed extras
- `woven-imprint ui` command — launches browser-based interface
- File migration in UI (upload ChatGPT JSON, SillyTavern cards, etc.)
- Export/import/delete characters from the UI
- Memory search and reflect actions in the UI
- PDF knowledge file extraction via pymupdf
- Custom GPT knowledge file import (`--knowledge` flag)
- Auto-open browser on `woven-imprint ui`

### Fixed
- `woven-imprint update` now also upgrades pipx-injected extras (gradio, openai, etc.)
- Gradio 6.0 compatibility
- Proper PNG chunk parsing for TavernAI character cards
- `/slash` commands in CLI chat to avoid collision with character messages
- Linux/WSL/Ubuntu 24.04+ install guide (externally-managed-environment)

## [0.1.1] - 2026-03-17

### Fixed
- Gradio 6.0 compatibility — removed deprecated `theme` and `type` parameters
- Documentation: externally-managed-environment fix for Linux/WSL/Ubuntu 24.04+
- Documentation: pipx inject instructions for optional extras (ui, pdf)
- Documentation: MCP tool count and missing migrate_from_text in tool table
- Documentation: architecture diagram accuracy (removed unimplemented Qdrant reference)

## [0.1.0] - 2026-03-17

### Added
- Three-tier memory system (buffer, core, bedrock) with SQLite storage
- Multi-strategy retrieval via Reciprocal Rank Fusion (semantic, keyword, recency, importance, relationship)
- Tier-aware recency decay (bedrock persists months, buffer fades in days)
- Two-phase retrieval: FTS pre-filter finds old memories beyond the recency window
- Four-level persona constraints (hard, temporal, soft, emergent)
- Birthdate-derived age with birthday detection and leap year handling
- NLI-inspired consistency checking with post-generation enforcement
- Character growth engine (soft constraints evolve from accumulated experience)
- Five-dimensional relationship tracking (trust, affection, respect, familiarity, tension)
- Bounded relationship changes (max +/-0.15 per interaction)
- Emotional state tracking (15 moods, natural decay toward neutral)
- Narrative arc awareness (6 phases, tension curves, story beat detection)
- Memory consolidation engine (buffer compression into core memories)
- Belief revision system (reinforce, contradict, invalidate with certainty scores)
- Conversation buffer with context window management and graceful overflow
- Multi-character interaction (two-character dialogue, group scenes)
- LLM providers: Ollama, OpenAI, Anthropic (any OpenAI-compatible)
- Embedding providers: Ollama (nomic-embed-text), OpenAI
- CLI tool: demo, create, chat, list, stats, export, delete, import, serve
- OpenAI-compatible API proxy server (model name = character name)
- MCP server for IDE integration (Claude Desktop, Cursor, Hermes, OpenClaw)
- Character import/export (JSON with re-embedding on import)
- Lightweight mode (skip emotion/arc tracking for faster responses)
- 146 unit tests, 13 evaluation benchmarks (94.8% avg score)
- Pride and Prejudice relationship evolution demo (16 scenes, 6 characters)
- Dockerfile for containerized deployment
- CI: lint (ruff), typecheck (pyright), test matrix (3.11/3.12/3.13), CodeQL, Dependabot
