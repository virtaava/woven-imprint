# Changelog

All notable changes to Woven Imprint will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
