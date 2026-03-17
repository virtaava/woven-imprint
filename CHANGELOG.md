# Changelog

All notable changes to Woven Imprint will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
