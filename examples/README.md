# Examples

## Prerequisites

All Python examples need Woven Imprint installed and an LLM backend configured.

**With Ollama (default)**:
```bash
pip install woven-imprint
ollama pull llama3.2            # or any chat model
ollama pull nomic-embed-text    # for memory embeddings
```

**With OpenAI** (no Ollama needed):
```bash
pip install woven-imprint[openai]
export WOVEN_IMPRINT_LLM_PROVIDER=openai
export WOVEN_IMPRINT_EMBEDDING_PROVIDER=openai
export WOVEN_IMPRINT_API_KEY_LLM=sk-your-key-here
```

## Examples

### [basic_usage.py](basic_usage.py) — Your First Character

Create a character, have a conversation, check the relationship, and see a reflection.
This covers the core loop: `create_character` → `chat` → `relationships` → `reflect`.

```bash
python examples/basic_usage.py
```

### [multi_character.py](multi_character.py) — Two Characters Talk

Two characters meet and interact in a scene. Demonstrates `interact()` for
two-character dialogue with automatic memory and relationship tracking on both sides.

```bash
python examples/multi_character.py
```

### [openai_proxy.py](openai_proxy.py) — Use via OpenAI API

Chat with characters through the standard OpenAI Python SDK. Any tool that
speaks the OpenAI API can use persistent characters — just point the `base_url`
to the Woven Imprint server.

**Step 1**: Start the server
```bash
woven-imprint serve --port 8650
```

**Step 2**: Create a character (in another terminal)
```bash
woven-imprint create "Marcus the Blacksmith"
```

**Step 3**: Run the example
```bash
pip install openai
python examples/openai_proxy.py
```

### [mcp_setup.md](mcp_setup.md) — IDE Integration

Configuration guide for using Woven Imprint characters in:
- **Claude Desktop** / **Cursor** — add to MCP config
- **Hermes Agent** — add to config.yaml
- **OpenClaw** — add to openclaw.json

No code needed — just config. Characters are available as MCP tools.

## Migrating Existing Characters

Already have a character in ChatGPT, SillyTavern, or a Custom GPT? Bring it over:

```bash
woven-imprint migrate conversations.json           # ChatGPT data export
woven-imprint migrate character_card.png            # SillyTavern / TavernAI card
woven-imprint migrate --text "You are Marcus..."    # Custom GPT instructions
woven-imprint migrate /path/to/claude/project/      # Claude Code project
woven-imprint migrate persona.md                    # Any text/markdown
```

The system extracts persona, memories, relationship baselines, and emotional state
automatically. See the platform setup guides ([Windows](../docs/setup-windows.md#step-6-bring-an-existing-character), [macOS](../docs/setup-mac.md#step-6-bring-an-existing-character), [Linux](../docs/setup-linux.md#step-6-bring-an-existing-character)) for details.

## Web Interface

Don't want to use the terminal? The web UI covers everything:

```bash
pip install woven-imprint[ui]
woven-imprint ui
```

Opens a browser with 4 tabs: Chat, Characters (create/export/import/delete),
Migrate (from text or file), and Settings.

## What to Try

1. **Launch the web UI** — `woven-imprint ui` (easiest way to explore)
2. **Or start with `basic_usage.py`** — understand the Python API
3. **Migrate an existing character** — `woven-imprint migrate` with your own data
4. **Run `multi_character.py`** — see how characters perceive each other differently
5. **Try the CLI** — `woven-imprint demo` for an interactive terminal chat
6. **Connect your IDE** — follow `mcp_setup.md` to use characters in Claude or Cursor
