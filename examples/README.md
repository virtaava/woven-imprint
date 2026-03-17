# Examples

## Prerequisites

All Python examples need Woven Imprint installed and an LLM backend running:

```bash
pip install woven-imprint
ollama pull llama3.2            # or any chat model
ollama pull nomic-embed-text    # for memory embeddings
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

No code needed — just config. Characters are available as MCP tools:
`chat`, `recall`, `get_relationship`, `reflect`, `evolve`, `end_session`.

## What to Try

1. **Start with `basic_usage.py`** — understand the core API
2. **Run `multi_character.py`** — see how characters perceive each other differently
3. **Try the CLI** — `woven-imprint demo` for an interactive chat with live feedback
4. **Connect your IDE** — follow `mcp_setup.md` to use characters in Claude or Cursor
