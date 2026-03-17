# Developer Guide

For Python developers, LLM practitioners, and integrators.

## Install

```bash
pip install woven-imprint           # core (requires only `requests`)
pip install woven-imprint[openai]    # + OpenAI/Azure/vLLM backend
pip install woven-imprint[anthropic] # + Anthropic Claude backend
pip install woven-imprint[mcp]       # + MCP server for IDE integration
pip install woven-imprint[pdf]       # + PDF knowledge file extraction
pip install woven-imprint[ui]        # + Gradio web interface
pip install woven-imprint[all]       # everything
```

From source:
```bash
git clone https://github.com/virtaava/woven-imprint.git
cd woven-imprint
pip install -e ".[dev]"
```

## LLM Backends

### Ollama (default)

```bash
ollama pull llama3.2 && ollama pull nomic-embed-text
```

```python
from woven_imprint import Engine
from woven_imprint.llm.ollama import OllamaLLM
from woven_imprint.embedding.ollama import OllamaEmbedding

engine = Engine(
    llm=OllamaLLM(model="llama3.2", num_ctx=8192),
    embedding=OllamaEmbedding(model="nomic-embed-text"),
)
```

### OpenAI

```python
from woven_imprint.llm import OpenAILLM
from woven_imprint.embedding import OpenAIEmbedding

engine = Engine(
    llm=OpenAILLM(model="gpt-4o-mini"),
    embedding=OpenAIEmbedding(model="text-embedding-3-small"),
)
```

Environment variables:
- **Mac/Linux**: `export OPENAI_API_KEY=sk-...`
- **Windows**: `$env:OPENAI_API_KEY = "sk-..."`

### Anthropic Claude

```python
from woven_imprint.llm import AnthropicLLM

engine = Engine(
    llm=AnthropicLLM(model="claude-sonnet-4-6"),
    embedding=OllamaEmbedding(),  # Claude has no embedding API
)
```

### Any OpenAI-compatible endpoint (vLLM, llama.cpp, LiteLLM)

```python
engine = Engine(
    llm=OpenAILLM(model="my-model", base_url="http://localhost:8000/v1", api_key="not-needed"),
    embedding=OllamaEmbedding(),
)
```

## Python API

```python
from woven_imprint import Engine

with Engine("characters.db") as engine:
    # Create
    char = engine.create_character(
        name="Marcus",
        birthdate="1995-08-22",
        persona={
            "backstory": "A blacksmith who lost his wife two years ago.",
            "personality": "gruff but kind, dry humor",
            "speaking_style": "short sentences, working-class dialect",
        },
    )

    # Chat (memories and relationships update automatically)
    response = char.chat("I need a sword.", user_id="player_1")

    # Inspect
    print(char.emotion.mood)
    print(char.relationships.describe("player_1"))
    print(char.recall("sword", limit=3))

    # Lifecycle
    char.reflect()                    # generate inner reflection
    char.consolidate()                # compress buffer → core memories
    char.evolve()                     # detect personality growth
    char.end_session()                # summarize and persist state
    char.export("marcus.json")        # full portable export
```

## Persona Structure

```python
persona = {
    # Shorthand (auto-classified)
    "backstory": "...",              # → hard constraint
    "personality": "...",            # → soft constraint
    "speaking_style": "...",         # → soft constraint

    # Explicit levels
    "hard": {"name": "Marcus", "species": "human"},         # never changes
    "temporal": {"location": "the forge"},                    # changes by event
    "soft": {"opinion_of_strangers": "wary at first"},       # evolves through experience
    # emergent traits form automatically through interaction
}
```

## Multi-Character Interaction

```python
from woven_imprint import Engine, interact, group_interaction

engine = Engine("world.db")
greta = engine.create_character("Greta", persona={...})
cael = engine.create_character("Cael", persona={...})

# Two characters talk
result = interact(greta, cael, situation="A stranger enters the tavern.", rounds=3)
for turn in result.turns:
    print(f"{turn.speaker}: {turn.response[:200]}")

# Group scene
results = group_interaction([greta, cael], situation="Town meeting.", rounds=2)
```

## Migration from Other Systems

```python
from woven_imprint.migrate import CharacterImporter

importer = CharacterImporter(engine)

char = importer.from_file("conversations.json")     # ChatGPT export
char = importer.from_file("card.png")                # SillyTavern card
char = importer.from_file("/path/to/claude/project") # Claude project
char = importer.from_text("You are Marcus...")        # any text

# With Custom GPT knowledge files
char = importer.from_custom_gpt(
    instructions="You are a product expert...",
    knowledge_files=["manual.pdf", "faq.txt"],
)

# Relationship baseline is auto-calculated from conversation history
print(char.relationships.describe("imported_user"))
```

## Integration

### MCP Server (Claude Desktop, Cursor, Hermes, OpenClaw)

See [MCP Setup](../examples/mcp_setup.md) for config. 13 tools available:
`list_characters`, `create_character`, `chat`, `recall`, `get_relationship`,
`reflect`, `evolve`, `new_session`, `end_session`, `consolidate`, `get_stats`,
`delete_character`, `migrate_from_text`.

### OpenAI-Compatible API Proxy

```bash
woven-imprint serve --port 8650
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8650/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="marcus",  # character name = model name
    messages=[{"role": "user", "content": "I need a sword."}],
)
```

### Web UI

```bash
woven-imprint ui                      # auto-detect browser
woven-imprint ui --browser chrome     # specific browser
woven-imprint ui --browser none       # don't open, just print URL
```

## Configuration

| Setting | Default | Env Var | CLI Flag |
|---------|---------|---------|----------|
| Database path | `~/.woven_imprint/characters.db` | — | `--db` |
| Ollama model | `llama3.2` | `WOVEN_IMPRINT_MODEL` | `--model` |
| MCP model | `llama3.2` | `WOVEN_IMPRINT_MODEL` | — |
| Lightweight mode | off | — | `character.lightweight = True` |

## CLI Reference

```bash
# Getting started
woven-imprint demo                        # Interactive demo
woven-imprint ui                          # Web interface

# Character management
woven-imprint create "Name"               # Create character
woven-imprint chat <name-or-id>           # Chat
woven-imprint list                        # List characters
woven-imprint stats <name-or-id>          # Character info
woven-imprint export <name-or-id>         # Export to JSON
woven-imprint import <path>               # Import from JSON
woven-imprint delete <name-or-id>         # Delete

# Migration
woven-imprint migrate <path>              # Auto-detect format
woven-imprint migrate --text "..."        # From text
woven-imprint migrate <path> -k f1 f2    # With knowledge files

# Server
woven-imprint serve --port 8650           # OpenAI-compatible API
woven-imprint ui --port 7860              # Web interface
woven-imprint ui --browser chrome         # Specify browser

# Maintenance
woven-imprint update                      # Update to latest version
woven-imprint --version                   # Show version
```

### Chat Slash Commands

During `woven-imprint chat` or `demo` — everything without `/` goes to the character:

```
/help       — list commands
/stats      — memory, emotion, relationships
/reflect    — character inner reflection
/memories   — search memories (prompts for term)
/recall X   — search memories for "X"
/quit       — end session and exit
```
