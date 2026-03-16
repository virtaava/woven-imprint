# Getting Started with Woven Imprint

## Installation

```bash
pip install woven-imprint
```

**Requirements:**
- Python 3.11+
- [Ollama](https://ollama.com) running locally (default LLM backend)

Pull the required models:
```bash
ollama pull qwen3-coder:30b    # or any chat model you prefer
ollama pull nomic-embed-text    # for memory embeddings
```

### Optional backends

```bash
pip install woven-imprint[openai]      # OpenAI / Azure / vLLM
pip install woven-imprint[anthropic]   # Anthropic Claude
pip install woven-imprint[mcp]         # MCP server for IDE integration
pip install woven-imprint[all]         # everything
```

---

## Quick Start (CLI)

The fastest way to see Woven Imprint in action:

```bash
# Launch the interactive demo — creates a character and starts chatting
woven-imprint demo

# Or with a different model
woven-imprint demo --model llama3.2
```

During the chat, type:
- `stats` — see memory counts, emotion, relationships
- `reflect` — have the character reflect on recent experiences
- `memories` — search the character's memories
- `quit` — end the session (generates a summary)

---

## Creating Characters

### From the CLI

```bash
woven-imprint create "Marcus the Blacksmith"
```

You'll be prompted for backstory, personality, speaking style, and birthdate.

### From Python

```python
from woven_imprint import Engine

engine = Engine("my_characters.db")

marcus = engine.create_character(
    name="Marcus",
    birthdate="1995-08-22",   # age auto-derived, increments on birthday
    persona={
        "backstory": "A seasoned blacksmith in a small village. Lost his wife two years ago.",
        "personality": "gruff but kind, protective, dry humor, uncomfortable with emotions",
        "speaking_style": "short sentences, working-class dialect, occasionally poetic",
        "occupation": "village blacksmith",
    },
)
```

### Persona structure

The `persona` dict supports four constraint levels:

```python
persona = {
    # Shorthand fields (auto-sorted into the right level)
    "backstory": "...",          # → hard constraint
    "personality": "...",        # → soft constraint
    "speaking_style": "...",     # → soft constraint
    "occupation": "...",         # → soft constraint

    # Or use explicit levels for fine control:
    "hard": {
        # NEVER change — immutable identity
        "name": "Marcus",
        "species": "human",
        "birthplace": "Ashford village",
    },
    "temporal": {
        # Change on schedule or by events — not through conversation
        "location": "the forge in Ashford",
        "title": "master blacksmith",
    },
    "soft": {
        # Evolve slowly through experience
        "personality": "gruff but kind",
        "opinion_of_strangers": "wary at first, warms up slowly",
        "habits": "works late, skips meals when focused",
    },
    # Emergent traits form automatically through interaction — don't define them
}
```

### Birthdate and age

```python
# Age is derived from birthdate — it increments automatically
char = engine.create_character("Alice", birthdate="1998-03-15", persona={...})

print(char.persona.age)             # 28 (as of 2026)
print(char.persona.is_birthday)     # True if today is March 15
print(char.persona.days_until_birthday)  # days until next birthday

# Characters without a birthdate can have a static age
char = engine.create_character("Old Sage", persona={"hard": {"age": 200}})
```

---

## Chatting

### Basic conversation

```python
response = marcus.chat("I need a sword forged by tomorrow.")
print(response)

# With user tracking (enables relationship development)
response = marcus.chat("It's urgent. Bandits are coming.", user_id="player_1")
```

### Sessions

Sessions group interactions and generate summaries when ended:

```python
marcus.start_session()                    # auto-started on first chat if needed
response = marcus.chat("Hello Marcus.")
response = marcus.chat("About that sword...")
summary = marcus.end_session()            # generates and stores a session summary
print(summary)

# Next session — Marcus remembers the previous one
response = marcus.chat("Any progress on my sword?")
```

---

## Memory

### How it works

Every interaction automatically:
1. Stores the exchange in **buffer** memory (raw conversation)
2. Extracts notable facts into **core** memory (every 3rd turn)
3. **Bedrock** memory holds the character's fundamental identity (seeded from persona)

### Searching memories

```python
memories = marcus.recall("sword commission", limit=5)
for m in memories:
    print(f"[{m['tier']}] {m['content'][:100]}")
```

### Reflection

Characters can reflect on accumulated experiences, generating higher-level insights:

```python
reflection = marcus.reflect()
print(reflection)
# "I've been taking on more work than I can handle. That stranger who
#  came in asking for a sword — there was something in their eyes..."
```

### Consolidation

When buffer memory grows large, consolidate it into denser core memories:

```python
stats = marcus.consolidate()
print(stats)  # {'clusters': 5, 'summarized': 47, 'created': 5, 'archived': 47}
```

### Belief revision

Characters can change their mind — old beliefs are preserved, not deleted:

```python
# Marcus believed the stranger was trustworthy
mem = marcus.memory.add("The stranger seems honest and trustworthy", tier="core")

# Later, he learns the truth
marcus.belief.contradict(mem["id"], "The stranger lied about the bandits — they were a thief")

# Old memory is marked 'contradicted', new one takes over in retrieval
# Marcus remembers he USED to trust them — characters can reference their own growth
```

---

## Relationships

Every interaction with a `user_id` automatically tracks the relationship across five dimensions:

| Dimension | Range | Meaning |
|-----------|-------|---------|
| trust | -1 to 1 | suspicion ↔ trust |
| affection | -1 to 1 | dislike ↔ warmth |
| respect | -1 to 1 | contempt ↔ admiration |
| familiarity | 0 to 1 | stranger → intimate knowledge |
| tension | 0 to 1 | calm → explosive |

```python
# Relationships update automatically during chat
marcus.chat("Thank you for saving my village.", user_id="player_1")

# Check the relationship
print(marcus.relationships.describe("player_1"))
# "Relationship with player_1 (stranger):
#   trust: neutral (0.08)
#   affection: neutral (0.05)
#   respect: neutral (0.06)
#   familiarity: acquaintances (0.12)
#   tension: calm (0.00)
#   trajectory: warming"

# Get raw dimensions
rel = marcus.relationships.get("player_1")
print(rel["dimensions"])  # {'trust': 0.08, 'affection': 0.05, ...}
```

Changes are bounded to ±0.15 per interaction — relationships develop gradually.
Familiarity only increases (you can't un-know someone).

---

## Emotional State

Characters have persistent mood that affects their responses:

```python
print(marcus.emotion.mood)       # "neutral"
print(marcus.emotion.intensity)  # 0.5

# After a dramatic conversation...
marcus.chat("Your wife... she would have wanted you to protect this place.", user_id="player_1")
print(marcus.emotion.mood)       # "melancholic"
print(marcus.emotion.intensity)  # 0.7

# Emotions naturally decay toward neutral over time
```

Available moods: joyful, content, excited, anxious, angry, sad, fearful,
disgusted, surprised, neutral, contemplative, melancholic, determined,
vulnerable, amused.

---

## Character Growth

After enough interactions, characters can evolve:

```python
# Detect and apply growth (needs 20+ core memories)
events = marcus.evolve()
for e in events:
    print(f"{e['trait']}: '{e['old_value']}' → '{e['new_value']}'")
    print(f"  Reason: {e['reason']}")
# personality: 'gruff but kind' → 'gruff but warming to trusted friends'
#   Reason: Repeated positive interactions with the player built confidence
```

Only soft constraints change. Hard facts never shift.

---

## Multi-Character Interaction

Two characters can talk to each other:

```python
from woven_imprint import Engine, interact

engine = Engine("tavern.db")
greta = engine.create_character("Greta", persona={...})
cael = engine.create_character("Cael", persona={...})

result = interact(
    greta, cael,
    situation="A dusty stranger enters the tavern at dusk.",
    rounds=3,         # 3 back-and-forth exchanges
    a_opens=True,     # Greta speaks first
)

for turn in result.turns:
    print(f"{turn.speaker}: {turn.response[:200]}")
```

Group scenes:

```python
from woven_imprint import group_interaction

results = group_interaction(
    [greta, cael, marcus],
    situation="A town meeting about the approaching bandits.",
    rounds=2,
)
```

---

## Export and Import

### Export a character

```python
# To dict
data = marcus.export()

# To file
marcus.export("marcus_backup.json")
```

The export includes everything: persona, all memories (buffer/core/bedrock),
relationships, emotional state, narrative arc, sessions. Embeddings are stripped
(recomputed on import).

### Import a character

```python
marcus = engine.import_character("marcus_backup.json")
# All memories are re-embedded on import
```

### Import persona from a markdown file

If you have a character defined in a markdown or text file, parse it into
a persona dict and create the character:

```python
from pathlib import Path

# Your character file (any format you like)
text = Path("marcus.md").read_text()

# Parse it however makes sense for your format
engine = Engine("characters.db")
marcus = engine.create_character(
    name="Marcus",
    birthdate="1995-08-22",
    persona={
        "backstory": text,  # or parse out specific sections
        "personality": "gruff but kind",
        "speaking_style": "short sentences, working-class dialect",
    },
)
```

---

## Using Different LLM Backends

### Ollama (default, local)

```python
from woven_imprint import Engine
from woven_imprint.llm.ollama import OllamaLLM
from woven_imprint.embedding.ollama import OllamaEmbedding

engine = Engine(
    db_path="characters.db",
    llm=OllamaLLM(model="llama3.2", num_ctx=8192),
    embedding=OllamaEmbedding(model="nomic-embed-text"),
)
```

### OpenAI

```python
from woven_imprint import Engine
from woven_imprint.llm import OpenAILLM
from woven_imprint.embedding import OpenAIEmbedding

engine = Engine(
    db_path="characters.db",
    llm=OpenAILLM(model="gpt-4o-mini"),
    embedding=OpenAIEmbedding(model="text-embedding-3-small"),
)
```

### Anthropic Claude

```python
from woven_imprint import Engine
from woven_imprint.llm import AnthropicLLM
from woven_imprint.embedding.ollama import OllamaEmbedding  # Claude has no embedding API

engine = Engine(
    db_path="characters.db",
    llm=AnthropicLLM(model="claude-sonnet-4-6"),
    embedding=OllamaEmbedding(),  # or OpenAIEmbedding()
)
```

### Any OpenAI-compatible server (vLLM, llama.cpp, LiteLLM)

```python
from woven_imprint.llm import OpenAILLM

engine = Engine(
    llm=OpenAILLM(model="my-model", base_url="http://localhost:8000/v1", api_key="not-needed"),
    embedding=OllamaEmbedding(),
)
```

---

## Integration with AI Systems

### MCP Server (Claude Desktop, Cursor, Hermes Agent)

Woven Imprint exposes all character operations as MCP tools.
See [examples/mcp_setup.md](../examples/mcp_setup.md) for configuration.

Tools available: `list_characters`, `create_character`, `chat`, `recall`,
`get_relationship`, `reflect`, `evolve`, `end_session`.

### OpenAI-Compatible API Server

Run characters as an OpenAI-compatible HTTP endpoint:

```bash
woven-imprint serve --port 8650
```

Any system that speaks OpenAI API can now use persistent characters:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8650/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="marcus",  # character name = model name
    messages=[{"role": "user", "content": "I need a sword."}],
)
```

---

## CLI Reference

```bash
woven-imprint demo                    # Interactive demo with pre-built character
woven-imprint create "Name"           # Create a new character
woven-imprint chat <name-or-id>       # Chat with existing character
woven-imprint list                    # List all characters
woven-imprint stats <name-or-id>      # Memory counts, relationships, emotion
woven-imprint export <name-or-id>     # Export to JSON
woven-imprint serve --port 8650       # Start OpenAI-compatible API server

# Global options
--db <path>          # Database path (default: ~/.woven_imprint/characters.db)
--model <name>       # Ollama model (default: qwen3-coder:30b)
```
