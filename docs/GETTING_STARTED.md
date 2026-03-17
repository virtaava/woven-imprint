# Getting Started with Woven Imprint

## Step 1: Install Python

You need Python 3.11 or newer. Check if you already have it:

**Windows**: Press `Win+R`, type `powershell`, press Enter. Then type:
```
python --version
```

**Mac**: Open Terminal (search for "Terminal" in Spotlight). Then type:
```
python3 --version
```

**Linux**: Open a terminal and type:
```
python3 --version
```

If you see `Python 3.11` or higher, skip to Step 2. Otherwise:

- **Windows**: Download from [python.org/downloads](https://www.python.org/downloads/).
  During installation, **check the box "Add Python to PATH"** — this is important.
- **Mac**: `brew install python@3.12` (install [Homebrew](https://brew.sh) first if needed)
- **Linux**: `sudo apt install python3 python3-pip python3-venv`

## Step 2: Install Woven Imprint

Open your terminal (PowerShell on Windows, Terminal on Mac/Linux) and run:

```
pip install woven-imprint
```

That's it. If you see "Successfully installed", you're ready.

<details>
<summary>Troubleshooting: "pip is not recognized"</summary>

Try `python -m pip install woven-imprint` or `python3 -m pip install woven-imprint`.
On Windows, you may need to close and reopen PowerShell after installing Python.
</details>

<details>
<summary>Alternative: install from source (for developers)</summary>

```
git clone https://github.com/virtaava/woven-imprint.git
cd woven-imprint
pip install -e .
```

This requires [Git](https://git-scm.com/downloads) to be installed.
</details>

## Step 3: Set Up an LLM

Woven Imprint needs a language model to power the characters. Pick one:

### Option A: Ollama (free, runs on your computer)

1. Download and install Ollama from [ollama.com](https://ollama.com)
2. Open a terminal and run these two commands (each downloads a model — the first
   is about 2 GB, the second about 300 MB, so this may take a few minutes):

```
ollama pull llama3.2
ollama pull nomic-embed-text
```

3. That's it. Woven Imprint uses Ollama automatically.

### Option B: OpenAI (requires API key, costs money per use)

1. Get an API key from [platform.openai.com](https://platform.openai.com)
2. Install the OpenAI extra: `pip install woven-imprint[openai]`
3. Set your API key:
   - **Windows PowerShell**: `$env:OPENAI_API_KEY = "sk-your-key-here"`
   - **Mac/Linux**: `export OPENAI_API_KEY=sk-your-key-here`

### Option C: Anthropic Claude (requires API key)

1. Get an API key from [console.anthropic.com](https://console.anthropic.com)
2. Install: `pip install woven-imprint[anthropic]`
3. Set your key:
   - **Windows PowerShell**: `$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"`
   - **Mac/Linux**: `export ANTHROPIC_API_KEY=sk-ant-your-key-here`

## Step 4: Try It

Run the interactive demo:

```
woven-imprint demo
```

You should see a character named Alice Blackwood respond to you. Type messages
and press Enter to chat. Type `quit` when you're done.

If something went wrong, check:
- Is Ollama running? (You should see the Ollama icon in your system tray)
- Did you pull the models? (`ollama pull llama3.2` and `ollama pull nomic-embed-text`)
- On Windows, try closing and reopening PowerShell

---

## What You Can Do

### Chat from the command line

```
woven-imprint demo                        # Quick demo with a pre-built character
woven-imprint create "Marcus"             # Create your own character
woven-imprint chat marcus                 # Chat with an existing character
woven-imprint list                        # See all your characters
woven-imprint stats marcus                # Check memory, relationships, emotion
```

### Migrate an existing character

Already have a character in ChatGPT, SillyTavern, or a Custom GPT? Bring it over:

```
woven-imprint migrate conversations.json           # ChatGPT data export
woven-imprint migrate character_card.png            # SillyTavern card
woven-imprint migrate --text "You are Marcus..."    # Custom GPT instructions
```

See [Migrate from Other Systems](#migrate-from-other-systems) below for details.

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

---

## Migrate from Other Systems

Already have a character in ChatGPT, SillyTavern, Claude, or a text file?
Bring it into Woven Imprint with one command.

### From ChatGPT

Export your data from ChatGPT (Settings → Data Controls → Export Data),
then use the `conversations.json` file:

```bash
woven-imprint migrate conversations.json
```

The system analyzes your conversation history to extract:
- Character persona (personality, backstory, speaking style)
- Key memories from the conversations
- Relationship baseline (trust, affection, familiarity based on how you interacted)
- Current emotional state (from the most recent messages)

### From SillyTavern / TavernAI

Character cards (JSON or PNG with embedded data) work directly:

```bash
woven-imprint migrate character_card.json
woven-imprint migrate character_card.png
```

### From Custom GPT Instructions

Copy your GPT's instructions and paste them:

```bash
woven-imprint migrate --text "You are Coach Rivera, a retired soccer coach..."
```

Or save them to a file first:

```bash
woven-imprint migrate my_gpt_instructions.txt
```

### From a Claude Project

Point to the project directory (reads CLAUDE.md and memory files):

```bash
woven-imprint migrate /path/to/project/
```

### From Any Text or Markdown

Any persona description, character sheet, or backstory document:

```bash
woven-imprint migrate persona.md
woven-imprint migrate character_sheet.txt
```

### Override the Name

If auto-detection gets the name wrong:

```bash
woven-imprint migrate conversations.json --name "Marcus"
```

### From Python

```python
from woven_imprint import Engine
from woven_imprint.migrate import CharacterImporter

engine = Engine("characters.db")
importer = CharacterImporter(engine)

# From any file (auto-detects format)
character = importer.from_file("conversations.json")

# From text
character = importer.from_text("You are Marcus, a gruff blacksmith...")

# Check what was imported
print(character.name)
print(character.persona.soft)
print(character.relationships.describe("imported_user"))
print(character.emotion.mood)
```

### Via MCP (Claude Desktop, Cursor)

```
"Migrate this character: You are a seasoned detective named Alice who
 speaks in clipped sentences and has a dark sense of humor..."
```

The `migrate_from_text` tool extracts persona, personality, and memories
automatically.

---

## CLI Reference

```bash
woven-imprint demo                        # Interactive demo with pre-built character
woven-imprint create "Name"               # Create a new character
woven-imprint chat <name-or-id>           # Chat with existing character
woven-imprint list                        # List all characters
woven-imprint stats <name-or-id>          # Memory counts, relationships, emotion
woven-imprint export <name-or-id>         # Export to JSON
woven-imprint delete <name-or-id>         # Delete a character
woven-imprint import <path>               # Import from Woven Imprint JSON export
woven-imprint migrate <path>              # Migrate from ChatGPT, SillyTavern, etc.
woven-imprint migrate --text "..."        # Migrate from pasted text
woven-imprint serve --port 8650           # Start OpenAI-compatible API server

# Global options
--db <path>          # Database path (default: ~/.woven_imprint/characters.db)
--model <name>       # Ollama model (env: WOVEN_IMPRINT_MODEL, default: llama3.2)
```
