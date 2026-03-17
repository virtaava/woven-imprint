# Getting Started with Woven Imprint

## Choose Your Path

**I'm new to this** — I want to create AI characters that remember me. I may not
have used Python or a terminal before.
→ Start at [Beginner Guide](#beginner-guide)

**I'm a developer** — I know Python, LLMs, and maybe MCP. I want the quick setup.
→ Jump to [Developer Guide](#developer-guide)

---

# Beginner Guide

## What You'll Need

- A computer (Windows, Mac, or Linux)
- About 15 minutes for setup
- About 3 GB of free disk space (for the AI model)

## Step 1: Install Python

Python is the programming language Woven Imprint runs on. You may already have it.

<details open>
<summary><strong>Windows</strong></summary>

1. Press `Win+R`, type `powershell`, press Enter
2. Type `python --version` and press Enter
3. If you see `Python 3.11` or higher, skip to Step 2
4. If not, go to [python.org/downloads](https://www.python.org/downloads/) and download the installer
5. **Important**: During installation, check the box that says **"Add Python to PATH"**
6. After installation, close PowerShell and open it again
7. Type `python --version` to confirm it works

</details>

<details>
<summary><strong>Mac</strong></summary>

1. Open Terminal (press `Cmd+Space`, type "Terminal", press Enter)
2. Type `python3 --version` and press Enter
3. If you see `Python 3.11` or higher, skip to Step 2
4. If not, install [Homebrew](https://brew.sh) first (paste the command from their site into Terminal)
5. Then run: `brew install python@3.12`
6. Verify: `python3 --version`

> On Mac, use `python3` and `pip3` instead of `python` and `pip` in all commands below.

</details>

<details>
<summary><strong>Linux</strong></summary>

1. Open a terminal
2. Type `python3 --version`
3. If you see 3.11 or higher, skip to Step 2
4. Ubuntu/Debian: `sudo apt update && sudo apt install -y python3 python3-pip`
5. Fedora: `sudo dnf install python3 python3-pip`

> On Linux, use `python3` and `pip3` instead of `python` and `pip` in all commands below.

</details>

## Step 2: Install Woven Imprint

Open your terminal (PowerShell on Windows, Terminal on Mac/Linux) and type:

```
pip install woven-imprint
```

You should see "Successfully installed woven-imprint" after a few seconds.

<details>
<summary>It says "pip is not recognized" or "command not found"</summary>

Try one of these instead:
```
python -m pip install woven-imprint
python3 -m pip install woven-imprint
```
On Windows, close and reopen PowerShell after installing Python.
</details>

<details>
<summary>It says "externally-managed-environment" (Linux / WSL / Ubuntu 24.04+)</summary>

Modern Ubuntu/Debian prevents installing packages into the system Python.
Create a virtual environment first:

```
python3 -m venv ~/woven-imprint-env
source ~/woven-imprint-env/bin/activate
pip install woven-imprint
```

You'll need to run `source ~/woven-imprint-env/bin/activate` each time you
open a new terminal before using `woven-imprint`. To make it automatic,
add it to your shell profile:

```
echo 'source ~/woven-imprint-env/bin/activate' >> ~/.bashrc
```

Alternatively, use `pipx` which manages the virtual environment for you:

```
sudo apt install pipx
pipx install woven-imprint
```
</details>

## Step 3: Install an AI Model

Your characters need an AI brain. The easiest free option is Ollama, which
runs models on your own computer.

<details open>
<summary><strong>Windows</strong></summary>

1. Go to [ollama.com](https://ollama.com) and click Download
2. Run the installer
3. After installation, you should see the Ollama icon in your system tray (bottom-right of the taskbar)
4. Open PowerShell and run these two commands:

```
ollama pull llama3.2
ollama pull nomic-embed-text
```

The first download is about 2 GB, the second about 300 MB. This will take a
few minutes depending on your internet speed. You'll see a progress bar.

</details>

<details>
<summary><strong>Mac</strong></summary>

1. Go to [ollama.com](https://ollama.com) and click Download
2. Open the downloaded file and drag Ollama to your Applications folder
3. Open Ollama from Applications — it will appear in your menu bar
4. Open Terminal and run:

```
ollama pull llama3.2
ollama pull nomic-embed-text
```

Wait for both downloads to complete (about 2.3 GB total).

</details>

<details>
<summary><strong>Linux</strong></summary>

1. Run: `curl -fsSL https://ollama.com/install.sh | sh`
2. Then pull the models:

```
ollama pull llama3.2
ollama pull nomic-embed-text
```

</details>

<details>
<summary>I don't want to install Ollama / I want to use ChatGPT's API instead</summary>

You can use OpenAI's API instead (requires an API key, costs money per use):

1. Get an API key from [platform.openai.com](https://platform.openai.com)
2. Run: `pip install woven-imprint[openai]`
3. Set your key:
   - **Windows**: `$env:OPENAI_API_KEY = "sk-your-key-here"`
   - **Mac/Linux**: `export OPENAI_API_KEY=sk-your-key-here`

Then add `--model openai` when running commands, or see the [Developer Guide](#option-b-openai-api)
for Python configuration.

</details>

## Step 4: Try It

### Option 1: Web interface (easiest)

```
pip install woven-imprint[ui]
woven-imprint ui
```

<details>
<summary>If you installed with pipx</summary>

Extras need to be injected separately with pipx:

```
pipx inject woven-imprint gradio
woven-imprint ui
```
</details>

<details>
<summary>If you get "externally-managed-environment"</summary>

Use the virtual environment you created in Step 2:

```
source ~/woven-imprint-env/bin/activate
pip install woven-imprint[ui]
woven-imprint ui
```
</details>

This opens a browser tab where you can create characters, chat, migrate
existing characters, and see stats — all without touching the terminal again.

### Option 2: Terminal

```
woven-imprint demo
```

You should see a character named Alice Blackwood introduce herself. Type a
message and press Enter to chat with her. Try asking her about a case or
telling her something personal — she'll remember it.

During the chat, you can use these commands (everything else is sent to the character):
- **/help** — list all commands
- **/stats** — see her memory, emotions, and how she feels about you
- **/reflect** — she thinks about what's happened so far
- **/memories** — search what she remembers (you'll be asked for a search term)
- **/recall sword** — search memories for "sword" directly
- **/quit** — end the conversation (she'll write a summary of your session)

<details>
<summary>It's not working</summary>

- **"command not found"**: close and reopen your terminal, then try again
- **Connection error**: make sure Ollama is running (check your system tray/menu bar)
- **"model not found"**: run `ollama pull llama3.2` and `ollama pull nomic-embed-text`
- **Very slow responses**: the first response may take 30-60 seconds while the model loads

</details>

## Step 5: Create Your Own Character

Now make your own character:

```
woven-imprint create "Marcus the Blacksmith"
```

You'll be asked a few questions:
- **Backstory**: who is this character? (e.g., "A gruff blacksmith who lost his wife two years ago")
- **Personality**: what are they like? (e.g., "kind but grumpy, dry humor")
- **Speaking style**: how do they talk? (e.g., "short sentences, working-class accent")
- **Birthdate**: optional, format YYYY-MM-DD (the character will age and know their birthday)

Press Enter to skip any question you don't want to fill in.

Then start chatting:

```
woven-imprint chat marcus
```

To see all your characters:

```
woven-imprint list
```

## Step 6: Bring an Existing Character

Already have a character somewhere else? You can import it.

### From ChatGPT

If you've been chatting with a character in ChatGPT:

1. Go to ChatGPT → Settings → Data Controls → Export Data
2. Wait for the email, download the zip file
3. Unzip it and find `conversations.json`
4. Run:

```
woven-imprint migrate conversations.json
```

The system will read your conversation history and create a character with:
- The personality it detects from how the AI responded
- Key memories from your conversations
- A relationship baseline (if you chatted a lot, familiarity will be high)
- The emotional tone from your most recent messages

### From a Custom GPT

Custom GPTs have two parts: instructions and knowledge files. You can import both.

**Instructions only:**

1. Go to your GPT → Configure → copy the Instructions text
2. Paste directly or save to a file:

```
woven-imprint migrate --text "You are Coach Rivera, a retired soccer coach..."
woven-imprint migrate my_gpt_instructions.txt
```

**Instructions + knowledge files:**

If your GPT has uploaded files (PDFs, text files, spreadsheets):

1. Copy the Instructions and save to a file (e.g., `instructions.txt`)
2. Download the knowledge files from your GPT
3. Include them with `--knowledge`:

```
woven-imprint migrate instructions.txt --knowledge product_manual.pdf faq.txt pricing.csv
```

The knowledge files are analyzed and key facts are stored as the character's
memories. The character will reference this knowledge in conversations.

### From SillyTavern / TavernAI

If you have a character card (JSON or PNG):

```
woven-imprint migrate character_card.json
woven-imprint migrate character_card.png
```

### From Any Text File

Any character description, backstory, or personality document:

```
woven-imprint migrate my_character.md
woven-imprint migrate character_sheet.txt
```

### Fix the Name

If the system guesses the wrong name:

```
woven-imprint migrate conversations.json --name "Marcus"
```

## What's Next?

- **Keep chatting** — your character remembers everything across sessions
- **Check the relationship** — `woven-imprint stats marcus` shows how your character feels about you
- **Create more characters** — they can even talk to each other (see [Developer Guide](#multi-character-interaction))
- **Connect to Claude or Cursor** — see [MCP Setup](../examples/mcp_setup.md)

---

# Developer Guide

Quick setup for those who know their way around Python and LLMs.

## Install

```bash
pip install woven-imprint           # core (requires only `requests`)
pip install woven-imprint[openai]    # + OpenAI/Azure/vLLM backend
pip install woven-imprint[anthropic] # + Anthropic Claude backend
pip install woven-imprint[mcp]       # + MCP server for IDE integration
pip install woven-imprint[all]       # everything
```

<details>
<summary>Install from source</summary>

```bash
git clone https://github.com/virtaava/woven-imprint.git
cd woven-imprint
pip install -e ".[dev]"
```
</details>

## LLM Backends

### Option A: Ollama (default)

```bash
ollama pull llama3.2 && ollama pull nomic-embed-text
```

Works out of the box. Override the model:

```python
from woven_imprint import Engine
from woven_imprint.llm.ollama import OllamaLLM
from woven_imprint.embedding.ollama import OllamaEmbedding

engine = Engine(
    llm=OllamaLLM(model="qwen3-coder:30b", num_ctx=8192),
    embedding=OllamaEmbedding(model="nomic-embed-text"),
)
```

### Option B: OpenAI API

```bash
pip install woven-imprint[openai]
export OPENAI_API_KEY=sk-...          # Mac/Linux
$env:OPENAI_API_KEY = "sk-..."       # Windows PowerShell
```

```python
from woven_imprint import Engine
from woven_imprint.llm import OpenAILLM
from woven_imprint.embedding import OpenAIEmbedding

engine = Engine(
    llm=OpenAILLM(model="gpt-4o-mini"),
    embedding=OpenAIEmbedding(model="text-embedding-3-small"),
)
```

### Option C: Anthropic Claude

```bash
pip install woven-imprint[anthropic]
export ANTHROPIC_API_KEY=sk-ant-...   # Mac/Linux
$env:ANTHROPIC_API_KEY = "sk-ant-..." # Windows PowerShell
```

```python
from woven_imprint import Engine
from woven_imprint.llm import AnthropicLLM
from woven_imprint.embedding.ollama import OllamaEmbedding  # Claude has no embedding API

engine = Engine(
    llm=AnthropicLLM(model="claude-sonnet-4-6"),
    embedding=OllamaEmbedding(),  # or OpenAIEmbedding()
)
```

### Option D: Any OpenAI-compatible endpoint (vLLM, llama.cpp, LiteLLM)

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

# Group scene
results = group_interaction([greta, cael, marcus], situation="Town meeting.", rounds=2)
```

## Migration from Other Systems

```python
from woven_imprint.migrate import CharacterImporter

importer = CharacterImporter(engine)

char = importer.from_file("conversations.json")    # ChatGPT export
char = importer.from_file("card.png")               # SillyTavern card
char = importer.from_file("/path/to/claude/project") # Claude project
char = importer.from_text("You are Marcus...")        # any text

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

## Configuration

| Setting | Default | Env Var | CLI Flag |
|---------|---------|---------|----------|
| Database path | `~/.woven_imprint/characters.db` | — | `--db` |
| Ollama model | `llama3.2` | `WOVEN_IMPRINT_MODEL` | `--model` |
| Lightweight mode | off | — | `character.lightweight = True` |

## CLI Reference

```bash
woven-imprint demo                        # Interactive demo
woven-imprint create "Name"               # Create character
woven-imprint chat <name-or-id>           # Chat
woven-imprint list                        # List characters
woven-imprint stats <name-or-id>          # Character info
woven-imprint export <name-or-id>         # Export to JSON
woven-imprint delete <name-or-id>         # Delete
woven-imprint import <path>               # Import Woven Imprint JSON
woven-imprint migrate <path>              # Migrate from other systems
woven-imprint migrate --text "..."        # Migrate from text
woven-imprint ui                          # Web interface (pip install woven-imprint[ui])
woven-imprint serve --port 8650           # OpenAI-compatible API
```
