# Setup Guide — Linux

Covers Ubuntu, Debian, Fedora, and other distributions.

## What You'll Need

- A Linux desktop or server
- About 15 minutes
- About 3 GB of free disk space (for the AI model)

## Step 1: Install Python

1. Open a terminal
2. Check: `python3 --version`
3. If you see 3.11 or higher, skip to Step 2

**Ubuntu / Debian**:
```
sudo apt update && sudo apt install -y python3 python3-pip python3-venv
```

**Fedora**:
```
sudo dnf install python3 python3-pip
```

> Use `python3` and `pip3` in all commands below.

## Step 2: Install Woven Imprint

### Recommended: use pipx (avoids "externally-managed-environment" errors)

```bash
sudo apt install pipx    # Ubuntu/Debian
pipx install woven-imprint
pipx inject woven-imprint fastapi uvicorn    # for the demo UI
```

### Alternative: virtual environment

```bash
python3 -m venv ~/woven-imprint-env
source ~/woven-imprint-env/bin/activate
pip install woven-imprint[demo]
```

You'll need to run `source ~/woven-imprint-env/bin/activate` each time you
open a new terminal. Make it automatic:
```bash
echo 'source ~/woven-imprint-env/bin/activate' >> ~/.bashrc
```

### Upgrading from a previous version

**pipx** (clean reinstall removes old gradio dependencies):
```bash
pipx uninstall woven-imprint
pipx install woven-imprint
pipx inject woven-imprint fastapi uvicorn
```

**venv**:
```bash
source ~/woven-imprint-env/bin/activate
pip install --upgrade woven-imprint[demo]
pip uninstall gradio -y    # remove old UI if present
```

Your character data (`~/.woven_imprint/characters.db`) and config carry over automatically — no migration needed.

## Step 3: Install an AI Model

```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama pull nomic-embed-text
```

About 2.3 GB total. You'll see a progress bar.

### Don't want to install Ollama?

Use OpenAI's API instead:

**pipx**:
```
pipx inject woven-imprint openai
export WOVEN_IMPRINT_LLM_PROVIDER=openai
export WOVEN_IMPRINT_EMBEDDING_PROVIDER=openai
export WOVEN_IMPRINT_API_KEY_LLM=sk-your-key-here
export WOVEN_IMPRINT_MODEL=gpt-4o-mini
```

**venv**:
```
pip install woven-imprint[openai]
export WOVEN_IMPRINT_LLM_PROVIDER=openai
export WOVEN_IMPRINT_EMBEDDING_PROVIDER=openai
export WOVEN_IMPRINT_API_KEY_LLM=sk-your-key-here
export WOVEN_IMPRINT_MODEL=gpt-4o-mini
```

Or set it permanently in `~/.woven_imprint/config.yaml` — see [Configuration](CONFIGURATION.md#openai-backend-no-local-ollama-needed).

## Step 4: Try It

### React demo UI (recommended)

```bash
woven-imprint demo
```

Opens your default browser. To skip opening and just print the URL:
```
woven-imprint demo --no-browser
```

### Terminal REPL

```
woven-imprint chat alice
```

Commands during chat (everything without / goes to the character):
- **/help** — list all commands
- **/stats** — memory, emotions, relationships
- **/reflect** — character reflects on experiences
- **/memories** — search memories
- **/quit** — end session

### Not working?

- **Connection error**: check if Ollama is running: `systemctl status ollama`
- **"model not found"**: run `ollama pull llama3.2`
- **Very slow**: first response can take 30-60 seconds while the model loads

## Step 5: Create Your Own Character

### From the web UI

Click the **Characters** tab → fill in the form → click **Create Character**.

### From the terminal

```
woven-imprint create "Marcus the Blacksmith"
woven-imprint chat marcus
```

## Step 6: Bring an Existing Character

### From ChatGPT

1. ChatGPT → Settings → Data Controls → Export Data
2. Download zip, unzip, find `conversations.json`

**Web UI**: Migrate tab → upload the file

**Terminal**: `woven-imprint migrate conversations.json`

### From a Custom GPT

**Web UI**: Migrate tab → paste instructions text

**Terminal**: `woven-imprint migrate --text "You are Coach Rivera..."`

With knowledge files: `woven-imprint migrate instructions.txt --knowledge manual.pdf`

For PDF support:
```
pipx inject woven-imprint pymupdf    # pipx
pip install woven-imprint[pdf]        # venv
```

### From SillyTavern

```
woven-imprint migrate character_card.json
woven-imprint migrate character_card.png
```

## Updating

```
woven-imprint update
```

Detects pipx vs pip automatically and upgrades everything including extras.

## What's Next?

- **Keep chatting** — characters remember everything across sessions
- **Connect to Claude or Cursor** — see [MCP Setup](../examples/mcp_setup.md)
- **Python API** — see [Developer Guide](DEVELOPER_GUIDE.md)
