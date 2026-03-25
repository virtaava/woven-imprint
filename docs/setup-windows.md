# Setup Guide — Windows

This guide covers both native Windows and WSL (Windows Subsystem for Linux).

## What You'll Need

- Windows 10 or 11
- About 15 minutes
- About 3 GB of free disk space (for the AI model)

## Step 1: Install Python

1. Press `Win+R`, type `powershell`, press Enter
2. Type `python --version` and press Enter
3. If you see `Python 3.11` or higher, skip to Step 2
4. Go to [python.org/downloads](https://www.python.org/downloads/) and download the installer
5. **Important**: Check the box **"Add Python to PATH"** during installation
6. Close and reopen PowerShell
7. Verify: `python --version`

## Step 2: Install Woven Imprint

In PowerShell:

```
pip install woven-imprint
```

You should see "Successfully installed woven-imprint".

### Troubleshooting

**"pip is not recognized"**: Close and reopen PowerShell. If that doesn't work, try:
```
python -m pip install woven-imprint
```

**Using WSL?** If you see "externally-managed-environment", use pipx:
```
sudo apt install pipx
pipx install woven-imprint
```

Or create a virtual environment:
```
python3 -m venv ~/woven-imprint-env
source ~/woven-imprint-env/bin/activate
pip install woven-imprint
```
You'll need to run the `source` command each time you open a new terminal.
Add it to your profile to make it automatic:
```
echo 'source ~/woven-imprint-env/bin/activate' >> ~/.bashrc
```

## Step 3: Install an AI Model

1. Go to [ollama.com](https://ollama.com) and click Download
2. Run the installer
3. After installation, you should see the Ollama icon in your system tray (bottom-right)
4. Open PowerShell (or your WSL terminal) and run:

```
ollama pull llama3.2
ollama pull nomic-embed-text
```

The first download is about 2 GB, the second about 300 MB. You'll see a progress bar.

### Don't want to install Ollama?

You can use OpenAI's API instead (requires an API key, costs money per use):

1. Get an API key from [platform.openai.com](https://platform.openai.com)
2. Run: `pip install woven-imprint[openai]`
3. Configure the provider:

**PowerShell**:
```
$env:WOVEN_IMPRINT_LLM_PROVIDER = "openai"
$env:WOVEN_IMPRINT_EMBEDDING_PROVIDER = "openai"
$env:WOVEN_IMPRINT_API_KEY_LLM = "sk-your-key-here"
$env:WOVEN_IMPRINT_MODEL = "gpt-4o-mini"
```

**WSL**:
```
export WOVEN_IMPRINT_LLM_PROVIDER=openai
export WOVEN_IMPRINT_EMBEDDING_PROVIDER=openai
export WOVEN_IMPRINT_API_KEY_LLM=sk-your-key-here
export WOVEN_IMPRINT_MODEL=gpt-4o-mini
```

Or set it permanently in `~/.woven_imprint/config.yaml` — see [Configuration](CONFIGURATION.md#openai-backend-no-local-ollama-needed).

## Step 4: Try It

### React demo UI (recommended)

```
pip install woven-imprint[demo]
woven-imprint demo
```

This opens a browser tab with everything you need — chat, create characters,
import existing ones, and see stats. No terminal needed after this.

**WSL users**: If it opens Chromium instead of your Windows browser, specify it:
```
woven-imprint demo --browser none
```
Then open `http://127.0.0.1:5173` in your Windows browser manually.
Or install wslu for automatic Windows browser support:
```
sudo apt install wslu
woven-imprint demo
```

### Terminal REPL

```
woven-imprint chat alice
```

You'll chat with a character named Alice Blackwood. Type messages and press Enter.

Commands during chat (use / prefix — everything else goes to the character):
- **/help** — list all commands
- **/stats** — memory, emotions, relationships
- **/reflect** — character reflects on experiences
- **/memories** — search memories
- **/quit** — end session

### Not working?

- **"command not found"**: close and reopen your terminal
- **Connection error**: make sure Ollama is running (check system tray)
- **"model not found"**: run `ollama pull llama3.2` and `ollama pull nomic-embed-text`
- **Very slow**: first response can take 30-60 seconds while the model loads

## Step 5: Create Your Own Character

### From the web UI

Click the **Characters** tab → fill in the form → click **Create Character**.

### From the terminal

```
woven-imprint create "Marcus the Blacksmith"
```

You'll be asked for backstory, personality, speaking style, and birthdate (optional).
Then chat: `woven-imprint chat marcus`

## Step 6: Bring an Existing Character

### From ChatGPT

1. Go to ChatGPT → Settings → Data Controls → Export Data
2. Wait for the email, download the zip, unzip it
3. Find `conversations.json`

**Web UI**: Go to the **Migrate** tab → upload the file

**Terminal**:
```
woven-imprint migrate conversations.json
```

### From a Custom GPT

1. Go to your GPT → Configure → copy the Instructions text

**Web UI**: Go to the **Migrate** tab → paste the text

**Terminal**:
```
woven-imprint migrate --text "You are Coach Rivera, a retired soccer coach..."
```

With knowledge files:
```
woven-imprint migrate instructions.txt --knowledge manual.pdf faq.txt
```

### From SillyTavern / TavernAI

```
woven-imprint migrate character_card.json
woven-imprint migrate character_card.png
```

### From any text file

```
woven-imprint migrate persona.md
```

## Updating

```
woven-imprint update
```

This auto-detects pip vs pipx and upgrades everything including extras (UI, PDF, etc.).

## What's Next?

- **Keep chatting** — characters remember everything across sessions
- **Check relationships** — `woven-imprint stats marcus` or click **Refresh Stats** in the UI
- **Connect to Claude or Cursor** — see [MCP Setup](../examples/mcp_setup.md)
- **Python API** — see [Developer Guide](DEVELOPER_GUIDE.md)
