# Setup Guide — macOS

## What You'll Need

- macOS 12 or later
- About 15 minutes
- About 3 GB of free disk space (for the AI model)

## Step 1: Install Python

1. Open Terminal (press `Cmd+Space`, type "Terminal", press Enter)
2. Type `python3 --version` and press Enter
3. If you see `Python 3.11` or higher, skip to Step 2
4. Install [Homebrew](https://brew.sh) if you don't have it (paste their command into Terminal)
5. Then: `brew install python@3.12`
6. Verify: `python3 --version`

> On macOS, use `python3` and `pip3` in all commands below.

## Step 2: Install Woven Imprint

```
pip3 install woven-imprint
```

You should see "Successfully installed woven-imprint".

### Troubleshooting

**"pip3 not found"**: Try `python3 -m pip install woven-imprint`

## Step 3: Install an AI Model

1. Go to [ollama.com](https://ollama.com) and click Download
2. Open the downloaded file and drag Ollama to Applications
3. Open Ollama from Applications — it appears in your menu bar
4. In Terminal:

```
ollama pull llama3.2
ollama pull nomic-embed-text
```

About 2.3 GB total. You'll see a progress bar.

### Don't want to install Ollama?

Use OpenAI's API instead:

1. Get an API key from [platform.openai.com](https://platform.openai.com)
2. Run: `pip3 install woven-imprint[openai]`
3. Configure the provider:

```
export WOVEN_IMPRINT_LLM_PROVIDER=openai
export WOVEN_IMPRINT_EMBEDDING_PROVIDER=openai
export WOVEN_IMPRINT_API_KEY_LLM=sk-your-key-here
export WOVEN_IMPRINT_MODEL=gpt-4o-mini
```

Or set it permanently in `~/.woven_imprint/config.yaml` — see [Configuration](CONFIGURATION.md#openai-backend-no-local-ollama-needed).

## Step 4: Try It

### Web interface (recommended)

```
pip3 install woven-imprint[ui]
woven-imprint ui
```

Opens your default browser with chat, character management, migration, and settings.

You can specify a browser: `woven-imprint ui --browser firefox`

### Terminal

```
woven-imprint demo
```

Commands during chat (everything without / goes to the character):
- **/help** — list all commands
- **/stats** — memory, emotions, relationships
- **/reflect** — character reflects on experiences
- **/memories** — search memories
- **/quit** — end session

### Not working?

- **Connection error**: make sure Ollama is running (check menu bar)
- **"model not found"**: run `ollama pull llama3.2`
- **Very slow**: first response can take 30-60 seconds while the model loads

## Step 5: Create Your Own Character

### From the web UI

Click the **Characters** tab → fill in the form → click **Create Character**.

### From Terminal

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

### From SillyTavern

```
woven-imprint migrate character_card.json
```

## Updating

```
woven-imprint update
```

## What's Next?

- **Keep chatting** — characters remember everything across sessions
- **Connect to Claude or Cursor** — see [MCP Setup](../examples/mcp_setup.md)
- **Python API** — see [Developer Guide](DEVELOPER_GUIDE.md)
