# Getting Started with Woven Imprint

## 1. Install

```bash
pip install woven-imprint[demo]
```

> **Ubuntu / WSL getting an "externally-managed-environment" error?**
> Modern Ubuntu blocks system-wide pip installs. Use pipx instead:
> ```bash
> sudo apt install pipx
> pipx install woven-imprint
> pipx inject woven-imprint fastapi uvicorn
> ```
> Or use a virtual environment — see the [Linux guide](setup-linux.md).

## 2. Run

```bash
woven-imprint demo
```

This starts the web UI and opens your browser automatically.
The UI is at **http://localhost:7860** — bookmark it or navigate there manually if the browser doesn't open.

## 3. Connect an LLM

**No model is configured by default** — you must set one up before chatting. Click **⚙ Settings** in the top bar.

**Option A — Ollama (local, free, no account needed)**

Install Ollama from [ollama.com](https://ollama.com), then pull a model and the embedding model:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```
In Settings, select **Ollama**. The UI discovers your installed models automatically. Pick one, click **Test Connection**, then **Save**.

**Option B — OpenAI, Anthropic, DeepSeek, or any API**

Select your provider in Settings, paste your API key. The UI queries the provider's API and shows available models. Pick one, click **Test Connection**, then **Save**.

---

Once connected you'll meet **Meridian**, a built-in demo character who knows how woven-imprint works. Chat with her, create your own characters from the **Characters** tab, or click the **?** icon in the top bar for the full [UI Guide](UI_GUIDE.md).

---

## Bringing an existing character?

If you have a character from ChatGPT, a Custom GPT, SillyTavern, or a text file, you can import it and pick up where you left off — memory and all.

→ **[Migration Guide](MIGRATION.md)**

---

## Platform guides

For a full walkthrough including Python installation and troubleshooting:

- **[Windows / WSL](setup-windows.md)**
- **[macOS](setup-mac.md)**
- **[Linux](setup-linux.md)**
- **[Docker](setup-docker.md)** — no Python needed

**Developer?** — See the [Developer Guide](DEVELOPER_GUIDE.md) for the Python API, MCP integration, and more.

---

## Upgrading

```bash
# pip
pip install --upgrade woven-imprint[demo]

# pipx
pipx upgrade woven-imprint
pipx inject woven-imprint --force fastapi uvicorn

# venv
source ~/woven-imprint-env/bin/activate
pip install --upgrade woven-imprint[demo]
```

Your character data (`~/.woven_imprint/characters.db`) is preserved across upgrades — no migration needed.

---

## Migrating from v0.4.x (Gradio UI)

The old Gradio UI (`woven-imprint ui`) was replaced by the React UI (`woven-imprint demo`) in v0.5.0.

```bash
# Old — no longer works
pip install woven-imprint[ui]
woven-imprint ui

# New
pip install woven-imprint[demo]
woven-imprint demo
```

**pipx (clean reinstall removes the old Gradio dependency):**
```bash
pipx uninstall woven-imprint
pipx install woven-imprint
pipx inject woven-imprint fastapi uvicorn
```

**venv:**
```bash
source ~/woven-imprint-env/bin/activate
pip install --upgrade woven-imprint[demo]
pip uninstall gradio -y
```

All your characters, memories, and relationships carry over automatically.

---

See [Configuration](CONFIGURATION.md) for all provider options and settings.
