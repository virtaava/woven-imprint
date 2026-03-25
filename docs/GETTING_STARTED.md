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

On first launch, click **⚙ Settings** to configure your LLM provider.

**Option A — Ollama (local, free, no account needed)**

Install Ollama from [ollama.com](https://ollama.com), then pull the models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```
In the Settings panel, select **Ollama** and click **Test Connection**. Done.

**Option B — OpenAI, Anthropic, DeepSeek, or any API**

Select your provider in Settings, paste your API key, pick a model, and click **Test Connection**. The UI discovers available models automatically.

---

Once connected you'll meet **Meridian**, a demo character who knows how woven-imprint works and can answer questions about it. Chat with her, then create your own characters from the **Characters** tab.

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
