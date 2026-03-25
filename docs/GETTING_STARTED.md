# Getting Started with Woven Imprint

## Choose Your Setup

### By Platform
- **[Windows / WSL](setup-windows.md)** — step-by-step for Windows and Windows Subsystem for Linux
- **[macOS](setup-mac.md)** — step-by-step for Mac
- **[Linux](setup-linux.md)** — step-by-step for Ubuntu, Debian, Fedora, and other distros
- **[Docker](setup-docker.md)** — no Python needed, fully containerized

### By Experience Level
- **New to this?** — Pick your platform above. Each guide walks you from zero to chatting.
- **Just want to try it?** — Use [Docker](setup-docker.md). Three commands and you're running.
- **Developer?** — Jump to the [Developer Guide](DEVELOPER_GUIDE.md) for Python API, MCP, and integrations.

---

## Fresh Installation

### Quick install (Python 3.11+ required)

```bash
pip install woven-imprint[demo]
woven-imprint demo
```

This opens a browser with the full demo UI. On first launch you'll be prompted to configure your LLM provider (Ollama, OpenAI, Anthropic, DeepSeek, NVIDIA NIM, or any OpenAI-compatible API). The UI discovers available models automatically.

### Ubuntu / WSL — "externally-managed-environment" error?

Modern Ubuntu (24.04+) blocks system-wide pip installs. Use one of these:

**Option A: pipx (recommended)**
```bash
sudo apt install pipx
pipx install woven-imprint
pipx inject woven-imprint fastapi uvicorn    # for the demo UI
woven-imprint demo
```

**Option B: virtual environment**
```bash
python3 -m venv ~/woven-imprint-env
source ~/woven-imprint-env/bin/activate
pip install woven-imprint[demo]
woven-imprint demo
```

Add to `.bashrc` to activate automatically:
```bash
echo 'source ~/woven-imprint-env/bin/activate' >> ~/.bashrc
```

### With Ollama (local, free)

```bash
pip install woven-imprint[demo]
ollama pull llama3.2 && ollama pull nomic-embed-text
woven-imprint demo
```

### With OpenAI

```bash
pip install "woven-imprint[demo,openai]"
woven-imprint demo
```

Configure your API key in the provider settings UI, or via environment:
```bash
export WOVEN_IMPRINT_LLM_PROVIDER=openai
export WOVEN_IMPRINT_API_KEY=sk-your-key-here
export WOVEN_IMPRINT_MODEL=gpt-4o-mini
```

### CLI only (no web UI)

```bash
pip install woven-imprint
woven-imprint chat MyCharacter
```

### Remote access (LAN / server)

```bash
woven-imprint demo --host 0.0.0.0 --port 7860
```

---

## Upgrading to Latest Version

**pip:**
```bash
pip install --upgrade woven-imprint[demo]
```

**pipx:**
```bash
pipx upgrade woven-imprint
pipx inject woven-imprint --force fastapi uvicorn
```

**venv:**
```bash
source ~/woven-imprint-env/bin/activate
pip install --upgrade woven-imprint[demo]
```

Your character data (`~/.woven_imprint/characters.db`) is fully compatible across versions — no migration needed.

---

## Upgrading from v0.4.x (Gradio-based UI)

The `[ui]` extra (Gradio) has been replaced by `[demo]` (React + FastAPI) in v0.5.0:

```bash
# Old commands (no longer work)
pip install woven-imprint[ui]
woven-imprint ui

# New commands
pip install woven-imprint[demo]
woven-imprint demo
```

**If you used pipx:**
```bash
pipx uninstall woven-imprint
pipx install woven-imprint
pipx inject woven-imprint fastapi uvicorn
```

**If you used a venv:**
```bash
source ~/woven-imprint-env/bin/activate
pip install --upgrade woven-imprint[demo]
pip uninstall gradio -y    # optional cleanup
```

**What carries over automatically:**
- All characters, memories, and relationships (`~/.woven_imprint/characters.db`)
- Configuration (`~/.woven_imprint/config.yaml`)
- No data migration needed

**What changed:**
- `woven-imprint ui` → `woven-imprint demo`
- Gradio web UI → React web UI with character management, X-Ray sidebar, provider discovery
- New `--host` flag for remote access (`woven-imprint demo --host 0.0.0.0`)

---

See [Configuration](CONFIGURATION.md) for all provider options and settings.
