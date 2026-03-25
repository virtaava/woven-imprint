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

## Quick Install (if you already have Python 3.11+)

```bash
pip install woven-imprint[demo]
woven-imprint demo
```

This opens a browser with the full demo UI. On first launch you'll be prompted to configure your LLM provider (Ollama, OpenAI, Anthropic, DeepSeek, NVIDIA NIM, or any OpenAI-compatible API). The UI discovers available models automatically.

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

### Remote access (Tailscale / LAN)

```bash
woven-imprint demo --host 0.0.0.0 --port 7860
```

## Upgrading from v0.4.x

The `[ui]` extra (Gradio) has been replaced by `[demo]` (React + FastAPI):

```bash
# Old (no longer works)
pip install woven-imprint[ui]
woven-imprint ui

# New
pip install woven-imprint[demo]
woven-imprint demo
```

Your character data (`~/.woven_imprint/characters.db`) is fully compatible — no migration needed. All characters, memories, and relationships carry over automatically.

If you used `pipx`:

```bash
pipx uninstall woven-imprint
pipx install "woven-imprint[demo]"
```

See [Configuration](CONFIGURATION.md) for all provider options and settings.
