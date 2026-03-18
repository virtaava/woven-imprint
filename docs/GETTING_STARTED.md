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

## Quick Install (if you already have Python 3.11+ and Ollama)

```bash
pip install woven-imprint
ollama pull llama3.2 && ollama pull nomic-embed-text
woven-imprint demo
```

Or use the web interface:

```bash
pip install woven-imprint[ui]
woven-imprint ui
```

### No Ollama? Use OpenAI instead

```bash
pip install woven-imprint[openai]
export WOVEN_IMPRINT_LLM_PROVIDER=openai
export WOVEN_IMPRINT_EMBEDDING_PROVIDER=openai
export WOVEN_IMPRINT_API_KEY_LLM=sk-your-key-here
export WOVEN_IMPRINT_MODEL=gpt-4o-mini
woven-imprint demo
```

See [Configuration](CONFIGURATION.md) for all provider options (Ollama, OpenAI, Anthropic, vLLM).
