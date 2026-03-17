# Getting Started with Woven Imprint

## Choose Your Setup

### By Platform
- **[Windows / WSL](setup-windows.md)** — step-by-step for Windows and Windows Subsystem for Linux
- **[macOS](setup-mac.md)** — step-by-step for Mac
- **[Linux](setup-linux.md)** — step-by-step for Ubuntu, Debian, Fedora, and other distros

### By Experience Level
- **New to this?** — Pick your platform above. Each guide walks you from zero to chatting.
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
