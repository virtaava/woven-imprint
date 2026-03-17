# MCP Server Setup

Woven Imprint characters in Claude Desktop, Cursor, Hermes Agent, or OpenClaw.

## Install

```bash
pip install woven-imprint
```

## Claude Desktop / Cursor

Add to your MCP config (`~/.config/claude/claude_desktop_config.json` or Cursor settings):

```json
{
  "mcpServers": {
    "woven-imprint": {
      "command": "python",
      "args": ["-m", "woven_imprint.mcp_server"]
    }
  }
}
```

## Hermes Agent

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  woven-imprint:
    command: python3
    args: ["-m", "woven_imprint.mcp_server"]
    timeout: 120
```

## OpenClaw

Add to `openclaw.json` under `mcp`:

```json
{
  "mcp": {
    "woven-imprint": {
      "command": "python3",
      "args": ["-m", "woven_imprint.mcp_server"]
    }
  }
}
```

## Available MCP Tools

Once connected, these tools are available:

| Tool | Description |
|------|-------------|
| `list_characters` | List all persistent characters |
| `create_character` | Create a new character with personality and backstory |
| `chat` | Send a message to a character, get an in-character response |
| `recall` | Search a character's memories |
| `get_relationship` | See how a character feels about someone |
| `reflect` | Have a character reflect on their experiences |
| `evolve` | Detect and apply character growth |
| `end_session` | End session and generate summary |
| `new_session` | Start a fresh conversation session |
| `consolidate` | Compress buffer memories into core memories |
| `get_stats` | Memory counts, emotion, arc phase, relationships |
| `delete_character` | Permanently delete a character |

## Usage in Claude

After setup, just ask Claude:

> "Create a character named Marcus who is a gruff blacksmith"

> "Chat with Marcus: I need a sword forged by tomorrow"

> "How does Marcus feel about me?"

> "What does Marcus remember about our conversation?"

Claude will use the MCP tools automatically. Marcus persists across conversations.

## OpenAI-Compatible API (Alternative)

For systems that speak OpenAI API but not MCP:

```bash
# Start the server
python -m woven_imprint.server.api --port 8650

# Point any OpenAI client to it
export OPENAI_BASE_URL=http://127.0.0.1:8650/v1
export OPENAI_API_KEY=not-needed
```

Then use `model="marcus"` in any OpenAI API call. The character name IS the model name.
