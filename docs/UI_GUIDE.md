# Demo UI Guide

The woven-imprint demo UI is a browser-based interface for chatting with AI characters that have persistent memory, emotions, and relationships. This guide walks through every part of the interface.

## Starting the UI

```bash
woven-imprint demo
```

Opens your browser at **http://localhost:7860**. Use `--no-browser` to skip auto-opening.

---

## First Launch

On a fresh install, no LLM provider is configured. The chat area shows a setup message asking you to open **Settings**. You must configure a provider before chatting.

---

## Top Bar

The top bar contains four elements:

| Element | Description |
|---------|-------------|
| **Character button** (left) | Shows the active character name. Click to open the character management drawer. |
| **Provider info** (center) | Shows the active provider and model (e.g. "ollama / mistral-small3.2"). A wifi icon indicates connection status. |
| **X-Ray toggle** | Show/hide the X-Ray sidebar panel. State is saved in your browser. |
| **Settings button** | Open the provider configuration modal. |

---

## Chat Panel

The main area where you interact with the character.

### Sending Messages

Type in the input field and press Enter or click the **Send** button (arrow icon). The character responds in-character, drawing on their persona, memory, and emotional state.

### Suggested Prompts

On the first message, prompt chips appear below the greeting. Click one to start the conversation quickly. These disappear after the first message.

### Reflect Button

Click **Reflect** to trigger the character's internal self-reflection. The character reviews their own memories and experiences and shares their thoughts as a system message (gold border). This can cause emotional shifts and personality insights. No user message is needed.

### New Session Button

Click **New Session** to end the current conversation session and start a fresh one. The character still remembers everything from previous sessions — sessions are conversation boundaries, not memory boundaries. Ending a session generates a summary that becomes a core memory.

---

## X-Ray Panel

The right sidebar shows the character's internal state updating in real time as you chat.

### Emotion

Displays the character's current mood, intensity (0-100%), and cause. An arc phase and tension level show where the character is in their narrative arc.

### Relationship Radar

A radar chart showing five dimensions of the character's relationship with you:
- **Trust** — how much the character trusts you
- **Affection** — emotional warmth
- **Respect** — how seriously they take you
- **Familiarity** — how well they know you
- **Tension** — unresolved conflict or friction

All values range from 0 to 1 and evolve naturally through conversation.

### Memory Feed

Shows the character's most recent memories with tier badges:
- **Bedrock** (gold) — permanent identity memories, never forgotten
- **Core** (blue) — important learned facts, very durable
- **Buffer** (gray) — recent conversation details, may be consolidated over time

Each memory shows its importance score as a percentage.

### Sessions

Lists the character's conversation sessions with timestamps. The active session is highlighted.

- **Pencil icon** — rename a session (click, type a name, press Enter to save or Escape/click away to cancel)
- **Play icon** — resume a past session (new messages will be tagged under that session)
- **Active badge** — shows which session is currently active

### Memory Search

Search the character's memories by keyword. Results are shown with tier badges.

---

## Settings Modal

Configure which LLM provider powers the character's responses.

### Provider Presets

Select from:
- **Ollama** — local, free, no account needed (default port 11434)
- **OpenAI** — requires API key
- **Anthropic** — requires API key
- **DeepSeek** — requires API key
- **NVIDIA NIM** — requires API key
- **Custom** — any OpenAI-compatible API (provide base URL and optional key)

### Model Discovery

After selecting a provider and entering credentials (if needed), the UI queries the provider's API to discover available models. Select one from the dropdown.

For Ollama, this lists all models you've pulled locally.

### Test Connection

Click **Test Connection** before saving. The UI sends a short test prompt to verify the provider responds correctly. The **Save** button is only enabled after a successful test.

---

## Character Drawer

Click the character name in the top bar to open the character management panel.

### Tabs

| Tab | Description |
|-----|-------------|
| **Characters** | List all characters. Click one to switch. Delete with the trash icon. |
| **Create** | Create a new character from scratch — name, personality, backstory, speaking style, birthdate. |
| **Import** | Import a character from a JSON export, SillyTavern card (JSON or PNG), or any text/markdown file. |
| **Migrate** | Paste text from a Custom GPT's instructions to create a character from it. |

### Switching Characters

Click a character name in the list to switch. This ends the current session with the old character and starts a new one with the selected character. All memories and relationships are preserved.

---

## Keyboard Shortcuts

| Key | Context | Action |
|-----|---------|--------|
| Enter | Chat input | Send message |
| Escape | Session rename | Cancel rename |

---

## Tips

- **Characters remember everything** across sessions and browser refreshes. Your session and character selection are saved in your browser.
- **Reflect often** — it helps the character develop deeper self-awareness and emotional range.
- **Check the X-Ray panel** to see how the character's emotions and relationship with you are evolving.
- **Name your sessions** to find them later — click the pencil icon next to any session.
- **Resume old sessions** when you want to continue a specific conversation thread.

---

## See Also

- **[Getting Started](GETTING_STARTED.md)** — install, run, connect an LLM
- **[Migration Guide](MIGRATION.md)** — bring characters from ChatGPT, SillyTavern, or text
- **[Configuration](CONFIGURATION.md)** — all provider and memory settings
- **[Developer Guide](DEVELOPER_GUIDE.md)** — Python API, MCP, CLI reference
