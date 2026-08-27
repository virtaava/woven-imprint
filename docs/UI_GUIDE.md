# Demo UI Guide

The woven-imprint demo UI is a browser-based interface for chatting with AI characters that have persistent memory, emotions, and relationships. This guide walks through every part of the interface.

## Starting the UI

```bash
woven-imprint demo
```

Opens your browser at **http://localhost:7860**. Use `--no-browser` to skip auto-opening. The
UI is a prebuilt static bundle shipped with the package — if you're developing the frontend
itself, see [Contributing](../CONTRIBUTING.md#frontend-demo-ui) for the `npm run build` step
that regenerates it.

**Local embedder note**: if your chat model and embedding model are served by two different
OpenAI-compatible endpoints (e.g. a big model for chat, a small one dedicated to embeddings),
set `llm.embedding_base_url` (or `WOVEN_IMPRINT_EMBEDDING_BASE_URL`) — otherwise the demo tries
to embed against `llm.base_url` and memory writes fail. See
[Configuration](CONFIGURATION.md#llm-settings).

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

### Pinned

A card above the Memory Feed, titled "Things the character always remembers," listing every
pinned memory. Pinned memories are rendered into a dedicated block in the character's system
prompt and are never dropped by the context budget — this is where you see (and undo) that.
Only appears once at least one memory is pinned.

### Memory Feed

Shows the character's most recent memories with tier badges:
- **Bedrock** (gold) — permanent identity memories, never forgotten
- **Core** (blue) — important learned facts, very durable
- **Buffer** (gray) — recent conversation details, may be consolidated over time

Each memory shows its importance score as a percentage and, when present, its date. Every
memory — pinned or not — has three controls:
- **Pin/unpin** (pin icon) — toggles whether the memory is always included in the prompt. A
  pinned memory also moves into the Pinned card above.
- **Edit** (pencil icon) — opens an inline textarea; **Save** re-embeds the memory with the new
  text, **Cancel** or **Escape** discards the edit.
- **Delete** (trash icon, with a confirmation) — permanently forgets the memory. If a fact was
  extracted from it, the fact is retracted (no longer considered current) rather than deleted.

The feed only shows unpinned memories not covered by the Pinned card above, so nothing appears
twice.

**Note**: the Memory Feed itself only shows the 10 most recent memories. Unpinning a memory that
isn't among those 10 moves it out of the Pinned card but it won't reappear in the feed below —
it still exists (search for it, or check the library/API directly) but the panel won't show it
again until it re-enters the recent-10 window or you reload.

### Facts

A card listing the character's current structured facts about you (statement, and "(since
YYYY-MM-DD)" when dated). Each fact has:
- **Edit** (pencil icon) — inline edit of the fact's object/value; saving updates both the fact
  (the statement is re-derived from the new object) and, if one is linked, the memory it came
  from — the memory's content is rewritten to match and re-embedded.
- **Retract** (with a confirmation) — marks the fact no longer current. It stays in the fact's
  history; a new fact can still supersede it later.

Only appears once the character has at least one current fact about you.

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

The drawer has **three** tabs (not four — earlier drafts of this guide listed a separate
"Migrate" tab, but migrate-from-text lives inside **Import**):

| Tab | Description |
|-----|-------------|
| **Characters** | List all characters. Click one to switch. Delete with the trash icon. |
| **Create** | Create a new character from scratch — name, personality, backstory, speaking style, birthdate. |
| **Import** | Two ways in, one tab: **Import from File** — a woven-imprint JSON export, a SillyTavern character card (JSON or PNG, V2 or V3), a ChatGPT conversation export, or a markdown/text persona file; **Migrate from Text** — paste Custom GPT instructions, persona text, or card text and give it a name to create a character from it directly. |

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
