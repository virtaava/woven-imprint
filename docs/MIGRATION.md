# Bringing an Existing Character to Woven Imprint

If you have a character you've built elsewhere — a Custom GPT, a ChatGPT conversation history, a SillyTavern card, or even just a text description — you can import it into woven-imprint and continue from there with full persistent memory.

---

## From ChatGPT (conversation history)

1. Go to **ChatGPT → Settings → Data Controls → Export Data**
2. Wait for the email, download the zip, and unzip it
3. Find `conversations.json` inside

**Web UI** — go to the **Import** tab, upload `conversations.json`, and follow the prompts.

**Terminal:**
```bash
woven-imprint migrate conversations.json
```

---

## From a Custom GPT

1. Open your GPT → **Configure** → copy the text from the **Instructions** field

**Web UI** — go to the **Import** tab, "Migrate from Text" section, paste the instructions text.

**Terminal:**
```bash
woven-imprint migrate --text "You are Coach Rivera, a retired soccer coach..."
```

If your GPT has knowledge files, pass them with `--knowledge`:
```bash
woven-imprint migrate instructions.txt --knowledge manual.pdf faq.txt
```

For PDF support, install the extra first:
```bash
pip install woven-imprint[pdf]        # pip / venv
pipx inject woven-imprint pymupdf    # pipx
```

---

## From SillyTavern / TavernAI

Pass the character card directly — both JSON and PNG formats are supported, spec **V2 and V3**:
```bash
woven-imprint migrate character_card.json
woven-imprint migrate character_card.png
```

**Web UI** — go to the **Import** tab, "Import from File", and choose the card (`.json` or
`.png`).

PNG cards embed the card JSON in a `tEXt` chunk. woven-imprint reads the V3 `ccv3` chunk when
present (preferring it over V2) and falls back to the V2 `chara` chunk otherwise, so cards from
either generation of SillyTavern import correctly.

Beyond `name`/`description`/`personality`/`first_mes`, the importer also reads:
- **`scenario`**, **`tags`** and **`alternate_greetings`** (combined with `first_mes`) → soft
  persona traits (personality, tags, greetings).
- **`system_prompt`** → a hard constraint (`hard.hard_constraints`) — the character's system
  prompt survives the move.
- **`creator_notes`** → kept on the character (`hard.creator_notes`) so a re-export
  (`export-card`, below) can write it back.

**Lorebook (`character_book`)**: every *enabled* entry becomes a memory, `content` prefixed
`[Lore: key1, key2, ...] ` (empty `keys` renders `[Lore] `):
- A **`constant`** entry (always active in SillyTavern) imports as a **pinned bedrock memory** —
  it lands in woven-imprint's own "Things you always remember" prompt block, the closest
  equivalent to SillyTavern always-on lore.
- A **keyed** (non-constant) entry imports as a **core memory**, retrieved normally by relevance.

Disabled entries are skipped. A `keys` field written as a comma-separated string, a bare
string, or a list with stray blank/non-string entries is all coerced to a clean list.

**Known limitation**: this is a lossy round-trip. Lorebook entries come back as plain text
memories (prefixed `[Lore: ...]`), not as woven-imprint's structured `FactStore` facts — a
lorebook has no subject/predicate/object shape to recover. If you `export-card` a
woven-imprint character and re-import the result, the current facts baked into its lorebook
come back as `[Lore: ...]`-wrapped core memories, not facts, and the fact ↔ memory linkage does
not survive the trip.

---

## From a Claude Code project

Point `migrate` at a project directory containing a `CLAUDE.md`:
```bash
woven-imprint migrate /path/to/claude/project/
```

The importer reads `CLAUDE.md` plus every other markdown file in the project (and anything
under `.claude/`) as the character's source material. This path is CLI/API-only — the demo's
file picker doesn't support choosing a directory.

---

## From any text or markdown file

If you have a persona written as plain text, markdown, or any other format:
```bash
woven-imprint migrate persona.md
woven-imprint migrate persona.txt
```

---

## What happens after import

The migrated character is created in your local database (`~/.woven_imprint/characters.db`) with its persona, backstory, and personality intact. From that point it behaves like any other woven-imprint character — memory, emotions, and relationships build up as you chat.

Open the demo UI to chat with the imported character:
```bash
woven-imprint demo
```

Then go to **http://localhost:7860**, open the **Characters** tab, and select your character.

---

## Exporting to SillyTavern

A woven-imprint character can also go the other direction — out as a SillyTavern V2 character
card:

```bash
woven-imprint export-card <name-or-id> [-o card.json]
```

or via the HTTP API (`GET /api/characters/{id}/card`), or in Python: `character.export_card()`.

The card's `character_book` (lorebook) is generated from the character's **pinned memories**
(as `constant` entries), **current facts about the user**, and its most important **core
memories** — everything else (`description`, `personality`, `scenario`, `first_mes` /
`alternate_greetings`, `system_prompt`, `tags`) maps from the matching persona fields above.

This is a one-way-lossy round trip: re-importing an exported card wraps every lorebook entry's
content as `[Lore: keys] ...` (memory text comes back prefixed, not verbatim), and facts come
back as plain core (or pinned bedrock, if `constant`) memories rather than structured facts —
see the known limitation above.

---

## See Also

- **[Getting Started](GETTING_STARTED.md)** — install, run, connect an LLM
- **[UI Guide](UI_GUIDE.md)** — walkthrough of every UI element
- **[Developer Guide](DEVELOPER_GUIDE.md)** — Python API for programmatic migration
