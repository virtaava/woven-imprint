# Bringing an Existing Character to Woven Imprint

If you have a character you've built elsewhere — a Custom GPT, a ChatGPT conversation history, a SillyTavern card, or even just a text description — you can import it into woven-imprint and continue from there with full persistent memory.

---

## From ChatGPT (conversation history)

1. Go to **ChatGPT → Settings → Data Controls → Export Data**
2. Wait for the email, download the zip, and unzip it
3. Find `conversations.json` inside

**Web UI** — go to the **Migrate** tab, upload `conversations.json`, and follow the prompts.

**Terminal:**
```bash
woven-imprint migrate conversations.json
```

---

## From a Custom GPT

1. Open your GPT → **Configure** → copy the text from the **Instructions** field

**Web UI** — go to the **Migrate** tab, paste the instructions text.

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

Pass the character card directly — both JSON and PNG formats are supported:
```bash
woven-imprint migrate character_card.json
woven-imprint migrate character_card.png
```

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

## See Also

- **[Getting Started](GETTING_STARTED.md)** — install, run, connect an LLM
- **[UI Guide](UI_GUIDE.md)** — walkthrough of every UI element
- **[Developer Guide](DEVELOPER_GUIDE.md)** — Python API for programmatic migration
