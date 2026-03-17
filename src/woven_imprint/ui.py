"""Web UI for Woven Imprint — powered by Gradio.

Launch with: woven-imprint ui
Auto-opens a browser tab with chat, character management, and settings.
"""

from __future__ import annotations


def launch(
    db_path: str | None = None,
    model: str = "llama3.2",
    port: int = 7860,
    browser: str | None = None,
):
    """Launch the Gradio web interface."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is required for the UI. Install it with:")
        print("  pip install woven-imprint[ui]")
        return

    import tempfile
    from pathlib import Path

    from .engine import Engine
    from .llm.ollama import OllamaLLM
    from .embedding.ollama import OllamaEmbedding

    db = db_path or str(Path.home() / ".woven_imprint" / "characters.db")
    Path(db).parent.mkdir(parents=True, exist_ok=True)

    engine = Engine(
        db_path=db,
        llm=OllamaLLM(model=model, num_ctx=8192),
        embedding=OllamaEmbedding(model="nomic-embed-text"),
    )

    _active: dict = {"char": None}

    # ── Helpers ──────────────────────────────────────────

    def char_choices():
        chars = engine.list_characters()
        if not chars:
            return []
        return [f"{c['name']} ({c['id'][:8]})" for c in chars]

    def _resolve_char(selection):
        if not selection:
            return None
        char_id = selection.split("(")[-1].rstrip(")")
        chars = engine.list_characters()
        match = next((c for c in chars if c["id"].startswith(char_id)), None)
        if not match:
            return None
        return engine.load_character(match["id"])

    def _format_stats(char):
        if not char:
            return "", "", ""

        info = f"**{char.name}**"
        if char.persona.age:
            info += f" (age {char.persona.age})"
        info += f"\n\n{char.persona.soft.get('personality', '')}"
        if char.persona.backstory:
            info += f"\n\n*{char.persona.backstory[:200]}*"

        stats = (
            f"**Memory** — buffer: {char.memory.count('buffer')} | "
            f"core: {char.memory.count('core')} | "
            f"bedrock: {char.memory.count('bedrock')}\n\n"
            f"**Emotion** — {char.emotion.mood} ({char.emotion.intensity:.1f})\n\n"
            f"**Arc** — {char.arc.current_phase.value} (tension {char.arc.tension:.1f})"
        )

        rels = char.relationships.get_all()
        rel_text = ""
        for r in rels:
            d = r["dimensions"]
            rel_text += (
                f"**{r['target_id'][:15]}** ({r.get('type', '?')}, {r.get('trajectory', '?')})\n"
                f"trust {d.get('trust', 0):.2f} | "
                f"affection {d.get('affection', 0):.2f} | "
                f"respect {d.get('respect', 0):.2f} | "
                f"familiarity {d.get('familiarity', 0):.2f} | "
                f"tension {d.get('tension', 0):.2f}\n\n"
            )
        if not rel_text:
            rel_text = "No relationships yet. Chat with a character to build one."

        return info, stats, rel_text

    # ── Chat Tab Functions ───────────────────────────────

    def select_character(selection):
        char = _resolve_char(selection)
        _active["char"] = char
        return _format_stats(char)

    def chat_fn(message, history):
        char = _active.get("char")
        if not char:
            return "Select a character first from the dropdown above."
        return char.chat(message, user_id="ui_user")

    def do_reflect():
        char = _active.get("char")
        if not char:
            return "No character selected."
        return char.reflect()

    def do_recall(query):
        char = _active.get("char")
        if not char:
            return "No character selected."
        if not query.strip():
            return "Enter a search term."
        memories = char.recall(query.strip(), limit=10)
        if not memories:
            return "No memories found."
        lines = []
        for m in memories:
            lines.append(f"**[{m['tier']}]** {m['content'][:200]}")
        return "\n\n".join(lines)

    def do_refresh():
        char = _active.get("char")
        return _format_stats(char)

    # ── Characters Tab Functions ─────────────────────────

    def create_character(name, personality, backstory, speaking_style, birthdate):
        if not name.strip():
            return "Name is required.", gr.update()
        persona = {}
        if personality.strip():
            persona["personality"] = personality.strip()
        if backstory.strip():
            persona["backstory"] = backstory.strip()
        if speaking_style.strip():
            persona["speaking_style"] = speaking_style.strip()

        char = engine.create_character(
            name=name.strip(),
            persona=persona,
            birthdate=birthdate.strip() or None,
        )
        _active["char"] = char
        return f"Created **{char.name}**!", gr.update(
            choices=char_choices(), value=f"{char.name} ({char.id[:8]})"
        )

    def delete_character(selection):
        char = _resolve_char(selection)
        if not char:
            return "No character selected.", gr.update()
        name = char.name
        engine.delete_character(char.id)
        if _active.get("char") and _active["char"].id == char.id:
            _active["char"] = None
        return f"Deleted **{name}**.", gr.update(choices=char_choices(), value=None)

    def export_character(selection):
        char = _resolve_char(selection)
        if not char:
            return None, "No character selected."
        path = tempfile.mktemp(suffix=".json", prefix=f"{char.name.lower().replace(' ', '_')}_")
        char.export(path)
        return path, f"Exported **{char.name}** — download below."

    def import_character(file):
        if not file:
            return "Upload a JSON file.", gr.update()
        char = engine.import_character(file.name)
        _active["char"] = char
        return f"Imported **{char.name}**!", gr.update(
            choices=char_choices(), value=f"{char.name} ({char.id[:8]})"
        )

    # ── Migrate Tab Functions ────────────────────────────

    def migrate_from_text(text, name):
        if not text.strip():
            return "Paste instructions or persona text.", gr.update()
        from .migrate import CharacterImporter

        importer = CharacterImporter(engine)
        char = importer.from_text(text.strip(), name=name.strip() or None)
        _active["char"] = char

        result = f"Migrated **{char.name}**\n\n"
        result += f"- Personality: {char.persona.soft.get('personality', '?')[:80]}\n"
        result += f"- Memories: {char.memory.count('core')} core, {char.memory.count('bedrock')} bedrock\n"
        rel = char.relationships.get("imported_user")
        if rel:
            d = rel["dimensions"]
            result += (
                f"- Relationship: trust={d.get('trust', 0):.2f}, "
                f"familiarity={d.get('familiarity', 0):.2f}"
            )
        return result, gr.update(choices=char_choices(), value=f"{char.name} ({char.id[:8]})")

    def migrate_from_file(file, name):
        if not file:
            return "Upload a file (ChatGPT JSON, SillyTavern card, markdown, etc.).", gr.update()
        from .migrate import CharacterImporter

        importer = CharacterImporter(engine)
        char = importer.from_file(file.name, name=name.strip() or None)
        _active["char"] = char

        result = f"Migrated **{char.name}** from `{Path(file.name).name}`\n\n"
        result += f"- Personality: {char.persona.soft.get('personality', '?')[:80]}\n"
        result += f"- Memories: {char.memory.count('core')} core, {char.memory.count('bedrock')} bedrock\n"
        rel = char.relationships.get("imported_user")
        if rel:
            d = rel["dimensions"]
            result += (
                f"- Relationship: trust={d.get('trust', 0):.2f}, "
                f"familiarity={d.get('familiarity', 0):.2f}"
            )
        return result, gr.update(choices=char_choices(), value=f"{char.name} ({char.id[:8]})")

    # ── Build UI ─────────────────────────────────────────

    with gr.Blocks(title="Woven Imprint") as app:
        gr.Markdown("# Woven Imprint\n*Persistent Character Infrastructure*")

        # Shared character selector at the top
        with gr.Row():
            character_dropdown = gr.Dropdown(
                choices=char_choices(),
                label="Active Character",
                interactive=True,
                scale=3,
            )
            char_info = gr.Markdown("Select or create a character to begin.")

        with gr.Tabs():
            # ── Tab 1: Chat ──────────────────────────
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.ChatInterface(fn=chat_fn)

                    with gr.Column(scale=1):
                        gr.Markdown("### Character State")
                        stats_display = gr.Markdown("")
                        rel_display = gr.Markdown("")
                        refresh_btn = gr.Button("Refresh Stats")

                        gr.Markdown("### Actions")
                        reflect_btn = gr.Button("Reflect")
                        reflect_output = gr.Markdown("")

                        recall_input = gr.Textbox(
                            label="Search Memories",
                            placeholder="sword, harbor case, etc.",
                        )
                        recall_btn = gr.Button("Search")
                        recall_output = gr.Markdown("")

            # ── Tab 2: Characters ────────────────────
            with gr.Tab("Characters"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Create New Character")
                        create_name = gr.Textbox(label="Name", placeholder="Marcus the Blacksmith")
                        create_personality = gr.Textbox(
                            label="Personality", placeholder="gruff but kind, dry humor"
                        )
                        create_backstory = gr.Textbox(
                            label="Backstory",
                            placeholder="A village blacksmith who lost his wife...",
                            lines=3,
                        )
                        create_style = gr.Textbox(
                            label="Speaking Style",
                            placeholder="short sentences, working-class dialect",
                        )
                        create_birthdate = gr.Textbox(
                            label="Birthdate", placeholder="YYYY-MM-DD (optional)"
                        )
                        create_btn = gr.Button("Create Character", variant="primary")
                        create_result = gr.Markdown("")

                    with gr.Column():
                        gr.Markdown("### Manage")
                        export_btn = gr.Button("Export Selected Character")
                        export_file = gr.File(label="Download", interactive=False)
                        export_result = gr.Markdown("")

                        gr.Markdown("---")
                        import_file = gr.File(
                            label="Import Character (Woven Imprint JSON)",
                            file_types=[".json"],
                        )
                        import_btn = gr.Button("Import")
                        import_result = gr.Markdown("")

                        gr.Markdown("---")
                        delete_btn = gr.Button("Delete Selected Character", variant="stop")
                        delete_result = gr.Markdown("")

            # ── Tab 3: Migrate ───────────────────────
            with gr.Tab("Migrate"):
                gr.Markdown(
                    "### Bring characters from other systems\n"
                    "Import from ChatGPT, SillyTavern, Custom GPTs, Claude projects, "
                    "or any text/markdown persona file."
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### From Text")
                        gr.Markdown(
                            "Paste Custom GPT instructions, persona description, or character sheet."
                        )
                        migrate_text_input = gr.Textbox(
                            label="Instructions / Persona",
                            placeholder="You are Coach Rivera, a retired soccer coach...",
                            lines=6,
                        )
                        migrate_text_name = gr.Textbox(
                            label="Name (optional)", placeholder="Auto-detected from text"
                        )
                        migrate_text_btn = gr.Button("Migrate from Text", variant="primary")
                        migrate_text_result = gr.Markdown("")

                    with gr.Column():
                        gr.Markdown("#### From File")
                        gr.Markdown(
                            "Upload a ChatGPT export (conversations.json), "
                            "SillyTavern card (.json/.png), Claude project, "
                            "or any .md/.txt persona file."
                        )
                        migrate_file_input = gr.File(
                            label="Upload File",
                            file_types=[".json", ".png", ".md", ".txt", ".csv"],
                        )
                        migrate_file_name = gr.Textbox(
                            label="Name (optional)", placeholder="Auto-detected from file"
                        )
                        migrate_file_btn = gr.Button("Migrate from File", variant="primary")
                        migrate_file_result = gr.Markdown("")

            # ── Tab 4: Settings ──────────────────────
            with gr.Tab("Settings"):
                gr.Markdown("### Configuration")
                gr.Markdown(f"**Database**: `{db}`")
                gr.Markdown(f"**LLM Model**: `{model}`")
                gr.Markdown("**Embedding**: `nomic-embed-text` via Ollama")

                gr.Markdown("### About")
                from . import __version__

                gr.Markdown(
                    f"**Woven Imprint** v{__version__}\n\n"
                    f"Persistent Character Infrastructure\n\n"
                    f"[GitHub](https://github.com/virtaava/woven-imprint) | "
                    f"[Documentation](https://github.com/virtaava/woven-imprint/blob/master/docs/GETTING_STARTED.md) | "
                    f"[PyPI](https://pypi.org/project/woven-imprint/)"
                )

        # ── Events ───────────────────────────────────────

        # Character selection → update info + stats
        character_dropdown.change(
            select_character,
            inputs=[character_dropdown],
            outputs=[char_info, stats_display, rel_display],
        )

        # Chat tab
        refresh_btn.click(do_refresh, outputs=[char_info, stats_display, rel_display])
        reflect_btn.click(do_reflect, outputs=[reflect_output])
        recall_btn.click(do_recall, inputs=[recall_input], outputs=[recall_output])

        # Characters tab
        create_btn.click(
            create_character,
            inputs=[
                create_name,
                create_personality,
                create_backstory,
                create_style,
                create_birthdate,
            ],
            outputs=[create_result, character_dropdown],
        )
        export_btn.click(
            export_character,
            inputs=[character_dropdown],
            outputs=[export_file, export_result],
        )
        import_btn.click(
            import_character,
            inputs=[import_file],
            outputs=[import_result, character_dropdown],
        )
        delete_btn.click(
            delete_character,
            inputs=[character_dropdown],
            outputs=[delete_result, character_dropdown],
        )

        # Migrate tab
        migrate_text_btn.click(
            migrate_from_text,
            inputs=[migrate_text_input, migrate_text_name],
            outputs=[migrate_text_result, character_dropdown],
        )
        migrate_file_btn.click(
            migrate_from_file,
            inputs=[migrate_file_input, migrate_file_name],
            outputs=[migrate_file_result, character_dropdown],
        )

    # Open browser — handle WSL specially (use Windows default browser)
    import platform
    import shutil
    import subprocess
    import threading

    def _open_browser():
        import time

        time.sleep(1.5)  # wait for server to start
        url = f"http://127.0.0.1:{port}"

        # User specified "none" — don't open
        if browser and browser.lower() == "none":
            return

        # User specified a browser by name
        if browser and browser.lower() != "auto":
            import webbrowser

            try:
                b = webbrowser.get(browser)
                b.open(url)
                return
            except webbrowser.Error:
                # Try as a direct command
                if shutil.which(browser):
                    subprocess.Popen([browser, url])
                    return
                print(f"  Browser '{browser}' not found, using auto-detect.")

        # Auto-detect: WSL → Windows default browser
        if (
            "microsoft" in platform.uname().release.lower()
            or "wsl" in platform.uname().release.lower()
        ):
            if shutil.which("wslview"):
                subprocess.Popen(["wslview", url])
                return
            try:
                subprocess.Popen(["cmd.exe", "/c", "start", url])
                return
            except FileNotFoundError:
                pass

        # macOS
        if platform.system() == "Darwin":
            subprocess.Popen(["open", url])
            return

        # Linux with xdg-open
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", url])
            return

        # Fallback
        import webbrowser

        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()
    app.launch(server_port=port, share=False, inbrowser=False)
