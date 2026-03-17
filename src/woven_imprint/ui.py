"""Simple web UI for trying Woven Imprint — powered by Gradio.

Launch with: woven-imprint ui
Opens a browser tab with character selection, chat, and stats.
"""

from __future__ import annotations


def launch(db_path: str | None = None, model: str = "llama3.2", port: int = 7860):
    """Launch the Gradio web interface."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is required for the UI. Install it with:")
        print("  pip install woven-imprint[ui]")
        return

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

    # Track active character per session
    _active: dict = {"char": None}

    def get_character_list():
        chars = engine.list_characters()
        if not chars:
            return ["No characters — create one below"]
        return [f"{c['name']} ({c['id'][:8]})" for c in chars]

    def select_character(selection):
        if not selection or "No characters" in selection:
            _active["char"] = None
            return "No character selected.", "", ""

        char_id = selection.split("(")[-1].rstrip(")")
        chars = engine.list_characters()
        match = next((c for c in chars if c["id"].startswith(char_id)), None)
        if not match:
            return "Character not found.", "", ""

        char = engine.load_character(match["id"])
        _active["char"] = char

        # Build info
        info = f"**{char.name}**"
        if char.persona.age:
            info += f" (age {char.persona.age})"
        info += f"\n\n{char.persona.soft.get('personality', '')}"
        if char.persona.backstory:
            info += f"\n\n*{char.persona.backstory[:200]}*"

        # Stats
        stats = (
            f"Buffer: {char.memory.count('buffer')} | "
            f"Core: {char.memory.count('core')} | "
            f"Bedrock: {char.memory.count('bedrock')}\n"
            f"Emotion: {char.emotion.mood} ({char.emotion.intensity:.1f})"
        )

        # Relationships
        rels = char.relationships.get_all()
        rel_text = ""
        for r in rels:
            d = r["dimensions"]
            rel_text += (
                f"**{r['target_id'][:15]}** ({r.get('type', '?')}): "
                f"trust {d.get('trust', 0):.2f}, "
                f"affection {d.get('affection', 0):.2f}, "
                f"familiarity {d.get('familiarity', 0):.2f}\n"
            )
        if not rel_text:
            rel_text = "No relationships yet."

        return info, stats, rel_text

    def chat(message, history):
        char = _active.get("char")
        if not char:
            return "Select a character first."

        response = char.chat(message, user_id="ui_user")
        return response

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
        return f"Created **{char.name}**!", gr.update(choices=get_character_list(), value=f"{char.name} ({char.id[:8]})")

    def migrate_text(text, name):
        if not text.strip():
            return "Paste the character instructions/persona text.", gr.update()

        from .migrate import CharacterImporter

        importer = CharacterImporter(engine)
        char = importer.from_text(text.strip(), name=name.strip() or None)
        _active["char"] = char

        result = f"Migrated **{char.name}**\n"
        result += f"- Personality: {char.persona.soft.get('personality', '?')[:80]}\n"
        result += f"- Memories: {char.memory.count('core')} core\n"

        rel = char.relationships.get("imported_user")
        if rel:
            d = rel["dimensions"]
            result += f"- Relationship: trust={d.get('trust', 0):.2f}, familiarity={d.get('familiarity', 0):.2f}"

        return result, gr.update(choices=get_character_list(), value=f"{char.name} ({char.id[:8]})")

    def refresh_stats():
        char = _active.get("char")
        if not char:
            return "", ""

        stats = (
            f"Buffer: {char.memory.count('buffer')} | "
            f"Core: {char.memory.count('core')} | "
            f"Bedrock: {char.memory.count('bedrock')}\n"
            f"Emotion: {char.emotion.mood} ({char.emotion.intensity:.1f})\n"
            f"Arc: {char.arc.current_phase.value} (tension {char.arc.tension:.1f})"
        )

        rels = char.relationships.get_all()
        rel_text = ""
        for r in rels:
            d = r["dimensions"]
            rel_text += (
                f"**{r['target_id'][:15]}** ({r.get('type', '?')}): "
                f"trust {d.get('trust', 0):.2f}, "
                f"affection {d.get('affection', 0):.2f}, "
                f"familiarity {d.get('familiarity', 0):.2f}\n"
            )
        if not rel_text:
            rel_text = "No relationships yet."

        return stats, rel_text

    # Build the UI
    with gr.Blocks(title="Woven Imprint", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Woven Imprint\n*Persistent Character Infrastructure*")

        with gr.Row():
            with gr.Column(scale=2):
                # Chat
                character_dropdown = gr.Dropdown(
                    choices=get_character_list(),
                    label="Character",
                    interactive=True,
                )
                char_info = gr.Markdown("Select a character to start chatting.")
                gr.ChatInterface(
                    fn=chat,
                    type="messages",
                )

            with gr.Column(scale=1):
                # Stats
                gr.Markdown("### Stats")
                stats_display = gr.Markdown("")
                rel_display = gr.Markdown("")
                refresh_btn = gr.Button("Refresh Stats")

                # Create
                gr.Markdown("### Create Character")
                create_name = gr.Textbox(label="Name", placeholder="Marcus the Blacksmith")
                create_personality = gr.Textbox(label="Personality", placeholder="gruff but kind, dry humor")
                create_backstory = gr.Textbox(label="Backstory", placeholder="A village blacksmith...", lines=2)
                create_style = gr.Textbox(label="Speaking Style", placeholder="short sentences, working-class")
                create_birthdate = gr.Textbox(label="Birthdate", placeholder="YYYY-MM-DD (optional)")
                create_btn = gr.Button("Create", variant="primary")
                create_result = gr.Markdown("")

                # Migrate
                gr.Markdown("### Migrate Character")
                migrate_text_input = gr.Textbox(
                    label="Paste instructions or persona",
                    placeholder="You are Coach Rivera, a retired soccer coach...",
                    lines=4,
                )
                migrate_name_input = gr.Textbox(label="Name (optional)", placeholder="Auto-detected")
                migrate_btn = gr.Button("Migrate")
                migrate_result = gr.Markdown("")

        # Events
        character_dropdown.change(
            select_character, inputs=[character_dropdown], outputs=[char_info, stats_display, rel_display]
        )
        refresh_btn.click(refresh_stats, outputs=[stats_display, rel_display])
        create_btn.click(
            create_character,
            inputs=[create_name, create_personality, create_backstory, create_style, create_birthdate],
            outputs=[create_result, character_dropdown],
        )
        migrate_btn.click(
            migrate_text, inputs=[migrate_text_input, migrate_name_input], outputs=[migrate_result, character_dropdown]
        )

    app.launch(server_port=port, share=False)
