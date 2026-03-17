"""CLI — interactive character management and chat."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import Engine
from .llm.ollama import OllamaLLM
from .embedding.ollama import OllamaEmbedding


DEFAULT_DB = str(Path.home() / ".woven_imprint" / "characters.db")


def _get_engine(db_path: str | None = None, model: str = "qwen3-coder:30b") -> Engine:
    db = db_path or DEFAULT_DB
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    return Engine(
        db_path=db,
        llm=OllamaLLM(model=model, num_ctx=8192),
        embedding=OllamaEmbedding(model="nomic-embed-text"),
    )


def _get_db_only_engine(db_path: str | None = None) -> Engine:
    """Engine for DB-only operations (list, stats) — no LLM needed."""
    from .storage.sqlite import SQLiteStorage

    db = db_path or DEFAULT_DB
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    # Use a dummy engine that only needs storage
    engine = object.__new__(Engine)
    engine.storage = SQLiteStorage(db)
    engine.llm = None  # type: ignore[assignment]
    engine.embedder = None  # type: ignore[assignment]
    return engine


def cmd_demo(args):
    """Run the interactive demo with a pre-built character."""
    engine = _get_engine(args.db, args.model)

    # Check if demo character exists
    chars = engine.list_characters()
    demo_char = next((c for c in chars if c["name"] == "Alice Blackwood"), None)

    if demo_char:
        alice = engine.load_character(demo_char["id"])
    else:
        alice = engine.create_character(
            name="Alice Blackwood",
            birthdate="1998-03-15",
            persona={
                "backstory": (
                    "A sharp-witted private detective in London who left the Metropolitan "
                    "Police after her partner was killed during an undercover operation. "
                    "She now works alone from a cramped office above a chip shop in Brixton."
                ),
                "personality": "witty, skeptical, observant, secretly lonely, dark humor",
                "speaking_style": "clipped sentences, dry humor, avoids emotional topics",
                "occupation": "private investigator",
            },
        )
        print(f"Created character: {alice.name}")

    print(f"\n{'=' * 50}")
    print("  WOVEN IMPRINT — Interactive Demo")
    print(f"  Character: {alice.name} (age {alice.persona.age})")
    print("  Type 'quit' to exit, 'stats' for character info")
    print(f"{'=' * 50}\n")

    _chat_loop(alice, engine)


def cmd_create(args):
    """Create a new character interactively."""
    engine = _get_engine(args.db, args.model)

    name = args.name
    print(f"Creating character: {name}")
    print("Answer the following (press Enter to skip):\n")

    backstory = input("  Backstory: ").strip()
    personality = input("  Personality: ").strip()
    speaking_style = input("  Speaking style: ").strip()
    birthdate = input("  Birthdate (YYYY-MM-DD): ").strip()

    persona = {}
    if backstory:
        persona["backstory"] = backstory
    if personality:
        persona["personality"] = personality
    if speaking_style:
        persona["speaking_style"] = speaking_style

    char = engine.create_character(
        name=name,
        persona=persona,
        birthdate=birthdate or None,
    )

    print(f"\nCreated: {char.name} (id: {char.id})")
    engine.close()


def cmd_chat(args):
    """Chat with an existing character."""
    engine = _get_engine(args.db, args.model)

    # Find character by name or ID
    chars = engine.list_characters()
    match = None
    query = args.character.lower()
    for c in chars:
        if c["id"] == args.character or c["name"].lower().startswith(query):
            match = c
            break

    if not match:
        print(f"Character not found: {args.character}")
        print("Available characters:")
        for c in chars:
            print(f"  {c['name']} (id: {c['id'][:12]})")
        engine.close()
        return

    char = engine.load_character(match["id"])
    print(f"\n  Chatting with {char.name}")
    print("  Type 'quit' to exit, 'stats' for info\n")
    _chat_loop(char, engine)


def cmd_list(args):
    """List all characters."""
    engine = _get_db_only_engine(args.db)
    chars = engine.list_characters()
    if not chars:
        print("No characters. Create one with: woven-imprint create 'Name'")
    else:
        print(f"{'Name':<30} {'ID':<15} {'Created'}")
        print("-" * 60)
        for c in chars:
            print(f"{c['name']:<30} {c['id'][:12]:<15} {c.get('created_at', '?')}")
    engine.close()


def cmd_stats(args):
    """Show character statistics."""
    engine = _get_db_only_engine(args.db)
    chars = engine.list_characters()
    query = args.character.lower()
    match = next(
        (c for c in chars if c["id"] == args.character or c["name"].lower().startswith(query)),
        None,
    )
    if not match:
        print(f"Character not found: {args.character}")
        engine.close()
        return

    char = engine.load_character(match["id"])
    print(f"\n  {char.name}")
    print(f"  Age: {char.persona.age}")
    print(f"  Emotion: {char.emotion.mood} (intensity {char.emotion.intensity:.1f})")
    print(f"  Arc phase: {char.arc.current_phase.value} (tension {char.arc.tension:.1f})")
    print("\n  Memories:")
    print(f"    Buffer: {char.memory.count('buffer')}")
    print(f"    Core:   {char.memory.count('core')}")
    print(f"    Bedrock:{char.memory.count('bedrock')}")

    rels = char.relationships.get_all()
    if rels:
        print("\n  Relationships:")
        for r in rels:
            d = r["dimensions"]
            print(
                f"    {r['target_id'][:15]}: trust={d['trust']:.2f} "
                f"affection={d['affection']:.2f} familiarity={d['familiarity']:.2f}"
            )

    if char.arc.beats:
        print(f"\n  Story beats: {len(char.arc.beats)}")
        for b in char.arc.beats[-3:]:
            print(f"    [{b.phase.value}] {b.description[:60]}")

    engine.close()


def cmd_export(args):
    """Export a character to JSON."""
    engine = _get_engine(args.db)
    chars = engine.list_characters()
    query = args.character.lower()
    match = next(
        (c for c in chars if c["id"] == args.character or c["name"].lower().startswith(query)),
        None,
    )
    if not match:
        print(f"Character not found: {args.character}")
        engine.close()
        return

    char = engine.load_character(match["id"])
    output = args.output or f"{char.name.lower().replace(' ', '_')}.json"
    char.export(output)
    print(f"Exported {char.name} to {output}")
    engine.close()


def cmd_delete(args):
    """Delete a character."""
    engine = _get_db_only_engine(args.db)
    chars = engine.list_characters()
    query = args.character.lower()
    match = next(
        (c for c in chars if c["id"] == args.character or c["name"].lower().startswith(query)),
        None,
    )
    if not match:
        print(f"Character not found: {args.character}")
        engine.close()
        return

    name = match["name"]
    if not args.yes:
        confirm = input(f"Delete {name} and all their data? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            engine.close()
            return

    engine.storage.delete_character(match["id"])
    print(f"Deleted: {name}")
    engine.close()


def cmd_import(args):
    """Import a character from JSON."""
    engine = _get_engine(args.db, args.model)
    char = engine.import_character(args.path)
    print(f"Imported: {char.name} (id: {char.id})")
    engine.close()


def cmd_serve(args):
    """Start the OpenAI-compatible API server."""
    from .server.api import run_server

    run_server(port=args.port, db_path=args.db, model=args.model)


def _chat_loop(char, engine):
    """Interactive chat REPL with live system feedback."""
    user_id = "cli_user"

    try:
        while True:
            try:
                user_input = input("  You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break

            if user_input.lower() == "stats":
                _print_live_stats(char)
                continue
            if user_input.lower() == "reflect":
                print(f"\n  [{char.name} reflects...]")
                reflection = char.reflect()
                print(f"  {reflection}\n")
                continue
            if user_input.lower() == "memories":
                mems = char.recall(input("  Search: ").strip(), limit=5)
                for m in mems:
                    print(f"    [{m['tier']}] {m['content'][:100]}")
                print()
                continue

            # Chat
            response = char.chat(user_input, user_id=user_id)
            print(f"\n  {char.name}: {response}\n")

            # Show live feedback (subtle)
            _print_subtle_feedback(char, user_id)

    except KeyboardInterrupt:
        print("\n")

    # End session
    summary = char.end_session()
    if summary:
        print(f"\n  [Session summary: {summary[:150]}...]")
    engine.close()


def _print_live_stats(char):
    """Print current character state."""
    print(f"\n  --- {char.name} ---")
    print(f"  Emotion: {char.emotion.mood} ({char.emotion.intensity:.1f})")
    print(f"  Arc: {char.arc.current_phase.value} (tension {char.arc.tension:.1f})")
    buf = char.memory.count("buffer")
    core = char.memory.count("core")
    bed = char.memory.count("bedrock")
    print(f"  Memory: {buf} buffer, {core} core, {bed} bedrock")

    rels = char.relationships.get_all()
    for r in rels:
        d = r["dimensions"]
        print(
            f"  Rel: trust={d['trust']:.2f} affection={d['affection']:.2f} "
            f"fam={d['familiarity']:.2f} tension={d['tension']:.2f}"
        )
    print()


def _print_subtle_feedback(char, user_id):
    """Show minimal system feedback after each turn."""
    parts = []

    # Emotion
    if char.emotion.mood != "neutral" and char.emotion.intensity > 0.3:
        parts.append(f"mood:{char.emotion.mood}")

    # Relationship change
    rel = char.relationships.get(user_id)
    if rel:
        d = rel["dimensions"]
        fam = d.get("familiarity", 0)
        if fam > 0:
            parts.append(f"fam:{fam:.2f}")

    # Story beat
    if char.arc.beats:
        latest = char.arc.beats[-1]
        if latest.turn_number == char.arc.turn_count:
            parts.append(f"beat:{latest.tags[0] if latest.tags else latest.phase.value}")

    if parts:
        print(f"  [{' | '.join(parts)}]\n")


def main():
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="woven-imprint",
        description="Woven Imprint — Persistent Character Infrastructure",
    )
    parser.add_argument("--version", action="version", version=f"woven-imprint {__version__}")
    parser.add_argument(
        "--db", default=None, help="Database path (default: ~/.woven_imprint/characters.db)"
    )
    import os

    default_model = os.environ.get("WOVEN_IMPRINT_MODEL", "llama3.2")
    parser.add_argument(
        "--model", default=default_model, help="Ollama model name (env: WOVEN_IMPRINT_MODEL)"
    )

    sub = parser.add_subparsers(dest="command")

    # demo
    sub.add_parser("demo", help="Interactive demo with a pre-built character")

    # create
    p_create = sub.add_parser("create", help="Create a new character")
    p_create.add_argument("name", help="Character name")

    # chat
    p_chat = sub.add_parser("chat", help="Chat with an existing character")
    p_chat.add_argument("character", help="Character name or ID")

    # list
    sub.add_parser("list", help="List all characters")

    # stats
    p_stats = sub.add_parser("stats", help="Show character statistics")
    p_stats.add_argument("character", help="Character name or ID")

    # export
    p_export = sub.add_parser("export", help="Export character to JSON")
    p_export.add_argument("character", help="Character name or ID")
    p_export.add_argument("-o", "--output", help="Output file path")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a character")
    p_delete.add_argument("character", help="Character name or ID")
    p_delete.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # import
    p_import = sub.add_parser("import", help="Import character from JSON")
    p_import.add_argument("path", help="Path to exported JSON file")

    # serve
    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible API server")
    p_serve.add_argument("--port", type=int, default=8650)

    args = parser.parse_args()

    commands = {
        "demo": cmd_demo,
        "create": cmd_create,
        "chat": cmd_chat,
        "list": cmd_list,
        "stats": cmd_stats,
        "export": cmd_export,
        "delete": cmd_delete,
        "import": cmd_import,
        "serve": cmd_serve,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
