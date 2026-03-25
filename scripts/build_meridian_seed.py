#!/usr/bin/env python3
"""Build the Meridian seed database from documentation.

This script is run once by the developers (not by end users).
It reads documentation files, chunks them into memory-sized pieces,
creates a Meridian character, and stores the doc chunks as bedrock
memories in a seed SQLite database.

Usage:
    python scripts/build_meridian_seed.py [--model MODEL] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_seed(output_path: str, model: str | None = None) -> None:
    """Build the Meridian seed database."""
    from woven_imprint.data.meridian_persona import MERIDIAN_PERSONA, MERIDIAN_BIRTHDATE
    from woven_imprint.engine import Engine
    from woven_imprint.providers import create_llm, create_embedding
    from woven_imprint.config import WovenConfig, LLMConfig

    # Read documentation files
    docs_dir = Path(__file__).parent.parent / "docs"
    doc_files = [
        "ARCHITECTURE.md",
        "CONFIGURATION.md",
        "DEVELOPER_GUIDE.md",
        "GETTING_STARTED.md",
    ]

    all_chunks = []
    for fname in doc_files:
        fpath = docs_dir / fname
        if fpath.exists():
            text = fpath.read_text()
            chunks = chunk_text(text)
            print(f"  {fname}: {len(chunks)} chunks")
            all_chunks.extend(chunks)

    if not all_chunks:
        print("No documentation found to seed.")
        return

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Create engine and character
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    # Create LLM with optional model override
    llm_kwargs = {}
    if model:
        llm_kwargs["model"] = model

    engine = Engine(
        db_path=str(output),
        llm=create_llm(**llm_kwargs) if not llm_kwargs else create_llm(),
        embedding=create_embedding(),
    )
    char = engine.create_character(
        name="Meridian",
        persona=MERIDIAN_PERSONA,
        birthdate=MERIDIAN_BIRTHDATE,
    )

    # Add doc chunks as bedrock memories
    for i, chunk in enumerate(all_chunks):
        char.memory.add(
            content=f"[Documentation] {chunk}",
            tier="bedrock",
            importance=0.85,
        )
        if (i + 1) % 10 == 0:
            print(f"  Added {i + 1}/{len(all_chunks)} memories...")

    print(f"\nSeed database saved to: {output}")
    print(f"Character: Meridian (ID: {char.id})")
    print(f"Memories: {len(all_chunks)} bedrock + persona seeds")
    engine.close()


def main():
    parser = argparse.ArgumentParser(description="Build Meridian seed database")
    parser.add_argument("--model", default=None, help="LLM model to use for embeddings")
    parser.add_argument(
        "--output",
        default="src/woven_imprint/data/meridian_seed.db",
        help="Output path for seed database",
    )
    args = parser.parse_args()
    build_seed(args.output, args.model)


if __name__ == "__main__":
    main()
