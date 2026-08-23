"""Shared paths, constants and provider factories for the parametric spike."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
REPO_DIR = SPIKE_DIR.parents[1]
OUT_DIR = SPIKE_DIR / "out"
DATA_DIR = SPIKE_DIR / "data"

sys.path.insert(0, str(REPO_DIR / "src"))

HF_DIR = Path.home() / "models" / "qwen3-4b-hf"
GGUF_DIR = Path.home() / "models" / "qwen3-4b-gguf"
GGUF_PATH = GGUF_DIR / "Qwen3-4B-Q8_0.gguf"

BRAIN_URL = "http://127.0.0.1:11800/v1"
BRAIN_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
EMBED_URL = "http://127.0.0.1:11801/v1"
SPIKE_PORT = 11810
SPIKE_URL = f"http://127.0.0.1:{SPIKE_PORT}/v1"

MINIMAL_SYSTEM = "You are Meridian."


def full_system_prompt() -> str:
    """Build Meridian's full production-style system prompt.

    MERIDIAN_PERSONA stores personality/speaking_style/hard_constraints as
    flat top-level keys, but PersonaModel only reads them from nested
    hard/soft/temporal dicts. Nest them here (mirroring the shorthand
    normalization woven_imprint.engine.create_character does for
    personality/speaking_style) so the hard constraints are included too.
    """
    from woven_imprint.data.meridian_persona import MERIDIAN_BIRTHDATE, MERIDIAN_PERSONA
    from woven_imprint.persona.model import PersonaModel

    persona = MERIDIAN_PERSONA
    normalized = {
        "name": persona.get("name", "Meridian"),
        "backstory": persona.get("backstory", ""),
        "hard": {"hard_constraints": persona.get("hard_constraints", "")},
        "soft": {
            "personality": persona.get("personality", ""),
            "speaking_style": persona.get("speaking_style", ""),
        },
    }
    return PersonaModel(normalized, MERIDIAN_BIRTHDATE).build_system_prompt()


def brain_llm():
    from woven_imprint.llm.openai_llm import OpenAILLM

    return OpenAILLM(model=BRAIN_MODEL, api_key="local", base_url=BRAIN_URL, timeout=300)


def spike_llm():
    from woven_imprint.llm.openai_llm import OpenAILLM

    return OpenAILLM(model="qwen3-4b", api_key="local", base_url=SPIKE_URL, timeout=300)


def spike_embedding():
    from woven_imprint.embedding.openai_embedding import OpenAIEmbedding

    return OpenAIEmbedding(model="nomic-embed-text", api_key="local", base_url=EMBED_URL)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
