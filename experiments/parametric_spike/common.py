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
    """OpenAILLM pointed at vllm-brain, with vLLM's reasoning template disabled.

    vllm-brain (Qwen/Qwen3.5-35B-A3B-FP8) is a thinking model that streams its
    chain-of-thought straight into `message.content` unless the request
    carries `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
    (see gen_persona_corpus.py's `_chat_no_thinking` for the empirical
    writeup). Fixed here at the source so every spike script gets a clean
    brain response via the normal .generate()/.generate_json_robust() API.
    """
    import re

    from woven_imprint.llm.openai_llm import OpenAILLM, _merge_system_messages
    from woven_imprint.llm.resilience import resilient_call

    class NoThinkOpenAILLM(OpenAILLM):
        """OpenAILLM that disables the vLLM reasoning template (Qwen3.5 leaks CoT into content otherwise)."""

        _EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}

        def generate(self, messages, temperature=0.7, max_tokens=2048):
            messages = _merge_system_messages(messages)
            response = resilient_call(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self._EXTRA,
                provider_name="openai",
            )
            return response.choices[0].message.content or ""

        def generate_json(self, messages, temperature=0.3):
            messages = _merge_system_messages(messages)
            try:
                response = resilient_call(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    extra_body=self._EXTRA,
                    provider_name="openai",
                )
                raw = response.choices[0].message.content or "{}"
                return json.loads(raw)
            except Exception:
                # Fallback: regular generation + parse (mirrors OpenAILLM.generate_json)
                raw = self.generate(messages, temperature=temperature)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    pass
                for match in re.finditer(r"\{.*\}", raw, re.DOTALL):
                    try:
                        return json.loads(match.group())
                    except json.JSONDecodeError:
                        continue
                raise ValueError(f"Could not parse JSON: {raw[:200]}")

    return NoThinkOpenAILLM(model=BRAIN_MODEL, api_key="local", base_url=BRAIN_URL, timeout=300)


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
