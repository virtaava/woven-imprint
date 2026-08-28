"""Shared dataclasses, paths, and provider constructors for eval/external.

Dataset-agnostic shapes that ``locomo.py``/``longmemeval.py`` normalize into,
and that the Task 3 runner consumes:

- ``Turn``/``Session``/``Question``/``Conversation`` — one message-level
  ingestion unit per benchmark (LoCoMo, LongMemEval-S).
- ``Probe`` — a LoCoMo-Plus Cognitive cue/trigger item stitched onto a base
  LoCoMo ``Conversation`` (see ``locomo.load_locomo_plus``).

All datetimes are timezone-aware UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from woven_imprint.llm.openai_llm import OpenAILLM

# eval/external/common.py -> eval/external -> eval -> repo root
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).parent / "data"
RUNS_DIR = Path(__file__).parent / "runs"
RESULTS_DIR = ROOT / "eval" / "results"

BRAIN_URL = "http://127.0.0.1:11800/v1"
BRAIN_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
EMBED_URL = "http://127.0.0.1:11801/v1"
EMBED_MODEL = "nomic-embed-text"
# vLLM's chat template streams Qwen3.5's chain-of-thought straight into
# message.content unless this is forwarded on every request.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass
class Turn:
    speaker: str
    role: str  # "user" | "assistant"
    text: str
    dia_id: str | None
    at: datetime  # aware UTC


@dataclass
class Session:
    session_id: str
    at: datetime
    turns: list[Turn]


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    category: str
    evidence: list[str]
    asked_at: datetime
    kind: str  # "qa" | "adversarial" | "abstain"


@dataclass
class Conversation:
    conv_id: str
    user_name: str
    character_name: str
    sessions: list[Session]
    questions: list[Question]
    transcript_text: str


@dataclass
class Probe:
    """A LoCoMo-Plus Cognitive cue/trigger item stitched onto a base conversation.

    ``cue`` is a synthetic Session built from the probe's ``cue_dialogue``
    (A/B lines mapped to the base conversation's user/character, turns spaced
    30s apart so they sort deterministically); ``cue.at`` is the upstream
    ``cue_time``. ``trigger_text`` is the trigger's A-line(s) only (the
    message the memory-mode harness sends to `Character.chat()`; B-lines, if
    any, are dropped since the product path always drives the trigger as a
    single user message). ``evidence_text`` renders the cue as
    ``f"{speaker}: {text}"`` lines (judge input). ``stitched_text`` is the
    full-context baseline transcript: every base session, then the cue, then
    the trigger, time-ordered, each block headed by ``DATE: ...`` and turns
    as ``{speaker} said, "{text}"`` (mirrors upstream's
    ``_build_conversation_context``/``_stitch_dialogue_for_plus`` in
    xjtuleeyf/Locomo-Plus's data/unified_input.py, combined per this task's
    brief).
    """

    probe_id: str
    relation_type: str
    time_gap: str
    base_conv_id: str
    cue: Session
    trigger_text: str
    trigger_at: datetime
    evidence_text: str
    stitched_text: str


def brain_llm(timeout: int = 300) -> OpenAILLM:
    """OpenAILLM pointed at vllm-brain with vLLM's reasoning template disabled."""
    from woven_imprint.llm.openai_llm import OpenAILLM

    return OpenAILLM(
        model=BRAIN_MODEL, api_key="local", base_url=BRAIN_URL, timeout=timeout, extra_body=NO_THINK
    )


EMBED_MAX_CHARS = 1200  # llama-embed rejects inputs over its 512-token physical batch


class TruncatingEmbedding:
    """EmbeddingProvider wrapper that embeds at most ``max_chars`` of each text.

    The local llama.cpp embedding server refuses inputs longer than its physical batch
    (512 tokens: "input (648 tokens) is too large to process"). LoCoMo turns never hit
    that, but LongMemEval-S turns can be multi-paragraph, so the harness embeds a bounded
    prefix (~300–400 tokens of English). Retrieval over such rows keys on the prefix only;
    the stored memory text is untouched.
    """

    def __init__(self, inner, max_chars: int = EMBED_MAX_CHARS) -> None:
        self._inner = inner
        self.max_chars = max_chars
        self.model = getattr(inner, "model", None)

    def _cut(self, text: str) -> str:
        return text if len(text) <= self.max_chars else text[: self.max_chars]

    def embed(self, text: str) -> list[float]:
        return self._inner.embed(self._cut(text))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_batch([self._cut(t) for t in texts])

    def dimensions(self) -> int:
        return self._inner.dimensions()


def embedder() -> TruncatingEmbedding:
    """llama-embed (nomic-embed-text, 768d) behind a prefix-truncating wrapper."""
    from woven_imprint.embedding.openai_embedding import OpenAIEmbedding

    return TruncatingEmbedding(
        OpenAIEmbedding(model=EMBED_MODEL, api_key="local", base_url=EMBED_URL)
    )


def parse_locomo_datetime(s: str) -> datetime:
    """Parse LoCoMo's session timestamp format, e.g. "1:56 pm on 8 May, 2023"."""
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
