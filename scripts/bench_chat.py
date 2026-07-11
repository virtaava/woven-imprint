"""Scripted chat benchmark — prints per-phase latency percentiles.

Usage (against any configured provider, e.g. local Ollama or an
OpenAI-compatible endpoint):

    WOVEN_IMPRINT_LLM_PROVIDER=ollama WOVEN_IMPRINT_MODEL=llama3.2 \
        python scripts/bench_chat.py --turns 12

Or point chat and embeddings at two different OpenAI-compatible
endpoints explicitly (bypasses config, which only has one `base_url`
shared between LLM and embedding providers):

    python scripts/bench_chat.py --turns 12 \
        --llm-base-url http://127.0.0.1:11800/v1 --llm-model my-chat-model \
        --embed-base-url http://127.0.0.1:11801/v1 --embed-model my-embed-model \
        --api-key sk-local

Writes raw per-turn metrics to bench_metrics.jsonl next to the script
and prints a summary table. Use --db to persist between runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from woven_imprint import Engine
from woven_imprint.llm.base import LLMProvider


class _MaxTokensCapLLM(LLMProvider):
    """Wraps an LLMProvider, optionally capping max_tokens on generation calls.

    Qwen3.5-35B-A3B-FP8 emits long "thinking" prose inline in the response
    content (no reasoning-parser separation configured on this deployment),
    which at the default max_tokens=2048 can produce responses long enough
    to overflow the embedding server's physical batch size when the
    response is stored as a memory. `max_tokens_cap` bounds generation
    length to keep the pipeline exercised realistically without that
    downstream failure.

    System-message merging (multiple leading system-role messages coalesced
    into one, required by strict OpenAI-compatible servers like vLLM's
    chat template) now happens inside the production provider itself — see
    `woven_imprint.llm.openai_llm._merge_system_messages` — so this bench
    talks to the same message-construction path production does. This shim
    only handles the max_tokens cap, which is bench/model-specific and
    doesn't belong in production generation.
    """

    def __init__(self, inner, max_tokens_cap: int | None = None):
        self.inner = inner
        self.max_tokens_cap = max_tokens_cap

    def _cap(self, max_tokens: int) -> int:
        if self.max_tokens_cap is None:
            return max_tokens
        return min(max_tokens, self.max_tokens_cap)

    def generate(self, messages, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        return self.inner.generate(
            messages, temperature=temperature, max_tokens=self._cap(max_tokens)
        )

    def generate_stream(self, messages, temperature: float = 0.7, max_tokens: int = 2048):
        return self.inner.generate_stream(
            messages, temperature=temperature, max_tokens=self._cap(max_tokens)
        )

    def generate_json(self, messages, temperature: float = 0.3):
        return self.inner.generate_json(messages, temperature=temperature)


SCRIPTED_TURNS = [
    "Hi! I'm Toni. I just moved to a small town by a lake.",
    "I work as a software developer, mostly on games.",
    "My sister Anna visits every summer. She loves rowing.",
    "Yesterday I found a strange old key in the boathouse.",
    "Do you remember what my sister's name is?",
    "The key had a carved raven on it. What do you make of that?",
    "I also adopted a cat last week. Her name is Viima.",
    "What do you remember about me so far?",
    "I'm nervous about a job interview on Friday.",
    "Tell me something you're curious about.",
    "The interview is for a lead developer role.",
    "Good night! Talk tomorrow.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=len(SCRIPTED_TURNS))
    parser.add_argument("--db", default=":memory:")
    parser.add_argument("--out", default=str(Path(__file__).parent / "bench_metrics.jsonl"))
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible base URL for chat. If set (with --llm-model), "
        "bypasses config-driven provider selection and constructs OpenAILLM directly.",
    )
    parser.add_argument("--llm-model", default=None, help="Chat model name for --llm-base-url.")
    parser.add_argument(
        "--embed-base-url",
        default=None,
        help="OpenAI-compatible base URL for embeddings. If set (with --embed-model), "
        "bypasses config-driven provider selection and constructs OpenAIEmbedding directly.",
    )
    parser.add_argument(
        "--embed-model", default=None, help="Embedding model name for --embed-base-url."
    )
    parser.add_argument(
        "--api-key",
        default="not-needed",
        help="API key for the explicit --llm-base-url/--embed-base-url providers.",
    )
    parser.add_argument(
        "--max-tokens-cap",
        type=int,
        default=400,
        help="Cap generate()/generate_stream() max_tokens when using --llm-base-url "
        "(protects against reasoning models producing responses too long for the "
        "embedding server's batch size when stored as memory). 0 disables the cap.",
    )
    args = parser.parse_args()

    llm = None
    embedding = None
    if args.llm_base_url and args.llm_model:
        from woven_imprint.llm.openai_llm import OpenAILLM

        llm = _MaxTokensCapLLM(
            OpenAILLM(model=args.llm_model, base_url=args.llm_base_url, api_key=args.api_key),
            max_tokens_cap=args.max_tokens_cap or None,
        )
    if args.embed_base_url and args.embed_model:
        from woven_imprint.embedding.openai_embedding import OpenAIEmbedding

        embedding = OpenAIEmbedding(
            model=args.embed_model, base_url=args.embed_base_url, api_key=args.api_key
        )

    engine = Engine(db_path=args.db, llm=llm, embedding=embedding)
    char = engine.create_character(
        "Benchling",
        persona={
            "backstory": "A curious lakeside spirit who remembers every visitor.",
            "personality": "warm, observant",
            "speaking_style": "short sentences",
        },
    )

    rows: list[dict] = []
    for i in range(args.turns):
        msg = SCRIPTED_TURNS[i % len(SCRIPTED_TURNS)]
        turn_wall_started = time.perf_counter()
        char.chat(msg, user_id="toni")
        if hasattr(char, "flush"):
            char.flush()  # include background cost in wall-time comparisons
        wall_ms = round((time.perf_counter() - turn_wall_started) * 1000.0, 2)
        row = dict(char.last_chat_metrics)
        # wall_ms is the honest total cost of the turn: chat() (which for
        # background=True returns after generate+consistency only) plus the
        # flush() above, which waits for any bookkeeping still in flight.
        # For background=False, subsystems already ran inside chat(), so
        # wall_ms and total_ms should be ~equal (flush is a no-op).
        row["wall_ms"] = wall_ms
        rows.append(row)
        print(
            f"turn {i + 1}: total={char.last_chat_metrics.get('total_ms', 0):.0f}ms "
            f"wall(incl. flush)={wall_ms:.0f}ms"
        )

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    keys = sorted({k for r in rows for k in r if k.endswith("_ms")})
    print(f"\n{'phase':<28}{'p50':>10}{'p95':>10}{'max':>10}")
    for key in keys:
        vals = sorted(r.get(key, 0.0) for r in rows)
        p50 = statistics.median(vals)
        p95 = vals[max(0, int(len(vals) * 0.95) - 1)]
        print(f"{key:<28}{p50:>10.0f}{p95:>10.0f}{vals[-1]:>10.0f}")


if __name__ == "__main__":
    main()
