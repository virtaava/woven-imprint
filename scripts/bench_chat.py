"""Scripted chat benchmark — prints per-phase latency percentiles.

Usage (against any configured provider, e.g. local Ollama or an
OpenAI-compatible endpoint):

    WOVEN_IMPRINT_LLM_PROVIDER=ollama WOVEN_IMPRINT_MODEL=llama3.2 \
        python scripts/bench_chat.py --turns 12

Writes raw per-turn metrics to bench_metrics.jsonl next to the script
and prints a summary table. Use --db to persist between runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from woven_imprint import Engine

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
    args = parser.parse_args()

    engine = Engine(db_path=args.db)
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
        char.chat(msg, user_id="toni")
        if hasattr(char, "flush"):
            char.flush()  # include background cost in wall-time comparisons
        rows.append(dict(char.last_chat_metrics))
        print(f"turn {i + 1}: total={char.last_chat_metrics.get('total_ms', 0):.0f}ms")

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
