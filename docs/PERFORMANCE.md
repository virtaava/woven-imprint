# Performance

Per-turn latency is measured by the instrumentation in `Character.chat()`
(`last_chat_metrics`) and collected via `scripts/bench_chat.py` or the
JSONL sink (`WOVEN_IMPRINT_METRICS_PATH`).

## How to run

    WOVEN_IMPRINT_LLM_PROVIDER=ollama WOVEN_IMPRINT_MODEL=<model> \
        python scripts/bench_chat.py --turns 12

## Baseline (pre-Phase-A)

**mock baseline — re-run against live model.** Recorded 2026-07-11 on DGX
Spark (GB10 Grace Blackwell, aarch64, 20 vCPU, DGX OS). A live OpenAI-compatible
chat endpoint is reachable at `http://127.0.0.1:11800/v1`
(`Qwen/Qwen3.5-35B-A3B-FP8`, served via vLLM), but it does not expose
`/v1/embeddings` — confirmed with a direct `curl` (`404 Not Found`). The
project's `LLMConfig` shares a single `base_url` between the LLM and
embedding providers, and no local Ollama instance is running (disabled
per infra convention), so there is no way to point the LLM at
`:11800` and embeddings at the separate `llama-embed` service
(`:11801`, nomic-embed-text) through env vars alone without code changes.
Per the task brief, no code changes were made to force this; instead this
table was generated with `FakeLLM`/`FakeEmbedder` (see `tests/helpers.py`)
driving the same 12 scripted turns as `scripts/bench_chat.py`, run
sequentially (`parallel=False`) against an in-memory SQLite DB.

    phase                              p50       p95       max
    build_context_ms                     0         0         0
    consistency_ms                       0         0         0
    conversation_buffer_ms               0         0         0
    generate_ms                          0         0         0
    maintenance_ms                       0         0         0
    relationship_context_ms              0         0         0
    retrieve_memories_ms                 0         1         1
    start_session_ms                     0         0         0
    store_response_memory_ms             0         0         0
    store_user_memory_ms                 0         0         0
    subsystems_ms                        0         0         0
    total_ms                             1         1         1

These numbers reflect in-process overhead only (no real LLM/embedding
network latency) — they are useful for spotting regressions in the
non-LLM code paths (context building, memory storage, subsystem
orchestration) but are not representative of real end-to-end latency.
Live numbers (against `vllm-brain` for chat and `llama-embed` for
embeddings, once the provider config supports independent base URLs)
get recorded here in Task 11.

## After Phase A

<filled in Task 11>
