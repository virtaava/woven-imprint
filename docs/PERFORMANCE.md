# Performance

Per-turn latency is measured by the instrumentation in `Character.chat()`
(`last_chat_metrics`) and collected via `scripts/bench_chat.py` or the
JSONL sink (`WOVEN_IMPRINT_METRICS_PATH`).

## How to run

    WOVEN_IMPRINT_LLM_PROVIDER=ollama WOVEN_IMPRINT_MODEL=<model> \
        python scripts/bench_chat.py --turns 12

Or, against two different OpenAI-compatible endpoints (chat and embeddings
on separate hosts/ports — config only supports one shared `base_url`):

    python scripts/bench_chat.py --turns 12 --db :memory: \
        --llm-base-url http://127.0.0.1:11800/v1 --llm-model <chat-model> \
        --embed-base-url http://127.0.0.1:11801/v1 --embed-model <embed-model> \
        --api-key <key>

To approximate pre-Phase-A synchronous bookkeeping on the same hardware for
comparison, add `WOVEN_IMPRINT_BACKGROUND=false` to either invocation.

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

Recorded 2026-07-11 on DGX Spark (GB10 Grace Blackwell, aarch64, 20 vCPU,
DGX OS). Live endpoints, both OpenAI-compatible:

- **Chat**: `http://127.0.0.1:11800/v1`, `Qwen/Qwen3.5-35B-A3B-FP8`, served
  via vLLM Docker (`vllm-brain`).
- **Embeddings**: `http://127.0.0.1:11801/v1`, `nomic-embed-text`, served
  via llama.cpp (`llama-embed`).

Run with `scripts/bench_chat.py --turns 12 --db :memory:`, using the new
`--llm-base-url`/`--embed-base-url` flags added in Task 11 (see "How to
run" above) to point chat and embeddings at these two separate services —
the config's shared `base_url` can't do this, which is why the Task 1
baseline above is a mock.

**Environment caveats specific to this model/deployment (bench-script-only
workarounds, in `scripts/bench_chat.py`, not `src/`):**
1. This vLLM deployment's OpenAI-compatible endpoint rejects requests with
   more than one `system`-role message (`400: "System message must be at
   the beginning."`). `Character._build_context` intentionally sends two
   (a stable persona prefix + a volatile block, for prefix-caching — see
   A4) — production behavior is unchanged. The bench script wraps the LLM
   provider to merge them before sending.
2. This deployment has no reasoning-parser configured, so
   Qwen3.5-35B-A3B-FP8's chain-of-thought is emitted inline as response
   text rather than separated. At the default `max_tokens=2048` this
   produced responses long enough to overflow the embedding server's
   physical batch size (512) when the response text was embedded as a
   memory. The bench script caps `max_tokens` at 400 (`--max-tokens-cap`)
   to keep the pipeline exercised end-to-end without that failure. This
   caps absolute `generate_ms` lower than an uncapped run would show, but
   does not affect the *relative* before/after comparison below, which is
   about where bookkeeping runs, not response length.

### "After (background on)" — `character.background: true` (config default)

    phase                              p50       p95       max
    build_context_ms                     0         0         0
    consistency_ms                    3180      6471     42471
    conversation_buffer_ms               0         0         0
    generate_ms                       8943      9388      9401
    maintenance_ms                       0         0         1
    relationship_context_ms              0         0         0
    retrieve_memories_ms                 7         9        18
    start_session_ms                     0         0         0
    store_response_memory_ms            23        33        55
    store_user_memory_ms                15        22        43
    subsystems_ms                        0         0         0
    total_ms                         12127     15649     51168
    wall_ms                          16586     55008    142591

`wall_ms` is `chat()` + `flush()` timed by the bench script itself (not
part of `last_chat_metrics`) — the honest total cost of a turn including
bookkeeping that already ran, just off the perceived critical path.

### "Comparison (sync mode)" — `WOVEN_IMPRINT_BACKGROUND=false`, same turns/hardware/model

    phase                              p50       p95       max
    build_context_ms                     0         0         0
    consistency_ms                    2962      5465      9369
    conversation_buffer_ms               0         0         0
    generate_ms                       8910      9317      9319
    maintenance_ms                       0         0         0
    relationship_context_ms              0         0         0
    retrieve_memories_ms                 8         9        17
    start_session_ms                     0         0         0
    store_response_memory_ms            22        24        29
    store_user_memory_ms                19        21        28
    subsystems_ms                     3865     17356    126201
    total_ms                         17074     29126    136883
    wall_ms                          17074     29126    136883

In sync mode `total_ms == wall_ms` (`flush()` is a no-op — nothing was
queued, `subsystems_ms` already includes the blocking bookkeeping cost).
This table approximates what the pre-Phase-A synchronous path would have
made the caller wait for.

### Interpretation

**Acceptance check (brief §5): perceived latency ≈ generate + consistency.**
In background mode, `total_ms` p50 (12127ms) is within 0.03% of
`generate_ms + consistency_ms` p50 (8943 + 3180 = 12123ms) — `chat()`
returns as soon as generation and consistency-checking finish, and
`subsystems_ms` (the bookkeeping *submission*, not execution) is ~0ms as
expected. The same holds at p95 (15649ms actual vs. 9388 + 6471 = 15859ms,
1.3% off — consistent, since p95-of-sums isn't exactly sum-of-p95s across
independent turns). **This confirms the architecture change worked**:
bookkeeping moved off the perceived hot path.

**What the user actually experiences vs. what they would have waited
pre-Phase-A:**

| | background=true (now) | background=false (sync, "before") |
|---|---|---|
| Perceived turn latency (`total_ms` p50) | **12.1s** | 17.1s |
| Perceived turn latency (`total_ms` p95) | 15.6s | 29.1s |
| Honest total cost incl. bookkeeping (`wall_ms` p50) | 16.6s | 17.1s |
| Worst observed turn (`max`) | 51.2s (a consistency-retry outlier) | 136.9s (a subsystems outlier — likely fact-extraction/relationship-assessment calls queued behind a slow LLM response) |

Background mode cuts what the caller perceives as turn latency by ~29% at
p50 and ~46% at p95 versus sync mode on identical hardware/model. The
*total* compute cost per turn (`wall_ms` in background mode) is comparable
to sync mode's `total_ms` — Phase A doesn't make the LLM calls cheaper, it
moves ~4 of them (emotion, arc, fact extraction, relationship assessment)
off the path the caller is blocked on. The tail is heavier in sync mode
(p95 29.1s, max 136.9s) because `subsystems_ms` there is a **sum** of
several sequential LLM calls in the same critical path the user is
waiting on; in background mode those same calls still happen and still
take that long in aggregate (compare `wall_ms` p95/max between the two
tables — 55.0s/142.6s background vs. 29.1s/136.9s sync, roughly the same
order of magnitude), but the caller isn't the one waiting on them.

These absolute numbers are specific to this reasoning model (heavy inline
chain-of-thought, few-second-plus generations even at `max_tokens=400`)
and this hardware; the qualitative result — perceived latency tracks
generate+consistency only, independent of bookkeeping cost — is the
architectural property Phase A set out to deliver, and it holds.
