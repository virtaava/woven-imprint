# Configuration Reference

Woven Imprint reads settings from three sources (highest priority wins):

1. **CLI flags** — `--model`, `--db`, `--port`, etc.
2. **Environment variables** — `WOVEN_IMPRINT_MODEL`, `OLLAMA_HOST`, etc.
3. **Config file** — `~/.woven_imprint/config.yaml`
4. **Built-in defaults** — if nothing else is set

## Quick Start

```bash
# Create a config file with all defaults documented
woven-imprint config --init

# View current settings
woven-imprint config
```

The generated file at `~/.woven_imprint/config.yaml` contains every option
with its default value. Uncomment and change what you need.

---

## LLM Settings

Controls which language model powers your characters.

```yaml
llm:
  model: llama3.2
  embedding_model: nomic-embed-text
  ollama_host: http://127.0.0.1:11434
  llm_provider: ollama          # ollama, openai, anthropic
  embedding_provider: ollama    # ollama, openai
  # api_key: null               # API key for openai/anthropic providers
  # base_url: null              # Custom base URL for provider
  # embedding_base_url: null    # Custom base URL for embedding provider (overrides base_url for embeddings)
  num_ctx: 8192
  temperature: 0.7
  temperature_json: 0.3
  max_tokens: 2048
  timeout: 120
  max_retries: 3
  retry_base_delay: 1.0
  retry_max_delay: 30.0
  circuit_breaker_threshold: 5
  circuit_breaker_cooldown: 30.0
```

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `model` | `llama3.2` | `WOVEN_IMPRINT_MODEL` | Model name for chat generation. Any model your provider supports works. Larger models (30B+) produce better characters but are slower. |
| `embedding_model` | `nomic-embed-text` | `WOVEN_IMPRINT_EMBEDDING_MODEL` | Model used for memory embeddings. Must be an embedding model, not a chat model. `nomic-embed-text` produces 768-dim vectors. |
| `ollama_host` | `http://127.0.0.1:11434` | `OLLAMA_HOST` | URL of the Ollama server. Change if Ollama runs on a different machine or port. For Docker: `http://ollama:11434`. |
| `llm_provider` | `ollama` | `WOVEN_IMPRINT_LLM_PROVIDER` | LLM provider backend. Supported: `ollama`, `openai`, `anthropic`, `gemma_edge`. All entry points (CLI, UI, MCP, API server) use this setting. |
| `embedding_provider` | `ollama` | `WOVEN_IMPRINT_EMBEDDING_PROVIDER` | Embedding provider backend. Supported: `ollama`, `openai`. |
| `api_key` | `null` | `WOVEN_IMPRINT_API_KEY_LLM` | API key for OpenAI or Anthropic providers. Not needed for Ollama. |
| `base_url` | `null` | `WOVEN_IMPRINT_BASE_URL` | Custom base URL for the provider. Use for vLLM, llama.cpp, LiteLLM, Azure endpoints, or a Gemma edge adapter bridge. |
| `embedding_base_url` | `null` | `WOVEN_IMPRINT_EMBEDDING_BASE_URL` | Custom base URL for the **embedding** provider (applies when `embedding_provider: openai`). Overrides `base_url` for embedding calls only — use when chat and embeddings are served by two different OpenAI-compatible endpoints (e.g. vLLM for chat, llama.cpp for embeddings). `null` = embeddings fall back to `base_url`. The `ollama` embedding provider uses `ollama_host` instead. |
| `num_ctx` | `8192` | `WOVEN_IMPRINT_NUM_CTX` | Context window size passed to Ollama. Higher = more conversation history but more VRAM. Most models support 4096-131072. |
| `temperature` | `0.7` | — | Sampling temperature for character responses. Lower = more deterministic, higher = more creative. |
| `temperature_json` | `0.3` | — | Temperature for JSON generation (fact extraction, relationship assessment). Lower for more reliable structured output. |
| `max_tokens` | `2048` | — | Maximum tokens per LLM response. |
| `timeout` | `120` | — | Seconds to wait for an LLM response before timing out. Increase if using large models on slow hardware. |
| `max_retries` | `3` | — | Number of retry attempts on transient failures (timeout, 502, 503, 429). Set to 0 to disable retries. |
| `retry_base_delay` | `1.0` | — | Initial delay between retries in seconds. Doubles each attempt (exponential backoff) with random jitter. |
| `retry_max_delay` | `30.0` | — | Maximum delay between retries. Caps the exponential growth. |
| `circuit_breaker_threshold` | `5` | — | Number of consecutive failures before the circuit breaker trips. Once tripped, all calls to that provider are rejected until cooldown expires. |
| `circuit_breaker_cooldown` | `30.0` | — | Seconds to wait after circuit breaker trips before retrying the provider. |

---

## Memory Settings

Controls how characters store, consolidate, and retrieve memories.

```yaml
memory:
  consolidation_threshold: 100
  consolidation_interval: 20
  state_save_interval: 10
  fact_extraction_interval: 3
  max_message_length: 50000
  max_facts_per_extraction: 5
  fact_density_scaling: true
  fact_importance: 0.75
  session_summary_importance: 0.85
  clustering_similarity: 0.75
  decay_bedrock: 0.9999
  decay_core: 0.999
  decay_buffer: 0.995
  tier_boost_bedrock: 0.35
  tier_boost_core: 0.2
  tier_boost_buffer: 0.0
  rrf_k: 60
  weight_semantic: 1.0
  weight_keyword: 1.0
  weight_recency: 1.0
  weight_importance: 1.0
  weight_relationship: 1.0
  # recency_anchor: created        # "created" | "accessed"
```

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `consolidation_threshold` | `100` | — | Number of buffer memories that triggers consolidation. When buffer exceeds this count, similar memories are clustered and summarized into core memories. Checked at session end (`end_session()`) and via explicit `Character.consolidate()` — no longer polled mid-chat. |
| `consolidation_interval` | `20` | — | **Unused as of Phase A.** Previously: check for consolidation every N chat turns. Consolidation checks moved to session end (`end_session()`) / explicit `Character.consolidate()` only — see Changelog "Auto-consolidation no longer runs mid-chat." Kept in config for backward-compatible file parsing; has no effect. |
| `state_save_interval` | `10` | — | Save emotion and narrative arc state to database every N turns. Protects against mid-session data loss. Lower = safer but more DB writes. |
| `fact_extraction_interval` | `3` | — | Extract notable facts from conversation every N turns. Every turn = comprehensive but expensive (1 LLM call per extraction). |
| `max_message_length` | `50000` | — | Maximum characters per user message. Messages exceeding this are silently truncated. ~12,500 tokens. |
| `max_facts_per_extraction` | `5` | `WOVEN_IMPRINT_MAX_FACTS` | Base maximum facts extracted per turn. With `fact_density_scaling` enabled, this scales up for long exchanges (2x for >2000 chars, max 15) and down for short ones (half, min 2). |
| `fact_density_scaling` | `true` | — | Scale fact extraction cap based on exchange length. Long exchanges produce more facts, short ones fewer. Disable for a fixed cap. |
| `fact_importance` | `0.75` | — | Importance score assigned to extracted facts. Higher = facts rank better in retrieval. Range: 0.0–1.0. |
| `session_summary_importance` | `0.85` | — | Importance score for session summaries. Higher than facts because summaries capture the essence of entire sessions. Range: 0.0–1.0. |
| `clustering_similarity` | `0.75` | — | Cosine similarity threshold for memory clustering during consolidation. Lower = more aggressive clustering (fewer, broader summaries). Higher = tighter clusters. Range: 0.0–1.0. |
| `decay_bedrock` | `0.9999` | — | Recency decay rate for bedrock memories (per hour). Half-life: ~290 days. Bedrock memories are nearly permanent — your character's core identity doesn't fade. |
| `decay_core` | `0.999` | — | Recency decay rate for core memories (per hour). Half-life: ~29 days. Session summaries and extracted facts fade over months. |
| `decay_buffer` | `0.995` | — | Recency decay rate for buffer memories (per hour). Half-life: ~5.8 days. Raw conversation observations fade within a week. |
| `tier_boost_bedrock` | `0.35` | — | Importance bonus added to bedrock memories during retrieval. Ensures identity-defining memories always surface. |
| `tier_boost_core` | `0.2` | — | Importance bonus for core memories. Ensures session summaries and facts outrank ephemeral buffer entries. |
| `tier_boost_buffer` | `0.0` | — | Importance bonus for buffer memories. Zero by default — buffer entries compete on content relevance alone. |
| `rrf_k` | `60` | — | Reciprocal Rank Fusion constant. Combines semantic/keyword/recency/importance/relationship rankings into one score: `1 / (rrf_k + rank)` per signal. Higher = flatter fusion (rank position matters less); lower = top ranks dominate more. Replaces the old tier-priority retrieval strategy (which had a seed-dominance bug — see Changelog). |
| `weight_semantic` | `1.0` | — | Weight applied to the semantic (embedding cosine similarity) signal in weighted RRF. Higher = semantic relevance matters more relative to other signals. |
| `weight_keyword` | `1.0` | — | Weight applied to the keyword (FTS) signal in weighted RRF. |
| `weight_recency` | `1.0` | — | Weight applied to the recency-decay signal in weighted RRF. |
| `weight_importance` | `1.0` | — | Weight applied to the stored importance score (plus tier boost) in weighted RRF. |
| `weight_relationship` | `1.0` | — | Weight applied to the relationship-target-match signal in weighted RRF (only relevant when a `relationship_target`/`user_id` is passed to retrieval). |
| `recency_anchor` | `created` | — | Which timestamp recency decay is anchored on: `created` (memory's creation time — decay is a fixed clock, independent of retrieval activity) or `accessed` (last-access time — frequently-recalled memories stay "fresh"). |

---

## Context Window Settings

Controls how the conversation history fits within the LLM's context window.

```yaml
context:
  total_tokens: 6000
  system_prompt_tokens: 1000
  memory_tokens: 1500
  conversation_tokens: 3000
  reserve_tokens: 500
  max_turns: 20
```

| Setting | Default | Description |
|---------|---------|-------------|
| `total_tokens` | `6000` | Total token budget for all content sent to the LLM. Should be less than your model's context window (`num_ctx`) to leave room for the response. |
| `system_prompt_tokens` | `1000` | Budget for the persona system prompt (name, backstory, personality, speaking style). |
| `memory_tokens` | `1500` | Budget for retrieved memories injected into the prompt. |
| `conversation_tokens` | `3000` | Budget for recent conversation history (sliding window). |
| `reserve_tokens` | `500` | Reserved for safety margin. |
| `max_turns` | `20` | Maximum conversation turns kept in the sliding window. Older turns are compressed into a summary. |

When the total exceeds the budget, the system degrades gracefully:
1. Compresses conversation history
2. Reduces retrieved memories
3. Drops optional context (emotion description, arc description)
4. The persona and current message are never dropped

---

## Relationship Settings

Controls how character relationships evolve.

```yaml
relationship:
  max_delta: 0.15
  key_moments_limit: 20
```

| Setting | Default | Description |
|---------|---------|-------------|
| `max_delta` | `0.15` | Maximum change per dimension per interaction. Prevents a single conversation from dramatically shifting a relationship. A value of 0.15 means it takes ~7 consistently positive interactions to move trust from 0.0 to 1.0. |
| `key_moments_limit` | `20` | Maximum number of pivotal moments stored per relationship. Oldest moments are dropped when the limit is exceeded. |

---

## Persona Settings

Controls character growth, emotion, and belief revision.

```yaml
persona:
  growth_threshold: 0.6
  growth_min_memories: 20
  emotion_decay_rate: 0.15
  emotion_neutral_intensity: 0.3
  belief_reinforce_delta: 0.15
```

| Setting | Default | Description |
|---------|---------|-------------|
| `growth_threshold` | `0.6` | Minimum confidence score for a growth event to be applied. The LLM assesses how confident it is that the character has genuinely changed. Below this threshold, the change is rejected. Range: 0.0–1.0. |
| `growth_min_memories` | `20` | Minimum core memories required before growth detection runs. Prevents premature personality changes from insufficient evidence. |
| `emotion_decay_rate` | `0.15` | How fast emotions decay toward neutral per turn. Higher = emotions fade faster. At 0.15, a strong emotion (intensity 0.9) takes ~6 turns to become negligible. |
| `emotion_neutral_intensity` | `0.3` | Intensity level when emotion resets to neutral. Not zero — characters maintain a baseline emotional presence. |
| `belief_reinforce_delta` | `0.15` | Certainty increase when a belief is reinforced. When a character's existing belief is confirmed, its certainty rises by this amount (capped at 1.0). |

---

## Character Defaults

Default behavior for new characters. Can be overridden per character.

```yaml
character:
  parallel: false
  lightweight: false
  enforce_consistency: true
  consistency_max_retries: 2
  consistency_temperature: 0.5
  consistency_fail_open_score: 0.8
  # consistency_stream_mode: log  # "off" | "log" — post-hoc consistency check in chat_stream()
  # metrics_path: null            # JSONL per-turn chat metrics (opt-in)
  background: true               # run bookkeeping (emotion/arc/relationship/facts) off the hot path
```

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `parallel` | `false` | `WOVEN_IMPRINT_PARALLEL` | Run subsystem updates (emotion, arc, fact extraction) in parallel threads. Set `true` for 3-4x faster turns with real LLMs. Keep `false` for testing or if you experience threading issues. Only takes effect when `background: false` — with `background: true` (the default), subsystem updates always run on the background worker thread instead. |
| `lightweight` | `false` | `WOVEN_IMPRINT_LIGHTWEIGHT` | Skip emotion tracking and narrative arc analysis. Reduces LLM calls from 5-7 to 2-3 per turn. Useful for slower models or batch operations. |
| `enforce_consistency` | `true` | `WOVEN_IMPRINT_ENFORCE_CONSISTENCY` | Run NLI-style consistency check on every response. Catches hard constraint violations (wrong name, contradicted backstory). Adds 1 LLM call per turn. |
| `consistency_max_retries` | `2` | — | Maximum regeneration attempts when a hard violation is detected. Higher = more likely to produce a consistent response, but slower. |
| `consistency_temperature` | `0.5` | — | Temperature for regeneration attempts after a consistency violation. Lower = more deterministic retry. |
| `consistency_fail_open_score` | `0.8` | — | Score returned when the consistency check itself fails (e.g., LLM returns unparseable JSON). 0.8 = optimistic fail-open. Lower if you want stricter behavior on check failures. |
| `consistency_stream_mode` | `log` | `WOVEN_IMPRINT_CONSISTENCY_STREAM_MODE` | Consistency-check behavior for `Character.chat_stream()`, where retraction is impossible (text is already streamed to the caller). `"log"` runs the check after the full response is streamed and records violations in `last_chat_metrics["stream_consistency_violations"]` plus a warning log; `"off"` skips the check entirely for streamed responses. `Character.chat()` (non-streaming) always enforces and retries — this setting only affects `chat_stream()`. |
| `metrics_path` | `null` | `WOVEN_IMPRINT_METRICS_PATH` | Path to a JSONL file that receives one record per `chat()`/`chat_stream()` call: `{"ts", "character_id", "metrics"}`, where `metrics` is the full `last_chat_metrics` phase breakdown. Opt-in — `null` disables the sink. Used by `scripts/bench_chat.py` and useful for production latency monitoring. |
| `background` | `true` | `WOVEN_IMPRINT_BACKGROUND` | Run bookkeeping (emotion assessment, arc tracking, fact extraction, relationship updates) on a per-character background thread after `chat()` returns, instead of blocking the caller. `Character.flush()` waits for pending work; `Character.close()` flushes and stops the worker. Set `false` to restore fully-synchronous pre-Phase-A behavior (bookkeeping completes before `chat()`/`chat_stream()` returns). |

---

## Server Settings

Controls the OpenAI-compatible API server and React demo UI.

```yaml
server:
  api_port: 8650
  api_key: null
  cors_origin: http://localhost
  demo_port: 7860
  demo_host: 127.0.0.1
  demo_browser: true
```

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `api_port` | `8650` | `WOVEN_IMPRINT_API_PORT` | Port for the OpenAI-compatible API proxy (`woven-imprint serve`). |
| `api_key` | `null` | `WOVEN_IMPRINT_API_KEY` | Bearer token required for API requests. `null` = no authentication (local dev only). Set this before exposing the API to a network. |
| `cors_origin` | `http://localhost` | — | Allowed CORS origin for the API server. Change to `*` only if you understand the security implications. |
| `demo_port` | `7860` | `WOVEN_IMPRINT_DEMO_PORT` | Port for the React demo UI (`woven-imprint demo`). |
| `demo_host` | `127.0.0.1` | `WOVEN_IMPRINT_DEMO_HOST` | Host to bind the demo UI to. Use `0.0.0.0` to expose on all network interfaces. |
| `demo_browser` | `true` | `WOVEN_IMPRINT_DEMO_BROWSER` | Whether to open a browser automatically when the demo starts. Set to `false` to launch without opening a browser. |

---

## Storage Settings

Controls where character data is stored.

```yaml
storage:
  db_path: ~/.woven_imprint/characters.db
  busy_timeout: 5000
```

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `db_path` | `~/.woven_imprint/characters.db` | `WOVEN_IMPRINT_DB` | Path to the SQLite database file. All characters, memories, relationships, and sessions are stored here. Use an absolute path for clarity. |
| `busy_timeout` | `5000` | — | Milliseconds to wait when SQLite is locked by another process. Relevant when running parallel mode or multiple instances. |

---

## Migration Settings

Controls how characters are imported from other systems (ChatGPT, SillyTavern, Claude, etc.).

```yaml
migration:
  max_messages: 0             # 0 = unlimited
  max_message_length: 0       # 0 = unlimited
  chunk_size: 50
```

| Setting | Default | Description |
|---------|---------|-------------|
| `max_messages` | `0` | Maximum messages to import from conversation exports. `0` = unlimited (imports everything). Previously hardcoded to 500. Set a limit if you want faster imports at the cost of less context. |
| `max_message_length` | `0` | Maximum characters per imported message. `0` = unlimited. Previously hardcoded to 2000. |
| `chunk_size` | `50` | When conversation history exceeds this count, analysis is split into chunks. Each chunk is analyzed independently, then results are synthesized into a unified character profile. Lower = more LLM calls but better analysis of each segment. |

---

## Maintenance Settings

Controls the offline batch-maintenance runner (`woven-imprint maintain`,
`Engine.run_maintenance()`, the `maintain` MCP tool / sidecar endpoint) — the
"nightly job" that consolidates, deduplicates, scores, reflects, and
regenerates callbacks. See the [Developer Guide](DEVELOPER_GUIDE.md#nightly-maintenance)
for the job list.

The budget counts logical LLM operations, not physical HTTP calls:
`generate_json_robust`'s bounded retry can make up to 2 physical calls per
budgeted operation, so `max_llm_calls_per_run: 50` can mean up to 100
physical calls against a model that produces malformed JSON.

```yaml
maintenance:
  max_llm_calls_per_run: 50
  consolidate_chunk_size: 500
  buffer_ttl_days: 14
  buffer_hygiene_max_importance: 0.55
  importance_scoring_batch: 30
  dedup_scan_limit: 200
  dedup_similarity: 0.92
  reinforce_similarity: 0.85
  contradiction_candidate_similarity: 0.70
  contradiction_max_pairs: 10
  reflect_importance_sum: 12.0
  callbacks_refresh_limit: 5
  callbacks_ready_cap: 10
  callbacks_refresh_on_session_end: true
```

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `max_llm_calls_per_run` | `50` | `WOVEN_IMPRINT_MAINTENANCE_BUDGET` | LLM-call budget for one maintenance run. The budget is shared across all jobs (and across all characters in an `Engine.run_maintenance()` run); jobs that would exceed it are skipped with reason `budget exhausted`. Override per run with `woven-imprint maintain --budget N`. |
| `consolidate_chunk_size` | `500` | — | Buffer memories loaded per consolidation pass. The maintenance `consolidate` job drains in bounded passes (up to 10 per run) until the buffer drops below the consolidation threshold or the budget runs out. |
| `buffer_ttl_days` | `14` | — | `buffer_hygiene` job: buffer memories older than this are candidates for archiving. |
| `buffer_hygiene_max_importance` | `0.55` | — | `buffer_hygiene` job: only stale buffer memories with importance **at or below** this are archived. Important memories survive the TTL sweep. |
| `importance_scoring_batch` | `30` | — | `score_importance` job: max unscored buffer memories (importance still at the 0.5 default) scored per run, in one LLM call. |
| `dedup_scan_limit` | `200` | — | Max core observation memories loaded for the `dedup`, `reinforce`, and `contradictions` scans. |
| `dedup_similarity` | `0.92` | — | `dedup` job: cosine similarity at or above this marks two core observations as duplicates — the less important one is archived, the survivor's belief is reinforced. Also the upper bound of the contradiction candidate band. |
| `reinforce_similarity` | `0.85` | — | `reinforce` job: a buffer memory this similar to a core observation reinforces that core memory's certainty (each core memory at most once per run). |
| `contradiction_candidate_similarity` | `0.70` | — | `contradictions` job: lower bound of the similarity band. Pairs with similarity in `[this, dedup_similarity)` are same-topic-but-not-identical — candidates for an LLM contradiction check. |
| `contradiction_max_pairs` | `10` | — | `contradictions` job: max candidate pairs sent to the LLM per run (highest-similarity first, one LLM call each). |
| `reflect_importance_sum` | `12.0` | — | `reflect` job: trigger a reflection when the summed importance of buffer memories newer than the last reflection reaches this. Importance-weighted, so a few big moments trigger as readily as many small ones. |
| `callbacks_refresh_limit` | `5` | — | Max callbacks generated per refresh (`callbacks` job, `refresh_callbacks()`, session-end refresh). One LLM call generates up to this many. |
| `callbacks_ready_cap` | `10` | — | Max `ready` callbacks kept per character. After a refresh, the lowest-salience callbacks beyond the cap are marked `expired`. |
| `callbacks_refresh_on_session_end` | `true` | — | Regenerate callbacks automatically in `end_session()` (one extra LLM call, non-fatal on failure). Disable if you only want callbacks refreshed by the nightly `maintain` run. |

---

## Environment Variables

All environment variables that Woven Imprint reads:

| Variable | Maps To | Example |
|----------|---------|---------|
| `WOVEN_IMPRINT_MODEL` | `llm.model` | `export WOVEN_IMPRINT_MODEL=qwen3-coder:30b` |
| `WOVEN_IMPRINT_EMBEDDING_MODEL` | `llm.embedding_model` | `export WOVEN_IMPRINT_EMBEDDING_MODEL=mxbai-embed-large` |
| `OLLAMA_HOST` | `llm.ollama_host` | `export OLLAMA_HOST=http://192.168.1.100:11434` |
| `WOVEN_IMPRINT_LLM_PROVIDER` | `llm.llm_provider` | `export WOVEN_IMPRINT_LLM_PROVIDER=openai` |
| `WOVEN_IMPRINT_EMBEDDING_PROVIDER` | `llm.embedding_provider` | `export WOVEN_IMPRINT_EMBEDDING_PROVIDER=openai` |
| `WOVEN_IMPRINT_API_KEY_LLM` | `llm.api_key` | `export WOVEN_IMPRINT_API_KEY_LLM=sk-...` |
| `WOVEN_IMPRINT_BASE_URL` | `llm.base_url` | `export WOVEN_IMPRINT_BASE_URL=http://localhost:8000/v1` |
| `WOVEN_IMPRINT_EMBEDDING_BASE_URL` | `llm.embedding_base_url` | `export WOVEN_IMPRINT_EMBEDDING_BASE_URL=http://localhost:11801/v1` |
| `WOVEN_IMPRINT_NUM_CTX` | `llm.num_ctx` | `export WOVEN_IMPRINT_NUM_CTX=32768` |
| `WOVEN_IMPRINT_DB` | `storage.db_path` | `export WOVEN_IMPRINT_DB=/data/characters.db` |
| `WOVEN_IMPRINT_API_KEY` | `server.api_key` | `export WOVEN_IMPRINT_API_KEY=my-secret` |
| `WOVEN_IMPRINT_API_PORT` | `server.api_port` | `export WOVEN_IMPRINT_API_PORT=9000` |
| `WOVEN_IMPRINT_UI_PORT` | `server.ui_port` | `export WOVEN_IMPRINT_UI_PORT=8080` |
| `WOVEN_IMPRINT_PARALLEL` | `character.parallel` | `export WOVEN_IMPRINT_PARALLEL=true` |
| `WOVEN_IMPRINT_LIGHTWEIGHT` | `character.lightweight` | `export WOVEN_IMPRINT_LIGHTWEIGHT=true` |
| `WOVEN_IMPRINT_ENFORCE_CONSISTENCY` | `character.enforce_consistency` | `export WOVEN_IMPRINT_ENFORCE_CONSISTENCY=false` |
| `WOVEN_IMPRINT_CONSISTENCY_STREAM_MODE` | `character.consistency_stream_mode` | `export WOVEN_IMPRINT_CONSISTENCY_STREAM_MODE=off` |
| `WOVEN_IMPRINT_MAX_FACTS` | `memory.max_facts_per_extraction` | `export WOVEN_IMPRINT_MAX_FACTS=10` |
| `WOVEN_IMPRINT_METRICS_PATH` | `character.metrics_path` | `export WOVEN_IMPRINT_METRICS_PATH=~/.woven_imprint/metrics.jsonl` |
| `WOVEN_IMPRINT_BACKGROUND` | `character.background` | `export WOVEN_IMPRINT_BACKGROUND=false` |
| `WOVEN_IMPRINT_MAINTENANCE_BUDGET` | `maintenance.max_llm_calls_per_run` | `export WOVEN_IMPRINT_MAINTENANCE_BUDGET=100` |

---

## Common Configurations

### Fast responses on weak hardware

```yaml
llm:
  model: llama3.2:3b
  num_ctx: 4096
character:
  lightweight: true
memory:
  fact_extraction_interval: 5
  consolidation_interval: 50
context:
  total_tokens: 3000
  max_turns: 10
```

### Maximum character quality

```yaml
llm:
  model: qwen3-coder:30b
  num_ctx: 16384
character:
  parallel: true
  enforce_consistency: true
memory:
  fact_extraction_interval: 2
context:
  total_tokens: 10000
  max_turns: 30
```

### OpenAI backend (no local Ollama needed)

```yaml
llm:
  llm_provider: openai
  embedding_provider: openai
  model: gpt-4o-mini
  embedding_model: text-embedding-3-small
  api_key: sk-...
```

Or via environment variables:
```bash
export WOVEN_IMPRINT_LLM_PROVIDER=openai
export WOVEN_IMPRINT_EMBEDDING_PROVIDER=openai
export WOVEN_IMPRINT_API_KEY_LLM=sk-...
export WOVEN_IMPRINT_MODEL=gpt-4o-mini
```

### Anthropic Claude backend

```yaml
llm:
  llm_provider: anthropic
  embedding_provider: ollama    # Claude has no embedding API
  model: claude-sonnet-4-6
  api_key: sk-ant-...
```

### vLLM / llama.cpp / any OpenAI-compatible endpoint

```yaml
llm:
  llm_provider: openai
  model: my-model
  base_url: http://localhost:8000/v1
  api_key: not-needed
```

### Docker / Remote Ollama

```yaml
llm:
  ollama_host: http://ollama:11434
storage:
  db_path: /data/characters.db
```

### Production API server

```yaml
server:
  api_key: your-secret-key-here
  cors_origin: https://yourdomain.com
  api_port: 8650
character:
  parallel: true
```

---

## Security Notice

Woven-imprint stores data locally in plaintext:

- `~/.woven_imprint/config.yaml` may contain provider API keys
- `~/.woven_imprint/characters.db` stores memories and relationships in plaintext SQLite

For sensitive use cases, we recommend OS-level disk encryption (e.g., LUKS, FileVault, BitLocker). Woven-imprint does not provide encrypted-at-rest storage.
