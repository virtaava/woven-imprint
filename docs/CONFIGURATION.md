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
  consolidation_keep_sources: true
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
  embedding_context: true
  rrf_k: 120
  weight_semantic: 1.0
  weight_keyword: 2.0
  weight_recency: 0.1
  weight_importance: 0.0
  weight_relationship: 0.0
  # recency_anchor: created        # "created" | "accessed"
  max_candidates: 5000
  relevance_gate: true
  relevance_semantic_topk: 100
  relevance_min_similarity: 0.000001
  fact_dedup_similarity: 0.92
```

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `consolidation_threshold` | `100` | — | Number of buffer memories that triggers consolidation. When buffer exceeds this count, similar memories are clustered and summarized into core memories. Checked at session end (`end_session()`) and via explicit `Character.consolidate()` — no longer polled mid-chat. Counts only *unconsolidated* buffer rows (see `consolidation_keep_sources`). |
| `consolidation_keep_sources` | `true` | — | When on, consolidation keeps source memories `active`/`buffer` (tagged `metadata.consolidated_into`) instead of archiving them, so verbatim sources stay retrievable after being summarized. When off, sources are archived as before (excluded from retrieval). |
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
| `embedding_context` | `true` | — | When on, buffer memory vectors are computed from a date+speaker contextualized string (`memory/store.py::build_embed_text`) instead of raw `content` — e.g. `"[2023-05-08] User: caroline: I adopted a cat"` instead of `"[User] I adopted a cat"`; character rows get `"[date] <Name>: body"`, everything else (facts/summaries/reflections/consolidations/events) gets `"[date] content"` unchanged. Stored `content` is never touched — this is embedding-only. Ported from the offline ranking experiment's winning template (`eval/external/runs/diagnostics/ranking/ranking_experiments_report.md` section (c)), with two fixes that experiment flagged: the pre-existing `"[User] "`/`"[Name] "` bracket tag is stripped before the speaker is injected (avoiding embedding the name twice), and no nomic `search_document:`/`search_query:` prefix is added (prefixes measured harmful on raw content in the same sweep). Set `false` to restore the pre-Tier-3d exact-content embedding. **This is a behavior change for existing databases**: memories written before this flag existed carry the old (raw-content) vector until re-embedded — run `woven-imprint reembed <character_id>` (or the opt-in `reembed` maintenance job, not in the nightly default set) to bring them in line with new writes; a DB with a mix of old and new vectors still works, it just retrieves less consistently until re-embedded. |
| `rrf_k` | `120` | — | Reciprocal Rank Fusion constant. Combines semantic/keyword/recency/importance/relationship rankings into one score: `1 / (rrf_k + rank)` per signal. Higher = flatter fusion (rank position matters less); lower = top ranks dominate more. Replaces the old tier-priority retrieval strategy (which had a seed-dominance bug — see Changelog). Raised from `60` to `120` alongside `embedding_context`/`weight_keyword` below: offline LoCoMo ranking experiments on contextualized docs (`ranking_experiments_report.md` section (b), 2026-08-30) found `rrf_k=120` (recency=importance=0) the best cell, recall@20 51.9% vs 49.4% at `60`. To be validated live by the Tier 3d v3 benchmark re-run; revert to `60` if v3 contradicts. |
| `weight_semantic` | `1.0` | — | Weight applied to the semantic (embedding cosine similarity) signal in weighted RRF. Higher = semantic relevance matters more relative to other signals. |
| `weight_keyword` | `2.0` | — | Weight applied to the keyword (FTS) signal in weighted RRF. Raised from `1.0` to `2.0`: at the `rrf_k=120`/contextualized-docs best cell, the offline keyword-weight sweep (`ranking_experiments_report.md` section (b)) measured evidence recall@20 climbing 51.9% (`1.0`) -> 54.7% (`2.0`) — the number cited by `embedding_context`/`rrf_k` above as the combined-change target (54.7% vs the pre-Tier-3d 50.8% baseline). @50 recall actually drops slightly at `2.0` (63.6% vs 65.1%) — a real trade-off, not an oversight; @20 was the metric the sweep optimized for. To be validated live by v3; revert to `1.0` if v3 contradicts. |
| `weight_recency` | `0.1` | — | Weight applied to the recency-decay signal in weighted RRF. Default `0.1` (was `1.0`): recency/importance ranking *inside the relevance gate* dilutes relevance rather than sharpening it — offline LoCoMo ranking experiments (`eval/external/ranking_experiments.py`, `eval/external/runs/diagnostics/ranking/ranking_experiments_report.md`, 2026-08-30) measured evidence recall@20 rising from 23.9% to 50.8% with `weight_recency`/`weight_importance` at 0. `0.1` costs only ~2 recall@20 points (48.7 vs 50.8) versus a bare `0.0` (controller ruling 2026-08-30) while still nudging ties toward newer memories. The long-horizon `recency_ordering` bench check tests the recency strategy itself by opting into `weight_recency=1.0` locally (see ARCHITECTURE.md#retrieval-relevance-gate) rather than relying on the `0.1` product default, since the benchmark's `HashEmbedder` makes the two memories it compares unequal in semantic relevance for reasons unrelated to recency. The decay machinery (`decay_bedrock`/`decay_core`/`decay_buffer`, `_recency_score`) is unchanged and still feeds this signal — raise it further to prefer recent memories among the already-relevant (gated) set more strongly. |
| `weight_importance` | `0.0` | — | Weight applied to the stored importance score (plus tier boost, plus a `relationship_target`/`user_id` affinity bonus of `+0.2`) in weighted RRF. Default `0.0` (was `1.0`) for the same reason as `weight_recency` — see above (importance had no equivalent long-horizon regression, so it stayed at `0.0` rather than `0.1`). At `0.0` the importance list is never built at all (see `retrieval.py`), so the tier boosts and the `+0.2` user-affinity bonus are dead weight until you raise this above `0.0` — they don't influence ranking on their own. |
| `weight_relationship` | `0.0` | — | Weight applied to the relationship-target-match signal in weighted RRF (only relevant when a `relationship_target`/`user_id` is passed to retrieval). Default `0.0` (was `1.0`): measured on identical DBs, answer-only re-runs scored LoCoMo J `0.476` with it at `1.0` vs `0.536` at `0.0` (2026-08-30) — the name-mention boost outranks genuine evidence. This is despite the tie-bias fix below being real and still in place: the strategy ranks only candidates whose boost is actually positive (see [ARCHITECTURE.md](ARCHITECTURE.md#retrieval-relevance-gate)), so it no longer injects an oldest-first bias on the untouched majority the way it used to — the fix just isn't enough to make the boost net-positive for recall. At `0.0` the relationship list is never built at all. Setting this above `0.0` re-enables the name-mention boost — the code is unchanged and available as an explicit opt-in. |
| `recency_anchor` | `created` | — | Which timestamp recency decay is anchored on: `created` (memory's creation time — decay is a fixed clock, independent of retrieval activity) or `accessed` (last-access time — frequently-recalled memories stay "fresh"). |
| `max_candidates` | `5000` | — | Hard cap on active memories scored per `retrieve()` call. Every active memory for the character is a retrieval candidate (no more 200-newest-core-rows window); if the character has more than this many active memories, the newest `max_candidates` are scored and older ones are reachable only via FTS keyword match. Raise it for characters with very long histories on capable hardware; lower it to bound retrieval latency. |
| `relevance_gate` | `true` | — | When true and the query is non-empty, recency/importance/relationship ranking is confined to memories that are semantically similar (similarity > `relevance_min_similarity`, top `relevance_semantic_topk` of those) or keyword-matching (FTS) — see [ARCHITECTURE.md](ARCHITECTURE.md#retrieval-relevance-gate). Narrows (does not eliminate) an off-topic bedrock/core flood outranking a fresh relevant fact via recency/importance floors alone — an off-topic memory with genuine positive similarity landing in the semantic top-K can still outrank on those signals. Falls back to scoring every active memory if nothing is relevant at all. Set `false` to restore the pre-gate fusion (those signals rank every active memory unconditionally). An empty query always bypasses the gate. |
| `relevance_semantic_topk` | `100` | — | Size of the semantic slice feeding the relevance gate's eligible set (unioned with all FTS keyword hits). Only used when `relevance_gate` is true. Raise it to let more semantically-adjacent memories compete on recency/importance; lower it to tighten the gate further. |
| `relevance_min_similarity` | `0.000001` | — | Semantic eligibility floor for the relevance gate: a memory must score strictly above this cosine similarity to count as "semantically relevant." Guards against float32 matmul noise on real dense embeddings (typically ~1e-8) being mistaken for a genuine positive match against a `sim > 0.0` floor. Only used when `relevance_gate` is true. |
| `fact_dedup_similarity` | `0.92` | — | Semantic dedup threshold for fact-derived core memories (`0` = off). Before inserting a fact-derived core row, `MemoryStore.add` compares its embedding against active core rows (FTS hits for the statement, limit 50, unioned with the newest 500 active core rows; cosine via the same numpy fast path retrieval uses). At or above this threshold, no new row is inserted — the existing row is reinforced instead: `importance` bumped by `0.05` (capped at `1.0`), `metadata.dup_count` incremented, `metadata.last_confirmed` stamped. The structured `facts` table row is still written either way — its own `(subject, predicate)` supersession is unaffected; a deduped fact's `memory_id` just points at the existing memory instead of a new one. Evidence: Tier 3c diagnostics found near-duplicate extracted facts dominating the retrieved top-20 (17.7/20 average) and 78/491 active core rows in one LoCoMo conversation falling into 6-word-prefix paraphrase groups (e.g. "considering a career in counseling and mental health" vs "...or mental health work"). To be validated live by the Tier 3d v3 benchmark re-run. |

Semantic scoring (cosine similarity across all candidates) uses `numpy` when it's
installed (`pip install woven-imprint[fast]`) — one matrix build + matmul per
`retrieve()` call instead of a pure-Python loop — and produces identical
rankings either way. `numpy` is an optional dependency; CI installs it, but a
plain `pip install woven-imprint` works without it.

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
  include_date: true
  facts_block: true
  facts_block_limit: 12
  pinned_block: true       # always include pinned memories in the prompt (see MemoryStore.pin)
  pinned_limit: 10         # max pinned memories rendered in the "Things you always remember" block
  memory_content_max_chars: 800  # per-memory content cap in rendered prompt lines (0 = unlimited)
```

| Setting | Default | Description |
|---------|---------|-------------|
| `total_tokens` | `6000` | Total token budget for all content sent to the LLM. Should be less than your model's context window (`num_ctx`) to leave room for the response. |
| `system_prompt_tokens` | `1000` | Budget for the persona system prompt (name, backstory, personality, speaking style). |
| `memory_tokens` | `1500` | Intended budget for retrieved memories; **not enforced yet** — the shared `total_tokens` pool bounds the whole prompt (follow-up). |
| `conversation_tokens` | `3000` | Budget for recent conversation history (sliding window). |
| `reserve_tokens` | `500` | Reserved for safety margin. |
| `max_turns` | `20` | Maximum conversation turns kept in the sliding window. Older turns are compressed into a summary. |
| `include_date` | `true` | Prefix the volatile context block with `Today is {weekday}, {YYYY-MM-DD}.` (from `woven_imprint.clock`). Retrieved memory lines are always rendered with their date, weekday, and a relative phrase (`2026-05-03 Sun, 3 weeks ago`) regardless of this setting — disabling it only removes the "Today is ..." line. |
| `facts_block` | `true` | Inject a "What you currently know about {user}" block into the volatile context, built from the character's structured facts (`Character.facts`). Superseded facts show as `(since YYYY-MM-DD, previously: X)`. A short "Things you have said about yourself" block follows when self-facts exist. Set to `false` to disable the block entirely (e.g. to save tokens or when structured facts aren't in use). |
| `facts_block_limit` | `12` | Maximum number of current user-facts included in the block, ordered by highest importance first, then by `recorded_at` **descending** (newest-recorded first) on ties, so the cap drops the oldest facts rather than the newest. Self-facts are capped separately at 5 and are not affected by this setting. |
| `pinned_block` | `true` | Render a "Things you always remember:" block into the volatile system prompt from memories pinned via `MemoryStore.pin()` (`Character.memory.pin`). Unlike other volatile content, this block counts toward the non-sheddable base and is never dropped by the token budget; pinned memories are also excluded from the ordinary retrieved-memories list so they don't appear twice. Set `false` to disable the block entirely — pins still exist (and can still be listed via `pinned()`/`GET /api/memory/pinned`), but they're no longer rendered specially and can appear in the ordinary memories list like any other memory. |
| `pinned_limit` | `10` | Maximum number of pinned memories rendered in the "Things you always remember" block, oldest-pinned-first. Has no effect on how many memories can be pinned — only on how many are shown in the prompt. |
| `memory_content_max_chars` | `800` | Maximum characters of a memory's `content` shown per line in the rendered memories block (`Character._format_memories`); `0` = unlimited. Was a hard-coded `200` — the LoCoMo abstention analysis (`eval/external/runs/diagnostics/abstain/locomo-mem-v2d/recommendation.md`) found 133/251 evidence lines cut by that cap, 23 with the gold answer's own words removed. Applies only to the rendered prompt line; stored `content` is never truncated, and this cap is still bounded by the overall `memory_tokens` prompt budget. |

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
  dynamics: true
  trust_gain_factor: 0.5
  betrayal_threshold: -0.10
  betrayal_damping_turns: 10
  betrayal_gain_damping: 0.25
  key_moment_threshold: 0.08
  trajectory_window: 5
  tension_decay_per_day: 0.05
```

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `max_delta` | `0.15` | — | Maximum change per dimension per interaction. Prevents a single conversation from dramatically shifting a relationship. A value of 0.15 means it takes ~7 consistently positive interactions to move trust from 0.0 to 1.0. |
| `key_moments_limit` | `20` | — | Maximum number of pivotal moments stored per relationship. Oldest moments are dropped when the limit is exceeded. |
| `dynamics` | `true` | `WOVEN_IMPRINT_RELATIONSHIP_DYNAMICS` | Route relationship updates through the code-driven dynamics formula (trust asymmetry, betrayal damping, tension decay, windowed trajectory, derived tier — see [ARCHITECTURE.md](ARCHITECTURE.md#relationship-model)) instead of the original clamp-only arithmetic. Set `false` for byte-identical pre-Tier-2 behavior; existing tests that pin that arithmetic set it explicitly. |
| `trust_gain_factor` | `0.5` | — | Multiplier applied to a *positive* clamped trust delta before it is added — trust rises slower than it falls. Only affects trust; other dimensions are unscaled. Only used when `dynamics` is true. |
| `betrayal_threshold` | `-0.10` | — | A single clamped trust delta at or below this value counts as a betrayal: it is applied at full magnitude (not scaled by `trust_gain_factor`), starts the damping window, and records a key moment. Only used when `dynamics` is true. |
| `betrayal_damping_turns` | `10` | — | Number of subsequent relationship updates for which positive trust gains are further scaled by `betrayal_gain_damping` after a betrayal. Decrements by one only on updates that *began* with the damping window already active; the update that starts the window (the betrayal itself, or one that occurs while already damped and resets the window) does not decrement. Only used when `dynamics` is true. |
| `betrayal_gain_damping` | `0.25` | — | Extra multiplier applied to positive trust deltas (on top of `trust_gain_factor`) while a betrayal's damping window (`betrayal_damping_turns`) is active. Only used when `dynamics` is true. |
| `key_moment_threshold` | `0.08` | — | Minimum `\|clamped delta\|` on any single dimension to record a dated key moment for that update (betrayals are recorded separately regardless of this threshold). Only used when `dynamics` is true. |
| `trajectory_window` | `5` | — | Number of most recent updates' net deltas (`trust+affection+respect`, post-scaling) and tension deltas kept in `state.recent` to derive `warming`/`cooling`/`volatile`/`stable`. Only used when `dynamics` is true; with `dynamics: false`, trajectory is derived from the current update's deltas alone. |
| `tension_decay_per_day` | `0.05` | — | Amount `tension` decays toward 0 per elapsed day (via the injectable clock) since the relationship's last update, applied before the current update's deltas. Only used when `dynamics` is true. |

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
  unified_assessment: true       # one LLM call per turn for emotion+relationship+beat+facts (vs 3-4 separate calls)
```

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `parallel` | `false` | `WOVEN_IMPRINT_PARALLEL` | Run subsystem updates (emotion, arc, fact extraction) in parallel threads. Set `true` for 3-4x faster turns with real LLMs. Keep `false` for testing or if you experience threading issues. Only takes effect when `background: false` — with `background: true` (the default), subsystem updates always run on the background worker thread instead. |
| `lightweight` | `false` | `WOVEN_IMPRINT_LIGHTWEIGHT` | Skip emotion tracking and narrative arc analysis. With the legacy per-engine path (`unified_assessment: false`), this drops separate emotion/arc calls, reducing LLM calls from 5-7 to 2-3 per turn. With `unified_assessment: true` (the default, carried from Tier 1), there is already only one bookkeeping call per turn (`_run_bookkeeping`) regardless of `lightweight` — setting `lightweight: true` there just tells that single call to skip the emotion/beat sections, it does not remove the call itself. Useful for slower models or batch operations either way. |
| `enforce_consistency` | `true` | `WOVEN_IMPRINT_ENFORCE_CONSISTENCY` | Run NLI-style consistency check on every response. Catches hard constraint violations (wrong name, contradicted backstory). Adds 1 LLM call per turn. |
| `consistency_max_retries` | `2` | — | Maximum regeneration attempts when a hard violation is detected. Higher = more likely to produce a consistent response, but slower. |
| `consistency_temperature` | `0.5` | — | Temperature for regeneration attempts after a consistency violation. Lower = more deterministic retry. |
| `consistency_fail_open_score` | `0.8` | — | Score returned when the consistency check itself fails (e.g., LLM returns unparseable JSON). 0.8 = optimistic fail-open. Lower if you want stricter behavior on check failures. |
| `consistency_stream_mode` | `log` | `WOVEN_IMPRINT_CONSISTENCY_STREAM_MODE` | Consistency-check behavior for `Character.chat_stream()`, where retraction is impossible (text is already streamed to the caller). `"log"` runs the check after the full response is streamed and records violations in `last_chat_metrics["stream_consistency_violations"]` plus a warning log; `"off"` skips the check entirely for streamed responses. `Character.chat()` (non-streaming) always enforces and retries — this setting only affects `chat_stream()`. |
| `metrics_path` | `null` | `WOVEN_IMPRINT_METRICS_PATH` | Path to a JSONL file that receives one record per `chat()`/`chat_stream()` call: `{"ts", "character_id", "metrics"}`, where `metrics` is the full `last_chat_metrics` phase breakdown. Opt-in — `null` disables the sink. Used by `scripts/bench_chat.py` and useful for production latency monitoring. |
| `background` | `true` | `WOVEN_IMPRINT_BACKGROUND` | Run bookkeeping (emotion assessment, arc tracking, fact extraction, relationship updates) on a per-character background thread after `chat()` returns, instead of blocking the caller. `Character.flush()` waits for pending work; `Character.close()` flushes and stops the worker. Set `false` to restore fully-synchronous pre-Phase-A behavior (bookkeeping completes before `chat()`/`chat_stream()` returns). |
| `unified_assessment` | `true` | `WOVEN_IMPRINT_UNIFIED_ASSESSMENT` | Make one LLM call per turn (`persona/assessment.py`'s `TurnAssessor`) covering emotion, relationship deltas, story beat, and fact extraction, instead of up to four separate per-engine calls. Set `false` to run the legacy per-engine path (kept for A/B; behavior is byte-identical to pre-Tier-1). `want_facts`/`want_beat` still follow `fact_extraction_interval` and the every-2nd-turn beat rule either way. |

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
| `WOVEN_IMPRINT_UNIFIED_ASSESSMENT` | `character.unified_assessment` | `export WOVEN_IMPRINT_UNIFIED_ASSESSMENT=false` |
| `WOVEN_IMPRINT_MAINTENANCE_BUDGET` | `maintenance.max_llm_calls_per_run` | `export WOVEN_IMPRINT_MAINTENANCE_BUDGET=100` |
| `WOVEN_IMPRINT_RELATIONSHIP_DYNAMICS` | `relationship.dynamics` | `export WOVEN_IMPRINT_RELATIONSHIP_DYNAMICS=false` |

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
