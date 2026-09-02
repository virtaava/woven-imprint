"""Centralized configuration — one file governs all defaults.

Priority (highest wins):
1. CLI flags / function arguments
2. Environment variables (WOVEN_IMPRINT_*, OLLAMA_HOST)
3. Config file (~/.woven_imprint/config.yaml)
4. Built-in defaults (this file)

Usage:
    from woven_imprint.config import get_config
    cfg = get_config()
    model = cfg.llm.model          # "llama3.2" or whatever user configured
    threshold = cfg.memory.consolidation_threshold  # 100 or user override
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    model: str = ""
    embedding_model: str = "nomic-embed-text"
    ollama_host: str = "http://127.0.0.1:11434"
    llm_provider: str = "ollama"  # ollama, openai, anthropic, gemma_edge
    embedding_provider: str = "ollama"  # ollama, openai
    api_key: str | None = None
    base_url: str | None = None
    embedding_base_url: str | None = None
    num_ctx: int = 8192
    temperature: float = 0.7
    temperature_json: float = 0.3
    max_tokens: int = 2048
    timeout: int = 120
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown: float = 30.0


@dataclass
class MemoryConfig:
    consolidation_threshold: int = 100
    # consolidation keeps source memories active (metadata.consolidated_into)
    # instead of archiving them
    consolidation_keep_sources: bool = True
    consolidation_interval: int = 20  # turns between auto-consolidation checks
    state_save_interval: int = 10  # turns between state saves
    fact_extraction_interval: int = 3  # extract facts every N turns
    max_message_length: int = 50_000
    max_facts_per_extraction: int = 5
    fact_density_scaling: bool = True
    fact_importance: float = 0.75
    session_summary_importance: float = 0.85
    clustering_similarity: float = 0.75
    decay_bedrock: float = 0.9999
    decay_core: float = 0.999
    decay_buffer: float = 0.995
    tier_boost_bedrock: float = 0.35
    tier_boost_core: float = 0.2
    tier_boost_buffer: float = 0.0
    # All memories (not just buffer-tier) are embedded with date+speaker context (see
    # memory/store.py::build_embed_text) instead of raw content —
    # e.g. "[2023-05-08] User: caroline: I adopted a cat" instead of
    # "[User] I adopted a cat". `false` restores the pre-Tier-3d exact-content
    # embedding. Existing DBs mix old (raw) and new (contextualized) vectors
    # until `woven-imprint reembed <character_id>` (or the opt-in `reembed`
    # maintenance job) is run — see CHANGELOG.
    embedding_context: bool = True
    # rrf_k 120 (was 60) and weight_keyword 2.0 (was 1.0) are tuned for
    # contextualized docs (embedding_context: true) — offline LoCoMo ranking
    # experiments (ranking_experiments_report.md section (b), 2026-08-30)
    # measured evidence recall@20 54.7% at this combination vs 50.8% on the
    # pre-Tier-3d baseline (raw content, rrf_k 60, weight_keyword 1.0). To be
    # validated live by the v3 benchmark re-run; revert both if v3 contradicts.
    rrf_k: int = 120
    weight_semantic: float = 1.0
    weight_keyword: float = 2.0
    # recency/importance ranking inside the relevance gate dilutes relevance;
    # LoCoMo evidence recall@20 23.9%->50.8% with both at 0 (ranking_experiments
    # 2026-08-30). weight_recency kept at a small 0.1 (not 0): the (a) sweep found
    # 0.1 costs only ~2 pts recall@20 (48.7 vs 50.8) while still breaking ties
    # newest-first when semantic/keyword relevance ties on a generic query — the
    # long-horizon recency_ordering bench check is the product-facing case for
    # that (controller ruling 2026-08-30). Set higher to prefer recent/important
    # memories among relevant ones more strongly.
    weight_recency: float = 0.1
    weight_importance: float = 0.0
    weight_relationship: float = 0.0
    recency_anchor: str = "created"  # "created" | "accessed"
    max_candidates: int = 5000  # cap on active memories scored per retrieve() call
    relevance_gate: bool = True  # gate recency/importance/relationship ranking to relevant memories
    relevance_semantic_topk: int = 100  # semantic cutoff feeding the relevance gate
    relevance_min_similarity: float = 1e-6  # semantic eligibility floor; above float32 matmul noise
    # Semantic dedup of fact-derived core memories (0 = off): before inserting a
    # fact-derived core row, MemoryStore.add compares its embedding against active
    # core rows (FTS hits for the statement ∪ newest 500 core rows); at or above this
    # cosine similarity, no new row is inserted — the existing one is reinforced
    # instead (importance +0.05 capped at 1.0, metadata.dup_count += 1,
    # metadata.last_confirmed stamped). Evidence: Tier 3c diagnostics found
    # near-duplicate extracted facts dominating the top-20 (17.7/20 avg; 78/491 active
    # core rows in conv-26 fell into 6-word-prefix paraphrase groups, e.g.
    # "considering a career in counseling and mental health" vs "...or mental health
    # work"). Structured facts' own (subject, predicate) supersession is unaffected —
    # this only gates the *core-memory* insert, not the `facts` table row.
    fact_dedup_similarity: float = 0.0
    # Tier 3f (docs/superpowers/specs/2026-09-01-tier3f-multihop-second-pass.md):
    # multi-hop second-pass expansion (0 = off). When > 0, after the first RRF
    # fusion, the top `retrieval_second_pass` fused hits become "seeds": salient
    # terms (capitalized tokens, digits, long lowercase words) are pulled from
    # their content and used for exactly one extra `fts_search` call, plus a
    # zero-cost semantic widening using the mean of the seeds' already-stored
    # embedding vectors (no embedder call — the vectors are already in hand).
    # This surfaces co-dependent evidence the first pass missed because it shares
    # terms with what WAS found, not with the original query (LoCoMo cat-1
    # multi-hop J 0.305 vs single-hop 0.709, v3b diagnostics). See
    # `MemoryRetriever.retrieve` for the widen-and-re-fuse implementation. Off by
    # default pending live measurement (controller ruling pending Task 2/3).
    retrieval_second_pass: int = 0
    # Tier 3i (docs/superpowers/specs/2026-09-02-tier3i-llm-query-expansion.md):
    # LLM-guided query expansion (0 = off). When > 0 AND the retriever was built
    # with an `llm` handle, one JSON call per retrieve() rewrites the query into
    # at most this many instance-level search queries (aggregation questions like
    # "how many X in total" need instance memories that are individually
    # dissimilar to the aggregate phrasing — LME-S-100 multi-session J 0.31,
    # Tier 3h diagnostics). Each expansion adds one embedding (single batch
    # call) + one fts_search and contributes two extra RRF ranked lists at
    # `query_expansion_weight`. Off by default pending the pre-registered
    # locomo-t3iA bar (spec §Measurement). Cost when on: +1 LLM call,
    # +1 embed_batch call, +N fts_search per retrieve.
    query_expansion: int = 0
    # RRF weight for each expansion query's two ranked lists (semantic + keyword).
    # The original query's lists keep their own weights, so the original ranking
    # stays dominant at the 0.5 default.
    query_expansion_weight: float = 0.5

    def __post_init__(self):
        if self.query_expansion < 0:
            raise ValueError("query_expansion must be >= 0")
        if self.query_expansion_weight <= 0:
            raise ValueError("query_expansion_weight must be > 0")


@dataclass
class ContextConfig:
    total_tokens: int = 6000
    system_prompt_tokens: int = 1000
    memory_tokens: int = 1500
    conversation_tokens: int = 3000
    reserve_tokens: int = 500
    max_turns: int = 20
    include_date: bool = True
    facts_block: bool = True
    facts_block_limit: int = 12
    pinned_block: bool = True  # always include pinned memories in the prompt (see MemoryStore.pin)
    pinned_limit: int = 10  # max pinned memories rendered in the "Things you always remember" block
    # Per-memory content cap in _format_memories (0 = unlimited). Was a hard-coded 200; the
    # LoCoMo abstention analysis (eval/external/runs/diagnostics/abstain/locomo-mem-v2d/
    # recommendation.md, 2026-08-30) found 133/251 evidence lines cut by that cap, 23 with the
    # gold answer's own words removed. Raised to 800 — still bounded by the overall
    # shared context.total_tokens prompt budget (memory_tokens is not enforced yet).
    memory_content_max_chars: int = 800
    # Include the weekday token (e.g. "Mon") in each rendered memory's date prefix
    # (`_format_memories`): "(2023-05-08 Mon, 3 months ago)" vs "(2023-05-08, 3 months
    # ago)" when false. The date and relative-time phrase are unaffected either way.
    # Tier 3e (docs/superpowers/specs/2026-08-31-tier3e-plus-chat-recovery.md): the
    # weekday token was one of two chat-path suspects for the Plus cognitive-cue
    # regression; gated here so it can be measured and disabled without reverting
    # the date/relative-phrase rendering it shares a line with.
    weekday_in_dates: bool = True
    # Append resolved absolute dates for a closed set of relative-time phrases found
    # in a memory's content (e.g. "last Saturday", "next month"), computed from that
    # memory's own created_at (`Character._format_memories` /
    # `character._relative_date_hints`): "(2023-05-08 Mon, 3 months ago; "tomorrow"
    # →2023-05-09)". Tier 3g (docs/superpowers/specs/
    # 2026-09-01-tier3g-relative-time-photo-render.md): the v3 abstain-with-evidence
    # analysis found b_relative_time questions (55/317) failing because the evidence
    # line states a relative phrase and the model must combine it with the line's own
    # formed date, and often doesn't. Pure datetime, no LLM. Default is PROVISIONAL
    # pending the locomo-t3gA measurement (pre-registered bar in the spec) — the
    # controller may flip it to False and ship opt-in if the bar isn't met.
    resolve_relative_dates: bool = True
    # Render a "(shared a photo: X)" parenthetical (the LoCoMo ingest convention —
    # see eval/external/locomo.py::_turn_text) as its own indented "    [photo] X"
    # continuation line instead of leaving it as an inline aside on the main content
    # line (`character._split_photo_captions`). Tier 3g: the v3 abstain-with-evidence
    # analysis found f_photo_caption questions (54/317) failing because the answer
    # sits inside that parenthetical and gets skipped. Default is PROVISIONAL pending
    # the same locomo-t3gA measurement as resolve_relative_dates above.
    photo_caption_line: bool = True


@dataclass
class RelationshipConfig:
    max_delta: float = 0.15
    key_moments_limit: int = 20
    dynamics: bool = True
    trust_gain_factor: float = 0.5
    betrayal_threshold: float = -0.10
    betrayal_damping_turns: int = 10
    betrayal_gain_damping: float = 0.25
    key_moment_threshold: float = 0.08
    trajectory_window: int = 5
    tension_decay_per_day: float = 0.05


@dataclass
class PersonaConfig:
    growth_threshold: float = 0.6
    growth_min_memories: int = 20
    emotion_decay_rate: float = 0.15
    emotion_neutral_intensity: float = 0.3
    belief_reinforce_delta: float = 0.15


@dataclass
class CharacterConfig:
    parallel: bool = False
    lightweight: bool = False
    enforce_consistency: bool = True
    consistency_max_retries: int = 2
    consistency_temperature: float = 0.5
    consistency_fail_open_score: float = 0.8
    consistency_stream_mode: str = "log"  # "off" | "log" — post-hoc check in chat_stream()
    metrics_path: str | None = None
    background: bool = True
    unified_assessment: bool = True


@dataclass
class MaintenanceConfig:
    max_llm_calls_per_run: int = 50
    consolidate_chunk_size: int = 500
    buffer_ttl_days: int = 14
    buffer_hygiene_max_importance: float = 0.55
    importance_scoring_batch: int = 30
    dedup_scan_limit: int = 200
    dedup_similarity: float = 0.92
    reinforce_similarity: float = 0.85
    contradiction_candidate_similarity: float = 0.70
    contradiction_max_pairs: int = 10
    reflect_importance_sum: float = 12.0
    callbacks_refresh_limit: int = 5
    callbacks_ready_cap: int = 10
    callbacks_refresh_on_session_end: bool = True


@dataclass
class ServerConfig:
    api_port: int = 8650
    sidecar_port: int = 8765
    api_key: str | None = None
    cors_origin: str = "http://localhost"
    ui_port: int = 7860
    ui_browser: str = "auto"
    demo_port: int = 7860
    demo_browser: bool = True


@dataclass
class MigrationConfig:
    max_messages: int = 0  # 0 = unlimited (was hardcoded 500)
    max_message_length: int = 0  # 0 = unlimited (was hardcoded 2000)
    chunk_size: int = 50


@dataclass
class StorageConfig:
    db_path: str = ""
    busy_timeout: int = 5000

    def __post_init__(self):
        if not self.db_path:
            self.db_path = str(Path.home() / ".woven_imprint" / "characters.db")


@dataclass
class WovenConfig:
    """Top-level configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    relationship: RelationshipConfig = field(default_factory=RelationshipConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    character: CharacterConfig = field(default_factory=CharacterConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)


# Global singleton
_config: WovenConfig | None = None
_config_path: str = str(Path.home() / ".woven_imprint" / "config.yaml")


def _load_yaml(path: str) -> dict:
    """Load YAML file if it exists."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml

        with open(p) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except ImportError:
        # No PyYAML — try simple key:value parsing
        data = {}
        current_section = None
        with open(p) as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith("#"):
                    continue
                if not line.startswith(" ") and line.endswith(":"):
                    current_section = line[:-1].strip()
                    data[current_section] = {}
                elif current_section and ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    # Parse value types
                    if val.lower() in ("true", "yes"):
                        val = True
                    elif val.lower() in ("false", "no"):
                        val = False
                    elif val.lower() in ("null", "none", "~"):
                        val = None
                    else:
                        try:
                            val = int(val)
                        except ValueError:
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                    data[current_section][key] = val
        return data


def _apply_dict(target, data: dict) -> None:
    """Apply dict values to a dataclass instance."""
    for key, val in data.items():
        if hasattr(target, key):
            current = getattr(target, key)
            if isinstance(current, bool):
                setattr(target, key, bool(val))
            elif isinstance(current, int) and not isinstance(val, bool):
                setattr(target, key, int(val))
            elif isinstance(current, float):
                setattr(target, key, float(val))
            elif isinstance(current, str) or current is None:
                setattr(target, key, str(val) if val is not None else None)


def _apply_env(cfg: WovenConfig) -> None:
    """Override config from environment variables."""
    env_map = {
        "WOVEN_IMPRINT_MODEL": ("llm", "model"),
        "WOVEN_IMPRINT_EMBEDDING_MODEL": ("llm", "embedding_model"),
        "OLLAMA_HOST": ("llm", "ollama_host"),
        "WOVEN_IMPRINT_LLM_PROVIDER": ("llm", "llm_provider"),
        "WOVEN_IMPRINT_EMBEDDING_PROVIDER": ("llm", "embedding_provider"),
        "WOVEN_IMPRINT_API_KEY_LLM": ("llm", "api_key"),
        "WOVEN_IMPRINT_BASE_URL": ("llm", "base_url"),
        "WOVEN_IMPRINT_EMBEDDING_BASE_URL": ("llm", "embedding_base_url"),
        "WOVEN_IMPRINT_NUM_CTX": ("llm", "num_ctx"),
        "WOVEN_IMPRINT_DB": ("storage", "db_path"),
        "WOVEN_IMPRINT_API_KEY": ("server", "api_key"),
        "WOVEN_IMPRINT_API_PORT": ("server", "api_port"),
        "WOVEN_IMPRINT_SIDECAR_PORT": ("server", "sidecar_port"),
        "WOVEN_IMPRINT_UI_PORT": ("server", "ui_port"),
        "WOVEN_IMPRINT_DEMO_PORT": ("server", "demo_port"),
        "WOVEN_IMPRINT_PARALLEL": ("character", "parallel"),
        "WOVEN_IMPRINT_LIGHTWEIGHT": ("character", "lightweight"),
        "WOVEN_IMPRINT_ENFORCE_CONSISTENCY": ("character", "enforce_consistency"),
        "WOVEN_IMPRINT_CONSISTENCY_STREAM_MODE": ("character", "consistency_stream_mode"),
        "WOVEN_IMPRINT_MAX_FACTS": ("memory", "max_facts_per_extraction"),
        "WOVEN_IMPRINT_METRICS_PATH": ("character", "metrics_path"),
        "WOVEN_IMPRINT_BACKGROUND": ("character", "background"),
        "WOVEN_IMPRINT_UNIFIED_ASSESSMENT": ("character", "unified_assessment"),
        "WOVEN_IMPRINT_MAINTENANCE_BUDGET": ("maintenance", "max_llm_calls_per_run"),
        "WOVEN_IMPRINT_RELATIONSHIP_DYNAMICS": ("relationship", "dynamics"),
    }

    for env_var, (section, key) in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            target = getattr(cfg, section)
            current = getattr(target, key)
            if isinstance(current, bool):
                setattr(target, key, val.lower() in ("true", "1", "yes"))
            elif isinstance(current, int) and not isinstance(current, bool):
                try:
                    setattr(target, key, int(val))
                except ValueError:
                    pass
            elif isinstance(current, float):
                try:
                    setattr(target, key, float(val))
                except ValueError:
                    pass
            else:
                setattr(target, key, val)


def get_config(config_path: str | None = None) -> WovenConfig:
    """Get the global configuration.

    Loads from:
    1. Built-in defaults
    2. Config file (~/.woven_imprint/config.yaml)
    3. Environment variables

    Call `reload_config()` to force re-read.
    """
    global _config

    if _config is not None:
        return _config

    cfg = WovenConfig()

    # Load config file
    path = config_path or _config_path
    file_data = _load_yaml(path)
    for section_name, section_data in file_data.items():
        if hasattr(cfg, section_name) and isinstance(section_data, dict):
            _apply_dict(getattr(cfg, section_name), section_data)

    # Apply environment overrides
    _apply_env(cfg)

    _config = cfg
    return cfg


def reload_config(config_path: str | None = None) -> WovenConfig:
    """Force reload configuration from file + env."""
    global _config
    _config = None
    return get_config(config_path)


def save_default_config(path: str | None = None) -> Path:
    """Write a default config.yaml with all options documented."""
    p = Path(path or _config_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    content = """# Woven Imprint Configuration
# All values shown are defaults. Uncomment and modify to override.

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

memory:
  consolidation_threshold: 100
  consolidation_keep_sources: true    # keep source memories active (metadata.consolidated_into) instead of archiving them
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
  # buffer memories are embedded with date+speaker context (see build_embed_text
  # in memory/store.py) instead of raw content; false = pre-Tier-3d raw-content
  # embedding. Existing DBs need `woven-imprint reembed <character_id>` after a
  # flag flip (or after an upgrade) to bring old vectors in line with new writes.
  embedding_context: true
  # rrf_k 120 / weight_keyword 2.0 are tuned for contextualized docs (above):
  # offline LoCoMo ranking experiments measured evidence recall@20 54.7% at this
  # combination vs 50.8% on the pre-Tier-3d baseline (rrf_k 60, weight_keyword 1.0,
  # raw content) — ranking_experiments_report.md section (b), 2026-08-30.
  rrf_k: 120
  weight_semantic: 1.0
  weight_keyword: 2.0
  # recency/importance ranking inside the relevance gate dilutes relevance;
  # LoCoMo evidence recall@20 23.9%->50.8% with both at 0 (ranking_experiments 2026-08-30).
  # weight_recency kept at a small 0.1 (not 0): costs ~2 pts recall@20 (48.7 vs 50.8) but
  # preserves newest-first tie-breaking when semantic/keyword relevance ties on a generic
  # query (controller ruling 2026-08-30). Set higher to prefer recent/important memories
  # among relevant ones more strongly.
  weight_recency: 0.1
  weight_importance: 0.0
  weight_relationship: 0.0
  # recency_anchor: created        # "created" | "accessed"
  # relevance_gate: true           # false = legacy fusion (recency/importance rank ALL candidates)
  # relevance_semantic_topk: 100   # semantic cutoff feeding the relevance gate
  # relevance_min_similarity: 0.000001  # semantic eligibility floor; above float32 matmul noise
  # fact_dedup_similarity: skip inserting a fact-derived core memory when it cosine-matches
  # an existing core row at/above this threshold (0 = off); reinforces the existing row
  # (importance +0.05, metadata.dup_count += 1) instead. Tier 3c found near-duplicate
  # extracted facts dominating the top-20 (17.7/20 avg; 78/491 core rows paraphrase-grouped).
  fact_dedup_similarity: 0.0
  # retrieval_second_pass: 0       # >0 = number of top fused hits to expand from (multi-hop
  # second pass); one extra fts_search + zero embed calls per retrieve. Off by default.
  # query_expansion: 0             # >0 = max LLM-generated expansion queries per retrieve
  # (aggregation/multi-hop recall); +1 LLM call + 1 embed_batch + N fts_search when on.
  # query_expansion_weight: 0.5    # RRF weight of each expansion's ranked lists.

context:
  total_tokens: 6000
  system_prompt_tokens: 1000
  memory_tokens: 1500
  conversation_tokens: 3000
  reserve_tokens: 500
  max_turns: 20
  include_date: true
  facts_block: true              # inject a "What you currently know about {user}" block
  facts_block_limit: 12          # max current user-facts in the block (importance desc, then newest first)
  memory_content_max_chars: 800  # per-memory content cap in the rendered prompt (0 = unlimited); was a hard-coded 200
  weekday_in_dates: true         # include the weekday token ("Mon") in rendered memory date prefixes
  # resolve_relative_dates: append absolute dates for closed-set relative phrases ("last
  # Saturday", "next month", ...) found in memory content, e.g. "(2023-05-08 Mon, 3
  # months ago; "tomorrow"→2023-05-09)". Tier 3g, default provisional pending measurement.
  resolve_relative_dates: true
  # photo_caption_line: render a "(shared a photo: X)" parenthetical as its own indented
  # "    [photo] X" continuation line instead of an inline aside. Tier 3g, default
  # provisional pending measurement.
  photo_caption_line: true

relationship:
  max_delta: 0.15
  key_moments_limit: 20
  dynamics: true                 # false = legacy byte-identical arithmetic (WOVEN_IMPRINT_RELATIONSHIP_DYNAMICS)
  trust_gain_factor: 0.5         # positive trust deltas are scaled by this; negative deltas are not
  betrayal_threshold: -0.10      # a single clamped trust delta <= this counts as a betrayal
  betrayal_damping_turns: 10     # trust gains damped for this many updates after a betrayal
  betrayal_gain_damping: 0.25    # multiplier applied to trust gains while damping is active
  key_moment_threshold: 0.08     # |clamped delta| >= this on any dimension records a key moment
  trajectory_window: 5           # trajectory derived from the sum of the last N net deltas
  tension_decay_per_day: 0.05    # tension decays toward 0 by this much per elapsed day

persona:
  growth_threshold: 0.6
  growth_min_memories: 20
  emotion_decay_rate: 0.15
  emotion_neutral_intensity: 0.3
  belief_reinforce_delta: 0.15

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
  unified_assessment: true       # one LLM call per turn: emotion, relationship deltas, story beat, facts (vs separate calls)

server:
  api_port: 8650
  sidecar_port: 8765
  api_key: null
  cors_origin: http://localhost
  ui_port: 7860
  ui_browser: auto
  demo_port: 7860
  demo_browser: true

storage:
  db_path: ~/.woven_imprint/characters.db
  busy_timeout: 5000

migration:
  max_messages: 0             # 0 = unlimited
  max_message_length: 0       # 0 = unlimited
  chunk_size: 50

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
"""
    p.write_text(content)
    return p


def _format_yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _dump_simple_yaml(data: dict, indent: int = 0) -> list[str]:
    lines: list[str] = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.extend(_dump_simple_yaml(value, indent + 2))
        else:
            lines.append(f"{pad}{key}: {_format_yaml_scalar(value)}")
    return lines


def save_config(cfg: WovenConfig | None = None, path: str | None = None) -> Path:
    """Persist config to config.yaml."""
    p = Path(path or _config_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg or get_config())

    try:
        import yaml

        p.write_text(yaml.safe_dump(data, sort_keys=False))
    except ImportError:
        p.write_text("\n".join(_dump_simple_yaml(data)) + "\n")

    return p
