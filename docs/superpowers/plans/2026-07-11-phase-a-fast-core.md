# Phase A — Fast, Correct Core: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Character.chat()` return as soon as the reply exists (moving 3–4 bookkeeping LLM calls to a background queue), add streaming, fix retrieval bias, make prompts cache-friendly, wire resilience into all providers, persist the conversation buffer, and capture real performance numbers.

**Architecture:** All changes live inside the existing Python package (`src/woven_imprint/`); no new runtime dependencies (stdlib only). Behavior-changing work is config-gated with backward-compatible defaults where possible; the one deliberate default change is `character.background = true` (bookkeeping goes async) with a `flush()` discipline for tests and batch callers.

**Tech Stack:** Python ≥3.11, sqlite3, pytest (existing suite: 289 tests, `tests/helpers.py` FakeLLM/FakeEmbedder), ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-07-10-companion-grade-evolution-design.md` §2 (Phase A: A1–A7).

## Global Constraints

- Python ≥3.11; CI = ruff lint+format, pyright, pytest 3.11–3.13 with `-x`, `python eval/run_eval.py`. All must stay green after every task.
- Core runtime dependencies stay exactly `requests` + `pyyaml` — new code is stdlib-only.
- Public API is additive: existing signatures (`chat`, `ingest`, `recall`, `reflect`, `consolidate`, `evolve`, sessions, export) keep working.
- All new knobs go through `config.py` dataclasses + env vars, following the existing `_apply_env` pattern.
- Run the full suite with `cd ~/sona/projects/woven-imprint && venv/bin/python -m pytest tests/ -x -q` (create/activate the project venv as the repo docs describe; plain `pytest` is fine if the venv is active).
- Commit after every task (conventional-commit style, matching repo history: `feat:`, `fix:`, `perf:`, `docs:`).

## File Structure (new/modified)

- **Create** `src/woven_imprint/metrics.py` — opt-in JSONL metrics sink (A7)
- **Create** `scripts/bench_chat.py` — scripted chat benchmark; prints per-phase latency table (A7)
- **Create** `docs/PERFORMANCE.md` — how to benchmark + recorded baseline/after numbers (A7)
- **Create** `src/woven_imprint/background.py` — per-character background worker queue (A1)
- **Create** `src/woven_imprint/embedding/cache.py` — content-hash LRU embedding cache (A3)
- **Modify** `src/woven_imprint/utils/rrf.py` — weights + configurable k (A3)
- **Modify** `src/woven_imprint/memory/retrieval.py` — drop tier-priority list, weighted RRF, created_at recency anchor (A3)
- **Modify** `src/woven_imprint/config.py` — new fields on MemoryConfig/CharacterConfig + env map entries
- **Modify** `src/woven_imprint/character.py` — background bookkeeping, chat_stream, session-turn persistence, prompt split, metrics sink hook
- **Modify** `src/woven_imprint/context.py` — `load_turns()`
- **Modify** `src/woven_imprint/llm/base.py` — `generate_stream()` default
- **Modify** `src/woven_imprint/llm/ollama.py`, `openai_llm.py`, `anthropic_llm.py`, `gemma_edge.py` — streaming + resilience parity
- **Modify** `src/woven_imprint/llm/resilience.py` — thread-safety lock, duck-typed retryable status codes
- **Modify** `src/woven_imprint/storage/sqlite.py` — indexes, migration v3 (`session_turns`), turn CRUD
- **Modify** `src/woven_imprint/engine.py` — wrap embedder in cache
- **Modify** `tests/helpers.py` — force `background = False` in `make_test_engine`
- **Test files:** `tests/test_metrics.py`, `tests/test_background.py`, `tests/test_resilience.py`, `tests/test_streaming.py`, `tests/test_session_turns.py`, additions to `tests/test_rrf.py`, `tests/test_retrieval.py`, `tests/test_embedding_cache.py` (new), `tests/test_character.py`

---

### Task 1: Metrics sink + benchmark script (baseline first)

**Files:**
- Create: `src/woven_imprint/metrics.py`
- Create: `scripts/bench_chat.py`
- Create: `docs/PERFORMANCE.md`
- Modify: `src/woven_imprint/config.py` (CharacterConfig + env map)
- Modify: `src/woven_imprint/character.py:264-266` (end of `chat()`)
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Character.last_chat_metrics: dict[str, float]` (already populated by `chat()`).
- Produces: `metrics.get_sink() -> MetricsSink | None`, `MetricsSink.write(character_id: str, metrics: dict) -> None`, `metrics.reset_sink() -> None`, config `character.metrics_path: str | None` (env `WOVEN_IMPRINT_METRICS_PATH`). Later tasks call `get_sink()` from `chat_stream` too.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import json

from woven_imprint import metrics
from woven_imprint.config import get_config, reload_config
from tests.helpers import make_test_engine


def _configure_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("WOVEN_IMPRINT_METRICS_PATH", str(tmp_path / "metrics.jsonl"))
    reload_config()
    metrics.reset_sink()


def test_sink_disabled_by_default(monkeypatch):
    monkeypatch.delenv("WOVEN_IMPRINT_METRICS_PATH", raising=False)
    reload_config()
    metrics.reset_sink()
    assert metrics.get_sink() is None


def test_chat_writes_metrics_line(tmp_path, monkeypatch):
    _configure_metrics(tmp_path, monkeypatch)
    engine = make_test_engine()
    char = engine.create_character("Metra")
    char.chat("Hello there")
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["character_id"] == char.id
    assert "total_ms" in rec["metrics"]
    assert "generate_ms" in rec["metrics"]
    # cleanup so other tests see defaults
    monkeypatch.delenv("WOVEN_IMPRINT_METRICS_PATH", raising=False)
    reload_config()
    metrics.reset_sink()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'woven_imprint.metrics'` (or ImportError).

- [ ] **Step 3: Implement**

Add to `CharacterConfig` in `src/woven_imprint/config.py` (after `consistency_fail_open_score`):

```python
    metrics_path: str | None = None
```

Add to `env_map` in `_apply_env`:

```python
        "WOVEN_IMPRINT_METRICS_PATH": ("character", "metrics_path"),
```

Create `src/woven_imprint/metrics.py`:

```python
"""Opt-in JSONL metrics sink for per-turn chat metrics.

Enabled by setting `character.metrics_path` in config or the
WOVEN_IMPRINT_METRICS_PATH env var. Each chat() appends one JSON line:
{"ts": ..., "character_id": ..., "metrics": {...}}.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .log import logger

_UNSET = object()
_sink: object = _UNSET
_lock = threading.Lock()


class MetricsSink:
    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def write(self, character_id: str, metrics: dict) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "character_id": character_id,
            "metrics": metrics,
        }
        try:
            with self._write_lock, open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.debug("Metrics write failed: %s", e)


def get_sink() -> MetricsSink | None:
    """Return the configured sink, or None if metrics are disabled."""
    global _sink
    with _lock:
        if _sink is _UNSET:
            from .config import get_config

            path = get_config().character.metrics_path
            _sink = MetricsSink(path) if path else None
        return _sink  # type: ignore[return-value]


def reset_sink() -> None:
    """Forget the cached sink (used after config reload / in tests)."""
    global _sink
    with _lock:
        _sink = _UNSET
```

In `src/woven_imprint/character.py`, at the end of `chat()` replace:

```python
        metrics["total_ms"] = round((time.perf_counter() - chat_started) * 1000.0, 2)
        self.last_chat_metrics = metrics
        return response
```

with:

```python
        metrics["total_ms"] = round((time.perf_counter() - chat_started) * 1000.0, 2)
        self.last_chat_metrics = metrics
        self._emit_metrics(metrics)
        return response
```

and add this method to `Character` (near `_save_state`):

```python
    def _emit_metrics(self, metrics: dict) -> None:
        from .metrics import get_sink

        sink = get_sink()
        if sink:
            sink.write(self.id, metrics)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_metrics.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Create the benchmark script**

Create `scripts/bench_chat.py`:

```python
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
```

- [ ] **Step 6: Record the BASELINE**

Run the benchmark against a live local model (any Ollama model, or the Spark endpoint via `WOVEN_IMPRINT_LLM_PROVIDER=openai WOVEN_IMPRINT_BASE_URL=http://127.0.0.1:11800/v1 WOVEN_IMPRINT_API_KEY_LLM=sk-local WOVEN_IMPRINT_MODEL=<model>`):

Run: `python scripts/bench_chat.py --turns 12`

Create `docs/PERFORMANCE.md` and paste the printed summary table under a "Baseline (pre-Phase-A)" heading, noting date, provider, model, and hardware. If no live LLM is reachable, record the table from a FakeLLM run instead and mark it "mock baseline — re-run against live model" (the structure of the doc is the deliverable; live numbers get refreshed in Task 11).

```markdown
# Performance

Per-turn latency is measured by the instrumentation in `Character.chat()`
(`last_chat_metrics`) and collected via `scripts/bench_chat.py` or the
JSONL sink (`WOVEN_IMPRINT_METRICS_PATH`).

## How to run

    WOVEN_IMPRINT_LLM_PROVIDER=ollama WOVEN_IMPRINT_MODEL=<model> \
        python scripts/bench_chat.py --turns 12

## Baseline (pre-Phase-A)

<paste table + environment description here>

## After Phase A

<filled in Task 11>
```

- [ ] **Step 7: Full suite + commit**

Run: `pytest tests/ -x -q` — expected: all pass.

```bash
git add src/woven_imprint/metrics.py src/woven_imprint/config.py src/woven_imprint/character.py scripts/bench_chat.py docs/PERFORMANCE.md tests/test_metrics.py
git commit -m "feat: opt-in JSONL metrics sink + chat benchmark script (A7)"
```

---

### Task 2: Weighted RRF and removal of the tier-priority strategy

**Files:**
- Modify: `src/woven_imprint/utils/rrf.py`
- Modify: `src/woven_imprint/memory/retrieval.py:146-159`
- Modify: `src/woven_imprint/config.py` (MemoryConfig)
- Test: `tests/test_rrf.py` (additions), `tests/test_retrieval.py` (additions)

**Interfaces:**
- Produces: `reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]`; MemoryConfig fields `rrf_k: int = 60`, `weight_semantic/weight_keyword/weight_recency/weight_importance/weight_relationship: float = 1.0`. The tier-priority ranked list is REMOVED from retrieval (tier keeps influencing decay + importance boost only).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rrf.py`:

```python
def test_rrf_weights_bias_fusion():
    from woven_imprint.utils.rrf import reciprocal_rank_fusion

    # list A prefers "x", list B prefers "y"; weighting B 3x must flip the winner
    lists = [["x", "y"], ["y", "x"]]
    unweighted = reciprocal_rank_fusion(lists)
    assert unweighted[0][1] == unweighted[1][1]  # tie without weights
    weighted = reciprocal_rank_fusion(lists, weights=[1.0, 3.0])
    assert weighted[0][0] == "y"


def test_rrf_configurable_k():
    from woven_imprint.utils.rrf import reciprocal_rank_fusion

    lists = [["a", "b"]]
    k10 = dict(reciprocal_rank_fusion(lists, k=10))
    k60 = dict(reciprocal_rank_fusion(lists, k=60))
    assert k10["a"] == 1 / 11
    assert k60["a"] == 1 / 61
```

Append to `tests/test_retrieval.py` (the seed-dominance regression test — check the file's existing fixture style and reuse its storage/embedder setup helpers if present; otherwise):

```python
def test_personal_core_fact_beats_bedrock_seed_flood():
    """Regression: many bedrock seeds must not drown a query-relevant core fact."""
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character(
        "Meridian", persona={"backstory": "A wizard of the old tower."}
    )
    # Flood bedrock with irrelevant seeds
    for i in range(30):
        char.memory.add(
            content=f"Ancient tower lore volume {i}: the stones hum at dusk.",
            tier="bedrock",
            role="observation",
            importance=0.9,
        )
    # One personal core fact
    char.memory.add(
        content="The user's sister is named Anna and she loves rowing.",
        tier="core",
        role="observation",
        importance=0.75,
    )
    results = char.retriever.retrieve("what is my sister's name", limit=5)
    contents = [m["content"] for m in results]
    assert any("Anna" in c for c in contents), contents
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/test_rrf.py tests/test_retrieval.py -v`
Expected: new tests FAIL (`TypeError: ... unexpected keyword argument 'weights'`; the seed-flood test may fail on assertion).

- [ ] **Step 3: Implement weighted RRF**

Replace the body of `src/woven_imprint/utils/rrf.py`'s fusion function (keep the module docstring):

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists: score(id) = sum_i weight_i / (k + rank_i + 1).

    Args:
        ranked_lists: One ranked list of IDs per strategy (best first).
        k: RRF dampening constant.
        weights: Optional per-list weights (defaults to 1.0 each).

    Returns:
        (id, score) tuples sorted by fused score, descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

- [ ] **Step 4: Add config fields**

In `MemoryConfig` (`src/woven_imprint/config.py`), append:

```python
    rrf_k: int = 60
    weight_semantic: float = 1.0
    weight_keyword: float = 1.0
    weight_recency: float = 1.0
    weight_importance: float = 1.0
    weight_relationship: float = 1.0
```

- [ ] **Step 5: Rewire retrieval**

In `src/woven_imprint/memory/retrieval.py`, delete the Strategy 5 block (lines 146-150: `_TIER_RANK` … `tier_ranked = …`) and replace the ranked-list assembly + fusion (lines 152-175) with:

```python
        from ..config import get_config

        mem_cfg = get_config().memory

        ranked_lists = [
            semantic_ranked,
            keyword_ranked,
            recency_ranked,
            importance_ranked,
        ]
        weights = [
            mem_cfg.weight_semantic,
            mem_cfg.weight_keyword,
            mem_cfg.weight_recency,
            mem_cfg.weight_importance,
        ]

        if relationship_target:
            rel_scores = []
            target_lower = relationship_target.lower()
            for m in all_memories:
                content_lower = m["content"].lower()
                meta = m.get("metadata", {})
                involves_target = (
                    target_lower in content_lower or meta.get("target_id") == relationship_target
                )
                rel_scores.append((m["id"], 1.0 if involves_target else 0.0))
            rel_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_lists.append([mid for mid, _ in rel_scores])
            weights.append(mem_cfg.weight_relationship)

        # Fuse with weighted RRF
        fused = reciprocal_rank_fusion(ranked_lists, k=mem_cfg.rrf_k, weights=weights)
```

Also update the class docstring strategy list (remove "Tier priority").

- [ ] **Step 6: Run tests, fix fallout, commit**

Run: `pytest tests/ -x -q` — some existing retrieval tests may assert bedrock-first ordering; update those assertions to the new intended behavior (tier influences decay/importance only), never by re-adding the tier list.

```bash
git add src/woven_imprint/utils/rrf.py src/woven_imprint/memory/retrieval.py src/woven_imprint/config.py tests/test_rrf.py tests/test_retrieval.py
git commit -m "fix: weighted RRF, drop tier-priority strategy (seed-dominance bug, A3)"
```

---

### Task 3: Recency anchored on created_at

**Files:**
- Modify: `src/woven_imprint/memory/retrieval.py:41-58,124-130`
- Modify: `src/woven_imprint/config.py` (MemoryConfig)
- Test: `tests/test_retrieval.py` (additions)

**Interfaces:**
- Produces: MemoryConfig field `recency_anchor: str = "created"` (`"created"` | `"accessed"`); `_recency_score(memory: dict, tier: str) -> float` now takes the memory dict.

- [ ] **Step 1: Write the failing test**

```python
def test_recency_anchor_created_ignores_touches(monkeypatch):
    """Retrieval touching accessed_at must not make old memories look fresh."""
    from woven_imprint.memory.retrieval import _recency_score

    old_created = "2020-01-01 00:00:00"
    fresh_touch = "2099-01-01 00:00:00"
    mem = {"created_at": old_created, "accessed_at": fresh_touch}
    score = _recency_score(mem, "buffer")
    assert score < 0.01  # decayed to ~nothing despite the fresh touch
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_retrieval.py::test_recency_anchor_created_ignores_touches -v`
Expected: FAIL (signature mismatch: current `_recency_score` takes a string).

- [ ] **Step 3: Implement**

Add to `MemoryConfig`:

```python
    recency_anchor: str = "created"  # "created" | "accessed"
```

Replace `_recency_score` in `retrieval.py`:

```python
def _recency_score(memory: dict, tier: str = "buffer") -> float:
    """Exponential decay based on hours since the anchor timestamp.

    Anchor is `created_at` by default (config memory.recency_anchor).
    Anchoring on `accessed_at` makes frequently-retrieved memories
    self-reinforcing — kept only as an opt-in legacy mode.
    """
    from ..config import get_config

    anchor_field = (
        "accessed_at" if get_config().memory.recency_anchor == "accessed" else "created_at"
    )
    decay_rate = _get_decay_rates().get(tier, 0.995)
    raw = memory.get(anchor_field) or memory.get("created_at") or ""
    try:
        anchored = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0.5
    if anchored.tzinfo is None:
        anchored = anchored.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    hours = max(0, (now - anchored).total_seconds() / 3600)
    return decay_rate**hours
```

Update the call site (Strategy 3):

```python
        recency_scores = [
            (m["id"], _recency_score(m, m.get("tier", "buffer"))) for m in all_memories
        ]
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/ -x -q` — expected all pass (fix any test calling `_recency_score` with the old signature).

```bash
git add src/woven_imprint/memory/retrieval.py src/woven_imprint/config.py tests/test_retrieval.py
git commit -m "fix: anchor recency decay on created_at, config-gated (A3)"
```

---

### Task 4: Embedding cache + missing DB indexes

**Files:**
- Create: `src/woven_imprint/embedding/cache.py`
- Modify: `src/woven_imprint/engine.py:32-39` (wrap embedder)
- Modify: `src/woven_imprint/storage/sqlite.py:83-86` (indexes in `_SCHEMA`)
- Test: `tests/test_embedding_cache.py`

**Interfaces:**
- Produces: `CachedEmbedder(inner: EmbeddingProvider, maxsize: int = 512)` implementing `embed/embed_batch/dimensions`, with `hits: int` and `misses: int` counters. `Engine` wraps whatever embedder it has (provided or factory-built) in `CachedEmbedder`. New indexes: `idx_memories_accessed`, `idx_memories_created`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embedding_cache.py
from tests.helpers import FakeEmbedder
from woven_imprint.embedding.cache import CachedEmbedder


def test_repeat_embed_hits_cache():
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner)
    v1 = cached.embed("hello world")
    v2 = cached.embed("hello world")
    assert v1 == v2
    assert cached.hits == 1
    assert cached.misses == 1


def test_lru_eviction():
    cached = CachedEmbedder(FakeEmbedder(), maxsize=2)
    cached.embed("a")
    cached.embed("b")
    cached.embed("c")  # evicts "a"
    cached.embed("a")
    assert cached.misses == 4


def test_engine_wraps_embedder():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    assert isinstance(engine.embedding, CachedEmbedder)


def test_indexes_exist():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    names = {
        r[0]
        for r in engine.storage._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_memories_accessed" in names
    assert "idx_memories_created" in names
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_embedding_cache.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement the cache**

Create `src/woven_imprint/embedding/cache.py`:

```python
"""Content-hash LRU cache wrapping any EmbeddingProvider.

Identical text is embedded once. Thread-safe (background workers embed too).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from .base import EmbeddingProvider


class CachedEmbedder(EmbeddingProvider):
    def __init__(self, inner: EmbeddingProvider, maxsize: int = 512):
        self.inner = inner
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> list[float]:
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
                return list(self._cache[key])
        vec = self.inner.embed(text)
        with self._lock:
            self.misses += 1
            self._cache[key] = list(vec)
            self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def dimensions(self) -> int:
        return self.inner.dimensions()
```

In `src/woven_imprint/engine.py`, after the embedder is resolved (whether passed in or factory-built), wrap it:

```python
        from .embedding.cache import CachedEmbedder

        if not isinstance(embedding, CachedEmbedder):
            embedding = CachedEmbedder(embedding)
        self.embedding = embedding
```

(Adapt to the exact local variable names at `engine.py:32-39`; the invariant is `self.embedding` ends up a `CachedEmbedder` exactly once.)

- [ ] **Step 4: Add the indexes**

In `_SCHEMA` in `src/woven_imprint/storage/sqlite.py`, after the existing `CREATE INDEX` lines:

```sql
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(character_id, accessed_at);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(character_id, created_at);
```

(`_SCHEMA` runs on every connection with IF NOT EXISTS, so existing DBs pick these up without a migration.)

- [ ] **Step 5: Run tests, commit**

Run: `pytest tests/ -x -q` — all pass.

```bash
git add src/woven_imprint/embedding/cache.py src/woven_imprint/engine.py src/woven_imprint/storage/sqlite.py tests/test_embedding_cache.py
git commit -m "perf: content-hash LRU embedding cache + accessed/created indexes (A3)"
```

---

### Task 5: Cache-friendly prompt split (stable prefix / volatile block)

**Files:**
- Modify: `src/woven_imprint/character.py:629-719` (`_build_context`)
- Test: `tests/test_character.py` (additions)

**Interfaces:**
- Produces: `_build_context` returns `[{system: <stable persona>}, {system: <volatile block>}?, *history, {user}]`. Message 0 is byte-identical across turns for an unchanged persona. Consumers (`consistency.enforce`, servers) already take the whole message list — no signature change.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_character.py
def test_prompt_prefix_stable_across_turns():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character(
        "Stable", persona={"backstory": "A lighthouse keeper.", "personality": "calm"}
    )
    char.chat("Hello, I'm Toni")
    first = [dict(m) for m in char.last_chat_messages]
    char.chat("I found a key with a raven on it")
    second = [dict(m) for m in char.last_chat_messages]

    # Message 0 = stable persona prefix, identical across turns
    assert first[0]["role"] == "system"
    assert first[0]["content"] == second[0]["content"]
    # Volatile state (memories etc.) must NOT be inside message 0
    assert "memories" not in first[0]["content"].lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_character.py::test_prompt_prefix_stable_across_turns -v`
Expected: FAIL (today everything is glued into one system message; memories/emotion make it differ per turn).

- [ ] **Step 3: Implement**

Rewrite the assembly part of `_build_context` (keep the budget/shedding logic, but accumulate optional parts into a separate `volatile` string instead of appending to `full_system`):

- Rename `full_system` usage: keep `system_prompt` as the untouched stable prefix.
- Everywhere the current code does `full_system += part`, do `volatile += part` instead (initialize `volatile = ""`).
- Budget math is unchanged (`base_size` already counts `system_prompt` separately).
- Final assembly becomes:

```python
        # Assemble final message list: stable prefix first, volatile second
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if volatile:
            messages.append({"role": "system", "content": volatile.lstrip("\n")})
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        return messages
```

Check `persona/consistency.py`'s `enforce()`: it reads `messages` to rebuild context — confirm it filters by role (it keeps last 3 message *pairs*); two system messages at the head are tolerated. If it hardcodes `messages[0]`, adjust it to collect all leading `system` messages.

- [ ] **Step 4: Run tests, fix fallout, commit**

Run: `pytest tests/ -x -q` — update any test asserting a single system message (search `tests/` for `messages[0]` and `role.*system` assertions).

```bash
git add src/woven_imprint/character.py src/woven_imprint/persona/consistency.py tests/test_character.py
git commit -m "perf: split prompt into stable persona prefix + volatile block (A4)"
```

---

### Task 6: Background bookkeeping queue

**Files:**
- Create: `src/woven_imprint/background.py`
- Modify: `src/woven_imprint/character.py` (chat/ingest subsystems + maintenance, `flush()`, `close()`)
- Modify: `src/woven_imprint/config.py` (CharacterConfig `background`, env var)
- Modify: `tests/helpers.py` (`make_test_engine` forces `background = False`)
- Test: `tests/test_background.py`

**Interfaces:**
- Produces: `BackgroundWorker` with `submit(label: str, fn, *args, **kwargs)`, `flush(timeout: float | None = None) -> bool`, `close()`, `success_counts: dict[str, int]`, `failure_counts: dict[str, int]`. `Character.background: bool` (config `character.background = True`, env `WOVEN_IMPRINT_BACKGROUND`), `Character.flush(timeout=None) -> bool`, `Character.close()`. Inline auto-consolidation is REMOVED from `chat()`/`ingest()` (stays in `end_session()` and explicit `consolidate()`).
- Consumes: `_run_subsystems_sequential(message, response, user_id)` (unchanged, now runs on the worker thread when background is on).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_background.py
import threading
import time

from tests.helpers import FakeEmbedder, FakeLLM, make_test_engine
from woven_imprint.background import BackgroundWorker
from woven_imprint.engine import Engine


def test_worker_runs_and_counts():
    worker = BackgroundWorker("test")
    hits = []
    worker.submit("job", hits.append, 1)
    assert worker.flush(timeout=5)
    assert hits == [1]
    assert worker.success_counts["job"] == 1
    worker.close()


def test_worker_records_failures():
    worker = BackgroundWorker("test")

    def boom():
        raise RuntimeError("nope")

    worker.submit("bad", boom)
    assert worker.flush(timeout=5)
    assert worker.failure_counts["bad"] == 1
    worker.close()


class SlowJSONLLM(FakeLLM):
    """generate_json blocks until released — proves chat() doesn't wait on it."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def generate_json(self, messages, **kw):
        self.release.wait(timeout=10)
        return super().generate_json(messages, **kw)


def test_chat_returns_before_bookkeeping():
    llm = SlowJSONLLM()
    engine = Engine(db_path=":memory:", llm=llm, embedding=FakeEmbedder())
    char = engine.create_character("Speedy")
    char.background = True
    char.parallel = False

    start = time.perf_counter()
    response = char.chat("hello", user_id="u1")
    elapsed = time.perf_counter() - start

    assert response == "I hear you."
    assert elapsed < 2.0  # did not wait for the blocked generate_json
    llm.release.set()
    assert char.flush(timeout=10)
    # Bookkeeping actually ran after flush
    assert char.get_relationship("u1") is not None


def test_flush_is_noop_when_sync():
    engine = make_test_engine()
    char = engine.create_character("Syncy")
    assert char.background is False  # helpers force sync mode
    char.chat("hi", user_id="u1")
    assert char.flush()  # trivially true
    assert char.get_relationship("u1") is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_background.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement the worker**

Create `src/woven_imprint/background.py`:

```python
"""Per-character background worker — serialized bookkeeping off the hot path.

One daemon thread per Character. Tasks run in submission order, so
DB writes stay ordered per character. Failures are counted per label
and logged at WARNING (never silently dropped).
"""

from __future__ import annotations

import queue
import threading
import time

from .log import logger

_STOP = object()


class BackgroundWorker:
    def __init__(self, name: str = ""):
        self._queue: queue.Queue = queue.Queue()
        self.success_counts: dict[str, int] = {}
        self.failure_counts: dict[str, int] = {}
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"woven-bg-{name}"
        )
        self._thread.start()

    def submit(self, label: str, fn, *args, **kwargs) -> None:
        if self._closed:
            raise RuntimeError("BackgroundWorker is closed")
        self._queue.put((label, fn, args, kwargs))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                return
            label, fn, args, kwargs = item
            try:
                fn(*args, **kwargs)
                self.success_counts[label] = self.success_counts.get(label, 0) + 1
            except Exception as e:
                logger.warning("Background task '%s' failed: %s", label, e)
                self.failure_counts[label] = self.failure_counts.get(label, 0) + 1
            finally:
                self._queue.task_done()

    def flush(self, timeout: float | None = None) -> bool:
        """Block until all submitted tasks finished. Returns False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._queue.unfinished_tasks:  # noqa: SLF001 — documented Queue attr
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def close(self, timeout: float | None = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout)
```

- [ ] **Step 4: Config + Character integration**

`CharacterConfig`: add `background: bool = True`. `env_map`: add `"WOVEN_IMPRINT_BACKGROUND": ("character", "background"),`.

`Character.__init__` (after the parallel/lightweight lines):

```python
        self.background: bool = _cfg.character.background
        self._worker = None  # lazily created BackgroundWorker
```

Add methods:

```python
    def _get_worker(self):
        if self._worker is None:
            from .background import BackgroundWorker

            self._worker = BackgroundWorker(self.id)
        return self._worker

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for queued background bookkeeping to finish."""
        if self._worker is None:
            return True
        return self._worker.flush(timeout=timeout)

    def close(self) -> None:
        """Flush and stop the background worker."""
        if self._worker is not None:
            self._worker.flush(timeout=10)
            self._worker.close()
            self._worker = None
```

In `chat()`, replace the subsystems block (`if self.parallel and not self.lightweight: ...else: ...`) with:

```python
        if self.background:
            self._get_worker().submit(
                "bookkeeping", self._run_subsystems_sequential, message, response, user_id
            )
        elif self.parallel and not self.lightweight:
            self._run_subsystems_parallel(message, response, user_id)
        else:
            self._run_subsystems_sequential(message, response, user_id)
```

Move `self._turn_count += 1` to BEFORE this block (extraction throttling reads it; incrementing first keeps the counted turn consistent whether sync or async).

In the periodic-maintenance block of `chat()` AND `ingest()`: delete the auto-consolidation branch (the `if self._turn_count % _cfg.memory.consolidation_interval == 0 and self.consolidator.needs_consolidation(): ...` block). Keep the `_save_state()` periodic save. Consolidation remains available via `Character.consolidate()` and inside `end_session()`.

In `end_session()`, first line after the session check: `self.flush(timeout=30)` — bookkeeping must land before the summary reads buffer memories.

In `tests/helpers.py` `_create_seq`, add `c.background = False` next to `c.parallel = False`.

- [ ] **Step 5: Run tests, fix fallout, commit**

Run: `pytest tests/ -x -q`. Expected fallout: tests that construct `Engine(...)` directly (not via `make_test_engine`) and assert post-chat state must either set `char.background = False` or call `char.flush()` after chats — fix them accordingly. Tests asserting auto-consolidation inside chat must move to explicit `char.consolidate()` calls.

```bash
git add src/woven_imprint/background.py src/woven_imprint/character.py src/woven_imprint/config.py tests/helpers.py tests/test_background.py
git commit -m "feat: background bookkeeping queue — chat returns after generate+consistency (A1)"
```

---

### Task 7: Provider streaming (`generate_stream`)

**Files:**
- Modify: `src/woven_imprint/llm/base.py`
- Modify: `src/woven_imprint/llm/ollama.py`, `src/woven_imprint/llm/openai_llm.py`, `src/woven_imprint/llm/anthropic_llm.py`
- Test: `tests/test_streaming.py`

**Interfaces:**
- Produces: `LLMProvider.generate_stream(messages, temperature=0.7, max_tokens=2048) -> Iterator[str]` — non-abstract default yields `self.generate(...)` as one chunk (so FakeLLM/GemmaEdge work unchanged). Native implementations for Ollama (NDJSON lines), OpenAI (delta stream), Anthropic (`messages.stream`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_streaming.py
import json

from tests.helpers import FakeLLM


def test_default_stream_yields_single_chunk():
    llm = FakeLLM()
    # FakeLLM doesn't define generate_stream — default must fall back to generate()
    from woven_imprint.llm.base import LLMProvider

    chunks = list(LLMProvider.generate_stream(llm, [{"role": "user", "content": "hi"}]))
    assert chunks == ["I hear you."]


def test_ollama_stream_parses_ndjson(monkeypatch):
    from woven_imprint.llm.ollama import OllamaLLM

    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}),
        json.dumps({"message": {"content": "lo"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]

    class FakeResp:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(line.encode() for line in lines)

    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResp())
    llm = OllamaLLM(model="test")
    chunks = list(llm.generate_stream([{"role": "user", "content": "hi"}]))
    assert "".join(chunks) == "Hello"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_streaming.py -v`
Expected: FAIL with AttributeError (`generate_stream` doesn't exist).

- [ ] **Step 3: Implement base default**

In `src/woven_imprint/llm/base.py` add (non-abstract, after `generate_json`):

```python
    def generate_stream(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ):
        """Yield response text chunks. Default: one chunk via generate().

        Providers with native streaming override this. Note: providers that
        post-process whole responses (e.g. think-tag stripping) may behave
        differently in stream mode — document per provider.
        """
        yield self.generate(messages, temperature=temperature, max_tokens=max_tokens)
```

Add `from typing import Any, Iterator` if needed for annotations (keep simple, no annotation is fine).

- [ ] **Step 4: Ollama native streaming**

In `OllamaLLM`:

```python
    def generate_stream(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ):
        """Stream chunks from Ollama's NDJSON /api/chat stream.

        Note: <think> tags are NOT stripped in stream mode (can't strip
        across chunk boundaries) — callers wanting stripped output use
        generate().
        """
        resp = self._post_stream(
            "/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": self.num_ctx,
                },
                "stream": True,
            },
        )
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                yield chunk
            if data.get("done"):
                break

    def _post_stream(self, endpoint: str, payload: dict):
        from .resilience import resilient_call

        def _do_post():
            resp = requests.post(
                f"{self.base_url}{endpoint}", json=payload, timeout=self.timeout, stream=True
            )
            resp.raise_for_status()
            return resp

        return resilient_call(_do_post, provider_name="ollama")
```

- [ ] **Step 5: OpenAI + Anthropic native streaming**

`OpenAILLM`:

```python
    def generate_stream(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield delta
```

`AnthropicLLM` (reuse the same system/message splitting as its `generate`; factor that into a helper `_split_messages(messages) -> tuple[str, list]` used by both):

```python
    def generate_stream(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ):
        system, chat_messages = self._split_messages(messages)
        with self.client.messages.stream(
            model=self.model,
            system=system or "You are a helpful assistant.",
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text
```

(GemmaEdge keeps the base default — its adapter contract has no stream endpoint yet.)

- [ ] **Step 6: Run tests, commit**

Run: `pytest tests/ -x -q` (OpenAI/Anthropic streaming paths are exercised only when SDKs installed; keep those under `pytest.importorskip` if you add tests for them).

```bash
git add src/woven_imprint/llm/base.py src/woven_imprint/llm/ollama.py src/woven_imprint/llm/openai_llm.py src/woven_imprint/llm/anthropic_llm.py tests/test_streaming.py
git commit -m "feat: generate_stream on providers with safe non-streaming default (A2)"
```

---

### Task 8: `Character.chat_stream()`

**Files:**
- Modify: `src/woven_imprint/character.py`
- Modify: `src/woven_imprint/config.py` (CharacterConfig `consistency_stream_mode`)
- Test: `tests/test_streaming.py` (additions)

**Interfaces:**
- Produces: `Character.chat_stream(message: str, user_id: str | None = None) -> Iterator[str]` — yields text chunks; after the final chunk, performs the same post-processing as `chat()` except consistency is post-hoc (config `character.consistency_stream_mode: str = "log"`, values `"off" | "log"`; violations are logged + counted in `last_chat_metrics["stream_consistency_violations"]`, the streamed text is never retracted — documented tradeoff from the spec).
- Consumes: `generate_stream` (Task 7), `BackgroundWorker` (Task 6), `_emit_metrics` (Task 1).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_streaming.py
class StreamingFakeLLM(FakeLLM):
    def generate_stream(self, messages, **kw):
        for chunk in ["I ", "hear ", "you."]:
            yield chunk


def test_chat_stream_yields_and_bookkeeps():
    from tests.helpers import FakeEmbedder
    from woven_imprint.engine import Engine

    engine = Engine(db_path=":memory:", llm=StreamingFakeLLM(), embedding=FakeEmbedder())
    char = engine.create_character("Streamy")
    char.background = False  # sync bookkeeping for deterministic assertions
    char.parallel = False

    chunks = list(char.chat_stream("hello", user_id="u1"))
    assert "".join(chunks) == "I hear you."
    # Same post-processing as chat(): buffers + memory + relationship
    assert char.get_relationship("u1") is not None
    buffer = char.memory.get_all(tier="buffer", limit=10)
    assert any("I hear you." in m["content"] for m in buffer)
    assert char._context.turn_count == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_streaming.py::test_chat_stream_yields_and_bookkeeps -v`
Expected: FAIL with AttributeError (`chat_stream` missing).

- [ ] **Step 3: Implement**

Add `consistency_stream_mode: str = "log"` to `CharacterConfig`.

Add to `Character` (after `chat()`); mirror `chat()`'s steps — store user memory, retrieve, relationship context, build context — then stream and post-process:

```python
    def chat_stream(self, message: str, user_id: str | None = None):
        """Like chat(), but yields response chunks as they generate.

        Consistency checking is post-hoc in stream mode: streamed text is
        never retracted. Violations are logged and counted in
        last_chat_metrics["stream_consistency_violations"].
        """
        metrics: dict[str, float] = {}
        chat_started = time.perf_counter()

        if not self._session_id:
            self.start_session()

        from .config import get_config

        _cfg = get_config()
        if len(message) > _cfg.memory.max_message_length:
            message = message[: _cfg.memory.max_message_length]

        self.memory.add(
            content=f"[User] {message}",
            tier="buffer",
            role="user",
            session_id=self._session_id,
            importance=0.5,
        )
        memories = self.retriever.retrieve(query=message, limit=10, relationship_target=user_id)
        rel_context = ""
        if user_id:
            self.relationships.get_or_create(user_id)
            rel_context = self.relationships.describe(user_id)
        messages = self._build_context(message, memories, rel_context)
        self.last_chat_messages = [dict(item) for item in messages]

        generate_started = time.perf_counter()
        chunks: list[str] = []
        for chunk in self.llm.generate_stream(messages, temperature=0.7):
            chunks.append(chunk)
            yield chunk
        response = "".join(chunks)
        metrics["generate_ms"] = round((time.perf_counter() - generate_started) * 1000.0, 2)

        # Post-hoc consistency (never retracts streamed text)
        if self.enforce_consistency and _cfg.character.consistency_stream_mode == "log":
            try:
                report = self.consistency.check(response, messages)
                violations = len(getattr(report, "hard_violations", []) or [])
                metrics["stream_consistency_violations"] = float(violations)
                if violations:
                    logger.warning(
                        "chat_stream: %d hard consistency violations (not retracted)",
                        violations,
                    )
            except Exception as e:
                logger.debug("Stream consistency check failed: %s", e)

        self._context.add_turn("user", message)
        self._context.add_turn("assistant", response)
        self.memory.add(
            content=f"[{self.name}] {response}",
            tier="buffer",
            role="character",
            session_id=self._session_id,
            importance=0.5,
        )

        self._turn_count += 1
        if self.background:
            self._get_worker().submit(
                "bookkeeping", self._run_subsystems_sequential, message, response, user_id
            )
        elif self.parallel and not self.lightweight:
            self._run_subsystems_parallel(message, response, user_id)
        else:
            self._run_subsystems_sequential(message, response, user_id)

        if self._turn_count % _cfg.memory.state_save_interval == 0:
            try:
                self._save_state()
            except Exception as e:
                logger.debug("Periodic state save failed: %s", e)

        metrics["total_ms"] = round((time.perf_counter() - chat_started) * 1000.0, 2)
        self.last_chat_metrics = metrics
        self._emit_metrics(metrics)
```

Check `ConsistencyChecker.check`'s exact signature in `persona/consistency.py` before wiring (the audit says `check()` takes the response + context; adapt the call to the real signature — do NOT guess).

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/ -x -q`.

```bash
git add src/woven_imprint/character.py src/woven_imprint/config.py tests/test_streaming.py
git commit -m "feat: Character.chat_stream with post-hoc consistency (A2)"
```

---

### Task 9: Resilience parity + tests

**Files:**
- Modify: `src/woven_imprint/llm/resilience.py`
- Modify: `src/woven_imprint/llm/gemma_edge.py:59-67` (`_post`)
- Modify: `src/woven_imprint/llm/openai_llm.py`, `src/woven_imprint/llm/anthropic_llm.py`
- Test: `tests/test_resilience.py`

**Interfaces:**
- Produces: thread-safe breaker registry (module lock); `_is_retryable` additionally treats any exception with an int `status_code` attribute in {429, 502, 503, 504} as retryable (covers openai/anthropic SDK errors without importing them); all four providers route calls through `resilient_call` with provider_names `"ollama" | "openai" | "anthropic" | "gemma_edge"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_resilience.py
import pytest
import requests

from woven_imprint.llm import resilience
from woven_imprint.llm.resilience import CircuitBreaker, _is_retryable, resilient_call, reset_breaker


@pytest.fixture(autouse=True)
def clean_breakers(monkeypatch):
    monkeypatch.setattr(resilience, "_breakers", {})
    # no real sleeping in tests
    monkeypatch.setattr(resilience.time, "sleep", lambda s: None)
    yield


def test_retries_transient_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.Timeout("slow")
        return "ok"

    assert resilient_call(flaky, provider_name="t1") == "ok"
    assert calls["n"] == 3


def test_permanent_error_not_retried():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        resilient_call(bad, provider_name="t2")
    assert calls["n"] == 1


def test_duck_typed_status_code_is_retryable():
    class SDKError(Exception):
        status_code = 429

    assert _is_retryable(SDKError()) is True

    class SDKPermanent(Exception):
        status_code = 400

    assert _is_retryable(SDKPermanent()) is False


def test_breaker_opens_after_threshold(monkeypatch):
    breaker = CircuitBreaker(threshold=2, cooldown=1000.0)
    monkeypatch.setattr(resilience, "_get_breaker", lambda name: breaker)

    def always_fails():
        raise ValueError("permanent")

    for _ in range(2):
        with pytest.raises(ValueError):
            resilient_call(always_fails, provider_name="t3")
    assert breaker.is_open
    with pytest.raises(ConnectionError, match="Circuit breaker open"):
        resilient_call(always_fails, provider_name="t3")


def test_breaker_resets_after_cooldown(monkeypatch):
    breaker = CircuitBreaker(threshold=1, cooldown=0.0)
    breaker.record_failure()
    assert breaker.is_open is False  # cooldown 0 → immediately reset


def test_gemma_edge_uses_resilience(monkeypatch):
    calls = {"n": 0}

    def flaky_post(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.Timeout("slow")

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"content": "hi"}

        return R()

    monkeypatch.setattr("requests.post", flaky_post)
    monkeypatch.setenv("WOVEN_IMPRINT_GEMMA_EDGE_URL", "http://x")
    from woven_imprint.llm.gemma_edge import GemmaEdgeLLM

    llm = GemmaEdgeLLM()
    assert llm.generate([{"role": "user", "content": "hi"}]) == "hi"
    assert calls["n"] == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_resilience.py -v`
Expected: `test_duck_typed_status_code_is_retryable` and `test_gemma_edge_uses_resilience` FAIL; the pure-breaker ones may pass — that's fine, they lock in current behavior.

- [ ] **Step 3: Implement**

In `resilience.py`:

1. Add after `_RETRYABLE_EXCEPTIONS`:

```python
def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is transient and worth retrying."""
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    # Duck-typed SDK errors (openai/anthropic APIStatusError expose .status_code)
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS_CODES
    return False
```

2. Thread-safety: add `import threading` and `_breakers_lock = threading.Lock()`; wrap the body of `_get_breaker` in `with _breakers_lock:`; in `CircuitBreaker`, guard `_failures`/`_tripped_at` mutation and `is_open` with an instance lock (`field(default_factory=threading.Lock, repr=False, compare=False)` won't work directly on a plain dataclass with the existing fields — simplest: convert `CircuitBreaker` to a regular class with the same attributes and an internal `self._lock = threading.Lock()`, preserving the public API: `threshold`, `cooldown`, `is_open`, `record_failure()`, `record_success()`).

In `gemma_edge.py`, replace `_post` with the resilient version (mirroring `ollama.py`):

```python
    def _post(self, endpoint: str, payload: dict) -> requests.Response:
        from .resilience import resilient_call

        def _do_post():
            resp = requests.post(
                f"{self.base_url}{endpoint}", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp

        return resilient_call(_do_post, provider_name="gemma_edge")
```

In `openai_llm.py`, wrap the two `self.client.chat.completions.create(...)` calls:

```python
        from .resilience import resilient_call

        response = resilient_call(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_name="openai",
        )
```

(same pattern for the json-mode call; `resilient_call(fn, *args, provider_name=..., **kwargs)` already forwards kwargs). In `anthropic_llm.py`, wrap `self.client.messages.create(...)` the same way with `provider_name="anthropic"`. Leave the SDK's own internal retries as-is (they compose — ours adds the circuit breaker).

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/ -x -q`.

```bash
git add src/woven_imprint/llm/resilience.py src/woven_imprint/llm/gemma_edge.py src/woven_imprint/llm/openai_llm.py src/woven_imprint/llm/anthropic_llm.py tests/test_resilience.py
git commit -m "fix: resilience parity across all providers + thread-safe breakers + tests (A5)"
```

---

### Task 10: Durable conversation buffer (`session_turns`, migration v3)

**Files:**
- Modify: `src/woven_imprint/storage/sqlite.py` (`_MIGRATIONS` + CRUD)
- Modify: `src/woven_imprint/context.py` (`load_turns`)
- Modify: `src/woven_imprint/character.py` (`chat`/`chat_stream`/`ingest` write turns; `resume_session` rehydrates)
- Test: `tests/test_session_turns.py`

**Interfaces:**
- Produces: migration v3 table `session_turns(id INTEGER PK AUTOINCREMENT, session_id, character_id, seq, role, content, created_at)`; `SQLiteStorage.add_session_turn(session_id, character_id, seq, role, content)`; `SQLiteStorage.get_session_turns(session_id, tail: int | None = None) -> list[dict]` (ordered by seq; `tail=N` returns last N); `ContextManager.load_turns(turns: list[dict])`; `Character._turn_seq: int` internal counter; `resume_session` loads the last `max_turns` turns into the context.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session_turns.py
from tests.helpers import make_test_engine


def test_turns_persisted_and_rehydrated():
    engine = make_test_engine(db_path=":memory:")
    char = engine.create_character("Memo")
    sid = char.start_session()
    char.chat("first message")
    char.chat("second message")

    turns = engine.storage.get_session_turns(sid)
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[0]["content"] == "first message"

    # Simulate cold start: fresh Character object, resume session
    char2 = engine.get_character(char.id)
    assert char2._context.turn_count == 0
    char2.resume_session(sid)
    assert char2._context.turn_count == 4
    msgs = char2._context.get_messages()
    assert msgs[0]["content"] == "first message"


def test_tail_limit():
    engine = make_test_engine(db_path=":memory:")
    char = engine.create_character("Taily")
    sid = char.start_session()
    for i in range(6):
        char.chat(f"msg {i}")
    tail = engine.storage.get_session_turns(sid, tail=4)
    assert len(tail) == 4
    assert tail[-1]["role"] == "assistant"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_session_turns.py -v`
Expected: FAIL with AttributeError (`get_session_turns` missing).

- [ ] **Step 3: Implement storage**

In `storage/sqlite.py`, extend `_MIGRATIONS`:

```python
_MIGRATIONS: dict[int, str] = {
    2: "ALTER TABLE sessions ADD COLUMN alias TEXT;",
    3: """
CREATE TABLE IF NOT EXISTS session_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_session_turns_session ON session_turns(session_id, seq);
""",
}
```

Add methods (near the sessions section):

```python
    def add_session_turn(
        self, session_id: str, character_id: str, seq: int, role: str, content: str
    ) -> None:
        self._conn.execute(
            "INSERT INTO session_turns (session_id, character_id, seq, role, content) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, character_id, seq, role, content),
        )
        self._commit()

    def get_session_turns(self, session_id: str, tail: int | None = None) -> list[dict]:
        rows = self._conn.execute(
            "SELECT seq, role, content, created_at FROM session_turns "
            "WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
        turns = [dict(r) for r in rows]
        if tail is not None:
            turns = turns[-tail:]
        return turns
```

- [ ] **Step 4: Implement ContextManager.load_turns**

In `context.py`:

```python
    def load_turns(self, turns: list[dict]) -> None:
        """Replace the buffer with persisted turns (oldest first)."""
        self.clear()
        for t in turns:
            self._turns.append(ConversationTurn(role=t["role"], content=t["content"]))
        self._enforce_limits()
```

- [ ] **Step 5: Wire into Character**

`__init__`: add `self._turn_seq: int = 0`.

Add helper:

```python
    def _persist_turn(self, role: str, content: str) -> None:
        if not self._session_id:
            return
        self._turn_seq += 1
        try:
            self.storage.add_session_turn(
                self._session_id, self.id, self._turn_seq, role, content
            )
        except Exception as e:
            logger.debug("Turn persistence failed: %s", e)
```

In `chat()` and `chat_stream()`, immediately after the `self._context.add_turn(...)` pair:

```python
        self._persist_turn("user", message)
        self._persist_turn("assistant", response)
```

In `ingest()`, after its `self._context.add_turn(role, content)`: `self._persist_turn(role, content)`.

In `start_session()`: reset `self._turn_seq = 0`.

Rewrite `resume_session()`:

```python
    def resume_session(self, session_id: str) -> str:
        """Resume a previous session, rehydrating recent turns into context."""
        self._session_id = session_id
        self._turn_count = 0
        self._context.clear()
        self.storage.reopen_session(session_id)
        try:
            turns = self.storage.get_session_turns(session_id, tail=self._context.max_turns)
            if turns:
                self._context.load_turns(turns)
                self._turn_seq = turns[-1]["seq"]
        except Exception as e:
            logger.debug("Session rehydration failed: %s", e)
        return session_id
```

- [ ] **Step 6: Run tests, commit**

Run: `pytest tests/ -x -q` (includes existing migration tests — v3 must apply cleanly on a v2 DB; `tests/test_migration.py` covers the pattern).

```bash
git add src/woven_imprint/storage/sqlite.py src/woven_imprint/context.py src/woven_imprint/character.py tests/test_session_turns.py
git commit -m "feat: persist conversation turns, rehydrate on resume (A6, migration v3)"
```

---

### Task 11: After-numbers, docs, changelog

**Files:**
- Modify: `docs/PERFORMANCE.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/DEVELOPER_GUIDE.md` (new APIs), `docs/CONFIGURATION.md` (new config keys)

**Interfaces:** none — documentation and measurement close-out.

- [ ] **Step 1: Re-run the benchmark**

Same command/environment as Task 1 Step 6 (same provider/model/hardware!). Paste the new table into `docs/PERFORMANCE.md` under "After Phase A", alongside a short interpretation: perceived `total_ms` should now approximate `generate_ms + consistency_ms` (bookkeeping moved off-path); note the `flush()`-inclusive wall time from the bench output as the honest total-cost figure.

- [ ] **Step 2: Acceptance check (spec §5)**

Verify from the bench output: perceived turn latency ≤ generate + consistency (+ small overhead). If `total_ms` still includes multi-LLM-call time, investigate before closing the task — `background` should default on outside tests.

- [ ] **Step 3: Update docs**

- `CHANGELOG.md`: add an `## [Unreleased]` section listing: background bookkeeping (`character.background`, `Character.flush()/close()`), `chat_stream()` + provider streaming, weighted RRF + tier-priority removal + `recency_anchor`, embedding cache, resilience on all providers, `session_turns` persistence + resume rehydration, metrics sink + bench script. Mark behavior changes: bookkeeping async by default; auto-consolidation no longer runs mid-chat (only at session end / explicit call); retrieval ranking changed.
- `docs/CONFIGURATION.md`: document every new key: `memory.rrf_k`, `memory.weight_*`, `memory.recency_anchor`, `character.background`, `character.metrics_path`, `character.consistency_stream_mode`, env vars `WOVEN_IMPRINT_BACKGROUND`, `WOVEN_IMPRINT_METRICS_PATH`. Also update `save_default_config`'s template string in `config.py` with the same keys.
- `docs/DEVELOPER_GUIDE.md`: short sections for `flush()/close()` discipline, `chat_stream` usage + stream-mode consistency tradeoff, and the metrics sink.

- [ ] **Step 4: Full CI-equivalent run + commit**

Run: `ruff check . && ruff format --check . && pyright && pytest tests/ -q && python eval/run_eval.py`
Expected: all green.

```bash
git add docs/PERFORMANCE.md CHANGELOG.md docs/CONFIGURATION.md docs/DEVELOPER_GUIDE.md src/woven_imprint/config.py
git commit -m "docs: Phase A performance results, config reference, changelog"
```

---

## Self-review (done at plan time)

- **Spec coverage:** A1 → Task 6 (+ inline-consolidation removal); A2 → Tasks 7–8; A3 → Tasks 2, 3, 4; A4 → Task 5; A5 → Task 9; A6 → Task 10; A7 → Tasks 1, 11. Health counters exist minimally as worker success/failure counts (full `health()` API is Phase B by spec).
- **Type consistency:** `BackgroundWorker.submit/flush/close`, `CachedEmbedder`, `generate_stream`, `add_session_turn/get_session_turns`, `load_turns`, `_persist_turn`, `_emit_metrics` used consistently across tasks.
- **Known judgment calls for the executor:** (1) exact local variable names in `engine.py` when wrapping the embedder; (2) `ConsistencyChecker.check` signature in Task 8 — read the file, don't guess; (3) test fallout in Tasks 2, 5, 6 is expected and must be fixed toward the new behavior, never by reverting the behavior change.
