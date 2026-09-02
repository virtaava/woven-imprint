# Tier 3i — LLM-Guided Query Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opt-in LLM-generated expansion queries at retrieval time so aggregation and multi-hop questions can reach instance-level evidence the original query's wording never matches.

**Architecture:** `MemoryRetriever` gains an optional `llm` handle. When `memory.query_expansion > 0` and an LLM is present, one JSON call rewrites the question into ≤N instance-level search queries; each surviving expansion gets one embedding (single `embed_batch` call) and one `fts_search`, contributing two extra RRF ranked lists at `memory.query_expansion_weight`. The off path (`query_expansion=0`, or no LLM handle) is byte-identical to today.

**Tech Stack:** Python 3.11, stdlib + existing project deps only (no new deps). SQLite FTS5, existing `reciprocal_rank_fusion`/`cosine_matrix` utils.

**Spec:** `docs/superpowers/specs/2026-09-02-tier3i-llm-query-expansion.md`

## Global Constraints

- Branch `feat/tier3i-query-expansion` off master `0ea7298`; commit per task.
- Never touch `experiments/parametric_spike/` or `kotlin/`.
- Lint/type scope: `.venv/bin/ruff check src tests eval demo` and `.venv/bin/pyright` must stay at 0 errors.
- Run tests with `.venv/bin/python -m pytest`.
- `query_expansion=0` (the default) must leave `retrieve()` behavior and its call pattern byte-identical: zero LLM calls, zero extra embed/FTS calls, identical result order.
- Expansion must NEVER raise out of `retrieve()`: any LLM/parse/embed failure degrades silently to the unexpanded ranking.
- Public API additive only.

---

### Task 1: Config fields `query_expansion` / `query_expansion_weight`

**Files:**
- Modify: `src/woven_imprint/config.py` (MemoryConfig, after `retrieval_second_pass: int = 0` ~line 123; and the commented example-YAML block near line 506)
- Test: `tests/test_config.py` (append)

**Interfaces:**
- Produces: `MemoryConfig.query_expansion: int = 0`, `MemoryConfig.query_expansion_weight: float = 0.5` — read later via `get_config().memory` inside `MemoryRetriever.retrieve` (Task 3).

- [ ] **Step 1: Write the failing test** — append to `tests/test_config.py`:

```python
def test_query_expansion_defaults():
    """Tier 3i (docs/superpowers/specs/2026-09-02-tier3i-llm-query-expansion.md):
    LLM query expansion ships opt-in — 0 queries by default, weight 0.5."""
    from woven_imprint.config import MemoryConfig

    cfg = MemoryConfig()
    assert cfg.query_expansion == 0
    assert cfg.query_expansion_weight == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py::test_query_expansion_defaults -v`
Expected: FAIL with `AttributeError: ... no attribute 'query_expansion'`

- [ ] **Step 3: Implement** — in `src/woven_imprint/config.py`, immediately after the `retrieval_second_pass: int = 0` field in `MemoryConfig`, add:

```python
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
```

In the commented example-YAML block (search for `# retrieval_second_pass: 0`), add below it:

```python
  # query_expansion: 0             # >0 = max LLM-generated expansion queries per retrieve
  # (aggregation/multi-hop recall); +1 LLM call + 1 embed_batch + N fts_search when on.
  # query_expansion_weight: 0.5    # RRF weight of each expansion's ranked lists.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py::test_query_expansion_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/woven_imprint/config.py tests/test_config.py
git commit -m "feat(tier3i): add query_expansion config fields (opt-in, default off)"
```

---

### Task 2: `_generate_expansions()` helper + prompt constant

**Files:**
- Modify: `src/woven_imprint/memory/retrieval.py` (module level, near `_salient_terms` ~line 82)
- Test: `tests/test_query_expansion.py` (create)

**Interfaces:**
- Consumes: `LLMProvider.generate_json_robust(messages, temperature=0.3) -> dict | list` (existing, `src/woven_imprint/llm/base.py`; raises `ValueError` after its one retry).
- Produces: `_generate_expansions(llm, query: str, n: int) -> list[str]` — module-level function in `retrieval.py`; returns `[]` on ANY failure; never raises. Also `_EXPANSION_PROMPT: str` module constant. Task 3 calls `_generate_expansions`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_query_expansion.py`:

```python
"""Tier 3i: LLM-guided query expansion in `MemoryRetriever.retrieve`.

Spec: docs/superpowers/specs/2026-09-02-tier3i-llm-query-expansion.md

`query_expansion` (default 0 = off) rewrites the query into instance-level
search queries via one LLM JSON call; each expansion adds one embedding +
one fts_search and two extra weighted-RRF ranked lists. Off path is
byte-identical; every failure mode degrades silently to the unexpanded
ranking.
"""

from __future__ import annotations

import uuid

import pytest

from woven_imprint.config import get_config
from woven_imprint.memory.retrieval import MemoryRetriever, _generate_expansions
from woven_imprint.storage.sqlite import SQLiteStorage


class FakeExpansionLLM:
    """Returns a canned generate_json_robust payload; counts calls."""

    def __init__(self, payload=None, exc: Exception | None = None):
        self.payload = payload if payload is not None else []
        self.exc = exc
        self.calls = 0
        self.last_messages = None

    def generate_json_robust(self, messages, temperature=0.3):
        self.calls += 1
        self.last_messages = messages
        if self.exc is not None:
            raise self.exc
        return self.payload


def test_generate_expansions_happy_path():
    llm = FakeExpansionLLM(["gallery opening attended", "museum visit", "art exhibit"])
    out = _generate_expansions(llm, "How many art events did I attend?", 3)
    assert out == ["gallery opening attended", "museum visit", "art exhibit"]
    assert llm.calls == 1


def test_generate_expansions_caps_dedups_and_drops_originals():
    llm = FakeExpansionLLM(
        [
            "  Museum Visit ",
            "museum visit",          # case-fold duplicate after strip
            "How many art events did I attend?",  # equals original -> dropped
            "",                       # empty -> dropped
            "concert attended",
            "play attended",          # beyond n=2 cap once dups removed
        ]
    )
    out = _generate_expansions(llm, "How many art events did I attend?", 2)
    assert out == ["Museum Visit", "concert attended"]


@pytest.mark.parametrize(
    "bad",
    [
        {"queries": ["a"]},   # dict, not list
        [1, 2, 3],             # non-string items
        "not json-list",       # plain string
    ],
)
def test_generate_expansions_garbage_payload_returns_empty(bad):
    llm = FakeExpansionLLM(bad)
    assert _generate_expansions(llm, "question", 3) == []


def test_generate_expansions_llm_error_returns_empty():
    llm = FakeExpansionLLM(exc=ValueError("no JSON"))
    assert _generate_expansions(llm, "question", 3) == []
    llm2 = FakeExpansionLLM(exc=RuntimeError("connection refused"))
    assert _generate_expansions(llm2, "question", 3) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_query_expansion.py -v`
Expected: FAIL with `ImportError: cannot import name '_generate_expansions'`

- [ ] **Step 3: Implement** — in `src/woven_imprint/memory/retrieval.py`, after `_salient_terms` (module level):

```python
# Tier 3i (docs/superpowers/specs/2026-09-02-tier3i-llm-query-expansion.md):
# prompt for the one expansion call. The model must return a bare JSON list of
# strings — instance-level search queries, not answers. Kept short: it runs
# once per retrieve() when `query_expansion` > 0.
_EXPANSION_PROMPT = (
    "You rewrite a question into search queries for a personal-memory database.\n"
    "The database stores one small dated memory per event (e.g. \"went to a gallery"
    " opening\", \"donated $50 at the bake sale\").\n"
    "Aggregate questions (how many / how much / in total / list all / what order)"
    " can only be answered by finding EVERY individual instance, so produce"
    " instance-level rephrasings naming the concrete things someone would have"
    " mentioned. For questions about connected entities, add a query naming the"
    " linking entity.\n"
    "Return ONLY a JSON list of at most {n} short search queries (3-8 words each),"
    " no explanations. Do not repeat the original question.\n"
    "Question: {query}"
)


def _generate_expansions(llm, query: str, n: int) -> list[str]:
    """One LLM JSON call -> up to `n` cleaned expansion queries.

    Returns [] on ANY failure (LLM error, non-list payload, non-string items,
    nothing left after cleaning) — expansion must never break retrieval.
    Cleaning: strip; drop empties; case-fold dedup; drop anything equal to the
    original query (case-folded); cap at `n`.
    """
    try:
        payload = llm.generate_json_robust(
            [{"role": "user", "content": _EXPANSION_PROMPT.format(n=n, query=query)}],
            temperature=0.0,
        )
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    seen: set[str] = {query.casefold().strip()}
    out: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            return []
        cleaned = item.strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= n:
            break
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_query_expansion.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src/woven_imprint/memory/retrieval.py tests/test_query_expansion.py
git commit -m "feat(tier3i): _generate_expansions helper + expansion prompt"
```

---

### Task 3: Expansion stage in `MemoryRetriever.retrieve` + optional `llm` handle

**Files:**
- Modify: `src/woven_imprint/memory/retrieval.py` (`__init__` ~line 231; `retrieve()` — insert the expansion block AFTER the Tier 3f `retrieval_second_pass` block, immediately BEFORE the `# Return top-N memories` comment ~line 526)
- Test: `tests/test_query_expansion.py` (append)

**Interfaces:**
- Consumes: `_generate_expansions(llm, query, n)` (Task 2); `MemoryConfig.query_expansion` / `.query_expansion_weight` (Task 1); existing `reciprocal_rank_fusion(ranked_lists, k, weights)`, `cosine_matrix(vec, vecs)`, `self.embedder.embed_batch(texts)`, `self.storage.fts_search(character_id, query, limit)`.
- Produces: `MemoryRetriever.__init__(self, storage, embedder, character_id, llm=None)` — Task 4 (Character) passes `llm=`. Local variables consumed inside `retrieve` only.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_query_expansion.py` (reuses `FakeExpansionLLM`; embedder/fixture pattern copied from `tests/test_retrieval_second_pass.py`):

```python
class WordEmbedder:
    """Deterministic bag-of-words embedder (first 10 words only), matching
    the FakeEmbedder pattern used in tests/test_retrieval_second_pass.py."""

    def __init__(self, dims: int = 50):
        self._vocab: dict[str, int] = {}
        self._next = 0
        self._dims = dims

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dims
        for word in text.lower().split()[:10]:
            if word not in self._vocab:
                self._vocab[word] = self._next % self._dims
                self._next += 1
            vec[self._vocab[word]] += 1.0
        mag = sum(x * x for x in vec) ** 0.5
        if mag > 0:
            vec = [x / mag for x in vec]
        return vec

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def dimensions(self):
        return self._dims


@pytest.fixture
def setup():
    storage = SQLiteStorage(":memory:")
    storage.save_character("c1", "Alice", {})
    embedder = WordEmbedder()
    cfg = get_config().memory
    orig_qe = cfg.query_expansion
    orig_sp = cfg.retrieval_second_pass
    yield storage, embedder, cfg
    cfg.query_expansion = orig_qe
    cfg.retrieval_second_pass = orig_sp
    storage.close()


def _add(storage, embedder, content, importance=0.5, tier="core"):
    mid = str(uuid.uuid4())
    storage.save_memory(
        {
            "id": mid,
            "character_id": "c1",
            "tier": tier,
            "content": content,
            "embedding": embedder.embed(content),
            "importance": importance,
        }
    )
    return mid


def _seed_aggregation_corpus(storage, embedder):
    """Instance memories that share no vocabulary with the aggregate question."""
    ids = {}
    ids["gallery"] = _add(storage, embedder, "went to a gallery opening downtown")
    ids["museum"] = _add(storage, embedder, "visited the modern museum with Sam")
    ids["concert"] = _add(storage, embedder, "enjoyed the symphony concert hall")
    for i in range(10):
        _add(storage, embedder, f"cooked dinner recipe number {i} tonight")
    return ids


def test_off_by_default_no_llm_calls_and_identical_results(setup):
    storage, embedder, cfg = setup
    _seed_aggregation_corpus(storage, embedder)
    llm = FakeExpansionLLM(["visited the modern museum"])
    baseline = MemoryRetriever(storage, embedder, "c1").retrieve("art events attended", limit=5)
    cfg.query_expansion = 0
    with_llm = MemoryRetriever(storage, embedder, "c1", llm=llm).retrieve(
        "art events attended", limit=5
    )
    assert llm.calls == 0
    assert [m["id"] for m in with_llm] == [m["id"] for m in baseline]


def test_config_on_but_no_llm_handle_is_silently_off(setup):
    storage, embedder, cfg = setup
    _seed_aggregation_corpus(storage, embedder)
    cfg.query_expansion = 3
    baseline_cfg_off = None
    cfg.query_expansion = 0
    baseline_cfg_off = MemoryRetriever(storage, embedder, "c1").retrieve(
        "art events attended", limit=5
    )
    cfg.query_expansion = 3
    no_handle = MemoryRetriever(storage, embedder, "c1").retrieve("art events attended", limit=5)
    assert [m["id"] for m in no_handle] == [m["id"] for m in baseline_cfg_off]


def test_expansion_surfaces_instance_memories(setup):
    storage, embedder, cfg = setup
    ids = _seed_aggregation_corpus(storage, embedder)
    cfg.query_expansion = 3
    llm = FakeExpansionLLM(
        [
            "went to a gallery opening",
            "visited the modern museum",
            "enjoyed the symphony concert",
        ]
    )
    retriever = MemoryRetriever(storage, embedder, "c1", llm=llm)
    results = retriever.retrieve("how many cultural outings in total", limit=5)
    assert llm.calls == 1
    got = {m["id"] for m in results}
    # The three instance memories share no words with the query; only the
    # expansion lists can rank them into the top 5 above the 10 dinner rows.
    assert {ids["gallery"], ids["museum"], ids["concert"]} <= got


def test_llm_failure_degrades_to_unexpanded_ranking(setup):
    storage, embedder, cfg = setup
    _seed_aggregation_corpus(storage, embedder)
    cfg.query_expansion = 0
    baseline = MemoryRetriever(storage, embedder, "c1").retrieve("cultural outings", limit=5)
    cfg.query_expansion = 3
    broken = FakeExpansionLLM(exc=RuntimeError("brain offline"))
    results = MemoryRetriever(storage, embedder, "c1", llm=broken).retrieve(
        "cultural outings", limit=5
    )
    assert broken.calls == 1
    assert [m["id"] for m in results] == [m["id"] for m in baseline]


def test_composes_with_second_pass_enabled(setup):
    storage, embedder, cfg = setup
    ids = _seed_aggregation_corpus(storage, embedder)
    cfg.query_expansion = 2
    cfg.retrieval_second_pass = 3
    llm = FakeExpansionLLM(["gallery opening downtown", "modern museum visited"])
    retriever = MemoryRetriever(storage, embedder, "c1", llm=llm)
    results = retriever.retrieve("how many cultural outings in total", limit=5)
    assert llm.calls == 1
    assert {ids["gallery"], ids["museum"]} <= {m["id"] for m in results}


def test_empty_query_skips_expansion(setup):
    storage, embedder, cfg = setup
    _seed_aggregation_corpus(storage, embedder)
    cfg.query_expansion = 3
    llm = FakeExpansionLLM(["anything"])
    MemoryRetriever(storage, embedder, "c1", llm=llm).retrieve("", limit=5)
    assert llm.calls == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_query_expansion.py -v`
Expected: Task-2 tests PASS; the new ones FAIL (`TypeError: __init__() got an unexpected keyword argument 'llm'` and/or assertion failures).

- [ ] **Step 3: Implement** — two edits in `src/woven_imprint/memory/retrieval.py`:

(a) Constructor (currently `def __init__(self, storage: SQLiteStorage, embedder: EmbeddingProvider, character_id: str):`):

```python
    def __init__(
        self,
        storage: SQLiteStorage,
        embedder: EmbeddingProvider,
        character_id: str,
        llm=None,
    ):
        self.storage = storage
        self.embedder = embedder
        self.character_id = character_id
        # Tier 3i: optional LLM handle for query expansion. None (the default)
        # keeps expansion off regardless of config — the retriever must work
        # LLM-free everywhere it does today.
        self.llm = llm
```

(b) In `retrieve()`, immediately BEFORE the `# Return top-N memories` comment (i.e. after the entire Tier 3f block; note the Tier 3f block rebinds `fused` when it runs):

```python
        # ── Tier 3i: LLM-guided query expansion (off by default) ──────────────
        # When `query_expansion` (N) > 0 AND this retriever holds an LLM handle,
        # one JSON call rewrites the query into <=N instance-level search
        # queries (aggregation questions — "how many X in total" — need
        # instance memories individually dissimilar to the aggregate phrasing;
        # LME-S-100 multi-session J 0.31). Each expansion gets one embedding
        # (single embed_batch call) + one fts_search; its semantic and keyword
        # rankings join the final RRF as extra lists at
        # `query_expansion_weight`. The already-built lists are reused as-is
        # (strategies 3-5 are NOT re-run, the relevance gate is not
        # re-derived): rows surfaced only by an expansion earn credit purely
        # through the expansion lists — the same additive-credit rationale as
        # the Tier 3f strategy-2 extension. Every failure (LLM, parse, embed,
        # FTS) degrades silently to the unexpanded ranking.
        if self.llm is not None and mem_cfg.query_expansion > 0 and query.strip():
            expansions = _generate_expansions(self.llm, query, mem_cfg.query_expansion)
            exp_vectors: list[list[float]] = []
            if expansions:
                try:
                    exp_vectors = self.embedder.embed_batch(expansions)
                except Exception:
                    expansions = []
            if expansions and len(exp_vectors) == len(expansions):
                exp_lists: list[list[str]] = []
                new_exp_rows = False
                exp_fts_hits: list[list[dict]] = []
                for exp in expansions:
                    try:
                        hits = self.storage.fts_search(self.character_id, exp, limit=50)
                    except Exception:
                        hits = []
                    exp_fts_hits.append(hits)
                    for m in hits:
                        if m["id"] not in memory_map:
                            memory_map[m["id"]] = m
                            all_memories.append(m)
                            new_exp_rows = True
                if new_exp_rows:
                    all_memories.sort(key=lambda m: m.get("rowid", 0))
                embedded_exp = [m for m in all_memories if m.get("embedding")]
                exp_weights: list[float] = []
                for exp_vec, hits in zip(exp_vectors, exp_fts_hits):
                    sims_e = cosine_matrix(exp_vec, [m["embedding"] for m in embedded_exp])
                    scored = list(zip((m["id"] for m in embedded_exp), sims_e))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    exp_lists.append([mid for mid, _ in scored])
                    exp_weights.append(mem_cfg.query_expansion_weight)
                    exp_lists.append([m["id"] for m in hits])
                    exp_weights.append(mem_cfg.query_expansion_weight)
                if exp_lists:
                    # `base_lists`/`base_weights` = whatever fusion last ran:
                    # the second-pass lists when Tier 3f was active, else the
                    # first-pass lists.
                    if mem_cfg.retrieval_second_pass > 0 and "ranked_lists2" in locals():
                        base_lists, base_weights = ranked_lists2, weights2
                    else:
                        base_lists, base_weights = ranked_lists, weights
                    fused = reciprocal_rank_fusion(
                        base_lists + exp_lists,
                        k=mem_cfg.rrf_k,
                        weights=base_weights + exp_weights,
                    )
```

IMPORTANT implementation notes for this step:
- The Tier 3f block defines `ranked_lists2`/`weights2` only when it actually ran AND had seeds; the `"ranked_lists2" in locals()` guard covers the `retrieval_second_pass>0`-but-no-seeds path. Keep it exactly as written.
- Do NOT touch anything above the insertion point. The `fused` rebind is the block's only output.

- [ ] **Step 4: Run the new tests + both neighbors' suites**

Run: `.venv/bin/python -m pytest tests/test_query_expansion.py tests/test_retrieval.py tests/test_retrieval_second_pass.py tests/test_retrieval_fullscan.py -v`
Expected: ALL PASS (off-path byte-identity keeps the older suites green).

- [ ] **Step 5: Commit**

```bash
git add src/woven_imprint/memory/retrieval.py tests/test_query_expansion.py
git commit -m "feat(tier3i): LLM query-expansion stage in retrieve(), opt-in via llm handle"
```

---

### Task 4: Character wires its LLM into the retriever

**Files:**
- Modify: `src/woven_imprint/character.py:257` (the `MemoryRetriever(storage, embedder, char_id)` construction)
- Test: `tests/test_query_expansion.py` (append)

**Interfaces:**
- Consumes: `MemoryRetriever.__init__(..., llm=None)` (Task 3); `Character.__init__` already binds `self.llm = llm` at line 244 before the retriever is built at line 257.
- Produces: `Character.retriever.llm is Character.llm` — this is what makes the eval runner's `char.retriever.retrieve(...)` path (eval/external/runner.py:308) expansion-capable with zero runner changes.

- [ ] **Step 1: Write the failing test** — append to `tests/test_query_expansion.py`:

```python
def test_character_wires_llm_into_retriever():
    """eval/external/runner.py calls char.retriever.retrieve() directly —
    the Character must hand its LLM to the retriever or benchmark runs can
    never exercise expansion."""
    from tests.helpers import make_test_engine  # existing suite-wide factory

    engine = make_test_engine()
    char = engine.create_character(name="Wire Test", backstory="test")
    assert char.retriever.llm is char.llm
```

NOTE: `make_test_engine()` is at `tests/helpers.py:78`; `create_character`'s exact
keyword names can be checked in any `tests/test_engine.py` test if the call above
does not match — adjust the call, not the assertion.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_query_expansion.py::test_character_wires_llm_into_retriever -v`
Expected: FAIL with `AssertionError` (retriever.llm is None)

- [ ] **Step 3: Implement** — in `src/woven_imprint/character.py` line 257 change:

```python
        self.retriever = MemoryRetriever(storage, embedder, char_id, llm=llm)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_query_expansion.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src/woven_imprint/character.py tests/test_query_expansion.py
git commit -m "feat(tier3i): Character passes its LLM to MemoryRetriever"
```

---

### Task 5: Full-suite gate + long-horizon bench

**Files:**
- No source changes expected; this task is the regression gate before measurement.

**Interfaces:**
- Consumes: everything above.
- Produces: a green tree at the branch tip; the controller then runs the `locomo-t3iA` measurement chain (NOT part of this plan's tasks).

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: 836 pre-existing + new tests, 0 failures (1 pre-existing skip OK).

- [ ] **Step 2: Lint + types**

Run: `.venv/bin/ruff check src tests eval demo && .venv/bin/pyright`
Expected: "All checks passed!" and "0 errors, 0 warnings".

- [ ] **Step 3: Long-horizon behavioral bench**

Run: `.venv/bin/python -m eval.bench_longhorizon`
Expected: 12/12 scenarios pass (expansion is off by default — this proves the off path).

- [ ] **Step 4: Commit (only if any fix was needed)**

```bash
git add -A && git commit -m "fix(tier3i): suite/bench fixes"
```
