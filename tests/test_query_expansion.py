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
            "museum visit",  # case-fold duplicate after strip
            "How many art events did I attend?",  # equals original -> dropped
            "",  # empty -> dropped
            "concert attended",
            "play attended",  # beyond n=2 cap once dups removed
        ]
    )
    out = _generate_expansions(llm, "How many art events did I attend?", 2)
    assert out == ["Museum Visit", "concert attended"]


@pytest.mark.parametrize(
    "bad",
    [
        {"queries": ["a"]},  # dict, not list
        [1, 2, 3],  # non-string items
        "not json-list",  # plain string
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
    orig_weight = cfg.query_expansion_weight
    yield storage, embedder, cfg
    cfg.query_expansion = orig_qe
    cfg.retrieval_second_pass = orig_sp
    cfg.query_expansion_weight = orig_weight
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


def test_query_expansion_weight_is_honored(setup):
    """`query_expansion_weight` controls how strongly the expansion lists'
    rank credit is mixed into the fused ranking. Same corpus/expansions as
    `test_expansion_surfaces_instance_memories`: a vanishingly small weight
    leaves the ranking indistinguishable from the unexpanded (query_expansion=0)
    baseline, while the 0.5 default visibly reorders it — proof the weight is
    actually read, not just accepted."""
    storage, embedder, cfg = setup
    _seed_aggregation_corpus(storage, embedder)
    query = "how many cultural outings in total"

    cfg.query_expansion = 0
    baseline = MemoryRetriever(storage, embedder, "c1").retrieve(query, limit=5)

    cfg.query_expansion = 3
    llm = FakeExpansionLLM(
        [
            "went to a gallery opening",
            "visited the modern museum",
            "enjoyed the symphony concert",
        ]
    )

    cfg.query_expansion_weight = 1e-6
    low_weight = MemoryRetriever(storage, embedder, "c1", llm=llm).retrieve(query, limit=5)

    cfg.query_expansion_weight = 0.5
    high_weight = MemoryRetriever(storage, embedder, "c1", llm=llm).retrieve(query, limit=5)

    # Negligible weight ~= no expansion influence at all: same ranking as the
    # query_expansion=0 baseline.
    assert [m["id"] for m in low_weight] == [m["id"] for m in baseline]
    # A real weight changes the ranking (a strict rank difference from the
    # negligible-weight run) — the expansion lists' credit is honored.
    assert [m["id"] for m in high_weight] != [m["id"] for m in low_weight]


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


def test_character_wires_llm_into_retriever():
    """eval/external/runner.py calls char.retriever.retrieve() directly —
    the Character must hand its LLM to the retriever or benchmark runs can
    never exercise expansion."""
    from tests.helpers import make_test_engine  # existing suite-wide factory

    engine = make_test_engine()
    char = engine.create_character("Wire Test", persona={})
    assert char.retriever.llm is char.llm
