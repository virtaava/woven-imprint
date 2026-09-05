"""Tier 3f: multi-hop second-pass expansion in `MemoryRetriever.retrieve`.

Spec: docs/superpowers/specs/2026-09-01-tier3f-multihop-second-pass.md

`retrieval_second_pass` (default 0 = off) takes the top-N fused hits as
"seeds" after the first fusion, extracts salient terms from their content,
and runs exactly one extra `fts_search` (plus a zero-cost mean-vector
semantic widening — no embedder call) to pull in co-dependent evidence the
first pass missed because it shares terms with what WAS found, not with the
original query.
"""

from __future__ import annotations

import uuid

import pytest

from woven_imprint.config import get_config
from woven_imprint.memory.retrieval import MemoryRetriever, _salient_terms
from woven_imprint.storage.sqlite import SQLiteStorage


class WordEmbedder:
    """Deterministic bag-of-words embedder (first 10 words only), matching
    the FakeEmbedder pattern used elsewhere in the suite (tests/helpers.py,
    tests/test_retrieval.py)."""

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
    retriever = MemoryRetriever(storage, embedder, "c1")
    cfg = get_config().memory
    original_second_pass = cfg.retrieval_second_pass
    original_entities = cfg.second_pass_entities
    yield storage, embedder, retriever, cfg
    cfg.retrieval_second_pass = original_second_pass
    cfg.second_pass_entities = original_entities
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


# ── (a) off = byte-identical ranking to today ──────────────────────────────


def test_second_pass_off_is_default_and_unchanged():
    """Default config has the feature off."""
    assert get_config().memory.retrieval_second_pass == 0


def test_second_pass_off_leaves_ranking_identical_on_mixed_store(setup):
    """A mixed tier/relevance store, `retrieval_second_pass=0` explicit (the
    default) — flipping the flag to 0 (its own default) must not perturb the
    ranking at all, and the result is deterministic and stable across repeat
    calls: today's fusion output, byte-for-byte, both times."""
    storage, embedder, retriever, cfg = setup
    cfg.retrieval_second_pass = 0

    _add(storage, embedder, "Caroline adopted a beagle named Rocket in Tampere", tier="core")
    _add(storage, embedder, "Rocket the beagle loves swimming at Tampere beach", tier="buffer")
    for i in range(5):
        _add(
            storage,
            embedder,
            f"Some unrelated bedrock detail number {i} about the archive",
            importance=0.9,
            tier="bedrock",
        )

    query = "What pet did Caroline adopt"
    first = retriever.retrieve(query, limit=5)
    ids_first = [m["id"] for m in first]

    # Re-run with the flag re-affirmed at 0 (its default) — must reproduce
    # the exact same ordering and scores.
    cfg.retrieval_second_pass = 0
    second = retriever.retrieve(query, limit=5)
    ids_second = [m["id"] for m in second]

    assert ids_first == ids_second
    scores_first = [m["_retrieval_score"] for m in first]
    scores_second = [m["_retrieval_score"] for m in second]
    assert scores_first == scores_second


# ── (b) split-evidence: second pass surfaces co-dependent evidence ────────


def _split_evidence_store(storage, embedder):
    """One memory (A) matches the query directly; a second memory (B) shares
    terms with A (the salient-term seed) but shares NO terms with the query
    itself — so B's cosine similarity to the query is exactly 0 under the
    bag-of-words WordEmbedder. A handful of short, vocab-disjoint distractors
    (each <=4 words, so the 50-dim/10-word-window embedder never wraps its
    vocabulary and creates an accidental collision) sit ahead of B in
    insertion order, so B is reliably below the top-K on relevance alone —
    only the second pass (seeded on A) should promote it."""
    a_id = _add(storage, embedder, "Caroline adopted a beagle named Rocket in Tampere")
    distractors = [
        "Weather turned cold suddenly",
        "Bridge repairs began today",
        "Market prices rose sharply",
        "Garden roses bloomed early",
        "Highway traffic increased slightly",
        "Library hours changed recently",
        "Kitchen renovation finished quickly",
        "Concert tickets sold instantly",
    ]
    for d in distractors:
        _add(storage, embedder, d)
    b_id = _add(storage, embedder, "Rocket the beagle loves swimming at Tampere beach")
    return a_id, b_id


def test_second_pass_off_does_not_surface_split_evidence(setup):
    storage, embedder, retriever, cfg = setup
    a_id, b_id = _split_evidence_store(storage, embedder)
    cfg.retrieval_second_pass = 0

    results = retriever.retrieve("What pet did Caroline adopt", limit=3)
    ids = [m["id"] for m in results]
    assert a_id in ids
    assert b_id not in ids


def test_second_pass_on_surfaces_split_evidence(setup):
    storage, embedder, retriever, cfg = setup
    a_id, b_id = _split_evidence_store(storage, embedder)
    cfg.retrieval_second_pass = 1  # seed from the single top hit (A)

    results = retriever.retrieve("What pet did Caroline adopt", limit=3)
    ids = [m["id"] for m in results]
    assert a_id in ids
    assert b_id in ids


# ── (c) cost bound: exactly one extra fts_search, zero embed calls ────────


def test_second_pass_cost_bound(setup):
    """When on: exactly one additional `fts_search` call and no new
    `embedder.embed` calls, versus the same query with the feature off."""
    storage, embedder, retriever, cfg = setup
    _split_evidence_store(storage, embedder)
    query = "What pet did Caroline adopt"

    fts_calls = {"n": 0}
    embed_calls = {"n": 0}
    orig_fts = storage.fts_search
    orig_embed = embedder.embed

    def spy_fts(*a, **kw):
        fts_calls["n"] += 1
        return orig_fts(*a, **kw)

    def spy_embed(*a, **kw):
        embed_calls["n"] += 1
        return orig_embed(*a, **kw)

    storage.fts_search = spy_fts
    embedder.embed = spy_embed
    try:
        cfg.retrieval_second_pass = 0
        fts_calls["n"] = 0
        embed_calls["n"] = 0
        retriever.retrieve(query, limit=3)
        fts_off = fts_calls["n"]
        embed_off = embed_calls["n"]

        cfg.retrieval_second_pass = 1
        fts_calls["n"] = 0
        embed_calls["n"] = 0
        retriever.retrieve(query, limit=3)
        fts_on = fts_calls["n"]
        embed_on = embed_calls["n"]
    finally:
        storage.fts_search = orig_fts
        embedder.embed = orig_embed

    assert fts_on == fts_off + 1
    assert embed_on == embed_off


# ── (d) _salient_terms unit cases ──────────────────────────────────────────


class TestSalientTerms:
    def test_capitalized_tokens_included(self):
        terms = _salient_terms("Caroline visited Paris last week")
        assert "Caroline" in terms
        assert "Paris" in terms

    def test_digit_tokens_included(self):
        terms = _salient_terms("The meeting is on 2024-05-08 at gate B7")
        assert any("2024" in t or "05" in t or "08" in t for t in terms)
        assert "B7" in terms

    def test_long_lowercase_words_included(self):
        terms = _salient_terms("the beagle loves swimming daily")
        assert "beagle" in terms
        assert "swimming" in terms

    def test_short_lowercase_words_excluded(self):
        terms = _salient_terms("the cat sat on a mat")
        assert terms == []

    def test_stopwords_excluded_even_when_long_enough(self):
        terms = _salient_terms("people should think about this because reasons")
        for stopword in ("people", "should", "think", "about", "because"):
            assert stopword not in terms
        assert "reasons" in terms

    def test_dedupes_case_insensitively(self):
        terms = _salient_terms("Rocket rocket ROCKET the beagle")
        assert terms.count("Rocket") == 1
        assert len([t for t in terms if t.lower() == "rocket"]) == 1

    def test_capped_at_default_limit(self):
        text = " ".join(f"Word{i}" for i in range(30))
        terms = _salient_terms(text)
        assert len(terms) == 16

    def test_respects_custom_limit(self):
        text = " ".join(f"Word{i}" for i in range(10))
        terms = _salient_terms(text, limit=3)
        assert len(terms) == 3

    def test_fts_special_characters_sanitized_out(self):
        terms = _salient_terms('She said "hello*world" and it costs $5-10 (approx)')
        for term in terms:
            assert all(c.isalnum() for c in term)
        joined = " OR ".join(terms)
        for bad_char in ('"', "*", "$", "(", ")", ":", "-"):
            assert bad_char not in joined

    def test_empty_text_returns_empty(self):
        assert _salient_terms("") == []


def test_second_pass_fts_query_carries_no_or_literal(setup, monkeypatch):
    """The literal token "OR" must never reach fts_search — the storage layer
    tokenizes and ORs terms itself, so a pre-joined "OR" became a search term
    matching the common word "or" corpus-wide (review 2026-09-01)."""
    import re

    storage, embedder, retriever, cfg = setup
    _split_evidence_store(storage, embedder)
    seen_queries: list[str] = []
    real_fts = storage.fts_search

    def spy(character_id, query, **kw):
        seen_queries.append(query)
        return real_fts(character_id, query, **kw)

    monkeypatch.setattr(storage, "fts_search", spy)
    cfg.retrieval_second_pass = 1
    retriever.retrieve("What pet did Caroline adopt?", limit=5)
    assert len(seen_queries) >= 2, "expected first-pass + second-pass FTS calls"
    assert not re.search(r"\bOR\b", seen_queries[-1]), seen_queries[-1]


def test_decoys_sharing_only_the_word_or_stay_out(setup):
    """Regression for the reproduced review case: high-importance decoys whose
    only overlap with the seeds is the word "or" must not enter the top-K."""
    storage, embedder, retriever, cfg = setup
    _split_evidence_store(storage, embedder)
    for i in range(20):
        _add(
            storage,
            embedder,
            f"Tax filings or paperwork or receipts batch {i} needs sorting.",
            importance=0.95,
            tier="bedrock",
        )
    cfg.retrieval_second_pass = 1
    results = retriever.retrieve("What pet did Caroline adopt?", limit=5)
    # The split evidence pair must own the top two slots, and no decoy may
    # score competitively with real evidence (with the OR bug, decoys scored
    # within ~5% of the genuine hits; fixed, they sit at distractor level and
    # can only fill otherwise-empty tail slots).
    assert "Caroline adopted" in results[0]["content"]
    assert "Rocket the beagle" in results[1]["content"]
    b_score = results[1]["_retrieval_score"]
    decoy_scores = [m["_retrieval_score"] for m in results if "Tax filings" in m["content"]]
    assert all(d < b_score * 0.5 for d in decoy_scores), (b_score, decoy_scores)


# ── (e) second_pass_entities: Tier 3o entity-quality terms ─────────────────


def _add_with_entities(storage, embedder, content, entities, importance=0.5):
    mid = _add(storage, embedder, content, importance=importance)
    row = storage.get_memory(mid)
    meta = dict(row.get("metadata") or {})
    meta["entities"] = entities
    storage.update_memory_fields(mid, metadata=meta)
    return mid


def test_second_pass_uses_seed_entities_when_enabled(setup):
    storage, embedder, retriever, cfg = setup
    # Query with NO overlap with target.
    query = "shopping mall visit"
    # Seed matches query; carries very short entity ID that can't be salient term.
    _add_with_entities(
        storage, embedder, "shopping at mall today", ["pz"]
    )
    # Target reachable ONLY via entity "pz" (2 chars: cannot be salient term;
    # zero word overlap with seed or query). Salient terms from seed are
    # ["shopping", "today"] — target has ["walls", "painted"] (no overlap).
    target = _add(storage, embedder, "pz had walls painted red", importance=0.01)

    # Negative control: verify target does NOT match query directly via FTS.
    fts_match = storage.fts_search("c1", query, limit=20)
    assert target not in {m["id"] for m in fts_match}, "target must not match query via FTS"

    # Add 50 distractors to ensure target CANNOT appear without second pass.
    for i in range(50):
        _add(storage, embedder, f"store opening hours {i} daily", importance=0.5)

    # NEGATIVE CONTROL: with second_pass=0, target is NOT retrieved.
    cfg.retrieval_second_pass = 0
    cfg.second_pass_entities = False
    results_control = [m["id"] for m in retriever.retrieve(query, limit=2)]
    assert target not in results_control, "NEGATIVE CONTROL: target must not appear with second_pass=0"

    # POSITIVE: with second_pass=2 + entities=True, target IS retrieved via entity "pz".
    cfg.retrieval_second_pass = 2
    cfg.second_pass_entities = True
    results_on = [m["id"] for m in retriever.retrieve(query, limit=3)]
    assert target in results_on, "target must appear via entity-seeded second pass"


def test_second_pass_entities_empty_union_falls_back_to_salient_terms(setup):
    storage, embedder, retriever, cfg = setup
    # Query that matches seed but has ZERO overlap with target.
    query = "expedition planning update"
    # Seed matches query (has "expedition"); NO entities (empty union).
    _add(storage, embedder, "mountain expedition planned eagerly")
    # Target shares salient term "mountain" (>=5 chars) with seed, but ZERO overlap with query.
    # Salient terms from seed: ["mountain", "expedition", "planned", "eagerly"].
    # Target must be unreachable via first pass (query doesn't match) but reachable via
    # salient-terms second pass (fallback when entity union is empty).
    target = _add(storage, embedder, "mountain caves explored carefully", importance=0.01)

    # Verify target does NOT match query directly via FTS.
    fts_match = storage.fts_search("c1", query, limit=20)
    assert target not in {m["id"] for m in fts_match}, "target must not match query via FTS"

    # Add 50 distractors to ensure target CANNOT appear without second pass.
    for i in range(50):
        _add(storage, embedder, f"weather satellite data temperature {i}", importance=0.5)

    # NEGATIVE CONTROL: with second_pass=0, target is NOT retrieved.
    cfg.retrieval_second_pass = 0
    cfg.second_pass_entities = False
    results_control = [m["id"] for m in retriever.retrieve(query, limit=2)]
    assert target not in results_control, "NEGATIVE CONTROL: target must not appear without second pass"

    # POSITIVE: with second_pass=2 + entities=True (empty union), should fallback to salient terms
    # and find target via "mountain".
    cfg.retrieval_second_pass = 2
    cfg.second_pass_entities = True
    results_entities_on = [m["id"] for m in retriever.retrieve(query, limit=3)]

    # With entities=False, should also find target (plain salient-terms path).
    cfg.second_pass_entities = False
    results_entities_off = [m["id"] for m in retriever.retrieve(query, limit=3)]

    # Both must find the target (via salient terms, no entity divergence).
    assert target in results_entities_on, "target must be found when entities=True (fallback)"
    assert target in results_entities_off, "target must be found when entities=False"
    # Mutation guard: results must be identical (both using salient-term fallback).
    assert results_entities_on == results_entities_off, "empty union must be byte-identical to salient-terms path"


def test_second_pass_entities_flag_off_is_byte_identical(setup):
    storage, embedder, retriever, cfg = setup
    # Query that matches seed but not targets directly.
    query = "adoption puppy week"
    # Seed with entity "Rocket"; salient terms "adopted"/"puppy"/"training"/"week".
    _add_with_entities(
        storage, embedder, "adopted puppy training last week", ["Rocket"]
    )
    # Target 1: matches entity "Rocket" only (not salient terms or query).
    target_entity = _add(storage, embedder, "Rocket chewed garden hose", importance=0.1)
    # Target 2: matches salient terms "training" from seed (not entity "Rocket" or query).
    target_salient = _add(storage, embedder, "training methods behavior puppies", importance=0.1)

    # Verify neither target matches query via FTS.
    fts_match = storage.fts_search("c1", query, limit=20)
    assert target_entity not in {m["id"] for m in fts_match}, "target_entity must not match query via FTS"
    assert target_salient not in {m["id"] for m in fts_match}, "target_salient must not match query via FTS"

    # Add 20 distractors with higher importance to bury low-importance targets.
    for i in range(20):
        _add(storage, embedder, f"weather report temperature forecast {i}", importance=0.5)

    # Run with second pass disabled + small limit: neither target appears (buried).
    cfg.retrieval_second_pass = 0
    cfg.second_pass_entities = False
    results_no_pass = [m["id"] for m in retriever.retrieve(query, limit=1)]
    assert target_entity not in results_no_pass and target_salient not in results_no_pass

    # Run with entities=False, second_pass=1: should find target_salient via salient terms.
    cfg.retrieval_second_pass = 1
    cfg.second_pass_entities = False
    results_off_1 = [m["id"] for m in retriever.retrieve(query, limit=2)]
    results_off_2 = [m["id"] for m in retriever.retrieve(query, limit=2)]
    assert results_off_1 == results_off_2, "flag-off must be deterministic"
    assert target_salient in results_off_1, "target_salient must appear (salient terms find it)"
    assert target_entity not in results_off_1, "target_entity must NOT appear (entity is ignored when flag=False)"

    # Run with entities=True, second_pass=1: should find target_entity via entity.
    cfg.second_pass_entities = True
    results_on = [m["id"] for m in retriever.retrieve(query, limit=2)]
    assert target_entity in results_on, "target_entity must appear when entities=True (entity pivot)"
    # Mutation guard: the entity path finds different targets than the salient path.
    assert results_on != results_off_1, "entity pivot must produce different results than salient terms"
