from collections import Counter
from datetime import timezone
from pathlib import Path

import pytest

from eval.external.common import DATA_DIR, parse_locomo_datetime
from eval.external.locomo import load_locomo, load_locomo_plus
from eval.external.longmemeval import _stratified_sample, load_longmemeval_s

FIX = Path(__file__).resolve().parent.parent / "eval" / "external" / "fixtures"


def test_parse_locomo_datetime():
    dt = parse_locomo_datetime("1:56 pm on 8 May, 2023")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (
        2023,
        5,
        8,
        13,
        56,
    ) and dt.tzinfo == timezone.utc


def test_load_locomo_mini_shapes():
    convs = load_locomo(FIX / "locomo_mini.json")
    assert len(convs) == 1
    c = convs[0]
    assert c.user_name and c.character_name and len(c.sessions) == 2
    assert all(t.role in ("user", "assistant") for s in c.sessions for t in s.turns)
    assert c.sessions[0].turns[0].role == "user"  # speaker_a is the user
    cats = sorted(q.category for q in c.questions)
    assert cats == ["1", "2", "3", "4", "5", "5"] or "5" in cats
    adv = [q for q in c.questions if q.kind == "adversarial"]
    assert adv and all(q.category == "5" for q in adv)
    assert c.questions[0].asked_at > c.sessions[-1].at
    assert c.sessions[0].turns[1].at > c.sessions[0].turns[0].at
    assert "[" in c.transcript_text and c.user_name in c.transcript_text


def test_load_locomo_plus_mini():
    from datetime import timedelta

    base = load_locomo(FIX / "locomo_mini.json")
    probes = load_locomo_plus(FIX / "locomo_plus_mini.json", base)
    assert len(probes) == 2 and probes[0].base_conv_id == base[0].conv_id
    p = probes[0]
    assert p.trigger_at == base[0].sessions[-1].at + timedelta(days=7)
    assert p.cue.at == p.trigger_at - timedelta(days=14)  # "two weeks later"
    assert p.cue.turns[0].role == "user" and p.cue.turns[0].speaker == base[0].user_name
    assert p.trigger_text and not p.trigger_text.startswith("A:")
    assert base[0].user_name in p.evidence_text and 'said, "' in p.stitched_text
    assert p.stitched_text.rstrip().endswith('"')  # trigger is the last line


def test_load_longmemeval_mini_and_sample():
    convs = load_longmemeval_s(FIX / "longmemeval_mini.json")
    assert len(convs) == 3
    kinds = {q.kind for c in convs for q in c.questions}
    assert {"qa", "abstain"} <= kinds
    sampled = load_longmemeval_s(FIX / "longmemeval_mini.json", sample=2, seed=1)
    assert len(sampled) == 2


def test_locomo_image_caption_folded_into_turn_text():
    convs = load_locomo(FIX / "locomo_mini.json")
    turn = convs[0].sessions[0].turns[2]
    assert "shared a photo: a cat sitting on a windowsill" in turn.text
    assert "windowsill" in convs[0].transcript_text


def test_longmemeval_sessions_sorted_chronologically(tmp_path):
    import json

    data = json.loads((FIX / "longmemeval_mini.json").read_text())
    item = data[0]
    # Reverse the haystack order in the file; loader must restore time order.
    item["haystack_sessions"] = list(reversed(item["haystack_sessions"]))
    item["haystack_dates"] = list(reversed(item["haystack_dates"]))
    p = tmp_path / "lme.json"
    p.write_text(json.dumps([item]))
    conv = load_longmemeval_s(p)[0]
    ats = [s.at for s in conv.sessions]
    assert ats == sorted(ats) and len(ats) == len(item["haystack_dates"])


# --- Stratified sample: prefix stability -------------------------------------------------


def _synthetic_items(counts: dict[str, int]) -> list[dict]:
    """Build a synthetic LongMemEval-shaped item list: ``{question_id, question_type}`` only
    (the fields ``_stratified_sample`` actually reads)."""
    items = []
    for qtype, n in counts.items():
        for j in range(n):
            items.append({"question_id": f"{qtype}-{j}", "question_type": qtype})
    return items


def _balance(items: list[dict]) -> int:
    counts = Counter(it["question_type"] for it in items)
    return max(counts.values()) - min(counts.values())


def test_stratified_sample_is_prefix_stable_and_balanced():
    # ~120 items across 6 types, uneven counts per type — each type still has more than
    # ceil(100 / 6) items so no type exhausts before the largest sample (100) is drawn (a type
    # smaller than that would cap out early and necessarily throw off the max-min<=1 balance
    # check below, which is an inherent property of round-robin sampling, not a bug).
    items = _synthetic_items({"t0": 17, "t1": 18, "t2": 19, "t3": 20, "t4": 21, "t5": 25})
    assert len(items) == 120

    sample_50 = _stratified_sample(items, 50, seed=7)
    sample_100 = _stratified_sample(items, 100, seed=7)

    assert len(sample_50) == 50 and len(sample_100) == 100
    ids_50 = [it["question_id"] for it in sample_50]
    ids_100 = [it["question_id"] for it in sample_100]
    assert ids_50 == ids_100[:50]

    assert _balance(sample_50) <= 1
    assert _balance(sample_100) <= 1


@pytest.mark.skipif(
    not (DATA_DIR / "longmemeval_s.json").exists(),
    reason="longmemeval_s.json not fetched (gitignored, 277 MB)",
)
def test_stratified_sample_prefix_stable_on_real_longmemeval_s():
    import json

    data = json.loads((DATA_DIR / "longmemeval_s.json").read_text())

    sample_50 = _stratified_sample(data, 50, seed=7)
    sample_100 = _stratified_sample(data, 100, seed=7)

    ids_50 = [it["question_id"] for it in sample_50]
    ids_100 = [it["question_id"] for it in sample_100]
    assert ids_50 == ids_100[:50]
    assert _balance(sample_50) <= 1
    assert _balance(sample_100) <= 1

    counts_50 = Counter(it["question_type"] for it in sample_50)
    counts_100 = Counter(it["question_type"] for it in sample_100)
    print(f"\nlongmemeval_s per-type counts, sample=50: {dict(counts_50)}")
    print(f"longmemeval_s per-type counts, sample=100: {dict(counts_100)}")


def test_truncating_embedding_bounds_input_and_keeps_interface():
    from eval.external.common import TruncatingEmbedding

    class Inner:
        model = "nomic-embed-text"

        def __init__(self):
            self.seen = []

        def embed(self, text):
            self.seen.append(text)
            return [1.0, 0.0]

        def embed_batch(self, texts):
            self.seen.extend(texts)
            return [[1.0, 0.0] for _ in texts]

        def dimensions(self):
            return 2

    inner = Inner()
    emb = TruncatingEmbedding(inner, max_chars=10)
    assert emb.embed("x" * 50) == [1.0, 0.0] and inner.seen[-1] == "x" * 10
    emb.embed_batch(["short", "y" * 30])
    assert inner.seen[-2:] == ["short", "y" * 10]
    assert emb.dimensions() == 2 and emb.model == "nomic-embed-text"
