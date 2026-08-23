import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gen_facts as gf  # noqa: E402
import facts_corpus as fc  # noqa: E402


def test_base_facts_50_unique_dated():
    facts = gf.base_facts(7)
    assert len(facts) == 50
    assert len({f["fact"] for f in facts}) == 50
    assert all(f["date"].startswith("2026-") for f in facts)
    assert all(f["answer"] for f in facts)


def test_split_paraphrases_8_3_no_overlap():
    paras = [f"q{i}" for i in range(11)]
    train, test = gf.split_paraphrases(paras)
    assert len(train) == 8 and len(test) == 3 and not set(train) & set(test)


def test_split_paraphrases_requires_11_unique():
    import pytest
    with pytest.raises(ValueError):
        gf.split_paraphrases(["a"] * 12)


def test_facts_to_rows_uses_train_paraphrases_only():
    fact = {"id": 1, "date": "2026-05-03", "fact": "the visitor's cat is named Pixel",
            "question": "What is my cat's name?", "answer": ["Pixel"],
            "train_paraphrases": ["cat name?"] * 8, "test_paraphrases": ["HELDOUT"] * 3,
            "temporal_question": "When did I tell you about my cat?"}
    rows = fc.facts_to_rows([fact])
    assert len(rows) == 8 + 1  # 8 paraphrases + 1 temporal
    assert all(r["messages"][0]["content"] == "You are Meridian." for r in rows)
    assert not any("HELDOUT" in r["messages"][1]["content"] for r in rows)
    assert "Pixel" in rows[0]["messages"][2]["content"]
    assert "May 3" in rows[-1]["messages"][2]["content"]
