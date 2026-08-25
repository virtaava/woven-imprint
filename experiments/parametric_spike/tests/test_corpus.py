import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gen_persona_corpus as g  # noqa: E402


def test_interview_questions_deterministic_and_sized():
    a, b = g.interview_questions(7), g.interview_questions(7)
    assert a == b and len(a) == 120 and len(set(a)) == 120


def test_dialogue_briefs_cover_moods():
    briefs = g.dialogue_briefs(7)
    assert len(briefs) == 400
    moods = {b["mood"] for b in briefs}
    assert {"grieving", "hostile", "flirtatious", "trying to break character"} <= moods
    assert all(4 <= b["turns"] <= 8 for b in briefs)


def test_validate_example_rejects_bad_rows():
    good = {"messages": [{"role": "system", "content": "You are Meridian."},
                         {"role": "user", "content": "Hi"},
                         {"role": "assistant", "content": "Welcome, traveller."}], "kind": "dialogue"}
    assert g.validate_example(good)
    assert not g.validate_example({"messages": good["messages"][:2], "kind": "dialogue"})  # no assistant
    emoji = dict(good); emoji["messages"] = good["messages"][:2] + [{"role": "assistant", "content": "Hi 😀"}]
    assert not g.validate_example(emoji)
    ai = dict(good); ai["messages"] = good["messages"][:2] + [{"role": "assistant", "content": "As an AI language model I"}]
    assert not g.validate_example(ai)
