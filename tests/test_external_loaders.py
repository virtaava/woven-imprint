from datetime import timezone
from pathlib import Path

from eval.external.common import parse_locomo_datetime
from eval.external.locomo import load_locomo, load_locomo_plus
from eval.external.longmemeval import load_longmemeval_s

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
