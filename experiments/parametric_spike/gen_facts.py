"""Create 50 dated visitor facts with teacher-written paraphrases. Writes data/facts.json once."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

_TEMPLATES = [
    ("the visitor's cat is named {x}", "What is my cat's name?", ["Luma", "Pixel", "Moss", "Juniper", "Otto"]),
    ("the visitor was born in {x}", "Which city was I born in?", ["Tampere", "Oulu", "Turku", "Kuopio", "Vaasa"]),
    ("the visitor's favourite food is {x}", "What is my favourite food?", ["rye bread", "salmon soup", "karjalanpiirakka", "mustikkapiirakka", "lohikeitto"]),
    ("the visitor plays {x} on weekends", "What do I play on weekends?", ["floorball", "chess", "the cello", "padel", "disc golf"]),
    ("the visitor's sister is called {x}", "What is my sister's name?", ["Aino", "Venla", "Sofia", "Helmi", "Elsa"]),
    ("the visitor works as {x}", "What is my job?", ["a lighthouse keeper", "a luthier", "a pharmacist", "a ferry captain", "a beekeeper"]),
    ("the visitor is building {x}", "What am I building?", ["a survival game", "a quiz app", "a greenhouse", "a sailing dinghy", "a synthesizer"]),
    ("the visitor's car is a {x}", "What car do I drive?", ["green Volvo", "red Saab", "blue Skoda", "white Toyota", "grey Dacia"]),
    ("the visitor is allergic to {x}", "What am I allergic to?", ["hazelnuts", "cats", "birch pollen", "shellfish", "penicillin"]),
    ("the visitor's lucky number is {x}", "What is my lucky number?", ["17", "42", "9", "23", "61"]),
]


def base_facts(seed: int) -> list[dict]:
    rng = random.Random(seed)
    start = dt.date(2026, 1, 5)
    facts = []
    fid = 1
    for tpl, q, values in _TEMPLATES:
        for v in values:
            d = start + dt.timedelta(days=rng.randint(0, 200))
            facts.append({"id": fid, "date": d.isoformat(), "fact": tpl.format(x=v), "question": q, "answer": [v]})
            fid += 1
    rng.shuffle(facts)
    for i, f in enumerate(facts, 1):
        f["id"] = i
    return facts


def split_paraphrases(paras: list[str]) -> tuple[list[str], list[str]]:
    uniq = list(dict.fromkeys(p.strip() for p in paras if p.strip()))
    if len(uniq) < 11:
        raise ValueError(f"need 11 unique paraphrases, got {len(uniq)}")
    return uniq[:8], uniq[8:11]


def first_person(fact: str) -> str:
    """'the visitor's cat is named X' -> 'my cat is named X'; 'the visitor is building X' -> 'I am building X'."""
    out = fact.replace("the visitor's ", "my ")
    for a, b in (("the visitor is ", "I am "), ("the visitor was ", "I was "), ("the visitor plays ", "I play "),
                 ("the visitor works ", "I work "), ("the visitor ", "I ")):
        out = out.replace(a, b)
    return out


def _paraphrase(llm, question: str, fact: str) -> list[str]:
    data = llm.generate_json_robust([
        {"role": "system", "content": "You write paraphrases for evaluation data. Return JSON only."},
        {"role": "user", "content": (
            f"A visitor once told Meridian that {fact}. Write 13 distinct ways the visitor might later ask "
            f"the question \"{question}\" — vary wording, formality and length; never include the answer. "
            'Return {"paraphrases": [..13 strings..]}')},
    ], temperature=0.8)
    return list(data.get("paraphrases", [])) if isinstance(data, dict) else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=common.DATA_DIR / "facts.json")
    args = ap.parse_args()
    llm = common.brain_llm()
    facts = base_facts(args.seed)
    for f in facts:
        paras = _paraphrase(llm, f["question"], f["fact"])
        for _ in range(2):
            if len(dict.fromkeys(paras)) >= 11:
                break
            paras += _paraphrase(llm, f["question"], f["fact"])
        f["train_paraphrases"], f["test_paraphrases"] = split_paraphrases(paras)
        f["temporal_question"] = f"When did I tell you that {first_person(f['fact'])}?"
        print(f"fact {f['id']:02d} ok")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
