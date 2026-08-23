"""Experiment 2: facts in weights (D) vs explicit memory store (E)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common, lora  # noqa: E402

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]

RESTART_EVERY = 40  # llama-server leaks host RAM per request; restart periodically to avoid OOM


def answer_correct(response: str, answers: list[str]) -> bool:
    r = response.lower()
    return any(a.lower() in r for a in answers)


def temporal_correct(response: str, date_iso: str) -> bool:
    d = dt.date.fromisoformat(date_iso)
    r = response.lower()
    if MONTHS[d.month - 1] not in r:
        return False
    m = re.search(MONTHS[d.month - 1] + r"\s+(\d{1,2})\b", r) or re.search(r"\b(\d{1,2})(st|nd|rd|th)?\s+of\s+" + MONTHS[d.month - 1], r)
    return True if m is None else int(m.group(1)) == d.day


def _queries(facts: list[dict]) -> list[dict]:
    q = []
    for f in facts:
        for p in f["test_paraphrases"]:
            q.append({"id": f["id"], "query": p, "kind": "recall", "answers": f["answer"], "date": f["date"]})
        q.append({"id": f["id"], "query": f["temporal_question"], "kind": "temporal", "answers": [], "date": f["date"]})
    return q


def run_D(facts, llm) -> list[dict]:
    lora.restart_server()
    lora.set_adapters({"facts": 1.0})
    items = []
    for i, q in enumerate(_queries(facts)):
        if i > 0 and i % RESTART_EVERY == 0:
            lora.restart_server()
            lora.set_adapters({"facts": 1.0})
        resp = llm.generate([{"role": "system", "content": common.MINIMAL_SYSTEM},
                             {"role": "user", "content": q["query"]}], temperature=0.0, max_tokens=120).strip()
        items.append(_item(q, resp))
    lora.set_adapters({})
    return items


def run_E(facts, llm) -> list[dict]:
    os.environ["WOVEN_IMPRINT_ENFORCE_CONSISTENCY"] = "false"
    os.environ["WOVEN_IMPRINT_LIGHTWEIGHT"] = "true"
    from woven_imprint import Engine
    from woven_imprint.config import reload_config
    from woven_imprint.data.meridian_persona import MERIDIAN_BIRTHDATE, MERIDIAN_PERSONA

    reload_config()
    lora.restart_server()
    lora.set_adapters({})
    engine = Engine(db_path=":memory:", llm=llm, embedding=common.spike_embedding())
    char = engine.create_character("Meridian", persona=MERIDIAN_PERSONA, birthdate=MERIDIAN_BIRTHDATE)
    lock = getattr(engine.storage, "_lock", None)
    for f in facts:
        d = dt.date.fromisoformat(f["date"])
        # Dated content: the spike's stand-in for the missing date line in _format_memories.
        mem = char.memory.add(f"The visitor told me on {d.strftime('%B')} {d.day}, {d.year} that {f['fact']}.",
                              tier="core", role="observation", importance=0.8)
        if lock is not None:
            with lock:
                engine.storage._conn.execute("UPDATE memories SET created_at=? WHERE id=?", (f["date"] + " 12:00:00", mem["id"]))
                engine.storage._conn.commit()
        else:
            engine.storage._conn.execute("UPDATE memories SET created_at=? WHERE id=?", (f["date"] + " 12:00:00", mem["id"]))
            engine.storage._conn.commit()
    items = []
    for i, q in enumerate(_queries(facts)):
        if i > 0 and i % RESTART_EVERY == 0:
            # E's engine talks to the same server via chat(); a restart between queries
            # is safe because each query starts a fresh session.
            lora.restart_server()
            lora.set_adapters({})
        char.start_session()
        resp = char.chat(q["query"], user_id="visitor").strip()
        items.append(_item(q, resp))
    return items


def _item(q: dict, resp: str) -> dict:
    ok = answer_correct(resp, q["answers"]) if q["kind"] == "recall" else temporal_correct(resp, q["date"])
    print(f"{q['kind']:8s} #{q['id']:02d} {'OK ' if ok else 'MISS'} {q['query'][:50]!r}")
    return {"id": q["id"], "query": q["query"], "kind": q["kind"], "response": resp, "correct": ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["D", "E"])
    ap.add_argument("--limit", type=int, default=None,
                     help="use only the first N facts (for smoke testing); default: all 50")
    args = ap.parse_args()
    facts = json.loads((common.DATA_DIR / "facts.json").read_text(encoding="utf-8"))
    if args.limit is not None:
        facts = facts[: args.limit]
    llm = common.spike_llm()
    for cond in args.conditions:
        items = run_D(facts, llm) if cond == "D" else run_E(facts, llm)
        rec = [i for i in items if i["kind"] == "recall"]
        tmp = [i for i in items if i["kind"] == "temporal"]
        summary = {"recall_acc": sum(i["correct"] for i in rec) / len(rec),
                   "temporal_acc": sum(i["correct"] for i in tmp) / len(tmp),
                   "n_recall": len(rec), "n_temporal": len(tmp)}
        out = common.OUT_DIR / f"facts_{cond}.json"
        out.write_text(json.dumps({"condition": cond, "items": items, "summary": summary}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(cond, summary, "->", out)


if __name__ == "__main__":
    main()
