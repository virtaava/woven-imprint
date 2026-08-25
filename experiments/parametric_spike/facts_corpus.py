"""facts.json -> out/facts_corpus.jsonl training rows (train paraphrases only)."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def _pretty(date_iso: str) -> str:
    d = dt.date.fromisoformat(date_iso)
    return f"{d.strftime('%B')} {d.day}"


def facts_to_rows(facts: list[dict]) -> list[dict]:
    rows = []
    for f in facts:
        when = _pretty(f["date"])
        for p in f["train_paraphrases"]:
            rows.append({"messages": [
                {"role": "system", "content": common.MINIMAL_SYSTEM},
                {"role": "user", "content": p},
                {"role": "assistant", "content": f"{f['answer'][0]}. You told me on {when}."}],
                "kind": "fact"})
        rows.append({"messages": [
            {"role": "system", "content": common.MINIMAL_SYSTEM},
            {"role": "user", "content": f["temporal_question"]},
            {"role": "assistant", "content": f"You told me on {when}."}],
            "kind": "fact_temporal"})
    return rows


def main() -> None:
    facts = json.loads((common.DATA_DIR / "facts.json").read_text(encoding="utf-8"))
    rows = facts_to_rows(facts)
    common.write_jsonl(common.OUT_DIR / "facts_corpus.jsonl", rows)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
