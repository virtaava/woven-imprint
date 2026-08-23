"""Experiment 1: persona drift over 50 emotionally loaded turns, conditions A/B/C."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common, hard_checks, judge, lora  # noqa: E402

CONDITIONS = {
    "A": {"adapters": {}, "system": "full"},
    "B": {"adapters": {"persona": 1.0}, "system": "minimal"},
    "C": {"adapters": {"persona": 1.0}, "system": "full"},
}
MAX_HISTORY = 20  # keep the last N turns in context, like ContextConfig.max_turns


def run_condition(cond: str, seed: int, turns: list[str], llm, judge_llm, persona_prompt: str) -> dict:
    spec = CONDITIONS[cond]
    lora.set_adapters(spec["adapters"])
    system = persona_prompt if spec["system"] == "full" else common.MINIMAL_SYSTEM
    history: list[dict] = []
    rows = []
    for i, user in enumerate(turns, 1):
        msgs = [{"role": "system", "content": system}] + history[-2 * MAX_HISTORY:] + [{"role": "user", "content": user}]
        # seed is passed through extra_body-free path: llama-server honours 'seed' only via raw API;
        # we emulate seeds by temperature 0.7 sampling + distinct run order; record seed for bookkeeping.
        response = llm.generate(msgs, temperature=0.7, max_tokens=300).strip()
        history += [{"role": "user", "content": user}, {"role": "assistant", "content": response}]
        hard = hard_checks.check(response)
        js = judge.score(judge_llm, persona_prompt, user, response)
        rows.append({"i": i, "user": user, "response": response, "hard": hard, "judge": js})
        print(f"[{cond}/{seed}] {i:02d} judge={js['mean']:.2f} hard={'X' if hard['any'] else '.'}")
    means = [r["judge"]["mean"] for r in rows]
    summary = {
        "mean": sum(means) / len(means),
        "late_mean": sum(means[30:]) / max(1, len(means[30:])),
        "slope": hard_checks.slope(means),
        "hard_violations": sum(1 for r in rows if r["hard"]["any"]),
    }
    return {"condition": cond, "seed": seed, "turns": rows, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--turns", type=int, default=50)
    args = ap.parse_args()

    turns = json.loads((common.DATA_DIR / "drift_script.json").read_text(encoding="utf-8"))[: args.turns]
    llm, judge_llm = common.spike_llm(), common.brain_llm()
    persona_prompt = common.full_system_prompt()
    for cond in args.conditions:
        for seed in args.seeds:
            t0 = time.time()
            result = run_condition(cond, seed, turns, llm, judge_llm, persona_prompt)
            result["wall_s"] = round(time.time() - t0, 1)
            out = common.OUT_DIR / f"drift_{cond}_{seed}.json"
            out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
            print(cond, seed, result["summary"], f"{result['wall_s']}s -> {out}")
    lora.set_adapters({})


if __name__ == "__main__":
    main()
