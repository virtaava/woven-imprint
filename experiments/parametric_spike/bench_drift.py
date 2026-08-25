"""Experiment 1: persona drift over 50 emotionally loaded turns, conditions A/B/C."""
from __future__ import annotations

import argparse
import json
import shutil
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
RESTART_EVERY = 25  # llama-server leaks host RAM per request; restart periodically to avoid OOM
LATE_FROM_TURN = 31  # late window is turns 31-50


def summarize(rows: list[dict]) -> dict:
    """Aggregate a condition/seed run's turns into a summary dict.

    Only turns whose judge result is valid (judge.score succeeded, possibly
    after one retry) feed mean/late_mean/slope — a judge-formatting failure
    must not be indistinguishable from genuine persona collapse (both used
    to silently zero-fill). hard_violations is unaffected: it is a
    deterministic regex check on the response text, independent of the judge.
    """
    valid = [r for r in rows if r["judge"].get("valid")]
    means = [r["judge"]["mean"] for r in valid]
    late_means = [r["judge"]["mean"] for r in valid if r["i"] >= LATE_FROM_TURN]
    return {
        "mean": sum(means) / len(means) if means else 0.0,
        "late_mean": sum(late_means) / max(1, len(late_means)),
        "slope": hard_checks.slope(means),
        "hard_violations": sum(1 for r in rows if r["hard"]["any"]),
        "judged": len(valid),
        "invalid_judgements": len(rows) - len(valid),
    }


def run_condition(cond: str, seed: int, turns: list[str], llm, judge_llm, persona_prompt: str) -> dict:
    spec = CONDITIONS[cond]
    system = persona_prompt if spec["system"] == "full" else common.MINIMAL_SYSTEM
    history: list[dict] = []
    rows = []
    for i, user in enumerate(turns, 1):
        if (i - 1) % RESTART_EVERY == 0:
            lora.restart_server()
            lora.set_adapters(spec["adapters"])
        msgs = [{"role": "system", "content": system}] + history[-2 * MAX_HISTORY:] + [{"role": "user", "content": user}]
        # seed is passed through extra_body-free path: llama-server honours 'seed' only via raw API;
        # we emulate seeds by temperature 0.7 sampling + distinct run order; record seed for bookkeeping.
        response = llm.generate(msgs, temperature=0.7, max_tokens=300).strip()
        history += [{"role": "user", "content": user}, {"role": "assistant", "content": response}]
        hard = hard_checks.check(response)
        js = judge.score(judge_llm, persona_prompt, user, response)
        rows.append({"i": i, "user": user, "response": response, "hard": hard, "judge": js})
        mean_str = f"{js['mean']:.2f}" if js.get("valid") else "INVALID"
        print(f"[{cond}/{seed}] {i:02d} judge={mean_str} hard={'X' if hard['any'] else '.'}")
    return {"condition": cond, "seed": seed, "turns": rows, "summary": summarize(rows)}


def rejudge_file(path: Path, judge_llm, persona_prompt: str) -> dict:
    """Re-run judge.score on every saved (user, response) pair in an existing
    drift_<cond>_<seed>.json without touching the spike server/adapters or
    regenerating any response. The pre-rejudge file is preserved as a .v1.json
    sibling the first time this runs (so a second --rejudge doesn't clobber
    the true original with an already-rejudged copy)."""
    result = json.loads(path.read_text(encoding="utf-8"))
    v1_path = path.with_suffix("").with_suffix(".v1.json")
    if not v1_path.exists():
        shutil.copyfile(path, v1_path)
    for row in result["turns"]:
        row["judge"] = judge.score(judge_llm, persona_prompt, row["user"], row["response"])
    result["summary"] = summarize(result["turns"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--turns", type=int, default=50)
    ap.add_argument("--rejudge", action="store_true",
                     help="re-score existing out/drift_<cond>_<seed>.json files with judge.py; "
                          "does not regenerate responses or touch the spike server/adapters")
    args = ap.parse_args()

    persona_prompt = common.full_system_prompt()
    judge_llm = common.brain_llm()

    if args.rejudge:
        for cond in args.conditions:
            for seed in args.seeds:
                path = common.OUT_DIR / f"drift_{cond}_{seed}.json"
                if not path.exists():
                    print(f"skip {cond} {seed}: {path} not found")
                    continue
                t0 = time.time()
                result = rejudge_file(path, judge_llm, persona_prompt)
                path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
                print(cond, seed, result["summary"], f"{round(time.time() - t0, 1)}s -> {path}")
        return

    turns = json.loads((common.DATA_DIR / "drift_script.json").read_text(encoding="utf-8"))[: args.turns]
    llm = common.spike_llm()
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
