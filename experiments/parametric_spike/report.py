"""Aggregate spike outputs into RESULTS.md with the H1/H2 decision."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def _load(glob: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(common.OUT_DIR.glob(glob))]


def main() -> None:
    drift = _load("drift_*.json")
    facts = {d["condition"]: d for d in _load("facts_*.json")}
    logs = {p.parent.name: json.loads(p.read_text()) for p in (common.OUT_DIR / "adapters").glob("*/train_log.json")}

    by = {}
    for d in drift:
        by.setdefault(d["condition"], []).append(d)
    agg = {c: {"mean": mean(r["summary"]["mean"] for r in rs),
               "late_mean": mean(r["summary"]["late_mean"] for r in rs),
               "slope": mean(r["summary"]["slope"] for r in rs),
               "hard": sum(r["summary"]["hard_violations"] for r in rs),
               "n": len(rs)} for c, rs in by.items()}

    a = agg.get("A")
    h1 = False
    if a:
        for c in ("B", "C"):
            if c in agg and agg[c]["mean"] >= a["mean"] + 0.05 and agg[c]["hard"] <= a["hard"] / 2:
                h1 = True
    D, E = facts.get("D", {}).get("summary"), facts.get("E", {}).get("summary")
    h2 = bool(D and E and E["recall_acc"] >= D["recall_acc"] + 0.20 and E["temporal_acc"] > D["temporal_acc"])

    lines = ["# Parametric layer spike — results", "",
             "Spec: docs/superpowers/specs/2026-08-23-parametric-layer-spike.md", "",
             "## Experiment 1 — persona drift (50 turns, Qwen3-4B Q8_0, judge Qwen3.5-35B)", "",
             "| cond | runs | judge mean | late mean (31–50) | slope/turn | hard violations |", "|---|---|---|---|---|---|"]
    for c in ("A", "B", "C"):
        if c in agg:
            g = agg[c]
            lines.append(f"| {c} | {g['n']} | {g['mean']:.3f} | {g['late_mean']:.3f} | {g['slope']:+.4f} | {g['hard']} |")
    lines += ["", f"**H1 (LoRA reduces drift): {'PASS' if h1 else 'FAIL'}**", "",
              "## Experiment 2 — facts in weights (D) vs explicit dated memory (E)", "",
              "| cond | recall acc (150 held-out paraphrases) | temporal acc (50) |", "|---|---|---|"]
    for c in ("D", "E"):
        if c in facts:
            s = facts[c]["summary"]
            lines.append(f"| {c} | {s['recall_acc']:.3f} | {s['temporal_acc']:.3f} |")
    lines += ["", f"**H2 (explicit store beats weights for facts): {'PASS' if h2 else 'FAIL'}**", "",
              "## Training cost on GB10", "", "| adapter | examples | steps | final loss | wall s | peak mem GB |", "|---|---|---|---|---|---|"]
    for name, l in sorted(logs.items()):
        lines.append(f"| {name} | {l['examples']} | {l['steps']} | {l['final_loss']:.3f} | {l['wall_s']} | {l['peak_mem_gb']} |")
    decision = ("BUILD the hybrid: persona/voice in a sleep-time-distilled LoRA, facts and relationships in the explicit store."
                if h1 and h2 else
                "DO NOT build the parametric layer as specified." if not h1 else
                "H1 passed but H2 failed: re-examine the explicit store before deciding (unexpected).")
    lines += ["", "## Decision", "", decision, ""]
    (common.SPIKE_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(decision)


if __name__ == "__main__":
    main()
