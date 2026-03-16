#!/usr/bin/env python3
"""Run all Woven Imprint evaluation suites."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.bench_memory import run_memory_suite
from eval.bench_persona import run_persona_suite


def main():
    print("=" * 60)
    print("WOVEN IMPRINT — Evaluation Suite")
    print("=" * 60)

    start = time.time()
    all_results = []

    # Memory & Core Mechanics
    print("\n--- Memory & Core Mechanics ---")
    mem_suite = run_memory_suite()
    print(mem_suite.summary())
    all_results.append(mem_suite)

    # Persona & Consistency
    print("\n--- Persona & Consistency ---")
    persona_suite = run_persona_suite()
    print(persona_suite.summary())
    all_results.append(persona_suite)

    # Summary
    total_passed = sum(s.passed for s in all_results)
    total_tests = sum(s.total for s in all_results)
    total_failed = sum(s.failed for s in all_results)
    avg_score = sum(s.avg_score for s in all_results) / len(all_results) if all_results else 0

    elapsed = (time.time() - start) * 1000

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed}/{total_tests} passed, {total_failed} failed")
    print(f"Average score: {avg_score:.1%}")
    print(f"Duration: {elapsed:.0f}ms")
    print("=" * 60)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    output = {
        "timestamp": timestamp,
        "total_passed": total_passed,
        "total_tests": total_tests,
        "avg_score": avg_score,
        "duration_ms": elapsed,
        "suites": [s.to_dict() for s in all_results],
    }

    output_path = output_dir / f"eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {output_path}")

    # Also save as latest
    latest_path = output_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
