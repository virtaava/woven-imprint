"""Eval framework — structured benchmarks for Woven Imprint capabilities."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""

    name: str
    passed: bool
    score: float  # 0.0 - 1.0
    details: dict = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class SuiteResult:
    """Result of a full benchmark suite run."""

    suite_name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def summary(self) -> str:
        lines = [
            f"Suite: {self.suite_name}",
            f"  Passed: {self.passed}/{self.total} ({self.avg_score:.1%} avg score)",
            f"  Duration: {self.total_duration_ms:.0f}ms",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.name}: {r.score:.2f}")
            if not r.passed and r.details:
                for k, v in r.details.items():
                    lines.append(f"         {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "avg_score": self.avg_score,
            "total_duration_ms": self.total_duration_ms,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details,
                    "duration_ms": r.duration_ms,
                }
                for r in self.results
            ],
        }

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def timed(fn):
    """Decorator to time a benchmark function."""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        if isinstance(result, BenchmarkResult):
            result.duration_ms = elapsed
        return result

    return wrapper
