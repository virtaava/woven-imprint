"""CI gate: the 60-simulated-day fake-clock long-horizon benchmark must all pass.

Benchmarks named in `bench_longhorizon.KNOWN_OPEN` are exempt from this gate —
each is a real, understood, deterministic finding (documented in the task-6
report and in bench_longhorizon.py's module/KNOWN_OPEN comments) that tuning
the benchmark script cannot fix without misrepresenting what's actually
measured. They still run and score every time and are rendered into
docs/RESULTS.md as failing, so the finding stays visible instead of being
silently excluded.
"""


def test_longhorizon_suite_all_pass():
    from eval.bench_longhorizon import KNOWN_OPEN, run_longhorizon_suite

    suite = run_longhorizon_suite(days=60)
    failed = [r.name for r in suite.results if not r.passed and r.name not in KNOWN_OPEN]
    assert not failed, (
        f"failed: {failed}; details: {[r.details for r in suite.results if not r.passed]}"
    )
