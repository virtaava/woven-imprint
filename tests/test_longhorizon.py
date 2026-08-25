"""CI gate: the 60-simulated-day fake-clock long-horizon benchmark must all pass."""


def test_longhorizon_suite_all_pass():
    from eval.bench_longhorizon import run_longhorizon_suite

    suite = run_longhorizon_suite(days=60)
    failed = [r.name for r in suite.results if not r.passed]
    assert not failed, (
        f"failed: {failed}; details: {[r.details for r in suite.results if not r.passed]}"
    )
