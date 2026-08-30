"""Smoke test: the standalone eval/external analysis scripts must import cleanly (no execution)
— they're run manually (``python -m eval.external.<script>``), not covered by any other test,
and each guards its heavy work behind ``if __name__ == "__main__":``, so a plain import should
be side-effect-free and cheap."""


def test_diagnose_recall_imports():
    import eval.external.diagnose_recall  # noqa: F401


def test_ranking_experiments_imports():
    import eval.external.ranking_experiments  # noqa: F401


def test_flip_analysis_imports():
    import eval.external.flip_analysis  # noqa: F401
