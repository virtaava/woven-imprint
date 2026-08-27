"""Tests for eval/render_results.py's external-benchmarks section.

The deterministic-suites render (`render()`) is exercised indirectly through
`render_full()`; these tests focus on the additive "## External benchmarks" section that
`render_external()`/`render_full()` append when `eval/results/external_latest.json` exists.
"""

from __future__ import annotations

from eval.render_results import render, render_external, render_full

_SUITE_DATA = {
    "timestamp": 1_800_000_000,
    "suites": [
        {
            "suite_name": "Memory & Core Mechanics",
            "results": [{"name": "recall_precision_at_5", "passed": True, "score": 1.0}],
        }
    ],
}


def _locomo_result(**overrides) -> dict:
    result = {
        "run_id": "locomo-mem-v1",
        "bench": "locomo",
        "mode": "memory",
        "timestamp": "2026-08-27T12:00:00+00:00",
        "summary": {
            "n_questions": 1540,
            "overall_j": 0.812,
            "overall_f1": 0.554,
            "per_category": {
                "1": {"n": 282, "j": 0.79, "f1": 0.51},
                "2": {"n": 321, "j": 0.80, "f1": 0.52},
                "3": {"n": 96, "j": 0.83, "f1": 0.58},
                "4": {"n": 841, "j": 0.83, "f1": 0.57},
            },
            "rule_scored": {"5": {"n": 446, "accuracy": 0.71}},
            "adversarial_accuracy": 0.71,
            "abstain_accuracy": None,
            "n_unparsed": 0,
            "mean_prompt_tokens_est": 612.3,
        },
        "conversations": [{"qid": "conv-1-q0", "judge_version": 2}],
    }
    result.update(overrides)
    return result


def _plus_result(**overrides) -> dict:
    result = {
        "run_id": "plus-mem-v1",
        "bench": "locomo_plus",
        "mode": "memory",
        "timestamp": "2026-08-27T13:00:00+00:00",
        "summary": {
            "n_probes": 401,
            "cognitive_accuracy": 0.63,
            "per_relation_type": {
                "causal": {"n": 101, "cognitive_accuracy": 0.6},
                "state": {"n": 100, "cognitive_accuracy": 0.65},
            },
            "per_time_gap": {
                "1 week": {"n": 150, "cognitive_accuracy": 0.7},
                "2 months": {"n": 50, "cognitive_accuracy": 0.5},
            },
            "mean_prompt_tokens_est": 300.0,
            "mean_seconds": 2.1,
            "mean_llm_calls": 2.0,
        },
    }
    result.update(overrides)
    return result


def test_render_external_empty_dict_returns_empty_string():
    assert render_external({}) == ""
    assert render_external(None) == ""  # type: ignore[arg-type]


def test_render_full_omits_section_when_external_missing():
    out = render_full(_SUITE_DATA, None)
    assert "External benchmarks" not in out
    assert out == render(_SUITE_DATA)


def test_render_full_omits_section_when_external_empty_dict():
    out = render_full(_SUITE_DATA, {})
    assert out == render(_SUITE_DATA)


def test_render_full_headline_score_line_unaffected_by_external_section():
    external = {"locomo:memory": _locomo_result()}
    out = render_full(_SUITE_DATA, external)
    # The deterministic headline line is unchanged and still precedes the new section.
    headline = "**1/1 passed** | Average score: 100.0%"
    assert headline in out
    assert out.index(headline) < out.index("## External benchmarks")


def test_render_external_locomo_table_rows():
    external = {"locomo:memory": _locomo_result()}
    section = render_external(external)

    assert "## External benchmarks (real LLM, local judge)" in section
    assert "### `locomo:memory`" in section
    assert "Run id: `locomo-mem-v1`" in section
    assert "Overall J (categories 1-4 / non-abstain): 81.2%" in section
    assert "(n=1540)" in section
    assert "Mean token-F1: 0.554" in section
    assert "Adversarial accuracy (category 5): 71.0%" in section
    assert "Abstain accuracy (`_abs`): n/a" in section
    assert "Mean prompt tokens (est.): 612" in section
    assert "Judge version: 2" in section
    # Per-category rows, numerically sorted.
    assert "| 1 | 282 | 79.0% | 0.510 |" in section
    assert "| 4 | 841 | 83.0% | 0.570 |" in section
    # Rule-scored (abstention) row, labelled and rendered separately from the judged categories.
    assert "category (abstention rule)" in section
    assert "| 5 | 446 | 71.0% |" in section


def test_render_external_judge_version_reports_the_set_present():
    """A run with a mix of judge_version 2 and un-rejudged (missing-field) records must report
    both, not just the first record's version."""
    external = {
        "locomo:memory": _locomo_result(
            conversations=[
                {"qid": "conv-1-q0", "judge_version": 2},
                {"qid": "conv-1-q1"},  # predates the field -> normalizes to "1"
            ]
        )
    }
    section = render_external(external)
    assert "Judge version: 1,2" in section


def test_render_external_locomo_plus_table_rows():
    external = {"locomo_plus:memory": _plus_result()}
    section = render_external(external)

    assert "### `locomo_plus:memory`" in section
    assert "Cognitive accuracy overall: 63.0% (n=401)" in section
    assert "| causal | 101 | 60.0% |" in section
    assert "| state | 100 | 65.0% |" in section
    assert "| 1 week | 150 | 70.0% |" in section
    assert "| 2 months | 50 | 50.0% |" in section


def test_render_external_multiple_keys_all_present():
    external = {
        "locomo:memory": _locomo_result(),
        "locomo:fullcontext": _locomo_result(run_id="locomo-full-v1", mode="fullcontext"),
        "locomo_plus:memory": _plus_result(),
    }
    section = render_external(external)
    assert "### `locomo:memory`" in section
    assert "### `locomo:fullcontext`" in section
    assert "### `locomo_plus:memory`" in section


def test_render_external_robust_to_missing_summary_keys():
    """A partial result (e.g. a killed/aggregating run) must render, not raise."""
    partial = {"run_id": "partial-v1", "bench": "locomo", "mode": "memory"}
    section = render_external({"locomo:memory": partial})
    assert "### `locomo:memory`" in section
    assert "n/a" in section


def test_render_external_robust_to_empty_result_value():
    section = render_external({"locomo:memory": {}})
    assert "### `locomo:memory`" in section


def test_render_external_robust_to_non_dict_result_value():
    """A malformed entry (wrong type entirely) is reported inline, not raised."""
    section = render_external({"locomo:memory": "not-a-dict"})
    assert "### `locomo:memory`" in section
    assert "could not render" in section


def test_render_external_longmemeval_uses_qa_renderer():
    result = _locomo_result(bench="longmemeval_s", run_id="lme-s-100-v1")
    section = render_external({"longmemeval_s:memory": result})
    assert "### `longmemeval_s:memory`" in section
    assert "Overall J (categories 1-4 / non-abstain)" in section
