import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_drift  # noqa: E402
import judge  # noqa: E402


class _FakeLLM:
    """Returns each item of `responses` in turn on successive generate_json_robust calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate_json_robust(self, messages, temperature=0.0):
        self.calls += 1
        return self._responses[self.calls - 1]


def test_retry_recovers_from_one_bad_response():
    good = {"in_character": 1.0, "voice": 0.5, "constraints": 0.5, "engagement": 1.0}
    llm = _FakeLLM([{"in_character": 0.5}, good])  # first call missing 3 axes
    out = judge.score(llm, "persona", "user turn", "response text")
    assert llm.calls == 2
    assert out["valid"] is True
    assert out["mean"] == sum(good.values()) / 4
    assert out["raw"] == good


def test_two_bad_responses_mark_invalid():
    llm = _FakeLLM([{"in_character": 0.5}, {"voice": "not-a-number"}])
    out = judge.score(llm, "persona", "user turn", "response text")
    assert llm.calls == 2
    assert out["valid"] is False
    assert out["mean"] is None
    for k in ("in_character", "voice", "constraints", "engagement"):
        assert out[k] is None
    assert out["raw"] == {"voice": "not-a-number"}


def test_first_call_success_does_not_retry():
    good = {"in_character": 1.0, "voice": 1.0, "constraints": 1.0, "engagement": 1.0}
    llm = _FakeLLM([good])
    out = judge.score(llm, "persona", "user turn", "response text")
    assert llm.calls == 1
    assert out["valid"] is True
    assert out["mean"] == 1.0


def _row(i, mean, valid=True, hard_any=False):
    js = {"in_character": mean, "voice": mean, "constraints": mean, "engagement": mean,
          "mean": mean, "valid": valid, "raw": {}}
    if not valid:
        js = {"in_character": None, "voice": None, "constraints": None, "engagement": None,
              "mean": None, "valid": False, "raw": {}}
    return {"i": i, "user": "u", "response": "r", "hard": {"any": hard_any}, "judge": js}


def test_summarize_excludes_invalid_turns():
    rows = [_row(i, 0.9) for i in range(1, 31)]
    rows += [_row(31, None, valid=False)]  # judge-formatting noise, not genuine collapse
    rows += [_row(i, 0.9) for i in range(32, 51)]
    summary = bench_drift.summarize(rows)
    assert summary["judged"] == 49
    assert summary["invalid_judgements"] == 1
    assert abs(summary["mean"] - 0.9) < 1e-9
    # late window (turns 31-50) has 19 valid turns, all 0.9 -- the invalid
    # turn 31 must not silently become a 0.0 that drags this down.
    assert abs(summary["late_mean"] - 0.9) < 1e-9


def test_summarize_hard_violations_unaffected_by_judge_validity():
    rows = [_row(1, 0.9, hard_any=True), _row(2, None, valid=False, hard_any=True)]
    summary = bench_drift.summarize(rows)
    assert summary["hard_violations"] == 2
